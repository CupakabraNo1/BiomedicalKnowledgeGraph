import csv
import json
from collections import defaultdict

import networkx as nx
import numpy as np

from . import config, data


def to_simple_undirected(G):
    """Turn the Neo4j graph into a simple undirected graph (no self-loops, no duplicate edges)."""
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))

    self_loops = duplicates = 0
    for source, target, attributes in G.edges(data=True):
        if source == target:
            self_loops += 1
        elif H.has_edge(source, target):
            duplicates += 1
        else:
            H.add_edge(source, target, **attributes)

    print(f"[to_simple_undirected]: {H.number_of_nodes():,} nodes, {H.number_of_edges():,} edges "
          f"({self_loops:,} self-loops, {duplicates:,} duplicates dropped)")
    return H


def build_node_index(G):
    """Number the nodes 0..n-1: diseases first, then genes, each sorted by identifier."""
    index = {node: i for i, node in enumerate(sorted(G.nodes))}

    first_gene = min(i for (kind, _), i in index.items() if kind == config.GENE_KIND)
    print(f"[build_node_index]: {len(index):,} nodes "
          f"(Disease 0 - {first_gene - 1}, Gene {first_gene} - {len(index) - 1})")
    return index


def prepare_graph():
    """Neo4j -> simple undirected graph with integer node ids."""
    G = data.load_from_db()
    H = to_simple_undirected(G)
    index = build_node_index(H)
    return nx.relabel_nodes(H, index), index


def build_train_graph(G, train_pairs):
    """All supporting edges + only the train disease-gene edges."""
    G_train = nx.Graph()
    G_train.add_nodes_from(G.nodes(data=True))
    G_train.add_edges_from(
        (u, v, attributes) for u, v, attributes in G.edges(data=True)
        if attributes["kind"] != config.TARGET_EDGE_KIND
    )
    for disease, gene in train_pairs:
        G_train.add_edge(int(disease), int(gene), kind=config.TARGET_EDGE_KIND)
    return G_train


def split_edges(G, rng):
    """Split disease-gene edges into train / val / test, disease by disease.

    Diseases with fewer than MIN_DEGREE_FOR_HOLDOUT genes go entirely to train.
    A node left without edges in G_train can't get an embedding (a random walk
    can't start from it), so it gets one of its held-out edges back.
    """
    genes_of = defaultdict(list)
    for u, v, kind in G.edges(data="kind"):
        if kind != config.TARGET_EDGE_KIND:
            continue
        disease, gene = (u, v) if G.nodes[u]["kind"] == config.DISEASE_KIND else (v, u)
        genes_of[disease].append(gene)

    train, val, test = [], [], []
    for disease in sorted(genes_of):
        genes = np.array(sorted(genes_of[disease]))

        if len(genes) < config.MIN_DEGREE_FOR_HOLDOUT:
            train.extend((disease, gene) for gene in genes)
            continue

        genes = genes[rng.permutation(len(genes))]
        n_test = max(1, round(config.TEST_FRACTION * len(genes)))
        n_val = max(1, round(config.VALIDATION_FRACTION * len(genes)))

        test.extend((disease, gene) for gene in genes[:n_test])
        val.extend((disease, gene) for gene in genes[n_test:n_test + n_val])
        train.extend((disease, gene) for gene in genes[n_test + n_val:])

    G_train = build_train_graph(G, train)

    moved = 0
    if config.PROTECT_ISOLATED:
        isolated = sorted(node for node, degree in G_train.degree() if degree == 0)
        for node in isolated:
            for pool in (val, test):
                candidates = [pair for pair in pool if node in pair]
                if candidates:
                    pair = min(candidates)
                    pool.remove(pair)
                    train.append(pair)
                    G_train.add_edge(pair[0], pair[1], kind=config.TARGET_EDGE_KIND)
                    moved += 1
                    break

    train_pos = np.array(sorted(train), dtype=np.int64)
    val_pos = np.array(sorted(val), dtype=np.int64)
    test_pos = np.array(sorted(test), dtype=np.int64)

    print(f"[split_edges]: {len(train_pos):,} train / {len(val_pos):,} val / "
          f"{len(test_pos):,} test edges of shape (?, {train_pos.shape[1]}), {moved:,} moved to train to avoid isolated nodes")
    return G_train, train_pos, val_pos, test_pos


def encode(pairs, n_nodes):
    """Each (disease, gene) pair as one number, so pairs fit in a set."""
    return pairs[:, 0] * n_nodes + pairs[:, 1]


def candidate_space(all_pos, n_nodes):
    """Diseases and genes that appear in at least one positive pair."""
    diseases = np.unique(all_pos[:, 0])
    genes = np.unique(all_pos[:, 1])
    print(f"[candidate_space]: {len(diseases)} diseases × {len(genes):,} genes = "
          f"{len(diseases) * len(genes):,} possible pairs")
    return {"diseases": diseases, "genes": genes, "n_nodes": int(n_nodes)}


def sample_negatives(pos, space, forbidden, ratio, rng, sampler=None):
    """Random (disease, gene) pairs that are not real edges, `ratio` per positive.

    sampler="uniform":     any disease
    sampler="per_disease": the same diseases as the positives, so disease
                           popularity can't tell positives from negatives

    Genes that hit a real edge or a repeat are redrawn, up to NEGATIVE_MAX_ROUNDS times.
    """
    sampler = sampler or config.NEGATIVE_SAMPLER
    genes, n_nodes = space["genes"], space["n_nodes"]
    need = len(pos) * ratio

    if sampler == "per_disease":
        disease = np.repeat(pos[:, 0], ratio)
    elif sampler == "uniform":
        disease = rng.choice(space["diseases"], size=need)
    else:
        raise ValueError(f"unknown sampler: {sampler!r}")

    gene = np.zeros(need, dtype=np.int64)
    todo = list(range(need))
    seen = set(forbidden)
    for _ in range(config.NEGATIVE_MAX_ROUNDS):
        if not todo:
            break
        retry = []
        for slot, g in zip(todo, rng.choice(genes, size=len(todo))):
            code = disease[slot] * n_nodes + g
            if code in seen:
                retry.append(slot)
            else:
                seen.add(code)
                gene[slot] = g
        todo = retry

    if todo:
        raise RuntimeError(f"[sample_negatives] {len(todo):,} of {need:,} pairs still unfilled")

    negatives = np.column_stack([disease, gene]).astype(np.int64)
    print(f"[sample_negatives]: {len(negatives):,} negatives "
          f"({sampler}, 1:{ratio} on {len(pos):,} positives)")
    return negatives


def cold_start_mask(train_pos, test_pos):
    """True where the test gene has no disease edge in train (a "new" gene)."""
    mask = ~np.isin(test_pos[:, 1], train_pos[:, 1])
    print(f"[cold_start_mask]: {mask.sum():,} of {len(mask):,} test edges are "
          f"cold start ({100 * mask.mean():.1f}%)")
    return mask


def save_splits(G, index, splits):
    """Write nodes.csv, edges.npz, splits.npz and splits_meta.json to data/processed."""
    config.PROCESSED_DATA.mkdir(parents=True, exist_ok=True)

    with open(config.NODES_FILE, "w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "kind", "identifier", "name"])
        for idx in sorted(index.values()):
            node = G.nodes[idx]
            writer.writerow([idx, node["kind"], node["identifier"], node.get("name", "")])

    save_edges(G)

    arrays = {name: value for name, value in splits.items() if isinstance(value, np.ndarray)}
    np.savez_compressed(config.SPLIT_FILE, **arrays)

    meta = {
        "seed": config.SEED,
        "sampler": splits["sampler"],
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "target_metaedge": list(config.TARGET_METAEDGE),
        "supporting_metaedges": [list(m) for m in config.SUPPORTING_METAEDGES],
        "fractions": {"val": config.VALIDATION_FRACTION, "test": config.TEST_FRACTION},
        "min_degree_for_holdout": config.MIN_DEGREE_FOR_HOLDOUT,
        "negative_ratio": {
            "train": config.NEGATIVE_RATIO_TRAIN,
            "val": config.NEGATIVE_RATIO_VALIDATION,
            "test": config.NEGATIVE_RATIO_VALIDATION,
        },
        "counts": {name: len(value) for name, value in arrays.items()},
        "cold_start_share": round(float(splits["test_cold"].mean()), 4),
    }
    config.SPLIT_META_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[save_splits]: written to {config.PROCESSED_DATA}")


def save_edges(G):
    """Edges as an (n, 2) array plus a kind code per edge."""
    kind_names = sorted({kind for _, _, kind in G.edges(data="kind")})
    code_of = {kind: code for code, kind in enumerate(kind_names)}

    edge_list = list(G.edges(data="kind"))
    edges = np.array([(u, v) for u, v, _ in edge_list], dtype=np.int64)
    codes = np.array([code_of[kind] for _, _, kind in edge_list], dtype=np.int8)

    np.savez_compressed(config.EDGES_FILE, edges=edges, kind_codes=codes,
                        kind_names=np.array(kind_names))
    print(f"[save_edges]: {len(edges):,} edges ({', '.join(kind_names)})")


def load_nodes():
    """Read nodes.csv back into index and node_attributes (idx -> kind, identifier, name).

    Gene identifiers are numbers and are turned back into int; disease ids stay "DOID:...".
    """
    index, node_attributes = {}, {}
    with open(config.NODES_FILE, newline="", encoding="utf8") as f:
        for row in csv.DictReader(f):
            idx = int(row["idx"])
            identifier = row["identifier"]
            if identifier.isdigit():
                identifier = int(identifier)
            index[(row["kind"], identifier)] = idx
            node_attributes[idx] = {"kind": row["kind"], "identifier": identifier,
                                    "name": row["name"]}
    return index, node_attributes


def load_graph():
    """Rebuild G from nodes.csv + edges.npz (no Neo4j needed)."""
    index, node_attributes = load_nodes()

    G = nx.Graph()
    G.add_nodes_from(node_attributes.items())
    with np.load(config.EDGES_FILE) as npz:
        edges, codes, kind_names = npz["edges"], npz["kind_codes"], npz["kind_names"]
    for (u, v), code in zip(edges, codes):
        G.add_edge(int(u), int(v), kind=str(kind_names[code]))

    print(f"[load_graph]: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G, index


def load_splits():
    """Load G, index and splits from data/processed; G_train is rebuilt."""
    meta = json.loads(config.SPLIT_META_FILE.read_text(encoding="utf-8"))
    G, index = load_graph()

    if (meta["n_nodes"], meta["n_edges"]) != (G.number_of_nodes(), G.number_of_edges()):
        raise RuntimeError("[load_splits] graph files don't match splits_meta.json")

    with np.load(config.SPLIT_FILE) as npz:
        splits = {name: npz[name] for name in npz.files}
    splits["sampler"] = meta["sampler"]
    splits["G_train"] = build_train_graph(G, splits["train_pos"])

    print(f"[load_splits]: {len(splits['train_pos']):,} train / "
          f"{len(splits['val_pos']):,} val / {len(splits['test_pos']):,} test")
    return G, index, splits
