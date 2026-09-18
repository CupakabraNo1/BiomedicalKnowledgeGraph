from . import config, data
import networkx as nx
import numpy as np
import csv, json
from collections import defaultdict
 
# positives 
def to_simple_undirected(G):
    H = nx.Graph()
    
    for node, attributes in G.nodes(data=True):
        H.add_node(node, **attributes)
    
    self_loops = parallel = 0    
    for source, target, attributes in G.edges(data=True):
        if source == target:
            self_loops += 1
            continue
        if H.has_edge(source, target):
            parallel += 1
            kinds = H[source][target].setdefault("kinds", [H[source][target]["kind"]])
            if attributes["kind"] not in kinds:
                kinds.append(attributes["kind"])
            continue
        H.add_edge(source, target, **attributes)
    print(f"[to_simple_undirected]: {H.number_of_nodes():,} nodes, "
          f"{H.number_of_edges():,} edges "
          f"({self_loops:,} self-loops, {parallel:,} parallel dropped)")
    return H

def build_node_index(G) -> dict:
    
    by_kind = defaultdict(list)
    for node, kind in G.nodes(data="kind"):
        by_kind[kind].append(node)
    index = {}
    for kind in sorted(by_kind):
        by_kind[kind].sort(key=lambda n: n[1])
        for node in by_kind[kind]:
            index[node] = len(index)
    
    spans = ", ".join(
        f"{kind} {index[nodes[0]]} - {index[nodes[-1]]}" for kind, nodes in sorted(by_kind.items())
    )
    print(f"[build_node_index]: {len(index):,} nodes ({spans})")
    return index

def prepare_graph():
    G = data.load_from_db()
    H = to_simple_undirected(G)
    index = build_node_index(H)
    H = nx.relabel_nodes(H, index)
    return H, index

def build_train_graph(G, target_edges):
    """Supporting edges (never hidden) + the given target edges."""
    G_train = nx.Graph()
    G_train.add_nodes_from(G.nodes(data=True))
    G_train.add_edges_from(
        (source, target, attributes) for source, target, attributes in G.edges(data=True)
        if attributes["kind"] != config.TARGET_EDGE_KIND
    )
    for disease, gene in target_edges:
        G_train.add_edge(int(disease), int(gene), kind=config.TARGET_EDGE_KIND)
    return G_train
    
def split_edges(G, index, rng):
    """Split target branches na train / val / test by disease and builds G_train"""
    node_of = {i: node for node, i in index.items()}
    by_disease = defaultdict(list)
    for source, target, kind in G.edges(data="kind"):
        if kind != config.TARGET_EDGE_KIND:
            continue
        disease, gene = (source, target) if G.nodes[source]["kind"] == config.DISEASE_KIND else (target, source)
        by_disease[disease].append(gene)
        
    print('[split_edges] node_of:', node_of)
    print('[split_edges] by_disese:', by_disease.items())
    
    # group by disease sorted by id 
    train, val, test = [], [], []
    for disease in sorted(by_disease):
        genes = np.array(sorted(by_disease[disease]))
        degree = len(genes)
        
        if degree < config.MIN_DEGREE_FOR_HOLDOUT:
            train.extend((disease, gene) for gene in genes)
            continue
        
        genes = genes[rng.permutation(degree)]
        n_test = max(1, round(config.TEST_FRACTION * degree))
        n_val = max(1, round(config.VALIDATION_FRACTION * degree))
        
        test.extend((disease, gene) for gene in genes[:n_test])
        val.extend((disease, gene) for gene in genes[n_test:n_test+n_val])
        train.extend((disease, gene) for gene in genes[n_test+n_val:])
    
    # create train graph: only train edges without 'associates'
    G_train = build_train_graph(G, train)
    
    # to protect isolated nodes (degree == 0) remove them from test and validation group
    # and put them in train group
    moved = 0
    if config.PROTECT_ISOLATED:
        isolated = sorted(node for node, deg in G_train.degree() if deg == 0)
        for nid in isolated:
            for pool in (val, test):
                candidates = [pair for pair in pool if nid in pair]
                if not candidates:
                    continue
                pair = min(candidates)
                pool.remove(pair)
                train.append(pair)
                G_train.add_edge(pair[0], pair[1], kind=config.TARGET_EDGE_KIND)
                moved += 1
                break
    
    # reshape in (n, 2) array of integers
    train_pos = np.array(sorted(train), dtype=np.int64)
    val_pos = np.array(sorted(val), dtype=np.int64)
    test_pos = np.array(sorted(test), dtype=np.int64)
    
    print(f"[split_edges]: {len(train_pos):,} train / {len(val_pos):,} val / "
          f"{len(test_pos):,} test grana, {moved:,} premešteno zbog izolovanih čvorova")
    
    return G_train, train_pos, val_pos, test_pos
    
# negatives
def encode(pairs, n_nodes):
    """Pairs (disease, gen) as single number"""
    return pairs[:, 0] * n_nodes + pairs[:, 1]

def candidate_space(all_pos, n_nodes):
    """"""
    diseases = np.unique(all_pos[:,0])
    genes, degrees = np.unique(all_pos[:,1], return_counts=True)
    
    largest = int(max(diseases.max(), genes.max()))
    if n_nodes <= largest:
        raise ValueError(f"n_nodes={n_nodes} mora biti > najvećeg indeksa čvora {largest}")
    print(f"[negative_space]: {len(diseases)} bolesti × {len(genes):,} gena = "
          f"{len(diseases) * len(genes):,} mogućih parova")
    return { 
            "diseases": diseases, 
            "genes": genes, 
            "gene_probs": degrees / degrees.sum(), 
            "n_nodes": int(n_nodes)
            }   


def sample_negatives(pos, space, forbidden, ratio, rng, sampler=None):
    """Draw `ratio` negative pairs per positive, from the disease x gene space."""
    
    sampler = sampler or config.NEGATIVE_SAMPLER
    genes, n_nodes = space["genes"], space["n_nodes"]
    probs = space["gene_probs"] if sampler == "degree_matched" else None
    
    need = len(pos) * ratio
    
    # One disease slot per negative we owe. per_disease copies the positives'
    # disease column, so "how many genes does this disease have" scores the
    # negatives exactly like the positives -- that baseline drops to AUC 0.500.
    if sampler == "per_disease":
        disease = np.repeat(pos[:, 0], ratio)
    elif sampler in ("uniform", "degree_matched"):
        disease = rng.choice(space["diseases"], size=need)
    else:
        raise ValueError(f"unknown sampler: {sampler!r}")
    
    gene = np.zeros(need, dtype=np.int64)
    todo = list(range(need))
    seen = set(forbidden)
    
    for _ in range(config.NEGATIVE_MAX_ROUNDS):
        if not todo:
            break
        draw = rng.choice(genes, size=len(todo), p=probs)
        retry = []
        for slot, g in zip(todo, draw):
            code = disease[slot] * n_nodes + g
            if code in seen:
                retry.append(slot)
                continue
            seen.add(code)
            gene[slot] = g
        todo = retry
        
    if todo:
        raise RuntimeError(
            f"[sample_negatives] {len(todo):,} of {need:,} pairs unfilled after "
            f"{config.NEGATIVE_MAX_ROUNDS} rounds -- the {sampler} space is too tight"
        )
        
    negatives = np.column_stack([disease, gene]).astype(np.int64)
    print(f"[sample_negatives]: {len(negatives):,} negatives "
    f"({sampler}, 1:{ratio} on {len(pos):,} positives)")
    return negatives


def cold_start_mask(train_pos, test_pos):
    """True where the test gene has no `associates` edge left in the train graph.

    Most genes have a single disease, so a good chunk of test edges are cold.
    That is reported, not repaired -- patching the split would leave only easy
    genes in test and inflate every number.
    """
    mask = ~np.isin(test_pos[:, 1], train_pos[:, 1])
    print(f"[cold_start_mask]: {mask.sum():,} of {len(mask):,} test edges are "
          f"cold start ({100 * mask.mean():.1f}%)")
    return mask

# save/load
def save_splits(G, index, splits):
    """Write processed data data/processed: nodes.csv (readable), edges.npz +
    splits.npz (arrays), splits_meta.json (settings)."""
    config.PROCESSED_DATA.mkdir(parents=True, exist_ok=True)

    with open(config.NODES_FILE, "w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "kind", "identifier", "name"])
        for _, idx in sorted(index.items(), key=lambda item: item[1]):
            attributes = G.nodes[idx]
            writer.writerow([idx, attributes["kind"], attributes["identifier"], attributes.get("name", "")])

    save_edges(G)

    arrays = {
        name: value for name, value in splits.items()
        if isinstance(value, np.ndarray)
    }
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
        "counts": {
            name: int(len(arr)) for name, arr in arrays.items()
        },
        "cold_start_share": round(float(splits["test_cold"].mean()), 4)
    }
    config.SPLIT_META_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[save_splits]: {config.NODES_FILE.name}, {config.EDGES_FILE.name}, "
          f"{config.SPLIT_FILE.name}, {config.SPLIT_META_FILE.name} written to "
          f"{config.PROCESSED_DATA}")


def save_edges(G):
    """Graph structure as arrays: (n, 2) endpoints plus one kind code per row."""
    kind_names = sorted({kind for _, _, kind in G.edges(data="kind")})
    code_of = {kind: code for code, kind in enumerate(kind_names)}

    edges = np.empty((G.number_of_edges(), 2), dtype=np.int64)
    codes = np.empty(G.number_of_edges(), dtype=np.int8)
    for row, (source, target, kind) in enumerate(G.edges(data="kind")):
        edges[row] = (source, target)
        codes[row] = code_of[kind]

    np.savez_compressed(config.EDGES_FILE, edges=edges, kind_codes=codes,
                        kind_names=np.array(kind_names))
    print(f"[save_edges]: {len(edges):,} edges ({', '.join(kind_names)})")


def load_nodes():
    """nodes.csv -> (index, attributes by idx) -- the inverse of what save_splits writes."""
    index, attributes = {}, {}
    with open(config.NODES_FILE, newline="", encoding="utf8") as f:
        for row in csv.DictReader(f):
            idx = int(row["idx"])
            identifier = row["identifier"]
            if identifier.lstrip("-").isdigit():
                identifier = int(identifier)
            index[(row["kind"], identifier)] = idx
            attributes[idx] = {"kind": row["kind"], "identifier": identifier,
                               "name": row["name"]}
    return index, attributes


def load_graph():
    """Rebuild the relabelled graph from nodes.csv + edges.npz, without touching Neo4j."""
    index, attributes = load_nodes()

    G = nx.Graph()
    G.add_nodes_from(attributes.items())
    with np.load(config.EDGES_FILE, allow_pickle=False) as npz:
        edges, codes, kind_names = npz["edges"], npz["kind_codes"], npz["kind_names"]
    for (source, target), code in zip(edges, codes):
        G.add_edge(int(source), int(target), kind=str(kind_names[code]))

    print(f"[load_graph]: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges "
          f"from {config.PROCESSED_DATA}")
    return G, index


def load_splits():
    """Rebuild processed data straight from data/processed."""
    meta = json.loads(config.SPLIT_META_FILE.read_text(encoding="utf-8"))
    G, index = load_graph()

    if meta["n_nodes"] != G.number_of_nodes() or meta["n_edges"] != G.number_of_edges():
        raise RuntimeError(
            f"[load_splits] meta records {meta['n_nodes']:,} nodes / "
            f"{meta['n_edges']:,} edges but nodes.csv + edges.npz hold "
            f"{G.number_of_nodes():,} / {G.number_of_edges():,} -- the files in "
            f"{config.PROCESSED_DATA} disagree"
        )

    with np.load(config.SPLIT_FILE) as npz:
        splits = {name: npz[name] for name in npz.files}
    splits["sampler"] = meta["sampler"]
    splits["G_train"] = build_train_graph(G, splits["train_pos"])

    print(f"[load_splits]: {len(splits['train_pos']):,} train / "
          f"{len(splits['val_pos']):,} val / {len(splits['test_pos']):,} test")
    return G, index, splits
