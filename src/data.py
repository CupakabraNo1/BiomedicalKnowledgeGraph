"""Loading the Hetionet graph from disk into a networkx graph."""
from __future__ import annotations

import bz2
import json
import re

import networkx as nx
from neo4j import GraphDatabase
from collections import defaultdict
from itertools import islice


from . import config

def get_driver():
    """Neo4j driver built from the configuration."""
    return GraphDatabase.driver(
        config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
    )


def scalar_props(data):
    """Keep only Neo4j-storable values from a `data` dict.

    Neo4j properties must be primitives or lists of primitives (no nested
    maps), so this keeps scalars and lists-of-scalars and drops anything else.
    """
    clean = {}
    for key, value in (data or {}).items():
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        elif isinstance(value, list) and all(
            isinstance(v, (str, int, float, bool)) for v in value
        ):
            clean[key] = value
    return clean


def create_nodes(G, nodes):
    for node in nodes:
        kind = node["kind"]
        G.add_node(
            (kind, node["identifier"]),
            kind=kind,
            identifier=node["identifier"],
            name=node.get("name"),
            **scalar_props(node.get("data")),
        )


def create_edges(G, edges):
    for edge in edges:
        source = tuple(edge["source_id"])   # [kind, id] -> (kind, id)
        target = tuple(edge["target_id"])

        G.add_edge(
            source, target,
            key=edge["kind"],
            kind=edge["kind"],
            direction=edge.get("direction", "both"),
            **scalar_props(edge.get("data")),
        )

def sanitize_reltype(kind):
    # "treats" -> "TREATS", "resembles" -> "RESEMBLES"
    return re.sub(r"[^A-Za-z0-9_]", "_", kind).upper()
 
def chunked(iterable, size):
    it = iter(iterable)
    while batch := list(islice(it, size)):
        yield batch
    
 
def ensure_constraints(session, kinds):
    """One uniqueness constraint per node label.

    Each constraint is backed by an index, without which every MATCH on
    `identifier` below would degrade into a full label scan.
    """
    for kind in sorted(kinds):
        session.run(
            f"CREATE CONSTRAINT IF NOT EXISTS "
            f"FOR (n:`{kind}`) REQUIRE n.identifier IS UNIQUE"
        ).consume()
    session.run("CALL db.awaitIndexes()").consume()


def load_nodes_to_db(session, G):
    """Write every node, grouped by kind so each label gets its own batched MERGE."""
    groups = defaultdict(list)
    for _, attrs in G.nodes(data=True):
        groups[attrs["kind"]].append({
            "identifier": attrs["identifier"],
            "props": {k: val for k, val in attrs.items() if k != "kind"},
        })

    ensure_constraints(session, groups)

    for kind, rows in groups.items():
        query = (
            f"UNWIND $rows AS row "
            f"MERGE (n:`{kind}` {{identifier: row.identifier}}) "
            f"SET n += row.props"
        )
        for batch in chunked(rows, config.BATCH_SIZE):
            session.run(query, rows=batch).consume()
        print(f"  nodes  {kind}: {len(rows):,}")


def load_edges_to_db(session, G):
    """Write every relationship, grouped by (source kind, edge kind, target kind)."""
    groups = defaultdict(list)
    for u, v, attrs in G.edges(data=True):
        key = (G.nodes[u]["kind"], attrs["kind"], G.nodes[v]["kind"])
        groups[key].append({
            "src": u[1],   # source identifier
            "tgt": v[1],   # target identifier
            "props": {k: val for k, val in attrs.items() if k != "kind"},
        })

    for (src_kind, edge_kind, tgt_kind), rows in groups.items():
        rel = sanitize_reltype(edge_kind)
        query = (
            f"UNWIND $rows AS row "
            f"MATCH (a:`{src_kind}` {{identifier: row.src}}) "
            f"MATCH (b:`{tgt_kind}` {{identifier: row.tgt}}) "
            f"MERGE (a)-[r:`{rel}`]->(b) "
            f"SET r += row.props"
        )
        for batch in chunked(rows, config.BATCH_SIZE):
            session.run(query, rows=batch).consume()
        print(f"  edges  {src_kind}-{edge_kind}->{tgt_kind}: {len(rows):,}")


def clear_db(drop_schema: bool = False, batch_size: int = 10_000):
    """Delete every node and relationship from Neo4j. Destructive, not reversible."""
    driver = get_driver()
    try:
        with driver.session() as session:
            before_nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            before_rels = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]

            session.run(
                f"MATCH ()-[r]->() CALL {{ WITH r DELETE r }} "
                f"IN TRANSACTIONS OF {batch_size} ROWS"
            ).consume()
            session.run(
                f"MATCH (n) CALL {{ WITH n DELETE n }} "
                f"IN TRANSACTIONS OF {batch_size} ROWS"
            ).consume()
            print(f"[clear_db]: deleted {before_nodes:,} nodes, {before_rels:,} relationships")

            if drop_schema:
                names = [r["name"] for r in session.run("SHOW CONSTRAINTS YIELD name")]
                for name in names:
                    session.run(f"DROP CONSTRAINT `{name}` IF EXISTS").consume()
                print(f"[clear_db]: dropped {len(names)} constraints")

            remaining = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            if remaining:
                raise RuntimeError(f"clear_db left {remaining:,} nodes behind")
    finally:
        driver.close()


def load_to_db(G):
    """Mirror the networkx graph into Neo4j: nodes first, then relationships."""
    driver = get_driver()
    try:
        with driver.session() as session:
            load_nodes_to_db(session, G)
            load_edges_to_db(session, G)
    finally:
        driver.close()


def metaedge_of(edge) -> tuple[str, str, str]:
    """The (source kind, edge kind, target kind) triple identifying an edge's type."""
    return (edge["source_id"][0], edge["kind"], edge["target_id"][0])


def selected_metaedges() -> set[tuple[str, str, str]]:
    """The target metaedge plus its supporting structure, as configured."""
    return {tuple(config.TARGET_METAEDGE),
            *(tuple(m) for m in config.SUPPORTING_METAEDGES)}


def load_graph(metaedges=None) -> nx.MultiDiGraph:
    """Load a Hetionet subgraph from data/raw/ and return it as a networkx graph."""
    metaedges = selected_metaedges() if metaedges is None else {tuple(m) for m in metaedges}

    HETIONET_FILE = config.RAW_DATA / config.HETIONET_FILENAME
    print(f"[load_graph]: reading {HETIONET_FILE}")

    opener = bz2.open if HETIONET_FILE.suffix == ".bz2" else open
    with opener(HETIONET_FILE, "rt", encoding="utf-8") as f:
        data = json.load(f)

    all_nodes = data.get("nodes", [])
    all_edges = data.get("edges", [])

    if not all_nodes or not all_edges:
        raise ValueError(
            f"{HETIONET_FILE} contains {len(all_nodes)} nodes and {len(all_edges)} edges. "
            "This is likely the metagraph (schema-only) file rather than the "
            "full hetnet — download hetionet-v1.0.json.bz2 instead."
        )

    edges = [e for e in all_edges if metaedge_of(e) in metaedges]
    if not edges:
        raise ValueError(
            f"No edges matched {sorted(metaedges)}. Check the metaedge triples "
            "against Hetionet's metaedge_tuples."
        )

    # Keep only the nodes those edges actually touch, so isolated nodes from the
    # rest of the hetnet do not inflate the subgraph.
    touched = {tuple(e["source_id"]) for e in edges} | {tuple(e["target_id"]) for e in edges}
    nodes = [n for n in all_nodes if (n["kind"], n["identifier"]) in touched]

    print(f"[load_graph]: selected {len(edges):,} of {len(all_edges):,} edges "
          f"across {len(metaedges)} metaedges")
    for meta in sorted(metaedges):
        count = sum(1 for e in edges if metaedge_of(e) == meta)
        target = "  (target)" if tuple(config.TARGET_METAEDGE) == meta else ""
        print(f"           {meta[0]}-{meta[1]}->{meta[2]}: {count:,}{target}")

    G = nx.MultiDiGraph()
    create_nodes(G, nodes)
    create_edges(G, edges)
    print(f"[load_graph]: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    load_to_db(G)
    return G
