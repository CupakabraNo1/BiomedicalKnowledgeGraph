"""Hetionet subgraph: file -> networkx -> Neo4j, and back."""
from __future__ import annotations

import bz2
import json
import re
from collections import defaultdict
from itertools import islice

import networkx as nx
from neo4j import GraphDatabase

from . import config, query


def get_driver():
    return GraphDatabase.driver(
        config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
    )


def selected_metaedges():
    """The target metaedge plus its supporting structure, as configured."""
    return {tuple(config.TARGET_METAEDGE),
            *(tuple(m) for m in config.SUPPORTING_METAEDGES)}


def metaedge_of(edge):
    """The (source kind, edge kind, target kind) triple identifying an edge's type."""
    return (edge["source_id"][0], edge["kind"], edge["target_id"][0])


def reltype(kind):
    """Edge kind as a Neo4j relationship type: "resembles" -> "RESEMBLES"."""
    return re.sub(r"[^A-Za-z0-9_]", "_", kind).upper()


def scalar_props(data):
    """Neo4j stores primitives and lists of primitives, nothing nested."""
    def ok(value):
        return isinstance(value, (str, int, float, bool))

    return {
        key: value for key, value in (data or {}).items()
        if ok(value) or (isinstance(value, list) and all(ok(v) for v in value))
    }


def chunked(items, size):
    it = iter(items)
    while batch := list(islice(it, size)):
        yield batch


# --- file -> networkx ---------------------------------------------------------

def create_nodes(G, nodes):
    for node in nodes:
        G.add_node(
            (node["kind"], node["identifier"]),
            kind=node["kind"],
            identifier=node["identifier"],
            name=node["name"],
            **scalar_props(node.get("data")),
        )


def create_edges(G, edges):
    for edge in edges:
        G.add_edge(
            tuple(edge["source_id"]),          # [kind, id] -> (kind, id)
            tuple(edge["target_id"]),
            key=edge["kind"],
            kind=edge["kind"],
            direction=edge.get("direction", "both"),
            **scalar_props(edge.get("data")),
        )


def load_hetionet(metaedges=None, *, write_to_db=True):
    """Read the configured subgraph out of the Hetionet dump in data/raw/."""
    metaedges = selected_metaedges() if metaedges is None else {tuple(m) for m in metaedges}

    with bz2.open(config.RAW_DATA / config.HETIONET_FILENAME, "rt", encoding="utf-8") as f:
        hetnet = json.load(f)

    edges = [e for e in hetnet["edges"] if metaedge_of(e) in metaedges]
    # Keep only the nodes those edges touch, so the rest of the hetnet does not
    # come along as isolated nodes.
    touched = {tuple(e["source_id"]) for e in edges} | {tuple(e["target_id"]) for e in edges}
    nodes = [n for n in hetnet["nodes"] if (n["kind"], n["identifier"]) in touched]

    G = nx.MultiDiGraph()
    create_nodes(G, nodes)
    create_edges(G, edges)
    print(f"[load_hetionet]: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    if write_to_db:
        load_to_db(G)
    return G


# --- networkx <-> Neo4j -------------------------------------------------------

def write_nodes(session, G):
    groups = defaultdict(list)
    for _, attrs in G.nodes(data=True):
        groups[attrs["kind"]].append({
            "identifier": attrs["identifier"],
            "props": {k: v for k, v in attrs.items() if k != "kind"},
        })

    for kind in groups:
        session.run(query.create_constraint(kind)).consume()
    session.run(query.AWAIT_INDEXES).consume()

    for kind, rows in groups.items():
        for batch in chunked(rows, config.NEO4J_BATCH_SIZE):
            session.run(query.merge_nodes(kind), rows=batch).consume()
        print(f"  {kind}: {len(rows):,} nodes")


def write_edges(session, G):
    groups = defaultdict(list)
    for u, v, attrs in G.edges(data=True):
        key = (G.nodes[u]["kind"], attrs["kind"], G.nodes[v]["kind"])
        groups[key].append({
            "src": u[1],   # source identifier
            "tgt": v[1],   # target identifier
            "props": {k: val for k, val in attrs.items() if k != "kind"},
        })

    for (src_kind, kind, tgt_kind), rows in groups.items():
        statement = query.merge_edges(src_kind, reltype(kind), tgt_kind)
        for batch in chunked(rows, config.NEO4J_BATCH_SIZE):
            session.run(statement, rows=batch).consume()
        print(f"  {src_kind}-{kind}->{tgt_kind}: {len(rows):,} edges")


def load_to_db(G):
    """Mirror the graph into Neo4j: nodes first, then relationships."""
    with get_driver() as driver, driver.session() as session:
        write_nodes(session, G)
        write_edges(session, G)


def load_from_db():
    """Read the graph back out of Neo4j — same shape as load_hetionet, much faster."""
    kinds = {reltype(kind): kind for _, kind, _ in selected_metaedges()}

    G = nx.MultiDiGraph()
    with get_driver() as driver, driver.session() as session:
        # Nodes first, or add_edge creates the endpoints without attributes.
        for rec in session.run(query.FETCH_NODES):
            props = rec["props"]
            G.add_node((rec["kind"], props["identifier"]), kind=rec["kind"], **props)

        for rec in session.run(query.FETCH_EDGES):
            kind = kinds[rec["reltype"]]
            G.add_edge(
                (rec["src_kind"], rec["src"]),
                (rec["tgt_kind"], rec["tgt"]),
                key=kind,
                kind=kind,
                **rec["props"],
            )

    print(f"[load_from_db]: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


def count_db():
    """Nodes and relationships currently in the database."""
    with get_driver() as driver, driver.session() as session:
        nodes = session.run(query.COUNT_NODES).single()["count"]
        edges = session.run(query.COUNT_EDGES).single()["count"]
    print(f"[count_db]: {nodes:,} nodes, {edges:,} relationships")
    return nodes, edges


def clear_db():
    """Delete every node and relationship. Destructive, not reversible."""
    with get_driver() as driver, driver.session() as session:
        session.run(query.DELETE_EDGES).consume()
        session.run(query.DELETE_NODES).consume()
    print("[clear_db]: database emptied")
