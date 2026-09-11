"""Script equivalent of notebooks/bio_kg.ipynb.

Runs the notebook's steps in order so the pipeline can be executed headlessly.
Both invocation styles work:

    python -m src.main      # as a package module
    python src/main.py      # directly, e.g. from an IDE run button
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import networkx as nx

if __package__:
    from . import config, data
else:
    # Run as a plain script: src/ is on sys.path but the project root is not,
    # so there is no parent package for a relative import to resolve against.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src import config, data


def step(title: str) -> None:
    """Print a section banner, standing in for the notebook's markdown cells."""
    print(f"\n=== {title} ===")


def check_db_data(driver) -> tuple[int, int]:
    """Count nodes and relationships currently stored in Neo4j."""
    with driver.session() as session:
        nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        rels = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]

    print(f"nodes: {nodes}, relationships: {rels}")
    return nodes, rels


def describe_graph(G: nx.MultiDiGraph, top: int = 5) -> None:
    """Summarise the loaded graph by node and edge kind."""
    node_kinds = Counter(kind for _, kind in G.nodes(data="kind"))
    edge_kinds = Counter(kind for _, _, kind in G.edges(data="kind"))

    print(f"{G.number_of_nodes():,} nodes across {len(node_kinds)} kinds, "
          f"{G.number_of_edges():,} edges across {len(edge_kinds)} kinds")

    print(f"top {top} node kinds:")
    for kind, count in node_kinds.most_common(top):
        print(f"  {kind:<24} {count:>9,}")

    print(f"top {top} edge kinds:")
    for kind, count in edge_kinds.most_common(top):
        print(f"  {kind:<24} {count:>9,}")


def main() -> int:
    step("Check connection to database")
    driver = data.get_driver()
    try:
        driver.verify_connectivity()
        print(f"connected to {config.NEO4J_URI}")
        check_db_data(driver)
    except Exception as exc:
        print(f"Neo4j unreachable at {config.NEO4J_URI}: {exc}", file=sys.stderr)
        print("Is the container up? Try: docker compose up -d", file=sys.stderr)
        return 1
    finally:
        driver.close()

    step("Load graph")
    try:
        G = data.load_graph()
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1

    step("Graph summary")
    describe_graph(G)

    return 0


if __name__ == "__main__":
    sys.exit(main())
