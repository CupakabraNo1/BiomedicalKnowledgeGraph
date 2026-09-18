"""Script equivalent of notebooks/bio_kg.ipynb.

Runs the notebook's steps in order so the pipeline can be executed headlessly:

    python -m src.main      # as a package module
    python src/main.py      # directly, e.g. from an IDE run button
"""
import sys
from collections import Counter
from pathlib import Path

if __package__:
    from . import config, data
else:
    # Run as a plain script: src/ is on sys.path but the project root is not,
    # so there is no parent package for a relative import to resolve against.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src import config, data


def step(title):
    """Print a section banner, standing in for the notebook's markdown cells."""
    print(f"\n=== {title} ===")


def describe_graph(G):
    """Summarise the loaded graph by node and edge kind."""
    node_kinds = Counter(kind for _, kind in G.nodes(data="kind"))
    edge_kinds = Counter(kind for _, _, kind in G.edges(data="kind"))

    print(f"{G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    for kind, count in node_kinds.most_common():
        print(f"  {kind:<12} {count:>9,} nodes")
    for kind, count in edge_kinds.most_common():
        print(f"  {kind:<12} {count:>9,} edges")


def main():
    step("Check connection to database")
    try:
        data.count_db()
    except Exception:
        sys.exit(f"Neo4j unreachable at {config.NEO4J_URI}. Try: docker compose up -d")

    step("Load graph")
    G = data.load_hetionet()

    step("Graph summary")
    describe_graph(G)


if __name__ == "__main__":
    main()
