import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
# Downloaded source files. Not data/neo4j/ — that is the database's own Docker
# volume and gets wiped whenever the container is reset.
RAW_DATA = DATA / "raw"
PROCESSED_DATA = DATA / "processed"
HETIONET_FILENAME = "hetionet-v1.0.json.bz2"
BATCH_SIZE = 20_000


TARGET_METAEDGE = ("Disease", "associates", "Gene")
SUPPORTING_METAEDGES = (
    ("Gene", "interacts", "Gene"),
    ("Disease", "resembles", "Disease"),
)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "mySecurePassword123")

SEED = 42
