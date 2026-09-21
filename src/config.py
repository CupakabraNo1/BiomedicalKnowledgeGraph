import os
from pathlib import Path
# db data
NEO4J_URI            = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER           = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD       = os.getenv("NEO4J_PASSWORD", "mySecurePassword123")

# files
ROOT                 = Path(__file__).resolve().parents[1]
DATA                 = ROOT / "data"
RAW_DATA             = DATA / "raw"
PROCESSED_DATA       = DATA / "processed"
HETIONET_FILENAME    = "hetionet-v1.0.json.bz2"
NODES_FILE           = PROCESSED_DATA / "nodes.csv"
EDGES_FILE           = PROCESSED_DATA / "edges.npz"
SPLIT_FILE           = PROCESSED_DATA / "splits.npz"
SPLIT_META_FILE      = PROCESSED_DATA / "splits_meta.json"
EMBEDDING_FILE       = PROCESSED_DATA / "embeddings.npy"
EMBEDDING_META_FILE  = PROCESSED_DATA / "embeddings_meta.json"


# load data
NEO4J_BATCH_SIZE     = 20_000
TARGET_METAEDGE      = ("Disease", "associates", "Gene")
SUPPORTING_METAEDGES = (
    ("Gene", "interacts", "Gene"),
    ("Disease", "resembles", "Disease"),
)
DISEASE_KIND, TARGET_EDGE_KIND, GENE_KIND = TARGET_METAEDGE

# procssing data
SEED                      = 42
VALIDATION_FRACTION       = 0.10
TEST_FRACTION             = 0.10
MIN_DEGREE_FOR_HOLDOUT    = 3
PROTECT_ISOLATED          = True
NEGATIVE_SAMPLER          = "uniform"
NEGATIVE_RATIO_TRAIN      = 1
NEGATIVE_RATIO_VALIDATION = 10
NEGATIVE_MAX_ROUNDS       = 20

# embeddings
EMB_DIM                   = 128
NUM_WALKS                 = 10
WALK_LENGTH               = 80
WINDOW                    = 10
P                         = 1.0
Q                         = 1.0
W2V_EPOCHS                = 10
W2V_NEGATIVE              = 5
W2V_WORKERS               = 8
