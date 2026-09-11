# Project: Link Prediction in a Biomedical Knowledge Graph

Course project (Deep Learning). A Hetionet subgraph is loaded into Neo4j; node2vec
representations feed a dense **autoencoder** that predicts missing gene–disease
links and flags anomalies.

## Scope

The prediction target is the `Disease–associates–Gene` metaedge. Two further
metaedges are loaded as supporting structure and are never predicted:
`Gene–interacts–Gene` and `Disease–resembles–Disease`. Without them the
gene–disease subgraph is bipartite, so Common Neighbors, Adamic–Adar and Jaccard
score exactly 0 on every gene–disease pair and the baseline comparison is empty.

Loaded subgraph: 15,763 nodes (15,627 genes, 136 diseases) and 160,330 edges, out
of the full hetnet's 47,031 nodes and 2,250,197 edges. The scope lives in
`src/config.py` as `TARGET_METAEDGE` and `SUPPORTING_METAEDGES`.

## Structure

```
Projekat_BioKG/
├── docker-compose.yml     # Neo4j 5.10
├── requirements.txt
├── data/
│   └── raw/               # hetionet-v1.0.json.bz2 (not version controlled)
├── notebooks/
│   └── bio_kg.ipynb
└── src/
    ├── config.py          # paths, Neo4j connection, subgraph scope
    ├── data.py            # Hetionet -> networkx -> Neo4j
    ├── main.py            # runs the notebook's steps headlessly
    ├── model.py           # autoencoder (stub)
    └── evaluate.py        # metrics (AUC, AP, Hits@K, MRR)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

docker compose up -d      # Neo4j at http://localhost:7474 (neo4j / mySecurePassword123)
```

Download the hetnet into `data/raw/`:

```bash
curl -L -o data/raw/hetionet-v1.0.json.bz2 \
  https://github.com/hetio/hetionet/raw/main/hetnet/json/hetionet-v1.0.json.bz2
```

## Usage

```bash
python -m src.main        # connection check -> load subgraph -> Neo4j -> summary
python src/main.py        # same, for an IDE run button
jupyter lab               # then open notebooks/bio_kg.ipynb
```

`src.data.clear_db()` empties the database. Run it before reloading with a
different scope: `load_to_db` merges rather than replaces, so edges from a
previous scope would otherwise linger.

Connection settings can be overridden via the environment variables `NEO4J_URI`,
`NEO4J_USER`, `NEO4J_PASSWORD` (see `src/config.py`).

## Status

| Step | Module | State |
|---|---|---|
| Load subgraph into Neo4j | `data.py` | done |
| Edge split + negative sampling | `data.py` | to do |
| node2vec embeddings | `embeddings.py` | to do — file not yet created |
| Edge features + baseline heuristics | `link_pred.py` | to do — file not yet created |
| Autoencoder | `model.py` | stub (`NotImplementedError`) |
| Metrics | `evaluate.py` | done |
