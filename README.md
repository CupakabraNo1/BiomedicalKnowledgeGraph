# Link Prediction in a Biomedical Knowledge Graph

Course project (Deep Learning). Given a Hetionet subgraph, predict missing
**disease–gene** links. Node embeddings from random walks feed a dense
**autoencoder**; its scores are compared against classic neighbourhood heuristics.

## Pipeline

```
Hetionet (json.bz2) ─► Neo4j ─► networkx graph ─► train / val / test split + negatives
                                                        │
                       G_train ─► random walks ─► Word2Vec embeddings (128-d)
                                                        │
                       edge features (e_disease − e_gene)² ─► autoencoder (+ logistic head)
                                                        │
                       scores vs. heuristics ─► AUC, AP, MRR, Hits@K ─► tables + figures
```

1. **Load** – `Disease–associates–Gene` (the target) plus two supporting edge
   types, `Gene–interacts–Gene` and `Disease–resembles–Disease`, are read from
   Hetionet and written to Neo4j. Without the supporting edges the graph is
   bipartite and the neighbourhood heuristics would score every pair 0.
   Subgraph: 15,763 nodes (136 diseases, 15,627 genes), 160,330 edges.
2. **Split** – disease–gene edges are split 80/10/10 per disease
   (10,157 / 1,233 / 1,233). Diseases with fewer than 3 genes go entirely to
   train, and a node that would be left without edges gets one held-out edge
   back. Val/test edges are removed from the training graph `G_train`.
   Negatives are random disease–gene pairs that are not edges (1:1 for train,
   1:10 for val/test).
3. **Embeddings** – uniform random walks on `G_train` (DeepWalk, i.e. node2vec
   with p = q = 1; 10 walks × 80 steps per node) + skip-gram Word2Vec
   (window 10, 10 epochs).
4. **Model** – an autoencoder (128 → 64 → **32** → 64 → 128, ReLU, dropout 0.2,
   MSE, Adam 1e-3, 100 epochs) learns to rebuild the edge features of real
   train edges only. Two scores per pair:
   - `autoencoder` – minus the reconstruction error (no negatives seen),
   - `autoencoder_head` – logistic regression on the 32-d latent code, fitted on
     train positives vs. train negatives. The head is not saved; it is refitted
     after loading the model.
5. **Evaluation** – compared with Common Neighbors, Adamic–Adar, Jaccard,
   Preferential Attachment and two popularity controls (disease / gene degree).
   Test edges whose gene has no disease in train (**cold start**, 27%) are
   reported separately.

## Results (test set)

Two negative sets are used. With **uniform** negatives the disease of a negative
is random, so "how many genes does this disease have" alone reaches the best
AUC (0.796). **Per-disease** negatives keep the same diseases as the positives,
which removes that shortcut — this is the main comparison.

| method (uniform negatives) | AUC | AP | AUC warm | AUC cold |
|---|---|---|---|---|
| disease_degree | **0.796** | 0.266 | 0.795 | **0.800** |
| autoencoder_head | 0.774 | **0.346** | **0.842** | 0.596 |
| preferential_attachment | 0.769 | 0.262 | 0.808 | 0.665 |
| autoencoder | 0.756 | 0.313 | 0.819 | 0.587 |
| adamic_adar | 0.748 | 0.344 | 0.804 | 0.599 |

| method (per-disease negatives) | AUC | AP | AUC warm | AUC cold |
|---|---|---|---|---|
| jaccard | **0.696** | 0.251 | 0.759 | 0.530 |
| autoencoder_head | **0.696** | **0.252** | **0.772** | 0.494 |
| adamic_adar | 0.694 | 0.213 | 0.752 | **0.540** |
| autoencoder | 0.669 | 0.199 | 0.740 | 0.482 |
| disease_degree | 0.500 | 0.091 | – | – |

- The autoencoder with a logistic head ties with Jaccard on AUC; it is slightly
  ahead on AP and on warm edges, so it does not beat the simple heuristics overall.
- Reconstruction error alone is weaker than the heuristics; the gain comes from
  the supervised head.
- On cold-start genes every method is at chance level (AUC ≈ 0.5).

Full tables (also MRR, Hits@10, Hits@50): `reports/results_uniform.csv`,
`reports/results_per_disease.csv`. Figures in `reports/figures/`: embedding
PCA, training loss, reconstruction error of edges vs. non-edges, AUC/AP bars
for both negative sets, warm vs. cold AUC, and a subgraph with the top
predicted genes for one disease.

## Structure

```
Projekat_BioKG/
├── docker-compose.yml       # Neo4j 5.10
├── requirements.txt
├── data/
│   ├── raw/                 # hetionet-v1.0.json.bz2 (downloaded, not in git)
│   ├── processed/           # nodes.csv, edges.npz, splits.npz, embeddings.npy (+ meta .json)
│   └── neo4j/               # Neo4j database files
├── models/                  # autoencoder_l2.keras + training history
├── reports/                 # result tables (.csv) and figures/
├── notebooks/
│   └── bio_kg.ipynb         # the whole project, step by step
└── src/
    ├── config.py            # paths, Neo4j connection, all hyperparameters
    ├── query.py             # Cypher queries
    ├── data.py              # Hetionet -> networkx -> Neo4j and back
    ├── processing.py        # simple graph, node ids, split, negatives, save/load
    ├── embeddings.py        # random walks, Word2Vec, save/load
    ├── link_pred.py         # edge features, heuristic scores
    ├── model.py             # Autoencoder (+ logistic head)
    ├── evaluate.py          # AUC, AP, MRR, Hits@K, results table
    ├── visualize.py         # figures and top candidate genes
    └── main.py              # only checks Neo4j and loads the subgraph
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

docker compose up -d      # Neo4j at http://localhost:7474 (neo4j / mySecurePassword123)

curl -L -o data/raw/hetionet-v1.0.json.bz2 \
  https://github.com/hetio/hetionet/raw/main/hetnet/json/hetionet-v1.0.json.bz2
```

Connection settings can be overridden with `NEO4J_URI`, `NEO4J_USER`,
`NEO4J_PASSWORD`.

## Usage

Open `notebooks/bio_kg.ipynb` (`jupyter lab`).

- **Section 1 (Pipeline)** builds everything from scratch: loads Neo4j, splits,
  trains the embeddings and the autoencoder, and saves them to `data/processed/`
  and `models/`. Needs Neo4j; only needed once.
- **Sections 2–5** load the saved files and produce the tables and figures.
  They do not need Neo4j.

All settings (split fractions, negative ratios, walk length, model size, seed)
are in `src/config.py`.

Notes:
- `data.clear_db()` empties Neo4j. Run it before loading a different subgraph —
  loading merges, it does not replace.
- Word2Vec with several workers is not fully reproducible, and neither is
  Keras training, so retraining the embeddings or the autoencoder gives
  slightly different numbers. The saved `embeddings.npy` and
  `models/autoencoder_l2.keras` are the ones behind the reported results.
