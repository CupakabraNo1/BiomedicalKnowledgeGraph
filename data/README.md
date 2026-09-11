# Data

Files are **not version controlled** (see `.gitignore`) — download them locally.

| Dataset | Source | Role |
|---|---|---|
| Hetionet | https://het.io | Ready-made biomedical graph (+ Neo4j version) |
| OGB `ogbl-biokg` | https://ogb.stanford.edu | Link prediction benchmark (split + evaluator) |
| DisGeNET | https://www.disgenet.org | Gene–disease associations |
| STRING | https://string-db.org | Protein–protein interactions |
| ClinVar (optional) | https://www.ncbi.nlm.nih.gov/clinvar | "Variant" nodes (DNA layer) |

- `raw/` — downloaded edge files / Neo4j export
- `processed/` — edge lists, embeddings (`.npy`), splits

> Tip: start from a subgraph (a single edge type, e.g. gene–disease), not the whole graph.
