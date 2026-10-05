# Data

Files here are **not version controlled** (see `.gitignore`).

| folder | contents | how it gets there |
|---|---|---|
| `raw/` | `hetionet-v1.0.json.bz2` (+ `hetionet-v1.0-metagraph.json`) | download, see below |
| `processed/` | `nodes.csv`, `edges.npz`, `splits.npz`, `splits_meta.json`, `embeddings.npy`, `embeddings_meta.json` | notebook section 1 |
| `neo4j/` | Neo4j database files | `docker compose up -d` + notebook section 1.1 |

Source: [Hetionet v1.0](https://het.io) — only `Disease–associates–Gene`,
`Gene–interacts–Gene` and `Disease–resembles–Disease` edges are used.

```bash
curl -L -o data/raw/hetionet-v1.0.json.bz2 \
  https://github.com/hetio/hetionet/raw/main/hetnet/json/hetionet-v1.0.json.bz2
```
