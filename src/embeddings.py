import json
import time

import networkx as nx
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config


def adjacency(G_train, n_nodes):
    """Adjacency matrix of the train graph (sparse CSR), row i = neighbours of node i."""
    A = nx.to_scipy_sparse_array(G_train, nodelist=range(n_nodes), format="csr", weight=None)
    if np.diff(A.indptr).min() == 0:
        raise ValueError("a node has no neighbours, a random walk can't start from it")
    return A


def random_walks(A, rng):
    """NUM_WALKS random walks of WALK_LENGTH steps from every node (DeepWalk style).

    All walks step at the same time: each moves to a random neighbour of its current node.
    """
    indptr, indices = A.indptr, A.indices
    degree = np.diff(indptr)

    start = np.tile(np.arange(len(degree), dtype=np.int32), config.NUM_WALKS)
    rng.shuffle(start)

    walks = np.empty((len(start), config.WALK_LENGTH), dtype=np.int32)
    walks[:, 0] = start

    current = start
    for step in range(1, config.WALK_LENGTH):
        offset = (rng.random(len(current)) * degree[current]).astype(np.int64)
        current = indices[indptr[current] + offset]
        walks[:, step] = current

    print(f"[random_walks]: {len(walks):,} walks x {config.WALK_LENGTH} nodes")
    return walks


def train_word2vec(walks, n_nodes):
    """Treat each walk as a sentence and learn a vector per node with skip-gram (row i = node i)."""
    corpus = [[str(node) for node in walk] for walk in walks]

    started = time.time()
    model = Word2Vec(
        corpus,
        vector_size=config.EMB_DIM,
        window=config.WINDOW,
        epochs=config.W2V_EPOCHS,
        negative=config.W2V_NEGATIVE,
        workers=config.W2V_WORKERS,
        seed=config.SEED,
        sg=1,
        min_count=0,
    )
    embeddings = np.vstack([model.wv[str(node)] for node in range(n_nodes)])

    print(f"[train_word2vec]: {n_nodes:,} x {config.EMB_DIM} in {time.time() - started:.0f} s")
    return embeddings.astype(np.float32)


def cosine(embeddings, pairs):
    """Cosine similarity of the two vectors in each (disease, gene) pair."""
    left, right = embeddings[pairs[:, 0]], embeddings[pairs[:, 1]]
    norm = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return (left * right).sum(axis=1) / np.maximum(norm, 1e-9)


def quality(embeddings, pos, neg):
    """Quick check of the embeddings: AUC and AP of plain cosine similarity."""
    scores = np.concatenate([cosine(embeddings, pos), cosine(embeddings, neg)])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    result = {
        "auc": float(roc_auc_score(labels, scores)),
        "ap": float(average_precision_score(labels, scores)),
    }
    print(f"[quality]: cosine AUC {result['auc']:.4f}, AP {result['ap']:.4f}")
    return result


def save_embeddings(embeddings, val_quality):
    """Write embeddings.npy and embeddings_meta.json."""
    config.PROCESSED_DATA.mkdir(parents=True, exist_ok=True)
    np.save(config.EMBEDDING_FILE, embeddings)

    meta = {
        "seed": config.SEED,
        "n_nodes": int(embeddings.shape[0]),
        "emb_dim": int(embeddings.shape[1]),
        "num_walks": config.NUM_WALKS,
        "walk_length": config.WALK_LENGTH,
        "window": config.WINDOW,
        "w2v_epochs": config.W2V_EPOCHS,
        "val_quality": val_quality,
    }
    config.EMBEDDING_META_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[save_embeddings]: {config.EMBEDDING_FILE.name} ({embeddings.nbytes / 1e6:.1f} MB)")


def load_embeddings(n_nodes):
    """Read embeddings.npy + meta and check the shape fits the current graph."""
    meta = json.loads(config.EMBEDDING_META_FILE.read_text(encoding="utf-8"))
    embeddings = np.load(config.EMBEDDING_FILE)

    if embeddings.shape != (n_nodes, config.EMB_DIM):
        raise RuntimeError(f"[load_embeddings] file has shape {embeddings.shape}, "
                           f"expected ({n_nodes}, {config.EMB_DIM})")

    print(f"[load_embeddings]: {embeddings.shape[0]:,} x {embeddings.shape[1]}")
    return embeddings, meta
