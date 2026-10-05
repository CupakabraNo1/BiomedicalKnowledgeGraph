import numpy as np

from . import config

EDGE_OPS = {
    "l2":       lambda left, right: (left - right) ** 2,
    "hadamard": lambda left, right: left * right,
}


def edge_features(embeddings, pairs, op=None):
    """One feature row per (disease, gene) pair."""
    op = op or config.EDGE_OP
    if op not in EDGE_OPS:
        raise ValueError(f"unknown EDGE_OP {op!r}, pick from {sorted(EDGE_OPS)}")
    left, right = embeddings[pairs[:, 0]], embeddings[pairs[:, 1]]
    features = EDGE_OPS[op](left, right).astype(np.float32)
    print(f"[edge_features] op={op}, pairs={len(pairs)}, "
          f"embeddings={embeddings.shape} -> features={features.shape}")
    print(f"[edge_features] mean={features.mean():.4f}, std={features.std():.4f}, "
          f"min={features.min():.4f}, max={features.max():.4f}")
    return features


def heuristic_scores(A, pairs):
    """Classic link prediction scores from the train graph (A = adjacency matrix).

    common_neighbors            - how many neighbours the two nodes share
    adamic_adar                 - same, but rare (low degree) neighbours count more
    jaccard                     - shared neighbours / all neighbours
    preferential_attachment     - degree × degree
    disease_degree, gene_degree - popularity alone, as a sanity check
    """
    binary = A.astype(np.float32)
    degree = np.asarray(binary.sum(axis=1)).ravel().astype(np.int64)
    print(f"[heuristic_scores] A={A.shape}, edges(nnz)={A.nnz}, pairs={len(pairs)}")
    print(f"[heuristic_scores] degree: mean={degree.mean():.2f}, max={degree.max()}, "
          f"isolated={(degree == 0).sum()}")

    inv_log = np.zeros(len(degree), dtype=np.float32)
    inv_log[degree > 1] = 1.0 / np.log(degree[degree > 1])
    weighted = binary.multiply(inv_log[None, :]).tocsr()

    left, right = pairs[:, 0], pairs[:, 1]
    common = np.asarray(binary[left].multiply(binary[right]).sum(axis=1)).ravel()
    adamic = np.asarray(binary[left].multiply(weighted[right]).sum(axis=1)).ravel()
    deg_left, deg_right = degree[left], degree[right]

    scores = {
        "common_neighbors": common,
        "adamic_adar": adamic,
        "jaccard": common / np.maximum(deg_left + deg_right - common, 1),
        "preferential_attachment": (deg_left * deg_right).astype(np.float64),
        "disease_degree": deg_left.astype(np.float64),
        "gene_degree": deg_right.astype(np.float64),
    }
    for name, values in scores.items():
        print(f"[heuristic_scores] {name:<24} mean={values.mean():.4f}, "
              f"max={values.max():.4f}, zeros={(values == 0).mean():.1%}")
    return scores


def all_scores(A, autoencoder, embeddings, pairs):
    """Heuristic scores + both autoencoder scores for the same pairs."""
    scores = heuristic_scores(A, pairs)
    x = edge_features(embeddings, pairs)
    scores["autoencoder"] = autoencoder.link_scores(x)
    scores["autoencoder_head"] = autoencoder.head_scores(x)
    for name in ("autoencoder", "autoencoder_head"):
        values = np.asarray(scores[name])
        print(f"[all_scores] {name:<16} shape={values.shape}, mean={values.mean():.4f}, "
              f"min={values.min():.4f}, max={values.max():.4f}")
    print(f"[all_scores] {len(scores)} score types: {list(scores)}")
    return scores
