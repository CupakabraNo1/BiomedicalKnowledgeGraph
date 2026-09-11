"""Link prediction evaluation (Phase 3).

All functions operate on scores of positive (existing edges) and negative
(non-pairs) samples. Higher score = more likely link.
"""
from __future__ import annotations

import numpy as np


def evaluate_scores(pos_scores: np.ndarray, neg_scores: np.ndarray) -> dict:
    """AUC-ROC and Average Precision over positive vs. negative scores."""
    from sklearn.metrics import roc_auc_score, average_precision_score

    y = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    s = np.concatenate([pos_scores, neg_scores])
    return {
        "auc": float(roc_auc_score(y, s)),
        "ap": float(average_precision_score(y, s)),
    }


def hits_at_k(pos_scores: np.ndarray, neg_scores: np.ndarray, k: int) -> float:
    """Fraction of positive links ranked above the k-th best negative."""
    if len(neg_scores) < k:
        return float("nan")
    thr = np.sort(neg_scores)[::-1][k - 1]
    return float(np.mean(pos_scores > thr))


def mrr(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """Mean Reciprocal Rank: each positive is ranked against all negatives."""
    neg = np.asarray(neg_scores)
    ranks = [1 + int(np.sum(neg > p)) for p in pos_scores]
    return float(np.mean([1.0 / r for r in ranks]))


def summarize(pos_scores, neg_scores, ks=(10, 50)) -> dict:
    """Combine all metrics into a single dictionary."""
    out = evaluate_scores(pos_scores, neg_scores)
    out["mrr"] = mrr(pos_scores, neg_scores)
    for k in ks:
        out[f"hits@{k}"] = hits_at_k(pos_scores, neg_scores, k)
    return out
