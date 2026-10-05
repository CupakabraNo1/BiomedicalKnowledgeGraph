import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def auc_ap(pos_scores, neg_scores):
    """AUC and Average Precision: how well the scores separate positives from negatives."""
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    scores = np.concatenate([pos_scores, neg_scores])
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "ap": float(average_precision_score(labels, scores)),
    }


def ranks(pos_scores, neg_scores):
    """Rank of every positive among all negatives (1 = best). A tie counts as half."""
    pos = np.asarray(pos_scores)[:, None]
    neg = np.asarray(neg_scores)[None, :]
    return 1 + (neg > pos).sum(axis=1) + 0.5 * (neg == pos).sum(axis=1)


def summarize(pos_scores, neg_scores, ks=(10, 50)):
    """All metrics for one method: auc, ap, mrr, hits@k."""
    result = auc_ap(pos_scores, neg_scores)
    r = ranks(pos_scores, neg_scores)
    result["mrr"] = float(np.mean(1.0 / r))
    for k in ks:
        result[f"hits@{k}"] = float(np.mean(r <= k))
    return result


def results_table(pos_scores, neg_scores, cold=None):
    """Table with one row per method.

    pos_scores / neg_scores: dict method name -> scores.
    cold: optional mask over the positives; adds auc_warm and auc_cold.
    """
    rows = {}
    for name in pos_scores:
        row = summarize(pos_scores[name], neg_scores[name])
        if cold is not None:
            row["auc_warm"] = auc_ap(pos_scores[name][~cold], neg_scores[name])["auc"]
            row["auc_cold"] = auc_ap(pos_scores[name][cold], neg_scores[name])["auc"]
        rows[name] = row

    n_pos = len(next(iter(pos_scores.values())))
    n_neg = len(next(iter(neg_scores.values())))
    print(f"[results_table]: {len(rows)} methods, {n_pos:,} positives vs {n_neg:,} negatives")
    return pd.DataFrame.from_dict(rows, orient="index").sort_values("auc", ascending=False)
