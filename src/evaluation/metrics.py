# src/evaluation/metrics.py
"""
Evaluation metrics to compare MGCS against ground-truth hallucination labels.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from typing import List


def compute_metrics(scores: List[float], labels: List[int], threshold: float = 0.5) -> dict:
    """
    Compute evaluation metrics given MGCS scores and binary hallucination labels.

    Convention: lower score → more likely hallucinated.
    Labels: 0 = faithful, 1 = hallucinated.

    Args:
        scores:    List of S_final values (higher = more similar to reference).
        labels:    List of ground-truth binary labels.
        threshold: Decision threshold on 1 - score for binary predictions.

    Returns:
        Dictionary of metric names to values.
    """
    scores_arr = np.array(scores)
    labels_arr = np.array(labels)

    # Invert: hallucination = low similarity → high (1 - score)
    inv_scores = 1.0 - scores_arr
    preds = (inv_scores >= threshold).astype(int)

    auroc = roc_auc_score(labels_arr, inv_scores)
    auprc = average_precision_score(labels_arr, inv_scores)
    report = classification_report(labels_arr, preds, output_dict=True)

    return {
        "AUROC": round(auroc, 4),
        "AUPRC": round(auprc, 4),
        "Accuracy": round(report["accuracy"], 4),
        "Precision_hallucinated": round(report["1"]["precision"], 4),
        "Recall_hallucinated": round(report["1"]["recall"], 4),
        "F1_hallucinated": round(report["1"]["f1-score"], 4),
    }


def summarize_scores(all_scores: List[dict]) -> dict:
    """
    Given a list of MGCSScores dicts, return mean per-level scores.
    """
    keys = ["s_word", "s_sent", "s_doc", "s_final"]
    return {
        k: round(float(np.mean([s[k] for s in all_scores])), 4)
        for k in keys
    }
