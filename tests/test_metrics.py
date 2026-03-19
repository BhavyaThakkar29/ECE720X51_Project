# tests/test_metrics.py
"""
Unit tests for the evaluation metrics module.
"""

import pytest
from src.evaluation.metrics import compute_metrics, summarize_scores


def test_compute_metrics_perfect_separation():
    # Faithful texts score 1.0, hallucinated score 0.0
    scores = [1.0, 1.0, 0.0, 0.0]
    labels = [0, 0, 1, 1]
    metrics = compute_metrics(scores, labels, threshold=0.5)
    assert metrics["AUROC"] == pytest.approx(1.0)
    assert metrics["Accuracy"] == pytest.approx(1.0)


def test_compute_metrics_random():
    scores = [0.5, 0.5, 0.5, 0.5]
    labels = [0, 1, 0, 1]
    metrics = compute_metrics(scores, labels)
    assert "AUROC" in metrics
    assert "F1_hallucinated" in metrics


def test_summarize_scores():
    all_scores = [
        {"s_word": 0.8, "s_sent": 0.7, "s_doc": 0.6, "s_final": 0.7},
        {"s_word": 0.6, "s_sent": 0.5, "s_doc": 0.4, "s_final": 0.5},
    ]
    summary = summarize_scores(all_scores)
    assert summary["s_word"] == pytest.approx(0.7, abs=1e-3)
    assert summary["s_final"] == pytest.approx(0.6, abs=1e-3)
