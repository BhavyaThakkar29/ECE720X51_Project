# tests/test_aggregator.py
"""
Unit tests for the MGCS aggregator using mock embedders.
Run with: pytest tests/
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.aggregation.aggregator import MGCSAggregator, MGCSScores


@pytest.fixture
def mock_aggregator():
    """Aggregator with mocked similarity scorers."""
    agg = MGCSAggregator.__new__(MGCSAggregator)
    agg.alpha = 0.3
    agg.beta = 0.4
    agg.gamma = 0.3

    agg.word_sim = MagicMock()
    agg.sent_sim = MagicMock()
    agg.doc_sim  = MagicMock()

    return agg


def test_weights_sum_to_one():
    with pytest.raises(AssertionError):
        MGCSAggregator(alpha=0.5, beta=0.5, gamma=0.5)


def test_sfinal_formula(mock_aggregator):
    mock_aggregator.word_sim.compute.return_value = 0.8
    mock_aggregator.sent_sim.compute.return_value = 0.6
    mock_aggregator.doc_sim.compute.return_value  = 0.7

    result = mock_aggregator.compute("generated text", "reference text")

    expected = 0.3 * 0.8 + 0.4 * 0.6 + 0.3 * 0.7
    assert isinstance(result, MGCSScores)
    assert abs(result.s_final - expected) < 1e-6


def test_identical_texts_returns_high_score(mock_aggregator):
    mock_aggregator.word_sim.compute.return_value = 1.0
    mock_aggregator.sent_sim.compute.return_value = 1.0
    mock_aggregator.doc_sim.compute.return_value  = 1.0

    result = mock_aggregator.compute("same text", "same text")
    assert result.s_final == pytest.approx(1.0)


def test_score_fields_present(mock_aggregator):
    mock_aggregator.word_sim.compute.return_value = 0.5
    mock_aggregator.sent_sim.compute.return_value = 0.5
    mock_aggregator.doc_sim.compute.return_value  = 0.5

    result = mock_aggregator.compute("a", "b")
    assert hasattr(result, "s_word")
    assert hasattr(result, "s_sent")
    assert hasattr(result, "s_doc")
    assert hasattr(result, "s_final")
