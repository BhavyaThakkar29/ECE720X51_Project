# tests/test_similarity.py
"""
Unit tests for individual similarity modules using mock embedders.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock

from src.similarity.word_similarity import WordSimilarity
from src.similarity.sentence_similarity import SentenceSimilarity
from src.similarity.document_similarity import DocumentSimilarity


# ── Word Similarity ──────────────────────────────────────────────────────────

def test_word_similarity_identical():
    ws = WordSimilarity.__new__(WordSimilarity)
    ws.embedder = MagicMock()
    # Identical embeddings → cosine similarity = 1.0
    emb = np.array([[1.0, 0.0], [0.0, 1.0]])
    ws.embedder.embed.return_value = emb
    score = ws.compute("text", "text")
    assert score == pytest.approx(1.0, abs=1e-5)


def test_word_similarity_orthogonal():
    ws = WordSimilarity.__new__(WordSimilarity)
    ws.embedder = MagicMock()
    gen_emb = np.array([[1.0, 0.0]])
    ref_emb = np.array([[0.0, 1.0]])
    ws.embedder.embed.side_effect = [gen_emb, ref_emb]
    score = ws.compute("generated", "reference")
    assert score == pytest.approx(0.0, abs=1e-5)


def test_word_similarity_empty_returns_zero():
    ws = WordSimilarity.__new__(WordSimilarity)
    ws.embedder = MagicMock()
    ws.embedder.embed.return_value = np.array([])
    score = ws.compute("", "reference")
    assert score == 0.0


# ── Sentence Similarity ──────────────────────────────────────────────────────

def test_sentence_similarity_identical():
    ss = SentenceSimilarity.__new__(SentenceSimilarity)
    ss.embedder = MagicMock()
    emb = np.array([[1.0, 0.0, 0.0]])
    ss.embedder.embed.return_value = emb
    score = ss.compute("A sentence.", "A sentence.")
    assert score == pytest.approx(1.0, abs=1e-5)


def test_sentence_similarity_range():
    ss = SentenceSimilarity.__new__(SentenceSimilarity)
    ss.embedder = MagicMock()
    gen_emb = np.array([[0.6, 0.8]])
    ref_emb = np.array([[0.8, 0.6]])
    ss.embedder.embed.side_effect = [gen_emb, ref_emb]
    score = ss.compute("gen", "ref")
    assert -1.0 <= score <= 1.0


# ── Document Similarity ──────────────────────────────────────────────────────

def test_document_similarity_identical():
    ds = DocumentSimilarity.__new__(DocumentSimilarity)
    ds.embedder = MagicMock()
    emb = np.array([1.0, 0.0, 0.0])
    ds.embedder.embed.return_value = emb
    score = ds.compute("doc", "doc")
    assert score == pytest.approx(1.0, abs=1e-5)


def test_document_similarity_opposite():
    ds = DocumentSimilarity.__new__(DocumentSimilarity)
    ds.embedder = MagicMock()
    ds.embedder.embed.side_effect = [
        np.array([1.0, 0.0]),
        np.array([-1.0, 0.0]),
    ]
    score = ds.compute("a", "b")
    assert score == pytest.approx(-1.0, abs=1e-5)
