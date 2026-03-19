# src/similarity/__init__.py
from .word_similarity import WordSimilarity
from .sentence_similarity import SentenceSimilarity
from .document_similarity import DocumentSimilarity

__all__ = ["WordSimilarity", "SentenceSimilarity", "DocumentSimilarity"]
