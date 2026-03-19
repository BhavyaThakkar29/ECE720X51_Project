# src/embeddings/sentence_embedder.py
"""
Sentence-level embedder using Sentence-Transformers.
Splits input text into sentences and returns one embedding per sentence.
"""

import numpy as np
import nltk
from sentence_transformers import SentenceTransformer

nltk.download("punkt_tab", quiet=True)


class SentenceEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        """
        Splits text into sentences using NLTK, then encodes each sentence.
        Returns array of shape (num_sentences, embedding_dim).
        """
        sentences = nltk.sent_tokenize(text)
        if not sentences:
            sentences = [text]
        embeddings = self.model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
        return embeddings  # (num_sentences, dim)

    def embed_single(self, sentence: str) -> np.ndarray:
        """Embed a single pre-split sentence."""
        return self.model.encode([sentence], convert_to_numpy=True, show_progress_bar=False)[0]
