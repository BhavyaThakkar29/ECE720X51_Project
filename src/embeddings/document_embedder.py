# src/embeddings/document_embedder.py
"""
Document-level embedder using Sentence-Transformers mean pooling.
Encodes the full text as a single fixed-size embedding.
"""

import numpy as np
from sentence_transformers import SentenceTransformer


class DocumentEmbedder:
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        """
        Returns a single document embedding of shape (embedding_dim,).
        Long texts are automatically truncated to the model's max sequence length.
        """
        embedding = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return embedding  # (dim,)
