# src/similarity/document_similarity.py
"""
Document-level cosine similarity between two full texts.
Each text is encoded as a single embedding; similarity is a single scalar.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.embeddings.document_embedder import DocumentEmbedder


class DocumentSimilarity:
    def __init__(self, embedder: DocumentEmbedder = None, model_name: str = "all-mpnet-base-v2"):
        self.embedder = embedder or DocumentEmbedder(model_name=model_name)

    def compute(self, generated: str, reference: str) -> float:
        """
        Compute document-level cosine similarity.

        Args:
            generated: Model-generated text.
            reference:  Ground-truth reference text.

        Returns:
            S_doc in [−1, 1].
        """
        gen_emb = self.embedder.embed(generated).reshape(1, -1)   # (1, d)
        ref_emb = self.embedder.embed(reference).reshape(1, -1)   # (1, d)
        score = cosine_similarity(gen_emb, ref_emb)[0][0]
        return float(score)
