# src/similarity/sentence_similarity.py
"""
Sentence-level cosine similarity between two texts.

Strategy:
  - Split both texts into sentences and embed each.
  - Use soft alignment (greedy max) across sentence pairs → S_sent.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.embeddings.sentence_embedder import SentenceEmbedder


class SentenceSimilarity:
    def __init__(self, embedder: SentenceEmbedder = None, model_name: str = "all-MiniLM-L6-v2"):
        self.embedder = embedder or SentenceEmbedder(model_name=model_name)

    def compute(self, generated: str, reference: str) -> float:
        """
        Compute soft sentence-level cosine similarity.

        Args:
            generated: Model-generated text.
            reference:  Ground-truth reference text.

        Returns:
            S_sent in [−1, 1].
        """
        gen_embs = self.embedder.embed(generated)   # (n_sents, d)
        ref_embs = self.embedder.embed(reference)   # (m_sents, d)

        if gen_embs.ndim == 1:
            gen_embs = gen_embs[np.newaxis, :]
        if ref_embs.ndim == 1:
            ref_embs = ref_embs[np.newaxis, :]

        sim_matrix = cosine_similarity(gen_embs, ref_embs)   # (n, m)
        max_sims = sim_matrix.max(axis=1)
        return float(max_sims.mean())
