# src/similarity/word_similarity.py
"""
Word-level cosine similarity between two texts.

Strategy:
  - Embed both texts at word level.
  - For each word in the generated text, find its max cosine similarity
    to any word in the reference text (soft alignment / greedy matching).
  - Average those max similarities → S_word.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.embeddings.word_embedder import WordEmbedder


class WordSimilarity:
    def __init__(self, embedder: WordEmbedder = None, model_name: str = "bert-base-uncased"):
        self.embedder = embedder or WordEmbedder(model_name=model_name)

    def compute(self, generated: str, reference: str) -> float:
        """
        Compute soft word-level cosine similarity.

        Args:
            generated: Model-generated text.
            reference:  Ground-truth reference text.

        Returns:
            S_word in [−1, 1], typically [0, 1] for natural language.
        """
        gen_embs = self.embedder.embed(generated)   # (n, d)
        ref_embs = self.embedder.embed(reference)   # (m, d)

        if gen_embs.size == 0 or ref_embs.size == 0:
            return 0.0

        # Pairwise cosine similarity matrix: (n, m)
        sim_matrix = cosine_similarity(gen_embs, ref_embs)

        # Greedy max alignment: for each generated word, best match in reference
        max_sims = sim_matrix.max(axis=1)   # (n,)
        return float(max_sims.mean())
