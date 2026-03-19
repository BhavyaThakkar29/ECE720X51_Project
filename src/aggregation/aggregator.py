# src/aggregation/aggregator.py
"""
Weighted aggregation of multi-granularity cosine similarity scores.

S_final = α·S_word + β·S_sent + γ·S_doc   (α + β + γ = 1)
"""

from dataclasses import dataclass
from src.similarity.word_similarity import WordSimilarity
from src.similarity.sentence_similarity import SentenceSimilarity
from src.similarity.document_similarity import DocumentSimilarity


@dataclass
class MGCSScores:
    s_word: float
    s_sent: float
    s_doc: float
    s_final: float
    alpha: float
    beta: float
    gamma: float

    def __repr__(self):
        return (
            f"MGCSScores(word={self.s_word:.4f}, sent={self.s_sent:.4f}, "
            f"doc={self.s_doc:.4f}, final={self.s_final:.4f})"
        )


class MGCSAggregator:
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.4,
        gamma: float = 0.3,
        word_model: str = "bert-base-uncased",
        sentence_model: str = "all-MiniLM-L6-v2",
        document_model: str = "all-mpnet-base-v2",
    ):
        assert abs(alpha + beta + gamma - 1.0) < 1e-6, "Weights must sum to 1.0"
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.word_sim = WordSimilarity(model_name=word_model)
        self.sent_sim = SentenceSimilarity(model_name=sentence_model)
        self.doc_sim = DocumentSimilarity(model_name=document_model)

    def compute(self, generated: str, reference: str) -> MGCSScores:
        """
        Compute all three similarity levels and return the weighted aggregate.

        Args:
            generated: Model-generated text.
            reference:  Ground-truth reference text.

        Returns:
            MGCSScores dataclass with per-level and final scores.
        """
        s_word = self.word_sim.compute(generated, reference)
        s_sent = self.sent_sim.compute(generated, reference)
        s_doc = self.doc_sim.compute(generated, reference)

        s_final = self.alpha * s_word + self.beta * s_sent + self.gamma * s_doc

        return MGCSScores(
            s_word=s_word,
            s_sent=s_sent,
            s_doc=s_doc,
            s_final=s_final,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
        )
