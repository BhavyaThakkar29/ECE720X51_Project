# src/embeddings/word_embedder.py
"""
Word-level embedder using BERT hidden states.
Returns token embeddings for each word in the input text.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel


class WordEmbedder:
    def __init__(self, model_name: str = "bert-base-uncased", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def embed(self, text: str) -> np.ndarray:
        """
        Returns mean-pooled token embeddings of shape (num_tokens, hidden_dim).
        Subword tokens are averaged back to whole-word embeddings.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
        )
        offset_mapping = inputs.pop("offset_mapping")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Last hidden state: (1, seq_len, hidden_dim)
        token_embeddings = outputs.last_hidden_state.squeeze(0).cpu().numpy()

        # Average subword tokens per word using offset mapping
        word_embeddings = self._aggregate_subwords(
            token_embeddings, offset_mapping.squeeze(0).numpy(), text
        )
        return word_embeddings  # (num_words, hidden_dim)

    def _aggregate_subwords(
        self, token_embeddings: np.ndarray, offset_mapping: np.ndarray, text: str
    ) -> np.ndarray:
        """Merge subword token embeddings into whole-word embeddings."""
        word_embeddings = []
        current_word_vecs = []
        current_start = -1

        for i, (start, end) in enumerate(offset_mapping):
            if start == 0 and end == 0:
                # Special tokens ([CLS], [SEP]) — skip
                if current_word_vecs:
                    word_embeddings.append(np.mean(current_word_vecs, axis=0))
                    current_word_vecs = []
                    current_start = -1
                continue
            if text[start:end].startswith(" ") or current_start == -1:
                if current_word_vecs:
                    word_embeddings.append(np.mean(current_word_vecs, axis=0))
                current_word_vecs = [token_embeddings[i]]
                current_start = start
            else:
                current_word_vecs.append(token_embeddings[i])

        if current_word_vecs:
            word_embeddings.append(np.mean(current_word_vecs, axis=0))

        return np.array(word_embeddings)  # (num_words, hidden_dim)
