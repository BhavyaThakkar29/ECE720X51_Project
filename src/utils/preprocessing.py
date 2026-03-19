# src/utils/preprocessing.py
"""
Text preprocessing utilities shared across embedding modules.
"""

import re
import unicodedata


def clean_text(text: str) -> str:
    """
    Basic text cleaning:
    - Normalize unicode
    - Collapse whitespace
    - Strip leading/trailing spaces
    """
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def truncate_text(text: str, max_words: int = 400) -> str:
    """Truncate text to a maximum number of whitespace-split words."""
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return text
