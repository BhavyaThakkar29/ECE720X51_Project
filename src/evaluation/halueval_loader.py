# src/evaluation/halueval_loader.py
"""
Loads HaluEval dataset splits from HuggingFace.
Returns list of dicts with keys: generated, reference, label (0=faithful, 1=hallucinated).

Available splits: 'qa', 'summarization', 'dialogue'
"""

from datasets import load_dataset
from typing import List, Dict


def load_halueval(split_name: str = "qa", max_samples: int = None) -> List[Dict]:
    """
    Load a HaluEval split.

    Args:
        split_name: One of 'qa', 'summarization', 'dialogue'.
        max_samples: Cap the number of samples (useful for quick testing).

    Returns:
        List of dicts: [{generated, reference, label}, ...]
    """
    dataset_map = {
        "qa": ("pminervini/HaluEval", "qa_samples"),
        "summarization": ("pminervini/HaluEval", "summarization_samples"),
        "dialogue": ("pminervini/HaluEval", "dialogue_samples"),
    }
    if split_name not in dataset_map:
        raise ValueError(f"split_name must be one of {list(dataset_map.keys())}")

    repo, config = dataset_map[split_name]
    dataset = load_dataset(repo, config, split="data")

    records = []
    for row in dataset:
        if split_name == "qa":
            records.append({
                "generated": row.get("hallucinated_answer", ""),
                "reference": row.get("right_answer", ""),
                "label": 1,   # hallucinated answer
            })
            records.append({
                "generated": row.get("right_answer", ""),
                "reference": row.get("right_answer", ""),
                "label": 0,   # faithful answer (self-reference)
            })
        elif split_name == "summarization":
            records.append({
                "generated": row.get("hallucinated_summary", ""),
                "reference": row.get("right_summary", ""),
                "label": 1,
            })
            records.append({
                "generated": row.get("right_summary", ""),
                "reference": row.get("right_summary", ""),
                "label": 0,
            })
        elif split_name == "dialogue":
            records.append({
                "generated": row.get("hallucinated_response", ""),
                "reference": row.get("right_response", ""),
                "label": 1,
            })
            records.append({
                "generated": row.get("right_response", ""),
                "reference": row.get("right_response", ""),
                "label": 0,
            })

        if max_samples and len(records) >= max_samples:
            break

    return records[:max_samples] if max_samples else records
