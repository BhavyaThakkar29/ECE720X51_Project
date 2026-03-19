# src/evaluation/evaluator.py
"""
End-to-end evaluation pipeline.
Loads a dataset, runs MGCS scoring, and reports metrics.
"""

import json
import logging
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict

from src.aggregation.aggregator import MGCSAggregator
from src.evaluation.metrics import compute_metrics, summarize_scores

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, aggregator: MGCSAggregator, output_dir: str = "results"):
        self.aggregator = aggregator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, records: List[Dict], dataset_name: str = "dataset") -> dict:
        """
        Run MGCS evaluation over a list of records.

        Args:
            records:      List of {generated, reference, label} dicts.
            dataset_name: Used for naming result files.

        Returns:
            Dictionary of evaluation metrics.
        """
        all_scores = []
        labels = []

        for record in tqdm(records, desc=f"Scoring [{dataset_name}]"):
            scores = self.aggregator.compute(record["generated"], record["reference"])
            all_scores.append({
                "s_word": scores.s_word,
                "s_sent": scores.s_sent,
                "s_doc": scores.s_doc,
                "s_final": scores.s_final,
                "label": record["label"],
            })
            labels.append(record["label"])

        s_finals = [s["s_final"] for s in all_scores]
        metrics = compute_metrics(s_finals, labels)
        mean_scores = summarize_scores(all_scores)

        result = {
            "dataset": dataset_name,
            "weights": {"alpha": self.aggregator.alpha, "beta": self.aggregator.beta, "gamma": self.aggregator.gamma},
            "mean_scores": mean_scores,
            "metrics": metrics,
        }

        # Save detailed scores and summary
        out_path = self.output_dir / f"{dataset_name}_scores.json"
        with open(out_path, "w") as f:
            json.dump(all_scores, f, indent=2)

        summary_path = self.output_dir / f"{dataset_name}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(result, f, indent=2)

        logger.info(f"Results saved to {self.output_dir}")
        return result
