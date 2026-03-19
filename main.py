# main.py
"""
Entry point for Multi-Granularity Cosine Similarity (MGCS) hallucination evaluation.

Usage:
    python main.py --config configs/config.yaml
    python main.py --dataset halueval_qa --alpha 0.3 --beta 0.4 --gamma 0.3 --max_samples 200
"""

import argparse
import yaml
import json

from src.aggregation.aggregator import MGCSAggregator
from src.evaluation.halueval_loader import load_halueval
from src.evaluation.evaluator import Evaluator
from src.utils.logger import get_logger

logger = get_logger("main")


def parse_args():
    parser = argparse.ArgumentParser(description="MGCS Hallucination Evaluator")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Override dataset: halueval_qa | halueval_summarization | halueval_dialogue")
    parser.add_argument("--alpha", type=float, default=None, help="Word-level weight")
    parser.add_argument("--beta", type=float, default=None, help="Sentence-level weight")
    parser.add_argument("--gamma", type=float, default=None, help="Document-level weight")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Allow CLI overrides
    alpha = args.alpha or cfg["weights"]["alpha"]
    beta  = args.beta  or cfg["weights"]["beta"]
    gamma = args.gamma or cfg["weights"]["gamma"]
    output_dir = args.output_dir or cfg["evaluation"]["output_dir"]

    logger.info(f"Weights → α={alpha}, β={beta}, γ={gamma}")

    aggregator = MGCSAggregator(
        alpha=alpha, beta=beta, gamma=gamma,
        word_model=cfg["models"]["word"],
        sentence_model=cfg["models"]["sentence"],
        document_model=cfg["models"]["document"],
    )
    evaluator = Evaluator(aggregator, output_dir=output_dir)

    datasets_to_run = (
        [args.dataset] if args.dataset
        else [d["name"] for d in cfg["datasets"]]
    )

    all_results = {}
    for dataset_name in datasets_to_run:
        logger.info(f"Loading dataset: {dataset_name}")

        # Map config dataset names to loader calls
        split_map = {
            "halueval_qa": "qa",
            "halueval_summarization": "summarization",
            "halueval_dialogue": "dialogue",
        }
        if dataset_name in split_map:
            records = load_halueval(
                split_name=split_map[dataset_name],
                max_samples=args.max_samples or cfg["evaluation"].get("max_samples"),
            )
        else:
            logger.warning(f"Unknown dataset '{dataset_name}', skipping.")
            continue

        logger.info(f"Running evaluation on {len(records)} samples...")
        result = evaluator.run(records, dataset_name=dataset_name)
        all_results[dataset_name] = result

        logger.info(f"\n{'='*50}")
        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"Mean Scores: {result['mean_scores']}")
        logger.info(f"Metrics:     {result['metrics']}")
        logger.info(f"{'='*50}\n")

    # Save combined results
    combined_path = f"{output_dir}/all_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"All results saved to {combined_path}")


if __name__ == "__main__":
    main()
