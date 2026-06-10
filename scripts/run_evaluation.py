"""CLI entry point for RAGAS evaluation.

Usage:
    python -m scripts.run_evaluation
    python -m scripts.run_evaluation --queries path/to/queries.json --output results.json --verbose
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Force offline mode for Hugging Face hub (embeddings and rerankers) to use cached local files
os.environ["HF_HUB_OFFLINE"] = "1"

from src.evaluation.ragas_eval import load_test_queries, run_evaluation


def main() -> None:
    """Run RAGAS evaluation from the command line."""
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation on INPPLC RAG pipeline",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=None,
        help="Path to test queries JSON (default: bundled queries)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation_results.json"),
        help="Path to save results JSON (default: evaluation_results.json)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-query progress",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="all",
        choices=["all", "fr", "ar", "en"],
        help="Language to evaluate on for benchmark queries (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of queries to evaluate (e.g., 10)",
    )

    args = parser.parse_args()

    queries = None
    if args.queries:
        queries = load_test_queries(args.queries, lang=args.lang)

    run_evaluation(
        queries=queries,
        output_path=args.output,
        verbose=args.verbose,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
