"""Run local inference pipeline and write prediction jsonl."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import dump_jsonl, load_jsonl
from src.inference.backends import generate_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for structured-output samples.")
    parser.add_argument("--input", required=True, help="Path to dataset split jsonl.")
    parser.add_argument("--output", required=True, help="Path to prediction jsonl.")
    parser.add_argument(
        "--backend",
        default="oracle",
        choices=("oracle", "empty_json", "invalid_json"),
        help="Inference backend for local pipeline checks.",
    )
    parser.add_argument(
        "--model-name",
        default="local-backend",
        help="Model name recorded in metadata.",
    )
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Experiment id recorded in metadata. Defaults to timestamp-based id.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.input)
    experiment_id = args.experiment_id or f"{args.backend}-{int(time.time())}"

    predictions = []
    for record in records:
        prediction_text, prediction_json = generate_prediction(record, backend=args.backend)
        predictions.append(
            {
                "sample_id": record["sample_id"],
                "prediction_text": prediction_text,
                "prediction_json": prediction_json,
                "metadata": {
                    "model_name": args.model_name,
                    "experiment_id": experiment_id,
                },
            }
        )

    dump_jsonl(args.output, predictions)
    print(f"Predictions written to {args.output}")


if __name__ == "__main__":
    main()
