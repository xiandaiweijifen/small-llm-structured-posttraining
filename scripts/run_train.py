"""Training entrypoint placeholder for phase-1 experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase-1 training entrypoint placeholder.")
    parser.add_argument(
        "--config",
        default="configs/train/sft_phase1.yaml",
        help="Path to training config yaml.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    experiment_name = config.get("experiment_name", "unknown_experiment")
    method = config.get("training", {}).get("method", "unknown")
    base_model = config.get("model", {}).get("base_model", "TODO")
    print("Training is not implemented in the local environment yet.")
    print(f"Experiment: {experiment_name}")
    print(f"Method: {method}")
    print(f"Base model: {base_model}")
    print("When training is ready, run this stage in Jupyter with the same config.")


if __name__ == "__main__":
    main()
