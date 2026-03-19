"""Split the phase-1 candidate dataset into train / val / test files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_builder import validate_records, write_dataset_splits
from src.data.io import load_jsonl
from src.data.splitting import assign_stratified_splits, summarize_phase1_splits
from src.evaluation.reporting import write_json_report
from src.schemas.registry import get_schema
from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build phase-1 train/val/test splits.")
    parser.add_argument(
        "--input",
        default="data/samples/phase1_candidate.jsonl",
        help="Path to candidate dataset jsonl.",
    )
    parser.add_argument(
        "--config",
        default="configs/dataset/phase1_ticket.yaml",
        help="Path to dataset config yaml.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/samples",
        help="Output directory for split files.",
    )
    parser.add_argument("--shuffle-seed", type=int, default=42, help="Random seed for split assignment.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    records = load_jsonl(args.input)
    schema = get_schema(config["schema_name"])
    validate_records(records, schema=schema, task_name=config["task_name"], schema_name=config["schema_name"])

    split_records = assign_stratified_splits(
        records=records,
        split_config=config["splits"],
        shuffle_seed=args.shuffle_seed,
    )
    write_dataset_splits(args.output_dir, split_records)

    summary = summarize_phase1_splits(split_records)
    summary_path = Path(args.output_dir) / "phase1_split_summary.json"
    write_json_report(summary_path, summary)
    print(f"Phase-1 splits written to {args.output_dir}")
    print(f"Split summary written to {summary_path}")


if __name__ == "__main__":
    main()
