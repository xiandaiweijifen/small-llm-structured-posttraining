"""Build a filtered phase-1 candidate dataset from mapped external data."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.filtering import (
    filter_candidate_records,
    sample_balanced_candidates,
    summarize_candidate_build,
)
from src.data.io import dump_jsonl, load_jsonl
from src.data.profiling import profile_dataset
from src.evaluation.reporting import write_json_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the phase-1 candidate dataset.")
    parser.add_argument(
        "--console-input",
        default="data/samples/console_ai_relabeled.jsonl",
        help="Path to relabeled Console-AI dataset.",
    )
    parser.add_argument(
        "--kameronb-input",
        default="data/samples/kameronb_relabeled.jsonl",
        help="Path to relabeled KameronB dataset.",
    )
    parser.add_argument(
        "--output",
        default="data/samples/phase1_candidate.jsonl",
        help="Path to output candidate dataset.",
    )
    parser.add_argument(
        "--profile-output",
        default="data/samples/phase1_candidate_profile.json",
        help="Path to candidate profile json.",
    )
    parser.add_argument(
        "--summary-output",
        default="data/samples/phase1_candidate_build_summary.json",
        help="Path to candidate build summary json.",
    )
    parser.add_argument("--shuffle-seed", type=int, default=42, help="Random seed for subsampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console_records = load_jsonl(args.console_input)
    kameronb_records = load_jsonl(args.kameronb_input)
    input_counts = {
        "console_ai_it_helpdesk_tickets": len(console_records),
        "kameronb_it_callcenter_tickets": len(kameronb_records),
    }

    combined = console_records + kameronb_records
    filtered = filter_candidate_records(combined)
    sampled = sample_balanced_candidates(
        filtered,
        max_per_source={
            "console_ai_it_helpdesk_tickets": 500,
            "kameronb_it_callcenter_tickets": 2500,
        },
        max_per_category_per_source=500,
        shuffle_seed=args.shuffle_seed,
    )

    dump_jsonl(args.output, sampled)
    write_json_report(args.profile_output, profile_dataset(sampled))
    write_json_report(
        args.summary_output,
        summarize_candidate_build(input_counts, filtered, sampled),
    )
    print(f"Candidate dataset written to {args.output}")
    print(f"Candidate profile written to {args.profile_output}")
    print(f"Candidate build summary written to {args.summary_output}")


if __name__ == "__main__":
    main()
