"""Build train/val/test reduced-schema splits for external adaptation experiments."""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.complexity import infer_complexity_bucket
from src.data.io import dump_jsonl, load_jsonl
from src.data.mappers import (
    choose_summary,
    compose_input_text,
    infer_blocking,
    normalize_priority,
    normalize_text,
)
from src.data.validation import validate_dataset_record
from src.evaluation.reporting import write_json_report
from src.schemas.registry import get_schema


SOURCE_NAME = "gorkemsevinc_customer_support_tickets"
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "exports" / SOURCE_NAME / "train.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "external_generalization"
ACTION_PREFIX_BY_CATEGORY = {
    "task": "Handle request",
    "bug": "Investigate issue",
    "feature": "Review and plan request",
    "incident": "Investigate and mitigate incident",
    "question": "Answer and clarify",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build canonicalized reduced-schema train/val/test splits for external adaptation."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input JSONL exported from Hugging Face.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where train/val/test reduced JSONL files should be written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splitting.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Fraction of data used for train split.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Fraction of data used for validation split.")
    return parser.parse_args()


def map_ticket_type_to_category(ticket_type: str | None, subject: str | None, text: str | None, priority: str) -> str:
    raw = (normalize_text(ticket_type) or "").lower()
    merged = " ".join(part for part in [raw, normalize_text(subject), normalize_text(text)] if part).lower()

    if raw == "product inquiry":
        return "question"
    if raw in {"refund request", "cancellation request"}:
        return "task"
    if raw == "billing inquiry":
        return "question"
    if raw == "technical issue":
        if priority == "urgent" and any(word in merged for word in ["cannot", "not turning on", "not working", "issue persists"]):
            return "incident"
        return "bug"

    if "inquiry" in raw or "question" in merged:
        return "question"
    if "request" in raw:
        return "task"
    if "issue" in raw:
        return "bug"
    return "task"


def map_ticket_type_to_component(ticket_type: str | None, product: str | None) -> str:
    raw = (normalize_text(ticket_type) or "").lower()
    product_text = (normalize_text(product) or "").lower()

    if raw == "technical issue":
        if any(token in product_text for token in ["office", "autocad", "photoshop", "software"]):
            return "software"
        if any(token in product_text for token in ["tv", "camera", "printer", "speaker"]):
            return "hardware"
        if any(token in product_text for token in ["laptop", "xps", "macbook", "thinkpad"]):
            return "hardware"
        return "error"
    if raw == "billing inquiry":
        return "account"
    if raw == "refund request":
        return "request"
    if raw == "cancellation request":
        return "deactivation"
    if raw == "product inquiry":
        return "software"
    return "software"


def build_project_record(index: int, raw_record: dict) -> dict:
    product = normalize_text(raw_record.get("Product Purchased")) or "unknown_product"
    ticket_type = normalize_text(raw_record.get("Ticket Type"))
    subject = normalize_text(raw_record.get("Ticket Subject"))
    combined_text = normalize_text(raw_record.get("Combined Text"))
    priority = normalize_priority(raw_record.get("Ticket Priority"))

    summary = choose_summary(subject, combined_text, product)
    category = map_ticket_type_to_category(ticket_type, subject, combined_text, priority)
    component = map_ticket_type_to_component(ticket_type, product)

    target_json = {
        "summary": summary,
        "category": category,
        "priority": priority,
        "requires_followup": True,
        "affected_systems": [
            {
                "name": product,
                "component": component,
            }
        ],
        "actions_requested": [
            {
                "action": f"{ACTION_PREFIX_BY_CATEGORY[category]}: {summary}",
                "owner": None,
                "deadline": None,
            }
        ],
        "constraints": {
            "environment": None,
            "blocking": infer_blocking(subject, combined_text, priority),
        },
    }

    return {
        "sample_id": f"{SOURCE_NAME}-{index:05d}",
        "task_name": "ticket_structured_output",
        "schema_name": "ticket_schema_v1_reduced",
        "complexity_bucket": infer_complexity_bucket(target_json),
        "input_text": compose_input_text(subject, combined_text),
        "target_json": target_json,
        "metadata": {
            "source_type": "email",
            "is_synthetic": False,
            "raw_source": SOURCE_NAME,
            "source_ticket_type": ticket_type,
            "source_priority": raw_record.get("Ticket Priority"),
        },
    }


def stratified_split(
    records: list[dict],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["target_json"]["category"]].append(record)

    train_records: list[dict] = []
    val_records: list[dict] = []
    test_records: list[dict] = []

    for bucket in grouped.values():
        shuffled = list(bucket)
        rng.shuffle(shuffled)
        total = len(shuffled)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        if train_end <= 0 and total > 0:
            train_end = 1
        if val_end <= train_end and total - train_end > 1:
            val_end = train_end + 1
        train_records.extend(shuffled[:train_end])
        val_records.extend(shuffled[train_end:val_end])
        test_records.extend(shuffled[val_end:])

    rng.shuffle(train_records)
    rng.shuffle(val_records)
    rng.shuffle(test_records)
    return train_records, val_records, test_records


def split_counts(records: list[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        if key == "category":
            value = record["target_json"]["category"]
        elif key == "priority":
            value = record["target_json"]["priority"]
        else:
            value = record["complexity_bucket"]
        counts[value] += 1
    return dict(sorted(counts.items()))


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_output = output_dir / f"{SOURCE_NAME}_train_reduced.jsonl"
    val_output = output_dir / f"{SOURCE_NAME}_val_reduced.jsonl"
    test_output = output_dir / f"{SOURCE_NAME}_test_reduced.jsonl"
    summary_output = output_dir / f"{SOURCE_NAME}_split_summary.json"

    raw_records = load_jsonl(input_path)
    mapped = [build_project_record(idx, row) for idx, row in enumerate(raw_records, start=1)]
    train_records, val_records, test_records = stratified_split(
        mapped,
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )

    schema = get_schema("ticket_schema_v1_reduced")
    for split_name, split_records in [
        ("train", train_records),
        ("val", val_records),
        ("test", test_records),
    ]:
        for record in split_records:
            record["metadata"]["split"] = split_name
            validate_dataset_record(
                record=record,
                schema=schema,
                expected_task_name="ticket_structured_output",
                expected_schema_name="ticket_schema_v1_reduced",
            )

    dump_jsonl(train_output, train_records)
    dump_jsonl(val_output, val_records)
    dump_jsonl(test_output, test_records)

    summary = {
        "input_path": str(input_path),
        "train_output": str(train_output),
        "val_output": str(val_output),
        "test_output": str(test_output),
        "raw_count": len(raw_records),
        "mapped_count": len(mapped),
        "train_count": len(train_records),
        "val_count": len(val_records),
        "test_count": len(test_records),
        "seed": int(args.seed),
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "train_category_counts": split_counts(train_records, "category"),
        "val_category_counts": split_counts(val_records, "category"),
        "test_category_counts": split_counts(test_records, "category"),
        "train_priority_counts": split_counts(train_records, "priority"),
        "test_complexity_counts": split_counts(test_records, "complexity"),
    }
    write_json_report(summary_output, summary)

    print("train_output =", train_output)
    print("val_output =", val_output)
    print("test_output =", test_output)
    print("summary_output =", summary_output)
    print("train_count =", len(train_records))
    print("val_count =", len(val_records))
    print("test_count =", len(test_records))


if __name__ == "__main__":
    main()
