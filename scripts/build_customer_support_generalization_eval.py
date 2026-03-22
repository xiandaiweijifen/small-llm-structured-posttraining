"""Build a reduced-schema external generalization eval set from customer support tickets."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.complexity import infer_complexity_bucket
from src.data.io import dump_jsonl, load_jsonl
from src.data.mappers import (
    build_action_text,
    choose_summary,
    compose_input_text,
    infer_blocking,
    normalize_priority,
    normalize_text,
)
from src.evaluation.reporting import write_json_report
from src.schemas.registry import get_schema
from src.data.validation import validate_dataset_record


SOURCE_NAME = "gorkemsevinc_customer_support_tickets"
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "exports" / SOURCE_NAME / "train.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "external_generalization" / "gorkemsevinc_customer_support_tickets_eval_reduced.jsonl"
DEFAULT_SUMMARY = PROJECT_ROOT / "data" / "external_generalization" / "gorkemsevinc_customer_support_tickets_eval_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a reduced-schema external generalization eval file from customer support tickets."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input JSONL exported from Hugging Face.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output project-format reduced-schema JSONL.")
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY), help="Summary JSON path.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=512,
        help="Maximum number of eval samples to keep. Use 0 or negative for full dataset.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling.")
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
    customer_email = normalize_text(raw_record.get("Customer Email"))
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
                "action": build_action_text(category, summary, combined_text),
                "owner": None,
                "deadline": None,
            }
        ],
        "constraints": {
            "environment": None,
            "blocking": infer_blocking(subject, combined_text, priority),
        },
    }

    record = {
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
            "customer_email": customer_email,
        },
    }
    return record


def sample_records(records: list[dict], max_samples: int, seed: int) -> list[dict]:
    if max_samples <= 0 or len(records) <= max_samples:
        return records
    rng = random.Random(seed)
    sampled = list(records)
    rng.shuffle(sampled)
    return sampled[:max_samples]


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary_output)

    raw_records = load_jsonl(input_path)
    mapped = [build_project_record(idx, row) for idx, row in enumerate(raw_records, start=1)]
    sampled = sample_records(mapped, int(args.max_samples), int(args.seed))

    schema = get_schema("ticket_schema_v1_reduced")
    for record in sampled:
        validate_dataset_record(
            record=record,
            schema=schema,
            expected_task_name="ticket_structured_output",
            expected_schema_name="ticket_schema_v1_reduced",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dump_jsonl(output_path, sampled)

    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "raw_count": len(raw_records),
        "mapped_count": len(mapped),
        "sampled_count": len(sampled),
        "max_samples": int(args.max_samples),
        "seed": int(args.seed),
        "complexity_counts": {
            bucket: sum(1 for row in sampled if row["complexity_bucket"] == bucket)
            for bucket in sorted({row["complexity_bucket"] for row in sampled})
        },
        "category_counts": {
            category: sum(1 for row in sampled if row["target_json"]["category"] == category)
            for category in sorted({row["target_json"]["category"] for row in sampled})
        },
        "priority_counts": {
            priority: sum(1 for row in sampled if row["target_json"]["priority"] == priority)
            for priority in sorted({row["target_json"]["priority"] for row in sampled})
        },
    }
    write_json_report(summary_path, summary)

    print("input_path =", input_path)
    print("output_path =", output_path)
    print("summary_path =", summary_path)
    print("raw_count =", len(raw_records))
    print("sampled_count =", len(sampled))


if __name__ == "__main__":
    main()
