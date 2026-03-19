"""Field-level error analysis helpers."""

from __future__ import annotations

from collections import Counter
from typing import Any

from src.evaluation.metrics import flatten_structure


def analyze_field_errors(
    gold_records: list[dict[str, Any]],
    pred_records: list[dict[str, Any]],
) -> dict[str, Any]:
    predictions_by_id = {record["sample_id"]: record for record in pred_records}

    field_totals: Counter[str] = Counter()
    field_matches: Counter[str] = Counter()
    field_mismatches: Counter[str] = Counter()
    mismatch_examples: dict[str, list[dict[str, Any]]] = {}

    for gold_record in gold_records:
        sample_id = gold_record["sample_id"]
        gold_flat = flatten_structure(gold_record["target_json"])
        pred_record = predictions_by_id.get(sample_id, {})
        pred_json = pred_record.get("prediction_json") or {}
        pred_flat = flatten_structure(pred_json) if isinstance(pred_json, dict) else {}

        for field_name, gold_value in gold_flat.items():
            field_totals[field_name] += 1
            pred_value = pred_flat.get(field_name)
            if pred_value == gold_value:
                field_matches[field_name] += 1
            else:
                field_mismatches[field_name] += 1
                mismatch_examples.setdefault(field_name, [])
                if len(mismatch_examples[field_name]) < 5:
                    mismatch_examples[field_name].append(
                        {
                            "sample_id": sample_id,
                            "gold": gold_value,
                            "pred": pred_value,
                        }
                    )

    per_field = {}
    for field_name in sorted(field_totals):
        total = field_totals[field_name]
        per_field[field_name] = {
            "total": total,
            "match_count": field_matches[field_name],
            "mismatch_count": field_mismatches[field_name],
            "match_rate": field_matches[field_name] / total if total else 0.0,
            "examples": mismatch_examples.get(field_name, []),
        }

    return {
        "num_samples": len(gold_records),
        "num_fields": len(per_field),
        "per_field": per_field,
        "worst_fields": worst_fields(per_field, top_k=15),
    }


def worst_fields(per_field: dict[str, dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    ranked = sorted(
        (
            {
                "field_name": field_name,
                "match_rate": stats["match_rate"],
                "mismatch_count": stats["mismatch_count"],
                "total": stats["total"],
            }
            for field_name, stats in per_field.items()
        ),
        key=lambda item: (item["match_rate"], -item["mismatch_count"], item["field_name"]),
    )
    return ranked[:top_k]
