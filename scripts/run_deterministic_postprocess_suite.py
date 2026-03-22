from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import dump_jsonl, load_jsonl
from src.evaluation.field_analysis import analyze_field_errors
from src.evaluation.metrics import evaluate_sample, summarize_results, try_parse_prediction_text
from src.evaluation.reporting import group_sample_results, write_json_report
from src.inference.repair import repair_prediction
from src.schemas.registry import get_schema

SCHEMA_NAME = "ticket_schema_v1_reduced"
SOURCE_EXPERIMENT = "qwen25_3b_stage7_canonical_action_component_structure_then_semantics_stage2_epoch9"
SUMMARY_PATH = PROJECT_ROOT / "docs" / "results" / "deterministic_postprocess_batch_summary.md"

ACTION_PREFIX_BY_CATEGORY = {
    "task": "Handle request",
    "bug": "Investigate issue",
    "feature": "Review and plan request",
    "incident": "Investigate and mitigate incident",
    "question": "Answer and clarify",
}

RUN_PRESETS = [
    "action_refresh",
    "component_from_name_strict",
    "component_from_name_majority",
    "action_component_majority",
]


def build_output_paths(experiment_name: str) -> dict[str, Path]:
    return {
        "prediction_path": PROJECT_ROOT / "results" / "predictions" / f"{experiment_name}_test.jsonl",
        "repaired_prediction_path": PROJECT_ROOT / "results" / "predictions" / f"{experiment_name}_test_repaired.jsonl",
        "raw_report_path": PROJECT_ROOT / "results" / "metrics" / f"{experiment_name}_test_report.json",
        "repaired_report_path": PROJECT_ROOT / "results" / "metrics" / f"{experiment_name}_test_repaired_report.json",
        "raw_field_path": PROJECT_ROOT / "results" / "metrics" / f"{experiment_name}_field_analysis.json",
        "repaired_field_path": PROJECT_ROOT / "results" / "metrics" / f"{experiment_name}_test_repaired_field_analysis.json",
    }


def build_component_maps(records: list[dict]) -> tuple[dict[str, str], dict[str, str]]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        target = record["target_json"]
        name = target["affected_systems"][0]["name"]
        component = target["affected_systems"][0]["component"]
        counts[name][component] += 1
    strict = {name: next(iter(counter)) for name, counter in counts.items() if len(counter) == 1}
    majority = {name: counter.most_common(1)[0][0] for name, counter in counts.items()}
    return strict, majority


def canonicalize_target_record(record: dict, component_map: dict[str, str]) -> dict:
    updated = deepcopy(record)
    target = updated["target_json"]
    name = target["affected_systems"][0]["name"]
    if name in component_map:
        target["affected_systems"][0]["component"] = component_map[name]
    category = target["category"]
    prefix = ACTION_PREFIX_BY_CATEGORY[category]
    target["actions_requested"][0]["action"] = f"{prefix}: {target['summary']}"
    return updated


def refresh_action_from_category(prediction_json: dict) -> None:
    category = prediction_json.get("category")
    summary = prediction_json.get("summary")
    if not isinstance(category, str) or not isinstance(summary, str):
        return
    prefix = ACTION_PREFIX_BY_CATEGORY.get(category)
    if prefix is None:
        return
    actions = prediction_json.get("actions_requested")
    if not isinstance(actions, list) or not actions or not isinstance(actions[0], dict):
        return
    actions[0]["action"] = f"{prefix}: {summary}"


def update_component_from_name(prediction_json: dict, component_map: dict[str, str]) -> None:
    systems = prediction_json.get("affected_systems")
    if not isinstance(systems, list) or not systems or not isinstance(systems[0], dict):
        return
    name = systems[0].get("name")
    if not isinstance(name, str):
        return
    component = component_map.get(name)
    if component is not None:
        systems[0]["component"] = component


def apply_variant(prediction_json: dict | None, preset_name: str, strict_map: dict[str, str], majority_map: dict[str, str]) -> dict | None:
    if not isinstance(prediction_json, dict):
        return prediction_json

    updated = deepcopy(prediction_json)
    if preset_name == "action_refresh":
        refresh_action_from_category(updated)
    elif preset_name == "component_from_name_strict":
        update_component_from_name(updated, strict_map)
    elif preset_name == "component_from_name_majority":
        update_component_from_name(updated, majority_map)
    elif preset_name == "action_component_majority":
        refresh_action_from_category(updated)
        update_component_from_name(updated, majority_map)
    return updated


def sample_eval_dicts(gold_records: list[dict], pred_records: list[dict], schema: dict):
    predictions_by_id = {record["sample_id"]: record for record in pred_records}
    results = []
    for gold_record in gold_records:
        pred_record = predictions_by_id.get(gold_record["sample_id"], {})
        sample_eval = evaluate_sample(
            sample_id=gold_record["sample_id"],
            prediction_text=pred_record.get("prediction_text"),
            prediction_json=pred_record.get("prediction_json"),
            target_json=gold_record["target_json"],
            schema=schema,
        )
        results.append(
            {
                **sample_eval.__dict__,
                "schema_name": gold_record["schema_name"],
                "complexity_bucket": gold_record.get("complexity_bucket", "unknown"),
            }
        )
    return results


def summarize_from_dicts(sample_results: list[dict]):
    from src.evaluation.metrics import SampleEvaluation

    return summarize_results(
        [
            SampleEvaluation(
                sample_id=item["sample_id"],
                valid_json=item["valid_json"],
                schema_compliant=item["schema_compliant"],
                field_exact_match=item["field_exact_match"],
                exact_match=item["exact_match"],
                primary_error=item["primary_error"],
            )
            for item in sample_results
        ]
    )


def run() -> None:
    print("project_root =", PROJECT_ROOT)
    print("source_experiment =", SOURCE_EXPERIMENT)
    print("scheduled_presets =", RUN_PRESETS)

    schema = get_schema(SCHEMA_NAME)
    train_records = load_jsonl(PROJECT_ROOT / "data" / "reduced" / "phase1_train_reduced.jsonl")
    test_records_raw = load_jsonl(PROJECT_ROOT / "data" / "reduced" / "phase1_test_reduced.jsonl")
    source_predictions = load_jsonl(PROJECT_ROOT / "results" / "predictions" / f"{SOURCE_EXPERIMENT}_test.jsonl")

    strict_map, majority_map = build_component_maps(train_records)
    test_records = [canonicalize_target_record(record, majority_map) for record in test_records_raw]

    print("strict_component_names =", len(strict_map))
    print("majority_component_names =", len(majority_map))
    print("test_samples =", len(test_records))

    batch_run_results = []

    for preset_name in RUN_PRESETS:
        experiment_name = f"qwen25_3b_stage8_{preset_name}"
        paths = build_output_paths(experiment_name)
        print("\n" + "=" * 80)
        print("running preset =", preset_name)
        print("=" * 80)

        predictions = []
        for record in source_predictions:
            prediction_json = apply_variant(record.get("prediction_json"), preset_name, strict_map, majority_map)
            predictions.append(
                {
                    **record,
                    "prediction_json": prediction_json,
                    "metadata": {
                        **record.get("metadata", {}),
                        "source_experiment": SOURCE_EXPERIMENT,
                        "postprocess_preset": preset_name,
                    },
                }
            )
        dump_jsonl(paths["prediction_path"], predictions)

        raw_sample_results = sample_eval_dicts(test_records, predictions, schema)
        raw_report = {
            "summary": summarize_from_dicts(raw_sample_results),
            "grouped_summary": {
                "by_complexity_bucket": {
                    name: summarize_from_dicts(items)
                    for name, items in group_sample_results(raw_sample_results, "complexity_bucket").items()
                }
            },
            "per_sample": raw_sample_results,
        }
        write_json_report(paths["raw_report_path"], raw_report)
        write_json_report(paths["raw_field_path"], analyze_field_errors(test_records, predictions))

        repaired_predictions = []
        for record in predictions:
            prediction_json = record.get("prediction_json")
            prediction_text = record.get("prediction_text")
            if prediction_json is None and isinstance(prediction_text, str):
                _, prediction_json = try_parse_prediction_text(prediction_text)
            repaired_json, repaired = repair_prediction(prediction_json, schema)
            repaired_predictions.append(
                {
                    **record,
                    "prediction_json": repaired_json,
                    "metadata": {**record.get("metadata", {}), "repaired": repaired},
                }
            )
        dump_jsonl(paths["repaired_prediction_path"], repaired_predictions)
        repaired_sample_results = sample_eval_dicts(test_records, repaired_predictions, schema)
        repaired_report = {"summary": summarize_from_dicts(repaired_sample_results), "per_sample": repaired_sample_results}
        write_json_report(paths["repaired_report_path"], repaired_report)
        write_json_report(paths["repaired_field_path"], analyze_field_errors(test_records, repaired_predictions))

        batch_run_results.append(
            {
                "preset_name": preset_name,
                "experiment_name": experiment_name,
                "raw_summary": raw_report["summary"],
                "repaired_summary": repaired_report["summary"],
            }
        )

    leaderboard = sorted(
        [
            {
                "preset_name": item["preset_name"],
                "experiment_name": item["experiment_name"],
                "field_exact_match": item["raw_summary"]["field_exact_match"],
                "end_to_end_exact_match": item["raw_summary"]["end_to_end_exact_match"],
            }
            for item in batch_run_results
        ],
        key=lambda item: item["end_to_end_exact_match"],
        reverse=True,
    )

    lines = [
        "# Deterministic Postprocess Batch Summary",
        "",
        f"- source experiment: `{SOURCE_EXPERIMENT}`",
        f"- strict component names: `{len(strict_map)}`",
        f"- majority component names: `{len(majority_map)}`",
        "",
        "## Leaderboard",
        "",
    ]
    for item in leaderboard:
        lines.extend(
            [
                f"### {item['preset_name']}",
                "",
                f"- experiment: `{item['experiment_name']}`",
                f"- field exact match: `{item['field_exact_match']:.4f}`",
                f"- end-to-end exact match: `{item['end_to_end_exact_match']:.4f}`",
                "",
            ]
        )
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("summary_path =", SUMMARY_PATH)
    print("leaderboard =")
    for item in leaderboard:
        print(item)


if __name__ == "__main__":
    run()
