from __future__ import annotations

import gc
import json
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import dump_jsonl, load_jsonl
from src.evaluation.field_analysis import analyze_field_errors
from src.evaluation.metrics import evaluate_sample, summarize_results, try_parse_prediction_text
from src.evaluation.reporting import group_sample_results, write_json_report
from src.inference.batch_generate import batched_generate_texts
from src.inference.repair import repair_prediction
from src.schemas.registry import get_schema

SCHEMA_NAME = "ticket_schema_v1_reduced"
SUMMARY_PATH = PROJECT_ROOT / "docs" / "results" / "big_model_reference_summary.md"
SKIP_COMPLETED = True

MODEL_PRESETS = [
    {
        "experiment_name": "qwen25_3b_reference_canonical_prompt",
        "model_name": "Qwen/Qwen2.5-3B-Instruct",
        "batch_size": 32,
    },
    {
        "experiment_name": "qwen25_7b_reference_canonical_prompt",
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "batch_size": 24,
    },
    {
        "experiment_name": "qwen25_14b_reference_canonical_prompt",
        "model_name": "Qwen/Qwen2.5-14B-Instruct",
        "batch_size": 12,
    },
    {
        "experiment_name": "qwen25_32b_reference_canonical_prompt",
        "model_name": "Qwen/Qwen2.5-32B-Instruct",
        "batch_size": 6,
    },
]

ACTION_PREFIX_BY_CATEGORY = {
    "task": "Handle request",
    "bug": "Investigate issue",
    "feature": "Review and plan request",
    "incident": "Investigate and mitigate incident",
    "question": "Answer and clarify",
}

CATEGORY_INCIDENT_PHRASES = [
    "customer cannot use",
    "outofmemoryexception",
    "process cannot access the file",
    "cannot resolve module",
    "critical for upcoming deadlines",
    "partition unavailable",
    "broker node down",
]
URGENT_PRIORITY_PHRASES = CATEGORY_INCIDENT_PHRASES + [
    "unable to log in",
]
BLOCKING_TRUE_PHRASES = [
    "outofmemoryexception",
    "process cannot access the file",
    "partition unavailable",
    "broker node down",
]

GENERATION_KWARGS = {
    "max_new_tokens": 256,
    "do_sample": False,
    "temperature": 1.0,
    "top_p": 1.0,
}


def build_majority_component_map(records: list[dict]) -> dict[str, str]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        target = record["target_json"]
        counts[target["affected_systems"][0]["name"]][target["affected_systems"][0]["component"]] += 1
    return {name: counter.most_common(1)[0][0] for name, counter in counts.items()}


def canonicalize_target_record(record: dict, component_map: dict[str, str]) -> dict:
    updated = deepcopy(record)
    target = updated["target_json"]
    name = target["affected_systems"][0]["name"]
    if name in component_map:
        target["affected_systems"][0]["component"] = component_map[name]
    category = target["category"]
    target["actions_requested"][0]["action"] = f"{ACTION_PREFIX_BY_CATEGORY[category]}: {target['summary']}"
    return updated


def parse_input_fields(input_text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for label in ["Subject", "Description"]:
        match = re.search(rf"{label}:\s*(.*?)(?=\n[A-Z][A-Za-z ]+:|$)", input_text, flags=re.S)
        if match:
            fields[label.lower()] = match.group(1).strip()
    if "subject" not in fields:
        fields["subject"] = input_text.strip()
    if "description" not in fields:
        fields["description"] = input_text.strip()
    return fields


def build_inference_messages(record: dict) -> list[dict]:
    conventions = "\n".join(
        [
            "Output conventions:",
            "- Return a JSON object only.",
            "- Use the reduced ticket schema fields only.",
            "- Set actions_requested[0].action exactly as one of:",
            '  - "Handle request: {summary}" when category = "task"',
            '  - "Investigate issue: {summary}" when category = "bug"',
            '  - "Review and plan request: {summary}" when category = "feature"',
            '  - "Investigate and mitigate incident: {summary}" when category = "incident"',
            '  - "Answer and clarify: {summary}" when category = "question"',
            "- Use a short normalized component label for affected_systems[0].component that is consistent with affected_systems[0].name.",
        ]
    )
    return [
        {
            "role": "system",
            "content": "You are an information extraction model. Return only JSON that matches the requested schema. Do not add explanations or markdown.",
        },
        {
            "role": "user",
            "content": (
                f"Task: extract a structured record for {record['task_name']}.\n"
                f"Schema name: {SCHEMA_NAME}\n"
                f"{conventions}\n"
                "Input text:\n"
                f"{record['input_text']}"
            ),
        },
    ]


def build_output_paths(experiment_name: str) -> dict[str, Path]:
    return {
        "raw_pred": PROJECT_ROOT / "results" / "predictions" / f"{experiment_name}_test.jsonl",
        "repair_pred": PROJECT_ROOT / "results" / "predictions" / f"{experiment_name}_test_repaired.jsonl",
        "det_pred": PROJECT_ROOT / "results" / "predictions" / f"{experiment_name}_test_deterministic.jsonl",
        "lex_pred": PROJECT_ROOT / "results" / "predictions" / f"{experiment_name}_test_lexical.jsonl",
        "raw_report": PROJECT_ROOT / "results" / "metrics" / f"{experiment_name}_test_report.json",
        "repair_report": PROJECT_ROOT / "results" / "metrics" / f"{experiment_name}_test_repaired_report.json",
        "det_report": PROJECT_ROOT / "results" / "metrics" / f"{experiment_name}_test_deterministic_report.json",
        "lex_report": PROJECT_ROOT / "results" / "metrics" / f"{experiment_name}_test_lexical_report.json",
        "raw_field": PROJECT_ROOT / "results" / "metrics" / f"{experiment_name}_field_analysis.json",
        "repair_field": PROJECT_ROOT / "results" / "metrics" / f"{experiment_name}_test_repaired_field_analysis.json",
        "det_field": PROJECT_ROOT / "results" / "metrics" / f"{experiment_name}_test_deterministic_field_analysis.json",
        "lex_field": PROJECT_ROOT / "results" / "metrics" / f"{experiment_name}_test_lexical_field_analysis.json",
    }


def outputs_complete(paths: dict[str, Path]) -> bool:
    return all(path.exists() for path in paths.values())


def load_model_and_tokenizer(model_name: str):
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def cleanup_model(*objects) -> None:
    for obj in objects:
        if obj is not None:
            del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def apply_stage8_postprocess(prediction_json: dict | None, component_map: dict[str, str]) -> dict | None:
    if not isinstance(prediction_json, dict):
        return prediction_json
    updated = deepcopy(prediction_json)
    category = updated.get("category")
    summary = updated.get("summary")
    if isinstance(category, str) and isinstance(summary, str) and category in ACTION_PREFIX_BY_CATEGORY:
        actions = updated.get("actions_requested")
        if isinstance(actions, list) and actions and isinstance(actions[0], dict):
            actions[0]["action"] = f"{ACTION_PREFIX_BY_CATEGORY[category]}: {summary}"
    systems = updated.get("affected_systems")
    if isinstance(systems, list) and systems and isinstance(systems[0], dict):
        name = systems[0].get("name")
        if isinstance(name, str) and name in component_map:
            systems[0]["component"] = component_map[name]
    return updated


def apply_stage9_lexical_postprocess(prediction_json: dict | None, input_text: str) -> dict | None:
    if not isinstance(prediction_json, dict):
        return prediction_json
    updated = deepcopy(prediction_json)
    fields = parse_input_fields(input_text)
    text = f"{fields.get('subject', '')} {fields.get('description', '')}".lower()

    if any(phrase in text for phrase in CATEGORY_INCIDENT_PHRASES):
        updated["category"] = "incident"
        actions = updated.get("actions_requested")
        summary = updated.get("summary")
        if isinstance(actions, list) and actions and isinstance(actions[0], dict) and isinstance(summary, str):
            actions[0]["action"] = f"{ACTION_PREFIX_BY_CATEGORY['incident']}: {summary}"

    if any(phrase in text for phrase in URGENT_PRIORITY_PHRASES):
        updated["priority"] = "urgent"

    if any(phrase in text for phrase in BLOCKING_TRUE_PHRASES):
        constraints = updated.get("constraints")
        if isinstance(constraints, dict):
            constraints["blocking"] = True

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


def evaluate_and_write(paths: dict[str, Path], key: str, gold_records: list[dict], pred_records: list[dict], schema: dict) -> dict:
    sample_results = sample_eval_dicts(gold_records, pred_records, schema)
    report = {
        "summary": summarize_from_dicts(sample_results),
        "grouped_summary": {
            "by_complexity_bucket": {
                name: summarize_from_dicts(items)
                for name, items in group_sample_results(sample_results, "complexity_bucket").items()
            }
        },
        "per_sample": sample_results,
    }
    write_json_report(paths[f"{key}_report"], report)
    write_json_report(paths[f"{key}_field"], analyze_field_errors(gold_records, pred_records))
    return report


def run() -> None:
    print("project_root =", PROJECT_ROOT)
    print("scheduled_models =", [item["model_name"] for item in MODEL_PRESETS])
    print("skip_completed =", SKIP_COMPLETED)

    schema = get_schema(SCHEMA_NAME)
    train_records = load_jsonl(PROJECT_ROOT / "data" / "reduced" / "phase1_train_reduced.jsonl")
    test_records_raw = load_jsonl(PROJECT_ROOT / "data" / "reduced" / "phase1_test_reduced.jsonl")
    component_map = build_majority_component_map(train_records)
    test_records = [canonicalize_target_record(record, component_map) for record in test_records_raw]
    input_text_by_id = {record["sample_id"]: record["input_text"] for record in test_records_raw}

    batch_summary = []

    for preset in MODEL_PRESETS:
        experiment_name = preset["experiment_name"]
        model_name = preset["model_name"]
        batch_size = preset["batch_size"]
        paths = build_output_paths(experiment_name)

        print("\n" + "=" * 80)
        print("model =", model_name)
        print("experiment =", experiment_name)
        print("batch_size =", batch_size)
        print("=" * 80)

        if SKIP_COMPLETED and outputs_complete(paths):
            raw_report = json.loads(paths["raw_report"].read_text(encoding="utf-8"))
            repair_report = json.loads(paths["repair_report"].read_text(encoding="utf-8"))
            det_report = json.loads(paths["det_report"].read_text(encoding="utf-8"))
            lex_report = json.loads(paths["lex_report"].read_text(encoding="utf-8"))
            batch_summary.append(
                {
                    "experiment_name": experiment_name,
                    "model_name": model_name,
                    "raw_summary": raw_report["summary"],
                    "repair_summary": repair_report["summary"],
                    "det_summary": det_report["summary"],
                    "lex_summary": lex_report["summary"],
                    "status": "skipped_existing",
                }
            )
            continue

        model = None
        tokenizer = None
        try:
            model, tokenizer = load_model_and_tokenizer(model_name)
            prediction_texts = batched_generate_texts(
                model=model,
                tokenizer=tokenizer,
                records=test_records_raw,
                build_messages=build_inference_messages,
                generation_kwargs=GENERATION_KWARGS,
                batch_size=batch_size,
            )

            raw_predictions = []
            for record, prediction_text in zip(test_records_raw, prediction_texts, strict=True):
                try:
                    prediction_json = json.loads(prediction_text)
                except json.JSONDecodeError:
                    prediction_json = None
                raw_predictions.append(
                    {
                        "sample_id": record["sample_id"],
                        "prediction_text": prediction_text,
                        "prediction_json": prediction_json,
                        "metadata": {
                            "model_name": model_name,
                            "experiment_id": experiment_name,
                        },
                    }
                )
            dump_jsonl(paths["raw_pred"], raw_predictions)
            raw_report = evaluate_and_write(paths, "raw", test_records, raw_predictions, schema)

            repaired_predictions = []
            for record in raw_predictions:
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
            dump_jsonl(paths["repair_pred"], repaired_predictions)
            repair_report = evaluate_and_write(paths, "repair", test_records, repaired_predictions, schema)

            deterministic_predictions = []
            for record in raw_predictions:
                deterministic_predictions.append(
                    {
                        **record,
                        "prediction_json": apply_stage8_postprocess(record.get("prediction_json"), component_map),
                        "metadata": {**record.get("metadata", {}), "postprocess": "deterministic"},
                    }
                )
            dump_jsonl(paths["det_pred"], deterministic_predictions)
            det_report = evaluate_and_write(paths, "det", test_records, deterministic_predictions, schema)

            lexical_predictions = []
            for record in deterministic_predictions:
                sample_id = record["sample_id"]
                lexical_predictions.append(
                    {
                        **record,
                        "prediction_json": apply_stage9_lexical_postprocess(record.get("prediction_json"), input_text_by_id[sample_id]),
                        "metadata": {**record.get("metadata", {}), "postprocess": "lexical"},
                    }
                )
            dump_jsonl(paths["lex_pred"], lexical_predictions)
            lex_report = evaluate_and_write(paths, "lex", test_records, lexical_predictions, schema)

            batch_summary.append(
                {
                    "experiment_name": experiment_name,
                    "model_name": model_name,
                    "raw_summary": raw_report["summary"],
                    "repair_summary": repair_report["summary"],
                    "det_summary": det_report["summary"],
                    "lex_summary": lex_report["summary"],
                    "status": "completed",
                }
            )
        finally:
            cleanup_model(model, tokenizer)

    def make_rows(key: str) -> list[dict]:
        return sorted(
            [
                {
                    "experiment_name": item["experiment_name"],
                    "model_name": item["model_name"],
                    "field_exact_match": item[f"{key}_summary"]["field_exact_match"],
                    "end_to_end_exact_match": item[f"{key}_summary"]["end_to_end_exact_match"],
                }
                for item in batch_summary
            ],
            key=lambda row: row["end_to_end_exact_match"],
            reverse=True,
        )

    raw_rows = make_rows("raw")
    repair_rows = make_rows("repair")
    det_rows = make_rows("det")
    lex_rows = make_rows("lex")

    lines = [
        "# Big Model Reference Summary",
        "",
        f"- target schema: `{SCHEMA_NAME}`",
        "- evaluation target: canonicalized reduced-schema target with canonical action and majority-mapped component",
        "- tracks: raw prompt-only, repair, deterministic postprocess, lexical postprocess",
        "",
    ]
    for title, rows in [
        ("Raw Prompt-Only", raw_rows),
        ("Prompt-Only + Repair", repair_rows),
        ("Prompt-Only + Deterministic Postprocess", det_rows),
        ("Prompt-Only + Lexical Postprocess", lex_rows),
    ]:
        lines.extend([f"## {title}", ""])
        for row in rows:
            lines.extend(
                [
                    f"### {row['model_name']}",
                    "",
                    f"- experiment: `{row['experiment_name']}`",
                    f"- field exact match: `{row['field_exact_match']:.4f}`",
                    f"- end-to-end exact match: `{row['end_to_end_exact_match']:.4f}`",
                    "",
                ]
            )
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("summary_path =", SUMMARY_PATH)


if __name__ == "__main__":
    run()
