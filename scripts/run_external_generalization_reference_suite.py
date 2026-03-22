from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import torch
from peft import PeftModel
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
BASE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
TRAINED_CHECKPOINT_DIR = PROJECT_ROOT / "results" / "checkpoints" / "qwen25_3b_stage7_canonical_action_component_structure_then_semantics_stage2_epoch9"
TRAIN_COMPONENT_SOURCE = PROJECT_ROOT / "data" / "reduced" / "phase1_train_reduced.jsonl"
SKIP_COMPLETED = True

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

USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
)

GENERATION_KWARGS = {
    "max_new_tokens": 256,
    "do_sample": False,
    "temperature": 1.0,
    "top_p": 1.0,
}

MODEL_PRESETS = [
    {
        "key": "trained_stage7_raw",
        "experiment_suffix": "trained_stage7_raw",
        "type": "trained_small",
        "model_name": BASE_MODEL_NAME,
        "batch_size": 16,
    },
    {
        "key": "trained_stage8_deterministic",
        "experiment_suffix": "trained_stage8_deterministic",
        "type": "trained_small_postprocess",
        "source_key": "trained_stage7_raw",
        "postprocess": "stage8",
    },
    {
        "key": "trained_stage9_lexical",
        "experiment_suffix": "trained_stage9_lexical",
        "type": "trained_small_postprocess",
        "source_key": "trained_stage8_deterministic",
        "postprocess": "stage9",
    },
    {
        "key": "prompt_14b_raw",
        "experiment_suffix": "prompt_14b_raw",
        "type": "prompt_only",
        "model_name": "Qwen/Qwen2.5-14B-Instruct",
        "batch_size": 12,
    },
    {
        "key": "prompt_14b_repair",
        "experiment_suffix": "prompt_14b_repair",
        "type": "prompt_only_repair",
        "source_key": "prompt_14b_raw",
    },
    {
        "key": "prompt_14b_lexical",
        "experiment_suffix": "prompt_14b_lexical",
        "type": "prompt_only_postprocess",
        "source_key": "prompt_14b_raw",
        "postprocess": "stage9",
    },
    {
        "key": "prompt_32b_raw",
        "experiment_suffix": "prompt_32b_raw",
        "type": "prompt_only",
        "model_name": "Qwen/Qwen2.5-32B-Instruct",
        "batch_size": 6,
    },
    {
        "key": "prompt_32b_repair",
        "experiment_suffix": "prompt_32b_repair",
        "type": "prompt_only_repair",
        "source_key": "prompt_32b_raw",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run external-dataset generalization references.")
    parser.add_argument("--input", required=True, help="Path to external project-format reduced-schema eval jsonl.")
    parser.add_argument("--dataset-tag", default=None, help="Short tag used in output filenames and summary names.")
    return parser.parse_args()


def sanitize_tag(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def build_majority_component_map(records: list[dict]) -> dict[str, str]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        target = record["target_json"]
        counts[target["affected_systems"][0]["name"]][target["affected_systems"][0]["component"]] += 1
    return {name: counter.most_common(1)[0][0] for name, counter in counts.items()}


def canonicalize_target_record(record: dict, component_map: dict[str, str]) -> dict:
    updated = deepcopy(record)
    updated.setdefault("schema_name", SCHEMA_NAME)
    updated.setdefault("complexity_bucket", "unknown")
    target = updated["target_json"]
    systems = target.get("affected_systems")
    if isinstance(systems, list) and systems and isinstance(systems[0], dict):
        name = systems[0].get("name")
        if isinstance(name, str) and name in component_map:
            systems[0]["component"] = component_map[name]
    category = target.get("category")
    summary = target.get("summary")
    actions = target.get("actions_requested")
    if (
        isinstance(category, str)
        and category in ACTION_PREFIX_BY_CATEGORY
        and isinstance(summary, str)
        and isinstance(actions, list)
        and actions
        and isinstance(actions[0], dict)
    ):
        actions[0]["action"] = f"{ACTION_PREFIX_BY_CATEGORY[category]}: {summary}"
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
                f"Task: extract a structured record for {record.get('task_name', 'ticket_structured_output')}.\n"
                f"Schema name: {SCHEMA_NAME}\n"
                f"{conventions}\n"
                "Input text:\n"
                f"{record['input_text']}"
            ),
        },
    ]


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


def apply_stage9_postprocess(prediction_json: dict | None, input_text: str) -> dict | None:
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


def build_output_paths(dataset_tag: str, experiment_suffix: str) -> dict[str, Path]:
    base = f"{dataset_tag}_{experiment_suffix}"
    return {
        "prediction_path": PROJECT_ROOT / "results" / "predictions" / f"{base}_test.jsonl",
        "report_path": PROJECT_ROOT / "results" / "metrics" / f"{base}_test_report.json",
        "field_path": PROJECT_ROOT / "results" / "metrics" / f"{base}_field_analysis.json",
    }


def outputs_complete(paths: dict[str, Path]) -> bool:
    return all(path.exists() for path in paths.values())


def cleanup_model(*objects) -> None:
    for obj in objects:
        if obj is not None:
            del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=BNB_CONFIG,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def load_trained_stage7_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=BNB_CONFIG,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, str(TRAINED_CHECKPOINT_DIR))
    model.eval()
    return model, tokenizer


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
                "schema_name": gold_record.get("schema_name", SCHEMA_NAME),
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


def write_eval_outputs(paths: dict[str, Path], gold_records: list[dict], pred_records: list[dict], schema: dict) -> dict:
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
    write_json_report(paths["report_path"], report)
    write_json_report(paths["field_path"], analyze_field_errors(gold_records, pred_records))
    return report


def generate_predictions(model, tokenizer, records: list[dict], batch_size: int, experiment_name: str) -> list[dict]:
    texts = batched_generate_texts(
        model=model,
        tokenizer=tokenizer,
        records=records,
        build_messages=build_inference_messages,
        generation_kwargs=GENERATION_KWARGS,
        batch_size=batch_size,
    )
    outputs = []
    for record, prediction_text in zip(records, texts, strict=True):
        try:
            prediction_json = json.loads(prediction_text)
        except json.JSONDecodeError:
            prediction_json = None
        outputs.append(
            {
                "sample_id": record["sample_id"],
                "prediction_text": prediction_text,
                "prediction_json": prediction_json,
                "metadata": {"experiment_id": experiment_name},
            }
        )
    return outputs


def run() -> None:
    args = parse_args()
    input_path = Path(args.input)
    dataset_tag = sanitize_tag(args.dataset_tag or input_path.stem)
    summary_path = PROJECT_ROOT / "docs" / "results" / f"{dataset_tag}_generalization_reference_summary.md"

    print("project_root =", PROJECT_ROOT)
    print("input_path =", input_path)
    print("dataset_tag =", dataset_tag)
    print("summary_path =", summary_path)

    schema = get_schema(SCHEMA_NAME)
    component_map = build_majority_component_map(load_jsonl(TRAIN_COMPONENT_SOURCE))
    gold_records_raw = load_jsonl(input_path)
    gold_records = [canonicalize_target_record(record, component_map) for record in gold_records_raw]
    input_text_by_id = {record["sample_id"]: record["input_text"] for record in gold_records}

    completed: dict[str, dict] = {}
    leaderboard_rows: list[dict] = []

    for preset in MODEL_PRESETS:
        experiment_name = f"{dataset_tag}_{preset['experiment_suffix']}"
        paths = build_output_paths(dataset_tag, preset["experiment_suffix"])

        if SKIP_COMPLETED and outputs_complete(paths):
            report = json.loads(paths["report_path"].read_text(encoding="utf-8"))
            completed[preset["key"]] = {
                "experiment_name": experiment_name,
                "prediction_path": paths["prediction_path"],
                "report": report,
            }
            leaderboard_rows.append(
                {
                    "key": preset["key"],
                    "experiment_name": experiment_name,
                    "field_exact_match": report["summary"]["field_exact_match"],
                    "end_to_end_exact_match": report["summary"]["end_to_end_exact_match"],
                }
            )
            print("skipped_existing =", experiment_name)
            continue

        print("\n" + "=" * 80)
        print("running =", preset["key"])
        print("experiment =", experiment_name)
        print("=" * 80)

        if preset["type"] in {"trained_small", "prompt_only"}:
            model = None
            tokenizer = None
            try:
                if preset["type"] == "trained_small":
                    model, tokenizer = load_trained_stage7_model()
                else:
                    model, tokenizer = load_model_and_tokenizer(preset["model_name"])

                predictions = generate_predictions(
                    model=model,
                    tokenizer=tokenizer,
                    records=gold_records,
                    batch_size=int(preset["batch_size"]),
                    experiment_name=experiment_name,
                )
                dump_jsonl(paths["prediction_path"], predictions)
                report = write_eval_outputs(paths, gold_records, predictions, schema)
            finally:
                cleanup_model(model, tokenizer)
        elif preset["type"] == "trained_small_postprocess":
            source = completed[preset["source_key"]]
            source_predictions = load_jsonl(source["prediction_path"])
            predictions = []
            for record in source_predictions:
                sample_id = record["sample_id"]
                if preset["postprocess"] == "stage8":
                    prediction_json = apply_stage8_postprocess(record.get("prediction_json"), component_map)
                else:
                    prediction_json = apply_stage9_postprocess(record.get("prediction_json"), input_text_by_id[sample_id])
                predictions.append(
                    {
                        **record,
                        "prediction_json": prediction_json,
                        "metadata": {**record.get("metadata", {}), "source_experiment": source["experiment_name"]},
                    }
                )
            dump_jsonl(paths["prediction_path"], predictions)
            report = write_eval_outputs(paths, gold_records, predictions, schema)
        elif preset["type"] == "prompt_only_repair":
            source = completed[preset["source_key"]]
            source_predictions = load_jsonl(source["prediction_path"])
            predictions = []
            for record in source_predictions:
                prediction_json = record.get("prediction_json")
                prediction_text = record.get("prediction_text")
                if prediction_json is None and isinstance(prediction_text, str):
                    _, prediction_json = try_parse_prediction_text(prediction_text)
                repaired_json, repaired = repair_prediction(prediction_json, schema)
                predictions.append(
                    {
                        **record,
                        "prediction_json": repaired_json,
                        "metadata": {**record.get("metadata", {}), "repaired": repaired, "source_experiment": source["experiment_name"]},
                    }
                )
            dump_jsonl(paths["prediction_path"], predictions)
            report = write_eval_outputs(paths, gold_records, predictions, schema)
        elif preset["type"] == "prompt_only_postprocess":
            source = completed[preset["source_key"]]
            source_predictions = load_jsonl(source["prediction_path"])
            predictions = []
            for record in source_predictions:
                sample_id = record["sample_id"]
                prediction_json = apply_stage9_postprocess(record.get("prediction_json"), input_text_by_id[sample_id])
                predictions.append(
                    {
                        **record,
                        "prediction_json": prediction_json,
                        "metadata": {**record.get("metadata", {}), "source_experiment": source["experiment_name"]},
                    }
                )
            dump_jsonl(paths["prediction_path"], predictions)
            report = write_eval_outputs(paths, gold_records, predictions, schema)
        else:
            raise ValueError(f"Unknown preset type: {preset['type']}")

        completed[preset["key"]] = {
            "experiment_name": experiment_name,
            "prediction_path": paths["prediction_path"],
            "report": report,
        }
        leaderboard_rows.append(
            {
                "key": preset["key"],
                "experiment_name": experiment_name,
                "field_exact_match": report["summary"]["field_exact_match"],
                "end_to_end_exact_match": report["summary"]["end_to_end_exact_match"],
            }
        )

    leaderboard_rows.sort(key=lambda item: item["end_to_end_exact_match"], reverse=True)

    lines = [
        "# External Generalization Reference Summary",
        "",
        f"- input dataset: `{input_path}`",
        f"- dataset tag: `{dataset_tag}`",
        f"- target schema: `{SCHEMA_NAME}`",
        "- evaluation target: canonicalized reduced-schema target with canonical action and majority-mapped component",
        "- compared tracks: strongest trained 3B, strongest 3B system line, 14B prompt-only, and 32B prompt-only references",
        "",
        "## Leaderboard",
        "",
    ]
    for row in leaderboard_rows:
        lines.extend(
            [
                f"### {row['key']}",
                "",
                f"- experiment: `{row['experiment_name']}`",
                f"- field exact match: `{row['field_exact_match']:.4f}`",
                f"- end-to-end exact match: `{row['end_to_end_exact_match']:.4f}`",
                "",
            ]
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("summary_path =", summary_path)
    print("leaderboard =")
    for row in leaderboard_rows:
        print(row)


if __name__ == "__main__":
    run()
