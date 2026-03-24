from __future__ import annotations

import gc
import json
import re
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

try:
    from vllm import LLM, SamplingParams
    try:
        from vllm.sampling_params import StructuredOutputsParams as _StructuredParamsClass
        _STRUCTURED_MODE = "structured_outputs"
    except Exception:
        from vllm.sampling_params import GuidedDecodingParams as _StructuredParamsClass
        _STRUCTURED_MODE = "guided_decoding"
except Exception as exc:  # pragma: no cover - runtime-only dependency
    raise ImportError(
        "This suite requires vLLM. Install it in the target environment with `pip install vllm`."
    ) from exc


SCHEMA_NAME = "ticket_schema_v1_reduced"
SUMMARY_PATH = PROJECT_ROOT / "docs" / "results" / "vllm_structured_reference_summary.md"
SKIP_COMPLETED = True
GUIDED_BACKEND: str | None = None

MODEL_PRESETS = [
    {
        "experiment_name": "qwen25_3b_vllm_reference",
        "model_name": "Qwen/Qwen2.5-3B-Instruct",
        "max_num_seqs": 64,
        "max_model_len": 2048,
    },
    {
        "experiment_name": "qwen25_7b_vllm_reference",
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "max_num_seqs": 48,
        "max_model_len": 2048,
    },
    {
        "experiment_name": "qwen25_14b_vllm_reference",
        "model_name": "Qwen/Qwen2.5-14B-Instruct",
        "max_num_seqs": 24,
        "max_model_len": 2048,
    },
    {
        "experiment_name": "qwen25_32b_vllm_reference",
        "model_name": "Qwen/Qwen2.5-32B-Instruct",
        "max_num_seqs": 12,
        "max_model_len": 2048,
    },
]

RUN_VARIANTS = [
    "raw",
    "raw_repair",
    "structured_json",
]

ACTION_PREFIX_BY_CATEGORY = {
    "task": "Handle request",
    "bug": "Investigate issue",
    "feature": "Review and plan request",
    "incident": "Investigate and mitigate incident",
    "question": "Answer and clarify",
}

GENERATION_KWARGS = {
    "max_tokens": 256,
    "temperature": 0.0,
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


def build_messages(record: dict) -> list[dict]:
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


def build_prompt(llm: LLM, record: dict) -> str:
    tokenizer = llm.get_tokenizer()
    messages = build_messages(record)
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n".join(f"{item['role']}: {item['content']}" for item in messages)


def make_sampling_params(schema: dict | None) -> SamplingParams:
    structured_param = None
    if schema is not None:
        kwargs = {"json": schema}
        if GUIDED_BACKEND is not None:
            kwargs["backend"] = GUIDED_BACKEND
        structured_param = _StructuredParamsClass(**kwargs)
    sampling_kwargs = {
        "max_tokens": GENERATION_KWARGS["max_tokens"],
        "temperature": GENERATION_KWARGS["temperature"],
        "top_p": GENERATION_KWARGS["top_p"],
    }
    if _STRUCTURED_MODE == "structured_outputs":
        sampling_kwargs["structured_outputs"] = structured_param
    else:
        sampling_kwargs["guided_decoding"] = structured_param
    return SamplingParams(**sampling_kwargs)


def build_output_paths(experiment_name: str, variant: str) -> dict[str, Path]:
    base = f"{experiment_name}_{variant}"
    return {
        "prediction_path": PROJECT_ROOT / "results" / "predictions" / f"{base}_test.jsonl",
        "report_path": PROJECT_ROOT / "results" / "metrics" / f"{base}_test_report.json",
        "field_path": PROJECT_ROOT / "results" / "metrics" / f"{base}_field_analysis.json",
    }


def outputs_complete(paths: dict[str, Path]) -> bool:
    return all(path.exists() for path in paths.values())


def cleanup_llm(llm: LLM | None) -> None:
    if llm is not None:
        del llm
    gc.collect()


def parse_prediction_text(prediction_text: str) -> dict | None:
    try:
        return json.loads(prediction_text)
    except json.JSONDecodeError:
        _, prediction_json = try_parse_prediction_text(prediction_text)
        return prediction_json


def generate_predictions(llm: LLM, records: list[dict], schema: dict | None, experiment_name: str, variant: str) -> list[dict]:
    prompts = [build_prompt(llm, record) for record in records]
    params = make_sampling_params(schema)
    outputs = llm.generate(prompts, params, use_tqdm=True)
    predictions = []
    for record, output in zip(records, outputs, strict=True):
        text = output.outputs[0].text if output.outputs else ""
        predictions.append(
            {
                "sample_id": record["sample_id"],
                "prediction_text": text,
                "prediction_json": parse_prediction_text(text),
                "metadata": {
                    "experiment_id": experiment_name,
                    "variant": variant,
                    "guided_backend": GUIDED_BACKEND,
                },
            }
        )
    return predictions


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


def build_repaired_predictions(source_predictions: list[dict], schema: dict, experiment_name: str, variant: str) -> list[dict]:
    repaired_predictions = []
    for record in source_predictions:
        repaired_json, repaired = repair_prediction(record.get("prediction_json"), schema)
        repaired_predictions.append(
            {
                **record,
                "prediction_json": repaired_json,
                "metadata": {
                    **record.get("metadata", {}),
                    "experiment_id": experiment_name,
                    "variant": variant,
                    "repaired": repaired,
                },
            }
        )
    return repaired_predictions


def write_summary(batch_summary: list[dict]) -> None:
    lines = [
        "# vLLM Structured Reference Summary",
        "",
        f"- target schema: `{SCHEMA_NAME}`",
        f"- guided backend: `{GUIDED_BACKEND}`",
        "- compared tracks: raw vLLM prompt-only, raw+repair, and vLLM JSON-schema structured outputs",
        "",
        "| Model | Variant | Field Exact Match | End-to-End Exact Match |",
        "| --- | --- | ---: | ---: |",
    ]
    for item in batch_summary:
        summary = item["report"]["summary"]
        lines.append(
            f"| {item['model_name']} | {item['variant']} | {summary['field_exact_match']:.4f} | {summary['end_to_end_exact_match']:.4f} |"
        )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> None:
    print("project_root =", PROJECT_ROOT)
    print("guided_backend =", GUIDED_BACKEND)
    print("structured_mode =", _STRUCTURED_MODE)
    print("scheduled_models =", [item["model_name"] for item in MODEL_PRESETS])
    print("skip_completed =", SKIP_COMPLETED)

    schema = get_schema(SCHEMA_NAME)
    train_records = load_jsonl(PROJECT_ROOT / "data" / "reduced" / "phase1_train_reduced.jsonl")
    test_records_raw = load_jsonl(PROJECT_ROOT / "data" / "reduced" / "phase1_test_reduced.jsonl")
    component_map = build_majority_component_map(train_records)
    test_records = [canonicalize_target_record(record, component_map) for record in test_records_raw]

    batch_summary: list[dict] = []

    for preset in MODEL_PRESETS:
        model_name = preset["model_name"]
        experiment_name = preset["experiment_name"]
        print("\n" + "=" * 80)
        print("model =", model_name)
        print("experiment =", experiment_name)
        print("=" * 80)

        llm = None
        try:
            need_llm = False
            for variant in RUN_VARIANTS:
                if not (SKIP_COMPLETED and outputs_complete(build_output_paths(experiment_name, variant))):
                    need_llm = True
                    break

            if need_llm:
                llm = LLM(
                    model=model_name,
                    trust_remote_code=True,
                    gpu_memory_utilization=0.92,
                    max_num_seqs=preset["max_num_seqs"],
                    max_model_len=preset["max_model_len"],
                    dtype="bfloat16",
                    tensor_parallel_size=1,
                )

            raw_predictions_cache: list[dict] | None = None

            for variant in RUN_VARIANTS:
                paths = build_output_paths(experiment_name, variant)
                if SKIP_COMPLETED and outputs_complete(paths):
                    report = json.loads(paths["report_path"].read_text(encoding="utf-8"))
                    batch_summary.append(
                        {
                            "model_name": model_name,
                            "experiment_name": experiment_name,
                            "variant": variant,
                            "report": report,
                            "status": "skipped_existing",
                        }
                    )
                    continue

                if variant == "raw":
                    raw_predictions_cache = generate_predictions(
                        llm=llm,
                        records=test_records_raw,
                        schema=None,
                        experiment_name=experiment_name,
                        variant=variant,
                    )
                    predictions = raw_predictions_cache
                elif variant == "raw_repair":
                    if raw_predictions_cache is None:
                        raw_predictions_cache = load_jsonl(build_output_paths(experiment_name, "raw")["prediction_path"])
                    predictions = build_repaired_predictions(raw_predictions_cache, schema, experiment_name, variant)
                elif variant == "structured_json":
                    predictions = generate_predictions(
                        llm=llm,
                        records=test_records_raw,
                        schema=schema,
                        experiment_name=experiment_name,
                        variant=variant,
                    )
                else:
                    raise ValueError(f"Unknown variant: {variant}")

                dump_jsonl(paths["prediction_path"], predictions)
                report = write_eval_outputs(paths, test_records, predictions, schema)
                batch_summary.append(
                    {
                        "model_name": model_name,
                        "experiment_name": experiment_name,
                        "variant": variant,
                        "report": report,
                        "status": "completed",
                    }
                )
        finally:
            cleanup_llm(llm)

    batch_summary.sort(
        key=lambda item: item["report"]["summary"]["end_to_end_exact_match"],
        reverse=True,
    )
    write_summary(batch_summary)
    print("summary_path =", SUMMARY_PATH)
    for item in batch_summary:
        summary = item["report"]["summary"]
        print(
            {
                "model_name": item["model_name"],
                "variant": item["variant"],
                "field_exact_match": summary["field_exact_match"],
                "end_to_end_exact_match": summary["end_to_end_exact_match"],
            }
        )


if __name__ == "__main__":
    run()
