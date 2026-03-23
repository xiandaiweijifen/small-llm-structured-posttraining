from __future__ import annotations

import gc
import json
import random
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

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
from src.training.formatters import DEFAULT_SYSTEM_PROMPT, build_user_prompt, convert_to_sft_records

BASE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
SCHEMA_NAME = "ticket_schema_v1_reduced"
BASELINE_CHECKPOINT_DIR = PROJECT_ROOT / "results" / "checkpoints" / "qwen25_3b_stage7_canonical_action_component_structure_then_semantics_stage2_epoch9"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external_generalization"
SOURCE_NAME = "gorkemsevinc_customer_support_tickets"
EXTERNAL_TRAIN = EXTERNAL_DIR / f"{SOURCE_NAME}_train_reduced.jsonl"
EXTERNAL_VAL = EXTERNAL_DIR / f"{SOURCE_NAME}_val_reduced.jsonl"
EXTERNAL_TEST = EXTERNAL_DIR / f"{SOURCE_NAME}_test_reduced.jsonl"
ARTIFACT_DIR = PROJECT_ROOT / "data" / "stage13_external_adaptation"
SUMMARY_PATH = PROJECT_ROOT / "docs" / "results" / "external_adaptation_batch_summary.md"
RUN_PRESETS = [
    "ext64_epoch3_lr1e4",
    "ext256_epoch3_lr1e4",
    "ext512_epoch3_lr1e4",
    "ext1024_epoch3_lr1e4",
    "ext256_mix_epoch3_lr1e4",
    "ext512_mix_epoch3_lr1e4",
    "ext1024_mix_epoch3_lr1e4",
]
SKIP_COMPLETED = True
INFERENCE_BATCH_SIZE = 24

PRESETS = {
    "ext64_epoch3_lr1e4": {
        "experiment_name": "qwen25_3b_stage13_ext64_epoch3_lr1e4",
        "external_samples": 64,
        "mix_in_domain": False,
        "learning_rate": 1e-4,
        "epochs": 3,
        "batch_size": 12,
        "grad_accum": 4,
        "seed": 42,
    },
    "ext256_epoch3_lr1e4": {
        "experiment_name": "qwen25_3b_stage13_ext256_epoch3_lr1e4",
        "external_samples": 256,
        "mix_in_domain": False,
        "learning_rate": 1e-4,
        "epochs": 3,
        "batch_size": 12,
        "grad_accum": 4,
        "seed": 42,
    },
    "ext512_epoch3_lr1e4": {
        "experiment_name": "qwen25_3b_stage13_ext512_epoch3_lr1e4",
        "external_samples": 512,
        "mix_in_domain": False,
        "learning_rate": 1e-4,
        "epochs": 3,
        "batch_size": 12,
        "grad_accum": 4,
        "seed": 42,
    },
    "ext1024_epoch3_lr1e4": {
        "experiment_name": "qwen25_3b_stage13_ext1024_epoch3_lr1e4",
        "external_samples": 1024,
        "mix_in_domain": False,
        "learning_rate": 1e-4,
        "epochs": 3,
        "batch_size": 12,
        "grad_accum": 4,
        "seed": 42,
    },
    "ext256_mix_epoch3_lr1e4": {
        "experiment_name": "qwen25_3b_stage13_ext256_mix_epoch3_lr1e4",
        "external_samples": 256,
        "mix_in_domain": True,
        "learning_rate": 1e-4,
        "epochs": 3,
        "batch_size": 12,
        "grad_accum": 4,
        "seed": 42,
    },
    "ext512_mix_epoch3_lr1e4": {
        "experiment_name": "qwen25_3b_stage13_ext512_mix_epoch3_lr1e4",
        "external_samples": 512,
        "mix_in_domain": True,
        "learning_rate": 1e-4,
        "epochs": 3,
        "batch_size": 12,
        "grad_accum": 4,
        "seed": 42,
    },
    "ext1024_mix_epoch3_lr1e4": {
        "experiment_name": "qwen25_3b_stage13_ext1024_mix_epoch3_lr1e4",
        "external_samples": 1024,
        "mix_in_domain": True,
        "learning_rate": 1e-4,
        "epochs": 3,
        "batch_size": 12,
        "grad_accum": 4,
        "seed": 42,
    },
}

ACTION_PREFIX_BY_CATEGORY = {
    "task": "Handle request",
    "bug": "Investigate issue",
    "feature": "Review and plan request",
    "incident": "Investigate and mitigate incident",
    "question": "Answer and clarify",
}

GENERATION_KWARGS = {
    "max_new_tokens": 256,
    "do_sample": False,
    "temperature": 1.0,
    "top_p": 1.0,
}

USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
try:
    import bitsandbytes  # type: ignore  # noqa: F401

    HAS_BNB_4BIT = True
except Exception:
    HAS_BNB_4BIT = False

BNB_CONFIG = (
    BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
    )
    if HAS_BNB_4BIT
    else None
)


def build_majority_component_map(records: list[dict]) -> dict[str, str]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        target = record["target_json"]
        counts[target["affected_systems"][0]["name"]][target["affected_systems"][0]["component"]] += 1
    return {name: counter.most_common(1)[0][0] for name, counter in counts.items()}


def canonicalize_in_domain_record(record: dict, component_map: dict[str, str]) -> dict:
    updated = deepcopy(record)
    target = updated["target_json"]
    name = target["affected_systems"][0]["name"]
    if name in component_map:
        target["affected_systems"][0]["component"] = component_map[name]
    category = target["category"]
    target["actions_requested"][0]["action"] = f"{ACTION_PREFIX_BY_CATEGORY[category]}: {target['summary']}"
    return updated


def bucket_counts(records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[record["complexity_bucket"]] += 1
    return dict(sorted(counts.items()))


def build_output_paths(experiment_name: str) -> dict[str, Path]:
    return {
        "checkpoint_dir": PROJECT_ROOT / "results" / "checkpoints" / experiment_name,
        "prediction_path": PROJECT_ROOT / "results" / "predictions" / f"{experiment_name}_test.jsonl",
        "repaired_prediction_path": PROJECT_ROOT / "results" / "predictions" / f"{experiment_name}_test_repaired.jsonl",
        "raw_report_path": PROJECT_ROOT / "results" / "metrics" / f"{experiment_name}_test_report.json",
        "repaired_report_path": PROJECT_ROOT / "results" / "metrics" / f"{experiment_name}_test_repaired_report.json",
        "raw_field_path": PROJECT_ROOT / "results" / "metrics" / f"{experiment_name}_field_analysis.json",
        "repaired_field_path": PROJECT_ROOT / "results" / "metrics" / f"{experiment_name}_test_repaired_field_analysis.json",
    }


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model():
    kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if BNB_CONFIG is not None:
        kwargs["quantization_config"] = BNB_CONFIG
    else:
        kwargs["torch_dtype"] = torch.bfloat16 if USE_BF16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, **kwargs)
    model.config.use_cache = False
    return model


def load_trainable_adapter_model(checkpoint_dir: Path):
    base_model = load_base_model()
    try:
        model = PeftModel.from_pretrained(base_model, str(checkpoint_dir), is_trainable=True)
    except TypeError:
        model = PeftModel.from_pretrained(base_model, str(checkpoint_dir))
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
    model.config.use_cache = False
    return model


def build_training_args(config: dict, output_dir: Path):
    return TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=float(config["learning_rate"]),
        num_train_epochs=float(config["epochs"]),
        per_device_train_batch_size=int(config["batch_size"]),
        per_device_eval_batch_size=int(config["batch_size"]),
        gradient_accumulation_steps=int(config["grad_accum"]),
        warmup_steps=20,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=25,
        save_steps=50,
        save_total_limit=2,
        bf16=USE_BF16,
        fp16=not USE_BF16,
        report_to="none",
        remove_unused_columns=False,
        seed=int(config["seed"]),
    )


def build_trainer(model, dataset, tokenizer, config: dict, output_dir: Path):
    return SFTTrainer(
        model=model,
        args=build_training_args(config, output_dir),
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )


def load_chat_dataset(train_file: Path, validation_file: Path, tokenizer):
    dataset = load_dataset("json", data_files={"train": str(train_file), "validation": str(validation_file)})

    def format_chat_example(example):
        example["text"] = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
        return example

    return dataset.map(format_chat_example)


def write_sft_split(records: list[dict], path: Path) -> Path:
    dump_jsonl(path, convert_to_sft_records(records, include_schema_definition=False))
    return path


def build_inference_messages(record: dict) -> list[dict]:
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(
                input_text=record["input_text"],
                task_name=record["task_name"],
                schema_name=record["schema_name"],
                include_schema_definition=False,
            ),
        },
    ]


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


def outputs_complete(paths: dict[str, Path]) -> bool:
    return all(
        path.exists()
        for path in [
            paths["prediction_path"],
            paths["repaired_prediction_path"],
            paths["raw_report_path"],
            paths["repaired_report_path"],
            paths["raw_field_path"],
            paths["repaired_field_path"],
        ]
    )


def cleanup_model(*objects):
    for obj in objects:
        if obj is not None:
            del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sample_stratified(records: list[dict], sample_size: int, seed: int) -> list[dict]:
    if sample_size <= 0 or len(records) <= sample_size:
        return list(records)
    rng = random.Random(seed)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["target_json"]["category"]].append(record)

    sampled: list[dict] = []
    total = len(records)
    for bucket in grouped.values():
        bucket_copy = list(bucket)
        rng.shuffle(bucket_copy)
        take = max(1, round(sample_size * len(bucket_copy) / total))
        sampled.extend(bucket_copy[:take])

    rng.shuffle(sampled)
    if len(sampled) > sample_size:
        sampled = sampled[:sample_size]
    elif len(sampled) < sample_size:
        seen = {row["sample_id"] for row in sampled}
        remainder = [row for row in records if row["sample_id"] not in seen]
        rng.shuffle(remainder)
        sampled.extend(remainder[: sample_size - len(sampled)])
    return sampled


def build_training_records(config: dict, external_train: list[dict], in_domain_train: list[dict]) -> list[dict]:
    external_subset = sample_stratified(external_train, int(config["external_samples"]), int(config["seed"]))
    if not config["mix_in_domain"]:
        return external_subset
    in_domain_subset = sample_stratified(in_domain_train, len(external_subset), int(config["seed"]) + 1000)
    combined = list(external_subset) + list(in_domain_subset)
    rng = random.Random(int(config["seed"]) + 2000)
    rng.shuffle(combined)
    return combined


def run():
    print("project_root =", PROJECT_ROOT)
    print("python =", sys.version)
    print("cuda_available =", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device =", torch.cuda.get_device_name(0))
        print("bf16_supported =", torch.cuda.is_bf16_supported())
    print("has_bnb_4bit =", HAS_BNB_4BIT)
    print("baseline_checkpoint_dir =", BASELINE_CHECKPOINT_DIR)
    print("scheduled_presets =", RUN_PRESETS)
    print("skip_completed =", SKIP_COMPLETED)
    print("inference_batch_size =", INFERENCE_BATCH_SIZE)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    component_map = build_majority_component_map(load_jsonl(PROJECT_ROOT / "data" / "reduced" / "phase1_train_reduced.jsonl"))
    in_domain_train = [
        canonicalize_in_domain_record(record, component_map)
        for record in load_jsonl(PROJECT_ROOT / "data" / "reduced" / "phase1_train_reduced.jsonl")
    ]
    external_train = load_jsonl(EXTERNAL_TRAIN)
    external_val = load_jsonl(EXTERNAL_VAL)
    external_test = load_jsonl(EXTERNAL_TEST)
    schema = get_schema(SCHEMA_NAME)

    print("external_train =", len(external_train))
    print("external_val =", len(external_val))
    print("external_test =", len(external_test))
    print("external_test_bucket_counts =", bucket_counts(external_test))

    batch_run_results = []
    for preset_name in RUN_PRESETS:
        config = PRESETS[preset_name]
        experiment_name = config["experiment_name"]
        paths = build_output_paths(experiment_name)
        print("\n" + "=" * 80)
        print("running preset =", preset_name)
        print("=" * 80)

        if SKIP_COMPLETED and outputs_complete(paths):
            raw_report = json.loads(paths["raw_report_path"].read_text(encoding="utf-8"))
            repaired_report = json.loads(paths["repaired_report_path"].read_text(encoding="utf-8"))
            batch_run_results.append(
                {
                    "preset_name": preset_name,
                    "experiment_name": experiment_name,
                    "status": "skipped_existing",
                    "raw_summary": raw_report["summary"],
                    "repaired_summary": repaired_report["summary"],
                    "train_count": int(config["external_samples"]) * (2 if config["mix_in_domain"] else 1),
                    "mix_in_domain": bool(config["mix_in_domain"]),
                }
            )
            continue

        train_records = build_training_records(config, external_train, in_domain_train)
        print("train_count =", len(train_records))
        print("mix_in_domain =", bool(config["mix_in_domain"]))
        print("train_bucket_counts =", bucket_counts(train_records))

        tokenizer = load_tokenizer()
        model = load_trainable_adapter_model(BASELINE_CHECKPOINT_DIR)
        trainer = None
        try:
            train_sft_path = ARTIFACT_DIR / f"{experiment_name}_train.jsonl"
            val_sft_path = ARTIFACT_DIR / f"{experiment_name}_val.jsonl"
            write_sft_split(train_records, train_sft_path)
            write_sft_split(external_val, val_sft_path)
            dataset = load_chat_dataset(train_sft_path, val_sft_path, tokenizer)
            output_root = paths["checkpoint_dir"]
            output_root.mkdir(parents=True, exist_ok=True)

            trainer = build_trainer(model=model, dataset=dataset, tokenizer=tokenizer, config=config, output_dir=output_root)
            train_result = trainer.train()
            trainer.save_model(str(output_root))
            print("train_loss =", train_result.training_loss)

            prediction_texts = batched_generate_texts(
                model=model,
                tokenizer=tokenizer,
                records=external_test,
                build_messages=build_inference_messages,
                generation_kwargs=GENERATION_KWARGS,
                batch_size=INFERENCE_BATCH_SIZE,
            )
            predictions = []
            for idx, (record, prediction_text) in enumerate(zip(external_test, prediction_texts, strict=True), 1):
                try:
                    prediction_json = json.loads(prediction_text)
                except json.JSONDecodeError:
                    prediction_json = None
                predictions.append(
                    {
                        "sample_id": record["sample_id"],
                        "prediction_text": prediction_text,
                        "prediction_json": prediction_json,
                        "metadata": {"model_name": BASE_MODEL_NAME, "experiment_id": experiment_name},
                    }
                )
                if idx % 50 == 0:
                    print(f"generated {idx} / {len(external_test)}")
            dump_jsonl(paths["prediction_path"], predictions)

            raw_sample_results = sample_eval_dicts(external_test, predictions, schema)
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
            write_json_report(paths["raw_field_path"], analyze_field_errors(external_test, predictions))

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
            repaired_sample_results = sample_eval_dicts(external_test, repaired_predictions, schema)
            repaired_report = {
                "summary": summarize_from_dicts(repaired_sample_results),
                "per_sample": repaired_sample_results,
            }
            write_json_report(paths["repaired_report_path"], repaired_report)
            write_json_report(paths["repaired_field_path"], analyze_field_errors(external_test, repaired_predictions))

            batch_run_results.append(
                {
                    "preset_name": preset_name,
                    "experiment_name": experiment_name,
                    "status": "completed",
                    "raw_summary": raw_report["summary"],
                    "repaired_summary": repaired_report["summary"],
                    "train_count": len(train_records),
                    "mix_in_domain": bool(config["mix_in_domain"]),
                }
            )
        finally:
            cleanup_model(trainer, model, tokenizer)

    lines = [
        "# External Adaptation Batch Summary",
        "",
        f"- baseline checkpoint: `{BASELINE_CHECKPOINT_DIR}`",
        f"- external train: `{EXTERNAL_TRAIN}`",
        f"- external val: `{EXTERNAL_VAL}`",
        f"- external test: `{EXTERNAL_TEST}`",
        f"- skip completed: `{SKIP_COMPLETED}`",
        "",
        "## Runs",
        "",
    ]
    for item in batch_run_results:
        raw = item["raw_summary"]
        repaired = item["repaired_summary"]
        lines.extend(
            [
                f"### {item['preset_name']}",
                "",
                f"- experiment: `{item['experiment_name']}`",
                f"- status: `{item['status']}`",
                f"- train count: `{item['train_count']}`",
                f"- mix in-domain: `{item['mix_in_domain']}`",
                f"- raw field exact match: `{raw['field_exact_match']:.4f}`",
                f"- raw end-to-end exact match: `{raw['end_to_end_exact_match']:.4f}`",
                f"- repaired field exact match: `{repaired['field_exact_match']:.4f}`",
                f"- repaired end-to-end exact match: `{repaired['end_to_end_exact_match']:.4f}`",
                "",
            ]
        )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("summary_path =", SUMMARY_PATH)

    leaderboard = sorted(
        [
            {
                "preset_name": item["preset_name"],
                "experiment_name": item["experiment_name"],
                "status": item["status"],
                "field_exact_match": item["raw_summary"]["field_exact_match"],
                "end_to_end_exact_match": item["raw_summary"]["end_to_end_exact_match"],
                "train_count": item["train_count"],
                "mix_in_domain": item["mix_in_domain"],
            }
            for item in batch_run_results
        ],
        key=lambda row: row["end_to_end_exact_match"],
        reverse=True,
    )
    print("leaderboard =")
    for row in leaderboard:
        print(row)


if __name__ == "__main__":
    run()
