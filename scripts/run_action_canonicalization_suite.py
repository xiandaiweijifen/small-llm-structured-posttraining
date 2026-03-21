from __future__ import annotations

import gc
import json
import sys
from copy import deepcopy
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import dump_jsonl, load_jsonl
from src.evaluation.field_analysis import analyze_field_errors
from src.evaluation.metrics import evaluate_sample, summarize_results, try_parse_prediction_text
from src.evaluation.reporting import group_sample_results, write_json_report
from src.inference.repair import repair_prediction
from src.schemas.registry import get_schema
from src.training.formatters import DEFAULT_SYSTEM_PROMPT, build_user_prompt, convert_to_sft_records

BASE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
SCHEMA_NAME = "ticket_schema_v1_reduced"
ARTIFACT_DIR = PROJECT_ROOT / "data" / "stage6_action_canonicalization"
SUMMARY_PATH = PROJECT_ROOT / "docs" / "results" / "action_canonicalization_batch_summary.md"
RUN_PRESETS = [
    "canonical_action_single_stage_epoch5_lr2e4",
    "canonical_action_structure_then_semantics_v1",
]
SKIP_COMPLETED = True

PRESETS = {
    "canonical_action_single_stage_epoch5_lr2e4": {
        "experiment_name": "qwen25_3b_stage6_canonical_action_single_stage_epoch5_lr2e4",
        "mode": "full",
        "learning_rate": 2e-4,
        "epochs": 5,
        "batch_size": 8,
        "grad_accum": 4,
        "seed": 42,
        "rank": 16,
        "alpha": 32,
        "dropout": 0.05,
    },
    "canonical_action_structure_then_semantics_v1": {
        "experiment_name": "qwen25_3b_stage6_canonical_action_structure_then_semantics_v1",
        "mode": "two_stage_structure_semantics",
        "learning_rate": 2e-4,
        "stage1_epochs": 1,
        "stage2_epochs": 5,
        "batch_size": 8,
        "grad_accum": 4,
        "seed": 42,
        "rank": 16,
        "alpha": 32,
        "dropout": 0.05,
        "structure_stage_buckets": ["simple", "medium"],
        "structure_stage_include_schema_definition": True,
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
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
)


def canonicalize_action(record: dict) -> dict:
    updated = deepcopy(record)
    target = updated["target_json"]
    category = target["category"]
    prefix = ACTION_PREFIX_BY_CATEGORY[category]
    summary = target["summary"]
    target["actions_requested"][0]["action"] = f"{prefix}: {summary}"
    return updated


def bucket_counts(records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        bucket = record["complexity_bucket"]
        counts[bucket] = counts.get(bucket, 0) + 1
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
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=BNB_CONFIG,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    return model


def build_training_args(config: dict, output_dir: Path, num_epochs: float):
    return TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=float(config["learning_rate"]),
        num_train_epochs=float(num_epochs),
        per_device_train_batch_size=int(config["batch_size"]),
        per_device_eval_batch_size=int(config["batch_size"]),
        gradient_accumulation_steps=int(config["grad_accum"]),
        warmup_steps=50,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=100,
        save_total_limit=2,
        bf16=USE_BF16,
        fp16=not USE_BF16,
        report_to="none",
        remove_unused_columns=False,
        seed=int(config["seed"]),
    )


def build_peft_config(config: dict):
    return LoraConfig(
        r=int(config["rank"]),
        lora_alpha=int(config["alpha"]),
        lora_dropout=float(config["dropout"]),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules="all-linear",
    )


def build_trainer(model, dataset, tokenizer, config: dict, output_dir: Path, num_epochs: float, peft_config=None):
    trainer_kwargs = {
        "model": model,
        "args": build_training_args(config, output_dir, num_epochs),
        "train_dataset": dataset["train"],
        "eval_dataset": dataset["validation"],
        "processing_class": tokenizer,
    }
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config
    return SFTTrainer(**trainer_kwargs)


def load_chat_dataset(train_file: Path, validation_file: Path, tokenizer):
    dataset = load_dataset("json", data_files={"train": str(train_file), "validation": str(validation_file)})

    def format_chat_example(example):
        example["text"] = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
        return example

    return dataset.map(format_chat_example)


def write_sft_split(records: list[dict], path: Path, include_schema_definition: bool = False) -> Path:
    dump_jsonl(path, convert_to_sft_records(records, include_schema_definition=include_schema_definition))
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


def generate_prediction_text(model, tokenizer, record: dict) -> str:
    messages = build_inference_messages(record)
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, **GENERATION_KWARGS)
    generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


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


def run():
    print("project_root =", PROJECT_ROOT)
    print("python =", sys.version)
    print("cuda_available =", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device =", torch.cuda.get_device_name(0))
        print("bf16_supported =", torch.cuda.is_bf16_supported())
    print("scheduled_presets =", RUN_PRESETS)
    print("skip_completed =", SKIP_COMPLETED)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    train_records = [canonicalize_action(record) for record in load_jsonl(PROJECT_ROOT / "data" / "reduced" / "phase1_train_reduced.jsonl")]
    val_records = [canonicalize_action(record) for record in load_jsonl(PROJECT_ROOT / "data" / "reduced" / "phase1_val_reduced.jsonl")]
    test_records = [canonicalize_action(record) for record in load_jsonl(PROJECT_ROOT / "data" / "reduced" / "phase1_test_reduced.jsonl")]
    schema = get_schema(SCHEMA_NAME)
    print("train =", len(train_records))
    print("val =", len(val_records))
    print("test =", len(test_records))
    print("train_bucket_counts =", bucket_counts(train_records))

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
                    "mode": config["mode"],
                    "raw_summary": raw_report["summary"],
                    "repaired_summary": repaired_report["summary"],
                }
            )
            continue

        tokenizer = load_tokenizer()
        model = load_base_model()
        trainer_objects = []
        try:
            val_sft_path = ARTIFACT_DIR / f"{experiment_name}_val.jsonl"
            write_sft_split(val_records, val_sft_path, include_schema_definition=False)
            output_root = paths["checkpoint_dir"]
            output_root.mkdir(parents=True, exist_ok=True)

            if config["mode"] == "two_stage_structure_semantics":
                structure_stage_records = [
                    record
                    for record in train_records
                    if record["complexity_bucket"] in set(config["structure_stage_buckets"])
                ]
                stage1_train_path = ARTIFACT_DIR / f"{experiment_name}_stage1_train.jsonl"
                stage2_train_path = ARTIFACT_DIR / f"{experiment_name}_stage2_train.jsonl"
                write_sft_split(
                    structure_stage_records,
                    stage1_train_path,
                    include_schema_definition=bool(config["structure_stage_include_schema_definition"]),
                )
                write_sft_split(train_records, stage2_train_path, include_schema_definition=False)

                stage1_dataset = load_chat_dataset(stage1_train_path, val_sft_path, tokenizer)
                stage1_output = output_root / "stage1_structure"
                stage1_trainer = build_trainer(
                    model=model,
                    dataset=stage1_dataset,
                    tokenizer=tokenizer,
                    config=config,
                    output_dir=stage1_output,
                    num_epochs=float(config["stage1_epochs"]),
                    peft_config=build_peft_config(config),
                )
                trainer_objects.append(stage1_trainer)
                stage1_result = stage1_trainer.train()
                stage1_trainer.save_model(str(stage1_output))
                model = stage1_trainer.model
                print("stage1_train_loss =", stage1_result.training_loss)

                stage2_dataset = load_chat_dataset(stage2_train_path, val_sft_path, tokenizer)
                stage2_output = output_root / "stage2_semantics"
                stage2_trainer = build_trainer(
                    model=model,
                    dataset=stage2_dataset,
                    tokenizer=tokenizer,
                    config=config,
                    output_dir=stage2_output,
                    num_epochs=float(config["stage2_epochs"]),
                    peft_config=None,
                )
                trainer_objects.append(stage2_trainer)
                stage2_result = stage2_trainer.train()
                stage2_trainer.save_model(str(output_root))
                print("stage2_train_loss =", stage2_result.training_loss)
            else:
                train_sft_path = ARTIFACT_DIR / f"{experiment_name}_train.jsonl"
                write_sft_split(train_records, train_sft_path, include_schema_definition=False)
                dataset = load_chat_dataset(train_sft_path, val_sft_path, tokenizer)
                trainer = build_trainer(
                    model=model,
                    dataset=dataset,
                    tokenizer=tokenizer,
                    config=config,
                    output_dir=output_root,
                    num_epochs=float(config["epochs"]),
                    peft_config=build_peft_config(config),
                )
                trainer_objects.append(trainer)
                train_result = trainer.train()
                trainer.save_model(str(output_root))
                print("train_loss =", train_result.training_loss)

            predictions = []
            for idx, record in enumerate(test_records, 1):
                prediction_text = generate_prediction_text(model, tokenizer, record)
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
                if idx % 25 == 0:
                    print(f"generated {idx} / {len(test_records)}")
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
                    "status": "completed",
                    "mode": config["mode"],
                    "raw_summary": raw_report["summary"],
                    "repaired_summary": repaired_report["summary"],
                }
            )
        finally:
            cleanup_model(*trainer_objects, model, tokenizer)

    lines = [
        "# Action Canonicalization Batch Summary",
        "",
        f"Skip completed: `{SKIP_COMPLETED}`",
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
                f"- mode: `{item['mode']}`",
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
