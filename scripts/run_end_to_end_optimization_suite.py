from __future__ import annotations

import gc
import json
import random
import sys
from collections import defaultdict
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
from src.inference.repair import repair_prediction
from src.schemas.registry import get_schema
from src.training.formatters import DEFAULT_SYSTEM_PROMPT, build_user_prompt, convert_to_sft_records

BASE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
SCHEMA_NAME = "ticket_schema_v1_reduced"
MINER_CHECKPOINT_DIR = PROJECT_ROOT / "results" / "checkpoints" / "qwen25_3b_stage2_structure_then_semantics_v1"
ARTIFACT_DIR = PROJECT_ROOT / "data" / "stage3_end_to_end_optimization"
SUMMARY_PATH = PROJECT_ROOT / "docs" / "results" / "end_to_end_optimization_batch_summary.md"

RUN_PRESETS = [
    "sts_v2_hard_only_x4_epoch1_lr5e5",
    "sts_v2_full_plus_hard_x2_epoch2_lr1e4",
    "sts_v2_full_plus_hard_x3_epoch2_lr1e4",
    "sts_v2_full_plus_hard_x2_epoch2_lr5e5",
]
SKIP_COMPLETED = True
SKIP_EXISTING_MINING = True

PRESETS = {
    "sts_v2_hard_only_x4_epoch1_lr5e5": {
        "experiment_name": "qwen25_3b_stage3_sts_v2_hard_only_x4_epoch1_lr5e5",
        "base_model": BASE_MODEL_NAME,
        "schema_name": SCHEMA_NAME,
        "init_from_checkpoint": MINER_CHECKPOINT_DIR,
        "learning_rate": 5e-5,
        "epochs": 1,
        "batch_size": 8,
        "grad_accum": 4,
        "seed": 42,
        "data_mode": "hard_only",
        "hard_repeat": 4,
    },
    "sts_v2_full_plus_hard_x2_epoch2_lr1e4": {
        "experiment_name": "qwen25_3b_stage3_sts_v2_full_plus_hard_x2_epoch2_lr1e4",
        "base_model": BASE_MODEL_NAME,
        "schema_name": SCHEMA_NAME,
        "init_from_checkpoint": MINER_CHECKPOINT_DIR,
        "learning_rate": 1e-4,
        "epochs": 2,
        "batch_size": 8,
        "grad_accum": 4,
        "seed": 42,
        "data_mode": "full_plus_hard",
        "hard_repeat": 2,
    },
    "sts_v2_full_plus_hard_x3_epoch2_lr1e4": {
        "experiment_name": "qwen25_3b_stage3_sts_v2_full_plus_hard_x3_epoch2_lr1e4",
        "base_model": BASE_MODEL_NAME,
        "schema_name": SCHEMA_NAME,
        "init_from_checkpoint": MINER_CHECKPOINT_DIR,
        "learning_rate": 1e-4,
        "epochs": 2,
        "batch_size": 8,
        "grad_accum": 4,
        "seed": 42,
        "data_mode": "full_plus_hard",
        "hard_repeat": 3,
    },
    "sts_v2_full_plus_hard_x2_epoch2_lr5e5": {
        "experiment_name": "qwen25_3b_stage3_sts_v2_full_plus_hard_x2_epoch2_lr5e5",
        "base_model": BASE_MODEL_NAME,
        "schema_name": SCHEMA_NAME,
        "init_from_checkpoint": MINER_CHECKPOINT_DIR,
        "learning_rate": 5e-5,
        "epochs": 2,
        "batch_size": 8,
        "grad_accum": 4,
        "seed": 42,
        "data_mode": "full_plus_hard",
        "hard_repeat": 2,
    },
}

IMPORTANT_FIELDS = [
    "actions_requested[0].action",
    "affected_systems[0].component",
    "category",
    "priority",
]

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


def bucket_counts(records: list[dict]) -> dict[str, int]:
    counts: defaultdict[str, int] = defaultdict(int)
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


def load_tokenizer(base_model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(base_model_name: str):
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=BNB_CONFIG,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    return model


def load_trainable_adapter_model(base_model_name: str, checkpoint_dir: Path):
    base_model = load_base_model(base_model_name)
    try:
        model = PeftModel.from_pretrained(base_model, str(checkpoint_dir), is_trainable=True)
    except TypeError:
        model = PeftModel.from_pretrained(base_model, str(checkpoint_dir))
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
    model.config.use_cache = False
    return model


def build_training_args(config: dict, output_dir: Path, num_epochs: int):
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


def build_trainer(model, dataset, tokenizer, config: dict, output_dir: Path, num_epochs: int):
    return SFTTrainer(
        model=model,
        args=build_training_args(config, output_dir, num_epochs),
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


def generate_prediction_text(model, tokenizer, record: dict) -> str:
    messages = build_inference_messages(record)
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, **GENERATION_KWARGS)
    generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def extract_field(obj: dict | None, field_name: str):
    if obj is None:
        return None
    current = obj
    for part in field_name.split("."):
        if "[" in part and part.endswith("]"):
            key, index_text = part[:-1].split("[")
            if not isinstance(current, dict):
                return None
            current = current.get(key)
            if not isinstance(current, list):
                return None
            index = int(index_text)
            if index >= len(current):
                return None
            current = current[index]
        else:
            if not isinstance(current, dict):
                return None
            current = current.get(part)
    return current


def analyze_train_hardness(records: list[dict], predictions: list[dict]):
    predictions_by_id = {item["sample_id"]: item for item in predictions}
    mined_rows: list[dict] = []
    hard_records: list[dict] = []
    for record in records:
        pred = predictions_by_id[record["sample_id"]]
        mismatched_fields = [
            field_name
            for field_name in IMPORTANT_FIELDS
            if extract_field(record["target_json"], field_name) != extract_field(pred.get("prediction_json"), field_name)
        ]
        mined_rows.append(
            {
                "sample_id": record["sample_id"],
                "complexity_bucket": record["complexity_bucket"],
                "mismatch_count": len(mismatched_fields),
                "mismatched_fields": mismatched_fields,
            }
        )
        if mismatched_fields:
            hard_records.append(record)
    return mined_rows, hard_records


def sample_eval_dicts(gold_records: list[dict], pred_records: list[dict], schema: dict):
    predictions_by_id = {record["sample_id"]: record for record in pred_records}
    results: list[dict] = []
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
    required = [
        paths["prediction_path"],
        paths["repaired_prediction_path"],
        paths["raw_report_path"],
        paths["repaired_report_path"],
        paths["raw_field_path"],
        paths["repaired_field_path"],
    ]
    return all(path.exists() for path in required)


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
    print("miner_checkpoint_dir =", MINER_CHECKPOINT_DIR)
    print("scheduled_presets =", RUN_PRESETS)
    print("skip_completed =", SKIP_COMPLETED)
    print("skip_existing_mining =", SKIP_EXISTING_MINING)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    train_records = load_jsonl(PROJECT_ROOT / "data" / "reduced" / "phase1_train_reduced.jsonl")
    val_records = load_jsonl(PROJECT_ROOT / "data" / "reduced" / "phase1_val_reduced.jsonl")
    test_records = load_jsonl(PROJECT_ROOT / "data" / "reduced" / "phase1_test_reduced.jsonl")
    print("train =", len(train_records))
    print("val =", len(val_records))
    print("test =", len(test_records))

    mining_predictions_path = ARTIFACT_DIR / "structure_then_semantics_v1_train_predictions.jsonl"
    mining_summary_path = ARTIFACT_DIR / "structure_then_semantics_v1_train_mining_summary.json"
    hard_subset_path = ARTIFACT_DIR / "structure_then_semantics_v1_hard_subset.jsonl"
    mined_rows_path = ARTIFACT_DIR / "structure_then_semantics_v1_hard_rows.jsonl"

    if SKIP_EXISTING_MINING and mining_summary_path.exists() and hard_subset_path.exists() and mined_rows_path.exists():
        mining_summary = json.loads(mining_summary_path.read_text(encoding="utf-8"))
        hard_train_records = load_jsonl(hard_subset_path)
    else:
        tokenizer = load_tokenizer(BASE_MODEL_NAME)
        miner_model = load_trainable_adapter_model(BASE_MODEL_NAME, MINER_CHECKPOINT_DIR)
        miner_model.eval()
        mined_predictions = []
        for idx, record in enumerate(train_records, 1):
            prediction_text = generate_prediction_text(miner_model, tokenizer, record)
            try:
                prediction_json = json.loads(prediction_text)
            except json.JSONDecodeError:
                prediction_json = None
            mined_predictions.append(
                {
                    "sample_id": record["sample_id"],
                    "prediction_text": prediction_text,
                    "prediction_json": prediction_json,
                }
            )
            if idx % 100 == 0:
                print(f"mined {idx} / {len(train_records)} train records")
        dump_jsonl(mining_predictions_path, mined_predictions)
        mined_rows, hard_train_records = analyze_train_hardness(train_records, mined_predictions)
        dump_jsonl(hard_subset_path, hard_train_records)
        dump_jsonl(mined_rows_path, mined_rows)
        mining_summary = {
            "num_train_records": len(train_records),
            "num_hard_records": len(hard_train_records),
            "hard_fraction": len(hard_train_records) / len(train_records),
            "hard_bucket_counts": bucket_counts(hard_train_records),
        }
        mining_summary_path.write_text(json.dumps(mining_summary, indent=2), encoding="utf-8")
        cleanup_model(miner_model, tokenizer)

    print("mining_summary =")
    print(json.dumps(mining_summary, indent=2))

    schema = get_schema(SCHEMA_NAME)
    val_sft_path = ARTIFACT_DIR / "phase1_val_reduced_sft.jsonl"
    write_sft_split(val_records, val_sft_path)
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
            result = {
                "preset_name": preset_name,
                "experiment_name": experiment_name,
                "status": "skipped_existing",
                "data_mode": config["data_mode"],
                "learning_rate": config["learning_rate"],
                "epochs": config["epochs"],
                "hard_repeat": config["hard_repeat"],
                "raw_summary": raw_report["summary"],
                "repaired_summary": repaired_report["summary"],
            }
            batch_run_results.append(result)
            print(json.dumps(result["raw_summary"], indent=2))
            continue

        if config["data_mode"] == "hard_only":
            train_subset = hard_train_records * int(config["hard_repeat"])
        elif config["data_mode"] == "full_plus_hard":
            train_subset = list(train_records) + (hard_train_records * int(config["hard_repeat"]))
        else:
            raise ValueError(f"Unsupported data_mode: {config['data_mode']}")
        random.Random(int(config["seed"])).shuffle(train_subset)
        print("train_subset_size =", len(train_subset))
        print("train_subset_bucket_counts =", bucket_counts(train_subset))

        train_sft_path = ARTIFACT_DIR / f"{experiment_name}_train.jsonl"
        write_sft_split(train_subset, train_sft_path)

        tokenizer = load_tokenizer(config["base_model"])
        model = load_trainable_adapter_model(config["base_model"], Path(config["init_from_checkpoint"]))
        trainer = None
        try:
            dataset = load_chat_dataset(train_sft_path, val_sft_path, tokenizer)
            output_root = paths["checkpoint_dir"]
            output_root.mkdir(parents=True, exist_ok=True)
            trainer = build_trainer(model, dataset, tokenizer, config, output_root, int(config["epochs"]))
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
                        "metadata": {"model_name": config["base_model"], "experiment_id": experiment_name},
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
                    "data_mode": config["data_mode"],
                    "learning_rate": config["learning_rate"],
                    "epochs": config["epochs"],
                    "hard_repeat": config["hard_repeat"],
                    "raw_summary": raw_report["summary"],
                    "repaired_summary": repaired_report["summary"],
                }
            )
            print(json.dumps(raw_report["summary"], indent=2))
        finally:
            cleanup_model(trainer, model, tokenizer)

    lines = [
        "# End-to-End Optimization Batch Summary",
        "",
        f"Skip completed: `{SKIP_COMPLETED}`",
        f"Hard mining checkpoint: `{MINER_CHECKPOINT_DIR}`",
        "",
        "## Hard Mining Summary",
        "",
        f"- num train records: `{mining_summary['num_train_records']}`",
        f"- num hard records: `{mining_summary['num_hard_records']}`",
        f"- hard fraction: `{mining_summary['hard_fraction']:.4f}`",
        f"- hard bucket counts: `{mining_summary['hard_bucket_counts']}`",
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
                f"- data mode: `{item['data_mode']}`",
                f"- learning rate: `{item['learning_rate']}`",
                f"- epochs: `{item['epochs']}`",
                f"- hard repeat: `{item['hard_repeat']}`",
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
