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
from src.inference.batch_generate import batched_generate_texts
from src.inference.repair import repair_prediction
from src.schemas.registry import get_schema
from src.training.formatters import DEFAULT_SYSTEM_PROMPT, build_user_prompt, convert_to_sft_records

BASE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
SCHEMA_NAME = "ticket_schema_v1_reduced"
BASELINE_CHECKPOINT_DIR = PROJECT_ROOT / "results" / "checkpoints" / "qwen25_3b_stage13_ext1024_epoch3_lr1e4"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external_generalization"
SOURCE_NAME = "gorkemsevinc_customer_support_tickets"
EXTERNAL_TRAIN = EXTERNAL_DIR / f"{SOURCE_NAME}_train_reduced.jsonl"
EXTERNAL_VAL = EXTERNAL_DIR / f"{SOURCE_NAME}_val_reduced.jsonl"
EXTERNAL_TEST = EXTERNAL_DIR / f"{SOURCE_NAME}_test_reduced.jsonl"
ARTIFACT_DIR = PROJECT_ROOT / "data" / "stage14_external_targeted_adaptation"
SUMMARY_PATH = PROJECT_ROOT / "docs" / "results" / "external_targeted_adaptation_batch_summary.md"
RUN_PRESETS = [
    "target_component_category_x1_epoch1_lr5e5",
    "target_component_category_x2_epoch1_lr5e5",
    "target_category_priority_x1_epoch1_lr5e5",
    "target_allcore_x1_epoch1_lr5e5",
    "target_allcore_x1_epoch1_lr1e4",
]
SKIP_COMPLETED = True
SKIP_EXISTING_MINING = True
INFERENCE_BATCH_SIZE = 24

PRESETS = {
    "target_component_category_x1_epoch1_lr5e5": {
        "experiment_name": "qwen25_3b_stage14_target_component_category_x1_epoch1_lr5e5",
        "subset_name": "component_or_category",
        "learning_rate": 5e-5,
        "epochs": 1,
        "target_repeat": 1,
        "batch_size": 12,
        "grad_accum": 4,
        "seed": 42,
    },
    "target_component_category_x2_epoch1_lr5e5": {
        "experiment_name": "qwen25_3b_stage14_target_component_category_x2_epoch1_lr5e5",
        "subset_name": "component_or_category",
        "learning_rate": 5e-5,
        "epochs": 1,
        "target_repeat": 2,
        "batch_size": 12,
        "grad_accum": 4,
        "seed": 42,
    },
    "target_category_priority_x1_epoch1_lr5e5": {
        "experiment_name": "qwen25_3b_stage14_target_category_priority_x1_epoch1_lr5e5",
        "subset_name": "category_or_priority",
        "learning_rate": 5e-5,
        "epochs": 1,
        "target_repeat": 1,
        "batch_size": 12,
        "grad_accum": 4,
        "seed": 42,
    },
    "target_allcore_x1_epoch1_lr5e5": {
        "experiment_name": "qwen25_3b_stage14_target_allcore_x1_epoch1_lr5e5",
        "subset_name": "all_core_fields",
        "learning_rate": 5e-5,
        "epochs": 1,
        "target_repeat": 1,
        "batch_size": 12,
        "grad_accum": 4,
        "seed": 42,
    },
    "target_allcore_x1_epoch1_lr1e4": {
        "experiment_name": "qwen25_3b_stage14_target_allcore_x1_epoch1_lr1e4",
        "subset_name": "all_core_fields",
        "learning_rate": 1e-4,
        "epochs": 1,
        "target_repeat": 1,
        "batch_size": 12,
        "grad_accum": 4,
        "seed": 42,
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
    kwargs = {"device_map": "auto", "trust_remote_code": True}
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
    return mined_rows


def build_targeted_subsets(train_records: list[dict], mined_rows: list[dict]) -> dict[str, list[dict]]:
    by_id = {record["sample_id"]: record for record in train_records}
    subsets = {
        "component_or_category": [],
        "category_or_priority": [],
        "all_core_fields": [],
    }
    for row in mined_rows:
        if row["mismatch_count"] == 0:
            continue
        fields = set(row["mismatched_fields"])
        record = by_id[row["sample_id"]]
        if fields.intersection({"affected_systems[0].component", "category"}):
            subsets["component_or_category"].append(record)
        if fields.intersection({"category", "priority"}):
            subsets["category_or_priority"].append(record)
        if fields.intersection(set(IMPORTANT_FIELDS)):
            subsets["all_core_fields"].append(record)
    return subsets


def generate_predictions(model, tokenizer, records: list[dict]) -> list[dict]:
    prediction_texts = batched_generate_texts(
        model=model,
        tokenizer=tokenizer,
        records=records,
        build_messages=build_inference_messages,
        generation_kwargs=GENERATION_KWARGS,
        batch_size=INFERENCE_BATCH_SIZE,
    )
    predictions = []
    for record, prediction_text in zip(records, prediction_texts, strict=True):
        try:
            prediction_json = json.loads(prediction_text)
        except json.JSONDecodeError:
            prediction_json = None
        predictions.append(
            {
                "sample_id": record["sample_id"],
                "prediction_text": prediction_text,
                "prediction_json": prediction_json,
                "metadata": {"model_name": BASE_MODEL_NAME},
            }
        )
    return predictions


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
    print("skip_existing_mining =", SKIP_EXISTING_MINING)
    print("inference_batch_size =", INFERENCE_BATCH_SIZE)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    external_train = load_jsonl(EXTERNAL_TRAIN)
    external_val = load_jsonl(EXTERNAL_VAL)
    external_test = load_jsonl(EXTERNAL_TEST)
    schema = get_schema(SCHEMA_NAME)

    mining_predictions_path = ARTIFACT_DIR / "external_train_predictions.jsonl"
    mining_rows_path = ARTIFACT_DIR / "external_train_hard_rows.jsonl"
    mining_summary_path = ARTIFACT_DIR / "external_train_mining_summary.json"

    if SKIP_EXISTING_MINING and mining_predictions_path.exists() and mining_rows_path.exists() and mining_summary_path.exists():
        mined_rows = load_jsonl(mining_rows_path)
        print("reusing existing external mining artifacts")
    else:
        tokenizer = load_tokenizer()
        model = load_trainable_adapter_model(BASELINE_CHECKPOINT_DIR)
        try:
            mining_predictions = generate_predictions(model, tokenizer, external_train)
            dump_jsonl(mining_predictions_path, mining_predictions)
            mined_rows = analyze_train_hardness(external_train, mining_predictions)
            dump_jsonl(mining_rows_path, mined_rows)
            summary = {
                "num_train_records": len(external_train),
                "num_hard_records": sum(1 for row in mined_rows if row["mismatch_count"] > 0),
                "hard_fraction": sum(1 for row in mined_rows if row["mismatch_count"] > 0) / len(external_train),
                "avg_mismatch_count": sum(row["mismatch_count"] for row in mined_rows) / len(mined_rows),
                "field_counts": {
                    field: sum(1 for row in mined_rows if field in row["mismatched_fields"])
                    for field in IMPORTANT_FIELDS
                },
            }
            write_json_report(mining_summary_path, summary)
        finally:
            cleanup_model(model, tokenizer)

    targeted_subsets = build_targeted_subsets(external_train, mined_rows)
    subset_summary = {
        name: {"num_records": len(records), "bucket_counts": bucket_counts(records)}
        for name, records in targeted_subsets.items()
    }
    print("targeted_subset_summary =")
    print(json.dumps(subset_summary, indent=2))

    val_sft_path = ARTIFACT_DIR / "external_val_sft.jsonl"
    write_sft_split(external_val, val_sft_path)
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
                    "subset_name": config["subset_name"],
                    "learning_rate": config["learning_rate"],
                    "epochs": config["epochs"],
                    "target_repeat": config["target_repeat"],
                    "raw_summary": raw_report["summary"],
                    "repaired_summary": repaired_report["summary"],
                }
            )
            continue

        target_subset = targeted_subsets[config["subset_name"]]
        train_subset = list(external_train) + (target_subset * int(config["target_repeat"]))
        random.Random(int(config["seed"])).shuffle(train_subset)
        print("target_subset_size =", len(target_subset))
        print("mixed_train_size =", len(train_subset))
        print("mixed_train_bucket_counts =", bucket_counts(train_subset))

        train_sft_path = ARTIFACT_DIR / f"{experiment_name}_train.jsonl"
        write_sft_split(train_subset, train_sft_path)

        tokenizer = load_tokenizer()
        model = load_trainable_adapter_model(BASELINE_CHECKPOINT_DIR)
        trainer = None
        try:
            dataset = load_chat_dataset(train_sft_path, val_sft_path, tokenizer)
            output_root = paths["checkpoint_dir"]
            output_root.mkdir(parents=True, exist_ok=True)
            trainer = build_trainer(model, dataset, tokenizer, config, output_root)
            train_result = trainer.train()
            trainer.save_model(str(output_root))
            print("train_loss =", train_result.training_loss)

            predictions = generate_predictions(model, tokenizer, external_test)
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
            repaired_report = {"summary": summarize_from_dicts(repaired_sample_results), "per_sample": repaired_sample_results}
            write_json_report(paths["repaired_report_path"], repaired_report)
            write_json_report(paths["repaired_field_path"], analyze_field_errors(external_test, repaired_predictions))

            batch_run_results.append(
                {
                    "preset_name": preset_name,
                    "experiment_name": experiment_name,
                    "status": "completed",
                    "subset_name": config["subset_name"],
                    "learning_rate": config["learning_rate"],
                    "epochs": config["epochs"],
                    "target_repeat": config["target_repeat"],
                    "raw_summary": raw_report["summary"],
                    "repaired_summary": repaired_report["summary"],
                }
            )
        finally:
            cleanup_model(trainer, model, tokenizer)

    lines = [
        "# External Targeted Adaptation Batch Summary",
        "",
        f"- baseline checkpoint: `{BASELINE_CHECKPOINT_DIR}`",
        f"- external train: `{EXTERNAL_TRAIN}`",
        f"- external val: `{EXTERNAL_VAL}`",
        f"- external test: `{EXTERNAL_TEST}`",
        f"- skip completed: `{SKIP_COMPLETED}`",
        f"- skip existing mining: `{SKIP_EXISTING_MINING}`",
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
                f"- subset: `{item['subset_name']}`",
                f"- learning rate: `{item['learning_rate']}`",
                f"- epochs: `{item['epochs']}`",
                f"- target repeat: `{item['target_repeat']}`",
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
