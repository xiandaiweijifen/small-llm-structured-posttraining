from __future__ import annotations

import gc
import json
import math
import random
import re
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
SOURCE_PREDICTION_EXPERIMENT = "qwen25_3b_stage14_target_allcore_x1_epoch1_lr5e5"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external_generalization"
SOURCE_NAME = "gorkemsevinc_customer_support_tickets"
EXTERNAL_TRAIN = EXTERNAL_DIR / f"{SOURCE_NAME}_train_reduced.jsonl"
EXTERNAL_VAL = EXTERNAL_DIR / f"{SOURCE_NAME}_val_reduced.jsonl"
EXTERNAL_TEST = EXTERNAL_DIR / f"{SOURCE_NAME}_test_reduced.jsonl"
ARTIFACT_DIR = PROJECT_ROOT / "data" / "stage16_external_frontier"
SUMMARY_PATH = PROJECT_ROOT / "docs" / "results" / "external_frontier_batch_summary.md"
TRAINING_PRESETS = [
    "ext2048_epoch3_lr1e4",
    "ext4096_epoch2_lr1e4",
    "extfull_epoch2_lr1e4",
    "ext2048_epoch3_lr5e5",
    "ext4096_epoch2_lr5e5",
    "extfull_epoch2_lr5e5",
]
POSTPROCESS_PRESETS = [
    "knn1_category_action",
    "knn3_category_action_majority",
    "knn3_priority_majority",
    "knn5_allcore_majority",
    "knn5_component_only",
]
SKIP_COMPLETED = True
INFERENCE_BATCH_SIZE = 32

TRAINING_CONFIGS = {
    "ext2048_epoch3_lr1e4": {
        "experiment_name": "qwen25_3b_stage16_ext2048_epoch3_lr1e4",
        "external_samples": 2048,
        "learning_rate": 1e-4,
        "epochs": 3,
        "batch_size": 16,
        "grad_accum": 4,
        "seed": 42,
    },
    "ext4096_epoch2_lr1e4": {
        "experiment_name": "qwen25_3b_stage16_ext4096_epoch2_lr1e4",
        "external_samples": 4096,
        "learning_rate": 1e-4,
        "epochs": 2,
        "batch_size": 16,
        "grad_accum": 4,
        "seed": 42,
    },
    "extfull_epoch2_lr1e4": {
        "experiment_name": "qwen25_3b_stage16_extfull_epoch2_lr1e4",
        "external_samples": None,
        "learning_rate": 1e-4,
        "epochs": 2,
        "batch_size": 16,
        "grad_accum": 4,
        "seed": 42,
    },
    "ext2048_epoch3_lr5e5": {
        "experiment_name": "qwen25_3b_stage16_ext2048_epoch3_lr5e5",
        "external_samples": 2048,
        "learning_rate": 5e-5,
        "epochs": 3,
        "batch_size": 16,
        "grad_accum": 4,
        "seed": 42,
    },
    "ext4096_epoch2_lr5e5": {
        "experiment_name": "qwen25_3b_stage16_ext4096_epoch2_lr5e5",
        "external_samples": 4096,
        "learning_rate": 5e-5,
        "epochs": 2,
        "batch_size": 16,
        "grad_accum": 4,
        "seed": 42,
    },
    "extfull_epoch2_lr5e5": {
        "experiment_name": "qwen25_3b_stage16_extfull_epoch2_lr5e5",
        "external_samples": None,
        "learning_rate": 5e-5,
        "epochs": 2,
        "batch_size": 16,
        "grad_accum": 4,
        "seed": 42,
    },
}

POSTPROCESS_CONFIGS = {
    "knn1_category_action": {
        "experiment_name": "qwen25_3b_stage16_knn1_category_action",
        "k": 1,
        "fields": ["category", "action"],
        "min_majority_ratio": 1.0,
    },
    "knn3_category_action_majority": {
        "experiment_name": "qwen25_3b_stage16_knn3_category_action_majority",
        "k": 3,
        "fields": ["category", "action"],
        "min_majority_ratio": 2 / 3,
    },
    "knn3_priority_majority": {
        "experiment_name": "qwen25_3b_stage16_knn3_priority_majority",
        "k": 3,
        "fields": ["priority"],
        "min_majority_ratio": 2 / 3,
    },
    "knn5_allcore_majority": {
        "experiment_name": "qwen25_3b_stage16_knn5_allcore_majority",
        "k": 5,
        "fields": ["category", "priority", "component", "action"],
        "min_majority_ratio": 0.6,
    },
    "knn5_component_only": {
        "experiment_name": "qwen25_3b_stage16_knn5_component_only",
        "k": 5,
        "fields": ["component"],
        "min_majority_ratio": 0.6,
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

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

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


def evaluate_and_write(gold_records: list[dict], predictions: list[dict], schema: dict, paths: dict[str, Path]):
    dump_jsonl(paths["prediction_path"], predictions)

    raw_sample_results = sample_eval_dicts(gold_records, predictions, schema)
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
    write_json_report(paths["raw_field_path"], analyze_field_errors(gold_records, predictions))

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
    repaired_sample_results = sample_eval_dicts(gold_records, repaired_predictions, schema)
    repaired_report = {"summary": summarize_from_dicts(repaired_sample_results), "per_sample": repaired_sample_results}
    write_json_report(paths["repaired_report_path"], repaired_report)
    write_json_report(paths["repaired_field_path"], analyze_field_errors(gold_records, repaired_predictions))
    return raw_report, repaired_report


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def build_retrieval_index(train_records: list[dict]) -> dict:
    token_counts: list[Counter] = []
    token_doc_freq: Counter = Counter()
    doc_norms: list[float] = []
    doc_labels: list[dict] = []
    inverted_index: dict[str, list[int]] = defaultdict(list)

    for idx, record in enumerate(train_records):
        counts = Counter(tokenize(record["input_text"]))
        token_counts.append(counts)
        for token in counts:
            token_doc_freq[token] += 1
            inverted_index[token].append(idx)
        doc_labels.append(
            {
                "category": record["target_json"]["category"],
                "priority": record["target_json"]["priority"],
                "component": record["target_json"]["affected_systems"][0]["component"],
            }
        )

    num_docs = len(train_records)
    idf = {token: math.log((1 + num_docs) / (1 + df)) + 1.0 for token, df in token_doc_freq.items()}
    for counts in token_counts:
        norm_sq = 0.0
        for token, count in counts.items():
            weight = count * idf[token]
            norm_sq += weight * weight
        doc_norms.append(math.sqrt(norm_sq) if norm_sq > 0 else 1.0)

    return {
        "token_counts": token_counts,
        "doc_norms": doc_norms,
        "idf": idf,
        "inverted_index": inverted_index,
        "doc_labels": doc_labels,
    }


def nearest_indices(query_text: str, retrieval_index: dict, k: int) -> list[int]:
    query_counts = Counter(tokenize(query_text))
    if not query_counts:
        return []
    query_norm_sq = sum((count * retrieval_index["idf"].get(token, 1.0)) ** 2 for token, count in query_counts.items())
    query_norm = math.sqrt(query_norm_sq) if query_norm_sq > 0 else 1.0
    scores: dict[int, float] = defaultdict(float)
    for token, q_count in query_counts.items():
        token_idf = retrieval_index["idf"].get(token)
        if token_idf is None:
            continue
        for doc_idx in retrieval_index["inverted_index"].get(token, []):
            scores[doc_idx] += q_count * retrieval_index["token_counts"][doc_idx].get(token, 0) * (token_idf ** 2)
    ranked = sorted(
        (
            (doc_idx, score / (query_norm * retrieval_index["doc_norms"][doc_idx]))
            for doc_idx, score in scores.items()
            if score > 0
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    return [doc_idx for doc_idx, _ in ranked[:k]]


def majority_vote(values: list[str]) -> tuple[str | None, float]:
    if not values:
        return None, 0.0
    counter = Counter(values)
    value, count = counter.most_common(1)[0]
    return value, count / len(values)


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


def apply_knn_postprocess(prediction_json: dict | None, source_record: dict, preset_name: str, retrieval_index: dict) -> dict | None:
    if not isinstance(prediction_json, dict):
        return prediction_json
    config = POSTPROCESS_CONFIGS[preset_name]
    updated = deepcopy(prediction_json)
    neighbor_ids = nearest_indices(source_record["input_text"], retrieval_index, int(config["k"]))
    if not neighbor_ids:
        return updated
    labels = [retrieval_index["doc_labels"][idx] for idx in neighbor_ids]

    if "category" in config["fields"]:
        category, ratio = majority_vote([item["category"] for item in labels])
        if category is not None and ratio >= float(config["min_majority_ratio"]):
            updated["category"] = category

    if "priority" in config["fields"]:
        priority, ratio = majority_vote([item["priority"] for item in labels])
        if priority is not None and ratio >= float(config["min_majority_ratio"]):
            updated["priority"] = priority

    if "component" in config["fields"]:
        component, ratio = majority_vote([item["component"] for item in labels])
        if component is not None and ratio >= float(config["min_majority_ratio"]):
            systems = updated.get("affected_systems")
            if isinstance(systems, list) and systems and isinstance(systems[0], dict):
                systems[0]["component"] = component

    if "action" in config["fields"]:
        refresh_action_from_category(updated)

    return updated


def build_train_subset(records: list[dict], sample_count: int | None, seed: int) -> list[dict]:
    if sample_count is None or sample_count >= len(records):
        return list(records)
    sampled = list(records)
    random.Random(seed).shuffle(sampled)
    return sampled[:sample_count]


def run():
    print("project_root =", PROJECT_ROOT)
    print("python =", sys.version)
    print("cuda_available =", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device =", torch.cuda.get_device_name(0))
        print("bf16_supported =", torch.cuda.is_bf16_supported())
    print("has_bnb_4bit =", HAS_BNB_4BIT)
    print("baseline_checkpoint_dir =", BASELINE_CHECKPOINT_DIR)
    print("source_prediction_experiment =", SOURCE_PREDICTION_EXPERIMENT)
    print("training_presets =", TRAINING_PRESETS)
    print("postprocess_presets =", POSTPROCESS_PRESETS)
    print("skip_completed =", SKIP_COMPLETED)
    print("inference_batch_size =", INFERENCE_BATCH_SIZE)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    external_train = load_jsonl(EXTERNAL_TRAIN)
    external_val = load_jsonl(EXTERNAL_VAL)
    external_test = load_jsonl(EXTERNAL_TEST)
    schema = get_schema(SCHEMA_NAME)

    val_sft_path = ARTIFACT_DIR / "external_val_sft.jsonl"
    write_sft_split(external_val, val_sft_path)
    training_results = []

    for preset_name in TRAINING_PRESETS:
        config = TRAINING_CONFIGS[preset_name]
        experiment_name = config["experiment_name"]
        paths = build_output_paths(experiment_name)
        print("\n" + "=" * 80)
        print("running training preset =", preset_name)
        print("=" * 80)

        if SKIP_COMPLETED and outputs_complete(paths):
            raw_report = json.loads(paths["raw_report_path"].read_text(encoding="utf-8"))
            repaired_report = json.loads(paths["repaired_report_path"].read_text(encoding="utf-8"))
            training_results.append(
                {
                    "preset_name": preset_name,
                    "experiment_name": experiment_name,
                    "status": "skipped_existing",
                    "train_count": config["external_samples"] if config["external_samples"] is not None else len(external_train),
                    "learning_rate": config["learning_rate"],
                    "epochs": config["epochs"],
                    "raw_summary": raw_report["summary"],
                    "repaired_summary": repaired_report["summary"],
                }
            )
            continue

        train_subset = build_train_subset(external_train, config["external_samples"], int(config["seed"]))
        print("train_size =", len(train_subset))
        print("bucket_counts =", bucket_counts(train_subset))

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
            raw_report, repaired_report = evaluate_and_write(external_test, predictions, schema, paths)
            training_results.append(
                {
                    "preset_name": preset_name,
                    "experiment_name": experiment_name,
                    "status": "completed",
                    "train_count": len(train_subset),
                    "learning_rate": config["learning_rate"],
                    "epochs": config["epochs"],
                    "raw_summary": raw_report["summary"],
                    "repaired_summary": repaired_report["summary"],
                }
            )
        finally:
            cleanup_model(trainer, model, tokenizer)

    retrieval_index = build_retrieval_index(external_train)
    write_json_report(
        ARTIFACT_DIR / "retrieval_index_summary.json",
        {
            "num_train_records": len(external_train),
            "avg_doc_tokens": sum(sum(counter.values()) for counter in retrieval_index["token_counts"]) / len(retrieval_index["token_counts"]),
            "num_idf_tokens": len(retrieval_index["idf"]),
        },
    )
    source_prediction_path = PROJECT_ROOT / "results" / "predictions" / f"{SOURCE_PREDICTION_EXPERIMENT}_test.jsonl"
    source_predictions = load_jsonl(source_prediction_path)
    test_by_id = {record["sample_id"]: record for record in external_test}
    postprocess_results = []

    for preset_name in POSTPROCESS_PRESETS:
        config = POSTPROCESS_CONFIGS[preset_name]
        experiment_name = config["experiment_name"]
        paths = build_output_paths(experiment_name)
        print("\n" + "=" * 80)
        print("running postprocess preset =", preset_name)
        print("=" * 80)

        if SKIP_COMPLETED and outputs_complete(paths):
            raw_report = json.loads(paths["raw_report_path"].read_text(encoding="utf-8"))
            repaired_report = json.loads(paths["repaired_report_path"].read_text(encoding="utf-8"))
            postprocess_results.append(
                {
                    "preset_name": preset_name,
                    "experiment_name": experiment_name,
                    "status": "skipped_existing",
                    "k": config["k"],
                    "fields": ",".join(config["fields"]),
                    "raw_summary": raw_report["summary"],
                    "repaired_summary": repaired_report["summary"],
                }
            )
            continue

        predictions = []
        for record in source_predictions:
            source_record = test_by_id[record["sample_id"]]
            prediction_json = apply_knn_postprocess(record.get("prediction_json"), source_record, preset_name, retrieval_index)
            predictions.append(
                {
                    **record,
                    "prediction_json": prediction_json,
                    "metadata": {
                        **record.get("metadata", {}),
                        "source_experiment": SOURCE_PREDICTION_EXPERIMENT,
                        "postprocess_preset": preset_name,
                    },
                }
            )
        raw_report, repaired_report = evaluate_and_write(external_test, predictions, schema, paths)
        postprocess_results.append(
            {
                "preset_name": preset_name,
                "experiment_name": experiment_name,
                "status": "completed",
                "k": config["k"],
                "fields": ",".join(config["fields"]),
                "raw_summary": raw_report["summary"],
                "repaired_summary": repaired_report["summary"],
            }
        )

    lines = [
        "# External Frontier Batch Summary",
        "",
        f"- baseline checkpoint: `{BASELINE_CHECKPOINT_DIR}`",
        f"- source prediction experiment for postprocess: `{SOURCE_PREDICTION_EXPERIMENT}`",
        f"- external train: `{EXTERNAL_TRAIN}`",
        f"- external val: `{EXTERNAL_VAL}`",
        f"- external test: `{EXTERNAL_TEST}`",
        f"- skip completed: `{SKIP_COMPLETED}`",
        f"- inference batch size: `{INFERENCE_BATCH_SIZE}`",
        "",
        "## Large-Scale External Adaptation Runs",
        "",
    ]
    for item in training_results:
        raw = item["raw_summary"]
        repaired = item["repaired_summary"]
        lines.extend(
            [
                f"### {item['preset_name']}",
                "",
                f"- experiment: `{item['experiment_name']}`",
                f"- status: `{item['status']}`",
                f"- train count: `{item['train_count']}`",
                f"- learning rate: `{item['learning_rate']}`",
                f"- epochs: `{item['epochs']}`",
                f"- raw field exact match: `{raw['field_exact_match']:.4f}`",
                f"- raw end-to-end exact match: `{raw['end_to_end_exact_match']:.4f}`",
                f"- repaired field exact match: `{repaired['field_exact_match']:.4f}`",
                f"- repaired end-to-end exact match: `{repaired['end_to_end_exact_match']:.4f}`",
                "",
            ]
        )

    lines.extend(["## Retrieval-Guided Postprocess Runs", ""])
    for item in postprocess_results:
        raw = item["raw_summary"]
        repaired = item["repaired_summary"]
        lines.extend(
            [
                f"### {item['preset_name']}",
                "",
                f"- experiment: `{item['experiment_name']}`",
                f"- status: `{item['status']}`",
                f"- k: `{item['k']}`",
                f"- fields: `{item['fields']}`",
                f"- raw field exact match: `{raw['field_exact_match']:.4f}`",
                f"- raw end-to-end exact match: `{raw['end_to_end_exact_match']:.4f}`",
                f"- repaired field exact match: `{repaired['field_exact_match']:.4f}`",
                f"- repaired end-to-end exact match: `{repaired['end_to_end_exact_match']:.4f}`",
                "",
            ]
        )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("summary_path =", SUMMARY_PATH)


if __name__ == "__main__":
    run()
