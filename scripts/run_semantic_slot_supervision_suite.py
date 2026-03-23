from __future__ import annotations

import gc
import json
import sys
from collections import Counter, defaultdict
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
from src.evaluation.metrics import evaluate_sample, summarize_results
from src.evaluation.reporting import group_sample_results, write_json_report
from src.inference.batch_generate import batched_generate_texts
from src.inference.repair import repair_prediction
from src.schemas.registry import get_schema
from src.training.formatters import DEFAULT_SYSTEM_PROMPT

BASE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
EVAL_SCHEMA_NAME = "ticket_schema_v1_reduced"
ARTIFACT_DIR = PROJECT_ROOT / "data" / "stage12_semantic_slot_supervision"
SUMMARY_PATH = PROJECT_ROOT / "docs" / "results" / "semantic_slot_supervision_batch_summary.md"
RUN_PRESETS = [
    "semantic_slot_single_stage_epoch7_lr2e4",
    "semantic_slot_single_stage_epoch9_lr2e4",
    "semantic_slot_single_stage_epoch11_lr2e4",
    "semantic_slot_single_stage_epoch9_lr1e4",
    "semantic_slot_structure_then_semantics_stage2_epoch9",
    "semantic_slot_structure_then_semantics_stage2_epoch11",
    "semantic_slot_structure_then_semantics_stage2_epoch9_lr1e4",
]
SKIP_COMPLETED = True

ACTION_PREFIX_BY_CATEGORY = {
    "task": "Handle request",
    "bug": "Investigate issue",
    "feature": "Review and plan request",
    "incident": "Investigate and mitigate incident",
    "question": "Answer and clarify",
}

ACTION_TEMPLATE_BY_CATEGORY = {
    "task": "ACTION_HANDLE_REQUEST",
    "bug": "ACTION_INVESTIGATE_ISSUE",
    "feature": "ACTION_REVIEW_AND_PLAN",
    "incident": "ACTION_INVESTIGATE_AND_MITIGATE_INCIDENT",
    "question": "ACTION_ANSWER_AND_CLARIFY",
}

CATEGORY_BY_ACTION_TEMPLATE = {value: key for key, value in ACTION_TEMPLATE_BY_CATEGORY.items()}

PRESETS = {
    "semantic_slot_single_stage_epoch7_lr2e4": {
        "experiment_name": "qwen25_3b_stage12_semantic_slot_single_stage_epoch7_lr2e4",
        "mode": "full",
        "learning_rate": 2e-4,
        "epochs": 7,
        "batch_size": 12,
        "grad_accum": 4,
        "seed": 42,
        "rank": 16,
        "alpha": 32,
        "dropout": 0.05,
    },
    "semantic_slot_single_stage_epoch9_lr2e4": {
        "experiment_name": "qwen25_3b_stage12_semantic_slot_single_stage_epoch9_lr2e4",
        "mode": "full",
        "learning_rate": 2e-4,
        "epochs": 9,
        "batch_size": 12,
        "grad_accum": 4,
        "seed": 42,
        "rank": 16,
        "alpha": 32,
        "dropout": 0.05,
    },
    "semantic_slot_single_stage_epoch11_lr2e4": {
        "experiment_name": "qwen25_3b_stage12_semantic_slot_single_stage_epoch11_lr2e4",
        "mode": "full",
        "learning_rate": 2e-4,
        "epochs": 11,
        "batch_size": 12,
        "grad_accum": 4,
        "seed": 42,
        "rank": 16,
        "alpha": 32,
        "dropout": 0.05,
    },
    "semantic_slot_single_stage_epoch9_lr1e4": {
        "experiment_name": "qwen25_3b_stage12_semantic_slot_single_stage_epoch9_lr1e4",
        "mode": "full",
        "learning_rate": 1e-4,
        "epochs": 9,
        "batch_size": 12,
        "grad_accum": 4,
        "seed": 42,
        "rank": 16,
        "alpha": 32,
        "dropout": 0.05,
    },
    "semantic_slot_structure_then_semantics_stage2_epoch9": {
        "experiment_name": "qwen25_3b_stage12_semantic_slot_structure_then_semantics_stage2_epoch9",
        "mode": "two_stage_structure_semantics",
        "learning_rate": 2e-4,
        "stage1_epochs": 1,
        "stage2_epochs": 9,
        "batch_size": 12,
        "grad_accum": 4,
        "seed": 42,
        "rank": 16,
        "alpha": 32,
        "dropout": 0.05,
        "structure_stage_buckets": ["simple", "medium"],
    },
    "semantic_slot_structure_then_semantics_stage2_epoch11": {
        "experiment_name": "qwen25_3b_stage12_semantic_slot_structure_then_semantics_stage2_epoch11",
        "mode": "two_stage_structure_semantics",
        "learning_rate": 2e-4,
        "stage1_epochs": 1,
        "stage2_epochs": 11,
        "batch_size": 12,
        "grad_accum": 4,
        "seed": 42,
        "rank": 16,
        "alpha": 32,
        "dropout": 0.05,
        "structure_stage_buckets": ["simple", "medium"],
    },
    "semantic_slot_structure_then_semantics_stage2_epoch9_lr1e4": {
        "experiment_name": "qwen25_3b_stage12_semantic_slot_structure_then_semantics_stage2_epoch9_lr1e4",
        "mode": "two_stage_structure_semantics",
        "learning_rate": 1e-4,
        "stage1_epochs": 1,
        "stage2_epochs": 9,
        "batch_size": 12,
        "grad_accum": 4,
        "seed": 42,
        "rank": 16,
        "alpha": 32,
        "dropout": 0.05,
        "structure_stage_buckets": ["simple", "medium"],
    },
}

GENERATION_KWARGS = {
    "max_new_tokens": 320,
    "do_sample": False,
    "temperature": 1.0,
    "top_p": 1.0,
}
INFERENCE_BATCH_SIZE = 24

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


def to_canonical_reduced_record(record: dict, component_map: dict[str, str]) -> dict:
    updated = deepcopy(record)
    target = updated["target_json"]
    name = target["affected_systems"][0]["name"]
    if name in component_map:
        target["affected_systems"][0]["component"] = component_map[name]
    category = target["category"]
    target["actions_requested"][0]["action"] = f"{ACTION_PREFIX_BY_CATEGORY[category]}: {target['summary']}"
    return updated


def to_semantic_slot_wrapper(record: dict, component_map: dict[str, str]) -> dict:
    canonical = to_canonical_reduced_record(record, component_map)
    target = canonical["target_json"]
    return {
        "semantic_slots": {
            "category": target["category"],
            "priority": target["priority"],
            "component": target["affected_systems"][0]["component"],
            "action_template": ACTION_TEMPLATE_BY_CATEGORY[target["category"]],
            "requires_followup": target["requires_followup"],
        },
        "target_json": deepcopy(target),
    }


def to_semantic_slot_record(record: dict, component_map: dict[str, str]) -> dict:
    updated = deepcopy(record)
    updated["target_json"] = to_semantic_slot_wrapper(record, component_map)
    updated["schema_name"] = "ticket_semantic_slot_wrapper_v1"
    return updated


def build_semantic_slot_prompt(input_text: str, include_schema_definition: bool = False) -> str:
    prompt = (
        "Task: extract a structured record for ticket_structured_output.\n"
        "Return a JSON object only.\n"
        "The JSON must contain exactly two top-level keys:\n"
        '1. "semantic_slots": object with keys '
        '"category", "priority", "component", "action_template", "requires_followup"\n'
        '2. "target_json": the final reduced-schema ticket JSON object\n'
    )
    if include_schema_definition:
        prompt += (
            "Semantic slot constraints:\n"
            '- category in ["task","bug","feature","incident","question"]\n'
            '- priority in ["low","medium","high","urgent"]\n'
            '- action_template in ['
            '"ACTION_HANDLE_REQUEST",'
            '"ACTION_INVESTIGATE_ISSUE",'
            '"ACTION_REVIEW_AND_PLAN",'
            '"ACTION_INVESTIGATE_AND_MITIGATE_INCIDENT",'
            '"ACTION_ANSWER_AND_CLARIFY"]\n'
            '- target_json must match the reduced ticket schema\n'
        )
    prompt += f"Input text:\n{input_text}"
    return prompt


def to_sft_record(sample: dict, include_schema_definition: bool = False) -> dict:
    assistant_content = json.dumps(sample["target_json"], ensure_ascii=False)
    return {
        "sample_id": sample["sample_id"],
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_semantic_slot_prompt(
                    sample["input_text"],
                    include_schema_definition=include_schema_definition,
                ),
            },
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "task_name": sample["task_name"],
            "schema_name": sample["schema_name"],
            "complexity_bucket": sample["complexity_bucket"],
            "source_type": sample["metadata"]["source_type"],
            "split": sample["metadata"].get("split"),
            "is_synthetic": sample["metadata"]["is_synthetic"],
            "semantic_slot_wrapper": True,
            "schema_conditioned_prompt": include_schema_definition,
        },
    }


def write_sft_split(records: list[dict], path: Path, include_schema_definition: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dump_jsonl(path, [to_sft_record(record, include_schema_definition=include_schema_definition) for record in records])


def load_chat_dataset(train_path: Path, val_path: Path, tokenizer):
    dataset = load_dataset("json", data_files={"train": str(train_path), "validation": str(val_path)})

    def format_chat_example(example):
        example["text"] = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return example

    return dataset.map(format_chat_example)


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model():
    load_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if BNB_CONFIG is not None:
        load_kwargs["quantization_config"] = BNB_CONFIG
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16 if USE_BF16 else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        **load_kwargs,
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
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
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


def build_inference_messages(record: dict) -> list[dict]:
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": build_semantic_slot_prompt(record["input_text"], include_schema_definition=False)},
    ]


def render_wrapper_prediction(prediction_json: dict | None) -> dict | None:
    if not isinstance(prediction_json, dict):
        return prediction_json

    semantic_slots = prediction_json.get("semantic_slots")
    target_json = deepcopy(prediction_json.get("target_json"))
    if not isinstance(target_json, dict):
        return prediction_json

    if isinstance(semantic_slots, dict):
        category = semantic_slots.get("category")
        priority = semantic_slots.get("priority")
        component = semantic_slots.get("component")
        action_template = semantic_slots.get("action_template")
        requires_followup = semantic_slots.get("requires_followup")

        if isinstance(category, str):
            target_json["category"] = category
        if isinstance(priority, str):
            target_json["priority"] = priority
        if isinstance(requires_followup, bool):
            target_json["requires_followup"] = requires_followup
        if (
            isinstance(component, str)
            and isinstance(target_json.get("affected_systems"), list)
            and target_json["affected_systems"]
            and isinstance(target_json["affected_systems"][0], dict)
        ):
            target_json["affected_systems"][0]["component"] = component

        summary = target_json.get("summary")
        inferred_category = target_json.get("category")
        if isinstance(action_template, str) and action_template in CATEGORY_BY_ACTION_TEMPLATE:
            inferred_category = CATEGORY_BY_ACTION_TEMPLATE[action_template]
            target_json["category"] = inferred_category
        if isinstance(summary, str) and isinstance(inferred_category, str):
            prefix = ACTION_PREFIX_BY_CATEGORY.get(inferred_category, "Investigate issue")
            if (
                isinstance(target_json.get("actions_requested"), list)
                and target_json["actions_requested"]
                and isinstance(target_json["actions_requested"][0], dict)
            ):
                target_json["actions_requested"][0]["action"] = f"{prefix}: {summary}"
    return target_json


def sample_eval_dicts(eval_records: list[dict], predictions: list[dict], schema: dict) -> list[dict]:
    by_id = {record["sample_id"]: record for record in eval_records}
    results = []
    for prediction in predictions:
        target_record = by_id[prediction["sample_id"]]
        sample_result = evaluate_sample(
            sample_id=prediction["sample_id"],
            prediction_text=prediction.get("prediction_text"),
            prediction_json=prediction.get("prediction_json"),
            target_json=target_record["target_json"],
            schema=schema,
        )
        results.append(
            {
                "sample_id": sample_result.sample_id,
                "valid_json": sample_result.valid_json,
                "schema_compliant": sample_result.schema_compliant,
                "field_exact_match": sample_result.field_exact_match,
                "exact_match": sample_result.exact_match,
                "primary_error": sample_result.primary_error,
                "schema_name": target_record["schema_name"],
                "complexity_bucket": target_record["complexity_bucket"],
            }
        )
    return results


def summarize_from_dicts(items: list[dict]) -> dict:
    from src.evaluation.error_types import ERROR_TYPES

    total = len(items)
    if total == 0:
        return {
            "num_samples": 0,
            "valid_json_rate": 0.0,
            "schema_compliance_rate": 0.0,
            "field_exact_match": 0.0,
            "end_to_end_exact_match": 0.0,
            "error_counts": {error: 0 for error in ERROR_TYPES},
        }

    error_counts = Counter(item["primary_error"] for item in items if item["primary_error"] is not None)
    return {
        "num_samples": total,
        "valid_json_rate": sum(item["valid_json"] for item in items) / total,
        "schema_compliance_rate": sum(item["schema_compliant"] for item in items) / total,
        "field_exact_match": sum(item["field_exact_match"] for item in items) / total,
        "end_to_end_exact_match": sum(item["exact_match"] for item in items) / total,
        "error_counts": {error: error_counts.get(error, 0) for error in ERROR_TYPES},
    }


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
    print("inference_batch_size =", INFERENCE_BATCH_SIZE)
    print("has_bnb_4bit =", HAS_BNB_4BIT)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    raw_train_records = load_jsonl(PROJECT_ROOT / "data" / "reduced" / "phase1_train_reduced.jsonl")
    raw_val_records = load_jsonl(PROJECT_ROOT / "data" / "reduced" / "phase1_val_reduced.jsonl")
    raw_test_records = load_jsonl(PROJECT_ROOT / "data" / "reduced" / "phase1_test_reduced.jsonl")
    component_map = build_majority_component_map(raw_train_records)

    train_records = [to_semantic_slot_record(record, component_map) for record in raw_train_records]
    val_records = [to_semantic_slot_record(record, component_map) for record in raw_val_records]
    test_records = [to_semantic_slot_record(record, component_map) for record in raw_test_records]
    eval_records = [to_canonical_reduced_record(record, component_map) for record in raw_test_records]
    schema = get_schema(EVAL_SCHEMA_NAME)

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
                write_sft_split(structure_stage_records, stage1_train_path, include_schema_definition=True)
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

            prediction_texts = batched_generate_texts(
                model=model,
                tokenizer=tokenizer,
                records=test_records,
                build_messages=build_inference_messages,
                generation_kwargs=GENERATION_KWARGS,
                batch_size=INFERENCE_BATCH_SIZE,
            )

            predictions = []
            for record, prediction_text in zip(test_records, prediction_texts, strict=True):
                try:
                    prediction_json = json.loads(prediction_text)
                except json.JSONDecodeError:
                    prediction_json = None
                predictions.append(
                    {
                        "sample_id": record["sample_id"],
                        "prediction_text": prediction_text,
                        "prediction_json": render_wrapper_prediction(prediction_json),
                        "metadata": {"model_name": BASE_MODEL_NAME, "experiment_id": experiment_name},
                    }
                )
            dump_jsonl(paths["prediction_path"], predictions)

            raw_sample_results = sample_eval_dicts(eval_records, predictions, schema)
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
            write_json_report(paths["raw_field_path"], analyze_field_errors(eval_records, predictions))

            repaired_predictions = []
            for record in predictions:
                repaired_json, repaired = repair_prediction(record.get("prediction_json"), schema)
                repaired_predictions.append(
                    {
                        **record,
                        "prediction_json": repaired_json,
                        "metadata": {**record.get("metadata", {}), "repaired": repaired},
                    }
                )
            dump_jsonl(paths["repaired_prediction_path"], repaired_predictions)
            repaired_sample_results = sample_eval_dicts(eval_records, repaired_predictions, schema)
            repaired_report = {"summary": summarize_from_dicts(repaired_sample_results), "per_sample": repaired_sample_results}
            write_json_report(paths["repaired_report_path"], repaired_report)
            write_json_report(paths["repaired_field_path"], analyze_field_errors(eval_records, repaired_predictions))

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
        "# Semantic Slot Supervision Batch Summary",
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


if __name__ == "__main__":
    run()
