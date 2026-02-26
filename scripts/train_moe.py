"""
MathScy MoE Training Script
GPU-ready script for training the domain-specialized MoE model.
Implements Branch-Train-Mix with DeepSeekMoE-style architecture.

Usage:
    # Train single domain expert:
    python scripts/train_moe.py --mode branch --domain algebraic_geometry

    # Full Branch-Train-Mix (all domains):
    python scripts/train_moe.py --mode branch

    # Multi-GPU with DeepSpeed:
    deepspeed --num_gpus=8 scripts/train_moe.py --mode branch --domain algebraic_geometry --deepspeed configs/deepspeed_zero2.json

    # Train on combined data (single/multi-GPU):
    python scripts/train_moe.py --mode single
    deepspeed --num_gpus=8 scripts/train_moe.py --mode single --deepspeed configs/deepspeed_zero2.json
"""

import argparse
import json
import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, Trainer,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType, prepare_model_for_kbit_training
from datasets import Dataset

BASE_DIR = "/scratch/ctoxtli/moexp"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

with open(os.path.join(BASE_DIR, "configs", "project_config.json")) as f:
    config = json.load(f)

EXPERT_DOMAINS = config["moe_config"]["expert_domains"]
TRAIN_CONFIG = config["moe_config"]["training"]


@dataclass
class DynamicPaddingCollator:
    """Pads batches dynamically to the longest sequence, saving GPU memory."""
    pad_token_id: int = 0

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_token_id] * pad_len)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad_len)
            batch["labels"].append(f["labels"] + [-100] * pad_len)
        return {k: torch.tensor(v) for k, v in batch.items()}


def load_domain_data(domain: str, max_examples: int = None) -> List[Dict]:
    """Load training data for a specific domain."""
    path = os.path.join(DATA_DIR, f"domain_{domain}.jsonl")
    if not os.path.exists(path):
        print(f"Warning: No data file for domain {domain} at {path}")
        return []

    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
            if max_examples and len(examples) >= max_examples:
                break
    return examples


def load_val_data(max_examples: int = None) -> List[Dict]:
    """Load validation data."""
    path = os.path.join(DATA_DIR, "moe_val.jsonl")
    if not os.path.exists(path):
        return []
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
            if max_examples and len(examples) >= max_examples:
                break
    return examples


def format_for_training(examples: List[Dict], tokenizer, max_seq_length: int = 2048) -> Dataset:
    """Format examples into tokenized dataset with label masking. No padding (done dynamically)."""
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for ex in examples:
        instruction = ex.get("instruction", "")
        response = ex.get("response", "")
        if isinstance(instruction, list):
            instruction = "\n".join(str(x) for x in instruction)
        if isinstance(response, list):
            response = "\n".join(str(x) for x in response)
        if not isinstance(instruction, str):
            instruction = str(instruction)
        if not isinstance(response, str):
            response = str(response)

        instruction_text = f"### Instruction:\n{instruction}\n\n### Response:\n"
        full_text = instruction_text + response + tokenizer.eos_token

        full_enc = tokenizer(
            full_text,
            truncation=True,
            max_length=max_seq_length,
            padding=False,  # No padding - done dynamically per batch
        )

        instr_enc = tokenizer(instruction_text, add_special_tokens=True)
        instr_len = len(instr_enc["input_ids"])

        labels = list(full_enc["input_ids"])
        for i in range(min(instr_len, len(labels))):
            labels[i] = -100

        input_ids_list.append(full_enc["input_ids"])
        attention_mask_list.append(full_enc["attention_mask"])
        labels_list.append(labels)

    dataset = Dataset.from_dict({
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
    })

    return dataset


def load_model_and_tokenizer(base_model_path: str, use_deepspeed: bool = False):
    """Load base model with proper QLoRA configuration."""
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model_kwargs = {
        "dtype": torch.bfloat16,
        "quantization_config": bnb_config,
    }

    if not use_deepspeed:
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs)

    # Prepare for QLoRA: enables gradient checkpointing + input_require_grads
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    return model, tokenizer


def train_domain_expert(
    domain: str,
    base_model_path: str,
    output_dir: str,
    resume_from: str = None,
    deepspeed_config: str = None,
    local_rank: int = -1,
):
    """Train a LoRA expert for a specific domain."""
    print(f"\n{'='*60}")
    print(f"Training domain expert: {domain}")
    print(f"{'='*60}")

    examples = load_domain_data(domain)
    if not examples:
        print(f"No data for {domain}, skipping.")
        return None

    val_examples = load_val_data(max_examples=500)
    print(f"Loaded {len(examples)} training examples, {len(val_examples)} validation examples")

    use_deepspeed = deepspeed_config is not None
    print(f"Loading base model: {base_model_path} (DeepSpeed: {use_deepspeed})")
    model, tokenizer = load_model_and_tokenizer(base_model_path, use_deepspeed=use_deepspeed)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=TRAIN_CONFIG["lora_rank"],
        lora_alpha=TRAIN_CONFIG["lora_alpha"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
    )

    if resume_from and os.path.exists(resume_from):
        print(f"Loading LoRA adapter from {resume_from}")
        model = PeftModel.from_pretrained(model, resume_from)
    else:
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    max_seq = TRAIN_CONFIG["max_seq_length"]
    train_dataset = format_for_training(examples, tokenizer, max_seq)
    eval_dataset = format_for_training(val_examples, tokenizer, max_seq) if val_examples else None

    # Memory-optimized training args for single A100 80GB
    training_args_kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": TRAIN_CONFIG["num_epochs"],
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "gradient_accumulation_steps": 4,  # effective batch = 32
        "learning_rate": TRAIN_CONFIG["learning_rate"],
        "warmup_ratio": TRAIN_CONFIG["warmup_ratio"],
        "logging_steps": 10,
        "save_steps": 100,
        "save_total_limit": 3,
        "bf16": True,
        "dataloader_pin_memory": True,
        "report_to": "none",
        "run_name": f"mathscy-{domain}",
        "eval_strategy": "steps" if eval_dataset else "no",
        "eval_steps": 100 if eval_dataset else None,
        "load_best_model_at_end": True if eval_dataset else False,
        "seed": 42,
        "gradient_checkpointing": True,
        "optim": "adamw_bnb_8bit",  # 8-bit optimizer saves memory
        "dataloader_num_workers": 0,
        "remove_unused_columns": False,
    }

    if deepspeed_config:
        training_args_kwargs["deepspeed"] = deepspeed_config
    if local_rank != -1:
        training_args_kwargs["local_rank"] = local_rank

    training_args = TrainingArguments(**training_args_kwargs)

    collator = DynamicPaddingCollator(pad_token_id=tokenizer.pad_token_id)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    # Check for existing checkpoints to resume from
    resume_checkpoint = None
    if os.path.isdir(output_dir):
        checkpoints = [
            os.path.join(output_dir, d) for d in os.listdir(output_dir)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
        ]
        if checkpoints:
            # Sort by step number and pick the latest
            checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
            resume_checkpoint = checkpoints[-1]
            print(f"Resuming from checkpoint: {resume_checkpoint}")

    print(f"Starting training for {domain}...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    final_path = os.path.join(output_dir, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Expert saved to {final_path}")

    return final_path


def branch_train_mix(base_model_path: str, deepspeed_config: str = None, local_rank: int = -1):
    """Full Branch-Train-Mix pipeline."""
    print("=" * 60)
    print("MathScy Branch-Train-Mix Pipeline")
    print("=" * 60)

    expert_paths = {}

    for domain in EXPERT_DOMAINS:
        output_dir = os.path.join(MODELS_DIR, f"expert_{domain}")
        checkpoint = os.path.join(output_dir, "final")

        if os.path.exists(checkpoint):
            print(f"Expert for {domain} already trained at {checkpoint}")
            expert_paths[domain] = checkpoint
            continue

        path = train_domain_expert(
            domain=domain,
            base_model_path=base_model_path,
            output_dir=output_dir,
            deepspeed_config=deepspeed_config,
            local_rank=local_rank,
        )
        if path:
            expert_paths[domain] = path

    print("\n" + "=" * 60)
    print("Phase 2: Assembling MoE from domain experts")
    print("=" * 60)

    registry = {
        "base_model": base_model_path,
        "experts": expert_paths,
        "domains_trained": list(expert_paths.keys()),
        "domains_skipped": [d for d in EXPERT_DOMAINS if d not in expert_paths],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    registry_path = os.path.join(MODELS_DIR, "expert_registry.json")
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"Expert registry saved to {registry_path}")
    print(f"Trained {len(expert_paths)}/{len(EXPERT_DOMAINS)} experts")

    return expert_paths


def main():
    parser = argparse.ArgumentParser(description="MathScy MoE Training")
    parser.add_argument("--mode", choices=["single", "branch"],
                       default="branch", help="Training mode")
    parser.add_argument("--domain", type=str, default=None,
                       help="Specific domain (for branch mode)")
    parser.add_argument("--base-model", type=str,
                       default="deepseek-ai/deepseek-math-7b-base",
                       help="Base model path")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from LoRA adapter checkpoint")
    parser.add_argument("--deepspeed", type=str, default=None,
                       help="DeepSpeed config file path")
    parser.add_argument("--local_rank", type=int, default=-1,
                       help="Local rank for distributed training")
    args = parser.parse_args()

    local_model = os.path.join(MODELS_DIR, "deepseek-math-7b-base")
    if os.path.exists(local_model):
        args.base_model = local_model
        print(f"Using local model: {local_model}")

    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    if args.mode == "branch" and args.domain:
        output_dir = os.path.join(MODELS_DIR, f"expert_{args.domain}")
        train_domain_expert(
            domain=args.domain,
            base_model_path=args.base_model,
            output_dir=output_dir,
            resume_from=args.resume,
            deepspeed_config=args.deepspeed,
            local_rank=args.local_rank,
        )
    elif args.mode == "branch":
        branch_train_mix(args.base_model, deepspeed_config=args.deepspeed, local_rank=args.local_rank)
    else:
        print("Training on combined data...")
        output_dir = os.path.join(MODELS_DIR, "moe_combined")
        examples = []
        for domain in EXPERT_DOMAINS:
            domain_examples = load_domain_data(domain)
            print(f"  {domain}: {len(domain_examples)} examples")
            examples.extend(domain_examples)

        if not examples:
            print("No training data found. Run prepare_training_data.py first.")
            sys.exit(1)

        use_deepspeed = args.deepspeed is not None
        model, tokenizer = load_model_and_tokenizer(args.base_model, use_deepspeed=use_deepspeed)

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=TRAIN_CONFIG["lora_rank"],
            lora_alpha=TRAIN_CONFIG["lora_alpha"],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        max_seq = TRAIN_CONFIG["max_seq_length"]
        train_dataset = format_for_training(examples, tokenizer, max_seq)
        val_examples = load_val_data(max_examples=500)
        eval_dataset = format_for_training(val_examples, tokenizer, max_seq) if val_examples else None

        print(f"Combined dataset: {len(train_dataset)} train, {len(eval_dataset) if eval_dataset else 0} val")

        training_args_kwargs = {
            "output_dir": output_dir,
            "num_train_epochs": TRAIN_CONFIG["num_epochs"],
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2,
            "gradient_accumulation_steps": 16,
            "learning_rate": TRAIN_CONFIG["learning_rate"],
            "warmup_ratio": TRAIN_CONFIG["warmup_ratio"],
            "logging_steps": 10,
            "save_steps": 100,
            "save_total_limit": 3,
            "bf16": True,
            "dataloader_pin_memory": True,
            "report_to": "none",
            "run_name": "mathscy-combined",
            "eval_strategy": "steps" if eval_dataset else "no",
            "eval_steps": 100 if eval_dataset else None,
            "seed": 42,
            "gradient_checkpointing": True,
            "optim": "adamw_bnb_8bit",
            "dataloader_num_workers": 0,
            "remove_unused_columns": False,
        }
        if args.deepspeed:
            training_args_kwargs["deepspeed"] = args.deepspeed
        if args.local_rank != -1:
            training_args_kwargs["local_rank"] = args.local_rank

        training_args = TrainingArguments(**training_args_kwargs)
        collator = DynamicPaddingCollator(pad_token_id=tokenizer.pad_token_id)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
        )

        print("Starting combined training...")
        trainer.train()

        final_path = os.path.join(output_dir, "final")
        model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        print(f"Combined model saved to {final_path}")


if __name__ == "__main__":
    main()
