"""
MathScy MoE Benchmark Evaluation Script
Evaluates the assembled MoE model on MATH and GSM8K benchmarks.

Usage:
    # Full evaluation (MATH + GSM8K):
    python scripts/evaluate_benchmarks.py --mode full

    # MATH only:
    python scripts/evaluate_benchmarks.py --mode math

    # GSM8K only:
    python scripts/evaluate_benchmarks.py --mode gsm8k

    # Base model baseline (no LoRA):
    python scripts/evaluate_benchmarks.py --mode math --baseline

    # Limit number of problems (for testing):
    python scripts/evaluate_benchmarks.py --mode math --max-problems 100
"""

import argparse
import json
import os
import re
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_from_disk

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = "/scratch/ctoxtli/moexp"
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

with open(os.path.join(BASE_DIR, "configs", "project_config.json")) as f:
    config = json.load(f)

EXPERT_DOMAINS = config["moe_config"]["expert_domains"]
DOMAIN_TO_IDX = {d: i for i, d in enumerate(EXPERT_DOMAINS)}
NUM_EXPERTS = len(EXPERT_DOMAINS)


# ─── Answer Extraction ───────────────────────────────────────────────

def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{...} in model output."""
    # Find the last \boxed{...} in the text
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return None


def extract_numeric_answer(text: str) -> Optional[str]:
    """Extract final numeric answer from text (for GSM8K-style)."""
    # GSM8K format: "#### <number>"
    match = re.search(r'####\s*([\-\d,\.]+)', text)
    if match:
        return match.group(1).replace(',', '').strip()

    # Look for "The answer is <number>"
    match = re.search(r'[Tt]he (?:final )?answer is[:\s]*([\-\d,\.\/]+)', text)
    if match:
        return match.group(1).replace(',', '').strip()

    # Last number in the text
    numbers = re.findall(r'[\-]?\d+(?:\.\d+)?', text)
    if numbers:
        return numbers[-1]
    return None


def normalize_answer(answer: str) -> str:
    """Normalize an answer for comparison."""
    if answer is None:
        return ""
    answer = answer.strip()
    # Remove trailing period
    answer = answer.rstrip('.')
    # Remove dollar signs and percent
    answer = answer.replace('$', '').replace('%', '').replace('\\%', '')
    # Normalize fractions: \frac{a}{b} -> a/b
    answer = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'\1/\2', answer)
    # Remove \text{} wrapping
    answer = re.sub(r'\\text\{([^{}]+)\}', r'\1', answer)
    # Remove \mathrm{} wrapping
    answer = re.sub(r'\\mathrm\{([^{}]+)\}', r'\1', answer)
    # Remove extra spaces
    answer = re.sub(r'\s+', ' ', answer).strip()
    # Try to evaluate simple fractions
    try:
        if '/' in answer and len(answer) < 20:
            parts = answer.split('/')
            if len(parts) == 2:
                val = float(parts[0]) / float(parts[1])
                if val == int(val):
                    return str(int(val))
                return f"{val:.6f}"
    except (ValueError, ZeroDivisionError):
        pass
    # Try to convert to number
    try:
        val = float(answer)
        if val == int(val) and '.' not in answer:
            return str(int(val))
        return f"{val:.6f}"
    except ValueError:
        pass
    return answer.lower()


def answers_match(pred: str, gold: str) -> bool:
    """Check if predicted answer matches gold answer."""
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    if not pred_norm or not gold_norm:
        return False
    return pred_norm == gold_norm


# ─── Router ──────────────────────────────────────────────────────────

class MathDomainRouter(nn.Module):
    """Sigmoid-based router (must match assemble_moe.py definition)."""
    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.gate = nn.Linear(hidden_size, num_experts, bias=True)
        self.register_buffer('expert_load_ema', torch.zeros(num_experts))
        self.register_buffer('balance_bias', torch.zeros(num_experts))
        self.balance_update_speed = 0.001

    def forward(self, hidden_states, attention_mask=None):
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden_states.mean(dim=1)
        logits = self.gate(pooled)
        balanced_logits = logits + self.balance_bias.detach()
        scores = torch.sigmoid(balanced_logits)
        topk_scores, topk_indices = torch.topk(scores, k=self.top_k, dim=-1)
        topk_weights = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-9)
        return topk_weights, topk_indices, scores


# ─── Model Loading ───────────────────────────────────────────────────

def load_base_model():
    """Load base model with QLoRA quantization."""
    model_path = "deepseek-ai/deepseek-math-7b-base"
    local_model = os.path.join(MODELS_DIR, "deepseek-math-7b-base")
    if os.path.exists(local_model):
        model_path = local_model

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # For generation

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb_config, device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    return model, tokenizer


def load_router(hidden_size: int) -> MathDomainRouter:
    """Load trained router."""
    router_path = os.path.join(MODELS_DIR, "moe_assembled", "router.pt")
    router = MathDomainRouter(hidden_size=hidden_size, num_experts=NUM_EXPERTS, top_k=2)
    router.load_state_dict(torch.load(router_path, weights_only=True))
    router = router.to("cuda" if torch.cuda.is_available() else "cpu")
    router.eval()
    print(f"Router loaded from {router_path}")
    return router


def route_problems(model, tokenizer, router, texts: List[str],
                   batch_size: int = 8, max_length: int = 512) -> List[Tuple[int, float]]:
    """Route problems to experts. Returns list of (expert_idx, weight) for top-1."""
    model.eval()
    router.eval()
    assignments = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            topk_w, topk_idx, scores = router(hidden, inputs["attention_mask"])

        for j in range(len(batch_texts)):
            expert_idx = topk_idx[j, 0].item()
            weight = topk_w[j, 0].item()
            assignments.append((expert_idx, weight))

        if i > 0 and (i // batch_size) % 20 == 0:
            print(f"  Routed {i}/{len(texts)} problems...")

    return assignments


# ─── Generation ──────────────────────────────────────────────────────

def format_math_prompt(problem: str) -> str:
    """Format a MATH problem for generation."""
    return (
        f"Solve the following math problem step by step. "
        f"Put your final answer in \\boxed{{}}.\n\n"
        f"Problem: {problem}\n\n"
        f"Solution:"
    )


def format_gsm8k_prompt(question: str) -> str:
    """Format a GSM8K problem for generation."""
    return (
        f"Solve the following math problem step by step. "
        f"End with 'The answer is <number>'.\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


def generate_solutions(model, tokenizer, prompts: List[str],
                       batch_size: int = 4, max_new_tokens: int = 1024) -> List[str]:
    """Generate solutions for a batch of prompts."""
    solutions = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=2048,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the generated part
        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"][j].shape[0]
            generated = tokenizer.decode(output[input_len:], skip_special_tokens=True)
            solutions.append(generated)

        if i > 0 and (i // batch_size) % 10 == 0:
            print(f"  Generated {i}/{len(prompts)} solutions...")

    return solutions


# ─── Evaluation ──────────────────────────────────────────────────────

def evaluate_math(model, tokenizer, router=None, max_problems: int = 0,
                  baseline: bool = False) -> Dict:
    """Evaluate on MATH benchmark."""
    print("\n" + "=" * 60)
    print("MATH Benchmark Evaluation")
    print("=" * 60)

    # Load dataset
    ds = load_from_disk(os.path.join(DATA_DIR, "math_benchmark"))
    if max_problems > 0:
        ds = ds.select(range(min(max_problems, len(ds))))
    print(f"Evaluating on {len(ds)} MATH problems...")

    problems = ds["problem"]
    solutions = ds["solution"]
    subjects = ds["subject"]
    levels = ds["level"]

    # Extract gold answers from solutions
    gold_answers = []
    for sol in solutions:
        ans = extract_boxed_answer(sol)
        gold_answers.append(ans if ans else "")

    # Route problems (MoE mode)
    if router and not baseline:
        print("\nRouting problems to experts...")
        prompts = [format_math_prompt(p) for p in problems]
        assignments = route_problems(model, tokenizer, router, problems)

        # Group by expert
        expert_groups = defaultdict(list)
        for idx, (expert_id, weight) in enumerate(assignments):
            expert_groups[expert_id].append(idx)

        print(f"\nExpert distribution:")
        for eid in sorted(expert_groups.keys()):
            domain = EXPERT_DOMAINS[eid]
            print(f"  {domain}: {len(expert_groups[eid])} problems")

        # Load expert registry
        with open(os.path.join(MODELS_DIR, "expert_registry.json")) as f:
            registry = json.load(f)

        # Generate per expert
        all_predictions = [""] * len(ds)
        for expert_id in sorted(expert_groups.keys()):
            domain = EXPERT_DOMAINS[expert_id]
            indices = expert_groups[expert_id]
            print(f"\n--- Expert: {domain} ({len(indices)} problems) ---")

            # Load LoRA adapter
            adapter_path = registry["domain_experts"][domain]
            expert_model = PeftModel.from_pretrained(model, adapter_path)
            expert_model.eval()

            # Also merge shared expert contribution
            # (In practice, we use top-1 expert for efficiency)
            batch_prompts = [prompts[i] for i in indices]
            batch_solutions = generate_solutions(
                expert_model, tokenizer, batch_prompts,
                batch_size=4, max_new_tokens=1024,
            )

            for i, sol in zip(indices, batch_solutions):
                all_predictions[i] = sol

            # Unload adapter to free memory
            del expert_model
            torch.cuda.empty_cache()

        predictions = all_predictions
    else:
        # Baseline: use base model directly
        print("\nGenerating solutions with base model (no adapters)...")
        prompts = [format_math_prompt(p) for p in problems]
        predictions = generate_solutions(
            model, tokenizer, prompts,
            batch_size=4, max_new_tokens=1024,
        )

    # Extract predicted answers
    pred_answers = []
    for pred in predictions:
        ans = extract_boxed_answer(pred)
        if ans is None:
            ans = extract_numeric_answer(pred)
        pred_answers.append(ans if ans else "")

    # Compute accuracy
    correct = 0
    total = len(ds)
    subject_correct = defaultdict(int)
    subject_total = defaultdict(int)
    level_correct = defaultdict(int)
    level_total = defaultdict(int)

    for i in range(total):
        is_correct = answers_match(pred_answers[i], gold_answers[i])
        if is_correct:
            correct += 1
            subject_correct[subjects[i]] += 1
            level_correct[levels[i]] += 1
        subject_total[subjects[i]] += 1
        level_total[levels[i]] += 1

    overall_acc = correct / total if total > 0 else 0

    # Print results
    mode_name = "Base Model (Baseline)" if baseline else "MoE Model"
    print(f"\n{'='*60}")
    print(f"MATH RESULTS - {mode_name}")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {correct}/{total} = {overall_acc:.1%}")

    print(f"\nPer-Subject Breakdown:")
    print(f"{'Subject':<30} {'Correct':>8} {'Total':>6} {'Acc':>8}")
    print("-" * 55)
    for subj in sorted(subject_total.keys()):
        acc = subject_correct[subj] / subject_total[subj]
        print(f"{subj:<30} {subject_correct[subj]:>8} {subject_total[subj]:>6} {acc:>7.1%}")

    print(f"\nPer-Level Breakdown:")
    print(f"{'Level':<30} {'Correct':>8} {'Total':>6} {'Acc':>8}")
    print("-" * 55)
    for level in sorted(level_total.keys()):
        acc = level_correct[level] / level_total[level]
        print(f"{level:<30} {level_correct[level]:>8} {level_total[level]:>6} {acc:>7.1%}")

    results = {
        "benchmark": "MATH",
        "mode": "baseline" if baseline else "moe",
        "overall": {
            "accuracy": round(overall_acc, 4),
            "correct": correct,
            "total": total,
        },
        "per_subject": {
            subj: {
                "accuracy": round(subject_correct[subj] / subject_total[subj], 4),
                "correct": subject_correct[subj],
                "total": subject_total[subj],
            }
            for subj in sorted(subject_total.keys())
        },
        "per_level": {
            level: {
                "accuracy": round(level_correct[level] / level_total[level], 4),
                "correct": level_correct[level],
                "total": level_total[level],
            }
            for level in sorted(level_total.keys())
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    return results


def evaluate_gsm8k(model, tokenizer, router=None, max_problems: int = 0,
                   baseline: bool = False) -> Dict:
    """Evaluate on GSM8K benchmark."""
    print("\n" + "=" * 60)
    print("GSM8K Benchmark Evaluation")
    print("=" * 60)

    # Load dataset
    ds = load_from_disk(os.path.join(DATA_DIR, "gsm8k_benchmark"))
    if max_problems > 0:
        ds = ds.select(range(min(max_problems, len(ds))))
    print(f"Evaluating on {len(ds)} GSM8K problems...")

    questions = ds["question"]
    answers = ds["answer"]

    # Extract gold answers (GSM8K format: "explanation\n#### <number>")
    gold_answers = []
    for ans in answers:
        match = re.search(r'####\s*([\-\d,\.]+)', ans)
        gold_answers.append(match.group(1).replace(',', '').strip() if match else "")

    # Route problems (MoE mode)
    if router and not baseline:
        print("\nRouting problems to experts...")
        prompts = [format_gsm8k_prompt(q) for q in questions]
        assignments = route_problems(model, tokenizer, router, list(questions))

        expert_groups = defaultdict(list)
        for idx, (expert_id, weight) in enumerate(assignments):
            expert_groups[expert_id].append(idx)

        print(f"\nExpert distribution:")
        for eid in sorted(expert_groups.keys()):
            domain = EXPERT_DOMAINS[eid]
            print(f"  {domain}: {len(expert_groups[eid])} problems")

        with open(os.path.join(MODELS_DIR, "expert_registry.json")) as f:
            registry = json.load(f)

        all_predictions = [""] * len(ds)
        for expert_id in sorted(expert_groups.keys()):
            domain = EXPERT_DOMAINS[expert_id]
            indices = expert_groups[expert_id]
            print(f"\n--- Expert: {domain} ({len(indices)} problems) ---")

            adapter_path = registry["domain_experts"][domain]
            expert_model = PeftModel.from_pretrained(model, adapter_path)
            expert_model.eval()

            batch_prompts = [prompts[i] for i in indices]
            batch_solutions = generate_solutions(
                expert_model, tokenizer, batch_prompts,
                batch_size=4, max_new_tokens=512,
            )

            for i, sol in zip(indices, batch_solutions):
                all_predictions[i] = sol

            del expert_model
            torch.cuda.empty_cache()

        predictions = all_predictions
    else:
        print("\nGenerating solutions with base model (no adapters)...")
        prompts = [format_gsm8k_prompt(q) for q in questions]
        predictions = generate_solutions(
            model, tokenizer, prompts,
            batch_size=4, max_new_tokens=512,
        )

    # Extract predicted answers
    pred_answers = []
    for pred in predictions:
        ans = extract_numeric_answer(pred)
        pred_answers.append(ans if ans else "")

    # Compute accuracy
    correct = sum(1 for p, g in zip(pred_answers, gold_answers) if answers_match(p, g))
    total = len(ds)
    overall_acc = correct / total if total > 0 else 0

    mode_name = "Base Model (Baseline)" if baseline else "MoE Model"
    print(f"\n{'='*60}")
    print(f"GSM8K RESULTS - {mode_name}")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {correct}/{total} = {overall_acc:.1%}")

    results = {
        "benchmark": "GSM8K",
        "mode": "baseline" if baseline else "moe",
        "overall": {
            "accuracy": round(overall_acc, 4),
            "correct": correct,
            "total": total,
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    return results


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MathScy Benchmark Evaluation")
    parser.add_argument("--mode", choices=["math", "gsm8k", "full"],
                        default="full", help="Which benchmarks to run")
    parser.add_argument("--baseline", action="store_true",
                        help="Run base model baseline (no LoRA adapters)")
    parser.add_argument("--max-problems", type=int, default=0,
                        help="Limit number of problems (0=all)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Generation batch size")
    args = parser.parse_args()

    print("Loading base model...")
    model, tokenizer = load_base_model()
    hidden_size = model.config.hidden_size
    print(f"Base model loaded. Hidden size: {hidden_size}")

    # Load router (unless baseline)
    router = None
    if not args.baseline:
        router = load_router(hidden_size)

    all_results = {}

    if args.mode in ["math", "full"]:
        math_results = evaluate_math(
            model, tokenizer, router,
            max_problems=args.max_problems,
            baseline=args.baseline,
        )
        all_results["math"] = math_results

        # Save MATH results
        suffix = "_baseline" if args.baseline else ""
        math_path = os.path.join(RESULTS_DIR, f"math_benchmark{suffix}.json")
        with open(math_path, "w") as f:
            json.dump(math_results, f, indent=2)
        print(f"\nMATH results saved to {math_path}")

    if args.mode in ["gsm8k", "full"]:
        gsm8k_results = evaluate_gsm8k(
            model, tokenizer, router,
            max_problems=args.max_problems,
            baseline=args.baseline,
        )
        all_results["gsm8k"] = gsm8k_results

        suffix = "_baseline" if args.baseline else ""
        gsm8k_path = os.path.join(RESULTS_DIR, f"gsm8k_benchmark{suffix}.json")
        with open(gsm8k_path, "w") as f:
            json.dump(gsm8k_results, f, indent=2)
        print(f"\nGSM8K results saved to {gsm8k_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    mode_name = "Base Model (Baseline)" if args.baseline else "MoE Model"
    print(f"Mode: {mode_name}")
    for bench, res in all_results.items():
        acc = res["overall"]["accuracy"]
        n = res["overall"]["total"]
        print(f"  {bench.upper()}: {acc:.1%} ({res['overall']['correct']}/{n})")


if __name__ == "__main__":
    main()
