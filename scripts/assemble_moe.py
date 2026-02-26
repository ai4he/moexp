"""
MathScy MoE Assembly Script
Assembles trained domain LoRA experts into a unified MoE model with sigmoid router.

Usage:
    # Train router:
    python scripts/assemble_moe.py --mode train-router

    # Evaluate router:
    python scripts/assemble_moe.py --mode evaluate

    # Full pipeline (train + evaluate):
    python scripts/assemble_moe.py --mode full
"""

import argparse
import json
import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple
from safetensors.torch import load_file

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = "/scratch/ctoxtli/moexp"
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

with open(os.path.join(BASE_DIR, "configs", "project_config.json")) as f:
    config = json.load(f)

EXPERT_DOMAINS = config["moe_config"]["expert_domains"]
DOMAIN_TO_IDX = {d: i for i, d in enumerate(EXPERT_DOMAINS)}
NUM_EXPERTS = len(EXPERT_DOMAINS)


class MathDomainRouter(nn.Module):
    """
    Sigmoid-based router with auxiliary-loss-free load balancing.
    Operates at sequence level: mean-pools hidden states then routes.
    """
    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size

        self.gate = nn.Linear(hidden_size, num_experts, bias=True)
        self.register_buffer('expert_load_ema', torch.zeros(num_experts))
        self.register_buffer('balance_bias', torch.zeros(num_experts))
        self.balance_update_speed = 0.001

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: [batch, seq_len] (1 for real tokens, 0 for padding)
        Returns:
            topk_weights: [batch, top_k] normalized routing weights
            topk_indices: [batch, top_k] selected expert indices
            all_scores: [batch, num_experts] raw sigmoid scores
        """
        # Mean pool over sequence (masking padding)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden_states.mean(dim=1)

        logits = self.gate(pooled)  # [B, E]
        balanced_logits = logits + self.balance_bias.detach()
        scores = torch.sigmoid(balanced_logits)  # [B, E]

        topk_scores, topk_indices = torch.topk(scores, k=self.top_k, dim=-1)
        topk_weights = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-9)

        if self.training:
            with torch.no_grad():
                expert_mask = torch.zeros_like(scores)
                expert_mask.scatter_(-1, topk_indices, 1.0)
                load = expert_mask.sum(dim=0)
                total = load.sum()
                load_fraction = load / (total + 1e-9)
                self.expert_load_ema = 0.99 * self.expert_load_ema + 0.01 * load_fraction
                target_load = 1.0 / self.num_experts
                self.balance_bias += self.balance_update_speed * (target_load - self.expert_load_ema)

        return topk_weights, topk_indices, scores


def load_base_model(model_path: str = "deepseek-ai/deepseek-math-7b-base"):
    """Load base model with QLoRA quantization."""
    local_model = os.path.join(MODELS_DIR, "deepseek-math-7b-base")
    if os.path.exists(local_model):
        model_path = local_model

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb_config, device_map="auto",
        dtype=torch.bfloat16,
    )
    return model, tokenizer


def extract_hidden_states(model, tokenizer, texts: List[str], batch_size: int = 8,
                          max_length: int = 512) -> torch.Tensor:
    """Extract mean-pooled hidden states from the base model for a list of texts."""
    model.eval()
    all_hidden = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Mean pool last hidden state
        hidden = outputs.hidden_states[-1]  # [B, T, H]
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        all_hidden.append(pooled.cpu())

        if (i // batch_size) % 50 == 0 and i > 0:
            print(f"  Extracted {i}/{len(texts)} hidden states")

    return torch.cat(all_hidden, dim=0)


def load_router_training_data(samples_per_domain: int = 1000):
    """Load balanced domain-labeled data for router training."""
    texts = []
    labels = []

    for domain in EXPERT_DOMAINS:
        path = os.path.join(DATA_DIR, f"domain_{domain}.jsonl")
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping")
            continue

        examples = []
        with open(path) as f:
            for line in f:
                examples.append(json.loads(line))

        random.shuffle(examples)
        sampled = examples[:samples_per_domain]

        for ex in sampled:
            inst = ex.get("instruction", "")
            resp = ex.get("response", "")
            if isinstance(inst, list):
                inst = "\n".join(str(x) for x in inst)
            if isinstance(resp, list):
                resp = "\n".join(str(x) for x in resp)
            texts.append(f"{inst}\n{resp}"[:1024])
            labels.append(DOMAIN_TO_IDX[domain])

    # Shuffle together
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    return list(texts), list(labels)


def train_router(model, tokenizer, save_path: str, samples_per_domain: int = 1000,
                 epochs: int = 10, lr: float = 1e-3, batch_size: int = 8):
    """Train the sigmoid router on domain-labeled data."""
    print("=" * 60)
    print("Phase 2b: Training MoE Router")
    print("=" * 60)

    # Load data
    print("Loading router training data...")
    texts, labels = load_router_training_data(samples_per_domain)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    print(f"Loaded {len(texts)} examples across {NUM_EXPERTS} domains")

    # Split train/val (90/10)
    split = int(0.9 * len(texts))
    train_texts, val_texts = texts[:split], texts[split:]
    train_labels, val_labels = labels_tensor[:split], labels_tensor[split:]

    # Extract hidden states
    print("Extracting hidden states from base model...")
    train_hidden = extract_hidden_states(model, tokenizer, train_texts, batch_size=batch_size)
    val_hidden = extract_hidden_states(model, tokenizer, val_texts, batch_size=batch_size)
    print(f"Hidden states: train={train_hidden.shape}, val={val_hidden.shape}")

    hidden_size = train_hidden.shape[1]

    # Initialize router
    router = MathDomainRouter(hidden_size=hidden_size, num_experts=NUM_EXPERTS, top_k=2)
    router = router.to("cuda" if torch.cuda.is_available() else "cpu")

    # Optimizer
    optimizer = torch.optim.AdamW(router.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    device = next(router.parameters()).device
    train_hidden = train_hidden.to(device)
    train_labels = train_labels.to(device)
    val_hidden = val_hidden.to(device)
    val_labels = val_labels.to(device)

    best_val_acc = 0.0
    best_state = None

    print(f"\nTraining router for {epochs} epochs...")
    for epoch in range(epochs):
        router.train()
        # Shuffle
        perm = torch.randperm(len(train_hidden))
        train_hidden_shuffled = train_hidden[perm]
        train_labels_shuffled = train_labels[perm]

        total_loss = 0
        correct = 0
        total = 0

        for i in range(0, len(train_hidden_shuffled), batch_size * 4):
            h = train_hidden_shuffled[i:i+batch_size*4].unsqueeze(1)  # [B, 1, H]
            y = train_labels_shuffled[i:i+batch_size*4]

            topk_w, topk_idx, scores = router(h)

            # Cross-entropy loss on all expert scores
            loss = F.cross_entropy(router.gate(h.squeeze(1)), y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
            predictions = scores.argmax(dim=-1)
            correct += (predictions == y).sum().item()
            total += len(y)

        scheduler.step()
        train_acc = correct / total
        train_loss = total_loss / total

        # Validation
        router.eval()
        with torch.no_grad():
            val_h = val_hidden.unsqueeze(1)
            _, _, val_scores = router(val_h)
            val_preds = val_scores.argmax(dim=-1)
            val_acc = (val_preds == val_labels).float().mean().item()

            # Top-2 accuracy
            val_top2 = torch.topk(val_scores, k=2, dim=-1).indices
            val_top2_hit = (val_top2 == val_labels.unsqueeze(-1)).any(dim=-1)
            val_top2_acc = val_top2_hit.float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in router.state_dict().items()}

        print(f"  Epoch {epoch+1}/{epochs}: loss={train_loss:.4f} "
              f"train_acc={train_acc:.3f} val_acc={val_acc:.3f} val_top2={val_top2_acc:.3f}")

    # Restore best
    if best_state:
        router.load_state_dict(best_state)

    # Save router
    os.makedirs(save_path, exist_ok=True)
    router_path = os.path.join(save_path, "router.pt")
    torch.save(router.state_dict(), router_path)
    print(f"\nRouter saved to {router_path}")
    print(f"Best validation accuracy: {best_val_acc:.3f}")

    return router


def evaluate_router(model, tokenizer, router, samples_per_domain: int = 200):
    """Evaluate router: accuracy, load balance, per-domain breakdown."""
    print("\n" + "=" * 60)
    print("Phase 3: Router Evaluation")
    print("=" * 60)

    # Load eval data
    texts, labels = load_router_training_data(samples_per_domain)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    print(f"Evaluating on {len(texts)} examples...")
    hidden = extract_hidden_states(model, tokenizer, texts, batch_size=8)

    device = next(router.parameters()).device
    hidden = hidden.to(device)
    labels_tensor = labels_tensor.to(device)

    router.eval()
    with torch.no_grad():
        h = hidden.unsqueeze(1)
        topk_w, topk_idx, scores = router(h)

    top1_preds = scores.argmax(dim=-1)
    top2_preds = torch.topk(scores, k=2, dim=-1).indices

    # Overall accuracy
    top1_acc = (top1_preds == labels_tensor).float().mean().item()
    top2_hit = (top2_preds == labels_tensor.unsqueeze(-1)).any(dim=-1)
    top2_acc = top2_hit.float().mean().item()

    # Per-domain accuracy
    domain_results = {}
    for domain in EXPERT_DOMAINS:
        idx = DOMAIN_TO_IDX[domain]
        mask = labels_tensor == idx
        if mask.sum() == 0:
            continue
        domain_top1 = (top1_preds[mask] == idx).float().mean().item()
        domain_top2 = (top2_preds[mask] == idx).any(dim=-1).float().mean().item() if mask.sum() > 0 else 0
        domain_results[domain] = {
            "top1_accuracy": round(domain_top1, 3),
            "top2_accuracy": round(domain_top2, 3),
            "num_samples": int(mask.sum()),
        }

    # Load balance
    expert_counts = torch.zeros(NUM_EXPERTS)
    for i in range(NUM_EXPERTS):
        expert_counts[i] = (top1_preds == i).sum().item()
    load_fractions = expert_counts / expert_counts.sum()
    max_load = load_fractions.max().item()
    min_load = load_fractions[load_fractions > 0].min().item()
    balance_ratio = max_load / min_load if min_load > 0 else float('inf')

    results = {
        "overall": {
            "top1_accuracy": round(top1_acc, 3),
            "top2_accuracy": round(top2_acc, 3),
            "num_samples": len(texts),
            "num_experts": NUM_EXPERTS,
        },
        "load_balance": {
            "expert_load_fractions": {EXPERT_DOMAINS[i]: round(load_fractions[i].item(), 3)
                                      for i in range(NUM_EXPERTS)},
            "balance_ratio_max_min": round(balance_ratio, 2),
            "target_fraction": round(1.0 / NUM_EXPERTS, 3),
        },
        "per_domain": domain_results,
        "router_bias": router.balance_bias.cpu().tolist(),
        "expert_load_ema": router.expert_load_ema.cpu().tolist(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Print results
    print(f"\n{'='*60}")
    print(f"ROUTER EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Top-1 Accuracy: {top1_acc:.1%}  (random baseline: {1/NUM_EXPERTS:.1%})")
    print(f"Top-2 Accuracy: {top2_acc:.1%}")
    print(f"Load Balance Ratio: {balance_ratio:.2f}  (target: 1.0)")
    print(f"\nPer-domain breakdown:")
    print(f"{'Domain':<25} {'Top-1':>8} {'Top-2':>8} {'N':>6}")
    print(f"{'-'*50}")
    for domain in EXPERT_DOMAINS:
        if domain in domain_results:
            r = domain_results[domain]
            print(f"{domain:<25} {r['top1_accuracy']:>7.1%} {r['top2_accuracy']:>7.1%} {r['num_samples']:>6}")
    print(f"\nLoad distribution:")
    for i, domain in enumerate(EXPERT_DOMAINS):
        bar = "#" * int(load_fractions[i].item() * 100)
        print(f"  {domain:<25} {load_fractions[i].item():>5.1%} {bar}")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, "router_evaluation.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


def create_expert_registry():
    """Create the expert registry JSON."""
    expert_paths = {}
    for domain in EXPERT_DOMAINS:
        path = os.path.join(MODELS_DIR, f"expert_{domain}", "final")
        if os.path.exists(path):
            expert_paths[domain] = path

    shared_path = os.path.join(MODELS_DIR, "expert_shared", "final")

    registry = {
        "base_model": "deepseek-ai/deepseek-math-7b-base",
        "shared_expert": shared_path if os.path.exists(shared_path) else None,
        "domain_experts": expert_paths,
        "router": os.path.join(MODELS_DIR, "moe_assembled", "router.pt"),
        "num_experts": NUM_EXPERTS,
        "top_k": 2,
        "domains": EXPERT_DOMAINS,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    os.makedirs(MODELS_DIR, exist_ok=True)
    registry_path = os.path.join(MODELS_DIR, "expert_registry.json")
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"Expert registry saved to {registry_path}")
    return registry


def main():
    parser = argparse.ArgumentParser(description="MathScy MoE Assembly")
    parser.add_argument("--mode", choices=["train-router", "evaluate", "full"],
                        default="full", help="Operation mode")
    parser.add_argument("--samples-per-domain", type=int, default=1000,
                        help="Training samples per domain for router")
    parser.add_argument("--eval-samples", type=int, default=200,
                        help="Evaluation samples per domain")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Router training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Router learning rate")
    args = parser.parse_args()

    save_path = os.path.join(MODELS_DIR, "moe_assembled")

    print("Loading base model...")
    model, tokenizer = load_base_model()
    print(f"Base model loaded. Hidden size: {model.config.hidden_size}")

    if args.mode in ["train-router", "full"]:
        router = train_router(
            model, tokenizer, save_path,
            samples_per_domain=args.samples_per_domain,
            epochs=args.epochs, lr=args.lr,
        )
        create_expert_registry()
    else:
        # Load existing router
        router_path = os.path.join(save_path, "router.pt")
        router = MathDomainRouter(
            hidden_size=model.config.hidden_size,
            num_experts=NUM_EXPERTS, top_k=2,
        )
        router.load_state_dict(torch.load(router_path, weights_only=True))
        router = router.to("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Router loaded from {router_path}")

    if args.mode in ["evaluate", "full"]:
        evaluate_router(model, tokenizer, router,
                        samples_per_domain=args.eval_samples)


if __name__ == "__main__":
    main()
