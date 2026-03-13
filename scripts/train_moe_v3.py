#!/usr/bin/env python3
"""
MathScy MoE v3 Training Script

Fixes applied vs v2 (notebook 06_new_moe_training.ipynb):
  1. Tighter gradient clipping: max_grad_norm=0.3 (was default 1.0)
  2. Lower learning rate: 2e-5 (was 1e-4) with cosine schedule
  3. More warmup: warmup_ratio=0.10 (was 0.03)
  4. Completion-length filter: skip examples where completion > MAX_COMPLETION_TOKENS
     to remove outlier long sequences that cause gradient explosions
  5. SpikeMonitorCallback: logs a WARNING and writes to spike_log.jsonl whenever
     loss > SPIKE_THRESHOLD * moving-avg, so issues are visible immediately
  6. Intermediate sanity check after first 20 steps of each expert — aborts and
     logs if loss is still above 5.0 (indicates data/config problem)

Usage:
    conda activate moexp
    export HF_HOME=/scratch/ctoxtli/cache
    python3 scripts/train_moe_v3.py [--domains algebra,number_theory] [--skip-existing]
"""

import os
import sys
import json
import math
import time
import shutil
import random
import logging
import argparse
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ.setdefault('HF_HOME', '/scratch/ctoxtli/cache')
os.environ.setdefault('TRANSFORMERS_CACHE', '/scratch/ctoxtli/cache')
os.environ.setdefault('HF_DATASETS_CACHE', '/scratch/ctoxtli/cache')

import torch
from datasets import Dataset as HFDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path('/scratch/ctoxtli/moexp')
DATA_DIR   = BASE_DIR / 'new_moe_to_train'
MODELS_DIR = BASE_DIR / 'models' / 'v3_moe'
PLOTS_DIR  = BASE_DIR / 'results' / 'training_plots_v3'
LOGS_DIR   = BASE_DIR / 'logs'

for d in [MODELS_DIR, PLOTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / 'train_moe_v3.log'),
    ],
)
log = logging.getLogger(__name__)

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Config ────────────────────────────────────────────────────────────────────
EXPERT_DOMAINS = [
    'algebraic_geometry', 'discrete_math', 'number_theory',
    'analysis', 'algebra', 'geometry_topology', 'probability_statistics',
]

CATEGORY_TO_DOMAIN = {
    'math.AG': 'algebraic_geometry', 'math.CO': 'discrete_math',
    'math.NT': 'number_theory',      'math.CA': 'analysis',
    'math.CV': 'analysis',           'math.FA': 'analysis',
    'math.AP': 'analysis',           'math.AC': 'algebra',
    'math.RA': 'algebra',            'math.GR': 'algebra',
    'math.RT': 'algebra',            'math.KT': 'algebra',
    'math.AT': 'geometry_topology',  'math.GT': 'geometry_topology',
    'math.GN': 'geometry_topology',  'math.DG': 'geometry_topology',
    'math.MG': 'geometry_topology',  'math.PR': 'probability_statistics',
    'math.ST': 'probability_statistics', 'math.IT': 'discrete_math',
    'math.LO': 'discrete_math',      'math.DS': 'analysis',
    'math.OC': 'analysis',           'math.NA': 'analysis',
    'math.SG': 'geometry_topology',  'math.SP': 'analysis',
}

@dataclass
class V3Config:
    base_model:          str  = 'deepseek-ai/deepseek-math-7b-base'
    lora_rank:           int  = 64
    lora_alpha:          int  = 128
    lora_dropout:        float = 0.05
    lora_targets:        list = field(default_factory=lambda: [
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
        'gate_proj', 'up_proj', 'down_proj',
    ])

    # ── v3 stability fixes ─────────────────────────────────────────────────
    learning_rate:       float = 2e-5     # was 1e-4 in v2
    max_grad_norm:       float = 0.3      # was default 1.0 in v2
    warmup_ratio:        float = 0.10     # was 0.03 in v2
    lr_scheduler:        str   = 'cosine'
    max_completion_tokens: int = 800      # filter outlier long completions

    batch_size:          int  = 4
    grad_accum:          int  = 8         # effective batch = 32
    num_epochs:          int  = 3
    max_seq_length:      int  = 2048
    weight_decay:        float = 0.01
    logging_steps:       int  = 5        # more frequent logging for early spike detection
    save_steps:          int  = 50
    eval_steps:          int  = 50

    # ── spike detection ────────────────────────────────────────────────────
    spike_threshold:     float = 5.0     # warn if loss > spike_threshold * moving_avg
    spike_window:        int   = 10      # steps in moving average window
    sanity_check_steps:  int   = 20      # abort if loss > 5.0 after this many steps
    sanity_loss_limit:   float = 5.0

CFG = V3Config()


# ── Data utilities ────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[Dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def categories_to_domain(cat_str: str) -> str:
    for cat in cat_str.replace(',', ' ').split():
        if cat in CATEGORY_TO_DOMAIN:
            return CATEGORY_TO_DOMAIN[cat]
    return 'algebra'  # fallback


def add_domain(entries: List[Dict]) -> List[Dict]:
    for e in entries:
        e['domain'] = categories_to_domain(e.get('categories', ''))
    return entries


def filter_long_completions(entries: List[Dict], tokenizer, max_tokens: int) -> List[Dict]:
    """Remove examples whose completion tokenizes to > max_tokens.
    This is the primary cause of gradient explosions in v2."""
    kept, dropped = [], 0
    for e in entries:
        completion = e.get('completion', '')
        toks = tokenizer(completion, add_special_tokens=False)['input_ids']
        if len(toks) <= max_tokens:
            kept.append(e)
        else:
            dropped += 1
    if dropped:
        log.warning(f'  Filtered {dropped} examples with completion > {max_tokens} tokens '
                    f'({dropped / (len(kept) + dropped):.1%} of data)')
    return kept


# ── Dataset building ──────────────────────────────────────────────────────────

class DynamicPaddingCollator:
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        max_len = max(len(f['input_ids']) for f in features)
        batch = {'input_ids': [], 'attention_mask': [], 'labels': []}
        for f in features:
            pad_len = max_len - len(f['input_ids'])
            batch['input_ids'].append(f['input_ids'] + [self.pad_token_id] * pad_len)
            batch['attention_mask'].append(f['attention_mask'] + [0] * pad_len)
            batch['labels'].append(f['labels'] + [-100] * pad_len)
        return {k: torch.tensor(v) for k, v in batch.items()}


def build_dataset(entries: List[Dict], tokenizer, max_seq: int,
                  domain_filter: Optional[str] = None) -> HFDataset:
    if domain_filter:
        entries = [e for e in entries if e.get('domain') == domain_filter]

    input_ids_list, attn_list, labels_list = [], [], []
    for entry in entries:
        prompt     = entry.get('prompt', '')
        completion = entry.get('completion', '')
        instr_text = f'### Instruction:\n{prompt}\n\n### Response:\n'
        full_text  = instr_text + completion + tokenizer.eos_token

        enc      = tokenizer(full_text, truncation=True, max_length=max_seq, padding=False)
        instr_len = len(tokenizer(instr_text, add_special_tokens=True)['input_ids'])

        labels = list(enc['input_ids'])
        for i in range(min(instr_len, len(labels))):
            labels[i] = -100

        # Skip examples with fewer than 8 supervised tokens (causes grad_norm=0 spikes)
        if sum(1 for l in labels if l != -100) < 8:
            continue

        input_ids_list.append(enc['input_ids'])
        attn_list.append(enc['attention_mask'])
        labels_list.append(labels)

    return HFDataset.from_dict({
        'input_ids':      input_ids_list,
        'attention_mask': attn_list,
        'labels':         labels_list,
    })


# ── Model utilities ────────────────────────────────────────────────────────────

def load_base_model(cfg: V3Config):
    local = BASE_DIR / 'models' / 'deepseek-math-7b-base'
    model_path = str(local) if local.exists() else cfg.base_model
    log.info(f'Loading base model from {model_path}')

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    log.info(f'Base model loaded. Hidden size: {model.config.hidden_size}')
    return model, tokenizer


def add_lora(model, cfg: V3Config):
    lora_cfg = LoraConfig(
        r=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
        target_modules=cfg.lora_targets, lora_dropout=cfg.lora_dropout,
        bias='none', task_type='CAUSAL_LM',
    )
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    log.info(f'LoRA applied: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)')
    return model


# ── Callbacks ─────────────────────────────────────────────────────────────────

class SpikeMonitorCallback(TrainerCallback):
    """
    Detects loss spikes and logs warnings to console + spike_log.jsonl.
    If loss > sanity_loss_limit after sanity_check_steps, stops training early
    and marks the run as failed so it can be restarted with different settings.
    """

    def __init__(self, domain: str, cfg: V3Config, plots_dir: Path):
        self.domain    = domain
        self.cfg       = cfg
        self.plots_dir = plots_dir
        self.spike_log_path = LOGS_DIR / 'spike_log.jsonl'

        self._window: deque = deque(maxlen=cfg.spike_window)
        self._train_losses: List[float] = []
        self._train_steps:  List[int]   = []
        self._spike_steps:  List[int]   = []
        self._failed = False

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kw):
        if logs is None or 'loss' not in logs:
            return

        step = state.global_step
        loss = logs['loss']
        self._train_losses.append(loss)
        self._train_steps.append(step)
        self._window.append(loss)

        # ── sanity check after first N steps ──────────────────────────────
        if step == self.cfg.sanity_check_steps and loss > self.cfg.sanity_loss_limit:
            msg = (f'[{self.domain}] SANITY FAIL at step {step}: '
                   f'loss={loss:.2f} > {self.cfg.sanity_loss_limit}. '
                   f'Stopping early — check data/LR.')
            log.error(msg)
            self._log_spike(step, loss, float('nan'), 'sanity_fail')
            self._failed = True
            control.should_training_stop = True
            return

        # ── spike detection ────────────────────────────────────────────────
        if len(self._window) >= 3:
            moving_avg = np.mean(list(self._window)[:-1])  # exclude current
            if moving_avg > 0 and loss > self.cfg.spike_threshold * moving_avg:
                msg = (f'[{self.domain}] SPIKE at step {step}: '
                       f'loss={loss:.2f}  moving_avg={moving_avg:.2f}  '
                       f'ratio={loss/moving_avg:.1f}x')
                log.warning(msg)
                self._spike_steps.append(step)
                self._log_spike(step, loss, moving_avg, 'spike')

        # ── live plot update every 50 steps ───────────────────────────────
        if step % 50 == 0 and len(self._train_steps) > 1:
            self._save_plot()

    def on_train_end(self, args, state, control, **kw):
        self._save_plot()
        n_spikes = len(self._spike_steps)
        status   = 'FAILED' if self._failed else 'OK'
        log.info(f'[{self.domain}] Training ended — status={status}, spikes={n_spikes}')

    def _log_spike(self, step: int, loss: float, avg: float, kind: str):
        record = {
            'domain': self.domain, 'step': step, 'loss': loss,
            'moving_avg': avg, 'kind': kind,
            'time': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(self.spike_log_path, 'a') as f:
            f.write(json.dumps(record) + '\n')

    def _save_plot(self):
        if not self._train_steps:
            return
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self._train_steps, self._train_losses, lw=1.2, color='steelblue', label='loss')
        for s in self._spike_steps:
            ax.axvline(s, color='red', alpha=0.4, lw=0.8)
        ax.set_title(f'{self.domain} — v3 training loss ({len(self._spike_steps)} spikes marked red)')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_yscale('symlog', linthresh=10)  # log scale to show both normal and spike ranges
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        out = self.plots_dir / f'{self.domain}_loss_v3.png'
        plt.savefig(out, dpi=120, bbox_inches='tight')
        plt.close()


# ── Training ──────────────────────────────────────────────────────────────────

def latest_checkpoint(output_dir: Path) -> Optional[str]:
    if not output_dir.is_dir():
        return None
    ckpts = sorted(
        [d for d in output_dir.iterdir() if d.name.startswith('checkpoint-')],
        key=lambda x: int(x.name.split('-')[-1]),
    )
    return str(ckpts[-1]) if ckpts else None


def train_expert(
    domain: str,
    train_entries: List[Dict],
    dev_entries:   List[Dict],
    cfg:           V3Config,
    skip_existing: bool = True,
) -> Optional[str]:
    output_dir = MODELS_DIR / f'expert_{domain}'
    final_path = output_dir / 'final'

    if skip_existing and final_path.exists():
        log.info(f'[{domain}] already trained at {final_path}, skipping.')
        return str(final_path)

    log.info(f'\n{"="*60}\nTraining v3 expert: {domain}\n{"="*60}')

    model, tokenizer = load_base_model(cfg)
    model = add_lora(model, cfg)

    # Filter long completions before building dataset (v3 fix #4)
    filtered_train = filter_long_completions(train_entries, tokenizer, cfg.max_completion_tokens)
    filtered_dev   = filter_long_completions(dev_entries,   tokenizer, cfg.max_completion_tokens)

    train_ds = build_dataset(filtered_train, tokenizer, cfg.max_seq_length, domain_filter=domain)
    dev_ds   = build_dataset(filtered_dev,   tokenizer, cfg.max_seq_length, domain_filter=domain)

    log.info(f'  Train: {len(train_ds):,}  Dev: {len(dev_ds):,}')

    if len(train_ds) == 0:
        log.warning(f'  No training examples for {domain}, skipping.')
        del model; torch.cuda.empty_cache()
        return None

    if len(dev_ds) < 10:
        log.warning(f'  Dev set tiny ({len(dev_ds)}); using full filtered dev split.')
        dev_ds = build_dataset(filtered_dev, tokenizer, cfg.max_seq_length)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,          # v3 fix #2
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,             # v3 fix #3
        lr_scheduler_type=cfg.lr_scheduler,        # v3 fix: explicit cosine
        max_grad_norm=cfg.max_grad_norm,           # v3 fix #1
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=3,
        eval_strategy='steps',
        eval_steps=cfg.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        bf16=True,
        dataloader_pin_memory=True,
        report_to='none',
        run_name=f'v3-moe-{domain}',
        seed=SEED,
        gradient_checkpointing=True,
        optim='adamw_bnb_8bit',
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    spike_cb = SpikeMonitorCallback(domain, cfg, PLOTS_DIR)
    collator  = DynamicPaddingCollator(pad_token_id=tokenizer.pad_token_id)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
        callbacks=[spike_cb],
    )

    resume = latest_checkpoint(output_dir)
    if resume:
        log.info(f'  Resuming from {resume}')

    trainer.train(resume_from_checkpoint=resume)

    if spike_cb._failed:
        log.error(f'[{domain}] Training failed sanity check. Adapter NOT saved.')
        del model, trainer; torch.cuda.empty_cache()
        return None

    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    log.info(f'  Saved to {final_path}')

    del model, trainer
    torch.cuda.empty_cache()
    return str(final_path)


def train_shared_expert(
    train_entries: List[Dict],
    dev_entries:   List[Dict],
    cfg:           V3Config,
    skip_existing: bool = True,
) -> Optional[str]:
    output_dir = MODELS_DIR / 'expert_shared'
    final_path = output_dir / 'final'

    if skip_existing and final_path.exists():
        log.info(f'[shared] already trained, skipping.')
        return str(final_path)

    log.info(f'\n{"="*60}\nTraining v3 shared expert\n{"="*60}')

    model, tokenizer = load_base_model(cfg)
    model = add_lora(model, cfg)

    # For shared expert, use balanced sample across all domains
    domain_counts = Counter(e.get('domain') for e in train_entries)
    min_count     = min(domain_counts.values())
    balanced: List[Dict] = []
    for domain in EXPERT_DOMAINS:
        domain_data = [e for e in train_entries if e.get('domain') == domain]
        balanced.extend(random.sample(domain_data, min(min_count, len(domain_data))))
    random.shuffle(balanced)

    filtered_train = filter_long_completions(balanced,    tokenizer, cfg.max_completion_tokens)
    filtered_dev   = filter_long_completions(dev_entries, tokenizer, cfg.max_completion_tokens)

    train_ds = build_dataset(filtered_train, tokenizer, cfg.max_seq_length)
    dev_ds   = build_dataset(filtered_dev,   tokenizer, cfg.max_seq_length)

    log.info(f'  Shared train: {len(train_ds):,}  Dev: {len(dev_ds):,}')

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler,
        max_grad_norm=cfg.max_grad_norm,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=3,
        eval_strategy='steps',
        eval_steps=cfg.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        bf16=True,
        dataloader_pin_memory=True,
        report_to='none',
        run_name='v3-moe-shared',
        seed=SEED,
        gradient_checkpointing=True,
        optim='adamw_bnb_8bit',
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    spike_cb = SpikeMonitorCallback('shared', cfg, PLOTS_DIR)
    collator  = DynamicPaddingCollator(pad_token_id=tokenizer.pad_token_id)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
        callbacks=[spike_cb],
    )

    resume = latest_checkpoint(output_dir)
    if resume:
        log.info(f'  Resuming from {resume}')

    trainer.train(resume_from_checkpoint=resume)

    if spike_cb._failed:
        log.error('[shared] Training failed sanity check. Adapter NOT saved.')
        del model, trainer; torch.cuda.empty_cache()
        return None

    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    log.info(f'  Saved to {final_path}')

    del model, trainer
    torch.cuda.empty_cache()
    return str(final_path)


# ── Registry ──────────────────────────────────────────────────────────────────

def save_registry(expert_paths: Dict[str, str], shared_path: Optional[str]):
    registry = {
        'base_model':    'deepseek-ai/deepseek-math-7b-base',
        'dataset':       'new_moe_to_train',
        'version':       'v3',
        'lora_rank':     CFG.lora_rank,
        'lora_alpha':    CFG.lora_alpha,
        'max_seq_length': CFG.max_seq_length,
        'num_epochs':    CFG.num_epochs,
        'learning_rate': CFG.learning_rate,
        'max_grad_norm': CFG.max_grad_norm,
        'warmup_ratio':  CFG.warmup_ratio,
        'max_completion_tokens': CFG.max_completion_tokens,
        'timestamp':     time.strftime('%Y-%m-%d %H:%M:%S'),
        'experts': {**expert_paths},
        'router':  str(MODELS_DIR / 'router.pt'),
        'expert_domains': EXPERT_DOMAINS,
        # normalised key for evaluate_conjectures_moe.py
        'domain_experts': expert_paths,
        'shared_expert': shared_path or '',
    }
    if shared_path:
        registry['experts']['shared'] = shared_path

    out = MODELS_DIR / 'v3_moe_registry.json'
    with open(out, 'w') as f:
        json.dump(registry, f, indent=2)
    log.info(f'Registry saved to {out}')
    return str(out)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train MathScy MoE v3 (stability fixes)')
    parser.add_argument('--domains', type=str, default=None,
                        help='Comma-separated domains to train (default: all 7 + shared)')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                        help='Skip domains that already have a final/ adapter')
    parser.add_argument('--no-skip', dest='skip_existing', action='store_false',
                        help='Re-train even if final/ adapter exists')
    parser.add_argument('--shared-only', action='store_true',
                        help='Only train the shared expert')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate (e.g. 5e-6 for unstable domains)')
    args = parser.parse_args()

    if args.lr is not None:
        CFG.learning_rate = args.lr
        log.info(f'Learning rate overridden to {CFG.learning_rate}')

    log.info('=' * 70)
    log.info('MathScy MoE v3 Training')
    log.info(f'  LR={CFG.learning_rate}  max_grad_norm={CFG.max_grad_norm}  '
             f'warmup={CFG.warmup_ratio}  max_completion_tokens={CFG.max_completion_tokens}')
    log.info('=' * 70)

    # GPU check
    if not torch.cuda.is_available():
        log.error('No GPU found. This script requires a CUDA GPU.')
        sys.exit(1)
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        log.info(f'  GPU {i}: {p.name}  {p.total_memory / 1e9:.1f} GB')

    # Load data
    log.info('\nLoading data...')
    train_data = add_domain(load_jsonl(str(DATA_DIR / 'train_theorem_prompt_completion_tagged.jsonl')))
    dev_data   = add_domain(load_jsonl(str(DATA_DIR / 'dev_theorem_prompt_completion_tagged.jsonl')))
    log.info(f'  Train: {len(train_data):,}  Dev: {len(dev_data):,}')

    domain_counts = Counter(e['domain'] for e in train_data)
    log.info('  Domain distribution (train):')
    for d in EXPERT_DOMAINS:
        log.info(f'    {d:32s}  {domain_counts.get(d, 0):5d}')

    # Determine which domains to train
    if args.shared_only:
        domains_to_train = []
    elif args.domains:
        domains_to_train = [d.strip() for d in args.domains.split(',')]
    else:
        domains_to_train = EXPERT_DOMAINS

    expert_paths: Dict[str, str] = {}

    # Train domain experts
    for domain in domains_to_train:
        path = train_expert(domain, train_data, dev_data, CFG, args.skip_existing)
        if path:
            expert_paths[domain] = path

    log.info(f'\n{len(expert_paths)}/{len(domains_to_train)} domain experts trained.')

    # Train shared expert
    shared_path = None
    if not args.domains or args.shared_only:
        shared_path = train_shared_expert(train_data, dev_data, CFG, args.skip_existing)

    # Save registry
    registry_path = save_registry(expert_paths, shared_path)
    log.info(f'\nDone. Registry at {registry_path}')

    # Print spike summary
    spike_log = LOGS_DIR / 'spike_log.jsonl'
    if spike_log.exists():
        with open(spike_log) as f:
            spikes = [json.loads(l) for l in f if l.strip()]
        if spikes:
            log.warning(f'\nSpike summary ({len(spikes)} total spikes):')
            domain_spikes = Counter(s['domain'] for s in spikes)
            for d, n in domain_spikes.most_common():
                log.warning(f'  {d}: {n} spikes')
        else:
            log.info('\nNo spikes detected.')


if __name__ == '__main__':
    main()
