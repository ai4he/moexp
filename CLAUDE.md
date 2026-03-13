# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**MathScy** implements a domain-specialized Mixture-of-Experts (MoE) LLM system for mathematical reasoning (conjecture formulation, proof construction, counterexample finding). It uses Branch-Train-Mix over DeepSeek-Math-7B-Base with 7 QLoRA domain experts + 1 shared expert, a sigmoid router, and a Self-Play Theorem Prover (STP) loop backed by Lean 4 formal verification.

## Environment Setup (REQUIRED at every session start)

```bash
# 1. Always activate the conda environment
conda activate moexp   # or: source activate moexp

# 2. Always set HuggingFace cache — models are pre-cached here, DO NOT re-download
export HF_HOME=/scratch/ctoxtli/cache

# 3. Verify models before downloading — check cache first
ls /scratch/ctoxtli/cache/hub | grep <model-name>
```

**Pre-cached models** (as of 2026-03-11):
- `deepseek-ai/deepseek-math-7b-base`
- `deepseek-ai/deepseek-math-7b-instruct`
- `deepseek-ai/DeepSeek-Prover-V2-7B`
- `deepseek-ai/DeepSeek-Prover-V2-671B`

## Common Commands

### Running Experiments (CPU/API — no GPU needed)
```bash
# Context investigation (Task 17 — already complete, results in results/task17_context_investigation.json)
python3 -u scripts/context_investigation.py > results/task17_log.txt 2>&1

# Zero-shot baseline (Task 16)
python3 scripts/zero_shot_baseline.py

# Conjecture generation and evaluation (multi-strategy, STP loop)
python3 scripts/evaluate_conjectures.py
```

### GPU-Required Commands
```bash
# Train all 7 domain experts (Branch-Train-Mix, ~19h on 1× A100)
python3 scripts/train_moe.py --train_all

# Train single expert
python3 scripts/train_moe.py --domain algebraic_geometry

# Assemble router + MoE (requires DeepSeek-Math-7B hidden states)
python3 scripts/assemble_moe.py

# Multi-GPU training with DeepSpeed
deepspeed --num_gpus=8 scripts/train_moe.py --deepspeed configs/deepspeed_zero2.json

# MATH/GSM8K benchmark evaluation
# (scripts not yet written — planned future work)
```

### Paper
```bash
cd paper && pdflatex -interaction=nonstopmode mathscy.tex
# Run twice to resolve cross-references
```

### Lean 4 Verification
```bash
cd lean_workspace && lake build
```

## Architecture

### Pipeline Stages
```
Data Prep (CPU)  →  Knowledge Extraction (Gemini API)  →  Training Data Assembly (CPU)
     ↓
Expert Training (GPU, A100/H100)  →  Router Assembly (GPU)
     ↓
Conjecture Generation (API)  →  STP Loop (API)  →  Lean 4 Verification (GPU + Lean)
     ↓
Evaluation & Paper
```

### MoE System (`scripts/train_moe.py`, `scripts/assemble_moe.py`)
- **Base**: `deepseek-ai/deepseek-math-7b-base`, 4-bit NF4 quantized
- **7 domain experts**: QLoRA adapters (rank=64, alpha=128) on `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`; each ~600 MB in `models/expert_*/final/`
- **Shared expert**: `models/expert_shared/` — always activated alongside top-2 domain experts
- **Router**: Sigmoid gating (`models/moe_assembled/router.pt`, 117 KB); selects top-2 experts from mean-pooled hidden states; 76.3% top-1 / 88.9% top-2 accuracy; load balance ratio 1.26
- **Domain taxonomy**: 7 domains mapped from ArXiv categories — see `configs/project_config.json` for full mapping

### Conjecture Generation & STP Loop (`scripts/evaluate_conjectures.py`)
Five strategies: pattern interpolation, composition, boundary exploration, theorem generation, cross-domain analogy. The STP loop runs conjecturer → prover → judge → next round. Results checkpointed per-domain in `results/stp_*_checkpoint.json`.

### Data Pipeline (`scripts/data_utils.py`, `scripts/prepare_training_data.py`)
- **Raw dataset**: `final_data_dedup_6.5m.jsonl` (23.4 GB, 6,472,665 entries)
- **Stratified sample**: `data/stratified_sample_100.jsonl` (3,400 entries, 100/category)
- **Extraction**: Gemini (`scripts/batch_gemini_extract.py`) + ArXiv LaTeX regex (`scripts/batch_arxiv_extract.py`)
- **Training data**: `data/domain_*.jsonl` — 15 per-domain files with 5 instruction task types: `theorem_understanding`, `concept_identification`, `generalization`, `statement_classification`, `contextualize`

### LLM API Layer (`scripts/llm_utils.py`)
Unified wrapper for:
- **Gemini** (`gemini-2.0-flash`): key rotation across keys in `working_Gemini_API_keys.json`; **only key index 3 is confirmed working**
- **HPC LLM** (OpenAI-compatible at `https://llm.rcd.clemson.edu/v1`): models `qwen3-30b-a3b-instruct-fp8`, `gptoss-20b`
- **Groq** (`llama-3.3-70b-versatile`): key in `groqcloud_key.txt`; 100K tokens/day limit
- **Mistral** (`mistral-small-latest`): key in `mistral_key.txt`; hits per-minute limits under sustained load

## Key Files

| File | Purpose |
|------|---------|
| `configs/project_config.json` | Master config: paths, hyperparameters, API settings, domain taxonomy |
| `scripts/evaluate_conjectures.py` | Main conjecture generation + STP loop (50 KB, most complex script) |
| `scripts/llm_utils.py` | All LLM API calls — edit here to change models or add providers |
| `models/expert_registry.json` | Expert-to-domain mapping used by router at inference |
| `results/proved_conjectures.lean` | Lean 4 formalizations (26 conjectures autoformalized, none formally verified) |
| `paper/mathscy.tex` | Paper source — ablation tables and results sections updated frequently |

## API Rate Limit Notes

- All APIs hit daily limits quickly. Check before running experiments.
- Groq 100K TPD exhausts in ~1 hour of sustained generation.
- Gemini 429s are common even with 6-key rotation; key index 3 is most reliable.
- OpenRouter is out of credits (402); requires credit purchase before use.
- Never hardcode API keys — GitHub push protection will block the push.

## Experimental Results Location

All results in `results/`:
- `task15_multi_judge_consensus.json` — 50/50 judge consensus (complete)
- `task16_zero_shot_baseline.json` — Zero-shot Gemini baseline (complete)
- `task17_context_investigation.json` — Context quality investigation (complete; good_context 0.864 > no_context 0.817 > random_context 0.780, ANOVA p=0.0016)
- `ranked_conjectures.json` — 153 conjectures ranked by quality
- `stp_*_checkpoint.json` — Per-domain STP results (7 files)
