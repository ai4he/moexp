#!/usr/bin/env python3
"""
MathScy MoE-Based Conjecture Generation Experiment

Uses the TRAINED MoE model (domain LoRA experts + sigmoid router) for conjecture
generation, instead of external API models. External APIs are used only for judging.

This enables a fair comparison between:
  - API-based generation (Groq/Mistral/Gemini) from evaluate_conjectures.py
  - MoE-based generation (trained domain experts) from this script

Usage:
    # Full experiment (generate + STP + evaluate + compare)
    python scripts/evaluate_conjectures_moe.py --mode full --judge mistral

    # Just generation (no STP, no judging)
    python scripts/evaluate_conjectures_moe.py --mode generate --domains algebra,number_theory

    # Just STP loop
    python scripts/evaluate_conjectures_moe.py --mode stp --rounds 2 --per-round 4 --judge mistral

    # Compare with existing API results
    python scripts/evaluate_conjectures_moe.py --mode compare
"""

import os
import sys
import json
import re
import time
import random
import argparse
import traceback
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Setup paths
PROJECT_DIR = "/scratch/ctoxtli/moexp"
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

sys.path.insert(0, os.path.join(PROJECT_DIR, "scripts"))

from assemble_moe import MathDomainRouter
from evaluate_conjectures import (
    load_api_keys,
    llm_generate,
    load_knowledge_base,
    format_results_for_prompt,
    parse_json_response,
    score_conjecture,
    rank_all_conjectures,
    generate_evaluation_report,
    print_report,
    STP_PROOF_PROMPT,
    STP_JUDGE_PROMPT,
    ANALOGY_PAIRS,
)

# ===== Expert Domains =====

EXPERT_DOMAINS = [
    "algebraic_geometry", "discrete_math", "number_theory",
    "analysis", "algebra", "geometry_topology", "probability_statistics"
]


# ===== MoE Inference Engine =====

class MoEInferenceEngine:
    """Manages the trained MoE model: base model, LoRA adapters, and router."""

    def __init__(self, registry_path: str):
        with open(registry_path) as f:
            self.registry = json.load(f)
        # Normalise key: new registry uses "experts", old uses "domain_experts"
        if "experts" in self.registry and "domain_experts" not in self.registry:
            self.registry["domain_experts"] = {
                k: v for k, v in self.registry["experts"].items() if k != "shared"
            }
            if "shared" in self.registry["experts"] and "shared_expert" not in self.registry:
                self.registry["shared_expert"] = self.registry["experts"]["shared"]
        self.base_model = None
        self.tokenizer = None
        self.router = None
        self.current_adapter = None
        self.adapters_loaded = set()

    def load_base(self):
        """Load base model with 4-bit quantization."""
        model_path = self.registry["base_model"]
        local_model = os.path.join(MODELS_DIR, "deepseek-math-7b-base")
        if os.path.exists(local_model):
            model_path = local_model

        print(f"Loading base model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        print(f"Base model loaded. Hidden size: {self.base_model.config.hidden_size}")

    def load_router(self):
        """Load trained sigmoid router."""
        hidden_size = self.base_model.config.hidden_size
        num_experts = len(EXPERT_DOMAINS)
        self.router = MathDomainRouter(
            hidden_size=hidden_size, num_experts=num_experts, top_k=2
        )
        router_path = self.registry["router"]
        state = torch.load(router_path, map_location="cpu", weights_only=True)
        self.router.load_state_dict(state)
        device = next(self.base_model.parameters()).device
        self.router.to(device).eval()
        print(f"Router loaded from {router_path}")

    def load_expert(self, domain: str):
        """Load a domain LoRA adapter. Uses named adapters for fast switching."""
        if domain not in self.registry["domain_experts"]:
            print(f"Warning: No expert for domain '{domain}'")
            return False

        adapter_path = self.registry["domain_experts"][domain]
        if not os.path.exists(adapter_path):
            print(f"Warning: Adapter path not found: {adapter_path}")
            return False

        if domain in self.adapters_loaded:
            # Already loaded, just switch
            self.base_model.set_adapter(domain)
            self.current_adapter = domain
            return True

        # First adapter: wrap with PeftModel
        if not self.adapters_loaded:
            self.base_model = PeftModel.from_pretrained(
                self.base_model, adapter_path, adapter_name=domain
            )
        else:
            # Additional adapters: load into existing PeftModel
            self.base_model.load_adapter(adapter_path, adapter_name=domain)

        self.base_model.set_adapter(domain)
        self.base_model.eval()
        self.adapters_loaded.add(domain)
        self.current_adapter = domain
        print(f"  Expert loaded: {domain} (from {adapter_path})")
        return True

    def route(self, text: str) -> Dict:
        """Route a text input through the sigmoid router."""
        if self.router is None:
            return {"top_experts": [], "scores": {}}

        # Get hidden states from base model
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(next(self.base_model.parameters()).device)

        with torch.no_grad():
            outputs = self.base_model(
                **inputs, output_hidden_states=True
            )
            hidden = outputs.hidden_states[-1]  # [1, T, H]
            mask = inputs["attention_mask"]

        with torch.no_grad():
            weights, indices, scores = self.router(hidden, mask)

        # Format results
        top_experts = []
        for i in range(indices.shape[1]):
            idx = indices[0, i].item()
            w = weights[0, i].item()
            top_experts.append({
                "domain": EXPERT_DOMAINS[idx],
                "weight": round(w, 4),
                "index": idx,
            })

        all_scores = {
            EXPERT_DOMAINS[i]: round(scores[0, i].item(), 4)
            for i in range(len(EXPERT_DOMAINS))
        }

        return {
            "top_experts": top_experts,
            "scores": all_scores,
        }

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.95,
        repetition_penalty: float = 1.15,
        no_repeat_ngram_size: int = 6,
    ) -> str:
        """Generate text using the currently-loaded expert adapter."""
        # Format using training prompt template
        formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"

        inputs = self.tokenizer(
            formatted, return_tensors="pt", truncation=True, max_length=2048
        ).to(next(self.base_model.parameters()).device)

        with torch.no_grad():
            outputs = self.base_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the generated part
        prompt_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][prompt_length:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Truncate at generation artifacts
        for marker in ["### Instruction:", "### Response:", "<|endoftext|>"]:
            if marker in text:
                text = text[:text.index(marker)]

        return text.strip()


# ===== MoE-Adapted Prompts (completion-friendly) =====
#
# These prompts are designed for a completion model fine-tuned on mathematical
# instruction data. The model works best with short, specific prompts that
# give it a single result to generalize or a concrete mathematical setup.

MOE_PATTERN_INTERPOLATION_PROMPT = """State a mathematical conjecture that generalizes the following known result. The conjecture should be a precise mathematical statement using standard notation.

Known result from {domain}:
{result}

Conjecture:"""

MOE_COMPOSITION_PROMPT = """Formulate a new conjecture that combines the following two results into a unified statement. Use precise mathematical notation.

Result 1: {result1}
Result 2: {result2}

Combined conjecture:"""

MOE_BOUNDARY_PROMPT = """The following result in {domain} holds under certain conditions. State a conjecture about what happens when the conditions are weakened or the conclusion is strengthened.

Known result: {result}

Boundary conjecture:"""

MOE_THEOREM_GEN_PROMPT = """Write a new theorem in {domain} involving {concepts}. Use precise mathematical notation with quantifiers and conditions.

Theorem."""

MOE_CROSS_DOMAIN_PROMPT = """The following result is known in {source_domain}:
{source_result}

State an analogous conjecture in {target_domain}, translating the key ideas appropriately.

Conjecture in {target_domain}:"""

MOE_STP_CONJECTURE_PROMPT = """State a novel mathematical conjecture in {domain} that extends or connects the following known results. The conjecture should be precise and non-trivial.

{context}

{previous_note}Conjecture:"""


def _pick_results(knowledge: List[Dict], n: int = 1) -> List[str]:
    """Pick n random results from knowledge base, formatted as text."""
    if not knowledge:
        return [""]
    selected = random.sample(knowledge, min(n, len(knowledge)))
    texts = []
    for e in selected:
        stmt = e.get("formal_statement", e.get("informal_description", ""))
        concepts = ", ".join(e.get("key_concepts", [])[:4])
        text = stmt[:400]
        if concepts:
            text += f" (Key concepts: {concepts})"
        texts.append(text)
    return texts


def _pick_concepts(knowledge: List[Dict], n: int = 3) -> str:
    """Pick n random key concepts from knowledge base entries."""
    all_concepts = []
    for e in knowledge:
        all_concepts.extend(e.get("key_concepts", []))
    if not all_concepts:
        return "algebraic structures"
    unique = list(set(all_concepts))
    selected = random.sample(unique, min(n, len(unique)))
    return ", ".join(selected)


# ===== Post-Processing =====

def extract_conjecture_from_text(raw_text: str, strategy: str, domain: str) -> Dict:
    """Extract structured conjecture data from free-form model output."""
    result = {
        "strategy": strategy,
        "domain": domain,
        "generator": "moe",
        "conjecture_statement": "",
        "informal_description": "",
        "confidence": 0.5,
        "estimated_difficulty": "medium",
        "lean4_sketch": "",
    }

    if not raw_text or len(raw_text.strip()) < 10:
        return result

    text = raw_text.strip()

    # Try to extract the main mathematical statement.
    # The model continues from "Conjecture. Let ..." so the output IS the statement.

    # First, try to find a self-contained statement ending with a period or $$
    # Look for the first complete mathematical statement (ends with . or $$ or \])
    statement_parts = []
    proof_parts = []
    in_proof = False

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        # Detect proof section
        if re.match(r'^(Proof|Sketch|Remark|Note|Observation)[.:]', stripped, re.IGNORECASE):
            in_proof = True
        if in_proof:
            proof_parts.append(stripped)
        else:
            statement_parts.append(stripped)

    # The conjecture statement is everything before the proof
    if statement_parts:
        # Join and clean up
        stmt = " ".join(statement_parts)
        # Truncate at proof-like markers that weren't caught
        for marker in ["Proof.", "Proof:", "Sketch of proof", "We prove this",
                        "The proof follows", "This follows from"]:
            idx = stmt.find(marker)
            if idx > 30:  # Only truncate if there's enough content before
                proof_parts.insert(0, stmt[idx:])
                stmt = stmt[:idx]
                break
        result["conjecture_statement"] = stmt.strip()
    else:
        result["conjecture_statement"] = text[:500]

    # Use proof section as informal description / proof sketch
    if proof_parts:
        result["informal_description"] = " ".join(proof_parts[:3])[:500]

    # If no proof section, try to extract an informal description from
    # parenthetical remarks or sentences without math
    if not result["informal_description"] and len(statement_parts) > 1:
        non_math = [s for s in statement_parts[1:]
                    if "$" not in s and "\\" not in s and len(s) > 20]
        if non_math:
            result["informal_description"] = non_math[0][:300]

    # Heuristic quality assessment based on content
    stmt = result["conjecture_statement"]
    has_math = "$" in stmt or "\\" in stmt
    has_quantifier = any(q in stmt.lower() for q in [
        "for all", "for every", "there exists", "if and only if",
        "let ", "suppose ", "assume ", "given ",
    ])
    is_long_enough = len(stmt) > 50

    # Assign a heuristic confidence based on mathematical content quality
    quality_signals = sum([has_math, has_quantifier, is_long_enough])
    result["confidence"] = 0.3 + 0.2 * quality_signals  # 0.3 to 0.9

    # Try to extract Lean 4 sketch if present
    lean_match = re.search(
        r'(?:lean|lean4|formalization)[:\s]*(.*?)(?:\n\n|\Z)',
        text, re.IGNORECASE | re.DOTALL
    )
    if lean_match:
        result["lean4_sketch"] = lean_match.group(1).strip()[:500]

    return result


def extract_multiple_conjectures(
    raw_text: str, strategy: str, domain: str, max_conjectures: int = 5
) -> List[Dict]:
    """Handle cases where the model generates multiple conjectures."""
    if not raw_text or len(raw_text.strip()) < 10:
        return []

    conjectures = []

    # Try splitting on numbered patterns
    splits = re.split(
        r'\n\s*(?:(?:\d+[\.\)]\s)|(?:[\(][a-zA-Z0-9][\)]\s)|(?:Conjecture\s+\d+[:\.]?\s)|(?:Theorem\s+\d+[:\.]?\s))',
        raw_text
    )

    if len(splits) > 1:
        for part in splits:
            if len(part.strip()) > 20:
                c = extract_conjecture_from_text(part, strategy, domain)
                if c["conjecture_statement"] and len(c["conjecture_statement"]) > 15:
                    conjectures.append(c)
                    if len(conjectures) >= max_conjectures:
                        break

    # Fallback: treat whole text as single conjecture
    if not conjectures:
        c = extract_conjecture_from_text(raw_text, strategy, domain)
        if c["conjecture_statement"] and len(c["conjecture_statement"]) > 15:
            conjectures.append(c)

    return conjectures


# ===== Multi-Strategy Conjecture Generation =====

def generate_conjectures_moe(
    engine: MoEInferenceEngine,
    domain: str,
    knowledge: List[Dict],
    include_theorems: bool = True,
    cross_domain_source: Optional[Tuple[str, List[Dict]]] = None,
    temperatures: List[float] = None,
    n_per_strategy: int = 3,
) -> List[Dict]:
    """Generate conjectures for a domain using the trained MoE model.

    For each strategy, we generate n_per_strategy conjectures by picking
    specific results from the knowledge base and asking the model to
    generalize/extend them. Each generation uses a different randomly
    selected result and temperature.
    """
    if temperatures is None:
        temperatures = [0.7, 0.85, 1.0]

    all_conjectures = []

    # Load the domain expert
    if not engine.load_expert(domain):
        print(f"  WARNING: Could not load expert for {domain}, using base model")

    # Log router decision if available
    sample_result = _pick_results(knowledge, 1)[0]
    routing = engine.route(sample_result[:300])
    if routing["top_experts"]:
        print(f"  Router scores: {routing['top_experts']}")

    # Strategy 1: Pattern interpolation (generalize single results)
    print(f"  Strategy: pattern_interpolation...")
    strategy_conjectures = []
    for i in range(n_per_strategy):
        result = _pick_results(knowledge, 1)[0]
        prompt = MOE_PATTERN_INTERPOLATION_PROMPT.format(
            domain=domain, result=result
        )
        temp = temperatures[i % len(temperatures)]
        try:
            raw = engine.generate(prompt, max_new_tokens=400, temperature=temp)
            if raw:
                conjs = extract_multiple_conjectures(raw, "pattern_interpolation", domain)
                strategy_conjectures.extend(conjs)
        except Exception as e:
            print(f"    Generation error: {e}")
    all_conjectures.extend(_dedup(strategy_conjectures))
    print(f"    -> {len(_dedup(strategy_conjectures))} generated")

    # Strategy 2: Composition (combine two results)
    print(f"  Strategy: composition...")
    strategy_conjectures = []
    for i in range(n_per_strategy):
        results = _pick_results(knowledge, 2)
        r1 = results[0] if len(results) > 0 else ""
        r2 = results[1] if len(results) > 1 else results[0] if results else ""
        prompt = MOE_COMPOSITION_PROMPT.format(result1=r1, result2=r2)
        temp = temperatures[i % len(temperatures)]
        try:
            raw = engine.generate(prompt, max_new_tokens=400, temperature=temp)
            if raw:
                conjs = extract_multiple_conjectures(raw, "composition", domain)
                strategy_conjectures.extend(conjs)
        except Exception as e:
            print(f"    Generation error: {e}")
    all_conjectures.extend(_dedup(strategy_conjectures))
    print(f"    -> {len(_dedup(strategy_conjectures))} generated")

    # Strategy 3: Boundary exploration (weaken/strengthen conditions)
    print(f"  Strategy: boundary_exploration...")
    strategy_conjectures = []
    for i in range(n_per_strategy):
        result = _pick_results(knowledge, 1)[0]
        prompt = MOE_BOUNDARY_PROMPT.format(domain=domain, result=result)
        temp = temperatures[i % len(temperatures)]
        try:
            raw = engine.generate(prompt, max_new_tokens=400, temperature=temp)
            if raw:
                conjs = extract_multiple_conjectures(raw, "boundary_exploration", domain)
                strategy_conjectures.extend(conjs)
        except Exception as e:
            print(f"    Generation error: {e}")
    all_conjectures.extend(_dedup(strategy_conjectures))
    print(f"    -> {len(_dedup(strategy_conjectures))} generated")

    # Strategy 4: Theorem generation (novel theorems from key concepts)
    if include_theorems:
        print(f"  Strategy: theorem_generation...")
        strategy_conjectures = []
        for i in range(n_per_strategy):
            concepts = _pick_concepts(knowledge, 3)
            prompt = MOE_THEOREM_GEN_PROMPT.format(domain=domain, concepts=concepts)
            temp = temperatures[i % len(temperatures)]
            try:
                raw = engine.generate(prompt, max_new_tokens=400, temperature=temp)
                if raw:
                    conjs = extract_multiple_conjectures(raw, "theorem_generation", domain)
                    strategy_conjectures.extend(conjs)
            except Exception as e:
                print(f"    Generation error: {e}")
        all_conjectures.extend(_dedup(strategy_conjectures))
        print(f"    -> {len(_dedup(strategy_conjectures))} generated")

    # Strategy 5: Cross-domain analogy
    if cross_domain_source:
        src_domain, src_knowledge = cross_domain_source
        print(f"  Strategy: cross_domain_analogy ({src_domain} -> {domain})...")

        cross_conjectures = []
        for i in range(n_per_strategy):
            src_result = _pick_results(src_knowledge, 1)[0]
            prompt = MOE_CROSS_DOMAIN_PROMPT.format(
                source_domain=src_domain, source_result=src_result,
                target_domain=domain,
            )
            # Generate from both domain experts
            for expert_domain in [domain, src_domain]:
                if engine.load_expert(expert_domain):
                    try:
                        temp = temperatures[i % len(temperatures)]
                        raw = engine.generate(prompt, max_new_tokens=400, temperature=temp)
                        if raw:
                            conjs = extract_multiple_conjectures(
                                raw, "cross_domain_analogy", domain
                            )
                            for c in conjs:
                                c["source_domain"] = src_domain
                                c["generating_expert"] = expert_domain
                            cross_conjectures.extend(conjs)
                    except Exception as e:
                        print(f"    Cross-domain error ({expert_domain}): {e}")

        unique_cross = _dedup(cross_conjectures)
        all_conjectures.extend(unique_cross)
        print(f"    -> {len(unique_cross)} generated")

    # Switch back to main domain expert
    engine.load_expert(domain)

    return all_conjectures


def _dedup(conjectures: List[Dict]) -> List[Dict]:
    """Deduplicate conjectures by statement text."""
    seen = set()
    unique = []
    for c in conjectures:
        key = c.get("conjecture_statement", "")[:200].strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


# ===== MoE-based STP Loop =====

def run_stp_round_moe(
    engine: MoEInferenceEngine,
    domain: str,
    knowledge: List[Dict],
    api_keys: dict,
    judge_provider: str = "mistral",
    n_conjectures: int = 5,
    previous_conjectures: List[str] = None,
) -> Dict:
    """STP round: MoE generates conjectures, external API proves and judges."""
    context = format_results_for_prompt(knowledge, max_entries=6)
    prev_text = "\n".join(previous_conjectures[-5:]) if previous_conjectures else ""

    round_results = {
        "domain": domain,
        "conjectures": [],
        "stats": {
            "total": 0, "proved": 0, "disproved": 0,
            "partially_proved": 0, "unknown": 0, "trivial": 0,
            "avg_quality": 0.0,
        },
    }

    engine.load_expert(domain)
    quality_scores = []

    for i in range(n_conjectures):
        print(f"    Conjecture {i+1}/{n_conjectures}...")

        # Step 1: MoE generates conjecture
        previous_note = ""
        if prev_text:
            previous_note = f"Avoid repeating these previously generated statements:\n{prev_text}\n\n"

        prompt = MOE_STP_CONJECTURE_PROMPT.format(
            domain=domain, context=context, previous_note=previous_note
        )

        try:
            raw = engine.generate(prompt, max_new_tokens=512, temperature=0.9)
        except Exception as e:
            print(f"      Generation error: {e}")
            continue

        conjecture = extract_conjecture_from_text(raw, "stp", domain)
        conj_text = conjecture["conjecture_statement"]
        if not conj_text or len(conj_text) < 15:
            print(f"      Empty/short generation, skipping")
            continue

        # Step 2: External API attempts proof
        proof_prompt = STP_PROOF_PROMPT.format(
            domain=domain,
            conjecture=conj_text,
            informal=conjecture.get("informal_description", ""),
            hint="Generated by trained MoE domain expert"
        )
        try:
            proof_response = llm_generate(
                proof_prompt, api_keys, provider=judge_provider,
                temperature=0.3, max_tokens=4096
            )
        except Exception as e:
            print(f"      Proof API error: {e}")
            proof_response = ""

        proof_result = parse_json_response(proof_response) if proof_response else []
        if not proof_result:
            proof_result = {
                "verdict": "unknown",
                "raw_response": (proof_response or "")[:500],
            }
        else:
            proof_result = proof_result[0] if isinstance(proof_result, list) else proof_result

        time.sleep(2)

        # Step 3: External API judges
        verdict = proof_result.get("verdict", "unknown")
        proof_text = proof_result.get("proof_or_counterexample", "")

        judge_prompt = STP_JUDGE_PROMPT.format(
            domain=domain,
            conjecture=conj_text,
            verdict=verdict,
            proof=proof_text[:2000],
        )
        try:
            judge_response = llm_generate(
                judge_prompt, api_keys, provider=judge_provider,
                temperature=0.2, max_tokens=2048
            )
        except Exception as e:
            print(f"      Judge API error: {e}")
            judge_response = ""

        judge_result = parse_json_response(judge_response) if judge_response else []
        if not judge_result:
            judge_result = {"overall_score": 0.5, "critique": "Judge unavailable"}
        else:
            judge_result = judge_result[0] if isinstance(judge_result, list) else judge_result

        # Record result (same format as API version)
        result_entry = {
            "conjecture": {
                "conjecture": conj_text,
                "informal": conjecture.get("informal_description", ""),
                "proof_hint": "",
                "lean4_sketch": conjecture.get("lean4_sketch", ""),
            },
            "proof_attempt": proof_result,
            "judge_evaluation": judge_result,
            "domain": domain,
            "quality_score": judge_result.get("overall_score", 0.5),
        }
        round_results["conjectures"].append(result_entry)

        # Update stats
        round_results["stats"]["total"] += 1
        v = verdict.lower().replace(" ", "_")
        if v in round_results["stats"]:
            round_results["stats"][v] += 1

        quality_scores.append(judge_result.get("overall_score", 0.5))

        if previous_conjectures is not None:
            previous_conjectures.append(conj_text[:200])

        print(f"      Verdict: {verdict} | Quality: {judge_result.get('overall_score', '?')}")

        time.sleep(3)

    if quality_scores:
        round_results["stats"]["avg_quality"] = sum(quality_scores) / len(quality_scores)

    return round_results


def run_stp_loop_moe(
    engine: MoEInferenceEngine,
    domain: str,
    knowledge: List[Dict],
    api_keys: dict,
    n_rounds: int = 3,
    n_per_round: int = 5,
    judge_provider: str = "mistral",
    checkpoint_path: str = None,
) -> List[Dict]:
    """Run multiple rounds of MoE-based STP."""
    all_rounds = []
    previous_conjectures = []

    # Resume from checkpoint
    start_round = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        all_rounds = checkpoint.get("rounds", [])
        start_round = len(all_rounds)
        previous_conjectures = checkpoint.get("previous_conjectures", [])
        print(f"  Resuming MoE STP from round {start_round + 1}")

    for round_num in range(start_round, n_rounds):
        print(f"\n  === MoE STP Round {round_num + 1}/{n_rounds} for {domain} ===")

        round_result = run_stp_round_moe(
            engine=engine,
            domain=domain,
            knowledge=knowledge,
            api_keys=api_keys,
            judge_provider=judge_provider,
            n_conjectures=n_per_round,
            previous_conjectures=previous_conjectures,
        )
        all_rounds.append(round_result)

        # Save checkpoint
        if checkpoint_path:
            checkpoint_data = {
                "domain": domain,
                "generator": "moe",
                "rounds": all_rounds,
                "previous_conjectures": previous_conjectures,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint_data, f, indent=2)

        stats = round_result["stats"]
        print(f"  Round {round_num + 1}: {stats['total']} conjectures, "
              f"avg quality: {stats['avg_quality']:.3f}")

    return all_rounds


# ===== Comparison Report =====

def generate_comparison_report(moe_report: Dict, api_report_path: str, output_path: str) -> Dict:
    """Compare MoE vs API conjecture generation results."""
    if not os.path.exists(api_report_path):
        print(f"  API report not found at {api_report_path}, skipping comparison")
        return {}

    with open(api_report_path) as f:
        api_report = json.load(f)

    moe_summary = moe_report.get("summary", {})
    api_summary = api_report.get("summary", {})

    comparison = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "overall": {
            "moe": {
                "total": moe_summary.get("total_conjectures", 0),
                "avg_quality": round(moe_summary.get("avg_quality_score", 0), 4),
            },
            "api": {
                "total": api_summary.get("total_conjectures", 0),
                "avg_quality": round(api_summary.get("avg_quality_score", 0), 4),
            },
            "quality_delta": round(
                moe_summary.get("avg_quality_score", 0) -
                api_summary.get("avg_quality_score", 0), 4
            ),
        },
        "per_domain": {},
        "per_strategy": {},
    }

    # Per-domain comparison
    all_domains = set(
        list(moe_report.get("per_domain", {}).keys()) +
        list(api_report.get("per_domain", {}).keys())
    )
    for domain in sorted(all_domains):
        moe_d = moe_report.get("per_domain", {}).get(domain, {})
        api_d = api_report.get("per_domain", {}).get(domain, {})
        comparison["per_domain"][domain] = {
            "moe_avg": round(moe_d.get("avg_quality", 0), 4),
            "moe_count": moe_d.get("count", 0),
            "moe_max": round(moe_d.get("max_quality", 0), 4),
            "api_avg": round(api_d.get("avg_quality", 0), 4),
            "api_count": api_d.get("count", 0),
            "api_max": round(api_d.get("max_quality", 0), 4),
            "delta": round(
                moe_d.get("avg_quality", 0) - api_d.get("avg_quality", 0), 4
            ),
        }

    # Per-strategy comparison
    all_strategies = set(
        list(moe_report.get("strategy_comparison", {}).keys()) +
        list(api_report.get("strategy_comparison", {}).keys())
    )
    for strategy in sorted(all_strategies):
        moe_s = moe_report.get("strategy_comparison", {}).get(strategy, {})
        api_s = api_report.get("strategy_comparison", {}).get(strategy, {})
        comparison["per_strategy"][strategy] = {
            "moe_avg": round(moe_s.get("avg_quality", 0), 4),
            "moe_count": moe_s.get("count", 0),
            "api_avg": round(api_s.get("avg_quality", 0), 4),
            "api_count": api_s.get("count", 0),
        }

    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)

    # Print comparison
    print("\n" + "=" * 70)
    print("MOE vs API COMPARISON")
    print("=" * 70)
    print(f"\n{'Metric':<30} {'MoE':>12} {'API':>12} {'Delta':>12}")
    print("-" * 66)
    print(f"{'Total conjectures':<30} {comparison['overall']['moe']['total']:>12} "
          f"{comparison['overall']['api']['total']:>12}")
    print(f"{'Avg quality':<30} {comparison['overall']['moe']['avg_quality']:>12.4f} "
          f"{comparison['overall']['api']['avg_quality']:>12.4f} "
          f"{comparison['overall']['quality_delta']:>+12.4f}")

    print(f"\n{'Domain':<25} {'MoE avg':>10} {'API avg':>10} {'Delta':>10}")
    print("-" * 55)
    for domain, stats in sorted(comparison["per_domain"].items()):
        if stats["moe_count"] > 0 or stats["api_count"] > 0:
            print(f"{domain:<25} {stats['moe_avg']:>10.4f} "
                  f"{stats['api_avg']:>10.4f} {stats['delta']:>+10.4f}")

    print(f"\n{'Strategy':<30} {'MoE avg':>10} {'API avg':>10}")
    print("-" * 50)
    for strategy, stats in sorted(comparison["per_strategy"].items()):
        if stats["moe_count"] > 0 or stats["api_count"] > 0:
            print(f"{strategy:<30} {stats['moe_avg']:>10.4f} {stats['api_avg']:>10.4f}")

    print("=" * 70)
    return comparison


# ===== Main =====

def main():
    parser = argparse.ArgumentParser(
        description="MathScy MoE-Based Conjecture Generation Experiment"
    )
    parser.add_argument(
        "--mode", choices=["full", "generate", "stp", "evaluate", "compare"],
        default="full", help="Experiment mode"
    )
    parser.add_argument(
        "--domains", type=str, default=None,
        help="Comma-separated list of domains (default: all 7)"
    )
    parser.add_argument(
        "--judge", type=str, default="mistral",
        help="External API provider for judging (mistral/groq/gemini)"
    )
    parser.add_argument(
        "--rounds", type=int, default=3,
        help="STP rounds per domain"
    )
    parser.add_argument(
        "--per-round", type=int, default=5,
        help="Conjectures per STP round"
    )
    parser.add_argument(
        "--temperatures", type=str, default="0.7,0.85,1.0",
        help="Comma-separated generation temperatures"
    )
    parser.add_argument(
        "--no-router", action="store_true",
        help="Skip loading the router (faster startup)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=RESULTS_DIR,
        help="Output directory"
    )
    parser.add_argument(
        "--registry", type=str,
        default=os.path.join(MODELS_DIR, "expert_registry.json"),
        help="Path to expert registry JSON (default: models/expert_registry.json)"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    temps = [float(t) for t in args.temperatures.split(",")]

    # Determine domains
    if args.domains:
        domains = [d.strip() for d in args.domains.split(",")]
    else:
        domains = EXPERT_DOMAINS

    print("=" * 70)
    print("MathScy MoE-Based Conjecture Generation Experiment")
    print("=" * 70)
    print(f"Mode:        {args.mode}")
    print(f"Domains:     {', '.join(domains)}")
    print(f"Judge:       {args.judge}")
    print(f"Temperatures: {temps}")

    # Output paths — prefix with "new_moe_" when using the new registry
    _is_new_moe = "new_moe" in args.registry
    _prefix = "new_moe_" if _is_new_moe else "moe_"
    moe_conj_path = os.path.join(args.output_dir, f"{_prefix}generated_conjectures.jsonl")
    moe_ranked_path = os.path.join(args.output_dir, f"{_prefix}ranked_conjectures.json")
    moe_report_path = os.path.join(args.output_dir, f"{_prefix}conjecture_evaluation_report.json")
    comparison_path = os.path.join(args.output_dir, f"{_prefix}vs_api_comparison.json")
    api_report_path = os.path.join(args.output_dir, "conjecture_evaluation_report.json")

    # ===== Initialize MoE Engine =====
    if args.mode != "compare":
        registry_path = args.registry
        print(f"\nInitializing MoE engine (registry: {registry_path})...")
        engine = MoEInferenceEngine(registry_path)
        engine.load_base()
        if not args.no_router:
            engine.load_router()
        print("MoE engine ready.\n")

    # Load API keys for judging
    if args.mode in ("full", "stp"):
        print("Loading API keys for judging...")
        api_keys = load_api_keys()
        available = [k for k in api_keys if api_keys[k]]
        print(f"  Available providers: {', '.join(available)}")
        if args.judge not in api_keys:
            fallback = available[0] if available else "gemini"
            print(f"  WARNING: {args.judge} not available, falling back to {fallback}")
            args.judge = fallback

    # Load knowledge base
    if args.mode != "compare":
        kb_path = os.path.join(DATA_DIR, "extracted_knowledge.jsonl")
        print(f"\nLoading knowledge base from {kb_path}...")
        knowledge_base = load_knowledge_base(kb_path)
        for d in sorted(knowledge_base.keys()):
            if d in domains:
                print(f"  {d}: {len(knowledge_base[d])} entries")

    # ===== Phase 1: Multi-Strategy Conjecture Generation =====
    all_multi_strategy = []

    if args.mode in ("full", "generate"):
        print("\n" + "=" * 60)
        print("PHASE 1: MoE Multi-Strategy Conjecture Generation")
        print("=" * 60)

        # Resume from existing file
        processed_domains = set()
        if os.path.exists(moe_conj_path):
            with open(moe_conj_path) as f:
                for line in f:
                    c = json.loads(line)
                    all_multi_strategy.append(c)
                    processed_domains.add(c.get("domain"))
            print(f"  Resumed {len(all_multi_strategy)} existing MoE conjectures "
                  f"from {len(processed_domains)} domains")

        for domain in domains:
            if domain in processed_domains:
                print(f"\n  Skipping {domain} (already processed)")
                continue

            if domain not in knowledge_base or len(knowledge_base[domain]) < 3:
                print(f"\n  Skipping {domain} (insufficient data: "
                      f"{len(knowledge_base.get(domain, []))} entries)")
                continue

            print(f"\n{'=' * 50}")
            print(f"Generating conjectures for: {domain}")
            print(f"{'=' * 50}")

            # Find cross-domain source
            cross_source = None
            for src, tgt in ANALOGY_PAIRS:
                if tgt == domain and src in knowledge_base and len(knowledge_base[src]) >= 3:
                    cross_source = (src, knowledge_base[src])
                    break

            domain_conjs = generate_conjectures_moe(
                engine=engine,
                domain=domain,
                knowledge=knowledge_base[domain],
                include_theorems=True,
                cross_domain_source=cross_source,
                temperatures=temps,
            )

            # Save incrementally
            with open(moe_conj_path, "a") as f:
                for c in domain_conjs:
                    f.write(json.dumps(c) + "\n")

            all_multi_strategy.extend(domain_conjs)
            processed_domains.add(domain)
            print(f"  Total for {domain}: {len(domain_conjs)} conjectures")

        print(f"\nPhase 1 complete: {len(all_multi_strategy)} MoE multi-strategy conjectures")

    elif args.mode in ("evaluate", "compare"):
        if os.path.exists(moe_conj_path):
            with open(moe_conj_path) as f:
                for line in f:
                    all_multi_strategy.append(json.loads(line))
            print(f"Loaded {len(all_multi_strategy)} existing MoE conjectures")

    # ===== Phase 2: STP Loop =====
    stp_results = {}

    if args.mode in ("full", "stp"):
        print("\n" + "=" * 60)
        print("PHASE 2: MoE Self-Play Theorem Prover (STP) Loop")
        print("=" * 60)

        for domain in domains:
            if domain not in knowledge_base or len(knowledge_base[domain]) < 3:
                print(f"  Skipping STP for {domain} (insufficient data)")
                continue

            checkpoint = os.path.join(
                args.output_dir, f"{_prefix}stp_{domain}_checkpoint.json"
            )

            print(f"\n{'=' * 50}")
            print(f"Running MoE STP for: {domain}")
            print(f"{'=' * 50}")

            rounds = run_stp_loop_moe(
                engine=engine,
                domain=domain,
                knowledge=knowledge_base[domain],
                api_keys=api_keys,
                n_rounds=args.rounds,
                n_per_round=args.per_round,
                judge_provider=args.judge,
                checkpoint_path=checkpoint,
            )
            stp_results[domain] = rounds

        total_stp = sum(
            sum(len(r.get("conjectures", [])) for r in rounds)
            for rounds in stp_results.values()
        )
        print(f"\nPhase 2 complete: {total_stp} MoE STP conjectures")

    elif args.mode in ("evaluate", "compare"):
        for domain in domains:
            checkpoint = os.path.join(
                args.output_dir, f"{_prefix}stp_{domain}_checkpoint.json"
            )
            if os.path.exists(checkpoint):
                with open(checkpoint) as f:
                    data = json.load(f)
                stp_results[domain] = data.get("rounds", [])
                print(f"Loaded MoE STP checkpoint for {domain}")

    # ===== Phase 3: Ranking & Evaluation =====
    if args.mode != "compare":
        print("\n" + "=" * 60)
        print("PHASE 3: Ranking & Evaluation Report")
        print("=" * 60)

        ranked = rank_all_conjectures(all_multi_strategy, stp_results)
        print(f"Total ranked conjectures: {len(ranked)}")

        # Save ranked conjectures
        with open(moe_ranked_path, "w") as f:
            serializable = []
            for r in ranked:
                entry = {k: v for k, v in r.items() if k != "full_entry"}
                serializable.append(entry)
            json.dump(serializable, f, indent=2)
        print(f"Ranked conjectures saved to {moe_ranked_path}")

        # Generate evaluation report
        moe_report = generate_evaluation_report(
            all_multi_strategy, stp_results, ranked, moe_report_path
        )
        print_report(moe_report)
    else:
        # Load existing report for comparison
        if os.path.exists(moe_report_path):
            with open(moe_report_path) as f:
                moe_report = json.load(f)
        else:
            print("No MoE report found. Run --mode full first.")
            return

    # ===== Phase 4: Comparison with API Results =====
    print("\n" + "=" * 60)
    print("PHASE 4: MoE vs API Comparison")
    print("=" * 60)

    generate_comparison_report(moe_report, api_report_path, comparison_path)
    print(f"\nComparison saved to {comparison_path}")

    print("\n" + "=" * 70)
    print("MoE experiment complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
