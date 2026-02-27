#!/usr/bin/env python3
"""
MathScy Comprehensive Conjecture Evaluation & Ablation Studies

12 tasks covering evaluation (T1-T6) and ablation studies (T7-T12):
  T1:  Rediscovery Detection (CPU-only)
  T2:  Cross-Validation of Proved Conjectures (9 Groq calls)
  T3:  LLM-Judge MoE Conjectures (113 Mistral calls)
  T4:  STP Loop Extension (~96 API calls)
  T5:  Conjecture Diversity Analysis (CPU-only)
  T6:  Strategy Effectiveness Deep Dive (CPU-only)
  T7:  Strategy Ablation (~120 Mistral calls)
  T8:  Temperature Ablation (~100 Mistral calls)
  T9:  Knowledge Context Ablation (~80 Mistral calls)
  T10: Domain Routing Ablation (~54 Mistral calls)
  T11: STP Round Ablation (CPU-only)
  T12: Cross-Domain Transfer Ablation (~84 Mistral calls)

Usage:
    python scripts/evaluate_conjectures_comprehensive.py --task all --resume
    python scripts/evaluate_conjectures_comprehensive.py --task eval       # T1-T6 only
    python scripts/evaluate_conjectures_comprehensive.py --task ablation   # T7-T12 only
    python scripts/evaluate_conjectures_comprehensive.py --task 3          # single task
    python scripts/evaluate_conjectures_comprehensive.py --task all --dry-run
"""

import os
import sys
import json
import time
import math
import random
import argparse
import re
import traceback
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

# ===== Paths =====
PROJECT_DIR = "/scratch/ctoxtli/moexp"
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
DATA_DIR = os.path.join(PROJECT_DIR, "data")

sys.path.insert(0, os.path.join(PROJECT_DIR, "scripts"))

from evaluate_conjectures import (
    load_api_keys, groq_generate, mistral_generate, llm_generate,
    load_knowledge_base, format_results_for_prompt, parse_json_response,
    score_conjecture, ANALOGY_PAIRS,
    STP_CONJECTURE_PROMPT, STP_PROOF_PROMPT, STP_JUDGE_PROMPT,
)

# ===== Constants =====
ALL_DOMAINS = [
    "algebraic_geometry", "discrete_math", "number_theory",
    "analysis", "algebra", "geometry_topology", "probability_statistics",
]

API_CALL_DELAY = 4  # seconds between API calls (increased to reduce rate limiting)

# Math keywords for rediscovery detection
MATH_KEYWORDS_SET = {
    "ring", "module", "group", "manifold", "prime", "ideal", "field", "algebra",
    "space", "bundle", "sheaf", "cohomology", "homology", "functor", "category",
    "morphism", "variety", "scheme", "divisor", "curve", "surface", "polynomial",
    "operator", "spectrum", "eigenvalue", "kernel", "dimension", "degree",
    "genus", "topology", "homotopy", "sequence", "complex", "tensor", "measure",
    "probability", "distribution", "convergence", "limit", "integral", "derivative",
    "differential", "equation", "inequality", "bound", "norm", "metric", "graph",
    "vertex", "edge", "tree", "lattice", "poset", "partition", "permutation",
    "conjecture", "theorem", "lemma", "corollary", "proposition", "isomorphism",
    "automorphism", "endomorphism", "subgroup", "quotient", "extension", "abelian",
    "nilpotent", "solvable", "finite", "infinite", "countable", "compact", "open",
    "closed", "dense", "connected", "hausdorff", "noetherian", "artinian",
    "regular", "singular", "smooth", "projective", "affine", "local", "global",
    "graded", "filtered", "exact", "injective", "surjective", "bijective",
    "commutative", "associative", "linear", "bilinear", "quadratic", "cubic",
    "rational", "irrational", "algebraic", "transcendental", "analytic",
    "holomorphic", "meromorphic", "symplectic", "riemannian", "euclidean",
    "hyperbolic", "elliptic", "parabolic", "modular", "galois", "frobenius",
    "euler", "gauss", "riemann", "hilbert", "banach", "sobolev", "lebesgue",
}

# ===== Prompts =====

STATEMENT_JUDGE_PROMPT = """You are a senior mathematics professor evaluating a generated mathematical conjecture.

Domain: {domain}
Conjecture: "{conjecture}"

Evaluate this conjecture on these criteria (score each 0.0-1.0):
1. Mathematical Correctness: Is the statement well-formed with correct notation?
2. Novelty: Does it go beyond trivially restating known results?
3. Non-triviality: Is it meaningful (not too easy or vacuously true)?
4. Significance: Would proving/disproving advance knowledge?
5. Formalizability: Could this be stated precisely in Lean 4?

Return ONLY valid JSON (no markdown, no code fences):
{{"correctness": 0.0, "novelty": 0.0, "non_triviality": 0.0, "significance": 0.0, "formalizability": 0.0, "overall_score": 0.0, "is_well_formed": true, "critique": "brief expert critique"}}"""

STATEMENT_JUDGE_WITH_STRATEGY_PROMPT = """You are a senior mathematics professor evaluating a generated mathematical conjecture.

Domain: {domain}
Strategy: {strategy}
Conjecture: "{conjecture}"

Score (0.0-1.0): correctness, novelty, non_triviality, significance, formalizability.
Return JSON: {{"correctness": 0.0, "novelty": 0.0, "non_triviality": 0.0, "significance": 0.0, "formalizability": 0.0, "overall_score": 0.0, "is_well_formed": true, "critique": "brief expert critique"}}"""

CROSS_VALIDATION_JUDGE_PROMPT = """You are a senior mathematics professor independently evaluating a conjecture and its proof.
Domain: {domain}
Conjecture: {conjecture}
Proof: {proof}
Evaluate on: correctness (0-1), novelty (0-1), non_triviality (0-1), significance (0-1), formalizability (0-1), proof_quality (0-1).
Return JSON with these scores, overall_score (0-1), verdict_agreement (agree/disagree/partially), and critique.

Return ONLY valid JSON (no markdown, no code fences):
{{"correctness": 0.0, "novelty": 0.0, "non_triviality": 0.0, "significance": 0.0, "formalizability": 0.0, "proof_quality": 0.0, "overall_score": 0.0, "verdict_agreement": "agree", "critique": "..."}}"""

CONJECTURE_GEN_PROMPT = """You are a creative mathematician working in {domain}.

Based on this mathematical context:
{context}

Using the {strategy} approach, formulate a novel mathematical conjecture.
The conjecture should be:
- Precise and well-formed
- Non-trivial (not an obvious consequence)
- Potentially provable or disprovable

State your conjecture clearly, starting with "Conjecture:" followed by the precise mathematical statement.

Return ONLY valid JSON (no markdown, no code fences):
{{"conjecture_statement": "precise mathematical statement", "informal_description": "plain English description", "confidence": 0.0, "estimated_difficulty": "easy|medium|hard"}}"""

CROSS_DOMAIN_GEN_PROMPT = """You are a creative mathematician finding connections between {source_domain} and {target_domain}.

Source domain ({source_domain}) context:
{source_context}

Target domain ({target_domain}) context:
{target_context}

Formulate a novel conjecture in {target_domain} inspired by results from {source_domain}.
The analogy should be mathematically meaningful, not superficial.

Return ONLY valid JSON (no markdown, no code fences):
{{"conjecture_statement": "precise mathematical statement in target domain", "informal_description": "plain English", "analogy_mapping": "how concepts map", "confidence": 0.0, "estimated_difficulty": "easy|medium|hard"}}"""


# =============================================================================
# Utility Functions
# =============================================================================

def checkpoint_load(path: str) -> Optional[Dict]:
    """Load a checkpoint file if it exists."""
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  Warning: Could not load checkpoint {path}: {e}")
    return None


def checkpoint_save(path: str, data: Dict):
    """Save checkpoint data to file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp_path, path)


def load_all_conjectures() -> Tuple[List[Dict], List[Dict]]:
    """Load all 266 conjectures: 153 API + 113 MoE."""
    api_conjectures = []
    api_path = os.path.join(RESULTS_DIR, "generated_conjectures.jsonl")
    if os.path.exists(api_path):
        with open(api_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        api_conjectures.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    moe_conjectures = []
    moe_path = os.path.join(RESULTS_DIR, "moe_generated_conjectures.jsonl")
    if os.path.exists(moe_path):
        with open(moe_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        moe_conjectures.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    return api_conjectures, moe_conjectures


def load_stp_checkpoints() -> Dict[str, Dict]:
    """Load all 7 STP checkpoint files."""
    checkpoints = {}
    for domain in ALL_DOMAINS:
        path = os.path.join(RESULTS_DIR, f"stp_{domain}_checkpoint.json")
        if os.path.exists(path):
            try:
                with open(path) as f:
                    checkpoints[domain] = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
    return checkpoints


def get_conjecture_text(entry: Dict) -> str:
    """Extract the main conjecture text from various entry formats."""
    for key in ("conjecture_statement", "theorem_statement", "statement", "conjecture"):
        val = entry.get(key, "")
        if isinstance(val, str) and len(val) > 10:
            return val
    return str(entry.get("conjecture_statement", entry.get("statement", "")))


def extract_math_keywords(text: str) -> set:
    """Extract mathematical keywords from a text string."""
    if not text:
        return set()
    text_lower = text.lower()
    words = set(re.findall(r'[a-zA-Z]{3,}', text_lower))
    math_words = words & MATH_KEYWORDS_SET
    # Also extract LaTeX commands
    latex_cmds = set(re.findall(r'\\([a-zA-Z]+)', text))
    # Extract terms inside $...$
    dollar_content = re.findall(r'\$([^$]+)\$', text)
    for content in dollar_content:
        inner_words = set(re.findall(r'[a-zA-Z]{3,}', content.lower()))
        math_words |= (inner_words & MATH_KEYWORDS_SET)
    math_words |= {cmd.lower() for cmd in latex_cmds if len(cmd) >= 3}
    return math_words


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def compute_word_set(text: str) -> set:
    """Extract a set of normalized words from text."""
    if not text:
        return set()
    words = re.findall(r'[a-zA-Z]{2,}', text.lower())
    return set(words)


def safe_mean(values: list) -> float:
    """Compute mean of a list, returning 0.0 for empty lists."""
    return sum(values) / len(values) if values else 0.0


def safe_std(values: list) -> float:
    """Compute standard deviation of a list."""
    if len(values) < 2:
        return 0.0
    m = safe_mean(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def safe_median(values: list) -> float:
    """Compute median of a list."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def shannon_entropy(counts: Dict[str, int]) -> float:
    """Compute Shannon entropy of a distribution."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


def bootstrap_test(group_a: list, group_b: list, n_bootstrap: int = 10000) -> float:
    """Bootstrap permutation test: returns approximate p-value for difference in means."""
    if not group_a or not group_b:
        return 1.0
    observed_diff = abs(safe_mean(group_a) - safe_mean(group_b))
    combined = group_a + group_b
    n_a = len(group_a)
    count_extreme = 0
    for _ in range(n_bootstrap):
        random.shuffle(combined)
        perm_a = combined[:n_a]
        perm_b = combined[n_a:]
        perm_diff = abs(safe_mean(perm_a) - safe_mean(perm_b))
        if perm_diff >= observed_diff:
            count_extreme += 1
    return count_extreme / n_bootstrap


def judge_conjecture(conjecture_text: str, domain: str, api_keys: dict,
                     provider: str = "mistral", strategy: str = None,
                     temperature: float = 0.2) -> Dict:
    """Judge a single conjecture using the statement-only prompt."""
    if strategy:
        prompt = STATEMENT_JUDGE_WITH_STRATEGY_PROMPT.format(
            domain=domain, strategy=strategy, conjecture=conjecture_text[:1500]
        )
    else:
        prompt = STATEMENT_JUDGE_PROMPT.format(
            domain=domain, conjecture=conjecture_text[:1500]
        )
    response = llm_generate(prompt, api_keys, provider=provider,
                            temperature=temperature, max_tokens=1024)
    results = parse_json_response(response) if response else []
    if results:
        result = results[0] if isinstance(results, list) else results
        # Ensure numeric scores
        for key in ("correctness", "novelty", "non_triviality", "significance",
                     "formalizability", "overall_score"):
            if key in result:
                try:
                    result[key] = float(result[key])
                except (ValueError, TypeError):
                    result[key] = 0.5
        return result
    return {
        "correctness": 0.5, "novelty": 0.5, "non_triviality": 0.5,
        "significance": 0.5, "formalizability": 0.5, "overall_score": 0.5,
        "is_well_formed": False, "critique": "Judge returned no valid response",
    }


def generate_conjecture(domain: str, context: str, strategy: str,
                        api_keys: dict, provider: str = "mistral",
                        temperature: float = 0.7) -> Dict:
    """Generate a single conjecture using the simplified prompt."""
    prompt = CONJECTURE_GEN_PROMPT.format(
        domain=domain, context=context, strategy=strategy
    )
    response = llm_generate(prompt, api_keys, provider=provider,
                            temperature=temperature, max_tokens=1024)
    results = parse_json_response(response) if response else []
    if results:
        result = results[0] if isinstance(results, list) else results
        if isinstance(result, dict):
            return result
    # Fallback: try to extract conjecture from raw text
    if response:
        match = re.search(r'Conjecture:\s*(.+?)(?:\n\n|\Z)', response, re.DOTALL)
        if match:
            return {
                "conjecture_statement": match.group(1).strip()[:500],
                "informal_description": "",
                "confidence": 0.5,
                "estimated_difficulty": "medium",
            }
        # Last resort
        return {
            "conjecture_statement": response.strip()[:500],
            "informal_description": "",
            "confidence": 0.3,
            "estimated_difficulty": "medium",
        }
    return {
        "conjecture_statement": "",
        "informal_description": "",
        "confidence": 0.0,
        "estimated_difficulty": "unknown",
    }


def print_task_header(task_num: int, title: str):
    """Print a formatted task header."""
    print(f"\n{'='*70}")
    print(f"  TASK {task_num}: {title}")
    print(f"{'='*70}")


def print_task_summary(task_num: int, result: Dict):
    """Print a brief task summary."""
    print(f"\n  Task {task_num} complete.")
    if "summary" in result:
        for k, v in result["summary"].items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")


# =============================================================================
# Task 1: Rediscovery Detection (CPU-only)
# =============================================================================

def run_task1_rediscovery(dry_run: bool = False, **kwargs) -> Dict:
    """Detect potential rediscoveries by comparing conjectures against ArXiv theorems."""
    print_task_header(1, "Rediscovery Detection")

    output_path = os.path.join(RESULTS_DIR, "task1_rediscovery_detection.json")

    # Load conjectures
    api_conjs, moe_conjs = load_all_conjectures()
    all_conjectures = []
    for c in api_conjs:
        all_conjectures.append({
            "source": "api",
            "domain": c.get("domain", "unknown"),
            "text": get_conjecture_text(c),
            "strategy": c.get("strategy", "unknown"),
        })
    for c in moe_conjs:
        all_conjectures.append({
            "source": "moe",
            "domain": c.get("domain", "unknown"),
            "text": get_conjecture_text(c),
            "strategy": c.get("strategy", "unknown"),
        })

    print(f"  Loaded {len(all_conjectures)} conjectures ({len(api_conjs)} API + {len(moe_conjs)} MoE)")

    if dry_run:
        return {"task": 1, "status": "dry_run", "n_conjectures": len(all_conjectures)}

    # Load ArXiv theorems
    arxiv_path = os.path.join(RESULTS_DIR, "arxiv_extracted_theorems.jsonl")
    print(f"  Loading ArXiv theorems from {arxiv_path}...")

    # Build inverted index: keyword -> list of theorem indices
    inverted_index = defaultdict(list)
    arxiv_theorems = []
    arxiv_keywords = []

    with open(arxiv_path) as f:
        line_count = 0
        for line in f:
            line_count += 1
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            body = entry.get("body", "")
            if not body or len(body) < 20:
                continue
            kw = extract_math_keywords(body)
            store_idx = len(arxiv_theorems)
            arxiv_theorems.append({
                "paper_id": entry.get("paper_id", ""),
                "type": entry.get("type", ""),
                "body": body[:500],  # truncate for storage
            })
            arxiv_keywords.append(kw)
            for word in kw:
                inverted_index[word].append(store_idx)

            if len(arxiv_theorems) % 20000 == 0:
                print(f"    Indexed {len(arxiv_theorems)} theorems...")

    print(f"  Indexed {len(arxiv_theorems)} ArXiv theorems, "
          f"{len(inverted_index)} unique keywords")

    # For each conjecture, find potential rediscoveries
    rediscoveries = []
    threshold = 0.4

    for ci, conj in enumerate(all_conjectures):
        conj_kw = extract_math_keywords(conj["text"])
        if not conj_kw:
            continue

        # Gather candidate theorem indices from inverted index
        candidate_counts = Counter()
        for word in conj_kw:
            for tidx in inverted_index.get(word, []):
                candidate_counts[tidx] += 1

        # Only check candidates that share at least 2 keywords
        best_sim = 0.0
        best_match = None
        for tidx, count in candidate_counts.most_common(100):
            if count < 2:
                break
            sim = jaccard_similarity(conj_kw, arxiv_keywords[tidx])
            if sim > best_sim:
                best_sim = sim
                best_match = tidx

        if best_sim >= threshold and best_match is not None:
            rediscoveries.append({
                "conjecture_index": ci,
                "conjecture_source": conj["source"],
                "conjecture_domain": conj["domain"],
                "conjecture_strategy": conj["strategy"],
                "conjecture_text": conj["text"][:300],
                "matched_theorem_index": best_match,
                "matched_paper_id": arxiv_theorems[best_match]["paper_id"],
                "matched_type": arxiv_theorems[best_match]["type"],
                "matched_body": arxiv_theorems[best_match]["body"][:300],
                "jaccard_similarity": round(best_sim, 4),
            })

        if (ci + 1) % 50 == 0:
            print(f"    Checked {ci + 1}/{len(all_conjectures)} conjectures, "
                  f"found {len(rediscoveries)} potential rediscoveries")

    # Compute statistics
    sim_by_domain = defaultdict(list)
    for r in rediscoveries:
        sim_by_domain[r["conjecture_domain"]].append(r["jaccard_similarity"])

    result = {
        "task": 1,
        "title": "Rediscovery Detection",
        "summary": {
            "total_conjectures": len(all_conjectures),
            "total_arxiv_theorems": len(arxiv_theorems),
            "total_keywords": len(inverted_index),
            "threshold": threshold,
            "potential_rediscoveries": len(rediscoveries),
            "rediscovery_rate": round(len(rediscoveries) / max(len(all_conjectures), 1), 4),
        },
        "per_domain": {
            domain: {
                "count": len(sims),
                "avg_similarity": round(safe_mean(sims), 4),
                "max_similarity": round(max(sims), 4) if sims else 0.0,
            }
            for domain, sims in sim_by_domain.items()
        },
        "rediscoveries": sorted(rediscoveries, key=lambda x: -x["jaccard_similarity"]),
        "status": "complete",
    }

    checkpoint_save(output_path, result)
    print_task_summary(1, result)
    print(f"  Potential rediscoveries found: {len(rediscoveries)}")
    for r in rediscoveries[:5]:
        print(f"    [{r['conjecture_domain']}] sim={r['jaccard_similarity']:.3f} "
              f"paper={r['matched_paper_id']}")

    return result


# =============================================================================
# Task 2: Cross-Validation of Proved Conjectures (9 Groq calls)
# =============================================================================

def run_task2_cross_validation(api_keys: dict, provider: str = "groq",
                               dry_run: bool = False, resume: bool = False,
                               **kwargs) -> Dict:
    """Cross-validate proved conjectures with independent LLM judge."""
    print_task_header(2, "Cross-Validation of Proved Conjectures")

    output_path = os.path.join(RESULTS_DIR, "task2_cross_validation.json")

    # Load proved conjectures from STP checkpoints
    checkpoints = load_stp_checkpoints()
    proved_entries = []

    for domain, ckpt in checkpoints.items():
        for round_data in ckpt.get("rounds", []):
            for entry in round_data.get("conjectures", []):
                verdict = entry.get("proof_attempt", {}).get("verdict", "")
                if verdict == "proved":
                    conj = entry.get("conjecture", {})
                    proof = entry.get("proof_attempt", {})
                    judge = entry.get("judge_evaluation", {})
                    proved_entries.append({
                        "domain": domain,
                        "conjecture_text": conj.get("conjecture", ""),
                        "proof_text": proof.get("proof_or_counterexample", ""),
                        "original_judge": judge,
                        "original_quality": entry.get("quality_score", 0.5),
                    })

    print(f"  Found {len(proved_entries)} proved conjectures across "
          f"{len(checkpoints)} domains")

    if dry_run:
        return {"task": 2, "status": "dry_run", "n_proved": len(proved_entries)}

    # Resume support
    existing = None
    if resume:
        existing = checkpoint_load(output_path)
    completed_indices = set()
    cross_validations = []
    if existing and "cross_validations" in existing:
        cross_validations = existing["cross_validations"]
        completed_indices = {cv["index"] for cv in cross_validations}
        print(f"  Resuming from checkpoint: {len(completed_indices)} already done")

    for i, entry in enumerate(proved_entries):
        if i in completed_indices:
            continue

        conj_text = entry["conjecture_text"]
        proof_text = entry["proof_text"]
        domain = entry["domain"]

        if not conj_text:
            continue

        print(f"  [{i+1}/{len(proved_entries)}] Cross-validating {domain} conjecture...")

        prompt = CROSS_VALIDATION_JUDGE_PROMPT.format(
            domain=domain,
            conjecture=conj_text[:1500],
            proof=proof_text[:2000],
        )

        response = llm_generate(prompt, api_keys, provider=provider,
                                temperature=0.2, max_tokens=1024)
        judge_result = parse_json_response(response) if response else []
        if judge_result:
            judge_result = judge_result[0] if isinstance(judge_result, list) else judge_result
        else:
            judge_result = {"overall_score": 0.5, "critique": "No valid response"}

        # Ensure numeric
        for key in ("correctness", "novelty", "non_triviality", "significance",
                     "formalizability", "proof_quality", "overall_score"):
            if key in judge_result:
                try:
                    judge_result[key] = float(judge_result[key])
                except (ValueError, TypeError):
                    judge_result[key] = 0.5

        cv_entry = {
            "index": i,
            "domain": domain,
            "conjecture_text": conj_text[:300],
            "cross_judge": judge_result,
            "original_quality": entry["original_quality"],
            "cross_quality": judge_result.get("overall_score", 0.5),
            "quality_diff": round(
                judge_result.get("overall_score", 0.5) - entry["original_quality"], 4
            ),
            "verdict_agreement": judge_result.get("verdict_agreement", "unknown"),
        }
        cross_validations.append(cv_entry)

        # Save checkpoint after each call
        partial_result = {
            "task": 2, "cross_validations": cross_validations,
            "status": "in_progress",
        }
        checkpoint_save(output_path, partial_result)

        print(f"    Original: {entry['original_quality']:.3f} | "
              f"Cross: {judge_result.get('overall_score', '?'):.3f} | "
              f"Agreement: {judge_result.get('verdict_agreement', '?')}")

        time.sleep(API_CALL_DELAY)

    # Compute comparison statistics
    original_scores = [cv["original_quality"] for cv in cross_validations]
    cross_scores = [cv["cross_quality"] for cv in cross_validations]
    diffs = [cv["quality_diff"] for cv in cross_validations]
    agreements = Counter(cv["verdict_agreement"] for cv in cross_validations)

    # Pearson correlation (manual to avoid scipy dependency)
    def pearson_r(xs, ys):
        n = len(xs)
        if n < 3:
            return 0.0
        mx, my = safe_mean(xs), safe_mean(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
        den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
        if den_x == 0 or den_y == 0:
            return 0.0
        return num / (den_x * den_y)

    result = {
        "task": 2,
        "title": "Cross-Validation of Proved Conjectures",
        "summary": {
            "n_proved": len(proved_entries),
            "n_validated": len(cross_validations),
            "avg_original_score": round(safe_mean(original_scores), 4),
            "avg_cross_score": round(safe_mean(cross_scores), 4),
            "avg_score_diff": round(safe_mean(diffs), 4),
            "std_score_diff": round(safe_std(diffs), 4),
            "score_correlation": round(pearson_r(original_scores, cross_scores), 4),
            "verdict_agreements": dict(agreements),
        },
        "per_domain": {},
        "cross_validations": cross_validations,
        "status": "complete",
    }

    # Per-domain breakdown
    domain_cvs = defaultdict(list)
    for cv in cross_validations:
        domain_cvs[cv["domain"]].append(cv)
    for domain, cvs in domain_cvs.items():
        orig = [cv["original_quality"] for cv in cvs]
        cross = [cv["cross_quality"] for cv in cvs]
        result["per_domain"][domain] = {
            "count": len(cvs),
            "avg_original": round(safe_mean(orig), 4),
            "avg_cross": round(safe_mean(cross), 4),
            "avg_diff": round(safe_mean([cv["quality_diff"] for cv in cvs]), 4),
        }

    checkpoint_save(output_path, result)
    print_task_summary(2, result)
    return result


# =============================================================================
# Task 3: LLM-Judge MoE Conjectures (113 Mistral calls)
# =============================================================================

def run_task3_moe_llm_judge(api_keys: dict, provider: str = "mistral",
                            dry_run: bool = False, resume: bool = False,
                            **kwargs) -> Dict:
    """Judge all 113 MoE conjectures with LLM."""
    print_task_header(3, "LLM-Judge MoE Conjectures")

    output_path = os.path.join(RESULTS_DIR, "task3_moe_llm_judge.json")

    # Load MoE conjectures
    _, moe_conjs = load_all_conjectures()
    print(f"  Loaded {len(moe_conjs)} MoE conjectures")

    if dry_run:
        return {"task": 3, "status": "dry_run", "n_conjectures": len(moe_conjs)}

    # Resume support
    judged = []
    completed_indices = set()
    if resume:
        existing = checkpoint_load(output_path)
        if existing and "judged_conjectures" in existing:
            judged = existing["judged_conjectures"]
            completed_indices = {j["index"] for j in judged}
            print(f"  Resuming: {len(completed_indices)} already judged")

    for i, conj in enumerate(moe_conjs):
        if i in completed_indices:
            continue

        text = get_conjecture_text(conj)
        domain = conj.get("domain", "unknown")
        strategy = conj.get("strategy", "unknown")

        if not text or len(text) < 10:
            print(f"  [{i+1}/{len(moe_conjs)}] Skipping (empty text)")
            judged.append({
                "index": i, "domain": domain, "strategy": strategy,
                "conjecture_text": text[:200],
                "llm_judge": {"overall_score": 0.0, "critique": "Empty conjecture"},
                "heuristic_score": 0.0,
            })
            continue

        print(f"  [{i+1}/{len(moe_conjs)}] Judging {domain}/{strategy}...")

        judge_result = judge_conjecture(
            text, domain, api_keys, provider=provider, strategy=strategy
        )

        # Compute heuristic score for comparison
        conf = conj.get("confidence", 0.5)
        try:
            conf = float(conf)
        except (ValueError, TypeError):
            conf = 0.5
        diff_map = {"easy": -0.1, "medium": 0.0, "hard": 0.1, "open_problem": 0.15}
        heuristic = 0.3 + 0.4 * conf + diff_map.get(
            conj.get("estimated_difficulty", "medium"), 0.0
        )
        lean = conj.get("lean4_sketch", "")
        if lean and len(lean) > 20:
            heuristic += 0.05
        heuristic = max(0.0, min(1.0, round(heuristic, 4)))

        entry = {
            "index": i,
            "domain": domain,
            "strategy": strategy,
            "conjecture_text": text[:300],
            "llm_judge": judge_result,
            "llm_score": judge_result.get("overall_score", 0.5),
            "heuristic_score": heuristic,
            "score_diff": round(judge_result.get("overall_score", 0.5) - heuristic, 4),
            "is_well_formed": judge_result.get("is_well_formed", None),
        }
        judged.append(entry)

        # Checkpoint every 10
        if (i + 1) % 10 == 0 or i == len(moe_conjs) - 1:
            partial = {"task": 3, "judged_conjectures": judged, "status": "in_progress"}
            checkpoint_save(output_path, partial)
            print(f"    Checkpoint saved ({len(judged)} judged)")

        time.sleep(API_CALL_DELAY)

    # Compile results
    llm_scores = [j["llm_score"] for j in judged if j.get("llm_score")]
    heur_scores = [j["heuristic_score"] for j in judged if j.get("heuristic_score")]
    diffs = [j["score_diff"] for j in judged if j.get("score_diff") is not None]
    well_formed = sum(1 for j in judged if j.get("is_well_formed") is True)

    # Per-domain and per-strategy breakdown
    domain_scores = defaultdict(lambda: {"llm": [], "heur": []})
    strategy_scores = defaultdict(lambda: {"llm": [], "heur": []})
    for j in judged:
        domain_scores[j["domain"]]["llm"].append(j.get("llm_score", 0))
        domain_scores[j["domain"]]["heur"].append(j.get("heuristic_score", 0))
        strategy_scores[j["strategy"]]["llm"].append(j.get("llm_score", 0))
        strategy_scores[j["strategy"]]["heur"].append(j.get("heuristic_score", 0))

    result = {
        "task": 3,
        "title": "LLM-Judge MoE Conjectures",
        "summary": {
            "total_conjectures": len(moe_conjs),
            "total_judged": len(judged),
            "avg_llm_score": round(safe_mean(llm_scores), 4),
            "avg_heuristic_score": round(safe_mean(heur_scores), 4),
            "avg_score_diff": round(safe_mean(diffs), 4),
            "std_score_diff": round(safe_std(diffs), 4),
            "well_formed_count": well_formed,
            "well_formed_rate": round(well_formed / max(len(judged), 1), 4),
        },
        "per_domain": {
            d: {
                "count": len(v["llm"]),
                "avg_llm": round(safe_mean(v["llm"]), 4),
                "avg_heur": round(safe_mean(v["heur"]), 4),
            }
            for d, v in domain_scores.items()
        },
        "per_strategy": {
            s: {
                "count": len(v["llm"]),
                "avg_llm": round(safe_mean(v["llm"]), 4),
                "avg_heur": round(safe_mean(v["heur"]), 4),
            }
            for s, v in strategy_scores.items()
        },
        "judged_conjectures": judged,
        "status": "complete",
    }

    checkpoint_save(output_path, result)
    print_task_summary(3, result)
    return result


# =============================================================================
# Task 4: STP Loop Extension (~96 API calls)
# =============================================================================

def run_task4_stp_extension(api_keys: dict, provider: str = "mistral",
                            judge_provider: str = "groq",
                            dry_run: bool = False, resume: bool = False,
                            knowledge_base: Dict = None, **kwargs) -> Dict:
    """Extend STP loop for domains with 0 proved conjectures."""
    print_task_header(4, "STP Loop Extension")

    output_path = os.path.join(RESULTS_DIR, "task4_stp_extension.json")
    target_domains = ["analysis", "geometry_topology", "algebraic_geometry", "discrete_math"]

    print(f"  Target domains (0 proved): {', '.join(target_domains)}")
    print(f"  Plan: 2 rounds x 4 conjectures x {len(target_domains)} domains")

    if dry_run:
        return {"task": 4, "status": "dry_run", "target_domains": target_domains}

    # Load knowledge base if not provided
    if knowledge_base is None:
        kb_path = os.path.join(DATA_DIR, "extracted_knowledge.jsonl")
        print(f"  Loading knowledge base...")
        knowledge_base = load_knowledge_base(kb_path)

    # Resume support
    existing = checkpoint_load(output_path) if resume else None
    completed_domains = set()
    extension_results = {}
    if existing and "domains" in existing:
        extension_results = existing["domains"]
        completed_domains = set(extension_results.keys())
        print(f"  Resuming: {len(completed_domains)} domains already done")

    for domain in target_domains:
        if domain in completed_domains:
            print(f"\n  Skipping {domain} (already completed)")
            continue

        if domain not in knowledge_base or len(knowledge_base[domain]) < 3:
            print(f"\n  Skipping {domain} (insufficient knowledge: "
                  f"{len(knowledge_base.get(domain, []))} entries)")
            continue

        print(f"\n  --- STP Extension for {domain} ---")

        # Load existing STP checkpoint to get previous conjectures
        stp_ckpt_path = os.path.join(RESULTS_DIR, f"stp_{domain}_checkpoint.json")
        previous_conjectures = []
        if os.path.exists(stp_ckpt_path):
            try:
                with open(stp_ckpt_path) as f:
                    stp_data = json.load(f)
                previous_conjectures = stp_data.get("previous_conjectures", [])
            except (json.JSONDecodeError, IOError):
                pass

        domain_results = {"rounds": [], "stats": {"total": 0, "proved": 0}}

        for round_num in range(2):
            print(f"\n    Round {round_num + 1}/2")
            context = format_results_for_prompt(knowledge_base[domain], max_entries=6)
            prev_text = "\n".join(previous_conjectures[-5:]) if previous_conjectures else "None yet."

            round_conjectures = []

            for conj_idx in range(4):
                print(f"      Conjecture {conj_idx + 1}/4...")

                # Step 1: Generate
                conj_prompt = STP_CONJECTURE_PROMPT.format(
                    domain=domain, context=context, previous=prev_text
                )
                conj_response = llm_generate(conj_prompt, api_keys, provider=provider,
                                             temperature=0.9, max_tokens=2048)
                if not conj_response:
                    print(f"        Empty generation response, skipping")
                    continue

                conjecture = parse_json_response(conj_response)
                if conjecture:
                    conjecture = conjecture[0] if isinstance(conjecture, list) else conjecture
                else:
                    conjecture = {"conjecture": conj_response.strip()[:500],
                                  "informal": "", "proof_hint": ""}

                conj_text = conjecture.get("conjecture", "")
                if not conj_text:
                    print(f"        No conjecture text, skipping")
                    continue

                time.sleep(API_CALL_DELAY)

                # Step 2: Prove
                proof_prompt = STP_PROOF_PROMPT.format(
                    domain=domain,
                    conjecture=conj_text[:1500],
                    informal=conjecture.get("informal", ""),
                    hint=conjecture.get("proof_hint", "No hint"),
                )
                proof_response = llm_generate(proof_prompt, api_keys, provider=provider,
                                              temperature=0.3, max_tokens=4096)
                proof_result = parse_json_response(proof_response) if proof_response else []
                if proof_result:
                    proof_result = proof_result[0] if isinstance(proof_result, list) else proof_result
                else:
                    proof_result = {"verdict": "unknown",
                                    "proof_or_counterexample": (proof_response or "")[:500]}

                time.sleep(API_CALL_DELAY)

                # Step 3: Judge
                verdict = proof_result.get("verdict", "unknown")
                proof_text = proof_result.get("proof_or_counterexample", "")

                actual_judge = judge_provider if judge_provider in api_keys else provider
                judge_prompt_text = STP_JUDGE_PROMPT.format(
                    domain=domain,
                    conjecture=conj_text[:1500],
                    verdict=verdict,
                    proof=proof_text[:2000],
                )
                judge_response = llm_generate(judge_prompt_text, api_keys,
                                              provider=actual_judge,
                                              temperature=0.2, max_tokens=2048)
                judge_result = parse_json_response(judge_response) if judge_response else []
                if judge_result:
                    judge_result = judge_result[0] if isinstance(judge_result, list) else judge_result
                else:
                    judge_result = {"overall_score": 0.5, "critique": "Judge unavailable"}

                quality = judge_result.get("overall_score", 0.5)
                try:
                    quality = float(quality)
                except (ValueError, TypeError):
                    quality = 0.5

                entry = {
                    "conjecture": conjecture,
                    "proof_attempt": proof_result,
                    "judge_evaluation": judge_result,
                    "quality_score": quality,
                    "verdict": verdict,
                }
                round_conjectures.append(entry)
                domain_results["stats"]["total"] += 1
                if verdict == "proved":
                    domain_results["stats"]["proved"] += 1

                previous_conjectures.append(conj_text[:200])

                print(f"        Verdict: {verdict} | Quality: {quality:.3f}")

                time.sleep(API_CALL_DELAY)

            domain_results["rounds"].append({
                "round": round_num + 1,
                "conjectures": round_conjectures,
            })

            # Save incrementally
            extension_results[domain] = domain_results
            partial = {"task": 4, "domains": extension_results, "status": "in_progress"}
            checkpoint_save(output_path, partial)

        extension_results[domain] = domain_results
        completed_domains.add(domain)
        print(f"    {domain}: {domain_results['stats']['total']} conjectures, "
              f"{domain_results['stats']['proved']} proved")

    # Compile final result
    total = sum(d["stats"]["total"] for d in extension_results.values())
    proved = sum(d["stats"]["proved"] for d in extension_results.values())

    result = {
        "task": 4,
        "title": "STP Loop Extension",
        "summary": {
            "target_domains": target_domains,
            "total_new_conjectures": total,
            "total_proved": proved,
            "prove_rate": round(proved / max(total, 1), 4),
        },
        "domains": extension_results,
        "status": "complete",
    }

    checkpoint_save(output_path, result)
    print_task_summary(4, result)
    return result


# =============================================================================
# Task 5: Conjecture Diversity Analysis (CPU-only)
# =============================================================================

def run_task5_diversity(dry_run: bool = False, **kwargs) -> Dict:
    """Analyze diversity of generated conjectures."""
    print_task_header(5, "Conjecture Diversity Analysis")

    output_path = os.path.join(RESULTS_DIR, "task5_diversity_analysis.json")

    api_conjs, moe_conjs = load_all_conjectures()
    all_conjectures = []
    for c in api_conjs:
        all_conjectures.append({
            "source": "api", "domain": c.get("domain", "unknown"),
            "strategy": c.get("strategy", "unknown"),
            "text": get_conjecture_text(c),
        })
    for c in moe_conjs:
        all_conjectures.append({
            "source": "moe", "domain": c.get("domain", "unknown"),
            "strategy": c.get("strategy", "unknown"),
            "text": get_conjecture_text(c),
        })

    print(f"  Loaded {len(all_conjectures)} conjectures")

    if dry_run:
        return {"task": 5, "status": "dry_run", "n_conjectures": len(all_conjectures)}

    # Classify by statement type
    type_patterns = {
        "existence": [r"there\s+exist", r"\\exists", r"exists\s+a"],
        "characterization": [r"if\s+and\s+only\s+if", r"\\iff", r"equivalent\s+to"],
        "bound": [r"at\s+most", r"at\s+least", r"bounded", r"\\leq", r"\\geq",
                  r"upper\s+bound", r"lower\s+bound"],
        "universal": [r"for\s+all", r"\\forall", r"for\s+every", r"for\s+any"],
        "conditional": [r"if\s+.*\s+then", r"\\Rightarrow", r"implies"],
        "equivalence": [r"isomorphic", r"equivalent", r"\\cong", r"\\simeq",
                       r"\\sim", r"homeomorphic", r"diffeomorphic"],
    }

    for conj in all_conjectures:
        text_lower = conj["text"].lower()
        conj["statement_types"] = []
        for stype, patterns in type_patterns.items():
            for pat in patterns:
                if re.search(pat, text_lower):
                    conj["statement_types"].append(stype)
                    break
        if not conj["statement_types"]:
            conj["statement_types"] = ["other"]

        # Extract key concepts
        conj["math_keywords"] = extract_math_keywords(conj["text"])
        conj["word_set"] = compute_word_set(conj["text"])

    # Statement type distribution
    type_counts = Counter()
    for conj in all_conjectures:
        for t in conj["statement_types"]:
            type_counts[t] += 1

    # Per-domain type distribution
    domain_type_counts = defaultdict(lambda: Counter())
    for conj in all_conjectures:
        for t in conj["statement_types"]:
            domain_type_counts[conj["domain"]][t] += 1

    # Concept distribution and entropy
    concept_counts = Counter()
    for conj in all_conjectures:
        for kw in conj["math_keywords"]:
            concept_counts[kw] += 1

    concept_entropy = shannon_entropy(concept_counts)

    # Per-domain concept entropy
    domain_concept_counts = defaultdict(Counter)
    for conj in all_conjectures:
        for kw in conj["math_keywords"]:
            domain_concept_counts[conj["domain"]][kw] += 1

    domain_entropies = {
        d: round(shannon_entropy(counts), 4)
        for d, counts in domain_concept_counts.items()
    }

    # Pairwise Jaccard distances within each domain
    domain_groups = defaultdict(list)
    for i, conj in enumerate(all_conjectures):
        domain_groups[conj["domain"]].append(i)

    domain_diversity = {}
    for domain, indices in domain_groups.items():
        if len(indices) < 2:
            domain_diversity[domain] = {
                "n_conjectures": len(indices),
                "avg_pairwise_jaccard": 0.0,
                "min_pairwise_jaccard": 0.0,
                "max_pairwise_jaccard": 0.0,
            }
            continue

        jaccard_dists = []
        # Sample pairs if too many
        pairs = []
        if len(indices) <= 50:
            for a_idx in range(len(indices)):
                for b_idx in range(a_idx + 1, len(indices)):
                    pairs.append((indices[a_idx], indices[b_idx]))
        else:
            for _ in range(min(500, len(indices) * (len(indices) - 1) // 2)):
                a, b = random.sample(indices, 2)
                pairs.append((a, b))

        for a, b in pairs:
            sim = jaccard_similarity(
                all_conjectures[a]["word_set"],
                all_conjectures[b]["word_set"],
            )
            jaccard_dists.append(1.0 - sim)  # distance = 1 - similarity

        domain_diversity[domain] = {
            "n_conjectures": len(indices),
            "avg_pairwise_jaccard": round(safe_mean(jaccard_dists), 4),
            "min_pairwise_jaccard": round(min(jaccard_dists), 4) if jaccard_dists else 0.0,
            "max_pairwise_jaccard": round(max(jaccard_dists), 4) if jaccard_dists else 0.0,
            "n_pairs_sampled": len(pairs),
        }

    # Source comparison (API vs MoE)
    api_words = set()
    moe_words = set()
    for conj in all_conjectures:
        if conj["source"] == "api":
            api_words |= conj["math_keywords"]
        else:
            moe_words |= conj["math_keywords"]

    result = {
        "task": 5,
        "title": "Conjecture Diversity Analysis",
        "summary": {
            "total_conjectures": len(all_conjectures),
            "n_statement_types": len(type_counts),
            "concept_entropy": round(concept_entropy, 4),
            "unique_concepts": len(concept_counts),
            "top_20_concepts": dict(concept_counts.most_common(20)),
        },
        "statement_type_distribution": dict(type_counts),
        "per_domain_types": {
            d: dict(counts) for d, counts in domain_type_counts.items()
        },
        "domain_concept_entropy": domain_entropies,
        "domain_diversity": domain_diversity,
        "source_comparison": {
            "api_unique_concepts": len(api_words),
            "moe_unique_concepts": len(moe_words),
            "shared_concepts": len(api_words & moe_words),
            "api_only": len(api_words - moe_words),
            "moe_only": len(moe_words - api_words),
            "concept_overlap": round(
                len(api_words & moe_words) / max(len(api_words | moe_words), 1), 4
            ),
        },
        "status": "complete",
    }

    checkpoint_save(output_path, result)
    print_task_summary(5, result)
    return result


# =============================================================================
# Task 6: Strategy Effectiveness Deep Dive (CPU-only)
# =============================================================================

def run_task6_strategy_effectiveness(dry_run: bool = False, **kwargs) -> Dict:
    """Deep analysis of strategy effectiveness."""
    print_task_header(6, "Strategy Effectiveness Deep Dive")

    output_path = os.path.join(RESULTS_DIR, "task6_strategy_effectiveness.json")

    # Load ranked conjectures (153 API conjectures with quality scores)
    ranked_path = os.path.join(RESULTS_DIR, "ranked_conjectures.json")
    if not os.path.exists(ranked_path):
        print(f"  ERROR: {ranked_path} not found")
        return {"task": 6, "status": "error", "message": "ranked_conjectures.json not found"}

    with open(ranked_path) as f:
        ranked = json.load(f)

    print(f"  Loaded {len(ranked)} ranked conjectures")

    if dry_run:
        return {"task": 6, "status": "dry_run", "n_conjectures": len(ranked)}

    # Group by (strategy, domain)
    pair_groups = defaultdict(list)
    strategy_groups = defaultdict(list)
    domain_groups = defaultdict(list)

    for entry in ranked:
        strategy = entry.get("strategy", "unknown")
        domain = entry.get("domain", "unknown")
        score = entry.get("quality_score", 0.0)
        try:
            score = float(score)
        except (ValueError, TypeError):
            score = 0.0

        pair_groups[(strategy, domain)].append(score)
        strategy_groups[strategy].append(score)
        domain_groups[domain].append(score)

    # Per-pair statistics
    pair_stats = {}
    for (strategy, domain), scores in pair_groups.items():
        pair_stats[f"{strategy}|{domain}"] = {
            "strategy": strategy,
            "domain": domain,
            "count": len(scores),
            "mean": round(safe_mean(scores), 4),
            "std": round(safe_std(scores), 4),
            "median": round(safe_median(scores), 4),
            "min": round(min(scores), 4) if scores else 0.0,
            "max": round(max(scores), 4) if scores else 0.0,
        }

    # Strategy comparison with statistical tests
    strategy_stats = {}
    strategy_names = sorted(strategy_groups.keys())
    for s in strategy_names:
        scores = strategy_groups[s]
        strategy_stats[s] = {
            "count": len(scores),
            "mean": round(safe_mean(scores), 4),
            "std": round(safe_std(scores), 4),
            "median": round(safe_median(scores), 4),
        }

    # Pairwise comparisons (try scipy first, bootstrap fallback)
    pairwise_tests = {}
    try:
        from scipy.stats import mannwhitneyu
        use_scipy = True
    except ImportError:
        use_scipy = False

    for i, s1 in enumerate(strategy_names):
        for s2 in strategy_names[i + 1:]:
            g1 = strategy_groups[s1]
            g2 = strategy_groups[s2]
            if len(g1) < 2 or len(g2) < 2:
                continue
            key = f"{s1}_vs_{s2}"
            if use_scipy:
                try:
                    stat, p_value = mannwhitneyu(g1, g2, alternative="two-sided")
                    pairwise_tests[key] = {
                        "test": "mann_whitney_u",
                        "statistic": round(float(stat), 4),
                        "p_value": round(float(p_value), 6),
                        "significant_005": p_value < 0.05,
                        "mean_diff": round(safe_mean(g1) - safe_mean(g2), 4),
                    }
                except Exception:
                    p_value = bootstrap_test(g1, g2, n_bootstrap=5000)
                    pairwise_tests[key] = {
                        "test": "bootstrap_permutation",
                        "p_value": round(p_value, 6),
                        "significant_005": p_value < 0.05,
                        "mean_diff": round(safe_mean(g1) - safe_mean(g2), 4),
                    }
            else:
                p_value = bootstrap_test(g1, g2, n_bootstrap=5000)
                pairwise_tests[key] = {
                    "test": "bootstrap_permutation",
                    "p_value": round(p_value, 6),
                    "significant_005": p_value < 0.05,
                    "mean_diff": round(safe_mean(g1) - safe_mean(g2), 4),
                }

    # Identify top and bottom pairs
    sorted_pairs = sorted(pair_stats.items(), key=lambda x: -x[1]["mean"])
    top_5 = [{"pair": k, **v} for k, v in sorted_pairs[:5]]
    bottom_5 = [{"pair": k, **v} for k, v in sorted_pairs[-5:]]

    result = {
        "task": 6,
        "title": "Strategy Effectiveness Deep Dive",
        "summary": {
            "total_conjectures": len(ranked),
            "n_strategies": len(strategy_groups),
            "n_domains": len(domain_groups),
            "n_strategy_domain_pairs": len(pair_stats),
            "best_strategy": sorted_pairs[0][0] if sorted_pairs else "N/A",
            "best_strategy_mean": sorted_pairs[0][1]["mean"] if sorted_pairs else 0.0,
        },
        "strategy_stats": strategy_stats,
        "domain_stats": {
            d: {
                "count": len(scores),
                "mean": round(safe_mean(scores), 4),
                "std": round(safe_std(scores), 4),
            }
            for d, scores in domain_groups.items()
        },
        "pair_stats": pair_stats,
        "pairwise_tests": pairwise_tests,
        "top_5_pairs": top_5,
        "bottom_5_pairs": bottom_5,
        "status": "complete",
    }

    checkpoint_save(output_path, result)
    print_task_summary(6, result)
    return result


# =============================================================================
# Task 7: Strategy Ablation (~120 Mistral calls)
# =============================================================================

def run_task7_strategy_ablation(api_keys: dict, provider: str = "mistral",
                                dry_run: bool = False, resume: bool = False,
                                knowledge_base: Dict = None, **kwargs) -> Dict:
    """Ablation study: compare strategies across domains."""
    print_task_header(7, "Strategy Ablation")

    output_path = os.path.join(RESULTS_DIR, "task7_strategy_ablation.json")

    strategies = [
        "pattern_interpolation", "composition", "boundary_exploration",
        "theorem_generation", "cross_domain_analogy",
    ]
    test_domains = ["algebra", "analysis", "discrete_math", "algebraic_geometry"]
    n_per_cell = 3

    print(f"  {len(strategies)} strategies x {len(test_domains)} domains x "
          f"{n_per_cell} conjectures = {len(strategies)*len(test_domains)*n_per_cell} generations")

    if dry_run:
        return {"task": 7, "status": "dry_run",
                "strategies": strategies, "domains": test_domains}

    if knowledge_base is None:
        kb_path = os.path.join(DATA_DIR, "extracted_knowledge.jsonl")
        print(f"  Loading knowledge base...")
        knowledge_base = load_knowledge_base(kb_path)

    # Resume
    existing = checkpoint_load(output_path) if resume else None
    ablation_results = {}
    completed_cells = set()
    if existing and "cells" in existing:
        ablation_results = existing.get("cells", {})
        completed_cells = set(ablation_results.keys())
        print(f"  Resuming: {len(completed_cells)} cells already done")

    for strategy in strategies:
        for domain in test_domains:
            cell_key = f"{strategy}|{domain}"
            if cell_key in completed_cells:
                continue

            print(f"\n  [{strategy}] [{domain}] Generating {n_per_cell} conjectures...")

            if domain not in knowledge_base or len(knowledge_base[domain]) < 3:
                print(f"    Skipping (insufficient knowledge)")
                ablation_results[cell_key] = {
                    "strategy": strategy, "domain": domain,
                    "conjectures": [], "scores": [], "skipped": True,
                }
                continue

            context = format_results_for_prompt(knowledge_base[domain], max_entries=6)
            cell_conjectures = []

            for ci in range(n_per_cell):
                # Generate
                if strategy == "cross_domain_analogy":
                    # Find a source domain
                    src_domain = None
                    for src, tgt in ANALOGY_PAIRS:
                        if tgt == domain and src in knowledge_base:
                            src_domain = src
                            break
                    if not src_domain:
                        src_domain = [d for d in test_domains if d != domain][0]
                    src_context = format_results_for_prompt(
                        knowledge_base.get(src_domain, []), max_entries=4
                    )
                    prompt = CROSS_DOMAIN_GEN_PROMPT.format(
                        source_domain=src_domain, target_domain=domain,
                        source_context=src_context, target_context=context,
                    )
                else:
                    prompt = CONJECTURE_GEN_PROMPT.format(
                        domain=domain, context=context, strategy=strategy,
                    )

                response = llm_generate(prompt, api_keys, provider=provider,
                                        temperature=0.7, max_tokens=1024)
                gen_result = parse_json_response(response) if response else []
                if gen_result:
                    gen_result = gen_result[0] if isinstance(gen_result, list) else gen_result
                else:
                    gen_result = {"conjecture_statement": (response or "")[:500]}

                conj_text = gen_result.get("conjecture_statement",
                            gen_result.get("conjecture", ""))
                if not conj_text:
                    conj_text = (response or "")[:500]

                time.sleep(API_CALL_DELAY)

                # Judge
                judge_result = judge_conjecture(
                    conj_text, domain, api_keys, provider=provider, strategy=strategy
                )
                time.sleep(API_CALL_DELAY)

                cell_conjectures.append({
                    "conjecture_text": conj_text[:300],
                    "generation": gen_result,
                    "judge": judge_result,
                    "score": judge_result.get("overall_score", 0.5),
                })

                print(f"    [{ci+1}/{n_per_cell}] Score: "
                      f"{judge_result.get('overall_score', '?'):.3f}")

            scores = [c["score"] for c in cell_conjectures]
            ablation_results[cell_key] = {
                "strategy": strategy,
                "domain": domain,
                "conjectures": cell_conjectures,
                "scores": scores,
                "mean_score": round(safe_mean(scores), 4),
                "std_score": round(safe_std(scores), 4),
            }
            completed_cells.add(cell_key)

            # Checkpoint
            partial = {"task": 7, "cells": ablation_results, "status": "in_progress"}
            checkpoint_save(output_path, partial)

    # Compile results
    strategy_means = defaultdict(list)
    domain_means = defaultdict(list)
    for cell_key, cell in ablation_results.items():
        if cell.get("skipped"):
            continue
        strategy_means[cell["strategy"]].append(cell.get("mean_score", 0))
        domain_means[cell["domain"]].append(cell.get("mean_score", 0))

    result = {
        "task": 7,
        "title": "Strategy Ablation",
        "summary": {
            "n_strategies": len(strategies),
            "n_domains": len(test_domains),
            "n_per_cell": n_per_cell,
            "total_cells": len(ablation_results),
            "best_strategy": max(strategy_means.items(),
                                 key=lambda x: safe_mean(x[1]))[0] if strategy_means else "N/A",
        },
        "strategy_summary": {
            s: {"mean": round(safe_mean(v), 4), "std": round(safe_std(v), 4)}
            for s, v in strategy_means.items()
        },
        "domain_summary": {
            d: {"mean": round(safe_mean(v), 4), "std": round(safe_std(v), 4)}
            for d, v in domain_means.items()
        },
        "cells": ablation_results,
        "status": "complete",
    }

    checkpoint_save(output_path, result)
    print_task_summary(7, result)
    return result


# =============================================================================
# Task 8: Temperature Ablation (~100 Mistral calls)
# =============================================================================

def run_task8_temperature_ablation(api_keys: dict, provider: str = "mistral",
                                   dry_run: bool = False, resume: bool = False,
                                   knowledge_base: Dict = None, **kwargs) -> Dict:
    """Ablation study: effect of temperature on conjecture quality and diversity."""
    print_task_header(8, "Temperature Ablation")

    output_path = os.path.join(RESULTS_DIR, "task8_temperature_ablation.json")

    temperatures = [0.3, 0.5, 0.7, 0.9, 1.1]
    test_domains = ["algebra", "analysis"]
    strategy = "pattern_interpolation"
    n_per_cell = 5

    total_calls = len(temperatures) * len(test_domains) * n_per_cell * 2  # gen + judge
    print(f"  {len(temperatures)} temps x {len(test_domains)} domains x "
          f"{n_per_cell} conjectures = ~{total_calls} API calls")

    if dry_run:
        return {"task": 8, "status": "dry_run",
                "temperatures": temperatures, "domains": test_domains}

    if knowledge_base is None:
        kb_path = os.path.join(DATA_DIR, "extracted_knowledge.jsonl")
        print(f"  Loading knowledge base...")
        knowledge_base = load_knowledge_base(kb_path)

    # Resume
    existing = checkpoint_load(output_path) if resume else None
    ablation_results = {}
    completed_cells = set()
    if existing and "cells" in existing:
        ablation_results = existing.get("cells", {})
        completed_cells = set(ablation_results.keys())
        print(f"  Resuming: {len(completed_cells)} cells already done")

    for temp in temperatures:
        for domain in test_domains:
            cell_key = f"t{temp}|{domain}"
            if cell_key in completed_cells:
                continue

            print(f"\n  [temp={temp}] [{domain}] Generating {n_per_cell} conjectures...")

            if domain not in knowledge_base or len(knowledge_base[domain]) < 3:
                print(f"    Skipping (insufficient knowledge)")
                continue

            context = format_results_for_prompt(knowledge_base[domain], max_entries=6)
            cell_conjectures = []

            for ci in range(n_per_cell):
                # Generate at specified temperature
                prompt = CONJECTURE_GEN_PROMPT.format(
                    domain=domain, context=context, strategy=strategy,
                )
                response = llm_generate(prompt, api_keys, provider=provider,
                                        temperature=temp, max_tokens=1024)
                gen_result = parse_json_response(response) if response else []
                if gen_result:
                    gen_result = gen_result[0] if isinstance(gen_result, list) else gen_result
                else:
                    gen_result = {"conjecture_statement": (response or "")[:500]}

                conj_text = gen_result.get("conjecture_statement",
                            gen_result.get("conjecture", ""))
                if not conj_text:
                    conj_text = (response or "")[:500]

                time.sleep(API_CALL_DELAY)

                # Judge (always at low temp for consistency)
                judge_result = judge_conjecture(
                    conj_text, domain, api_keys, provider=provider
                )
                time.sleep(API_CALL_DELAY)

                cell_conjectures.append({
                    "conjecture_text": conj_text[:300],
                    "score": judge_result.get("overall_score", 0.5),
                    "judge": judge_result,
                    "word_set": list(compute_word_set(conj_text))[:50],
                })

                print(f"    [{ci+1}/{n_per_cell}] Score: "
                      f"{judge_result.get('overall_score', '?'):.3f}")

            # Compute quality and diversity
            scores = [c["score"] for c in cell_conjectures]

            # Diversity: average pairwise Jaccard distance
            jaccard_dists = []
            for a in range(len(cell_conjectures)):
                for b in range(a + 1, len(cell_conjectures)):
                    ws_a = set(cell_conjectures[a].get("word_set", []))
                    ws_b = set(cell_conjectures[b].get("word_set", []))
                    jaccard_dists.append(1.0 - jaccard_similarity(ws_a, ws_b))

            ablation_results[cell_key] = {
                "temperature": temp,
                "domain": domain,
                "strategy": strategy,
                "conjectures": [{k: v for k, v in c.items() if k != "word_set"}
                                for c in cell_conjectures],
                "scores": scores,
                "mean_score": round(safe_mean(scores), 4),
                "std_score": round(safe_std(scores), 4),
                "avg_diversity": round(safe_mean(jaccard_dists), 4),
            }
            completed_cells.add(cell_key)

            # Checkpoint
            partial = {"task": 8, "cells": ablation_results, "status": "in_progress"}
            checkpoint_save(output_path, partial)

    # Compile
    temp_quality = defaultdict(list)
    temp_diversity = defaultdict(list)
    for cell in ablation_results.values():
        t = cell.get("temperature", 0)
        temp_quality[t].append(cell.get("mean_score", 0))
        temp_diversity[t].append(cell.get("avg_diversity", 0))

    result = {
        "task": 8,
        "title": "Temperature Ablation",
        "summary": {
            "temperatures": temperatures,
            "domains": test_domains,
            "strategy": strategy,
            "n_per_cell": n_per_cell,
        },
        "temperature_effects": {
            str(t): {
                "avg_quality": round(safe_mean(v), 4),
                "avg_diversity": round(safe_mean(temp_diversity.get(t, [])), 4),
            }
            for t, v in sorted(temp_quality.items())
        },
        "cells": ablation_results,
        "status": "complete",
    }

    checkpoint_save(output_path, result)
    print_task_summary(8, result)
    return result


# =============================================================================
# Task 9: Knowledge Context Ablation (~80 Mistral calls)
# =============================================================================

def run_task9_context_ablation(api_keys: dict, provider: str = "mistral",
                               dry_run: bool = False, resume: bool = False,
                               knowledge_base: Dict = None, **kwargs) -> Dict:
    """Ablation study: effect of context size on conjecture quality."""
    print_task_header(9, "Knowledge Context Ablation")

    output_path = os.path.join(RESULTS_DIR, "task9_context_ablation.json")

    context_sizes = [1, 3, 6, 10]
    test_domains = ["algebra", "discrete_math"]
    strategy = "pattern_interpolation"
    n_per_cell = 5

    total_calls = len(context_sizes) * len(test_domains) * n_per_cell * 2
    print(f"  {len(context_sizes)} sizes x {len(test_domains)} domains x "
          f"{n_per_cell} = ~{total_calls} API calls")

    if dry_run:
        return {"task": 9, "status": "dry_run",
                "context_sizes": context_sizes, "domains": test_domains}

    if knowledge_base is None:
        kb_path = os.path.join(DATA_DIR, "extracted_knowledge.jsonl")
        print(f"  Loading knowledge base...")
        knowledge_base = load_knowledge_base(kb_path)

    # Resume
    existing = checkpoint_load(output_path) if resume else None
    ablation_results = {}
    completed_cells = set()
    if existing and "cells" in existing:
        ablation_results = existing.get("cells", {})
        completed_cells = set(ablation_results.keys())
        print(f"  Resuming: {len(completed_cells)} cells already done")

    for ctx_size in context_sizes:
        for domain in test_domains:
            cell_key = f"ctx{ctx_size}|{domain}"
            if cell_key in completed_cells:
                continue

            print(f"\n  [context={ctx_size}] [{domain}] "
                  f"Generating {n_per_cell} conjectures...")

            if domain not in knowledge_base or len(knowledge_base[domain]) < ctx_size:
                print(f"    Skipping (insufficient knowledge for size {ctx_size})")
                continue

            cell_conjectures = []

            for ci in range(n_per_cell):
                context = format_results_for_prompt(
                    knowledge_base[domain], max_entries=ctx_size
                )
                prompt = CONJECTURE_GEN_PROMPT.format(
                    domain=domain, context=context, strategy=strategy,
                )
                response = llm_generate(prompt, api_keys, provider=provider,
                                        temperature=0.7, max_tokens=1024)
                gen_result = parse_json_response(response) if response else []
                if gen_result:
                    gen_result = gen_result[0] if isinstance(gen_result, list) else gen_result
                else:
                    gen_result = {"conjecture_statement": (response or "")[:500]}

                conj_text = gen_result.get("conjecture_statement",
                            gen_result.get("conjecture", ""))
                if not conj_text:
                    conj_text = (response or "")[:500]

                time.sleep(API_CALL_DELAY)

                judge_result = judge_conjecture(
                    conj_text, domain, api_keys, provider=provider
                )
                time.sleep(API_CALL_DELAY)

                cell_conjectures.append({
                    "conjecture_text": conj_text[:300],
                    "score": judge_result.get("overall_score", 0.5),
                    "judge": judge_result,
                })

                print(f"    [{ci+1}/{n_per_cell}] Score: "
                      f"{judge_result.get('overall_score', '?'):.3f}")

            scores = [c["score"] for c in cell_conjectures]
            ablation_results[cell_key] = {
                "context_size": ctx_size,
                "domain": domain,
                "conjectures": cell_conjectures,
                "scores": scores,
                "mean_score": round(safe_mean(scores), 4),
                "std_score": round(safe_std(scores), 4),
            }
            completed_cells.add(cell_key)

            partial = {"task": 9, "cells": ablation_results, "status": "in_progress"}
            checkpoint_save(output_path, partial)

    # Compile
    size_quality = defaultdict(list)
    for cell in ablation_results.values():
        s = cell.get("context_size", 0)
        size_quality[s].append(cell.get("mean_score", 0))

    result = {
        "task": 9,
        "title": "Knowledge Context Ablation",
        "summary": {
            "context_sizes": context_sizes,
            "domains": test_domains,
            "n_per_cell": n_per_cell,
        },
        "context_size_effects": {
            str(s): {
                "avg_quality": round(safe_mean(v), 4),
                "std_quality": round(safe_std(v), 4),
            }
            for s, v in sorted(size_quality.items())
        },
        "cells": ablation_results,
        "status": "complete",
    }

    checkpoint_save(output_path, result)
    print_task_summary(9, result)
    return result


# =============================================================================
# Task 10: Domain Routing Ablation (~54 Mistral calls)
# =============================================================================

def run_task10_domain_routing_ablation(api_keys: dict, provider: str = "mistral",
                                       dry_run: bool = False, resume: bool = False,
                                       knowledge_base: Dict = None, **kwargs) -> Dict:
    """Ablation study: effect of domain routing correctness."""
    print_task_header(10, "Domain Routing Ablation")

    output_path = os.path.join(RESULTS_DIR, "task10_domain_routing_ablation.json")

    test_domains = ["algebra", "analysis", "discrete_math"]
    conditions = ["correct_domain", "wrong_domain", "no_context"]
    n_per_cell = 3

    total_calls = len(test_domains) * len(conditions) * n_per_cell * 2
    print(f"  {len(test_domains)} domains x {len(conditions)} conditions x "
          f"{n_per_cell} = ~{total_calls} API calls")

    if dry_run:
        return {"task": 10, "status": "dry_run",
                "domains": test_domains, "conditions": conditions}

    if knowledge_base is None:
        kb_path = os.path.join(DATA_DIR, "extracted_knowledge.jsonl")
        print(f"  Loading knowledge base...")
        knowledge_base = load_knowledge_base(kb_path)

    # Create a mapping of wrong domains
    wrong_domain_map = {
        "algebra": "geometry_topology",
        "analysis": "discrete_math",
        "discrete_math": "number_theory",
    }

    # Resume
    existing = checkpoint_load(output_path) if resume else None
    ablation_results = {}
    completed_cells = set()
    if existing and "cells" in existing:
        ablation_results = existing.get("cells", {})
        completed_cells = set(ablation_results.keys())
        print(f"  Resuming: {len(completed_cells)} cells already done")

    strategy = "pattern_interpolation"

    for domain in test_domains:
        for condition in conditions:
            cell_key = f"{condition}|{domain}"
            if cell_key in completed_cells:
                continue

            print(f"\n  [{condition}] [{domain}] Generating {n_per_cell} conjectures...")

            cell_conjectures = []

            for ci in range(n_per_cell):
                if condition == "correct_domain":
                    if domain not in knowledge_base or len(knowledge_base[domain]) < 3:
                        break
                    context = format_results_for_prompt(
                        knowledge_base[domain], max_entries=6
                    )
                elif condition == "wrong_domain":
                    wrong_d = wrong_domain_map.get(domain, "algebra")
                    if wrong_d not in knowledge_base or len(knowledge_base[wrong_d]) < 3:
                        break
                    context = format_results_for_prompt(
                        knowledge_base[wrong_d], max_entries=6
                    )
                else:  # no_context
                    context = "(No mathematical context provided.)"

                prompt = CONJECTURE_GEN_PROMPT.format(
                    domain=domain, context=context, strategy=strategy,
                )
                response = llm_generate(prompt, api_keys, provider=provider,
                                        temperature=0.7, max_tokens=1024)
                gen_result = parse_json_response(response) if response else []
                if gen_result:
                    gen_result = gen_result[0] if isinstance(gen_result, list) else gen_result
                else:
                    gen_result = {"conjecture_statement": (response or "")[:500]}

                conj_text = gen_result.get("conjecture_statement",
                            gen_result.get("conjecture", ""))
                if not conj_text:
                    conj_text = (response or "")[:500]

                time.sleep(API_CALL_DELAY)

                judge_result = judge_conjecture(
                    conj_text, domain, api_keys, provider=provider
                )
                time.sleep(API_CALL_DELAY)

                cell_conjectures.append({
                    "conjecture_text": conj_text[:300],
                    "score": judge_result.get("overall_score", 0.5),
                    "judge": judge_result,
                })

                print(f"    [{ci+1}/{n_per_cell}] Score: "
                      f"{judge_result.get('overall_score', '?'):.3f}")

            scores = [c["score"] for c in cell_conjectures]
            ablation_results[cell_key] = {
                "condition": condition,
                "domain": domain,
                "conjectures": cell_conjectures,
                "scores": scores,
                "mean_score": round(safe_mean(scores), 4),
                "std_score": round(safe_std(scores), 4),
            }
            completed_cells.add(cell_key)

            partial = {"task": 10, "cells": ablation_results, "status": "in_progress"}
            checkpoint_save(output_path, partial)

    # Compile
    condition_quality = defaultdict(list)
    for cell in ablation_results.values():
        cond = cell.get("condition", "")
        condition_quality[cond].append(cell.get("mean_score", 0))

    result = {
        "task": 10,
        "title": "Domain Routing Ablation",
        "summary": {
            "domains": test_domains,
            "conditions": conditions,
            "n_per_cell": n_per_cell,
        },
        "condition_effects": {
            c: {
                "avg_quality": round(safe_mean(v), 4),
                "std_quality": round(safe_std(v), 4),
            }
            for c, v in condition_quality.items()
        },
        "cells": ablation_results,
        "status": "complete",
    }

    checkpoint_save(output_path, result)
    print_task_summary(10, result)
    return result


# =============================================================================
# Task 11: STP Round Ablation (CPU-only)
# =============================================================================

def run_task11_stp_round_ablation(dry_run: bool = False, **kwargs) -> Dict:
    """Analyze STP quality improvement across rounds."""
    print_task_header(11, "STP Round Ablation")

    output_path = os.path.join(RESULTS_DIR, "task11_stp_round_ablation.json")

    checkpoints = load_stp_checkpoints()
    print(f"  Loaded {len(checkpoints)} STP checkpoint files")

    if dry_run:
        return {"task": 11, "status": "dry_run", "n_checkpoints": len(checkpoints)}

    per_domain = {}
    all_round1_scores = []
    all_round2_scores = []

    for domain, ckpt in checkpoints.items():
        rounds = ckpt.get("rounds", [])
        round_scores = []

        for ri, round_data in enumerate(rounds):
            conjectures = round_data.get("conjectures", [])
            scores = []
            verdicts = Counter()

            for entry in conjectures:
                quality = entry.get("quality_score", 0.5)
                try:
                    quality = float(quality)
                except (ValueError, TypeError):
                    quality = 0.5
                scores.append(quality)

                verdict = entry.get("proof_attempt", {}).get("verdict", "unknown")
                verdicts[verdict] += 1

            round_info = {
                "round": ri + 1,
                "n_conjectures": len(conjectures),
                "scores": scores,
                "mean_score": round(safe_mean(scores), 4),
                "std_score": round(safe_std(scores), 4),
                "median_score": round(safe_median(scores), 4),
                "verdicts": dict(verdicts),
            }
            round_scores.append(round_info)

            if ri == 0:
                all_round1_scores.extend(scores)
            elif ri == 1:
                all_round2_scores.extend(scores)

        # Compute improvement
        if len(round_scores) >= 2:
            r1_mean = round_scores[0]["mean_score"]
            r2_mean = round_scores[1]["mean_score"]
            improvement = round(r2_mean - r1_mean, 4)
        else:
            improvement = 0.0

        per_domain[domain] = {
            "n_rounds": len(rounds),
            "rounds": round_scores,
            "improvement_r1_to_r2": improvement,
            "total_conjectures": sum(r["n_conjectures"] for r in round_scores),
        }

    # Overall round comparison
    round_comparison = {
        "round_1": {
            "n_scores": len(all_round1_scores),
            "mean": round(safe_mean(all_round1_scores), 4),
            "std": round(safe_std(all_round1_scores), 4),
            "median": round(safe_median(all_round1_scores), 4),
        },
        "round_2": {
            "n_scores": len(all_round2_scores),
            "mean": round(safe_mean(all_round2_scores), 4),
            "std": round(safe_std(all_round2_scores), 4),
            "median": round(safe_median(all_round2_scores), 4),
        },
    }

    if all_round1_scores and all_round2_scores:
        round_comparison["improvement"] = round(
            safe_mean(all_round2_scores) - safe_mean(all_round1_scores), 4
        )
        # Bootstrap test for significance
        p_val = bootstrap_test(all_round1_scores, all_round2_scores, n_bootstrap=5000)
        round_comparison["bootstrap_p_value"] = round(p_val, 6)
        round_comparison["significant_005"] = p_val < 0.05

    # Verdict trajectory
    r1_verdicts = Counter()
    r2_verdicts = Counter()
    for domain_data in per_domain.values():
        for rd in domain_data.get("rounds", []):
            if rd["round"] == 1:
                for v, c in rd["verdicts"].items():
                    r1_verdicts[v] += c
            elif rd["round"] == 2:
                for v, c in rd["verdicts"].items():
                    r2_verdicts[v] += c

    result = {
        "task": 11,
        "title": "STP Round Ablation",
        "summary": {
            "n_domains": len(checkpoints),
            "total_rounds": sum(d["n_rounds"] for d in per_domain.values()),
            "total_conjectures": sum(d["total_conjectures"] for d in per_domain.values()),
            "avg_improvement": round(
                safe_mean([d["improvement_r1_to_r2"] for d in per_domain.values()]), 4
            ),
        },
        "round_comparison": round_comparison,
        "verdict_trajectory": {
            "round_1": dict(r1_verdicts),
            "round_2": dict(r2_verdicts),
        },
        "per_domain": per_domain,
        "status": "complete",
    }

    checkpoint_save(output_path, result)
    print_task_summary(11, result)
    return result


# =============================================================================
# Task 12: Cross-Domain Transfer Ablation (~84 Mistral calls)
# =============================================================================

def run_task12_cross_domain_transfer(api_keys: dict, provider: str = "mistral",
                                     dry_run: bool = False, resume: bool = False,
                                     knowledge_base: Dict = None, **kwargs) -> Dict:
    """Ablation study: cross-domain transfer effectiveness."""
    print_task_header(12, "Cross-Domain Transfer Ablation")

    output_path = os.path.join(RESULTS_DIR, "task12_cross_domain_transfer.json")

    analogy_pairs = ANALOGY_PAIRS  # 7 pairs
    n_per_condition = 3

    total_calls = len(analogy_pairs) * 2 * n_per_condition * 2  # cross + baseline, gen + judge
    print(f"  {len(analogy_pairs)} pairs x 2 conditions x {n_per_condition} conjectures "
          f"= ~{total_calls} API calls")

    if dry_run:
        return {"task": 12, "status": "dry_run", "n_pairs": len(analogy_pairs)}

    if knowledge_base is None:
        kb_path = os.path.join(DATA_DIR, "extracted_knowledge.jsonl")
        print(f"  Loading knowledge base...")
        knowledge_base = load_knowledge_base(kb_path)

    # Resume
    existing = checkpoint_load(output_path) if resume else None
    ablation_results = {}
    completed_cells = set()
    if existing and "cells" in existing:
        ablation_results = existing.get("cells", {})
        completed_cells = set(ablation_results.keys())
        print(f"  Resuming: {len(completed_cells)} cells already done")

    for src_domain, tgt_domain in analogy_pairs:
        # --- Cross-domain condition ---
        cross_key = f"cross|{src_domain}->{tgt_domain}"
        if cross_key not in completed_cells:
            print(f"\n  [cross] {src_domain} -> {tgt_domain}")

            src_available = (src_domain in knowledge_base and
                            len(knowledge_base[src_domain]) >= 3)
            tgt_available = (tgt_domain in knowledge_base and
                            len(knowledge_base[tgt_domain]) >= 3)

            if not src_available or not tgt_available:
                print(f"    Skipping (insufficient knowledge)")
                ablation_results[cross_key] = {
                    "condition": "cross_domain",
                    "source": src_domain, "target": tgt_domain,
                    "conjectures": [], "scores": [], "skipped": True,
                }
            else:
                cross_conjectures = []
                for ci in range(n_per_condition):
                    src_context = format_results_for_prompt(
                        knowledge_base[src_domain], max_entries=4
                    )
                    tgt_context = format_results_for_prompt(
                        knowledge_base[tgt_domain], max_entries=4
                    )
                    prompt = CROSS_DOMAIN_GEN_PROMPT.format(
                        source_domain=src_domain, target_domain=tgt_domain,
                        source_context=src_context, target_context=tgt_context,
                    )
                    response = llm_generate(prompt, api_keys, provider=provider,
                                            temperature=0.7, max_tokens=1024)
                    gen_result = parse_json_response(response) if response else []
                    if gen_result:
                        gen_result = gen_result[0] if isinstance(gen_result, list) else gen_result
                    else:
                        gen_result = {"conjecture_statement": (response or "")[:500]}

                    conj_text = gen_result.get("conjecture_statement",
                                gen_result.get("conjecture", ""))
                    if not conj_text:
                        conj_text = (response or "")[:500]

                    time.sleep(API_CALL_DELAY)

                    judge_result = judge_conjecture(
                        conj_text, tgt_domain, api_keys, provider=provider
                    )
                    time.sleep(API_CALL_DELAY)

                    cross_conjectures.append({
                        "conjecture_text": conj_text[:300],
                        "score": judge_result.get("overall_score", 0.5),
                        "judge": judge_result,
                    })
                    print(f"    [cross {ci+1}/{n_per_condition}] Score: "
                          f"{judge_result.get('overall_score', '?'):.3f}")

                scores = [c["score"] for c in cross_conjectures]
                ablation_results[cross_key] = {
                    "condition": "cross_domain",
                    "source": src_domain, "target": tgt_domain,
                    "conjectures": cross_conjectures,
                    "scores": scores,
                    "mean_score": round(safe_mean(scores), 4),
                }

            completed_cells.add(cross_key)
            partial = {"task": 12, "cells": ablation_results, "status": "in_progress"}
            checkpoint_save(output_path, partial)

        # --- Baseline (same-domain) condition ---
        baseline_key = f"baseline|{tgt_domain}"
        if baseline_key not in completed_cells:
            print(f"\n  [baseline] {tgt_domain} (same-domain)")

            if tgt_domain not in knowledge_base or len(knowledge_base[tgt_domain]) < 3:
                print(f"    Skipping (insufficient knowledge)")
                ablation_results[baseline_key] = {
                    "condition": "same_domain",
                    "target": tgt_domain,
                    "conjectures": [], "scores": [], "skipped": True,
                }
            else:
                baseline_conjectures = []
                for ci in range(n_per_condition):
                    context = format_results_for_prompt(
                        knowledge_base[tgt_domain], max_entries=6
                    )
                    prompt = CONJECTURE_GEN_PROMPT.format(
                        domain=tgt_domain, context=context,
                        strategy="pattern_interpolation",
                    )
                    response = llm_generate(prompt, api_keys, provider=provider,
                                            temperature=0.7, max_tokens=1024)
                    gen_result = parse_json_response(response) if response else []
                    if gen_result:
                        gen_result = gen_result[0] if isinstance(gen_result, list) else gen_result
                    else:
                        gen_result = {"conjecture_statement": (response or "")[:500]}

                    conj_text = gen_result.get("conjecture_statement",
                                gen_result.get("conjecture", ""))
                    if not conj_text:
                        conj_text = (response or "")[:500]

                    time.sleep(API_CALL_DELAY)

                    judge_result = judge_conjecture(
                        conj_text, tgt_domain, api_keys, provider=provider
                    )
                    time.sleep(API_CALL_DELAY)

                    baseline_conjectures.append({
                        "conjecture_text": conj_text[:300],
                        "score": judge_result.get("overall_score", 0.5),
                        "judge": judge_result,
                    })
                    print(f"    [baseline {ci+1}/{n_per_condition}] Score: "
                          f"{judge_result.get('overall_score', '?'):.3f}")

                scores = [c["score"] for c in baseline_conjectures]
                ablation_results[baseline_key] = {
                    "condition": "same_domain",
                    "target": tgt_domain,
                    "conjectures": baseline_conjectures,
                    "scores": scores,
                    "mean_score": round(safe_mean(scores), 4),
                }

            completed_cells.add(baseline_key)
            partial = {"task": 12, "cells": ablation_results, "status": "in_progress"}
            checkpoint_save(output_path, partial)

    # Compile: compare cross-domain vs baseline per target domain
    cross_scores_by_target = defaultdict(list)
    baseline_scores_by_target = defaultdict(list)

    for key, cell in ablation_results.items():
        if cell.get("skipped"):
            continue
        if cell["condition"] == "cross_domain":
            cross_scores_by_target[cell["target"]].extend(cell.get("scores", []))
        elif cell["condition"] == "same_domain":
            baseline_scores_by_target[cell["target"]].extend(cell.get("scores", []))

    all_cross = []
    all_baseline = []
    per_target = {}

    for tgt in set(list(cross_scores_by_target.keys()) +
                   list(baseline_scores_by_target.keys())):
        c_scores = cross_scores_by_target.get(tgt, [])
        b_scores = baseline_scores_by_target.get(tgt, [])
        all_cross.extend(c_scores)
        all_baseline.extend(b_scores)

        per_target[tgt] = {
            "cross_mean": round(safe_mean(c_scores), 4),
            "baseline_mean": round(safe_mean(b_scores), 4),
            "cross_n": len(c_scores),
            "baseline_n": len(b_scores),
            "transfer_benefit": round(safe_mean(c_scores) - safe_mean(b_scores), 4),
        }

    result = {
        "task": 12,
        "title": "Cross-Domain Transfer Ablation",
        "summary": {
            "n_pairs": len(analogy_pairs),
            "n_per_condition": n_per_condition,
            "avg_cross_score": round(safe_mean(all_cross), 4),
            "avg_baseline_score": round(safe_mean(all_baseline), 4),
            "avg_transfer_benefit": round(
                safe_mean(all_cross) - safe_mean(all_baseline), 4
            ),
        },
        "per_target_domain": per_target,
        "cells": ablation_results,
        "status": "complete",
    }

    checkpoint_save(output_path, result)
    print_task_summary(12, result)
    return result


# =============================================================================
# Report Compilation
# =============================================================================

def compile_report(results_dir: str = RESULTS_DIR) -> Dict:
    """Compile all task results into a single comprehensive report."""
    print(f"\n{'='*70}")
    print(f"  COMPILING COMPREHENSIVE REPORT")
    print(f"{'='*70}")

    report = {
        "title": "MathScy Comprehensive Conjecture Evaluation & Ablation Studies",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tasks": {},
    }

    task_files = {
        1: "task1_rediscovery_detection.json",
        2: "task2_cross_validation.json",
        3: "task3_moe_llm_judge.json",
        4: "task4_stp_extension.json",
        5: "task5_diversity_analysis.json",
        6: "task6_strategy_effectiveness.json",
        7: "task7_strategy_ablation.json",
        8: "task8_temperature_ablation.json",
        9: "task9_context_ablation.json",
        10: "task10_domain_routing_ablation.json",
        11: "task11_stp_round_ablation.json",
        12: "task12_cross_domain_transfer.json",
    }

    for task_num, filename in task_files.items():
        path = os.path.join(results_dir, filename)
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                # Store only summary and key fields (not full conjecture lists)
                summary_data = {
                    "title": data.get("title", f"Task {task_num}"),
                    "status": data.get("status", "unknown"),
                    "summary": data.get("summary", {}),
                }
                # Include select extra fields
                for key in ("per_domain", "strategy_stats", "round_comparison",
                             "temperature_effects", "context_size_effects",
                             "condition_effects", "per_target_domain",
                             "statement_type_distribution", "source_comparison",
                             "pairwise_tests", "strategy_summary", "domain_summary",
                             "verdict_trajectory"):
                    if key in data:
                        summary_data[key] = data[key]

                report["tasks"][str(task_num)] = summary_data
                print(f"  Task {task_num}: {data.get('title', 'N/A')} - "
                      f"{data.get('status', 'unknown')}")
            except (json.JSONDecodeError, IOError) as e:
                print(f"  Task {task_num}: ERROR loading - {e}")
                report["tasks"][str(task_num)] = {"status": "error", "error": str(e)}
        else:
            print(f"  Task {task_num}: Not found ({filename})")
            report["tasks"][str(task_num)] = {"status": "not_run"}

    # Global summary
    completed = sum(
        1 for t in report["tasks"].values() if t.get("status") == "complete"
    )
    report["global_summary"] = {
        "tasks_completed": completed,
        "tasks_total": 12,
        "completion_rate": round(completed / 12, 4),
    }

    # Key findings
    findings = []

    # T1 findings
    t1 = report["tasks"].get("1", {}).get("summary", {})
    if t1:
        findings.append(
            f"Rediscovery: {t1.get('potential_rediscoveries', 'N/A')} potential "
            f"rediscoveries found (rate: {t1.get('rediscovery_rate', 'N/A')})"
        )

    # T2 findings
    t2 = report["tasks"].get("2", {}).get("summary", {})
    if t2:
        findings.append(
            f"Cross-validation: score correlation {t2.get('score_correlation', 'N/A')}, "
            f"avg diff {t2.get('avg_score_diff', 'N/A')}"
        )

    # T3 findings
    t3 = report["tasks"].get("3", {}).get("summary", {})
    if t3:
        findings.append(
            f"MoE LLM-judge: avg score {t3.get('avg_llm_score', 'N/A')}, "
            f"well-formed rate {t3.get('well_formed_rate', 'N/A')}"
        )

    # T6 findings
    t6 = report["tasks"].get("6", {}).get("summary", {})
    if t6:
        findings.append(
            f"Best strategy-domain pair: {t6.get('best_strategy', 'N/A')} "
            f"(mean {t6.get('best_strategy_mean', 'N/A')})"
        )

    # T8 findings
    t8_effects = report["tasks"].get("8", {}).get("temperature_effects", {})
    if t8_effects:
        best_temp = max(t8_effects.items(),
                        key=lambda x: x[1].get("avg_quality", 0), default=None)
        if best_temp:
            findings.append(
                f"Best temperature: {best_temp[0]} "
                f"(quality {best_temp[1].get('avg_quality', 'N/A')})"
            )

    # T11 findings
    t11 = report["tasks"].get("11", {}).get("round_comparison", {})
    if t11:
        r1 = t11.get("round_1", {}).get("mean", "N/A")
        r2 = t11.get("round_2", {}).get("mean", "N/A")
        findings.append(f"STP rounds: R1 avg={r1}, R2 avg={r2}")

    report["key_findings"] = findings

    # Save
    report_path = os.path.join(results_dir, "comprehensive_evaluation_report.json")
    checkpoint_save(report_path, report)
    print(f"\n  Report saved to {report_path}")

    # Print findings
    print(f"\n  --- Key Findings ---")
    for f in findings:
        print(f"    * {f}")

    return report


# =============================================================================
# Main
# =============================================================================

TASK_MAP = {
    1: ("Rediscovery Detection", run_task1_rediscovery),
    2: ("Cross-Validation of Proved Conjectures", run_task2_cross_validation),
    3: ("LLM-Judge MoE Conjectures", run_task3_moe_llm_judge),
    4: ("STP Loop Extension", run_task4_stp_extension),
    5: ("Conjecture Diversity Analysis", run_task5_diversity),
    6: ("Strategy Effectiveness Deep Dive", run_task6_strategy_effectiveness),
    7: ("Strategy Ablation", run_task7_strategy_ablation),
    8: ("Temperature Ablation", run_task8_temperature_ablation),
    9: ("Knowledge Context Ablation", run_task9_context_ablation),
    10: ("Domain Routing Ablation", run_task10_domain_routing_ablation),
    11: ("STP Round Ablation", run_task11_stp_round_ablation),
    12: ("Cross-Domain Transfer Ablation", run_task12_cross_domain_transfer),
}

EVAL_TASKS = [1, 2, 3, 4, 5, 6]
ABLATION_TASKS = [7, 8, 9, 10, 11, 12]


def main():
    parser = argparse.ArgumentParser(
        description="MathScy Comprehensive Conjecture Evaluation & Ablation Studies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/evaluate_conjectures_comprehensive.py --task all --resume
  python scripts/evaluate_conjectures_comprehensive.py --task eval
  python scripts/evaluate_conjectures_comprehensive.py --task ablation
  python scripts/evaluate_conjectures_comprehensive.py --task 3
  python scripts/evaluate_conjectures_comprehensive.py --task all --dry-run
        """,
    )
    parser.add_argument(
        "--task", type=str, default="all",
        help="Task(s) to run: 1-12, 'eval' (T1-T6), 'ablation' (T7-T12), or 'all'",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoints if available",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without making API calls",
    )
    parser.add_argument(
        "--provider", type=str, default="mistral",
        help="Primary LLM provider (mistral/groq/gemini)",
    )
    parser.add_argument(
        "--judge-provider", type=str, default="groq",
        help="LLM provider for judging in Task 2/4 (groq/mistral/gemini)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("  MathScy Comprehensive Evaluation & Ablation Studies")
    print("=" * 70)
    print(f"  Task:     {args.task}")
    print(f"  Resume:   {args.resume}")
    print(f"  Dry-run:  {args.dry_run}")
    print(f"  Provider: {args.provider}")
    print(f"  Judge:    {args.judge_provider}")
    print()

    # Determine which tasks to run
    if args.task == "all":
        task_nums = list(range(1, 13))
    elif args.task == "eval":
        task_nums = EVAL_TASKS
    elif args.task == "ablation":
        task_nums = ABLATION_TASKS
    else:
        try:
            task_nums = [int(args.task)]
        except ValueError:
            print(f"ERROR: Invalid task specifier: {args.task}")
            print("  Valid: 1-12, 'eval', 'ablation', 'all'")
            sys.exit(1)

    # Validate task numbers
    for tn in task_nums:
        if tn not in TASK_MAP:
            print(f"ERROR: Unknown task number: {tn}")
            sys.exit(1)

    # Load API keys (skip for CPU-only tasks in dry-run)
    cpu_only_tasks = {1, 5, 6, 11}
    needs_api = any(tn not in cpu_only_tasks for tn in task_nums)

    api_keys = {}
    if needs_api and not args.dry_run:
        print("Loading API keys...")
        api_keys = load_api_keys()
        available = [k for k in api_keys if api_keys.get(k)]
        print(f"  Available providers: {', '.join(available)}")

        if args.provider not in api_keys or not api_keys.get(args.provider):
            fallback = next((k for k in ("mistral", "groq", "gemini") if api_keys.get(k)), None)
            if fallback:
                print(f"  WARNING: {args.provider} not available, falling back to {fallback}")
                args.provider = fallback
            elif not args.dry_run:
                print(f"  ERROR: No API providers available!")
                sys.exit(1)

    # Preload knowledge base for tasks that need it
    knowledge_base = None
    kb_needed_tasks = {4, 7, 8, 9, 10, 12}
    if any(tn in kb_needed_tasks for tn in task_nums) and not args.dry_run:
        kb_path = os.path.join(DATA_DIR, "extracted_knowledge.jsonl")
        if os.path.exists(kb_path):
            print(f"\nPreloading knowledge base from {kb_path}...")
            knowledge_base = load_knowledge_base(kb_path)
            for d in sorted(knowledge_base.keys()):
                if d in ALL_DOMAINS:
                    print(f"  {d}: {len(knowledge_base[d])} entries")
        else:
            print(f"  WARNING: Knowledge base not found at {kb_path}")

    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Run tasks
    all_results = {}
    print(f"\nRunning {len(task_nums)} task(s): {task_nums}")

    for tn in task_nums:
        title, func = TASK_MAP[tn]
        print(f"\n{'#'*70}")
        print(f"  Starting Task {tn}: {title}")
        print(f"{'#'*70}")

        try:
            result = func(
                api_keys=api_keys,
                provider=args.provider,
                judge_provider=args.judge_provider,
                dry_run=args.dry_run,
                resume=args.resume,
                knowledge_base=knowledge_base,
            )
            all_results[tn] = result
            status = result.get("status", "complete")
            print(f"\n  Task {tn} finished with status: {status}")

        except Exception as e:
            print(f"\n  ERROR in Task {tn}: {e}")
            traceback.print_exc()
            all_results[tn] = {"task": tn, "status": "error", "error": str(e)}

    # Compile report
    if not args.dry_run:
        report = compile_report(RESULTS_DIR)
    else:
        print("\n  [Dry-run] Skipping report compilation")
        report = {"status": "dry_run"}

    print(f"\n{'='*70}")
    print(f"  ALL TASKS COMPLETE")
    print(f"{'='*70}")
    completed = sum(1 for r in all_results.values()
                    if r.get("status") in ("complete", "dry_run"))
    errored = sum(1 for r in all_results.values() if r.get("status") == "error")
    print(f"  Completed: {completed}/{len(task_nums)}")
    if errored:
        print(f"  Errors:    {errored}")
    print(f"  Results in: {RESULTS_DIR}")
    print()


if __name__ == "__main__":
    main()
