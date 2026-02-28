#!/usr/bin/env python3
"""
Context Investigation Experiment (Task 17)

Investigates the no-context anomaly from Table 14 of the MathScy paper,
where no-context (0.309) outperforms correct-domain context (0.180) in
conjecture generation. This experiment generates 63 conjectures total
(3 conditions x 7 domains x 3 per cell) and evaluates them with Mistral judge.

Conditions:
  1. no_context: Just the domain name, no retrieved context
  2. good_context: Domain name + well-known theorem from that domain
  3. random_context: Domain name + theorem from an unrelated domain

Usage:
    python scripts/context_investigation.py
"""

import os
import sys
import json
import time
import re
import traceback
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional
import urllib.request
import urllib.error

# ===== Configuration =====

PROJECT_DIR = "/scratch/ctoxtli/moexp"
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

with open(os.path.join(PROJECT_DIR, "mistral_key.txt")) as _kf:
    MISTRAL_API_KEY = _kf.read().strip()

# Use Mistral for both generation and judging
GENERATION_MODEL = "mistral-small-latest"
JUDGE_MODEL = "mistral-small-latest"

DELAY_SECONDS = 3
CONJECTURES_PER_CELL = 3  # 3 per (domain, condition) pair

# ===== Domains and Context =====

DOMAINS = [
    "algebra",
    "number_theory",
    "algebraic_geometry",
    "analysis",
    "discrete_math",
    "geometry_topology",
    "probability_statistics",
]

GOOD_CONTEXT = {
    "algebra": "Wedderburn's theorem: Every finite division ring is a field.",
    "number_theory": "Fermat's Last Theorem: For n >= 3, x^n + y^n = z^n has no positive integer solutions.",
    "algebraic_geometry": "Bezout's theorem: Two projective curves of degrees d and e intersect in exactly de points (counted with multiplicity).",
    "analysis": "The Hahn-Banach theorem: Every bounded linear functional on a subspace can be extended to the whole space.",
    "discrete_math": "Ramsey's theorem: For any r,s >= 2, there exists R(r,s) such that any 2-coloring of edges of K_n with n >= R(r,s) contains a red K_r or blue K_s.",
    "geometry_topology": "The Poincare conjecture (now theorem): Every simply connected, closed 3-manifold is homeomorphic to the 3-sphere.",
    "probability_statistics": "The central limit theorem: The sum of n iid random variables with finite variance converges in distribution to a normal.",
}

# Random context: each domain gets a theorem from an unrelated domain
RANDOM_CONTEXT_MAP = {
    "algebra": "analysis",
    "number_theory": "probability_statistics",
    "algebraic_geometry": "discrete_math",
    "analysis": "algebra",
    "discrete_math": "algebraic_geometry",
    "geometry_topology": "number_theory",
    "probability_statistics": "geometry_topology",
}

# ===== Prompt Templates =====

GENERATION_PROMPT = """You are a research mathematician specializing in {domain}. Generate a novel mathematical conjecture.
{context_section}
Provide: 1) formal statement, 2) informal description, 3) motivation, 4) difficulty (easy/medium/hard), 5) proof sketch.
Format as JSON with keys: statement, informal, motivation, difficulty, proof_sketch

Return ONLY valid JSON, no code fences or extra text."""

CONTEXT_SECTIONS = {
    "no_context": "",
    "good_context": "Here is a foundational result in your area:\n{theorem}\nGenerate a conjecture that extends, generalizes, or builds upon ideas from this area.",
    "random_context": "Here is a result from another area of mathematics:\n{theorem}\nGenerate a novel conjecture in {domain}, potentially drawing cross-disciplinary inspiration.",
}

JUDGE_PROMPT = """You are a senior mathematics professor evaluating generated conjectures.

Domain: {domain}
Conjecture: {conjecture}

Evaluate this conjecture on the following criteria (score each 0.0-1.0):

1. **Mathematical Correctness**: Is the statement well-formed and uses correct notation?
2. **Novelty**: Does it go beyond trivially restating known results?
3. **Non-triviality**: Is it a meaningful statement (not too easy or vacuously true)?
4. **Significance**: Would proving/disproving this advance mathematical knowledge?
5. **Formalizability**: Could this be stated precisely in Lean 4?
6. **Proof Quality**: Is the proof sketch rigorous and plausible?

Return JSON:
{{
    "correctness": 0.0-1.0,
    "novelty": 0.0-1.0,
    "non_triviality": 0.0-1.0,
    "significance": 0.0-1.0,
    "formalizability": 0.0-1.0,
    "proof_quality": 0.0-1.0,
    "overall_score": 0.0-1.0,
    "critique": "brief expert critique of the conjecture"
}}

Return ONLY valid JSON, no code fences."""

# ===== API Functions =====

def mistral_call(prompt: str, temperature: float = 0.7, model: str = None,
              max_tokens: int = 1500, max_retries: int = 4) -> Optional[str]:
    """Make a Mistral API call with robust retry logic."""
    url = "https://api.mistral.ai/v1/chat/completions"
    if model is None:
        model = GENERATION_MODEL
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode("utf-8")

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=90) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                content = result["choices"][0]["message"]["content"]
                return content
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            print(f"    [Mistral] HTTP {e.code} (attempt {attempt+1}): {body[:200]}")
            if e.code == 429:
                # Parse retry-after if available, otherwise exponential backoff
                wait = (attempt + 1) * 20
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
            elif e.code in (500, 502, 503):
                wait = (attempt + 1) * 10
                print(f"    Server error, retrying in {wait}s...")
                time.sleep(wait)
            elif attempt < max_retries - 1:
                time.sleep(5)
            else:
                return None
        except Exception as e:
            print(f"    [Mistral] Error (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                return None
    return None


def parse_json_response(text: str) -> Optional[Dict]:
    """Parse JSON from LLM response, handling code fences and extra text."""
    if not text:
        return None

    # Try direct parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try extracting from code fences
    patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

    # Last resort: find first { to last }
    start = text.find('{')
    end = text.rfind('}')
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            pass

    return None


# ===== Experiment Logic =====

def build_generation_prompt(domain: str, condition: str) -> str:
    """Build the generation prompt for a given domain and condition."""
    if condition == "no_context":
        context_section = CONTEXT_SECTIONS["no_context"]
    elif condition == "good_context":
        theorem = GOOD_CONTEXT[domain]
        context_section = CONTEXT_SECTIONS["good_context"].format(theorem=theorem)
    elif condition == "random_context":
        source_domain = RANDOM_CONTEXT_MAP[domain]
        theorem = GOOD_CONTEXT[source_domain]
        context_section = CONTEXT_SECTIONS["random_context"].format(
            theorem=theorem, domain=domain
        )
    else:
        raise ValueError(f"Unknown condition: {condition}")

    return GENERATION_PROMPT.format(domain=domain, context_section=context_section)


def generate_conjecture(domain: str, condition: str, trial: int) -> Dict:
    """Generate a single conjecture and return structured result."""
    prompt = build_generation_prompt(domain, condition)
    print(f"  Generating [{condition}] {domain} #{trial+1}...")

    raw_response = mistral_call(prompt, temperature=0.7, max_tokens=1500)
    if not raw_response:
        print(f"    FAILED: No response from generator")
        return {
            "domain": domain,
            "condition": condition,
            "trial": trial,
            "raw_response": None,
            "parsed": None,
            "generation_success": False,
        }

    parsed = parse_json_response(raw_response)
    if not parsed:
        print(f"    WARNING: Could not parse JSON from response")

    conjecture_text = ""
    if parsed:
        conjecture_text = parsed.get("statement", "") or parsed.get("conjecture_statement", "")
        if not conjecture_text:
            conjecture_text = json.dumps(parsed, indent=2)
    else:
        conjecture_text = raw_response[:500]

    print(f"    Generated: {conjecture_text[:100]}...")

    return {
        "domain": domain,
        "condition": condition,
        "trial": trial,
        "prompt": prompt,
        "raw_response": raw_response,
        "parsed": parsed,
        "conjecture_text": conjecture_text,
        "generation_success": True,
    }


def judge_conjecture(entry: Dict) -> Dict:
    """Judge a single conjecture using Mistral."""
    if not entry.get("generation_success"):
        entry["judge_result"] = None
        entry["overall_score"] = 0.0
        entry["scores"] = {}
        return entry

    domain = entry["domain"]
    conjecture = entry.get("conjecture_text", "")

    # Build a richer conjecture description for the judge
    parsed = entry.get("parsed")
    if parsed:
        conjecture_parts = []
        if parsed.get("statement"):
            conjecture_parts.append(f"Statement: {parsed['statement']}")
        if parsed.get("informal"):
            conjecture_parts.append(f"Informal: {parsed['informal']}")
        if parsed.get("motivation"):
            conjecture_parts.append(f"Motivation: {parsed['motivation']}")
        if parsed.get("difficulty"):
            conjecture_parts.append(f"Difficulty: {parsed['difficulty']}")
        if parsed.get("proof_sketch"):
            conjecture_parts.append(f"Proof sketch: {parsed['proof_sketch']}")
        if conjecture_parts:
            conjecture = "\n".join(conjecture_parts)

    prompt = JUDGE_PROMPT.format(domain=domain, conjecture=conjecture)
    print(f"  Judging [{entry['condition']}] {domain} #{entry['trial']+1}...")

    raw_judge = mistral_call(prompt, temperature=0.3, max_tokens=1000)
    judge_result = parse_json_response(raw_judge) if raw_judge else None

    if judge_result:
        overall = float(judge_result.get("overall_score", 0.5))
        scores = {
            "correctness": float(judge_result.get("correctness", 0.5)),
            "novelty": float(judge_result.get("novelty", 0.5)),
            "non_triviality": float(judge_result.get("non_triviality", 0.5)),
            "significance": float(judge_result.get("significance", 0.5)),
            "formalizability": float(judge_result.get("formalizability", 0.5)),
            "proof_quality": float(judge_result.get("proof_quality", 0.5)),
        }
        critique = judge_result.get("critique", "")
        print(f"    Score: {overall:.3f} | Critique: {critique[:80]}...")
    else:
        overall = 0.5
        scores = {k: 0.5 for k in ["correctness", "novelty", "non_triviality",
                                     "significance", "formalizability", "proof_quality"]}
        critique = "Judge parse failed"
        print(f"    WARNING: Judge response parse failed, using default 0.5")

    entry["judge_raw"] = raw_judge
    entry["judge_result"] = judge_result
    entry["overall_score"] = overall
    entry["scores"] = scores
    entry["critique"] = critique
    return entry


def run_experiment() -> Dict:
    """Run the full context investigation experiment."""
    print("=" * 70)
    print("CONTEXT INVESTIGATION EXPERIMENT (Task 17)")
    print("Investigating no-context anomaly from Table 14")
    print("=" * 70)
    print(f"Generator: Mistral {GENERATION_MODEL} (temp=0.7)")
    print(f"Judge: Mistral {JUDGE_MODEL} (temp=0.3)")
    print(f"Domains: {len(DOMAINS)}")
    print(f"Conditions: no_context, good_context, random_context")
    print(f"Conjectures per cell: {CONJECTURES_PER_CELL}")
    print(f"Total conjectures: {len(DOMAINS) * 3 * CONJECTURES_PER_CELL}")
    print(f"Delay between API calls: {DELAY_SECONDS}s")
    print("=" * 70)

    conditions = ["no_context", "good_context", "random_context"]
    all_results = []
    generation_failures = 0
    judge_failures = 0

    # Phase 1: Generate all conjectures
    print("\n===== PHASE 1: GENERATION =====\n")
    for domain in DOMAINS:
        print(f"\n--- Domain: {domain} ---")
        for condition in conditions:
            for trial in range(CONJECTURES_PER_CELL):
                result = generate_conjecture(domain, condition, trial)
                all_results.append(result)
                if not result["generation_success"]:
                    generation_failures += 1
                time.sleep(DELAY_SECONDS)

    print(f"\n  Generation complete: {len(all_results)} attempted, "
          f"{generation_failures} failures")

    # Phase 2: Judge all conjectures
    print("\n===== PHASE 2: JUDGING =====\n")
    for i, entry in enumerate(all_results):
        entry = judge_conjecture(entry)
        all_results[i] = entry
        if entry.get("judge_result") is None and entry.get("generation_success"):
            judge_failures += 1
        time.sleep(DELAY_SECONDS)

    print(f"\n  Judging complete: {judge_failures} judge failures")

    # Phase 3: Analysis
    print("\n===== PHASE 3: ANALYSIS =====\n")
    analysis = analyze_results(all_results, conditions)

    # Build final output
    output = {
        "experiment": "context_investigation",
        "description": "Investigating no-context anomaly: does providing domain context hurt conjecture quality?",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "generator": f"Mistral {GENERATION_MODEL} (temp=0.7)",
            "judge": f"Mistral {JUDGE_MODEL} (temp=0.3)",
            "domains": DOMAINS,
            "conditions": conditions,
            "conjectures_per_cell": CONJECTURES_PER_CELL,
            "total_attempted": len(all_results),
            "generation_failures": generation_failures,
            "judge_failures": judge_failures,
            "good_context_theorems": GOOD_CONTEXT,
            "random_context_map": RANDOM_CONTEXT_MAP,
        },
        "analysis": analysis,
        "conjectures": [
            {
                "domain": r["domain"],
                "condition": r["condition"],
                "trial": r["trial"],
                "conjecture_text": r.get("conjecture_text", ""),
                "parsed": r.get("parsed"),
                "overall_score": r.get("overall_score", 0.0),
                "scores": r.get("scores", {}),
                "critique": r.get("critique", ""),
                "generation_success": r.get("generation_success", False),
            }
            for r in all_results
        ],
    }

    return output


def analyze_results(all_results: List[Dict], conditions: List[str]) -> Dict:
    """Compute per-condition averages, per-domain breakdowns, and t-tests."""
    from scipy import stats as scipy_stats

    # Group scores by condition
    condition_scores = defaultdict(list)
    domain_condition_scores = defaultdict(lambda: defaultdict(list))
    criteria_by_condition = defaultdict(lambda: defaultdict(list))

    for r in all_results:
        if not r.get("generation_success"):
            continue
        cond = r["condition"]
        domain = r["domain"]
        score = r.get("overall_score", 0.0)
        condition_scores[cond].append(score)
        domain_condition_scores[domain][cond].append(score)

        # Per-criteria
        for criterion, val in r.get("scores", {}).items():
            criteria_by_condition[cond][criterion].append(val)

    # Per-condition summary
    condition_summary = {}
    for cond in conditions:
        scores = condition_scores[cond]
        if scores:
            mean = sum(scores) / len(scores)
            condition_summary[cond] = {
                "n": len(scores),
                "mean": round(mean, 4),
                "std": round((sum((s - mean)**2 for s in scores) / len(scores)) ** 0.5, 4),
                "min": round(min(scores), 4),
                "max": round(max(scores), 4),
                "scores": [round(s, 4) for s in scores],
            }
        else:
            condition_summary[cond] = {"n": 0, "mean": 0, "std": 0, "scores": []}

    # Per-criteria breakdown by condition
    criteria_summary = {}
    for cond in conditions:
        criteria_summary[cond] = {}
        for criterion, vals in criteria_by_condition[cond].items():
            if vals:
                criteria_summary[cond][criterion] = round(sum(vals) / len(vals), 4)

    # Per-domain breakdown
    domain_summary = {}
    for domain in DOMAINS:
        domain_summary[domain] = {}
        for cond in conditions:
            scores = domain_condition_scores[domain][cond]
            if scores:
                domain_summary[domain][cond] = {
                    "n": len(scores),
                    "mean": round(sum(scores) / len(scores), 4),
                    "scores": [round(s, 4) for s in scores],
                }
            else:
                domain_summary[domain][cond] = {"n": 0, "mean": 0, "scores": []}

    # Statistical tests (t-tests between conditions)
    statistical_tests = {}

    # Pairwise t-tests (Welch's t-test, unequal variance)
    pairs = [
        ("no_context", "good_context"),
        ("no_context", "random_context"),
        ("good_context", "random_context"),
    ]
    for cond_a, cond_b in pairs:
        scores_a = condition_scores[cond_a]
        scores_b = condition_scores[cond_b]
        if len(scores_a) >= 2 and len(scores_b) >= 2:
            t_stat, p_value = scipy_stats.ttest_ind(scores_a, scores_b, equal_var=False)
            statistical_tests[f"{cond_a}_vs_{cond_b}"] = {
                "t_statistic": round(float(t_stat), 4),
                "p_value": round(float(p_value), 4),
                "significant_at_005": bool(p_value < 0.05),
                "significant_at_010": bool(p_value < 0.10),
                "mean_a": round(sum(scores_a) / len(scores_a), 4),
                "mean_b": round(sum(scores_b) / len(scores_b), 4),
                "n_a": len(scores_a),
                "n_b": len(scores_b),
                "effect_direction": f"{cond_a} > {cond_b}" if sum(scores_a)/len(scores_a) > sum(scores_b)/len(scores_b) else f"{cond_b} > {cond_a}",
            }

    # One-way ANOVA across all three conditions
    all_groups = [condition_scores[c] for c in conditions if len(condition_scores[c]) >= 2]
    if len(all_groups) == 3:
        f_stat, anova_p = scipy_stats.f_oneway(*all_groups)
        statistical_tests["anova_3way"] = {
            "f_statistic": round(float(f_stat), 4),
            "p_value": round(float(anova_p), 4),
            "significant_at_005": bool(anova_p < 0.05),
        }

    # Print summary
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n{'Condition':<20} {'N':>4} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 60)
    for cond in conditions:
        s = condition_summary[cond]
        print(f"{cond:<20} {s['n']:>4} {s['mean']:>8.4f} {s['std']:>8.4f} "
              f"{s.get('min', 0):>8.4f} {s.get('max', 0):>8.4f}")

    print(f"\nPer-criteria means by condition:")
    print(f"{'Criterion':<20}", end="")
    for cond in conditions:
        print(f" {cond:>14}", end="")
    print()
    print("-" * 62)
    all_criteria = ["correctness", "novelty", "non_triviality", "significance",
                    "formalizability", "proof_quality"]
    for criterion in all_criteria:
        print(f"{criterion:<20}", end="")
        for cond in conditions:
            val = criteria_summary.get(cond, {}).get(criterion, 0)
            print(f" {val:>14.4f}", end="")
        print()

    print(f"\nPer-domain means:")
    print(f"{'Domain':<25}", end="")
    for cond in conditions:
        print(f" {cond:>14}", end="")
    print()
    print("-" * 67)
    for domain in DOMAINS:
        print(f"{domain:<25}", end="")
        for cond in conditions:
            val = domain_summary[domain][cond].get("mean", 0)
            print(f" {val:>14.4f}", end="")
        print()

    print(f"\nStatistical Tests:")
    print("-" * 60)
    for test_name, result in statistical_tests.items():
        if "t_statistic" in result:
            sig = "***" if result["p_value"] < 0.01 else ("**" if result["p_value"] < 0.05 else ("*" if result["p_value"] < 0.10 else "ns"))
            print(f"  {test_name}: t={result['t_statistic']:.3f}, p={result['p_value']:.4f} {sig}")
            print(f"    {result['effect_direction']}")
        elif "f_statistic" in result:
            sig = "***" if result["p_value"] < 0.01 else ("**" if result["p_value"] < 0.05 else ("*" if result["p_value"] < 0.10 else "ns"))
            print(f"  {test_name}: F={result['f_statistic']:.3f}, p={result['p_value']:.4f} {sig}")

    # Interpretation
    no_ctx_mean = condition_summary["no_context"]["mean"]
    good_ctx_mean = condition_summary["good_context"]["mean"]
    random_ctx_mean = condition_summary["random_context"]["mean"]

    print(f"\n{'=' * 60}")
    print("INTERPRETATION")
    print(f"{'=' * 60}")
    if no_ctx_mean > good_ctx_mean:
        print(f"  ANOMALY CONFIRMED: no_context ({no_ctx_mean:.4f}) > good_context ({good_ctx_mean:.4f})")
        print(f"  Delta: {no_ctx_mean - good_ctx_mean:+.4f}")
    else:
        print(f"  ANOMALY NOT CONFIRMED: good_context ({good_ctx_mean:.4f}) >= no_context ({no_ctx_mean:.4f})")
        print(f"  Delta: {good_ctx_mean - no_ctx_mean:+.4f}")

    if random_ctx_mean > good_ctx_mean:
        print(f"  Cross-domain effect: random_context ({random_ctx_mean:.4f}) > good_context ({good_ctx_mean:.4f})")
    else:
        print(f"  No cross-domain effect: good_context ({good_ctx_mean:.4f}) >= random_context ({random_ctx_mean:.4f})")

    nv_gc = statistical_tests.get("no_context_vs_good_context", {})
    if nv_gc.get("significant_at_005"):
        print(f"  The difference between no_context and good_context IS statistically significant (p={nv_gc['p_value']:.4f})")
    elif nv_gc.get("significant_at_010"):
        print(f"  The difference is marginally significant (p={nv_gc['p_value']:.4f})")
    else:
        print(f"  The difference is NOT statistically significant (p={nv_gc.get('p_value', 'N/A')})")

    return {
        "condition_summary": condition_summary,
        "criteria_by_condition": criteria_summary,
        "domain_summary": domain_summary,
        "statistical_tests": statistical_tests,
        "anomaly_confirmed": no_ctx_mean > good_ctx_mean,
        "interpretation": {
            "no_context_mean": no_ctx_mean,
            "good_context_mean": good_ctx_mean,
            "random_context_mean": random_ctx_mean,
            "no_vs_good_delta": round(no_ctx_mean - good_ctx_mean, 4),
            "random_vs_good_delta": round(random_ctx_mean - good_ctx_mean, 4),
        },
    }


def main():
    start_time = time.time()
    output = run_experiment()

    elapsed = time.time() - start_time
    output["elapsed_seconds"] = round(elapsed, 1)

    # Save results
    output_path = os.path.join(RESULTS_DIR, "task17_context_investigation.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n\nResults saved to: {output_path}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Total conjectures: {len(output['conjectures'])}")


if __name__ == "__main__":
    main()
