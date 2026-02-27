#!/usr/bin/env python3
"""
MathScy MoE Conjecture Evaluation - Lite (Heuristic-Only)

Evaluates MoE-generated conjectures using purely heuristic quality metrics.
No API calls, no GPU required. Designed for honest assessment of base-model
MoE conjectures, which are expected to be lower quality than API-generated ones
since the base model (deepseek-math-7b) is not instruction-tuned.

Scoring dimensions:
  1. Statement quality: length, math symbols, sentence completeness
  2. Specificity: presence of concrete mathematical objects/concepts
  3. Formalizability: precise conditions/conclusions vs hand-wavy text
  4. Penalties: LaTeX artifacts, truncation, questions, meta-commentary

Usage:
    python scripts/evaluate_moe_conjectures_lite.py
"""

import os
import sys
import json
import re
import math
from collections import defaultdict
from datetime import datetime

# Paths
PROJECT_DIR = "/scratch/ctoxtli/moexp"
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
MOE_CONJECTURES_PATH = os.path.join(RESULTS_DIR, "moe_generated_conjectures.jsonl")
API_REPORT_PATH = os.path.join(RESULTS_DIR, "conjecture_evaluation_report.json")
MOE_REPORT_PATH = os.path.join(RESULTS_DIR, "moe_conjecture_evaluation_report.json")
COMPARISON_PATH = os.path.join(RESULTS_DIR, "moe_vs_api_comparison.json")


# ---------------------------------------------------------------------------
# Mathematical keyword / concept dictionaries
# ---------------------------------------------------------------------------

MATH_SYMBOLS = ['$', '\\', '\\frac', '\\sum', '\\int', '\\lim', '\\prod',
                '\\forall', '\\exists', '\\in', '\\subset', '\\geq', '\\leq',
                '\\to', '\\rightarrow', '\\mapsto', '\\cong', '\\simeq',
                '\\otimes', '\\oplus', '\\wedge', '\\cap', '\\cup',
                '\\mathbb', '\\mathcal', '\\mathfrak']

SPECIFIC_MATH_OBJECTS = [
    # Algebraic structures
    'ring', 'field', 'group', 'module', 'algebra', 'ideal', 'subgroup',
    'lattice', 'monoid', 'semigroup', 'vector space', 'Lie algebra',
    # Geometry / topology
    'manifold', 'variety', 'scheme', 'bundle', 'sheaf', 'curve', 'surface',
    'hypersurface', 'singularity', 'divisor', 'cohomology', 'homology',
    'homotopy', 'Lagrangian', 'symplectic', 'Riemannian',
    # Number theory
    'prime', 'integer', 'rational', 'algebraic number', 'Galois',
    'modular form', 'L-function', 'zeta function', 'discriminant',
    # Analysis
    'Banach', 'Hilbert', 'operator', 'eigenvalue', 'convergence',
    'integral', 'measure', 'norm', 'compact', 'bounded', 'continuous',
    # Combinatorics / discrete
    'graph', 'tree', 'partition', 'permutation', 'chromatic', 'matroid',
    'hypergraph', 'vertex', 'edge', 'degree', 'path',
    # Probability / statistics
    'random variable', 'distribution', 'expectation', 'variance',
    'probability', 'stochastic', 'martingale', 'Markov',
    'covariance', 'estimator',
    # Formal / precise language
    'isomorphism', 'homeomorphism', 'diffeomorphism', 'embedding',
    'functor', 'category', 'morphism', 'exact sequence',
    'dimension', 'rank', 'degree', 'genus',
]

FORMAL_INDICATORS = [
    # Quantifiers & logical structure
    'for all', 'for any', 'for every', 'there exists', 'there exist',
    'if and only if', 'iff', 'implies', 'then', 'suppose', 'assume',
    'let ', 'given ', 'such that', 'satisfying', 'where ',
    # Conclusion words
    'is isomorphic', 'is equivalent', 'is equal', 'divides',
    'converges', 'is bounded', 'is continuous', 'is compact',
    'is finite', 'is infinite', 'has dimension', 'admits',
]

# Patterns indicating low-quality / artifact text
LATEX_ARTIFACT_PATTERNS = [
    r'\\ref\{',           # unresolved references
    r'\\cite\{',          # unresolved citations
    r'\\label\{',         # label commands
    r'\\begin\{',         # raw environment starts
    r'\\end\{',           # raw environment ends
    r'\\section',         # section headers
    r'\\subsection',      # subsection headers
    r'%%+',               # LaTeX comments
    r'\\qed',             # proof termination
    r'\\square',          # proof box
    r'\\endproof',        # endproof
    r'\\endgroup',        # TeX grouping
    r'\\backspace',       # TeX command
    r'Context before:',   # prompt leakage
    r'Domain before',     # prompt leakage
    r'See Theorem',       # reference to elsewhere
    r'See Lemma',         # reference to elsewhere
    r'See Remark',        # reference to elsewhere
    r'See Corollary',     # reference to elsewhere
    r'as follows\.\.',    # trailing ellipsis
]

TRUNCATION_INDICATORS = [
    r'\.\.\.$',           # ends with ...
    r'[a-z]$',            # ends mid-word (lowercase, no punctuation)
    r' o$',               # cut off "or", "of", etc.
    r' t$',               # cut off "the", "that", etc.
    r' a$',               # cut off article
    r' b$',               # cut off "by", "but", etc.
    r'sucht ',            # misspelling from truncation
]

META_COMMENTARY_PATTERNS = [
    r"I'm not sure",
    r"not sure if",
    r"you could",
    r"could formulate",
    r"could involve",
    r"could explore",
    r"could potentially",
    r"This could be",
    r"This is some text",
    r"This is an example",
    r"might look like",
    r"post-processing",
    r"We present it here",
    r"other examples can be generated",
    r"replacing these terms",
    r"Generalize this to",
    r"Investigate whether",
    r"Investigate bounds",
    r"Consider weakening",
    r"Consider strengthening",
    r"making explicit reference",
]

QUESTION_PATTERNS = [
    r'\?$',               # ends with question mark
    r'\?\s*$',
    r'^Analogue of .* for .*\?',
    r'Are there any',
    r'What should replace',
]


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def score_statement_quality(text: str) -> float:
    """Score basic statement quality: length, math content, completeness."""
    score = 0.0
    length = len(text)

    # Length scoring (0 to 0.3)
    if length < 15:
        score += 0.0
    elif length < 30:
        score += 0.05
    elif length < 60:
        score += 0.15
    elif length < 150:
        score += 0.25
    elif length < 300:
        score += 0.30
    elif length < 500:
        score += 0.25
    else:
        # Very long - likely has artifacts
        score += 0.10

    # Contains math symbols (0 to 0.4)
    math_count = 0
    for sym in MATH_SYMBOLS:
        if sym in text:
            math_count += 1
    if math_count == 0:
        score += 0.0
    elif math_count <= 2:
        score += 0.15
    elif math_count <= 5:
        score += 0.30
    else:
        score += 0.40

    # Sentence completeness (0 to 0.3)
    # A good conjecture should start with a capital letter and end with
    # punctuation (period, colon, or closing math delimiter)
    starts_well = bool(re.match(r'^[A-Z$\\]', text.strip()))
    ends_well = bool(re.search(r'[.;:)$}]\.?\s*$', text.strip()))
    if starts_well and ends_well:
        score += 0.30
    elif starts_well or ends_well:
        score += 0.15
    else:
        score += 0.05

    return min(score, 1.0)


def score_specificity(text: str) -> float:
    """Score specificity: presence of concrete mathematical objects."""
    text_lower = text.lower()
    score = 0.0

    # Count specific mathematical objects mentioned
    obj_count = 0
    for obj in SPECIFIC_MATH_OBJECTS:
        if obj.lower() in text_lower:
            obj_count += 1

    if obj_count == 0:
        score = 0.05
    elif obj_count == 1:
        score = 0.25
    elif obj_count <= 3:
        score = 0.55
    elif obj_count <= 6:
        score = 0.75
    else:
        score = 0.90

    # Bonus for inline math with specific variables/notation
    inline_math = re.findall(r'\$[^$]+\$', text)
    if len(inline_math) >= 3:
        score = min(score + 0.10, 1.0)
    elif len(inline_math) >= 1:
        score = min(score + 0.05, 1.0)

    return score


def score_formalizability(text: str) -> float:
    """Score how formalizable the statement is: precise conditions/conclusions."""
    text_lower = text.lower()
    score = 0.0

    # Count formal indicators
    formal_count = 0
    for ind in FORMAL_INDICATORS:
        if ind.lower() in text_lower:
            formal_count += 1

    if formal_count == 0:
        score = 0.05
    elif formal_count == 1:
        score = 0.25
    elif formal_count <= 3:
        score = 0.55
    elif formal_count <= 5:
        score = 0.75
    else:
        score = 0.90

    # Check for explicit hypothesis-conclusion structure
    has_hypothesis = any(w in text_lower for w in ['let ', 'suppose ', 'assume ', 'given ', 'if '])
    has_conclusion = any(w in text_lower for w in ['then ', 'implies', 'is isomorphic',
                                                    'is equivalent', 'converges',
                                                    'there exists', 'we have',
                                                    'it follows', 'holds'])
    if has_hypothesis and has_conclusion:
        score = min(score + 0.15, 1.0)
    elif has_hypothesis or has_conclusion:
        score = min(score + 0.05, 1.0)

    return score


def compute_penalties(text: str) -> float:
    """Compute penalty score (0 to 1, where 1 = maximum penalty)."""
    penalties = 0.0
    text_stripped = text.strip()

    # LaTeX artifacts
    artifact_count = 0
    for pattern in LATEX_ARTIFACT_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            artifact_count += 1
    if artifact_count >= 4:
        penalties += 0.50
    elif artifact_count >= 2:
        penalties += 0.30
    elif artifact_count >= 1:
        penalties += 0.15

    # Truncation
    truncated = False
    for pattern in TRUNCATION_INDICATORS:
        if re.search(pattern, text_stripped):
            truncated = True
            break
    if truncated:
        penalties += 0.15

    # Meta-commentary / non-mathematical text
    meta_count = 0
    for pattern in META_COMMENTARY_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            meta_count += 1
    if meta_count >= 3:
        penalties += 0.40
    elif meta_count >= 2:
        penalties += 0.25
    elif meta_count >= 1:
        penalties += 0.15

    # Question-like text (conjectures should be statements, not questions)
    for pattern in QUESTION_PATTERNS:
        if re.search(pattern, text_stripped, re.IGNORECASE):
            penalties += 0.15
            break

    # Very short / content-free
    if len(text_stripped) < 20:
        penalties += 0.30
    elif len(text_stripped) < 40:
        penalties += 0.10

    # Starts with "Context before:" - raw prompt leakage
    if text_stripped.startswith("Context before:") or text_stripped.startswith("Domain before"):
        penalties += 0.40

    # Ends with raw LaTeX closing (}], %%, etc.)
    if re.search(r'[%}\\]\s*$', text_stripped) and not text_stripped.endswith('$'):
        penalties += 0.10

    # Contains "Combination of Result" or similar trivial text
    trivial_phrases = [
        'Combination of Result',
        'Conjecture for combined result',
        'combined result',
        'This is some text before',
    ]
    for phrase in trivial_phrases:
        if phrase.lower() in text.lower():
            penalties += 0.25
            break

    return min(penalties, 1.0)


def score_conjecture(entry: dict) -> dict:
    """Score a single conjecture and return detailed scores."""
    text = entry.get("conjecture_statement", "")

    sq = score_statement_quality(text)
    sp = score_specificity(text)
    fm = score_formalizability(text)
    pen = compute_penalties(text)

    # Weighted combination
    # Weights: statement_quality=0.25, specificity=0.30, formalizability=0.30, penalty=0.15
    raw_score = 0.25 * sq + 0.30 * sp + 0.30 * fm
    # Apply penalty as a multiplicative reduction plus direct subtraction
    final_score = max(0.0, raw_score * (1.0 - 0.5 * pen) - 0.15 * pen)
    final_score = round(min(final_score, 1.0), 4)

    return {
        "conjecture_statement": text[:200] + ("..." if len(text) > 200 else ""),
        "domain": entry.get("domain", "unknown"),
        "strategy": entry.get("strategy", "unknown"),
        "quality_score": final_score,
        "breakdown": {
            "statement_quality": round(sq, 4),
            "specificity": round(sp, 4),
            "formalizability": round(fm, 4),
            "penalty": round(pen, 4),
        }
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def load_conjectures(path: str) -> list:
    """Load conjectures from JSONL file."""
    conjectures = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                conjectures.append(json.loads(line))
    return conjectures


def build_moe_report(scored: list) -> dict:
    """Build the MoE evaluation report."""
    total = len(scored)
    scores = [s["quality_score"] for s in scored]
    avg_score = sum(scores) / total if total > 0 else 0.0

    # Per-domain
    per_domain = defaultdict(list)
    for s in scored:
        per_domain[s["domain"]].append(s["quality_score"])

    per_domain_report = {}
    for domain, domain_scores in sorted(per_domain.items()):
        per_domain_report[domain] = {
            "count": len(domain_scores),
            "avg_quality": round(sum(domain_scores) / len(domain_scores), 4),
            "max_quality": round(max(domain_scores), 4),
        }

    # Per-strategy
    per_strategy = defaultdict(list)
    for s in scored:
        per_strategy[s["strategy"]].append(s["quality_score"])

    strategy_report = {}
    for strat, strat_scores in sorted(per_strategy.items()):
        strategy_report[strat] = {
            "count": len(strat_scores),
            "avg_quality": round(sum(strat_scores) / len(strat_scores), 4),
        }

    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scoring_method": "heuristic_only (no API calls, no GPU)",
        "summary": {
            "total_conjectures": total,
            "avg_quality_score": round(avg_score, 4),
            "min_quality_score": round(min(scores), 4) if scores else 0.0,
            "max_quality_score": round(max(scores), 4) if scores else 0.0,
            "median_quality_score": round(sorted(scores)[total // 2], 4) if scores else 0.0,
        },
        "per_domain": per_domain_report,
        "strategy_comparison": strategy_report,
        "score_distribution": {
            "0.0-0.1": sum(1 for s in scores if s < 0.1),
            "0.1-0.2": sum(1 for s in scores if 0.1 <= s < 0.2),
            "0.2-0.3": sum(1 for s in scores if 0.2 <= s < 0.3),
            "0.3-0.4": sum(1 for s in scores if 0.3 <= s < 0.4),
            "0.4-0.5": sum(1 for s in scores if 0.4 <= s < 0.5),
            "0.5-0.6": sum(1 for s in scores if 0.5 <= s < 0.6),
            "0.6-0.7": sum(1 for s in scores if 0.6 <= s < 0.7),
            "0.7-0.8": sum(1 for s in scores if 0.7 <= s < 0.8),
            "0.8-0.9": sum(1 for s in scores if 0.8 <= s < 0.9),
            "0.9-1.0": sum(1 for s in scores if 0.9 <= s <= 1.0),
        },
        "scored_conjectures": scored,
    }

    return report


def build_comparison(moe_report: dict, api_report: dict) -> dict:
    """Build comparison between MoE and API results."""
    moe_summary = moe_report["summary"]
    api_summary = api_report["summary"]

    # Per-domain comparison
    domain_comparison = {}
    all_domains = set(list(moe_report["per_domain"].keys()) +
                      list(api_report["per_domain"].keys()))
    for domain in sorted(all_domains):
        moe_d = moe_report["per_domain"].get(domain, {})
        api_d = api_report["per_domain"].get(domain, {})
        moe_avg = moe_d.get("avg_quality", 0.0)
        api_avg = api_d.get("avg_quality", 0.0)
        gap = round(api_avg - moe_avg, 4) if moe_avg > 0 and api_avg > 0 else None
        domain_comparison[domain] = {
            "moe_count": moe_d.get("count", 0),
            "api_count": api_d.get("count", 0),
            "moe_avg_quality": round(moe_avg, 4),
            "api_avg_quality": round(api_avg, 4),
            "quality_gap": gap,
        }

    # Strategy comparison
    strategy_comparison = {}
    all_strategies = set(list(moe_report["strategy_comparison"].keys()) +
                         list(api_report["strategy_comparison"].keys()))
    for strat in sorted(all_strategies):
        moe_s = moe_report["strategy_comparison"].get(strat, {})
        api_s = api_report["strategy_comparison"].get(strat, {})
        moe_avg = moe_s.get("avg_quality", 0.0)
        api_avg = api_s.get("avg_quality", 0.0)
        gap = round(api_avg - moe_avg, 4) if moe_avg > 0 and api_avg > 0 else None
        strategy_comparison[strat] = {
            "moe_count": moe_s.get("count", 0),
            "api_count": api_s.get("count", 0),
            "moe_avg_quality": round(moe_avg, 4),
            "api_avg_quality": round(api_avg, 4),
            "quality_gap": gap,
        }

    moe_avg_total = moe_summary["avg_quality_score"]
    api_avg_total = api_summary["avg_quality_score"]
    overall_gap = round(api_avg_total - moe_avg_total, 4)

    comparison = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": (
            "Comparison of MoE-generated conjectures (base model, heuristic scoring) "
            "vs API-generated conjectures (instruction-tuned models, API-judge scoring). "
            "MoE conjectures are expected to be lower quality because the base model "
            "(deepseek-math-7b-base) is not instruction-tuned."
        ),
        "overall": {
            "moe_total_conjectures": moe_summary["total_conjectures"],
            "api_total_conjectures": api_summary["total_conjectures"],
            "moe_avg_quality": round(moe_avg_total, 4),
            "api_avg_quality": round(api_avg_total, 4),
            "quality_gap_api_minus_moe": overall_gap,
            "moe_scoring_method": "heuristic (no API)",
            "api_scoring_method": "LLM judge (Mistral/Groq)",
        },
        "per_domain": domain_comparison,
        "per_strategy": strategy_comparison,
        "analysis": {
            "key_findings": [],
        },
    }

    # Generate key findings
    findings = comparison["analysis"]["key_findings"]

    findings.append(
        f"Overall quality gap: API conjectures score {overall_gap:.4f} higher "
        f"than MoE conjectures ({api_avg_total:.4f} vs {moe_avg_total:.4f})."
    )

    # Find best/worst MoE domains
    moe_domains = moe_report["per_domain"]
    if moe_domains:
        best_domain = max(moe_domains.items(), key=lambda x: x[1]["avg_quality"])
        worst_domain = min(moe_domains.items(), key=lambda x: x[1]["avg_quality"])
        findings.append(
            f"Best MoE domain: {best_domain[0]} (avg {best_domain[1]['avg_quality']:.4f}), "
            f"worst: {worst_domain[0]} (avg {worst_domain[1]['avg_quality']:.4f})."
        )

    # Find best/worst MoE strategies
    moe_strats = moe_report["strategy_comparison"]
    if moe_strats:
        best_strat = max(moe_strats.items(), key=lambda x: x[1]["avg_quality"])
        worst_strat = min(moe_strats.items(), key=lambda x: x[1]["avg_quality"])
        findings.append(
            f"Best MoE strategy: {best_strat[0]} (avg {best_strat[1]['avg_quality']:.4f}), "
            f"worst: {worst_strat[0]} (avg {worst_strat[1]['avg_quality']:.4f})."
        )

    # Smallest domain gap
    gaps = [(d, v["quality_gap"]) for d, v in domain_comparison.items() if v["quality_gap"] is not None]
    if gaps:
        smallest_gap = min(gaps, key=lambda x: x[1])
        largest_gap = max(gaps, key=lambda x: x[1])
        findings.append(
            f"Smallest domain quality gap: {smallest_gap[0]} ({smallest_gap[1]:.4f}), "
            f"largest: {largest_gap[0]} ({largest_gap[1]:.4f})."
        )

    findings.append(
        "Note: MoE uses heuristic scoring (pattern-based) while API uses LLM-judge scoring. "
        "These scoring methods are not directly comparable but provide useful relative signals."
    )

    findings.append(
        "The MoE base model (deepseek-math-7b-base) generates text continuations rather than "
        "structured conjectures, leading to frequent LaTeX artifacts, truncations, and meta-commentary."
    )

    return comparison


def print_comparison_table(comparison: dict):
    """Print a formatted comparison table to stdout."""
    overall = comparison["overall"]

    print("=" * 80)
    print("  MoE vs API Conjecture Quality Comparison")
    print("=" * 80)
    print()
    print(f"  MoE scoring method:  {overall['moe_scoring_method']}")
    print(f"  API scoring method:  {overall['api_scoring_method']}")
    print()

    # Overall summary
    print("-" * 80)
    print(f"  {'Metric':<35} {'MoE':>12} {'API':>12} {'Gap':>12}")
    print("-" * 80)
    print(f"  {'Total conjectures':<35} {overall['moe_total_conjectures']:>12} "
          f"{overall['api_total_conjectures']:>12} {'':>12}")
    print(f"  {'Avg quality score':<35} {overall['moe_avg_quality']:>12.4f} "
          f"{overall['api_avg_quality']:>12.4f} {overall['quality_gap_api_minus_moe']:>+12.4f}")
    print()

    # Per-domain
    print("-" * 80)
    print(f"  {'Domain':<25} {'MoE Avg':>10} {'API Avg':>10} {'Gap':>10} {'MoE N':>7} {'API N':>7}")
    print("-" * 80)
    for domain, vals in sorted(comparison["per_domain"].items()):
        gap_str = f"{vals['quality_gap']:+.4f}" if vals["quality_gap"] is not None else "N/A"
        print(f"  {domain:<25} {vals['moe_avg_quality']:>10.4f} {vals['api_avg_quality']:>10.4f} "
              f"{gap_str:>10} {vals['moe_count']:>7} {vals['api_count']:>7}")
    print()

    # Per-strategy
    print("-" * 80)
    print(f"  {'Strategy':<25} {'MoE Avg':>10} {'API Avg':>10} {'Gap':>10} {'MoE N':>7} {'API N':>7}")
    print("-" * 80)
    for strat, vals in sorted(comparison["per_strategy"].items()):
        gap_str = f"{vals['quality_gap']:+.4f}" if vals["quality_gap"] is not None else "N/A"
        print(f"  {strat:<25} {vals['moe_avg_quality']:>10.4f} {vals['api_avg_quality']:>10.4f} "
              f"{gap_str:>10} {vals['moe_count']:>7} {vals['api_count']:>7}")
    print()

    # Key findings
    print("-" * 80)
    print("  Key Findings:")
    print("-" * 80)
    for i, finding in enumerate(comparison["analysis"]["key_findings"], 1):
        # Wrap long findings
        lines = []
        words = finding.split()
        current = ""
        for word in words:
            if len(current) + len(word) + 1 > 72:
                lines.append(current)
                current = word
            else:
                current = current + " " + word if current else word
        if current:
            lines.append(current)
        print(f"  {i}. {lines[0]}")
        for line in lines[1:]:
            print(f"     {line}")
    print()
    print("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  MathScy MoE Conjecture Evaluation (Heuristic-Only, Lite)")
    print("=" * 70)
    print()

    # Step 1: Load MoE conjectures
    print(f"[1/6] Loading MoE conjectures from {MOE_CONJECTURES_PATH}...")
    if not os.path.exists(MOE_CONJECTURES_PATH):
        print(f"  ERROR: File not found: {MOE_CONJECTURES_PATH}")
        sys.exit(1)
    conjectures = load_conjectures(MOE_CONJECTURES_PATH)
    print(f"  Loaded {len(conjectures)} conjectures")
    print()

    # Step 2: Score each conjecture
    print("[2/6] Scoring conjectures with heuristic metrics...")
    scored = []
    for entry in conjectures:
        scored.append(score_conjecture(entry))

    scores = [s["quality_score"] for s in scored]
    print(f"  Scored {len(scored)} conjectures")
    print(f"  Score range: [{min(scores):.4f}, {max(scores):.4f}]")
    print(f"  Mean: {sum(scores)/len(scores):.4f}, "
          f"Median: {sorted(scores)[len(scores)//2]:.4f}")
    print()

    # Step 3: Print per-domain summary
    print("[3/6] Per-domain summary:")
    per_domain = defaultdict(list)
    for s in scored:
        per_domain[s["domain"]].append(s["quality_score"])
    print(f"  {'Domain':<25} {'Count':>6} {'Avg':>8} {'Max':>8}")
    print(f"  {'-'*25} {'-'*6} {'-'*8} {'-'*8}")
    for domain in sorted(per_domain.keys()):
        ds = per_domain[domain]
        print(f"  {domain:<25} {len(ds):>6} {sum(ds)/len(ds):>8.4f} {max(ds):>8.4f}")
    print()

    # Step 4: Print per-strategy summary
    print("[4/6] Per-strategy summary:")
    per_strategy = defaultdict(list)
    for s in scored:
        per_strategy[s["strategy"]].append(s["quality_score"])
    print(f"  {'Strategy':<25} {'Count':>6} {'Avg':>8} {'Max':>8}")
    print(f"  {'-'*25} {'-'*6} {'-'*8} {'-'*8}")
    for strat in sorted(per_strategy.keys()):
        ss = per_strategy[strat]
        print(f"  {strat:<25} {len(ss):>6} {sum(ss)/len(ss):>8.4f} {max(ss):>8.4f}")
    print()

    # Step 5: Build and save MoE report
    print(f"[5/6] Saving MoE evaluation report to {MOE_REPORT_PATH}...")
    moe_report = build_moe_report(scored)
    with open(MOE_REPORT_PATH, 'w') as f:
        json.dump(moe_report, f, indent=2)
    print(f"  Saved.")
    print()

    # Step 6: Load API report and build comparison
    print(f"[6/6] Building comparison with API results...")
    if not os.path.exists(API_REPORT_PATH):
        print(f"  WARNING: API report not found at {API_REPORT_PATH}")
        print(f"  Skipping comparison.")
        return

    with open(API_REPORT_PATH, 'r') as f:
        api_report = json.load(f)

    comparison = build_comparison(moe_report, api_report)
    with open(COMPARISON_PATH, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"  Saved comparison to {COMPARISON_PATH}")
    print()

    # Print comparison table
    print_comparison_table(comparison)

    print(f"\nOutput files:")
    print(f"  MoE report:   {MOE_REPORT_PATH}")
    print(f"  Comparison:   {COMPARISON_PATH}")
    print()


if __name__ == "__main__":
    main()
