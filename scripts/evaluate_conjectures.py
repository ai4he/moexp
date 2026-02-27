"""
MathScy Conjecture & Theorem Generation Evaluation

Evaluates the MoE system's ability to generate novel mathematical conjectures
and theorems using multiple strategies and the Self-Play Theorem Prover (STP) loop.

Supports Gemini, Groq Cloud, and Mistral APIs for generation and evaluation.

Usage:
    python scripts/evaluate_conjectures.py --mode full
    python scripts/evaluate_conjectures.py --mode generate --domains algebraic_geometry,algebra
    python scripts/evaluate_conjectures.py --mode stp --domains algebraic_geometry --rounds 3
    python scripts/evaluate_conjectures.py --mode evaluate  # Score existing results
"""

import os
import sys
import json
import time
import random
import argparse
import traceback
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Setup paths
PROJECT_DIR = "/scratch/ctoxtli/moexp"
sys.path.insert(0, os.path.join(PROJECT_DIR, "scripts"))

from llm_utils import gemini_generate
from data_utils import get_domain_group, MATH_DOMAIN_GROUPS

# ===== API Configuration =====

def load_api_keys():
    """Load all available API keys."""
    keys = {}

    # Gemini (key 4 / index 3 works)
    with open(os.path.join(PROJECT_DIR, "working_Gemini_API_keys.json")) as f:
        gemini_keys = json.load(f)
    keys["gemini"] = gemini_keys[3]

    # Groq Cloud
    groq_path = os.path.join(PROJECT_DIR, "groqcloud_key.txt")
    if os.path.exists(groq_path):
        with open(groq_path) as f:
            keys["groq"] = f.read().strip()

    # Mistral
    mistral_path = os.path.join(PROJECT_DIR, "mistral_key.txt")
    if os.path.exists(mistral_path):
        with open(mistral_path) as f:
            keys["mistral"] = f.read().strip()

    # OpenRouter
    openrouter_path = os.path.join(PROJECT_DIR, "openrouter_key.txt")
    if os.path.exists(openrouter_path):
        with open(openrouter_path) as f:
            keys["openrouter"] = f.read().strip()

    return keys


def groq_generate(prompt: str, api_key: str, model: str = "llama-3.3-70b-versatile",
                  temperature: float = 0.7, max_tokens: int = 4096, max_retries: int = 4) -> str:
    """Call Groq Cloud API."""
    import requests
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            elif resp.status_code == 429:
                wait = min(2 ** attempt * 5, 60)
                print(f"  Groq rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Groq error {resp.status_code}: {resp.text[:200]}")
                time.sleep(2 ** attempt)
        except Exception as e:
            print(f"  Groq request error: {e}")
            time.sleep(2 ** attempt)
    return ""


def mistral_generate(prompt: str, api_key: str, model: str = "mistral-small-latest",
                     temperature: float = 0.7, max_tokens: int = 4096, max_retries: int = 4) -> str:
    """Call Mistral API."""
    import requests
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            elif resp.status_code == 429:
                wait = min(2 ** attempt * 5, 60)
                print(f"  Mistral rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Mistral error {resp.status_code}: {resp.text[:200]}")
                time.sleep(2 ** attempt)
        except Exception as e:
            print(f"  Mistral request error: {e}")
            time.sleep(2 ** attempt)
    return ""


def openrouter_generate(prompt: str, api_key: str, model: str = "google/gemini-2.0-flash-001",
                        temperature: float = 0.7, max_tokens: int = 4096, max_retries: int = 4) -> str:
    """Call OpenRouter API."""
    import requests
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/ai4he/moexp",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            elif resp.status_code == 429:
                wait = min(2 ** attempt * 5, 60)
                print(f"  OpenRouter rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  OpenRouter error {resp.status_code}: {resp.text[:200]}")
                time.sleep(2 ** attempt)
        except Exception as e:
            print(f"  OpenRouter request error: {e}")
            time.sleep(2 ** attempt)
    return ""


def llm_generate(prompt: str, api_keys: dict, provider: str = "gemini", **kwargs) -> str:
    """Unified LLM generation interface."""
    if provider == "gemini":
        return gemini_generate(prompt, key=api_keys["gemini"], **kwargs)
    elif provider == "groq":
        return groq_generate(prompt, api_key=api_keys["groq"], **kwargs)
    elif provider == "mistral":
        return mistral_generate(prompt, api_key=api_keys["mistral"], **kwargs)
    elif provider == "openrouter":
        return openrouter_generate(prompt, api_key=api_keys["openrouter"], **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ===== Knowledge Base =====

def load_knowledge_base(path: str) -> Dict[str, List[Dict]]:
    """Load extracted knowledge and organize by MoE expert domain."""
    domain_knowledge = defaultdict(list)

    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            domain = entry.get("domain_group", "other")
            extracted = entry.get("extracted", {})
            if not isinstance(extracted, dict):
                continue
            if not extracted.get("parse_error"):
                domain_knowledge[domain].append({
                    "id": entry.get("id"),
                    "domain": domain,
                    "statement_type": extracted.get("statement_type", "unknown"),
                    "formal_statement": extracted.get("formal_statement", ""),
                    "informal_description": extracted.get("informal_description", ""),
                    "key_concepts": extracted.get("key_concepts", []),
                    "potential_generalizations": extracted.get("potential_generalizations", ""),
                    "related_theorems": extracted.get("related_theorems", []),
                })

    return domain_knowledge


def format_results_for_prompt(entries: List[Dict], max_entries: int = 8) -> str:
    """Format knowledge base entries as text for LLM prompts."""
    selected = random.sample(entries, min(max_entries, len(entries)))
    parts = []
    for i, e in enumerate(selected, 1):
        stmt = e.get("formal_statement", e.get("informal_description", "N/A"))
        concepts = ", ".join(e.get("key_concepts", [])[:5])
        stype = e.get("statement_type", "unknown")
        parts.append(
            f"Result {i} ({stype}):\n"
            f"  Statement: {stmt[:400]}\n"
            f"  Key concepts: {concepts}\n"
        )
    return "\n".join(parts)


# ===== Conjecture Generation Strategies =====

PATTERN_INTERPOLATION_PROMPT = """You are a mathematical researcher specializing in {domain}.

Analyze the following collection of theorems/results from {domain} and identify common patterns.
Then generate novel conjectures by interpolating between these patterns.

Known Results:
{results}

Instructions:
1. Identify structural patterns shared across these results
2. Find parameters or conditions that vary between results
3. Interpolate: what happens for intermediate values or conditions?
4. Generate conjectures that fill "gaps" in the pattern

Return a JSON array of 3-5 conjecture objects, each with:
{{
    "strategy": "pattern_interpolation",
    "conjecture_statement": "precise mathematical statement in LaTeX",
    "informal_description": "plain English description",
    "pattern_identified": "what pattern was found",
    "source_results": ["which results inspire this"],
    "confidence": 0.0-1.0,
    "estimated_difficulty": "easy|medium|hard|open_problem",
    "lean4_sketch": "approximate Lean 4 formalization"
}}

Return ONLY valid JSON, no markdown formatting or code fences."""


CROSS_DOMAIN_PROMPT = """You are a creative mathematician who finds deep connections between different fields.

Source domain ({source_domain}) results:
{source_results}

Target domain ({target_domain}) context:
{target_results}

Instructions:
1. Identify structural similarities between source and target domains
2. Find theorems in the source domain that have no known analogue in the target
3. Formulate analogous conjectures for the target domain
4. Ensure the analogy is mathematically meaningful, not just superficial

Return a JSON array of 3-5 conjecture objects, each with:
{{
    "strategy": "cross_domain_analogy",
    "conjecture_statement": "precise mathematical statement in target domain (LaTeX)",
    "informal_description": "plain English description",
    "source_theorem": "the theorem being analogized",
    "analogy_mapping": "how concepts map between domains",
    "confidence": 0.0-1.0,
    "estimated_difficulty": "easy|medium|hard|open_problem",
    "lean4_sketch": "approximate Lean 4 formalization"
}}

Return ONLY valid JSON, no markdown formatting or code fences."""


COMPOSITION_PROMPT = """You are a mathematical researcher who combines existing results to discover new ones.

Given these results from {domain}:
{results}

Instructions:
1. Consider pairs or triples of results that share key concepts
2. Ask: what happens when we apply one result's technique to another's objects?
3. Look for results that, combined, might imply something new
4. Check if the converse of any combined result could hold

Return a JSON array of 3-5 conjecture objects, each with:
{{
    "strategy": "theorem_composition",
    "conjecture_statement": "precise mathematical statement in LaTeX",
    "informal_description": "plain English description",
    "composed_from": ["result 1", "result 2"],
    "composition_method": "how the results were combined",
    "confidence": 0.0-1.0,
    "estimated_difficulty": "easy|medium|hard|open_problem",
    "lean4_sketch": "approximate Lean 4 formalization"
}}

Return ONLY valid JSON, no markdown formatting or code fences."""


BOUNDARY_PROMPT = """You are a mathematical researcher who probes the limits of known results.

Given these results from {domain}:
{results}

Instructions:
1. For each result, identify its hypotheses/conditions
2. Ask: what if we weaken a hypothesis? Strengthen a conclusion?
3. What happens at boundary cases (n→∞, ε→0, dimension changes)?
4. Are the conditions necessary? Can any be removed?
5. Generate conjectures about these boundary behaviors

Return a JSON array of 3-5 conjecture objects, each with:
{{
    "strategy": "boundary_exploration",
    "conjecture_statement": "precise mathematical statement in LaTeX",
    "informal_description": "plain English description",
    "original_result": "the result being pushed",
    "boundary_type": "weakened_hypothesis|strengthened_conclusion|limit_case|necessity",
    "confidence": 0.0-1.0,
    "estimated_difficulty": "easy|medium|hard|open_problem",
    "lean4_sketch": "approximate Lean 4 formalization"
}}

Return ONLY valid JSON, no markdown formatting or code fences."""


THEOREM_GENERATION_PROMPT = """You are a mathematical researcher specializing in {domain}.

Given the following known results from {domain}, generate novel theorems that could plausibly be true
and provable. Focus on statements that fill gaps in the existing theory.

Known Results:
{results}

Instructions:
1. Identify gaps or missing connections between the known results
2. Formulate precise mathematical statements (theorems) that bridge these gaps
3. Each theorem should be non-trivial, precise, and include proof hints
4. Focus on statements that advance the field, not trivial consequences

Return a JSON array of 3-5 theorem objects, each with:
{{
    "strategy": "theorem_generation",
    "theorem_statement": "precise mathematical statement in LaTeX",
    "informal_description": "plain English description",
    "proof_sketch": "outline of how this might be proved",
    "key_techniques": ["mathematical techniques needed"],
    "confidence": 0.0-1.0,
    "estimated_difficulty": "easy|medium|hard|open_problem",
    "lean4_sketch": "approximate Lean 4 formalization"
}}

Return ONLY valid JSON, no markdown formatting or code fences."""


def _fix_latex_json(text: str) -> str:
    """Fix common issues with LaTeX in JSON: unescaped backslashes."""
    import re
    # Inside JSON string values, LaTeX commands like \geq need to be \\geq
    # But we need to be careful not to double-escape already escaped ones
    # Strategy: find string values and fix backslashes
    try:
        json.loads(text)
        return text  # Already valid
    except json.JSONDecodeError:
        pass

    # Replace single backslashes that aren't already escaped
    # Common LaTeX commands that appear in math
    latex_cmds = [
        'geq', 'leq', 'mathbb', 'mathcal', 'mathfrak', 'mathrm',
        'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'varepsilon',
        'theta', 'lambda', 'mu', 'nu', 'pi', 'sigma', 'tau', 'phi',
        'omega', 'infty', 'partial', 'nabla', 'forall', 'exists',
        'times', 'otimes', 'oplus', 'cdot', 'circ', 'cup', 'cap',
        'subset', 'subseteq', 'supset', 'supseteq', 'in', 'notin',
        'rightarrow', 'leftarrow', 'Rightarrow', 'Leftarrow',
        'langle', 'rangle', 'lfloor', 'rfloor', 'lceil', 'rceil',
        'frac', 'sqrt', 'sum', 'prod', 'int', 'oint', 'lim',
        'dim', 'ker', 'coker', 'Hom', 'Ext', 'Tor',
        'text', 'operatorname', 'overline', 'underline', 'hat', 'tilde',
        'quad', 'qquad', 'hspace', 'vspace',
        'begin', 'end', 'left', 'right',
        'QQ', 'ZZ', 'RR', 'CC', 'FF', 'PP',
        'ell', 'wp', 'aleph', 'beth',
    ]

    for cmd in latex_cmds:
        # Replace \cmd with \\cmd where not already escaped
        text = re.sub(r'(?<!\\)\\' + cmd + r'(?![a-zA-Z])', r'\\\\' + cmd, text)

    # Also handle \\ in common patterns
    text = text.replace('\\_', '\\\\_')

    return text


def parse_json_response(response: str) -> list:
    """Parse JSON array or object from LLM response, handling code fences and LaTeX."""
    if not response:
        return []

    import re
    clean = response.strip()

    def try_parse(text: str):
        """Try parsing JSON, with LaTeX fix and control character fallbacks."""
        text = text.strip()
        if not text:
            return None

        def _try_load(t):
            result = json.loads(t)
            if isinstance(result, dict):
                return [result]
            elif isinstance(result, list):
                return [x for x in result if isinstance(x, dict)]
            return None

        try:
            return _try_load(text)
        except json.JSONDecodeError:
            pass
        # Try fixing control characters (newlines/tabs inside JSON strings)
        try:
            fixed_ctrl = re.sub(r'[\x00-\x1f]', lambda m: '\\n' if m.group() == '\n' else '\\t' if m.group() == '\t' else '', text)
            return _try_load(fixed_ctrl)
        except json.JSONDecodeError:
            pass
        # Try fixing LaTeX escaping
        try:
            fixed = _fix_latex_json(text)
            return _try_load(fixed)
        except (json.JSONDecodeError, Exception):
            pass
        # Try both fixes combined
        try:
            fixed_both = re.sub(r'[\x00-\x1f]', lambda m: '\\n' if m.group() == '\n' else '\\t' if m.group() == '\t' else '', _fix_latex_json(text))
            return _try_load(fixed_both)
        except (json.JSONDecodeError, Exception):
            pass
        return None

    # Handle code fences (```json ... ``` or ``` ... ```)
    fence_pattern = re.compile(r'```(?:json)?\s*\n(.*?)\n\s*```', re.DOTALL)
    fence_matches = fence_pattern.findall(clean)
    if fence_matches:
        for match in fence_matches:
            result = try_parse(match)
            if result:
                return result

    # Simple fence removal
    if clean.startswith("```"):
        lines = clean.split("\n")
        clean = "\n".join(lines[1:])
    if clean.endswith("```"):
        clean = clean.rsplit("```", 1)[0]
    clean = clean.strip()

    # Try parsing as-is
    result = try_parse(clean)
    if result:
        return result

    # Try finding JSON object in the response
    obj_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', clean, re.DOTALL)
    if obj_match:
        result = try_parse(obj_match.group())
        if result:
            return result

    # Try most aggressive: find outermost { ... }
    brace_start = clean.find('{')
    brace_end = clean.rfind('}')
    if brace_start >= 0 and brace_end > brace_start:
        result = try_parse(clean[brace_start:brace_end + 1])
        if result:
            return result

    # Try finding JSON array
    arr_start = clean.find('[')
    arr_end = clean.rfind(']')
    if arr_start >= 0 and arr_end > arr_start:
        result = try_parse(clean[arr_start:arr_end + 1])
        if result:
            return result

    return []


# ===== Multi-Strategy Conjecture Generation =====

ANALOGY_PAIRS = [
    ("number_theory", "algebraic_geometry"),
    ("algebra", "algebraic_geometry"),
    ("geometry_topology", "algebraic_geometry"),
    ("analysis", "probability_statistics"),
    ("discrete_math", "algebra"),
    ("number_theory", "analysis"),
    ("algebra", "geometry_topology"),
]


def generate_conjectures_for_domain(
    domain: str,
    knowledge: List[Dict],
    api_keys: dict,
    provider: str = "gemini",
    include_theorems: bool = True,
    cross_domain_source: Optional[Tuple[str, List[Dict]]] = None,
) -> List[Dict]:
    """Generate conjectures for a domain using multiple strategies."""
    all_conjectures = []
    results_text = format_results_for_prompt(knowledge)

    strategies = [
        ("pattern_interpolation", PATTERN_INTERPOLATION_PROMPT),
        ("composition", COMPOSITION_PROMPT),
        ("boundary_exploration", BOUNDARY_PROMPT),
    ]

    if include_theorems:
        strategies.append(("theorem_generation", THEOREM_GENERATION_PROMPT))

    for strategy_name, prompt_template in strategies:
        print(f"  Strategy: {strategy_name}...")

        if strategy_name in ("pattern_interpolation", "composition", "boundary_exploration", "theorem_generation"):
            if strategy_name == "theorem_generation":
                prompt = prompt_template.format(domain=domain, results=results_text)
            else:
                prompt = prompt_template.format(domain=domain, results=results_text)

        response = llm_generate(prompt, api_keys, provider=provider,
                                temperature=0.8, max_tokens=4096)

        conjectures = parse_json_response(response)
        for c in conjectures:
            c["domain"] = domain
            c["strategy"] = strategy_name
            c["generator"] = provider
        all_conjectures.extend(conjectures)
        print(f"    -> {len(conjectures)} generated")

        time.sleep(2)

    # Cross-domain analogy
    if cross_domain_source:
        src_domain, src_knowledge = cross_domain_source
        print(f"  Strategy: cross_domain_analogy ({src_domain} -> {domain})...")
        src_text = format_results_for_prompt(src_knowledge)
        tgt_text = format_results_for_prompt(knowledge)

        prompt = CROSS_DOMAIN_PROMPT.format(
            source_domain=src_domain, source_results=src_text,
            target_domain=domain, target_results=tgt_text
        )

        response = llm_generate(prompt, api_keys, provider=provider,
                                temperature=0.8, max_tokens=4096)

        conjectures = parse_json_response(response)
        for c in conjectures:
            c["domain"] = domain
            c["strategy"] = "cross_domain_analogy"
            c["source_domain"] = src_domain
            c["generator"] = provider
        all_conjectures.extend(conjectures)
        print(f"    -> {len(conjectures)} generated")

    return all_conjectures


# ===== STP (Self-Play Theorem Prover) Loop =====

STP_CONJECTURE_PROMPT = """You are a creative mathematician in {domain}.

Given this mathematical context:
{context}

Previously generated conjectures (avoid repeating):
{previous}

Generate ONE novel mathematical conjecture that:
1. Is precise, well-defined, and falsifiable
2. Is non-trivial (not immediately obvious or a direct restatement)
3. Could potentially be proved or disproved using known techniques
4. Advances understanding in {domain}
5. Can be formalized in Lean 4

Return JSON:
{{
    "conjecture": "precise LaTeX statement",
    "informal": "plain English description",
    "motivation": "why this is interesting and plausible",
    "lean4_sketch": "approximate Lean 4 code",
    "proof_hint": "suggested approach to prove or disprove",
    "novelty_claim": "how this differs from known results"
}}

Return ONLY valid JSON, no code fences."""


STP_PROOF_PROMPT = """You are a rigorous mathematical proof assistant specializing in {domain}.

Analyze the following conjecture and attempt to either prove it, disprove it with a counterexample,
or provide a detailed analysis of why it's difficult.

Conjecture: {conjecture}
Informal description: {informal}
Proof hint from conjecturer: {hint}

Provide a thorough mathematical analysis:

Return JSON:
{{
    "verdict": "proved|disproved|partially_proved|unknown",
    "proof_or_counterexample": "the detailed proof, counterexample, or analysis",
    "key_steps": ["list of key logical steps"],
    "lean4_proof_sketch": "approximate Lean 4 proof (if proved)",
    "counterexample": "specific counterexample (if disproved)",
    "difficulty_assessment": "easy|medium|hard|very_hard|open_problem",
    "confidence": 0.0-1.0,
    "mathematical_rigor": 0.0-1.0,
    "feedback_for_conjecturer": "what makes this a good/bad conjecture and how to improve"
}}

Return ONLY valid JSON, no code fences."""


STP_JUDGE_PROMPT = """You are a senior mathematics professor evaluating generated conjectures and their proof attempts.

Domain: {domain}
Conjecture: {conjecture}
Proof attempt verdict: {verdict}
Proof/analysis: {proof}

Evaluate this conjecture on the following criteria (score each 0.0-1.0):

1. **Mathematical Correctness**: Is the statement well-formed and uses correct notation?
2. **Novelty**: Does it go beyond trivially restating known results?
3. **Non-triviality**: Is it a meaningful statement (not too easy or vacuously true)?
4. **Significance**: Would proving/disproving this advance mathematical knowledge?
5. **Formalizability**: Could this be stated precisely in Lean 4?
6. **Proof Quality**: Is the proof attempt rigorous and correct?

Return JSON:
{{
    "correctness": 0.0-1.0,
    "novelty": 0.0-1.0,
    "non_triviality": 0.0-1.0,
    "significance": 0.0-1.0,
    "formalizability": 0.0-1.0,
    "proof_quality": 0.0-1.0,
    "overall_score": 0.0-1.0,
    "verdict_agreement": "agree|disagree|partially_agree",
    "critique": "brief expert critique of the conjecture and proof",
    "is_publishable": true/false,
    "suggested_improvements": "how to improve the conjecture"
}}

Return ONLY valid JSON, no code fences."""


def run_stp_round(
    domain: str,
    knowledge: List[Dict],
    api_keys: dict,
    provider: str = "gemini",
    judge_provider: str = "groq",
    n_conjectures: int = 5,
    previous_conjectures: List[str] = None,
) -> Dict:
    """Run one round of Self-Play Theorem Prover."""
    context = format_results_for_prompt(knowledge, max_entries=6)
    prev_text = "\n".join(previous_conjectures[-5:]) if previous_conjectures else "None yet."

    round_results = {
        "domain": domain,
        "conjectures": [],
        "stats": {"total": 0, "proved": 0, "disproved": 0,
                  "partially_proved": 0, "unknown": 0, "trivial": 0,
                  "avg_quality": 0.0}
    }

    quality_scores = []

    for i in range(n_conjectures):
        print(f"    Conjecture {i+1}/{n_conjectures}...")

        # Step 1: Generate conjecture
        conj_prompt = STP_CONJECTURE_PROMPT.format(
            domain=domain, context=context, previous=prev_text
        )
        conj_response = llm_generate(conj_prompt, api_keys, provider=provider,
                                      temperature=0.9, max_tokens=2048)
        if not conj_response:
            print(f"      Empty API response for conjecture generation")
            continue
        conjecture = parse_json_response(conj_response)
        if not conjecture:
            # Fallback: try to use the raw response as the conjecture text
            # Extract text between code fences if present
            raw = conj_response
            if "```" in raw:
                import re
                fence = re.search(r'```(?:json)?\s*\n(.*?)\n\s*```', raw, re.DOTALL)
                if fence:
                    raw = fence.group(1)
            print(f"      Using fallback conjecture extraction")
            conjecture = {"conjecture": raw[:500], "informal": "", "proof_hint": "", "lean4_sketch": ""}
        else:
            conjecture = conjecture[0] if isinstance(conjecture, list) else conjecture
            if not isinstance(conjecture, dict):
                conjecture = {"conjecture": str(conjecture)[:500], "informal": "", "proof_hint": "", "lean4_sketch": ""}

        conj_text = conjecture.get("conjecture", "")
        if not conj_text:
            print(f"      No conjecture text found in parsed response")
            continue

        time.sleep(2)

        # Step 2: Attempt proof
        proof_prompt = STP_PROOF_PROMPT.format(
            domain=domain,
            conjecture=conj_text,
            informal=conjecture.get("informal", ""),
            hint=conjecture.get("proof_hint", "No hint provided")
        )
        proof_response = llm_generate(proof_prompt, api_keys, provider=provider,
                                       temperature=0.3, max_tokens=4096)
        proof_result = parse_json_response(proof_response) if proof_response else []
        if not proof_result:
            proof_result = {"verdict": "unknown", "raw_response": (proof_response or "")[:500]}
        else:
            proof_result = proof_result[0] if isinstance(proof_result, list) else proof_result

        time.sleep(2)

        # Step 3: Judge evaluation (use different provider for independence)
        verdict = proof_result.get("verdict", "unknown")
        proof_text = proof_result.get("proof_or_counterexample", "")

        actual_judge = judge_provider if judge_provider in api_keys else provider
        judge_prompt = STP_JUDGE_PROMPT.format(
            domain=domain,
            conjecture=conj_text,
            verdict=verdict,
            proof=proof_text[:2000]
        )
        judge_response = llm_generate(judge_prompt, api_keys, provider=actual_judge,
                                       temperature=0.2, max_tokens=2048)
        judge_result = parse_json_response(judge_response) if judge_response else []
        if not judge_result:
            judge_result = {"overall_score": 0.5, "critique": "Judge unavailable"}
        else:
            judge_result = judge_result[0] if isinstance(judge_result, list) else judge_result

        # Record result
        result_entry = {
            "conjecture": conjecture,
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

        # Track for dedup
        if previous_conjectures is not None:
            previous_conjectures.append(conj_text[:200])

        print(f"      Verdict: {verdict} | Quality: {judge_result.get('overall_score', '?'):.2f}")

        time.sleep(3)

    if quality_scores:
        round_results["stats"]["avg_quality"] = sum(quality_scores) / len(quality_scores)

    return round_results


def run_stp_loop(
    domain: str,
    knowledge: List[Dict],
    api_keys: dict,
    n_rounds: int = 3,
    n_per_round: int = 5,
    provider: str = "gemini",
    judge_provider: str = "groq",
    checkpoint_path: str = None,
) -> List[Dict]:
    """Run multiple rounds of STP."""
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
        print(f"  Resuming STP from round {start_round + 1}")

    for round_num in range(start_round, n_rounds):
        print(f"\n  === STP Round {round_num + 1}/{n_rounds} for {domain} ===")

        round_result = run_stp_round(
            domain=domain,
            knowledge=knowledge,
            api_keys=api_keys,
            provider=provider,
            judge_provider=judge_provider,
            n_conjectures=n_per_round,
            previous_conjectures=previous_conjectures,
        )
        all_rounds.append(round_result)

        # Save checkpoint
        if checkpoint_path:
            checkpoint_data = {
                "domain": domain,
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


# ===== Conjecture Quality Scoring =====

def score_conjecture(entry: Dict) -> float:
    """Score a conjecture using judge evaluation and heuristics."""
    judge = entry.get("judge_evaluation", {})

    # If judge provided an overall score, weight it heavily
    if "overall_score" in judge:
        judge_score = float(judge.get("overall_score", 0.5))
    else:
        # Compute from components
        weights = {
            "correctness": 0.20,
            "novelty": 0.25,
            "non_triviality": 0.20,
            "significance": 0.15,
            "formalizability": 0.10,
            "proof_quality": 0.10,
        }
        judge_score = sum(
            weights.get(k, 0) * float(judge.get(k, 0.5))
            for k in weights
        )

    # Heuristic adjustments
    conjecture = entry.get("conjecture", {})
    proof = entry.get("proof_attempt", {})
    verdict = proof.get("verdict", "unknown")

    # Bonus for proved conjectures (non-trivial ones are valuable)
    if verdict == "proved" and proof.get("difficulty_assessment", "") not in ("easy", "trivial"):
        judge_score = min(1.0, judge_score + 0.1)

    # Bonus for having Lean 4 sketch
    lean_sketch = conjecture.get("lean4_sketch", "")
    if lean_sketch and len(lean_sketch) > 30 and "theorem" in lean_sketch.lower():
        judge_score = min(1.0, judge_score + 0.05)

    # Penalty for trivial results
    if verdict == "trivial" or proof.get("difficulty_assessment") == "easy":
        judge_score = max(0.0, judge_score - 0.15)

    return round(judge_score, 4)


def rank_all_conjectures(multi_strategy: List[Dict], stp_results: Dict) -> List[Dict]:
    """Rank all conjectures from both multi-strategy and STP generation."""
    all_scored = []

    # Score multi-strategy conjectures (no judge evaluation, use heuristics)
    for c in multi_strategy:
        score = 0.5  # Base score

        # Confidence from the generator
        conf = c.get("confidence", 0.5)
        if isinstance(conf, (int, float)):
            score = 0.3 + 0.4 * conf

        # Difficulty bonus (harder = more interesting)
        diff_map = {"easy": -0.1, "medium": 0.0, "hard": 0.1, "open_problem": 0.15}
        diff = c.get("estimated_difficulty", "medium")
        score += diff_map.get(diff, 0.0)

        # Lean 4 sketch bonus
        lean = c.get("lean4_sketch", "")
        if lean and len(lean) > 20:
            score += 0.05

        all_scored.append({
            "source": "multi_strategy",
            "domain": c.get("domain", "unknown"),
            "strategy": c.get("strategy", "unknown"),
            "statement": c.get("conjecture_statement", c.get("theorem_statement", "")),
            "informal": c.get("informal_description", ""),
            "lean4_sketch": c.get("lean4_sketch", ""),
            "confidence": conf,
            "difficulty": diff,
            "quality_score": round(score, 4),
            "full_entry": c,
        })

    # Score STP conjectures (have judge evaluations)
    for domain, rounds in stp_results.items():
        for round_data in rounds:
            for entry in round_data.get("conjectures", []):
                score = score_conjecture(entry)
                conj = entry.get("conjecture", {})
                proof = entry.get("proof_attempt", {})
                judge = entry.get("judge_evaluation", {})

                all_scored.append({
                    "source": "stp",
                    "domain": domain,
                    "strategy": "stp",
                    "statement": conj.get("conjecture", ""),
                    "informal": conj.get("informal", ""),
                    "lean4_sketch": conj.get("lean4_sketch", ""),
                    "verdict": proof.get("verdict", "unknown"),
                    "proof_sketch": proof.get("proof_or_counterexample", "")[:500],
                    "confidence": proof.get("confidence", 0.5),
                    "difficulty": proof.get("difficulty_assessment", "unknown"),
                    "quality_score": score,
                    "judge_critique": judge.get("critique", ""),
                    "is_publishable": judge.get("is_publishable", False),
                    "full_entry": entry,
                })

    # Sort by quality score descending
    all_scored.sort(key=lambda x: -x["quality_score"])
    return all_scored


# ===== Evaluation Report =====

def generate_evaluation_report(
    multi_strategy: List[Dict],
    stp_results: Dict,
    ranked: List[Dict],
    output_path: str,
) -> Dict:
    """Generate comprehensive evaluation report."""
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {},
        "multi_strategy": {},
        "stp": {},
        "per_domain": {},
        "top_conjectures": [],
        "strategy_comparison": {},
    }

    # Overall summary
    total_multi = len(multi_strategy)
    total_stp = sum(
        sum(len(r.get("conjectures", [])) for r in rounds)
        for rounds in stp_results.values()
    )
    report["summary"] = {
        "total_conjectures": total_multi + total_stp,
        "multi_strategy_count": total_multi,
        "stp_count": total_stp,
        "domains_covered": list(set(
            [c.get("domain", "?") for c in multi_strategy] +
            list(stp_results.keys())
        )),
        "avg_quality_score": (
            sum(r["quality_score"] for r in ranked) / len(ranked)
            if ranked else 0.0
        ),
    }

    # Multi-strategy breakdown
    strategy_counts = Counter(c.get("strategy", "?") for c in multi_strategy)
    report["multi_strategy"] = {
        "total": total_multi,
        "by_strategy": dict(strategy_counts),
        "by_domain": dict(Counter(c.get("domain", "?") for c in multi_strategy)),
    }

    # STP breakdown
    stp_stats = {}
    for domain, rounds in stp_results.items():
        domain_stats = {"rounds": len(rounds), "total": 0, "proved": 0,
                        "disproved": 0, "unknown": 0, "avg_quality": 0.0}
        qualities = []
        for r in rounds:
            stats = r.get("stats", {})
            domain_stats["total"] += stats.get("total", 0)
            domain_stats["proved"] += stats.get("proved", 0)
            domain_stats["disproved"] += stats.get("disproved", 0)
            domain_stats["unknown"] += stats.get("unknown", 0)
            if stats.get("avg_quality", 0) > 0:
                qualities.append(stats["avg_quality"])
        if qualities:
            domain_stats["avg_quality"] = sum(qualities) / len(qualities)
        stp_stats[domain] = domain_stats
    report["stp"] = stp_stats

    # Per-domain analysis
    domain_scores = defaultdict(list)
    for r in ranked:
        domain_scores[r["domain"]].append(r["quality_score"])

    for domain, scores in domain_scores.items():
        report["per_domain"][domain] = {
            "count": len(scores),
            "avg_quality": sum(scores) / len(scores),
            "max_quality": max(scores),
            "min_quality": min(scores),
        }

    # Strategy comparison
    strategy_scores = defaultdict(list)
    for r in ranked:
        strategy_scores[r["strategy"]].append(r["quality_score"])

    for strategy, scores in strategy_scores.items():
        report["strategy_comparison"][strategy] = {
            "count": len(scores),
            "avg_quality": sum(scores) / len(scores),
            "max_quality": max(scores),
        }

    # Top conjectures (without full_entry to keep report concise)
    for r in ranked[:20]:
        entry = {k: v for k, v in r.items() if k != "full_entry"}
        report["top_conjectures"].append(entry)

    # Save report
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


def print_report(report: Dict):
    """Print evaluation report to console."""
    summary = report["summary"]
    print("\n" + "=" * 70)
    print("MATHSCY CONJECTURE & THEOREM GENERATION EVALUATION")
    print("=" * 70)

    print(f"\nTotal conjectures generated: {summary['total_conjectures']}")
    print(f"  Multi-strategy: {summary['multi_strategy_count']}")
    print(f"  STP loop:       {summary['stp_count']}")
    print(f"  Domains:        {', '.join(summary['domains_covered'])}")
    print(f"  Avg quality:    {summary['avg_quality_score']:.3f}")

    # Strategy comparison
    print("\n--- Strategy Comparison ---")
    for strategy, stats in sorted(report["strategy_comparison"].items(),
                                   key=lambda x: -x[1]["avg_quality"]):
        print(f"  {strategy:30s} | count: {stats['count']:3d} | "
              f"avg: {stats['avg_quality']:.3f} | max: {stats['max_quality']:.3f}")

    # Per-domain
    print("\n--- Per-Domain Results ---")
    for domain, stats in sorted(report["per_domain"].items(),
                                 key=lambda x: -x[1]["avg_quality"]):
        print(f"  {domain:25s} | count: {stats['count']:3d} | "
              f"avg: {stats['avg_quality']:.3f} | max: {stats['max_quality']:.3f}")

    # STP stats
    if report["stp"]:
        print("\n--- STP Loop Results ---")
        for domain, stats in report["stp"].items():
            print(f"  {domain}: {stats['total']} conjectures over {stats['rounds']} rounds")
            print(f"    proved: {stats['proved']}, disproved: {stats['disproved']}, "
                  f"unknown: {stats['unknown']}, avg_quality: {stats['avg_quality']:.3f}")

    # Top 10 conjectures
    print("\n--- Top 10 Conjectures ---")
    for i, entry in enumerate(report["top_conjectures"][:10], 1):
        print(f"\n  {i}. [{entry['domain']}] Score: {entry['quality_score']:.3f} "
              f"({entry['strategy']}) {'[PUBLISHABLE]' if entry.get('is_publishable') else ''}")
        stmt = entry.get("statement", "")[:200]
        print(f"     {stmt}")
        informal = entry.get("informal", "")[:150]
        if informal:
            print(f"     -> {informal}")

    print("\n" + "=" * 70)


# ===== Main =====

def main():
    parser = argparse.ArgumentParser(description="MathScy Conjecture Generation Evaluation")
    parser.add_argument("--mode", choices=["full", "generate", "stp", "evaluate"],
                        default="full", help="Evaluation mode")
    parser.add_argument("--domains", type=str, default=None,
                        help="Comma-separated list of domains (default: all 7)")
    parser.add_argument("--provider", type=str, default="gemini",
                        help="LLM provider for generation (gemini/groq/mistral)")
    parser.add_argument("--judge", type=str, default="groq",
                        help="LLM provider for judging (gemini/groq/mistral)")
    parser.add_argument("--rounds", type=int, default=3,
                        help="STP rounds per domain")
    parser.add_argument("--per-round", type=int, default=5,
                        help="Conjectures per STP round")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(PROJECT_DIR, "results"),
                        help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load API keys
    print("Loading API keys...")
    api_keys = load_api_keys()
    available = [k for k in api_keys if api_keys[k]]
    print(f"  Available providers: {', '.join(available)}")

    if args.provider not in api_keys:
        print(f"  WARNING: {args.provider} not available, falling back to gemini")
        args.provider = "gemini"

    # Determine domains
    all_expert_domains = [
        "algebraic_geometry", "discrete_math", "number_theory",
        "analysis", "algebra", "geometry_topology", "probability_statistics"
    ]

    if args.domains:
        domains = [d.strip() for d in args.domains.split(",")]
    else:
        domains = all_expert_domains

    print(f"  Domains: {', '.join(domains)}")

    # Load knowledge base
    kb_path = os.path.join(PROJECT_DIR, "data", "extracted_knowledge.jsonl")
    print(f"\nLoading knowledge base from {kb_path}...")
    knowledge_base = load_knowledge_base(kb_path)
    for d in sorted(knowledge_base.keys()):
        if d in domains:
            print(f"  {d}: {len(knowledge_base[d])} entries")

    # Paths
    conjectures_path = os.path.join(args.output_dir, "generated_conjectures.jsonl")
    ranked_path = os.path.join(args.output_dir, "ranked_conjectures.json")
    report_path = os.path.join(args.output_dir, "conjecture_evaluation_report.json")

    # ===== Phase 1: Multi-Strategy Conjecture Generation =====
    all_multi_strategy = []

    if args.mode in ("full", "generate"):
        print("\n" + "=" * 60)
        print("PHASE 1: Multi-Strategy Conjecture Generation")
        print("=" * 60)

        # Load existing conjectures
        processed_domains = set()
        if os.path.exists(conjectures_path):
            with open(conjectures_path) as f:
                for line in f:
                    c = json.loads(line)
                    all_multi_strategy.append(c)
                    processed_domains.add(c.get("domain"))
            print(f"  Resumed {len(all_multi_strategy)} existing conjectures "
                  f"from {len(processed_domains)} domains")

        for domain in domains:
            if domain in processed_domains:
                print(f"\n  Skipping {domain} (already processed)")
                continue

            if domain not in knowledge_base or len(knowledge_base[domain]) < 3:
                print(f"\n  Skipping {domain} (only {len(knowledge_base.get(domain, []))} entries)")
                continue

            print(f"\n{'='*50}")
            print(f"Generating conjectures for: {domain}")
            print(f"{'='*50}")

            # Find cross-domain source
            cross_source = None
            for src, tgt in ANALOGY_PAIRS:
                if tgt == domain and src in knowledge_base and len(knowledge_base[src]) >= 3:
                    cross_source = (src, knowledge_base[src])
                    break

            domain_conjs = generate_conjectures_for_domain(
                domain=domain,
                knowledge=knowledge_base[domain],
                api_keys=api_keys,
                provider=args.provider,
                include_theorems=True,
                cross_domain_source=cross_source,
            )

            # Save incrementally
            with open(conjectures_path, "a") as f:
                for c in domain_conjs:
                    f.write(json.dumps(c) + "\n")

            all_multi_strategy.extend(domain_conjs)
            processed_domains.add(domain)
            print(f"  Total for {domain}: {len(domain_conjs)} conjectures")

        print(f"\nPhase 1 complete: {len(all_multi_strategy)} multi-strategy conjectures")

    elif args.mode == "evaluate":
        # Load existing conjectures
        if os.path.exists(conjectures_path):
            with open(conjectures_path) as f:
                for line in f:
                    all_multi_strategy.append(json.loads(line))
            print(f"Loaded {len(all_multi_strategy)} existing multi-strategy conjectures")

    # ===== Phase 2: STP Loop =====
    stp_results = {}

    if args.mode in ("full", "stp"):
        print("\n" + "=" * 60)
        print("PHASE 2: Self-Play Theorem Prover (STP) Loop")
        print("=" * 60)

        for domain in domains:
            if domain not in knowledge_base or len(knowledge_base[domain]) < 3:
                print(f"  Skipping STP for {domain} (insufficient data)")
                continue

            checkpoint = os.path.join(args.output_dir, f"stp_{domain}_checkpoint.json")

            print(f"\n{'='*50}")
            print(f"Running STP for: {domain}")
            print(f"{'='*50}")

            rounds = run_stp_loop(
                domain=domain,
                knowledge=knowledge_base[domain],
                api_keys=api_keys,
                n_rounds=args.rounds,
                n_per_round=args.per_round,
                provider=args.provider,
                judge_provider=args.judge,
                checkpoint_path=checkpoint,
            )
            stp_results[domain] = rounds

        total_stp = sum(
            sum(len(r.get("conjectures", [])) for r in rounds)
            for rounds in stp_results.values()
        )
        print(f"\nPhase 2 complete: {total_stp} STP conjectures")

    elif args.mode == "evaluate":
        # Load existing STP results
        for domain in domains:
            checkpoint = os.path.join(args.output_dir, f"stp_{domain}_checkpoint.json")
            if os.path.exists(checkpoint):
                with open(checkpoint) as f:
                    data = json.load(f)
                stp_results[domain] = data.get("rounds", [])
                print(f"Loaded STP checkpoint for {domain}")

    # ===== Phase 3: Ranking & Evaluation =====
    print("\n" + "=" * 60)
    print("PHASE 3: Ranking & Evaluation Report")
    print("=" * 60)

    ranked = rank_all_conjectures(all_multi_strategy, stp_results)

    # Save ranked conjectures
    ranked_for_save = [{k: v for k, v in r.items() if k != "full_entry"} for r in ranked]
    with open(ranked_path, "w") as f:
        json.dump(ranked_for_save, f, indent=2)
    print(f"Saved {len(ranked)} ranked conjectures to {ranked_path}")

    # Generate and print report
    report = generate_evaluation_report(
        all_multi_strategy, stp_results, ranked, report_path
    )
    print_report(report)

    # Save Lean 4 verification queue (top 50)
    lean_queue = []
    for r in ranked[:50]:
        lean_queue.append({
            "domain": r["domain"],
            "quality_score": r["quality_score"],
            "conjecture_latex": r.get("statement", ""),
            "informal": r.get("informal", ""),
            "lean4_sketch": r.get("lean4_sketch", ""),
            "strategy": r["strategy"],
            "source": r["source"],
            "verification_status": "pending",
        })

    lean_queue_path = os.path.join(args.output_dir, "lean_verification_queue.json")
    with open(lean_queue_path, "w") as f:
        json.dump(lean_queue, f, indent=2)
    print(f"\nSaved top {len(lean_queue)} conjectures to Lean 4 verification queue")

    # Summary file
    summary_path = os.path.join(args.output_dir, "conjecture_generation_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_conjectures": len(ranked),
            "multi_strategy": len(all_multi_strategy),
            "stp": sum(
                sum(len(r.get("conjectures", [])) for r in rounds)
                for rounds in stp_results.values()
            ),
            "domains": domains,
            "avg_quality": report["summary"]["avg_quality_score"],
            "top_score": ranked[0]["quality_score"] if ranked else 0,
            "strategies_used": list(report["strategy_comparison"].keys()),
        }, f, indent=2)

    print(f"\nAll results saved to {args.output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
