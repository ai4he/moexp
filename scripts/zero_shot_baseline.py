"""
Zero-Shot Conjecture Generation Baseline for MathScy

Generates conjectures using a zero-shot prompt with Gemini-2.0-Flash,
then evaluates them with the Mistral LLM judge.

This provides a baseline comparison against the MathScy MoE pipeline's conjecture generation.

Usage:
    python scripts/zero_shot_baseline.py
"""

import os
import sys
import json
import time
import re
import traceback
import numpy as np
from datetime import datetime

# ===== Configuration =====

PROJECT_DIR = "/scratch/ctoxtli/moexp"
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

# Load API keys
with open(os.path.join(PROJECT_DIR, "working_Gemini_API_keys.json")) as f:
    GEMINI_API_KEYS = json.load(f)  # Try all keys, rotate on rate limit

with open(os.path.join(PROJECT_DIR, "mistral_key.txt")) as f:
    MISTRAL_API_KEY = f.read().strip()

with open(os.path.join(PROJECT_DIR, "groqcloud_key.txt")) as f:
    GROQ_API_KEY = f.read().strip()

GEMINI_MODEL = "gemini-2.0-flash"
MISTRAL_MODEL = "mistral-small-latest"
GROQ_MODEL = "llama-3.3-70b-versatile"

DOMAINS = [
    "algebraic_geometry",
    "discrete_math",
    "number_theory",
    "analysis",
    "algebra",
    "geometry_topology",
    "probability_statistics",
]

CONJECTURES_PER_DOMAIN = 2
GENERATION_TEMPERATURE = 0.7
API_DELAY = 3  # seconds between API calls

# ===== Prompt Templates =====

GENERATION_PROMPT = """You are a mathematician. Generate a novel mathematical conjecture in the domain of {domain}.
Provide:
1. A formal statement of the conjecture
2. A brief informal description
3. Motivation for why this conjecture might be interesting
4. A difficulty estimate (easy/medium/hard)
5. A brief proof sketch or evidence supporting the conjecture

Format as JSON with keys: statement, informal, motivation, difficulty, proof_sketch"""

JUDGE_PROMPT = """You are an expert mathematical judge. Evaluate the following conjecture on these criteria (score 0-1 each):
- correctness: How likely is this conjecture to be mathematically correct?
- novelty: How novel is this conjecture relative to known results?
- non_triviality: Is this a non-trivial statement?
- significance: How significant would this result be if true?
- formalizability: Can this be formalized in a proof assistant?
- proof_quality: How rigorous is the proof sketch?

Conjecture: {statement}
Domain: {domain}
Proof sketch: {proof_sketch}

Return JSON with keys: correctness, novelty, non_triviality, significance, formalizability, proof_quality, overall_quality (weighted average), critique"""

# ===== Helper Functions =====

def fix_json_escapes(text: str) -> str:
    """Fix invalid JSON escape sequences (e.g., LaTeX \\to, \\theta, \\frac) by doubling
    backslashes that are not valid JSON escapes."""
    # Step 1: Fix obvious non-JSON escapes (backslash + non-escape char)
    text = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)
    # Step 2: Fix \n/\t/\b/\f/\r followed by alpha (likely LaTeX, not JSON escape)
    text = re.sub(r'\\([ntbfr])(?=[a-zA-Z])', r'\\\\\1', text)
    return text


def extract_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks and LaTeX content."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract content from markdown code block if present
    code_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    json_str = code_block_match.group(1).strip() if code_block_match else None

    # If no code block, try to find outermost braces
    if json_str is None:
        brace_match = re.search(r'\{.*\}', text, re.DOTALL)
        if brace_match:
            json_str = brace_match.group(0)

    if json_str is None:
        json_str = text.strip()

    # Try direct parse of extracted string
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Try with fixed escapes
    try:
        fixed = fix_json_escapes(json_str)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Last resort: brute force replace all backslashes, then fix quotes
    try:
        brute = json_str.replace('\\', '\\\\')
        brute = brute.replace('\\\\"', '\\"')
        return json.loads(brute)
    except json.JSONDecodeError:
        pass

    raise ValueError(f"Could not parse JSON from response: {text[:300]}...")


def gemini_generate(prompt: str, temperature: float = 0.7, max_tokens: int = 2048) -> str:
    """Generate text using the Gemini API directly via REST. Cycles through all API keys."""
    import requests

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
    }

    # Try each key, cycling through them
    for key_idx, api_key in enumerate(GEMINI_API_KEYS):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
        try:
            response = requests.post(url, json=payload, timeout=60)
            if response.status_code == 429:
                print(f"    Gemini key {key_idx+1} rate limited, trying next...")
                continue
            response.raise_for_status()
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            if "429" in str(e):
                continue
            if key_idx < len(GEMINI_API_KEYS) - 1:
                print(f"    Gemini key {key_idx+1} failed: {str(e)[:80]}, trying next...")
                continue
            else:
                raise

    # If all keys failed, wait and retry with first working key
    print(f"    All Gemini keys rate limited, waiting 60s...")
    time.sleep(60)
    for api_key in GEMINI_API_KEYS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
        try:
            response = requests.post(url, json=payload, timeout=60)
            if response.status_code == 200:
                data = response.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            continue

    raise RuntimeError("Gemini generation failed: all keys exhausted")


def mistral_generate(prompt: str, temperature: float = 0.1, max_tokens: int = 2048) -> str:
    """Generate text using the Mistral API via OpenAI-compatible endpoint."""
    from openai import OpenAI

    client = OpenAI(
        base_url="https://api.mistral.ai/v1",
        api_key=MISTRAL_API_KEY,
    )

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MISTRAL_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate" in error_str.lower():
                wait = 10 * (attempt + 1)
                print(f"    Mistral rate limited, waiting {wait}s...")
                time.sleep(wait)
            elif attempt < 2:
                print(f"    Mistral attempt {attempt+1} failed: {e}, retrying...")
                time.sleep(5)
            else:
                raise
    raise RuntimeError("Mistral generation failed after all retries")


def groq_generate(prompt: str, temperature: float = 0.1, max_tokens: int = 2048) -> str:
    """Generate text using the Groq API."""
    from openai import OpenAI

    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY,
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def generate_conjecture(domain: str, attempt: int = 1) -> dict:
    """Generate a single conjecture using zero-shot prompting via Gemini."""
    prompt = GENERATION_PROMPT.format(domain=domain.replace("_", " "))

    try:
        raw_text = gemini_generate(prompt, temperature=GENERATION_TEMPERATURE)
        parsed = extract_json(raw_text)

        # Validate required keys
        required_keys = ["statement", "informal", "motivation", "difficulty", "proof_sketch"]
        for key in required_keys:
            if key not in parsed:
                parsed[key] = "N/A"

        return {
            "domain": domain,
            "attempt": attempt,
            "raw_response": raw_text,
            **parsed,
            "generation_model": GEMINI_MODEL,
            "generation_method": "zero_shot",
            "temperature": GENERATION_TEMPERATURE,
        }
    except Exception as e:
        print(f"  [ERROR] Generation failed for {domain} attempt {attempt}: {e}")
        return {
            "domain": domain,
            "attempt": attempt,
            "error": str(e),
            "statement": "GENERATION_FAILED",
            "informal": "",
            "motivation": "",
            "difficulty": "",
            "proof_sketch": "",
            "generation_model": GEMINI_MODEL,
            "generation_method": "zero_shot",
            "temperature": GENERATION_TEMPERATURE,
        }


def judge_conjecture(conjecture: dict) -> dict:
    """Evaluate a conjecture using the LLM judge. Try Mistral first, fallback to Groq."""
    if conjecture.get("statement", "") == "GENERATION_FAILED":
        return {
            "correctness": 0.0,
            "novelty": 0.0,
            "non_triviality": 0.0,
            "significance": 0.0,
            "formalizability": 0.0,
            "proof_quality": 0.0,
            "overall_quality": 0.0,
            "critique": "Conjecture generation failed.",
            "judge_model": "none",
        }

    prompt = JUDGE_PROMPT.format(
        statement=conjecture.get("statement", ""),
        domain=conjecture.get("domain", "").replace("_", " "),
        proof_sketch=conjecture.get("proof_sketch", ""),
    )

    # Try Mistral first (most reliable)
    raw_text = None
    judge_model_used = None
    try:
        raw_text = mistral_generate(prompt, temperature=0.1)
        judge_model_used = MISTRAL_MODEL
        print(f"    (Judged via Mistral)")
    except Exception as e:
        print(f"    Mistral failed ({type(e).__name__}: {str(e)[:80]}), trying Groq...")

    # Fallback to Groq
    if raw_text is None:
        try:
            raw_text = groq_generate(prompt, temperature=0.1)
            judge_model_used = GROQ_MODEL
            print(f"    (Judged via Groq)")
        except Exception as e:
            print(f"    [ERROR] All judges failed: {e}")
            return {
                "correctness": 0.0,
                "novelty": 0.0,
                "non_triviality": 0.0,
                "significance": 0.0,
                "formalizability": 0.0,
                "proof_quality": 0.0,
                "overall_quality": 0.0,
                "critique": f"All judges failed: {str(e)[:200]}",
                "judge_model": "none",
            }

    try:
        parsed = extract_json(raw_text)

        # Ensure all score keys exist and are floats
        score_keys = ["correctness", "novelty", "non_triviality", "significance",
                       "formalizability", "proof_quality", "overall_quality"]
        for key in score_keys:
            if key not in parsed:
                parsed[key] = 0.0
            else:
                try:
                    parsed[key] = float(parsed[key])
                except (ValueError, TypeError):
                    parsed[key] = 0.0

        if "critique" not in parsed:
            parsed["critique"] = ""

        parsed["judge_model"] = judge_model_used
        return parsed
    except Exception as e:
        print(f"    [ERROR] Failed to parse judge response: {e}")
        print(f"    Raw response: {raw_text[:200]}...")
        return {
            "correctness": 0.0,
            "novelty": 0.0,
            "non_triviality": 0.0,
            "significance": 0.0,
            "formalizability": 0.0,
            "proof_quality": 0.0,
            "overall_quality": 0.0,
            "critique": f"Judge response parsing failed: {str(e)[:200]}",
            "judge_model": judge_model_used,
        }


# ===== Main Experiment =====

def run_experiment():
    """Run the zero-shot baseline experiment."""
    print("=" * 70)
    print("Zero-Shot Conjecture Generation Baseline Experiment")
    print("=" * 70)
    print(f"Generation model: {GEMINI_MODEL} (via Gemini API)")
    print(f"Judge model: {MISTRAL_MODEL} (via Mistral, fallback to Groq)")
    print(f"Domains: {len(DOMAINS)}")
    print(f"Conjectures per domain: {CONJECTURES_PER_DOMAIN}")
    print(f"Total conjectures: {len(DOMAINS) * CONJECTURES_PER_DOMAIN}")
    print(f"Temperature: {GENERATION_TEMPERATURE}")
    print()

    all_conjectures = []
    per_domain_results = {}

    # Phase 1: Generate conjectures
    print("\n" + "=" * 50)
    print("PHASE 1: Generating conjectures (zero-shot)")
    print("=" * 50)

    for domain in DOMAINS:
        domain_display = domain.replace("_", " ").title()
        print(f"\n--- Domain: {domain_display} ---")

        for attempt in range(1, CONJECTURES_PER_DOMAIN + 1):
            print(f"  Generating conjecture {attempt}/{CONJECTURES_PER_DOMAIN}...")
            conjecture = generate_conjecture(domain, attempt)

            if conjecture.get("statement") != "GENERATION_FAILED":
                stmt_preview = conjecture['statement'][:100]
                print(f"    Statement: {stmt_preview}...")
                print(f"    Difficulty: {conjecture.get('difficulty', 'N/A')}")
            else:
                print(f"    [FAILED] {conjecture.get('error', 'unknown error')[:100]}")

            all_conjectures.append(conjecture)
            time.sleep(API_DELAY)

    successful = sum(1 for c in all_conjectures if c.get("statement") != "GENERATION_FAILED")
    print(f"\nGenerated {len(all_conjectures)} conjectures total.")
    print(f"Successful: {successful}, Failed: {len(all_conjectures) - successful}")

    # Phase 2: Judge conjectures
    print("\n" + "=" * 50)
    print("PHASE 2: Evaluating conjectures (LLM judge)")
    print("=" * 50)

    for i, conjecture in enumerate(all_conjectures):
        domain_display = conjecture["domain"].replace("_", " ").title()
        print(f"\n  Judging conjecture {i+1}/{len(all_conjectures)} "
              f"({domain_display}, attempt {conjecture['attempt']})...")

        scores = judge_conjecture(conjecture)
        conjecture["scores"] = scores
        print(f"    Overall quality: {scores['overall_quality']:.3f}")
        print(f"    Correctness: {scores['correctness']:.2f}, Novelty: {scores['novelty']:.2f}, "
              f"Non-triviality: {scores['non_triviality']:.2f}")
        if scores.get("critique"):
            critique_preview = scores["critique"][:120]
            print(f"    Critique: {critique_preview}...")

        time.sleep(API_DELAY)

    # Phase 3: Aggregate results
    print("\n" + "=" * 50)
    print("PHASE 3: Aggregating results")
    print("=" * 50)

    all_qualities = []
    for domain in DOMAINS:
        domain_conjectures = [c for c in all_conjectures if c["domain"] == domain]
        qualities = [c["scores"]["overall_quality"] for c in domain_conjectures
                     if c.get("scores", {}).get("overall_quality") is not None]

        domain_avg = float(np.mean(qualities)) if qualities else 0.0
        domain_std = float(np.std(qualities)) if qualities else 0.0
        all_qualities.extend(qualities)

        per_domain_results[domain] = {
            "num_conjectures": len(domain_conjectures),
            "num_successful": sum(1 for c in domain_conjectures
                                  if c.get("statement") != "GENERATION_FAILED"),
            "avg_quality": round(domain_avg, 4),
            "std_quality": round(domain_std, 4),
            "avg_correctness": round(float(np.mean(
                [c["scores"]["correctness"] for c in domain_conjectures])), 4),
            "avg_novelty": round(float(np.mean(
                [c["scores"]["novelty"] for c in domain_conjectures])), 4),
            "avg_non_triviality": round(float(np.mean(
                [c["scores"]["non_triviality"] for c in domain_conjectures])), 4),
            "avg_significance": round(float(np.mean(
                [c["scores"]["significance"] for c in domain_conjectures])), 4),
            "avg_formalizability": round(float(np.mean(
                [c["scores"]["formalizability"] for c in domain_conjectures])), 4),
            "avg_proof_quality": round(float(np.mean(
                [c["scores"]["proof_quality"] for c in domain_conjectures])), 4),
        }

        domain_display = domain.replace("_", " ").title()
        print(f"  {domain_display}: avg_quality={domain_avg:.4f} (std={domain_std:.4f})")

    overall_avg = float(np.mean(all_qualities)) if all_qualities else 0.0
    overall_std = float(np.std(all_qualities)) if all_qualities else 0.0

    print(f"\n  OVERALL: avg_quality={overall_avg:.4f} (std={overall_std:.4f})")

    # Determine actual judge model used
    judge_models_used = set()
    for c in all_conjectures:
        jm = c.get("scores", {}).get("judge_model")
        if jm and jm != "none":
            judge_models_used.add(jm)
    judge_str = ", ".join(sorted(judge_models_used)) if judge_models_used else MISTRAL_MODEL

    # Prepare output - remove raw_response to keep output clean
    conjectures_output = []
    for c in all_conjectures:
        output_entry = {k: v for k, v in c.items() if k != "raw_response"}
        conjectures_output.append(output_entry)

    # Build final results
    results = {
        "task": "task16_zero_shot_baseline",
        "description": "Zero-shot conjecture generation baseline",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": "gemini-2.0-flash",
        "model_full": GEMINI_MODEL,
        "judge": judge_str,
        "generation_method": "zero_shot",
        "temperature": GENERATION_TEMPERATURE,
        "domains": DOMAINS,
        "conjectures_per_domain": CONJECTURES_PER_DOMAIN,
        "total_conjectures": len(all_conjectures),
        "successful_conjectures": successful,
        "per_domain_results": per_domain_results,
        "overall_avg_quality": round(overall_avg, 4),
        "overall_std_quality": round(overall_std, 4),
        "overall_avg_correctness": round(float(np.mean(
            [c["scores"]["correctness"] for c in all_conjectures])), 4),
        "overall_avg_novelty": round(float(np.mean(
            [c["scores"]["novelty"] for c in all_conjectures])), 4),
        "overall_avg_non_triviality": round(float(np.mean(
            [c["scores"]["non_triviality"] for c in all_conjectures])), 4),
        "overall_avg_significance": round(float(np.mean(
            [c["scores"]["significance"] for c in all_conjectures])), 4),
        "overall_avg_formalizability": round(float(np.mean(
            [c["scores"]["formalizability"] for c in all_conjectures])), 4),
        "overall_avg_proof_quality": round(float(np.mean(
            [c["scores"]["proof_quality"] for c in all_conjectures])), 4),
        "conjectures": conjectures_output,
    }

    # Save results
    output_path = os.path.join(RESULTS_DIR, "task16_zero_shot_baseline.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Print comparison summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"Zero-shot baseline overall quality: {overall_avg:.4f} (std={overall_std:.4f})")
    print(f"MathScy pipeline overall quality:   0.6931 (from main evaluation)")
    print(f"Delta:                              {overall_avg - 0.6931:+.4f}")
    print()
    print("Per-domain comparison:")
    print(f"{'Domain':<25} {'Zero-shot':>10} {'MathScy':>10} {'Delta':>10}")
    print("-" * 57)
    # MathScy pipeline averages from evaluation report (STP results)
    mathscy_avgs = {
        "algebraic_geometry": 0.738,
        "discrete_math": 0.663,
        "number_theory": 0.700,
        "analysis": 0.689,
        "algebra": 0.850,
        "geometry_topology": 0.633,
        "probability_statistics": 0.679,
    }
    for domain in DOMAINS:
        zs = per_domain_results[domain]["avg_quality"]
        ms = mathscy_avgs.get(domain, 0.0)
        delta = zs - ms
        domain_display = domain.replace("_", " ").title()
        print(f"  {domain_display:<23} {zs:>10.4f} {ms:>10.4f} {delta:>+10.4f}")

    print("\nDone!")
    return results


if __name__ == "__main__":
    run_experiment()
