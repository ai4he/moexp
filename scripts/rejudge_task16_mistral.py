#!/usr/bin/env python3
"""Re-judge task16 zero-shot baseline conjectures using Mistral API."""

import json
import time
import re
import sys
import requests
import numpy as np
from datetime import datetime

def log(msg):
    print(msg, flush=True)

# Config
RESULTS_PATH = "/scratch/ctoxtli/moexp/results/task16_zero_shot_baseline.json"
MISTRAL_KEY_PATH = "/scratch/ctoxtli/moexp/mistral_key.txt"
MISTRAL_ENDPOINT = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL = "mistral-small-latest"
DELAY_BETWEEN_CALLS = 5  # seconds

# Read API key
with open(MISTRAL_KEY_PATH) as f:
    MISTRAL_API_KEY = f.read().strip()
log(f"API key loaded: {MISTRAL_API_KEY[:8]}...")

# Read existing results
with open(RESULTS_PATH) as f:
    results = json.load(f)
log(f"Loaded results with {len(results['conjectures'])} conjectures")

def build_judge_prompt(domain, statement):
    """Build the judge prompt for evaluating a conjecture."""
    return f"""You are an expert mathematician acting as a judge for AI-generated mathematical conjectures.

Evaluate the following conjecture from the domain of {domain.replace('_', ' ')}:

CONJECTURE:
{statement}

Rate this conjecture on each of the following 6 criteria using a score from 0.0 to 1.0:

1. **correctness** (0.0-1.0): Is the conjecture mathematically well-formed and plausibly true? Does it use correct notation and definitions?
2. **novelty** (0.0-1.0): Is this conjecture genuinely new or interesting, rather than a restatement of known results?
3. **non_triviality** (0.0-1.0): Is the conjecture non-trivial? Would proving it require substantial mathematical work?
4. **significance** (0.0-1.0): If true, would this conjecture have meaningful implications for the field?
5. **formalizability** (0.0-1.0): Could this conjecture be formally stated in a proof assistant like Lean 4?
6. **proof_quality** (0.0-1.0): Does the conjecture come with a reasonable proof sketch or approach? (Score 0.3 if no proof is provided but the statement is clear.)

Also provide:
- **overall_score** (0.0-1.0): A weighted average reflecting the overall quality of this conjecture.
- **critique**: A brief (2-3 sentence) critique explaining your scores.

Respond ONLY with a JSON object in this exact format:
{{
  "correctness": 0.0,
  "novelty": 0.0,
  "non_triviality": 0.0,
  "significance": 0.0,
  "formalizability": 0.0,
  "proof_quality": 0.0,
  "overall_score": 0.0,
  "critique": "Your critique here."
}}"""


def call_mistral(prompt, max_retries=3):
    """Call Mistral API with retry logic."""
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MISTRAL_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 1000
    }

    for attempt in range(max_retries):
        try:
            log(f"  API call attempt {attempt+1}/{max_retries}...")
            resp = requests.post(MISTRAL_ENDPOINT, headers=headers, json=payload, timeout=30)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                log(f"  Got response ({len(content)} chars)")
                return content
            elif resp.status_code == 429:
                wait_time = 10 * (attempt + 1)
                log(f"  Rate limited (429). Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                log(f"  API error {resp.status_code}: {resp.text[:200]}")
                time.sleep(5)
        except requests.exceptions.Timeout:
            log(f"  Request timed out (30s)")
            time.sleep(5)
        except Exception as e:
            log(f"  Request error: {e}")
            time.sleep(5)

    return None


def parse_judge_response(response_text):
    """Parse JSON from Mistral response, stripping code fences if present."""
    if response_text is None:
        return None

    text = response_text.strip()

    # Strip ```json ... ``` code fences
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    # Try to find JSON object
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if json_match:
        text = json_match.group(0)

    try:
        scores = json.loads(text)
        # Validate required keys
        required = ["correctness", "novelty", "non_triviality", "significance",
                     "formalizability", "proof_quality", "overall_score"]
        for key in required:
            if key not in scores:
                log(f"  Missing key: {key}")
                return None
            scores[key] = float(scores[key])
        return scores
    except (json.JSONDecodeError, ValueError) as e:
        log(f"  JSON parse error: {e}")
        log(f"  Raw text: {text[:300]}")
        return None


def main():
    log("=" * 60)
    log("Re-judging Task 16 Zero-Shot Baseline with Mistral API")
    log("=" * 60)
    log(f"Model: {MISTRAL_MODEL}")
    log(f"Endpoint: {MISTRAL_ENDPOINT}")
    log("")

    conjectures = results["conjectures"]
    successful_indices = [i for i, c in enumerate(conjectures)
                          if c.get("statement") != "GENERATION_FAILED"]

    log(f"Total conjectures: {len(conjectures)}")
    log(f"Successful (to judge): {len(successful_indices)}")
    log("")

    judged_count = 0

    for idx in successful_indices:
        conj = conjectures[idx]
        domain = conj["domain"]
        statement = conj["statement"]

        log(f"Judging [{idx}] domain={domain}...")
        prompt = build_judge_prompt(domain, statement)
        response = call_mistral(prompt)

        if response:
            scores = parse_judge_response(response)
            if scores:
                conj["judge_scores"] = scores
                conj["judge_model"] = MISTRAL_MODEL
                judged_count += 1
                log(f"  overall_score={scores['overall_score']:.3f}, "
                      f"correctness={scores['correctness']:.2f}, "
                      f"novelty={scores['novelty']:.2f}, "
                      f"non_triviality={scores['non_triviality']:.2f}")
            else:
                log(f"  FAILED to parse response")
                conj["judge_scores"] = {}
        else:
            log(f"  FAILED to get API response")
            conj["judge_scores"] = {}

        # Delay between calls
        if idx != successful_indices[-1]:
            log(f"  Waiting {DELAY_BETWEEN_CALLS}s...")
            time.sleep(DELAY_BETWEEN_CALLS)

    log("")
    log("=" * 60)
    log("RECOMPUTING STATISTICS")
    log("=" * 60)

    # Update metadata
    results["judge"] = MISTRAL_MODEL
    results["judge_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Recompute per-domain statistics
    criteria = ["correctness", "novelty", "non_triviality", "significance",
                "formalizability", "proof_quality"]

    all_scores = []
    domain_scores = {}

    for conj in conjectures:
        domain = conj["domain"]
        if domain not in domain_scores:
            domain_scores[domain] = []

        scores = conj.get("judge_scores", {})
        if scores and "overall_score" in scores:
            domain_scores[domain].append(scores)
            all_scores.append(scores)

    # Update per_domain_results
    for domain in results["per_domain_results"]:
        dr = results["per_domain_results"][domain]
        ds = domain_scores.get(domain, [])

        if ds:
            dr["avg_quality"] = round(float(np.mean([s["overall_score"] for s in ds])), 3)
            dr["std_quality"] = round(float(np.std([s["overall_score"] for s in ds])), 3)
            for crit in criteria:
                dr[f"avg_{crit}"] = round(float(np.mean([s[crit] for s in ds])), 3)
        else:
            dr["avg_quality"] = 0.0
            dr["std_quality"] = 0.0
            for crit in criteria:
                dr[f"avg_{crit}"] = 0.0

    # Update overall statistics
    if all_scores:
        results["overall_avg_quality"] = round(float(np.mean([s["overall_score"] for s in all_scores])), 3)
        results["overall_std_quality"] = round(float(np.std([s["overall_score"] for s in all_scores])), 3)
        for crit in criteria:
            results[f"overall_avg_{crit}"] = round(float(np.mean([s[crit] for s in all_scores])), 3)
    else:
        results["overall_avg_quality"] = 0.0
        results["overall_std_quality"] = 0.0
        for crit in criteria:
            results[f"overall_avg_{crit}"] = 0.0

    # Save updated results
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    log(f"\nSaved updated results to {RESULTS_PATH}")

    # Print summary
    log("")
    log("=" * 60)
    log("SUMMARY: Zero-Shot Baseline (Task 16)")
    log("=" * 60)
    log(f"Judge model: {MISTRAL_MODEL}")
    log(f"Successfully judged: {judged_count}/{len(successful_indices)} conjectures")
    log("")

    if all_scores:
        log(f"Overall average quality: {results['overall_avg_quality']:.3f} (+/- {results['overall_std_quality']:.3f})")
        log("")
        log("Per-criteria averages:")
        for crit in criteria:
            log(f"  {crit:20s}: {results[f'overall_avg_{crit}']:.3f}")
        log("")

        log("Per-domain averages:")
        log(f"  {'Domain':<25s} {'Avg Quality':>12s} {'# Judged':>10s}")
        log(f"  {'-'*25} {'-'*12} {'-'*10}")
        for domain in results["domains"]:
            dr = results["per_domain_results"][domain]
            n_judged = len(domain_scores.get(domain, []))
            log(f"  {domain:<25s} {dr['avg_quality']:>12.3f} {n_judged:>10d}")
        log("")

        log("NOTE: This is the zero-shot baseline (no domain context, no pipeline).")
        log("      Compare with MoE pipeline conjectures (avg quality ~0.693) to measure improvement.")
    else:
        log("No conjectures were successfully judged.")


if __name__ == "__main__":
    main()
