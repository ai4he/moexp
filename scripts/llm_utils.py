"""
LLM Utility functions for MathScy project.
Supports Gemini API and HPC LLM API (OpenAI-compatible).
"""

import json
import time
import requests
import os
import random
from typing import List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===== Gemini API =====

GEMINI_KEYS_PATH = "/scratch/ctoxtli/moexp/working_Gemini_API_keys.json"

def load_gemini_keys() -> List[str]:
    with open(GEMINI_KEYS_PATH) as f:
        return json.load(f)

# Key rotation state
_gemini_key_idx = 0
_gemini_keys = None

def get_next_gemini_key() -> str:
    """Round-robin key rotation."""
    global _gemini_key_idx, _gemini_keys
    if _gemini_keys is None:
        _gemini_keys = load_gemini_keys()
    key = _gemini_keys[_gemini_key_idx % len(_gemini_keys)]
    _gemini_key_idx += 1
    return key

def gemini_generate(
    prompt: str,
    model: str = "gemini-2.0-flash",
    key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    max_retries: int = 6,
    system_instruction: Optional[str] = None,
    **kwargs,
) -> str:
    """Call Gemini API with retry and key rotation."""
    if key is None:
        key = get_next_gemini_key()

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
    }

    # Allow caller to request structured JSON output
    if kwargs.get("json_mode"):
        payload["generationConfig"]["responseMimeType"] = "application/json"

    if system_instruction:
        payload["systemInstruction"] = {
            "parts": [{"text": system_instruction}]
        }

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, timeout=60)
            if resp.status_code == 200:
                result = resp.json()
                candidates = result.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        text = parts[0].get("text", "")
                        if text:
                            return text
                # Empty response - likely transient API issue, retry with backoff
                if attempt < max_retries - 1:
                    wait = min(2 ** attempt * 2, 30)
                    print(f"  Empty response, retrying in {wait}s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait)
                    continue
                return ""
            elif resp.status_code == 429:
                # Rate limited - try another key
                wait = min(2 ** attempt, 60)
                print(f"  Rate limited on key, waiting {wait}s and trying next key...")
                time.sleep(wait)
                key = get_next_gemini_key()
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
            elif resp.status_code == 404:
                print(f"  Model {model} not available with this key, trying next...")
                key = get_next_gemini_key()
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
            else:
                print(f"  Gemini error {resp.status_code}: {resp.text[:200]}")
                time.sleep(2 ** attempt)
        except Exception as e:
            print(f"  Gemini request error: {e}")
            time.sleep(2 ** attempt)

    return ""

def gemini_batch_generate(
    prompts: List[str],
    model: str = "gemini-2.0-flash",
    max_workers: int = 3,
    delay_between: float = 0.5,
    **kwargs,
) -> List[str]:
    """Process multiple prompts with parallel execution."""
    results = [""] * len(prompts)

    def process_one(idx_prompt):
        idx, prompt = idx_prompt
        time.sleep(idx * delay_between)  # Stagger requests
        return idx, gemini_generate(prompt, model=model, **kwargs)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_one, (i, p)): i for i, p in enumerate(prompts)}
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result

    return results


# ===== HPC LLM API (OpenAI-compatible) =====

HPC_BASE_URL = "https://llm.rcd.clemson.edu/v1"
HPC_API_KEY = os.environ.get("HPC_LLM_API_KEY", "")

def set_hpc_api_key(key: str):
    """Set the HPC LLM API key."""
    global HPC_API_KEY
    HPC_API_KEY = key
    os.environ["HPC_LLM_API_KEY"] = key

def hpc_chat_completion(
    messages: List[Dict[str, str]],
    model: str = "qwen3-30b-a3b-instruct-fp8",
    temperature: float = 0.7,
    max_tokens: int = 4096,
    max_retries: int = 3,
) -> str:
    """Call HPC LLM API (OpenAI-compatible)."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {HPC_API_KEY}",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{HPC_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,
            )
            if resp.status_code == 200:
                result = resp.json()
                return result["choices"][0]["message"]["content"]
            elif resp.status_code == 429:
                wait = min(2 ** attempt * 5, 60)
                print(f"  HPC rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  HPC error {resp.status_code}: {resp.text[:200]}")
                time.sleep(2 ** attempt)
        except Exception as e:
            print(f"  HPC request error: {e}")
            time.sleep(2 ** attempt)

    return ""

def hpc_generate(prompt: str, model: str = "qwen3-30b-a3b-instruct-fp8", **kwargs) -> str:
    """Convenience wrapper for simple prompt -> response."""
    messages = [{"role": "user", "content": prompt}]
    return hpc_chat_completion(messages, model=model, **kwargs)

def hpc_embed(texts: List[str], model: str = "qwen3-embedding-4b") -> List[List[float]]:
    """Get embeddings from HPC LLM API."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {HPC_API_KEY}",
    }

    payload = {
        "model": model,
        "input": texts,
    }

    try:
        resp = requests.post(
            f"{HPC_BASE_URL}/embeddings",
            headers=headers,
            json=payload,
            timeout=120,
        )
        if resp.status_code == 200:
            result = resp.json()
            return [d["embedding"] for d in result["data"]]
        else:
            print(f"  Embedding error {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"  Embedding request error: {e}")

    return []


# ===== Math theorem extraction prompts =====

THEOREM_EXTRACTION_PROMPT = """You are a mathematical knowledge extraction system. Given the following LaTeX content from a math paper, extract structured information.

Paper category: {categories}
Paper abstract: {abstract}

Content to analyze:
{text}

Context before:
{previous_context}

Context after:
{following_context}

Extract and return a JSON object with the following fields:
{{
    "statement_type": "theorem|lemma|proposition|corollary|conjecture|definition",
    "formal_statement": "the precise mathematical statement in clean LaTeX",
    "informal_description": "a plain English description of what the statement says",
    "key_concepts": ["list", "of", "key", "mathematical", "concepts"],
    "mathematical_domain": "primary math domain (e.g., algebraic geometry, combinatorics, etc.)",
    "prerequisites": ["mathematical concepts needed to understand this"],
    "related_theorems": ["names of related theorems if mentioned"],
    "proof_sketch": "brief description of proof approach if available",
    "potential_generalizations": "possible ways to generalize this result",
    "lean4_formalization_difficulty": "easy|medium|hard|very_hard"
}}

Return ONLY valid JSON, no markdown formatting."""


CONJECTURE_GENERATION_PROMPT = """You are a creative mathematical researcher specializing in {domain}.

Based on the following collection of theorems and results from recent papers in {domain}, generate novel conjectures that could potentially be true.

Known results:
{known_results}

Guidelines:
1. Look for patterns across the known results
2. Consider generalizations of existing theorems
3. Consider analogies from related mathematical domains
4. Ensure conjectures are non-trivial but plausible
5. Each conjecture should be precise enough to be formalized in Lean 4

Generate 3-5 novel conjectures. For each conjecture provide:
{{
    "conjecture_statement": "precise mathematical statement in LaTeX",
    "informal_description": "plain English explanation",
    "motivation": "why this conjecture might be true, based on which known results",
    "related_known_results": ["which known results inspire this"],
    "estimated_difficulty": "easy|medium|hard|open_problem",
    "verification_approach": "how one might try to prove or disprove this",
    "lean4_sketch": "approximate Lean 4 formalization (best effort)"
}}

Return a JSON array of conjecture objects."""


if __name__ == "__main__":
    # Quick test
    print("Testing Gemini API...")
    result = gemini_generate("What is the Fundamental Theorem of Algebra? Reply in one sentence.")
    print(f"Gemini response: {result[:200]}")

    print("\nGemini API utility module loaded successfully.")
