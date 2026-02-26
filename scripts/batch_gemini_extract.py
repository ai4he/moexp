"""
Batch Gemini-powered Knowledge Extraction
Processes entries from the stratified sample through Gemini for structured extraction.

Usage:
    python scripts/batch_gemini_extract.py [--max-entries 100] [--batch-size 5]
"""

import json
import os
import sys
import re
import time
import argparse
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from llm_utils import gemini_generate, THEOREM_EXTRACTION_PROMPT
from data_utils import get_domain_group, SAMPLE_DIR

BASE_DIR = "/scratch/ctoxtli/moexp"
DATA_DIR = os.path.join(BASE_DIR, "data")


def fix_latex_json(text: str) -> str:
    """Fix unescaped LaTeX backslashes in JSON strings.

    Gemini sometimes produces \ell, \nabla, \frac etc. without proper JSON escaping.
    This converts unescaped LaTeX commands (backslash + 2+ letters) to double-escaped form.
    """
    # Match backslash followed by 2+ word chars that aren't valid JSON escapes
    # Valid JSON escapes: \n \t \r \b \f \\ \" \/ \uXXXX
    def _fix(m):
        seq = m.group(1)
        # Valid JSON escape sequences (single char after backslash)
        if seq in ('n', 't', 'r', 'b', 'f'):
            return m.group(0)  # Keep as-is
        return '\\\\' + seq

    return re.sub(r'\\([a-zA-Z]{2,})', r'\\\\\\1', text)


def parse_json_response(response: str) -> dict:
    """Parse JSON from Gemini response with multiple fallback strategies."""
    if not response or not response.strip():
        return {'raw_response': '', 'parse_error': True, 'error_type': 'empty'}

    clean = response.strip()

    # Strip markdown code fences
    if clean.startswith('```'):
        # Remove opening fence (```json or ```)
        first_nl = clean.find('\n')
        clean = clean[first_nl + 1:] if first_nl != -1 else clean[3:]
    if clean.endswith('```'):
        clean = clean[:clean.rfind('```')]
    clean = clean.strip()

    # Strategy 1: Direct parse
    try:
        result = json.loads(clean)
        # If Gemini returns a list, take the first dict element
        if isinstance(result, list) and result and isinstance(result[0], dict):
            return result[0]
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Strategy 2: Fix LaTeX backslashes then parse
    try:
        fixed = fix_latex_json(clean)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Strategy 3: Fix common brace issues (\{ and \} in JSON)
    try:
        fixed2 = fix_latex_json(clean)
        fixed2 = fixed2.replace('\\{', '\\\\{').replace('\\}', '\\\\}')
        return json.loads(fixed2)
    except json.JSONDecodeError:
        pass

    # Strategy 4: Extract first JSON object if response contains extra text
    try:
        # Find first { and last }
        start = clean.find('{')
        end = clean.rfind('}')
        if start != -1 and end > start:
            subset = clean[start:end+1]
            fixed = fix_latex_json(subset)
            return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    return {'raw_response': response[:2000], 'parse_error': True, 'error_type': 'json_decode'}


def extract_one(entry: dict, gemini_key: str) -> dict:
    """Extract structured knowledge from one entry using Gemini."""
    prompt = THEOREM_EXTRACTION_PROMPT.format(
        categories=entry.get('categories', 'unknown'),
        abstract=entry.get('abstract', 'N/A')[:500],
        text=entry.get('text', '')[:2000],
        previous_context=entry.get('previous context', '')[:500],
        following_context=entry.get('following context', '')[:500],
    )

    response = gemini_generate(
        prompt,
        model='gemini-2.0-flash',
        key=gemini_key,
        temperature=0.3,
        max_tokens=2048,
        json_mode=True,
    )

    return parse_json_response(response)


def batch_extract(
    sample_path: str,
    output_path: str,
    gemini_key: str,
    max_entries: int = 100,
    delay: float = 4.0,
):
    """Process entries through Gemini extraction with checkpointing."""
    # Load sample
    sample = []
    with open(sample_path) as f:
        for line in f:
            sample.append(json.loads(line))
    print(f"Loaded {len(sample)} entries from sample")

    # Resume from checkpoint
    already_done = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                e = json.loads(line)
                already_done.add(e.get('id', ''))
        print(f"Already processed: {len(already_done)} entries")

    to_process = [e for e in sample if e.get('id', '') not in already_done][:max_entries]
    print(f"Processing {len(to_process)} entries...")

    stats = Counter()
    consecutive_failures = 0
    with open(output_path, 'a') as outf:
        for i, entry in enumerate(to_process):
            try:
                extracted = extract_one(entry, gemini_key)

                output = {
                    'id': entry.get('id', ''),
                    'categories': entry.get('categories', ''),
                    'type': entry.get('type', ''),
                    'domain_group': get_domain_group(entry.get('categories', '')),
                    'original_text': entry.get('text', '')[:500],
                    'extracted': extracted,
                }
                outf.write(json.dumps(output) + '\n')
                outf.flush()

                if extracted.get('parse_error'):
                    stats['parse_error'] += 1
                    consecutive_failures += 1
                else:
                    stats['success'] += 1
                    consecutive_failures = 0

                domain = output['domain_group']
                stats[f'domain_{domain}'] += 1

                status = extracted.get('statement_type', '?')
                if extracted.get('parse_error'):
                    status = f"FAIL({extracted.get('error_type', '?')})"
                print(f"  [{i+1}/{len(to_process)}] {entry.get('id', '?')} "
                      f"({entry.get('type', '?')}): {status} - "
                      f"{extracted.get('mathematical_domain', '?')}")

            except Exception as e:
                print(f"  Error: {entry.get('id', '?')}: {e}")
                stats['error'] += 1
                consecutive_failures += 1

            # Adaptive delay: back off when seeing consecutive failures
            current_delay = delay
            if consecutive_failures >= 5:
                current_delay = delay * 3
                if consecutive_failures % 10 == 0:
                    print(f"  Warning: {consecutive_failures} consecutive failures, "
                          f"using {current_delay}s delay")
            elif consecutive_failures >= 2:
                current_delay = delay * 1.5
            time.sleep(current_delay)

    print(f"\n=== Extraction Complete ===")
    for k, v in stats.most_common():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Gemini knowledge extraction")
    parser.add_argument("--max-entries", type=int, default=100)
    parser.add_argument("--delay", type=float, default=4.0)
    parser.add_argument("--sample-path", default=os.path.join(SAMPLE_DIR, "stratified_sample_100.jsonl"))
    parser.add_argument("--output-path", default=os.path.join(DATA_DIR, "extracted_knowledge.jsonl"))
    args = parser.parse_args()

    # Load working Gemini key
    keys = json.load(open(os.path.join(BASE_DIR, "working_Gemini_API_keys.json")))
    gemini_key = keys[3]  # Key 4 (index 3) confirmed working

    os.makedirs(DATA_DIR, exist_ok=True)
    batch_extract(
        sample_path=args.sample_path,
        output_path=args.output_path,
        gemini_key=gemini_key,
        max_entries=args.max_entries,
        delay=args.delay,
    )
