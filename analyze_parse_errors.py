#!/usr/bin/env python3
"""Analyze parse errors in extracted_knowledge.jsonl"""
import json
import re
from collections import Counter

errors = []
successes = []
with open('/scratch/ctoxtli/moexp/data/extracted_knowledge.jsonl') as f:
    for i, line in enumerate(f):
        obj = json.loads(line)
        ext = obj.get('extracted', {})
        if isinstance(ext, dict) and ext.get('parse_error'):
            raw = ext.get('raw_response', '')
            errors.append({'line': i, 'raw': raw, 'raw_len': len(raw)})
        else:
            successes.append({'line': i, 'ext': ext})

print(f"Total entries: {len(errors) + len(successes)}")
print(f"Successes: {len(successes)}")
print(f"Errors: {len(errors)}")
print()

# Split errors
truncated = [e for e in errors if len(e['raw']) > 0]
empty = [e for e in errors if len(e['raw']) == 0]

print(f"=== ERROR BREAKDOWN ===")
print(f"Empty responses: {len(empty)} ({100*len(empty)/len(errors):.1f}% of errors)")
print(f"Truncated at 500 chars: {len(truncated)} ({100*len(truncated)/len(errors):.1f}% of errors)")
print()

# Analyze empty responses - look at context
print("=== EMPTY RESPONSE ANALYSIS ===")
empty_lines = set(e['line'] for e in empty)
# Check if empty responses cluster together
empty_sorted = sorted(e['line'] for e in empty)
gaps = []
for i in range(1, len(empty_sorted)):
    gaps.append(empty_sorted[i] - empty_sorted[i-1])
if gaps:
    print(f"Line range: {empty_sorted[0]} to {empty_sorted[-1]}")
    print(f"Median gap between empty errors: {sorted(gaps)[len(gaps)//2]}")
    # Check for consecutive runs
    runs = 1
    max_run = 1
    cur_run = 1
    for i in range(1, len(empty_sorted)):
        if empty_sorted[i] == empty_sorted[i-1] + 1:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 1
    print(f"Max consecutive empty errors: {max_run}")
print()

# Analyze truncated - recovery attempts
print("=== TRUNCATED RESPONSE RECOVERY ANALYSIS ===")

def try_recover(raw):
    """Try to recover valid JSON from truncated markdown-wrapped response"""
    # Strip markdown fences
    cleaned = raw.strip()
    if cleaned.startswith('```'):
        # Remove opening fence
        first_newline = cleaned.find('\n')
        if first_newline >= 0:
            cleaned = cleaned[first_newline+1:]
        else:
            return None, "no newline after fence"

    # Remove closing fence if present
    if cleaned.rstrip().endswith('```'):
        cleaned = cleaned[:cleaned.rstrip().rfind('```')]

    cleaned = cleaned.strip()

    # Try direct parse
    try:
        return json.loads(cleaned), "direct"
    except:
        pass

    # Try to close the JSON
    attempt = cleaned

    # Count unmatched quotes (rough)
    in_string = False
    escaped = False
    for ch in attempt:
        if escaped:
            escaped = False
            continue
        if ch == '\\':
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string

    if in_string:
        attempt += '"'

    # Close brackets and braces
    open_brackets = attempt.count('[') - attempt.count(']')
    open_braces = attempt.count('{') - attempt.count('}')

    attempt += ']' * max(0, open_brackets)
    attempt += '}' * max(0, open_braces)

    try:
        return json.loads(attempt), "bracket_close"
    except:
        pass

    # Try more aggressive: find last complete key-value pair
    # Look for last comma or last complete value
    for trim_amount in range(1, min(100, len(cleaned))):
        trimmed = cleaned[:-trim_amount].rstrip().rstrip(',')

        open_brackets = trimmed.count('[') - trimmed.count(']')
        open_braces = trimmed.count('{') - trimmed.count('}')

        in_str = False
        esc = False
        for ch in trimmed:
            if esc:
                esc = False
                continue
            if ch == '\\':
                esc = True
                continue
            if ch == '"':
                in_str = not in_str

        if in_str:
            trimmed += '"'

        trimmed += ']' * max(0, open_brackets)
        trimmed += '}' * max(0, open_braces)

        try:
            result = json.loads(trimmed)
            return result, f"trimmed_{trim_amount}"
        except:
            continue

    return None, "unrecoverable"

recovered = 0
recovery_methods = Counter()
recovered_fields = Counter()
unrecoverable_samples = []

for e in truncated:
    result, method = try_recover(e['raw'])
    if result is not None:
        recovered += 1
        recovery_methods[method] += 1
        for key in result:
            recovered_fields[key] += 1
    else:
        unrecoverable_samples.append(e)

print(f"Recovered: {recovered}/{len(truncated)} ({100*recovered/len(truncated):.1f}%)")
print(f"\nRecovery methods:")
for method, count in recovery_methods.most_common():
    print(f"  {method}: {count}")

print(f"\nFields present in recovered entries:")
for field, count in recovered_fields.most_common():
    print(f"  {field}: {count}/{recovered} ({100*count/recovered:.1f}%)")

print(f"\nUnrecoverable: {len(unrecoverable_samples)}")
if unrecoverable_samples:
    print("Sample unrecoverable (first 3):")
    for e in unrecoverable_samples[:3]:
        print(f"  Line {e['line']}: {e['raw'][:200]}...")
print()

# Analyze what fields are MISSING in recovered entries
expected_fields = ['statement_type', 'formal_statement', 'informal_description',
                   'key_concepts', 'mathematical_domain', 'prerequisites',
                   'related_theorems', 'proof_sketch']
print("=== FIELD COMPLETENESS IN RECOVERED ENTRIES ===")
for field in expected_fields:
    count = recovered_fields.get(field, 0)
    print(f"  {field}: {count}/{recovered} ({100*count/recovered:.1f}%)")

# Compare with successful entries
print("\n=== SUCCESSFUL ENTRY FIELD ANALYSIS ===")
success_fields = Counter()
for s in successes:
    ext = s['ext']
    if isinstance(ext, dict):
        for key in ext:
            success_fields[key] += 1

for field in expected_fields:
    count = success_fields.get(field, 0)
    print(f"  {field}: {count}/{len(successes)} ({100*count/len(successes):.1f}%)")

# Overall summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
total = len(errors) + len(successes)
print(f"Total entries: {total}")
print(f"Already successful: {len(successes)} ({100*len(successes)/total:.1f}%)")
print(f"Empty responses (API failures): {len(empty)} ({100*len(empty)/total:.1f}%)")
print(f"Truncated at 500 chars: {len(truncated)} ({100*len(truncated)/total:.1f}%)")
print(f"  - Recoverable from truncated: {recovered} ({100*recovered/total:.1f}%)")
print(f"  - Unrecoverable from truncated: {len(truncated)-recovered} ({100*(len(truncated)-recovered)/total:.1f}%)")
print(f"\nPotential success rate after fixes: {len(successes)+recovered}/{total} = {100*(len(successes)+recovered)/total:.1f}%")
print(f"Remaining failures (need re-extraction): {len(empty) + len(truncated) - recovered}/{total} = {100*(len(empty) + len(truncated) - recovered)/total:.1f}%")
