#!/usr/bin/env python3
"""Deep analysis of parse errors - empty responses and error patterns by entry type"""
import json
from collections import Counter

entries = []
with open('/scratch/ctoxtli/moexp/data/extracted_knowledge.jsonl') as f:
    for i, line in enumerate(f):
        obj = json.loads(line)
        ext = obj.get('extracted', {})
        is_error = isinstance(ext, dict) and ext.get('parse_error')
        raw = ext.get('raw_response', '') if is_error else ''
        entries.append({
            'line': i,
            'is_error': is_error,
            'is_empty': is_error and len(raw) == 0,
            'is_truncated': is_error and len(raw) == 500,
            'type': obj.get('type', 'unknown'),
            'domain': obj.get('domain_group', 'unknown'),
            'categories': obj.get('categories', 'unknown'),
            'text_len': len(obj.get('original_text', '')),
            'raw': raw,
        })

# Error rate by entry type
print("=== ERROR RATE BY ENTRY TYPE ===")
type_stats = {}
for e in entries:
    t = e['type']
    if t not in type_stats:
        type_stats[t] = {'total': 0, 'errors': 0, 'empty': 0, 'truncated': 0}
    type_stats[t]['total'] += 1
    if e['is_error']:
        type_stats[t]['errors'] += 1
    if e['is_empty']:
        type_stats[t]['empty'] += 1
    if e['is_truncated']:
        type_stats[t]['truncated'] += 1

for t, s in sorted(type_stats.items(), key=lambda x: -x[1]['total']):
    if s['total'] >= 5:
        print(f"  {t:20s}: {s['errors']:4d}/{s['total']:4d} errors ({100*s['errors']/s['total']:5.1f}%) "
              f"[empty={s['empty']}, trunc={s['truncated']}]")

# Error rate by domain
print("\n=== ERROR RATE BY DOMAIN ===")
domain_stats = {}
for e in entries:
    d = e['domain']
    if d not in domain_stats:
        domain_stats[d] = {'total': 0, 'errors': 0, 'empty': 0, 'truncated': 0}
    domain_stats[d]['total'] += 1
    if e['is_error']:
        domain_stats[d]['errors'] += 1
    if e['is_empty']:
        domain_stats[d]['empty'] += 1
    if e['is_truncated']:
        domain_stats[d]['truncated'] += 1

for d, s in sorted(domain_stats.items(), key=lambda x: -x[1]['total']):
    print(f"  {d:30s}: {s['errors']:4d}/{s['total']:4d} errors ({100*s['errors']/s['total']:5.1f}%) "
          f"[empty={s['empty']}, trunc={s['truncated']}]")

# Check if empty responses are clustered (batch failures?)
print("\n=== EMPTY RESPONSE DISTRIBUTION ===")
empty_indices = [e['line'] for e in entries if e['is_empty']]
if empty_indices:
    # Break into contiguous runs
    runs = []
    current_run = [empty_indices[0]]
    for idx in empty_indices[1:]:
        if idx == current_run[-1] + 1:
            current_run.append(idx)
        else:
            runs.append(current_run)
            current_run = [idx]
    runs.append(current_run)

    print(f"Number of contiguous runs of empty errors: {len(runs)}")
    print(f"Run length distribution:")
    run_lens = [len(r) for r in runs]
    for length in sorted(set(run_lens), reverse=True)[:10]:
        count = run_lens.count(length)
        print(f"  Length {length}: {count} runs")

    print(f"\nLargest runs (start_line, length):")
    for r in sorted(runs, key=len, reverse=True)[:5]:
        print(f"  Lines {r[0]}-{r[-1]} (length {len(r)})")

# Check original_text length distribution for errors vs successes
print("\n=== ORIGINAL TEXT LENGTH: ERRORS vs SUCCESSES ===")
error_text_lens = [e['text_len'] for e in entries if e['is_error'] and not e['is_empty']]
success_text_lens = [e['text_len'] for e in entries if not e['is_error']]
empty_text_lens = [e['text_len'] for e in entries if e['is_empty']]

if error_text_lens:
    print(f"Truncated errors text len: mean={sum(error_text_lens)/len(error_text_lens):.0f}, "
          f"median={sorted(error_text_lens)[len(error_text_lens)//2]}")
if success_text_lens:
    print(f"Successes text len:        mean={sum(success_text_lens)/len(success_text_lens):.0f}, "
          f"median={sorted(success_text_lens)[len(success_text_lens)//2]}")
if empty_text_lens:
    print(f"Empty errors text len:     mean={sum(empty_text_lens)/len(empty_text_lens):.0f}, "
          f"median={sorted(empty_text_lens)[len(empty_text_lens)//2]}")

# Check if markdown stripping in batch_gemini_extract.py is the issue
# Simulate the current parsing logic on truncated responses
print("\n=== SIMULATING CURRENT PARSER ON TRUNCATED RESPONSES ===")
truncated = [e for e in entries if e['is_truncated']]
for e in truncated[:5]:
    raw = e['raw']
    print(f"\nLine {e['line']} (type={e['type']}, domain={e['domain']}):")
    print(f"  Raw starts with: {repr(raw[:80])}")

    clean = raw.strip()
    if clean.startswith('```'):
        clean = clean.split('\n', 1)[1] if '\n' in clean else clean[3:]
        print(f"  After strip opening fence: {repr(clean[:60])}...")
    if clean.endswith('```'):
        clean = clean.rsplit('```', 1)[0]
        print(f"  After strip closing fence (no change - truncated, no closing fence)")
    else:
        print(f"  No closing fence found (truncated)")
    print(f"  Ends with: ...{repr(clean[-60:])}")
    try:
        json.loads(clean)
        print(f"  JSON parse: SUCCESS (unexpected!)")
    except json.JSONDecodeError as ex:
        print(f"  JSON parse error: {ex.msg} at pos {ex.pos}")

# Check how many truncated entries have the opening ```json pattern
print("\n=== MARKDOWN FENCE PATTERNS IN TRUNCATED RESPONSES ===")
fence_patterns = Counter()
for e in truncated:
    raw = e['raw']
    if raw.startswith('```json\n'):
        fence_patterns['```json\\n'] += 1
    elif raw.startswith('```json'):
        fence_patterns['```json (no newline)'] += 1
    elif raw.startswith('```\n'):
        fence_patterns['```\\n'] += 1
    elif raw.startswith('```'):
        fence_patterns['``` (other)'] += 1
    else:
        fence_patterns['no fence'] += 1

for pattern, count in fence_patterns.most_common():
    print(f"  {pattern}: {count}")
