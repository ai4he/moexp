import json
import random
from collections import Counter, defaultdict

# First pass: count categories and types
print("=== Scanning dataset for category distribution ===")
cat_counter = Counter()
type_counter = Counter()
env_counter = Counter()
total = 0

# Track unique primary categories
primary_cat_counter = Counter()

with open('/scratch/ctoxtli/moexp/final_data_dedup_6.5m.jsonl', 'r') as f:
    for i, line in enumerate(f):
        if i % 500000 == 0:
            print(f"  Processed {i:,} lines...", flush=True)
        try:
            entry = json.loads(line.strip())
            cats = entry.get('categories', '')
            # Categories can be space-separated like "math.CO cs.CG"
            for cat in cats.split():
                cat_counter[cat] += 1
            # Primary category is the first one
            if cats:
                primary_cat_counter[cats.split()[0]] += 1
            type_counter[entry.get('type', 'unknown')] += 1
            env_counter[entry.get('env', 'unknown')] += 1
            total += 1
        except:
            continue

print(f"\n=== Total entries: {total:,} ===")
print(f"\n=== Unique categories: {len(cat_counter)} ===")
print(f"=== Unique primary categories: {len(primary_cat_counter)} ===")

print("\n=== Top 50 categories (all) ===")
for cat, count in cat_counter.most_common(50):
    print(f"  {cat}: {count:,} ({100*count/total:.1f}%)")

print("\n=== Math-specific categories ===")
math_cats = {k: v for k, v in cat_counter.items() if k.startswith('math.')}
for cat, count in sorted(math_cats.items(), key=lambda x: -x[1]):
    print(f"  {cat}: {count:,} ({100*count/total:.1f}%)")

print(f"\n=== Entry types ===")
for t, count in type_counter.most_common(20):
    print(f"  {t}: {count:,}")

print(f"\n=== Environments ===")
for e, count in env_counter.most_common(20):
    print(f"  {e}: {count:,}")

# Save category stats
stats = {
    'total_entries': total,
    'categories': dict(cat_counter),
    'primary_categories': dict(primary_cat_counter),
    'types': dict(type_counter),
    'environments': dict(env_counter),
    'math_categories': math_cats
}
with open('/scratch/ctoxtli/moexp/dataset_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)
print("\nStats saved to /scratch/ctoxtli/moexp/dataset_stats.json")
