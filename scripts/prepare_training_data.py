"""
Prepare unified MoE training data from all extraction sources.
Combines Gemini structured extraction + ArXiv raw theorems into
domain-labeled training examples for each MoE expert.

Usage:
    python scripts/prepare_training_data.py
"""

import json
import os
import sys
import random
from collections import Counter, defaultdict
from typing import List, Dict

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import get_domain_group, get_primary_category, MATH_DOMAIN_GROUPS

BASE_DIR = "/scratch/ctoxtli/moexp"
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

EXPERT_DOMAINS = [
    "algebraic_geometry", "discrete_math", "number_theory",
    "analysis", "algebra", "geometry_topology",
    "probability_statistics", "computational",
]


def load_gemini_extractions() -> List[Dict]:
    """Load Gemini-extracted structured knowledge."""
    path = os.path.join(DATA_DIR, "extracted_knowledge.jsonl")
    if not os.path.exists(path):
        return []
    entries = []
    with open(path) as f:
        for line in f:
            e = json.loads(line)
            ext = e.get("extracted", {})
            # Handle cases where extracted is a list (take first element)
            if isinstance(ext, list):
                ext = ext[0] if ext and isinstance(ext[0], dict) else {}
                e["extracted"] = ext
            if not ext.get("parse_error"):
                entries.append(e)
    return entries


def load_arxiv_theorems() -> List[Dict]:
    """Load ArXiv-extracted raw theorems."""
    path = os.path.join(RESULTS_DIR, "arxiv_extracted_theorems.jsonl")
    if not os.path.exists(path):
        return []
    entries = []
    with open(path) as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


def load_sample_metadata() -> Dict[str, Dict]:
    """Load paper metadata from stratified sample for domain labeling."""
    path = os.path.join(DATA_DIR, "stratified_sample_100.jsonl")
    if not os.path.exists(path):
        return {}
    metadata = {}
    with open(path) as f:
        for line in f:
            e = json.loads(line)
            pid = e.get("id", "")
            if pid and pid not in metadata:
                metadata[pid] = {
                    "categories": e.get("categories", ""),
                    "abstract": e.get("abstract", ""),
                    "domain": get_domain_group(e.get("categories", "")),
                }
    return metadata


def create_training_examples(
    gemini_entries: List[Dict],
    arxiv_theorems: List[Dict],
    sample_metadata: Dict[str, Dict],
) -> Dict[str, List[Dict]]:
    """Create training examples organized by domain."""
    domain_examples = defaultdict(list)

    # From Gemini extractions
    for entry in gemini_entries:
        domain = entry.get("domain_group", "other")
        extracted = entry.get("extracted", {})

        # Task: Theorem understanding
        if extracted.get("formal_statement"):
            domain_examples[domain].append({
                "task": "theorem_understanding",
                "instruction": (
                    f"Explain the following mathematical statement in plain language:\n\n"
                    f"{extracted['formal_statement']}"
                ),
                "response": extracted.get("informal_description", ""),
                "domain": domain,
                "source": "gemini_extraction",
            })

        # Task: Concept identification
        if extracted.get("key_concepts"):
            domain_examples[domain].append({
                "task": "concept_identification",
                "instruction": (
                    f"Identify the key mathematical concepts in the following:\n\n"
                    f"{entry.get('original_text', '')[:500]}"
                ),
                "response": (
                    f"Key concepts: {', '.join(extracted['key_concepts'])}.\n"
                    f"Domain: {extracted.get('mathematical_domain', domain)}.\n"
                    f"Prerequisites: {', '.join(extracted.get('prerequisites', []))}."
                ),
                "domain": domain,
                "source": "gemini_extraction",
            })

        # Task: Generalization
        if extracted.get("potential_generalizations"):
            domain_examples[domain].append({
                "task": "generalization",
                "instruction": (
                    f"Suggest possible generalizations of the following result:\n\n"
                    f"{extracted.get('formal_statement', entry.get('original_text', '')[:500])}"
                ),
                "response": extracted["potential_generalizations"],
                "domain": domain,
                "source": "gemini_extraction",
            })

    # From ArXiv raw theorems
    for thm in arxiv_theorems:
        paper_id = thm.get("paper_id", "")
        meta = sample_metadata.get(paper_id, {})
        domain = meta.get("domain", "other")

        thm_type = thm.get("type", "theorem")
        body = thm.get("body", "").strip()
        if not body or len(body) < 10:
            continue

        # Task: Statement classification
        domain_examples[domain].append({
            "task": "statement_classification",
            "instruction": (
                f"Classify the following mathematical statement and identify its domain:\n\n"
                f"\\begin{{{thm_type}}}{body[:500]}\\end{{{thm_type}}}"
            ),
            "response": (
                f"This is a {thm_type} in {domain.replace('_', ' ')}."
            ),
            "domain": domain,
            "source": "arxiv_extraction",
        })

        # Task: Context understanding (if context available)
        if thm.get("context_before") or thm.get("context_after"):
            domain_examples[domain].append({
                "task": "contextualize",
                "instruction": (
                    f"Given the following mathematical result, explain its context:\n\n"
                    f"{body[:500]}"
                ),
                "response": (
                    f"Context before: {thm.get('context_before', '')[:200]}\n"
                    f"Context after: {thm.get('context_after', '')[:200]}"
                ),
                "domain": domain,
                "source": "arxiv_extraction",
            })

    return domain_examples


def save_training_data(domain_examples: Dict[str, List[Dict]]):
    """Save training data in multiple formats."""
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. Combined file
    all_examples = []
    for domain, examples in domain_examples.items():
        all_examples.extend(examples)

    combined_path = os.path.join(DATA_DIR, "moe_training_combined.jsonl")
    random.shuffle(all_examples)
    with open(combined_path, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Combined: {len(all_examples)} examples -> {combined_path}")

    # 2. Per-domain files (for Branch-Train-Mix)
    # Write files for all domains that have data (not just EXPERT_DOMAINS)
    for domain, examples in sorted(domain_examples.items()):
        if not examples:
            continue
        domain_path = os.path.join(DATA_DIR, f"domain_{domain}.jsonl")
        with open(domain_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        marker = " *" if domain in EXPERT_DOMAINS else ""
        print(f"  {domain}: {len(examples)} examples -> {domain_path}{marker}")

    # 3. Train/val split (90/10)
    random.seed(42)
    random.shuffle(all_examples)
    split_idx = int(len(all_examples) * 0.9)
    train = all_examples[:split_idx]
    val = all_examples[split_idx:]

    for name, data in [("train", train), ("val", val)]:
        path = os.path.join(DATA_DIR, f"moe_{name}.jsonl")
        with open(path, "w") as f:
            for ex in data:
                f.write(json.dumps(ex) + "\n")
        print(f"  {name}: {len(data)} examples -> {path}")

    # Stats summary
    stats = {
        "total_examples": len(all_examples),
        "train_size": len(train),
        "val_size": len(val),
        "domains": {d: len(exs) for d, exs in domain_examples.items()},
        "tasks": dict(Counter(ex["task"] for ex in all_examples).most_common()),
        "sources": dict(Counter(ex["source"] for ex in all_examples).most_common()),
    }
    stats_path = os.path.join(DATA_DIR, "training_data_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to {stats_path}")
    print(json.dumps(stats, indent=2))

    return stats


if __name__ == "__main__":
    print("Loading extraction results...")
    gemini = load_gemini_extractions()
    arxiv = load_arxiv_theorems()
    metadata = load_sample_metadata()

    print(f"Gemini extractions: {len(gemini)}")
    print(f"ArXiv theorems: {len(arxiv)}")
    print(f"Sample metadata: {len(metadata)} papers")

    print("\nCreating training examples...")
    domain_examples = create_training_examples(gemini, arxiv, metadata)

    print("\nSaving training data...")
    stats = save_training_data(domain_examples)
