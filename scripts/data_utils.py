"""
Data utilities for MathScy project.
Handles dataset loading, stratified sampling, and ArXiv paper downloading.
"""

import json
import os
import random
import tarfile
import tempfile
import time
import re
import gzip
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import requests

DATA_PATH = "/scratch/ctoxtli/moexp/final_data_dedup_6.5m.jsonl"
SAMPLE_DIR = "/scratch/ctoxtli/moexp/data"
CACHE_DIR = "/scratch/ctoxtli/moexp/cache"

# Math category groupings for MoE expert assignment
# Each group will correspond to one or more MoE experts
MATH_DOMAIN_GROUPS = {
    "algebraic_geometry": ["math.AG"],
    "combinatorics": ["math.CO"],
    "number_theory": ["math.NT"],
    "analysis": ["math.AP", "math.CA", "math.FA", "math.CV", "math.SP"],
    "algebra": ["math.AC", "math.RA", "math.GR", "math.RT"],
    "geometry_topology": ["math.GT", "math.DG", "math.GN", "math.AT", "math.MG", "math.SG"],
    "probability_statistics": ["math.PR", "math.ST"],
    "logic_foundations": ["math.LO", "math.CT"],
    "optimization_control": ["math.OC", "math.NA"],
    "dynamical_systems": ["math.DS"],
    "differential_equations": ["math.AP"],
    "mathematical_physics": ["math-ph", "math.MP"],
    "discrete_math": ["math.CO", "cs.DM"],
    "computational": ["cs.CG", "cs.SC", "cs.CC", "cs.LO"],
    "quantum_info": ["math.QA", "quant-ph"],
}

# Reverse mapping: category -> domain group
CAT_TO_DOMAIN = {}
for domain, cats in MATH_DOMAIN_GROUPS.items():
    for cat in cats:
        CAT_TO_DOMAIN[cat] = domain


def get_primary_category(categories: str) -> str:
    """Extract the primary (first) category from space-separated categories."""
    if not categories:
        return "unknown"
    return categories.split()[0]


def get_domain_group(categories: str) -> str:
    """Map categories to a domain group for MoE expert assignment."""
    primary = get_primary_category(categories)
    return CAT_TO_DOMAIN.get(primary, "other")


def scan_dataset_categories(max_lines: int = None) -> Dict:
    """Scan the full dataset and return category statistics."""
    cat_counter = Counter()
    primary_cat_counter = Counter()
    type_counter = Counter()
    domain_counter = Counter()
    total = 0

    with open(DATA_PATH, 'r') as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            if i % 1000000 == 0:
                print(f"  Scanned {i:,} lines...")
            try:
                entry = json.loads(line.strip())
                cats = entry.get('categories', '')
                for cat in cats.split():
                    cat_counter[cat] += 1
                primary = get_primary_category(cats)
                primary_cat_counter[primary] += 1
                domain_counter[get_domain_group(cats)] += 1
                type_counter[entry.get('type', 'unknown')] += 1
                total += 1
            except:
                continue

    return {
        'total': total,
        'categories': dict(cat_counter),
        'primary_categories': dict(primary_cat_counter),
        'types': dict(type_counter),
        'domain_groups': dict(domain_counter),
    }


def create_stratified_sample(
    n_per_category: int = 50,
    target_categories: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    seed: int = 42,
) -> List[Dict]:
    """
    Create a stratified sample from the dataset.
    Samples n_per_category entries from each primary math category.
    """
    random.seed(seed)

    # Reservoir sampling per category
    reservoirs = defaultdict(list)
    counts = Counter()

    print("Creating stratified sample...")
    with open(DATA_PATH, 'r') as f:
        for i, line in enumerate(f):
            if i % 1000000 == 0:
                print(f"  Processed {i:,} lines...")
            try:
                entry = json.loads(line.strip())
                primary_cat = get_primary_category(entry.get('categories', ''))

                # Filter to target categories if specified
                if target_categories and primary_cat not in target_categories:
                    continue

                # Only include math-related categories
                if not (primary_cat.startswith('math.') or primary_cat in ['math-ph', 'cs.CG', 'cs.SC', 'cs.LO']):
                    continue

                counts[primary_cat] += 1
                n = counts[primary_cat]

                if n <= n_per_category:
                    reservoirs[primary_cat].append(entry)
                else:
                    # Reservoir sampling
                    j = random.randint(0, n - 1)
                    if j < n_per_category:
                        reservoirs[primary_cat][j] = entry
            except:
                continue

    # Combine all samples
    sample = []
    for cat, entries in sorted(reservoirs.items()):
        print(f"  {cat}: {len(entries)} entries (from {counts[cat]:,} total)")
        sample.extend(entries)

    print(f"\nTotal sample size: {len(sample)} entries from {len(reservoirs)} categories")

    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            for entry in sample:
                f.write(json.dumps(entry) + '\n')
        print(f"Sample saved to {output_path}")

    return sample


def download_arxiv_source(arxiv_id: str, output_dir: str = None) -> Optional[str]:
    """
    Download LaTeX source for an ArXiv paper.
    arxiv_id format: "0704_0002" -> converts to "0704.0002"
    Returns path to extracted source directory or None on failure.
    """
    if output_dir is None:
        output_dir = os.path.join(CACHE_DIR, "arxiv_sources")
    os.makedirs(output_dir, exist_ok=True)

    # Convert underscore format to dot format
    arxiv_id_dot = arxiv_id.replace('_', '.')

    # Check if already downloaded
    source_dir = os.path.join(output_dir, arxiv_id)
    if os.path.exists(source_dir) and os.listdir(source_dir):
        return source_dir

    # Try downloading from ArXiv e-print API
    url = f"https://arxiv.org/e-print/{arxiv_id_dot}"

    try:
        resp = requests.get(url, timeout=30, headers={
            'User-Agent': 'MathScy Research Project (ctoxtli@clemson.edu)'
        })

        if resp.status_code == 200:
            os.makedirs(source_dir, exist_ok=True)

            content_type = resp.headers.get('content-type', '')

            if 'application/x-eprint-tar' in content_type or 'gzip' in content_type:
                # Tar.gz file
                with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
                    tmp.write(resp.content)
                    tmp_path = tmp.name

                try:
                    with tarfile.open(tmp_path, 'r:gz') as tar:
                        tar.extractall(source_dir)
                except:
                    # Maybe it's just gzip'd single file
                    try:
                        import gzip
                        with gzip.open(tmp_path, 'rb') as gz:
                            content = gz.read()
                        with open(os.path.join(source_dir, 'main.tex'), 'wb') as f:
                            f.write(content)
                    except:
                        pass

                os.unlink(tmp_path)
            else:
                # Plain text (single .tex file)
                with open(os.path.join(source_dir, 'main.tex'), 'wb') as f:
                    f.write(resp.content)

            return source_dir
        elif resp.status_code == 429:
            print(f"  Rate limited downloading {arxiv_id_dot}, waiting...")
            time.sleep(5)
            return None
        else:
            print(f"  Failed to download {arxiv_id_dot}: HTTP {resp.status_code}")
            return None

    except Exception as e:
        print(f"  Error downloading {arxiv_id_dot}: {e}")
        return None


def extract_latex_from_source(source_dir: str) -> Optional[str]:
    """Extract the main LaTeX content from a downloaded ArXiv source."""
    if not source_dir or not os.path.exists(source_dir):
        return None

    # Look for main .tex file
    tex_files = []
    for root, dirs, files in os.walk(source_dir):
        for f in files:
            if f.endswith('.tex'):
                tex_files.append(os.path.join(root, f))

    if not tex_files:
        return None

    # Heuristic: prefer files with 'main' in name, or the largest .tex file
    main_file = None
    for tf in tex_files:
        basename = os.path.basename(tf).lower()
        if 'main' in basename or 'paper' in basename:
            main_file = tf
            break

    if main_file is None:
        # Use the largest .tex file
        main_file = max(tex_files, key=os.path.getsize)

    try:
        with open(main_file, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except:
        return None


def extract_theorems_from_latex(latex_content: str) -> List[Dict]:
    """
    Extract theorem-like environments from raw LaTeX content.
    Returns list of dicts with type, label, content, context.
    """
    if not latex_content:
        return []

    theorems = []

    # Pattern to match theorem-like environments
    env_names = [
        'theorem', 'lemma', 'proposition', 'corollary', 'conjecture',
        'definition', 'remark', 'example', 'claim', 'fact',
    ]

    for env in env_names:
        # Match \begin{env}[optional title]...\end{env}
        pattern = rf'\\begin\{{{env}\}}(\[.*?\])?(.*?)\\end\{{{env}\}}'
        matches = re.finditer(pattern, latex_content, re.DOTALL)

        for match in matches:
            title = match.group(1) or ""
            body = match.group(2).strip()

            # Get surrounding context (200 chars before and after)
            start = max(0, match.start() - 200)
            end = min(len(latex_content), match.end() + 200)
            context_before = latex_content[start:match.start()]
            context_after = latex_content[match.end():end]

            theorems.append({
                'type': env,
                'title': title.strip('[]') if title else "",
                'body': body,
                'context_before': context_before,
                'context_after': context_after,
                'char_position': match.start(),
            })

    return theorems


def batch_download_arxiv_sources(
    arxiv_ids: List[str],
    output_dir: str = None,
    delay: float = 3.0,
    max_downloads: int = None,
) -> Dict[str, Optional[str]]:
    """Download LaTeX sources for multiple ArXiv papers with rate limiting."""
    results = {}
    downloaded = 0

    for i, arxiv_id in enumerate(arxiv_ids):
        if max_downloads and downloaded >= max_downloads:
            break

        print(f"  [{i+1}/{len(arxiv_ids)}] Downloading {arxiv_id}...")
        source_dir = download_arxiv_source(arxiv_id, output_dir)
        results[arxiv_id] = source_dir

        if source_dir:
            downloaded += 1

        # Rate limiting - ArXiv allows ~1 request per 3 seconds
        if i < len(arxiv_ids) - 1:
            time.sleep(delay)

    print(f"\nDownloaded {downloaded}/{len(arxiv_ids)} papers")
    return results


if __name__ == "__main__":
    # Quick test: create a small stratified sample
    sample = create_stratified_sample(
        n_per_category=10,
        output_path=os.path.join(SAMPLE_DIR, "test_sample_10.jsonl"),
    )
    print(f"\nSample entries: {len(sample)}")
    if sample:
        print(f"First entry categories: {sample[0].get('categories', 'N/A')}")
        print(f"First entry type: {sample[0].get('type', 'N/A')}")
