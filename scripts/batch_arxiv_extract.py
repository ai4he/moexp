"""
Batch ArXiv LaTeX Download + Theorem Extraction
Processes papers from the stratified sample, downloads LaTeX sources,
and extracts theorem environments.

Usage:
    python scripts/batch_arxiv_extract.py [--max-papers 50] [--delay 3]
"""

import json
import os
import sys
import time
import argparse
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import (
    download_arxiv_source, extract_latex_from_source,
    extract_theorems_from_latex, get_domain_group,
    SAMPLE_DIR, CACHE_DIR
)

BASE_DIR = "/scratch/ctoxtli/moexp"
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def load_sample_paper_ids(sample_path: str) -> list:
    """Load unique paper IDs from stratified sample."""
    paper_ids = set()
    with open(sample_path) as f:
        for line in f:
            entry = json.loads(line)
            pid = entry.get('id', '')
            if pid:
                paper_ids.add(pid)
    return sorted(list(paper_ids))


def batch_process(
    sample_path: str,
    max_papers: int = 50,
    delay: float = 3.0,
    checkpoint_path: str = None,
):
    """Download and extract theorems from sample papers."""
    if checkpoint_path is None:
        checkpoint_path = os.path.join(RESULTS_DIR, "arxiv_extraction_checkpoint.json")

    # Load checkpoint
    processed = {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            processed = json.load(f)
        print(f"Resuming from checkpoint: {len(processed)} papers already processed")

    # Load paper IDs
    paper_ids = load_sample_paper_ids(sample_path)
    print(f"Total unique papers in sample: {len(paper_ids)}")

    # Filter already processed
    to_process = [pid for pid in paper_ids if pid not in processed][:max_papers]
    print(f"Papers to process: {len(to_process)}")

    # Output file for extracted theorems
    theorems_path = os.path.join(RESULTS_DIR, "arxiv_extracted_theorems.jsonl")

    stats = Counter()
    for i, pid in enumerate(to_process):
        print(f"\n[{i+1}/{len(to_process)}] Processing {pid}...")

        result = {
            'paper_id': pid,
            'status': 'pending',
            'theorems': [],
            'latex_length': 0,
        }

        # Download
        source_dir = download_arxiv_source(pid)
        if source_dir:
            # Extract LaTeX
            latex = extract_latex_from_source(source_dir)
            if latex:
                result['latex_length'] = len(latex)
                # Extract theorems
                theorems = extract_theorems_from_latex(latex)
                result['theorems'] = theorems
                result['status'] = 'success'
                stats['success'] += 1
                stats['theorems'] += len(theorems)
                print(f"  Success: {len(latex)} chars, {len(theorems)} theorems")
            else:
                result['status'] = 'no_latex'
                stats['no_latex'] += 1
                print(f"  No LaTeX content found")
        else:
            result['status'] = 'download_failed'
            stats['download_failed'] += 1
            print(f"  Download failed")

        # Save theorem entries
        with open(theorems_path, 'a') as f:
            for thm in result.get('theorems', []):
                entry = {
                    'paper_id': pid,
                    'type': thm['type'],
                    'title': thm.get('title', ''),
                    'body': thm['body'],
                    'context_before': thm.get('context_before', ''),
                    'context_after': thm.get('context_after', ''),
                }
                f.write(json.dumps(entry) + '\n')

        # Update checkpoint
        processed[pid] = {
            'status': result['status'],
            'n_theorems': len(result.get('theorems', [])),
            'latex_length': result.get('latex_length', 0),
        }
        with open(checkpoint_path, 'w') as f:
            json.dump(processed, f)

        # Rate limiting
        time.sleep(delay)

    # Summary
    print(f"\n=== Batch Processing Complete ===")
    print(f"Total processed: {len(to_process)}")
    for status, count in stats.most_common():
        print(f"  {status}: {count}")

    return processed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch ArXiv theorem extraction")
    parser.add_argument("--max-papers", type=int, default=50)
    parser.add_argument("--delay", type=float, default=3.0)
    parser.add_argument("--sample-path", default=os.path.join(SAMPLE_DIR, "stratified_sample_100.jsonl"))
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    batch_process(
        sample_path=args.sample_path,
        max_papers=args.max_papers,
        delay=args.delay,
    )
