"""
validate_dataset.py
-------------------
Quick validation + stats script for any JSONL dataset file.

Shows:
  - Total record count
  - Task type distribution
  - Language distribution
  - Token length percentiles (p10, p50, p90, p99)
  - Quality score distribution
  - Sample records

Usage:
    python scripts/clean/validate_dataset.py data/processed/train.jsonl
    python scripts/clean/validate_dataset.py data/processed/train.jsonl --samples 3
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def percentile(data: list[int | float], p: float) -> float:
    if not data:
        return 0.0
    data = sorted(data)
    idx = int(len(data) * p / 100)
    return data[min(idx, len(data) - 1)]


def main():
    parser = argparse.ArgumentParser(description="Validate and inspect a Sable-7B JSONL dataset")
    parser.add_argument("file", help="Path to JSONL file")
    parser.add_argument("--samples", type=int, default=2, help="Number of sample records to print")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)

    records = []
    errors  = 0
    task_types   = Counter()
    languages    = Counter()
    token_counts = []
    quality_scores = []
    license_counts = Counter()

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                continue

            records.append(rec)
            task_types[rec.get("task_type", "unknown")] += 1
            languages[rec.get("language", "unknown")] += 1

            toks = rec.get("metadata", {}).get("token_count_estimate", 0)
            token_counts.append(toks)

            qual = rec.get("metadata", {}).get("quality_score")
            if qual is not None:
                quality_scores.append(qual)

            lic = rec.get("metadata", {}).get("license", "UNKNOWN")
            license_counts[lic] += 1

    total = len(records)
    print(f"\n{'='*60}")
    print(f"Dataset: {path.name}")
    print(f"{'='*60}")
    print(f"  Total records : {total:,}")
    print(f"  Parse errors  : {errors:,}")
    print()

    if total == 0:
        print("No valid records found.")
        return

    print("Task type distribution:")
    for task, count in task_types.most_common():
        bar = "█" * int(40 * count / total)
        print(f"  {task:<20} {count:>6,}  {bar}")
    print()

    print("Language distribution:")
    for lang, count in languages.most_common():
        bar = "█" * int(40 * count / total)
        print(f"  {lang:<20} {count:>6,}  {bar}")
    print()

    print("Token length percentiles:")
    for p in [10, 25, 50, 75, 90, 99]:
        print(f"  p{p:<3} = {percentile(token_counts, p):>6.0f} tokens")
    print(f"  mean = {sum(token_counts)/len(token_counts):>6.0f} tokens")
    print()

    if quality_scores:
        print("Quality score percentiles:")
        for p in [10, 25, 50, 75, 90]:
            print(f"  p{p:<3} = {percentile(quality_scores, p):.3f}")
        print(f"  mean = {sum(quality_scores)/len(quality_scores):.3f}")
        print()

    print("License distribution:")
    for lic, count in license_counts.most_common():
        print(f"  {lic:<30} {count:>6,}")
    print()

    # Print sample records
    print(f"Sample records (first {min(args.samples, total)}):")
    print("-" * 60)
    for rec in records[:args.samples]:
        print(f"ID          : {rec.get('id', '?')}")
        print(f"Task        : {rec.get('task_type', '?')}")
        print(f"Language    : {rec.get('language', '?')}")
        print(f"License     : {rec.get('metadata', {}).get('license', '?')}")
        print(f"Quality     : {rec.get('metadata', {}).get('quality_score', '?')}")
        print(f"Tokens (est): {rec.get('metadata', {}).get('token_count_estimate', '?')}")
        msgs = rec.get("messages", [])
        for msg in msgs:
            role    = msg.get("role", "?")
            content = msg.get("content", "")[:200].replace("\n", " ")
            print(f"  [{role}] {content}...")
        print("-" * 60)


if __name__ == "__main__":
    main()
