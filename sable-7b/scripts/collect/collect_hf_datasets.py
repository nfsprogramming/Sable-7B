"""
collect_hf_datasets.py
----------------------
Pulls permissively-licensed coding datasets from Hugging Face Hub and
converts them into the Sable-7B JSONL training schema.

Usage:
    python scripts/collect/collect_hf_datasets.py --output data/raw/hf_datasets

Datasets collected (all Apache-2.0 / MIT / cc-by):
  - m-a-p/CodeFeedback-Filtered-Instruction  (debugging + code Q&A)
  - ise-uiuc/Magicoder-OSS-Instruct-75K      (real OSS-grounded instructions)
  - theblackcat102/evol-codealpaca-v1        (evolved code alpaca)
  - nickrosh/Evol-Instruct-Code-80k-v1       (evolved instruct code)
  - ise-uiuc/Magicoder-Evol-Instruct-110K    (evolved magicoder instructions)
  - sahil2801/CodeAlpaca-20k                 (baseline code alpaca)

All examples are streamed — never loads full dataset into RAM.
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterator

# ---------------------------------------------------------------------------
# Language detection helpers
# ---------------------------------------------------------------------------

LANG_PATTERNS = {
    "python":     [r"\.py\b", r"\bpython\b", r"\bdef \w+\(", r"\bimport \w"],
    "javascript": [r"\.js\b", r"\bjavascript\b", r"\bconsole\.log\b", r"\bconst \w"],
    "typescript": [r"\.ts\b", r"\btypescript\b", r"\binterface \w", r": string\b"],
    "java":       [r"\.java\b", r"\bjava\b", r"\bpublic class\b", r"\bSystem\.out\b"],
    "c":          [r"\.c\b", r"\bprintf\(", r"#include <stdio"],
    "cpp":        [r"\.cpp\b", r"\bc\+\+\b", r"\bstd::\w", r"#include <iostream"],
    "go":         [r"\.go\b", r"\bgolang\b", r"\bfunc \w+\(", r"\bfmt\.Print"],
    "rust":       [r"\.rs\b", r"\brust\b", r"\bfn \w+\(", r"\blet mut\b"],
    "dart":       [r"\.dart\b", r"\bdart\b", r"\bvoid main\(\)", r"\bFlutter\b"],
    "kotlin":     [r"\.kt\b", r"\bkotlin\b", r"\bfun \w+\(", r"\bval \w"],
}

TARGET_LANGS = set(LANG_PATTERNS.keys())

def detect_language(text: str) -> str:
    text_lower = text.lower()
    for lang, patterns in LANG_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text_lower):
                return lang
    return "python"  # default fallback


SYSTEM_PROMPT = (
    "You are Sable, a specialist software engineering assistant. "
    "You excel at debugging, refactoring, code review, test generation, "
    "repository-level coding, and agentic software engineering tasks. "
    "Be precise, thorough, and always explain your reasoning."
)


def make_id(source_ref: str, content: str) -> str:
    h = hashlib.md5((source_ref + content[:200]).encode()).hexdigest()[:12]
    return f"{source_ref.replace('/', '-')}-{h}"


def write_jsonl(records: Iterator[dict], out_path: Path, source_name: str) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
            if count % 500 == 0:
                print(f"  [{source_name}] {count} records written...", flush=True)
    return count


# ---------------------------------------------------------------------------
# Dataset-specific converters
# ---------------------------------------------------------------------------

def collect_codefeedback(out_dir: Path) -> int:
    """
    m-a-p/CodeFeedback-Filtered-Instruction
    Columns: query, answer
    task_type: mostly debugging / code Q&A → label as 'debugging' or 'reasoning'
    License: Apache-2.0
    """
    from datasets import load_dataset
    print("Collecting: CodeFeedback-Filtered-Instruction ...", flush=True)
    ds = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train", streaming=True)

    def convert(row):
        query = (row.get("query") or "").strip()
        answer = (row.get("answer") or "").strip()
        if not query or not answer:
            return None
        lang = detect_language(query + " " + answer)
        task = "debugging" if any(w in query.lower() for w in ["bug", "error", "fix", "wrong", "fail", "crash", "exception"]) else "reasoning"
        return {
            "id": make_id("hf-codefeedback", query),
            "task_type": task,
            "language": lang,
            "source": "real_world",
            "source_ref": "hf:m-a-p/CodeFeedback-Filtered-Instruction",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": query},
                {"role": "assistant", "content": answer},
            ],
            "metadata": {
                "num_files": 1,
                "token_count_estimate": len((query + answer).split()),
                "difficulty": "medium",
                "license": "Apache-2.0",
            },
        }

    records = (r for row in ds if (r := convert(row)) is not None)
    return write_jsonl(records, out_dir / "codefeedback.jsonl", "CodeFeedback")


def collect_magicoder(out_dir: Path) -> int:
    """
    ise-uiuc/Magicoder-OSS-Instruct-75K
    Columns: problem, solution, (lang)
    License: Apache-2.0
    """
    from datasets import load_dataset
    print("Collecting: Magicoder-OSS-Instruct-75K ...", flush=True)
    ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train", streaming=True)

    def convert(row):
        problem  = (row.get("problem") or "").strip()
        solution = (row.get("solution") or "").strip()
        lang_raw = (row.get("lang") or "").strip().lower()
        if not problem or not solution:
            return None
        lang = lang_raw if lang_raw in TARGET_LANGS else detect_language(problem + solution)
        return {
            "id": make_id("hf-magicoder", problem),
            "task_type": "repo_context",
            "language": lang,
            "source": "real_world",
            "source_ref": "hf:ise-uiuc/Magicoder-OSS-Instruct-75K",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": problem},
                {"role": "assistant", "content": solution},
            ],
            "metadata": {
                "num_files": 1,
                "token_count_estimate": len((problem + solution).split()),
                "difficulty": "medium",
                "license": "Apache-2.0",
            },
        }

    records = (r for row in ds if (r := convert(row)) is not None)
    return write_jsonl(records, out_dir / "magicoder.jsonl", "Magicoder")


def collect_evol_codealpaca(out_dir: Path) -> int:
    """
    theblackcat102/evol-codealpaca-v1
    Columns: instruction, output
    License: Apache-2.0 (check per-use)
    """
    from datasets import load_dataset
    print("Collecting: evol-codealpaca-v1 ...", flush=True)
    ds = load_dataset("theblackcat102/evol-codealpaca-v1", split="train", streaming=True)

    def convert(row):
        instruction = (row.get("instruction") or "").strip()
        output      = (row.get("output") or "").strip()
        if not instruction or not output:
            return None
        lang = detect_language(instruction + " " + output)
        task = "refactoring" if any(w in instruction.lower() for w in ["refactor", "improve", "optimize", "clean"]) else "reasoning"
        return {
            "id": make_id("hf-evol-codealpaca", instruction),
            "task_type": task,
            "language": lang,
            "source": "real_world",
            "source_ref": "hf:theblackcat102/evol-codealpaca-v1",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": instruction},
                {"role": "assistant", "content": output},
            ],
            "metadata": {
                "num_files": 1,
                "token_count_estimate": len((instruction + output).split()),
                "difficulty": "medium",
                "license": "Apache-2.0",
            },
        }

    records = (r for row in ds if (r := convert(row)) is not None)
    return write_jsonl(records, out_dir / "evol_codealpaca.jsonl", "EvolCodeAlpaca")


def collect_evol_instruct(out_dir: Path) -> int:
    """
    nickrosh/Evol-Instruct-Code-80k-v1
    Columns: instruction, output
    License: Apache-2.0
    """
    from datasets import load_dataset
    print("Collecting: Evol-Instruct-Code-80k-v1 ...", flush=True)
    ds = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train", streaming=True)

    def convert(row):
        instruction = (row.get("instruction") or "").strip()
        output      = (row.get("output") or "").strip()
        if not instruction or not output:
            return None
        lang = detect_language(instruction + " " + output)
        task = "debugging" if any(w in instruction.lower() for w in ["debug", "bug", "fix", "error"]) else "reasoning"
        return {
            "id": make_id("hf-evol-instruct", instruction),
            "task_type": task,
            "language": lang,
            "source": "real_world",
            "source_ref": "hf:nickrosh/Evol-Instruct-Code-80k-v1",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": instruction},
                {"role": "assistant", "content": output},
            ],
            "metadata": {
                "num_files": 1,
                "token_count_estimate": len((instruction + output).split()),
                "difficulty": "medium",
                "license": "Apache-2.0",
            },
        }

    records = (r for row in ds if (r := convert(row)) is not None)
    return write_jsonl(records, out_dir / "evol_instruct.jsonl", "EvolInstruct")


def collect_magicoder_evol(out_dir: Path) -> int:
    """
    ise-uiuc/Magicoder-Evol-Instruct-110K
    Columns: instruction, response
    License: Apache-2.0
    """
    from datasets import load_dataset
    print("Collecting: Magicoder-Evol-Instruct-110K ...", flush=True)
    ds = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split="train", streaming=True)

    def convert(row):
        instruction = (row.get("instruction") or "").strip()
        response    = (row.get("response") or "").strip()
        if not instruction or not response:
            return None
        lang = detect_language(instruction + " " + response)
        task = "reasoning"
        return {
            "id": make_id("hf-magicoder-evol", instruction),
            "task_type": task,
            "language": lang,
            "source": "real_world",
            "source_ref": "hf:ise-uiuc/Magicoder-Evol-Instruct-110K",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": instruction},
                {"role": "assistant", "content": response},
            ],
            "metadata": {
                "num_files": 1,
                "token_count_estimate": len((instruction + response).split()),
                "difficulty": "hard",
                "license": "Apache-2.0",
            },
        }

    records = (r for row in ds if (r := convert(row)) is not None)
    return write_jsonl(records, out_dir / "magicoder_evol.jsonl", "MagicoderEvol")


def collect_codealpaca_20k(out_dir: Path) -> int:
    """
    sahil2801/CodeAlpaca-20k
    Columns: instruction, input, output
    License: Apache-2.0
    """
    from datasets import load_dataset
    print("Collecting: CodeAlpaca-20k ...", flush=True)
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train", streaming=True)

    def convert(row):
        instruction = (row.get("instruction") or "").strip()
        inp         = (row.get("input") or "").strip()
        output      = (row.get("output") or "").strip()
        
        user_content = f"{instruction}\n\n{inp}" if inp else instruction
        if not user_content or not output:
            return None
            
        lang = detect_language(user_content + " " + output)
        return {
            "id": make_id("hf-codealpaca-20k", user_content),
            "task_type": "reasoning",
            "language": lang,
            "source": "real_world",
            "source_ref": "hf:sahil2801/CodeAlpaca-20k",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
                {"role": "assistant", "content": output},
            ],
            "metadata": {
                "num_files": 1,
                "token_count_estimate": len((user_content + output).split()),
                "difficulty": "medium",
                "license": "Apache-2.0",
            },
        }

    records = (r for row in ds if (r := convert(row)) is not None)
    return write_jsonl(records, out_dir / "codealpaca_20k.jsonl", "CodeAlpaca20k")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

COLLECTORS = {
    "codefeedback":   collect_codefeedback,
    "magicoder":      collect_magicoder,
    "evol_codealpaca": collect_evol_codealpaca,
    "evol_instruct":  collect_evol_instruct,
    "magicoder_evol": collect_magicoder_evol,
    "codealpaca_20k": collect_codealpaca_20k,
}


def main():
    parser = argparse.ArgumentParser(description="Collect HuggingFace coding datasets for Sable-7B")
    parser.add_argument("--output", default="data/raw/hf_datasets", help="Output directory for raw JSONL files")
    parser.add_argument("--datasets", nargs="+", choices=list(COLLECTORS), default=list(COLLECTORS),
                        help="Which datasets to collect (default: all)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for name in args.datasets:
        try:
            count = COLLECTORS[name](out_dir)
            print(f"  ✅ {name}: {count:,} records → {out_dir / (name + '.jsonl')}")
            total += count
        except Exception as e:
            print(f"  ❌ {name} failed: {e}", file=sys.stderr)

    print(f"\nDone. Total records collected: {total:,}")
    print(f"Output directory: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
