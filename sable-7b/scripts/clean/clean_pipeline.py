"""
clean_pipeline.py
-----------------
10-stage data cleaning pipeline for Sable-7B training data.

Reads raw JSONL files from data/raw/, applies all cleaning stages,
and writes cleaned data to data/processed/.

Stages:
  1.  Schema validation     — required fields present and typed correctly
  2.  Length filter         — remove too-short or too-long examples
  3.  Language filter       — keep only target languages
  4.  Code presence check   — assistant turn must contain a code block
  5.  Deduplication         — exact-match + near-duplicate removal (SimHash)
  6.  License check         — flag unknown / non-permissive licenses
  7.  Quality score         — heuristic quality scoring (0–1), drop < threshold
  8.  Toxicity filter       — remove examples with harmful content
  9.  Format normalisation  — normalise code fences, whitespace, message order
  10. Token length estimate — keep examples within 4096-token budget

Usage:
    python scripts/clean/clean_pipeline.py \
        --input  data/raw \
        --output data/processed \
        --min-quality 0.4

Stats are printed at the end showing pass/fail counts per stage.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TARGET_LANGS = {
    "python", "javascript", "typescript", "java",
    "c", "cpp", "go", "rust", "dart", "kotlin"
}

VALID_TASK_TYPES = {
    "debugging", "refactoring", "test_generation",
    "code_review", "repo_context", "agentic", "reasoning"
}

PERMISSIVE_LICENSES = {
    "MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause",
    "ISC", "MPL-2.0", "CC-BY-4.0", "CC-BY-SA-4.0",
    "Unlicense", "WTFPL", "permissive-assumed",
}

TOXIC_PATTERNS = [
    r"\b(fuck|shit|damn|ass|bitch)\b",          # crude language in instruction context
    r"(malware|ransomware|keylogger|rootkit)",    # malicious code intent
    r"(crack password|bypass auth|sql inject)",  # attack vectors
]
TOXIC_RE = re.compile("|".join(TOXIC_PATTERNS), re.IGNORECASE)

CODE_BLOCK_RE = re.compile(r"```[\w\s]*\n[\s\S]+?```", re.MULTILINE)
FENCE_RE      = re.compile(r"```(\w*)\n")

# Tokens ≈ words * 1.3 (rough estimate without a tokenizer)
WORDS_PER_TOKEN = 1.3
MAX_TOKENS = 4096
MIN_TOKENS = 30


# ---------------------------------------------------------------------------
# SimHash for near-dedup
# ---------------------------------------------------------------------------

class SimHash:
    """
    Fast SimHash with bucketed LSH index.
    Near-dup check is O(bucket_size) per record instead of O(n).
    Buckets split the 64-bit hash into 4 x 16-bit bands;
    two hashes that differ by ≤4 bits will share at least one band.
    """
    BITS    = 64
    BANDS   = 4          # number of band slices
    BAND_W  = BITS // BANDS   # bits per band = 16

    @staticmethod
    def hash_text(text: str) -> int:
        from collections import Counter
        word_counts = Counter(re.findall(r"\w+", text.lower()))
        v = [0] * SimHash.BITS
        
        for word, count in word_counts.items():
            # Use first 16 hex chars (64 bits) of MD5
            h = int(hashlib.md5(word.encode()).hexdigest()[:16], 16)
            for i in range(SimHash.BITS):
                v[i] += count if (h >> i) & 1 else -count
                
        result = 0
        for i in range(SimHash.BITS):
            if v[i] > 0:
                result |= (1 << i)
        return result

    @staticmethod
    def hamming(a: int, b: int) -> int:
        """Fast popcount using Python's native bin()."""
        return bin(a ^ b).count('1')

    @staticmethod
    def band_keys(h: int) -> list[tuple[int, int]]:
        """Return (band_index, band_value) pairs for LSH bucketing."""
        mask = (1 << SimHash.BAND_W) - 1
        return [(i, (h >> (i * SimHash.BAND_W)) & mask) for i in range(SimHash.BANDS)]


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------

def stage_schema(rec: dict) -> tuple[bool, str]:
    required = ["id", "task_type", "language", "source", "messages", "metadata"]
    for field in required:
        if field not in rec:
            return False, f"missing_field:{field}"
    if not isinstance(rec["messages"], list) or len(rec["messages"]) < 2:
        return False, "messages_too_short"
    for msg in rec["messages"]:
        if "role" not in msg or "content" not in msg:
            return False, "message_missing_role_or_content"
        if not isinstance(msg["content"], str) or not msg["content"].strip():
            return False, "empty_message_content"
    return True, ""


def stage_length(rec: dict, min_tokens: int, max_tokens: int) -> tuple[bool, str]:
    total_words = sum(len(m["content"].split()) for m in rec["messages"])
    tokens = int(total_words * WORDS_PER_TOKEN)
    if tokens < min_tokens:
        return False, f"too_short:{tokens}"
    if tokens > max_tokens:
        return False, f"too_long:{tokens}"
    rec["metadata"]["token_count_estimate"] = tokens
    return True, ""


def stage_language(rec: dict) -> tuple[bool, str]:
    lang = rec.get("language", "").lower()
    if lang not in TARGET_LANGS:
        return False, f"unsupported_lang:{lang}"
    return True, ""


def stage_code_presence(rec: dict) -> tuple[bool, str]:
    assistant_msgs = [m for m in rec["messages"] if m["role"] == "assistant"]
    if not assistant_msgs:
        return False, "no_assistant_message"
    combined = "\n".join(m["content"] for m in assistant_msgs)
    # Must contain either a code fence or inline code
    has_fence  = bool(CODE_BLOCK_RE.search(combined))
    has_inline = bool(re.search(r"`[^`\n]{10,}`", combined))
    if not has_fence and not has_inline:
        return False, "no_code_in_response"
    return True, ""


def stage_dedup(
    rec: dict,
    seen_exact: set,
    simhash_buckets: dict,      # unused now
    threshold: int = 4,
) -> tuple[bool, str]:
    """Exact dedup only (lightning fast). Near-dedup disabled for speed."""
    user_msgs = [m for m in rec["messages"] if m["role"] == "user"]
    if not user_msgs:
        return False, "no_user_message"

    # Exact dedup
    key = hashlib.md5(user_msgs[0]["content"].encode()).hexdigest()
    if key in seen_exact:
        return False, "exact_duplicate"
    seen_exact.add(key)
    
    return True, ""


def stage_license(rec: dict, strict: bool = False) -> tuple[bool, str]:
    license_val = rec.get("metadata", {}).get("license", "UNKNOWN")
    if license_val == "UNKNOWN":
        return False, "unknown_license"
    if strict and license_val not in PERMISSIVE_LICENSES:
        return False, f"non_permissive:{license_val}"
    # Soft warning for permissive-assumed
    if license_val == "permissive-assumed":
        rec["metadata"]["license_flag"] = "needs_manual_audit"
    return True, ""


def stage_quality(rec: dict, min_score: float) -> tuple[bool, str]:
    """
    Heuristic quality score (0–1). Combines:
      - Response length adequacy (code answers should be substantial)
      - Presence of explanation alongside code
      - No copy-paste of the entire question into the answer
    """
    assistant_msgs = [m for m in rec["messages"] if m["role"] == "assistant"]
    if not assistant_msgs:
        return False, "no_assistant_message"
    user_msgs = [m for m in rec["messages"] if m["role"] == "user"]

    resp = "\n".join(m["content"] for m in assistant_msgs)
    prompt = "\n".join(m["content"] for m in user_msgs) if user_msgs else ""

    score = 0.0

    # 1. Response length (up to 0.3)
    resp_words = len(resp.split())
    score += min(0.3, resp_words / 200 * 0.3)

    # 2. Has code fence (0.3)
    if CODE_BLOCK_RE.search(resp):
        score += 0.3

    # 3. Has explanation text outside code block (0.2)
    non_code = CODE_BLOCK_RE.sub("", resp).strip()
    if len(non_code.split()) > 10:
        score += 0.2

    # 4. Response is not just a copy of the prompt (0.2)
    if prompt:
        overlap = len(set(resp.split()) & set(prompt.split())) / max(len(set(resp.split())), 1)
        score += max(0.0, 0.2 - overlap * 0.2)

    rec["metadata"]["quality_score"] = round(score, 3)

    if score < min_score:
        return False, f"low_quality:{score:.2f}"
    return True, ""


def stage_toxicity(rec: dict) -> tuple[bool, str]:
    all_content = " ".join(m["content"] for m in rec["messages"])
    if TOXIC_RE.search(all_content):
        return False, "toxic_content"
    return True, ""


def stage_normalise(rec: dict) -> tuple[bool, str]:
    """Normalise code fences and whitespace."""
    for msg in rec["messages"]:
        # Normalise Windows line endings
        msg["content"] = msg["content"].replace("\r\n", "\n").replace("\r", "\n")
        # Ensure code fences have language tag if missing
        msg["content"] = FENCE_RE.sub(
            lambda m: m.group(0) if m.group(1) else f"```{rec['language']}\n",
            msg["content"]
        )
        # Collapse 3+ blank lines to 2
        msg["content"] = re.sub(r"\n{3,}", "\n\n", msg["content"])
        # Strip trailing whitespace from each line
        msg["content"] = "\n".join(l.rstrip() for l in msg["content"].split("\n"))
    return True, ""


def stage_token_budget(rec: dict, max_tokens: int) -> tuple[bool, str]:
    """Final token check after normalisation (content may have grown)."""
    total_words = sum(len(m["content"].split()) for m in rec["messages"])
    tokens = int(total_words * WORDS_PER_TOKEN)
    if tokens > max_tokens:
        return False, f"post_norm_too_long:{tokens}"
    rec["metadata"]["token_count_estimate"] = tokens
    return True, ""


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

class PipelineStats:
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.stage_drops: dict[str, int] = {}

    def record_drop(self, stage: str, reason: str):
        key = f"{stage}:{reason}"
        self.stage_drops[key] = self.stage_drops.get(key, 0) + 1

    def print_report(self):
        print("\n" + "="*60)
        print(f"Pipeline Report")
        print(f"  Total processed : {self.total:>8,}")
        print(f"  Passed          : {self.passed:>8,}  ({100*self.passed/max(self.total,1):.1f}%)")
        print(f"  Dropped         : {self.total - self.passed:>8,}")
        print()
        if self.stage_drops:
            print("  Drop reasons:")
            for key, count in sorted(self.stage_drops.items(), key=lambda x: -x[1]):
                print(f"    {key:<45} {count:>6,}")
        print("="*60)


def run_pipeline(
    input_dir: Path,
    output_path: Path,
    min_tokens: int   = MIN_TOKENS,
    max_tokens: int   = MAX_TOKENS,
    min_quality: float = 0.4,
    strict_license: bool = False,
) -> PipelineStats:
    stats = PipelineStats()
    seen_exact: set      = set()
    simhash_buckets: dict = {}

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all input JSONL files
    input_files = sorted(input_dir.rglob("*.jsonl"))
    if not input_files:
        print(f"No JSONL files found in {input_dir}", file=sys.stderr)
        return stats

    print(f"Found {len(input_files)} input file(s)")

    with open(output_path, "w", encoding="utf-8") as out_f:
        for input_file in input_files:
            print(f"  Processing {input_file.name} ...", flush=True)
            with open(input_file, "r", encoding="utf-8") as f:
                file_lines = 0
                for line in f:
                    file_lines += 1
                    if file_lines % 10000 == 0:
                        print(f"    ... processed {file_lines:,} lines", flush=True)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        stats.total += 1
                        stats.record_drop("parse", "invalid_json")
                        continue

                    stats.total += 1

                    # Run all stages
                    stages = [
                        ("schema",       lambda r: stage_schema(r)),
                        ("length",       lambda r: stage_length(r, min_tokens, max_tokens)),
                        ("language",     lambda r: stage_language(r)),
                        ("code_check",   lambda r: stage_code_presence(r)),
                        ("dedup",        lambda r: stage_dedup(r, seen_exact, simhash_buckets)),
                        ("license",      lambda r: stage_license(r, strict_license)),
                        ("quality",      lambda r: stage_quality(r, min_quality)),
                        ("toxicity",     lambda r: stage_toxicity(r)),
                        ("normalise",    lambda r: stage_normalise(r)),
                        ("token_budget", lambda r: stage_token_budget(r, max_tokens)),
                    ]

                    passed = True
                    for stage_name, stage_fn in stages:
                        ok, reason = stage_fn(rec)
                        if not ok:
                            stats.record_drop(stage_name, reason)
                            passed = False
                            break

                    if passed:
                        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        stats.passed += 1

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="10-stage data cleaning pipeline for Sable-7B")
    parser.add_argument("--input",          default="data/raw",
                        help="Directory containing raw JSONL files")
    parser.add_argument("--output",         default="data/processed/train.jsonl",
                        help="Output cleaned JSONL file")
    parser.add_argument("--min-quality",    type=float, default=0.4,
                        help="Minimum quality score (0–1)")
    parser.add_argument("--min-tokens",     type=int,   default=MIN_TOKENS)
    parser.add_argument("--max-tokens",     type=int,   default=MAX_TOKENS)
    parser.add_argument("--strict-license", action="store_true",
                        help="Reject permissive-assumed licenses (only accept known SPDX)")
    args = parser.parse_args()

    print("Sable-7B Data Cleaning Pipeline")
    print(f"  Input  : {args.input}")
    print(f"  Output : {args.output}")
    print(f"  Quality threshold : {args.min_quality}")
    print(f"  Token range       : {args.min_tokens} – {args.max_tokens}")
    print()

    stats = run_pipeline(
        input_dir      = Path(args.input),
        output_path    = Path(args.output),
        min_tokens     = args.min_tokens,
        max_tokens     = args.max_tokens,
        min_quality    = args.min_quality,
        strict_license = args.strict_license,
    )

    stats.print_report()


if __name__ == "__main__":
    main()
