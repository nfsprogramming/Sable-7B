"""
collect_github_bugfix.py
------------------------
Collects bug-fix examples from GitHub via the REST API by:
  1. Searching for commits with keywords ("fix bug", "fixes #", "patch", etc.)
  2. Fetching the diff for each matching commit
  3. Extracting before/after code blocks from the diff
  4. Writing to the Sable-7B JSONL schema

Requirements:
  - GITHUB_TOKEN env var (set it before running)
  - pip install PyGithub

Usage:
    $env:GITHUB_TOKEN = "ghp_..."
    python scripts/collect/collect_github_bugfix.py --output data/raw/github_bugfix --max-commits 5000
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterator

try:
    from github import Github, RateLimitExceededException, GithubException
except ImportError:
    print("ERROR: PyGithub not installed. Run: pip install PyGithub")
    sys.exit(1)


SYSTEM_PROMPT = (
    "You are Sable, a specialist software engineering assistant. "
    "You excel at debugging, refactoring, code review, test generation, "
    "repository-level coding, and agentic software engineering tasks. "
    "Be precise, thorough, and always explain your reasoning."
)

# Languages we care about and their file extensions
LANG_EXT = {
    ".py":   "python",
    ".js":   "javascript",
    ".ts":   "typescript",
    ".java": "java",
    ".c":    "c",
    ".cpp":  "cpp",
    ".cc":   "cpp",
    ".go":   "go",
    ".rs":   "rust",
}

# Repos to search — permissive-licensed OSS projects known to have good bug fixes
TARGET_REPOS = [
    # Python
    "psf/requests",
    "pallets/flask",
    "django/django",
    "tiangolo/fastapi",
    "aio-libs/aiohttp",
    # JavaScript / TypeScript
    "expressjs/express",
    "axios/axios",
    "microsoft/TypeScript",
    # Go
    "gin-gonic/gin",
    "golang/go",
    # Rust
    "rust-lang/rust",
    "tokio-rs/tokio",
]

BUG_KEYWORDS = [
    "fix bug", "fixes #", "bugfix", "bug fix", "patch",
    "fix error", "fix crash", "resolve issue", "fix regression",
    "fix typo", "correct", "repair"
]


def make_id(repo: str, sha: str) -> str:
    h = hashlib.md5((repo + sha).encode()).hexdigest()[:12]
    return f"gh-{repo.replace('/', '-')}-{h}"


def parse_diff_for_lang(diff: str, lang: str) -> tuple[str, str]:
    """Extract before/after code blocks from a unified diff."""
    before_lines, after_lines = [], []
    for line in diff.split("\n"):
        if line.startswith("---") or line.startswith("+++") or line.startswith("@@"):
            continue
        if line.startswith("-"):
            before_lines.append(line[1:])
        elif line.startswith("+"):
            after_lines.append(line[1:])
        else:
            before_lines.append(line[1:] if line.startswith(" ") else line)
            after_lines.append(line[1:] if line.startswith(" ") else line)
    return "\n".join(before_lines).strip(), "\n".join(after_lines).strip()


def get_lang_from_filename(filename: str) -> str | None:
    ext = Path(filename).suffix.lower()
    return LANG_EXT.get(ext)


def commits_from_repo(g: Github, repo_name: str, max_per_repo: int) -> Iterator[dict]:
    try:
        repo = g.get_repo(repo_name)
        license_name = repo.license.spdx_id if repo.license else "UNKNOWN"

        # Only process permissive licenses
        if license_name not in {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "MPL-2.0", "NOASSERTION"}:
            print(f"  Skipping {repo_name}: non-permissive license ({license_name})")
            return

        count = 0
        for keyword in BUG_KEYWORDS:
            if count >= max_per_repo:
                break
            try:
                commits = repo.get_commits()
                for commit in commits:
                    if count >= max_per_repo:
                        break
                    msg = commit.commit.message or ""
                    if keyword.lower() not in msg.lower():
                        continue
                    # Fetch files changed
                    try:
                        files = commit.files
                    except GithubException:
                        continue

                    for f in files:
                        lang = get_lang_from_filename(f.filename)
                        if not lang:
                            continue
                        patch = f.patch or ""
                        if len(patch) < 50 or len(patch) > 8000:
                            continue
                        before, after = parse_diff_for_lang(patch, lang)
                        if not before or not after or before == after:
                            continue

                        user_content = (
                            f"### Repository: {repo_name}\n"
                            f"### File: {f.filename}\n"
                            f"### Commit message: {msg.strip()[:300]}\n\n"
                            f"### Buggy code:\n```{lang}\n{before}\n```\n\n"
                            f"### Task:\nFix the bug described in the commit message."
                        )
                        assistant_content = (
                            f"Here is the corrected code:\n\n"
                            f"```{lang}\n{after}\n```\n\n"
                            f"**Explanation:** This commit {msg.strip()[:150]}"
                        )

                        yield {
                            "id": make_id(repo_name, commit.sha),
                            "task_type": "debugging",
                            "language": lang,
                            "source": "real_world",
                            "source_ref": f"github:{repo_name}@{commit.sha[:8]}",
                            "messages": [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user",   "content": user_content},
                                {"role": "assistant", "content": assistant_content},
                            ],
                            "metadata": {
                                "num_files": 1,
                                "token_count_estimate": len((user_content + assistant_content).split()),
                                "difficulty": "medium",
                                "license": license_name,
                                "repo": repo_name,
                                "sha": commit.sha[:8],
                            },
                        }
                        count += 1
                        break  # one file per commit

            except RateLimitExceededException:
                reset_time = g.rate_limiting_resettime
                wait = max(0, reset_time - time.time()) + 5
                print(f"  Rate limited — sleeping {wait:.0f}s ...", flush=True)
                time.sleep(wait)
            except GithubException as e:
                print(f"  GitHub API error on {repo_name}/{keyword}: {e}", flush=True)
                continue

    except GithubException as e:
        print(f"  Failed to access {repo_name}: {e}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Collect GitHub bug-fix examples for Sable-7B")
    parser.add_argument("--output",      default="data/raw/github_bugfix")
    parser.add_argument("--max-commits", type=int, default=5000,
                        help="Max total commits to collect")
    parser.add_argument("--max-per-repo", type=int, default=500,
                        help="Max commits per repo")
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("ERROR: Set GITHUB_TOKEN environment variable before running.")
        print("  $env:GITHUB_TOKEN = 'ghp_...'")
        sys.exit(1)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "github_bugfix.jsonl"

    g = Github(token)
    total = 0

    with open(out_file, "w", encoding="utf-8") as f:
        for repo_name in TARGET_REPOS:
            if total >= args.max_commits:
                break
            print(f"Processing {repo_name} ...", flush=True)
            repo_count = 0
            for rec in commits_from_repo(g, repo_name, args.max_per_repo):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total += 1
                repo_count += 1
                if total >= args.max_commits:
                    break
            print(f"  ✅ {repo_name}: {repo_count} records", flush=True)

    print(f"\nDone. Total: {total:,} records → {out_file}")


if __name__ == "__main__":
    main()
