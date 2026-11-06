# Dataset Sources

> ⚠️ Always verify license terms per-repository/dataset before use (see `11-licensing-and-security.md`).

## Real-world code sources

- **GitHub public repositories** (via GitHub API / GH Archive)
  - Bug-fix commits: filter commits with messages matching patterns like `fix`, `bug`, `resolve`, `patch`
  - PR diffs + review comments for code review data
  - Issue → linked PR pairs for debugging context
- **Open-source issue trackers** (GitHub Issues, GitLab Issues) linked to resolving commits
- **Existing open code-instruction datasets** (e.g., permissively licensed subsets of code-alpaca-style, CommitPackFT-style, or similar public code datasets) — check license per dataset
- **Test suites from open-source repos** for test-generation examples (source file + its existing test file, aligned pairs)

## Synthetic data generation

- Use a strong existing model (e.g., a larger instruct model or the base Qwen2.5-Coder itself) to generate:
  - Additional debugging scenarios (inject synthetic bugs into clean code, pair with fix + explanation)
  - Refactoring pairs (ask model to refactor clean code, human/model review for quality)
  - Agentic transcripts (simulate plan → act → observe loops on toy repos)
- Always tag synthetic data distinctly from real-world data in metadata (see `06-jsonl-format.md`) so quality/regressions can be traced back to source type.

## Repo-level context sources

- Clone small-to-medium permissively licensed repos to construct multi-file context windows (e.g., a function + the files it imports from).
- Use static analysis (e.g., import graphs) to select *relevant* surrounding files rather than dumping entire repos.

## Exclusions

- No proprietary/private codebases without explicit permission.
- No repos with restrictive licenses (GPL-family) if the resulting model or its outputs might be redistributed — confirm policy before inclusion.
- No datasets containing secrets/credentials (run secret-scanning as part of cleaning — see `07-data-cleaning-pipeline.md`).
