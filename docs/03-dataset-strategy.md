# Dataset Strategy

## Philosophy

Quality and task-relevance over volume. This is a **specialist** model — every example should reinforce one of the seven core skills, not generic Q&A.

## Sizing Plan (Phased)

| Phase | Size | Purpose |
|---|---|---|
| Phase 1 (MVP) | ~5 GB curated | Validate pipeline, training loop, and eval harness end-to-end |
| Phase 2 (Scale-up) | 10–20 GB curated | Only if Phase 1 shows clear improvement over base model on evals |
| Phase 3 (Optional) | 20 GB+ | Only if compute/time budget allows; diminishing returns expected on 7B + QLoRA |

Do not jump straight to the largest dataset — validate the pipeline cheaply first.

## Skill-to-Data Mapping

| Skill | Data type needed |
|---|---|
| Debugging & bug fixing | (buggy code, error/traceback, fixed code) triples |
| Repository-level coding | Multi-file diffs, PRs with context from >1 file |
| Refactoring | (before, after) code pairs with a stated refactor goal |
| Test generation/fixing | (source code → generated tests), (failing test → fix) pairs |
| Code reasoning | Step-by-step explanations of bugs/logic, chain-of-thought over code |
| Code review | (PR diff → review comments) pairs |
| Agentic SWE | Multi-step task transcripts: plan → tool call → observation → next action |

## Language Distribution Target (rough guide, adjust based on available data)

- Python: 30%
- JavaScript/TypeScript: 25%
- Java: 10%
- C/C++: 10%
- Go: 8%
- Rust: 7%
- Dart: 5%
- Kotlin: 5%

## Deduplication & Quality Bar

- Remove near-duplicate examples (hash + embedding similarity threshold).
- Exclude code with no meaningful signal (e.g., trivial boilerplate, auto-generated files, minified code).
- Prefer real-world examples (real bugs/PRs) over synthetic when available; use synthetic to fill gaps in underrepresented skills/languages.
- Cap max token length per example based on target training context (see `05-qlora-training-plan.md`) to avoid excessive truncation.

See `04-dataset-sources.md` for where data comes from and `06-jsonl-format.md` / `07-data-cleaning-pipeline.md` for structure and processing.
