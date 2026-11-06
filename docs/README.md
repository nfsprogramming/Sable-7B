# Sable-7B — Project Documentation

A specialist Software Engineering / Coding Agent fine-tune of **Qwen2.5-Coder-7B-Instruct**, trained via **QLoRA (4-bit) + Model Soup**, targeting a laptop with an **RTX 4050 (6GB VRAM)**, **i7-14700HX**, **16 GB RAM**, and a **1 TB SSD**.

## Contents

| File | Description |
|---|---|
| [01-project-spec.md](01-project-spec.md) | Goals, non-goals, target skills/languages, definition of done |
| [02-model-selection.md](02-model-selection.md) | Why Qwen2.5-Coder-7B-Instruct, alternatives considered |
| [03-dataset-strategy.md](03-dataset-strategy.md) | Phased dataset sizing, skill-to-data mapping, language mix |
| [04-dataset-sources.md](04-dataset-sources.md) | Where training data comes from, real-world + synthetic |
| [05-jsonl-format.md](05-jsonl-format.md) | Training data schema, repo-level and agentic conventions |
| [06-data-cleaning-pipeline.md](06-data-cleaning-pipeline.md) | End-to-end cleaning/processing stages |
| [07-qlora-training-plan.md](07-qlora-training-plan.md) | LoRA config, hardware constraints, training loop, soup step |
| [08-hardware-and-storage.md](08-hardware-and-storage.md) | Storage budget breakdown, VRAM feasibility notes |
| [09-evaluation-strategy.md](09-evaluation-strategy.md) | Per-skill eval methods, regression checks, promotion criteria |
| [10-experiment-logging.md](10-experiment-logging.md) | What to log per run, versioning discipline |
| [11-licensing-and-security.md](11-licensing-and-security.md) | Model/data licensing, secret scanning, security practices |
| [12-development-roadmap.md](12-development-roadmap.md) | Milestone-by-milestone roadmap from setup to deployment |

## Quick Summary

- **Base model:** Qwen2.5-Coder-7B-Instruct (7.61B params)
- **Method:** QLoRA 4-bit + Soup
- **Dataset:** Start ~5 GB curated, scale to 10–20 GB if results improve
- **Focus:** Debugging, repo-level coding, refactoring, test gen/fixing, code reasoning, code review, agentic SWE
- **Languages:** Python, JS/TS, Java, C/C++, Go, Rust, Dart, Kotlin
- **Storage reserve:** 150–200 GB of the 1 TB SSD
