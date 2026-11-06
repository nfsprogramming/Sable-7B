# Sable-7B

> A specialist Software Engineering & Coding Agent model — fine-tuned from [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) using QLoRA and Model Soup weight averaging.

**Narrow and deep, not broad and shallow.**

---

## Overview

Sable-7B is an open-source fine-tuning and data-engineering pipeline designed to specialize `Qwen2.5-Coder-7B-Instruct` for autonomous software engineering workflows. Rather than functioning as a broad, generic conversational model, Sable-7B concentrates on seven core engineering skills:

| Skill | Description |
|---|---|
| 🐛 **Debugging & Bug Fixing** | Diagnosing errors from tracebacks and producing exact, regression-free fixes |
| 🏗️ **Repository-Level Coding** | Multi-file reasoning, project structure awareness, and coordinated cross-file edits |
| 🔧 **Refactoring** | Transforming code structure and maintainability while preserving strict behavior |
| 🧪 **Test Generation & Fixing** | Authoring targeted unit/integration tests and repairing failing assertions |
| 🧠 **Code Reasoning** | Explaining logic step-by-step and diagnosing root causes |
| 👀 **Code Review** | Reviewing pull requests and diffs for syntax, security, and performance issues |
| 🤖 **Agentic SWE** | Multi-step task planning, tool-calling execution, and verification loops |

**Target Languages:** Python · JavaScript / TypeScript · Java · C / C++ · Go · Rust · Dart · Kotlin

---

## Features (Implemented)

- **10-Stage Data Cleaning Pipeline (`clean_pipeline.py`)**: Filters raw instruction datasets through schema validation, token length bounds, language filtering, code fence enforcement, SimHash exact deduplication, license compliance, heuristic quality scoring, toxicity screening, and whitespace/newline normalisation.
- **Permissive Data Collection (`collect_hf_datasets.py` & `collect_github_bugfix.py`)**: Automated collectors for vetted coding datasets from Hugging Face Hub (CodeFeedback, Magicoder, Evol-CodeAlpaca) and real-world GitHub commit patches via the REST API.
- **Dataset Inspection & Validation (`validate_dataset.py`)**: Distribution analysis tool computing token percentiles (p10, p50, p90, p99), language mix, task type breakdown, and quality scores.
- **Consumer Hardware QLoRA Configurations**:
  - `smoke_test.yaml`: Minimal test configuration with layer streaming to verify VRAM headroom on NVIDIA RTX 4050 6GB.
  - `soup.yaml`: Production 3-epoch QLoRA fine-tuning config with NF4 quantization, cosine learning rate scheduler, and gradient accumulation.
- **Kaggle Cloud Training Setup (`sable-7b/kaggle/`)**: Pre-configured Kaggle notebook (`train.ipynb`) and kernel descriptor (`kernel-metadata.json`) for acceleration on free Tesla T4 GPUs.
- **Comprehensive Architecture & Specification Suite (`docs/`)**: 12 detailed specifications covering dataset curation, hardware constraints, evaluation rubrics, licensing, and roadmap milestones.
- **Automated Test Suite (`sable-7b/tests/`)**: Pytest validation suite verifying pipeline stages, normalization rules, and quality metrics.

---

## Tech Stack

- **Base Model:** Qwen2.5-Coder-7B-Instruct (7.61B parameters)
- **Frameworks & Libraries:**
  - [PyTorch](https://pytorch.org/) & [Hugging Face Transformers](https://github.com/huggingface/transformers)
  - [PEFT](https://github.com/huggingface/peft) (Parameter-Efficient Fine-Tuning)
  - [TRL](https://github.com/huggingface/trl) (Transformer Reinforcement Learning)
  - [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) (NF4 4-bit Quantization)
  - [Accelerate](https://github.com/huggingface/accelerate)
  - [Soup CLI](libs/soup/README.md) (v0.73.3) for local layer streaming and training orchestration
  - [PyGithub](https://github.com/PyGithub/PyGithub) for mining bug-fix commits
  - [Pytest](https://docs.pytest.org/) for automated test verification

---

## Repository Structure

```
Sable-7B/
├── .gitignore                   # Curated ignore rules for data, weights, and caches
├── LICENSE                      # MIT License
├── README.md                    # Project documentation and guide
├── docs/                        # Complete technical specifications (12 modules)
│   ├── 01-project-spec.md
│   ├── 02-model-selection.md
│   ├── 03-dataset-strategy.md
│   ├── 04-dataset-sources.md
│   ├── 05-jsonl-format.md
│   ├── 06-data-cleaning-pipeline.md
│   ├── 07-qlora-training-plan.md
│   ├── 08-hardware-and-storage.md
│   ├── 09-evaluation-strategy.md
│   ├── 10-experiment-logging.md
│   ├── 11-licensing-and-security.md
│   ├── 12-development-roadmap.md
│   └── README.md
├── libs/
│   └── soup/                    # Vendored Soup CLI (v0.73.3, Apache-2.0)
└── sable-7b/
    ├── requirements.txt         # Python dependencies
    ├── configs/
    │   ├── smoke_test.yaml      # 1-epoch VRAM smoke test config
    │   └── soup.yaml            # Full QLoRA training config
    ├── data/
    │   ├── raw/                 # Downloaded raw datasets (gitignored)
    │   ├── processed/           # Cleaned train.jsonl / val.jsonl (gitignored)
    │   └── SOURCES.md           # Dataset sources and licensing ledger
    ├── experiments/
    │   ├── experiment_log.md    # Training run logs and hyperparameter notes
    │   └── CURRENT_BEST.md      # Best checkpoint tracking
    ├── kaggle/
    │   ├── kernel-metadata.json # Kaggle API kernel configuration
    │   └── train.ipynb          # Kaggle T4 training notebook
    ├── output/                  # Checkpoint and LoRA adapter outputs (gitignored)
    ├── scripts/
    │   ├── clean/               # 10-stage data cleaning & dataset validator
    │   │   ├── clean_pipeline.py
    │   │   └── validate_dataset.py
    │   ├── collect/             # Data mining scripts
    │   │   ├── collect_github_bugfix.py
    │   │   └── collect_hf_datasets.py
    │   └── eval/                # Evaluation scripts placeholder
    └── tests/
        └── test_pipeline.py     # Automated unit tests for data pipeline
```

---

## Prerequisites

- **OS:** Linux, macOS, or Windows 10/11
- **Python:** 3.10 to 3.12
- **Hardware (Training):**
  - NVIDIA GPU with ≥6 GB VRAM (e.g. RTX 4050 Laptop with layer streaming enabled), or cloud GPU (Kaggle Tesla T4, Google Colab, RunPod).
  - Minimum 16 GB system RAM.
  - ~150–200 GB available disk space for datasets and checkpoint caching.
- **Hardware (Data Preprocessing):** Any standard multi-core CPU.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nfsprogramming/Sable-7B.git
   cd Sable-7B
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv

   # On Windows (PowerShell):
   .venv\Scripts\Activate.ps1

   # On Linux / macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r sable-7b/requirements.txt
   ```

4. **(Optional) Install Soup CLI:**
   ```bash
   pip install soup-cli[train]
   ```

---

## Usage Guide

### 1. Collect Raw Training Data

Collect permissively licensed coding datasets from Hugging Face Hub:
```bash
python sable-7b/scripts/collect/collect_hf_datasets.py --output sable-7b/data/raw/hf_datasets
```

Collect bug-fix commit pairs from public GitHub repositories:
```bash
# Requires GITHUB_TOKEN environment variable for higher API rate limits
$env:GITHUB_TOKEN="your_token_here"  # Windows PowerShell
# export GITHUB_TOKEN="your_token_here"  # Linux / macOS

python sable-7b/scripts/collect/collect_github_bugfix.py --output sable-7b/data/raw/github_bugfix --max-commits 5000
```

### 2. Clean & Deduplicate Data

Execute the 10-stage cleaning pipeline:
```bash
python sable-7b/scripts/clean/clean_pipeline.py \
    --input sable-7b/data/raw \
    --output sable-7b/data/processed \
    --min-quality 0.4
```

### 3. Validate Processed Dataset

Inspect token length distributions and task breakdown:
```bash
python sable-7b/scripts/clean/validate_dataset.py sable-7b/data/processed/train.jsonl --samples 3
```

### 4. Run Smoke Test (Local Hardware)

Verify VRAM fit on local hardware without saving heavy checkpoints:
```bash
cd sable-7b
soup train --config configs/smoke_test.yaml
```

### 5. Cloud Fine-Tuning on Kaggle

To train using a free Kaggle Tesla T4 GPU:
1. Navigate to `sable-7b/kaggle/`.
2. Push or upload `train.ipynb` with `kernel-metadata.json`.
3. Enable GPU accelerator and Internet access in the Kaggle sidebar.
4. Run the notebook to train and export LoRA adapters.

---

## Running Tests

Run the automated test suite to verify pipeline stages, deduplication, and parsing:

```bash
pytest sable-7b/tests
```

---

## Status & Limitations

- **Current Status:** 🚧 Active development — data collection and cleaning pipelines are functional. Model training workflows have undergone smoke testing; no production model weights are packaged in this Git repository.
- **Hardware Requirement:** Fine-tuning an 7.61B model locally on a 6 GB VRAM card requires 4-bit NF4 quantization and layer streaming (`stream_layers: true`). Context length is budgeted between 1024 and 2048 tokens.
- **Evaluation Sandbox:** Any code execution during evaluation must be conducted in an isolated, sandboxed environment without host network access.

---

## Contributing

Contributions, bug reports, and dataset suggestions are welcome.
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/dataset-enhancement`).
3. Ensure all tests pass (`pytest sable-7b/tests`).
4. Commit your changes and open a Pull Request.

---

## License

- **Sable-7B Codebase & Documentation:** [MIT License](LICENSE) &copy; 2026 NFS Programming.
- **Vendored Dependencies:** `libs/soup` is licensed under the [Apache-2.0 License](libs/soup/LICENSE).
- **Base Model Weights:** Subject to the [Qwen Research License Agreement](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct).
