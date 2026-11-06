# Experiment Logging

## Why

With limited hardware and long training runs, every run is expensive — good logging prevents repeating mistakes and enables reliable comparison across runs.

## What to log per run

- **Run ID** (date + short hash)
- Base model version/commit
- Dataset version (which Phase, which cleaning pipeline version)
- LoRA hyperparameters (rank, alpha, target modules, dropout)
- Training hyperparameters (learning rate, batch size, gradient accumulation steps, context length, epochs/steps, optimizer)
- Hardware/environment (driver version, CUDA version, bitsandbytes version, transformers/peft versions)
- Training loss curve (saved, not just final value)
- Validation loss curve
- Wall-clock training time
- Final checkpoint location + size
- Evaluation results (per task_type, per language, vs. base model) — link to `09-evaluation-strategy.md` results
- Notes: anything unusual observed (OOM near the end, loss spikes, data issues discovered mid-run)

## Suggested format

A simple `experiments/experiment_log.md` (or `.csv`/JSONL for structured querying) with one entry per run, plus a `experiments/<run_id>/` folder containing:
- config file used (exact hyperparameters, versioned)
- loss curve plot/data
- eval results file
- notes.md

## Versioning discipline

- Never overwrite a previous run's config or logs — always create a new run ID.
- Tag the dataset version used for each run so results are reproducible.
- Keep a single `CURRENT_BEST.md` pointer file noting which run/checkpoint is currently the best candidate and why.
