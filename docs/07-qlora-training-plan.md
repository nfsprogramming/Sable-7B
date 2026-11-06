# QLoRA Training Plan

## Method Summary

- **Base model:** Qwen2.5-Coder-7B-Instruct, loaded in 4-bit (NF4) quantization.
- **Fine-tuning method:** LoRA adapters on top of the frozen 4-bit base (QLoRA).
- **Post-training:** "Soup" — train a few LoRA runs with varied seeds/data order/hyperparameters, then average the resulting adapter weights (or merged model weights) to reduce variance and improve robustness.

## Suggested LoRA configuration (starting point)

- Rank (r): 16 (increase to 32 only if quality plateaus and VRAM allows)
- Alpha: 32
- Target modules: attention projection layers (q_proj, k_proj, v_proj, o_proj) + optionally MLP projections if VRAM allows
- Dropout: 0.05
- Precision: bf16/fp16 compute with 4-bit NF4 base weights

## RTX 4050 6GB constraints

- **Batch size:** 1 (use gradient accumulation to simulate larger effective batch size, e.g., accumulate 8–16 steps)
- **Context length:** Start at 1024–2048 tokens; only increase if VRAM headroom allows after profiling. Repo-level examples may need chunking to fit.
- **Gradient checkpointing:** Enabled (trades compute for memory)
- **Optimizer:** Paged AdamW 8-bit (bitsandbytes) to reduce optimizer state memory
- **Mixed precision:** bf16 if supported, else fp16

## Training loop plan

1. Smoke test: train on a tiny subset (a few hundred examples) for a few steps to confirm the pipeline runs end-to-end without OOM.
2. Phase 1 run: train on the ~5 GB curated dataset (see `03-dataset-strategy.md`).
3. Evaluate against baseline (see `09-evaluation-strategy.md`).
4. If improved: proceed to Phase 2 (10–20 GB) with the same recipe, adjusting only what the eval results indicate needs adjusting.
5. Run 2–4 independent LoRA training runs (varied seed/data shuffling) for the "Soup" step.
6. Average adapter weights (or merge each into the base separately and average the merged weights) to produce the final model.

## Checkpointing strategy

- Save adapter-only checkpoints (small, a few hundred MB) at regular intervals rather than full merged models, to save disk space during training.
- Only merge LoRA into the base model for the final candidate(s) selected after evaluation.

## Estimated training time

Highly dependent on dataset size and context length; expect multi-day wall-clock time for Phase 1 on a single 6 GB laptop GPU. Plan for overnight/background runs rather than interactive iteration.
