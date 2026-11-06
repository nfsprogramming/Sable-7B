# Hardware & Storage Requirements

## Target machine

- GPU: NVIDIA RTX 4050 (6 GB VRAM)
- CPU: Intel i7-14700HX
- RAM: 16 GB
- Storage: 1 TB SSD

## Storage budget breakdown

| Item | Estimated size |
|---|---|
| Base 7B model (fp16/original weights) | ~15 GB |
| Quantized (4-bit) model | ~5–8 GB |
| Curated training dataset | ~10–20 GB |
| Training checkpoints (adapters, intermediate) | ~20–50 GB |
| Processed/cache data (tokenized, intermediate pipeline artifacts) | ~20–40 GB |
| Final model + adapters | ~5–15 GB |
| Temporary files / scratch space | ~20 GB |
| **Recommended free storage reserved** | **150–200 GB** |

## Guidance

- 150–200 GB reserved is sufficient for the Phase 1 (~5 GB dataset) and Phase 2 (10–20 GB dataset) plan — no need to reserve hundreds of GB upfront.
- Keep raw data, processed data, and checkpoints on separate subfolders/volumes if possible for easier cleanup between experiments.
- Periodically prune old/inferior checkpoints once an experiment's best checkpoint is identified (see `10-experiment-logging.md`).

## VRAM feasibility notes

- 6 GB VRAM is workable for QLoRA fine-tuning of a 7B model **only** with:
  - 4-bit base quantization
  - Batch size 1 with gradient accumulation
  - Reduced context length (1024–2048 tokens to start)
  - Gradient checkpointing enabled
  - 8-bit paged optimizer states
- Attempting larger batch sizes, longer context, or a bigger base model (14B+) on this GPU is not realistic without offloading strategies that would significantly slow training.
