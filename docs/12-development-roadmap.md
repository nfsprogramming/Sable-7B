# Development Roadmap

## Milestone 0 — Setup
- [ ] Set up local environment (transformers, peft, bitsandbytes, accelerate, trl or equivalent training framework)
- [ ] Download and verify Qwen2.5-Coder-7B-Instruct base weights
- [ ] Confirm QLoRA smoke test runs without OOM on a tiny toy dataset

## Milestone 1 — Data Pipeline (Phase 1, ~5 GB)
- [ ] Implement collection scripts for chosen sources (`04-dataset-sources.md`)
- [ ] Implement cleaning pipeline stages (`06-data-cleaning-pipeline.md`)
- [ ] Produce validated `train.jsonl` / `val.jsonl` / `test.jsonl`
- [ ] Manual spot-check pass

## Milestone 2 — First Training Run
- [ ] Train Phase 1 QLoRA run per `07-qlora-training-plan.md`
- [ ] Log run per `10-experiment-logging.md`
- [ ] Run evaluation per `09-evaluation-strategy.md`
- [ ] Decide: iterate on Phase 1 recipe, or proceed to scale-up

## Milestone 3 — Scale-Up (Phase 2, 10–20 GB) [conditional]
- [ ] Only if Milestone 2 shows clear, non-regressing improvement
- [ ] Expand dataset per `03-dataset-strategy.md`
- [ ] Repeat training + evaluation

## Milestone 4 — Model Soup
- [ ] Run 2–4 varied training runs (seed/order/hyperparameter variation)
- [ ] Average adapters into a single "souped" model
- [ ] Evaluate souped model against best individual run

## Milestone 5 — Packaging
- [ ] Merge final LoRA adapter into base weights (if desired for deployment simplicity)
- [ ] Re-quantize final merged model for local inference
- [ ] Write model card documenting training data summary, intended use, known limitations
- [ ] Confirm licensing compliance per `11-licensing-and-security.md`

## Milestone 6 — Local Deployment
- [ ] Set up local inference (e.g., via a local serving framework compatible with the quantized format)
- [ ] Integrate into coding workflow/agent harness for real-world testing
- [ ] Collect real usage feedback to inform any future v2 dataset additions

## Stretch Goals (post-v1)
- [ ] Explore 14B base model if hardware/compute access improves
- [ ] Expand agentic capability with real tool-use environments (not just static transcripts)
- [ ] Continual/incremental fine-tuning pipeline as new curated data becomes available
