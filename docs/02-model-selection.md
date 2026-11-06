# Model Selection

## Chosen Base: Qwen2.5-Coder-7B-Instruct

### Why this model

- Purpose-built for code (not a general LLM retrofitted for code).
- Strong baseline performance on code generation, completion, and repair benchmarks.
- 7B size is the sweet spot for QLoRA fine-tuning on a 6 GB VRAM consumer GPU.
- Instruct-tuned already, so it understands chat/agentic instruction formats — fine-tuning only needs to *specialize* behavior, not teach instruction-following from zero.
- Wide language coverage overlapping the target stack (Python, JS/TS, Java, C/C++, Go, Rust, Dart, Kotlin).

### Alternatives considered (and why not used for v1)

| Model | Reason not chosen for v1 |
|---|---|
| Qwen2.5-Coder-14B/32B | Exceeds 6 GB VRAM budget even with 4-bit QLoRA at usable context lengths |
| Qwen2.5-Coder-1.5B/3B | Too small — insufficient reasoning depth for repo-level/agentic tasks |
| DeepSeek-Coder-6.7B | Comparable class, but Qwen2.5-Coder benchmarks edge it out on recent code-repair tasks |
| CodeLlama-7B/13B | Older architecture, generally weaker on current code benchmarks |
| General-purpose LLMs (Llama-3-8B-Instruct, Mistral-7B) | Not code-specialized at the base level; would need far more data to reach the same coding competence |

### Future path

Once v1 (7B) is validated end-to-end, consider scaling to a 14B variant if:
- A GPU with more VRAM becomes available, or
- Cloud/rented GPU time is used for training only (inference still local/quantized).
