# Sable-7B — Project Specification

## 1. Goal

Fine-tune a **specialist Software Engineering / Coding Agent model** — not a general-purpose chatbot.

The model should be exceptional at:

- 🐛 Debugging & bug fixing
- 🏗️ Repository-level coding (multi-file context, project structure awareness)
- 🔧 Refactoring
- 🧪 Test generation & test fixing
- 🧠 Code reasoning (explaining *why* code fails, tracing logic)
- 👀 Code review (spotting issues, suggesting improvements)
- 🤖 Agentic software engineering (tool use, multi-step task execution, plan-then-act workflows)

Target languages: **Python, JavaScript/TypeScript, Java, C/C++, Go, Rust, Dart, Kotlin**

## 2. Non-Goals

- General world knowledge / trivia
- Creative writing
- Long, unrelated conversational chat
- Multilingual (natural language) support beyond English code comments/docs

The explicit philosophy: **narrow and deep, not broad and shallow.**

## 3. Base Model

- **Model:** Qwen2.5-Coder-7B-Instruct
- **Parameters:** 7.61B
- Pretrained specifically for code generation, reasoning, and code fixing — a strong base for further specialization rather than teaching coding ability from scratch.

## 4. Training Method

- **Technique:** QLoRA (4-bit quantized base + LoRA adapters)
- **Post-training merge strategy:** "Soup" — averaging weights across multiple LoRA checkpoints/runs to improve generalization and reduce variance from any single run.

## 5. Target Hardware

- GPU: NVIDIA RTX 4050 (6 GB VRAM)
- RAM: 16 GB
- CPU: Intel i7-14700HX
- Storage: 1 TB SSD (reserve ~150–200 GB for this project)

## 6. Definition of Done (v1)

A first version is considered successful if it:

1. Outperforms the base Qwen2.5-Coder-7B-Instruct on a held-out debugging/refactoring eval set.
2. Maintains coherent multi-file / repo-level context handling.
3. Does not regress general code-generation quality vs. the base model.
4. Runs inference locally within available VRAM (quantized).

See `05-qlora-training-plan.md` and `09-evaluation-strategy.md` for details.
