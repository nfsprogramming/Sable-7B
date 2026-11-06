# Evaluation Strategy

## Principle

Evaluate against the **base model** (Qwen2.5-Coder-7B-Instruct, unmodified) on every axis the fine-tune targets. A fine-tune is only worth keeping if it improves targeted skills **without regressing** general coding ability.

## Held-out test set

- Constructed from `test.jsonl` (see `05-jsonl-format.md`) — never used in training or hyperparameter selection.
- Stratified across all seven task types and all eight target languages.

## Per-skill evaluation

| Skill | Evaluation approach |
|---|---|
| Debugging & bug fixing | Does the model's fix actually resolve the bug? Run against test cases where available; otherwise use model/human-graded correctness |
| Repository-level coding | Multi-file consistency check — does generated code correctly reference/import across the provided files? |
| Refactoring | Behavior preservation (existing tests still pass) + readability/quality scoring |
| Test generation/fixing | Generated tests actually run and meaningfully cover the target function; fixed tests pass |
| Code reasoning | Human/model-graded rubric on explanation correctness and clarity |
| Code review | Overlap/quality comparison against real human review comments held out from training |
| Agentic SWE | Task completion rate on multi-step held-out agentic scenarios |

## Regression checks

- Run standard public code benchmarks (e.g., a subset of HumanEval/MBPP-style problems, if licensing allows local use) before and after fine-tuning to catch generic capability regression.

## Comparison protocol

1. Run identical prompts through base model and fine-tuned model.
2. Blind pairwise comparison where feasible (grader doesn't know which output is which).
3. Track win/tie/loss rate per task_type, not just an aggregate score — a model can improve overall while regressing on one skill.

## Promotion criteria

A candidate checkpoint is promoted to "current best" only if it:
- Wins or ties on every task_type category (no silent regressions), and
- Wins outright on at least the primary targeted skills for that training phase.
