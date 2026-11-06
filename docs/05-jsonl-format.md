# Training Data Format (JSONL)

All processed training data is stored as JSONL (one JSON object per line) for streaming-friendly loading.

## Standard schema

```json
{
  "id": "unique-example-id",
  "task_type": "debugging | refactoring | test_generation | code_review | repo_context | agentic | reasoning",
  "language": "python | javascript | typescript | java | c | cpp | go | rust | dart | kotlin",
  "source": "real_world | synthetic",
  "source_ref": "e.g. github repo + commit sha, or 'synthetic-gen-v1'",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": {
    "num_files": 1,
    "token_count_estimate": 850,
    "difficulty": "easy | medium | hard",
    "license": "MIT | Apache-2.0 | ..."
  }
}
```

## Notes on fields

- **messages**: Uses the standard chat format matching Qwen2.5-Coder-Instruct's chat template. Multi-turn agentic examples include multiple user/assistant/tool turns.
- **task_type**: Used for stratified sampling during training and for per-category evaluation later.
- **source**: Critical for tracing whether quality issues stem from real vs. synthetic data.
- **license**: Tracked per-example for auditability, even if filtering already happened upstream.

## Repo-level / multi-file example convention

For repository-level tasks, concatenate relevant files in the `user` turn with clear delimiters:

```
### File: src/utils/parser.py
<file content>

### File: src/main.py
<file content>

### Task:
<instruction, e.g. "Fix the bug causing IndexError in parser.py">
```

## Agentic transcript convention

Represent each step as an alternating sequence:

```
user: <task>
assistant: <plan / thought>
assistant (tool_call): <action>
tool: <observation/result>
assistant: <next step or final answer>
```

Tool calls should follow a consistent structured format (e.g., a JSON action block) so the model learns a predictable agentic interface.

## Splits

- `train.jsonl`
- `val.jsonl` (held out, ~2–5% of data, stratified by task_type)
- `test.jsonl` (held out, never touched until final eval — see `09-evaluation-strategy.md`)
