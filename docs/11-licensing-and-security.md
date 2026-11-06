# Licensing & Security

## Base model license

- Confirm and record the current Qwen2.5-Coder-7B-Instruct license terms (check the model card at time of download — license terms can vary by release/version).
- Verify whether the license permits fine-tuning and redistribution of derivative weights for your intended use (personal/local use vs. any future sharing/publishing).

## Training data licensing

- Every source repo/dataset must have its license recorded (see `04-dataset-sources.md`, `metadata.license` field in `05-jsonl-format.md`).
- Avoid copyleft-licensed code (e.g., GPL) if there is any intent to distribute the fine-tuned model or its outputs, unless legal implications are understood and accepted.
- When in doubt, exclude rather than include — a smaller, clean dataset is safer than a larger, legally ambiguous one.

## Security practices

- **Secret scanning**: mandatory pipeline stage (see `06-data-cleaning-pipeline.md`) — scan all collected code for API keys, tokens, passwords, private keys before they enter the training set.
- **No PII**: strip personal information (names/emails in commit metadata) from training examples where not functionally necessary.
- **Sandboxing**: if executing untrusted code (e.g., to verify tests pass/fail during data cleaning or evaluation), always run in an isolated sandbox/container with no network access and resource limits.
- **Model output caution**: since this is an agentic coding model, treat its generated code and tool-call actions as untrusted output requiring review before execution in any privileged environment, especially during early evaluation phases.

## Documentation & auditability

- Keep a `SOURCES.md` ledger listing every external dataset/repo used, its license, and date accessed — useful both for compliance and for reproducibility.
