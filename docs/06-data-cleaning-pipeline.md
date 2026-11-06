# Data Cleaning & Processing Pipeline

## Pipeline stages

1. **Raw collection**
   - Pull from sources listed in `04-dataset-sources.md`.
   - Store raw, unprocessed data separately from cleaned data (never overwrite raw).

2. **License filtering**
   - Drop anything without a clear permissive license.
   - Record license per source in metadata.

3. **Secret scanning**
   - Run a secrets-detection pass (API keys, tokens, credentials, private keys) and reject/redact matches.

4. **Deduplication**
   - Exact-match hash dedup first (cheap).
   - Near-duplicate dedup via embedding similarity (catch copy-pasted/boilerplate variants).

5. **Quality filtering**
   - Remove auto-generated/minified/vendored files.
   - Remove examples with broken syntax (parse-check per language).
   - Remove examples below a minimum meaningful length (trivial one-liners) unless specifically targeting a skill that needs them.

6. **Task-type labeling**
   - Rule-based labeling first (commit message patterns, file diff shape).
   - Model-assisted labeling/verification for ambiguous cases.

7. **Format conversion**
   - Convert into the JSONL schema defined in `05-jsonl-format.md`.
   - Validate every record against the schema (reject malformed records rather than silently coercing).

8. **Tokenization dry-run**
   - Tokenize with the target tokenizer to estimate token counts and flag examples exceeding context length (truncate, split, or drop per policy).

9. **Split generation**
   - Stratified train/val/test split by `task_type` and `language` to keep distributions consistent across splits.

10. **Manual spot-check**
    - Randomly sample ~50–100 examples per task_type for human review before large-scale training runs.

## Tooling suggestions

- Use streaming/chunked processing (don't load the full dataset into RAM at once) given the 16 GB RAM constraint.
- Log pipeline stage statistics (records in → records out) at every stage for auditability.
