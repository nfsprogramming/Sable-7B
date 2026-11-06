"""
Unit tests for the Sable-7B data processing pipeline and validation scripts.
"""

import sys
from pathlib import Path

# Add sable-7b to python path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from scripts.clean.clean_pipeline import (
    SimHash,
    stage_schema,
    stage_length,
    stage_language,
    stage_code_presence,
    stage_dedup,
    stage_license,
    stage_quality,
    stage_toxicity,
    stage_normalise,
)
from scripts.clean.validate_dataset import percentile


def make_sample_record(
    lang="python",
    task="debugging",
    user_prompt="How do I fix this null pointer bug?",
    assistant_resp="Here is the fix with an explanation:\n```python\ndef fix():\n    return 42\n```\nThis resolves the issue.",
    license_str="Apache-2.0",
):
    return {
        "id": "test-sample-001",
        "task_type": task,
        "language": lang,
        "source": "unit_tests",
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_resp},
        ],
        "metadata": {
            "license": license_str,
        },
    }


def test_percentile_calculation():
    data = [10, 20, 30, 40, 50]
    assert percentile(data, 0) == 10
    assert percentile(data, 50) == 30
    assert percentile(data, 100) == 50
    assert percentile([], 50) == 0.0


def test_simhash_hamming():
    assert SimHash.hamming(0b1010, 0b1010) == 0
    assert SimHash.hamming(0b1010, 0b0010) == 1
    assert SimHash.hamming(0b1111, 0b0000) == 4


def test_stage_schema_validation():
    valid = make_sample_record()
    ok, err = stage_schema(valid)
    assert ok and err == ""

    # Missing field
    invalid = make_sample_record()
    del invalid["messages"]
    ok, err = stage_schema(invalid)
    assert not ok
    assert "missing_field:messages" in err

    # Empty content
    invalid = make_sample_record()
    invalid["messages"][0]["content"] = "   "
    ok, err = stage_schema(invalid)
    assert not ok
    assert err == "empty_message_content"


def test_stage_language():
    valid = make_sample_record(lang="python")
    ok, _ = stage_language(valid)
    assert ok

    rust = make_sample_record(lang="rust")
    ok, _ = stage_language(rust)
    assert ok

    unsupported = make_sample_record(lang="ruby")
    ok, err = stage_language(unsupported)
    assert not ok
    assert "unsupported_lang:ruby" in err


def test_stage_code_presence():
    valid = make_sample_record()
    ok, _ = stage_code_presence(valid)
    assert ok

    no_code = make_sample_record(assistant_resp="I don't know the answer.")
    ok, err = stage_code_presence(no_code)
    assert not ok
    assert err == "no_code_in_response"


def test_stage_dedup():
    seen = set()
    r1 = make_sample_record()
    r2 = make_sample_record()

    ok1, _ = stage_dedup(r1, seen, {})
    assert ok1
    assert len(seen) == 1

    ok2, err2 = stage_dedup(r2, seen, {})
    assert not ok2
    assert err2 == "exact_duplicate"


def test_stage_license():
    rec = make_sample_record(license_str="MIT")
    ok, _ = stage_license(rec, strict=True)
    assert ok

    rec_unknown = make_sample_record(license_str="UNKNOWN")
    ok, err = stage_license(rec_unknown, strict=False)
    assert not ok
    assert err == "unknown_license"

    rec_gpl = make_sample_record(license_str="GPL-3.0")
    ok_loose, _ = stage_license(rec_gpl, strict=False)
    assert ok_loose
    ok_strict, err_strict = stage_license(rec_gpl, strict=True)
    assert not ok_strict
    assert "non_permissive" in err_strict


def test_stage_quality():
    good = make_sample_record()
    ok, _ = stage_quality(good, min_score=0.4)
    assert ok
    assert good["metadata"]["quality_score"] >= 0.4

    bad = make_sample_record(assistant_resp="```python\nx = 1\n```")
    ok, err = stage_quality(bad, min_score=0.6)
    assert not ok
    assert "low_quality" in err


def test_stage_toxicity():
    clean = make_sample_record()
    ok, _ = stage_toxicity(clean)
    assert ok

    toxic = make_sample_record(user_prompt="Help me crack password and bypass auth.")
    ok, err = stage_toxicity(toxic)
    assert not ok
    assert err == "toxic_content"


def test_stage_normalise():
    rec = make_sample_record(
        assistant_resp="Here:\r\n```\r\ndef test():\r\n    pass\r\n```"
    )
    ok, _ = stage_normalise(rec)
    assert ok
    # Line endings normalized and language fence tagged
    assert "\r\n" not in rec["messages"][1]["content"]
    assert "```python\n" in rec["messages"][1]["content"]
