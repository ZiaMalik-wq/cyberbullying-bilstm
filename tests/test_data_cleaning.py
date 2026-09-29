"""
tests/test_data_cleaning.py
===========================

Unit and integration tests for src/data_cleaning.apply_policy_d.

Unit tests use synthetic in-memory DataFrames.
The integration test loads the actual raw dataset if available.

Run with:
    python -m pytest tests/test_data_cleaning.py -v
"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from src.data_cleaning import apply_policy_d

TEXT_COL = "tweet_text"
LABEL_COL = "cyberbullying_type"
RAW_PATH = (
    pathlib.Path(__file__).parent.parent / "data" / "raw" / "cyberbullying_tweets.csv"
)

# Expected row count from the prior quantitative analysis (Policy D)
EXPECTED_FINAL_ROW_COUNT = 44_378


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_df(rows: list[tuple[str, str]]) -> pd.DataFrame:
    """Create a minimal DataFrame from (text, label) tuples."""
    return pd.DataFrame(rows, columns=[TEXT_COL, LABEL_COL])


# ---------------------------------------------------------------------------
# 1. Conflicting tweet texts are completely removed
# ---------------------------------------------------------------------------

def test_conflicting_texts_are_removed() -> None:
    """All rows for a tweet_text with >1 label must be absent from output."""
    df = make_df([
        ("tweet_conflict", "age"),
        ("tweet_conflict", "gender"),   # same text, different label → conflict
        ("tweet_safe", "religion"),
    ])
    result = apply_policy_d(df, TEXT_COL, LABEL_COL)
    assert "tweet_conflict" not in result[TEXT_COL].values, (
        "Conflicting tweet text must not appear in cleaned output."
    )


def test_conflicting_both_copies_removed() -> None:
    """Both copies of a conflicting text must be removed, not just one."""
    df = make_df([
        ("tweet_A", "age"),
        ("tweet_A", "gender"),
        ("tweet_B", "religion"),
    ])
    result = apply_policy_d(df, TEXT_COL, LABEL_COL)
    # tweet_A had 2 rows; both must be gone
    assert len(result[result[TEXT_COL] == "tweet_A"]) == 0


def test_non_conflicting_texts_preserved() -> None:
    """Texts that never appear with a conflicting label must be retained."""
    df = make_df([
        ("tweet_conflict", "age"),
        ("tweet_conflict", "gender"),
        ("tweet_safe", "religion"),
    ])
    result = apply_policy_d(df, TEXT_COL, LABEL_COL)
    assert "tweet_safe" in result[TEXT_COL].values


# ---------------------------------------------------------------------------
# 2. Consistent duplicates collapse to one row
# ---------------------------------------------------------------------------

def test_consistent_duplicate_collapses_to_one() -> None:
    """A tweet appearing twice with the same label must appear exactly once."""
    df = make_df([
        ("tweet_dup", "age"),
        ("tweet_dup", "age"),   # consistent duplicate
        ("tweet_unique", "gender"),
    ])
    result = apply_policy_d(df, TEXT_COL, LABEL_COL)
    count = (result[TEXT_COL] == "tweet_dup").sum()
    assert count == 1, f"Expected 1 row for consistent duplicate, got {count}."


def test_consistent_duplicate_label_preserved() -> None:
    """The label retained for a consistent duplicate must be correct."""
    df = make_df([
        ("tweet_dup", "religion"),
        ("tweet_dup", "religion"),
    ])
    result = apply_policy_d(df, TEXT_COL, LABEL_COL)
    assert result[result[TEXT_COL] == "tweet_dup"][LABEL_COL].iloc[0] == "religion"


# ---------------------------------------------------------------------------
# 3. Retained tweet text is unchanged
# ---------------------------------------------------------------------------

def test_retained_tweet_text_unchanged() -> None:
    """The tweet_text value for retained rows must be exactly the original string."""
    original_text = "Hello World! #test @user 123 — no cleaning applied"
    df = make_df([
        (original_text, "gender"),
        ("conflict_text", "age"),
        ("conflict_text", "ethnicity"),
    ])
    result = apply_policy_d(df, TEXT_COL, LABEL_COL)
    retained = result[result[TEXT_COL] == original_text]
    assert len(retained) == 1
    assert retained[TEXT_COL].iloc[0] == original_text, (
        "tweet_text must not be modified by Policy D."
    )


def test_special_characters_in_text_unchanged() -> None:
    """URLs, @mentions, #hashtags, emojis must pass through unmodified."""
    special = "RT @user: Check https://example.com #tag 😊"
    df = make_df([(special, "not_cyberbullying")])
    result = apply_policy_d(df, TEXT_COL, LABEL_COL)
    assert result[TEXT_COL].iloc[0] == special


# ---------------------------------------------------------------------------
# 4. Labels of retained rows remain unchanged
# ---------------------------------------------------------------------------

def test_labels_of_retained_rows_unchanged() -> None:
    """Labels of non-conflicting rows must not be altered."""
    df = make_df([
        ("safe_tweet", "other_cyberbullying"),
        ("conflict", "age"),
        ("conflict", "gender"),
    ])
    result = apply_policy_d(df, TEXT_COL, LABEL_COL)
    row = result[result[TEXT_COL] == "safe_tweet"]
    assert row[LABEL_COL].iloc[0] == "other_cyberbullying"


# ---------------------------------------------------------------------------
# 5. Output contains no duplicate tweet_text values
# ---------------------------------------------------------------------------

def test_no_duplicate_tweet_texts_in_output() -> None:
    """After Policy D, no tweet_text value should appear more than once."""
    df = make_df([
        ("dup_consistent", "age"),
        ("dup_consistent", "age"),
        ("dup_conflict", "gender"),
        ("dup_conflict", "religion"),
        ("unique", "ethnicity"),
    ])
    result = apply_policy_d(df, TEXT_COL, LABEL_COL)
    n_dups = result[TEXT_COL].duplicated().sum()
    assert n_dups == 0, f"Expected 0 duplicate tweet_texts, found {n_dups}."


# ---------------------------------------------------------------------------
# 6. Output contains no conflicting labels
# ---------------------------------------------------------------------------

def test_no_conflicting_labels_in_output() -> None:
    """No tweet_text in the output should map to more than one label."""
    df = make_df([
        ("conflict_a", "age"),
        ("conflict_a", "gender"),
        ("conflict_b", "religion"),
        ("conflict_b", "not_cyberbullying"),
        ("clean", "ethnicity"),
        ("clean_dup", "ethnicity"),
        ("clean_dup", "ethnicity"),
    ])
    result = apply_policy_d(df, TEXT_COL, LABEL_COL)
    label_nunique = result.groupby(TEXT_COL)[LABEL_COL].nunique()
    n_conflicts = (label_nunique > 1).sum()
    assert n_conflicts == 0, (
        f"Expected 0 conflicting texts, found {n_conflicts}."
    )


# ---------------------------------------------------------------------------
# 7. Error handling
# ---------------------------------------------------------------------------

def test_raises_on_missing_text_column() -> None:
    df = pd.DataFrame({"wrong_col": ["a"], LABEL_COL: ["age"]})
    with pytest.raises(ValueError, match="tweet_text"):
        apply_policy_d(df, TEXT_COL, LABEL_COL)


def test_raises_on_missing_label_column() -> None:
    df = pd.DataFrame({TEXT_COL: ["a"], "wrong_label": ["age"]})
    with pytest.raises(ValueError, match="cyberbullying_type"):
        apply_policy_d(df, TEXT_COL, LABEL_COL)


def test_empty_dataframe_returns_empty() -> None:
    df = pd.DataFrame(columns=[TEXT_COL, LABEL_COL])
    result = apply_policy_d(df, TEXT_COL, LABEL_COL)
    assert len(result) == 0


def test_all_unique_rows_unchanged() -> None:
    """A DataFrame with no duplicates at all must be returned unchanged."""
    df = make_df([
        ("tweet_1", "age"),
        ("tweet_2", "gender"),
        ("tweet_3", "religion"),
    ])
    result = apply_policy_d(df, TEXT_COL, LABEL_COL)
    assert len(result) == 3
    assert set(result[TEXT_COL]) == {"tweet_1", "tweet_2", "tweet_3"}


# ---------------------------------------------------------------------------
# 8. Integration test — expected final row count on actual raw dataset
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not RAW_PATH.exists(),
    reason="Raw dataset not available at data/raw/cyberbullying_tweets.csv",
)
def test_expected_row_count_on_real_data() -> None:
    """Policy D applied to the actual raw dataset must yield exactly 44,378 rows."""
    df = pd.read_csv(RAW_PATH, encoding="utf-8")
    result = apply_policy_d(df, TEXT_COL, LABEL_COL)
    assert len(result) == EXPECTED_FINAL_ROW_COUNT, (
        f"Expected {EXPECTED_FINAL_ROW_COUNT:,} rows after Policy D, "
        f"got {len(result):,}."
    )


@pytest.mark.skipif(
    not RAW_PATH.exists(),
    reason="Raw dataset not available at data/raw/cyberbullying_tweets.csv",
)
def test_no_duplicates_on_real_data() -> None:
    """After Policy D on real data, no tweet_text duplicates must remain."""
    df = pd.read_csv(RAW_PATH, encoding="utf-8")
    result = apply_policy_d(df, TEXT_COL, LABEL_COL)
    n_dups = result[TEXT_COL].duplicated().sum()
    assert n_dups == 0, f"Expected 0 duplicate tweet_texts, found {n_dups}."


@pytest.mark.skipif(
    not RAW_PATH.exists(),
    reason="Raw dataset not available at data/raw/cyberbullying_tweets.csv",
)
def test_no_conflicting_labels_on_real_data() -> None:
    """After Policy D on real data, no conflicting labels must remain."""
    df = pd.read_csv(RAW_PATH, encoding="utf-8")
    result = apply_policy_d(df, TEXT_COL, LABEL_COL)
    label_nunique = result.groupby(TEXT_COL)[LABEL_COL].nunique()
    n_conflicts = int((label_nunique > 1).sum())
    assert n_conflicts == 0, (
        f"Expected 0 conflicting texts after Policy D, found {n_conflicts}."
    )


@pytest.mark.skipif(
    not RAW_PATH.exists(),
    reason="Raw dataset not available at data/raw/cyberbullying_tweets.csv",
)
def test_class_counts_on_real_data() -> None:
    """Per-class counts after Policy D must match previously verified values."""
    expected_counts = {
        "age": 7992,
        "ethnicity": 7952,
        "gender": 7772,
        "not_cyberbullying": 6428,
        "other_cyberbullying": 6243,
        "religion": 7991,
    }
    df = pd.read_csv(RAW_PATH, encoding="utf-8")
    result = apply_policy_d(df, TEXT_COL, LABEL_COL)
    actual_counts = result[LABEL_COL].value_counts().to_dict()
    for label, expected in expected_counts.items():
        actual = actual_counts.get(label, 0)
        assert actual == expected, (
            f"Class [{label}]: expected {expected:,}, got {actual:,}."
        )
