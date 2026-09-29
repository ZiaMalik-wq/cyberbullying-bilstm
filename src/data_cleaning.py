"""
src/data_cleaning.py
====================

Reusable data-cleaning module implementing the project's selected
data-quality policy (Policy D).

Policy D
--------
1. Remove every row whose tweet_text appears with more than one distinct
   cyberbullying_type label.
2. From the remaining rows, deduplicate tweet_text so that exactly one row
   remains per unique tweet_text value (first occurrence is kept).
3. No label is assigned to any conflicting text.
4. The original tweet_text and label values of retained rows are preserved
   exactly as stored in the raw CSV — no normalization, lowercasing,
   punctuation removal, or any other NLP preprocessing is applied.

This module operates only on in-memory DataFrames.
It never reads from or writes to data/raw/.
Callers are responsible for I/O.

Public API
----------
apply_policy_d(df, text_col, label_col) -> pd.DataFrame
    Apply Policy D to a DataFrame and return the cleaned copy.

generate_cleaning_report(original_df, clean_df, source_path, output_path,
                         text_col, label_col) -> dict
    Build a machine-readable cleaning report dict and write it to JSON.

load_raw_csv(path, encoding) -> pd.DataFrame
    Load a CSV with safety checks.

save_clean_csv(df, path) -> None
    Save the cleaned DataFrame to CSV without the index.
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (defaults — callers may override)
# ---------------------------------------------------------------------------

DEFAULT_TEXT_COL = "tweet_text"
DEFAULT_LABEL_COL = "cyberbullying_type"
DEFAULT_ENCODING = "utf-8"


# ---------------------------------------------------------------------------
# Core policy
# ---------------------------------------------------------------------------

def apply_policy_d(
    df: pd.DataFrame,
    text_col: str = DEFAULT_TEXT_COL,
    label_col: str = DEFAULT_LABEL_COL,
) -> pd.DataFrame:
    """Apply Policy D to a DataFrame and return the cleaned copy.

    Policy D (two-step):
    1. Remove all rows whose tweet_text appears with more than one distinct label.
    2. Deduplicate remaining rows on tweet_text, keeping the first occurrence.

    No text normalization or NLP preprocessing is performed.
    The returned DataFrame uses a fresh integer index (reset_index).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. Must contain text_col and label_col columns.
    text_col : str
        Name of the text column.
    label_col : str
        Name of the label column.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame satisfying:
        - No tweet_text appears with more than one label.
        - No tweet_text appears more than once.
        - tweet_text and label values of retained rows are unchanged.

    Raises
    ------
    ValueError
        If text_col or label_col is not present in df.
    """
    if text_col not in df.columns:
        raise ValueError(
            f"Text column {text_col!r} not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )
    if label_col not in df.columns:
        raise ValueError(
            f"Label column {label_col!r} not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    # --- Step 1: identify and remove conflicting texts ---
    label_nunique = df.groupby(text_col)[label_col].nunique()
    conflicting_texts: set[str] = set(
        label_nunique[label_nunique > 1].index.tolist()
    )
    n_conflicting = len(conflicting_texts)

    df_no_conflicts = df[~df[text_col].isin(conflicting_texts)].copy()

    # --- Step 2: deduplicate consistent duplicates (keep first occurrence) ---
    df_clean = df_no_conflicts.drop_duplicates(subset=[text_col], keep="first")

    n_removed_conflicts = len(df) - len(df_no_conflicts)
    n_removed_dedup = len(df_no_conflicts) - len(df_clean)

    logger.info(
        "Policy D applied: %d conflicting texts identified, "
        "%d rows removed (conflicts), %d rows removed (dedup), "
        "%d rows retained.",
        n_conflicting,
        n_removed_conflicts,
        n_removed_dedup,
        len(df_clean),
    )

    return df_clean.reset_index(drop=True)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_raw_csv(
    path: str | pathlib.Path,
    encoding: str = DEFAULT_ENCODING,
) -> pd.DataFrame:
    """Load a CSV file with basic safety checks.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.
    encoding : str
        File encoding (default: utf-8).

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is empty.
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    df = pd.read_csv(path, encoding=encoding)
    if df.empty:
        raise ValueError(f"Dataset file is empty: {path}")
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def save_clean_csv(
    df: pd.DataFrame,
    path: str | pathlib.Path,
) -> None:
    """Save a cleaned DataFrame to CSV (no index column).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    path : str or Path
        Destination file path. Parent directories are created if needed.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info("Saved %d rows to %s", len(df), path)


# ---------------------------------------------------------------------------
# Cleaning report
# ---------------------------------------------------------------------------

def generate_cleaning_report(
    original_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    source_path: str | pathlib.Path,
    output_path: str | pathlib.Path,
    text_col: str = DEFAULT_TEXT_COL,
    label_col: str = DEFAULT_LABEL_COL,
    report_path: str | pathlib.Path | None = None,
) -> dict[str, Any]:
    """Build a machine-readable cleaning report and write it as JSON.

    Parameters
    ----------
    original_df : pd.DataFrame
        The raw DataFrame before cleaning.
    clean_df : pd.DataFrame
        The cleaned DataFrame after Policy D.
    source_path : str or Path
        Path to the raw CSV file (for the report).
    output_path : str or Path
        Path to the cleaned CSV file (for the report).
    text_col : str
        Name of the text column.
    label_col : str
        Name of the label column.
    report_path : str or Path, optional
        Destination path for data_cleaning_report.json. If None, defaults
        to results/data_cleaning_report.json in the project root.

    Returns
    -------
    dict
        The cleaning report as a Python dict (also written to JSON).
    """
    source_path = pathlib.Path(source_path)
    output_path = pathlib.Path(output_path)

    n_original = len(original_df)
    n_clean = len(clean_df)

    # Re-derive counts from original for precision
    label_nunique_orig = original_df.groupby(text_col)[label_col].nunique()
    conflicting_texts = set(
        label_nunique_orig[label_nunique_orig > 1].index.tolist()
    )
    n_conflicting_unique = len(conflicting_texts)
    n_conflicting_rows = int(
        original_df[text_col].isin(conflicting_texts).sum()
    )

    # Consistent duplicate redundant copies
    df_no_conflicts = original_df[
        ~original_df[text_col].isin(conflicting_texts)
    ]
    n_redundant = int(
        df_no_conflicts.duplicated(subset=[text_col], keep="first").sum()
    )
    consistent_dup_texts = int(
        df_no_conflicts[df_no_conflicts[text_col].duplicated(keep=False)][text_col].nunique()
    )

    # Clean dataset stats
    n_unique_clean = int(clean_df[text_col].nunique())
    n_dup_remaining = int(clean_df.duplicated(subset=[text_col]).sum())
    label_nunique_clean = clean_df.groupby(text_col)[label_col].nunique()
    n_conflict_remaining = int((label_nunique_clean > 1).sum())

    all_labels = sorted(original_df[label_col].dropna().unique())
    clean_counts = clean_df[label_col].value_counts()
    class_distribution = {
        label: {
            "count": int(clean_counts.get(label, 0)),
            "pct": round(
                float(clean_counts.get(label, 0)) / n_clean * 100, 4
            )
            if n_clean > 0
            else 0.0,
        }
        for label in all_labels
    }

    report: dict[str, Any] = {
        "policy": "D",
        "policy_name": "Policy D",
        "policy_description": (
            "Remove all rows for tweet_text values appearing with more than one label; "
            "then deduplicate remaining tweet_text values to one row each (first occurrence)."
        ),
        "source_file": source_path.name,
        "output_file": str(output_path),
        "original_row_count": n_original,
        "conflicting_text_count": n_conflicting_unique,
        "conflicting_unique_texts": n_conflicting_unique,
        "conflicting_rows_removed": n_conflicting_rows,
        "consistent_duplicate_groups": consistent_dup_texts,
        "redundant_duplicate_rows_removed": n_redundant,
        "total_rows_removed": n_conflicting_rows + n_redundant,
        "final_row_count": n_clean,
        "final_unique_text_count": n_unique_clean,
        "final_duplicate_text_count": n_dup_remaining,
        "final_conflicting_text_count": n_conflict_remaining,
        "pct_retained": round(n_clean / n_original * 100, 4) if n_original else 0.0,
        "final_class_counts": {label: v["count"] for label, v in class_distribution.items()},
        "final_class_percentages": {label: v["pct"] for label, v in class_distribution.items()},
        "final_class_distribution": class_distribution,
    }

    if report_path is None:
        report_path = (
            pathlib.Path(__file__).resolve().parent.parent
            / "results"
            / "data_cleaning_report.json"
        )
    else:
        report_path = pathlib.Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    logger.info("Cleaning report written to %s", report_path)

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run Policy D cleaning pipeline: raw -> data/processed/cyberbullying_clean.csv."""
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stdout,
    )

    root = pathlib.Path(__file__).parent.parent
    raw_path = root / "data" / "raw" / "cyberbullying_tweets.csv"
    clean_path = root / "data" / "processed" / "cyberbullying_clean.csv"
    report_path = root / "results" / "data_cleaning_report.json"

    # Load
    original_df = load_raw_csv(raw_path)

    # Apply Policy D
    clean_df = apply_policy_d(original_df)

    # Save
    save_clean_csv(clean_df, clean_path)

    # Report
    report = generate_cleaning_report(
        original_df, clean_df, raw_path, clean_path, report_path=report_path
    )

    print("\n" + "=" * 60)
    print("  Policy D Cleaning -- Results")
    print("=" * 60)
    print(f"  Source          : {raw_path.name}")
    print(f"  Output          : {clean_path}")
    print(f"  Original rows   : {report['original_row_count']:,}")
    print(f"  Conflicting rows removed   : {report['conflicting_rows_removed']:,}")
    print(f"  Redundant copies removed   : {report['redundant_duplicate_rows_removed']:,}")
    print(f"  Total rows removed         : {report['total_rows_removed']:,}")
    print(f"  Final rows      : {report['final_row_count']:,}")
    print(f"  Unique texts    : {report['final_unique_text_count']:,}")
    print(f"  Dup texts left  : {report['final_duplicate_text_count']:,}")
    print(f"  Conflict texts  : {report['final_conflicting_text_count']:,}")
    print(f"  % retained      : {report['pct_retained']:.2f}%")
    print()
    print(f"  {'Label':<25}  {'Count':>8}  {'Pct':>8}")
    print(f"  {'-'*25}  {'-'*8}  {'-'*8}")
    for label, v in report["final_class_distribution"].items():
        print(f"  {label:<25}  {v['count']:>8,}  {v['pct']:>7.2f}%")
    print()
    print(f"  Report written  : {report_path}")
    print()

    # Verification against expected Policy D numbers
    EXPECTED = {
        "final_row_count": 44378,
        "final_unique_text_count": 44378,
        "final_duplicate_text_count": 0,
        "final_conflicting_text_count": 0,
        "class_counts": {
            "age": 7992,
            "ethnicity": 7952,
            "gender": 7772,
            "not_cyberbullying": 6428,
            "other_cyberbullying": 6243,
            "religion": 7991,
        },
    }

    print("=" * 60)
    print("  Verification Against Expected Policy D Values")
    print("=" * 60)
    discrepancies: list[str] = []

    def check(name: str, actual: int, expected: int) -> None:
        status = "[OK]" if actual == expected else "[MISMATCH]"
        print(f"  {status:<10} {name}: {actual:,} (expected {expected:,})")
        if actual != expected:
            discrepancies.append(
                f"{name}: actual={actual}, expected={expected}"
            )

    check("Final row count", report["final_row_count"], EXPECTED["final_row_count"])
    check("Unique texts", report["final_unique_text_count"], EXPECTED["final_unique_text_count"])
    check("Dup texts remaining", report["final_duplicate_text_count"], EXPECTED["final_duplicate_text_count"])
    check("Conflict texts remaining", report["final_conflicting_text_count"], EXPECTED["final_conflicting_text_count"])
    for label, expected_count in EXPECTED["class_counts"].items():
        actual_count = report["final_class_distribution"][label]["count"]
        check(f"Class [{label}]", actual_count, expected_count)

    print()
    if discrepancies:
        print("  STOP: discrepancies found:")
        for d in discrepancies:
            print(f"    - {d}")
        print("  Investigate before proceeding.")
        sys.exit(1)
    else:
        print("  All values match expected Policy D results.")
    print()


if __name__ == "__main__":
    main()
