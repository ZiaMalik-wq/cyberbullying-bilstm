"""
src/inspect_dataset.py
======================

Read-only dataset inspection script.

Purpose
-------
Produce a comprehensive factual report on the raw dataset so that
docs/DATASET.md can be filled with verified values.

Rules
-----
* NEVER modifies data/raw/ or any file inside it.
* NEVER drops rows, renames columns, or alters values.
* NEVER creates train/val/test splits.
* NEVER creates TextVectorization or any model component.
* All operations are purely analytical / read-only.

Usage
-----
    python src/inspect_dataset.py

Output
------
Prints a structured report to stdout.
Also writes a JSON summary to:
    results/dataset_inspection.json
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd

# Force UTF-8 output so non-ASCII characters in tweets print correctly
# on Windows terminals that default to cp1252.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = pathlib.Path(__file__).parent.parent
DATA_FILE = ROOT / "data" / "raw" / "cyberbullying_tweets.csv"
OUTPUT_JSON = ROOT / "results" / "dataset_inspection.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    """Print a section header."""
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def check_url(text: str) -> bool:
    """Return True if text contains a URL pattern."""
    return bool(re.search(r"https?://\S+|www\.\S+", text, re.IGNORECASE))


def check_mention(text: str) -> bool:
    """Return True if text contains an @mention."""
    return bool(re.search(r"@\w+", text))


def check_hashtag(text: str) -> bool:
    """Return True if text contains a #hashtag."""
    return bool(re.search(r"#\w+", text))


def check_emoji(text: str) -> bool:
    """Return True if text contains emoji characters."""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002700-\U000027BF"
        "\U0001F900-\U0001F9FF"
        "\U00002600-\U000026FF"
        "\U00002B50-\U00002B55"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "]+",
        flags=re.UNICODE,
    )
    return bool(emoji_pattern.search(text))


def check_repeated_chars(text: str, n: int = 3) -> bool:
    """Return True if any character is repeated n or more times consecutively."""
    pattern = rf"(.)\1{{{n - 1},}}"
    return bool(re.search(pattern, text))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # -----------------------------------------------------------------------
    # 1. File information
    # -----------------------------------------------------------------------
    section("1. File Information")

    if not DATA_FILE.exists():
        print(f"ERROR: File not found: {DATA_FILE}")
        sys.exit(1)

    size_bytes = DATA_FILE.stat().st_size
    print(f"  Filename  : {DATA_FILE.name}")
    print(f"  Path      : {DATA_FILE}")
    print(f"  Size      : {size_bytes:,} bytes  ({size_bytes / 1024 / 1024:.2f} MB)")

    # Detect encoding by trying UTF-8 first, then fallbacks
    df = None
    detected_encoding = None
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(DATA_FILE, encoding=enc)
            detected_encoding = enc
            break
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue

    if df is None:
        print("ERROR: Could not read file with any tested encoding.")
        sys.exit(1)

    print(f"  Encoding  : {detected_encoding}")
    print(f"  Format    : CSV")

    # -----------------------------------------------------------------------
    # 2. Shape and columns
    # -----------------------------------------------------------------------
    section("2. Shape and Columns")

    n_rows, n_cols = df.shape
    print(f"  Rows      : {n_rows:,}")
    print(f"  Columns   : {n_cols}")
    print(f"  Column names : {list(df.columns)}")
    print()
    print(df.dtypes.to_string())

    # -----------------------------------------------------------------------
    # 3. Identify text and label columns
    # -----------------------------------------------------------------------
    section("3. Text and Label Columns")

    object_cols = df.select_dtypes(include="str").columns.tolist()
    text_col: str | None = None
    label_col: str | None = None

    if len(object_cols) >= 2:
        nuniques = {c: df[c].nunique() for c in object_cols}
        label_col = min(nuniques, key=nuniques.__getitem__)
        text_col = max(nuniques, key=nuniques.__getitem__)
    elif len(object_cols) == 1:
        text_col = object_cols[0]

    print(f"  Detected text column  : {text_col!r}")
    print(f"  Detected label column : {label_col!r}")

    # -----------------------------------------------------------------------
    # 4. Unique labels and class distribution
    # -----------------------------------------------------------------------
    section("4. Unique Labels and Class Distribution")

    label_counts: pd.Series = pd.Series(dtype=int)
    label_pct: pd.Series = pd.Series(dtype=float)
    unique_labels: list = []

    if label_col:
        label_counts = df[label_col].value_counts(dropna=False)
        label_pct = df[label_col].value_counts(normalize=True, dropna=False) * 100
        dist_df = pd.DataFrame({"count": label_counts, "pct": label_pct.round(2)})
        print(dist_df.to_string())
        unique_labels = df[label_col].dropna().unique().tolist()
        print(f"\n  Total unique labels : {len(unique_labels)}")
        print(f"  Labels              : {sorted(str(l) for l in unique_labels)}")
    else:
        print("  Label column not detected.")

    # -----------------------------------------------------------------------
    # 5. Missing values
    # -----------------------------------------------------------------------
    section("5. Missing Values")

    missing = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df) * 100).round(4)
    missing_df = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    print(missing_df.to_string())

    # -----------------------------------------------------------------------
    # 6. Exact duplicate rows
    # -----------------------------------------------------------------------
    section("6. Exact Duplicate Rows")

    n_exact_dupes = int(df.duplicated().sum())
    print(f"  Exact duplicate rows : {n_exact_dupes:,}")

    # -----------------------------------------------------------------------
    # 7. Duplicate text samples
    # -----------------------------------------------------------------------
    section("7. Duplicate Text Samples")

    n_text_dupes = 0
    n_conflicting = 0

    if text_col:
        n_text_dupes = int(df.duplicated(subset=[text_col]).sum())
        if label_col:
            text_label_conflict = df.groupby(text_col)[label_col].nunique()
            n_conflicting = int((text_label_conflict > 1).sum())
        print(f"  Duplicate text rows (same tweet_text)    : {n_text_dupes:,}")
        print(f"  Texts with conflicting labels            : {n_conflicting:,}")
    else:
        print("  Text column not detected.")

    # -----------------------------------------------------------------------
    # 8. Empty / whitespace-only text samples
    # -----------------------------------------------------------------------
    section("8. Empty or Whitespace-Only Texts")

    n_null = 0
    n_empty = 0

    if text_col:
        is_null = df[text_col].isnull()
        is_empty = df[text_col].astype(str).str.strip() == ""
        n_null = int(is_null.sum())
        n_empty = int((is_empty & ~is_null).sum())
        print(f"  Null text values     : {n_null:,}")
        print(f"  Empty/whitespace     : {n_empty:,}")

    # -----------------------------------------------------------------------
    # 9. Character-length statistics
    # -----------------------------------------------------------------------
    section("9. Character-Length Statistics (text column)")

    char_stats: dict = {}

    if text_col:
        char_len = df[text_col].dropna().astype(str).str.len()
        char_stats = {
            "min": int(char_len.min()),
            "max": int(char_len.max()),
            "mean": round(float(char_len.mean()), 2),
            "median": round(float(char_len.median()), 2),
            "std": round(float(char_len.std()), 2),
            "p25": round(float(char_len.quantile(0.25)), 2),
            "p75": round(float(char_len.quantile(0.75)), 2),
            "p90": round(float(char_len.quantile(0.90)), 2),
            "p95": round(float(char_len.quantile(0.95)), 2),
            "p99": round(float(char_len.quantile(0.99)), 2),
        }
        for k, v in char_stats.items():
            print(f"  {k:8s} : {v}")

    # -----------------------------------------------------------------------
    # 10. Word/token-length statistics
    # -----------------------------------------------------------------------
    section("10. Word-Length Statistics (whitespace-split tokens)")

    word_stats: dict = {}

    if text_col:
        word_len = df[text_col].dropna().astype(str).str.split().str.len()
        word_stats = {
            "min": int(word_len.min()),
            "max": int(word_len.max()),
            "mean": round(float(word_len.mean()), 2),
            "median": round(float(word_len.median()), 2),
            "std": round(float(word_len.std()), 2),
            "p25": round(float(word_len.quantile(0.25)), 2),
            "p75": round(float(word_len.quantile(0.75)), 2),
            "p90": round(float(word_len.quantile(0.90)), 2),
            "p95": round(float(word_len.quantile(0.95)), 2),
            "p99": round(float(word_len.quantile(0.99)), 2),
        }
        for k, v in word_stats.items():
            print(f"  {k:8s} : {v}")

    # -----------------------------------------------------------------------
    # 11. Social-media feature presence
    # -----------------------------------------------------------------------
    section("11. Social-Media Feature Presence")

    social_stats: dict = {}

    if text_col:
        texts = df[text_col].dropna().astype(str)
        n_total = len(texts)

        has_url = texts.apply(check_url)
        has_mention = texts.apply(check_mention)
        has_hashtag = texts.apply(check_hashtag)
        has_emoji = texts.apply(check_emoji)
        has_repeated = texts.apply(check_repeated_chars)

        social_stats = {
            "urls": int(has_url.sum()),
            "mentions": int(has_mention.sum()),
            "hashtags": int(has_hashtag.sum()),
            "emojis": int(has_emoji.sum()),
            "repeated_chars_3plus": int(has_repeated.sum()),
        }

        for k, v in social_stats.items():
            pct = v / n_total * 100
            print(f"  Rows with {k:25s}: {v:6,}  ({pct:.1f}%)")

    # -----------------------------------------------------------------------
    # 12. Representative examples per class
    # -----------------------------------------------------------------------
    section("12. Representative Examples Per Class (2 per class)")

    if text_col and label_col:
        for label in sorted(str(l) for l in df[label_col].dropna().unique()):
            subset = df[df[label_col] == label][text_col].dropna()
            print(f"\n  --- {label} ---")
            for i, ex in enumerate(subset.head(2)):
                short = str(ex)[:200].replace("\n", " ")
                print(f"  [{i+1}] {short}")

    # -----------------------------------------------------------------------
    # 13. Non-ASCII character check
    # -----------------------------------------------------------------------
    section("13. Non-ASCII Character Check")

    n_non_ascii = 0

    if text_col:
        texts_str = df[text_col].dropna().astype(str)
        has_non_ascii = texts_str.apply(lambda t: any(ord(c) > 127 for c in t))
        n_non_ascii = int(has_non_ascii.sum())
        print(f"  Rows with non-ASCII chars : {n_non_ascii:,}  ({n_non_ascii / len(texts_str) * 100:.1f}%)")
        non_ascii_examples = texts_str[has_non_ascii].head(3).tolist()
        for ex in non_ascii_examples:
            print(f"  EXAMPLE: {ex[:150]}")

    # -----------------------------------------------------------------------
    # 14. Write JSON summary
    # -----------------------------------------------------------------------
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "filename": DATA_FILE.name,
        "size_bytes": size_bytes,
        "encoding": detected_encoding,
        "format": "csv",
        "n_rows": n_rows,
        "n_cols": n_cols,
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "text_col": text_col,
        "label_col": label_col,
        "unique_labels": sorted(str(l) for l in unique_labels),
        "class_distribution": {
            str(k): {"count": int(v), "pct": round(float(label_pct[k]), 4)}
            for k, v in label_counts.items()
        }
        if label_col
        else {},
        "missing_values": {c: int(missing[c]) for c in df.columns},
        "exact_duplicate_rows": n_exact_dupes,
        "duplicate_text_rows": n_text_dupes,
        "texts_with_conflicting_labels": n_conflicting,
        "null_text_values": n_null,
        "empty_whitespace_texts": n_empty,
        "char_length_stats": char_stats,
        "word_length_stats": word_stats,
        "social_media_features": social_stats,
        "non_ascii_rows": n_non_ascii,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    section("DONE")
    print(f"  JSON summary written to: {OUTPUT_JSON}")
    print()


if __name__ == "__main__":
    main()
