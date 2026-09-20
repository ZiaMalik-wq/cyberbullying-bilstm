"""
src/analyze_duplicates.py
=========================

Read-only analysis of duplicate rows and conflicting labels in the raw dataset.

Purpose
-------
Investigate the duplicate and conflicting-label problem identified during
initial dataset inspection, before any preprocessing or splitting decisions
are made.

Rules
-----
* NEVER modifies data/raw/ or any file inside it.
* NEVER drops rows, renames columns, or alters values.
* NEVER creates train/val/test splits.
* NEVER creates TextVectorization or any model component.
* All operations are purely analytical / read-only.

Usage
-----
    python src/analyze_duplicates.py

Outputs
-------
* Prints a structured report to stdout.
* Writes results/duplicate_analysis.json
"""

from __future__ import annotations

import json
import pathlib
import sys

import pandas as pd

# Force UTF-8 output on Windows terminals
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = pathlib.Path(__file__).parent.parent
DATA_FILE = ROOT / "data" / "raw" / "cyberbullying_tweets.csv"
OUTPUT_JSON = ROOT / "results" / "duplicate_analysis.json"
OUTPUT_MD = ROOT / "docs" / "DUPLICATE_ANALYSIS.md"

TEXT_COL = "tweet_text"
LABEL_COL = "cyberbullying_type"

MAX_EXAMPLE_LEN = 120  # chars shown per example tweet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    bar = "=" * 65
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def short(text: str) -> str:
    """Truncate text for display."""
    s = str(text).replace("\n", " ").strip()
    return s[:MAX_EXAMPLE_LEN] + "…" if len(s) > MAX_EXAMPLE_LEN else s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not DATA_FILE.exists():
        print(f"ERROR: File not found: {DATA_FILE}")
        sys.exit(1)

    df = pd.read_csv(DATA_FILE, encoding="utf-8")
    total_rows = len(df)

    # -----------------------------------------------------------------------
    # 1. Exact duplicate rows (text AND label identical)
    # -----------------------------------------------------------------------
    section("1. Exact Duplicate Rows (tweet_text AND label identical)")

    exact_dup_mask = df.duplicated(keep=False)
    n_exact_dup_rows = int(df.duplicated().sum())          # non-first occurrences
    n_exact_dup_all = int(exact_dup_mask.sum())            # all copies

    exact_dup_df = df[exact_dup_mask].copy()
    n_exact_dup_groups = int(
        exact_dup_df.groupby([TEXT_COL, LABEL_COL]).ngroups
    )

    print(f"  Total rows that are exact duplicates (all copies)  : {n_exact_dup_all:,}")
    print(f"  Non-first exact duplicate rows                     : {n_exact_dup_rows:,}")
    print(f"  Unique (text, label) pairs with duplicates         : {n_exact_dup_groups:,}")

    print("\n  Sample exact duplicates (text truncated):")
    shown_pairs: set = set()
    for _, row in exact_dup_df.iterrows():
        pair = (row[TEXT_COL][:60], row[LABEL_COL])
        if pair not in shown_pairs:
            shown_pairs.add(pair)
            print(f"    [{row[LABEL_COL]}]  {short(row[TEXT_COL])}")
        if len(shown_pairs) >= 4:
            break

    # -----------------------------------------------------------------------
    # 2. Unique tweet_text values
    # -----------------------------------------------------------------------
    section("2. Unique tweet_text Values")

    n_unique_texts = int(df[TEXT_COL].nunique())
    n_texts_appearing_more_than_once = int(
        df[TEXT_COL].value_counts()[df[TEXT_COL].value_counts() > 1].shape[0]
    )

    print(f"  Total rows                                : {total_rows:,}")
    print(f"  Unique tweet_text values                  : {n_unique_texts:,}")
    print(f"  tweet_text appearing more than once       : {n_texts_appearing_more_than_once:,}")
    print(f"  tweet_text appearing exactly once         : {n_unique_texts - n_texts_appearing_more_than_once:,}")

    # -----------------------------------------------------------------------
    # 3. Frequency distribution of repeated tweet_text values
    # -----------------------------------------------------------------------
    section("3. Frequency Distribution of Repeated tweet_text")

    text_counts = df[TEXT_COL].value_counts()
    repeated = text_counts[text_counts > 1]

    freq_dist: dict[int, int] = {}
    for cnt in repeated:
        freq_dist[int(cnt)] = freq_dist.get(int(cnt), 0) + 1

    print(f"  {'Appearances':>12}  {'Unique texts with that count':>30}")
    for k in sorted(freq_dist):
        print(f"  {k:>12}  {freq_dist[k]:>30,}")

    # -----------------------------------------------------------------------
    # 4-9. Group by tweet_text → label analysis
    # -----------------------------------------------------------------------
    section("4–9. Consistent vs Conflicting Repeated Texts")

    # Group by tweet_text, collect unique labels per group
    grouped = (
        df.groupby(TEXT_COL)[LABEL_COL]
        .apply(lambda s: tuple(sorted(s.unique())))
        .reset_index()
        .rename(columns={LABEL_COL: "label_tuple"})
    )
    grouped["n_labels"] = grouped["label_tuple"].apply(len)
    grouped["n_rows"] = grouped[TEXT_COL].map(df[TEXT_COL].value_counts())

    repeated_texts = grouped[grouped["n_rows"] > 1].copy()
    consistent = repeated_texts[repeated_texts["n_labels"] == 1]
    conflicting = repeated_texts[repeated_texts["n_labels"] > 1]

    n_consistent_unique = len(consistent)
    n_conflicting_unique = len(conflicting)
    n_consistent_rows = int(consistent["n_rows"].sum())
    n_conflicting_rows = int(conflicting["n_rows"].sum())
    max_labels_per_text = int(grouped["n_labels"].max())

    print(f"\n  Repeated texts (appear > 1 time)           : {len(repeated_texts):,}")
    print(f"\n  Consistent duplicates (same label always)  : {n_consistent_unique:,} unique texts")
    print(f"    → covering                                : {n_consistent_rows:,} rows")
    print(f"\n  Conflicting duplicates (labels differ)     : {n_conflicting_unique:,} unique texts")
    print(f"    → covering                                : {n_conflicting_rows:,} rows")
    print(f"\n  Maximum labels assigned to a single text   : {max_labels_per_text}")

    # -----------------------------------------------------------------------
    # 6. Label-combination enumeration for conflicting texts
    # -----------------------------------------------------------------------
    section("6. Label-Combination Counts (conflicting texts only)")

    combo_counts: dict[str, int] = {}
    for _, row in conflicting.iterrows():
        key = " + ".join(row["label_tuple"])
        combo_counts[key] = combo_counts.get(key, 0) + 1

    combo_sorted = sorted(combo_counts.items(), key=lambda x: -x[1])

    print(f"\n  {'Combination':<55}  {'Unique texts':>12}  {'Rows':>8}")
    print(f"  {'-'*55}  {'-'*12}  {'-'*8}")
    combo_row_counts: dict[str, int] = {}
    for combo, cnt in combo_sorted:
        # sum rows for this combination
        mask = conflicting["label_tuple"].apply(lambda t: " + ".join(t) == combo)
        row_cnt = int(conflicting[mask]["n_rows"].sum())
        combo_row_counts[combo] = row_cnt
        print(f"  {combo:<55}  {cnt:>12,}  {row_cnt:>8,}")

    # -----------------------------------------------------------------------
    # 10. Representative examples per conflicting combination
    # -----------------------------------------------------------------------
    section("10. Representative Examples per Conflicting Combination")

    # Build a lookup: text → list of (label, ...) from original df
    examples_shown: dict[str, list] = {}

    for combo, _ in combo_sorted:
        labels_in_combo = set(combo.split(" + "))
        # find one conflicting text for this combo
        match = conflicting[
            conflicting["label_tuple"].apply(lambda t: set(t) == labels_in_combo)
        ].head(1)
        if match.empty:
            continue
        example_text = match.iloc[0][TEXT_COL]
        # retrieve all rows for this text from original df
        rows_for_text = df[df[TEXT_COL] == example_text][[TEXT_COL, LABEL_COL]]
        examples_shown[combo] = rows_for_text.to_dict("records")

        print(f"\n  Combination: {combo}")
        print(f"  Text (truncated): {short(example_text)}")
        print(f"  Label occurrences:")
        label_freq = rows_for_text[LABEL_COL].value_counts()
        for lbl, lbl_cnt in label_freq.items():
            print(f"    {lbl_cnt}×  {lbl}")

    # -----------------------------------------------------------------------
    # Write JSON
    # -----------------------------------------------------------------------
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "total_rows": total_rows,
        "unique_tweet_texts": n_unique_texts,
        "texts_appearing_more_than_once": n_texts_appearing_more_than_once,
        "texts_appearing_exactly_once": n_unique_texts - n_texts_appearing_more_than_once,
        "exact_duplicate_rows": {
            "all_copies": n_exact_dup_all,
            "non_first_copies": n_exact_dup_rows,
            "unique_pairs_with_duplicates": n_exact_dup_groups,
        },
        "repeated_text_analysis": {
            "consistent_unique_texts": n_consistent_unique,
            "consistent_rows": n_consistent_rows,
            "conflicting_unique_texts": n_conflicting_unique,
            "conflicting_rows": n_conflicting_rows,
            "max_labels_per_text": max_labels_per_text,
        },
        "duplicate_text_frequency_distribution": {
            str(k): v for k, v in sorted(freq_dist.items())
        },
        "conflicting_label_combinations": [
            {
                "combination": combo,
                "unique_texts": cnt,
                "rows": combo_row_counts[combo],
            }
            for combo, cnt in combo_sorted
        ],
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    section("DONE")
    print(f"  JSON written to : {OUTPUT_JSON}")
    print()

    # -----------------------------------------------------------------------
    # Write DUPLICATE_ANALYSIS.md
    # -----------------------------------------------------------------------
    _write_markdown(summary, combo_sorted, examples_shown, combo_row_counts)
    print(f"  MD  written to  : {OUTPUT_MD}")
    print()


def _write_markdown(
    s: dict,
    combo_sorted: list[tuple[str, int]],
    examples_shown: dict[str, list],
    combo_row_counts: dict[str, int],
) -> None:
    """Write docs/DUPLICATE_ANALYSIS.md with all findings."""

    total = s["total_rows"]
    conflicting_rows = s["repeated_text_analysis"]["conflicting_rows"]
    conflicting_unique = s["repeated_text_analysis"]["conflicting_unique_texts"]

    lines: list[str] = [
        "# Duplicate and Conflicting-Label Analysis",
        "",
        "> [!IMPORTANT]",
        "> This is a read-only analysis. The raw dataset has not been modified.",
        "> No preprocessing, splitting, or cleaning decisions have been made.",
        "",
        f"Analysis performed on `data/raw/cyberbullying_tweets.csv` ({total:,} rows).",
        "",
        "---",
        "",
        "## 1. Dataset Overview",
        "",
        "| Metric | Count |",
        "|--------|------:|",
        f"| Total rows | {total:,} |",
        f"| Unique `tweet_text` values | {s['unique_tweet_texts']:,} |",
        f"| Texts appearing exactly once | {s['texts_appearing_exactly_once']:,} |",
        f"| Texts appearing more than once | {s['texts_appearing_more_than_once']:,} |",
        "",
        "---",
        "",
        "## 2. Exact Duplicate Rows",
        "",
        "Exact duplicate rows are rows where **both** `tweet_text` and",
        "`cyberbullying_type` are identical.",
        "",
        "| Metric | Count |",
        "|--------|------:|",
        f"| All copies of exact duplicate rows | {s['exact_duplicate_rows']['all_copies']:,} |",
        f"| Non-first (redundant) exact duplicate rows | {s['exact_duplicate_rows']['non_first_copies']:,} |",
        f"| Unique (text, label) pairs with duplicates | {s['exact_duplicate_rows']['unique_pairs_with_duplicates']:,} |",
        "",
        "Exact duplicate rows are straightforward: the same tweet with the same",
        "label appears more than once. Removing the extra copies would lose no",
        "information.",
        "",
        "---",
        "",
        "## 3. Repeated `tweet_text` Frequency Distribution",
        "",
        "| Appearances | Unique texts |",
        "|------------:|-------------:|",
    ]

    for k, v in sorted(s["duplicate_text_frequency_distribution"].items(), key=lambda x: int(x[0])):
        lines.append(f"| {k} | {v:,} |")

    lines += [
        "",
        "---",
        "",
        "## 4. Consistent vs Conflicting Repeated Texts",
        "",
        "| Category | Unique texts | Rows covered |",
        "|----------|-------------:|-------------:|",
        f"| Consistent duplicates (same label every time) | {s['repeated_text_analysis']['consistent_unique_texts']:,} | {s['repeated_text_analysis']['consistent_rows']:,} |",
        f"| **Conflicting duplicates (labels differ)** | **{conflicting_unique:,}** | **{conflicting_rows:,}** |",
        "",
        f"- Maximum labels assigned to a single text: **{s['repeated_text_analysis']['max_labels_per_text']}**",
        f"- Conflicting rows represent **{conflicting_rows / total * 100:.1f}%** of the total dataset.",
        "",
        "---",
        "",
        "## 5. Conflicting Label Combinations",
        "",
        "Each row below represents a distinct set of labels observed for the",
        "same tweet text.",
        "",
        "| Combination | Unique texts | Rows |",
        "|-------------|-------------:|-----:|",
    ]

    for combo, cnt in combo_sorted:
        row_cnt = combo_row_counts[combo]
        lines.append(f"| `{combo}` | {cnt:,} | {row_cnt:,} |")

    lines += [
        "",
        "---",
        "",
        "## 6. Representative Examples per Conflicting Combination",
        "",
        "> Texts are truncated to 120 characters for readability.",
        "> Full texts are available in the raw dataset.",
        "",
    ]

    for combo, _ in combo_sorted:
        if combo not in examples_shown:
            continue
        records = examples_shown[combo]
        if not records:
            continue
        example_text = records[0][TEXT_COL]
        short_text = (
            example_text[:120].replace("\n", " ").strip() + "…"
            if len(example_text) > 120
            else example_text.replace("\n", " ").strip()
        )
        label_freq: dict[str, int] = {}
        for r in records:
            lbl = r[LABEL_COL]
            label_freq[lbl] = label_freq.get(lbl, 0) + 1

        lines.append(f"### `{combo}`")
        lines.append("")
        lines.append(f"**Text (truncated):** {short_text}")
        lines.append("")
        lines.append("**Label occurrences for this text:**")
        lines.append("")
        for lbl, cnt in sorted(label_freq.items(), key=lambda x: -x[1]):
            lines.append(f"- `{lbl}` × {cnt}")
        lines.append("")

    lines += [
        "---",
        "",
        "## 7. Data-Leakage Implications",
        "",
        "> [!CAUTION]",
        "> **The conflicting-label problem is a data-leakage risk.**",
        ">",
        f"> {conflicting_unique:,} unique tweet texts ({conflicting_rows:,} rows, "
        f"{conflicting_rows / s['total_rows'] * 100:.1f}% of the dataset)",
        "> appear with more than one label.",
        ">",
        "> If a train/validation/test split is performed on the raw dataset",
        "> **before** resolving conflicts, the same tweet text could appear in",
        "> both the training set and the test set — potentially with a different",
        "> label. This would cause:",
        ">",
        "> - Near-duplicate memorization rather than generalization",
        "> - Artificially inflated test performance",
        "> - Misleading evaluation",
        "",
        "---",
        "",
        "## 8. Possible Handling Strategies",
        "",
        "> [!IMPORTANT]",
        "> **No strategy has been chosen yet.**",
        "> The following options are presented for review. A decision will be",
        "> made after this report has been reviewed.",
        "",
        "| # | Strategy | Description | Risk |",
        "|---|----------|-------------|------|",
        "| A | Drop all conflicting rows | Remove all rows whose `tweet_text` appears with >1 label | Loses data; may reduce dataset size significantly |",
        "| B | Keep one label per conflicting text (majority vote) | For each text, keep only rows matching the most frequent label | May introduce label noise; loses minority rows |",
        "| C | Keep one label per conflicting text (first occurrence) | Keep the first-seen label for each text | Arbitrary; depends on row order |",
        "| D | Deduplicate texts before splitting, then split | Remove duplicate `tweet_text` values entirely (keeping one row per text) then split | Simplest leakage prevention; reduces dataset |",
        "| E | Split first, then remove cross-split duplicates | Perform split, then identify and remove any text that appears in both train and test | Cleaner split but more complex implementation |",
        "",
        "---",
        "",
        "## 9. Notes",
        "",
        f"- Analysis performed on: `data/raw/cyberbullying_tweets.csv`",
        f"- Script: `src/analyze_duplicates.py`",
        f"- Machine-readable results: `results/duplicate_analysis.json`",
        f"- The raw dataset has **not been modified**.",
        f"- No preprocessing or splitting decisions have been made.",
        "",
    ]

    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
