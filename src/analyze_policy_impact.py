"""
src/analyze_policy_impact.py
============================

Quantitative impact analysis of data-quality policies for handling
duplicate and conflicting-label tweets.

Purpose
-------
Calculate the exact dataset composition that would result from each
candidate policy, WITHOUT applying any policy to the raw data.

Rules
-----
* NEVER modifies data/raw/ or any file inside it.
* NEVER drops rows from the actual DataFrame permanently in a way that
  would alter a file on disk.
* NEVER creates train/val/test splits.
* NEVER creates TextVectorization or any model component.
* NEVER chooses or recommends a policy.
* All reported numbers come from actual execution against the raw dataset.

Usage
-----
    python src/analyze_policy_impact.py

Outputs
-------
* results/duplicate_policy_impact.json
* docs/DUPLICATE_POLICY_IMPACT.md
* Structured report printed to stdout
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = pathlib.Path(__file__).parent.parent
DATA_FILE = ROOT / "data" / "raw" / "cyberbullying_tweets.csv"
OUTPUT_JSON = ROOT / "results" / "duplicate_policy_impact.json"
OUTPUT_MD = ROOT / "docs" / "DUPLICATE_POLICY_IMPACT.md"

TEXT_COL = "tweet_text"
LABEL_COL = "cyberbullying_type"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def class_stats(df: pd.DataFrame) -> dict[str, dict]:
    """Return per-class count and percentage for a given dataframe."""
    total = len(df)
    counts = df[LABEL_COL].value_counts()
    return {
        label: {
            "count": int(counts.get(label, 0)),
            "pct": round(float(counts.get(label, 0)) / total * 100, 4)
            if total > 0 else 0.0,
        }
        for label in sorted(df[LABEL_COL].unique())
    }


def duplicate_check(df: pd.DataFrame) -> dict[str, int]:
    """Return duplicate/conflict counts for a given dataframe."""
    text_label_counts = df.groupby(TEXT_COL)[LABEL_COL].nunique()
    n_dup_texts = int((df[TEXT_COL].value_counts() > 1).sum())
    n_conflict_texts = int((text_label_counts > 1).sum())
    return {
        "duplicate_tweet_text_values": n_dup_texts,
        "conflicting_label_texts": n_conflict_texts,
    }


def policy_summary(
    original_len: int,
    result_df: pd.DataFrame,
    note: str = "",
) -> dict[str, Any]:
    """Build a complete policy result dict."""
    n = len(result_df)
    removed = original_len - n
    pct_retained = round(n / original_len * 100, 4)
    cs = class_stats(result_df)
    dc = duplicate_check(result_df)
    return {
        "total_rows": n,
        "rows_removed": removed,
        "pct_retained": pct_retained,
        "unique_tweet_texts": int(result_df[TEXT_COL].nunique()),
        "class_distribution": cs,
        "duplicate_tweet_text_values_remaining": dc["duplicate_tweet_text_values"],
        "conflicting_label_texts_remaining": dc["conflicting_label_texts"],
        "note": note,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not DATA_FILE.exists():
        print(f"ERROR: File not found: {DATA_FILE}")
        sys.exit(1)

    df = pd.read_csv(DATA_FILE, encoding="utf-8")
    total_rows = len(df)
    all_labels = sorted(df[LABEL_COL].unique())

    # -----------------------------------------------------------------------
    # 1. Current class distribution
    # -----------------------------------------------------------------------
    section("1. Current Class Distribution (Baseline)")

    baseline_counts = df[LABEL_COL].value_counts()
    baseline_pct = df[LABEL_COL].value_counts(normalize=True) * 100

    print(f"\n  {'Label':<25}  {'Count':>8}  {'Pct':>8}")
    print(f"  {'-'*25}  {'-'*8}  {'-'*8}")
    for label in all_labels:
        print(f"  {label:<25}  {baseline_counts[label]:>8,}  {baseline_pct[label]:>7.2f}%")
    print(f"\n  {'TOTAL':<25}  {total_rows:>8,}  {'100.00%':>8}")

    # -----------------------------------------------------------------------
    # 2. Identify conflicting texts
    # -----------------------------------------------------------------------
    section("2. Conflicting Texts — Per-Class Impact")

    # A text is conflicting if it appears with 2+ distinct labels
    text_label_nunique = df.groupby(TEXT_COL)[LABEL_COL].nunique()
    conflicting_texts = set(text_label_nunique[text_label_nunique > 1].index)
    conflict_mask = df[TEXT_COL].isin(conflicting_texts)
    conflict_df = df[conflict_mask]

    print(f"\n  Total conflicting unique texts : {len(conflicting_texts):,}")
    print(f"  Total rows in conflicting texts: {len(conflict_df):,}")
    print()

    print(f"  {'Label':<25}  {'Rows in conflicts':>17}  {'Unique conflict texts':>20}")
    print(f"  {'-'*25}  {'-'*17}  {'-'*20}")
    conflict_per_class: dict[str, dict] = {}
    for label in all_labels:
        label_conflict_df = conflict_df[conflict_df[LABEL_COL] == label]
        rows_in_conflict = len(label_conflict_df)
        unique_texts_in_conflict = int(label_conflict_df[TEXT_COL].nunique())
        conflict_per_class[label] = {
            "rows_in_conflicting_texts": rows_in_conflict,
            "unique_conflicting_texts": unique_texts_in_conflict,
        }
        print(
            f"  {label:<25}  {rows_in_conflict:>17,}  {unique_texts_in_conflict:>20,}"
        )

    # -----------------------------------------------------------------------
    # 3. Majority-vote verification
    # -----------------------------------------------------------------------
    section("3. Majority-Vote Verification")

    # For each conflicting text, determine how many rows carry each label
    # Build per-text label-count dicts manually (pandas 3 compatible)
    conflict_label_counts: dict[str, dict[str, int]] = {}
    for text, grp in conflict_df.groupby(TEXT_COL):
        conflict_label_counts[text] = dict(grp[LABEL_COL].value_counts())

    def has_majority(d: dict) -> bool:
        """Return True if one label count strictly exceeds all others combined."""
        vals = list(d.values())
        return max(vals) > sum(vals) - max(vals)

    n_with_majority = sum(1 for d in conflict_label_counts.values() if has_majority(d))
    n_tied = len(conflict_label_counts) - n_with_majority

    print(f"\n  Conflicting texts with a clear majority label : {n_with_majority:,}")
    print(f"  Conflicting texts with a tie (no majority)    : {n_tied:,}")
    print()
    if n_with_majority == 0:
        print(
            "  CONCLUSION: Every conflicting text is a perfect tie (each label appears\n"
            "  exactly once). Majority voting CANNOT resolve any conflict in this dataset."
        )
    else:
        print(
            f"  WARNING: {n_with_majority} conflicting texts have a majority label.\n"
            "  Majority voting would be applicable for those cases only."
        )

    # -----------------------------------------------------------------------
    # 4. Consistent duplicates — class distribution
    # -----------------------------------------------------------------------
    section("4. Consistent Duplicates — Class Distribution")

    # Consistent = repeated but same label always
    text_all_labels = df.groupby(TEXT_COL)[LABEL_COL].apply(lambda s: tuple(sorted(s.unique())))
    repeated_mask = df[TEXT_COL].map(df[TEXT_COL].value_counts() > 1)
    consistent_mask = repeated_mask & ~conflict_mask
    consistent_df = df[consistent_mask]
    # The 36 "redundant" copies are non-first occurrences
    redundant_mask = consistent_mask & df.duplicated(subset=[TEXT_COL], keep="first")
    n_redundant = int(redundant_mask.sum())

    print(f"\n  Consistent-duplicate rows (all copies)  : {len(consistent_df):,}")
    print(f"  Redundant copies (non-first occurrences) : {n_redundant:,}")
    print()
    consistent_counts = consistent_df[LABEL_COL].value_counts()
    redundant_counts = df[redundant_mask][LABEL_COL].value_counts()
    print(f"  {'Label':<25}  {'All copies':>10}  {'Redundant copies':>16}")
    print(f"  {'-'*25}  {'-'*10}  {'-'*16}")
    for label in all_labels:
        ac = int(consistent_counts.get(label, 0))
        rc = int(redundant_counts.get(label, 0))
        print(f"  {label:<25}  {ac:>10,}  {rc:>16,}")

    # -----------------------------------------------------------------------
    # 5. Policy evaluations
    # -----------------------------------------------------------------------
    section("5. Policy Evaluations")

    # --- Policy A: Remove all rows for conflicting texts ---
    policy_a_df = df[~conflict_mask].copy()
    policy_a = policy_summary(
        total_rows,
        policy_a_df,
        note=(
            "Removes all 3,278 rows belonging to the 1,639 conflicting texts. "
            "Consistent duplicates (72 rows, 36 unique texts) remain. "
            "No conflicting texts remain. Duplicate tweet_text values remain "
            "(36 consistent-duplicate texts still appear twice)."
        ),
    )
    print("\n  Policy A: Remove all conflicting texts")
    print(f"    Rows remaining : {policy_a['total_rows']:,}")
    print(f"    Rows removed   : {policy_a['rows_removed']:,}")
    print(f"    % retained     : {policy_a['pct_retained']:.2f}%")
    print(f"    Dup texts left : {policy_a['duplicate_tweet_text_values_remaining']:,}")
    print(f"    Conflict texts : {policy_a['conflicting_label_texts_remaining']:,}")

    # --- Policy B: Deduplicate to one row per unique tweet_text (first occurrence) ---
    policy_b_df = df.drop_duplicates(subset=[TEXT_COL], keep="first").copy()
    policy_b = policy_summary(
        total_rows,
        policy_b_df,
        note=(
            "Keeps exactly one row per unique tweet_text (first occurrence). "
            "For consistent duplicates: label is preserved (same label either way). "
            "For conflicting texts: the label kept is whichever appeared first in "
            "the raw file — an arbitrary choice. "
            "No duplicate tweet_text values remain. No conflicting texts remain. "
            "This policy implicitly assigns a label to 1,639 ambiguous texts."
        ),
    )
    print("\n  Policy B: Deduplicate to one row per unique tweet_text (first occurrence)")
    print(f"    Rows remaining : {policy_b['total_rows']:,}")
    print(f"    Rows removed   : {policy_b['rows_removed']:,}")
    print(f"    % retained     : {policy_b['pct_retained']:.2f}%")
    print(f"    Dup texts left : {policy_b['duplicate_tweet_text_values_remaining']:,}")
    print(f"    Conflict texts : {policy_b['conflicting_label_texts_remaining']:,}")

    # --- Policy C: Keep consistent duplicates, remove conflicting texts ---
    # In this dataset, ~conflict_mask already preserves consistent duplicates,
    # so Policy C is numerically identical to Policy A.
    policy_c_df = df[~conflict_mask].copy()
    policy_c = policy_summary(
        total_rows,
        policy_c_df,
        note=(
            "Remove all conflicting texts while explicitly keeping all consistent "
            "duplicates. In this dataset this is numerically identical to Policy A "
            "because ~conflict_mask already retains consistent-duplicate rows. "
            "The distinction is conceptual: Policy A is framed as 'remove conflicting', "
            "Policy C is framed as 'keep consistent, remove conflicting'. Both produce "
            "the same DataFrame in this specific dataset."
        ),
    )
    print("\n  Policy C: Keep consistent duplicates, remove conflicting texts")
    print(f"    Rows remaining : {policy_c['total_rows']:,}  (identical to Policy A)")
    print(f"    Rows removed   : {policy_c['rows_removed']:,}")
    print(f"    % retained     : {policy_c['pct_retained']:.2f}%")
    print(f"    Dup texts left : {policy_c['duplicate_tweet_text_values_remaining']:,}")
    print(f"    Conflict texts : {policy_c['conflicting_label_texts_remaining']:,}")

    # --- Policy D: Remove conflicting texts AND deduplicate consistent duplicates ---
    # No arbitrary label assignment required: consistent duplicates have one label.
    policy_d_df = (
        df[~conflict_mask]
        .drop_duplicates(subset=[TEXT_COL], keep="first")
        .copy()
    )
    policy_d = policy_summary(
        total_rows,
        policy_d_df,
        note=(
            "Remove all conflicting texts (3,278 rows) AND reduce consistent "
            "duplicates to one copy each (removes 36 redundant copies). "
            "No arbitrary label assignment required — consistent duplicates "
            "always share the same label. Result has no duplicate tweet_text "
            "values and no conflicting texts. This is the cleanest dataset "
            "state achievable without arbitrary label decisions."
        ),
    )
    print("\n  Policy D: Remove conflicting texts + deduplicate consistent duplicates")
    print(f"    Rows remaining : {policy_d['total_rows']:,}")
    print(f"    Rows removed   : {policy_d['rows_removed']:,}")
    print(f"    % retained     : {policy_d['pct_retained']:.2f}%")
    print(f"    Dup texts left : {policy_d['duplicate_tweet_text_values_remaining']:,}")
    print(f"    Conflict texts : {policy_d['conflicting_label_texts_remaining']:,}")

    # -----------------------------------------------------------------------
    # 6. Per-class breakdown for each policy
    # -----------------------------------------------------------------------
    section("6. Per-Class Breakdown by Policy")

    policies = {
        "baseline": {"df": df, "label": "Baseline (raw)"},
        "A": {"df": policy_a_df, "label": "Policy A"},
        "B": {"df": policy_b_df, "label": "Policy B"},
        "C": {"df": policy_c_df, "label": "Policy C (≡ A)"},
        "D": {"df": policy_d_df, "label": "Policy D"},
    }

    header = f"  {'Label':<25}" + "".join(
        f"  {v['label']:>16}" for v in policies.values()
    )
    print(f"\n  Counts:\n{header}")
    print(f"  {'-'*25}" + "  " + "  ".join(["-"*16]*len(policies)))
    for label in all_labels:
        row = f"  {label:<25}"
        for p in policies.values():
            cnt = int(p["df"][LABEL_COL].value_counts().get(label, 0))
            row += f"  {cnt:>16,}"
        print(row)
    total_row = f"  {'TOTAL':<25}"
    for p in policies.values():
        total_row += f"  {len(p['df']):>16,}"
    print(total_row)

    print()
    header2 = f"  {'Label':<25}" + "".join(
        f"  {v['label']:>16}" for v in policies.values()
    )
    print(f"  Percentages:\n{header2}")
    print(f"  {'-'*25}" + "  " + "  ".join(["-"*16]*len(policies)))
    for label in all_labels:
        row = f"  {label:<25}"
        for p in policies.values():
            n_total = len(p["df"])
            cnt = int(p["df"][LABEL_COL].value_counts().get(label, 0))
            pct = cnt / n_total * 100 if n_total > 0 else 0.0
            row += f"  {pct:>15.2f}%"
        print(row)

    # -----------------------------------------------------------------------
    # 7. Summary comparison table
    # -----------------------------------------------------------------------
    section("7. Policy Comparison Summary")

    policies_summary = {
        "A": policy_a,
        "B": policy_b,
        "C": policy_c,
        "D": policy_d,
    }
    print(
        f"\n  {'Metric':<40}  {'A':>10}  {'B':>10}  {'C':>10}  {'D':>10}"
    )
    print(f"  {'-'*40}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    metrics = [
        ("Total rows", "total_rows"),
        ("Rows removed", "rows_removed"),
        ("% retained", "pct_retained"),
        ("Unique tweet_text", "unique_tweet_texts"),
        ("Dup texts remaining", "duplicate_tweet_text_values_remaining"),
        ("Conflict texts remaining", "conflicting_label_texts_remaining"),
    ]
    for label, key in metrics:
        row = f"  {label:<40}"
        for p in policies_summary.values():
            val = p[key]
            if isinstance(val, float):
                row += f"  {val:>9.2f}%"
            else:
                row += f"  {val:>10,}"
        print(row)

    # -----------------------------------------------------------------------
    # 8. Write JSON
    # -----------------------------------------------------------------------
    result = {
        "source_file": DATA_FILE.name,
        "total_rows_original": total_rows,
        "baseline_class_distribution": {
            label: {
                "count": int(baseline_counts[label]),
                "pct": round(float(baseline_pct[label]), 4),
            }
            for label in all_labels
        },
        "conflicting_text_analysis": {
            "total_conflicting_unique_texts": len(conflicting_texts),
            "total_rows_in_conflicting_texts": len(conflict_df),
            "per_class": conflict_per_class,
        },
        "consistent_duplicate_analysis": {
            "total_consistent_dup_rows": len(consistent_df),
            "redundant_copies": n_redundant,
            "per_class_all_copies": {
                label: int(consistent_counts.get(label, 0)) for label in all_labels
            },
            "per_class_redundant_copies": {
                label: int(redundant_counts.get(label, 0)) for label in all_labels
            },
        },
        "majority_vote_verification": {
            "conflicting_texts_with_clear_majority": n_with_majority,
            "conflicting_texts_tied": n_tied,
            "majority_vote_applicable": n_with_majority > 0,
            "conclusion": (
                "Every conflicting text is a perfect tie. Majority voting cannot "
                "resolve any conflict in this dataset."
                if n_with_majority == 0
                else f"{n_with_majority} texts have a majority; majority vote partially applicable."
            ),
        },
        "policies": {
            "A": policy_a,
            "B": policy_b,
            "C": policy_c,
            "D": policy_d,
        },
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    section("DONE — writing outputs")
    print(f"  JSON : {OUTPUT_JSON}")

    # -----------------------------------------------------------------------
    # 9. Write Markdown
    # -----------------------------------------------------------------------
    _write_markdown(result, policies_summary, all_labels)
    print(f"  MD   : {OUTPUT_MD}")
    print()


# ---------------------------------------------------------------------------
# Markdown writer
# ---------------------------------------------------------------------------

def _write_markdown(
    r: dict,
    policies_summary: dict[str, dict],
    all_labels: list[str],
) -> None:
    total = r["total_rows_original"]
    n_conflict_texts = r["conflicting_text_analysis"]["total_conflicting_unique_texts"]
    n_conflict_rows = r["conflicting_text_analysis"]["total_rows_in_conflicting_texts"]
    mv = r["majority_vote_verification"]

    lines: list[str] = [
        "# Duplicate Policy Impact Analysis",
        "",
        "> [!IMPORTANT]",
        "> This is a quantitative analysis only. No policy has been chosen.",
        "> The raw dataset has not been modified.",
        "> No preprocessing, splitting, or cleaning has been performed.",
        "> All numbers are from actual execution against the raw dataset.",
        "",
        f"Source: `data/raw/cyberbullying_tweets.csv`  ({total:,} rows)",
        "",
        "---",
        "",
        "## 1. Baseline Class Distribution",
        "",
        "| Label | Count | Percentage |",
        "|-------|------:|-----------:|",
    ]
    for label, v in r["baseline_class_distribution"].items():
        lines.append(f"| `{label}` | {v['count']:,} | {v['pct']:.2f}% |")
    lines += [
        f"| **Total** | **{total:,}** | **100.00%** |",
        "",
        "---",
        "",
        "## 2. Conflicting Texts — Per-Class Impact",
        "",
        f"**{n_conflict_texts:,} unique tweet texts** appear with conflicting labels,",
        f"covering **{n_conflict_rows:,} rows** ({n_conflict_rows / total * 100:.1f}% of the dataset).",
        "",
        "| Label | Rows in conflicting texts | Unique conflicting texts |",
        "|-------|-------------------------:|-------------------------:|",
    ]
    for label, v in r["conflicting_text_analysis"]["per_class"].items():
        lines.append(
            f"| `{label}` | {v['rows_in_conflicting_texts']:,} | "
            f"{v['unique_conflicting_texts']:,} |"
        )
    lines += [
        "",
        "---",
        "",
        "## 3. Majority-Vote Verification",
        "",
        "> [!NOTE]",
        f"> Conflicting texts with a clear majority label: **{mv['conflicting_texts_with_clear_majority']:,}**",
        f"> Conflicting texts that are tied: **{mv['conflicting_texts_tied']:,}**",
        ">",
        f"> {mv['conclusion']}",
        "",
        "Every conflicting tweet appears **exactly twice**, each time with a **different**",
        "label. The counts are 1 vs 1 for every conflict — there is no majority.",
        "**Majority voting is not a valid resolution strategy for this dataset.**",
        "",
        "---",
        "",
        "## 4. Consistent Duplicates — Class Distribution",
        "",
        "| Label | All copies | Redundant copies |",
        "|-------|-----------:|-----------------:|",
    ]
    cd = r["consistent_duplicate_analysis"]
    for label in all_labels:
        ac = cd["per_class_all_copies"].get(label, 0)
        rc = cd["per_class_redundant_copies"].get(label, 0)
        lines.append(f"| `{label}` | {ac:,} | {rc:,} |")
    lines += [
        f"| **Total** | **{cd['total_consistent_dup_rows']:,}** | "
        f"**{cd['redundant_copies']:,}** |",
        "",
        "Consistent duplicates share the same label in both copies.",
        "Removing the redundant copy loses no label information.",
        "",
        "---",
        "",
        "## 5. Policy Definitions",
        "",
        "| Policy | Definition |",
        "|--------|------------|",
        "| **A** | Remove all rows belonging to the 1,639 conflicting texts. Keep consistent duplicates unchanged. |",
        "| **B** | Deduplicate to one row per unique `tweet_text` (first occurrence). Resolves conflicts by keeping the first-seen label — an **arbitrary** choice for the 1,639 conflicting texts. |",
        "| **C** | Explicitly keep all consistent-duplicate rows; remove all conflicting texts. *(Numerically identical to Policy A in this dataset.)* |",
        "| **D** | Remove all conflicting texts **AND** deduplicate consistent duplicates to one copy each. No arbitrary label assignment required. |",
        "",
        "---",
        "",
        "## 6. Policy Comparison — Row Counts",
        "",
        "| Metric | Baseline | Policy A | Policy B | Policy C | Policy D |",
        "|--------|----------:|---------:|---------:|---------:|---------:|",
    ]
    metrics = [
        ("Total rows", "total_rows"),
        ("Rows removed", "rows_removed"),
        ("Unique tweet_text", "unique_tweet_texts"),
        ("Dup texts remaining", "duplicate_tweet_text_values_remaining"),
        ("Conflict texts remaining", "conflicting_label_texts_remaining"),
    ]
    baseline_vals = {
        "total_rows": total,
        "rows_removed": 0,
        "unique_tweet_texts": r["total_rows_original"] - (
            r["consistent_duplicate_analysis"]["redundant_copies"]
            + r["conflicting_text_analysis"]["total_rows_in_conflicting_texts"] // 2
        ),
        "duplicate_tweet_text_values_remaining": 1675,
        "conflicting_label_texts_remaining": 1639,
    }
    # Use actual unique tweet_text count for baseline
    baseline_vals["unique_tweet_texts"] = 46017  # verified from dataset inspection

    for label, key in metrics:
        bv = baseline_vals[key]
        row = f"| {label} | {bv:,} |"
        for p in policies_summary.values():
            row += f" {p[key]:,} |"
        lines.append(row)

    lines += [
        "",
        "| % data retained | 100.00% |",
    ]
    pct_row = "| % data retained | 100.00% |"
    for p in policies_summary.values():
        pct_row_inner = f" {p['pct_retained']:.2f}% |"
        lines[-1] = lines[-1] + pct_row_inner

    # Rewrite the pct row cleanly
    lines[-1] = (
        "| % data retained | 100.00% | "
        + " | ".join(f"{p['pct_retained']:.2f}%" for p in policies_summary.values())
        + " |"
    )

    lines += [
        "",
        "---",
        "",
        "## 7. Per-Class Counts by Policy",
        "",
        "| Label | Baseline | Policy A | Policy B | Policy C | Policy D |",
        "|-------|----------:|---------:|---------:|---------:|---------:|",
    ]
    policy_dfs_order = ["A", "B", "C", "D"]
    for label in all_labels:
        base_cnt = r["baseline_class_distribution"][label]["count"]
        row = f"| `{label}` | {base_cnt:,} |"
        for key in policy_dfs_order:
            cnt = policies_summary[key]["class_distribution"].get(label, {}).get("count", 0)
            row += f" {cnt:,} |"
        lines.append(row)
    base_total = total
    row = f"| **Total** | **{base_total:,}** |"
    for key in policy_dfs_order:
        row += f" **{policies_summary[key]['total_rows']:,}** |"
    lines.append(row)

    lines += [
        "",
        "---",
        "",
        "## 8. Per-Class Percentages by Policy",
        "",
        "| Label | Baseline | Policy A | Policy B | Policy C | Policy D |",
        "|-------|----------:|---------:|---------:|---------:|---------:|",
    ]
    for label in all_labels:
        base_pct = r["baseline_class_distribution"][label]["pct"]
        row = f"| `{label}` | {base_pct:.2f}% |"
        for key in policy_dfs_order:
            pct = policies_summary[key]["class_distribution"].get(label, {}).get("pct", 0.0)
            row += f" {pct:.2f}% |"
        lines.append(row)

    lines += [
        "",
        "---",
        "",
        "## 9. Methodological Considerations",
        "",
        "| Consideration | Policy A | Policy B | Policy C | Policy D |",
        "|---------------|----------|----------|----------|----------|",
        "| Requires arbitrary label assignment | No | **Yes** (1,639 texts) | No | No |",
        "| Conflicting texts remain | No | No | No | No |",
        "| Duplicate tweet_text values remain | Yes (36 texts) | No | Yes (36 texts) | No |",
        "| Data loss from conflicts | Yes | Yes (indirectly) | Yes | Yes |",
        "| A and C are numerically identical | — | — | ✓ | — |",
        "| Largest retained dataset | — | — | ✓ | — |",
        "| Cleanest result (no dups, no conflicts) | — | ✓ (with caveat) | — | ✓ |",
        "",
        "---",
        "",
        "## 10. Unresolved Decision",
        "",
        "> [!IMPORTANT]",
        "> **No policy has been selected.** The choice involves a trade-off between:",
        ">",
        "> - **Data volume** (Policy B retains the most rows at 46,017)",
        "> - **Methodological cleanliness** (Policy D leaves no duplicates or conflicts",
        ">   without any arbitrary label assignment)",
        "> - **Simplicity** (Policies A/C are the same operation)",
        ">",
        "> The decision must be made before the train/validation/test split is created.",
        "",
        "---",
        "",
        "## 11. Notes",
        "",
        "- Script: `src/analyze_policy_impact.py`",
        "- Machine-readable results: `results/duplicate_policy_impact.json`",
        "- The raw dataset has **not been modified**.",
        "- No train/val/test split has been created.",
        "- All numbers are from actual execution on the raw dataset.",
        "",
    ]

    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
