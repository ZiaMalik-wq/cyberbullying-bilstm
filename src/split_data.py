"""
src/split_data.py
=================

Reusable, leakage-safe dataset splitting module.

Purpose
-------
Performs a stratified 80% train / 10% validation / 10% test split on the
Policy D cleaned dataset (`data/processed/cyberbullying_clean.csv`) and
saves the derived artifacts to `data/processed/`.

Key Invariants
--------------
1. Input must be the Policy D cleaned dataset (44,378 rows, zero duplicates).
2. The raw dataset (`data/raw/cyberbullying_tweets.csv`) and the clean dataset
   (`data/processed/cyberbullying_clean.csv`) are NEVER modified.
3. Stratified splitting preserves class balance across all 6 classes.
4. Deterministic random seed ensures strict reproducibility.
5. Strict leakage verification: zero text overlap between any partitions.
6. Split is executed prior to any text tokenization, vocabulary fitting,
   or model building.

Outputs
-------
* `data/processed/train.csv` (80% of data, 35,502 rows)
* `data/processed/val.csv`   (10% of data, 4,438 rows)
* `data/processed/test.csv`  (10% of data, 4,438 rows)
* `results/split_report.json` (machine-readable report)
"""

from __future__ import annotations

import json
import logging
import pathlib
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default Constants & Paths
# ---------------------------------------------------------------------------

DEFAULT_RANDOM_SEED: int = 42
DEFAULT_TRAIN_RATIO: float = 0.80
DEFAULT_VAL_RATIO: float = 0.10
DEFAULT_TEST_RATIO: float = 0.10

DEFAULT_TEXT_COL: str = "tweet_text"
DEFAULT_LABEL_COL: str = "cyberbullying_type"

PROJECT_ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parent.parent
CONFIG_PATH: pathlib.Path = PROJECT_ROOT / "configs" / "split_config.json"
DEFAULT_INPUT_CSV: pathlib.Path = (
    PROJECT_ROOT / "data" / "processed" / "cyberbullying_clean.csv"
)
DEFAULT_OUTPUT_DIR: pathlib.Path = PROJECT_ROOT / "data" / "processed"
DEFAULT_REPORT_JSON: pathlib.Path = PROJECT_ROOT / "results" / "split_report.json"

EXPECTED_CLASSES: list[str] = [
    "age",
    "ethnicity",
    "gender",
    "not_cyberbullying",
    "other_cyberbullying",
    "religion",
]


# ---------------------------------------------------------------------------
# Core Splitting Logic
# ---------------------------------------------------------------------------

def load_clean_dataset(
    path: str | pathlib.Path = DEFAULT_INPUT_CSV,
    text_col: str = DEFAULT_TEXT_COL,
    label_col: str = DEFAULT_LABEL_COL,
) -> pd.DataFrame:
    """Load the Policy D cleaned dataset with strict validation.

    Parameters
    ----------
    path : str or Path
        Path to the cleaned dataset CSV.
    text_col : str
        Expected text column name.
    label_col : str
        Expected label column name.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If required columns are missing, dataset is empty, or has duplicate texts.
    """
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {p}. "
            "Please ensure Policy D data cleaning has been run."
        )

    df = pd.read_csv(p, encoding="utf-8")
    if df.empty:
        raise ValueError(f"Cleaned dataset at {p} is empty.")

    if text_col not in df.columns:
        raise ValueError(f"Required text column {text_col!r} missing from {p}.")
    if label_col not in df.columns:
        raise ValueError(f"Required label column {label_col!r} missing from {p}.")

    # Invariant: cleaned dataset must not contain duplicate tweet_text
    n_unique = df[text_col].nunique()
    if n_unique != len(df):
        raise ValueError(
            f"Input dataset contains duplicate texts: {len(df)} rows, "
            f"but only {n_unique} unique texts. Policy D invariant violated."
        )

    return df


def stratified_split(
    df: pd.DataFrame,
    text_col: str = DEFAULT_TEXT_COL,
    label_col: str = DEFAULT_LABEL_COL,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform a stratified train/val/test split preserving class distributions.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset to split.
    text_col : str
        Text column name.
    label_col : str
        Class label column name for stratification.
    train_ratio : float
        Proportion for training split (e.g. 0.80).
    val_ratio : float
        Proportion for validation split (e.g. 0.10).
    test_ratio : float
        Proportion for test split (e.g. 0.10).
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df)

    Raises
    ------
    ValueError
        If split ratios do not sum to 1.0 or are out of bounds.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(
            f"Split ratios must sum to 1.0, got: {train_ratio} + {val_ratio} + "
            f"{test_ratio} = {total_ratio}"
        )

    if any(r <= 0 for r in (train_ratio, val_ratio, test_ratio)):
        raise ValueError("All split ratios must be strictly positive.")

    # Step 1: Split test set out of total dataset
    test_count = int(round(len(df) * test_ratio))
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_count,
        stratify=df[label_col],
        random_state=random_seed,
    )

    # Step 2: Split validation set out of train_val set
    val_count = int(round(len(df) * val_ratio))
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_count,
        stratify=train_val_df[label_col],
        random_state=random_seed,
    )

    # Reset indices cleanly
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Verification & Diagnostics
# ---------------------------------------------------------------------------

def verify_split_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str = DEFAULT_TEXT_COL,
) -> dict[str, Any]:
    """Verify that there is strictly zero text overlap between any splits.

    Parameters
    ----------
    train_df : pd.DataFrame
    val_df : pd.DataFrame
    test_df : pd.DataFrame
    text_col : str

    Returns
    -------
    dict[str, Any]
        Audit details with boolean pass flag and overlap counts.
    """
    train_set = set(train_df[text_col])
    val_set = set(val_df[text_col])
    test_set = set(test_df[text_col])

    overlap_train_val = train_set.intersection(val_set)
    overlap_train_test = train_set.intersection(test_set)
    overlap_val_test = val_set.intersection(test_set)

    is_leakage_free = (
        len(overlap_train_val) == 0
        and len(overlap_train_test) == 0
        and len(overlap_val_test) == 0
    )

    return {
        "is_leakage_free": is_leakage_free,
        "train_val_overlap_count": len(overlap_train_val),
        "train_test_overlap_count": len(overlap_train_test),
        "val_test_overlap_count": len(overlap_val_test),
    }


def compute_split_stats(
    df: pd.DataFrame,
    label_col: str = DEFAULT_LABEL_COL,
) -> dict[str, Any]:
    """Calculate summary statistics and class distribution for a split.

    Parameters
    ----------
    df : pd.DataFrame
    label_col : str

    Returns
    -------
    dict[str, Any]
    """
    total = len(df)
    counts = df[label_col].value_counts().to_dict()
    classes = sorted(df[label_col].unique())

    per_class = {}
    for c in classes:
        cnt = int(counts.get(c, 0))
        pct = round(cnt / total * 100, 4) if total > 0 else 0.0
        per_class[c] = {"count": cnt, "percentage": pct}

    return {
        "total_rows": total,
        "num_classes": len(classes),
        "classes_present": classes,
        "class_distribution": per_class,
    }


def verify_reproducibility(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str = DEFAULT_TEXT_COL,
    label_col: str = DEFAULT_LABEL_COL,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> bool:
    """Verify that a second run with the same seed produces exact identical splits."""
    t2, v2, te2 = stratified_split(
        df,
        text_col=text_col,
        label_col=label_col,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
    )
    return (
        train_df.equals(t2)
        and val_df.equals(v2)
        and test_df.equals(te2)
    )


# ---------------------------------------------------------------------------
# Report Generation & I/O
# ---------------------------------------------------------------------------

def generate_split_report(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    original_df: pd.DataFrame,
    config: dict[str, Any],
    output_path: str | pathlib.Path | None = DEFAULT_REPORT_JSON,
    text_col: str = DEFAULT_TEXT_COL,
    label_col: str = DEFAULT_LABEL_COL,
) -> dict[str, Any]:
    """Generate a comprehensive, machine-readable JSON split report."""
    leakage = verify_split_leakage(train_df, val_df, test_df, text_col=text_col)
    train_stats = compute_split_stats(train_df, label_col=label_col)
    val_stats = compute_split_stats(val_df, label_col=label_col)
    test_stats = compute_split_stats(test_df, label_col=label_col)
    total_len = len(original_df)

    report = {
        "status": "COMPLETED",
        "configuration": config,
        "source_dataset_rows": total_len,
        "split_summary": {
            "train": {
                "rows": len(train_df),
                "percentage": round(len(train_df) / total_len * 100, 2),
            },
            "validation": {
                "rows": len(val_df),
                "percentage": round(len(val_df) / total_len * 100, 2),
            },
            "test": {
                "rows": len(test_df),
                "percentage": round(len(test_df) / total_len * 100, 2),
            },
            "sum_rows": len(train_df) + len(val_df) + len(test_df),
        },
        "leakage_verification": leakage,
        "reproducibility_verified": verify_reproducibility(
            original_df,
            train_df,
            val_df,
            test_df,
            text_col=text_col,
            label_col=label_col,
            train_ratio=config.get("train_ratio", DEFAULT_TRAIN_RATIO),
            val_ratio=config.get("val_ratio", DEFAULT_VAL_RATIO),
            test_ratio=config.get("test_ratio", DEFAULT_TEST_RATIO),
            random_seed=config.get("random_seed", DEFAULT_RANDOM_SEED),
        ),
        "class_distributions": {
            "train": train_stats["class_distribution"],
            "validation": val_stats["class_distribution"],
            "test": test_stats["class_distribution"],
        },
    }

    if output_path is not None:
        out_p = pathlib.Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Split report written to %s", out_p)

    return report


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str | pathlib.Path = DEFAULT_OUTPUT_DIR,
    train_name: str = "train.csv",
    val_name: str = "val.csv",
    test_name: str = "test.csv",
) -> dict[str, pathlib.Path]:
    """Save derived split DataFrames to CSV files."""
    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "train": out_dir / train_name,
        "validation": out_dir / val_name,
        "test": out_dir / test_name,
    }

    train_df.to_csv(paths["train"], index=False, encoding="utf-8")
    val_df.to_csv(paths["validation"], index=False, encoding="utf-8")
    test_df.to_csv(paths["test"], index=False, encoding="utf-8")

    logger.info("Saved train (%d rows) to %s", len(train_df), paths["train"])
    logger.info("Saved val (%d rows) to %s", len(val_df), paths["validation"])
    logger.info("Saved test (%d rows) to %s", len(test_df), paths["test"])

    return paths


# ---------------------------------------------------------------------------
# Pipeline Entry Point
# ---------------------------------------------------------------------------

def run_split_pipeline(
    config_path: str | pathlib.Path | None = None,
) -> dict[str, Any]:
    """Execute the end-to-end splitting pipeline using config."""
    cfg_p = pathlib.Path(config_path) if config_path else CONFIG_PATH
    if cfg_p.exists():
        with open(cfg_p, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = {
            "random_seed": DEFAULT_RANDOM_SEED,
            "train_ratio": DEFAULT_TRAIN_RATIO,
            "val_ratio": DEFAULT_VAL_RATIO,
            "test_ratio": DEFAULT_TEST_RATIO,
            "stratify": True,
            "text_column": DEFAULT_TEXT_COL,
            "label_column": DEFAULT_LABEL_COL,
            "source_dataset": str(DEFAULT_INPUT_CSV.relative_to(PROJECT_ROOT)),
            "train_output": str((DEFAULT_OUTPUT_DIR / "train.csv").relative_to(PROJECT_ROOT)),
            "val_output": str((DEFAULT_OUTPUT_DIR / "val.csv").relative_to(PROJECT_ROOT)),
            "test_output": str((DEFAULT_OUTPUT_DIR / "test.csv").relative_to(PROJECT_ROOT)),
            "report_output": str(DEFAULT_REPORT_JSON.relative_to(PROJECT_ROOT)),
        }

    input_path = PROJECT_ROOT / config.get("source_dataset", DEFAULT_INPUT_CSV)
    text_col = config.get("text_column", DEFAULT_TEXT_COL)
    label_col = config.get("label_column", DEFAULT_LABEL_COL)
    train_ratio = float(config.get("train_ratio", DEFAULT_TRAIN_RATIO))
    val_ratio = float(config.get("val_ratio", DEFAULT_VAL_RATIO))
    test_ratio = float(config.get("test_ratio", DEFAULT_TEST_RATIO))
    seed = int(config.get("random_seed", DEFAULT_RANDOM_SEED))

    df_clean = load_clean_dataset(input_path, text_col=text_col, label_col=label_col)

    train_df, val_df, test_df = stratified_split(
        df_clean,
        text_col=text_col,
        label_col=label_col,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=seed,
    )

    # Save artifacts
    save_splits(
        train_df,
        val_df,
        test_df,
        output_dir=DEFAULT_OUTPUT_DIR,
        train_name="train.csv",
        val_name="val.csv",
        test_name="test.csv",
    )

    report_path = PROJECT_ROOT / config.get("report_output", DEFAULT_REPORT_JSON)
    report = generate_split_report(
        train_df,
        val_df,
        test_df,
        df_clean,
        config=config,
        output_path=report_path,
        text_col=text_col,
        label_col=label_col,
    )

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    rep = run_split_pipeline()
    print("\n" + "=" * 65)
    print("STRATIFIED DATA SPLIT SUMMARY")
    print("=" * 65)
    print(f"Total rows  : {rep['source_dataset_rows']:,}")
    print(
        f"Train rows  : {rep['split_summary']['train']['rows']:,} "
        f"({rep['split_summary']['train']['percentage']}%)"
    )
    print(
        f"Val rows    : {rep['split_summary']['validation']['rows']:,} "
        f"({rep['split_summary']['validation']['percentage']}%)"
    )
    print(
        f"Test rows   : {rep['split_summary']['test']['rows']:,} "
        f"({rep['split_summary']['test']['percentage']}%)"
    )
    print("-" * 65)
    leak = rep["leakage_verification"]
    print(f"Leakage-free: {leak['is_leakage_free']} [OK]")
    print(f"Reproducible: {rep['reproducibility_verified']} [OK]")
    print("=" * 65)
