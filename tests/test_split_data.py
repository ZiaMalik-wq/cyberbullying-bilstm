"""
tests/test_split_data.py
========================

Unit and integration tests for leakage-safe train/val/test splitting.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import pytest
import pandas as pd
import numpy as np

from src.split_data import (
    load_clean_dataset,
    stratified_split,
    verify_split_leakage,
    compute_split_stats,
    verify_reproducibility,
    save_splits,
    generate_split_report,
    EXPECTED_CLASSES,
    DEFAULT_RANDOM_SEED,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "raw" / "cyberbullying_tweets.csv"
CLEAN_CSV = ROOT / "data" / "processed" / "cyberbullying_clean.csv"
TRAIN_CSV = ROOT / "data" / "processed" / "train.csv"
VAL_CSV = ROOT / "data" / "processed" / "val.csv"
TEST_CSV = ROOT / "data" / "processed" / "test.csv"
REPORT_JSON = ROOT / "results" / "split_report.json"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_df() -> pd.DataFrame:
    """Fixture providing the actual Policy D cleaned DataFrame."""
    assert CLEAN_CSV.exists(), f"Cleaned dataset missing at {CLEAN_CSV}"
    return pd.read_csv(CLEAN_CSV, encoding="utf-8")


@pytest.fixture
def dummy_df() -> pd.DataFrame:
    """Fixture providing a controlled dummy DataFrame with 6 classes."""
    data = []
    classes = EXPECTED_CLASSES
    for c in classes:
        for i in range(100):  # 100 samples per class = 600 total
            data.append({"tweet_text": f"Sample {c} number {i}", "cyberbullying_type": c})
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

def test_load_clean_dataset_valid(clean_df: pd.DataFrame) -> None:
    """Test that load_clean_dataset loads 44,378 rows with unique texts."""
    df = load_clean_dataset(CLEAN_CSV)
    assert len(df) == 44378
    assert df["tweet_text"].nunique() == 44378
    assert set(df["cyberbullying_type"].unique()) == set(EXPECTED_CLASSES)


def test_load_clean_dataset_missing_file(tmp_path: pathlib.Path) -> None:
    """Test that missing file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_clean_dataset(tmp_path / "non_existent.csv")


def test_load_clean_dataset_duplicate_texts(tmp_path: pathlib.Path) -> None:
    """Test that dataset with duplicate texts raises ValueError."""
    bad_df = pd.DataFrame({
        "tweet_text": ["dup", "dup"],
        "cyberbullying_type": ["age", "age"]
    })
    bad_file = tmp_path / "bad.csv"
    bad_df.to_csv(bad_file, index=False)
    with pytest.raises(ValueError, match="duplicate texts"):
        load_clean_dataset(bad_file)


def test_stratified_split_proportions(dummy_df: pd.DataFrame) -> None:
    """Test split proportions on controlled 600-sample dummy dataset (80/10/10)."""
    train_df, val_df, test_df = stratified_split(
        dummy_df,
        train_ratio=0.80,
        val_ratio=0.10,
        test_ratio=0.10,
        random_seed=42,
    )
    assert len(train_df) == 480  # 80% of 600
    assert len(val_df) == 60    # 10% of 600
    assert len(test_df) == 60   # 10% of 600
    assert len(train_df) + len(val_df) + len(test_df) == 600


def test_stratified_split_class_coverage(dummy_df: pd.DataFrame) -> None:
    """Test that every class has exact expected counts in each split."""
    train_df, val_df, test_df = stratified_split(dummy_df, random_seed=42)
    for c in EXPECTED_CLASSES:
        assert (train_df["cyberbullying_type"] == c).sum() == 80
        assert (val_df["cyberbullying_type"] == c).sum() == 10
        assert (test_df["cyberbullying_type"] == c).sum() == 10


def test_stratified_split_invalid_ratios(dummy_df: pd.DataFrame) -> None:
    """Test that ratios not summing to 1.0 or non-positive values error."""
    with pytest.raises(ValueError, match="must sum to 1.0"):
        stratified_split(dummy_df, train_ratio=0.7, val_ratio=0.1, test_ratio=0.1)

    with pytest.raises(ValueError, match="strictly positive"):
        stratified_split(dummy_df, train_ratio=0.9, val_ratio=0.1, test_ratio=0.0)


def test_text_disjointness_zero_leakage(dummy_df: pd.DataFrame) -> None:
    """Test verify_split_leakage flags no overlap on clean split."""
    train_df, val_df, test_df = stratified_split(dummy_df, random_seed=42)
    audit = verify_split_leakage(train_df, val_df, test_df)
    assert audit["is_leakage_free"] is True
    assert audit["train_val_overlap_count"] == 0
    assert audit["train_test_overlap_count"] == 0
    assert audit["val_test_overlap_count"] == 0


def test_leakage_detection_catches_overlap() -> None:
    """Test verify_split_leakage catches simulated overlapping texts."""
    train_df = pd.DataFrame({"tweet_text": ["text_A", "text_B"], "cyberbullying_type": ["age", "gender"]})
    val_df = pd.DataFrame({"tweet_text": ["text_B", "text_C"], "cyberbullying_type": ["gender", "religion"]})
    test_df = pd.DataFrame({"tweet_text": ["text_D", "text_E"], "cyberbullying_type": ["ethnicity", "age"]})

    audit = verify_split_leakage(train_df, val_df, test_df)
    assert audit["is_leakage_free"] is False
    assert audit["train_val_overlap_count"] == 1


def test_reproducibility(dummy_df: pd.DataFrame) -> None:
    """Test that identical seeds yield identical DataFrames."""
    t1, v1, te1 = stratified_split(dummy_df, random_seed=42)
    assert verify_reproducibility(dummy_df, t1, v1, te1, random_seed=42) is True


def test_different_seeds_produce_different_splits(dummy_df: pd.DataFrame) -> None:
    """Test that different seeds produce distinct permutations."""
    t1, _, _ = stratified_split(dummy_df, random_seed=42)
    t2, _, _ = stratified_split(dummy_df, random_seed=99)
    assert not t1["tweet_text"].equals(t2["tweet_text"])


# ---------------------------------------------------------------------------
# Integration Tests on Actual Saved Artifacts
# ---------------------------------------------------------------------------

def test_actual_artifacts_exist() -> None:
    """Verify train.csv, val.csv, test.csv, and split_report.json exist."""
    assert TRAIN_CSV.exists(), f"Missing {TRAIN_CSV}"
    assert VAL_CSV.exists(), f"Missing {VAL_CSV}"
    assert TEST_CSV.exists(), f"Missing {TEST_CSV}"
    assert REPORT_JSON.exists(), f"Missing {REPORT_JSON}"


def test_actual_artifacts_row_counts() -> None:
    """Verify exact row counts: train=35502, val=4438, test=4438 (sum=44378)."""
    train_df = pd.read_csv(TRAIN_CSV, encoding="utf-8")
    val_df = pd.read_csv(VAL_CSV, encoding="utf-8")
    test_df = pd.read_csv(TEST_CSV, encoding="utf-8")

    assert len(train_df) == 35502
    assert len(val_df) == 4438
    assert len(test_df) == 4438
    assert len(train_df) + len(val_df) + len(test_df) == 44378


def test_actual_artifacts_zero_leakage() -> None:
    """Verify zero tweet_text overlap between actual train, val, and test files."""
    train_df = pd.read_csv(TRAIN_CSV, encoding="utf-8")
    val_df = pd.read_csv(VAL_CSV, encoding="utf-8")
    test_df = pd.read_csv(TEST_CSV, encoding="utf-8")

    train_set = set(train_df["tweet_text"])
    val_set = set(val_df["tweet_text"])
    test_set = set(test_df["tweet_text"])

    assert len(train_set.intersection(val_set)) == 0
    assert len(train_set.intersection(test_set)) == 0
    assert len(val_set.intersection(test_set)) == 0


def test_actual_artifacts_all_classes_preserved() -> None:
    """Verify that all 6 classes are present in actual train, val, and test files."""
    train_df = pd.read_csv(TRAIN_CSV, encoding="utf-8")
    val_df = pd.read_csv(VAL_CSV, encoding="utf-8")
    test_df = pd.read_csv(TEST_CSV, encoding="utf-8")

    for df, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        present = set(df["cyberbullying_type"].unique())
        assert present == set(EXPECTED_CLASSES), f"Classes missing in {name}: {set(EXPECTED_CLASSES) - present}"


def test_actual_report_matches_artifacts() -> None:
    """Verify split_report.json matches actual artifact counts and statuses."""
    with open(REPORT_JSON, "r", encoding="utf-8") as f:
        rep = json.load(f)

    assert rep["status"] == "COMPLETED"
    assert rep["source_dataset_rows"] == 44378
    assert rep["split_summary"]["train"]["rows"] == 35502
    assert rep["split_summary"]["validation"]["rows"] == 4438
    assert rep["split_summary"]["test"]["rows"] == 4438
    assert rep["leakage_verification"]["is_leakage_free"] is True
    assert rep["reproducibility_verified"] is True


def test_source_data_unmodified() -> None:
    """Verify that raw dataset and cleaned Policy D dataset have not been altered."""
    assert RAW_CSV.exists()
    assert CLEAN_CSV.exists()

    df_raw = pd.read_csv(RAW_CSV, encoding="utf-8")
    df_clean = pd.read_csv(CLEAN_CSV, encoding="utf-8")

    assert len(df_raw) == 47692
    assert len(df_clean) == 44378
    assert df_clean["tweet_text"].nunique() == 44378
