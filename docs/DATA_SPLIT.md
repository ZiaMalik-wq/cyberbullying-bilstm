# Leakage-Safe Train / Validation / Test Split

## 1. Executive Summary

Following Policy D data cleaning (`docs/DATA_QUALITY_POLICY.md`), the cyberbullying classification dataset was partitioned into **Training (80%)**, **Validation (10%)**, and **Test (10%)** sets.

The splitting procedure was executed using stratified sampling with a fixed random seed (`seed = 42`) prior to any text preprocessing, tokenization, vocabulary learning, or model development.

Strict leakage audits confirm **zero sample overlap** between any partitions, full preservation of all six classes in every partition, and complete reproducibility.

---

## 2. Splitting Methodology & Invariants

### 2.1 Pre-Vectorization Splitting Invariant

Splitting is performed **strictly before** any NLP preprocessing, feature engineering, or `TextVectorization` vocabulary learning:

- **Zero Vocabulary Leakage**: The vocabulary of token-to-index mappings must reflect real-world deployment where test data is unseen. Fitting a tokenizer or vectorizer across the full dataset before splitting introduces data leakage (the model gains access to the frequency and vocabulary distribution of test tokens).
- **Zero Evaluation Contamination**: Model selection, early stopping, and hyperparameter decisions use the validation set exclusively. The test set is sequestered untouched for final performance reporting.

### 2.2 Stratified Partitioning Protocol

Because the Policy D cleaned dataset exhibits minor variations in class sizes (from 6,243 `other_cyberbullying` to 7,992 `age`), standard random splitting could introduce slight class proportion drift. Stratified sampling guarantees that each of the six classes is partitioned in exact 80/10/10 proportions:

1. **Test Set Extraction**: 10.0% ($4,438$ samples) stratified by `cyberbullying_type` (`random_state=42`).
2. **Validation Set Extraction**: 10.0% of total ($4,438$ samples) stratified from the remaining 90% partition (`random_state=42`).
3. **Training Set**: Remaining 80.0% ($35,502$ samples).

### 2.3 Source Data Immutability

- Raw dataset (`data/raw/cyberbullying_tweets.csv`, 47,692 rows) remains strictly read-only and unmodified.
- Clean dataset (`data/processed/cyberbullying_clean.csv`, 44,378 rows) remains strictly read-only and unmodified.

---

## 3. Quantitative Split Summary

| Partition | Proportion | Row Count | Unique `tweet_text` | Null Values | Classes Present |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Training (`train.csv`)** | **80.00%** | **35,502** | 35,502 | 0 | 6 / 6 |
| **Validation (`val.csv`)** | **10.00%** | **4,438** | 4,438 | 0 | 6 / 6 |
| **Test (`test.csv`)** | **10.00%** | **4,438** | 4,438 | 0 | 6 / 6 |
| **Total** | **100.00%** | **44,378** | **44,378** | **0** | **6 / 6** |

---

## 4. Class Distribution Across Splits

| Class Label | Clean Dataset Count | Train Count (80%) | Val Count (10%) | Test Count (10%) | Train % | Val % | Test % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `age` | 7,992 | 6,393 | 800 | 799 | 18.01% | 18.03% | 18.00% |
| `religion` | 7,991 | 6,393 | 799 | 799 | 18.01% | 18.00% | 18.00% |
| `ethnicity` | 7,952 | 6,362 | 795 | 795 | 17.92% | 17.91% | 17.91% |
| `gender` | 7,772 | 6,218 | 777 | 777 | 17.51% | 17.51% | 17.51% |
| `not_cyberbullying` | 6,428 | 5,142 | 643 | 643 | 14.48% | 14.49% | 14.49% |
| `other_cyberbullying` | 6,243 | 4,994 | 624 | 625 | 14.07% | 14.06% | 14.08% |
| **Total** | **44,378** | **35,502** | **4,438** | **4,438** | **100.00%** | **100.00%** | **100.00%** |

---

## 5. Leakage & Reproducibility Audit

A comprehensive programmatic audit verified the following invariants:

| Verification Check | Target / Invariant | Measured Result | Status |
| :--- | :---: | :---: | :---: |
| **Train $\cap$ Val Overlap** | 0 samples | 0 | **PASSED** |
| **Train $\cap$ Test Overlap** | 0 samples | 0 | **PASSED** |
| **Val $\cap$ Test Overlap** | 0 samples | 0 | **PASSED** |
| **Class Coverage** | All 6 classes in every split | 6 / 6 in all splits | **PASSED** |
| **Sum Preservation** | $35,502 + 4,438 + 4,438 = 44,378$ | 44,378 rows | **PASSED** |
| **Deterministic Seed** | Re-run with seed 42 produces identical files | Identical bit-for-bit | **PASSED** |
| **Source Immutability** | Source clean CSV unchanged | Unmodified | **PASSED** |

---

## 6. Implementation & Artifacts

- **Reusable Split Module**: [`src/split_data.py`](file:///d:/Semester7/ANN/Lab/LabProject/cyberbullying-bilstm/src/split_data.py)
- **Centralized Configuration**: [`configs/split_config.json`](file:///d:/Semester7/ANN/Lab/LabProject/cyberbullying-bilstm/configs/split_config.json)
- **Derived Datasets**:
  - `data/processed/train.csv` (35,502 rows)
  - `data/processed/val.csv` (4,438 rows)
  - `data/processed/test.csv` (4,438 rows)
- **Machine-Readable Report**: [`results/split_report.json`](file:///d:/Semester7/ANN/Lab/LabProject/cyberbullying-bilstm/results/split_report.json)
- **Automated Tests**: [`tests/test_split_data.py`](file:///d:/Semester7/ANN/Lab/LabProject/cyberbullying-bilstm/tests/test_split_data.py) (16/16 passed)
