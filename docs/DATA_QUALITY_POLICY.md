# Data Quality Policy: Policy D

## 1. Executive Summary

Following a quantitative duplicate analysis (`docs/DUPLICATE_ANALYSIS.md`) and policy impact evaluation (`docs/DUPLICATE_POLICY_IMPACT.md`), **Policy D** has been formally selected as the data-quality policy for the cyberbullying classification dataset.

This policy resolves all duplicate and conflicting-label records prior to train/validation/test splitting, ensuring data integrity, preventing data leakage across splits, and eliminating label ambiguity.

The raw dataset (`data/raw/cyberbullying_tweets.csv`) remains strictly **unmodified**. All cleaning operations produce a derived dataset saved to `data/processed/cyberbullying_clean.csv`.

---

## 2. Policy Definition

**Policy D** applies a two-step deterministic cleaning procedure:

1. **Remove Conflicting Texts**:
   Every row whose `tweet_text` appears with more than one distinct `cyberbullying_type` label is completely excluded. No label is assigned to any conflicting text.
2. **Deduplicate Consistent Duplicates**:
   For `tweet_text` values that appear multiple times with the *same* label (consistent duplicates), retain exactly one copy (the first occurrence) and remove the redundant duplicates.

### Key Characteristics

- **Zero NLP Preprocessing**: Duplicate detection and conflict removal operate on the original `tweet_text` values exactly as stored in the raw CSV. No lowercasing, punctuation stripping, stopword removal, tokenization, or text normalization is performed at this stage.
- **Exact Preservation**: Retained tweets preserve their exact text strings and original assigned labels.
- **No Arbitrary Ground Truth**: No heuristic or arbitrary label choices are made.

---

## 3. Rationale for Policy Selection

### Why Majority Voting Is Impossible

An intuitive approach to resolving label conflicts is majority voting (assigning the label with the most annotations). However, empirical analysis of all 1,639 conflicting `tweet_text` values revealed:

- **100% of conflicting texts have exactly two occurrences** (one row with Label A, one row with Label B).
- **Zero conflicting texts have 3 or more occurrences.**
- Every single conflict is an exact **1-vs-1 tie**.

Majority voting is therefore mathematically impossible without introducing arbitrary tie-breaking rules.

### Rejection of Policy B (Arbitrary First-Seen Label)

Policy B retains 46,017 rows by keeping the first occurrence of each unique text. For conflicting texts, this arbitrarily assigns whichever label happened to appear first in the CSV order.

Because the row ordering in the raw dataset is not a semantic signal, assigning ground-truth labels based on first appearance introduces arbitrary label noise and corrupts evaluation integrity. Therefore, **Policy B is NOT a scientifically justified conflict-resolution strategy**, and its higher retention rate does not justify its adoption.

### Prevention of Data Leakage and Label Ambiguity

- **Cross-Split Leakage**: If duplicate texts remain in the dataset, the same text could appear in both the training set and the test set, leading to falsely inflated test performance.
- **Ground-Truth Ambiguity**: Training or testing a neural network on contradictory supervision (identical input with different targets) harms model convergence and invalidates metric interpretations.

---

## 4. Quantitative Impact Summary

| Metric | Raw Dataset | Cleaned Dataset (Policy D) | Delta / Removed |
| :--- | :---: | :---: | :---: |
| **Total Rows** | 47,692 | **44,378** | -3,314 (-6.95%) |
| **Unique `tweet_text`** | 46,017 | **44,378** | -1,639 |
| **Duplicate Texts Remaining** | 1,675 | **0** | -1,675 |
| **Conflicting Texts Remaining** | 1,639 | **0** | -1,639 |
| **Conflicting Rows Removed** | — | **3,278** | (all copies of 1,639 texts) |
| **Redundant Copies Removed** | — | **36** | (from 36 consistent duplicate groups) |
| **Data Retention Rate** | 100.00% | **93.05%** | — |

---

## 5. Class Distribution: Before and After

| Class Label | Raw Count | Raw % | Cleaned Count | Cleaned % | Net Change |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `age` | 7,992 | 16.76% | 7,992 | 18.01% | 0 (0.00%) |
| `ethnicity` | 7,961 | 16.69% | 7,952 | 17.92% | -9 (-0.11%) |
| `gender` | 7,973 | 16.72% | 7,772 | 17.51% | -201 (-2.52%) |
| `not_cyberbullying` | 7,945 | 16.66% | 6,428 | 14.48% | -1,517 (-19.09%) |
| `other_cyberbullying` | 7,823 | 16.40% | 6,243 | 14.07% | -1,580 (-20.19%) |
| `religion` | 7,998 | 16.77% | 7,991 | 18.01% | -7 (-0.09%) |
| **Total** | **47,692** | **100.00%** | **44,378** | **100.00%** | **-3,314** |

### Observations on Distribution Shift

- `age` is completely unaffected (0 conflicts, 0 consistent duplicates).
- `ethnicity` and `religion` experience negligible row loss (<0.15%).
- `not_cyberbullying` and `other_cyberbullying` bear 94.5% of all conflict removals (1,509 and 1,580 rows respectively). This highlights substantial annotator confusion between general non-cyberbullying and unstructured "other" cyberbullying.
- The resulting class distribution remains sufficiently balanced for multiclass learning, while eliminating all contradictory supervisory signals.

---

## 6. Implementation & Verification

- **Implementation Module**: [`src/data_cleaning.py`](file:///d:/Semester7/ANN/Lab/LabProject/cyberbullying-bilstm/src/data_cleaning.py)
- **Unit & Integration Tests**: [`tests/test_data_cleaning.py`](file:///d:/Semester7/ANN/Lab/LabProject/cyberbullying-bilstm/tests/test_data_cleaning.py) (18/18 passed)
- **Machine-Readable Report**: [`results/data_cleaning_report.json`](file:///d:/Semester7/ANN/Lab/LabProject/cyberbullying-bilstm/results/data_cleaning_report.json)
- **Derived Dataset**: `data/processed/cyberbullying_clean.csv` (excluded from git tracking via `.gitignore`)
- **Raw Dataset**: `data/raw/cyberbullying_tweets.csv` (read-only, untouched)
