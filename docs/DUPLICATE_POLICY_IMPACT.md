# Duplicate Policy Impact Analysis

> [!IMPORTANT]
> This is a quantitative analysis only. No policy has been chosen.
> The raw dataset has not been modified.
> No preprocessing, splitting, or cleaning has been performed.
> All numbers are from actual execution against the raw dataset.

Source: `data/raw/cyberbullying_tweets.csv`  (47,692 rows)

---

## 1. Baseline Class Distribution

| Label | Count | Percentage |
| ------- | ------: | -----------: |
| `age` | 7,992 | 16.76% |
| `ethnicity` | 7,961 | 16.69% |
| `gender` | 7,973 | 16.72% |
| `not_cyberbullying` | 7,945 | 16.66% |
| `other_cyberbullying` | 7,823 | 16.40% |
| `religion` | 7,998 | 16.77% |
| **Total** | **47,692** | **100.00%** |

---

## 2. Conflicting Texts — Per-Class Impact

**1,639 unique tweet texts** appear with conflicting labels,
covering **3,278 rows** (6.9% of the dataset).

| Label | Rows in conflicting texts | Unique conflicting texts |
| ------- | -------------------------: | -------------------------: |
| `age` | 0 | 0 |
| `ethnicity` | 7 | 7 |
| `gender` | 176 | 176 |
| `not_cyberbullying` | 1,509 | 1,509 |
| `other_cyberbullying` | 1,580 | 1,580 |
| `religion` | 6 | 6 |

---

## 3. Majority-Vote Verification

> [!NOTE]
> Conflicting texts with a clear majority label: **0**
> Conflicting texts that are tied: **1,639**
>
> Every conflicting text is a perfect tie. Majority voting cannot resolve any conflict in this dataset.

Every conflicting tweet appears **exactly twice**, each time with a **different**
label. The counts are 1 vs 1 for every conflict — there is no majority.
**Majority voting is not a valid resolution strategy for this dataset.**

---

## 4. Consistent Duplicates — Class Distribution

| Label | All copies | Redundant copies |
| ------- | -----------: | -----------------: |
| `age` | 0 | 0 |
| `ethnicity` | 4 | 2 |
| `gender` | 50 | 25 |
| `not_cyberbullying` | 16 | 8 |
| `other_cyberbullying` | 0 | 0 |
| `religion` | 2 | 1 |
| **Total** | **72** | **36** |

Consistent duplicates share the same label in both copies.
Removing the redundant copy loses no label information.

---

## 5. Policy Definitions

| Policy | Definition |
| -------- | ------------ |
| **A** | Remove all rows belonging to the 1,639 conflicting texts. Keep consistent duplicates unchanged. |
| **B** | Deduplicate to one row per unique `tweet_text` (first occurrence). Resolves conflicts by keeping the first-seen label — an **arbitrary** choice for the 1,639 conflicting texts. |
| **C** | Explicitly keep all consistent-duplicate rows; remove all conflicting texts. *(Numerically identical to Policy A in this dataset.)* |
| **D** | Remove all conflicting texts **AND** deduplicate consistent duplicates to one copy each. No arbitrary label assignment required. |

---

## 6. Policy Comparison — Row Counts

| Metric | Baseline | Policy A | Policy B | Policy C | Policy D |
| -------- | ----------: | ---------: | ---------: | ---------: | ---------: |
| Total rows | 47,692 | 44,414 | 46,017 | 44,414 | 44,378 |
| Rows removed | 0 | 3,278 | 1,675 | 3,278 | 3,314 |
| Unique tweet_text | 46,017 | 44,378 | 46,017 | 44,378 | 44,378 |
| Dup texts remaining | 1,675 | 36 | 0 | 36 | 0 |
| Conflict texts remaining | 1,639 | 0 | 0 | 0 | 0 |
| % data retained | 100.00% | 93.13% | 96.49% | 93.13% | 93.05% |

---

## 7. Per-Class Counts by Policy

| Label | Baseline | Policy A | Policy B | Policy C | Policy D |
| ------- | ----------: | ---------: | ---------: | ---------: | ---------: |
| `age` | 7,992 | 7,992 | 7,992 | 7,992 | 7,992 |
| `ethnicity` | 7,961 | 7,954 | 7,952 | 7,954 | 7,952 |
| `gender` | 7,973 | 7,797 | 7,898 | 7,797 | 7,772 |
| `not_cyberbullying` | 7,945 | 6,436 | 7,937 | 6,436 | 6,428 |
| `other_cyberbullying` | 7,823 | 6,243 | 6,243 | 6,243 | 6,243 |
| `religion` | 7,998 | 7,992 | 7,995 | 7,992 | 7,991 |
| **Total** | **47,692** | **44,414** | **46,017** | **44,414** | **44,378** |

---

## 8. Per-Class Percentages by Policy

| Label | Baseline | Policy A | Policy B | Policy C | Policy D |
| ------- | ----------: | ---------: | ---------: | ---------: | ---------: |
| `age` | 16.76% | 17.99% | 17.37% | 17.99% | 18.01% |
| `ethnicity` | 16.69% | 17.91% | 17.28% | 17.91% | 17.92% |
| `gender` | 16.72% | 17.56% | 17.16% | 17.56% | 17.51% |
| `not_cyberbullying` | 16.66% | 14.49% | 17.25% | 14.49% | 14.48% |
| `other_cyberbullying` | 16.40% | 14.06% | 13.57% | 14.06% | 14.07% |
| `religion` | 16.77% | 17.99% | 17.37% | 17.99% | 18.01% |

---

## 9. Methodological Considerations

| Consideration | Policy A | Policy B | Policy C | Policy D |
| --------------- | ---------- | ---------- | ---------- | ---------- |
| Requires arbitrary label assignment | No | **Yes** (1,639 texts) | No | No |
| Conflicting texts remain | No | No | No | No |
| Duplicate tweet_text values remain | Yes (36 texts) | No | Yes (36 texts) | No |
| Data loss from conflicts | Yes | Yes (indirectly) | Yes | Yes |
| A and C are numerically identical | Yes | — | Yes | — |
| Largest retained dataset | — | Yes (46,017) | — | — |
| Cleanest result (no dups, no conflicts, scientifically justified) | — | No (arbitrary labels) | — | **Yes** |

---

## 10. Selected Policy: Policy D

> [!IMPORTANT]
> **Policy D has been formally selected as the data-quality policy.**
>
> Although Policy B retains the largest row count (46,017 rows), it relies on an arbitrary first-occurrence label assignment for 1,639 conflicting texts and is **NOT** a scientifically justified conflict-resolution strategy. Its larger retained dataset does not make it preferable.
>
> **Policy D** was selected because it is the only approach that completely eliminates duplicate texts and conflicting ground-truth labels without introducing arbitrary label assignments.
>
> Detailed specification and implementation details are documented in [docs/DATA_QUALITY_POLICY.md](file:///d:/Semester7/ANN/Lab/LabProject/cyberbullying-bilstm/docs/DATA_QUALITY_POLICY.md).

---

## 11. Notes

- Script: `src/analyze_policy_impact.py`
- Machine-readable results: `results/duplicate_policy_impact.json`
- The raw dataset has **not been modified**.
- No train/val/test split has been created.
- All numbers are from actual execution on the raw dataset.
