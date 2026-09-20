# Duplicate and Conflicting-Label Analysis

> [!IMPORTANT]
> This is a read-only analysis. The raw dataset has not been modified.
> No preprocessing, splitting, or cleaning decisions have been made.

Analysis performed on `data/raw/cyberbullying_tweets.csv` (47,692 rows).

---

## 1. Dataset Overview

| Metric | Count |
| -------- | ------: |
| Total rows | 47,692 |
| Unique `tweet_text` values | 46,017 |
| Texts appearing exactly once | 44,342 |
| Texts appearing more than once | 1,675 |

---

## 2. Exact Duplicate Rows

Exact duplicate rows are rows where **both** `tweet_text` and
`cyberbullying_type` are identical.

| Metric | Count |
| -------- | ------: |
| All copies of exact duplicate rows | 72 |
| Non-first (redundant) exact duplicate rows | 36 |
| Unique (text, label) pairs with duplicates | 36 |

Exact duplicate rows are straightforward: the same tweet with the same
label appears more than once. Removing the extra copies would lose no
information.

---

## 3. Repeated `tweet_text` Frequency Distribution

| Appearances | Unique texts  |
|------------:|--------------:|
|           2 |        1,675  |

---

## 4. Consistent vs Conflicting Repeated Texts

| Category                                       | Unique texts | Rows covered |
|------------------------------------------------|-------------:|-------------:|
| Consistent duplicates (same label every time)  |           36 |           72 |
| **Conflicting duplicates (labels differ)**     |    **1,639** |    **3,278** |

- Maximum labels assigned to a single text: **2**
- Conflicting rows represent **6.9%** of the total dataset.

---

## 5. Conflicting Label Combinations

Each row below represents a distinct set of labels observed for the
same tweet text.

| Combination | Unique texts | Rows |
| ------------- | -------------: | -----: |
| `not_cyberbullying + other_cyberbullying` | 1,456 | 2,912 |
| `gender + other_cyberbullying` | 124 | 248 |
| `gender + not_cyberbullying` | 50 | 100 |
| `ethnicity + religion` | 4 | 8 |
| `ethnicity + not_cyberbullying` | 2 | 4 |
| `ethnicity + gender` | 1 | 2 |
| `gender + religion` | 1 | 2 |
| `not_cyberbullying + religion` | 1 | 2 |

---

## 6. Representative Examples per Conflicting Combination

> Texts are truncated to 120 characters for readability.
> Full texts are available in the raw dataset.

### `not_cyberbullying + other_cyberbullying`

**Text (truncated):** #MKR omg my dad and I are screaming at the TV.

**Label occurrences for this text:**

- `not_cyberbullying` × 1
- `other_cyberbullying` × 1

### `gender + other_cyberbullying`

**Text (truncated):** #Kat could Skank for Australia at the next Olympics. #MKR

**Label occurrences for this text:**

- `gender` × 1
- `other_cyberbullying` × 1

### `gender + not_cyberbullying`

**Text (truncated):** #MKR I really hope they get out-sassed

**Label occurrences for this text:**

- `not_cyberbullying` × 1
- `gender` × 1

### `ethnicity + religion`

**Text (truncated):** @GarrettaBrown85 @5Candrew Why do people even talk about white privilege when the majority of food stamp recipients are…

**Label occurrences for this text:**

- `religion` × 1
- `ethnicity` × 1

### `ethnicity + not_cyberbullying`

**Text (truncated):** @SFtheWolf @max2000warlord people confuse empathy with being scared, and that says more about them than anything.

**Label occurrences for this text:**

- `not_cyberbullying` × 1
- `ethnicity` × 1

### `ethnicity + gender`

**Text (truncated):** @YTM1staWu1fy Sweden, man.  The Swedish men are cattle, at this point.

**Label occurrences for this text:**

- `gender` × 1
- `ethnicity` × 1

### `gender + religion`

**Text (truncated):** @moderncomments Liberals suddenly consider ISIS a threat.

**Label occurrences for this text:**

- `gender` × 1
- `religion` × 1

### `not_cyberbullying + religion`

**Text (truncated):** You're not going to get the response you're looking for. Give up. @PeerWorker

**Label occurrences for this text:**

- `not_cyberbullying` × 1
- `religion` × 1

---

## 7. Data-Leakage Implications

> [!CAUTION]
> **The conflicting-label problem is a data-leakage risk.**
>
> 1,639 unique tweet texts (3,278 rows, 6.9% of the dataset)
> appear with more than one label.
>
> If a train/validation/test split is performed on the raw dataset
> **before** resolving conflicts, the same tweet text could appear in
> both the training set and the test set — potentially with a different
> label. This would cause:
>
> - Near-duplicate memorization rather than generalization
> - Artificially inflated test performance
> - Misleading evaluation

---

## 8. Possible Handling Strategies

> [!IMPORTANT]
> **No strategy has been chosen yet.**
> The following options are presented for review. A decision will be
> made after this report has been reviewed.

| # | Strategy | Description | Risk |
| --- | ---------- | ------------- | ------ |
| A | Drop all conflicting rows | Remove all rows whose `tweet_text` appears with >1 label | Loses data; may reduce dataset size significantly |
| B | Keep one label per conflicting text (majority vote) | For each text, keep only rows matching the most frequent label | May introduce label noise; loses minority rows |
| C | Keep one label per conflicting text (first occurrence) | Keep the first-seen label for each text | Arbitrary; depends on row order |
| D | Deduplicate texts before splitting, then split | Remove duplicate `tweet_text` values entirely (keeping one row per text) then split | Simplest leakage prevention; reduces dataset |
| E | Split first, then remove cross-split duplicates | Perform split, then identify and remove any text that appears in both train and test | Cleaner split but more complex implementation |

---

## 9. Notes

- Analysis performed on: `data/raw/cyberbullying_tweets.csv`
- Script: `src/analyze_duplicates.py`
- Machine-readable results: `results/duplicate_analysis.json`
- The raw dataset has **not been modified**.
- No preprocessing or splitting decisions have been made.
