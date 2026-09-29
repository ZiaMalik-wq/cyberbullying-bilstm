# Fine-Grained Cyberbullying Classification Using Bi-LSTM

## 1. Project Overview

This project is a university Artificial Neural Networks and Deep Learning
laboratory project.

The objective is to develop a deep-learning based text classification system
that identifies different categories of cyberbullying in social-media text.

The primary model will be a Bidirectional Long Short-Term Memory (Bi-LSTM)
network trained entirely from scratch.

The project will be developed over approximately four weeks.

---

## 2. Problem Statement

Traditional binary cyberbullying detection only determines whether a text
contains cyberbullying.

This project investigates fine-grained multiclass classification, where a text
sample is classified into one of six categories:

1. Age
2. Ethnicity
3. Gender
4. Religion
5. Other Cyberbullying
6. Not Cyberbullying

The model receives text as input and predicts one of the six categories.

---

## 3. Main Research Question

How effectively can a Bi-LSTM trained entirely from scratch classify different
types of cyberbullying from social-media text, and how does it compare with
alternative recurrent neural architectures?

---

## 4. Secondary Research Questions

1. How does a Simple RNN compare with LSTM, GRU, and Bi-LSTM?
2. How does sequence length affect classification performance?
3. How does embedding dimension affect classification performance?
4. How does recurrent-layer size affect classification performance?
5. Which cyberbullying categories are most difficult to distinguish?
6. What patterns occur in incorrectly classified samples?

---

## 5. Dataset

The project uses the Cyberbullying Classification dataset containing social-media
text samples belonging to six categories.

The raw dataset (`data/raw/cyberbullying_tweets.csv`) contains 47,692 rows and
2 columns: `tweet_text` and `cyberbullying_type`. The raw dataset is strictly
read-only and must never be modified.

### Selected Data-Quality Policy: Policy D

Prior to data splitting, text preprocessing, or modeling, quantitative duplicate
and conflict analyses were conducted (`docs/DUPLICATE_ANALYSIS.md`,
`docs/DUPLICATE_POLICY_IMPACT.md`). The project formally selected **Policy D**
as documented in `docs/DATA_QUALITY_POLICY.md`:

1. **Remove Conflicting Texts**: All 1,639 text groups with contradictory labels
   (3,278 rows) are completely excluded. Because all conflicts are exact 1-vs-1
   ties, majority voting is impossible, and arbitrary label assignment (Policy B)
   is rejected as unscientific.
2. **Deduplicate Consistent Duplicates**: 36 redundant copies from consistent
   duplicate groups are removed, leaving exactly one copy per unique text.
3. **Derived Clean Dataset**: The resulting dataset contains exactly 44,378 rows,
   44,378 unique `tweet_text` values, 0 duplicates, and 0 conflicting labels.
   It is saved to `data/processed/cyberbullying_clean.csv`.
4. **Preservation of Raw Text**: Duplicate detection and cleaning were executed
   on exact string matches before any NLP preprocessing, text normalization,
   tokenization, or dataset splitting.

---

## 6. Classes

The intended class categories are:

1. Age
2. Ethnicity
3. Gender
4. Religion
5. Other Cyberbullying
6. Not Cyberbullying

The actual labels and their representation in the dataset must be verified
before implementation.

A single canonical label mapping must be used throughout the project.

---

## 7. Deep Learning Constraints

The project MUST NOT use pretrained models.

Forbidden:

- BERT
- RoBERTa
- DistilBERT
- GPT
- pretrained Transformer models
- Word2Vec pretrained embeddings
- GloVe pretrained embeddings
- FastText pretrained embeddings
- sentence-transformer models
- any external pretrained language representation

The embedding layer must be initialized and trained from scratch using the
project's training data.

---

## 8. Technology Stack

Primary language:

Python 3

Deep learning:

TensorFlow / Keras

Data processing:

- Pandas
- NumPy

Machine learning evaluation:

- Scikit-learn

Visualization:

- Matplotlib
- Seaborn

Experiment monitoring:

- TensorBoard

Development environment:

- Google Colab
- Kaggle Notebook as an alternative

The local development machine is:

Dell Vostro 15 3530
16 GB RAM
512 GB SSD

Heavy model training should use an available Colab/Kaggle GPU.

---

## 9. Text Processing

The primary text vectorization method should use:

Keras TextVectorization

The vocabulary must be learned using training data only.

The preprocessing pipeline must avoid data leakage.

Do not aggressively remove linguistic information without considering its
possible effect on cyberbullying classification.

Social-media characteristics such as hashtags, mentions, punctuation,
capitalization, repeated characters, slang, and URLs must be considered
carefully rather than automatically removed.

---

## 10. Data Splitting

The dataset should be divided into:

- Training set
- Validation set
- Test set

A stratified split should be used when appropriate.

The test set must remain untouched until final evaluation.

No test-set information may influence:

- vocabulary construction
- hyperparameter selection
- architecture selection
- preprocessing decisions

---

## 11. Baseline and Comparison Models

The project should investigate the following models:

1. Neural baseline
2. Simple RNN
3. LSTM
4. GRU
5. Bi-LSTM

All models must use trainable embeddings rather than pretrained embeddings.

---

## 12. Main Bi-LSTM Architecture

The initial conceptual architecture is:

Input Text
    ↓
TextVectorization
    ↓
Trainable Embedding
    ↓
Bidirectional LSTM
    ↓
Dropout
    ↓
Dense
    ↓
Dropout
    ↓
Dense(6)
    ↓
Softmax

Exact hyperparameters must remain configurable and should be determined
through experimentation rather than being assumed to be optimal.

---

## 13. Training

The initial optimizer should be Adam.

The model should use an appropriate multiclass cross-entropy loss.

Training should support:

- early stopping
- model checkpointing
- validation monitoring
- reproducible random seeds

Training hyperparameters must be configurable.

---

## 14. Evaluation Metrics

The project must report:

### Overall metrics

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Weighted F1

### Class-level metrics

- Precision
- Recall
- F1
- Support

### Additional analysis

- Confusion matrix
- Training/validation loss curves
- Training/validation accuracy curves
- Parameter count
- Training duration
- Best validation performance

---

## 15. Experimentation

Experiments should investigate:

### Architecture

- Neural baseline
- Simple RNN
- LSTM
- GRU
- Bi-LSTM

### Hyperparameters

Potential variables:

- vocabulary size
- sequence length
- embedding dimension
- recurrent units
- dropout
- batch size
- learning rate

Experiments should remain reasonably small and manageable for a four-week
university project.

---

## 16. Ablation Study

The final model should undergo selected ablation experiments.

Possible comparisons include:

- LSTM vs Bi-LSTM
- with vs without dropout
- different sequence lengths
- different embedding dimensions
- different recurrent-layer sizes

The purpose is to determine which architectural choices contribute to the
observed performance.

---

## 17. Error Analysis

The final project must analyze incorrectly classified examples.

Investigate possible causes such as:

- ambiguous language
- sarcasm
- slang
- spelling variations
- short text
- overlapping categories
- insufficient context
- unusual linguistic patterns

Do not fabricate explanations. Base conclusions on actual examples.

---

## 18. Reproducibility

Experiments should use explicit random seeds where practical.

Each experiment should record:

- experiment ID
- model type
- random seed
- dataset split
- vocabulary size
- sequence length
- embedding dimension
- recurrent units
- dropout
- optimizer
- learning rate
- batch size
- epochs
- validation metrics
- final metrics
- training time
- parameter count

---

## 19. Academic Integrity

The project must contain only experimentally verified results.

Never fabricate:

- accuracy
- precision
- recall
- F1
- dataset statistics
- training time
- parameter counts
- experimental outcomes

If code has not been executed, its result must be marked as pending.

The student should be able to understand and explain every important part of
the implementation.

---

## 20. Project Scope

Priority order:

1. Correctness
2. Experimental validity
3. Reproducibility
4. Clean implementation
5. Meaningful analysis
6. Documentation
7. Optional demonstration interface

A GUI or web interface is optional and must not take priority over the
machine-learning experiments.

---

## 21. Development Workflow

The project should follow:

Explore
→ Plan
→ Implement
→ Test
→ Review
→ Document

Large changes should not be implemented blindly.

Before significant implementation:

- inspect existing files
- identify dependencies
- identify assumptions
- propose a plan

After implementation:

- run relevant tests
- inspect outputs
- review for data leakage
- review for reproducibility
- document the change

---

## 22. Important Rule

Never claim that code, experiments, or tests were executed unless they were
actually executed and the output was observed.

Never invent experimental results.
