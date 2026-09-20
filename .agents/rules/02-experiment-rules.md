---
trigger: always_on
---

# Experiment Rules

Every experiment must answer a question.

Do not run experiments merely to maximize a metric.

---

## Experiment Structure

Every experiment should have:

- Experiment ID
- Research question
- Hypothesis
- Model
- Independent variable
- Controlled variables
- Configuration
- Metrics
- Result
- Interpretation

---

## Architecture Experiments

Compare:

- baseline
- Simple RNN
- LSTM
- GRU
- Bi-LSTM

---

## Hyperparameter Experiments

Potential variables:

- sequence length
- vocabulary size
- embedding dimension
- recurrent units
- dropout
- learning rate
- batch size

Do not perform an unnecessarily large grid search.

---

## Controlled Experiments

When investigating a single variable, keep other important variables fixed.

Do not change multiple major variables and then attribute the result to one
variable.

---

## Test Set

Never use test-set performance to select:
- architecture
- hyperparameters
- preprocessing
- sequence length
- vocabulary size

The test set is used for final evaluation.

---

## Results

Only actual executed results may be recorded as results.

Use `Pending` for experiments that have not been executed.

---

## Interpretation

Do not automatically assume that the highest accuracy represents the best
model.

Consider:
- macro F1
- per-class behavior
- confusion matrix
- generalization
- parameter count
- training cost
- stability