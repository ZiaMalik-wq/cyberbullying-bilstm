---
trigger: always_on
---

# Machine Learning Rules

## Framework

Use:

- Python
- TensorFlow
- Keras
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- TensorBoard

Do not introduce PyTorch unless explicitly requested.

---

## Text Vectorization

Use Keras TextVectorization for the primary text-to-sequence pipeline.

The vocabulary must be learned from training data only.

Keep the vectorization pipeline reproducible.

---

## Model Architecture

The main architecture is:

Text
→ TextVectorization
→ Trainable Embedding
→ Bidirectional LSTM
→ Dropout
→ Dense
→ Dropout
→ Dense(6)
→ Softmax

Exact values must be configurable.

---

## Comparison Models

Implement comparable models for:

1. Neural baseline
2. Simple RNN
3. LSTM
4. GRU
5. Bi-LSTM

Comparison should be fair.

Keep important training conditions consistent unless an experiment explicitly
investigates a different condition.

---

## Training

Use Adam initially.

Use an appropriate multiclass loss.

Support:

- early stopping
- model checkpointing
- validation monitoring

Do not train indefinitely.

---

## Evaluation

Always calculate:

- accuracy
- macro precision
- macro recall
- macro F1
- weighted F1

For final evaluation calculate:

- per-class precision
- per-class recall
- per-class F1
- confusion matrix

---

## Class Labels

Maintain one canonical mapping between class names and numerical labels.

Do not define different mappings in different modules.

---

## Performance

Use GPU when available.

The project must remain usable in Google Colab and Kaggle environments.

Do not assume that a local NVIDIA GPU exists.

---

## Visualization

Generate:

- class distribution
- text length distribution
- training/validation loss
- training/validation accuracy
- confusion matrix
- class-level metrics where useful

---

## Model Saving

Save models in a reproducible format.

Do not commit large model artifacts to Git unless explicitly required.