# Results

This document records experimentally verified results.

## Model Benchmark (Lab 05 & Lab 06 Adaptation)

Implemented and configured in `notebooks/project_progress.ipynb`:

| Model Architecture | Input Representation | Regularization / Callbacks | Parameters | Val Loss | Val Acc | Val Macro F1 | Test Acc | Test Macro F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense Baseline | TF-IDF (5,000 features) | None (256 ReLU units) | 1,281,798 | *Pending* | *Pending* | *Pending* | — | — |
| Dense L1 | TF-IDF (5,000 features) | L1 ($\lambda = 3 \times 10^{-4}$) | 1,281,798 | *Pending* | *Pending* | *Pending* | — | — |
| Dense L2 | TF-IDF (5,000 features) | L2 ($\lambda = 3 \times 10^{-4}$) | 1,281,798 | *Pending* | *Pending* | *Pending* | — | — |
| Dense Dropout | TF-IDF (5,000 features) | Dropout ($p = 0.50$) | 1,281,798 | *Pending* | *Pending* | *Pending* | — | — |
| Basic LSTM | Sequences (dim=50, scratch) | Dropout ($p = 0.10$) | 561,156 | *Pending* | *Pending* | *Pending* | — | — |
| Improved Stacked LSTM | Sequences (dim=64, scratch) | Stacked 64->32 + Dropout + EarlyStopping | 679,270 | *Pending* | *Pending* | *Pending* | — | — |

*Note: Execution results will populate upon running `notebooks/project_progress.ipynb` in Google Colab.*

## Primary Project Recurrent Architectures

| Architecture | Status |
| --- | --- |
| Simple RNN | Planned |
| GRU | Planned |
| Bi-LSTM (Primary Model) | Planned |
