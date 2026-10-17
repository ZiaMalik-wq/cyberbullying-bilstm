"""
src/modeling.py
===============

Reusable modeling, vectorization, and evaluation utilities for the
Cyberbullying Classification project (Lab Assignment 02 Adaptation).

Contains model builders, tokenizers, vectorizers, and metric computers for:
1. TF-IDF vectorization and 6-class label encoding.
2. Dense feedforward networks (Baseline, L1, L2, Dropout).
3. Trainable Embedding + LSTM networks (Basic LSTM, Improved Stacked LSTM).
4. Multi-class evaluation (Accuracy, Macro F1, Weighted F1, Per-Class Report).

Constraints:
- No pretrained models or pretrained embeddings (all trained from scratch).
- All vocabularies and encoders fitted strictly on training data.
- Six target classes: age, ethnicity, gender, not_cyberbullying, other_cyberbullying, religion.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Embedding, Input, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.regularizers import l1, l2

logger = logging.getLogger(__name__)

EXPECTED_CLASSES: List[str] = [
    "age",
    "ethnicity",
    "gender",
    "not_cyberbullying",
    "other_cyberbullying",
    "religion",
]


# ---------------------------------------------------------------------------
# Label Encoding & Text Vectorization
# ---------------------------------------------------------------------------

def create_label_encoder(train_labels: pd.Series) -> LabelEncoder:
    """Fit a LabelEncoder on training labels.

    Ensures that class indexing is deterministic and consistent.
    """
    le = LabelEncoder()
    le.fit(train_labels)
    # Check that all 6 expected classes are known
    for c in EXPECTED_CLASSES:
        if c not in le.classes_:
            raise ValueError(f"Expected class '{c}' not found in training labels.")
    return le


def fit_tfidf_vectorizer(
    train_texts: pd.Series,
    max_features: int = 5000,
    stop_words: str = "english",
) -> TfidfVectorizer:
    """Fit a TF-IDF vectorizer strictly on training texts.

    Parameters
    ----------
    train_texts : pd.Series
        Training text series.
    max_features : int
        Maximum vocabulary features (matches PDF Lab 05 = 5000).
    stop_words : str
        Stopword removal setting.

    Returns
    -------
    TfidfVectorizer
        Fitted TF-IDF vectorizer.
    """
    tfidf = TfidfVectorizer(max_features=max_features, stop_words=stop_words)
    tfidf.fit(train_texts.astype(str))
    return tfidf


def fit_sequence_tokenizer(
    train_texts: pd.Series,
    vocab_size: int = 10000,
    oov_token: str = "<OOV>",
) -> Tokenizer:
    """Fit a Keras Tokenizer strictly on training texts.

    Parameters
    ----------
    train_texts : pd.Series
        Training text series.
    vocab_size : int
        Maximum vocabulary size (matches PDF Lab 06 = 10000).
    oov_token : str
        Out-of-vocabulary marker.

    Returns
    -------
    Tokenizer
        Fitted Keras Tokenizer.
    """
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(train_texts.astype(str))
    return tokenizer


def tokenize_and_pad(
    tokenizer: Tokenizer,
    texts: pd.Series,
    maxlen: int = 100,
    padding: str = "post",
    truncating: str = "post",
) -> np.ndarray:
    """Convert texts to padded integer sequences.

    Parameters
    ----------
    tokenizer : Tokenizer
        Fitted Tokenizer.
    texts : pd.Series
        Text series to convert.
    maxlen : int
        Maximum sequence length.
    padding : str
        Padding position ('post' matches PDF).
    truncating : str
        Truncating position ('post' matches PDF).

    Returns
    -------
    np.ndarray
        Padded integer token array of shape (N, maxlen).
    """
    seqs = tokenizer.texts_to_sequences(texts.astype(str))
    # Safeguard: map any empty sequence (e.g. pure punctuation like ':-)') to OOV token
    oov_idx = tokenizer.word_index.get("<OOV>", 1)
    seqs = [s if len(s) > 0 else [oov_idx] for s in seqs]
    padded = pad_sequences(
        seqs,
        maxlen=maxlen,
        padding=padding,
        truncating=truncating,
    )
    return padded


# ---------------------------------------------------------------------------
# Dense Model Architectures (Lab 05 Adaptation)
# ---------------------------------------------------------------------------

def build_dense_baseline(
    input_dim: int = 5000,
    num_classes: int = 6,
    units: int = 256,
    learning_rate: float = 0.001,
) -> Sequential:
    """Build simple Dense baseline network (PDF Page 4)."""
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(units, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ], name="Dense_Baseline")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_dense_l1(
    input_dim: int = 5000,
    num_classes: int = 6,
    units: int = 256,
    l1_val: float = 0.0003,
    learning_rate: float = 0.001,
) -> Sequential:
    """Build Dense network with L1 kernel regularization (PDF Page 4)."""
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(units, activation="relu", kernel_regularizer=l1(l1_val)),
        Dense(num_classes, activation="softmax"),
    ], name="Dense_L1_Regularized")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_dense_l2(
    input_dim: int = 5000,
    num_classes: int = 6,
    units: int = 256,
    l2_val: float = 0.0003,
    learning_rate: float = 0.001,
) -> Sequential:
    """Build Dense network with L2 kernel regularization (PDF Page 5)."""
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(units, activation="relu", kernel_regularizer=l2(l2_val)),
        Dense(num_classes, activation="softmax"),
    ], name="Dense_L2_Regularized")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_dense_dropout(
    input_dim: int = 5000,
    num_classes: int = 6,
    units: int = 256,
    dropout_rate: float = 0.5,
    learning_rate: float = 0.001,
) -> Sequential:
    """Build Dense network with Dropout regularization (PDF Page 6)."""
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(units, activation="relu"),
        Dropout(dropout_rate),
        Dense(num_classes, activation="softmax"),
    ], name="Dense_Dropout")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# LSTM Model Architectures (Lab 06 Adaptation)
# ---------------------------------------------------------------------------

def build_basic_lstm(
    vocab_size: int = 10000,
    embedding_dim: int = 50,
    maxlen: int = 100,
    num_classes: int = 6,
    lstm_units: int = 100,
    dropout: float = 0.1,
    recurrent_dropout: float = 0.0,
    learning_rate: float = 0.001,
) -> Sequential:
    """Build Basic LSTM model with trainable Embedding layer (PDF Page 8).

    Note: Embedding is initialized randomly and trained strictly from scratch.
    """
    model = Sequential([
        Input(shape=(maxlen,)),
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            mask_zero=False,
        ),
        Dropout(dropout),
        LSTM(lstm_units, dropout=dropout, recurrent_dropout=recurrent_dropout),
        Dense(num_classes, activation="softmax"),
    ], name="Basic_LSTM")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_improved_lstm(
    vocab_size: int = 10000,
    embedding_dim: int = 64,
    maxlen: int = 100,
    num_classes: int = 6,
    lstm_units_1: int = 64,
    lstm_units_2: int = 32,
    dense_units: int = 64,
    dropout_rate: float = 0.3,
    learning_rate: float = 0.002,
) -> Sequential:
    """Build Improved Stacked LSTM network with Dense layer (PDF Page 9).

    Incorporates:
    - Trainable Embedding layer (dim=64) from scratch
    - Stacked LSTM layers (64 -> 32)
    - Intermediate Dense projection (64)
    - Systematic Dropout (0.3)
    """
    model = Sequential([
        Input(shape=(maxlen,)),
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            mask_zero=False,
        ),
        Dropout(dropout_rate),
        LSTM(lstm_units_1, return_sequences=True, dropout=dropout_rate),
        Dropout(dropout_rate),
        LSTM(lstm_units_2, dropout=dropout_rate),
        Dropout(dropout_rate),
        Dense(dense_units, activation="relu"),
        Dropout(dropout_rate),
        Dense(num_classes, activation="softmax"),
    ], name="Improved_Stacked_LSTM")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Metric Evaluation Helpers
# ---------------------------------------------------------------------------

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute comprehensive classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth integer labels.
    y_pred_probs : np.ndarray
        Predicted probabilities of shape (N, num_classes).
    class_names : Optional[List[str]]
        Names corresponding to label indices.

    Returns
    -------
    dict
        Dictionary containing accuracy, macro_f1, weighted_f1, and classification report.
    """
    y_pred = np.argmax(y_pred_probs, axis=1)
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report_dict,
        "y_pred": y_pred,
    }
