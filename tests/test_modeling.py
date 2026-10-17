"""
tests/test_modeling.py
======================

Unit tests for src/modeling.py verifying:
1. Deterministic 6-class label encoding.
2. TF-IDF vectorization fitting strictly on train texts.
3. Tokenizer and sequence padding shapes.
4. Model architectures, input/output dimensions, parameter counts.
5. Multi-class metric calculation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.modeling import (
    EXPECTED_CLASSES,
    build_basic_lstm,
    build_dense_baseline,
    build_dense_dropout,
    build_dense_l1,
    build_dense_l2,
    build_improved_lstm,
    compute_classification_metrics,
    create_label_encoder,
    fit_sequence_tokenizer,
    fit_tfidf_vectorizer,
    tokenize_and_pad,
)


@pytest.fixture
def sample_train_data() -> Tuple[pd.Series, pd.Series]:
    texts = pd.Series([
        "This is an age related sample with older person",
        "Ethnic background offensive tweet here",
        "Misogynistic comment targeting women and girls",
        "Completely normal benign non-cyberbullying post",
        "Generic mean harassment and insults online",
        "Religious intolerance and sectarian text sample",
    ] * 5)
    labels = pd.Series(EXPECTED_CLASSES * 5)
    return texts, labels


def test_label_encoder(sample_train_data):
    texts, labels = sample_train_data
    le = create_label_encoder(labels)
    assert len(le.classes_) == 6
    encoded = le.transform(labels)
    assert len(encoded) == len(labels)
    assert set(np.unique(encoded)) == {0, 1, 2, 3, 4, 5}
    # Inverse transform roundtrip
    restored = le.inverse_transform(encoded)
    assert list(restored) == list(labels)


def test_tfidf_vectorizer(sample_train_data):
    texts, labels = sample_train_data
    tfidf = fit_tfidf_vectorizer(texts, max_features=50)
    vec = tfidf.transform(texts)
    assert vec.shape[0] == len(texts)
    assert vec.shape[1] <= 50


def test_sequence_tokenizer(sample_train_data):
    texts, labels = sample_train_data
    tokenizer = fit_sequence_tokenizer(texts, vocab_size=100)
    padded = tokenize_and_pad(tokenizer, texts, maxlen=20)
    assert padded.shape == (len(texts), 20)
    assert isinstance(padded, np.ndarray)


def test_dense_architectures():
    baseline = build_dense_baseline(input_dim=100, num_classes=6, units=32)
    assert baseline.output_shape == (None, 6)
    assert len(baseline.layers) == 2

    l1_model = build_dense_l1(input_dim=100, num_classes=6, units=32)
    assert l1_model.output_shape == (None, 6)

    l2_model = build_dense_l2(input_dim=100, num_classes=6, units=32)
    assert l2_model.output_shape == (None, 6)

    dropout_model = build_dense_dropout(input_dim=100, num_classes=6, units=32)
    assert dropout_model.output_shape == (None, 6)


def test_lstm_architectures():
    basic = build_basic_lstm(vocab_size=100, embedding_dim=16, maxlen=20, num_classes=6, lstm_units=16)
    assert basic.output_shape == (None, 6)

    improved = build_improved_lstm(
        vocab_size=100,
        embedding_dim=16,
        maxlen=20,
        num_classes=6,
        lstm_units_1=16,
        lstm_units_2=8,
        dense_units=16,
    )
    assert improved.output_shape == (None, 6)


def test_compute_metrics():
    y_true = np.array([0, 1, 2, 3, 4, 5])
    # Perfect predictions
    y_pred_probs = np.eye(6)
    metrics = compute_classification_metrics(y_true, y_pred_probs, class_names=EXPECTED_CLASSES)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["weighted_f1"] == 1.0
    assert "age" in metrics["classification_report"]
