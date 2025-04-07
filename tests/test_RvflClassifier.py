#!/usr/bin/env python
# Created by "Thieu" at 11:27, 17/08/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from graforvfl import RvflClassifier


@pytest.fixture
def sample_data():
    X, y = make_classification(n_samples=200, n_features=20, n_informative=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def test_fit_predict_shape(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    model = RvflClassifier(size_hidden=10, act_name='sigmoid', weight_initializer="random_normal", reg_alpha=0.1,
                           seed=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    assert pred.shape == y_test.shape
    assert np.all(np.isin(pred, np.unique(y_train)))


def test_predict_proba_shape(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    model = RvflClassifier(size_hidden=10, act_name='sigmoid', weight_initializer="random_normal", reg_alpha=0.1,
                           seed=42)
    model.fit(X_train, y_train)
    probas = model.predict_proba(X_test)
    assert probas.shape == (X_test.shape[0], len(np.unique(y_train)))


def test_score_accuracy(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    model = RvflClassifier(size_hidden=10, act_name='sigmoid', weight_initializer="random_normal", reg_alpha=0.1,
                           seed=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    assert 0 <= score <= 1


def test_scores_dict(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    model = RvflClassifier(size_hidden=10, act_name='sigmoid', weight_initializer="random_normal", reg_alpha=0.1,
                           seed=42)
    model.fit(X_train, y_train)
    scores = model.scores(X_test, y_test, list_metrics=["AS", "RS", "F1S"])
    assert isinstance(scores, dict)
    assert all(metric in scores for metric in ["AS", "RS", "F1S"])


def test_evaluate_output(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    model = RvflClassifier(size_hidden=10, act_name='sigmoid', weight_initializer="random_normal", reg_alpha=0.1,
                           seed=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results = model.evaluate(y_test, y_pred, list_metrics=["AS", "RS", "F1S"])
    assert isinstance(results, dict)
    assert all(metric in results for metric in ["AS", "RS", "F1S"])
