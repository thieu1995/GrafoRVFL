#!/usr/bin/env python
# Created by "Thieu" at 15:45, 15/08/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from graforvfl import RvflRegressor


@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=200, n_features=10, noise=0.1, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def test_model_initialization():
    model = RvflRegressor(
        size_hidden=20,
        act_name='relu',
        weight_initializer='he_uniform',
        reg_alpha=0.1,
        seed=42
    )
    assert model.size_hidden == 20
    assert model.act_name == 'relu'
    assert model.weight_initializer == 'he_uniform'
    assert model.reg_alpha == 0.1
    assert model.seed == 42


def test_model_fit_and_predict(regression_data):
    X_train, X_test, y_train, y_test = regression_data
    model = RvflRegressor(seed=123)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == y_test.shape


def test_score_method(regression_data):
    X_train, X_test, y_train, y_test = regression_data
    model = RvflRegressor(seed=123)
    model.fit(X_train, y_train)
    r2 = model.score(X_test, y_test)
    assert isinstance(r2, float)
    assert -1 <= r2 <= 1  # R² score range


def test_scores_method(regression_data):
    X_train, X_test, y_train, y_test = regression_data
    model = RvflRegressor(seed=123)
    model.fit(X_train, y_train)
    scores = model.scores(X_test, y_test, list_metrics=["MSE", "MAE", "R2"])
    assert isinstance(scores, dict)
    assert "MSE" in scores
    assert "MAE" in scores
    assert "R2" in scores


def test_evaluate_method(regression_data):
    X_train, X_test, y_train, y_test = regression_data
    model = RvflRegressor(seed=123)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results = model.evaluate(y_test, y_pred, list_metrics=["RMSE", "MAPE"])
    assert isinstance(results, dict)
    assert "RMSE" in results
    assert "MAPE" in results
