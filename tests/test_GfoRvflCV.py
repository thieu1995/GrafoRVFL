#!/usr/bin/env python
# Created by "Thieu" at 15:45, 15/08/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pytest
import numpy as np
from sklearn.datasets import load_breast_cancer, make_regression
from graforvfl import GfoRvflCV, Data, IntegerVar, StringVar


@pytest.fixture
def classification_data():
    X, y = load_breast_cancer(return_X_y=True)
    data = Data(X, y)
    data.split_train_test(test_size=0.2, random_state=42, inplace=True)
    data.X_train, scaler_X = data.scale(data.X_train)
    data.X_test = scaler_X.transform(data.X_test)
    data.y_train, scaler_y = data.encode_label(data.y_train)
    data.y_test = scaler_y.transform(data.y_test)
    return data


@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=200, n_features=20, noise=0.1, random_state=42)
    data = Data(X, y)
    data.split_train_test(test_size=0.2, random_state=42, inplace=True)
    data.X_train, scaler_X = data.scale(data.X_train)
    data.X_test = scaler_X.transform(data.X_test)
    data.y_train, scaler_y = data.scale(data.y_train)
    data.y_test = scaler_y.transform(data.y_test.reshape(-1, 1))
    return data


@pytest.fixture
def common_bounds():
    return [
        IntegerVar(lb=5, ub=100, name="size_hidden"),
        StringVar(valid_sets=["relu", "sigmoid"], name="act_name"),
        StringVar(valid_sets=["glorot_uniform", "he_uniform"], name="weight_initializer"),
    ]


@pytest.mark.parametrize("problem_type, scoring", [
    ("classification", "AS"),
    ("regression", "MSE")
])
def test_initialization(problem_type, scoring, common_bounds):
    model = GfoRvflCV(problem_type=problem_type, scoring=scoring, bounds=common_bounds)
    assert model.problem_type == problem_type
    assert model.scoring == scoring
    assert model.bounds is not None
    assert model.optim is not None


def test_fit_and_predict_classification(classification_data, common_bounds):
    opt_params = {"name": "WOA", "epoch": 3, "pop_size": 5}
    model = GfoRvflCV(problem_type="classification", scoring="AS", bounds=common_bounds,
                      optim="OriginalWOA", optim_params=opt_params, cv=3, seed=42)
    model.fit(classification_data.X_train, classification_data.y_train)
    preds = model.predict(classification_data.X_test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == classification_data.X_test.shape[0]


def test_fit_and_predict_regression(regression_data, common_bounds):
    opt_params = {"name": "WOA", "epoch": 3, "pop_size": 5}
    model = GfoRvflCV(problem_type="regression", scoring="MSE", bounds=common_bounds,
                      optim="OriginalWOA", optim_params=opt_params, cv=3, seed=42)
    model.fit(regression_data.X_train, regression_data.y_train)
    preds = model.predict(regression_data.X_test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == regression_data.X_test.shape[0]


def test_predict_before_fit_raises(common_bounds):
    model = GfoRvflCV(problem_type="classification", scoring="AS", bounds=common_bounds)
    with pytest.raises(ValueError, match="Model is not trained"):
        model.predict(np.random.rand(10, 5))


def test_score_and_scores_methods(classification_data, common_bounds):
    opt_params = {"name": "WOA", "epoch": 3, "pop_size": 5}
    model = GfoRvflCV(problem_type="classification", scoring="AS", bounds=common_bounds,
                      optim="OriginalWOA", optim_params=opt_params, cv=3, seed=42)
    model.fit(classification_data.X_train, classification_data.y_train)
    score = model.score(classification_data.X_test, classification_data.y_test)
    scores = model.scores(classification_data.X_test, classification_data.y_test)
    assert isinstance(score, float)
    assert isinstance(scores, dict)
    assert "AS" in scores or "RS" in scores


def test_evaluate_method_output(classification_data, common_bounds):
    opt_params = {"name": "WOA", "epoch": 3, "pop_size": 5}
    model = GfoRvflCV(problem_type="classification", scoring="AS", bounds=common_bounds,
                      optim="OriginalWOA", optim_params=opt_params, cv=3, seed=42)
    model.fit(classification_data.X_train, classification_data.y_train)
    preds = model.predict(classification_data.X_test)
    evals = model.evaluate(classification_data.y_test, preds, list_metrics=["AS", "F1S"])
    assert isinstance(evals, dict)
    assert "AS" in evals and "F1S" in evals
