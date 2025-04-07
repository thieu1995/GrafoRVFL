#!/usr/bin/env python
# Created by "Thieu" at 09:52, 07/04/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pytest
import numpy as np
from sklearn.datasets import load_breast_cancer
from graforvfl import Data
from graforvfl import GfoRvflTuner, IntegerVar, StringVar


@pytest.fixture(scope="module")
def data_and_bounds():
    # Load sample classification data
    X, y = load_breast_cancer(return_X_y=True)
    data = Data(X, y)
    data.split_train_test(test_size=0.2, random_state=42, inplace=True)

    # Define search space
    bounds = [
        IntegerVar(lb=5, ub=20, name="size_hidden"),
        StringVar(valid_sets=["relu", "tanh"], name="act_name"),
        StringVar(valid_sets=["he_uniform", "glorot_uniform"], name="weight_initializer")
    ]
    return data, bounds


def test_initialization(data_and_bounds):
    _, bounds = data_and_bounds
    tuner = GfoRvflTuner(problem_type="classification",
                         bounds=bounds,
                         optim="BaseGA",
                         optim_param_grid={"epoch": [10], "pop_size": [20]},
                         scoring="AS",
                         cv=3,
                         search_type="random",
                         n_iter=2,
                         seed=42,
                         verbose=False)

    assert tuner.problem_type == "classification"
    assert tuner.scoring == "AS"
    assert tuner.search_type == "random"
    assert tuner.n_iter == 2
    assert tuner.best_score == -np.inf


def test_param_combinations_grid():
    tuner = GfoRvflTuner(problem_type="classification",
                         bounds=[],
                         optim="BaseGA",
                         optim_param_grid={"epoch": [10, 20], "pop_size": [30, 40]},
                         scoring="AS",
                         cv=3,
                         search_type="grid")

    combos = tuner._get_param_combinations()
    assert len(combos) == 4  # 2 * 2 combinations


def test_param_combinations_random():
    tuner = GfoRvflTuner(problem_type="classification",
                         bounds=[],
                         optim="BaseGA",
                         optim_param_grid={"epoch": [10, 20], "pop_size": [30, 40]},
                         scoring="AS",
                         cv=3,
                         search_type="random",
                         n_iter=3,
                         seed=1)

    combos = tuner._get_param_combinations()
    assert len(combos) == 3
    for combo in combos:
        assert combo["epoch"] in [10, 20]
        assert combo["pop_size"] in [30, 40]


def test_fit_classification(data_and_bounds):
    data, bounds = data_and_bounds
    tuner = GfoRvflTuner(problem_type="classification",
                         bounds=bounds,
                         optim="BaseGA",
                         optim_param_grid={"epoch": [5], "pop_size": [10]},
                         scoring="AS",
                         cv=2,
                         search_type="grid",
                         seed=1,
                         verbose=False)

    tuner.fit(data.X_train, data.y_train)

    assert tuner.best_optim_params is not None
    assert tuner.best_searcher is not None
    assert tuner.best_score > 0


def test_predict_after_fit(data_and_bounds):
    data, bounds = data_and_bounds
    tuner = GfoRvflTuner(problem_type="classification",
                         bounds=bounds,
                         optim="BaseGA",
                         optim_param_grid={"epoch": [5], "pop_size": [10]},
                         scoring="AS",
                         cv=2,
                         search_type="grid",
                         seed=1,
                         verbose=False)

    tuner.fit(data.X_train, data.y_train)
    preds = tuner.predict(data.X_test)
    assert len(preds) == len(data.y_test)


def test_score_after_fit(data_and_bounds):
    data, bounds = data_and_bounds
    tuner = GfoRvflTuner(problem_type="classification",
                         bounds=bounds,
                         optim="BaseGA",
                         optim_param_grid={"epoch": [5], "pop_size": [10]},
                         scoring="AS",
                         cv=2,
                         search_type="grid",
                         seed=1,
                         verbose=False)

    tuner.fit(data.X_train, data.y_train)
    score = tuner.score(data.X_test, data.y_test)
    assert 0 <= score <= 1
