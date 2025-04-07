#!/usr/bin/env python
# Created by "Thieu" at 09:57, 07/04/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from graforvfl import GfoRvflComparator, IntegerVar


# Dummy data
X_train = np.random.rand(20, 5)
y_train = np.random.rand(20)
X_test = np.random.rand(5, 5)
y_test = np.random.rand(5)


# Dummy optimizer class
class DummyOptimizer:
    def __init__(self, name="DummyOpt"):
        self.name = name


@pytest.fixture
def comparator():
    return GfoRvflComparator(
        problem_type="regression",
        bounds=[IntegerVar(lb=1, ub=10, name="size_hidden")],
        optim_list=[DummyOptimizer()],
        optim_params_list=[{"pop_size": 10, "epochs": 5}],
        scoring="MSE",
        cv=2,
        seed=42,
        verbose=False
    )


def test_init_valid_comparator(comparator):
    assert comparator.problem_type == "regression"
    assert comparator.scoring == "MSE"
    assert isinstance(comparator.optim_list, list)
    assert comparator.cv == 2


def test_init_invalid_lengths():
    with pytest.raises(ValueError):
        GfoRvflComparator(
            optim_list=[DummyOptimizer(), DummyOptimizer()],
            optim_params_list=[{"a": 1}],
        )


@patch("graforvfl.network.gfo_rvfl_comparator.GfoRvflCV")
def test_run_returns_expected_format(mock_model, comparator, tmp_path):
    # Mock fit and scores
    instance = MagicMock()
    instance.optim.name = "DummyOpt"
    instance.loss_train = list(np.random.rand(10))
    instance.best_estimator.scores.return_value = {
        "MSE": 0.1,
        "NSE": 0.9,
        "KGE": 0.8,
        "R": 0.7,
        "MAE": 0.05
    }
    instance.best_params = {"param": 42}
    mock_model.return_value = instance

    models, losses, metrics_df = comparator.run(
        X_train, y_train, X_test, y_test, n_trials=2,
        path_save=str(tmp_path),
        save_results=False,
        save_models=False
    )

    assert len(models) == 1  # One optimizer
    assert isinstance(metrics_df, pd.DataFrame)
    assert metrics_df.shape[0] == 2  # Two trials
    assert "optimizer" in metrics_df.columns
    assert isinstance(losses[0], dict)
    assert all("trial_" in key for key in losses[0].keys())


@patch("graforvfl.network.gfo_rvfl_comparator.GfoRvflCV")
def test_run_with_zero_trials(mock_model, comparator, tmp_path):
    # Ensure n_trials < 1 resets to 1
    instance = MagicMock()
    instance.optim.name = "DummyOpt"
    instance.loss_train = list(np.random.rand(10))
    instance.best_estimator.scores.return_value = {
        "MSE": 0.1,
        "NSE": 0.9,
        "KGE": 0.8,
        "R": 0.7,
        "MAE": 0.05
    }
    instance.best_params = {"param": 42}
    mock_model.return_value = instance

    models, losses, metrics_df = comparator.run(
        X_train, y_train, X_test, y_test, n_trials=0,
        path_save=str(tmp_path),
        save_results=False,
        save_models=False
    )

    assert len(models[0]) == 1  # Should default to 1 trial
