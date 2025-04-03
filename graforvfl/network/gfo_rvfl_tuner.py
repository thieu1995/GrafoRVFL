#!/usr/bin/env python
# Created by "Thieu" at 22:28, 02/04/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import itertools
import numpy as np
import random
from graforvfl.shared.scorer import get_all_classification_metrics, get_all_regression_metrics
from sklearn.model_selection import train_test_split
from graforvfl.network.gfo_rvfl_cv import GfoRvflCV


class GfoRvflTuner:
    """
    Hyperparameter tuner for the metaheuristic algorithm used in GfoRvflCV.

    Parameters
    ----------
    base_model : GfoRvflCV instance
        A base instance of GfoRvflCV with fixed RVFL hyperparameters.

    optim_param_grid : dict
        Dictionary of hyperparameter ranges for the metaheuristic algorithm.

    search_type : str, default="random"
        - "grid" for exhaustive grid search.
        - "random" for randomized search.

    n_iter : int, default=10
        Number of random search iterations (only used when search_type="random").

    cv : int, default=3
        Number of cross-validation folds.

    scoring : str, default="accuracy"
        The evaluation metric used to compare different optimization settings.

    random_state : int, default=None
        Random seed for reproducibility.

    verbose : bool, default=True
        Whether to print progress.

    Attributes
    ----------
    best_optim_params : dict
        The best found hyperparameters for the metaheuristic optimizer.

    best_score : float
        The best evaluation score.

    best_searcher : GfoRvflCV
        The best trained model using the optimized metaheuristic parameters.

    """

    SUPPORTED_CLS_METRICS = get_all_classification_metrics()
    SUPPORTED_REG_METRICS = get_all_regression_metrics()

    def __init__(self, problem_type="regression", bounds=None,
                 optim="OriginalWOA", optim_param_grid=None,
                 scoring="MSE", cv=None,
                 search_type="random", n_iter=10, seed=None, verbose=True,  **kwargs):
        self.problem_type = problem_type
        self.bounds = bounds
        self.optim = optim
        self.optim_param_grid = optim_param_grid
        self.scoring = scoring
        self.cv = cv
        self.search_type = search_type
        self.n_iter = n_iter
        self.seed = seed
        self.verbose = verbose
        self.generator = np.random.default_rng(seed)

        self.best_optim_params = None
        self.best_searcher = None

        if problem_type == "regression":
            self.minmax = self.SUPPORTED_REG_METRICS[scoring]
        else:
            self.minmax = self.SUPPORTED_CLS_METRICS[scoring]

        if self.minmax == "min":
            self.best_score = np.inf
        else:
            self.best_score = -np.inf

    def _get_param_combinations(self):
        """Generate parameter combinations based on search type."""
        param_keys = list(self.optim_param_grid.keys())
        param_values = list(self.optim_param_grid.values())

        if self.search_type == "grid":
            return [dict(zip(param_keys, values)) for values in itertools.product(*param_values)]
        elif self.search_type == "random":
            random.seed(self.seed)
            return [
                {k: random.choice(v) for k, v in self.optim_param_grid.items()}
                for _ in range(self.n_iter)
            ]
        else:
            raise ValueError("search_type must be 'grid' or 'random'.")

    def fit(self, X, y):
        """Optimize the metaheuristic parameters for GfoRvflCV."""
        param_combinations = self._get_param_combinations()

        for idx, optim_params in enumerate(param_combinations):
            if self.verbose:
                print(f"Testing {idx+1}/{len(param_combinations)}: {optim_params}")

            # Clone base model and update optimization parameters
            model = GfoRvflCV(problem_type=self.problem_type, bounds=self.bounds,
                              optim=self.optim, optim_params=optim_params,
                              scoring=self.scoring, cv=self.cv, seed=self.seed, verbose=self.verbose)

            # Perform cross-validation
            scores = []
            for _ in range(self.cv):
                X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                                  test_size=1.0/self.cv, random_state=self.seed)
                model.fit(X_train, y_train)
                score = model.best_estimator.score(X_val, y_val)
                scores.append(score)
            avg_score = np.mean(scores)

            # Update best parameters
            if ((self.minmax == "max" and avg_score > self.best_score) or (self.minmax == "min" and avg_score < self.best_score)):
                self.best_score = avg_score
                self.best_optim_params = optim_params
                self.best_searcher = model

        if self.verbose:
            print(f"Best optimizer parameters: {self.best_optim_params}")
            print(f"Best score: {self.best_score}")

        return self

    def predict(self, X):
        """Predict using the best found estimator."""
        if self.best_searcher is None:
            raise ValueError("Tuner has not been fitted yet. Call fit() first.")
        return self.best_searcher.predict(X)

    def score(self, X, y):
        """Evaluate the best model on given data."""
        if self.best_searcher is None:
            raise ValueError("Tuner has not been fitted yet. Call fit() first.")
        return self.best_searcher.score(X, y)
