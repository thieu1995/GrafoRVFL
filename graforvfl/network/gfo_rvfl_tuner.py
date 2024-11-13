#!/usr/bin/env python
# Created by "Thieu" at 10:04, 04/12/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from mealpy import Problem, get_optimizer_by_name, Optimizer
from permetrics import ClassificationMetric, RegressionMetric
from graforvfl.shared.scorer import get_all_classification_metrics, get_all_regression_metrics
from graforvfl.shared import boundary_controller
from graforvfl import RvflRegressor, RvflClassifier


class HyperparameterProblem(Problem):
    """
    This class defines the Hyper-parameter tuning problem that will be used for Mealpy library.

    Parameters
    ----------
    bounds : from Mealpy library.

    minmax : from Mealpy library.

    X : array-like of shape (n_samples, n_features)
        Test samples. For some estimators this may be a precomputed kernel matrix or a list of generic objects instead with shape
        ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted`` is the number of samples used in the fitting for the estimator.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        True values for `X`.

    model_class : RvflRegressor or RvflClassifier
        The class definition of RVFL network for regression or classification problem.

    metric_class : RegressionMetric or ClassificationMetric
        The class definition of Performance Metrics for regression or classification problem.

    obj_name : str
        The name of the loss function used in network

    cv : int, default=5
        The k fold cross-validation method

    seed: int, default=None
        Determines random number generation for weights and bias initialization.
        Pass an int for reproducible results across multiple function calls.
    """

    def __init__(self, bounds=None, minmax="max", X=None, y=None, model_class=None, metric_class=None, obj_name=None, cv=5, seed=None, **kwargs):
        self.model_class = model_class
        self.model = None
        self.X = X
        self.y = y
        self.metric_class = metric_class
        self.obj_name = obj_name
        self.cv = cv
        self.kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
        super().__init__(bounds, minmax, **{**kwargs, "seed":seed})

    def obj_func(self, x):
        x_decoded = self.decode_solution(x)
        self.model = self.model_class(**x_decoded, seed=self.seed)
        scores = []
        # Perform custom cross-validation
        for train_idx, test_idx in self.kf.split(self.X):
            # Split the data into training and test sets
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            # Train the model on the training set
            self.model.fit(X_train, y_train)
            # Make predictions on the test set
            y_pred = self.model.predict(X_test)
            # Calculate accuracy for the current fold
            mt = self.metric_class(y_test, y_pred)
            score = mt.get_metric_by_name(self.obj_name)[self.obj_name]
            # Accumulate accuracy across folds
            scores.append(score)
        return np.mean(scores)


class GfoRvflTuner:
    """
    Defines the Gradient Free Optimization-based Random Vector Functional Link Network.

    Parameters
    ----------
    problem_type : str, default="regression"
        The problem type

    bounds : from Mealpy library, default=None
        The boundary for RVFL hyper-parameters. It can be an instance of these classes:
        [FloatVar, BoolVar, StringVar, IntegerVar, PermutationVar, BinaryVar, MixedSetVar]

    cv : int, default=5
        The k fold cross-validation method.

    scoring : str
        The name of objective for the problem, also depend on the problem is classification and regression.

    optimizer : str or instance of Optimizer class (from Mealpy library), default = "BaseGA"
        The Metaheuristic Algorithm that use to solve the feature selection problem.
        Current supported list, please check it here: https://github.com/thieu1995/mealpy.
        If a custom optimizer is passed, make sure it is an instance of `Optimizer` class.

    optimizer_paras : None or dict of parameter, default=None
        The parameter for the `optimizer` object.
        If `None`, the default parameters of optimizer is used (defined in https://github.com/thieu1995/mealpy.)
        If `dict` is passed, make sure it has at least `epoch` and `pop_size` parameters.

    verbose : bool, default=False
        Whether to print progress messages to stdout.

    seed: int, default=None
        Determines random number generation for weights and bias initialization.
        Pass an int for reproducible results across multiple function calls.

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from mealpy import StringVar, IntegerVar
    >>> from graforvfl import Data, GfoRvflTuner

    >>> ## Load data object
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> data = Data(X, y)

    >>> ## Split train and test
    >>> data.split_train_test(test_size=0.2, random_state=2, inplace=True)
    >>> print(data.X_train.shape, data.X_test.shape)

    >>> ## Scaling dataset
    >>> data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=("standard", "minmax"))
    >>> data.X_test = scaler_X.transform(data.X_test)

    >>> data.y_train, scaler_y = data.encode_label(data.y_train)
    >>> data.y_test = scaler_y.transform(data.y_test)

    >>> # Design the boundary (parameters)
    >>> my_bounds = [
    >>>     IntegerVar(lb=2, ub=1000, name="size_hidden"),
    >>>     StringVar(valid_sets=("none", "relu", "leaky_relu", "celu", "prelu", "gelu",
    >>>         "elu", "selu", "rrelu", "tanh", "sigmoid"), name="act_name"),
    >>>     StringVar(valid_sets=("orthogonal", "he_uniform", "he_normal", "glorot_uniform", "glorot_normal",
    >>>         "lecun_uniform", "lecun_normal", "random_uniform", "random_normal"), name="weight_initializer")
    >>> ]

    >>> opt_paras = {"name": "WOA", "epoch": 10, "pop_size": 20}
    >>> model = GfoRvflTuner(problem_type="classification", bounds=my_bounds, cv=3, scoring="AS",
    >>>                   optimizer="OriginalWOA", optimizer_paras=opt_paras, verbose=True, seed=42)
    >>> model.fit(data.X_train, data.y_train)
    >>> print(model.best_params)
    >>> print(model.best_estimator)
    >>> print(model.best_estimator.scores(data.X_test, data.y_test, list_metrics=("PS", "RS", "NPV", "F1S", "F2S")))
    """

    SUPPORTED_CLS_METRICS = get_all_classification_metrics()
    SUPPORTED_REG_METRICS = get_all_regression_metrics()

    def __init__(self, problem_type="regression", bounds=None, cv=5, scoring="MSE",
                 optimizer="OriginalWOA", optimizer_paras=None, verbose=True, seed=None):
        if problem_type == "regression":
            self.network_class = RvflRegressor
            self.scoring = boundary_controller.check_str("scoring", scoring, self.SUPPORTED_REG_METRICS)
            self.minmax = self.SUPPORTED_REG_METRICS[self.scoring]
            self.metric_class = RegressionMetric
        else:
            self.network_class = RvflClassifier
            self.scoring = boundary_controller.check_str("scoring", scoring, self.SUPPORTED_CLS_METRICS)
            self.minmax = self.SUPPORTED_CLS_METRICS[self.scoring]
            self.metric_class = ClassificationMetric
        self.seed = seed
        self.problem_type = problem_type
        self.bounds = bounds
        self.cv = cv
        self.verbose = "console" if verbose else "None"
        self.optimizer_paras = optimizer_paras
        self.optimizer = self._set_optimizer(optimizer, optimizer_paras)
        self.best_params = None
        self.best_estimator = None
        self.loss_train = None

    def _set_optimizer(self, optimizer=None, optimizer_paras=None):
        if type(optimizer) is str:
            opt_class = get_optimizer_by_name(optimizer)
            if type(optimizer_paras) is dict:
                return opt_class(**optimizer_paras)
            else:
                return opt_class(epoch=250, pop_size=20)
        elif isinstance(optimizer, Optimizer):
            if type(optimizer_paras) is dict:
                return optimizer.set_parameters(optimizer_paras)
            return optimizer
        else:
            raise TypeError(f"optimizer needs to set as a string and supported by Mealpy library.")

    def fit(self, X, y):
        self.problem = HyperparameterProblem(self.bounds, self.minmax, X, y, self.network_class, self.metric_class,
                                             obj_name=self.scoring, cv=self.cv, log_to=self.verbose, seed=self.seed)
        self.optimizer.solve(self.problem, seed=self.seed)
        self.best_params = self.optimizer.problem.decode_solution(self.optimizer.g_best.solution)
        self.best_estimator = self.network_class(**self.best_params, seed=self.seed)
        self.best_estimator.fit(X, y)
        self.loss_train = self.optimizer.history.list_global_best_fit
        return self

    def predict(self, X):
        if self.best_params is None or self.best_estimator is None:
            raise ValueError(f"Model is not trained, please call the fit() function.")
        return self.best_estimator.predict(X)

    def save_convergence(self, save_path="history", filename="convergence.csv"):
        """
        Save the convergence (fitness value) during the training process to csv file.

        Parameters
        ----------
        save_path : saved path (relative path, consider from current executed script path)
        filename : name of the file, needs to have ".csv" extension
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        if self.loss_train is None:
            print(f"{self.__class__.__name__} network doesn't have training loss!")
        else:
            data = {"epoch": list(range(1, len(self.loss_train) + 1)), "loss": self.loss_train}
            pd.DataFrame(data).to_csv(f"{save_path}/{filename}", index=False)

    def save_performance_metrics(self, y_true, y_pred, list_metrics=("RMSE", "MAE"), save_path="history", filename="metrics.csv"):
        """
        Save evaluation metrics to csv file

        Parameters
        ----------
        y_true : ground truth data
        y_pred : predicted output
        list_metrics : list of evaluation metrics
        save_path : saved path (relative path, consider from current executed script path)
        filename : name of the file, needs to have ".csv" extension
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        results = self.best_estimator.evaluate(y_true, y_pred, list_metrics)
        df = pd.DataFrame.from_dict(results, orient='index').T
        df.to_csv(f"{save_path}/{filename}", index=False)

    def save_y_predicted(self, X, y_true, save_path="history", filename="y_predicted.csv"):
        """
        Save the predicted results to csv file

        Parameters
        ----------
        X : The features data, nd.ndarray
        y_true : The ground truth data
        save_path : saved path (relative path, consider from current executed script path)
        filename : name of the file, needs to have ".csv" extension
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        y_pred = self.predict(X)
        data = {"y_true": np.squeeze(np.asarray(y_true)), "y_pred": np.squeeze(np.asarray(y_pred))}
        pd.DataFrame(data).to_csv(f"{save_path}/{filename}", index=False)

    def save_model(self, save_path="history", filename="network.pkl"):
        """
        Save network to pickle file

        Parameters
        ----------
        save_path : saved path (relative path, consider from current executed script path)
        filename : name of the file, needs to have ".pkl" extension
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        if filename[-4:] != ".pkl":
            filename += ".pkl"
        pickle.dump(self, open(f"{save_path}/{filename}", 'wb'))

    @staticmethod
    def load_model(load_path="history", filename="network.pkl"):
        """
        Load a saved model from a pickle file.

        Parameters
        ----------
        load_path : str, default="history"
            Directory containing the saved file.
        filename : str, default="network.pkl"
            Name of the file (must end with `.pkl`).

        Returns
        -------
        model : BaseRVFL
            Loaded model instance.
        """
        if filename[-4:] != ".pkl":
            filename += ".pkl"
        return pickle.load(open(f"{load_path}/{filename}", 'rb'))
