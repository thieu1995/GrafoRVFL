#!/usr/bin/env python
# Created by "Thieu" at 09:48, 17/08/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from permetrics import RegressionMetric, ClassificationMetric
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge
from graforvfl.shared import activator, boundary_controller, randomer
from graforvfl.shared.scorer import get_all_regression_metrics, get_all_classification_metrics


class BaseRVFL(BaseEstimator):
    """
    This class defines the general Random Vector Functional Link (RVFL) networks

    Parameters
    ----------
    size_hidden : int, default=10
        The number of hidden nodes

    act_name : str, default="sigmoid"
        The activation of the hidden layer. The supported values are:
        ["none", "relu", "leaky_relu", "celu", "prelu", "gelu", "elu", "selu", "rrelu", "tanh", "hard_tanh",
        "sigmoid", "hard_sigmoid", "log_sigmoid", "silu", "swish", "hard_swish", "soft_plus", "mish",
        "soft_sign", "tanh_shrink", "soft_shrink", "hard_shrink", "softmin", "softmax", "log_softmax" }

    weight_initializer : str, default="random_uniform"
        The weight initialization methods. The supported methods are:
        ["orthogonal", "he_uniform", "he_normal", "glorot_uniform", "glorot_normal",
        "lecun_uniform", "lecun_normal", "random_uniform", "random_normal"]
        For definition of these methods, please check it at: https://keras.io/api/layers/initializers/

    trainer : str, default = "OLS"
        The utilized method for training weights of hidden-output layer and weights of input-output layer.
            + MPI: Moore-Penrose inversion
            + OLS: Ordinary Least Squares (OLS) without regularization
            + L2: OLS regression with regularization

    alpha : float (Optional), default=0.5
        The penalty value for L2 method. Only effect when `trainer`="L2".
    """

    SUPPORTED_CLS_METRICS = get_all_classification_metrics()
    SUPPORTED_REG_METRICS = get_all_regression_metrics()
    CLS_OBJ_LOSSES = None
    SUPPORTED_WEIGHT_INITIALIZER = [
        "orthogonal", "he_uniform", "he_normal", "glorot_uniform", "glorot_normal",
        "lecun_uniform", "lecun_normal", "random_uniform", "random_normal"
    ]

    def __init__(self, size_hidden=10, act_name='sigmoid', weight_initializer="random_uniform", trainer="OLS", alpha=0.5):
        self.size_hidden = size_hidden
        self.act_name = act_name
        self.act_func = getattr(activator, self.act_name)
        self.weight_initializer, self.weight_randomer = self._get_weight_initializer(weight_initializer)
        self.trainer = trainer
        self.alpha = alpha
        self.weights = {}
        self.obj_scaler, self.loss_train = None, None
        self.n_labels, self.obj_scaler = None, None

    def _get_weight_initializer(self, name):
        if type(name) is str:
            wi = boundary_controller.check_str("weight_initializer", name, self.SUPPORTED_WEIGHT_INITIALIZER)
            wr = getattr(randomer, f"{wi}_initializer")
            return wi, wr
        else:
            raise ValueError(f"weight_initializer should be a string and belongs to {self.SUPPORTED_WEIGHT_INITIALIZER}")

    @staticmethod
    def _check_method(method=None, list_supported_methods=None):
        if type(method) is str:
            return boundary_controller.check_str("method", method, list_supported_methods)
        else:
            raise ValueError(f"method should be a string and belongs to {list_supported_methods}")

    def _trained(self, trainer="OLS", D=None, y=None):
        if trainer == "MPI":
            return np.linalg.pinv(D) @ y
        elif trainer == "L2":
            ridge_model = Ridge(alpha=self.alpha, fit_intercept=False)
            return ridge_model.fit(D, y).coef_.T
        else:
            ridge_model = Ridge(alpha=0.0, fit_intercept=False)
            return ridge_model.fit(D, y).coef_.T

    def fit(self, X, y):
        self.size_input = X.shape[1]
        if type(y) in (list, tuple, np.ndarray):
            y = np.squeeze(np.asarray(y))
            if y.ndim == 1:
                self.size_output = 1
            elif y.ndim == 2:
                self.size_output = y.shape[1]
            else:
                raise TypeError("Invalid y array shape, it should be 1D vector or 2D matrix.")
        else:
            raise TypeError("Invalid y array type, it should be list, tuple or np.ndarray")
        self.weights["Wh"] = self.weight_randomer((self.size_hidden, self.size_input))
        self.weights["bh"] = self.weight_randomer(self.size_hidden)
        H = self.act_func(X @ self.weights["Wh"].T + self.weights["bh"])
        D = np.concatenate((X, H), axis=1)
        self.weights["Wioho"] = self._trained(self.trainer, D, y)
        return self

    def predict(self, X):
        H = self.act_func(X @ self.weights["Wh"].T + self.weights["bh"])
        D = np.concatenate((X, H), axis=1)
        y_pred = D @ self.weights["Wioho"]
        return y_pred

    def predict_proba(self, X):
        H = self.act_func(X @ self.weights["Wh"].T + self.weights["bh"])
        D = np.concatenate((X, H), axis=1)
        y_pred = D @ self.weights["Wioho"]
        return y_pred

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

    def get_weights_size(self):
        return np.sum([item.size for item in self.weights.values()])

    def __evaluate_reg(self, y_true, y_pred, list_metrics=("MSE", "MAE")):
        rm = RegressionMetric(y_true=y_true, y_pred=y_pred, decimal=8)
        return rm.get_metrics_by_list_names(list_metrics)

    def __evaluate_cls(self, y_true, y_pred, list_metrics=("AS", "RS")):
        cm = ClassificationMetric(y_true, y_pred, decimal=8)
        return cm.get_metrics_by_list_names(list_metrics)

    def __score_reg(self, X, y, method="RMSE"):
        method = self._check_method(method, list(self.SUPPORTED_REG_METRICS.keys()))
        y_pred = self.predict(X)
        return RegressionMetric(y, y_pred, decimal=8).get_metric_by_name(method)[method]

    def __scores_reg(self, X, y, list_methods=("MSE", "MAE")):
        y_pred = self.predict(X)
        return self.__evaluate_reg(y_true=y, y_pred=y_pred, list_metrics=list_methods)

    def __score_cls(self, X, y, method="AS"):
        method = self._check_method(method, list(self.SUPPORTED_CLS_METRICS.keys()))
        return_prob = False
        if self.n_labels > 2:
            if method in self.CLS_OBJ_LOSSES:
                return_prob = True
        if return_prob:
            y_pred = self.predict_proba(X)
        else:
            y_pred = self.predict(X)
        cm = ClassificationMetric(y_true=y, y_pred=y_pred, decimal=8)
        return cm.get_metric_by_name(method)[method]

    def __scores_cls(self, X, y, list_methods=("AS", "RS")):
        list_errors = list(set(list_methods) & set(self.CLS_OBJ_LOSSES))
        list_scores = list((set(self.SUPPORTED_CLS_METRICS.keys()) - set(self.CLS_OBJ_LOSSES)) & set(list_methods))
        t1 = {}
        if len(list_errors) > 0:
            return_prob = False
            if self.n_labels > 2:
                return_prob = True
            if return_prob:
                y_pred = self.predict_proba(X)
            else:
                y_pred = self.predict(X)
            t1 = self.__evaluate_cls(y_true=y, y_pred=y_pred, list_metrics=list_errors)
        y_pred = self.predict(X)
        t2 = self.__evaluate_cls(y_true=y, y_pred=y_pred, list_metrics=list_scores)
        return {**t2, **t1}

    def evaluate(self, y_true, y_pred, list_metrics=None):
        """Return the list of performance metrics of the prediction.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Predicted values for `X`.

        list_metrics : list
            You can get metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        results : dict
            The results of the list metrics
        """
        pass

    def score(self, X, y, method=None):
        """Return the metric of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a precomputed kernel matrix or a list of generic objects instead with shape
            ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted`` is the number of samples used in the fitting for the estimator.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        method : str, default="RMSE"
            You can get metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        result : float
            The result of selected metric
        """
        pass

    def scores(self, X, y, list_methods=None):
        """Return the list of metrics of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a precomputed kernel matrix or a list of generic objects instead with shape
            ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted`` is the number of samples used in the fitting for the estimator.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        list_methods : list, default=("MSE", "MAE")
            You can get metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        results : dict
            The results of the list metrics
        """
        pass

    def save_loss_train(self, save_path="history", filename="loss.csv"):
        """
        Save the loss (convergence) during the training process to csv file.

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

    def save_metrics(self, y_true, y_pred, list_metrics=("RMSE", "MAE"), save_path="history", filename="metrics.csv"):
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
        results = self.evaluate(y_true, y_pred, list_metrics)
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
        if filename[-4:] != ".pkl":
            filename += ".pkl"
        return pickle.load(open(f"{load_path}/{filename}", 'rb'))
