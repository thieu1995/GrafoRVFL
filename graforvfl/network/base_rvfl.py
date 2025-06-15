#!/usr/bin/env python
# Created by "Thieu" at 09:48, 17/08/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import inspect
import pprint
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
    This class defines the general Random Vector Functional Link (RVFL) network.
    It is a single-hidden layer network with direct connection between input and output.

    Parameters
    ----------
    size_hidden : int, default=10
        Number of nodes in the hidden layer.

    act_name : str, default="sigmoid"
        Name of the activation function for the hidden layer. Supported values include:
        ["none", "relu", "leaky_relu", "celu", "prelu", "gelu", "elu", "selu", "rrelu", "tanh", "hard_tanh",
        "sigmoid", "hard_sigmoid", "log_sigmoid", "silu", "swish", "hard_swish", "soft_plus", "mish",
        "soft_sign", "tanh_shrink", "soft_shrink", "hard_shrink", "softmin", "softmax", "log_softmax" ]

    weight_initializer : str, default="random_uniform"
        Method for initializing weights (input-hidden weights). Supported methods include:
        ["orthogonal", "he_uniform", "he_normal", "glorot_uniform", "glorot_normal",
        "lecun_uniform", "lecun_normal", "random_uniform", "random_normal"]
        For definition of these methods, please check it at: https://keras.io/api/layers/initializers/

    reg_alpha : float (Optional), default=None
        Regularization parameter for L2 training. Effective only when `reg_alpha` > 0.

    seed: int, default=None
        Determines random number generation for weights and bias initialization.
        Pass an int for reproducible results across multiple function calls.

    Attributes
    ----------
    weights : dict
        Dictionary containing the initialized weights for hidden layers and output layers.
    act_func : callable
        The activation function applied to the hidden layer.
    size_input : int
        Number of features in the input data.
    size_output : int
        Number of outputs based on the target data dimensionality.
    loss_train : list
        Stores the loss history during training, if applicable.
    """

    SUPPORTED_CLS_METRICS = get_all_classification_metrics()
    SUPPORTED_REG_METRICS = get_all_regression_metrics()
    CLS_OBJ_LOSSES = None
    SUPPORTED_WEIGHT_INITIALIZER = [
        "orthogonal", "he_uniform", "he_normal", "glorot_uniform", "glorot_normal",
        "lecun_uniform", "lecun_normal", "random_uniform", "random_normal"
    ]
    SUPPORTED_ACTIVATION = ["none", "relu", "leaky_relu", "celu", "prelu", "gelu", "elu", "selu", "rrelu",
                            "tanh", "hard_tanh", "sigmoid", "hard_sigmoid", "log_sigmoid", "silu", "swish",
                            "hard_swish", "soft_plus", "mish", "soft_sign", "tanh_shrink", "soft_shrink",
                            "hard_shrink", "softmin", "softmax", "log_softmax"]

    def __init__(self, size_hidden=10, act_name='sigmoid', weight_initializer="random_uniform", reg_alpha=None, seed=None):
        self.size_hidden = size_hidden
        self.act_name = act_name
        self.weight_initializer = weight_initializer
        self.reg_alpha = reg_alpha
        self.seed = seed
        self.is_fitted = False

        self.act_func, self.weight_randomer = None, None
        self.size_input, self.size_output = None, None
        self.weights = {}
        self.loss_train = None, None
        self.feature_names, self.label_name = None, None

    # def __repr__(self, **kwargs):
    #     """Pretty-print parameters like scikit-learn's Estimator."""
    #     param_order = list(inspect.signature(self.__init__).parameters.keys())
    #     param_dict = {k: getattr(self, k) for k in param_order}
    #
    #     param_str = ", ".join(f"{k}={repr(v)}" for k, v in param_dict.items())
    #     if len(param_str) <= 80:
    #         return f"{self.__class__.__name__}({param_str})"
    #     else:
    #         formatted_params = ",\n  ".join(f"{k}={pprint.pformat(v)}" for k, v in param_dict.items())
    #         return f"{self.__class__.__name__}(\n  {formatted_params}\n)"

    def _to_numpy(self, data, is_X=True):
        if isinstance(data, pd.DataFrame):
            if is_X:
                if self.feature_names is None:
                    self.feature_names = data.columns.tolist()
            else:
                if self.label_name is None:
                    self.label_name = data.columns.tolist()
            return data.values
        elif isinstance(data, pd.Series):
            if is_X:
                if self.feature_names is None:
                    self.feature_names = [data.name if data.name is not None else "feature_0"]
            else:
                if self.label_name is None:
                    self.label_name = [data.name if data.name is not None else "label_0"]
            return data.values.reshape(-1, 1)
        elif isinstance(data, (list, tuple, np.ndarray)):
            data_new = data
            if data.ndim == 1:
                data_new = data.reshape(-1, 1)
            if is_X:
                if self.feature_names is None:
                    self.feature_names = [f"feature_{i}" for i in range(data_new.shape[1])]
            else:
                if self.label_name is None:
                    self.label_name = [f"label_{i}" for i in range(data_new.shape[1])]
            return data
        else:
            raise TypeError("Input X must be a numpy array or pandas DataFrame/Series.")

    def _get_weight_initializer(self, name):
        if isinstance(name, str):
            wi = boundary_controller.check_str("weight_initializer", name, self.SUPPORTED_WEIGHT_INITIALIZER)
            wr = getattr(randomer, f"{wi}_initializer")
            return wr
        else:
            raise ValueError(f"weight_initializer should be a string and belongs to {self.SUPPORTED_WEIGHT_INITIALIZER}")

    def _init_weights(self):
        self.weights["Wh"] = self.weight_randomer((self.size_hidden, self.size_input), seed=self.seed)
        self.weights["bh"] = self.weight_randomer(self.size_hidden, seed=self.seed).flatten()

    def _get_D(self, X):
        H = self.act_func(X @ self.weights["Wh"].T + self.weights["bh"])
        return np.concatenate([X, H], axis=1)

    def _check_input_output(self, X, y):
        ## Check X, y
        self.size_input = X.shape[1]
        if y.ndim == 1:
            self.size_output = 1
        elif y.ndim == 2:
            self.size_output = y.shape[1]
        else:
            raise TypeError("Invalid y array shape, it should be 1D vector or 2D matrix.")

    def fit(self, X, y):
        """
        Fit the RVFL model to the training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training input features.

        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            Target values.

        Returns
        -------
        self : BaseRVFL
            The fitted model.
        """
        ## Check X, y
        X = self._to_numpy(X, is_X=True)
        y = self._to_numpy(y, is_X=False)
        self._check_input_output(X, y)
        ## Check parameters
        self.act_func = getattr(activator, self.act_name)
        self.weight_randomer = self._get_weight_initializer(self.weight_initializer)

        ## Train the model
        self._init_weights()
        D = self._get_D(X)
        if self.reg_alpha is None or self.reg_alpha == 0:         # Standard OLS (reg_alpha = 0)
            self.weights["Wioho"] = np.linalg.pinv(D) @ y
        else:
            ridge_model = Ridge(alpha=self.reg_alpha, fit_intercept=False, random_state=self.seed)
            self.weights["Wioho"] = ridge_model.fit(D, y).coef_.T
        self.is_fitted = True
        self.P = np.linalg.inv(D.T @ D + 1e-8 * np.eye(D.shape[1]))
        return self

    def partial_fit(self, X, y):
        X = self._to_numpy(X, is_X=True)
        y = self._to_numpy(y, is_X=False).reshape(-1, 1)
        self._check_input_output(X, y)
        if not self.is_fitted:
            ## Check parameters
            self.act_func = getattr(activator, self.act_name)
            self.weight_randomer = self._get_weight_initializer(self.weight_initializer)

            self._init_weights()
            D = self._get_D(X)
            self.weights["Wioho"] = np.zeros((D.shape[1], self.size_output))
            self.P = np.eye(D.shape[1]) * 1e5
            self.is_fitted = True

        # Batch update
        D = self._get_D(X)
        self.weights["Wioho"] = np.reshape(self.weights["Wioho"], (D.shape[1], self.size_output))
        for idx in range(D.shape[0]):
            d_i = D[idx:idx + 1, :]     # (1, D.shape[1])
            y_i = y[idx:idx + 1, :]     # (1, self.size_output)
            P_dT = self.P @ d_i.T
            k = P_dT / (1.0 + d_i @ P_dT)
            self.weights["Wioho"] += k @ (y_i - d_i @ self.weights["Wioho"])
            self.P = self.P - k @ d_i @ self.P
        return self

    def predict(self, X):
        """
        Predict target values using the fitted RVFL model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_pred : ndarray
            Predicted target values.
        """
        X = self._to_numpy(X, is_X=True)
        D = self._get_D(X)
        return D @ self.weights["Wioho"]

    def __call__(self, X):
        return self.predict(X)

    def get_weights(self):
        """
        Retrieve the current weights of the RVFL model.

        Returns
        -------
        weights : dict
            Dictionary containing the current model weights.
        """
        return self.weights

    def set_weights(self, weights):
        """
        Set the weights for the RVFL model.

        Parameters
        ----------
        weights : dict
            Dictionary containing the weights to set.
        """
        self.weights = weights

    def get_weights_size(self):
        """
        Calculate the total number of parameters in the model.

        Returns
        -------
        size : int
            Total number of parameters across all weights.
        """
        return np.sum([item.size for item in self.weights.values()])

    def _evaluate_reg(self, y_true, y_pred, list_metrics=("MSE", "MAE")):
        y_true = self._to_numpy(y_true, is_X=False)
        rm = RegressionMetric(y_true=y_true, y_pred=y_pred)
        return rm.get_metrics_by_list_names(list_metrics)

    def _evaluate_cls(self, y_true, y_pred, list_metrics=("AS", "RS")):
        y_true = self._to_numpy(y_true, is_X=False)
        cm = ClassificationMetric(y_true, y_pred)
        return cm.get_metrics_by_list_names(list_metrics)

    def _score_reg(self, X, y, metric="RMSE"):
        y_pred = self.predict(X)
        return RegressionMetric(y, y_pred).get_metric_by_name(metric)[metric]

    def _scores_reg(self, X, y, list_metrics=("MSE", "MAE")):
        y_pred = self.predict(X)
        return self._evaluate_reg(y_true=y, y_pred=y_pred, list_metrics=list_metrics)

    def _score_cls(self, X, y, metric="AS"):
        return_prob = False
        if self.n_labels > 2:
            if metric in self.CLS_OBJ_LOSSES:
                return_prob = True
        if return_prob:
            y_pred = self.predict_proba(X)
        else:
            y_pred = self.predict(X)
        cm = ClassificationMetric(y_true=y, y_pred=y_pred)
        return cm.get_metric_by_name(metric)[metric]

    def _scores_cls(self, X, y, list_metrics=("AS", "RS")):
        list_errors = list(set(list_metrics) & set(self.CLS_OBJ_LOSSES))
        t1 = {}
        if len(list_errors) > 0:
            return_prob = False
            if self.n_labels > 2:
                return_prob = True
            if return_prob:
                y_pred = self.predict_proba(X)
            else:
                y_pred = self.predict(X)
            t1 = self._evaluate_cls(y_true=y, y_pred=y_pred, list_metrics=list_errors)
        y_pred = self.predict(X)
        t2 = self._evaluate_cls(y_true=y, y_pred=y_pred, list_metrics=list_metrics)
        return {**t2, **t1}

    def score(self, X, y):
        """Default interface for score function"""
        pass

    def scores(self, X, y, list_metrics=None):
        """Default interface for scores function"""
        pass

    def evaluate(self, y_true, y_pred, list_metrics=None):
        """Default interface for evaluate function"""
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
        y_true : ndarray
            Ground truth target values.
        y_pred : ndarray
            Predicted target values.
        list_metrics : list of str, default=("RMSE", "MAE")
            List of metrics to calculate.
        save_path : str, default="history"
            Directory to save the file.
        filename : str, default="metrics.csv"
            Name of the file (must end with `.csv`).
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
        X : ndarray
            Input features.
        y_true : ndarray
            Ground truth target values.
        save_path : str, default="history"
            Directory to save the file.
        filename : str, default="y_predicted.csv"
            Name of the file (must end with `.csv`).
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
        save_path : str, default="history"
            Directory to save the file.
        filename : str, default="network.pkl"
            Name of the file (must end with `.pkl`).
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
