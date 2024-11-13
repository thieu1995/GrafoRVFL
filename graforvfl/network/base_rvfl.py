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

    trainer : str, default = "MPI"
        The utilized method for training weights of hidden-output layer and weights of input-output layer.
            + MPI: Moore-Penrose inversion (Ordinary Least Squares without regularization)
            + L2: Ordinary Least Squares (OLS) regression with regularization

    alpha : float (Optional), default=0.5
        Regularization parameter for L2 training. Effective only when `trainer="L2"`.

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

    def __init__(self, size_hidden=10, act_name='sigmoid', weight_initializer="random_uniform", trainer="MPI", alpha=0.5, seed=None):
        self.size_hidden = size_hidden
        self.act_name = act_name
        self.act_func = getattr(activator, self.act_name)
        self.seed = seed
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

    def _trained(self, trainer="MPI", D=None, y=None):
        if trainer == "MPI":        # Standard OLS (alpha = 0)
            return np.linalg.pinv(D) @ y
        else:   # trainer == "L2":
            ridge_model = Ridge(alpha=self.alpha, fit_intercept=False, random_state=self.seed)
            return ridge_model.fit(D, y).coef_.T

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
        self.weights["Wh"] = self.weight_randomer((self.size_hidden, self.size_input), seed=self.seed)
        self.weights["bh"] = self.weight_randomer(self.size_hidden, seed=self.seed).flatten()
        H = self.act_func(X @ self.weights["Wh"].T + self.weights["bh"])
        D = np.concatenate((X, H), axis=1)
        self.weights["Wioho"] = self._trained(self.trainer, D, y)
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
        H = self.act_func(X @ self.weights["Wh"].T + self.weights["bh"])
        D = np.concatenate((X, H), axis=1)
        y_pred = D @ self.weights["Wioho"]
        return y_pred

    def predict_proba(self, X):
        """
        Predict probabilities (or scores) for classification tasks.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_pred : ndarray
            Predicted probabilities or scores.
        """
        H = self.act_func(X @ self.weights["Wh"].T + self.weights["bh"])
        D = np.concatenate((X, H), axis=1)
        y_pred = D @ self.weights["Wioho"]
        return y_pred

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

    def __evaluate_reg(self, y_true, y_pred, list_metrics=("MSE", "MAE")):
        rm = RegressionMetric(y_true=y_true, y_pred=y_pred)
        return rm.get_metrics_by_list_names(list_metrics)

    def __evaluate_cls(self, y_true, y_pred, list_metrics=("AS", "RS")):
        cm = ClassificationMetric(y_true, y_pred)
        return cm.get_metrics_by_list_names(list_metrics)

    def __score_reg(self, X, y, metric="RMSE"):
        y_pred = self.predict(X)
        return RegressionMetric(y, y_pred).get_metric_by_name(metric)[metric]

    def __scores_reg(self, X, y, list_metrics=("MSE", "MAE")):
        y_pred = self.predict(X)
        return self.__evaluate_reg(y_true=y, y_pred=y_pred, list_metrics=list_metrics)

    def __score_cls(self, X, y, metric="AS"):
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

    def __scores_cls(self, X, y, list_metrics=("AS", "RS")):
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
            t1 = self.__evaluate_cls(y_true=y, y_pred=y_pred, list_metrics=list_errors)
        y_pred = self.predict(X)
        t2 = self.__evaluate_cls(y_true=y, y_pred=y_pred, list_metrics=list_metrics)
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
