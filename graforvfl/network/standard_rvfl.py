#!/usr/bin/env python
# Created by "Thieu" at 09:52, 17/08/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin
from graforvfl.network.base_rvfl import BaseRVFL
from graforvfl.shared.scaler import ObjectiveScaler, OneHotEncoder


class RvflRegressor(BaseRVFL, RegressorMixin):
    """
    Defines the ELM network for Regression problems that inherit the BaseRVFL and RegressorMixin classes.

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

    trainer : str, default = "MPI"
        The utilized method for training weights of hidden-output layer and weights of input-output layer.
            + MPI: Moore-Penrose inversion
            + OLS: Ordinary Least Squares (OLS) without regularization
            + L2: OLS regression with regularization

    alpha : float (Optional), default=0.5
        The penalty value for L2 method. Only effect when `trainer`="L2".

    Examples
    --------
    >>> from graforvfl import RvflRegressor, Data
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=200, random_state=1)
    >>> data = Data(X, y)
    >>> data.split_train_test(test_size=0.2, random_state=1)
    >>> model = RvflRegressor(size_hidden=10, act_name='sigmoid', weight_initializer="random_normal", trainer="OLS", alpha=0.5)
    >>> model.fit(data.X_train, data.y_train)
    >>> pred = model.predict(data.X_test)
    >>> print(pred)
    """

    def __init__(self, size_hidden=10, act_name='sigmoid', weight_initializer="random_normal", trainer="MPI", alpha=0.5):
        super().__init__(size_hidden=size_hidden, act_name=act_name, weight_initializer=weight_initializer, trainer=trainer, alpha=alpha)

    def score(self, X, y, method="RMSE"):
        return self._BaseRVFL__score_reg(X, y, method)

    def scores(self, X, y, list_methods=("MSE", "MAE")):
        return self._BaseRVFL__scores_reg(X, y, list_methods)

    def evaluate(self, y_true, y_pred, list_metrics=("MSE", "MAE")):
        return self._BaseRVFL__evaluate_reg(y_true, y_pred, list_metrics)


class RvflClassifier(BaseRVFL, ClassifierMixin):
    """
    Defines the general class of Metaheuristic-based ELM network for Classification problems that inherit the BaseRVFL and ClassifierMixin classes.

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

    trainer : str, default = "MPI"
        The utilized method for training weights of hidden-output layer and weights of input-output layer.
            + MPI: Moore-Penrose inversion
            + OLS: Ordinary Least Squares (OLS) without regularization
            + L2: OLS regression with regularization

    alpha : float (Optional), default=0.5
        The penalty value for L2 method. Only effect when `trainer`="L2".

    Examples
    --------
    >>> from graforvfl import Data, RvflClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, random_state=1)
    >>> data = Data(X, y)
    >>> data.split_train_test(test_size=0.2, random_state=1)
    >>> model = RvflClassifier(size_hidden=10, act_name='sigmoid', weight_initializer="random_normal", trainer="OLS", alpha=0.5)
    >>> model.fit(data.X_train, data.y_train)
    >>> pred = model.predict(data.X_test)
    >>> print(pred)
    array([1, 0, 1, 0, 1])
    """

    CLS_OBJ_LOSSES = ["CEL", "HL", "KLDL", "BSL"]

    def __init__(self, size_hidden=10, act_name='sigmoid', weight_initializer="random_normal", trainer="MPI", alpha=0.5):
        super().__init__(size_hidden=size_hidden, act_name=act_name, weight_initializer=weight_initializer, trainer=trainer, alpha=alpha)
        self.n_labels = None
        self.obj_scaler = None

    def fit(self, X, y):
        self.size_input = X.shape[1]
        if type(y) in (list, tuple, np.ndarray):
            y = np.squeeze(np.asarray(y))
            if y.ndim == 1:
                self.n_labels = len(np.unique(y))
                self.size_output = self.n_labels
            else:
                raise TypeError("Invalid y array shape, it should be 1D vector containing labels 0, 1, 2,.. and so on.")
        else:
            raise TypeError("Invalid y array type, it should be list, tuple or np.ndarray")
        ohe_scaler = OneHotEncoder(sparse=False)
        ohe_scaler.fit(np.reshape(y, (-1, 1)))
        self.obj_scaler = ObjectiveScaler(obj_name="softmax", ohe_scaler=ohe_scaler)
        y_scaled = self.obj_scaler.transform(y)

        self.weights["Wh"] = self.weight_randomer((self.size_hidden, self.size_input))
        self.weights["bh"] = self.weight_randomer(self.size_hidden).flatten()
        H = self.act_func(X @ self.weights["Wh"].T + self.weights["bh"])
        D = np.concatenate((X, H), axis=1)
        self.weights["Wioho"] = self._trained(self.trainer, D, y_scaled)
        return self

    def predict_proba(self, X):
        H = self.act_func(X @ self.weights["Wh"].T + self.weights["bh"])
        D = np.concatenate((X, H), axis=1)
        y_pred = D @ self.weights["Wioho"]
        return y_pred

    def predict(self, X):
        y_pred = self.predict_proba(X)
        return self.obj_scaler.inverse_transform(y_pred)

    def score(self, X, y, method="AS"):
        return self._BaseRVFL__score_cls(X, y, method)

    def scores(self, X, y, list_methods=("AS", "RS")):
        return self._BaseRVFL__scores_cls(X, y, list_methods)

    def evaluate(self, y_true, y_pred, list_metrics=("AS", "RS")):
        return self._BaseRVFL__evaluate_cls(y_true, y_pred, list_metrics)
