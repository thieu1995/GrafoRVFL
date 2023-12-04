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

    trainer : str, default = "OLS"
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

    def __init__(self, size_hidden=10, act_name='sigmoid', weight_initializer="random_normal", trainer="OLS", alpha=0.5):
        super().__init__(size_hidden=size_hidden, act_name=act_name, weight_initializer=weight_initializer, trainer=trainer, alpha=alpha)

    def score(self, X, y, method="RMSE"):
        return self._BaseRVFL__score_reg(X, y, method)

    def scores(self, X, y, list_methods=("MSE", "MAE")):
        return self._BaseRVFL__scores_reg(X, y, list_methods)

    def evaluate(self, y_true, y_pred, list_metrics=("MSE", "MAE")):
        return self._BaseRVFL__evaluate_reg(y_true, y_pred, list_metrics)

