#!/usr/bin/env python
# Created by "Thieu" at 09:52, 17/08/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from scipy import special as ss
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.linear_model import Ridge
from graforvfl.network.base_rvfl import BaseRVFL
from graforvfl.shared import activator
from graforvfl.shared.scaler import OneHotEncoder


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
        "soft_sign", "tanh_shrink", "soft_shrink", "hard_shrink", "softmin", "softmax", "log_softmax" ]

    weight_initializer : str, default="random_uniform"
        The weight initialization methods. The supported methods are:
        ["orthogonal", "he_uniform", "he_normal", "glorot_uniform", "glorot_normal",
        "lecun_uniform", "lecun_normal", "random_uniform", "random_normal"]
        For definition of these methods, please check it at: https://keras.io/api/layers/initializers/

    reg_alpha : float (Optional), default=None
        Regularization parameter for L2 training. Effective only when `reg_alpha` > 0.

    seed: int (Optional), default=None
        Determines random number generation for weights and bias initialization.
        Pass an int for reproducible results across multiple function calls.

    Examples
    --------
    >>> from graforvfl import RvflRegressor, Data
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=200, random_state=1)
    >>> data = Data(X, y)
    >>> data.split_train_test(test_size=0.2, random_state=1)
    >>> model = RvflRegressor(size_hidden=10, act_name='sigmoid', weight_initializer="random_normal", reg_alpha=0.5, seed=42)
    >>> model.fit(data.X_train, data.y_train)
    >>> pred = model.predict(data.X_test)
    >>> print(pred)
    """

    def __init__(self, size_hidden=10, act_name='sigmoid', weight_initializer="random_normal", reg_alpha=None, seed=None):
        super().__init__(size_hidden=size_hidden, act_name=act_name, weight_initializer=weight_initializer, reg_alpha=reg_alpha, seed=seed)

    def score(self, X, y):
        """Return the real R2 (Coefficient of Determination) metric, not (Pearson’s Correlation Index)^2 like Scikit-Learn library.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a precomputed kernel matrix or a list of generic objects instead with shape
            ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted`` is the number of samples used in the fitting for the estimator.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        Returns
        -------
        result : float
            The result of selected metric
        """
        return self._score_reg(X, y, "R2")

    def scores(self, X, y, list_metrics=("MSE", "MAE")):
        """Return the list of regression metrics of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a precomputed kernel matrix or a list of generic objects instead with shape
            ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted`` is the number of samples used in the fitting for the estimator.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        list_metrics : list, default=("MSE", "MAE")
            You can get regression metrics from Permetrics library: https://permetrics.readthedocs.io/en/latest/pages/regression.html

        Returns
        -------
        results : dict
            The results of the list metrics
        """
        return self._scores_reg(X, y, list_metrics)

    def evaluate(self, y_true, y_pred, list_metrics=("MSE", "MAE")):
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
        return self._evaluate_reg(y_true, y_pred, list_metrics)


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
        "soft_sign", "tanh_shrink", "soft_shrink", "hard_shrink", "softmin", "softmax", "log_softmax" ]

    weight_initializer : str, default="random_uniform"
        The weight initialization methods. The supported methods are:
        ["orthogonal", "he_uniform", "he_normal", "glorot_uniform", "glorot_normal",
        "lecun_uniform", "lecun_normal", "random_uniform", "random_normal"]
        For definition of these methods, please check it at: https://keras.io/api/layers/initializers/

    reg_alpha : float (Optional), default=None
        Regularization parameter for L2 training. Effective only when `reg_alpha` > 0.

    seed: int (Optional), default=None
        Determines random number generation for weights and bias initialization.
        Pass an int for reproducible results across multiple function calls.

    Examples
    --------
    >>> from graforvfl import Data, RvflClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, random_state=1)
    >>> data = Data(X, y)
    >>> data.split_train_test(test_size=0.2, random_state=1)
    >>> model = RvflClassifier(size_hidden=10, act_name='sigmoid', weight_initializer="random_normal", reg_alpha=0.5, seed=42)
    >>> model.fit(data.X_train, data.y_train)
    >>> pred = model.predict(data.X_test)
    >>> print(pred)
    array([1, 0, 1, 0, 1])
    """

    CLS_OBJ_LOSSES = ["CEL", "HL", "KLDL", "BSL"]

    def __init__(self, size_hidden=10, act_name='sigmoid', weight_initializer="random_normal", reg_alpha=None, seed=None):
        super().__init__(size_hidden=size_hidden, act_name=act_name, weight_initializer=weight_initializer, reg_alpha=reg_alpha, seed=seed)
        self.n_labels, self.ohe_scaler, self.classes_ = None, None, None

    def _check_input_output(self, X, y):
        ## Check X, y
        self.size_input = X.shape[1]
        y = np.squeeze(np.array(y))
        if y.ndim == 0:  # Single label
            y = np.array([y])  # Convert to 1D array
        if y.ndim == 1:
            self.size_output = self.n_labels = len(np.unique(y))
            self.classes_ = np.unique(y)
        else:
            raise TypeError("Invalid y array shape, it should be 1D vector containing labels 0, 1, 2,.. and so on.")

    def fit(self, X, y):
        """
        Fit the RVFLClassifier model on the entire training dataset.

        This method trains the RVFL network using either ordinary least squares (OLS)
        or ridge regression, depending on the value of `reg_alpha`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.

        y : array-like of shape (n_samples,)
            Target class labels corresponding to X.

        Returns
        -------
        self : object
            Returns the fitted model.
        """
        ## Check X, y
        X = self._to_numpy(X, is_X=True)
        y = self._to_numpy(y, is_X=False).reshape(-1, 1)  # Ensure y is a column vector
        self._check_input_output(X, y)
        ## Check parameters
        self.act_func = getattr(activator, self.act_name)
        self.weight_randomer = self._get_weight_initializer(self.weight_initializer)

        # Transform y to one-hot encoding
        self.ohe_scaler = OneHotEncoder().fit(y)
        y_scaled = self.ohe_scaler.transform(y)  # Transform y to one-hot encoding

        ## Train the model
        self._init_weights()
        D = self._get_D(X)
        if self.reg_alpha is None or self.reg_alpha == 0:         # Standard OLS (reg_alpha = 0)
            self.weights["Wioho"] = np.linalg.pinv(D) @ y_scaled
        else:                           # trainer == "L2":
            ridge_model = Ridge(alpha=self.reg_alpha, fit_intercept=False, random_state=self.seed)
            self.weights["Wioho"] = ridge_model.fit(D, y_scaled).coef_.T
        self.is_fitted = True
        self.P = np.linalg.inv(D.T @ D + 1e-8 * np.eye(D.shape[1]))
        return self

    def partial_fit(self, X, y, classes=None):
        """
        Perform an incremental update to the model using a mini-batch of data.

        This method supports online or real-time learning. The first call to `partial_fit`
        must include the full list of target class labels via the `classes` parameter
        to initialize the output layer and encoder.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples for the current batch.

        y : array-like of shape (n_samples,)
            Target class labels for the current batch.

        classes : array-like of shape (n_classes,), optional
            List of all possible class labels. Must be provided in the first call only.

        Returns
        -------
        self : object
            Returns the partially fitted model.

        Raises
        ------
        TypeError
            If `classes` is not provided in the first call or not of correct type.
        """
        X = self._to_numpy(X, is_X=True)
        y = self._to_numpy(y, is_X=False).reshape(-1, 1)  # Ensure y is a column vector
        if not self.is_fitted:
            ## Check parameters
            self.act_func = getattr(activator, self.act_name)
            self.weight_randomer = self._get_weight_initializer(self.weight_initializer)

            # Transform y to one-hot encoding
            if classes is None or not isinstance(classes, (list, tuple, np.ndarray)):
                raise TypeError("classes must be a list, tuple, or numpy array of class labels for first partial_fit call.")
            self.size_output = self.n_labels = len(classes)
            self.classes_ = classes
            self.ohe_scaler = OneHotEncoder().fit(np.reshape(classes, (-1, 1)))
            self.size_input = X.shape[1]

            self._init_weights()
            D = self._get_D(X)
            self.weights["Wioho"] = np.zeros((D.shape[1], self.size_output))
            self.P = np.eye(D.shape[1]) * 1e5
            self.is_fitted = True

        # Batch update
        y = self.ohe_scaler.transform(y)  # Transform y to one-hot encoding
        D = self._get_D(X)
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
        Predict class labels for the input samples X.

        This method computes the output probabilities using the hidden and direct layers,
        then uses the inverse of the one-hot encoder to return class predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted class labels.
        """
        y_logits = self.predict_proba(X)
        return self.ohe_scaler.inverse_transform(y_logits)

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
        X = self._to_numpy(X, is_X=True)
        D = self._get_D(X)
        y_raw = D @ self.weights["Wioho"]
        if self.size_output > 1:
            return ss.softmax(y_raw, axis=1)
        else:   # if binary, use sigmoid
            return np.column_stack([1 - ss.expit(y_raw), ss.expit(y_raw)])

    def score(self, X, y):
        """Return the real Accuracy Score metric

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a precomputed kernel matrix or a list of generic objects instead with shape
            ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted`` is the number of samples used in the fitting for the estimator.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        Returns
        -------
        result : float
            The result of selected metric
        """
        return self._score_cls(X, y, "AS")

    def scores(self, X, y, list_metrics=("AS", "RS")):
        """Return the list of classification metrics of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
           Test samples. For some estimators this may be a precomputed kernel matrix or a list of generic objects instead with shape
           ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted`` is the number of samples used in the fitting for the estimator.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
           True values for `X`.

        list_metrics : list, default=("AS", "RS")
           You can get classification metrics from Permetrics library: https://permetrics.readthedocs.io/en/latest/pages/classification.html

        Returns
        -------
        results : dict
           The results of the list metrics
        """
        return self._scores_cls(X, y, list_metrics)

    def evaluate(self, y_true, y_pred, list_metrics=("AS", "RS")):
        """Return the list of classification performance metrics of the prediction.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Predicted values for `X`.

        list_metrics : list
            You can get classification metrics from Permetrics library: https://permetrics.readthedocs.io/en/latest/pages/classification.html

        Returns
        -------
        results : dict
            The results of the list metrics
        """
        return self._evaluate_cls(y_true, y_pred, list_metrics)
