#!/usr/bin/env python
# Created by "Thieu" at 12:36, 17/09/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from scipy.stats import boxcox, yeojohnson
from scipy.special import inv_boxcox
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler


class OneHotEncoder:
    def __init__(self):
        self.categories_ = None

    def fit(self, X):
        """Fit the encoder to unique categories in X."""
        self.categories_ = np.unique(X)
        return self

    def transform(self, X):
        """Transform X into one-hot encoded format."""
        if self.categories_ is None:
            raise ValueError("The encoder has not been fitted yet.")
        one_hot = np.zeros((X.shape[0], len(self.categories_)), dtype=int)
        for i, val in enumerate(X):
            index = np.where(self.categories_ == val)[0][0]
            one_hot[i, index] = 1
        return one_hot

    def fit_transform(self, X):
        """Fit the encoder to X and transform X."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, one_hot):
        """Convert one-hot encoded format back to original categories."""
        if self.categories_ is None:
            raise ValueError("The encoder has not been fitted yet.")
        if one_hot.shape[1] != len(self.categories_):
            raise ValueError("The shape of the input does not match the number of categories.")
        original = np.array([self.categories_[np.argmax(row)] for row in one_hot])
        return original


class LabelEncoder:
    """
    Encode categorical features as integer labels.
    """

    def __init__(self):
        self.unique_labels = None
        self.label_to_index = {}

    def fit(self, y):
        """
        Fit label encoder to a given set of labels.

        Parameters
        ----------
        y : array-like
            Labels to encode.
        """
        self.unique_labels = np.unique(y)
        self.label_to_index = {label: i for i, label in enumerate(self.unique_labels)}

    def transform(self, y):
        """
        Transform labels to encoded integer labels.

        Parameters
        ----------
        y : array-like
            Labels to encode.

        Returns:
        --------
        encoded_labels : array-like
            Encoded integer labels.
        """
        if self.unique_labels is None:
            raise ValueError("Label encoder has not been fit yet.")
        return np.array([self.label_to_index[label] for label in y])

    def fit_transform(self, y):
        """Fit label encoder and return encoded labels.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        y : array-like of shape (n_samples,)
            Encoded labels.
        """
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y):
        """
        Transform integer labels to original labels.

        Parameters
        ----------
        y : array-like
            Encoded integer labels.

        Returns
        -------
        original_labels : array-like
            Original labels.
        """
        if self.unique_labels is None:
            raise ValueError("Label encoder has not been fit yet.")
        return np.array([self.unique_labels[i] if i in self.label_to_index.values() else "unknown" for i in y])


class ObjectiveScaler:
    """
    For label scaler in classification (binary and multiple classification)
    """
    def __init__(self, obj_name="sigmoid", ohe_scaler=None):
        """
        ohe_scaler: Need to be an instance of One-Hot-Encoder for softmax scaler (multiple classification problem)
        """
        self.obj_name = obj_name
        self.ohe_scaler = ohe_scaler

    def transform(self, data):
        if self.obj_name == "sigmoid" or self.obj_name == "self":
            return data
        elif self.obj_name == "hinge":
            data = np.squeeze(np.array(data))
            data[np.where(data == 0)] = -1
            return data
        elif self.obj_name == "softmax":
            data = self.ohe_scaler.fit_transform(np.reshape(data, (-1, 1)))
            return data

    def inverse_transform(self, data):
        if self.obj_name == "sigmoid":
            data = np.squeeze(np.array(data))
            data = np.rint(data).astype(int)
        elif self.obj_name == "hinge":
            data = np.squeeze(np.array(data))
            data = np.ceil(data).astype(int)
            data[np.where(data == -1)] = 0
        elif self.obj_name == "softmax":
            data = np.squeeze(np.array(data))
            data = np.argmax(data, axis=1)
        return data


class Log1pScaler(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        # LogETransformer doesn't require fitting, so we simply return self.
        return self

    def transform(self, X):
        # Apply the natural logarithm to each element of the input data
        return np.log1p(X)

    def inverse_transform(self, X):
        # Apply the exponential function to reverse the logarithmic transformation
        return np.expm1(X)


class LogeScaler(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        # LogETransformer doesn't require fitting, so we simply return self.
        return self

    def transform(self, X):
        # Apply the natural logarithm (base e) to each element of the input data
        return np.log(X)

    def inverse_transform(self, X):
        # Apply the exponential function to reverse the logarithmic transformation
        return np.exp(X)


class SqrtScaler(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        # SqrtScaler doesn't require fitting, so we simply return self.
        return self

    def transform(self, X):
        # Apply the square root transformation to each element of the input data
        return np.sqrt(X)

    def inverse_transform(self, X):
        # Apply the square of each element to reverse the square root transformation
        return X ** 2


class BoxCoxScaler(BaseEstimator, TransformerMixin):

    def __init__(self, lmbda=None):
        self.lmbda = lmbda

    def fit(self, X, y=None):
        # Estimate the lambda parameter from the data if not provided
        if self.lmbda is None:
            _, self.lmbda = boxcox(X.flatten())
        return self

    def transform(self, X):
        # Apply the Box-Cox transformation to the data
        X_new = boxcox(X.flatten(), lmbda=self.lmbda)
        return X_new.reshape(X.shape)

    def inverse_transform(self, X):
        # Inverse transform using the original lambda parameter
        return inv_boxcox(X, self.lmbda)


class YeoJohnsonScaler(BaseEstimator, TransformerMixin):

    def __init__(self, lmbda=None):
        self.lmbda = lmbda

    def fit(self, X, y=None):
        # Estimate the lambda parameter from the data if not provided
        if self.lmbda is None:
            _, self.lmbda = yeojohnson(X.flatten())
        return self

    def transform(self, X):
        # Apply the Yeo-Johnson transformation to the data
        X_new = boxcox(X.flatten(), lmbda=self.lmbda)
        return X_new.reshape(X.shape)

    def inverse_transform(self, X):
        # Inverse transform using the original lambda parameter
        return inv_boxcox(X, self.lmbda)


class SinhArcSinhScaler(BaseEstimator, TransformerMixin):
    # https://stats.stackexchange.com/questions/43482/transformation-to-increase-kurtosis-and-skewness-of-normal-r-v
    def __init__(self, epsilon=0.1, delta=1.0):
        self.epsilon = epsilon
        self.delta = delta

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.sinh(self.delta * np.arcsinh(X) - self.epsilon)

    def inverse_transform(self, X):
        return np.sinh((np.arcsinh(X) + self.epsilon) / self.delta)
