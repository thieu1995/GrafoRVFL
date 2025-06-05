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
    """
    A simple implementation of one-hot encoding for 1D categorical data.

    Attributes:
        categories_ (np.ndarray): Sorted array of unique categories fitted from the input data.
    """
    def __init__(self):
        """Initialize the encoder with no categories."""
        self.categories_ = None

    def fit(self, X):
        """
        Fit the encoder to the unique categories in X.

        Args:
            X (array-like): 1D array of categorical values.

        Returns:
            self: Fitted OneHotEncoder instance.
        """
        X = np.asarray(X).ravel()
        self.categories_ = np.unique(X)
        return self

    def transform(self, X):
        """
        Transform input data into one-hot encoded format.

        Args:
            X (array-like): 1D array of categorical values.

        Returns:
            np.ndarray: One-hot encoded array of shape (n_samples, n_categories).

        Raises:
            ValueError: If the encoder has not been fitted or unknown category is found.
        """
        if self.categories_ is None:
            raise ValueError("The encoder has not been fitted yet.")

        X = np.asarray(X).ravel()
        one_hot = np.zeros((X.shape[0], len(self.categories_)), dtype=int)

        for i, val in enumerate(X):
            indices = np.where(self.categories_ == val)[0]
            if len(indices) == 0:
                raise ValueError(f"Unknown category encountered during transform: {val}")
            one_hot[i, indices[0]] = 1
        return one_hot

    def fit_transform(self, X):
        """
        Fit the encoder to X and transform X.

        Args:
            X (array-like): 1D array of categorical values.

        Returns:
            np.ndarray: One-hot encoded array of shape (n_samples, n_categories).
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, one_hot):
        """
        Convert one-hot encoded data back to original categories.

        Args:
            one_hot (np.ndarray): 2D array of one-hot encoded data.

        Returns:
            np.ndarray: 1D array of original categorical values.

        Raises:
            ValueError: If the encoder has not been fitted or shape mismatch occurs.
        """
        if self.categories_ is None:
            raise ValueError("The encoder has not been fitted yet.")
        if one_hot.shape[1] != len(self.categories_):
            raise ValueError("The shape of the input does not match the number of categories.")
        return np.array([self.categories_[np.argmax(row)] for row in one_hot])


class LabelEncoder:
    """
    Encode categorical labels as integer indices and decode them back.

    This class maps unique categorical labels to integers from 0 to n_classes - 1.
    """

    def __init__(self):
        """
        Initialize the label encoder.
        """
        self.unique_labels = None
        self.label_to_index = {}

    def fit(self, y):
        """
        Fit the encoder by finding unique labels in the input data.

        Parameters
        ----------
        y : array-like
            Input labels.

        Returns
        -------
        self : LabelEncoder
            Fitted LabelEncoder instance.
        """
        y = np.asarray(y).ravel()
        self.unique_labels = np.unique(y)
        self.label_to_index = {label: i for i, label in enumerate(self.unique_labels)}
        return self

    def transform(self, y):
        """
        Transform labels to integer indices.

        Parameters
        ----------
        y : array-like
            Labels to encode.

        Returns
        -------
        encoded_labels : np.ndarray
            Encoded integer labels.

        Raises
        ------
        ValueError
            If the encoder has not been fitted or unknown labels are found.
        """
        if self.unique_labels is None:
            raise ValueError("Label encoder has not been fit yet.")
        y = np.asarray(y).ravel()
        encoded = []
        for label in y:
            if label not in self.label_to_index:
                raise ValueError(f"Unknown label: {label}")
            encoded.append(self.label_to_index[label])
        return np.array(encoded)

    def fit_transform(self, y):
        """
        Fit the encoder and transform labels in one step.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Input labels.

        Returns
        -------
        np.ndarray
            Encoded integer labels.
        """
        return self.fit(y).transform(y)

    def inverse_transform(self, y):
        """
        Transform integer indices back to original labels.

        Parameters
        ----------
        y : array-like of int
            Encoded integer labels.

        Returns
        -------
        original_labels : np.ndarray
            Original labels.

        Raises
        ------
        ValueError
            If the encoder has not been fitted or index is out of bounds.
        """
        if self.unique_labels is None:
            raise ValueError("Label encoder has not been fit yet.")
        y = np.asarray(y).ravel()
        return np.array([self.unique_labels[i] if 0 <= i < len(self.unique_labels) else "unknown" for i in y])


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
