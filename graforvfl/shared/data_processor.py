#!/usr/bin/env python
# Created by "Thieu" at 23:33, 10/08/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from graforvfl.shared.scaler import *


class TimeSeriesDifferencer:
    """
    A class for applying and reversing differencing on time series data.

    Differencing helps remove trends and seasonality from time series for better modeling.
    """

    def __init__(self, interval=1):
        """
        Initialize the differencer with a specified interval.

        Parameters
        ----------
        interval : int
            The lag interval to use for differencing. Must be >= 1.
        """
        if interval < 1:
            raise ValueError("Interval for differencing must be at least 1.")
        self.interval = interval
        self.original_data = None

    def difference(self, X):
        """
        Apply differencing to the input time series.

        Parameters
        ----------
        X : array-like
            The original time series data.

        Returns
        -------
        np.ndarray
            The differenced time series of length (len(X) - interval).
        """
        X = np.asarray(X)
        if X.ndim != 1:
            raise ValueError("Input must be a one-dimensional array.")
        self.original_data = X.copy()
        return np.array([X[i] - X[i - self.interval] for i in range(self.interval, len(X))])

    def inverse_difference(self, diff_data):
        """
        Reverse the differencing transformation using the stored original data.

        Parameters
        ----------
        diff_data : array-like
            The differenced data to invert.

        Returns
        -------
        np.ndarray
            The reconstructed original data (excluding the first `interval` values).

        Raises
        ------
        ValueError
            If the original data is not available.
        """
        if self.original_data is None:
            raise ValueError("Original data is required for inversion. Call difference() first.")
        diff_data = np.asarray(diff_data)
        return np.array([
            diff_data[i - self.interval] + self.original_data[i - self.interval]
            for i in range(self.interval, len(self.original_data))
        ])


class FeatureEngineering:
    """
    A class for performing custom feature engineering on numeric datasets.
    """

    def __init__(self):
        """
        Initialize the FeatureEngineering class.

        Currently, this class has no parameters but can be extended in the future.
        """
        pass

    def create_threshold_binary_features(self, X, threshold):
        """
        Add binary indicator columns to mark values below a given threshold.
        Each original column is followed by a new column indicating whether
        each value is below the threshold (1 if True, 0 otherwise).

        Parameters
        ----------
        X : numpy.ndarray
            The input 2D matrix of shape (n_samples, n_features).

        threshold : float
            The threshold value used to determine binary flags.

        Returns
        -------
        numpy.ndarray
            A new 2D matrix of shape (n_samples, 2 * n_features),
            where each original column is followed by its binary indicator column.

        Raises
        ------
        ValueError
            If `X` is not a NumPy array or not 2D.
            If `threshold` is not a numeric type.
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("Input X should be a NumPy array.")
        if X.ndim != 2:
            raise ValueError("Input X must be a 2D array.")
        if not isinstance(threshold, (int, float)):
            raise ValueError("Threshold should be a numeric value.")

        # Create a new matrix to hold original and new binary columns
        X_new = np.zeros((X.shape[0], X.shape[1] * 2), dtype=X.dtype)

        for idx in range(X.shape[1]):
            feature_values = X[:, idx]
            indicator_column = (feature_values < threshold).astype(int)
            X_new[:, idx * 2] = feature_values
            X_new[:, idx * 2 + 1] = indicator_column

        return X_new


class DataTransformer(BaseEstimator, TransformerMixin):
    """
    The class is used to transform data using different scaling techniques.

    Parameters
    ----------
    scaling_methods : str, tuple, list, or np.ndarray
        The name of the scaler you want to use. Supported scaler names are: 'standard', 'minmax', 'max-abs',
        'log1p', 'loge', 'sqrt', 'sinh-arc-sinh', 'robust', 'box-cox', 'yeo-johnson'.

    list_dict_paras : dict or list of dict
        The parameters for the scaler. If you have only one scaler, please use a dict. Otherwise, please use a list of dict.
    """

    SUPPORTED_SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler, "max-abs": MaxAbsScaler,
                         "log1p": Log1pScaler, "loge": LogeScaler, "sqrt": SqrtScaler,
                         "sinh-arc-sinh": SinhArcSinhScaler, "robust": RobustScaler,
                         "box-cox": BoxCoxScaler, "yeo-johnson": YeoJohnsonScaler}

    def __init__(self, scaling_methods=('standard', ), list_dict_paras=None):
        """
        Initialize the DataTransformer.

        Parameters
        ----------
        scaling_methods : str or list/tuple of str
            One or more scaling methods to apply in sequence.
            Must be keys in SUPPORTED_SCALERS.

        list_dict_paras : dict or list of dict, optional
            Parameters for each scaler. If only one method is provided,
            a single dict is expected. If multiple methods are provided,
            a list of parameter dictionaries should be given.
        """
        if isinstance(scaling_methods, str):
            if list_dict_paras is None:
                self.list_dict_paras = [{}]
            elif isinstance(list_dict_paras, dict):
                self.list_dict_paras = [list_dict_paras]
            else:
                raise TypeError("Expected a single dict for list_dict_paras when using one scaling method.")
            self.scaling_methods = [scaling_methods]
        elif isinstance(scaling_methods, (list, tuple, np.ndarray)):
            if list_dict_paras is None:
                self.list_dict_paras = [{} for _ in range(len(scaling_methods))]
            elif isinstance(list_dict_paras, (list, tuple, np.ndarray)):
                self.list_dict_paras = list(list_dict_paras)
            else:
                raise TypeError("list_dict_paras should be a list/tuple of dicts when using multiple scaling methods.")
            self.scaling_methods = list(scaling_methods)
        else:
            raise TypeError("scaling_methods must be a str, list, tuple, or np.ndarray")

        self.scalers = [self._get_scaler(technique, paras) for (technique, paras) in
                        zip(self.scaling_methods, self.list_dict_paras)]
        self.size_input_ = None

    @staticmethod
    def _ensure_2d(X):
        if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
            X = X.values
        else:
            X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)    # convert (n,) to (n, 1)
        elif X.ndim != 2:
            raise ValueError(f"Input X must be 1D or 2D, but got shape {X.shape}")
        return X

    def _get_scaler(self, technique, paras):
        if technique in self.SUPPORTED_SCALERS.keys():
            if not isinstance(paras, dict):
                paras = {}
            return self.SUPPORTED_SCALERS[technique](**paras)
        else:
            raise ValueError(f"Unsupported scaling technique: '{technique}'. Supported techniques: {list(self.SUPPORTED_SCALERS)}")

    def fit(self, X, y=None):
        """
        Fit the sequence of scalers on the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        y : Ignored
            Not used, exists for compatibility with sklearn's pipeline.

        Returns
        -------
        self : object
            Fitted transformer.
        """
        X = self._ensure_2d(X)
        self.size_input_ = X.shape[1]
        for idx, _ in enumerate(self.scalers):
            X = self.scalers[idx].fit_transform(X)
        return self

    def transform(self, X):
        """
        Transform the input data using the sequence of fitted scalers.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to transform.

        Returns
        -------
        X_transformed : array-like
            Transformed data.
        """
        X = self._ensure_2d(X)
        if hasattr(self, 'size_input_') and X.shape[1] != self.size_input_:
            raise ValueError("Input dimension does not match the one seen during fit.")
        for scaler in self.scalers:
            X = scaler.transform(X)
        return X

    def inverse_transform(self, X):
        """
        Reverse the transformations applied to the data.

        Parameters
        ----------
        X : array-like
            Transformed data to invert.

        Returns
        -------
        X_original : array-like
            Original data before transformation.
        """
        X = self._ensure_2d(X)
        for scaler in reversed(self.scalers):
            X = scaler.inverse_transform(X)
        return X


class Data:
    """
    The structure of our supported Data class

    Parameters
    ----------
    X : np.ndarray
        The features of your data

    y : np.ndarray
        The labels of your data
    """

    SUPPORT = {
        "scaler": list(DataTransformer.SUPPORTED_SCALERS.keys())
    }

    def __init__(self, X=None, y=None, name="Unknown"):
        self.X = self._to_numpy(X, is_X=True)
        self.y = self._to_numpy(y, is_X=False)
        self.name = name
        self.X_train, self.y_train, self.X_test, self.y_test = None, None, None, None

    @staticmethod
    def _to_numpy(data=None, is_X=True):
        if isinstance(data, pd.DataFrame):
            return data.values
        elif isinstance(data, pd.Series):
            return data.values.reshape(-1, 1) if is_X else data.values.ravel()
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                return data.reshape(-1, 1) if is_X else data
            return data
        else:
            raise TypeError(f"Input {'X' if is_X else 'y'} must be a numpy array or pandas DataFrame/Series.")

    @staticmethod
    def scale(X, scaling_methods=('standard', ), list_dict_paras=None):
        X = np.squeeze(np.asarray(X))
        if X.ndim == 1:
            X = np.reshape(X, (-1, 1))
        if X.ndim >= 3:
            raise TypeError(f"Invalid X data type. It should be array-like with shape (n samples, m features)")
        scaler = DataTransformer(scaling_methods=scaling_methods, list_dict_paras=list_dict_paras)
        data = scaler.fit_transform(X)
        return data, scaler

    @staticmethod
    def encode_label(y):
        y = np.squeeze(np.asarray(y))
        if y.ndim != 1:
            raise TypeError(f"Invalid y data type. It should be a vector / array-like with shape (n samples,)")
        scaler = LabelEncoder()
        data = scaler.fit_transform(y)
        return data, scaler

    def split_train_test(self, test_size=0.2, train_size=None,
                         random_state=41, shuffle=True, stratify=None, inplace=True):
        """
        The wrapper of the split_train_test function in scikit-learn library.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size,
                        train_size=train_size, random_state=random_state, shuffle=shuffle, stratify=stratify)
        if not inplace:
            return self.X_train, self.X_test, self.y_train, self.y_test

    def set_train_test(self, X_train=None, y_train=None, X_test=None, y_test=None):
        """
        Function use to set your own X_train, y_train, X_test, y_test in case you don't want to use our split function

        Parameters
        ----------
        X_train : np.ndarray
        y_train : np.ndarray
        X_test : np.ndarray
        y_test : np.ndarray
        """
        self.X_train = self._to_numpy(X_train, is_X=True)
        self.y_train = self._to_numpy(y_train, is_X=False)
        self.X_test = self._to_numpy(X_test, is_X=True)
        self.y_test = self._to_numpy(y_test, is_X=False)
        return self
