import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Functions


class CustomScaler(BaseEstimator, TransformerMixin):
    """
    Custom scaler for time series data that standardises data (z-score normalisation) based on average and std of the
    entire curve. The class follows the structure of sklearn's TransformerMixin. Curves can be stacked horizontally in a data matrix,
    with each column representing a time point. A vartracker list is used to track which columns correspond to which variables.
    """

    def __init__(self):
        self.vartracker_ = None
        self.unique_vars_ = None
        self.mean_ = None
        self.std_ = None

    def fit(self, X, y=None, vartracker=None):
        """
        Fit the scaler to the data.

        Parameters:
        X (np.ndarray): Data matrix to fit the scaler to.
        y (np.ndarray): Optional target values (not used). Just here because of sklearn's TransformerMixin structure.
        vartracker (list): List of variable names for each data instance. If None, the entire data are treated as a single variable.

        Returns:
        self: Fitted scaler object.
        """

        # Get vartracker
        self.vartracker_ = vartracker
        if self.vartracker_ is None:
            self.mean_ = np.mean(X)
            self.std_ = np.std(X)
        else:
            self.vartracker_ = np.array(vartracker)
            self.unique_vars_ = np.unique(self.vartracker_)

            # Get mean and std for each key in datadict
            self.mean_ = {
                var: np.mean(X[:, self.vartracker_ == var])
                for var in np.unique(self.vartracker_)
            }
            self.std_ = {
                var: np.std(X[:, self.vartracker_ == var])
                for var in np.unique(self.vartracker_)
            }

        return self

    def transform(self, X):
        """
        Transform the data using the fitted scaler.

        Parameters:
        X (np.ndarray): Data matrix to transform.

        Returns:
        Xz (np.ndarray): Standardised data matrix.
        """

        Xz = np.zeros(X.shape)

        # Standardise data
        for var in self.unique_vars_:
            Xz[:, self.vartracker_ == var] = (
                X[:, self.vartracker_ == var] - self.mean_[var]
            ) / self.std_[var]

        return Xz

    def fit_transform(self, X, y=None, vartracker=None):
        """
        Fit the scaler to the data and transform it.

        Parameters:
        X (np.ndarray): Data matrix to fit and transform.
        y (np.ndarray): Optional target values (not used). Just here because of sklearn's TransformerMixin structure.
        vartracker (list): List tracking the variable name for each data instance. If None, the entire data is treated as a single variable.

        Returns:
        Xz (np.ndarray): Standardised data matrix.
        """

        self.fit(X, y, vartracker)
        Xz = self.transform(X)

        return Xz

    def inverse_transform(self, Xz, y=None):
        """
        Inverse transform the standardised data back to the original scale.

        Parameters:
        Xz (np.ndarray): Standardised data matrix to inverse transform.
        y (np.ndarray): Optional target values (not used). Just here because of sklearn's TransformerMixin structure.

        Returns:
        X (np.ndarray): Data matrix in the original scale.
        """

        X = np.zeros(Xz.shape)

        # Standardise data
        for var in self.unique_vars_:
            X[:, self.vartracker_ == var] = (
                Xz[:, self.vartracker_ == var] * self.std_[var] + self.mean_[var]
            )

        return X
