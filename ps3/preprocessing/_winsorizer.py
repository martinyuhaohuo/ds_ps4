import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# TODO: Write a simple Winsorizer transformer which takes a lower and upper quantile and cuts the
# data accordingly
class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile, upper_quantile):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        X_transformed = X.copy()
        X_transformed = np.asarray(X_transformed)
        self.lower_quantile_ = np.quantile(X_transformed, self.lower_quantile, axis=0)
        self.upper_quantile_ = np.quantile(X_transformed, self.upper_quantile, axis=0)
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed = np.asarray(X_transformed)
        X_transformed = np.where(X_transformed < self.lower_quantile_, self.lower_quantile_, X_transformed)
        X_transformed = np.where(X_transformed > self.upper_quantile_, self.upper_quantile_, X_transformed)
        return X_transformed

        
