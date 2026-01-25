"""
Feature scaling for Nutri-Score prediction.

Implements automatic scaler selection based on skewness and manual method selection
following scikit-learn Transformer API.
"""
import joblib
import pandas as pd
import numpy as np
from typing import Optional, Dict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy.stats import skew


SCALERS = {
    'standard': StandardScaler,
    'minmax': MinMaxScaler,
    'robust': RobustScaler
}

METADATA_COLS = ['nutriscore_grade', 'split_group', 'product_name', 'brands', 'code']


class FeatureScaler(BaseEstimator, TransformerMixin):
    """
    Scales numerical features using configurable scaling strategies.
    
    Parameters
    ----------
    method : {'auto', 'standard', 'minmax', 'robust'}, default='auto'
        Scaling method. If 'auto', selects scaler based on feature skewness.
    skew_threshold : float, default=1.0
        Skewness threshold for automatic method selection. Features with
        |skewness| > threshold use MinMaxScaler, else StandardScaler.
    
    Attributes
    ----------
    scalers_ : dict[str, sklearn.preprocessing.Transformer]
        Fitted scalers for each numerical feature.
    
    Notes
    -----
    Automatically identifies and scales all numerical columns, excluding
    metadata columns (target, split_group, product_name, brands, code).
    """
    
    def __init__(self, method: str = 'auto', skew_threshold: float = 1.0):
        self.method = method
        self.skew_threshold = skew_threshold
        self.scalers_: Dict[str, BaseEstimator] = {}
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureScaler':
        """
        Fit scalers for all numerical features.
        
        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Input features containing numerical columns.
        y : pd.Series, optional
            Ignored. Present for API compatibility.
        
        Returns
        -------
        self : FeatureScaler
            Returns self for method chaining.
        """
        self.scalers_ = {}
        
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        features_to_scale = [col for col in numerical_cols if col not in METADATA_COLS]
        
        for feature in features_to_scale:
            if self.method == 'auto':
                scaler = self._select_scaler_by_skewness(X[feature])
            else:
                scaler = SCALERS[self.method]()
            
            scaler.fit(X[feature].values.reshape(-1, 1))
            self.scalers_[feature] = scaler
        
        return self
    
    def _select_scaler_by_skewness(self, X_feature: pd.Series) -> BaseEstimator:
        """
        Select scaler based on feature skewness.
        
        Parameters
        ----------
        X_feature : pd.Series of shape (n_samples,)
            Feature values to analyze.
        
        Returns
        -------
        scaler : sklearn.preprocessing.Transformer
            MinMaxScaler if |skewness| > threshold, else StandardScaler.
        """
        feature_skewness = abs(skew(X_feature.dropna()))
        return MinMaxScaler() if feature_skewness > self.skew_threshold else StandardScaler()
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform numerical features using fitted scalers.
        
        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Input features to transform.
        
        Returns
        -------
        X_scaled : pd.DataFrame of shape (n_samples, n_features)
            Scaled features. Non-numerical and metadata columns preserved.
        """
        X_scaled = X.copy()
        
        for feature, scaler in self.scalers_.items():
            if feature in X_scaled.columns:
                X_scaled[feature] = scaler.transform(
                    X_scaled[feature].values.reshape(-1, 1)
                ).flatten()
        
        return X_scaled
    
    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Fit scalers and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def save(self, path: str) -> None:
        """Save fitted scaler to disk."""
        joblib.dump(self, path)
    
    @classmethod
    def load(cls, path: str) -> 'FeatureScaler':
        """Load fitted scaler from disk."""
        return joblib.load(path)
