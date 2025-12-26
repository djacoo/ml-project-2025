"""
Feature scaling module for Nutri-Score prediction
This implementation follows the scikit-learn Transformer API, allowing 
integration into pipelines and cross-validation
Supports StandardScaler, MinMaxScaler, and RobustScaler.
"""
import joblib
import pandas as pd
from pathlib import Path
from typing import List, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# numerical features for Nutri-Score
NUMERICAL_FEATURES = [
    'energy-kcal_100g',
    'fat_100g',
    'saturated-fat_100g',
    'carbohydrates_100g',
    'sugars_100g',
    'fiber_100g',
    'proteins_100g',
    'salt_100g',
    'additives_n'
]

SCALERS = {
    'standard': StandardScaler,
    'minmax': MinMaxScaler,
    'robust': RobustScaler
}

# Inheriting from BaseEstimator and TransformerMixin for:
# - Automatic fit_transform implementation
# - Compatibility with sklearn Pipeline objects
# - Access to get_params/set_params for hyperparameter optimization (e.g., GridSearchCV)
class FeatureScaler(BaseEstimator, TransformerMixin):
    """
    Wrapper for Nutri-Score feature scaling
    """

    def __init__(
        self,
        method: str = 'robust',
        features: Optional[List[str]] = None
    ):
        self.method = method
        self.features = features or NUMERICAL_FEATURES
        self._init_scaler()

    def _init_scaler(self):
        if self.method not in SCALERS:
            raise ValueError(f"Unknown scaling method: {self.method}")
        self.scaler_ = SCALERS[self.method]()

    def fit(self, X, y=None):
        
        self._check_features(X)
        self.scaler_.fit(X[self.features])
        return self

    def transform(self, X):
        self._check_features(X)
        
        X_scaled = X.copy()
        X_scaled[self.features] = self.scaler_.transform(X[self.features])
        return X_scaled

    def _check_features(self, X):
        
        for feature in self.features:
            if feature not in X.columns:
                raise ValueError(f"Missing feature in DataFrame: {feature}")
                
    def save(self, path: str):
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str):
        
        return joblib.load(path)