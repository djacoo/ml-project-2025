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
from scipy.stats import skew

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

    def __init__(self, method: str = 'auto',skew_threshold: float = 0.5):
        self.method = method
        self.scalers= {}
        self.skew_threshold = skew_threshold

    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        self.scalers = {}
        for feature in NUMERICAL_FEATURES:
            if self.method == 'auto':
                self.scalers[feature] = self.skewness_scaler(feature,X=X)
            else:
                self.scalers[feature] = SCALERS[self.method]()
            self.scalers[feature].fit(X[feature].values.reshape(-1, 1))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_scaled = X.copy()
        for feature in NUMERICAL_FEATURES:
            X_scaled[feature] = self.scalers[feature].transform(X[feature].values.reshape(-1, 1))
        return X_scaled

    def skewness_scaler(self, feature: str,X: pd.DataFrame):
        skewness = abs(skew(X[feature]))
        if skewness > self.skew_threshold:
            return MinMaxScaler()
        else:
            return StandardScaler()

    def save(self, path: str) -> None:
        joblib.dump(self, path)
        
    @classmethod
    def load(cls, path: str):
        return joblib.load(path)
    