"""
Outlier removal transformers for scikit-learn Pipeline compatibility.

This module provides scikit-learn compatible transformers for:
- Missing value handling
- Outlier detection and removal

All transformers follow the scikit-learn API (BaseEstimator, TransformerMixin).
"""

import pandas as pd
from typing import Optional
from sklearn.base import BaseEstimator, TransformerMixin

from data.preprocessing import MissingValueHandler, OutlierHandler


class MissingValueTransformer(BaseEstimator, TransformerMixin):
    """
    Wrapper for MissingValueHandler to make it compatible with scikit-learn Pipeline.
    
    This transformer handles missing values by:
    - Dropping features with >95% missing data (configurable threshold)
    - Dropping rows with missing target variable
    - Imputing numerical features with median
    - Imputing additives_n with 0
    - Handling categorical features with 'unknown' placeholder
    
    Parameters:
    -----------
    threshold_drop_feature : float, default=0.95
        Drop features with missing percentage above this threshold
    target_col : str, default='nutriscore_grade'
        Name of the target column
    """
    
    def __init__(self, threshold_drop_feature: float = 0.95, target_col: str = 'nutriscore_grade'):
        self.threshold_drop_feature = threshold_drop_feature
        self.target_col = target_col
        self.handler = MissingValueHandler(threshold_drop_feature=threshold_drop_feature)
        self.dropped_features_ = None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit the missing value handler (analyzes missing patterns).
        
        Args:
            X: Input dataframe
            y: Target series (optional, used to identify target column)
            
        Returns:
            self
        """
        # Analyze missing values
        self.handler.analyze_missing_values(X)
        self.dropped_features_ = self.handler.dropped_features.copy()
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by handling missing values.
        
        Args:
            X: Input dataframe
            
        Returns:
            Dataframe with missing values handled
        """
        return self.handler.handle_missing_values(X, target_col=self.target_col)
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


class OutlierRemovalTransformer(BaseEstimator, TransformerMixin):
    """
    Wrapper for OutlierHandler to make it compatible with scikit-learn Pipeline.
    
    This transformer removes outliers by:
    - Removing negative values (nutritional values cannot be negative)
    - Applying domain-specific valid ranges (e.g., fat <= 100g per 100g)
    - Removing statistical outliers using IQR method (3×IQR threshold)
    
    Parameters:
    -----------
    target_col : str, default='nutriscore_grade'
        Name of the target column
    """
    
    def __init__(self, target_col: str = 'nutriscore_grade'):
        self.target_col = target_col
        self.handler = OutlierHandler()
        self.rows_removed_ = 0
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit the outlier handler (analyzes outliers).
        
        Args:
            X: Input dataframe
            y: Target series (optional, not used)
            
        Returns:
            self
        """
        # Analyze outliers
        self.handler.detect_outliers(X)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by removing outliers.
        
        Args:
            X: Input dataframe
            
        Returns:
            Dataframe with outliers removed
        """
        initial_rows = len(X)
        df_clean = self.handler.remove_outliers(X, target_col=self.target_col)
        self.rows_removed_ = initial_rows - len(df_clean)
        return df_clean
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
