"""
Outlier detection and removal transformers.

Provides scikit-learn compatible transformers for missing value handling and
outlier removal following domain-specific rules and statistical methods.
"""
import pandas as pd
from typing import Optional
from sklearn.base import BaseEstimator, TransformerMixin

from data.preprocessing import MissingValueHandler, OutlierHandler


class MissingValueTransformer(BaseEstimator, TransformerMixin):
    """
    Handles missing values through imputation and feature/row removal.
    
    Parameters
    ----------
    threshold_drop_feature : float, default=0.95
        Drop features with missing percentage above this threshold.
    target_col : str, default='nutriscore_grade'
        Target column name. Rows with missing target are dropped.
    
    Attributes
    ----------
    dropped_features_ : list[str]
        Features dropped due to high missing percentage.
    
    Notes
    -----
    Imputation strategy:
    - Numerical features: median imputation
    - additives_n: zero imputation
    - Categorical features: 'unknown' placeholder
    """
    
    def __init__(
        self, threshold_drop_feature: float = 0.95, target_col: str = 'nutriscore_grade'
    ):
        self.threshold_drop_feature = threshold_drop_feature
        self.target_col = target_col
        self.handler = MissingValueHandler(threshold_drop_feature=threshold_drop_feature)
        self.dropped_features_: Optional[list] = None
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'MissingValueTransformer':
        """
        Analyze missing value patterns.
        
        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Input features.
        y : pd.Series, optional
            Ignored. Present for API compatibility.
        
        Returns
        -------
        self : MissingValueTransformer
            Returns self for method chaining.
        """
        self.handler.analyze_missing_values(X)
        self.dropped_features_ = self.handler.dropped_features.copy()
        return self
    
    def transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Handle missing values through imputation and removal.
        
        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Input features to transform.
        y : pd.Series of shape (n_samples,), optional
            Target variable. Added to X if target_col not present.
        
        Returns
        -------
        X_processed : pd.DataFrame
            Features with missing values handled. Shape may differ due to
            row/column removal.
        """
        X_work = X.copy()
        if self.target_col not in X_work.columns and y is not None:
            X_work[self.target_col] = y.values
        
        return self.handler.handle_missing_values(X_work, target_col=self.target_col)
    
    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X, y)


class OutlierRemovalTransformer(BaseEstimator, TransformerMixin):
    """
    Removes outliers using domain rules and statistical methods.
    
    Parameters
    ----------
    target_col : str, default='nutriscore_grade'
        Target column name. Preserved during transformation.
    
    Attributes
    ----------
    rows_removed_ : int
        Number of rows removed during last transform call.
    
    Notes
    -----
    Removal strategy:
    - Domain rules: negative values, values > 100g per 100g (macros),
      energy > 3000 kcal, salt > 50g
    - Statistical: IQR method with 3×IQR threshold
    """
    
    def __init__(self, target_col: str = 'nutriscore_grade'):
        self.target_col = target_col
        self.handler = OutlierHandler()
        self.rows_removed_ = 0
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'OutlierRemovalTransformer':
        """
        Analyze outlier patterns.
        
        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Input features.
        y : pd.Series, optional
            Ignored. Present for API compatibility.
        
        Returns
        -------
        self : OutlierRemovalTransformer
            Returns self for method chaining.
        """
        self.handler.detect_outliers(X)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Remove outliers from data.
        
        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Input features to transform.
        
        Returns
        -------
        X_clean : pd.DataFrame
            Features with outliers removed. Shape (n_samples - n_removed, n_features).
        """
        initial_rows = len(X)
        df_clean = self.handler.remove_outliers(X, target_col=self.target_col)
        self.rows_removed_ = initial_rows - len(df_clean)
        return df_clean
    
    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
