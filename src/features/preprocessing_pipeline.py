"""
Complete preprocessing pipeline for Nutri-Score prediction.

This module provides a scikit-learn Pipeline that combines all preprocessing steps:
1. Missing value handling
2. Outlier removal
3. Feature encoding
4. Feature scaling
5. PCA dimensionality reduction

All transformers follow the scikit-learn API (BaseEstimator, TransformerMixin).
"""

import pandas as pd
import numpy as np
from typing import Optional, List
from sklearn.pipeline import Pipeline

from features.outlier_removal import MissingValueTransformer, OutlierRemovalTransformer
from features.encoding import FeatureEncoder
from features.scaling import FeatureScaler
from features.dimensionality_reduction import FeatureReducer


def create_preprocessing_pipeline(
    missing_threshold: float = 0.95,
    top_n_countries: int = 15,
    scaling_method: str = 'auto',
    scaling_skew_threshold: float = 1.0,
    pca_variance_threshold: float = 0.95,
    target_col: str = 'nutriscore_grade',
    include_pca: bool = True
) -> Pipeline:
    """
    Create a complete preprocessing pipeline for Nutri-Score prediction.
    
    Pipeline steps:
    1. Missing value handling (imputation, dropping high-missing features)
    2. Outlier removal (domain rules, statistical outliers)
    3. Feature encoding (categorical features: one-hot, target encoding)
    4. Feature scaling (standardization/normalization)
    5. PCA dimensionality reduction (optional)
    
    Args:
        missing_threshold: Threshold for dropping features with high missing percentage
        top_n_countries: Number of top countries to keep for encoding
        scaling_method: Scaling method ('auto', 'standard', 'minmax', 'robust')
        scaling_skew_threshold: Skewness threshold for auto scaling method
        pca_variance_threshold: Variance threshold for PCA (0.0 to 1.0)
        target_col: Name of target column
        include_pca: Whether to include PCA in the pipeline
        
    Returns:
        sklearn Pipeline object with all preprocessing steps
        
    Example:
        >>> pipeline = create_preprocessing_pipeline()
        >>> X_processed = pipeline.fit_transform(X_train, y_train)
        >>> X_test_processed = pipeline.transform(X_test)
    """
    steps = [
        ('missing_values', MissingValueTransformer(
            threshold_drop_feature=missing_threshold,
            target_col=target_col
        )),
        ('outlier_removal', OutlierRemovalTransformer(target_col=target_col)),
        ('encoding', FeatureEncoder(top_n_countries=top_n_countries)),
        ('scaling', FeatureScaler(method=scaling_method, skew_threshold=scaling_skew_threshold)),
    ]
    
    if include_pca:
        steps.append(('pca', FeatureReducer(variance_threshold=pca_variance_threshold)))
    
    return Pipeline(steps)


class PreprocessingPipeline:
    """
    High-level wrapper for the preprocessing pipeline with additional utilities.
    
    This class provides a convenient interface for preprocessing that handles:
    - Target column separation
    - Split group preservation
    - Non-feature column preservation
    - Pipeline fitting and transformation
    """
    
    def __init__(
        self,
        missing_threshold: float = 0.95,
        top_n_countries: int = 15,
        scaling_method: str = 'auto',
        scaling_skew_threshold: float = 1.0,
        pca_variance_threshold: float = 0.95,
        target_col: str = 'nutriscore_grade',
        split_group_col: str = 'split_group',
        preserve_cols: Optional[List[str]] = None,
        include_pca: bool = True
    ):
        """
        Initialize the preprocessing pipeline.
        
        Args:
            missing_threshold: Threshold for dropping features with high missing percentage
            top_n_countries: Number of top countries to keep for encoding
            scaling_method: Scaling method ('auto', 'standard', 'minmax', 'robust')
            scaling_skew_threshold: Skewness threshold for auto scaling method
            pca_variance_threshold: Variance threshold for PCA (0.0 to 1.0)
            target_col: Name of target column
            split_group_col: Name of split group column (train/val/test)
            preserve_cols: List of columns to preserve (e.g., ['product_name', 'brands'])
            include_pca: Whether to include PCA in the pipeline
        """
        self.target_col = target_col
        self.split_group_col = split_group_col
        self.preserve_cols = preserve_cols or []
        
        # Create the sklearn pipeline
        self.pipeline = create_preprocessing_pipeline(
            missing_threshold=missing_threshold,
            top_n_countries=top_n_countries,
            scaling_method=scaling_method,
            scaling_skew_threshold=scaling_skew_threshold,
            pca_variance_threshold=pca_variance_threshold,
            target_col=target_col,
            include_pca=include_pca
        )
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit the preprocessing pipeline on training data.
        
        Args:
            X: Input dataframe with features
            y: Target series (required for target encoding)
            
        Returns:
            self
        """
        # Separate features from metadata
        metadata_cols = [self.target_col, self.split_group_col] + self.preserve_cols
        existing_metadata = [col for col in metadata_cols if col in X.columns]
        
        X_features = X.drop(columns=existing_metadata, errors='ignore')
        
        # Fit pipeline (encoding step needs y for target encoding)
        self.pipeline.fit(X_features, y)
        
        return self
    
    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Transform data using the fitted pipeline.
        
        Args:
            X: Input dataframe with features
            y: Target series (optional, for consistency)
            
        Returns:
            Transformed dataframe with preserved columns added back
        """
        # Separate features from metadata
        metadata_cols = [self.target_col, self.split_group_col] + self.preserve_cols
        existing_metadata = [col for col in metadata_cols if col in X.columns]
        
        metadata = X[existing_metadata].copy() if existing_metadata else pd.DataFrame()
        X_features = X.drop(columns=existing_metadata, errors='ignore')
        
        # Transform features
        X_transformed = self.pipeline.transform(X_features)
        
        # Add back metadata columns
        if not metadata.empty:
            for col in metadata.columns:
                X_transformed[col] = metadata[col].values
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X, y)
    
    def get_pipeline_steps(self) -> List[str]:
        """Get list of pipeline step names."""
        return [name for name, _ in self.pipeline.steps]
    
    def get_transformer(self, step_name: str):
        """Get a specific transformer from the pipeline by name."""
        return self.pipeline.named_steps.get(step_name)
    
    def save(self, path: str):
        """Save the pipeline to disk."""
        import joblib
        joblib.dump(self, path)
    
    @classmethod
    def load(cls, path: str):
        """Load a saved pipeline from disk."""
        import joblib
        return joblib.load(path)
