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
        scaling_method: Scaling method ('auto', 'standard', 'minmax')
        scaling_skew_threshold: Skewness threshold for auto scaling method
        pca_variance_threshold: Variance threshold for PCA (0.0 to 1.0)
        target_col: Name of target column to transform
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
    - Split group preservation and usage (fits only on 'train' if split_group exists)
    - Non-feature column preservation
    - Pipeline fitting and transformation
    
    **Important:** If `split_group` column exists in the data, the pipeline will
    automatically fit only on the 'train' split to prevent data leakage. The
    `transform()` method can then be applied to the entire dataset (train/val/test).
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
        
        If split_group column exists, fits only on 'train' split to prevent data leakage.
        Otherwise, fits on all provided data.
        
        Args:
            X: Input dataframe with features
            y: Target series (required for target encoding)
            
        Returns:
            self
        """
        # Check if split_group exists and extract train subset for fitting
        if self.split_group_col in X.columns:
            train_mask = X[self.split_group_col] == 'train'
            train_indices = X.index[train_mask]
            print(f"\nUsing 'split_group' column: Fitting pipeline on TRAIN subset only")
            print(f"  Train: {train_mask.sum():,} samples")
            print(f"  Val:   {(X[self.split_group_col] == 'val').sum():,} samples")
            print(f"  Test:  {(X[self.split_group_col] == 'test').sum():,} samples")
            
            # Use only train data for fitting
            X_fit = X.loc[train_indices].copy()
            if y is not None:
                y_fit = y.loc[train_indices]
            else:
                y_fit = None
        else:
            print("\nNo 'split_group' column found. Fitting on all data.")
            print("(Note: For reproducible splits, run 'scripts/split_data.py' first)")
            X_fit = X.copy()
            y_fit = y
        
        # Separate features from metadata
        metadata_cols = [self.split_group_col] + self.preserve_cols
        existing_metadata = [col for col in metadata_cols if col in X_fit.columns]
        
        # Keep target in X_features for MissingValueTransformer and OutlierRemovalTransformer
        # They need it to drop rows with missing target
        X_features = X_fit.drop(columns=existing_metadata, errors='ignore')
        
        # If target is not in X but y is provided, add it
        if self.target_col not in X_features.columns and y_fit is not None:
            X_features[self.target_col] = y_fit.values
            y_work = None  # y is now in X_features
        else:
            y_work = y_fit
        
        # Fit each step manually to handle row removal and y alignment
        X_current = X_features.copy()
        y_current = y_work
        
        for step_name, transformer in self.pipeline.steps:
            # Extract target from X_current if it exists (for encoding step)
            if self.target_col in X_current.columns and y_current is None:
                y_current = pd.Series(X_current[self.target_col].values, index=X_current.index)
            
            # Fit the transformer
            if step_name in ['encoding']:
                # Encoding needs y for target encoding
                if y_current is None:
                    raise ValueError(f"Target encoding requires y parameter, but y is None and {self.target_col} not in X")
                transformer.fit(X_current, y_current)
            else:
                transformer.fit(X_current, y_current)
            
            # Transform to get updated X (some transformers remove rows)
            X_transformed = transformer.transform(X_current)
            
            # If rows were removed, align y_current
            if len(X_transformed) < len(X_current):
                # Use indices from transformed dataframe to align y
                remaining_indices = X_transformed.index
                if y_current is not None:
                    y_current = y_current.loc[remaining_indices]
                # Also update target in X if it exists
                if self.target_col in X_transformed.columns:
                    if y_current is not None:
                        X_transformed[self.target_col] = y_current.values
            
            X_current = X_transformed
        
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
        # Separate features from metadata (but keep target for transformers that need it)
        metadata_cols = [self.split_group_col] + self.preserve_cols
        existing_metadata = [col for col in metadata_cols if col in X.columns]
        
        # Preserve target if it exists in X
        target_in_X = self.target_col in X.columns
        if target_in_X:
            existing_metadata.append(self.target_col)
        
        metadata = X[existing_metadata].copy() if existing_metadata else pd.DataFrame()
        X_features = X.drop(columns=existing_metadata, errors='ignore')
        
        # If target is not in X_features but y is provided, add it temporarily
        # (needed for MissingValueTransformer and OutlierRemovalTransformer)
        if self.target_col not in X_features.columns and y is not None:
            X_features[self.target_col] = y.values
        
        # Transform features
        X_transformed = self.pipeline.transform(X_features)
        
        # Remove target from transformed data if it was added temporarily
        if self.target_col in X_transformed.columns and not target_in_X:
            X_transformed = X_transformed.drop(columns=[self.target_col])
        
        # Add back metadata columns (aligned with X_transformed indices)
        if not metadata.empty:
            for col in metadata.columns:
                if col != self.target_col or target_in_X:  # Only add target if it was in original X
                    # Align metadata with X_transformed indices (some rows may have been removed)
                    aligned_metadata = metadata.loc[X_transformed.index, col]
                    X_transformed[col] = aligned_metadata.values
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X, y)
    
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
        """
        Load a saved pipeline from disk.
        
        Note: When loading, ensure that the 'src' directory is in Python path:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
            pipeline = PreprocessingPipeline.load('models/preprocessing_pipeline.joblib')
        """
        import joblib
        import sys
        from pathlib import Path
        
        # Ensure src is in path for module imports
        current_file = Path(__file__).resolve()
        src_dir = current_file.parent.parent
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        
        return joblib.load(path)
