"""
Complete preprocessing pipeline for Nutri-Score prediction.

Provides scikit-learn Pipeline combining missing value handling, outlier removal,
feature encoding, scaling, and PCA dimensionality reduction.
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

from features.outlier_removal import MissingValueTransformer, OutlierRemovalTransformer
from features.encoding import FeatureEncoder
from features.feature_engineering import FeatureEngineer
from features.scaling import FeatureScaler
from features.dimensionality_reduction import FeatureReducer


def create_preprocessing_pipeline(
    missing_threshold: float = 0.95,
    top_n_countries: int = 15,
    scaling_method: str = 'auto',
    scaling_skew_threshold: float = 1.0,
    pca_variance_threshold: float = 0.95,
    target_col: str = 'nutriscore_grade',
    include_feature_engineering: bool = True,
    feature_engineering_kwargs: Optional[dict] = None,
    remove_statistical_outliers: bool = False,
    include_pca: bool = True
) -> Pipeline:
    """
    Create preprocessing pipeline with configurable steps.
    
    Parameters
    ----------
    missing_threshold : float, default=0.95
        Threshold for dropping features with high missing percentage.
    top_n_countries : int, default=15
        Number of top countries to retain for multi-label encoding.
    scaling_method : {'auto', 'standard', 'minmax', 'robust'}, default='auto'
        Scaling method. 'auto' selects based on feature skewness.
    scaling_skew_threshold : float, default=1.0
        Skewness threshold for automatic scaler selection.
    pca_variance_threshold : float, default=0.95
        Minimum cumulative explained variance for PCA component selection.
    target_col : str, default='nutriscore_grade'
        Target column name.
    include_feature_engineering : bool, default=True
        Whether to include feature engineering step (derived features).
    feature_engineering_kwargs : dict, optional
        Keyword arguments to pass to FeatureEngineer (e.g., 
        {'add_ratios': True, 'add_energy_density': True}).
        If None, uses FeatureEngineer defaults.
    remove_statistical_outliers : bool, default=False
        Whether to remove statistical outliers (3×IQR method).
        Default is False because statistical outliers in nutritional data
        often represent valid product categories, not measurement errors.
    include_pca : bool, default=True
        Whether to include PCA dimensionality reduction step.
    
    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        Fitted pipeline with steps: missing_values, outlier_removal,
        encoding, [feature_engineering], scaling, [pca].
    """
    steps = [
        ('missing_values', MissingValueTransformer(
            threshold_drop_feature=missing_threshold,
            target_col=target_col
        )),
        ('outlier_removal', OutlierRemovalTransformer(
            target_col=target_col,
            remove_statistical_outliers=remove_statistical_outliers
        )),
        ('encoding', FeatureEncoder(top_n_countries=top_n_countries)),
    ]
    
    # Add feature engineering step if enabled
    if include_feature_engineering:
        fe_kwargs = feature_engineering_kwargs or {}
        steps.append(('feature_engineering', FeatureEngineer(**fe_kwargs)))
    
    steps.append(('scaling', FeatureScaler(method=scaling_method, skew_threshold=scaling_skew_threshold)))
    
    if include_pca:
        steps.append(('pca', FeatureReducer(variance_threshold=pca_variance_threshold)))
    
    return Pipeline(steps)


class PreprocessingPipeline:
    """
    High-level preprocessing pipeline wrapper with metadata handling.
    
    Automatically fits on 'train' split when split_group column exists to prevent
    data leakage. Preserves metadata columns (target, split_group, product_name, etc.)
    through transformation.
    
    Parameters
    ----------
    missing_threshold : float, default=0.95
        Threshold for dropping features with high missing percentage.
    top_n_countries : int, default=15
        Number of top countries to retain for encoding.
    scaling_method : {'auto', 'standard', 'minmax', 'robust'}, default='auto'
        Scaling method.
    scaling_skew_threshold : float, default=1.0
        Skewness threshold for automatic scaler selection.
    pca_variance_threshold : float, default=0.95
        Minimum cumulative explained variance for PCA.
    target_col : str, default='nutriscore_grade'
        Target column name.
    split_group_col : str, default='split_group'
        Split group column name (train/val/test).
    preserve_cols : list[str], optional
        Additional columns to preserve (e.g., ['product_name', 'brands']).
    include_feature_engineering : bool, default=True
        Whether to include feature engineering step (derived features).
    feature_engineering_kwargs : dict, optional
        Keyword arguments to pass to FeatureEngineer.
    remove_statistical_outliers : bool, default=False
        Whether to remove statistical outliers (3×IQR method).
        Default is False because statistical outliers in nutritional data
        often represent valid product categories, not measurement errors.
    include_pca : bool, default=True
        Whether to include PCA step.
    
    Attributes
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Fitted preprocessing pipeline.
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
        include_feature_engineering: bool = True,
        feature_engineering_kwargs: Optional[dict] = None,
        remove_statistical_outliers: bool = False,
        include_pca: bool = True
    ):
        self.target_col = target_col
        self.split_group_col = split_group_col
        self.preserve_cols = preserve_cols or []
        
        self.pipeline = create_preprocessing_pipeline(
            missing_threshold=missing_threshold,
            top_n_countries=top_n_countries,
            scaling_method=scaling_method,
            scaling_skew_threshold=scaling_skew_threshold,
            pca_variance_threshold=pca_variance_threshold,
            target_col=target_col,
            include_feature_engineering=include_feature_engineering,
            feature_engineering_kwargs=feature_engineering_kwargs,
            remove_statistical_outliers=remove_statistical_outliers,
            include_pca=include_pca
        )
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'PreprocessingPipeline':
        """
        Fit pipeline on training data.
        
        If split_group column exists, fits only on 'train' split to prevent
        data leakage. Otherwise fits on all provided data.
        
        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Input features. May contain target and metadata columns.
        y : pd.Series of shape (n_samples,), optional
            Target variable. Required for target encoding.
        
        Returns
        -------
        self : PreprocessingPipeline
            Returns self for method chaining.
        
        Raises
        ------
        ValueError
            If target encoding is required but y is None and target_col not in X.
        UserWarning
            If split_group column is expected but not found in data.
        
        Notes
        -----
        The pipeline automatically extracts the train split if split_group column
        exists, ensuring transformers are fitted only on training data to prevent
        data leakage.
        """
        # Step 1: Extract train split if split_group exists
        X_fit, y_fit = self._extract_train_split(X, y)
        
        # Step 2: Prepare features and extract target
        X_features, y_target = self._prepare_features_and_target(X_fit, y_fit)
        
        # Step 3: Fit pipeline steps
        X_current = X_features.copy()
        
        # Add target to X_current for transformers that need it (missing_values, outlier_removal)
        if y_target is not None and self.target_col not in X_current.columns:
            X_current[self.target_col] = y_target.values
        
        for step_name, transformer in self.pipeline.steps:
            # Validate target availability for encoding step
            if step_name == 'encoding' and y_target is None:
                raise ValueError(
                    f"Target encoding requires y parameter, but y is None and "
                    f"{self.target_col} not in X"
                )
            
            # Ensure target is available for encoding step
            # If target is in X_current, extract it to y_target for consistency
            if step_name == 'encoding' and self.target_col in X_current.columns:
                if y_target is None:
                    y_target = pd.Series(X_current[self.target_col].values, index=X_current.index)
                # Ensure indices are aligned
                elif not X_current.index.equals(y_target.index):
                    y_target = y_target.reindex(X_current.index)
            
            # Fit transformer
            transformer.fit(X_current, y_target)
            
            # Transform data
            X_transformed = transformer.transform(X_current)
            
            # Remove target from transformed data (keep it separate)
            if self.target_col in X_transformed.columns:
                X_transformed = X_transformed.drop(columns=[self.target_col])
            
            # Align target if rows were removed
            if len(X_transformed) < len(X_current) and y_target is not None:
                y_target = y_target.loc[X_transformed.index]
            
            # Add target back to X_transformed for next transformer if needed
            # Use aligned indices to ensure consistency
            if y_target is not None and self.target_col not in X_transformed.columns:
                y_aligned = y_target.reindex(X_transformed.index)
                X_transformed[self.target_col] = y_aligned.values
            
            X_current = X_transformed
        
        return self
    
    def _extract_train_split(
        self, X: pd.DataFrame, y: Optional[pd.Series]
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Extract train split if split_group exists, else return all data.
        
        Raises a warning if split_group column is expected but not found,
        as this may cause data leakage if test data is included.
        """
        if self.split_group_col in X.columns:
            train_mask = X[self.split_group_col] == 'train'
            train_indices = X.index[train_mask]
            return X.loc[train_indices].copy(), y.loc[train_indices] if y is not None else None
        
        # Warning if split_group column not found
        import warnings
        warnings.warn(
            f"'{self.split_group_col}' column not found in data. "
            f"Fitting on all data. This may cause data leakage if test/validation data is included. "
            f"For proper train/val/test split, ensure '{self.split_group_col}' column exists.",
            UserWarning,
            stacklevel=2
        )
        return X.copy(), y
    
    def _prepare_features_and_target(
        self, X: pd.DataFrame, y: Optional[pd.Series]
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Separate features from metadata and extract target.
        
        Strategy:
        1. Remove metadata columns (split_group, preserve_cols)
        2. Extract target from X if present, otherwise use y parameter
        3. Remove target from X_features (keep it separate)
        
        Returns
        -------
        X_features : pd.DataFrame
            Features without metadata and target columns.
        y_target : pd.Series or None
            Target variable extracted from X or provided as y parameter.
        """
        # Remove metadata columns
        metadata_cols = [self.split_group_col] + self.preserve_cols
        existing_metadata = [col for col in metadata_cols if col in X.columns]
        X_features = X.drop(columns=existing_metadata, errors='ignore')
        
        # Extract target: prefer X[target_col] if present, otherwise use y parameter
        if self.target_col in X_features.columns:
            y_target = pd.Series(X_features[self.target_col].values, index=X_features.index)
            X_features = X_features.drop(columns=[self.target_col])
        elif y is not None:
            y_target = y
        else:
            y_target = None
        
        return X_features, y_target
    
    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Transform data using fitted pipeline.
        
        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Input features to transform. May contain target and metadata columns.
        y : pd.Series of shape (n_samples,), optional
            Target variable. Used only if target_col not in X.
        
        Returns
        -------
        X_transformed : pd.DataFrame
            Transformed features with metadata columns preserved.
            Shape may differ from input due to row removal in preprocessing steps.
        
        Raises
        ------
        ValueError
            If pipeline has not been fitted (call fit() first).
        
        Notes
        -----
        Metadata columns (target, split_group, preserve_cols) are preserved
        through the transformation and added back to the output.
        """
        # Step 1: Separate metadata and features
        metadata_cols = [self.split_group_col] + self.preserve_cols
        existing_metadata = [col for col in metadata_cols if col in X.columns]
        
        # Step 2: Extract target if present in X, otherwise use y parameter
        target_in_X = self.target_col in X.columns
        if target_in_X:
            existing_metadata.append(self.target_col)
            y_target = pd.Series(X[self.target_col].values, index=X.index)
        else:
            y_target = y
        
        # Step 3: Remove metadata and target from features
        metadata = X[existing_metadata].copy() if existing_metadata else pd.DataFrame()
        X_features = X.drop(columns=existing_metadata, errors='ignore')
        
        # Step 4: Add target to features if needed (for transformers that require it)
        if self.target_col not in X_features.columns and y_target is not None:
            X_features[self.target_col] = y_target.values
        
        # Step 5: Transform features
        X_transformed = self.pipeline.transform(X_features)
        
        # Step 6: Remove target from transformed features (keep it separate)
        if self.target_col in X_transformed.columns:
            X_transformed = X_transformed.drop(columns=[self.target_col])
        
        # Step 7: Restore metadata columns (aligned with transformed indices)
        if not metadata.empty:
            for col in metadata.columns:
                if col != self.target_col or target_in_X:
                    aligned_metadata = metadata.reindex(X_transformed.index)
                    X_transformed[col] = aligned_metadata[col].values
        
        return X_transformed
    
    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Fit pipeline and transform in one step."""
        self.fit(X, y)
        return self.transform(X, y)
    
    def get_pipeline_steps(self) -> List[str]:
        """Return list of pipeline step names."""
        return [name for name, _ in self.pipeline.steps]
    
    def get_transformer(self, step_name: str) -> Optional[BaseEstimator]:
        """Return transformer for given step name."""
        return self.pipeline.named_steps.get(step_name)
    
    def save(self, path: str) -> None:
        """Save fitted pipeline to disk."""
        import joblib
        joblib.dump(self, path)
    
    @classmethod
    def load(cls, path: str) -> 'PreprocessingPipeline':
        """
        Load saved pipeline from disk.
        
        Parameters
        ----------
        path : str
            Path to saved pipeline file.
        
        Returns
        -------
        pipeline : PreprocessingPipeline
            Loaded pipeline instance.
        
        Notes
        -----
        Automatically ensures 'src' directory is in Python path for module imports.
        """
        import joblib
        import sys
        from pathlib import Path
        
        current_file = Path(__file__).resolve()
        src_dir = current_file.parent.parent
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        
        return joblib.load(path)
