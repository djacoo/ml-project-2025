"""
Dimensionality reduction module for Nutri-Score prediction using PCA.
This implementation follows the scikit-learn Transformer API, allowing 
integration into pipelines and cross-validation.
"""
import joblib
import pandas as pd
import numpy as np
from typing import Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA


class FeatureReducer(BaseEstimator, TransformerMixin):
    """
    Wrapper for PCA-based dimensionality reduction.
    
    Parameters:
    -----------
    variance_threshold : float, default=0.95
        Minimum variance to retain (0.0 to 1.0). PCA will select the minimum
        number of components that explain at least this variance.
    n_components : int or None, default=None
        Number of components to keep. If None, uses variance_threshold.
        If specified, overrides variance_threshold.
    """
    
    def __init__(self, variance_threshold: float = 0.95, n_components: Optional[int] = None):
        self.variance_threshold = variance_threshold
        self.n_components = n_components
        self.pca = None
        self.feature_columns_ = None
        self.n_components_selected_ = None
        self.explained_variance_ratio_ = None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit PCA on the data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features (excluding target and split_group)
        y : pd.Series, optional
            Target variable (not used, kept for API compatibility)
        """
        # Select only numerical columns for PCA
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if len(numerical_cols) == 0:
            raise ValueError("No numerical columns found for PCA")
        
        # Exclude target and metadata columns if present
        exclude_cols = ['nutriscore_grade', 'split_group', 'product_name', 'brands', 'code']
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        if len(numerical_cols) == 0:
            raise ValueError("No numerical feature columns found for PCA (all excluded)")
        
        # Store feature column names
        self.feature_columns_ = numerical_cols
        
        # Convert to numpy array for PCA
        X_array = X[numerical_cols].values
        
        # Determine number of components
        if self.n_components is not None:
            n_comp = min(self.n_components, X_array.shape[1])
        else:
            # Fit PCA with all components to compute explained variance
            pca_full = PCA()
            pca_full.fit(X_array)
            cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)
            n_comp = np.argmax(cumsum_variance >= self.variance_threshold) + 1
            # Ensure at least 1 component
            n_comp = max(1, n_comp)
        
        self.n_components_selected_ = n_comp
        
        # Fit PCA with selected number of components
        self.pca = PCA(n_components=n_comp)
        self.pca.fit(X_array)
        
        # Store explained variance ratio
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted PCA.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features to transform
            
        Returns:
        --------
        pd.DataFrame
            Transformed data with principal components (PC1, PC2, ...)
        """
        if self.pca is None:
            raise ValueError("PCA has not been fitted. Call fit() first.")
        
        # Convert to numpy array
        X_array = X[self.feature_columns_].values
        
        # Transform
        X_transformed = self.pca.transform(X_array)
        
        # Create DataFrame with principal component names
        pc_columns = [f'PC{i+1}' for i in range(self.n_components_selected_)]
        X_pca = pd.DataFrame(X_transformed, columns=pc_columns, index=X.index)
        
        # Preserve non-feature columns (target, split_group, etc.)
        preserve_cols = ['nutriscore_grade', 'split_group', 'product_name', 'brands', 'code']
        for col in preserve_cols:
            if col in X.columns:
                X_pca[col] = X[col].values
        
        return X_pca
    
    def get_explained_variance_ratio(self) -> np.ndarray:
        """Return explained variance ratio for each component."""
        if self.pca is None:
            raise ValueError("PCA has not been fitted. Call fit() first.")
        return self.explained_variance_ratio_
    
    def get_cumulative_variance(self) -> np.ndarray:
        """Return cumulative explained variance ratio."""
        if self.pca is None:
            raise ValueError("PCA has not been fitted. Call fit() first.")
        return np.cumsum(self.explained_variance_ratio_)
    
    def save(self, path: str) -> None:
        """Save the fitted reducer to disk."""
        joblib.dump(self, path)
    
    @classmethod
    def load(cls, path: str):
        """Load a fitted reducer from disk."""
        return joblib.load(path)

