"""
Dimensionality reduction using Principal Component Analysis (PCA).

Implements variance-threshold based component selection following scikit-learn
Transformer API for integration into preprocessing pipelines.
"""
import joblib
import pandas as pd
import numpy as np
from typing import Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA


METADATA_COLS = ['nutriscore_grade', 'split_group', 'product_name', 'brands', 'code']


class FeatureReducer(BaseEstimator, TransformerMixin):
    """
    PCA-based dimensionality reduction with variance threshold selection.
    
    Parameters
    ----------
    variance_threshold : float, default=0.95
        Minimum cumulative explained variance to retain. Selects minimum number
        of components such that cumulative variance >= threshold.
    n_components : int, optional
        Fixed number of components. If specified, overrides variance_threshold.
    
    Attributes
    ----------
    pca_ : sklearn.decomposition.PCA
        Fitted PCA transformer.
    feature_columns_ : list[str]
        Numerical feature columns used for PCA (metadata columns excluded).
    n_components_selected_ : int
        Number of principal components selected.
    explained_variance_ratio_ : np.ndarray of shape (n_components_selected_,)
        Explained variance ratio for each selected component.
    
    Notes
    -----
    Automatically excludes metadata columns (target, split_group, product_name,
    brands, code) from PCA transformation. These columns are preserved in output.
    """
    
    def __init__(
        self, variance_threshold: float = 0.95, n_components: Optional[int] = None
    ):
        self.variance_threshold = variance_threshold
        self.n_components = n_components
        self.pca_: Optional[PCA] = None
        self.feature_columns_: Optional[list] = None
        self.n_components_selected_: Optional[int] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureReducer':
        """
        Fit PCA on numerical features.
        
        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Input features. Only numerical columns are used for PCA.
        y : pd.Series, optional
            Ignored. Present for API compatibility.
        
        Returns
        -------
        self : FeatureReducer
            Returns self for method chaining.
        
        Raises
        ------
        ValueError
            If no numerical columns found or all columns excluded as metadata.
        """
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if len(numerical_cols) == 0:
            raise ValueError("No numerical columns found for PCA")
        
        self.feature_columns_ = [
            col for col in numerical_cols if col not in METADATA_COLS
        ]
        
        if len(self.feature_columns_) == 0:
            raise ValueError("No numerical feature columns found for PCA (all excluded)")
        
        X_array = X[self.feature_columns_].values
        
        if self.n_components is not None:
            n_comp = min(self.n_components, X_array.shape[1])
        else:
            pca_full = PCA()
            pca_full.fit(X_array)
            cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)
            n_comp = np.argmax(cumsum_variance >= self.variance_threshold) + 1
            n_comp = max(1, n_comp)
        
        self.n_components_selected_ = n_comp
        self.pca_ = PCA(n_components=n_comp)
        self.pca_.fit(X_array)
        self.explained_variance_ratio_ = self.pca_.explained_variance_ratio_
        
        cumulative_variance = np.cumsum(self.explained_variance_ratio_)[-1]
        print(f"\nPCA fitted: {n_comp} components selected")
        print(f"Explained variance: {cumulative_variance:.4f} ({cumulative_variance*100:.2f}%)")
        print(f"Feature reduction: {len(self.feature_columns_)} -> {n_comp} components")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features to principal component space.
        
        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Input features to transform.
        
        Returns
        -------
        X_pca : pd.DataFrame of shape (n_samples, n_components_selected_)
            Transformed features as principal components (PC1, PC2, ...).
            Metadata columns preserved and added to output.
        
        Raises
        ------
        ValueError
            If PCA not fitted.
        """
        if self.pca_ is None:
            raise ValueError("PCA has not been fitted. Call fit() first.")
        
        # Validate that all required feature columns are present
        missing_cols = [col for col in self.feature_columns_ if col not in X.columns]
        if missing_cols:
            raise ValueError(
                f"Missing feature columns required for PCA transformation: {missing_cols}. "
                f"Expected columns: {self.feature_columns_[:10]}..."
                if len(self.feature_columns_) > 10 else f"Expected columns: {self.feature_columns_}"
            )
        
        X_array = X[self.feature_columns_].values
        X_transformed = self.pca_.transform(X_array)
        
        pc_columns = [f'PC{i+1}' for i in range(self.n_components_selected_)]
        X_pca = pd.DataFrame(X_transformed, columns=pc_columns, index=X.index)
        
        for col in METADATA_COLS:
            if col in X.columns:
                X_pca[col] = X[col].values
        
        return X_pca
    
    def get_explained_variance_ratio(self) -> np.ndarray:
        """
        Return explained variance ratio for each component.
        
        Returns
        -------
        explained_variance_ratio : np.ndarray of shape (n_components_selected_,)
            Explained variance ratio per component.
        
        Raises
        ------
        ValueError
            If PCA not fitted.
        """
        if self.pca_ is None:
            raise ValueError("PCA has not been fitted. Call fit() first.")
        return self.explained_variance_ratio_
    
    def get_cumulative_variance(self) -> np.ndarray:
        """
        Return cumulative explained variance ratio.
        
        Returns
        -------
        cumulative_variance : np.ndarray of shape (n_components_selected_,)
            Cumulative explained variance ratio.
        
        Raises
        ------
        ValueError
            If PCA not fitted.
        """
        if self.pca_ is None:
            raise ValueError("PCA has not been fitted. Call fit() first.")
        return np.cumsum(self.explained_variance_ratio_)
    
    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Fit PCA and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def save(self, path: str) -> None:
        """Save fitted reducer to disk."""
        joblib.dump(self, path)
    
    @classmethod
    def load(cls, path: str) -> 'FeatureReducer':
        """Load fitted reducer from disk."""
        return joblib.load(path)
