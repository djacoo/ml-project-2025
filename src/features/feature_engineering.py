"""
Feature engineering for creating derived nutritional features.

Implements creation of macro nutrient ratios, energy density, caloric contributions,
and boolean flags based on WHO recommendations following scikit-learn Transformer API.
"""
import joblib
import pandas as pd
import numpy as np
from typing import Optional
from sklearn.base import BaseEstimator, TransformerMixin


NUTRITIONAL_FEATURES = [
    "energy-kcal_100g",
    "fat_100g",
    "saturated-fat_100g",
    "carbohydrates_100g",
    "sugars_100g",
    "fiber_100g",
    "proteins_100g",
    "salt_100g"
]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Creates derived nutritional features from base nutritional values.
    
    Parameters
    ----------
    add_ratios : bool, default=True
        Whether to add macro nutrient ratio features.
    add_energy_density : bool, default=True
        Whether to add energy density feature (kcal/g).
    add_caloric_contributions : bool, default=True
        Whether to add caloric contribution features.
    add_boolean_flags : bool, default=True
        Whether to add boolean flags for high nutrient levels (WHO thresholds).
    
    Notes
    -----
    Derived features:
    - Ratios: fat_to_protein, sugar_to_carb, saturated_to_total_fat
    - Energy density: calories per gram
    - Caloric contributions: calories from fat/carbs/protein
    - Boolean flags: high_fat (>20g), high_sugar (>15g), high_salt (>1.5g)
    """
    
    def __init__(
        self,
        add_ratios: bool = True,
        add_energy_density: bool = True,
        add_caloric_contributions: bool = True,
        add_boolean_flags: bool = True
    ):
        self.add_ratios = add_ratios
        self.add_energy_density = add_energy_density
        self.add_caloric_contributions = add_caloric_contributions
        self.add_boolean_flags = add_boolean_flags
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureEngineer':
        """
        Fit method (no fitting required for feature engineering).
        
        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Input features.
        y : pd.Series, optional
            Ignored. Present for API compatibility.
        
        Returns
        -------
        self : FeatureEngineer
            Returns self for method chaining.
        """
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features by adding derived nutritional features.
        
        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Input features containing nutritional columns.
        
        Returns
        -------
        X_engineered : pd.DataFrame of shape (n_samples, n_features + n_derived)
            Original features plus derived features.
        """
        X_engineered = X.copy()
        
        if self.add_ratios:
            X_engineered = self._add_ratios(X_engineered)
        
        if self.add_energy_density:
            X_engineered = self._add_energy_density(X_engineered)
        
        if self.add_caloric_contributions:
            X_engineered = self._add_caloric_contributions(X_engineered)
        
        if self.add_boolean_flags:
            X_engineered = self._add_boolean_flags(X_engineered)
        
        # Impute any NaN values created by feature engineering (e.g., division by zero in ratios)
        # This ensures compatibility with downstream transformers like PCA
        derived_cols = [col for col in X_engineered.columns if col not in X.columns]
        if derived_cols:
            for col in derived_cols:
                if X_engineered[col].isna().any():
                    # For ratio features, impute with 0 (when denominator is 0, ratio is undefined)
                    # For other features, impute with 0 as well
                    X_engineered[col] = X_engineered[col].fillna(0.0)
        
        return X_engineered
    
    def _add_ratios(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add macro nutrient ratio features."""
        if 'fat_100g' in X.columns and 'proteins_100g' in X.columns:
            X['fat_to_protein_ratio'] = np.divide(
                X['fat_100g'], X['proteins_100g'],
                out=np.full_like(X['fat_100g'], np.nan, dtype=float),
                where=X['proteins_100g'] != 0
            )
        
        if 'sugars_100g' in X.columns and 'carbohydrates_100g' in X.columns:
            X['sugar_to_carb_ratio'] = np.divide(
                X['sugars_100g'], X['carbohydrates_100g'],
                out=np.full_like(X['sugars_100g'], np.nan, dtype=float),
                where=X['carbohydrates_100g'] != 0
            )
        
        if 'saturated-fat_100g' in X.columns and 'fat_100g' in X.columns:
            X['saturated_to_total_fat_ratio'] = np.divide(
                X['saturated-fat_100g'], X['fat_100g'],
                out=np.full_like(X['saturated-fat_100g'], np.nan, dtype=float),
                where=X['fat_100g'] != 0
            )
        
        return X
    
    def _add_energy_density(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add energy density feature (kcal/g)."""
        if 'energy-kcal_100g' in X.columns:
            X['energy_density'] = X['energy-kcal_100g'] / 100.0
        return X
    
    def _add_caloric_contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add caloric contribution features (kcal from macros)."""
        if 'fat_100g' in X.columns:
            X['calories_from_fat'] = X['fat_100g'] * 9.0
        
        if 'carbohydrates_100g' in X.columns:
            X['calories_from_carbs'] = X['carbohydrates_100g'] * 4.0
        
        if 'proteins_100g' in X.columns:
            X['calories_from_protein'] = X['proteins_100g'] * 4.0
        
        return X
    
    def _add_boolean_flags(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add boolean flags for high nutrient levels (WHO thresholds)."""
        if 'fat_100g' in X.columns:
            X['high_fat'] = (X['fat_100g'] > 20.0).astype(int)
        
        if 'sugars_100g' in X.columns:
            X['high_sugar'] = (X['sugars_100g'] > 15.0).astype(int)
        
        if 'salt_100g' in X.columns:
            X['high_salt'] = (X['salt_100g'] > 1.5).astype(int)
        
        return X
    
    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def save(self, path: str) -> None:
        """Save fitted engineer to disk."""
        joblib.dump(self, path)
    
    @classmethod
    def load(cls, path: str) -> 'FeatureEngineer':
        """Load fitted engineer from disk."""
        return joblib.load(path)
