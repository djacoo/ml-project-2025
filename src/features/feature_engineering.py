import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Nutritional features used for feature engineering
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
    Feature engineer for creating derived nutritional features.

    Creates the following derived features:
    1. Macro nutrient ratios:
       - fat_to_protein_ratio: Ratio of fat to protein (high values = high fat, low protein)
       - sugar_to_carb_ratio: Proportion of sugars in carbohydrates (0-1 range)
       - saturated_to_total_fat_ratio: Proportion of saturated fat in total fat (0-1 range)

    2. Energy density:
       - energy_density: Calories per gram (kcal/g)

    3. Caloric contributions:
       - calories_from_fat: Estimated calories from fat (fat * 9 kcal/g)
       - calories_from_carbs: Estimated calories from carbs (carbs * 4 kcal/g)
       - calories_from_protein: Estimated calories from protein (protein * 4 kcal/g)

    4. Boolean flags (based on WHO recommendations):
       - high_fat: Fat > 20g per 100g
       - high_sugar: Sugar > 15g per 100g
       - high_salt: Salt > 1.5g per 100g

    Parameters:
    -----------
    add_ratios : bool, default=True
        Whether to add macro nutrient ratios
    add_energy_density : bool, default=True
        Whether to add energy density feature
    add_caloric_contributions : bool, default=True
        Whether to add caloric contribution features
    add_boolean_flags : bool, default=True
        Whether to add boolean flags for high nutrient levels
    """

    def __init__(
        self,
        add_ratios=True,
        add_energy_density=True,
        add_caloric_contributions=True,
        add_boolean_flags=True
    ):
        self.add_ratios = add_ratios
        self.add_energy_density = add_energy_density
        self.add_caloric_contributions = add_caloric_contributions
        self.add_boolean_flags = add_boolean_flags

    def fit(self, X, y=None):
        """
        Fit method (no fitting required for feature engineering).

        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        y : array-like, optional
            Target variable (not used)

        Returns:
        --------
        self
        """
        return self

    def transform(self, X):
        """
        Transform input features by adding derived features.

        Parameters:
        -----------
        X : pd.DataFrame
            Input features

        Returns:
        --------
        X_engineered : pd.DataFrame
            DataFrame with original and derived features
        """
        X_engineered = X.copy()

        # 1. Create macro nutrient ratios
        if self.add_ratios:
            X_engineered = self._add_ratios(X_engineered)

        # 2. Calculate energy density
        if self.add_energy_density:
            X_engineered = self._add_energy_density(X_engineered)

        # 3. Calculate caloric contributions
        if self.add_caloric_contributions:
            X_engineered = self._add_caloric_contributions(X_engineered)

        # 4. Create boolean flags
        if self.add_boolean_flags:
            X_engineered = self._add_boolean_flags(X_engineered)

        return X_engineered

    def _add_ratios(self, X):
        """Add macro nutrient ratio features."""
        # Fat to protein ratio
        # Higher values indicate high fat and/or low protein
        X['fat_to_protein_ratio'] = np.where(
            X['proteins_100g'] > 0,
            X['fat_100g'] / X['proteins_100g'],
            0  # If protein is 0, set ratio to 0
        )
        # Replace inf values (when protein is very small but not 0) with a high value
        X['fat_to_protein_ratio'] = X['fat_to_protein_ratio'].replace([np.inf, -np.inf], 0)

        # Sugar to carb ratio
        # Represents proportion of carbs that are sugars (0-1 range, can exceed 1 due to data errors)
        X['sugar_to_carb_ratio'] = np.where(
            X['carbohydrates_100g'] > 0,
            X['sugars_100g'] / X['carbohydrates_100g'],
            0  # If carbs is 0, set ratio to 0
        )
        # Clip to [0, 1] range as sugars cannot exceed total carbs (handle data errors)
        X['sugar_to_carb_ratio'] = X['sugar_to_carb_ratio'].clip(0, 1)
        X['sugar_to_carb_ratio'] = X['sugar_to_carb_ratio'].replace([np.inf, -np.inf], 0)

        # Saturated to total fat ratio
        # Represents proportion of fat that is saturated (0-1 range)
        X['saturated_to_total_fat_ratio'] = np.where(
            X['fat_100g'] > 0,
            X['saturated-fat_100g'] / X['fat_100g'],
            0  # If fat is 0, set ratio to 0
        )
        # Clip to [0, 1] range as saturated fat cannot exceed total fat
        X['saturated_to_total_fat_ratio'] = X['saturated_to_total_fat_ratio'].clip(0, 1)
        X['saturated_to_total_fat_ratio'] = X['saturated_to_total_fat_ratio'].replace([np.inf, -np.inf], 0)

        return X

    def _add_energy_density(self, X):
        """
        Add energy density feature (kcal per gram).

        Energy density indicates how calorie-dense a food is.
        Higher values = more calories per gram = more energy-dense food.
        """
        # Energy density = kcal per 100g / 100 = kcal per gram
        X['energy_density'] = X['energy-kcal_100g'] / 100.0

        # Handle any inf or nan values
        X['energy_density'] = X['energy_density'].replace([np.inf, -np.inf], 0)
        X['energy_density'] = X['energy_density'].fillna(0)

        return X

    def _add_caloric_contributions(self, X):
        """
        Add caloric contribution features.

        Estimates how many calories come from each macronutrient:
        - Fat: 9 kcal per gram
        - Carbohydrates: 4 kcal per gram
        - Protein: 4 kcal per gram
        """
        # Calories from fat (9 kcal per gram)
        X['calories_from_fat'] = X['fat_100g'] * 9

        # Calories from carbs (4 kcal per gram)
        X['calories_from_carbs'] = X['carbohydrates_100g'] * 4

        # Calories from protein (4 kcal per gram)
        X['calories_from_protein'] = X['proteins_100g'] * 4

        # Handle any inf or nan values
        for col in ['calories_from_fat', 'calories_from_carbs', 'calories_from_protein']:
            X[col] = X[col].replace([np.inf, -np.inf], 0)
            X[col] = X[col].fillna(0)

        return X

    def _add_boolean_flags(self, X):
        """
        Add boolean flags for high nutrient levels based on WHO recommendations.

        Thresholds based on WHO guidelines for "high" nutrient content:
        - High fat: > 20g per 100g
        - High sugar: > 15g per 100g (simplified from WHO's 22.5g for "high")
        - High salt: > 1.5g per 100g
        """
        # High fat flag (> 20g per 100g)
        X['high_fat'] = (X['fat_100g'] > 20).astype(int)

        # High sugar flag (> 15g per 100g)
        X['high_sugar'] = (X['sugars_100g'] > 15).astype(int)

        # High salt flag (> 1.5g per 100g)
        X['high_salt'] = (X['salt_100g'] > 1.5).astype(int)

        return X

    def save(self, path):
        """Save the feature engineer to a file."""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        """Load a feature engineer from a file."""
        return joblib.load(path)
