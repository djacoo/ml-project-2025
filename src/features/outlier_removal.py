"""
Outlier detection and removal transformers.

Provides scikit-learn compatible transformers for missing value handling and
outlier removal following domain-specific rules and statistical methods.
"""
import pandas as pd
from typing import Optional, Dict
from sklearn.base import BaseEstimator, TransformerMixin

from data.preprocessing import MissingValueHandler


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
        
        Raises
        ------
        ValueError
            If target_col is required but not found in X and y is None.
        
        Notes
        -----
        Rows with missing target variable are dropped. Features with >95% missing
        data are removed. Numerical features are imputed with median, categorical
        with 'unknown', and additives_n with 0.
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
    remove_statistical_outliers : bool, default=False
        Whether to remove statistical outliers (3×IQR method).
        Default is False because statistical outliers in nutritional data
        often represent valid product categories, not measurement errors.
    
    Attributes
    ----------
    rows_removed_ : int
        Number of rows removed during last transform call.
    outlier_report_ : dict
        Dictionary containing outlier detection and removal statistics.
    valid_ranges_ : dict
        Valid ranges for nutritional values (per 100g).
    
    Notes
    -----
    Removal strategy:
    - Domain rules: negative values, values > 100g per 100g (macros),
      energy > 3000 kcal, salt > 50g
    - Statistical: IQR method with 3×IQR threshold (optional, controlled by flag)
    """
    
    def __init__(
        self, 
        target_col: str = 'nutriscore_grade',
        remove_statistical_outliers: bool = False
    ):
        self.target_col = target_col
        self.remove_statistical_outliers = remove_statistical_outliers
        
        # Define valid ranges for nutritional values (per 100g)
        # Note: energy_100g removed (redundant with energy-kcal_100g)
        self.valid_ranges_ = {
            'fat_100g': (0, 100),              # 0-100g per 100g
            'saturated-fat_100g': (0, 100),    # 0-100g per 100g
            'carbohydrates_100g': (0, 100),    # 0-100g per 100g
            'sugars_100g': (0, 100),           # 0-100g per 100g
            'fiber_100g': (0, 100),            # 0-100g per 100g (extreme but possible)
            'proteins_100g': (0, 100),         # 0-100g per 100g
            'salt_100g': (0, 50),              # 0-50g per 100g (very salty max)
        }
        
        self.rows_removed_ = 0
        self.outlier_report_: Optional[Dict] = None
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict:
        """
        Detect various types of outliers in the dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe

        Returns
        -------
        dict
            Dictionary with outlier statistics
        """
        outlier_info = {}

        for col in self.valid_ranges_.keys():
            if col not in df.columns:
                continue

            col_info = {}

            # Check negative values
            negative_mask = df[col] < 0
            col_info['negative_count'] = int(negative_mask.sum())

            # Check values outside valid range
            min_val, max_val = self.valid_ranges_[col]
            below_min = (df[col] < min_val).sum()
            above_max = (df[col] > max_val).sum()
            col_info['below_min'] = int(below_min)
            col_info['above_max'] = int(above_max)
            col_info['valid_range'] = self.valid_ranges_[col]

            # Statistical outliers using IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR  # Using 3*IQR for extreme outliers only
            upper_bound = Q3 + 3 * IQR

            statistical_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            col_info['statistical_outliers'] = int(statistical_outliers)
            col_info['IQR_bounds'] = (float(lower_bound), float(upper_bound))

            # Current stats
            col_info['min'] = float(df[col].min())
            col_info['max'] = float(df[col].max())
            col_info['median'] = float(df[col].median())

            outlier_info[col] = col_info

        return outlier_info
    
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
        self.outlier_report_ = {}
        self.outlier_report_['before_cleaning'] = self._detect_outliers(X)
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
        
        Notes
        -----
        Removal strategy:
        - Step 1: Analyze outliers (if not done in fit)
        - Step 2: Remove invalid values (negative, outside valid ranges)
        - Step 3: Remove statistical outliers (if remove_statistical_outliers=True)
        - Step 4: Final verification
        
        Invalid values are always removed. Statistical outliers are only removed
        if remove_statistical_outliers=True (default: False).
        """
        df_clean = X.copy()
        initial_rows = len(df_clean)

        print("\n" + "="*70)
        print("OUTLIER REMOVAL")
        print("="*70)
        print(f"\nInitial dataset: {initial_rows:,} rows")

        # Step 1: Analyze outliers before removal (if not already done in fit)
        if self.outlier_report_ is None:
            self.outlier_report_ = {}
        if 'before_cleaning' not in self.outlier_report_:
            print("\n1. Analyzing outliers...")
            self.outlier_report_['before_cleaning'] = self._detect_outliers(df_clean)
        else:
            print("\n1. Using pre-analyzed outlier patterns...")

        # Print summary of issues found
        print("\nOutlier Summary (Before Cleaning):")
        print("-" * 70)
        total_issues = 0
        for col, info in self.outlier_report_['before_cleaning'].items():
            issues = info['negative_count'] + info['above_max']
            if issues > 0:
                total_issues += issues
                print(f"{col:25s}: {info['negative_count']:6,} negative, "
                      f"{info['above_max']:6,} above max ({info['valid_range'][1]})")

        print(f"\nTotal problematic values: {total_issues:,}")

        # Step 2: Remove rows with invalid values
        print("\n2. Removing invalid values...")
        rows_removed = 0
        removal_reasons = {}

        for col in self.valid_ranges_.keys():
            if col not in df_clean.columns:
                continue

            min_val, max_val = self.valid_ranges_[col]

            # Create mask for invalid values (negative or outside valid range)
            invalid_mask = (df_clean[col] < min_val) | (df_clean[col] > max_val)
            invalid_count = invalid_mask.sum()

            if invalid_count > 0:
                # Store removal reason
                removal_reasons[col] = {
                    'count': int(invalid_count),
                    'reason': f'Outside valid range ({min_val}, {max_val})'
                }

                # Remove invalid rows
                df_clean = df_clean[~invalid_mask]
                rows_removed += invalid_count

                print(f"   - {col:25s}: Removed {invalid_count:,} rows")

        # Step 3: Remove statistical outliers (optional, controlled by flag)
        if self.remove_statistical_outliers:
            print("\n3. Removing extreme statistical outliers...")

            nutritional_cols = list(self.valid_ranges_.keys())
            existing_cols = [col for col in nutritional_cols if col in df_clean.columns]

            # Create combined mask for all statistical outliers
            combined_extreme_mask = pd.Series(False, index=df_clean.index)

            for col in existing_cols:
                # Use 3*IQR for very extreme outliers only
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                statistical_lower = Q1 - 3 * IQR
                statistical_upper = Q3 + 3 * IQR

                # Get valid physical ranges
                min_val, max_val = self.valid_ranges_[col]
                
                # Clamp statistical bounds to valid physical ranges
                # Lower bound: clamp to 0 (negative values already removed in Step 2)
                # Upper bound: use statistical upper, but don't exceed physical max
                lower_bound = max(0, statistical_lower)
                upper_bound = min(statistical_upper, max_val)
                
                # Only remove upper outliers (very high values that are statistically extreme)
                # Lower outliers (negative/very low) already handled in Step 2
                extreme_mask = (df_clean[col] > upper_bound)
                extreme_count = extreme_mask.sum()

                if extreme_count > 0:
                    # Combine with overall mask (a row is outlier if outlier in any column)
                    combined_extreme_mask = combined_extreme_mask | extreme_mask
                    
                    if col not in removal_reasons:
                        removal_reasons[col] = {
                            'count': int(extreme_count),
                            'reason': f'Extreme statistical outlier (3*IQR, upper bound: {upper_bound:.2f})'
                        }

                    print(f"   - {col:25s}: {extreme_count:,} extreme outliers detected "
                          f"(removing values > {upper_bound:.2f}, "
                          f"IQR: {statistical_lower:.2f} - {statistical_upper:.2f}, "
                          f"clamped: {lower_bound:.2f} - {upper_bound:.2f})")

            # Remove all statistical outliers at once
            if combined_extreme_mask.any():
                statistical_outliers_count = combined_extreme_mask.sum()
                df_clean = df_clean[~combined_extreme_mask]
                rows_removed += statistical_outliers_count
                print(f"\n   Removed {statistical_outliers_count:,} rows with statistical outliers")
            else:
                print("\n   No statistical outliers to remove")
        else:
            print("\n3. Statistical outlier removal skipped (remove_statistical_outliers=False)")

        # Step 4: Final verification
        print("\n4. Final verification...")
        final_rows = len(df_clean)
        total_removed = initial_rows - final_rows
        removal_pct = (total_removed / initial_rows) * 100

        print(f"   Initial rows: {initial_rows:,}")
        print(f"   Final rows: {final_rows:,}")
        print(f"   Rows removed: {total_removed:,} ({removal_pct:.2f}%)")

        # Analyze outliers after removal
        self.outlier_report_['after_cleaning'] = self._detect_outliers(df_clean)
        self.outlier_report_['removal_summary'] = removal_reasons
        self.outlier_report_['rows_removed'] = int(total_removed)
        self.outlier_report_['removal_percentage'] = float(removal_pct)

        # Verify no invalid values remain
        print("\n5. Validation check...")
        issues_remaining = 0
        for col, info in self.outlier_report_['after_cleaning'].items():
            issues = info['negative_count'] + info['above_max']
            if issues > 0:
                issues_remaining += issues
                print(f"   ⚠️  {col}: {issues:,} issues remaining")

        if issues_remaining == 0:
            print("   All invalid values successfully removed!")
        else:
            print(f"   ⚠️  WARNING: {issues_remaining:,} issues remaining")

        print("\n" + "="*70)

        self.rows_removed_ = total_removed
        return df_clean
    
    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
