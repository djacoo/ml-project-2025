"""
Data preprocessing module for the Nutri-Score prediction dataset.

This module implements missing value handling (imputation, dropping).

Note: Outlier removal logic has been moved to src/features/outlier_removal.py
(OutlierRemovalTransformer) to keep it within the pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
import sys

try:
    from .countries_mappings import COUNTRY_OVERRIDES
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.countries_mappings import COUNTRY_OVERRIDES

import pycountry

class MissingValueHandler:
    """
    Handles missing values in the Open Food Facts dataset.

    Strategy:
    - Drop features with >95% missing data (not useful for prediction)
    - Drop rows with missing target variable (nutriscore_grade)
    - Impute numerical nutritional features with median
    - Impute additives_n with 0 (assume no additives if not specified)
    - Handle categorical features with 'unknown' placeholder
    """

    def __init__(self, threshold_drop_feature: float = 0.95):
        """
        Initialize the missing value handler.

        Args:
            threshold_drop_feature: Drop features with missing percentage above this threshold
        """
        self.threshold_drop_feature = threshold_drop_feature
        self.imputation_stats = {}
        self.dropped_features = []
        self.missing_report = {}

    def analyze_missing_values(self, df: pd.DataFrame) -> Dict:
        """
        Analyze missing value patterns in the dataset.

        Args:
            df: Input dataframe

        Returns:
            Dictionary with missing value statistics
        """
        missing_info = {}

        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100

            missing_info[col] = {
                'missing_count': int(missing_count),
                'missing_percentage': round(missing_pct, 2),
                'dtype': str(df[col].dtype),
                'unique_values': int(df[col].nunique()) if missing_count < len(df) else 0
            }

        # Sort by missing percentage
        missing_info = dict(sorted(missing_info.items(),
                                  key=lambda x: x[1]['missing_percentage'],
                                  reverse=True))

        return missing_info

    def handle_missing_values(self, df: pd.DataFrame,
                            target_col: str = 'nutriscore_grade') -> pd.DataFrame:
        """
        Handle missing values according to the defined strategy.

        Args:
            df: Input dataframe
            target_col: Name of the target column

        Returns:
            Processed dataframe with missing values handled
        """
        df_processed = df.copy()
        initial_rows = len(df_processed)
        initial_cols = len(df_processed.columns)

        print("\n" + "="*70)
        print("MISSING VALUE HANDLING")
        print("="*70)
        print(f"\nInitial dataset: {initial_rows:,} rows × {initial_cols} columns")

        # Step 1: Analyze missing values
        print("\n1. Analyzing missing value patterns...")
        self.missing_report = self.analyze_missing_values(df_processed)

        # Print summary
        print("\nMissing Value Summary (Top 10):")
        print("-" * 70)
        for i, (col, info) in enumerate(list(self.missing_report.items())[:10], 1):
            if info['missing_percentage'] > 0:
                print(f"{i:2d}. {col:40s} {info['missing_percentage']:6.2f}% "
                      f"({info['missing_count']:,} missing)")

        # Step 2: Drop features with >95% missing
        print(f"\n2. Dropping features with >{self.threshold_drop_feature*100:.0f}% missing data...")
        high_missing_cols = [col for col, info in self.missing_report.items()
                            if info['missing_percentage'] > self.threshold_drop_feature * 100
                            and col != target_col]

        if high_missing_cols:
            print(f"   Dropping {len(high_missing_cols)} features:")
            for col in high_missing_cols:
                pct = self.missing_report[col]['missing_percentage']
                print(f"   - {col:40s} ({pct:.2f}% missing)")
                self.dropped_features.append(col)

            df_processed = df_processed.drop(columns=high_missing_cols)
        else:
            print("   No features to drop.")

        # Step 3: Drop rows with missing target
        print(f"\n3. Handling target variable '{target_col}'...")
        target_missing = df_processed[target_col].isnull().sum()
        if target_missing > 0:
            print(f"   Dropping {target_missing:,} rows with missing target")
            df_processed = df_processed[df_processed[target_col].notna()]
        else:
            print("   No missing values in target variable")

        # Step 4: Handle numerical features
        print("\n4. Imputing numerical features...")
        numerical_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()

        for col in numerical_cols:
            missing_count = df_processed[col].isnull().sum()
            if missing_count > 0:
                # Special handling for additives_n: impute with 0
                if col == 'additives_n':
                    df_processed[col] = df_processed[col].fillna(0)
                    self.imputation_stats[col] = {
                        'method': 'constant',
                        'value': 0,
                        'missing_count': int(missing_count),
                        'rationale': 'Assume no additives if not specified'
                    }
                    print(f"   - {col:40s} imputed with 0 ({missing_count:,} values)")
                else:
                    # Use median for other numerical features
                    median_val = df_processed[col].median()
                    df_processed[col] = df_processed[col].fillna(median_val)
                    self.imputation_stats[col] = {
                        'method': 'median',
                        'value': float(median_val),
                        'missing_count': int(missing_count),
                        'rationale': 'Median imputation for nutritional values'
                    }
                    print(f"   - {col:40s} imputed with median={median_val:.2f} "
                          f"({missing_count:,} values)")

        # Step 5: Handle categorical features
        print("\n5. Imputing categorical features...")
        categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()

        # Exclude target column
        categorical_cols = [col for col in categorical_cols if col != target_col]

        for col in categorical_cols:
            missing_count = df_processed[col].isnull().sum()
            if missing_count > 0:
                # Use 'unknown' for missing categorical values
                df_processed[col] = df_processed[col].fillna('unknown')
                self.imputation_stats[col] = {
                    'method': 'constant',
                    'value': 'unknown',
                    'missing_count': int(missing_count),
                    'rationale': 'Placeholder for missing categorical data'
                }
                print(f"   - {col:40s} imputed with 'unknown' ({missing_count:,} values)")

        # Step 6: Final verification
        print("\n6. Final verification...")
        final_rows = len(df_processed)
        final_cols = len(df_processed.columns)
        remaining_missing = df_processed.isnull().sum().sum()

        print(f"   Final dataset: {final_rows:,} rows × {final_cols} columns")
        print(f"   Rows dropped: {initial_rows - final_rows:,} "
              f"({(initial_rows - final_rows)/initial_rows*100:.2f}%)")
        print(f"   Columns dropped: {initial_cols - final_cols}")
        print(f"   Remaining missing values: {remaining_missing:,}")

        if remaining_missing > 0:
            print("\n   ⚠️  WARNING: Missing values still present!")
            print("\n   Columns with remaining missing values:")
            for col in df_processed.columns:
                missing = df_processed[col].isnull().sum()
                if missing > 0:
                    print(f"      - {col}: {missing:,} missing")
        else:
            print("\n   All missing values handled successfully!")

        print("\n" + "="*70)

        return df_processed

    def save_imputation_report(self, output_path: Path):
        """
        Save detailed report of imputation strategies and statistics.

        Args:
            output_path: Path to save the JSON report
        """
        report = {
            'strategy': {
                'threshold_drop_feature': self.threshold_drop_feature,
                'description': 'Drop features >95% missing, impute others based on type'
            },
            'dropped_features': self.dropped_features,
            'imputation_statistics': self.imputation_stats,
            'missing_value_analysis': self.missing_report
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nImputation report saved to: {output_path}")


def preprocess_dataset(input_path: Path,
                      output_path: Path,
                      report_path: Path = None) -> Tuple[pd.DataFrame, MissingValueHandler]:
    """
    Main preprocessing function to handle missing values and outliers.
    This function also cleans the 'countries' column.

    Args:
        input_path: Path to input CSV file
        output_path: Path to save processed CSV file
        report_path: Path to save imputation report (optional)

    Returns:
        Tuple of (processed dataframe, missing value handler instance)
    """
    print("\n" + "="*70)
    print("DATA PREPROCESSING - MISSING VALUE HANDLING")
    print("="*70)

    # Load data
    print(f"\nLoading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} rows x {len(df.columns)} columns")

    # Initialize handler and process
    handler = MissingValueHandler(threshold_drop_feature=0.95)
    df_processed = handler.handle_missing_values(df)

    # Clean countries column if it exists
    if 'countries' in df_processed.columns:
        print("\n" + "="*70)
        print("CLEANING COUNTRIES COLUMN")
        print("="*70)
        print(f"\nCleaning 'countries' column...")
        df_processed['countries'] = df_processed['countries'].apply(clean_countries_column)
        print("Countries column cleaned and overwritten")
    
    # Remove unnecessary columns
    print("\n" + "="*70)
    print("REMOVING UNNECESSARY COLUMNS")
    print("="*70)
    
    columns_to_remove = []
    
    # Remove energy_100g (redundant with energy-kcal_100g)
    if 'energy_100g' in df_processed.columns:
        columns_to_remove.append('energy_100g')
        print(f"  - Removing 'energy_100g' (redundant with 'energy-kcal_100g')")
    
    # Remove main_category and categories if they have > 10k unique values
    for col in ['main_category', 'categories']:
        if col in df_processed.columns:
            n_unique = df_processed[col].nunique()
            if n_unique > 10000:
                columns_to_remove.append(col)
                print(f"  - Removing '{col}' ({n_unique:,} unique values, > 10k threshold)")
    
    if columns_to_remove:
        df_processed = df_processed.drop(columns=columns_to_remove)
        print(f"\nRemoved {len(columns_to_remove)} column(s)")
    else:
        print("\nNo columns to remove")
    
    # Save processed data
    print(f"\nSaving processed data to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(output_path, index=False)
    print(f"Processed data saved successfully")

    # Save report
    if report_path:
        handler.save_imputation_report(report_path)

    return df_processed, handler

def normalize_single_country(raw_name: str) -> str:
    """
    Convert a raw country name to the standard ISO name.
    
    Strategy: Override -> Lookup ISO -> None
    
    Args:
        raw_name: Raw country name string
        
    Returns:
        Standardized country name or None if not found
    """
    # Base cleaning
    name = str(raw_name).lower().strip()
    name = name.replace("en:", "").replace("-", " ")
    
    if not name:
        return None

    # Check the dictionary in countries_mappings.py
    if name in COUNTRY_OVERRIDES:
        return COUNTRY_OVERRIDES[name]

    # Lookup in ISO standard using Pycountry
    try:
        country = pycountry.countries.lookup(name)
        country_name = country.name
        
        # Normalize: pycountry may return names in different languages
        # (e.g., "Deutschland" instead of "Germany")
        # Check if the result (case-insensitive) should map to a standard name
        country_name_lower = country_name.lower()
        
        # If pycountry returned a name that's in our override dict, use the standard name
        if country_name_lower in COUNTRY_OVERRIDES:
            return COUNTRY_OVERRIDES[country_name_lower]
        
        # Check if the result matches any standard name we use (case-insensitive)
        # This ensures consistency (e.g., if pycountry returns "Germany" and we use "Germany", keep it)
        standard_names = set(COUNTRY_OVERRIDES.values())
        for std_name in standard_names:
            if country_name_lower == std_name.lower():
                return std_name  # Return the standard name we use
        
        return country_name
    except LookupError:
        return None

def clean_countries_column(entry) -> str:
    """
    Clean and normalize country entries.
    
    Pipeline: Split -> Normalize -> Deduplicate -> Join
    
    Args:
        entry: Country entry (can be single country or comma-separated list)
        
    Returns:
        Cleaned, sorted, comma-separated country string or "unknown"
    """
    if pd.isna(entry) or entry == "unknown":
        return "unknown"
    
    valid_countries = set()
    
    # Split to handle lists
    for raw_item in str(entry).split(','):
        clean_name = normalize_single_country(raw_item)
        if clean_name:
            valid_countries.add(clean_name)
    
    # Return "unknown" if no valid countries found
    if not valid_countries:
        return "unknown"
    
    # Return sorted string for determinism
    return ",".join(sorted(valid_countries))

if __name__ == "__main__":
    # Define paths
    project_root = Path(__file__).parent.parent.parent
    input_file = project_root / "data" / "processed" / "openfoodfacts_filtered.csv"
    output_file = project_root / "data" / "processed" / "openfoodfacts_preprocessed.csv"
    report_file = project_root / "data" / "processed" / "imputation_report.json"

    # Run preprocessing
    df_processed, handler = preprocess_dataset(input_file, output_file, report_file)

    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)
    print(f"\nProcessed dataset shape: {df_processed.shape}")
    print(f"Output file: {output_file}")
    print(f"Report file: {report_file}")
