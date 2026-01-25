"""
Script to remove outliers and invalid values from the preprocessed dataset.

This script applies domain-specific rules and statistical methods to clean
nutritional data, removing impossible values and extreme outliers.

Uses OutlierRemovalTransformer for scikit-learn compatibility.
If split_group exists, fits only on train set and applies to whole dataset.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from features.outlier_removal import OutlierRemovalTransformer


def main():
    """Main execution function for outlier removal."""
    # Define paths
    project_root = Path(__file__).parent.parent
    input_file = project_root / "data" / "processed" / "openfoodfacts_preprocessed.csv"
    output_file = project_root / "data" / "processed" / "openfoodfacts_cleaned.csv"
    report_file = project_root / "data" / "processed" / "outlier_removal_report.json"
    models_dir = project_root / "models"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("OUTLIER REMOVAL PROCESS")
    print("="*70)
    
    # Load preprocessed data
    print(f"\nLoading data from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df):,} rows x {len(df.columns)} columns")
    
    # Extract target and check for split_group
    target_col = 'nutriscore_grade'
    split_group_col = 'split_group'
    
    y = df[target_col] if target_col in df.columns else None
    split_group = df[split_group_col].copy() if split_group_col in df.columns else None
    
    # Remove target and split_group from X for processing
    cols_to_drop = [target_col] if target_col in df.columns else []
    if split_group_col in df.columns:
        cols_to_drop.append(split_group_col)
    
    X = df.drop(columns=cols_to_drop, errors='ignore')
    
    # Initialize transformer
    transformer = OutlierRemovalTransformer(target_col=target_col)
    
    # Check if split_group exists (for reproducible train/test split)
    if split_group is not None:
        print(f"\nUsing existing 'split_group' column:")
        train_mask = split_group == 'train'
        print(f"  Train: {train_mask.sum():,} samples")
        print(f"  Val:   {(split_group == 'val').sum():,} samples")
        print(f"  Test:  {(split_group == 'test').sum():,} samples")
        
        # Get train indices for fitting
        train_idx = df.index[train_mask]
        X_train = X.iloc[train_idx].copy()
        
        # Fit transformer only on train set
        print("\nFitting outlier removal transformer on TRAIN subset")
        transformer.fit(X_train, y=y.iloc[train_idx] if y is not None else None)
        
        # Apply transformation to whole dataset
        print("Applying transformation to the whole dataset")
        X_clean = transformer.transform(X)
        
        # Get rows removed info
        rows_removed = transformer.rows_removed_
        
    else:
        # No split_group: fit and transform on all data
        print("\nNo 'split_group' column found. Fitting on all data.")
        print("(Note: For reproducible splits, run 'scripts/split_data.py' first)")
        
        # Fit and transform
        X_clean = transformer.fit_transform(X, y=y)
        rows_removed = transformer.rows_removed_
    
    # Add back target and split_group columns
    # Note: OutlierRemovalTransformer preserves original indices for remaining rows
    # So we can use X_clean.index to get the corresponding values from original data
    if y is not None:
        # Use indices from cleaned dataframe to get corresponding target values
        remaining_indices = X_clean.index
        y_clean = y.loc[remaining_indices]
        X_clean[target_col] = y_clean.values
    
    if split_group is not None:
        # Use indices from cleaned dataframe to get corresponding split_group values
        remaining_indices = X_clean.index
        split_group_clean = split_group.loc[remaining_indices]
        X_clean[split_group_col] = split_group_clean.values
    
    # Save cleaned data
    print(f"\nSaving cleaned data to: {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    X_clean.to_csv(output_file, index=False)
    print(f"Cleaned data saved successfully")
    
    # Save outlier removal report (using handler's report)
    if hasattr(transformer.handler, 'outlier_report'):
        report_data = {
            'rows_removed': int(rows_removed),
            'rows_before': int(len(X)),
            'rows_after': int(len(X_clean)),
            'removal_percentage': float(rows_removed / len(X) * 100) if len(X) > 0 else 0.0,
            'outlier_report': transformer.handler.outlier_report,
            'removal_stats': transformer.handler.removal_stats if hasattr(transformer.handler, 'removal_stats') else {}
        }
        
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        print(f"Outlier removal report saved to: {report_file}")
    
    # Final summary
    print("\n" + "="*70)
    print("OUTLIER REMOVAL COMPLETE")
    print("="*70)
    print(f"\nCleaned dataset shape: {X_clean.shape}")
    print(f"Output file: {output_file}")
    print(f"Report file: {report_file}")
    print("\nSummary:")
    print(f"  Rows before: {len(X):,}")
    print(f"  Rows after: {len(X_clean):,}")
    print(f"  Rows removed: {rows_removed:,} ({rows_removed / len(X) * 100:.2f}%)")
    
    if split_group is not None:
        print(f"\n  Split preserved: {split_group_col} column maintained")
    
    print("="*70)


if __name__ == "__main__":
    main()
