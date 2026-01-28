"""
Script to split cleaned dataset into train/validation/test sets.

This script:
1. Loads the cleaned dataset
2. Performs stratified split into train/val/test
3. Adds split_group column to the dataset
4. Saves the dataset with split_group
5. Saves separate files for each split
6. Saves metadata about the split
"""

import sys
import pandas as pd
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.data_loader import (
    split_data,
    add_split_group_column,
    verify_stratification,
    save_splits,
    save_split_metadata
)


def main():
    # Configuration
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    random_state = 42
    target_col = 'nutriscore_grade'
    
    project_root = Path(__file__).parent.parent
    input_file = project_root / "data" / "processed" / "openfoodfacts_cleaned.csv"
    output_file = project_root / "data" / "processed" / "openfoodfacts_cleaned.csv"  # Overwrite with split_group
    splits_dir = project_root / "data" / "processed" / "splits"
    metadata_file = project_root / "data" / "processed" / "split_metadata.json"
    
    print("="*70)
    print("DATA SPLITTING - TRAIN/VALIDATION/TEST")
    print("="*70)
    
    # Load cleaned data
    print(f"\nLoading cleaned data from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df):,} rows x {len(df.columns)} columns")
    
    # Check if target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    # Check if split_group already exists
    if 'split_group' in df.columns:
        print("\n⚠ Warning: 'split_group' column already exists in dataset")
        print("Overwriting existing split...")
        df = df.drop(columns=['split_group'])
    
    # Perform split
    print(f"\nSplitting data into train/val/test ({train_ratio*100:.0f}%/{val_ratio*100:.0f}%/{test_ratio*100:.0f}%)...")
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        df,
        target_col=target_col,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
        stratify=True
    )
    
    print(f"Split completed:")
    print(f"  Train: {len(X_train):,} samples ({len(X_train)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(X_val):,} samples ({len(X_val)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(X_test):,} samples ({len(X_test)/len(df)*100:.1f}%)")
    
    # Verify stratification
    print("\nVerifying class distribution...")
    stats = verify_stratification(y_train, y_val, y_test)
    
    print("\nClass Distribution:")
    print("-" * 70)
    all_classes = sorted(set(y_train) | set(y_val) | set(y_test))
    
    print(f"{'Class':<10} {'Train':<15} {'Val':<15} {'Test':<15} {'Max Dev':<10}")
    print("-" * 70)
    
    for cls in all_classes:
        train_pct = stats['train_distribution'].get(cls, 0.0) * 100
        val_pct = stats['val_distribution'].get(cls, 0.0) * 100
        test_pct = stats['test_distribution'].get(cls, 0.0) * 100
        
        train_count = stats['train_counts'].get(cls, 0)
        val_count = stats['val_counts'].get(cls, 0)
        test_count = stats['test_counts'].get(cls, 0)
        
        # Calculate max deviation for this class
        max_dev = max(abs(val_pct - train_pct), abs(test_pct - train_pct))
        
        print(f"{cls:<10} {train_count:>6} ({train_pct:>5.1f}%) {val_count:>6} ({val_pct:>5.1f}%) {test_count:>6} ({test_pct:>5.1f}%) {max_dev:>6.2f}%")
    
    print("-" * 70)
    print(f"Max deviation across all classes: {stats['max_deviation']*100:.2f}%")
    
    if stats['is_balanced']:
        print("Stratification is balanced (max deviation < 5%)")
    else:
        print("⚠ Warning: Stratification shows some imbalance (max deviation >= 5%)")
    
    # Get indices for split_group column
    train_indices = X_train.index.values
    val_indices = X_val.index.values
    test_indices = X_test.index.values
    
    # Add split_group column to original dataframe
    print("\nAdding 'split_group' column to dataset...")
    df_with_split = add_split_group_column(df, train_indices, val_indices, test_indices)
    
    # Save dataset with split_group
    print(f"\nSaving dataset with 'split_group' column to: {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_with_split.to_csv(output_file, index=False)
    print("Dataset saved")
    
    # Save separate split files
    print(f"\nSaving separate split files to: {splits_dir}")
    saved_files = save_splits(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        output_dir=splits_dir
    )
    print("Split files saved:")
    for name, path in saved_files.items():
        print(f"  - {name}: {path.name}")
    
    # Save metadata
    print(f"\nSaving split metadata to: {metadata_file}")
    config = {
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'random_state': random_state,
        'target_col': target_col,
        'total_samples': len(df),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test)
    }
    save_split_metadata(stats, config, metadata_file)
    print("Metadata saved")
    
    # Final summary
    print("\n" + "="*70)
    print("DATA SPLITTING COMPLETE")
    print("="*70)
    print(f"\nDataset with 'split_group': {output_file}")
    print(f"Split files directory: {splits_dir}")
    print(f"Metadata file: {metadata_file}")
    print("\nNext steps:")
    print("  1. Run 'scripts/apply_encoding.py' (will use existing split_group)")
    print("  2. Run 'scripts/apply_scaling.py' (scales all numerical features)")
    print("  3. Run 'scripts/apply_pca.py'")
    print("="*70)


if __name__ == "__main__":
    main()

