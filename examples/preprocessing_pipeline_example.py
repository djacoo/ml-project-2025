"""
Example usage of the PreprocessingPipeline.

This example shows how to:
1. Create a preprocessing pipeline
2. Fit it on training data
3. Transform training, validation, and test sets
4. Use the pipeline with train/test splits
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from features.preprocessing_pipeline import PreprocessingPipeline


def example_basic_usage():
    """Basic usage example."""
    print("="*70)
    print("EXAMPLE 1: Basic Usage")
    print("="*70)
    
    # Load data
    data_file = Path("../data/processed/openfoodfacts_filtered.csv")
    df = pd.read_csv(data_file)
    
    # Separate target
    y = df['nutriscore_grade']
    X = df.drop(columns=['nutriscore_grade'])
    
    # Create pipeline
    pipeline = PreprocessingPipeline(
        missing_threshold=0.95,
        top_n_countries=15,
        scaling_method='auto',
        include_pca=True
    )
    
    # Fit and transform
    X_processed = pipeline.fit_transform(X, y)
    X_processed['nutriscore_grade'] = y.values
    
    print(f"Original shape: {X.shape}")
    print(f"Processed shape: {X_processed.shape}")
    print(f"Pipeline steps: {pipeline.get_pipeline_steps()}")


def example_train_test_split():
    """Example with train/test split."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Train/Test Split")
    print("="*70)
    
    # Load data
    data_file = Path("../data/processed/openfoodfacts_filtered.csv")
    df = pd.read_csv(data_file)
    
    # Separate target
    y = df['nutriscore_grade']
    X = df.drop(columns=['nutriscore_grade'])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train: {len(X_train):,} samples")
    print(f"Test: {len(X_test):,} samples")
    
    # Create and fit pipeline on training data only
    pipeline = PreprocessingPipeline(
        missing_threshold=0.95,
        top_n_countries=15,
        scaling_method='auto',
        include_pca=True
    )
    
    print("\nFitting pipeline on training data...")
    pipeline.fit(X_train, y_train)
    
    # Transform both train and test
    print("Transforming training data...")
    X_train_processed = pipeline.transform(X_train, y_train)
    
    print("Transforming test data...")
    X_test_processed = pipeline.transform(X_test, y_test)
    
    print(f"\nTrain processed shape: {X_train_processed.shape}")
    print(f"Test processed shape: {X_test_processed.shape}")
    
    # Add targets back
    X_train_processed['nutriscore_grade'] = y_train.values
    X_test_processed['nutriscore_grade'] = y_test.values
    
    return X_train_processed, X_test_processed


def example_with_split_group():
    """Example using existing split_group column."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Using split_group Column")
    print("="*70)
    
    # Load data with split_group
    data_file = Path("../data/processed/openfoodfacts_cleaned.csv")
    df = pd.read_csv(data_file)
    
    if 'split_group' not in df.columns:
        print("⚠ split_group column not found. Run split_data.py first.")
        return
    
    # Separate by split
    train_mask = df['split_group'] == 'train'
    val_mask = df['split_group'] == 'val'
    test_mask = df['split_group'] == 'test'
    
    X_train = df[train_mask].drop(columns=['nutriscore_grade', 'split_group'])
    y_train = df[train_mask]['nutriscore_grade']
    
    X_val = df[val_mask].drop(columns=['nutriscore_grade', 'split_group'])
    y_val = df[val_mask]['nutriscore_grade']
    
    X_test = df[test_mask].drop(columns=['nutriscore_grade', 'split_group'])
    y_test = df[test_mask]['nutriscore_grade']
    
    print(f"Train: {len(X_train):,} samples")
    print(f"Val:   {len(X_val):,} samples")
    print(f"Test:  {len(X_test):,} samples")
    
    # Create pipeline
    pipeline = PreprocessingPipeline(
        missing_threshold=0.95,
        top_n_countries=15,
        scaling_method='auto',
        split_group_col='split_group',
        include_pca=True
    )
    
    # Fit on train only
    print("\nFitting pipeline on training data...")
    pipeline.fit(X_train, y_train)
    
    # Transform all splits
    print("Transforming splits...")
    X_train_processed = pipeline.transform(X_train, y_train)
    X_val_processed = pipeline.transform(X_val, y_val)
    X_test_processed = pipeline.transform(X_test, y_test)
    
    # Add targets and split_group back
    X_train_processed['nutriscore_grade'] = y_train.values
    X_train_processed['split_group'] = 'train'
    
    X_val_processed['nutriscore_grade'] = y_val.values
    X_val_processed['split_group'] = 'val'
    
    X_test_processed['nutriscore_grade'] = y_test.values
    X_test_processed['split_group'] = 'test'
    
    print(f"\nTrain processed: {X_train_processed.shape}")
    print(f"Val processed:   {X_val_processed.shape}")
    print(f"Test processed:  {X_test_processed.shape}")
    
    return X_train_processed, X_val_processed, X_test_processed


def example_custom_pipeline():
    """Example with custom pipeline configuration."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Custom Pipeline Configuration")
    print("="*70)
    
    # Load data
    data_file = Path("../data/processed/openfoodfacts_filtered.csv")
    df = pd.read_csv(data_file)
    
    y = df['nutriscore_grade']
    X = df.drop(columns=['nutriscore_grade'])
    
    # Create custom pipeline (no PCA, different scaling)
    pipeline = PreprocessingPipeline(
        missing_threshold=0.90,  # More aggressive feature dropping
        top_n_countries=20,      # More countries
        scaling_method='standard',  # Force StandardScaler
        pca_variance_threshold=0.99,  # Keep more variance
        include_pca=False  # Skip PCA
    )
    
    print("Custom pipeline configuration:")
    print(f"  Missing threshold: 0.90")
    print(f"  Top N countries: 20")
    print(f"  Scaling method: standard")
    print(f"  Include PCA: False")
    
    X_processed = pipeline.fit_transform(X, y)
    
    print(f"\nOriginal: {X.shape}")
    print(f"Processed: {X_processed.shape}")


if __name__ == "__main__":
    # Run examples
    example_basic_usage()
    example_train_test_split()
    example_with_split_group()
    example_custom_pipeline()
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED")
    print("="*70)
