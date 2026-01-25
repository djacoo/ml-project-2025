"""
Script to run the complete preprocessing pipeline using scikit-learn Pipeline.

This script demonstrates how to use the PreprocessingPipeline class to apply
all preprocessing steps in a single pipeline.

Usage:
    python scripts/run_preprocessing_pipeline.py
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from features.preprocessing_pipeline import PreprocessingPipeline


def main():
    project_root = Path(__file__).parent.parent
    
    # Input and output files
    input_file = project_root / "data" / "processed" / "openfoodfacts_filtered.csv"
    output_file = project_root / "data" / "processed" / "openfoodfacts_pipeline_processed.csv"
    models_dir = project_root / "models"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("COMPLETE PREPROCESSING PIPELINE")
    print("="*70)
    
    # Load data
    print(f"\nLoading data from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df):,} rows x {len(df.columns)} columns")
    
    # Check if target column exists
    target_col = 'nutriscore_grade'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    # Separate features and target
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    # Create preprocessing pipeline
    print("\nCreating preprocessing pipeline...")
    pipeline = PreprocessingPipeline(
        missing_threshold=0.95,
        top_n_countries=15,
        scaling_method='auto',
        scaling_skew_threshold=1.0,
        pca_variance_threshold=0.95,
        target_col=target_col,
        split_group_col=None,  # No split_group in initial data
        preserve_cols=['product_name', 'brands', 'code'],
        include_pca=True
    )
    
    print(f"Pipeline steps: {pipeline.get_pipeline_steps()}")
    
    # Fit and transform
    print("\nFitting pipeline on full dataset...")
    print("(Note: In production, fit only on train set)")
    X_processed = pipeline.fit_transform(X, y)
    
    # Add target back
    X_processed[target_col] = y.values
    
    # Save processed data
    print(f"\nSaving processed data to: {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    X_processed.to_csv(output_file, index=False)
    print(f"Processed data saved: {X_processed.shape[0]:,} rows x {X_processed.shape[1]} columns")
    
    # Save pipeline
    pipeline_path = models_dir / "preprocessing_pipeline.joblib"
    pipeline.save(str(pipeline_path))
    print(f"Pipeline saved to: {pipeline_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("PIPELINE SUMMARY")
    print("="*70)
    print(f"Original features: {len(X.columns)}")
    print(f"Processed features: {len([c for c in X_processed.columns if c not in [target_col, 'product_name', 'brands', 'code']])}")
    print(f"Target column: {target_col} (preserved)")
    print(f"Preserved columns: product_name, brands, code")
    
    # Show pipeline steps info
    print("\nPipeline Steps Applied:")
    for step_name in pipeline.get_pipeline_steps():
        transformer = pipeline.get_transformer(step_name)
        print(f"  - {step_name}: {type(transformer).__name__}")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
