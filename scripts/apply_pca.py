"""
Script to apply PCA dimensionality reduction.
Fits on a reproducible train split, applies to whole dataset,
and marks the split in the output file.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from features.dimensionality_reduction import FeatureReducer

def main():
    # Configure PCA parameters
    variance_threshold = 0.95  # Keep 95% of variance
    
    project_root = Path(__file__).parent.parent
    input_file = project_root / "data" / "processed" / "openfoodfacts_encoded.csv"
    output_file = project_root / "data" / "processed" / "openfoodfacts_pca.csv"
    models_dir = project_root / "models"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("APPLYING PCA DIMENSIONALITY REDUCTION")
    print("="*70)
    
    print(f"Loading: {input_file}")
    df = pd.read_csv(input_file)
    
    # Extract target and split_group
    target_col = 'nutriscore_grade'
    split_group_col = 'split_group'
    
    y = df[target_col]
    split_group = df[split_group_col].copy()
    
    # Remove non-numeric columns and metadata from X for PCA
    # Non-numeric columns: product_name, brands
    # Note: categories and main_category already removed in preprocessing
    # Note: energy_100g already removed in preprocessing
    # Metadata: nutriscore_grade (target), split_group
    non_numeric_cols = ['product_name', 'brands']
    
    # Filter to only existing columns
    existing_non_numeric = [col for col in non_numeric_cols if col in df.columns]
    X = df.drop(columns=[target_col, split_group_col] + existing_non_numeric, errors='ignore')
    
    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])
    
    # Store non-numeric columns to add back later
    non_numeric_data = df[existing_non_numeric].copy() if existing_non_numeric else pd.DataFrame()
    
    # Use existing split_group for train/test split
    print("Using existing 'split_group' column for train/test split")
    indices = np.arange(len(df))
    train_mask = split_group == 'train'
    train_idx = indices[train_mask]
    
    # Create train subset for fitting PCA
    X_train = X.iloc[train_idx].copy()
    
    # Fit PCA only on train set
    pca_path = models_dir / "pca.joblib"
    print(f"Fitting new PCA on TRAIN subset (variance threshold: {variance_threshold})")
    reducer = FeatureReducer(variance_threshold=variance_threshold)
    reducer.fit(X_train)
    reducer.save(str(pca_path))
    
    # Print PCA information
    n_components = reducer.n_components_selected_
    explained_variance = reducer.get_cumulative_variance()[-1]
    print(f"PCA fitted: {n_components} components selected")
    print(f"Explained variance: {explained_variance:.4f} ({explained_variance*100:.2f}%)")
    print("PCA saved")
    
    print("Applying transformation to the whole dataset")
    X_pca = reducer.transform(X)
    
    # Add back target, split_group, and non-numeric columns
    X_pca[target_col] = y.values
    X_pca[split_group_col] = split_group.values
    for col in non_numeric_data.columns:
        X_pca[col] = non_numeric_data[col].values
    
    # Save the PCA-transformed dataset
    output_file.parent.mkdir(parents=True, exist_ok=True)
    X_pca.to_csv(output_file, index=False)
    
    print("\n" + "="*70)
    print(f"PCA-transformed data saved to {output_file}")
    print(f"Original features: {X.shape[1]}")
    print(f"PCA components: {n_components}")
    print(f"Reduction: {X.shape[1] - n_components} features ({((X.shape[1] - n_components) / X.shape[1] * 100):.1f}%)")
    print("="*70)

if __name__ == "__main__":
    main()

