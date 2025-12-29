"""
Script to apply feature encoding.
Fits on a reproducible train split, applies to whole dataset,
and marks the split in the output file.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from features.encoding import FeatureEncoder

def main():
    # Configure number of top countries to keep
    top_n_countries = 15
    project_root = Path(__file__).parent.parent
    input_file = project_root / "data" / "processed" / "openfoodfacts_scaled.csv"
    output_file = project_root / "data" / "processed" / "openfoodfacts_encoded.csv"
    models_dir = project_root / "models"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("APPLYING ENCODING")
    print("="*70)
    
    print(f"Loading: {input_file}")
    df = pd.read_csv(input_file)
    
    # Extract target and split_group
    target_col = 'nutriscore_grade'
    split_group_col = 'split_group'
    
    y = df[target_col]
    split_group = df[split_group_col].copy()
    
    # Remove split_group and target from X for encoding
    X = df.drop(columns=[target_col, split_group_col])
    
    # Use existing split_group for train/test split
    print("Using existing 'split_group' column for train/test split")
    indices = np.arange(len(df))
    train_mask = split_group == 'train'
    train_idx = indices[train_mask]
    
    # Create train subset for fitting encoder
    X_train = X.iloc[train_idx].copy()
    y_train = y.iloc[train_idx]
    
    # Fit Encoder only on train set
    encoder_path = models_dir / "encoder.joblib"
    print(f"Fitting new encoder on TRAIN subset (top {top_n_countries} countries)")
    encoder = FeatureEncoder(top_n_countries=top_n_countries)
    encoder.fit(X_train, y=y_train)
    encoder.save(str(encoder_path))
    print("Encoder fitted and saved")
    
    print("Applying transformation to the whole dataset")
    df_encoded = encoder.transform(X)
    
    # Add back target and split_group columns
    df_encoded[target_col] = y.values
    df_encoded[split_group_col] = split_group.values
    
    # Save the encoded dataset
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_encoded.to_csv(output_file, index=False)
    
    print("\n" + "="*70)
    print(f"Encoded data saved to {output_file}")
    print(f"Column 'split_group' added to track train/test rows")
    print("="*70)

if __name__ == "__main__":
    main()

