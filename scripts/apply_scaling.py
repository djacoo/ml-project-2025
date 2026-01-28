"""
Script to apply feature scaling.
Fits on a reproducible train split, applies to whole dataset,
and preserves the split_group column from the input dataset.
"""

import sys
import pandas as pd
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from features.scaling import FeatureScaler

def main():
    project_root = Path(__file__).parent.parent
    input_file = project_root / "data" / "processed" / "openfoodfacts_encoded.csv"
    output_file = project_root / "data" / "processed" / "openfoodfacts_scaled.csv"
    models_dir = project_root / "models"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("APPLYING SCALING")
    print("="*70)
    
    print(f"Loading: {input_file}")
    df = pd.read_csv(input_file)
    
    # Check if split_group exists
    split_group_col = 'split_group'
    if split_group_col not in df.columns:
        raise ValueError(
            f"'{split_group_col}' column not found in dataset. "
            f"Please run 'scripts/split_data.py' first to create train/val/test splits."
        )
    
    # Extract split_group
    split_group = df[split_group_col].copy()
    
    # Get train indices (for fitting scaler)
    train_mask = split_group == 'train'
    train_idx = df.index[train_mask]
    
    print(f"Using existing 'split_group' column:")
    print(f"  Train: {train_mask.sum():,} samples")
    print(f"  Val:   {(split_group == 'val').sum():,} samples")
    print(f"  Test:  {(split_group == 'test').sum():,} samples")
    
    # Create train subset for fitting scaler
    X_train = df.iloc[train_idx].copy()
    
    # Fit Scaler only on train set
    scaler_path = models_dir / "scaler.joblib"
    
    if scaler_path.exists():
        print(f"\nLoading existing scaler from {scaler_path}")
        scaler = FeatureScaler.load(str(scaler_path))
    else:
        print("\nFitting new scaler on TRAIN subset")
        scaler = FeatureScaler(method='auto', skew_threshold=1.0)
        scaler.fit(X_train)
        scaler.save(str(scaler_path))
        print("Scaler fitted and saved")
        
    print("Applying transformation to the whole dataset")
    df_scaled = scaler.transform(df)
    
    # Preserve split_group column
    df_scaled[split_group_col] = split_group.values
    
    # Save the scaled dataset
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_scaled.to_csv(output_file, index=False)
    
    print("\n" + "="*70)
    print(f"Scaled data saved to {output_file}")
    print(f"Column 'split_group' preserved from input dataset")
    print("="*70)

if __name__ == "__main__":
    main()