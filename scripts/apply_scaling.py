"""
Script to apply feature scaling.
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

from features.scaling import FeatureScaler

def main():
    project_root = Path(__file__).parent.parent
    input_file = project_root / "data" / "processed" / "openfoodfacts_cleaned.csv"
    output_file = project_root / "data" / "processed" / "openfoodfacts_scaled.csv"
    models_dir = project_root / "models"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("APPLYING SCALING")
    print("="*70)
    
    print(f"Loading: {input_file}")
    df = pd.read_csv(input_file)
    
    indices = np.arange(len(df))
    target_col = 'nutriscore_grade' 
    y = df[target_col] if target_col in df.columns else None
    train_idx, test_idx = train_test_split(
        indices, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    # creating the train subset based on the indices
    X_train = df.iloc[train_idx].copy()
    
    # Fit Scaler only on train set
    scaler_path = models_dir / "scaler.joblib"
    
    if scaler_path.exists():
        print(f"Loading existing scaler from {scaler_path}")
        scaler = FeatureScaler.load(str(scaler_path))
    else:
        print("Fitting new scaler on TRAIN subset")
        scaler = FeatureScaler(method='auto', skew_threshold=1.0)
        scaler.fit(X_train)
        scaler.save(str(scaler_path))
        print("Scaler fitted and saved")
        
    print("Applying transformation to the whole dataset")
    df_scaled = scaler.transform(df)
    
    #adding split group column to track train/test rows
    df_scaled['split_group'] = 'test'
    df_scaled.iloc[train_idx, df_scaled.columns.get_loc('split_group')] = 'train'
    
    #saving the scaled dataset
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_scaled.to_csv(output_file, index=False)
    
    print("\n" + "="*70)
    print(f"Scaled data saved to {output_file}")
    print(f"Column 'split_group' added to track train/test rows")
    print("="*70)

if __name__ == "__main__":
    main()