"""
Script to apply feature scaling to numerical features.

This script:
1. Loads cleaned dataset
2. Automatically chooses scaling method based on distribution skewness
3. Fits scaler on training data
4. Saves scaler for reuse

Usage:
    python scripts/apply_scaling.py
"""

import sys
from pathlib import Path
from scipy.stats import skew

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from features.scaling import FeatureScaler, NUMERICAL_FEATURES
from sklearn.model_selection import train_test_split
import pandas as pd


def choose_scaling_method(X_train):
    """Choose scaling method based on feature distribution skewness."""
    # Calculate skewness for each feature
    skewness = {
        f: skew(X_train[f].dropna()) 
        for f in NUMERICAL_FEATURES 
        if f in X_train.columns
    }
    
    # Count highly skewed features (|skew| > 1)
    #replace with a for loop
    highly_skewed = 0
    for s in skewness.values():
        if abs(s) > 1:
            highly_skewed += 1
    ratio = highly_skewed / len(skewness) if len(skewness) > 0 else 0
    
    # Choose method: MinMaxScaler if >50% features are highly skewed, else StandardScaler
    method = 'minmax' if ratio > 0.5 else 'standard'
    
    print(f"Highly skewed features: {highly_skewed}/{len(skewness)} ({ratio*100:.0f}%)")
    print(f"Chosen method: {method.upper()}")
    
    return method


def main():
    """Main execution function for feature scaling."""
    # Define paths
    project_root = Path(__file__).parent.parent
    input_file = project_root / "data" / "processed" / "openfoodfacts_cleaned.csv"
    scaler_dir = project_root / "models"
    
    print("="*70)
    print("FEATURE SCALING")
    print("="*70)
    
    # Load cleaned data
    print(f"\nLoading data from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
    
    # Prepare data
    X = df.drop(columns=['nutriscore_grade', 'code'])
    y = df['nutriscore_grade']
    
    # Split data (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    
    # Choose scaling method
    print(f"\nAnalyzing feature distributions...")
    method = choose_scaling_method(X_train)
    
    # Fit scaler on training data
    print(f"\nFitting scaler on training data...")
    scaler = FeatureScaler(method=method)
    scaler.fit(X_train)
    print(f"Scaler fitted")
    
    # Save scaler
    scaler_path = scaler_dir / f'scaler_{method}.pkl'
    scaler_dir.mkdir(parents=True, exist_ok=True)
    scaler.save(str(scaler_path))
    print(f"Scaler saved to: {scaler_path}")
    
    # Final summary
    print("\n" + "="*70)
    print("SCALING COMPLETE")
    print("="*70)
    print(f"Method: {method.upper()}")
    print(f"Features scaled: {len(NUMERICAL_FEATURES)}")
    print(f"Scaler saved: {scaler_path}")


if __name__ == "__main__":
    main()

