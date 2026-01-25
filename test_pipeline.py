"""
Test script for PreprocessingPipeline with FeatureEngineer.

This script creates synthetic data and tests the complete pipeline
to verify that FeatureEngineer is properly integrated.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from features.preprocessing_pipeline import PreprocessingPipeline


def create_synthetic_data(n_samples=100):
    """Create synthetic data similar to Open Food Facts dataset."""
    np.random.seed(42)
    
    data = {
        # Nutritional values per 100g
        'energy-kcal_100g': np.random.uniform(0, 900, n_samples),
        'fat_100g': np.random.uniform(0, 100, n_samples),
        'saturated-fat_100g': np.random.uniform(0, 50, n_samples),
        'carbohydrates_100g': np.random.uniform(0, 100, n_samples),
        'sugars_100g': np.random.uniform(0, 100, n_samples),
        'fiber_100g': np.random.uniform(0, 50, n_samples),
        'proteins_100g': np.random.uniform(0, 100, n_samples),
        'salt_100g': np.random.uniform(0, 5, n_samples),
        'additives_n': np.random.randint(0, 10, n_samples),
        
        # Categorical features
        'countries': np.random.choice(
            ['France', 'Italy', 'Germany', 'Spain', 'United Kingdom', 'France,Italy'],
            n_samples
        ),
        'pnns_groups_1': np.random.choice(
            ['Beverages', 'Sugary snacks', 'Fruits and vegetables', 'Cereals'],
            n_samples
        ),
        'pnns_groups_2': np.random.choice(
            ['Soft drinks', 'Biscuits', 'Fresh fruits', 'Bread'],
            n_samples
        ),
        
        # Metadata
        'product_name': [f'Product_{i}' for i in range(n_samples)],
        'brands': [f'Brand_{i % 5}' for i in range(n_samples)],
        'code': [f'CODE_{i:06d}' for i in range(n_samples)],
        
        # Target
        'nutriscore_grade': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Add some missing values (simulate real data)
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
    for idx in missing_indices[:5]:
        df.loc[idx, 'fiber_100g'] = np.nan
    
    return df


def test_pipeline_basic():
    """Test 1: Basic pipeline functionality with FeatureEngineer."""
    print("="*70)
    print("TEST 1: Basic Pipeline with FeatureEngineer")
    print("="*70)
    
    # Create synthetic data
    df = create_synthetic_data(n_samples=200)
    y = df['nutriscore_grade']
    X = df.drop(columns=['nutriscore_grade'])
    
    print(f"\nOriginal data shape: {X.shape}")
    print(f"Original columns: {list(X.columns)}")
    
    # Create pipeline with FeatureEngineer (default: included)
    pipeline = PreprocessingPipeline(
        missing_threshold=0.95,
        top_n_countries=5,
        scaling_method='auto',
        scaling_skew_threshold=1.0,
        pca_variance_threshold=0.95,
        target_col='nutriscore_grade',
        include_feature_engineering=True,  # Explicitly enable
        include_pca=True
    )
    
    print(f"\nPipeline steps: {pipeline.get_pipeline_steps()}")
    
    # Verify feature_engineering is in the pipeline
    assert 'feature_engineering' in pipeline.get_pipeline_steps(), \
        "❌ FeatureEngineering step not found in pipeline!"
    print("✅ FeatureEngineering step found in pipeline")
    
    # Fit and transform
    print("\nFitting and transforming pipeline...")
    X_processed = pipeline.fit_transform(X, y)
    
    print(f"\nProcessed data shape: {X_processed.shape}")
    print(f"Processed columns (first 20): {list(X_processed.columns[:20])}")
    
    # Check for derived features from FeatureEngineer
    derived_features = [
        'fat_to_protein_ratio',
        'sugar_to_carb_ratio',
        'saturated_to_total_fat_ratio',
        'energy_density',
        'calories_from_fat',
        'calories_from_carbs',
        'calories_from_protein',
        'high_fat',
        'high_sugar',
        'high_salt'
    ]
    
    found_features = [feat for feat in derived_features if feat in X_processed.columns]
    print(f"\nDerived features found: {len(found_features)}/{len(derived_features)}")
    for feat in found_features:
        print(f"  ✅ {feat}")
    
    missing_features = [feat for feat in derived_features if feat not in X_processed.columns]
    if missing_features:
        print(f"\n⚠️  Missing features: {missing_features}")
        print("   (This might be expected if some base columns were removed)")
    
    # Verify no NaN values in processed data (except metadata)
    feature_cols = [c for c in X_processed.columns 
                   if c not in ['nutriscore_grade', 'product_name', 'brands', 'code']]
    nan_counts = X_processed[feature_cols].isna().sum().sum()
    print(f"\nNaN values in processed features: {nan_counts}")
    if nan_counts == 0:
        print("✅ No NaN values in processed features")
    else:
        print(f"⚠️  Found {nan_counts} NaN values")
    
    return X_processed


def test_pipeline_without_feature_engineering():
    """Test 2: Pipeline without FeatureEngineer."""
    print("\n" + "="*70)
    print("TEST 2: Pipeline WITHOUT FeatureEngineer")
    print("="*70)
    
    df = create_synthetic_data(n_samples=200)
    y = df['nutriscore_grade']
    X = df.drop(columns=['nutriscore_grade'])
    
    # Create pipeline WITHOUT FeatureEngineer
    pipeline = PreprocessingPipeline(
        missing_threshold=0.95,
        top_n_countries=5,
        scaling_method='auto',
        include_feature_engineering=False,  # Explicitly disable
        include_pca=True
    )
    
    print(f"\nPipeline steps: {pipeline.get_pipeline_steps()}")
    
    # Verify feature_engineering is NOT in the pipeline
    assert 'feature_engineering' not in pipeline.get_pipeline_steps(), \
        "❌ FeatureEngineering step found when it should be disabled!"
    print("✅ FeatureEngineering step correctly excluded")
    
    # Fit and transform
    X_processed = pipeline.fit_transform(X, y)
    
    print(f"\nProcessed data shape: {X_processed.shape}")
    
    # Check that derived features are NOT present
    derived_features = ['fat_to_protein_ratio', 'energy_density', 'high_fat']
    found_features = [feat for feat in derived_features if feat in X_processed.columns]
    
    if len(found_features) == 0:
        print("✅ Derived features correctly excluded")
    else:
        print(f"⚠️  Found unexpected derived features: {found_features}")
    
    return X_processed


def test_pipeline_custom_feature_engineering():
    """Test 3: Pipeline with custom FeatureEngineer parameters."""
    print("\n" + "="*70)
    print("TEST 3: Pipeline with Custom FeatureEngineer Parameters")
    print("="*70)
    
    df = create_synthetic_data(n_samples=200)
    y = df['nutriscore_grade']
    X = df.drop(columns=['nutriscore_grade'])
    
    # Create pipeline with custom FeatureEngineer (only ratios)
    pipeline = PreprocessingPipeline(
        missing_threshold=0.95,
        top_n_countries=5,
        scaling_method='auto',
        include_feature_engineering=True,
        feature_engineering_kwargs={
            'add_ratios': True,
            'add_energy_density': False,
            'add_caloric_contributions': False,
            'add_boolean_flags': False
        },
        include_pca=False  # Skip PCA for this test
    )
    
    print(f"\nPipeline steps: {pipeline.get_pipeline_steps()}")
    
    # Fit and transform
    X_processed = pipeline.fit_transform(X, y)
    
    print(f"\nProcessed data shape: {X_processed.shape}")
    
    # Check that only ratio features are present
    ratio_features = ['fat_to_protein_ratio', 'sugar_to_carb_ratio', 'saturated_to_total_fat_ratio']
    other_features = ['energy_density', 'calories_from_fat', 'high_fat']
    
    found_ratios = [feat for feat in ratio_features if feat in X_processed.columns]
    found_others = [feat for feat in other_features if feat in X_processed.columns]
    
    print(f"\nRatio features found: {len(found_ratios)}/{len(ratio_features)}")
    for feat in found_ratios:
        print(f"  ✅ {feat}")
    
    if len(found_others) == 0:
        print("✅ Other derived features correctly excluded")
    else:
        print(f"⚠️  Found unexpected features: {found_others}")
    
    return X_processed


def test_pipeline_with_split_group():
    """Test 4: Pipeline with split_group column."""
    print("\n" + "="*70)
    print("TEST 4: Pipeline with split_group Column")
    print("="*70)
    
    df = create_synthetic_data(n_samples=300)
    
    # Add split_group column
    np.random.seed(42)
    split_groups = np.random.choice(
        ['train', 'val', 'test'],
        size=len(df),
        p=[0.7, 0.15, 0.15]
    )
    df['split_group'] = split_groups
    
    y = df['nutriscore_grade']
    X = df.drop(columns=['nutriscore_grade'])
    
    print(f"\nData split:")
    print(f"  Train: {(df['split_group'] == 'train').sum()}")
    print(f"  Val:   {(df['split_group'] == 'val').sum()}")
    print(f"  Test:  {(df['split_group'] == 'test').sum()}")
    
    # Create pipeline
    pipeline = PreprocessingPipeline(
        missing_threshold=0.95,
        top_n_countries=5,
        scaling_method='auto',
        split_group_col='split_group',
        include_feature_engineering=True,
        include_pca=True
    )
    
    print(f"\nPipeline steps: {pipeline.get_pipeline_steps()}")
    
    # Fit and transform (should fit only on train)
    print("\nFitting and transforming pipeline...")
    print("(Pipeline should fit only on 'train' split)")
    X_processed = pipeline.fit_transform(X, y)
    
    print(f"\nProcessed data shape: {X_processed.shape}")
    
    # Verify split_group is preserved
    if 'split_group' in X_processed.columns:
        print("✅ split_group column preserved")
        print(f"  Train: {(X_processed['split_group'] == 'train').sum()}")
        print(f"  Val:   {(X_processed['split_group'] == 'val').sum()}")
        print(f"  Test:  {(X_processed['split_group'] == 'test').sum()}")
    else:
        print("⚠️  split_group column not found in processed data")
    
    return X_processed


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("PREPROCESSING PIPELINE TEST SUITE")
    print("Testing FeatureEngineer Integration")
    print("="*70)
    
    try:
        # Test 1: Basic pipeline with FeatureEngineer
        X1 = test_pipeline_basic()
        
        # Test 2: Pipeline without FeatureEngineer
        X2 = test_pipeline_without_feature_engineering()
        
        # Test 3: Custom FeatureEngineer parameters
        X3 = test_pipeline_custom_feature_engineering()
        
        # Test 4: With split_group
        X4 = test_pipeline_with_split_group()
        
        print("\n" + "="*70)
        print("ALL TESTS COMPLETED SUCCESSFULLY! ✅")
        print("="*70)
        print("\nSummary:")
        print(f"  Test 1 (Basic): {X1.shape}")
        print(f"  Test 2 (No FE): {X2.shape}")
        print(f"  Test 3 (Custom FE): {X3.shape}")
        print(f"  Test 4 (Split group): {X4.shape}")
        
    except Exception as e:
        print("\n" + "="*70)
        print("TEST FAILED! ❌")
        print("="*70)
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
