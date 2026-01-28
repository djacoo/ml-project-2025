# Preprocessing Pipeline Usage Guide

## Overview

The `PreprocessingPipeline` class provides a complete scikit-learn compatible pipeline that combines all preprocessing steps:

1. **Missing Value Handling** - Imputation and feature dropping
2. **Outlier Removal** - Domain rules and statistical outlier detection
3. **Feature Encoding** - Categorical encoding (one-hot, target encoding)
4. **Feature Scaling** - Standardization/normalization
5. **PCA Dimensionality Reduction** - Optional dimensionality reduction

## Quick Start

### Basic Usage

```python
from features.preprocessing_pipeline import PreprocessingPipeline
import pandas as pd

# Load your data
df = pd.read_csv('data/processed/openfoodfacts_filtered.csv')
y = df['nutriscore_grade']
X = df.drop(columns=['nutriscore_grade'])

# Create and use pipeline
pipeline = PreprocessingPipeline(
    missing_threshold=0.95,
    top_n_countries=15,
    scaling_method='auto',
    include_pca=True
)

# Fit and transform
X_processed = pipeline.fit_transform(X, y)
X_processed['nutriscore_grade'] = y.values
```

### With Train/Test Split

```python
from sklearn.model_selection import train_test_split
from features.preprocessing_pipeline import PreprocessingPipeline

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create pipeline
pipeline = PreprocessingPipeline(include_pca=True)

# Fit on training data only
pipeline.fit(X_train, y_train)

# Transform both sets
X_train_processed = pipeline.transform(X_train, y_train)
X_test_processed = pipeline.transform(X_test, y_test)
```

### With Existing Split Group

```python
# Load data with split_group column
df = pd.read_csv('data/processed/openfoodfacts_cleaned.csv')

# Separate by split
train_mask = df['split_group'] == 'train'
X_train = df[train_mask].drop(columns=['nutriscore_grade', 'split_group'])
y_train = df[train_mask]['nutriscore_grade']

# Create pipeline with split_group support
pipeline = PreprocessingPipeline(
    split_group_col='split_group',
    preserve_cols=['product_name', 'brands']
)

# Fit and transform
pipeline.fit(X_train, y_train)
X_train_processed = pipeline.transform(X_train, y_train)
```

## API Reference

### PreprocessingPipeline

#### Parameters

- `missing_threshold` (float, default=0.95): Threshold for dropping features with high missing percentage
- `top_n_countries` (int, default=15): Number of top countries to keep for encoding
- `scaling_method` (str, default='auto'): Scaling method ('auto', 'standard', 'minmax', 'robust')
- `scaling_skew_threshold` (float, default=1.0): Skewness threshold for auto scaling
- `pca_variance_threshold` (float, default=0.95): Variance threshold for PCA (0.0 to 1.0)
- `target_col` (str, default='nutriscore_grade'): Name of target column
- `split_group_col` (str, default='split_group'): Name of split group column
- `preserve_cols` (list, default=None): List of columns to preserve (e.g., ['product_name', 'brands'])
- `include_pca` (bool, default=True): Whether to include PCA in the pipeline

#### Methods

- `fit(X, y)`: Fit the pipeline on training data
- `transform(X, y=None)`: Transform data using fitted pipeline
- `fit_transform(X, y)`: Fit and transform in one step
- `get_pipeline_steps()`: Get list of pipeline step names
- `get_transformer(step_name)`: Get a specific transformer by name
- `save(path)`: Save the pipeline to disk
- `load(path)`: Load a saved pipeline from disk

### create_preprocessing_pipeline()

Function to create a raw sklearn Pipeline (without the high-level wrapper).

```python
from features.preprocessing_pipeline import create_preprocessing_pipeline

pipeline = create_preprocessing_pipeline(
    missing_threshold=0.95,
    top_n_countries=15,
    scaling_method='auto',
    include_pca=True
)

# Use like any sklearn Pipeline
X_processed = pipeline.fit_transform(X_train, y_train)
X_test_processed = pipeline.transform(X_test)
```

## Pipeline Steps

The pipeline consists of the following steps (in order):

1. **missing_values** - `MissingValueTransformer`
   - Analyzes and handles missing values
   - Drops features with >95% missing (configurable)
   - Imputes numerical features with median
   - Imputes categorical features with 'unknown'

2. **outlier_removal** - `OutlierRemovalTransformer`
   - Removes invalid values (negative, out of range)
   - Applies domain-specific rules
   - Removes statistical outliers (3×IQR)

3. **encoding** - `FeatureEncoder`
   - MultiLabelBinarizer for countries (top N)
   - OneHotEncoder for pnns_groups_1
   - TargetEncoder for pnns_groups_2 (requires y)

4. **scaling** - `FeatureScaler`
   - Automatically scales all numerical features
   - Chooses StandardScaler or MinMaxScaler based on skewness (if method='auto')
   - Excludes target, split_group, and preserved columns

5. **pca** - `FeatureReducer` (optional)
   - PCA dimensionality reduction
   - Selects components to explain variance threshold
   - Default: 95% variance explained

## Examples

See `examples/preprocessing_pipeline_example.py` for complete examples including:
- Basic usage
- Train/test split
- Using split_group column
- Custom configuration

## Notes

- **Target Encoding**: The `FeatureEncoder` uses TargetEncoder for `pnns_groups_2`, which requires `y` during `fit()`. Always pass `y` when fitting the pipeline.
- **Data Leakage Prevention**: Always fit the pipeline only on training data, then transform validation and test sets.
- **Preserved Columns**: Columns like `product_name`, `brands`, `code` are preserved through the pipeline and added back to the output.
- **Split Group**: If `split_group_col` is specified, the column is preserved and added back to transformed data.

## Integration with Model Training

```python
from features.preprocessing_pipeline import PreprocessingPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create and fit preprocessing pipeline
preprocessing = PreprocessingPipeline(include_pca=True)
X_train_processed = preprocessing.fit_transform(X_train, y_train)
X_test_processed = preprocessing.transform(X_test)

# Remove non-feature columns for model training
feature_cols = [c for c in X_train_processed.columns 
                if c not in ['nutriscore_grade', 'split_group', 'product_name', 'brands']]
X_train_features = X_train_processed[feature_cols]
X_test_features = X_test_processed[feature_cols]

# Train model
model = RandomForestClassifier()
model.fit(X_train_features, y_train)

# Evaluate
y_pred = model.predict(X_test_features)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
```
