# Legacy Scripts

This directory contains deprecated preprocessing scripts that have been replaced by the unified `PreprocessingPipeline` class.

## Deprecated Scripts

- `remove_outliers.py` - Replaced by `MissingValueTransformer` and `OutlierRemovalTransformer` in the pipeline
- `apply_encoding.py` - Replaced by `FeatureEncoder` in the pipeline
- `apply_scaling.py` - Replaced by `FeatureScaler` in the pipeline
- `apply_pca.py` - Replaced by `FeatureReducer` in the pipeline

## Why These Scripts Are Deprecated

The new `PreprocessingPipeline` class (`src/features/preprocessing_pipeline.py`) provides:
- **Unified interface**: All preprocessing steps in one pipeline
- **Consistency**: Ensures same transformations are applied to train/test/inference
- **Production-ready**: Can be saved/loaded for deployment
- **scikit-learn compatible**: Works seamlessly with sklearn workflows

## When to Use Legacy Scripts

These scripts are kept for:
- **Debugging**: Step-by-step validation of individual preprocessing steps
- **Development**: Testing changes to specific transformers
- **Documentation**: Understanding the original implementation

## Recommended Usage

**For normal preprocessing, use:**
```bash
python scripts/run_preprocessing_pipeline.py
```

**For debugging individual steps, you can still use these legacy scripts, but note:**
- They may not reflect the latest pipeline implementation
- They create intermediate files that the pipeline doesn't need
- They don't ensure consistency with the production pipeline

## Migration Guide

If you have workflows using these scripts, migrate to:

```python
from features.preprocessing_pipeline import PreprocessingPipeline

# Create and fit pipeline
pipeline = PreprocessingPipeline(include_pca=True)
X_processed = pipeline.fit_transform(X_train, y_train)

# Transform test data
X_test_processed = pipeline.transform(X_test)
```

See `docs/preprocessing_pipeline_usage.md` for complete documentation.
