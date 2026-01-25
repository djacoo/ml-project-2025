# Machine Learning Project 2025

## Nutri-Score Prediction using Classical ML

This project predicts the Nutri-Score (A, B, C, D, E) of food products based on nutritional information using classical machine learning algorithms.

### Project Information

- **Course**: Machine Learning
- **Academic Year**: 2025/2026
- **Task**: Multi-class Classification (5 classes)
- **Dataset**: Open Food Facts (100,000 products)

## Repository Structure

```
ml-project-2025/
├── data/
│   ├── raw/              # Original downloaded data
│   └── processed/        # Cleaned and preprocessed data
├── notebooks/            # Jupyter notebooks for EDA and experiments
├── src/
│   ├── data/            # Data loading and preprocessing
│   ├── features/        # Feature engineering and scaling
│   ├── models/          # Model implementations
│   ├── evaluation/      # Evaluation metrics and visualization
│   └── utils/           # Helper functions
├── scripts/             # Standalone scripts (download, train, etc.)
├── models/              # Saved trained models
├── results/             # Evaluation results and figures
├── report/              # Technical report drafts
├── presentation/        # Presentation slides
└── tests/               # Unit tests

```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/djacoo/ml-project-2025.git
cd ml-project-2025
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Data

```bash
python scripts/download_data.py
```

### 5. Run Preprocessing Pipeline

**Recommended approach (using scikit-learn Pipeline):**

```bash
# 1. Handle missing values (imputation) - still needed as separate step
python src/data/preprocessing.py

# 2. Create train/validation/test splits
python scripts/split_data.py

# 3. Run complete preprocessing pipeline (includes: outlier removal, encoding, scaling, PCA)
python scripts/run_preprocessing_pipeline.py
```

The pipeline script applies all preprocessing steps in sequence:
1. Missing value handling (already done in step 1)
2. Outlier removal
3. Feature encoding (categorical → numerical)
4. Feature scaling (standardization/normalization)
5. PCA dimensionality reduction (optional)

**Output:** `data/processed/openfoodfacts_pipeline_processed.csv` with all preprocessing applied.

**Alternative (legacy step-by-step scripts - for debugging only):**

If you need to run preprocessing steps individually for debugging, legacy scripts are available in `scripts/legacy/`:
- `scripts/legacy/remove_outliers.py` - Outlier removal
- `scripts/legacy/apply_encoding.py` - Feature encoding
- `scripts/legacy/apply_scaling.py` - Feature scaling
- `scripts/legacy/apply_pca.py` - PCA reduction

**Note:** These scripts are deprecated and kept only for debugging purposes. The pipeline approach is recommended as it ensures consistency and is easier to use in production. See `scripts/legacy/README.md` for more details.

## Development Workflow

This project follows **Git Flow** methodology.

### Quick Start for Development

1. Always create feature branches from `dev`:
   ```bash
   git checkout dev
   git pull origin dev
   git checkout -b feature/issue-X-description
   ```

2. Work on your feature and commit:
   ```bash
   git add .
   git commit -m "feat: description (closes #X)"
   ```

3. Push and create pull request:
   ```bash
   git push -u origin feature/issue-X-description
   gh pr create --base dev
   ```

## Current Progress

### Phase 1: Data Acquisition & Preprocessing ✅ **COMPLETED**
- [x] Repository setup and structure
- [x] Data download (100k products from Open Food Facts)
- [x] Exploratory Data Analysis (EDA)
  - Statistical analysis of all features
  - Missing value analysis
  - Class distribution analysis
  - Feature correlation analysis
- [x] **Issue #2: Missing Value Handling**
  - Implemented `MissingValueHandler` class in `src/data/preprocessing.py`
  - Comprehensive imputation strategy (median, constant, placeholder)
  - Validation notebook with before/after comparison
  - Zero missing values in final dataset (100,000 rows × 20 columns)
- [x] **Issue #3: Remove Outliers and Invalid Values**
  - Implemented `OutlierHandler` class with domain-specific rules
  - Removed 3,903 rows (3.9%) with invalid nutritional values
  - Domain validation (energy 0-3000 kcal, macros 0-100g, salt 0-50g)
  - Statistical outlier detection (IQR method)
  - Final dataset: 96,097 rows × 20 columns, all values validated
- [x] **Issue #4: Normalize Numerical Features**
  - Implemented `FeatureScaler` class in `src/features/scaling.py`
  - StandardScaler applied to numerical features
  - Fitted on train split, applied to entire dataset
  - Scaler saved to `models/scaler.joblib`
  - Validation notebook: `notebooks/scaling_validation.ipynb`
- [x] **Issue #5: Encode Categorical Variables**
  - Implemented `FeatureEncoder` class in `src/features/encoding.py`
  - One-hot encoding for top 15 countries (others grouped as "other")
  - Encoder saved to `models/encoder.joblib`
  - Validation notebook: `notebooks/encoding_validation.ipynb`
- [x] **Issue #6: Create Derived Features**
  - Implemented `FeatureEngineer` class in `src/features/feature_engineering.py`
  - Created macro nutrient ratios (fat/protein, sugar/carb, saturated/total fat)
  - Added energy density and caloric contribution features
  - Created boolean flags for high nutrient levels (WHO recommendations)
  - Validation notebook: `notebooks/feature_engineering_validation.ipynb`
- [x] **Issue #7: Apply PCA for Dimensionality Reduction**
  - Implemented `FeatureReducer` class in `src/features/dimensionality_reduction.py`
  - PCA applied with 95% variance threshold
  - PCA model saved to `models/pca.joblib`
  - Validation notebook: `notebooks/pca_validation.ipynb`
- [x] **Issue #8: Create Train/Val/Test Splits**
  - Stratified split: 70% train, 15% validation, 15% test
  - Total samples: 98,468 (after preprocessing)
  - Train: 68,927 samples | Val: 14,770 samples | Test: 14,771 samples
  - Class distribution preserved across splits (max deviation: 0.000046)
  - Splits saved to `data/processed/splits/`
  - Metadata saved to `data/processed/split_metadata.json`

**Phase 1 Summary:** All preprocessing steps completed. Dataset ready for model training with 98,468 samples split into train/val/test sets. All preprocessing models (scaler, encoder, PCA) saved and ready for inference.

### Phase 2: Model Training
- [ ] 7 Classical ML models to implement

### Phase 3: Evaluation & Analysis
- [ ] Performance metrics and comparison

### Phase 4: Documentation
- [ ] Technical report (8-20 pages)
- [ ] Presentation slides (~10 min)

**Track progress:** [Phase 1 Milestone](https://github.com/djacoo/ml-project-2025/milestone/1)

## Models to Implement

1. Logistic Regression (baseline)
2. K-Nearest Neighbors
3. Support Vector Machine
4. Decision Tree
5. Random Forest
6. Gradient Boosting (XGBoost/LightGBM)
7. Naive Bayes

## Evaluation Metrics

- Accuracy
- Precision, Recall, F1-Score (per class)
- Confusion Matrix
- ROC Curves and AUC
- Training time comparison

## License

See [LICENSE](LICENSE) file for details.
