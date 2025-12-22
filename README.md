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

### Phase 1: Data Acquisition & Preprocessing
- [x] Repository setup and structure
- [x] Data download (100k products from Open Food Facts)
- [x] Exploratory Data Analysis (EDA)
  - Statistical analysis of all features
  - Missing value analysis
  - Class distribution analysis
  - Feature correlation analysis
- [x] **Issue #2: Missing Value Handling** ✓
  - Implemented `MissingValueHandler` class in `src/data/preprocessing.py`
  - Comprehensive imputation strategy (median, constant, placeholder)
  - Validation notebook with before/after comparison
  - Zero missing values in final dataset (100,000 rows × 20 columns)
- [ ] Issue #3: Remove Outliers and Invalid Values (Next)
- [ ] Issue #4: Normalize Numerical Features
- [ ] Issue #5: Encode Categorical Variables
- [ ] Issue #6: Create Derived Features
- [ ] Issue #7: Apply PCA for Dimensionality Reduction
- [ ] Issue #8: Create Train/Val/Test Splits

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
