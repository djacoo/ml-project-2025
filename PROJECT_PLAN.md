# Machine Learning Project 2025 - Project Plan

## Food Product Nutri-Score Prediction using Classical ML

**Course:** Machine Learning
**Academic Year:** 2025/2026

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Development Workflow](#development-workflow)
3. [Timeline & Milestones](#timeline--milestones)
4. [Tasks Breakdown](#tasks-breakdown)
5. [Repository Structure](#repository-structure)
6. [Setup Instructions](#setup-instructions)
7. [Progress Tracking](#progress-tracking)
8. [Deliverables Checklist](#deliverables-checklist)

---

## Project Overview

### Problem Statement
Predict the Nutri-Score (A, B, C, D, E) of food products based on nutritional information and product characteristics using classical machine learning algorithms.

### Data Source
- **API:** Open Food Facts (https://world.openfoodfacts.org/data)
- **Format:** CSV/JSONL
- **Expected Size:** 50,000-100,000 products

### Machine Learning Task
- **Type:** Multi-class Classification
- **Classes:** 5 (Nutri-Score: A, B, C, D, E)
- **Approach:** Classical ML

---

## Development Workflow

### Git Flow Structure

This project follows Git Flow methodology:

**Branches:**
- `main` - Production-ready code (stable releases)
- `dev` - Development branch (integration branch)
- `feature/*` - Feature branches for individual issues
- `bugfix/*` - Bug fix branches
- `hotfix/*` - Urgent production fixes

**Workflow:**
1. Create feature branch from `dev`: `git checkout -b feature/issue-X-description`
2. Work on issue and commit changes
3. Push and create PR to `dev` branch
4. After review and merge, delete feature branch
5. When phase complete, merge `dev` to `main`

### Issue Tracking

All work is tracked through GitHub Issues organized by milestones:

**Phase 1 Issues** (Milestone: Phase 1 - Data Preprocessing):
- Issue #1: Complete EDA with Statistical Analysis
- Issue #2: Handle Missing Values
- Issue #3: Remove Outliers and Invalid Values
- Issue #4: Normalize Numerical Features
- Issue #5: Encode Categorical Variables
- Issue #6: Create Derived Features
- Issue #7: Apply PCA for Dimensionality Reduction
- Issue #8: Create Train/Val/Test Splits

**Labels:**
- `phase-1`, `phase-2`, etc. - Phase organization
- `eda` - Exploratory data analysis
- `preprocessing` - Data preprocessing tasks
- `feature-engineering` - Feature engineering tasks

**Links:**
- [All Issues](https://github.com/djacoo/ml-project-2025/issues)
- [Milestones](https://github.com/djacoo/ml-project-2025/milestones)
- [Phase 1 Milestone](https://github.com/djacoo/ml-project-2025/milestone/1)

---

## Timeline & Milestones

### Data Acquisition & Exploration

- [x] Set up development environment
- [x] Create project repository structure
- [x] Set up Git Flow workflow (main, dev branches)
- [x] Create GitHub Issues and Milestones for Phase 1
- [x] Download Open Food Facts dataset (100,000 products)
- [x] Perform initial data exploration
- [x] Document dataset characteristics
- [x] Create EDA notebook (`notebooks/eda.ipynb`)

---

### Data Preprocessing & Feature Engineering

- [x] Handle missing values (Issue #2 - COMPLETED)
  - Implemented MissingValueHandler class
  - Dropped features with >95% missing (2 features)
  - Median imputation for nutritional features
  - Zero imputation for additives_n
  - 'unknown' placeholder for categorical features
  - Validation notebook with before/after analysis
  - 100% data retention (100,000 rows preserved)
- [ ] Remove outliers and invalid data (Issue #3 - Next)
- [ ] Normalize nutritional values per 100g
- [ ] Encode categorical variables
- [ ] Create derived features
- [ ] Apply dimensionality reduction (PCA)
- [ ] Prepare train/validation/test splits
- [ ] Document preprocessing pipeline

**Deliverable:** Clean dataset ready for modeling

---

### Model Training & Hyperparameter Tuning

- [ ] Implement baseline (Logistic Regression)
- [ ] Implement K-NN classifier
- [ ] Implement SVM with multiple kernels
- [ ] Implement Decision Tree
- [ ] Implement Random Forest
- [ ] Implement Gradient Boosting (XGBoost/LightGBM)
- [ ] Implement Naive Bayes
- [ ] Hyperparameter tuning for each model
- [ ] Cross-validation for all models
- [ ] Save trained models

**Deliverable:** Trained models with optimized parameters

---

### Evaluation & Analysis

- [ ] Calculate all evaluation metrics
- [ ] Generate confusion matrices
- [ ] Create ROC curves
- [ ] Perform feature importance analysis
- [ ] Analyze failure cases
- [ ] Create comparison visualizations
- [ ] Document results

**Deliverable:** Complete evaluation results

---

### Report Writing & Presentation Preparation

- [ ] Write technical report sections
- [ ] Create figures and tables for report
- [ ] Proofread and format report
- [ ] Create presentation slides
- [ ] Prepare demo/visualization
- [ ] Practice presentation
- [ ] Submit materials 1 week before exam

**Deliverable:** Final report + presentation + code

---

## Tasks Breakdown

### Phase 1: Setup & Infrastructure

#### Environment Setup
```bash
- [ ] Install Python 3.8+
- [ ] Create virtual environment
- [ ] Install required libraries
- [ ] Set up Jupyter Notebook
- [ ] Configure Git/GitHub
```

#### Repository Organization
```bash
- [ ] Create directory structure
- [ ] Initialize Git repository
- [ ] Create .gitignore
- [ ] Write README.md
- [ ] Set up requirements.txt
```

---

### Phase 2: Data Pipeline

#### Data Collection
```python
- [ ] Download Open Food Facts dataset
- [ ] Extract relevant product fields
- [ ] Filter products with Nutri-Score labels
- [ ] Save raw data locally
- [ ] Document data source and version
```

#### Exploratory Data Analysis
```python
- [ ] Dataset size and dimensions
- [ ] Class distribution (A/B/C/D/E)
- [ ] Missing value analysis
- [ ] Feature correlation analysis
- [ ] Statistical summaries
- [ ] Visualization of key features
```

#### Data Preprocessing
```python
- [ ] Missing value imputation
- [ ] Outlier detection and removal
- [ ] Feature normalization/scaling
- [ ] Categorical encoding
- [ ] Feature selection
- [ ] Train/test split (80/20)
```

#### Feature Engineering
```python
- [ ] Nutritional ratios (fat/protein, sugar/carb)
- [ ] Energy density calculation
- [ ] Nutrient balance scores
- [ ] Product category encoding
- [ ] PCA for dimensionality reduction
```

---

### Phase 3: Model Development

#### Model 1: Logistic Regression
```python
- [ ] Implement multi-class logistic regression
- [ ] Test regularization: L1, L2
- [ ] Hyperparameter tuning (C values)
- [ ] Cross-validation
- [ ] Evaluate and document results
```

#### Model 2: K-Nearest Neighbors
```python
- [ ] Implement K-NN classifier
- [ ] Test K values: [3, 5, 7, 9, 11]
- [ ] Test distance metrics
- [ ] Apply feature scaling
- [ ] Evaluate and document results
```

#### Model 3: Support Vector Machine
```python
- [ ] Implement SVM classifier
- [ ] Test kernels: linear, RBF, polynomial
- [ ] Grid search for C and gamma
- [ ] Handle multi-class classification
- [ ] Evaluate and document results
```

#### Model 4: Decision Tree
```python
- [ ] Implement Decision Tree
- [ ] Test max_depth values
- [ ] Test min_samples_split values
- [ ] Visualize tree structure
- [ ] Evaluate and document results
```

#### Model 5: Random Forest
```python
- [ ] Implement Random Forest
- [ ] Test n_estimators: [50, 100, 200]
- [ ] Extract feature importance
- [ ] Analyze out-of-bag scores
- [ ] Evaluate and document results
```

#### Model 6: Gradient Boosting
```python
- [ ] Implement XGBoost/LightGBM
- [ ] Tune learning rate
- [ ] Tune number of estimators
- [ ] Early stopping implementation
- [ ] Evaluate and document results
```

#### Model 7: Naive Bayes
```python
- [ ] Implement Gaussian Naive Bayes
- [ ] Test on normalized vs raw features
- [ ] Evaluate and document results
```

---

### Phase 4: Evaluation & Analysis

#### Performance Metrics
```python
- [ ] Accuracy for all models
- [ ] Precision, Recall, F1-Score (per class)
- [ ] Confusion matrices
- [ ] ROC curves and AUC scores
- [ ] Classification reports
- [ ] Training time comparison
```

#### Analysis
```python
- [ ] Model comparison table
- [ ] Feature importance ranking
- [ ] Learning curves
- [ ] Error analysis
- [ ] Failure case identification
- [ ] Visualization of results
```

---

### Phase 5: Documentation

#### Technical Report

#### Code Documentation
```python
- [ ] Docstrings for all functions
- [ ] Comments for complex logic
- [ ] README with usage instructions
- [ ] Requirements.txt
- [ ] Example usage notebooks
```

#### Presentation
```markdown
- [ ] Title slide
- [ ] Problem definition (1-2 slides)
- [ ] Dataset overview (1 slide)
- [ ] Methodology (2-3 slides)
- [ ] Results (2-3 slides)
- [ ] Demo/Visualization (1 slide)
- [ ] Conclusions (1 slide)
- [ ] Total: ~10 slides for 10 minutes
```

---

