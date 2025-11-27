# Machine Learning Project 2025 - Project Plan

## Food Product Nutri-Score Prediction using Classical ML

**Course:** Machine Learning
**Academic Year:** 2025/2026

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Timeline & Milestones](#timeline--milestones)
3. [Tasks Breakdown](#tasks-breakdown)
4. [Repository Structure](#repository-structure)
5. [Setup Instructions](#setup-instructions)
6. [Progress Tracking](#progress-tracking)
7. [Deliverables Checklist](#deliverables-checklist)

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

## Timeline & Milestones

### Data Acquisition & Exploration

- [ ] Set up development environment
- [ ] Create project repository structure
- [ ] Download Open Food Facts dataset
- [ ] Perform initial data exploration
- [ ] Document dataset characteristics
- [ ] Create EDA notebook

**Deliverable:** EDA report with dataset statistics

---

### Data Preprocessing & Feature Engineering

- [ ] Handle missing values
- [ ] Remove outliers and invalid data
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

