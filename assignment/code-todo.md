# Code Todo List

---

## 1. Setup stuff

### Getting started
- [ ] make project folders
- [ ] setup virtual env (venv or conda)
- [ ] requirements.txt with libraries I need:
  - [ ] numpy, pandas
  - [ ] scikit-learn
  - [ ] matplotlib, seaborn for plots
  - [ ] jupyter
  - [ ] others depending on what I do
- [ ] init git repo
- [ ] .gitignore (don't push data, models, pycache)
- [ ] basic README

### Dataset
- [ ] find dataset (kaggle or papers with code probably)
- [ ] download it
- [ ] organize in folders:
  - [ ] data/raw/ for original
  - [ ] data/processed/ for cleaned
- [ ] write down dataset info somewhere:
  - [ ] how many samples
  - [ ] number of features
  - [ ] classes (if classification)
  - [ ] format
- [ ] check if dataset has usage restrictions

---

## 2. Look at the data (EDA)

### First look
- [ ] notebook for EDA (maybe 01_eda.ipynb)
- [ ] load data and check it out
  - [ ] shape, head/tail
  - [ ] data types
  - [ ] memory usage
- [ ] basic stats (mean, median, std, etc.)

### Check data quality
- [ ] missing values?
  - [ ] which columns
  - [ ] how many
  - [ ] visualize patterns
- [ ] duplicates?
- [ ] outliers?
  - [ ] box plots
  - [ ] IQR or z-score
- [ ] if classification -> class balance
  - [ ] count per class
  - [ ] plot distribution

### Visualizations
- [ ] histograms for numeric stuff
- [ ] bar plots for categories
- [ ] correlation heatmap
- [ ] pair plots maybe
- [ ] target variable distribution
- [ ] specific viz depending on data type (images, text, time series, etc.)

---

## 3. Preprocessing

### Cleaning
- [ ] preprocessing notebook (02_preprocessing.ipynb)
- [ ] deal with missing values
  - [ ] remove rows? impute mean/median? forward fill?
  - [ ] implement and document why
- [ ] remove duplicates if any
- [ ] handle outliers (keep, remove, cap?)

### Feature stuff
- [ ] extract features if needed
  - [ ] images -> HoG, SIFT, color histograms
  - [ ] text -> TF-IDF, embeddings
  - [ ] tabular -> polynomial, interactions
- [ ] categorical variables
  - [ ] one-hot encoding
  - [ ] label encoding
  - [ ] target encoding?
- [ ] scaling
  - [ ] standardization (z-score)
  - [ ] min-max
  - [ ] robust scaler
- [ ] create new features if I can think of any

### Dimensionality reduction
- [ ] PCA if needed
  - [ ] decide how many components
  - [ ] plot explained variance
- [ ] maybe try t-SNE for viz
- [ ] LDA if classification
- [ ] feature selection methods

### Split data
- [ ] train/test split (70-30 or 80-20)
  - [ ] stratify if classification
- [ ] validation set if not doing CV
- [ ] save processed data
  - [ ] to data/processed/
  - [ ] csv or pickle

---

## 4. Models

### Baseline
- [ ] modeling notebook (03_modeling.ipynb)
- [ ] simple baseline
  - [ ] classification -> dummy classifier or logistic regression
  - [ ] regression -> mean predictor or linear regression
  - [ ] clustering -> random or simple k-means
- [ ] eval baseline
- [ ] write down metrics

### Pick models
- [ ] choose 2-3 models to compare
  - [ ] Model 1: ________________
  - [ ] Model 2: ________________
  - [ ] Model 3: ________________
- [ ] implement with default params first
- [ ] make it modular
  - [ ] train function
  - [ ] predict function
  - [ ] eval function

### Tune hyperparameters
- [ ] define search space for each model
- [ ] grid search or random search
- [ ] cross-validation (k=5 or 10)
  - [ ] stratified for classification
- [ ] document best params
- [ ] save models

### Train final models
- [ ] train with best params
- [ ] pipeline:
  - [ ] load data
  - [ ] train
  - [ ] save model
  - [ ] log stuff
- [ ] track time and memory

---

## 5. Evaluation

### Metrics
- [ ] pick metrics based on task
  - [ ] classification -> accuracy, precision, recall, f1, roc-auc
  - [ ] regression -> mse, rmse, mae, r2
  - [ ] clustering -> silhouette, davies-bouldin
- [ ] eval functions
- [ ] test all models
- [ ] comparison table

### Analysis
- [ ] confusion matrix (classification)
- [ ] per-class performance
- [ ] look at failures
  - [ ] visualize wrong predictions
  - [ ] find patterns in errors
- [ ] compare models
- [ ] statistical tests maybe

### Visualize results
- [ ] comparison plots
  - [ ] bar charts
  - [ ] roc curves
  - [ ] pred vs actual
- [ ] model behavior
  - [ ] feature importance
  - [ ] decision boundaries
  - [ ] clusters
- [ ] figures for report

### Cross-validation
- [ ] k-fold CV on final model
- [ ] mean and std of metrics
- [ ] plot CV results
  - [ ] box plots
  - [ ] learning curves
- [ ] check overfitting/underfitting

---

## 6. Clean up code

### Organization
- [ ] organize into modules
  - [ ] src/data/ for loading/preprocessing
  - [ ] src/features/
  - [ ] src/models/
  - [ ] src/visualization/
  - [ ] src/utils/
- [ ] main script (main.py or train.py)
  - [ ] with args
- [ ] move notebook code to scripts (optional but cleaner)

### Code quality
- [ ] docstrings for functions
  - [ ] params
  - [ ] returns
  - [ ] examples
- [ ] comments for tricky parts
- [ ] follow pep8
- [ ] remove unused stuff
- [ ] error handling
- [ ] type hints maybe

### Documentation
- [ ] update README with:
  - [ ] what the project does
  - [ ] how to install
  - [ ] how to run
  - [ ] dataset info
  - [ ] results
  - [ ] folder structure
- [ ] requirements.txt or environment.yml
- [ ] config file for hyperparams
- [ ] license if needed

### Reproducibility
- [ ] set random seeds everywhere
  - [ ] np.random.seed()
  - [ ] random.seed()
  - [ ] random_state in sklearn
- [ ] save preprocessing params (scaler, pca, etc.)
- [ ] document versions

---

## 7. Testing

### Test code
- [ ] test data loading
- [ ] test preprocessing
- [ ] test training
- [ ] test evaluation
- [ ] try different splits
- [ ] edge cases

### Full pipeline
- [ ] end to end script
  - [ ] raw data -> results
- [ ] test from scratch
- [ ] time it
- [ ] document steps

### Verify
- [ ] metrics are correct
- [ ] cross-check with different methods
- [ ] reproducible (same seed = same results)
- [ ] check output files

---

## 8. GitHub

### Repo structure
- [ ] proper folders:
  ```
  project/
  ├── data/
  │   ├── raw/
  │   ├── processed/
  │   └── README.md
  ├── notebooks/
  ├── src/
  │   ├── data/
  │   ├── features/
  │   ├── models/
  │   └── utils/
  ├── models/
  ├── results/
  ├── .gitignore
  ├── README.md
  ├── requirements.txt
  └── main.py
  ```

### Git stuff
- [ ] review files before commit
- [ ] no sensitive data
- [ ] no huge files (use git lfs if needed)
- [ ] good commit messages
- [ ] push to github

### Documentation
- [ ] good README
- [ ] data README
- [ ] comments in code
- [ ] example outputs

### Final checks
- [ ] clone in new folder and test
- [ ] all deps in requirements.txt
- [ ] installation works
- [ ] relative paths (not absolute)
- [ ] repo is public if needed
- [ ] add collaborator if group

---

## 9. Extra stuff (if there is time)

### Advanced
- [ ] ensemble methods
- [ ] interpretability (SHAP, LIME)
- [ ] interactive plots (plotly)
- [ ] streamlit dashboard maybe
- [ ] logging

### Optimization
- [ ] profile code
- [ ] optimize slow parts
- [ ] parallel processing
- [ ] better data structures
- [ ] cache results

### Deployment
- [ ] api for predictions
- [ ] docker
- [ ] web interface
- [ ] cloud deploy

---

## Before submission

- [ ] code documented
- [ ] all models done
- [ ] all metrics calculated
- [ ] reproducible
- [ ] github clean
- [ ] good README
- [ ] no plagiarism (cite sources)
- [ ] runs without errors
- [ ] both contributed if group
- [ ] link ready
- [ ] submit 1 week before exam

---