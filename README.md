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

This project follows **Git Flow** methodology. See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed workflow instructions.

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

- [x] Repository setup and structure
- [x] Data download (100k products)
- [x] Initial EDA started
- [ ] Phase 1: Data Preprocessing (In Progress)
  - See [milestone](https://github.com/djacoo/ml-project-2025/milestone/1)
- [ ] Phase 2: Model Training
- [ ] Phase 3: Evaluation
- [ ] Phase 4: Report Writing

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

## Documentation

- **Project Plan**: [PROJECT_PLAN.md](PROJECT_PLAN.md)
- **Issues**: [GitHub Issues](https://github.com/djacoo/ml-project-2025/issues)
- **Milestones**: [GitHub Milestones](https://github.com/djacoo/ml-project-2025/milestones)
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md)

## Contact

For questions or issues, please open an issue on GitHub.

## License

See [LICENSE](LICENSE) file for details.
