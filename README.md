# Machine Learning Project

Machine Learning course project - A.Y. 2025/2026

## Project Description

[Brief description of the project - to be added]

## Dataset

This project uses the **Fashion-MNIST** dataset, a collection of Zalando's article images consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image associated with a label from 10 classes.

### Classes
0. T-shirt/top
1. Trouser
2. Pullover
3. Dress
4. Coat
5. Sandal
6. Shirt
7. Sneaker
8. Bag
9. Ankle boot

### Dataset Files
The dataset is located in `data/raw/` and includes:
- `fashion-mnist_train.csv` - Training data in CSV format (60,000 samples)
- `fashion-mnist_test.csv` - Test data in CSV format (10,000 samples)
- Original IDX format files (train-images-idx3-ubyte, train-labels-idx1-ubyte, etc.)

## Project Structure

```
ml-project-2025/
├── data/
│   ├── raw/              # Original, immutable data
│   ├── processed/        # Cleaned and processed data
│   └── interim/          # Intermediate data transformations
├── notebooks/
│   ├── 01_eda.ipynb     # Exploratory data analysis
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── data/            # Scripts for data processing
│   ├── features/        # Feature engineering
│   ├── models/          # Model training and prediction
│   ├── visualization/   # Plotting and visualization
│   └── utils/           # Utility functions
├── models/
│   └── saved_models/    # Trained model files
├── results/
│   ├── figures/         # Generated plots and figures
│   └── metrics/         # Performance metrics
├── assignment/          # Project requirements and todo lists
├── requirements.txt     # Python dependencies
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/djacoo/ml-project-2025.git
cd ml-project-2025
```

2. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```
