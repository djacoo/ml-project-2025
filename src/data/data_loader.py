"""
Data loading and splitting utilities for train/validation/test sets.

This module provides functions to:
- Split data into train/val/test sets with stratification
- Save and load split datasets
- Verify class distribution across splits
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split


def split_data(
    df: pd.DataFrame,
    target_col: str = 'nutriscore_grade',
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split dataset into train, validation, and test sets with stratification.
    
    Args:
        df: Input dataframe
        target_col: Name of target column
        train_ratio: Proportion of data for training (default: 0.7)
        val_ratio: Proportion of data for validation (default: 0.15)
        test_ratio: Proportion of data for testing (default: 0.15)
        random_state: Random seed for reproducibility
        stratify: Whether to stratify by target variable
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        
    Raises:
        ValueError: If ratios don't sum to 1.0
    """
    # Validate ratios
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0, atol=1e-6):
        raise ValueError(f"Ratios must sum to 1.0, got {ratio_sum}")
    
    # Extract target
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    # First split: train vs (val + test)
    train_size = train_ratio
    temp_size = val_ratio + test_ratio
    
    stratify_param = y if stratify else None
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=temp_size,
        random_state=random_state,
        stratify=stratify_param
    )
    
    # Second split: val vs test from temp
    # Adjust val_ratio for the temp subset
    val_size_adjusted = val_ratio / temp_size
    
    stratify_param_temp = y_temp if stratify else None
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_size_adjusted),  # test_ratio / temp_size
        random_state=random_state,
        stratify=stratify_param_temp
    )
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def add_split_group_column(
    df: pd.DataFrame,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray
) -> pd.DataFrame:
    """
    Add split_group column to dataframe indicating train/val/test assignment.
    
    Args:
        df: Input dataframe
        train_indices: Indices for training set
        val_indices: Indices for validation set
        test_indices: Indices for test set
        
    Returns:
        Dataframe with 'split_group' column added
    """
    df_with_split = df.copy()
    df_with_split['split_group'] = 'test'  # Default to test
    
    # Set train (using loc for index-based assignment)
    df_with_split.loc[train_indices, 'split_group'] = 'train'
    
    # Set val
    df_with_split.loc[val_indices, 'split_group'] = 'val'
    
    # Set test (already set, but explicit for clarity)
    df_with_split.loc[test_indices, 'split_group'] = 'test'
    
    return df_with_split


def verify_stratification(
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series
) -> Dict:
    """
    Verify that class distribution is balanced across splits.
    
    Args:
        y_train: Training target values
        y_val: Validation target values
        y_test: Test target values
        
    Returns:
        Dictionary with stratification statistics
    """
    # Calculate distributions
    train_dist = y_train.value_counts(normalize=True).sort_index()
    val_dist = y_val.value_counts(normalize=True).sort_index()
    test_dist = y_test.value_counts(normalize=True).sort_index()
    
    # Calculate counts
    train_counts = y_train.value_counts().sort_index()
    val_counts = y_val.value_counts().sort_index()
    test_counts = y_test.value_counts().sort_index()
    
    # Calculate differences (max deviation from train distribution)
    all_classes = sorted(set(y_train) | set(y_val) | set(y_test))
    max_deviation = 0.0
    
    for cls in all_classes:
        train_pct = train_dist.get(cls, 0.0)
        val_pct = val_dist.get(cls, 0.0)
        test_pct = test_dist.get(cls, 0.0)
        
        val_dev = abs(val_pct - train_pct)
        test_dev = abs(test_pct - train_pct)
        max_deviation = max(max_deviation, val_dev, test_dev)
    
    stats = {
        'train_distribution': train_dist.to_dict(),
        'val_distribution': val_dist.to_dict(),
        'test_distribution': test_dist.to_dict(),
        'train_counts': train_counts.to_dict(),
        'val_counts': val_counts.to_dict(),
        'test_counts': test_counts.to_dict(),
        'max_deviation': max_deviation,
        'is_balanced': max_deviation < 0.05  # Consider balanced if < 5% deviation
    }
    
    return stats


def save_splits(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path
) -> Dict[str, Path]:
    """
    Save train/val/test splits to disk as CSV files.
    
    Args:
        X_train, X_val, X_test: Feature dataframes
        y_train, y_val, y_test: Target series
        output_dir: Directory to save files
        
    Returns:
        Dictionary mapping split names to file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV
    X_train.to_csv(output_dir / 'X_train.csv', index=False)
    y_train.to_csv(output_dir / 'y_train.csv', index=False)
    X_val.to_csv(output_dir / 'X_val.csv', index=False)
    y_val.to_csv(output_dir / 'y_val.csv', index=False)
    X_test.to_csv(output_dir / 'X_test.csv', index=False)
    y_test.to_csv(output_dir / 'y_test.csv', index=False)
    
    saved_files = {
        'X_train': output_dir / 'X_train.csv',
        'y_train': output_dir / 'y_train.csv',
        'X_val': output_dir / 'X_val.csv',
        'y_val': output_dir / 'y_val.csv',
        'X_test': output_dir / 'X_test.csv',
        'y_test': output_dir / 'y_test.csv'
    }
    
    return saved_files


def load_train_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Load training data from CSV files."""
    X_train = pd.read_csv(data_dir / 'X_train.csv')
    y_train = pd.read_csv(data_dir / 'y_train.csv').squeeze()
    return X_train, y_train


def load_val_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Load validation data from CSV files."""
    X_val = pd.read_csv(data_dir / 'X_val.csv')
    y_val = pd.read_csv(data_dir / 'y_val.csv').squeeze()
    return X_val, y_val


def load_test_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Load test data from CSV files."""
    X_test = pd.read_csv(data_dir / 'X_test.csv')
    y_test = pd.read_csv(data_dir / 'y_test.csv').squeeze()
    return X_test, y_test


def save_split_metadata(
    stats: Dict,
    config: Dict,
    output_path: Path
) -> None:
    """
    Save split metadata (statistics and configuration) to JSON.
    
    Args:
        stats: Stratification statistics from verify_stratification
        config: Split configuration (ratios, random_state, etc.)
        output_path: Path to save metadata JSON
    """
    metadata = {
        'configuration': config,
        'statistics': {
            'train_size': {k: int(v) for k, v in stats['train_counts'].items()},
            'val_size': {k: int(v) for k, v in stats['val_counts'].items()},
            'test_size': {k: int(v) for k, v in stats['test_counts'].items()},
            'train_distribution': {k: float(v) for k, v in stats['train_distribution'].items()},
            'val_distribution': {k: float(v) for k, v in stats['val_distribution'].items()},
            'test_distribution': {k: float(v) for k, v in stats['test_distribution'].items()},
            'max_deviation': float(stats['max_deviation']),
            'is_balanced': bool(stats['is_balanced'])
        }
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)

