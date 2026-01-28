"""
Open Food Facts Data Downloader

This script downloads the Open Food Facts dataset, filters it for products
with Nutri-Score labels, and saves it.

Usage:
    python scripts/download_data.py

Author: Parretti Jacopo VR536104, Fraccaroli Cesare VR533061
"""

import requests
import pandas as pd
import gzip
import shutil
import os
from pathlib import Path
from tqdm import tqdm
import json

# config
DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Create directories
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Open FoodFacts dataset
OPENFOODFACTS_URL = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
RAW_FILE = RAW_DATA_DIR / "openfoodfacts_raw.csv.gz"
FILTERED_FILE = PROCESSED_DATA_DIR / "openfoodfacts_filtered.csv"

# Columns for Nutri-Score prediction
RELEVANT_COLUMNS = [
    # Target variable
    'nutriscore_grade',

    # Product identification
    'code',
    'product_name',
    'brands',
    'categories',
    'countries',

    # Nutritional values per 100g
    'energy_100g',
    'energy-kcal_100g',
    'fat_100g',
    'saturated-fat_100g',
    'carbohydrates_100g',
    'sugars_100g',
    'fiber_100g',
    'proteins_100g',
    'salt_100g',
    'sodium_100g',

    # Additional nutritional information
    'fruits-vegetables-nuts_100g',
    'fruits-vegetables-nuts-estimate_100g',

    # Product characteristics
    'additives_n',
    'ingredients_n',
    'nutrition_grade_fr',
    'pnns_groups_1',
    'pnns_groups_2',
    'main_category',
]


def download_dataset(url, output_path):
    """
    Download the Open Food Facts dataset from the given URL.

    Args:
        url (str): URL to download from
        output_path (Path): Path where to save the downloaded file to

    Returns:
        bool: True if download successful, False otherwise
    """
    try:
        print(f"Downloading dataset from {url}...")

        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)

        print(f"Dataset downloaded successfully to {output_path}")
        return True

    except Exception as e:
        print(f"✗ Error downloading dataset: {e}")
        return False


def load_and_filter_data(input_path, output_path, sample_size=None, chunksize=10000):
    """
    Load the raw dataset, filter for relevant data, and save.
    Processes data in chunks to avoid loading entire dataset into memory.

    Args:
        input_path (Path): Path to the raw compressed CSV file
        output_path (Path): Path where to save the filtered CSV
        sample_size (int, optional): If provided, randomly sample this many rows
        chunksize (int): Number of rows to process at a time (default: 10000)

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    try:
        print("\nLoading and filtering dataset...")
        print(f"Processing in chunks of {chunksize:,} rows to avoid memory issues...")

        # Initialize counters and storage
        total_rows = 0
        filtered_rows = 0
        valid_rows = 0
        grade_counts = {}
        missing_columns = []
        existing_columns = None
        all_filtered_chunks = []

        # Read the compressed CSV in chunks
        print("Reading CSV file in chunks...")
        chunk_iterator = pd.read_csv(
            input_path,
            compression='gzip',
            sep='\t',
            low_memory=False,
            on_bad_lines='skip',  # Skip malformed rows
            chunksize=chunksize
        )

        # Process each chunk
        for df_chunk in tqdm(chunk_iterator, desc="Processing chunks"):
            total_rows += len(df_chunk)

            # Determine existing columns on first chunk
            if existing_columns is None:
                existing_columns = [col for col in RELEVANT_COLUMNS if col in df_chunk.columns]
                missing_columns = [col for col in RELEVANT_COLUMNS if col not in df_chunk.columns]

            # Filter for products with Nutri-Score labels
            df_filtered = df_chunk[df_chunk['nutriscore_grade'].notna()].copy()
            filtered_rows += len(df_filtered)

            if len(df_filtered) == 0:
                continue

            # Keep only relevant columns
            df_filtered = df_filtered[existing_columns]

            # Filter for valid Nutri-Score grades (a, b, c, d, e)
            valid_grades = ['a', 'b', 'c', 'd', 'e']
            df_filtered = df_filtered[df_filtered['nutriscore_grade'].str.lower().isin(valid_grades)]
            valid_rows += len(df_filtered)

            if len(df_filtered) == 0:
                continue

            # Update grade counts
            chunk_grade_counts = df_filtered['nutriscore_grade'].str.upper().value_counts()
            for grade, count in chunk_grade_counts.items():
                grade_counts[grade] = grade_counts.get(grade, 0) + count

            # Collect chunks for later processing
            all_filtered_chunks.append(df_filtered)

        print(f"Processed {total_rows:,} total products")
        print(f"Found {filtered_rows:,} products with Nutri-Score labels")
        print(f"Filtered to {valid_rows:,} products with valid Nutri-Score (a-e)")

        if missing_columns:
            print(f"\nNote: The following columns are not in the dataset:")
            for col in missing_columns:
                print(f"  - {col}")

        print(f"Kept {len(existing_columns)} relevant columns")

        # Combine all filtered chunks
        print(f"\nCombining filtered chunks...")
        df_filtered = pd.concat(all_filtered_chunks, ignore_index=True)
        
        # Sample if requested
        if sample_size and len(df_filtered) > sample_size:
            df_filtered = df_filtered.sample(n=sample_size, random_state=42)
            print(f"Randomly sampled {sample_size:,} products")
            
            # Recalculate grade distribution for sampled data
            grade_counts = df_filtered['nutriscore_grade'].str.upper().value_counts().sort_index()
        
        # Save filtered data
        print(f"\nSaving filtered dataset to {output_path}...")
        df_filtered.to_csv(output_path, index=False)
        print(f"Filtered dataset saved successfully")

        # Display class distribution
        print("\nNutri-Score distribution:")
        grade_counts_sorted = dict(sorted(grade_counts.items()))
        for grade, count in grade_counts_sorted.items():
            percentage = (count / valid_rows) * 100 if valid_rows > 0 else 0
            print(f"  Grade {grade}: {count:,} ({percentage:.1f}%)")

        # Save metadata
        metadata = {
            'total_products': total_rows,
            'filtered_products': valid_rows,
            'columns': len(existing_columns),
            'missing_columns': missing_columns,
            'grade_distribution': grade_counts_sorted,
            'data_source': OPENFOODFACTS_URL,
            'download_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        metadata_path = PROCESSED_DATA_DIR / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {metadata_path}")

        return df_filtered

    except Exception as e:
        print(f"✗ Error processing dataset: {e}")
        raise


def main():
    """Main execution function."""
    print("=" * 70)
    print("Open Food Facts Data Acquisition")
    print("=" * 70)

    # Check if filtered data already exists
    if FILTERED_FILE.exists():
        print(f"\n⚠ Filtered dataset already exists at {FILTERED_FILE}")
        print("Exiting without re-downloading.")
        return

    # Download dataset
    if not RAW_FILE.exists():
        success = download_dataset(OPENFOODFACTS_URL, RAW_FILE)
        if not success:
            print("\nFailed to download dataset. Exiting.")
            return
    else:
        print(f"Raw dataset already exists at {RAW_FILE}")

    # Load and filter data
    df = load_and_filter_data(RAW_FILE, FILTERED_FILE, sample_size=100000)

    print("\n" + "=" * 70)
    print("Data Acquisition Complete!")
    print("=" * 70)
    print(f"\nFiltered dataset location: {FILTERED_FILE}")
    print(f"Dataset shape: {df.shape}")
    print("\nNext steps:")
    print("  1. Run the EDA notebook: notebooks/01_exploratory_data_analysis.ipynb")
    print("  2. Review the data characteristics")
    print("  3. Proceed with data preprocessing and feature engineering")


if __name__ == "__main__":
    main()
