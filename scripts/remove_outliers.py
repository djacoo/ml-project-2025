"""
Script to remove outliers and invalid values from the preprocessed dataset.

This script applies domain-specific rules and statistical methods to clean
nutritional data, removing impossible values and extreme outliers.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.preprocessing import OutlierHandler
import pandas as pd


def main():
    """Main execution function for outlier removal."""
    # Define paths
    project_root = Path(__file__).parent.parent
    input_file = project_root / "data" / "processed" / "openfoodfacts_preprocessed.csv"
    output_file = project_root / "data" / "processed" / "openfoodfacts_cleaned.csv"
    report_file = project_root / "data" / "processed" / "outlier_removal_report.json"

    print("="*70)
    print("OUTLIER REMOVAL PROCESS")
    print("="*70)

    # Load preprocessed data
    print(f"\nLoading data from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"✓ Loaded {len(df):,} rows × {len(df.columns)} columns")

    # Initialize outlier handler
    handler = OutlierHandler()

    # Remove outliers
    df_clean = handler.remove_outliers(df)

    # Save cleaned data
    print(f"\nSaving cleaned data to: {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_file, index=False)
    print(f"✓ Cleaned data saved successfully")

    # Save outlier removal report
    handler.save_outlier_report(report_file)

    # Final summary
    print("\n" + "="*70)
    print("OUTLIER REMOVAL COMPLETE")
    print("="*70)
    print(f"\nCleaned dataset shape: {df_clean.shape}")
    print(f"Output file: {output_file}")
    print(f"Report file: {report_file}")
    print("\nSummary:")
    print(f"  Rows before: {len(df):,}")
    print(f"  Rows after: {len(df_clean):,}")
    print(f"  Rows removed: {len(df) - len(df_clean):,} "
          f"({(len(df) - len(df_clean)) / len(df) * 100:.2f}%)")


if __name__ == "__main__":
    main()
