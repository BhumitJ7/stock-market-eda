"""
main.py

Single entry point: downloads the dataset, cleans the metadata,
runs the full EDA suite, and saves all plots to assets/.

Usage:
    python src/main.py
"""

import kagglehub
from clean import load_and_clean
from eda import run_full_eda


def main():
    path = kagglehub.dataset_download("jacksoncrow/stock-market-dataset")
    print("Dataset downloaded to:", path)

    df = load_and_clean(path)
    print("\nCleaned shape:", df.shape)

    run_full_eda(df, output_dir="assets")


if __name__ == "__main__":
    main()
