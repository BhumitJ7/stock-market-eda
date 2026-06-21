
"""
clean.py

Loads and cleans symbols_valid_meta.csv from the
jacksoncrow/stock-market-dataset Kaggle dataset.

Cleaning philosophy: validate an assumption before acting on it.
Columns are only dropped once we've confirmed the invariant that
justifies dropping them actually holds for every row.
"""

import os
import pandas as pd


def load_raw(dataset_path: str, filename: str = "symbols_valid_meta.csv") -> pd.DataFrame:
    """Load the raw metadata CSV from the downloaded dataset path."""
    csv_path = os.path.join(dataset_path, filename)
    return pd.read_csv(csv_path)


def validate_assumptions(df: pd.DataFrame) -> dict:
    """
    Check the invariants that justify dropping certain columns.
    Returns a dict of {check_name: bool} so callers can inspect
    results before any destructive operation happens.
    """
    return {
        "Nasdaq Traded all 'Y'": (df["Nasdaq Traded"] == "Y").all(),
        "Test Issue all 'N'": (df["Test Issue"] == "N").all(),
        "Financial Status mostly null/constant": df["Financial Status"].nunique(dropna=True) <= 1,
        "Symbol == NASDAQ Symbol": (df["Symbol"] == df["NASDAQ Symbol"]).all(),
    }


def clean(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Clean the raw metadata dataframe.

    - Drops 'Nasdaq Traded', 'Test Issue', 'NASDAQ Symbol' only if their
      underlying invariant is confirmed to hold.
    - Leaves 'Financial Status' in place regardless of missingness, since
      its nulls are meaningful (most issuers simply aren't flagged).
    - Adds derived columns useful for downstream EDA:
        'Symbol Length', 'Has Special Char', 'Security Name Clean'
    """
    df = df.copy()
    checks = validate_assumptions(df)

    drop_candidates = {
        "Nasdaq Traded": checks["Nasdaq Traded all 'Y'"],
        "Test Issue": checks["Test Issue all 'N'"],
        "NASDAQ Symbol": checks["Symbol == NASDAQ Symbol"],
    }
    drop_cols = [col for col, ok in drop_candidates.items() if ok]
    df.drop(columns=drop_cols, inplace=True)

    if verbose:
        print("Validation checks:")
        for k, v in checks.items():
            print(f"  {k}: {v}")
        print(f"Dropped columns: {drop_cols}")

    # Derived fields used across the EDA module
    df["Security Name Clean"] = (
        df["Security Name"]
        .str.lower()
        .str.replace(r"[^a-z0-9 ]", "", regex=True)
        .str.strip()
    )
    df["Symbol Length"] = df["Symbol"].str.len()
    df["Has Special Char"] = df["Symbol"].str.contains(r"[^A-Za-z]", regex=True)

    return df


def load_and_clean(dataset_path: str, verbose: bool = True) -> pd.DataFrame:
    """Convenience wrapper: load raw CSV and return the cleaned dataframe."""
    raw = load_raw(dataset_path)
    if verbose:
        print("Raw shape:", raw.shape)
    return clean(raw, verbose=verbose)


if __name__ == "__main__":
    import kagglehub

    path = kagglehub.dataset_download("jacksoncrow/stock-market-dataset")
    cleaned = load_and_clean(path)
    print("\nCleaned shape:", cleaned.shape)
    print(cleaned.head())
