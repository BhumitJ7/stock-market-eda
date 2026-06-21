
"""
eda.py

Exploratory data analysis functions for the cleaned
symbols_valid_meta dataframe (see clean.py).

Each function is self-contained: it takes the cleaned dataframe,
prints/returns a summary, and optionally saves a plot to `output_dir`.
Designed to be run interactively (notebook) or as a script.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def missing_value_report(df: pd.DataFrame, output_dir: str = None) -> pd.DataFrame:
    """Return a table of missing value counts/percentages, and plot if any exist."""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    report = report[report["missing_count"] > 0].sort_values("missing_pct", ascending=False)

    if not report.empty and output_dir:
        report["missing_pct"].plot(kind="barh", figsize=(6, 3), color="indianred")
        plt.title("Missing Data by Column (%)")
        plt.xlabel("% missing")
        plt.tight_layout()
        _save(output_dir, "missing_values.png")

    return report


def duplicate_report(df: pd.DataFrame) -> dict:
    """Check exact duplicate rows, duplicate symbols, and near-duplicate names."""
    near_dupes = df[df.duplicated("Security Name Clean", keep=False)].sort_values(
        "Security Name Clean"
    )
    return {
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "duplicate_symbols": int(df["Symbol"].duplicated().sum()),
        "near_duplicate_name_rows": near_dupes[["Symbol", "Security Name"]],
    }


def symbol_structure_report(df: pd.DataFrame, output_dir: str = None) -> pd.Series:
    """Summarize symbol length distribution and flag special-character symbols."""
    length_desc = df["Symbol Length"].describe()

    if output_dir:
        df["Symbol Length"].value_counts().sort_index().plot(
            kind="bar", figsize=(8, 3), color="steelblue"
        )
        plt.title("Distribution of Ticker Symbol Length")
        plt.xlabel("Length")
        plt.ylabel("Count")
        plt.tight_layout()
        _save(output_dir, "symbol_length_distribution.png")

    print("Symbols with non-alphabetic characters:", df["Has Special Char"].sum())
    print(df.loc[df["Has Special Char"], "Symbol"].head(10).tolist())

    return length_desc


def etf_flag_consistency_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-check the ETF flag against fund/trust-like keywords in the
    security name. Returns rows flagged ETF == 'N' with a fund-like name —
    a guardrail against silent labeling errors, not necessarily a bug
    (closed-end funds and trusts are legitimately non-ETFs).
    """
    etf_keywords = r"etf|fund|trust|index"
    name_suggests_etf = df["Security Name"].str.lower().str.contains(etf_keywords, regex=True)

    mismatch_table = pd.crosstab(df["ETF"], name_suggests_etf)
    print("ETF flag vs. name-suggests-ETF crosstab:")
    print(mismatch_table)

    suspicious = df[(df["ETF"] == "N") & name_suggests_etf]
    print(f"\nFlagged non-ETF but fund-like name: {len(suspicious)}")

    return suspicious[["Symbol", "Security Name"]]


def round_lot_report(df: pd.DataFrame) -> pd.DataFrame:
    """Identify securities that don't use the standard 100-share round lot."""
    outliers = df[df["Round Lot Size"] != 100]
    print(df["Round Lot Size"].value_counts())
    print(f"\nNon-standard round lot size: {len(outliers)} rows")
    return outliers[["Symbol", "Security Name", "Round Lot Size"]]


def exchange_category_heatmap(df: pd.DataFrame, output_dir: str = None) -> pd.DataFrame:
    """Cross-tab of Listing Exchange x Market Category, plotted as a heatmap."""
    ct = pd.crosstab(df["Listing Exchange"], df["Market Category"])

    if output_dir:
        plt.figure(figsize=(8, 4))
        sns.heatmap(ct, annot=True, fmt="d", cmap="Blues")
        plt.title("Market Category by Listing Exchange")
        plt.tight_layout()
        _save(output_dir, "exchange_category_heatmap.png")

    return ct


def etf_distribution_by_exchange(df: pd.DataFrame, output_dir: str = None) -> pd.DataFrame:
    """Stacked bar chart of ETF vs non-ETF counts per listing exchange."""
    ct = pd.crosstab(df["Listing Exchange"], df["ETF"])

    if output_dir:
        fig, ax = plt.subplots(figsize=(8, 5))
        ct.plot(kind="bar", stacked=True, ax=ax, color=["#4C72B0", "#DD8452"])
        plt.title("ETF vs Non-ETF Distribution by Exchange")
        plt.xlabel("Listing Exchange")
        plt.ylabel("Number of Securities")
        plt.xticks(rotation=0)
        plt.legend(title="ETF")
        plt.tight_layout()
        _save(output_dir, "etf_distribution_by_exchange.png")

    return ct


def summary_table(df: pd.DataFrame, etf_mismatch_count: int) -> pd.DataFrame:
    """One-row summary of key dataset-level stats."""
    summary = pd.DataFrame({
        "n_rows": [len(df)],
        "n_symbols_unique": [df["Symbol"].nunique()],
        "n_etfs": [(df["ETF"] == "Y").sum()],
        "n_exchanges": [df["Listing Exchange"].nunique()],
        "pct_standard_lot": [(df["Round Lot Size"] == 100).mean() * 100],
        "n_etf_flag_mismatches": [etf_mismatch_count],
    })
    return summary.T.rename(columns={0: "value"})


def _save(output_dir: str, filename: str):
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
    plt.close()


def run_full_eda(df: pd.DataFrame, output_dir: str = "assets") -> None:
    """Run every EDA check in sequence and print a summary at the end."""
    print("\n=== Missing values ===")
    print(missing_value_report(df, output_dir))

    print("\n=== Duplicates ===")
    dupes = duplicate_report(df)
    print("Exact duplicate rows:", dupes["exact_duplicate_rows"])
    print("Duplicate symbols:", dupes["duplicate_symbols"])
    print(dupes["near_duplicate_name_rows"])

    print("\n=== Symbol structure ===")
    print(symbol_structure_report(df, output_dir))

    print("\n=== ETF flag consistency ===")
    mismatches = etf_flag_consistency_report(df)

    print("\n=== Round lot size ===")
    print(round_lot_report(df))

    print("\n=== Exchange x Market Category ===")
    print(exchange_category_heatmap(df, output_dir))

    print("\n=== ETF distribution by exchange ===")
    print(etf_distribution_by_exchange(df, output_dir))

    print("\n=== Summary ===")
    print(summary_table(df, len(mismatches)))


if __name__ == "__main__":
    from clean import load_and_clean
    import kagglehub

    path = kagglehub.dataset_download("jacksoncrow/stock-market-dataset")
    cleaned = load_and_clean(path)
    run_full_eda(cleaned, output_dir="assets")
