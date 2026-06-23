"""
sharpe_over_shy.py

Compute Sharpe ratios relative to SHY.

The main performance tables use a standard Sharpe ratio with a zero
risk-free rate, which is common in simple backtests but methodologically
imperfect.

This script adds a complementary metric:

    Sharpe over SHY = mean(r_strategy - r_SHY) / std(r_strategy - r_SHY) * sqrt(12)

SHY is not a perfect risk-free asset, but it is the defensive asset used
inside the strategy universe. It is therefore a practical proxy for the
cash-like return available to the strategy.

Run from the project root with:

    python src/haa/sharpe_over_shy.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# Make imports robust when running the file directly from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from common.data import load_prices, MONTHLY_PRICES_PATH
from common.metrics import compute_simple_returns


# ============================================================
# Paths and configuration
# ============================================================

DATA_DIR = Path("data")

COMPARISON_RETURNS_PATH = DATA_DIR / "strategy_comparison_returns.csv"
SHARPE_OVER_SHY_OUTPUT_PATH = DATA_DIR / "sharpe_over_shy_summary.csv"

PERIODS_PER_YEAR = 12

VALIDATION_PERIODS = {
    "Full Sample": ("2005-01-31", "2025-12-31"),
    "In-Sample": ("2005-01-31", "2015-12-31"),
    "Out-of-Sample": ("2016-01-31", "2025-12-31"),
}


# ============================================================
# Loading utilities
# ============================================================

def load_strategy_returns(
    path: Path = COMPARISON_RETURNS_PATH,
) -> pd.DataFrame:
    """
    Load strategy comparison returns.

    Parameters
    ----------
    path:
        CSV path produced by compare_strategies.py.

    Returns
    -------
    pd.DataFrame
        Strategy and benchmark return matrix.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}. "
            "Run python src/haa/compare_strategies.py first."
        )

    returns = pd.read_csv(path, index_col=0, parse_dates=True)

    if returns.empty:
        raise ValueError("Strategy return matrix is empty.")

    return returns


def load_shy_returns() -> pd.Series:
    """
    Load monthly SHY returns from monthly price data.

    Returns
    -------
    pd.Series
        Monthly SHY returns.
    """
    prices = load_prices(MONTHLY_PRICES_PATH)

    if "SHY" not in prices.columns:
        raise ValueError("SHY is missing from monthly price data.")

    returns = compute_simple_returns(prices)
    shy_returns = returns["SHY"].copy()
    shy_returns.name = "SHY"

    return shy_returns


# ============================================================
# Excess-return metrics
# ============================================================

def compute_sharpe_over_shy(
    strategy_returns: pd.Series,
    shy_returns: pd.Series,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> pd.Series:
    """
    Compute excess-return metrics relative to SHY.

    Parameters
    ----------
    strategy_returns:
        Strategy monthly returns.
    shy_returns:
        SHY monthly returns.
    periods_per_year:
        Number of periods per year.

    Returns
    -------
    pd.Series
        Metrics:
        - total_excess_return;
        - annualized_excess_return;
        - excess_volatility;
        - sharpe_over_shy.
    """
    aligned = pd.concat(
        [
            strategy_returns.rename("strategy"),
            shy_returns.rename("shy"),
        ],
        axis=1,
    ).dropna()

    if aligned.empty:
        raise ValueError("No common dates between strategy returns and SHY returns.")

    excess_returns = aligned["strategy"] - aligned["shy"]

    total_excess_return = float(excess_returns.sum())
    annualized_excess_return = float(excess_returns.mean() * periods_per_year)
    excess_volatility = float(excess_returns.std(ddof=1) * np.sqrt(periods_per_year))

    if excess_volatility == 0 or np.isnan(excess_volatility):
        sharpe_over_shy = np.nan
    else:
        sharpe_over_shy = annualized_excess_return / excess_volatility

    return pd.Series(
        {
            "total_excess_return": total_excess_return,
            "annualized_excess_return": annualized_excess_return,
            "excess_volatility": excess_volatility,
            "sharpe_over_shy": sharpe_over_shy,
        }
    )


def slice_period(
    series_or_frame: pd.Series | pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.Series | pd.DataFrame:
    """
    Slice a Series or DataFrame over a date range.
    """
    return series_or_frame.loc[
        (series_or_frame.index >= start_date)
        & (series_or_frame.index <= end_date)
    ]


def build_sharpe_over_shy_summary(
    strategy_returns: pd.DataFrame,
    shy_returns: pd.Series,
) -> pd.DataFrame:
    """
    Build Sharpe-over-SHY summary for all samples and strategies.

    Parameters
    ----------
    strategy_returns:
        Strategy and benchmark return matrix.
    shy_returns:
        SHY return series.

    Returns
    -------
    pd.DataFrame
        Multi-index DataFrame with levels:
        - sample;
        - strategy.
    """
    all_rows = []

    for sample_name, (start_date, end_date) in VALIDATION_PERIODS.items():
        period_strategy_returns = slice_period(
            strategy_returns,
            start_date=start_date,
            end_date=end_date,
        )

        period_shy_returns = slice_period(
            shy_returns,
            start_date=start_date,
            end_date=end_date,
        )

        if not isinstance(period_strategy_returns, pd.DataFrame):
            raise TypeError("period_strategy_returns should be a DataFrame.")

        if not isinstance(period_shy_returns, pd.Series):
            raise TypeError("period_shy_returns should be a Series.")

        for strategy in period_strategy_returns.columns:
            metrics = compute_sharpe_over_shy(
                strategy_returns=period_strategy_returns[strategy],
                shy_returns=period_shy_returns,
                periods_per_year=PERIODS_PER_YEAR,
            )

            row = metrics.to_dict()
            row["sample"] = sample_name
            row["start_date"] = start_date
            row["end_date"] = end_date
            row["strategy"] = strategy

            all_rows.append(row)

    summary = pd.DataFrame(all_rows)
    summary = summary.set_index(["sample", "strategy"])

    ordered_columns = [
        "start_date",
        "end_date",
        "total_excess_return",
        "annualized_excess_return",
        "excess_volatility",
        "sharpe_over_shy",
    ]

    summary = summary[ordered_columns]

    return summary


# ============================================================
# Display helpers
# ============================================================

def format_sharpe_over_shy_summary(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Format Sharpe-over-SHY metrics for display.
    """
    formatted = summary.copy()

    percentage_columns = [
        "total_excess_return",
        "annualized_excess_return",
        "excess_volatility",
    ]

    for column in percentage_columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(
                lambda x: "--" if pd.isna(x) else f"{100 * x:.2f}%"
            )

    if "sharpe_over_shy" in formatted.columns:
        formatted["sharpe_over_shy"] = formatted["sharpe_over_shy"].map(
            lambda x: "--" if pd.isna(x) else f"{x:.2f}"
        )

    return formatted


def print_sample_summary(
    summary: pd.DataFrame,
    sample_name: str,
) -> None:
    """
    Print one formatted sample summary.
    """
    sample_summary = summary.loc[sample_name].copy()

    display_summary = sample_summary.drop(
        columns=["start_date", "end_date"],
        errors="ignore",
    )

    print()
    print("=" * 80)
    print(sample_name)
    print("=" * 80)
    print(format_sharpe_over_shy_summary(display_summary))

    print()
    print("Ranking by Sharpe over SHY:")
    print(
        sample_summary["sharpe_over_shy"]
        .sort_values(ascending=False)
        .map(lambda x: f"{x:.2f}")
    )

    print()
    print("Ranking by annualized excess return over SHY:")
    print(
        sample_summary["annualized_excess_return"]
        .sort_values(ascending=False)
        .map(lambda x: f"{100 * x:.2f}%")
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    """
    Run the Sharpe-over-SHY analysis.
    """
    print("=" * 80)
    print("Sharpe over SHY analysis")
    print("=" * 80)

    strategy_returns = load_strategy_returns()
    shy_returns = load_shy_returns()

    summary = build_sharpe_over_shy_summary(
        strategy_returns=strategy_returns,
        shy_returns=shy_returns,
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SHARPE_OVER_SHY_OUTPUT_PATH)

    print()
    print("Strategy return matrix:")
    print(strategy_returns.tail())
    print()
    print(f"Shape: {strategy_returns.shape}")
    print(f"Date range: {strategy_returns.index.min().date()} to {strategy_returns.index.max().date()}")

    print()
    print("SHY returns:")
    print(shy_returns.tail())

    for sample_name in VALIDATION_PERIODS:
        print_sample_summary(
            summary=summary,
            sample_name=sample_name,
        )

    print()
    print("Saved outputs:")
    print(f"- {SHARPE_OVER_SHY_OUTPUT_PATH}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()