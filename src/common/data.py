"""
data.py

Data loading utilities for the regime-aware sector rotation project.

This module downloads adjusted price data with yfinance and prepares
daily and monthly price matrices.

The project works mainly at monthly frequency:
- signals are computed at month-end;
- allocations are implemented for the following month.

Run from the project root with:

    python src/common/data.py
"""

from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import yfinance as yf


# ============================================================
# Configuration
# ============================================================

DEFAULT_START_DATE = "2003-12-01"
DEFAULT_END_DATE = None

DATA_DIR = Path("data")
DAILY_PRICES_PATH = DATA_DIR / "daily_prices.csv"
MONTHLY_PRICES_PATH = DATA_DIR / "monthly_prices.csv"


REPLICATION_TICKERS = [
    "SPY",  # Large-cap U.S. equities
    "MDY",  # Mid-cap U.S. equities
    "IJR",  # Small-cap U.S. equities
    "TIP",  # TIPS ETF, used as canary asset
    "SHY",  # Short-duration Treasury ETF, defensive asset
    "VTI",  # Total U.S. equity market benchmark
]


SECTOR_TICKERS = [
    "XLK",  # Technology
    "XLY",  # Consumer Discretionary
    "XLI",  # Industrials
    "XLF",  # Financials
    "XLE",  # Energy
    "XLP",  # Consumer Staples
    "XLU",  # Utilities
    "XLV",  # Health Care
]


ALL_TICKERS = sorted(set(REPLICATION_TICKERS + SECTOR_TICKERS))


# ============================================================
# Core functions
# ============================================================

def download_adjusted_prices(
    tickers: Iterable[str],
    start: str = DEFAULT_START_DATE,
    end: Optional[str] = DEFAULT_END_DATE,
) -> pd.DataFrame:
    """
    Download adjusted daily close prices for a list of tickers.

    Parameters
    ----------
    tickers:
        Iterable of ticker symbols.
    start:
        Start date in YYYY-MM-DD format.
    end:
        End date in YYYY-MM-DD format. If None, yfinance downloads up to today.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by date, with one column per ticker.

    Notes
    -----
    yfinance with auto_adjust=True adjusts OHLC prices for dividends and splits.
    We then keep the adjusted close-like 'Close' field.
    """
    tickers = list(tickers)

    if not tickers:
        raise ValueError("The ticker list cannot be empty.")

    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if data.empty:
        raise ValueError("Downloaded data is empty.")

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns.get_level_values(0):
            raise ValueError("Could not find 'Close' prices in downloaded data.")
        prices = data["Close"].copy()
    else:
        if "Close" not in data.columns:
            raise ValueError("Could not find 'Close' prices in downloaded data.")
        prices = data[["Close"]].copy()
        prices.columns = tickers

    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    prices = prices.dropna(how="all")

    return prices

def validate_price_matrix(prices: pd.DataFrame) -> None:
    """
    Validate a price matrix.

    Parameters
    ----------
    prices:
        DataFrame indexed by date with one column per ticker.

    Notes
    -----
    This validation is intentionally strict. A single fully missing ticker
    should make the data download fail, because otherwise downstream backtests
    may silently reuse incomplete price data.
    """
    if prices.empty:
        raise ValueError("Price matrix is empty.")

    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("Price matrix index must be a DatetimeIndex.")

    if prices.isna().all(axis=None):
        raise ValueError("Price matrix contains only missing values.")

    fully_missing_columns = prices.columns[prices.isna().all()].tolist()
    if fully_missing_columns:
        raise ValueError(
            "The following tickers contain only missing values: "
            f"{fully_missing_columns}. "
            "The download likely failed for these tickers."
        )

    non_positive = (prices <= 0).any(axis=0)
    non_positive_columns = non_positive[non_positive].index.tolist()
    if non_positive_columns:
        raise ValueError(
            "The following tickers contain non-positive prices: "
            f"{non_positive_columns}"
        )

def to_monthly_prices(daily_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Convert daily prices to month-end prices.

    Parameters
    ----------
    daily_prices:
        Daily adjusted price matrix.

    Returns
    -------
    pd.DataFrame
        Monthly adjusted price matrix using the last available observation
        of each calendar month.
    """
    validate_price_matrix(daily_prices)

    monthly_prices = daily_prices.resample("ME").last()
    monthly_prices = monthly_prices.dropna(how="all")

    return monthly_prices


def save_prices(prices: pd.DataFrame, path: Path) -> None:
    """
    Save a price matrix to CSV.

    Parameters
    ----------
    prices:
        Price matrix.
    path:
        Destination CSV path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path, index=True)


def load_prices(path: Path) -> pd.DataFrame:
    """
    Load a price matrix from CSV.

    Parameters
    ----------
    path:
        CSV path.

    Returns
    -------
    pd.DataFrame
        Price matrix indexed by date.
    """
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    prices.index.name = "Date"
    return prices


def download_and_save_project_data(
    tickers: Iterable[str] = ALL_TICKERS,
    start: str = DEFAULT_START_DATE,
    end: Optional[str] = DEFAULT_END_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download daily prices and save both daily and monthly datasets.

    Parameters
    ----------
    tickers:
        Tickers to download.
    start:
        Start date.
    end:
        End date.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Daily prices and monthly prices.
    """
    daily_prices = download_adjusted_prices(
        tickers=tickers,
        start=start,
        end=end,
    )

    validate_price_matrix(daily_prices)

    monthly_prices = to_monthly_prices(daily_prices)
    validate_price_matrix(monthly_prices)

    save_prices(daily_prices, DAILY_PRICES_PATH)
    save_prices(monthly_prices, MONTHLY_PRICES_PATH)

    return daily_prices, monthly_prices


# ============================================================
# Script entry point
# ============================================================

def main() -> None:
    """
    Download and save project data.
    """
    print("=" * 80)
    print("Downloading project data")
    print("=" * 80)

    print("Tickers:")
    print(", ".join(ALL_TICKERS))

    daily_prices, monthly_prices = download_and_save_project_data(
        tickers=ALL_TICKERS,
        start=DEFAULT_START_DATE,
        end=DEFAULT_END_DATE,
    )

    print()
    print("Daily prices:")
    print(daily_prices.tail())
    print()
    print(f"Saved to: {DAILY_PRICES_PATH}")
    print(f"Shape: {daily_prices.shape}")
    print(f"Date range: {daily_prices.index.min().date()} to {daily_prices.index.max().date()}")

    print()
    print("Monthly prices:")
    print(monthly_prices.tail())
    print()
    print(f"Saved to: {MONTHLY_PRICES_PATH}")
    print(f"Shape: {monthly_prices.shape}")
    print(f"Date range: {monthly_prices.index.min().date()} to {monthly_prices.index.max().date()}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()