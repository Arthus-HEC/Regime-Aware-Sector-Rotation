"""
metrics.py

Performance and backtesting metrics for the regime-aware sector rotation project.

The project works at monthly frequency. By default, annualized metrics use
12 periods per year.

This module contains generic functions:
- returns computation;
- portfolio return computation;
- transaction cost adjustment;
- CAGR;
- annualized volatility;
- Sharpe ratio;
- drawdown;
- turnover;
- performance summary tables.

Run from the project root with:

    python src/common/metrics.py
"""

from pathlib import Path
import sys
from typing import Optional

import numpy as np
import pandas as pd


# Make imports robust when running the file directly from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from common.data import load_prices, MONTHLY_PRICES_PATH


# ============================================================
# Return utilities
# ============================================================

def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple returns from a price matrix.

    Parameters
    ----------
    prices:
        Price matrix indexed by date, with one column per asset.

    Returns
    -------
    pd.DataFrame
        Simple returns.

    Notes
    -----
    Returns are defined as:

        R_t = P_t / P_{t-1} - 1
    """
    if prices.empty:
        raise ValueError("Price matrix is empty.")

    returns = prices.pct_change()
    returns = returns.dropna(how="all")

    return returns


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log-returns from a price matrix.

    Parameters
    ----------
    prices:
        Price matrix indexed by date, with one column per asset.

    Returns
    -------
    pd.DataFrame
        Log-return matrix.
    """
    if prices.empty:
        raise ValueError("Price matrix is empty.")

    log_returns = np.log(prices / prices.shift(1))
    log_returns = log_returns.dropna(how="all")

    return log_returns


def compute_portfolio_returns(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.Series:
    """
    Compute portfolio returns from asset returns and portfolio weights.

    Parameters
    ----------
    returns:
        Asset return matrix indexed by date.
    weights:
        Portfolio weights indexed by date with the same asset columns.

    Returns
    -------
    pd.Series
        Portfolio returns.

    Important convention
    --------------------
    This function assumes that weights are already aligned with returns.

    That means the row weights.loc[t] is the portfolio held during the period
    whose return is returns.loc[t].

    The replication script will handle the signal lag separately to avoid
    look-ahead bias.
    """
    if returns.empty:
        raise ValueError("Return matrix is empty.")

    if weights.empty:
        raise ValueError("Weight matrix is empty.")

    common_index = returns.index.intersection(weights.index)
    common_columns = returns.columns.intersection(weights.columns)

    if len(common_index) == 0:
        raise ValueError("Returns and weights have no common dates.")

    if len(common_columns) == 0:
        raise ValueError("Returns and weights have no common assets.")

    aligned_returns = returns.loc[common_index, common_columns]
    aligned_weights = weights.loc[common_index, common_columns]

    portfolio_returns = (aligned_weights * aligned_returns).sum(axis=1)
    portfolio_returns.name = "portfolio_return"

    return portfolio_returns


# ============================================================
# Transaction costs and turnover
# ============================================================

def compute_turnover(weights: pd.DataFrame) -> pd.Series:
    """
    Compute one-way portfolio turnover.

    Parameters
    ----------
    weights:
        Portfolio weights indexed by date.

    Returns
    -------
    pd.Series
        One-way turnover at each date.

    Notes
    -----
    Turnover is defined as:

        turnover_t = 0.5 * sum_i |w_{t,i} - w_{t-1,i}|

    This convention means that switching from 100% asset A to 100% asset B
    produces turnover equal to 1, not 2.
    """
    if weights.empty:
        raise ValueError("Weight matrix is empty.")

    turnover = 0.5 * weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()
    turnover.name = "turnover"

    return turnover


def apply_transaction_costs(
    portfolio_returns: pd.Series,
    turnover: pd.Series,
    transaction_cost_bps: float = 5.0,
) -> pd.Series:
    """
    Apply transaction costs to portfolio returns.

    Parameters
    ----------
    portfolio_returns:
        Gross portfolio returns.
    turnover:
        One-way turnover series.
    transaction_cost_bps:
        Transaction cost in basis points per unit of one-way turnover.

    Returns
    -------
    pd.Series
        Net portfolio returns.
    """
    if transaction_cost_bps < 0:
        raise ValueError("Transaction costs cannot be negative.")

    common_index = portfolio_returns.index.intersection(turnover.index)

    cost_rate = transaction_cost_bps / 10_000.0

    net_returns = (
        portfolio_returns.loc[common_index]
        - cost_rate * turnover.loc[common_index]
    )

    net_returns.name = "portfolio_return_net"

    return net_returns


# ============================================================
# Equity curve and risk metrics
# ============================================================

def compute_equity_curve(
    returns: pd.Series,
    initial_value: float = 1.0,
) -> pd.Series:
    """
    Compute cumulative wealth from a return series.

    Parameters
    ----------
    returns:
        Periodic simple returns.
    initial_value:
        Initial portfolio value.

    Returns
    -------
    pd.Series
        Cumulative wealth index.
    """
    if initial_value <= 0:
        raise ValueError("Initial value must be strictly positive.")

    equity_curve = initial_value * (1.0 + returns.fillna(0.0)).cumprod()
    equity_curve.name = "equity_curve"

    return equity_curve


def compute_cagr(
    returns: pd.Series,
    periods_per_year: int = 12,
) -> float:
    """
    Compute compound annual growth rate.

    Parameters
    ----------
    returns:
        Periodic simple returns.
    periods_per_year:
        Number of periods per year.

    Returns
    -------
    float
        CAGR.
    """
    returns = returns.dropna()

    if returns.empty:
        return np.nan

    total_growth = float((1.0 + returns).prod())
    n_periods = len(returns)

    if total_growth <= 0:
        return np.nan

    years = n_periods / periods_per_year
    cagr = total_growth ** (1.0 / years) - 1.0

    return cagr


def compute_annualized_return(
    returns: pd.Series,
    periods_per_year: int = 12,
) -> float:
    """
    Compute annualized arithmetic mean return.

    Parameters
    ----------
    returns:
        Periodic simple returns.
    periods_per_year:
        Number of periods per year.

    Returns
    -------
    float
        Annualized arithmetic mean return.
    """
    returns = returns.dropna()

    if returns.empty:
        return np.nan

    return float(returns.mean() * periods_per_year)


def compute_annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = 12,
) -> float:
    """
    Compute annualized volatility.

    Parameters
    ----------
    returns:
        Periodic simple returns.
    periods_per_year:
        Number of periods per year.

    Returns
    -------
    float
        Annualized volatility.
    """
    returns = returns.dropna()

    if returns.empty:
        return np.nan

    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def compute_sharpe_ratio(
    returns: pd.Series,
    periods_per_year: int = 12,
    annual_risk_free_rate: float = 0.0,
) -> float:
    """
    Compute annualized Sharpe ratio.

    Parameters
    ----------
    returns:
        Periodic simple returns.
    periods_per_year:
        Number of periods per year.
    annual_risk_free_rate:
        Annual risk-free rate.

    Returns
    -------
    float
        Annualized Sharpe ratio.
    """
    returns = returns.dropna()

    if returns.empty:
        return np.nan

    periodic_rf = annual_risk_free_rate / periods_per_year
    excess_returns = returns - periodic_rf

    volatility = excess_returns.std(ddof=1)

    if volatility == 0 or np.isnan(volatility):
        return np.nan

    sharpe = (
        excess_returns.mean()
        / volatility
        * np.sqrt(periods_per_year)
    )

    return float(sharpe)


def compute_drawdown(
    equity_curve: pd.Series,
) -> pd.Series:
    """
    Compute drawdown from an equity curve.

    Parameters
    ----------
    equity_curve:
        Cumulative wealth index.

    Returns
    -------
    pd.Series
        Drawdown series.
    """
    if equity_curve.empty:
        raise ValueError("Equity curve is empty.")

    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    drawdown.name = "drawdown"

    return drawdown


def compute_max_drawdown(
    returns: pd.Series,
) -> float:
    """
    Compute maximum drawdown from a return series.

    Parameters
    ----------
    returns:
        Periodic simple returns.

    Returns
    -------
    float
        Maximum drawdown.
    """
    equity_curve = compute_equity_curve(returns)
    drawdown = compute_drawdown(equity_curve)

    return float(drawdown.min())


# ============================================================
# Performance summary
# ============================================================

def compute_performance_summary(
    returns: pd.Series,
    periods_per_year: int = 12,
    annual_risk_free_rate: float = 0.0,
    turnover: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Compute a standard performance summary.

    Parameters
    ----------
    returns:
        Periodic simple returns.
    periods_per_year:
        Number of periods per year.
    annual_risk_free_rate:
        Annual risk-free rate.
    turnover:
        Optional turnover series.

    Returns
    -------
    pd.Series
        Performance metrics.
    """
    returns = returns.dropna()

    if returns.empty:
        raise ValueError("Return series is empty.")

    total_return = float((1.0 + returns).prod() - 1.0)

    summary = {
        "total_return": total_return,
        "cagr": compute_cagr(returns, periods_per_year),
        "annualized_return": compute_annualized_return(returns, periods_per_year),
        "annualized_volatility": compute_annualized_volatility(
            returns,
            periods_per_year,
        ),
        "sharpe_ratio": compute_sharpe_ratio(
            returns,
            periods_per_year,
            annual_risk_free_rate,
        ),
        "max_drawdown": compute_max_drawdown(returns),
    }

    if turnover is not None:
        aligned_turnover = turnover.reindex(returns.index).fillna(0.0)
        summary["total_turnover"] = float(aligned_turnover.sum())
        summary["average_turnover"] = float(aligned_turnover.mean())
    else:
        summary["total_turnover"] = np.nan
        summary["average_turnover"] = np.nan

    return pd.Series(summary)


def format_performance_summary(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Format a performance summary table for display.

    Parameters
    ----------
    summary:
        DataFrame with strategies as rows and metrics as columns.

    Returns
    -------
    pd.DataFrame
        Formatted summary table.
    """
    formatted = summary.copy()

    percentage_columns = [
        "total_return",
        "cagr",
        "annualized_return",
        "annualized_volatility",
        "max_drawdown",
        "average_turnover",
    ]

    for column in percentage_columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(
                lambda x: "--" if pd.isna(x) else f"{100 * x:.2f}%"
            )

    numeric_columns = [
        "sharpe_ratio",
        "total_turnover",
    ]

    for column in numeric_columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(
                lambda x: "--" if pd.isna(x) else f"{x:.2f}"
            )

    return formatted


# ============================================================
# Period slicing
# ============================================================

def slice_period(
    series_or_frame: pd.Series | pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.Series | pd.DataFrame:
    """
    Slice a Series or DataFrame over a date range (inclusive).

    Parameters
    ----------
    series_or_frame:
        Series or DataFrame indexed by date.
    start_date:
        Start date (inclusive).
    end_date:
        End date (inclusive).

    Returns
    -------
    pd.Series | pd.DataFrame
        Sliced object.
    """
    return series_or_frame.loc[
        (series_or_frame.index >= start_date)
        & (series_or_frame.index <= end_date)
    ]


# ============================================================
# Script entry point
# ============================================================

def main() -> None:
    """
    Quick test of the metrics module using monthly prices.
    """
    print("=" * 80)
    print("Testing performance metrics")
    print("=" * 80)

    prices = load_prices(MONTHLY_PRICES_PATH)
    returns = compute_simple_returns(prices)

    print()
    print("Monthly returns:")
    print(returns.tail())

    benchmark = returns["VTI"].dropna()

    summary = compute_performance_summary(
        returns=benchmark,
        periods_per_year=12,
        annual_risk_free_rate=0.0,
    )

    summary_table = pd.DataFrame(
        [summary],
        index=["VTI Buy and Hold"],
    )

    print()
    print("Raw performance summary:")
    print(summary_table)

    print()
    print("Formatted performance summary:")
    print(format_performance_summary(summary_table))

    print()
    print("Done.")


if __name__ == "__main__":
    main()