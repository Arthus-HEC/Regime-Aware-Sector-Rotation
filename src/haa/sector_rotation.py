"""
sector_rotation.py

Sector rotation extension of the HAA-style regime-aware strategy.

The replicated paper rotates between market-cap ETFs:
    SPY, MDY, IJR

This extension rotates between sector ETFs.

The logic is:

1. Compute 13612 momentum scores:
       M_t = (R_1m + R_3m + R_6m + R_12m) / 4

2. Use TIP as the canary asset:
       if M_t(TIP) > 0 -> risk-on
       if M_t(TIP) <= 0 -> risk-off

3. In risk-on regimes:
       allocate 100% to the strongest offensive sector ETF.

4. In risk-off regimes:
       allocate 100% to the strongest defensive asset among defensive sectors and SHY.

5. Signals are computed at month-end and implemented for the following month.

Run from the project root with:

    python src/haa/sector_rotation.py
"""

from pathlib import Path
import sys
from typing import Iterable

import pandas as pd


# Make imports robust when running the file directly from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from common.data import load_prices, MONTHLY_PRICES_PATH
from common.metrics import (
    compute_simple_returns,
    compute_portfolio_returns,
    compute_turnover,
    apply_transaction_costs,
    compute_performance_summary,
    format_performance_summary,
    compute_equity_curve,
    compute_drawdown,
    build_one_asset_weights,
    lag_weights_to_next_month,
)
from haa.momentum import (
    DEFAULT_HORIZONS,
    compute_multi_horizon_momentum,
    compute_absolute_momentum_signal,
    select_top_momentum_asset,
)


# ============================================================
# Configuration
# ============================================================

SECTOR_CANARY_ASSET = "TIP"

SECTOR_OFFENSIVE_ASSETS = [
    "XLK",  # Technology
    "XLY",  # Consumer Discretionary
    "XLI",  # Industrials
    "XLF",  # Financials
    "XLE",  # Energy
]

SECTOR_DEFENSIVE_ASSETS = [
    "XLP",  # Consumer Staples
    "XLU",  # Utilities
    "XLV",  # Health Care
    "SHY",  # Short-duration Treasury ETF
]

SECTOR_BENCHMARKS = [
    "VTI",
    "SPY",
]

BACKTEST_START_DATE = "2005-01-31"
BACKTEST_END_DATE = "2025-12-31"

TRANSACTION_COST_BPS = 5.0
PERIODS_PER_YEAR = 12

OUTPUT_DIR = Path("data")
RETURNS_OUTPUT_PATH = OUTPUT_DIR / "sector_rotation_returns.csv"
WEIGHTS_OUTPUT_PATH = OUTPUT_DIR / "sector_rotation_weights.csv"
SIGNALS_OUTPUT_PATH = OUTPUT_DIR / "sector_rotation_signals.csv"
SUMMARY_OUTPUT_PATH = OUTPUT_DIR / "sector_rotation_summary.csv"


# ============================================================
# Signal construction
# ============================================================

def build_target_assets(
    risk_on: pd.Series,
    selected_offensive_asset: pd.Series,
    selected_defensive_asset: pd.Series,
) -> pd.Series:
    """
    Build the target asset selected at each signal date.

    Parameters
    ----------
    risk_on:
        Binary risk-on signal.
    selected_offensive_asset:
        Best offensive sector ETF at each date.
    selected_defensive_asset:
        Best defensive asset at each date.

    Returns
    -------
    pd.Series
        Target asset at each signal date.
    """
    common_index = (
        risk_on.index
        .intersection(selected_offensive_asset.index)
        .intersection(selected_defensive_asset.index)
    )

    target_assets = selected_offensive_asset.loc[common_index].copy()
    target_assets.loc[risk_on.loc[common_index] == 0] = (
        selected_defensive_asset.loc[common_index]
    )

    target_assets.name = "target_asset"

    return target_assets


def build_sector_rotation_signals(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build sector rotation signals.

    Parameters
    ----------
    prices:
        Monthly price matrix.

    Returns
    -------
    pd.DataFrame
        Signal table containing:
        - canary momentum;
        - risk-on signal;
        - selected offensive sector;
        - selected defensive asset;
        - target asset;
        - target asset momentum.
    """
    momentum = compute_multi_horizon_momentum(
        prices=prices,
        horizons=DEFAULT_HORIZONS,
    )

    risk_on = compute_absolute_momentum_signal(
        momentum=momentum,
        canary_asset=SECTOR_CANARY_ASSET,
    )

    selected_offensive_asset = select_top_momentum_asset(
        momentum=momentum,
        assets=SECTOR_OFFENSIVE_ASSETS,
    )
    selected_offensive_asset.name = "selected_offensive_asset"

    selected_defensive_asset = select_top_momentum_asset(
        momentum=momentum,
        assets=SECTOR_DEFENSIVE_ASSETS,
    )
    selected_defensive_asset.name = "selected_defensive_asset"

    target_asset = build_target_assets(
        risk_on=risk_on,
        selected_offensive_asset=selected_offensive_asset,
        selected_defensive_asset=selected_defensive_asset,
    )

    target_asset_momentum = pd.Series(
        data=[
            momentum.loc[date, asset]
            for date, asset in target_asset.items()
        ],
        index=target_asset.index,
        name="target_asset_momentum",
    )

    signals = pd.concat(
        [
            momentum[SECTOR_CANARY_ASSET].rename("canary_momentum"),
            risk_on,
            selected_offensive_asset,
            selected_defensive_asset,
            target_asset,
            target_asset_momentum,
        ],
        axis=1,
    )

    signals = signals.dropna()

    return signals


def build_sector_rotation_weights(
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build sector rotation signals and lagged portfolio weights.

    Parameters
    ----------
    prices:
        Monthly price matrix.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Signals and lagged weights.
    """
    signals = build_sector_rotation_signals(prices)

    universe = sorted(
        set(SECTOR_OFFENSIVE_ASSETS + SECTOR_DEFENSIVE_ASSETS)
    )

    signal_weights = build_one_asset_weights(
        target_assets=signals["target_asset"],
        universe=universe,
    )

    weights = lag_weights_to_next_month(signal_weights)

    return signals, weights


# ============================================================
# Backtest
# ============================================================

def run_sector_rotation_backtest(
    prices: pd.DataFrame,
    transaction_cost_bps: float = TRANSACTION_COST_BPS,
    backtest_start_date: str = BACKTEST_START_DATE,
    backtest_end_date: str | None = BACKTEST_END_DATE,
) -> dict[str, object]:
    """
    Run the sector rotation backtest.

    Parameters
    ----------
    prices:
        Monthly price matrix.
    transaction_cost_bps:
        Transaction cost in basis points.
    backtest_start_date:
        First date included in the performance comparison.
    backtest_end_date:
        Last date included in the performance comparison.

    Returns
    -------
    dict
        Backtest outputs.
    """
    required_assets = (
        [SECTOR_CANARY_ASSET]
        + SECTOR_OFFENSIVE_ASSETS
        + SECTOR_DEFENSIVE_ASSETS
        + SECTOR_BENCHMARKS
    )

    missing_assets = [asset for asset in required_assets if asset not in prices.columns]
    if missing_assets:
        raise ValueError(f"Missing required assets in prices: {missing_assets}")

    returns = compute_simple_returns(prices)

    signals, weights = build_sector_rotation_weights(prices)

    strategy_gross_returns = compute_portfolio_returns(
        returns=returns,
        weights=weights,
    )
    strategy_gross_returns.name = "Sector Rotation Gross"

    turnover = compute_turnover(weights)

    strategy_net_returns = apply_transaction_costs(
        portfolio_returns=strategy_gross_returns,
        turnover=turnover,
        transaction_cost_bps=transaction_cost_bps,
    )
    strategy_net_returns.name = "Sector Rotation Net"

    return_series = [
        strategy_gross_returns,
        strategy_net_returns,
    ]

    for benchmark in SECTOR_BENCHMARKS:
        benchmark_returns = returns[benchmark].copy()
        benchmark_returns.name = f"{benchmark} Buy and Hold"
        return_series.append(benchmark_returns)

    combined_returns = pd.concat(return_series, axis=1)

    combined_turnover = turnover.rename("Sector Rotation Turnover")

    if backtest_start_date is not None:
        combined_returns = combined_returns.loc[
            combined_returns.index >= backtest_start_date
        ]
        combined_turnover = combined_turnover.loc[
            combined_turnover.index >= backtest_start_date
        ]
        weights = weights.loc[weights.index >= backtest_start_date]
        signals = signals.loc[signals.index >= backtest_start_date]

    if backtest_end_date is not None:
        combined_returns = combined_returns.loc[
            combined_returns.index <= backtest_end_date
        ]
        combined_turnover = combined_turnover.loc[
            combined_turnover.index <= backtest_end_date
        ]
        weights = weights.loc[weights.index <= backtest_end_date]
        signals = signals.loc[signals.index <= backtest_end_date]

    combined_returns = combined_returns.dropna()
    turnover_aligned = combined_turnover.reindex(combined_returns.index).fillna(0.0)

    performance_rows = {}

    for column in combined_returns.columns:
        if column in ["Sector Rotation Gross", "Sector Rotation Net"]:
            strategy_turnover = turnover_aligned
        else:
            strategy_turnover = None

        performance_rows[column] = compute_performance_summary(
            returns=combined_returns[column],
            periods_per_year=PERIODS_PER_YEAR,
            turnover=strategy_turnover,
        )

    performance_summary = pd.DataFrame(performance_rows).T

    equity_curves = pd.DataFrame(
        {
            column: compute_equity_curve(combined_returns[column])
            for column in combined_returns.columns
        }
    )

    drawdowns = pd.DataFrame(
        {
            column: compute_drawdown(equity_curves[column])
            for column in equity_curves.columns
        }
    )

    return {
        "signals": signals,
        "weights": weights,
        "returns": combined_returns,
        "turnover": turnover_aligned,
        "performance_summary": performance_summary,
        "equity_curves": equity_curves,
        "drawdowns": drawdowns,
    }


# ============================================================
# Saving and display
# ============================================================

def save_sector_rotation_outputs(results: dict[str, object]) -> None:
    """
    Save sector rotation outputs to CSV files.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    signals = results["signals"]
    returns = results["returns"]
    weights = results["weights"]
    performance_summary = results["performance_summary"]

    if not isinstance(signals, pd.DataFrame):
        raise TypeError("results['signals'] must be a DataFrame.")
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("results['returns'] must be a DataFrame.")
    if not isinstance(weights, pd.DataFrame):
        raise TypeError("results['weights'] must be a DataFrame.")
    if not isinstance(performance_summary, pd.DataFrame):
        raise TypeError("results['performance_summary'] must be a DataFrame.")

    signals.to_csv(SIGNALS_OUTPUT_PATH)
    returns.to_csv(RETURNS_OUTPUT_PATH)
    weights.to_csv(WEIGHTS_OUTPUT_PATH)
    performance_summary.to_csv(SUMMARY_OUTPUT_PATH)


def main() -> None:
    """
    Run the sector rotation extension.
    """
    print("=" * 80)
    print("HAA-style sector rotation extension")
    print("=" * 80)

    prices = load_prices(MONTHLY_PRICES_PATH)

    print()
    print("Monthly prices loaded:")
    print(prices.tail())
    print()
    print(f"Shape: {prices.shape}")
    print(f"Date range: {prices.index.min().date()} to {prices.index.max().date()}")

    results = run_sector_rotation_backtest(
        prices=prices,
        transaction_cost_bps=TRANSACTION_COST_BPS,
        backtest_start_date=BACKTEST_START_DATE,
        backtest_end_date=BACKTEST_END_DATE,
    )

    signals = results["signals"]
    weights = results["weights"]
    returns = results["returns"]
    performance_summary = results["performance_summary"]

    if not isinstance(signals, pd.DataFrame):
        raise TypeError("signals should be a DataFrame.")
    if not isinstance(weights, pd.DataFrame):
        raise TypeError("weights should be a DataFrame.")
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns should be a DataFrame.")
    if not isinstance(performance_summary, pd.DataFrame):
        raise TypeError("performance_summary should be a DataFrame.")

    print()
    print("Signals preview:")
    print(signals.tail(12))

    print()
    print("Weights preview:")
    print(weights.tail(12))

    print()
    print("Returns preview:")
    print(returns.tail(12))

    print()
    print("Raw performance summary:")
    print(performance_summary)

    print()
    print("Formatted performance summary:")
    print(format_performance_summary(performance_summary))

    print()
    print("Risk-on distribution:")
    print(signals["risk_on"].value_counts(normalize=True).rename("frequency"))

    print()
    print("Selected offensive sector distribution:")
    print(signals["selected_offensive_asset"].value_counts(normalize=True).rename("frequency"))

    print()
    print("Selected defensive asset distribution:")
    print(signals["selected_defensive_asset"].value_counts(normalize=True).rename("frequency"))

    print()
    print("Realized target asset distribution:")
    realized_asset = weights.idxmax(axis=1)
    print(realized_asset.value_counts(normalize=True).rename("frequency"))

    save_sector_rotation_outputs(results)

    print()
    print("Saved outputs:")
    print(f"- {SIGNALS_OUTPUT_PATH}")
    print(f"- {RETURNS_OUTPUT_PATH}")
    print(f"- {WEIGHTS_OUTPUT_PATH}")
    print(f"- {SUMMARY_OUTPUT_PATH}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()