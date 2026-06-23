"""
sector_rotation_top2_shy.py

Defensive version of the top-2 sector rotation strategy.

The strategy is:

Risk-on regime:
    Allocate 50% to each of the top-2 offensive sector ETFs.

Risk-off regime:
    Allocate 100% to SHY.

This tests whether the drawdown problem of the sector rotation extension
comes from remaining too exposed to equity sectors in unfavorable regimes.

Run from the project root with:

    python src/haa/sector_rotation_top2_shy.py
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
)
from haa.momentum import (
    DEFAULT_HORIZONS,
    compute_multi_horizon_momentum,
    compute_absolute_momentum_signal,
)
from haa.sector_rotation_topk import (
    select_top_k_assets,
    lag_weights_to_next_month,
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

DEFENSIVE_ASSET = "SHY"

SECTOR_BENCHMARKS = [
    "VTI",
    "SPY",
]

TOP_K = 2

BACKTEST_START_DATE = "2005-01-31"
BACKTEST_END_DATE = "2025-12-31"

TRANSACTION_COST_BPS = 5.0
PERIODS_PER_YEAR = 12

OUTPUT_DIR = Path("data")
RETURNS_OUTPUT_PATH = OUTPUT_DIR / "sector_rotation_top2_shy_returns.csv"
WEIGHTS_OUTPUT_PATH = OUTPUT_DIR / "sector_rotation_top2_shy_weights.csv"
SIGNALS_OUTPUT_PATH = OUTPUT_DIR / "sector_rotation_top2_shy_signals.csv"
SUMMARY_OUTPUT_PATH = OUTPUT_DIR / "sector_rotation_top2_shy_summary.csv"


# ============================================================
# Signal construction
# ============================================================

def build_top2_shy_signals(
    prices: pd.DataFrame,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    """
    Build signals for the top-2 sector rotation with SHY risk-off.

    Parameters
    ----------
    prices:
        Monthly price matrix.
    top_k:
        Number of offensive sectors selected in risk-on regimes.

    Returns
    -------
    pd.DataFrame
        Signal table containing:
        - canary momentum;
        - risk-on signal;
        - top-k offensive sectors.
    """
    momentum = compute_multi_horizon_momentum(
        prices=prices,
        horizons=DEFAULT_HORIZONS,
    )

    risk_on = compute_absolute_momentum_signal(
        momentum=momentum,
        canary_asset=SECTOR_CANARY_ASSET,
    )

    top_offensive_assets = select_top_k_assets(
        momentum=momentum,
        assets=SECTOR_OFFENSIVE_ASSETS,
        top_k=top_k,
    )
    top_offensive_assets = top_offensive_assets.add_prefix("offensive_")

    signals = pd.concat(
        [
            momentum[SECTOR_CANARY_ASSET].rename("canary_momentum"),
            risk_on,
            top_offensive_assets,
        ],
        axis=1,
    )

    signals["defensive_asset"] = DEFENSIVE_ASSET
    signals = signals.dropna()

    return signals


def build_top2_shy_weights_from_signals(
    signals: pd.DataFrame,
    universe: Iterable[str],
    top_k: int = TOP_K,
) -> pd.DataFrame:
    """
    Convert top-2 offensive signals and SHY risk-off into portfolio weights.

    Parameters
    ----------
    signals:
        Signal table.
    universe:
        Tradable universe.
    top_k:
        Number of offensive sectors held in risk-on regimes.

    Returns
    -------
    pd.DataFrame
        Signal-date portfolio weights.
    """
    universe = list(universe)

    weights = pd.DataFrame(
        data=0.0,
        index=signals.index,
        columns=universe,
    )

    offensive_columns = [f"offensive_rank_{i + 1}" for i in range(top_k)]
    required_columns = ["risk_on"] + offensive_columns + ["defensive_asset"]

    missing_columns = [col for col in required_columns if col not in signals.columns]

    if missing_columns:
        raise ValueError(f"Missing signal columns: {missing_columns}")

    equal_weight = 1.0 / top_k

    for date, row in signals.iterrows():
        if row["risk_on"] == 1:
            selected_assets = [row[col] for col in offensive_columns]

            for asset in selected_assets:
                if asset not in weights.columns:
                    raise ValueError(f"Selected asset {asset} is not in the universe.")
                weights.loc[date, asset] += equal_weight

        else:
            defensive_asset = row["defensive_asset"]

            if defensive_asset not in weights.columns:
                raise ValueError(
                    f"Defensive asset {defensive_asset} is not in the universe."
                )

            weights.loc[date, defensive_asset] = 1.0

    return weights


def build_top2_shy_weights(
    prices: pd.DataFrame,
    top_k: int = TOP_K,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build signals and lagged weights for the top-2 SHY strategy.

    Parameters
    ----------
    prices:
        Monthly price matrix.
    top_k:
        Number of offensive sectors held in risk-on regimes.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Signals and lagged portfolio weights.
    """
    signals = build_top2_shy_signals(
        prices=prices,
        top_k=top_k,
    )

    universe = sorted(
        set(SECTOR_OFFENSIVE_ASSETS + [DEFENSIVE_ASSET])
    )

    signal_weights = build_top2_shy_weights_from_signals(
        signals=signals,
        universe=universe,
        top_k=top_k,
    )

    weights = lag_weights_to_next_month(signal_weights)

    return signals, weights


# ============================================================
# Backtest
# ============================================================

def run_top2_shy_backtest(
    prices: pd.DataFrame,
    top_k: int = TOP_K,
    transaction_cost_bps: float = TRANSACTION_COST_BPS,
    backtest_start_date: str = BACKTEST_START_DATE,
    backtest_end_date: str | None = BACKTEST_END_DATE,
) -> dict[str, object]:
    """
    Run the top-2 sector rotation with SHY risk-off backtest.
    """
    required_assets = (
        [SECTOR_CANARY_ASSET]
        + SECTOR_OFFENSIVE_ASSETS
        + [DEFENSIVE_ASSET]
        + SECTOR_BENCHMARKS
    )

    missing_assets = [asset for asset in required_assets if asset not in prices.columns]
    if missing_assets:
        raise ValueError(f"Missing required assets in prices: {missing_assets}")

    returns = compute_simple_returns(prices)

    signals, weights = build_top2_shy_weights(
        prices=prices,
        top_k=top_k,
    )

    strategy_gross_returns = compute_portfolio_returns(
        returns=returns,
        weights=weights,
    )
    strategy_gross_returns.name = "Sector Top-2 SHY Gross"

    turnover = compute_turnover(weights)

    strategy_net_returns = apply_transaction_costs(
        portfolio_returns=strategy_gross_returns,
        turnover=turnover,
        transaction_cost_bps=transaction_cost_bps,
    )
    strategy_net_returns.name = "Sector Top-2 SHY Net"

    return_series = [
        strategy_gross_returns,
        strategy_net_returns,
    ]

    for benchmark in SECTOR_BENCHMARKS:
        benchmark_returns = returns[benchmark].copy()
        benchmark_returns.name = f"{benchmark} Buy and Hold"
        return_series.append(benchmark_returns)

    combined_returns = pd.concat(return_series, axis=1)
    combined_turnover = turnover.rename("Sector Top-2 SHY Turnover")

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
        if column.startswith("Sector Top-2 SHY"):
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

def save_top2_shy_outputs(results: dict[str, object]) -> None:
    """
    Save top-2 SHY strategy outputs to CSV files.
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
    Run the defensive top-2 sector rotation strategy.
    """
    print("=" * 80)
    print("HAA-style sector rotation extension: Top-2 with SHY risk-off")
    print("=" * 80)

    prices = load_prices(MONTHLY_PRICES_PATH)

    print()
    print("Monthly prices loaded:")
    print(prices.tail())
    print()
    print(f"Shape: {prices.shape}")
    print(f"Date range: {prices.index.min().date()} to {prices.index.max().date()}")

    results = run_top2_shy_backtest(
        prices=prices,
        top_k=TOP_K,
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
    print("Realized average weights:")
    print(weights.mean().sort_values(ascending=False).rename("average_weight"))

    print()
    print("Realized asset usage frequency:")
    asset_usage = (weights > 0).mean().sort_values(ascending=False)
    print(asset_usage.rename("frequency"))

    save_top2_shy_outputs(results)

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