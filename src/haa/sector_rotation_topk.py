"""
sector_rotation_topk.py

Diversified sector rotation extension of the HAA-style strategy.

Instead of allocating 100% to the single best sector, this version allocates
equally across the top-k assets:

- in risk-on regimes:
      top-k offensive sector ETFs

- in risk-off regimes:
      top-k defensive assets

Signals are computed at month-end and implemented for the following month.

Run from the project root with:

    python src/haa/sector_rotation_topk.py
"""

from pathlib import Path
import sys
from typing import Iterable

import numpy as np
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
    lag_weights_to_next_month,
)
from haa.momentum import (
    DEFAULT_HORIZONS,
    compute_multi_horizon_momentum,
    compute_absolute_momentum_signal,
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

TOP_K = 2

BACKTEST_START_DATE = "2005-01-31"
BACKTEST_END_DATE = "2025-12-31"

TRANSACTION_COST_BPS = 5.0
PERIODS_PER_YEAR = 12

OUTPUT_DIR = Path("data")
RETURNS_OUTPUT_PATH = OUTPUT_DIR / "sector_rotation_top2_returns.csv"
WEIGHTS_OUTPUT_PATH = OUTPUT_DIR / "sector_rotation_top2_weights.csv"
SIGNALS_OUTPUT_PATH = OUTPUT_DIR / "sector_rotation_top2_signals.csv"
SUMMARY_OUTPUT_PATH = OUTPUT_DIR / "sector_rotation_top2_summary.csv"


# ============================================================
# Signal construction
# ============================================================

def select_top_k_assets(
    momentum: pd.DataFrame,
    assets: Iterable[str],
    top_k: int,
) -> pd.DataFrame:
    """
    Select the top-k assets by momentum at each date.

    Parameters
    ----------
    momentum:
        Momentum score matrix.
    assets:
        Candidate assets.
    top_k:
        Number of assets to select.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns rank_1, rank_2, ..., rank_k.
    """
    assets = list(assets)

    if top_k <= 0:
        raise ValueError("top_k must be strictly positive.")

    if top_k > len(assets):
        raise ValueError("top_k cannot exceed the number of candidate assets.")

    missing_assets = [asset for asset in assets if asset not in momentum.columns]
    if missing_assets:
        raise ValueError(f"Missing assets in momentum matrix: {missing_assets}")

    arr = momentum[assets].to_numpy()
    col_names = np.array(assets)
    top_k_indices = np.argsort(-arr, axis=1)[:, :top_k]

    selected = pd.DataFrame(
        col_names[top_k_indices],
        index=momentum.index,
        columns=[f"rank_{i + 1}" for i in range(top_k)],
    )

    return selected


def build_topk_sector_rotation_signals(
    prices: pd.DataFrame,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    """
    Build top-k sector rotation signals.

    Parameters
    ----------
    prices:
        Monthly price matrix.
    top_k:
        Number of assets selected in each regime.

    Returns
    -------
    pd.DataFrame
        Signal table containing:
        - canary momentum;
        - risk-on signal;
        - top-k offensive assets;
        - top-k defensive assets.
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

    top_defensive_assets = select_top_k_assets(
        momentum=momentum,
        assets=SECTOR_DEFENSIVE_ASSETS,
        top_k=top_k,
    )
    top_defensive_assets = top_defensive_assets.add_prefix("defensive_")

    signals = pd.concat(
        [
            momentum[SECTOR_CANARY_ASSET].rename("canary_momentum"),
            risk_on,
            top_offensive_assets,
            top_defensive_assets,
        ],
        axis=1,
    )

    signals = signals.dropna()

    return signals


def build_topk_weights_from_signals(
    signals: pd.DataFrame,
    universe: Iterable[str],
    top_k: int = TOP_K,
) -> pd.DataFrame:
    """
    Convert top-k signals into equal-weighted portfolio weights.

    Parameters
    ----------
    signals:
        Signal table.
    universe:
        Tradable universe.
    top_k:
        Number of assets held in each regime.

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
    defensive_columns = [f"defensive_rank_{i + 1}" for i in range(top_k)]

    required_columns = ["risk_on"] + offensive_columns + defensive_columns
    missing_columns = [col for col in required_columns if col not in signals.columns]

    if missing_columns:
        raise ValueError(f"Missing signal columns: {missing_columns}")

    equal_weight = 1.0 / top_k

    all_assets = signals[offensive_columns + defensive_columns].to_numpy().flatten()
    invalid = set(all_assets) - set(universe)
    if invalid:
        raise ValueError(f"Selected assets not in universe: {sorted(invalid)}")

    risk_on_mask = (signals["risk_on"] == 1).astype(float)
    risk_off_mask = 1.0 - risk_on_mask

    for col in offensive_columns:
        dummies = pd.get_dummies(signals[col]).reindex(columns=universe, fill_value=0)
        weights += dummies.mul(risk_on_mask * equal_weight, axis=0)

    for col in defensive_columns:
        dummies = pd.get_dummies(signals[col]).reindex(columns=universe, fill_value=0)
        weights += dummies.mul(risk_off_mask * equal_weight, axis=0)

    return weights


def build_topk_sector_rotation_weights(
    prices: pd.DataFrame,
    top_k: int = TOP_K,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build top-k sector rotation signals and lagged portfolio weights.
    """
    signals = build_topk_sector_rotation_signals(
        prices=prices,
        top_k=top_k,
    )

    universe = sorted(
        set(SECTOR_OFFENSIVE_ASSETS + SECTOR_DEFENSIVE_ASSETS)
    )

    signal_weights = build_topk_weights_from_signals(
        signals=signals,
        universe=universe,
        top_k=top_k,
    )

    weights = lag_weights_to_next_month(signal_weights)

    return signals, weights


# ============================================================
# Backtest
# ============================================================

def run_topk_sector_rotation_backtest(
    prices: pd.DataFrame,
    top_k: int = TOP_K,
    transaction_cost_bps: float = TRANSACTION_COST_BPS,
    backtest_start_date: str = BACKTEST_START_DATE,
    backtest_end_date: str | None = BACKTEST_END_DATE,
) -> dict[str, object]:
    """
    Run the top-k sector rotation backtest.
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

    signals, weights = build_topk_sector_rotation_weights(
        prices=prices,
        top_k=top_k,
    )

    strategy_gross_returns = compute_portfolio_returns(
        returns=returns,
        weights=weights,
    )
    strategy_gross_returns.name = f"Sector Rotation Top-{top_k} Gross"

    turnover = compute_turnover(weights)

    strategy_net_returns = apply_transaction_costs(
        portfolio_returns=strategy_gross_returns,
        turnover=turnover,
        transaction_cost_bps=transaction_cost_bps,
    )
    strategy_net_returns.name = f"Sector Rotation Top-{top_k} Net"

    return_series = [
        strategy_gross_returns,
        strategy_net_returns,
    ]

    for benchmark in SECTOR_BENCHMARKS:
        benchmark_returns = returns[benchmark].copy()
        benchmark_returns.name = f"{benchmark} Buy and Hold"
        return_series.append(benchmark_returns)

    combined_returns = pd.concat(return_series, axis=1)
    combined_turnover = turnover.rename(f"Sector Rotation Top-{top_k} Turnover")

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
        if column.startswith(f"Sector Rotation Top-{top_k}"):
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

def save_topk_sector_rotation_outputs(results: dict[str, object]) -> None:
    """
    Save top-k sector rotation outputs to CSV files.
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
    Run the top-k sector rotation extension.
    """
    print("=" * 80)
    print(f"HAA-style sector rotation extension: Top-{TOP_K}")
    print("=" * 80)

    prices = load_prices(MONTHLY_PRICES_PATH)

    print()
    print("Monthly prices loaded:")
    print(prices.tail())
    print()
    print(f"Shape: {prices.shape}")
    print(f"Date range: {prices.index.min().date()} to {prices.index.max().date()}")

    results = run_topk_sector_rotation_backtest(
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

    save_topk_sector_rotation_outputs(results)

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