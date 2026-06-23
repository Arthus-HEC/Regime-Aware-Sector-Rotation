"""
robustness_canary.py

Robustness test for the sector Top-2 SHY strategy.

The baseline strategy uses TIP as a canary asset. This script tests whether
the results are sensitive to the choice of canary.

For each canary asset, the strategy is:

Risk-on:
    50% top offensive sector 1
    50% top offensive sector 2

Risk-off:
    100% SHY

Only the canary asset changes.

Run from the project root with:

    python src/haa/robustness_canary.py
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

CANARY_ASSETS = [
    "TIP",
    "SPY",
    "VTI",
    "SHY",
]

SECTOR_OFFENSIVE_ASSETS = [
    "XLK",  # Technology
    "XLY",  # Consumer Discretionary
    "XLI",  # Industrials
    "XLF",  # Financials
    "XLE",  # Energy
]

DEFENSIVE_ASSET = "SHY"

BENCHMARKS = [
    "VTI",
    "SPY",
]

TOP_K = 2

BACKTEST_START_DATE = "2005-01-31"
BACKTEST_END_DATE = "2025-12-31"

TRANSACTION_COST_BPS = 5.0
PERIODS_PER_YEAR = 12

OUTPUT_DIR = Path("data")
CANARY_SUMMARY_OUTPUT_PATH = OUTPUT_DIR / "robustness_canary_summary.csv"
CANARY_RETURNS_OUTPUT_PATH = OUTPUT_DIR / "robustness_canary_returns.csv"
CANARY_RISK_ON_OUTPUT_PATH = OUTPUT_DIR / "robustness_canary_risk_on_frequency.csv"


# ============================================================
# Signal and weights
# ============================================================

def build_top2_shy_signals_with_canary(
    prices: pd.DataFrame,
    canary_asset: str,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    """
    Build signals for the Top-2 SHY strategy with a chosen canary.

    Parameters
    ----------
    prices:
        Monthly price matrix.
    canary_asset:
        Asset used to define the risk-on / risk-off regime.
    top_k:
        Number of offensive sectors selected in risk-on regimes.

    Returns
    -------
    pd.DataFrame
        Signal table.
    """
    momentum = compute_multi_horizon_momentum(
        prices=prices,
        horizons=DEFAULT_HORIZONS,
    )

    risk_on = compute_absolute_momentum_signal(
        momentum=momentum,
        canary_asset=canary_asset,
    )

    top_offensive_assets = select_top_k_assets(
        momentum=momentum,
        assets=SECTOR_OFFENSIVE_ASSETS,
        top_k=top_k,
    )
    top_offensive_assets = top_offensive_assets.add_prefix("offensive_")

    signals = pd.concat(
        [
            momentum[canary_asset].rename("canary_momentum"),
            risk_on,
            top_offensive_assets,
        ],
        axis=1,
    )

    signals["canary_asset"] = canary_asset
    signals["defensive_asset"] = DEFENSIVE_ASSET

    signals = signals.dropna()

    return signals


def build_top2_shy_weights_from_signals(
    signals: pd.DataFrame,
    universe: Iterable[str],
    top_k: int = TOP_K,
) -> pd.DataFrame:
    """
    Convert signals into portfolio weights.

    In risk-on regimes, the strategy holds the top-k offensive sectors.
    In risk-off regimes, the strategy holds 100% SHY.
    """
    universe = list(universe)

    weights = pd.DataFrame(
        data=0.0,
        index=signals.index,
        columns=universe,
    )

    offensive_columns = [f"offensive_rank_{i + 1}" for i in range(top_k)]

    required_columns = ["risk_on"] + offensive_columns + ["defensive_asset"]
    missing_columns = [column for column in required_columns if column not in signals.columns]

    if missing_columns:
        raise ValueError(f"Missing signal columns: {missing_columns}")

    equal_weight = 1.0 / top_k

    invalid_off = set(signals[offensive_columns].to_numpy().flatten()) - set(universe)
    if invalid_off:
        raise ValueError(f"Offensive assets not in universe: {sorted(invalid_off)}")

    invalid_def = set(signals["defensive_asset"].unique()) - set(universe)
    if invalid_def:
        raise ValueError(f"Defensive assets not in universe: {sorted(invalid_def)}")

    risk_on_mask = (signals["risk_on"] == 1).astype(float)
    risk_off_mask = 1.0 - risk_on_mask

    for col in offensive_columns:
        dummies = pd.get_dummies(signals[col]).reindex(columns=universe, fill_value=0)
        weights += dummies.mul(risk_on_mask * equal_weight, axis=0)

    dummies_def = pd.get_dummies(signals["defensive_asset"]).reindex(columns=universe, fill_value=0)
    weights += dummies_def.mul(risk_off_mask, axis=0)

    return weights


def build_top2_shy_weights_with_canary(
    prices: pd.DataFrame,
    canary_asset: str,
    top_k: int = TOP_K,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build signals and lagged weights for a given canary.
    """
    signals = build_top2_shy_signals_with_canary(
        prices=prices,
        canary_asset=canary_asset,
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

def run_top2_shy_canary_backtest(
    prices: pd.DataFrame,
    canary_asset: str,
    top_k: int = TOP_K,
    transaction_cost_bps: float = TRANSACTION_COST_BPS,
    backtest_start_date: str = BACKTEST_START_DATE,
    backtest_end_date: str | None = BACKTEST_END_DATE,
) -> dict[str, object]:
    """
    Run the Top-2 SHY strategy with a chosen canary.
    """
    required_assets = (
        [canary_asset]
        + SECTOR_OFFENSIVE_ASSETS
        + [DEFENSIVE_ASSET]
        + BENCHMARKS
    )

    missing_assets = [asset for asset in required_assets if asset not in prices.columns]

    if missing_assets:
        raise ValueError(f"Missing required assets in prices: {missing_assets}")

    returns = compute_simple_returns(prices)

    signals, weights = build_top2_shy_weights_with_canary(
        prices=prices,
        canary_asset=canary_asset,
        top_k=top_k,
    )

    strategy_name = f"Top-2 SHY Canary={canary_asset}"

    strategy_gross_returns = compute_portfolio_returns(
        returns=returns,
        weights=weights,
    )
    strategy_gross_returns.name = f"{strategy_name} Gross"

    turnover = compute_turnover(weights)

    strategy_net_returns = apply_transaction_costs(
        portfolio_returns=strategy_gross_returns,
        turnover=turnover,
        transaction_cost_bps=transaction_cost_bps,
    )
    strategy_net_returns.name = f"{strategy_name} Net"

    if backtest_start_date is not None:
        strategy_net_returns = strategy_net_returns.loc[
            strategy_net_returns.index >= backtest_start_date
        ]
        turnover = turnover.loc[
            turnover.index >= backtest_start_date
        ]
        signals = signals.loc[
            signals.index >= backtest_start_date
        ]
        weights = weights.loc[
            weights.index >= backtest_start_date
        ]

    if backtest_end_date is not None:
        strategy_net_returns = strategy_net_returns.loc[
            strategy_net_returns.index <= backtest_end_date
        ]
        turnover = turnover.loc[
            turnover.index <= backtest_end_date
        ]
        signals = signals.loc[
            signals.index <= backtest_end_date
        ]
        weights = weights.loc[
            weights.index <= backtest_end_date
        ]

    strategy_net_returns = strategy_net_returns.dropna()
    turnover_aligned = turnover.reindex(strategy_net_returns.index).fillna(0.0)

    summary = compute_performance_summary(
        returns=strategy_net_returns,
        periods_per_year=PERIODS_PER_YEAR,
        turnover=turnover_aligned,
    )

    risk_on_frequency = float(signals["risk_on"].mean())

    average_weights = weights.mean().sort_values(ascending=False)

    return {
        "canary_asset": canary_asset,
        "returns": strategy_net_returns,
        "turnover": turnover_aligned,
        "signals": signals,
        "weights": weights,
        "summary": summary,
        "risk_on_frequency": risk_on_frequency,
        "average_weights": average_weights,
    }


def build_benchmark_summaries(
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build benchmark returns and summaries.
    """
    returns = compute_simple_returns(prices)

    benchmark_returns = {}

    for benchmark in BENCHMARKS:
        series = returns[benchmark].copy()
        series.name = f"{benchmark} Buy and Hold"

        series = series.loc[series.index >= BACKTEST_START_DATE]
        series = series.loc[series.index <= BACKTEST_END_DATE]
        series = series.dropna()

        benchmark_returns[series.name] = series

    benchmark_returns_df = pd.DataFrame(benchmark_returns).dropna()

    rows = {}

    for column in benchmark_returns_df.columns:
        rows[column] = compute_performance_summary(
            returns=benchmark_returns_df[column],
            periods_per_year=PERIODS_PER_YEAR,
            turnover=None,
        )

    benchmark_summary = pd.DataFrame(rows).T

    return benchmark_returns_df, benchmark_summary


# ============================================================
# Main
# ============================================================

def main() -> None:
    """
    Run canary robustness tests.
    """
    print("=" * 80)
    print("Robustness test: canary asset")
    print("=" * 80)

    prices = load_prices(MONTHLY_PRICES_PATH)

    print()
    print("Monthly prices loaded:")
    print(f"Shape: {prices.shape}")
    print(f"Date range: {prices.index.min().date()} to {prices.index.max().date()}")

    strategy_returns = {}
    summary_rows = {}
    risk_on_rows = {}

    for canary_asset in CANARY_ASSETS:
        print()
        print("-" * 80)
        print(f"Running canary test: {canary_asset}")
        print("-" * 80)

        result = run_top2_shy_canary_backtest(
            prices=prices,
            canary_asset=canary_asset,
            top_k=TOP_K,
            transaction_cost_bps=TRANSACTION_COST_BPS,
            backtest_start_date=BACKTEST_START_DATE,
            backtest_end_date=BACKTEST_END_DATE,
        )

        strategy_name = f"Top-2 SHY Canary={canary_asset}"

        strategy_returns[strategy_name] = result["returns"]
        summary_rows[strategy_name] = result["summary"]

        risk_on_rows[strategy_name] = {
            "risk_on_frequency": result["risk_on_frequency"],
        }

        print("Performance:")
        print(
            format_performance_summary(
                pd.DataFrame([result["summary"]], index=[strategy_name])
            )
        )

        print()
        print("Risk-on frequency:")
        print(f"{100 * result['risk_on_frequency']:.2f}%")

        print()
        print("Average weights:")
        print(result["average_weights"].map(lambda x: f"{100 * x:.2f}%"))

    benchmark_returns, benchmark_summary = build_benchmark_summaries(prices)

    all_returns = pd.concat(
        [
            pd.DataFrame(strategy_returns),
            benchmark_returns,
        ],
        axis=1,
    ).dropna()

    summary = pd.concat(
        [
            pd.DataFrame(summary_rows).T,
            benchmark_summary,
        ],
        axis=0,
    )

    risk_on_summary = pd.DataFrame(risk_on_rows).T

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_returns.to_csv(CANARY_RETURNS_OUTPUT_PATH)
    summary.to_csv(CANARY_SUMMARY_OUTPUT_PATH)
    risk_on_summary.to_csv(CANARY_RISK_ON_OUTPUT_PATH)

    print()
    print("=" * 80)
    print("Final canary robustness summary")
    print("=" * 80)

    print()
    print("Formatted summary:")
    print(format_performance_summary(summary))

    print()
    print("Risk-on frequencies:")
    print(risk_on_summary.map(lambda x: f"{100 * x:.2f}%"))

    print()
    print("Ranking by CAGR:")
    print(summary["cagr"].sort_values(ascending=False).map(lambda x: f"{100 * x:.2f}%"))

    print()
    print("Ranking by Sharpe ratio:")
    print(summary["sharpe_ratio"].sort_values(ascending=False).map(lambda x: f"{x:.2f}"))

    print()
    print("Ranking by maximum drawdown:")
    print(summary["max_drawdown"].sort_values(ascending=False).map(lambda x: f"{100 * x:.2f}%"))

    print()
    print("Saved outputs:")
    print(f"- {CANARY_RETURNS_OUTPUT_PATH}")
    print(f"- {CANARY_SUMMARY_OUTPUT_PATH}")
    print(f"- {CANARY_RISK_ON_OUTPUT_PATH}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()