"""
replication.py

Replication of a HAA-style market-cap rotation strategy.

The replicated logic is:

1. Compute 13612 momentum scores:
       M_t = (R_1m + R_3m + R_6m + R_12m) / 4

2. Use TIP as the canary asset:
       if M_t(TIP) > 0 -> risk-on
       if M_t(TIP) <= 0 -> risk-off

3. In risk-on regimes:
       allocate 100% to the best momentum asset among SPY, MDY, IJR

4. In risk-off regimes:
       allocate 100% to SHY

5. Signals are computed at month-end and implemented for the following month.

Run from the project root with:

    python src/haa/replication.py
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
    REPLICATION_OFFENSIVE_ASSETS,
    REPLICATION_CANARY_ASSET,
    REPLICATION_DEFENSIVE_ASSET,
    REPLICATION_BENCHMARK,
    build_momentum_signals,
)


# ============================================================
# Configuration
# ============================================================

BACKTEST_START_DATE = "2005-01-31"
BACKTEST_END_DATE = None

TRANSACTION_COST_BPS = 5.0
PERIODS_PER_YEAR = 12

OUTPUT_DIR = Path("data")
RETURNS_OUTPUT_PATH = OUTPUT_DIR / "haa_replication_returns.csv"
WEIGHTS_OUTPUT_PATH = OUTPUT_DIR / "haa_replication_weights.csv"
SUMMARY_OUTPUT_PATH = OUTPUT_DIR / "haa_replication_summary.csv"


# ============================================================
# Strategy construction
# ============================================================

def build_target_assets(
    signals: pd.DataFrame,
    defensive_asset: str = REPLICATION_DEFENSIVE_ASSET,
) -> pd.Series:
    """
    Build the target asset selected at each signal date.

    Parameters
    ----------
    signals:
        Signal table containing risk_on and selected_asset.
    defensive_asset:
        Asset held when the canary signal is risk-off.

    Returns
    -------
    pd.Series
        Target asset at each signal date.

    Interpretation
    --------------
    If risk_on = 1, the strategy holds the selected offensive asset.
    If risk_on = 0, the strategy holds the defensive asset.
    """
    required_columns = {"risk_on", "selected_asset"}

    missing_columns = required_columns.difference(signals.columns)
    if missing_columns:
        raise ValueError(f"Missing signal columns: {missing_columns}")

    target_assets = signals["selected_asset"].copy()
    target_assets.loc[signals["risk_on"] == 0] = defensive_asset
    target_assets.name = "target_asset"

    return target_assets


def build_one_asset_weights(
    target_assets: pd.Series,
    universe: Iterable[str],
) -> pd.DataFrame:
    """
    Convert selected target assets into one-hot portfolio weights.

    Parameters
    ----------
    target_assets:
        Series containing selected target asset at each signal date.
    universe:
        Assets that can appear in the portfolio.

    Returns
    -------
    pd.DataFrame
        Weight matrix indexed by signal date.
    """
    universe = list(universe)

    if not universe:
        raise ValueError("Universe cannot be empty.")

    weights = pd.DataFrame(
        data=0.0,
        index=target_assets.index,
        columns=universe,
    )

    for date, asset in target_assets.items():
        if asset not in weights.columns:
            raise ValueError(f"Selected asset {asset} is not in the universe.")
        weights.loc[date, asset] = 1.0

    return weights


def lag_weights_to_next_month(
    signal_weights: pd.DataFrame,
) -> pd.DataFrame:
    """
    Lag signal weights by one month.

    Parameters
    ----------
    signal_weights:
        Weights decided at the end of each month.

    Returns
    -------
    pd.DataFrame
        Weights held during the following month.

    Important
    ---------
    If weights are computed using information available at month-end t,
    they can only be implemented for the return from t to t+1.

    Therefore, for the return indexed by date t, we use the weights decided
    at date t-1.
    """
    weights_for_returns = signal_weights.shift(1)
    weights_for_returns = weights_for_returns.dropna(how="all")

    return weights_for_returns


def build_replication_weights(
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build HAA replication signals and lagged portfolio weights.

    Parameters
    ----------
    prices:
        Monthly price matrix.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Signals and lagged weights.
    """
    signals = build_momentum_signals(
        prices=prices,
        offensive_assets=REPLICATION_OFFENSIVE_ASSETS,
        canary_asset=REPLICATION_CANARY_ASSET,
        horizons=DEFAULT_HORIZONS,
    )

    target_assets = build_target_assets(
        signals=signals,
        defensive_asset=REPLICATION_DEFENSIVE_ASSET,
    )

    universe = sorted(
        set(REPLICATION_OFFENSIVE_ASSETS + [REPLICATION_DEFENSIVE_ASSET])
    )

    signal_weights = build_one_asset_weights(
        target_assets=target_assets,
        universe=universe,
    )

    weights = lag_weights_to_next_month(signal_weights)

    signals = signals.copy()
    signals["target_asset"] = target_assets

    return signals, weights


# ============================================================
# Backtest
# ============================================================

def run_replication_backtest(
    prices: pd.DataFrame,
    transaction_cost_bps: float = TRANSACTION_COST_BPS,
    backtest_start_date: str = BACKTEST_START_DATE,
    backtest_end_date: str | None = BACKTEST_END_DATE,
) -> dict[str, object]:
    """
    Run the HAA replication backtest.

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
        Backtest outputs:
        - signals;
        - weights;
        - returns;
        - turnover;
        - performance summary;
        - equity curves;
        - drawdowns.
    """
    required_assets = (
        REPLICATION_OFFENSIVE_ASSETS
        + [
            REPLICATION_CANARY_ASSET,
            REPLICATION_DEFENSIVE_ASSET,
            REPLICATION_BENCHMARK,
        ]
    )

    missing_assets = [asset for asset in required_assets if asset not in prices.columns]
    if missing_assets:
        raise ValueError(f"Missing required assets in prices: {missing_assets}")

    returns = compute_simple_returns(prices)

    signals, weights = build_replication_weights(prices)

    strategy_gross_returns = compute_portfolio_returns(
        returns=returns,
        weights=weights,
    )
    strategy_gross_returns.name = "HAA Gross"

    turnover = compute_turnover(weights)

    strategy_net_returns = apply_transaction_costs(
        portfolio_returns=strategy_gross_returns,
        turnover=turnover,
        transaction_cost_bps=transaction_cost_bps,
    )
    strategy_net_returns.name = "HAA Net"

    benchmark_returns = returns[REPLICATION_BENCHMARK].copy()
    benchmark_returns.name = f"{REPLICATION_BENCHMARK} Buy and Hold"

    combined_returns = pd.concat(
        [
            strategy_gross_returns,
            strategy_net_returns,
            benchmark_returns,
        ],
        axis=1,
    )

    combined_turnover = turnover.rename("HAA Turnover")

    if backtest_start_date is not None:
        combined_returns = combined_returns.loc[combined_returns.index >= backtest_start_date]
        combined_turnover = combined_turnover.loc[combined_turnover.index >= backtest_start_date]
        weights = weights.loc[weights.index >= backtest_start_date]
        signals = signals.loc[signals.index >= backtest_start_date]

    if backtest_end_date is not None:
        combined_returns = combined_returns.loc[combined_returns.index <= backtest_end_date]
        combined_turnover = combined_turnover.loc[combined_turnover.index <= backtest_end_date]
        weights = weights.loc[weights.index <= backtest_end_date]
        signals = signals.loc[signals.index <= backtest_end_date]

    combined_returns = combined_returns.dropna()

    turnover_aligned = combined_turnover.reindex(combined_returns.index).fillna(0.0)

    performance_rows = {}

    performance_rows["HAA Gross"] = compute_performance_summary(
        returns=combined_returns["HAA Gross"],
        periods_per_year=PERIODS_PER_YEAR,
        turnover=turnover_aligned,
    )

    performance_rows["HAA Net"] = compute_performance_summary(
        returns=combined_returns["HAA Net"],
        periods_per_year=PERIODS_PER_YEAR,
        turnover=turnover_aligned,
    )

    performance_rows[f"{REPLICATION_BENCHMARK} Buy and Hold"] = compute_performance_summary(
        returns=combined_returns[f"{REPLICATION_BENCHMARK} Buy and Hold"],
        periods_per_year=PERIODS_PER_YEAR,
        turnover=None,
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

def save_replication_outputs(results: dict[str, object]) -> None:
    """
    Save replication outputs to CSV files.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    returns = results["returns"]
    weights = results["weights"]
    performance_summary = results["performance_summary"]

    if not isinstance(returns, pd.DataFrame):
        raise TypeError("results['returns'] must be a DataFrame.")
    if not isinstance(weights, pd.DataFrame):
        raise TypeError("results['weights'] must be a DataFrame.")
    if not isinstance(performance_summary, pd.DataFrame):
        raise TypeError("results['performance_summary'] must be a DataFrame.")

    returns.to_csv(RETURNS_OUTPUT_PATH)
    weights.to_csv(WEIGHTS_OUTPUT_PATH)
    performance_summary.to_csv(SUMMARY_OUTPUT_PATH)


def main() -> None:
    """
    Run the HAA replication.
    """
    print("=" * 80)
    print("HAA-style market-cap rotation replication")
    print("=" * 80)

    prices = load_prices(MONTHLY_PRICES_PATH)

    print()
    print("Monthly prices loaded:")
    print(prices.tail())
    print()
    print(f"Shape: {prices.shape}")
    print(f"Date range: {prices.index.min().date()} to {prices.index.max().date()}")

    results = run_replication_backtest(
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
    print("Realized target asset distribution:")
    realized_asset = weights.idxmax(axis=1)
    print(realized_asset.value_counts(normalize=True).rename("frequency"))

    save_replication_outputs(results)

    print()
    print("Saved outputs:")
    print(f"- {RETURNS_OUTPUT_PATH}")
    print(f"- {WEIGHTS_OUTPUT_PATH}")
    print(f"- {SUMMARY_OUTPUT_PATH}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()