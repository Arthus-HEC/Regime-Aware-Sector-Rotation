"""
momentum.py

Momentum signal utilities for the regime-aware sector rotation project.

The core signal is the 13612 momentum score used in HAA-style strategies:

    M_t = (R_1m + R_3m + R_6m + R_12m) / 4

where R_hm is the trailing total return over h months.

Run from the project root with:

    python src/haa/momentum.py
"""

from pathlib import Path
import sys
from typing import Iterable, Sequence

import pandas as pd


# Make imports robust when running the file directly from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from common.data import load_prices, MONTHLY_PRICES_PATH


# ============================================================
# Configuration
# ============================================================

DEFAULT_HORIZONS = (1, 3, 6, 12)

REPLICATION_OFFENSIVE_ASSETS = [
    "SPY",  # Large-cap U.S. equities
    "MDY",  # Mid-cap U.S. equities
    "IJR",  # Small-cap U.S. equities
]

REPLICATION_CANARY_ASSET = "TIP"
REPLICATION_DEFENSIVE_ASSET = "SHY"
REPLICATION_BENCHMARK = "VTI"


# ============================================================
# Momentum functions
# ============================================================

def compute_trailing_return(
    prices: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """
    Compute trailing total returns over a given number of months.

    Parameters
    ----------
    prices:
        Monthly price matrix indexed by date, with one column per asset.
    horizon:
        Lookback horizon in months.

    Returns
    -------
    pd.DataFrame
        Trailing return matrix over the selected horizon.
    """
    if horizon <= 0:
        raise ValueError("The horizon must be strictly positive.")

    if prices.empty:
        raise ValueError("The price matrix is empty.")

    returns = prices / prices.shift(horizon) - 1.0
    return returns


def compute_multi_horizon_momentum(
    prices: pd.DataFrame,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """
    Compute a multi-horizon momentum score.

    By default, this computes the 13612 momentum signal:

        M_t = (R_1m + R_3m + R_6m + R_12m) / 4

    Parameters
    ----------
    prices:
        Monthly price matrix indexed by date.
    horizons:
        Lookback horizons in months.

    Returns
    -------
    pd.DataFrame
        Momentum score matrix indexed by date.
    """
    if not horizons:
        raise ValueError("At least one horizon must be provided.")

    trailing_returns = [
        compute_trailing_return(prices=prices, horizon=h)
        for h in horizons
    ]

    momentum = sum(trailing_returns) / len(trailing_returns)
    momentum = momentum.dropna(how="all")

    return momentum


def compute_absolute_momentum_signal(
    momentum: pd.DataFrame,
    canary_asset: str,
) -> pd.Series:
    """
    Compute a binary risk-on / risk-off signal from a canary asset.

    Parameters
    ----------
    momentum:
        Momentum score matrix.
    canary_asset:
        Asset used as the regime indicator.

    Returns
    -------
    pd.Series
        Binary signal indexed by date:
        - 1 if canary momentum is positive;
        - 0 otherwise.
    """
    if canary_asset not in momentum.columns:
        raise ValueError(f"Canary asset '{canary_asset}' is not in momentum columns.")

    signal = (momentum[canary_asset] > 0).astype(int)
    signal.name = "risk_on"

    return signal


def rank_assets_by_momentum(
    momentum: pd.DataFrame,
    assets: Iterable[str],
) -> pd.DataFrame:
    """
    Rank a set of assets by their momentum score.

    Parameters
    ----------
    momentum:
        Momentum score matrix.
    assets:
        Assets to rank.

    Returns
    -------
    pd.DataFrame
        Rank matrix. Rank 1 means highest momentum.
    """
    assets = list(assets)

    missing_assets = [asset for asset in assets if asset not in momentum.columns]
    if missing_assets:
        raise ValueError(f"Missing assets in momentum matrix: {missing_assets}")

    ranks = momentum[assets].rank(axis=1, ascending=False, method="first")
    return ranks


def select_top_momentum_asset(
    momentum: pd.DataFrame,
    assets: Iterable[str],
) -> pd.Series:
    """
    Select the asset with the highest momentum score at each date.

    Parameters
    ----------
    momentum:
        Momentum score matrix.
    assets:
        Candidate assets.

    Returns
    -------
    pd.Series
        Selected ticker at each date.
    """
    assets = list(assets)

    missing_assets = [asset for asset in assets if asset not in momentum.columns]
    if missing_assets:
        raise ValueError(f"Missing assets in momentum matrix: {missing_assets}")

    selected = momentum[assets].idxmax(axis=1)
    selected.name = "selected_asset"

    return selected


def build_momentum_signals(
    prices: pd.DataFrame,
    offensive_assets: Iterable[str],
    canary_asset: str,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """
    Build the main HAA-style signal table.

    Parameters
    ----------
    prices:
        Monthly price matrix.
    offensive_assets:
        Risk assets among which the strategy selects in favorable regimes.
    canary_asset:
        Asset used to determine the risk-on / risk-off regime.
    horizons:
        Momentum horizons.

    Returns
    -------
    pd.DataFrame
        Signal table containing:
        - canary momentum;
        - risk-on signal;
        - selected offensive asset;
        - selected offensive asset momentum.
    """
    offensive_assets = list(offensive_assets)

    momentum = compute_multi_horizon_momentum(
        prices=prices,
        horizons=horizons,
    )

    risk_on = compute_absolute_momentum_signal(
        momentum=momentum,
        canary_asset=canary_asset,
    )

    selected_asset = select_top_momentum_asset(
        momentum=momentum,
        assets=offensive_assets,
    )

    selected_asset_momentum = pd.Series(
        data=[
            momentum.loc[date, asset]
            for date, asset in selected_asset.items()
        ],
        index=selected_asset.index,
        name="selected_asset_momentum",
    )

    signals = pd.concat(
        [
            momentum[canary_asset].rename("canary_momentum"),
            risk_on,
            selected_asset,
            selected_asset_momentum,
        ],
        axis=1,
    )

    signals = signals.dropna()

    return signals


# ============================================================
# Script entry point
# ============================================================

def main() -> None:
    """
    Compute and display the main 13612 momentum signals.
    """
    print("=" * 80)
    print("Computing HAA-style 13612 momentum signals")
    print("=" * 80)

    prices = load_prices(MONTHLY_PRICES_PATH)

    print()
    print("Monthly prices loaded:")
    print(prices.tail())
    print()
    print(f"Shape: {prices.shape}")
    print(f"Date range: {prices.index.min().date()} to {prices.index.max().date()}")

    momentum = compute_multi_horizon_momentum(
        prices=prices,
        horizons=DEFAULT_HORIZONS,
    )

    print()
    print("Momentum scores:")
    print(momentum.tail())
    print()
    print(f"Momentum shape: {momentum.shape}")

    signals = build_momentum_signals(
        prices=prices,
        offensive_assets=REPLICATION_OFFENSIVE_ASSETS,
        canary_asset=REPLICATION_CANARY_ASSET,
        horizons=DEFAULT_HORIZONS,
    )

    print()
    print("Replication signals:")
    print(signals.tail(12))

    print()
    print("Risk-on distribution:")
    print(signals["risk_on"].value_counts(normalize=True).rename("frequency"))

    print()
    print("Selected offensive asset distribution:")
    print(signals["selected_asset"].value_counts(normalize=True).rename("frequency"))

    print()
    print("Done.")


if __name__ == "__main__":
    main()