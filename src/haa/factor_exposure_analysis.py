"""
factor_exposure_analysis.py

ETF proxy factor exposure analysis for the regime-aware sector rotation project.

The goal is to check whether strategy performance is explained by broad
market and style exposures rather than by a genuine allocation effect.

This is not a full Fama-French factor model. It is a lightweight
ETF-based diagnostic using only the assets already present in the project.

Run from the project root with:

    python src/haa/factor_exposure_analysis.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================
# Imports
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path("data")

MONTHLY_PRICES_PATH = DATA_DIR / "monthly_prices.csv"
COMPARISON_RETURNS_PATH = DATA_DIR / "strategy_comparison_returns.csv"

FACTOR_SUMMARY_PATH = DATA_DIR / "factor_regression_summary.csv"
FACTOR_COEFFICIENTS_PATH = DATA_DIR / "factor_regression_coefficients.csv"

PERIODS_PER_YEAR = 12

TARGET_STRATEGIES = [
    "HAA Market-Cap Net",
    "Sector Top-1 Net",
    "Sector Top-2 Net",
    "Sector Top-2 SHY Net",
    "VTI Buy and Hold",
    "SPY Buy and Hold",
]

MODEL_SPECS = {
    "market_only": [
        "MKT_SPY",
    ],
    "market_and_defensive": [
        "MKT_SPY",
        "SHY",
    ],
    "style_proxy": [
        "MKT_SPY",
        "SIZE_IJR_MINUS_SPY",
        "MID_MDY_MINUS_SPY",
        "TIP_MINUS_SHY",
    ],
    "sector_proxy": [
        "MKT_SPY",
        "OFFENSIVE_SECTORS_MINUS_SPY",
        "DEFENSIVE_SECTORS_MINUS_SPY",
        "TIP_MINUS_SHY",
    ],
}


# ============================================================
# Loading
# ============================================================

def load_monthly_prices(
    path: Path = MONTHLY_PRICES_PATH,
) -> pd.DataFrame:
    """
    Load monthly ETF prices.

    Parameters
    ----------
    path:
        Path to monthly prices.

    Returns
    -------
    pd.DataFrame
        Monthly price matrix.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}. Run python src/common/data.py first."
        )

    prices = pd.read_csv(path, index_col=0, parse_dates=True)

    if prices.empty:
        raise ValueError("Monthly price file is empty.")

    return prices


def load_strategy_returns(
    path: Path = COMPARISON_RETURNS_PATH,
) -> pd.DataFrame:
    """
    Load strategy comparison returns.

    Parameters
    ----------
    path:
        Path to strategy comparison returns.

    Returns
    -------
    pd.DataFrame
        Strategy return matrix.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}. "
            "Run python src/haa/compare_strategies.py first."
        )

    returns = pd.read_csv(path, index_col=0, parse_dates=True)

    if returns.empty:
        raise ValueError("Strategy comparison returns file is empty.")

    missing = [col for col in TARGET_STRATEGIES if col not in returns.columns]
    if missing:
        raise ValueError(f"Missing expected strategy columns: {missing}")

    return returns[TARGET_STRATEGIES].dropna()


# ============================================================
# Factor construction
# ============================================================

def compute_simple_returns(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute monthly simple returns.

    Parameters
    ----------
    prices:
        Monthly price matrix.

    Returns
    -------
    pd.DataFrame
        Monthly simple returns.
    """
    returns = prices.pct_change(fill_method=None)
    returns = returns.dropna(how="all")
    return returns


def build_etf_proxy_factors(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build ETF proxy factors from the project universe.

    Parameters
    ----------
    prices:
        Monthly price matrix.

    Returns
    -------
    pd.DataFrame
        Factor return matrix.
    """
    required_assets = [
        "SPY",
        "SHY",
        "TIP",
        "IJR",
        "MDY",
        "XLK",
        "XLY",
        "XLI",
        "XLF",
        "XLE",
        "XLP",
        "XLU",
        "XLV",
    ]

    missing = [asset for asset in required_assets if asset not in prices.columns]
    if missing:
        raise ValueError(f"Missing assets for factor construction: {missing}")

    returns = compute_simple_returns(prices)

    offensive_sectors = ["XLK", "XLY", "XLI", "XLF", "XLE"]
    defensive_sectors = ["XLP", "XLU", "XLV"]

    factors = pd.DataFrame(index=returns.index)

    factors["MKT_SPY"] = returns["SPY"]
    factors["SHY"] = returns["SHY"]
    factors["SIZE_IJR_MINUS_SPY"] = returns["IJR"] - returns["SPY"]
    factors["MID_MDY_MINUS_SPY"] = returns["MDY"] - returns["SPY"]
    factors["TIP_MINUS_SHY"] = returns["TIP"] - returns["SHY"]
    factors["OFFENSIVE_SECTORS_MINUS_SPY"] = (
        returns[offensive_sectors].mean(axis=1) - returns["SPY"]
    )
    factors["DEFENSIVE_SECTORS_MINUS_SPY"] = (
        returns[defensive_sectors].mean(axis=1) - returns["SPY"]
    )

    return factors.dropna()


# ============================================================
# OLS implementation
# ============================================================

def run_ols(
    y: pd.Series,
    x: pd.DataFrame,
) -> dict[str, object]:
    """
    Run OLS with an intercept using NumPy.

    Parameters
    ----------
    y:
        Dependent variable.
    x:
        Factor matrix.

    Returns
    -------
    dict[str, object]
        Regression outputs.
    """
    data = pd.concat([y.rename("y"), x], axis=1).dropna()

    if data.empty:
        raise ValueError("Regression data is empty.")

    y_array = data["y"].to_numpy(dtype=float)
    x_array = data[x.columns].to_numpy(dtype=float)

    n_obs = len(data)
    x_design = np.column_stack(
        [
            np.ones(n_obs),
            x_array,
        ]
    )

    coefficient_names = ["alpha"] + list(x.columns)

    beta, _, _, _ = np.linalg.lstsq(
        x_design,
        y_array,
        rcond=None,
    )

    fitted = x_design @ beta
    residuals = y_array - fitted

    n_parameters = x_design.shape[1]
    degrees_freedom = n_obs - n_parameters

    if degrees_freedom <= 0:
        raise ValueError("Not enough observations for regression.")

    residual_variance = float(
        (residuals @ residuals) / degrees_freedom
    )

    xtx_inv = np.linalg.pinv(x_design.T @ x_design)
    covariance_matrix = residual_variance * xtx_inv

    standard_errors = np.sqrt(np.diag(covariance_matrix))
    t_stats = beta / standard_errors

    total_sum_squares = float(
        ((y_array - y_array.mean()) @ (y_array - y_array.mean()))
    )
    residual_sum_squares = float(residuals @ residuals)

    if total_sum_squares == 0:
        r_squared = np.nan
    else:
        r_squared = 1.0 - residual_sum_squares / total_sum_squares

    residual_volatility = float(
        pd.Series(residuals).std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)
    )

    coefficients = pd.DataFrame(
        {
            "coefficient": coefficient_names,
            "estimate": beta,
            "standard_error": standard_errors,
            "t_stat": t_stats,
        }
    )

    return {
        "n_obs": n_obs,
        "r_squared": r_squared,
        "residual_volatility": residual_volatility,
        "coefficients": coefficients,
    }


# ============================================================
# Analysis
# ============================================================

def run_factor_exposure_analysis(
    strategy_returns: pd.DataFrame,
    factors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run all factor regressions.

    Parameters
    ----------
    strategy_returns:
        Strategy return matrix.
    factors:
        Factor return matrix.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Regression summary and coefficient table.
    """
    summary_rows = []
    coefficient_frames = []

    for strategy in TARGET_STRATEGIES:
        for model_name, factor_names in MODEL_SPECS.items():
            x = factors[factor_names]
            y = strategy_returns[strategy]

            result = run_ols(y=y, x=x)

            coefficients = result["coefficients"].copy()
            coefficients["strategy"] = strategy
            coefficients["model"] = model_name

            alpha_row = coefficients.loc[
                coefficients["coefficient"] == "alpha"
            ].iloc[0]

            market_beta = np.nan
            if "MKT_SPY" in coefficients["coefficient"].values:
                market_beta = float(
                    coefficients.loc[
                        coefficients["coefficient"] == "MKT_SPY",
                        "estimate",
                    ].iloc[0]
                )

            summary_rows.append(
                {
                    "strategy": strategy,
                    "model": model_name,
                    "n_obs": result["n_obs"],
                    "annualized_alpha": alpha_row["estimate"] * PERIODS_PER_YEAR,
                    "alpha_t_stat": alpha_row["t_stat"],
                    "r_squared": result["r_squared"],
                    "market_beta": market_beta,
                    "residual_volatility": result["residual_volatility"],
                }
            )

            coefficient_frames.append(coefficients)

    summary = pd.DataFrame(summary_rows)
    coefficients = pd.concat(
        coefficient_frames,
        axis=0,
        ignore_index=True,
    )

    return summary, coefficients


# ============================================================
# Display
# ============================================================

def format_factor_summary(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Format summary for display.

    Parameters
    ----------
    summary:
        Raw factor summary.

    Returns
    -------
    pd.DataFrame
        Formatted table.
    """
    formatted = summary.copy()

    formatted["annualized_alpha"] = formatted["annualized_alpha"].map(
        lambda x: f"{100 * x:.2f}%"
    )
    formatted["alpha_t_stat"] = formatted["alpha_t_stat"].map(
        lambda x: f"{x:.2f}"
    )
    formatted["r_squared"] = formatted["r_squared"].map(
        lambda x: f"{x:.3f}"
    )
    formatted["market_beta"] = formatted["market_beta"].map(
        lambda x: f"{x:.3f}"
    )
    formatted["residual_volatility"] = formatted["residual_volatility"].map(
        lambda x: f"{100 * x:.2f}%"
    )

    return formatted


# ============================================================
# Main
# ============================================================

def main() -> None:
    """
    Run ETF proxy factor exposure analysis.
    """
    print("=" * 80)
    print("ETF proxy factor exposure analysis")
    print("=" * 80)

    prices = load_monthly_prices()
    strategy_returns = load_strategy_returns()
    factors = build_etf_proxy_factors(prices)

    common_index = strategy_returns.index.intersection(factors.index)
    strategy_returns = strategy_returns.loc[common_index]
    factors = factors.loc[common_index]

    print()
    print("Strategy returns:")
    print(strategy_returns.tail())
    print()
    print(f"Shape: {strategy_returns.shape}")
    print(f"Date range: {strategy_returns.index.min().date()} to {strategy_returns.index.max().date()}")

    print()
    print("ETF proxy factors:")
    print(factors.tail())
    print()
    print(f"Shape: {factors.shape}")

    summary, coefficients = run_factor_exposure_analysis(
        strategy_returns=strategy_returns,
        factors=factors,
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    summary.to_csv(FACTOR_SUMMARY_PATH, index=False)
    coefficients.to_csv(FACTOR_COEFFICIENTS_PATH, index=False)

    print()
    print("Regression summary:")
    display_columns = [
        "strategy",
        "model",
        "annualized_alpha",
        "alpha_t_stat",
        "r_squared",
        "market_beta",
        "residual_volatility",
    ]
    print(
        format_factor_summary(summary[display_columns])
        .to_string(index=False)
    )

    print()
    print("Saved outputs:")
    print(f"- {FACTOR_SUMMARY_PATH}")
    print(f"- {FACTOR_COEFFICIENTS_PATH}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()