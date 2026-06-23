"""
plots.py

Plotting utilities for the regime-aware sector rotation project.

This module generates:
- cumulative performance plots;
- drawdown plots;
- portfolio allocation plots;
- target asset distribution plots.

Run from the project root with:

    python src/common/plots.py
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


# Make imports robust when running the file directly from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from common.metrics import compute_equity_curve, compute_drawdown


FIGURES_DIR = Path("figures")

RETURNS_PATH = Path("data/haa_replication_returns.csv")
WEIGHTS_PATH = Path("data/haa_replication_weights.csv")


def save_figure(path: Path) -> None:
    """
    Save the current matplotlib figure.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_cumulative_performance(
    returns: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot cumulative performance for several return series.
    """
    equity_curves = pd.DataFrame(
        {
            column: compute_equity_curve(returns[column])
            for column in returns.columns
        }
    )

    plt.figure(figsize=(10, 6))

    for column in equity_curves.columns:
        plt.plot(equity_curves.index, equity_curves[column], label=column)

    plt.title("Cumulative Performance")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_figure(output_path)


def plot_drawdowns(
    returns: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot drawdowns for several return series.
    """
    equity_curves = pd.DataFrame(
        {
            column: compute_equity_curve(returns[column])
            for column in returns.columns
        }
    )

    drawdowns = pd.DataFrame(
        {
            column: compute_drawdown(equity_curves[column])
            for column in equity_curves.columns
        }
    )

    plt.figure(figsize=(10, 6))

    for column in drawdowns.columns:
        plt.plot(drawdowns.index, drawdowns[column], label=column)

    plt.title("Drawdown Comparison")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_figure(output_path)


def plot_weights(
    weights: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot portfolio weights over time.
    """
    plt.figure(figsize=(10, 6))

    plt.stackplot(
        weights.index,
        [weights[column] for column in weights.columns],
        labels=weights.columns,
    )

    plt.title("Portfolio Allocation Over Time")
    plt.xlabel("Date")
    plt.ylabel("Portfolio weight")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)

    save_figure(output_path)


def plot_asset_distribution(
    weights: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot the realized target asset distribution.
    """
    selected_asset = weights.idxmax(axis=1)
    distribution = selected_asset.value_counts(normalize=True).sort_values(ascending=False)

    plt.figure(figsize=(8, 5))
    plt.bar(distribution.index, distribution.values)

    plt.title("Realized Target Asset Distribution")
    plt.xlabel("Asset")
    plt.ylabel("Frequency")
    plt.grid(True, axis="y", alpha=0.3)

    save_figure(output_path)


def main() -> None:
    """
    Generate replication figures.
    """
    print("=" * 80)
    print("Generating replication figures")
    print("=" * 80)

    returns = pd.read_csv(RETURNS_PATH, index_col=0, parse_dates=True)
    weights = pd.read_csv(WEIGHTS_PATH, index_col=0, parse_dates=True)

    print()
    print("Returns loaded:")
    print(returns.tail())

    print()
    print("Weights loaded:")
    print(weights.tail())

    plot_cumulative_performance(
        returns=returns,
        output_path=FIGURES_DIR / "haa_replication_cumulative_performance.png",
    )

    plot_drawdowns(
        returns=returns,
        output_path=FIGURES_DIR / "haa_replication_drawdowns.png",
    )

    plot_weights(
        weights=weights,
        output_path=FIGURES_DIR / "haa_replication_weights.png",
    )

    plot_asset_distribution(
        weights=weights,
        output_path=FIGURES_DIR / "haa_replication_asset_distribution.png",
    )

    print()
    print("Saved figures:")
    print(f"- {FIGURES_DIR / 'haa_replication_cumulative_performance.png'}")
    print(f"- {FIGURES_DIR / 'haa_replication_drawdowns.png'}")
    print(f"- {FIGURES_DIR / 'haa_replication_weights.png'}")
    print(f"- {FIGURES_DIR / 'haa_replication_asset_distribution.png'}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()