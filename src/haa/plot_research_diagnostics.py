"""
plot_research_diagnostics.py

Additional research diagnostic figures for the regime-aware sector rotation project.

This script generates figures used in the README mini-paper:
- walk-forward performance comparison;
- walk-forward selection counts;
- bootstrap confidence intervals for CAGR differences;
- bootstrap confidence intervals for Sharpe differences;
- ETF proxy factor alpha estimates;
- ETF proxy market beta estimates.

Run from the project root with:

    python src/haa/plot_research_diagnostics.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_DIR = Path("data")
FIGURES_DIR = Path("figures")

WALK_FORWARD_SUMMARY_PATH = DATA_DIR / "walk_forward_summary.csv"
WALK_FORWARD_SELECTION_PATH = DATA_DIR / "walk_forward_selection.csv"
BOOTSTRAP_SUMMARY_PATH = DATA_DIR / "bootstrap_significance_summary.csv"
FACTOR_SUMMARY_PATH = DATA_DIR / "factor_regression_summary.csv"


def save_figure(path: Path) -> None:
    """
    Save the current matplotlib figure.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def load_csv(path: Path) -> pd.DataFrame:
    """
    Load a required CSV file.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"File is empty: {path}")

    return df


def plot_walk_forward_cagr() -> None:
    """
    Plot CAGR comparison over the walk-forward evaluation period.
    """
    summary = pd.read_csv(WALK_FORWARD_SUMMARY_PATH, index_col=0)

    values = summary["cagr"].sort_values(ascending=True) * 100.0

    plt.figure(figsize=(10, 6))
    plt.barh(values.index, values.values)
    plt.title("Walk-forward evaluation: CAGR")
    plt.xlabel("CAGR (%)")
    plt.grid(True, axis="x", alpha=0.3)

    save_figure(FIGURES_DIR / "walk_forward_cagr_comparison.png")


def plot_walk_forward_selection_counts() -> None:
    """
    Plot how often each strategy is selected by the walk-forward rules.
    """
    selection = load_csv(WALK_FORWARD_SELECTION_PATH)

    counts = (
        selection
        .groupby(["selection_metric", "selected_strategy"])
        .size()
        .unstack(fill_value=0)
    )

    counts.T.plot(kind="barh", figsize=(10, 6))
    plt.title("Walk-forward selection counts")
    plt.xlabel("Number of selected test years")
    plt.ylabel("Selected strategy")
    plt.grid(True, axis="x", alpha=0.3)
    plt.legend(title="Selection metric")

    save_figure(FIGURES_DIR / "walk_forward_selection_counts.png")


def plot_bootstrap_cagr_intervals() -> None:
    """
    Plot bootstrap confidence intervals for CAGR differences.
    """
    summary = load_csv(BOOTSTRAP_SUMMARY_PATH)

    df = summary.loc[summary["metric"] == "diff_cagr"].copy()
    df = df.sort_values("observed_difference")

    observed = df["observed_difference"] * 100.0
    lower = df["ci_2_5"] * 100.0
    upper = df["ci_97_5"] * 100.0

    xerr = [
        observed - lower,
        upper - observed,
    ]

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        observed,
        df["benchmark"],
        xerr=xerr,
        fmt="o",
        capsize=4,
    )
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.title("Bootstrap confidence intervals: CAGR difference")
    plt.xlabel("Sector Top-2 SHY minus benchmark CAGR difference (%)")
    plt.grid(True, axis="x", alpha=0.3)

    save_figure(FIGURES_DIR / "bootstrap_cagr_confidence_intervals.png")


def plot_bootstrap_sharpe_intervals() -> None:
    """
    Plot bootstrap confidence intervals for Sharpe ratio differences.
    """
    summary = load_csv(BOOTSTRAP_SUMMARY_PATH)

    df = summary.loc[summary["metric"] == "diff_sharpe_ratio"].copy()
    df = df.sort_values("observed_difference")

    observed = df["observed_difference"]
    lower = df["ci_2_5"]
    upper = df["ci_97_5"]

    xerr = [
        observed - lower,
        upper - observed,
    ]

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        observed,
        df["benchmark"],
        xerr=xerr,
        fmt="o",
        capsize=4,
    )
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.title("Bootstrap confidence intervals: Sharpe difference")
    plt.xlabel("Sector Top-2 SHY minus benchmark Sharpe difference")
    plt.grid(True, axis="x", alpha=0.3)

    save_figure(FIGURES_DIR / "bootstrap_sharpe_confidence_intervals.png")


def plot_factor_alpha_for_target() -> None:
    """
    Plot annualized alpha estimates for Sector Top-2 SHY across proxy factor models.
    """
    summary = load_csv(FACTOR_SUMMARY_PATH)

    df = summary.loc[
        summary["strategy"] == "Sector Top-2 SHY Net"
    ].copy()

    df["annualized_alpha_pct"] = df["annualized_alpha"] * 100.0

    plt.figure(figsize=(9, 5))
    plt.bar(df["model"], df["annualized_alpha_pct"])
    plt.title("Sector Top-2 SHY: ETF proxy annualized alpha")
    plt.ylabel("Annualized alpha (%)")
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, axis="y", alpha=0.3)

    save_figure(FIGURES_DIR / "factor_alpha_sector_top2_shy.png")


def plot_factor_market_beta() -> None:
    """
    Plot market beta estimates under the sector proxy model.
    """
    summary = load_csv(FACTOR_SUMMARY_PATH)

    df = summary.loc[
        summary["model"] == "sector_proxy"
    ].copy()

    df = df.sort_values("market_beta", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(df["strategy"], df["market_beta"])
    plt.title("Market beta under ETF sector proxy model")
    plt.xlabel("Market beta to SPY")
    plt.grid(True, axis="x", alpha=0.3)

    save_figure(FIGURES_DIR / "factor_market_beta_sector_proxy.png")


def main() -> None:
    """
    Generate all research diagnostic figures.
    """
    print("=" * 80)
    print("Generating research diagnostic figures")
    print("=" * 80)

    plot_walk_forward_cagr()
    plot_walk_forward_selection_counts()
    plot_bootstrap_cagr_intervals()
    plot_bootstrap_sharpe_intervals()
    plot_factor_alpha_for_target()
    plot_factor_market_beta()

    print()
    print("Saved figures:")
    print(f"- {FIGURES_DIR / 'walk_forward_cagr_comparison.png'}")
    print(f"- {FIGURES_DIR / 'walk_forward_selection_counts.png'}")
    print(f"- {FIGURES_DIR / 'bootstrap_cagr_confidence_intervals.png'}")
    print(f"- {FIGURES_DIR / 'bootstrap_sharpe_confidence_intervals.png'}")
    print(f"- {FIGURES_DIR / 'factor_alpha_sector_top2_shy.png'}")
    print(f"- {FIGURES_DIR / 'factor_market_beta_sector_proxy.png'}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()