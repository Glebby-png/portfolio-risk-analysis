"""
visualization.py
----------------
All plotting functions.  Each returns a matplotlib Figure so callers
can save or display as needed.  Designed for ~5 analyst-grade charts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


PALETTE    = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
REGIME_SHADING = {
    "COVID Crash":     ("#e74c3c", "2020-02-20", "2020-04-07"),
    "Inflation Shock": ("#e67e22", "2022-01-01", "2022-12-31"),
}

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#f8f9fa",
    "axes.grid":         True,
    "grid.color":        "white",
    "grid.linewidth":    1.0,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


def _shade_regimes(ax, shade=True):
    """Optionally shade key periods on a time-series axis."""
    if not shade:
        return
    for label, (color, start, end) in REGIME_SHADING.items():
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=0.12, color=color, label=label)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Cumulative Returns
# ─────────────────────────────────────────────────────────────────────────────

def plot_cumulative_returns(
    log_rets: pd.DataFrame,
    port_rets: pd.Series,
    shade: bool = True,
) -> plt.Figure:
    """Growth of $1 invested across each asset and the equal-weight portfolio."""
    cum = (1 + log_rets).cumprod()
    port_cum = (1 + port_rets).cumprod()

    fig, ax = plt.subplots(figsize=(12, 5))
    _shade_regimes(ax, shade)

    for i, col in enumerate(cum.columns):
        ax.plot(cum.index, cum[col], label=col, color=PALETTE[i], linewidth=1.6)
    ax.plot(port_cum.index, port_cum, label="Portfolio (EW)",
            color="black", linewidth=2.2, linestyle="--")

    ax.set_title("Cumulative Returns — Multi-Asset Portfolio", fontsize=14, fontweight="bold")
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Correlation Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(
    corr: pd.DataFrame,
    title: str = "Asset Return Correlation Matrix",
) -> plt.Figure:
    """Annotated Pearson correlation heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)   # upper triangle mask

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=-1, vmax=1,
        linewidths=0.5,
        ax=ax,
        mask=mask,
        annot_kws={"size": 11},
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Rolling Correlation
# ─────────────────────────────────────────────────────────────────────────────

def plot_rolling_correlation(
    rolling_corr: pd.Series,
    shade: bool = True,
) -> plt.Figure:
    """Rolling pair-wise correlation over time with regime shading."""
    fig, ax = plt.subplots(figsize=(12, 4))
    _shade_regimes(ax, shade)

    ax.plot(rolling_corr.index, rolling_corr, color="#4C72B0", linewidth=1.6)
    ax.axhline(0,  color="black",  linewidth=0.8, linestyle="--")
    ax.axhline(0.5, color="#e74c3c", linewidth=0.8, linestyle=":", alpha=0.7)

    ax.set_title(f"Rolling Correlation — {rolling_corr.name}", fontsize=13, fontweight="bold")
    ax.set_ylabel("Pearson Correlation")
    ax.set_ylim(-1, 1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Drawdown Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_drawdown(
    drawdowns_dict: dict[str, pd.Series],
    shade: bool = True,
) -> plt.Figure:
    """
    Drawdown chart for multiple series.

    drawdowns_dict: { label → drawdown Series }
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    _shade_regimes(ax, shade)

    colors = PALETTE + ["black", "purple"]
    for i, (label, dd) in enumerate(drawdowns_dict.items()):
        ax.fill_between(dd.index, dd * 100, 0,
                        alpha=0.25, color=colors[i % len(colors)])
        ax.plot(dd.index, dd * 100,
                label=label, color=colors[i % len(colors)], linewidth=1.4)

    ax.set_title("Maximum Drawdown — Peak-to-Trough Declines", fontsize=13, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Volatility Bar Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_volatility_comparison(
    vol_by_regime: dict[str, pd.Series],
) -> plt.Figure:
    """
    Grouped bar chart of annualised volatility per asset across regimes.
    Visually demonstrates how volatility spikes during crisis periods.
    """
    df = pd.DataFrame(vol_by_regime) * 100         # convert to %
    n_assets  = len(df)
    n_regimes = len(df.columns)
    x = np.arange(n_assets)
    width = 0.7 / n_regimes

    fig, ax = plt.subplots(figsize=(10, 5))
    regime_colors = ["#2ecc71", "#e74c3c", "#e67e22", "#3498db"]

    for i, regime in enumerate(df.columns):
        bars = ax.bar(
            x + i * width - (n_regimes - 1) * width / 2,
            df[regime],
            width=width * 0.9,
            label=regime,
            color=regime_colors[i % len(regime_colors)],
            alpha=0.85,
        )
        # value labels on top of bars
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(df.index, fontsize=11)
    ax.set_ylabel("Annualised Volatility (%)")
    ax.set_title("Volatility by Asset Across Market Regimes", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Regime Correlation Comparison (small multiples)
# ─────────────────────────────────────────────────────────────────────────────

def plot_regime_heatmaps(regime_corrs: dict[str, pd.DataFrame]) -> plt.Figure:
    """Side-by-side correlation heatmaps for each regime."""
    n = len(regime_corrs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, (regime, corr) in zip(axes, regime_corrs.items()):
        sns.heatmap(
            corr, annot=True, fmt=".2f",
            cmap="RdYlGn", vmin=-1, vmax=1,
            linewidths=0.5, ax=ax,
            annot_kws={"size": 9},
            cbar=(ax == axes[-1]),
        )
        ax.set_title(regime, fontsize=11, fontweight="bold")

    fig.suptitle("Correlation Matrices by Market Regime", fontsize=14,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig
