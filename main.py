"""
main.py
-------
End-to-end pipeline.  Runs the full analysis and saves all outputs.

Usage:
    python main.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_loader  import load_prices, REGIMES, TICKERS
from returns      import log_returns, portfolio_returns, cumulative_returns
from risk_metrics import risk_summary, annualised_volatility, drawdown_series
from correlation  import (
    correlation_matrix,
    regime_correlations,
    rolling_correlation,
    diversification_ratio,
    pairwise_regime_shift,
)
from visualization import (
    plot_cumulative_returns,
    plot_correlation_heatmap,
    plot_rolling_correlation,
    plot_drawdown,
    plot_volatility_comparison,
    plot_regime_heatmaps,
)

OUTPUT = "output"
os.makedirs(OUTPUT, exist_ok=True)


def save(fig: plt.Figure, name: str):
    path = os.path.join(OUTPUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {path}")


def main():
    print("=" * 60)
    print("  Multi-Asset Portfolio Risk & Correlation Analysis")
    print("=" * 60)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("\n[1] Loading price data …")
    prices  = load_prices(start="2018-01-01")
    log_ret = log_returns(prices)
    ew_weights = np.ones(len(prices.columns)) / len(prices.columns)
    port_ret   = portfolio_returns(log_ret, ew_weights)

    # ── 2. Risk Summary ───────────────────────────────────────────────────────
    print("\n[2] Computing risk metrics …")
    summary = risk_summary(log_ret, port_ret)
    print("\n" + summary.to_string())
    summary.to_csv(os.path.join(OUTPUT, "risk_summary.csv"))

    # ── 3. VaR Detail ─────────────────────────────────────────────────────────
    from risk_metrics import var_summary
    print("\n[3] VaR breakdown for equal-weight portfolio:")
    print(var_summary(port_ret).to_string())

    # ── 4. Diversification Ratio ──────────────────────────────────────────────
    dr = diversification_ratio(log_ret, ew_weights)
    print(f"\n[4] Diversification Ratio (equal-weight): {dr:.4f}")
    print("    (>1 = diversification is reducing portfolio risk)")

    # ── 5. Regime Correlation Shifts ──────────────────────────────────────────
    print("\n[5] SPY vs BTC — correlation shift across regimes:")
    shift = pairwise_regime_shift(log_ret, "SPY", "BTC-USD")
    print(shift.to_string())

    # ── 6. Visualizations ─────────────────────────────────────────────────────
    print("\n[6] Generating charts …")

    # 6a. Cumulative returns
    fig = plot_cumulative_returns(log_ret, port_ret)
    save(fig, "01_cumulative_returns.png")

    # 6b. Full-sample correlation heatmap
    corr = correlation_matrix(log_ret)
    fig  = plot_correlation_heatmap(corr, "Full-Sample Correlation Matrix (2018–present)")
    save(fig, "02_correlation_heatmap_full.png")

    # 6c. Regime heatmaps
    regime_corrs = regime_correlations(log_ret)
    fig = plot_regime_heatmaps(regime_corrs)
    save(fig, "03_correlation_by_regime.png")

    # 6d. Rolling SPY vs BTC correlation
    roll_corr = rolling_correlation(log_ret, "SPY", "BTC-USD", window=60)
    fig = plot_rolling_correlation(roll_corr)
    save(fig, "04_rolling_corr_spy_btc.png")

    # 6e. Drawdowns
    dd_dict = {col: drawdown_series(log_ret[col]) for col in log_ret.columns}
    dd_dict["Portfolio (EW)"] = drawdown_series(port_ret)
    fig = plot_drawdown(dd_dict)
    save(fig, "05_drawdowns.png")

    # 6f. Volatility by regime
    vol_by_regime = {}
    for regime, (start, end) in REGIMES.items():
        subset = log_ret.loc[start:end]
        if len(subset) > 20:
            vol_by_regime[regime] = annualised_volatility(subset)
    fig = plot_volatility_comparison(vol_by_regime)
    save(fig, "06_volatility_by_regime.png")

    # ── 7. Key Insights ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  KEY ANALYTICAL INSIGHTS")
    print("=" * 60)

    spy_btc_covid = regime_corrs.get("COVID Crash", pd.DataFrame())
    spy_btc_normal = regime_corrs.get("Pre-COVID (Normal)", pd.DataFrame())

    if not spy_btc_covid.empty and not spy_btc_normal.empty:
        corr_normal = spy_btc_normal.loc["SPY", "BTC-USD"]
        corr_covid  = spy_btc_covid.loc["SPY", "BTC-USD"]
        print(f"\n• SPY–BTC correlation rose from {corr_normal:.2f} (pre-COVID) "
              f"to {corr_covid:.2f} during the COVID crash.")
        print("  → Diversification benefits deteriorated as BTC behaved like a risk asset.")

    if "Inflation Shock" in regime_corrs and "Pre-COVID (Normal)" in regime_corrs:
        bond_corr_normal = spy_btc_normal.loc["SPY", "TLT"] if "TLT" in spy_btc_normal else float("nan")
        bond_corr_inf    = regime_corrs["Inflation Shock"].loc["SPY", "TLT"] if "TLT" in regime_corrs["Inflation Shock"] else float("nan")
        if not (np.isnan(bond_corr_normal) or np.isnan(bond_corr_inf)):
            print(f"\n• SPY–TLT (bond) correlation: {bond_corr_normal:.2f} (normal) → "
                  f"{bond_corr_inf:.2f} (2022 inflation shock).")
            print("  → The traditional 60/40 hedge broke down when both fell simultaneously.")

    print(f"\n• Portfolio diversification ratio: {dr:.3f}")
    print(f"  → Equal-weight portfolio reduces risk vs. simple asset average.")
    print(f"\n• Max drawdown (portfolio):  {summary.loc['Portfolio', 'Max Drawdown (%)']:.1f}%")
    print(f"  Max drawdown (BTC alone):  {summary.loc['BTC-USD',   'Max Drawdown (%)']:.1f}%")
    print(f"\n• Sharpe ratios:")
    for asset in summary.index:
        print(f"    {asset:<20}  {summary.loc[asset, 'Sharpe Ratio']:>6.3f}")

    print(f"\n✓ All outputs saved to /{OUTPUT}/")


if __name__ == "__main__":
    main()
