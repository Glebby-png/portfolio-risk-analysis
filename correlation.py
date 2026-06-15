"""
correlation.py
--------------
Correlation analysis across full sample and across market regimes.
The "regime-aware" view is the analytical edge of this project.
"""

import numpy as np
import pandas as pd
from data_loader import REGIMES


def correlation_matrix(log_rets: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix of log returns."""
    return log_rets.corr()


def regime_correlations(log_rets: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Compute correlation matrices for each named market regime.

    Returns a dict: { regime_name → correlation DataFrame }
    """
    result = {}
    for name, (start, end) in REGIMES.items():
        subset = log_rets.loc[start:end]
        if len(subset) > 20:                   # need enough data
            result[name] = subset.corr()
    return result


def rolling_correlation(
    log_rets: pd.DataFrame,
    asset_a: str,
    asset_b: str,
    window: int = 60,
) -> pd.Series:
    """
    60-day rolling Pearson correlation between two assets.

    Reveals how the relationship between two assets evolves over time —
    especially useful for showing BTC's shift from uncorrelated to risk-on.
    """
    corr = log_rets[asset_a].rolling(window).corr(log_rets[asset_b])
    corr.name = f"{asset_a} vs {asset_b} ({window}d roll)"
    return corr


def diversification_ratio(
    log_rets: pd.DataFrame,
    weights: np.ndarray | None = None,
) -> float:
    """
    Diversification Ratio = (weighted sum of individual vols) / portfolio vol

    DR > 1 means diversification is reducing risk.
    DR close to 1 means assets are moving together (no diversification benefit).

    Higher DR → better diversification.
    """
    n = log_rets.shape[1]
    if weights is None:
        weights = np.ones(n) / n

    individual_vols = log_rets.std().values * np.sqrt(252)
    port_vol = np.sqrt(weights @ log_rets.cov().values * 252 @ weights)
    weighted_avg_vol = weights @ individual_vols

    return float(weighted_avg_vol / port_vol)


def pairwise_regime_shift(
    log_rets: pd.DataFrame,
    asset_a: str,
    asset_b: str,
) -> pd.DataFrame:
    """
    Show how the correlation between two assets changes across all regimes.
    Useful for detecting 'correlation contagion' during crises.
    """
    rows = []
    for regime, (start, end) in REGIMES.items():
        subset = log_rets.loc[start:end]
        if len(subset) > 20:
            corr = subset[asset_a].corr(subset[asset_b])
            rows.append({"Regime": regime, "Correlation": round(corr, 4)})
    return pd.DataFrame(rows).set_index("Regime")
