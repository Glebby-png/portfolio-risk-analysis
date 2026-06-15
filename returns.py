"""
returns.py
----------
Computes log returns and portfolio-level returns.

Formula reference
-----------------
Log return:   r_t = ln(P_t / P_{t-1})
Portfolio return (single period): R_p = w^T · r
"""

import numpy as np
import pandas as pd


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns from a price DataFrame.

    r_t = ln(P_t / P_{t-1})

    Log returns are preferred because they are:
      • Time-additive  (multi-period return = sum of log returns)
      • Symmetric      (no floor at -100%)
      • Better behaved statistically for volatility modeling
    """
    return np.log(prices / prices.shift(1)).dropna()


def portfolio_returns(
    log_rets: pd.DataFrame,
    weights: np.ndarray | list | None = None,
) -> pd.Series:
    """f
    Compute daily portfolio log returns using a fixed weight vector.

    Parameters
    ----------
    log_rets : DataFrame of individual asset log returns
    weights  : array-like, must sum to 1.0.  Defaults to equal weight.

    Returns
    -------
    pd.Series  — daily portfolio return series
    """
    n = log_rets.shape[1]
    if weights is None:
        weights = np.ones(n) / n          # equalizes weights if not given

    weights = np.asarray(weights, dtype=float)
    assert abs(weights.sum() - 1.0) < 1e-6, "Weights must sum to 1."

    port = (log_rets * weights).sum(axis=1)
    port.name = "Portfolio"
    return port


def cumulative_returns(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Convert a return series to a cumulative growth index (starts at 1.0)."""
    # it should not be this: 
    # return (1 + returns).cumprod(); this is for simple returns
    
    return np.exp(returns.cumsum())