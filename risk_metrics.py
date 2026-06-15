"""
risk_metrics.py
---------------
Core risk and performance statistics used throughout the analysis.

Formulas documented inline
"""

import numpy as np
import pandas as pd
from scipy import stats


TRADING_DAYS = 252          # annualisation constant
RISK_FREE_RATE = 0.04       # approximate current annual risk-free rate


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Volatility
# ─────────────────────────────────────────────────────────────────────────────

def annualised_volatility(returns: pd.Series | pd.DataFrame) -> pd.Series | float:
    """
    Annualised volatility = daily std-dev × √252

    σ_annual = σ_daily × root(T)

    Higher σ → greater uncertainty → higher risk.
    """
    return returns.std() * np.sqrt(TRADING_DAYS)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Sharpe Ratio
# ─────────────────────────────────────────────────────────────────────────────

def sharpe_ratio(
    returns: pd.Series,
    risk_free: float = RISK_FREE_RATE,
) -> float:
    """
    Sharpe Ratio = (E[R_p] - R_f) / σ_p   (annualised)

    Measures reward per unit of total risk.
    A Sharpe > 1.0 is generally considered good; > 2.0 is exceptional.
    """
    daily_rf = np.log(1 + risk_free) / TRADING_DAYS
    excess   = returns - daily_rf
    return (excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Value at Risk (VaR)
# ─────────────────────────────────────────────────────────────────────────────

def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical (non-parametric) VaR.

    VaR_α = -Quantile_{1-α}(R)

    Interpretation:  On the worst (1-α)% of days, the portfolio is
    expected to lose at least |VaR| of its value.

    Example: 95% VaR = -2.1%  →  only 5% of days saw worse than -2.1%.
    """
    return float(np.percentile(returns, (1 - confidence) * 100))


def parametric_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Parametric (Gaussian) VaR.

    VaR_α = μ - z_α × σ

    Assumes returns are normally distributed — a simplification, but
    useful for comparison and fast computation.
    """
    mu, sigma = returns.mean(), returns.std()
    z = stats.norm.ppf(1 - confidence)
    return float(mu + z * sigma)


def conditional_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Conditional VaR (CVaR / Expected Shortfall).

    CVaR_α = E[R | R < VaR_α]
S
    Average of all losses beyond the VaR threshold.
    More conservative and increasingly preferred by risk managers.
    """
    threshold = historical_var(returns, confidence)
    tail = returns[returns <= threshold]
    return float(tail.mean())


def var_summary(returns: pd.Series, confidence: float = 0.95) -> pd.Series:
    """Return all three VaR metrics as a Series."""
    return pd.Series({
        f"Historical VaR ({int(confidence*100)}%)":  historical_var(returns, confidence),
        f"Parametric VaR ({int(confidence*100)}%)":  parametric_var(returns, confidence),
        f"CVaR / ES    ({int(confidence*100)}%)":    conditional_var(returns, confidence),
    })


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Maximum Drawdown
# ─────────────────────────────────────────────────────────────────────────────

def drawdown_series(returns: pd.Series) -> pd.Series:
    """
    Drawdown at each point in time.

    DD_t = (P_t - max(P_{0..t})) / max(P_{0..t})

    A value of -0.30 means the portfolio is 30% below its prior peak.
    """
    cum   = (1 + returns).cumprod()
    peak  = cum.cummax()
    return (cum - peak) / peak


def max_drawdown(returns: pd.Series) -> float:
    """Maximum (worst) drawdown over the full period."""
    return float(drawdown_series(returns).min())


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Summary Table
# ─────────────────────────────────────────────────────────────────────────────

def risk_summary(log_rets: pd.DataFrame, port_rets: pd.Series) -> pd.DataFrame:
    """
    Produce a single DataFrame combining all key metrics for each asset
    plus the overall portfolio.
    """
    all_rets = pd.concat([log_rets, port_rets.rename("Portfolio")], axis=1)
    rows = []
    for col in all_rets.columns:
        r = all_rets[col]
        rows.append({
            "Asset":             col,
            "Ann. Return (%)":   round(r.mean() * TRADING_DAYS * 100, 2),
            "Ann. Volatility (%)": round(annualised_volatility(r) * 100, 2),
            "Sharpe Ratio":      round(sharpe_ratio(r), 3),
            "Max Drawdown (%)":  round(max_drawdown(r) * 100, 2),
            "Hist. VaR 95% (%)": round(historical_var(r) * 100, 2),
            "CVaR 95% (%)":      round(conditional_var(r) * 100, 2),
        })
    return pd.DataFrame(rows).set_index("Asset")
