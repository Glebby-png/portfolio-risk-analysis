"""
data_loader.py
--------------
Pulls adjusted close prices for a multi-asset universe via yfinance,
aligns everything to a shared business-day calendar, and forward-fills
weekend/holiday gaps (most relevant for BTC-USD which trades 24/7).
"""

import yfinance as yf
import pandas as pd


# ── Default asset universe ────────────────────────────────────────────────────
TICKERS = {
    "SPY": "Equities (S&P 500)",
    "TLT": "Bonds (20+ yr Treasury)",
    "GLD": "Gold",
    "BTC-USD": "Bitcoin",
}

# ── Market regimes used throughout the project ────────────────────────────────
REGIMES = {
    "Pre-COVID (Normal)": ("2019-01-01", "2020-02-01"),
    "COVID Crash":        ("2020-02-01", "2020-12-31"),
    "Inflation Shock":    ("2022-01-01", "2022-12-31"),
    "Recovery / AI Bull": ("2023-01-01", "2024-12-31"),
}


def load_prices(
    tickers: list[str] = None,
    start: str = "2018-01-01",
    end: str = None,
) -> pd.DataFrame:
    """
    Download adjusted close prices and align to business-day calendar.

    Parameters
    ----------
    tickers : list of ticker strings (defaults to TICKERS keys)
    start   : ISO date string
    end     : ISO date string  (None → today)

    Returns
    -------
    pd.DataFrame  — columns = tickers, index = DatetimeIndex (business days)
    """
    if tickers is None:
        
        tickers = list(TICKERS.keys())

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]

    # Reindex to business days; forward-fill weekend BTC prices into Monday
    prices = (
        raw
        .reindex(pd.bdate_range(raw.index.min(), raw.index.max()))
        .ffill()
        .dropna()
    )

    print(f"Loaded {len(tickers)} assets | {prices.index[0].date()} → {prices.index[-1].date()} | {len(prices)} trading days")
    return prices


def slice_regime(prices: pd.DataFrame, regime: str) -> pd.DataFrame:
    """Return the price slice corresponding to a named regime."""
    start, end = REGIMES[regime]
    return prices.loc[start:end]
