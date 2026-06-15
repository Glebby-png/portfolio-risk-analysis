# portfolio-risk-analysis
# Multi-Asset Portfolio Risk & Correlation Dynamics Across Market Regimes

A quantitative risk study analyzing how diversification behaves across major asset classes how that effectiveness changes during different market regimes.

---

## Research Question

> *How effective is multi-asset diversification, and how does correlation structure break down during market stress?*

---

## Asset

| Ticker   | Asset Class | Role |
|----------|-------------|------|
| SPY      | Equities    | S&P 500 — growth / risk-on |
| TLT      | Bonds       | 20+ yr Treasuries — defensive hedge |
| GLD      | Gold        | Inflation / fear hedge |
| BTC-USD  | Crypto      | Speculative / emerging macro asset |

---

## Market Regimes Analyzed

| Regime | Period | Characteristics |
|--------|--------|-----------------|
| Pre-COVID | 2019 | Calm trending bull market |
| COVID Crash | Feb–Dec 2020 | Acute liquidity crisis, cross-asset selloff |
| Inflation Shock | 2022 | Rate hike cycle; bonds & stocks fell together |
| Recovery / AI Bull | 2023–2024 | Risk-on recovery, equity outperformance |

---

## Methodology

### Returns
Log returns used throughout:

$$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

### Volatility (annualised)
$$\sigma_{\text{annual}} = \sigma_{\text{daily}} \times \sqrt{252}$$

### Sharpe Ratio
$$\text{Sharpe} = \frac{E[R_p - R_f]}{\sigma_p} \times \sqrt{252}$$

### Value at Risk (Historical)
$$
\text{VaR}_{α} = -Q_{1-α}(R)
$$

### Maximum Drawdown
$$\text{DD}_t = \frac{P_t - \max(P_{0\ldots t})}{\max(P_{0\ldots t})}$$

### Weekend / Holiday Alignment
BTC trades 24/7; equities do not. Solution: reindex all assets to a shared business-day calendar and forward-fill weekend BTC prices into Monday.

```python
prices = prices.asfreq("B").ffill()
```

---

## Key Findings

1. **Correlation contagion:** Cross-asset correlations increased sharply during both the COVID crash and 2022 inflation shock, reducing diversification benefits precisely when they were most needed.

2. **Bond-equity hedge failure (2022):** The classic SPY–TLT negative correlation turned positive during the inflation shock, invalidating the 60/40 hedging thesis when both assets fell together.

3. **BTC's evolving macro character:** Rolling correlation shows BTC shifted from largely uncorrelated (2018–2019) to a high-beta risk asset post-2021, reducing its diversification value.

4. **Gold as resilient hedge:** GLD maintained lower equity correlation across both crisis periods, providing genuine tail-risk protection when bonds failed.

5. **Equal-weight diversification reduces drawdowns** materially compared to single-asset exposure.

---

## Project Structure

```
portfolio-risk-analysis/
├── data_loader.py
├── returns.py
├── risk_metrics.py
├── correlation.py
├── visualization.py
├── main.py
├── analysis.ipynb
├── risk_summary.csv
├── README.md
└── requirements.txt
```

---

## Visualizations

| Chart | Description |
|-------|-------------|
| Cumulative Returns | Growth of $1 invested, with crisis period shading |
| Correlation Heatmap | Full-sample Pearson correlation matrix |
| Regime Heatmaps | Side-by-side correlations across all market regimes |
| Rolling Correlation | 60-day SPY vs BTC — reveals regime shift over time |
| Drawdown Chart | Peak-to-trough declines for all assets + portfolio |
| Volatility Comparison | Annualised vol by asset across regimes |

---

## Quick Start

## Installation

```bash
git clone https://github.com/Glebby-png/portfolio-risk-analysis.git
cd portfolio-risk-analysis

pip install -r requirements.txt
```

*Data sourced from Yahoo Finance via `yfinance`.*
