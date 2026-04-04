# Trade Scanner — Pair Trading Strategy

## Project Overview

Statistical arbitrage system that identifies cointegrated stock pairs within S&P 500 sectors and backtests mean-reversion trading strategies.

**Conclusion on file**: The strategy does not reliably beat the market after transaction costs. This repo is primarily for research and education.

## Architecture

```
src/
  scanner/       # Pair discovery: correlation → cointegration → statistical filters
  backtest/      # Simulation: batch (index.py) + single pair (single_pair.py)
  common/        # Shared utilities: yfinance data fetching, Excel I/O
  constants/     # S&P 500 sector mappings (JSON)
```

Reports output to `__reports__/` (gitignored).

## Key Entry Points

| Script | Purpose |
|--------|---------|
| `src/scanner/index.py` | Run full sector scan → outputs Excel with candidate pairs |
| `src/backtest/index.py` | Batch backtest all pairs from scanner output |
| `src/backtest/single_pair.py` | Interactive single-pair backtest with charts |

## Domain Rules

- **Walk-forward split**: Always use first 50% of data for pair discovery; never leak test data into training.
- **Hedge ratio**: Computed via OLS regression (Y ~ X); use static (not rolling) for stability.
- **Z-score window**: 60-day rolling mean/std for spread normalization.
- **Entry threshold**: ±2.0 z-score; exit at 0; stop loss at ±3.5.
- **Filter pipeline order**: correlation → cointegration → ADF stationarity → half-life → Hurst → zero-crossings → mean reversion rate → current z-score bounds.

## Filter Thresholds (current config)

| Filter | Threshold |
|--------|-----------|
| Correlation | ≥ 0.7 |
| Cointegration p-value | < 0.02 |
| ADF p-value on spread | < 0.05 |
| Half-life | 5–30 days |
| Hurst exponent | < 0.5 |
| Zero-crossings (2yr) | ≥ 7 |
| Mean reversion success rate | ≥ 60% |
| Current z-score | within [−3.5, +3.5] |

## Dependencies

Install via: `pip install -r requirements.txt`

Core: `yfinance`, `pandas`, `numpy`, `scipy`, `statsmodels`, `matplotlib`, `openpyxl`

## What NOT to Do

- Do not use rolling hedge ratios — tested and found to be unstable.
- Do not test pairs across different sectors — cointegration assumptions break down.
- Do not skip the walk-forward split — lookahead bias inflates results significantly.
- Do not remove the `gc.collect()` calls after each sector — memory usage spikes without them.
