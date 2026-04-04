# Backtest — Design & Conventions

## Overview

The backtest simulates a dollar-neutral pairs trading strategy on cointegrated pairs identified by the scanner.
All metrics are computed after transaction costs and use walk-forward beta estimation to prevent lookahead bias.

---

## Architecture

| File | Role |
|------|------|
| `index.py` | Batch runner — reads scanner Excel, runs `SinglePairBacktest` for each pair |
| `single_pair.py` | Per-pair orchestration — data fetch, walk-forward split, trade simulation, results |
| `utils.py` | `BacktestUtils` — simulation engine, daily return builder, all performance metrics |

---

## Walk-Forward Beta Estimation

The hedge ratio (beta) is estimated using OLS on the **first 50% of the data only** (training window).
It is then applied as a fixed constant to the full backtest period.

This matches the scanner's walk-forward split and eliminates the most impactful form of lookahead bias:
using a beta computed from future data to construct past spread/z-score signals.

```
Total data:  |<------- training (50%) ------->|<------- backtest (50%) ------->|
Beta fitted:  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Beta applied: ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

The spread is: `Y - beta * X - intercept`
The z-score is: rolling 60-day normalisation of the spread (backward-looking, no lookahead).

---

## Trade Logic

**Entry** (one position at a time):
- LONG spread when z-score ≤ −2.0 and `is_good_entry` filters pass
- SHORT spread when z-score ≥ +2.0 and `is_good_entry` filters pass

**Exit**:
- Mean reversion: z-score crosses 0
- Stop loss: z-score exceeds **fixed absolute** threshold of ±3.5 (LONG exits at z < −3.5, SHORT at z > +3.5)

The stop loss is **absolute** (not relative to entry z-score). This ensures pairs entering at more extreme z-scores receive the same protection as less extreme entries.

**Entry filters** (`is_good_entry`):
- Rolling half-life: 5–30 days
- Rolling Hurst exponent: < 0.5
- Rolling ADF p-value on spread: < 0.05

---

## Transaction Costs

Each trade includes transaction costs of **0.1% per side** (2 sides on entry + 2 sides on exit = 0.2% of capital per round-trip).

```python
transaction_cost = total_capital_deployed * 0.001 * 2
```

This approximates realistic brokerage commissions + bid-ask spread. Results without costs would be higher;
the gap represents the friction that erodes the strategy's edge in practice.

---

## Position Sizing

```
hedge_ratio = abs(beta)
stock1_allocation = capital / (1 + hedge_ratio)
stock2_allocation = capital - stock1_allocation

stock1_shares = stock1_allocation / entry_price1
stock2_shares = stock2_allocation / entry_price2
```

Capital is **fixed** (not compounded). Each trade is sized as if starting capital is always the initial amount.
CAGR and annualized return describe what a fixed-notional account would produce.

---

## Performance Metrics

### Daily Return Series

All risk-adjusted metrics (Sharpe, Sortino, Volatility) are computed on a **daily fractional return series**
built from the trade log and price data — not on a per-trade P&L series.

`build_daily_returns` assigns each calendar day within a trade's holding period its actual P&L from price
changes, divided by capital. Days outside any trade have 0 return. This gives proper daily resolution
for annualisation and accounts for intra-trade drawdown.

### Metrics Reference

| Metric | Formula | Notes |
|--------|---------|-------|
| Sharpe | `(mean(r - rf_daily) / std(r - rf_daily)) × √252` | r = daily returns; rf = 2%/252 |
| Sortino | `(mean(r - rf_daily) / semi_dev) × √252` | semi_dev = √mean(min(r−target, 0)²) |
| Volatility | `std(daily_returns) × √252` | annualised fractional vol |
| Max Drawdown % | `max_drawdown_$ / peak_equity × 100` | peak = highest equity reached, not initial capital |
| Calmar | `CAGR% / max_drawdown%` | |
| CAGR | `(final_equity / initial_capital)^(1/years) − 1` | final_equity = initial + total PnL |
| Win Rate | winning trades / total trades | |
| Profit Factor | gross profit / gross loss | |

### `Can Trade` Field

Reports `YES` only if:
1. Current z-score ≥ entry threshold, AND
2. All `is_good_entry` rolling filters pass at the last bar (half-life, Hurst, ADF)

A `YES` result means the pair is currently signalling a tradeable entry under all active filters.

---

## Known Limitations

- **Fixed capital**: No compounding. CAGR assumes reinvestment but trade sizing does not adjust.
- **No slippage model**: Transaction costs are flat %. Market impact is not modelled.
- **Static beta**: Beta is fixed from training window. Structural breaks after the split point are not detected.
- **Daily bars**: Intra-day slippage and gap risk are not captured.
- **No short-selling cost**: Borrow fees for the short leg are not modelled.
