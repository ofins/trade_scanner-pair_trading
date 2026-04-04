# Testing Guidelines

## Current State

No formal test suite exists. Validation is done through:
- Walk-forward splits to prevent lookahead bias
- `test.py` utility for sector data generation
- Manual runs of `single_pair.py` for spot-checking

## When Adding Tests

If adding tests, use `pytest`. Place test files in a `tests/` directory mirroring `src/`:
```
tests/
  test_scanner_utils.py
  test_backtest_utils.py
  test_common_utils.py
```

## What to Test

**Scanner utils** (`src/scanner/utils.py`):
- `calculate_half_life()`: Verify output is positive float for known mean-reverting series.
- `calculate_hurst_exponent()`: Known random walk should return ~0.5; known AR(1) should return < 0.5.
- `test_stationarity()`: ADF on a stationary series should return p-value < 0.05.

**Backtest utils** (`src/backtest/utils.py`):
- `backtest_pair()`: Use synthetic price series with known spread behavior; verify trade count and direction.
- Performance metrics: Sharpe, Sortino, max drawdown — verify against hand-calculated values.

**Do NOT mock yfinance** — use pre-downloaded CSV fixtures instead. Mocking the data layer has caused false-positive test results in the past.

## Manual Validation Checklist

Before committing changes to filter logic:
1. Run scanner on Technology sector (largest, most pairs).
2. Confirm number of passing pairs is in expected range (historically 0–15).
3. Spot-check 2–3 pairs manually with `analyze-pair` command.
4. Run `single_pair.py` on a known good pair (e.g., from a previous scan result).
