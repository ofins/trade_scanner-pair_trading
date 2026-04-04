# Financial & Statistical Conventions

## Price Data

- Always use **adjusted close prices** for all calculations (accounts for dividends, splits).
- Use **log returns** for correlation calculations: `np.log(prices / prices.shift(1))`.
- Data source: `yfinance` daily OHLCV, minimum 2 years of history preferred.

## Spread Definition

```
spread = Y - (beta * X)
```
- `Y` is the dependent variable (stock being "priced")
- `X` is the independent variable (the hedge)
- `beta` is the OLS regression coefficient (hedge ratio)

## Cointegration Testing

- Use **Engle-Granger two-step test** (via `statsmodels.tsa.stattools.coint`).
- Test both directions (X→Y and Y→X); use whichever yields the lower p-value.
- p-value threshold: **< 0.02** (stricter than the standard 0.05 to reduce false positives).

## Half-Life Calculation

Uses AR(1) model on spread changes:
```
Δspread(t) = θ × spread(t-1) + ε
half_life = -log(2) / θ
```
Valid range: **5–30 days**. Outside this range the pair is either too noisy or too slow.

## Hurst Exponent

- Computed via log-variance vs log-lag regression over lags 2–20.
- **H < 0.5**: Mean-reverting (desirable)
- **H = 0.5**: Random walk
- **H > 0.5**: Trending (avoid)

## Z-Score

```
z = (spread - rolling_mean(60)) / rolling_std(60)
```
- Entry: |z| ≥ 2.0
- Exit (profit): z crosses 0
- Exit (stop loss): |z| ≥ 3.5

## Performance Metrics

| Metric | Formula / Note |
|--------|---------------|
| Sharpe | (mean daily return / std daily return) × √252 |
| Sortino | (mean daily return / downside std) × √252 |
| Calmar | CAGR / max drawdown |
| Max Drawdown | Peak-to-trough decline in equity curve |
| CAGR | (final equity / initial equity)^(1/years) - 1 |
| Win Rate | Profitable trades / total trades |
| Profit Factor | Gross profit / gross loss |

## Position Sizing

```python
hedge_ratio = abs(beta)
stock_Y_allocation = capital / (1 + hedge_ratio)
stock_X_allocation = capital - stock_Y_allocation
```

Shares are calculated from allocation / entry price. No leverage applied.

## Important Caveats

- Transaction costs are NOT included in backtest results — real-world performance will be lower.
- Hedge ratio is **static** (computed once from full in-sample data), not rolling.
- Sector constraint: only test pairs within the same GICS sector to ensure economic rationale.
