# /project:backtest

Run backtests on pairs identified by the scanner.

## Steps

1. Ask the user: batch backtest (all pairs from scanner output) or single pair?

### Batch Backtest
1. Find the most recent scanner output in `__reports__/` or ask the user to specify a file.
2. Read `src/backtest/index.py` to confirm configuration.
3. Run `python src/backtest/index.py` with the specified input file.
4. After completion, summarize:
   - Total pairs tested
   - % profitable pairs
   - Average Sharpe ratio
   - Best and worst performing pairs
   - Distribution of win rates

### Single Pair Backtest
1. Ask for the two ticker symbols (e.g., `JPM` and `WFC`).
2. Ask for the date range.
3. Run `python src/backtest/single_pair.py` and provide the tickers when prompted.
4. Report: total return, Sharpe ratio, max drawdown, win rate, number of trades.
5. Note any chart files generated.

## Notes

- Outputs are saved to `__reports__/` with timestamps.
- Single pair backtest generates a 3-panel chart (spread, z-score, equity curve).
- A Sharpe ratio > 1.0 is considered acceptable; > 1.5 is strong.
