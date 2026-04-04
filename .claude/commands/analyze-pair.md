# /project:analyze-pair

Deep-dive statistical analysis of a specific stock pair.

## Steps

1. Ask the user for two ticker symbols and a date range (default: last 2 years).
2. Fetch data using `CommonUtils.fetch_data()`.
3. Compute and report:
   - **Correlation**: Pearson correlation of log-returns
   - **Cointegration**: Engle-Granger test, p-value, and interpretation
   - **Hedge ratio**: OLS beta (Y ~ X), include R² and residual plot summary
   - **Spread stats**: mean, std, current value, current z-score
   - **Half-life**: Days to 50% mean reversion (AR(1) model)
   - **Hurst exponent**: Mean reversion strength (target < 0.5)
   - **ADF test on spread**: p-value and conclusion
   - **Zero-crossings**: Count over the full period
4. Verdict: Does this pair pass all scanner filters? If not, which filters fail?
5. If the pair passes filters, estimate a current trade signal (long spread / short spread / no signal).

## Output Format

```
=== Pair Analysis: TICK1 / TICK2 ===
Period: YYYY-MM-DD to YYYY-MM-DD

FILTER RESULTS:
  Correlation:     0.XX  [PASS/FAIL]
  Cointegration:   p=0.XX  [PASS/FAIL]
  ADF on spread:   p=0.XX  [PASS/FAIL]
  Half-life:       XX days  [PASS/FAIL]
  Hurst exponent:  0.XX  [PASS/FAIL]
  Zero-crossings:  XX  [PASS/FAIL]

CURRENT SIGNAL: [LONG SPREAD / SHORT SPREAD / NO SIGNAL]
Current z-score: X.XX
```
