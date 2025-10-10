## Finding highly correlated pairs

### Compute correlation matrix

```python
data.corr()
```

computes pairwise correlation between columns using **Pearson correlation** by default.

minimum correlation coefficient is above 0.7

For more information on **Pearson Correlation Coefficient**, [Click here](https://www.scribbr.com/statistics/pearson-correlation-coefficient/)

Examples

```python
import pandas as pd

data = pd.DataFrame({
    "AAPL": [150, 152, 153, 155],
    "MSFT": [300, 305, 307, 310],
    "TSLA": [700, 680, 710, 720]
})

corr_matrix = data.corr()
print(corr_matrix)

# OUTPUT
        AAPL     MSFT      TSLA
AAPL   1.000  0.998     0.658
MSFT   0.998  1.000     0.642
TSLA   0.658  0.642     1.000
```

- Diagonal = correlation with itself (1.0)
- Off-diagonal = correlation between different stocks
- For example, AAPL and MSFT are very highly correlated (0.998)
- AAPL and TSLA are moderately correlated (0.658)

### Testing both directions

After finding correlated pairs of stocks, we test them in both directions to find main stock and stock for hedging.

### Using Engle-Granger two-step cointegration test

```python
coint(y0, y1)
```

This function from statsmodel performs the Engle-Granger two-step cointegration test where `y0` regress onto `y1`.

It will return `score`, `pvalue`, and `crit_values`.

Score - The ADF test statistic for the null hypothesis that there is no cointegration.

pvalue - The p-value corresponding to that statistic. Smaller means stronger evidence for cointegration.

```python
y0=α+βy1+ϵ
```

### Testing for stationarity

Augmented Dickey–Fuller (ADF) test to check if a time series is stationary

- If the series is stationary, it doesn’t drift — it fluctuates around a fixed mean.
- If it’s non-stationary, it tends to trend upward or downward (like most stock prices).

**Why It’s Used in Pairs Trading**

When you test two stocks for cointegration, you regress one on the other.

y=α+βx+ϵ

You then test whether the residual (ε) — the spread between the two — is stationary.

If the spread is stationary, it means the two stocks move together long-term and the spread mean-reverts — perfect for trading.
