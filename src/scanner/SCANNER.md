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
