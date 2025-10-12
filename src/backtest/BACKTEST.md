## Overview

### Performance and efficiency metrics overview

Cumulative return: The total percentage gain or loss over the backtesting period.

Compound Annual Growth Rate (CAGR): The geometric mean of the annual returns, which shows the smoothed, average growth rate per year.

Win rate: The percentage of trades that are profitable. This metric is most useful when analyzed alongside the average win/loss ratio, since a low win rate can still be profitable with sufficiently large winning trades.
Profit factor: The ratio of the gross profits to the gross losses. A value above 1 indicates a profitable strategy.

Average trade: The typical amount of profit or loss per trade. This can be analyzed in conjunction with the expectancy, which measures the average expected profit per dollar risked.

Equity curve: A visual plot of the cumulative profit and loss over time. A smoothly rising curve indicates a healthy and consistent strategy.

---

Maximum Drawdown (MDD): The largest peak-to-trough decline in your portfolio's value during the backtest. This represents the worst-case loss and a key test of psychological tolerance.

Annualized volatility: The standard deviation of the strategy's returns. It measures the fluctuation of returns and helps quantify the strategy's risk.

Stress and regime testing: A backtest should include a variety of market conditions, including bull and bear markets, to confirm the strategy is robust and adaptable.

Beta: Measures your strategy's sensitivity to overall market movements. A low beta suggests lower sensitivity to market fluctuations, which can be useful for creating diversified portfolios.

---

Sharpe ratio: The most widely used metric for comparing risk-adjusted returns. It measures the excess return (above the risk-free rate) per unit of total risk (standard deviation).
Interpretation: A higher Sharpe ratio is better. Ratios above 1 are generally considered good, above 2 are very good, and above 3 are excellent.

Sortino ratio: Similar to the Sharpe ratio, but it focuses only on downside volatility, or the standard deviation of negative returns. This metric is useful if you are more concerned with limiting losses than with overall volatility.

Calmar ratio: Compares the strategy's Compound Annual Growth Rate (CAGR) to its Maximum Drawdown (MDD), giving a clear picture of return relative to the worst-case loss.

### Sharpe ratio

Sharpe ratio by itself is not a reliable indicator as it only assumes Sharpe ratio up to this point, and does not reliable indicate if it would stay this way.

Instead, measure **rolling sharpe ratio**, which looks for pairs whose sharpe ratio stays positive and consistent and not necessarily the highest.

### calculate_reversion_quality

### check_liquidity

### calculate_correlation_stability

### calculate_volatility_regime

## Backtesting

### Initial tests

#### Consider live-metrics at each entry

Initial tests are done looking for specific metrics when scanning stocks then backtesting each pair 2-5y. However, this does not take into consideration of these _metrics_ during entry of each trade, where these metrics may fluctuate.

**Hypothesis**
Taking these metrics into consideration may result in better performance.

### Z-score

- test different z-score interval in same timeframe (e.g. 4h vs 1d in span of 2years)
