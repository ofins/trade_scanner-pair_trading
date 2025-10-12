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

**Observation**
Adding more restriction actually results in poorer results.
This is likely due to filters being _too restrcitive_ which is filtering out many good opportunities. This makes trade pool too small and leads to bigger variance.

#### Control

**SAMPLE A (SP500, 40 pairs)**

**SAMPLE B (Random, 13 pairs)**
Avg/trade = 69; avg mdd = 166; avg mdd% = 6.64; winrate = 74%

#### Only adjust half-life metrics

**SAMPLE A (SP500)**

10 <= hl <= 90 : Avg/trade = 65; avg mdd = 110; avg mdd% = 4.4; winrate = 75%

10 <= hl <= 30 : Avg/trade = 64; avg mdd = 97; avg mdd% = 3.9; winrate = 75%

2 <= hl <= 20 : Avg/trade = 86; avg mdd = 101; avg mdd% = 4.1; winrate = 76%

2 <= hl <= 10 : Avg/trade = 82; avg mdd = 61; avg mdd% = 2.4; winrate = 77%

2 <= hl <= 5 : Avg/trade = 90; avg mdd = 18; avg mdd% = 0.7; winrate = 70%

0 <= hl <= 3 : too restrictive for most trades.

0 <= hl <= 2 : unable to take trades.

**SAMPLE B (Random, 13 pairs)**

10 <= hl <= 30 : Avg/trade = 74; avg mdd = 73; avg mdd% = 2.93; winrate = 80%

2 <= hl <= 20 : Avg/trade = 72; avg mdd = 163; avg mdd% = 6.55; winrate = 75%

2 <= hl <= 5 : Avg/trade = 68; avg mdd = 1; avg mdd% = 0.3; winrate = 64%

#### Only adjust Hurst Exponential metrics

**SAMPLE A (SP500)**

hurst < 50: Avg/trade = 71; avg mdd = 145; avg mdd% = 5.83; winrate = 75%

hurst < 30: Avg/trade = 77; avg mdd = 65; avg mdd% = 2.61; winrate = 77%

hurst < 20: Avg/trade = 75; avg mdd = 33; avg mdd% = 1.34; winrate = 71%

#### Only adjust ADP P-value

**SAMPLE A (SP500)**

ADP < 0.2: Avg/trade = 50; avg mdd = 0; avg mdd% = 3.63; winrate = 70%

ADP < 0.1: Avg/trade = 48; avg mdd = 70; avg mdd% = 2.8; winrate = 68%

#### Halflife + Hurst

**SAMPLE A (SP500)**

half-life (2-5), Hurst < 0.5
Avg/trade = 86; avg mdd = 13; avg mdd% = 0.55; winrate = 70%

half-life (2-10), Hurst < 0.5
Avg/trade = 81; avg mdd = 49; avg mdd% = 1.98; winrate = 77%

half-life (2-20), Hurst < 0.5
Avg/trade = 86; avg mdd = 101; avg mdd% = 4.06; winrate = 76%

half-life (2-20), Hurst < 0.3
Avg/trade = 79; avg mdd = 45; avg mdd% = 1.82; winrate = 78%

**SAMPLE B (Random, 13 pairs)**

half-life (10-30), Hurst < 0.3
Avg/trade = 54; avg mdd = 45; avg mdd% = 1.79; winrate = 80%

half-life (10-30), Hurst < 053
Avg/trade = 73; avg mdd = 75; avg mdd% = 3.03; winrate = 79%

### Z-score

- test different z-score interval in same timeframe (e.g. 4h vs 1d in span of 2years)

### Dynamic position sizing

Use Relaxed Filters as Position Sizing

Instead of rejecting trades, adjust position size:

def calculate*position_size(base_capital, metrics):
score = 1.0
if metrics['hurst'] < 0.45: score *= 1.2 # Boost strong mean reversion
if metrics['half_life'] > 60: score \_= 0.8 # Reduce slow pairs
return base_capital \* score
