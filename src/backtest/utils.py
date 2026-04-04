
import numpy as np
import pandas as pd


class BacktestUtils:
    @staticmethod
    def backtest_pair(df: pd.DataFrame, stock1:str, stock2:str, zscore_window:int, entry_threshold:float, capital: float, transaction_cost_pct: float = 0.001, stop_loss_zscore: float = 3.5)-> pd.DataFrame:
        """ Backtest a single pair trading strategy on two stocks with given parameters """
        trades = []
        position: dict = None 

        for i in range(zscore_window, len(df)):
            current_date = df.index[i]
            zscore = df['ZScore'].iloc[i]

            # if zscore is NaN, skip.
            if pd.isna(zscore):
                continue

            # Entry logic
            if position is None:
                if zscore <= -entry_threshold and BacktestUtils.is_good_entry(df, i, debug=False):
                    # LONG spread entry
                    # Spread = stock2 - beta * stock1 is below mean
                    # Stock 2 is undervalued relative to Stock 1
                    # Action: BUY stock2, SELL stock1
                    position = {
                        'type': 'LONG',
                        'entry_date': current_date,
                        'entry_zscore': zscore,
                        'stock1_price': df[stock1].iloc[i],
                        'stock2_price': df[stock2].iloc[i],
                        'hedge_ratio': df['Hedge_Ratio'].iloc[i]
                    }
                elif zscore >= entry_threshold and BacktestUtils.is_good_entry(df, i, debug=False): 
                    # SHORT spread entry
                    # Spread = stock2 - beta * stock1 is above mean
                    # Stock 2 is overvalued relative to Stock 1
                    # Action: SELL stock2, BUY stock1
                    position = {
                        'type': 'SHORT',
                        'entry_date': current_date,
                        'entry_zscore': zscore,
                        'stock1_price': df[stock1].iloc[i],
                        'stock2_price': df[stock2].iloc[i],
                        'hedge_ratio': df['Hedge_Ratio'].iloc[i]
                    }
            elif position is not None:
                exit_signal = False
                exit_reason = ''

                # Exit at mean (z-score crosses zero)
                if position['type'] == 'LONG' and zscore >= 0:
                    exit_signal = True
                    exit_reason = 'Mean Reversion'
                elif position['type'] == 'SHORT' and zscore <= 0:
                    exit_signal = True
                    exit_reason = 'Mean Reversion'

                # Stop loss: fixed absolute z-score level, same regardless of entry point
                elif position['type'] == 'LONG' and zscore < -stop_loss_zscore:
                    exit_signal = True
                    exit_reason = 'Stop Loss'
                elif position['type'] == 'SHORT' and zscore > stop_loss_zscore:
                    exit_signal = True
                    exit_reason = 'Stop Loss'

                if exit_signal:
                    # Calculate PnL
                    exit_stock1_price = df[stock1].iloc[i]
                    exit_stock2_price = df[stock2].iloc[i]

                    # Calculate position sizes
                    hedge_ratio = abs(position['hedge_ratio'])
                    stock1_allocation = capital / (1 + hedge_ratio)
                    stock2_allocation = capital - stock1_allocation

                    stock1_shares = stock1_allocation / position['stock1_price']
                    stock2_shares = stock2_allocation / position['stock2_price']

                    if position['type'] == 'LONG':
                        # LONG spread: BUY stock2, SELL stock1
                        stock1_pnl = stock1_shares * (position['stock1_price'] - exit_stock1_price)
                        stock2_pnl = stock2_shares * (exit_stock2_price - position['stock2_price'])
                    else:
                        # SHORT spread: SELL stock2, BUY stock1
                        stock1_pnl = stock1_shares * (exit_stock1_price - position['stock1_price'])
                        stock2_pnl = stock2_shares * (position['stock2_price'] - exit_stock2_price)

                    transaction_cost = (stock1_allocation + stock2_allocation) * transaction_cost_pct * 2
                    total_pnl = stock1_pnl + stock2_pnl - transaction_cost
                    pnl_percent = (total_pnl / capital) * 100

                    trades.append({
                        'Entry Date': position['entry_date'],
                        'Exit Date': current_date,
                        'Position': position['type'],
                        'Days held': (current_date - position['entry_date']).days,
                        'Win': total_pnl > 0,
                        'Entry ZScore': position['entry_zscore'],
                        'Exit ZScore': zscore,
                        'Entry Stock1 Price': position['stock1_price'],
                        'Exit Stock1 Price': exit_stock1_price,
                        'Entry Stock2 Price': position['stock2_price'],
                        'Exit Stock2 Price': exit_stock2_price,
                        'Stock1_Shares': stock1_shares,
                        'Stock2_Shares': stock2_shares,
                        'Stock1_Capital': stock1_allocation,
                        'Stock2_Capital': stock2_allocation,
                        'Total_Capital': stock1_allocation + stock2_allocation,
                        'Hedge Ratio': hedge_ratio,
                        'Stock1 Shares': stock1_shares,
                        'Stock2 Shares': stock2_shares,
                        'Transaction_Cost': transaction_cost,
                        'PnL ($)': total_pnl,
                        'PnL (%)': pnl_percent,
                        'Exit Reason': exit_reason
                    })

                    position = None  # Reset position
        
        return pd.DataFrame(trades)

    @staticmethod
    def build_daily_returns(df: pd.DataFrame, trades_df: pd.DataFrame, stock1: str, stock2: str, capital: float) -> pd.Series:
        """
        Build a daily fractional return series from completed trades and price data.

        For each day strictly after a trade's entry date up to and including exit date,
        computes the daily P&L from price moves using the trade's share counts, then
        divides by capital to produce a fractional return. Overlapping trades are summed.

        LONG  (buy stock2, sell stock1): pnl = shares2 * Δprice2 - shares1 * Δprice1
        SHORT (sell stock2, buy stock1): pnl = shares1 * Δprice1 - shares2 * Δprice2
        """
        if trades_df.empty:
            return pd.Series(0.0, index=df.index)

        price1_diff = df[stock1].diff()
        price2_diff = df[stock2].diff()
        daily_pnl = pd.Series(0.0, index=df.index)

        for _, trade in trades_df.iterrows():
            mask = (df.index > trade['Entry Date']) & (df.index <= trade['Exit Date'])
            trade_dates = df.index[mask]
            if len(trade_dates) == 0:
                continue
            shares1 = trade['Stock1_Shares']
            shares2 = trade['Stock2_Shares']
            if trade['Position'] == 'LONG':
                pnl = shares2 * price2_diff.loc[trade_dates] - shares1 * price1_diff.loc[trade_dates]
            else:
                pnl = shares1 * price1_diff.loc[trade_dates] - shares2 * price2_diff.loc[trade_dates]
            daily_pnl.loc[trade_dates] += pnl

        return daily_pnl / capital

    """ Filters """

    @staticmethod
    def is_good_entry(df: pd.DataFrame, index: int, debug: bool = False) -> bool:
        """
        Entry filter logic that considers current half-life, hurst, ADF p-value.

        Filters are applied progressively - each metric must pass to proceed.
        Adjust thresholds based on your backtesting results.

        Args:
            df: DataFrame with metrics
            index: Current index to check
            debug: If True, print why entries are rejected
        """
        half_life = df['Half_Life'].iloc[index]
        hurst = df['Hurst'].iloc[index]
        adf_p_value = df['ADF_PValue'].iloc[index]

        # Check for NaN values - if any critical metric is NaN, reject entry
        if pd.isna(half_life) or pd.isna(hurst) or pd.isna(adf_p_value):
            if debug:
                print(f"[{df.index[index]}] Rejected: NaN values (HL={half_life:.2f}, H={hurst:.3f}, ADF={adf_p_value:.3f})")
            return False

        # Filter 1: Half-life (should be reasonable for mean reversion)
        # Aligned with scanner filters: 5-30 days
        if not (5 <= half_life <= 30):
            if debug:
                print(f"[{df.index[index]}] Rejected: Half-life out of range: {half_life:.2f} days (need 5-30)")
            return False

        # Filter 2: Hurst exponent (< 0.5 indicates mean reversion)
        # < 0.4: strong mean reversion
        # 0.4-0.5: moderate mean reversion
        # > 0.5: trending (avoid)
        if hurst >= 0.5:
            if debug:
                print(f"[{df.index[index]}] Rejected: Hurst too high (trending): {hurst:.3f}")
            return False

        # Filter 3: ADF p-value (< 0.05 is statistically significant stationarity)
        # Aligned with scanner: require statistically significant stationarity
        if adf_p_value >= 0.05:
            if debug:
                print(f"[{df.index[index]}] Rejected: ADF p-value too high: {adf_p_value:.3f}")
            return False

        if debug:
            print(f"[{df.index[index]}] ✓ ACCEPTED: HL={half_life:.2f}, H={hurst:.3f}, ADF={adf_p_value:.3f}")
        return True

    """ Performance metrics """
    @staticmethod
    def calculate_average_metrics(results: list[dict]) -> dict:
        """ Calculate average metrics across multiple backtest results """
        import math

        if not results:
            return {}

        avg_metrics = {}
        excluded = {'Ticker1', 'Ticker2'}

        keys = [key for key in results[0].keys() if key not in excluded]

        # Do not count zeros, None, or inf in averages
        for key in keys:
            if isinstance(results[0][key], (int, float)):
                # Filter out zeros, None, inf, and -inf
                valid_values = [
                    result[key] for result in results
                    if result[key] != 0
                    and result[key] is not None
                    and not math.isinf(result[key])
                ]

                if len(valid_values) > 0:
                    avg_metrics[key] = sum(valid_values) / len(valid_values)
                else:
                    avg_metrics[key] = 0  # Default to 0 if no valid values
            else:
                avg_metrics[key] = results[0][key]

        return avg_metrics

    """ Compute various performance metrics from trades DataFrame """
    @staticmethod
    def calculate_max_drawdown(pnl_series: pd.Series, initial_capital: float) -> tuple[float, float]:
        """
        Calculate the maximum drawdown from a series of P&L values.

        Args:
            pnl_series: Series of P&L values from individual trades
            initial_capital: Starting capital for the strategy

        Returns:
            Tuple of (max_drawdown_dollars, max_drawdown_percentage relative to initial capital)
        """
        if pnl_series.empty:
            return 0.0, 0.0

        # Calculate equity curve starting from initial capital
        equity_curve = initial_capital + pnl_series.cumsum()

        # Calculate running maximum (peak equity)
        running_max = equity_curve.expanding().max()

        # Calculate drawdown at each point (peak - current equity)
        drawdown = running_max - equity_curve

        # Maximum drawdown in dollars
        max_drawdown_dollars = drawdown.max() if not pd.isna(drawdown.max()) else 0.0

        # Calculate percentage drawdown relative to PEAK equity
        peak_equity = running_max.max()
        max_drawdown_pct = (max_drawdown_dollars / peak_equity) * 100 if peak_equity > 0 else 0.0

        return max_drawdown_dollars, max_drawdown_pct

    @staticmethod
    def calculate_annualized_return(total_pnl: float, initial_capital: float, date_index: pd.DatetimeIndex) -> float:
        """ Calculate annualized return given total PnL, initial capital, and date index """
        if initial_capital <= 0 or date_index.empty:
            return 0.0
        
        total_days = (date_index[-1] - date_index[0]).days
        if total_days <= 0:
            return 0.0
        
        years = total_days / 365.25
        total_return = total_pnl / initial_capital
        annualized_return = (1 + total_return) ** (1 / years) - 1
        return annualized_return * 100  # Return as percentage
    
    @staticmethod
    def calculate_cagr(total_pnl: float, initial_capital: float, date_index: pd.DatetimeIndex) -> float:
        """
        Calculate Compound Annual Growth Rate (CAGR)
        
        Args:
            total_pnl: Total profit/loss
            initial_capital: Starting capital
            date_index: DatetimeIndex for calculating time period
        
        Returns:
            CAGR as percentage
        """
        if initial_capital <= 0 or date_index.empty:
            return 0.0
        
        total_days = (date_index[-1] - date_index[0]).days
        if total_days <= 0:
            return 0.0
        
        years = total_days / 365.25
        final_value = initial_capital + total_pnl
        
        if final_value <= 0 or years <= 0:
            return 0.0
        
        cagr = ((final_value / initial_capital) ** (1 / years)) - 1
        return cagr * 100  # Return as percentage
    
    @staticmethod
    def calculate_sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calculate annualized Sharpe ratio from a daily fractional return series.

        Args:
            daily_returns: Series of daily fractional returns from build_daily_returns
            risk_free_rate: Annual risk-free rate (default 2%)

        Returns:
            Annualized Sharpe ratio
        """
        if daily_returns.empty or daily_returns.std() == 0:
            return 0.0

        daily_rf = risk_free_rate / 252
        excess = daily_returns - daily_rf

        return (excess.mean() / excess.std()) * np.sqrt(252) if excess.std() > 0 else 0.0
    
    @staticmethod
    def calculate_volatility(daily_returns: pd.Series, annualized: bool = True) -> float:
        """
        Calculate volatility from a daily fractional return series.

        Args:
            daily_returns: Series of daily fractional returns from build_daily_returns
            annualized: Whether to annualize (default True, multiplies by sqrt(252))

        Returns:
            Volatility as a fractional value (e.g. 0.15 = 15% annualised)
        """
        if daily_returns.empty:
            return 0.0

        volatility = daily_returns.std()
        if annualized:
            volatility *= np.sqrt(252)

        return volatility
    
    @staticmethod
    def calculate_max_consecutive_losses(trades_df: pd.DataFrame) -> int:
        """
        Calculate maximum consecutive losing trades
        
        Args:
            trades_df: DataFrame with 'Win' column (boolean)
        
        Returns:
            Maximum number of consecutive losses
        """
        if trades_df.empty or 'Win' not in trades_df.columns:
            return 0
        
        max_losses = 0
        current_losses = 0
        
        for win in trades_df['Win']:
            if not win:
                current_losses += 1
                max_losses = max(max_losses, current_losses)
            else:
                current_losses = 0
        
        return max_losses
    
    @staticmethod
    def calculate_max_consecutive_wins(trades_df: pd.DataFrame) -> int:
        """
        Calculate maximum consecutive winning trades
        
        Args:
            trades_df: DataFrame with 'Win' column (boolean)
        
        Returns:
            Maximum number of consecutive wins
        """
        if trades_df.empty or 'Win' not in trades_df.columns:
            return 0
        
        max_wins = 0
        current_wins = 0
        
        for win in trades_df['Win']:
            if win:
                current_wins += 1
                max_wins = max(max_wins, current_wins)
            else:
                current_wins = 0
        
        return max_wins
    
    @staticmethod
    def calculate_profit_factor(trades_df: pd.DataFrame) -> float:
        """
        Calculate profit factor (gross profit / gross loss)
        
        Args:
            trades_df: DataFrame with 'PnL ($)' column
        
        Returns:
            Profit factor
        """
        if trades_df.empty or 'PnL ($)' not in trades_df.columns:
            return 0.0
        
        gross_profit = trades_df[trades_df['PnL ($)'] > 0]['PnL ($)'].sum()
        gross_loss = abs(trades_df[trades_df['PnL ($)'] < 0]['PnL ($)'].sum())
        
        return gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
    
    @staticmethod
    def calculate_average_trade_duration(trades_df: pd.DataFrame) -> float:
        """
        Calculate average trade duration in days
        
        Args:
            trades_df: DataFrame with 'Days held' column
        
        Returns:
            Average trade duration in days
        """
        if trades_df.empty or 'Days held' not in trades_df.columns:
            return 0.0
        
        return trades_df['Days held'].mean()
    
    @staticmethod
    def calculate_recovery_factor(total_pnl: float, max_drawdown: float) -> float:
        """
        Calculate recovery factor (total return / max drawdown)
        
        Args:
            total_pnl: Total profit/loss
            max_drawdown: Maximum drawdown
        
        Returns:
            Recovery factor
        """
        if max_drawdown == 0:
            return float('inf') if total_pnl > 0 else 0.0
        
        return total_pnl / max_drawdown
    
    @staticmethod
    def calculate_calmar_ratio(annualized_return: float, max_drawdown_pct: float) -> float:
        """
        Calculate Calmar ratio (annualized return / max drawdown percentage)
        
        Args:
            annualized_return: Annualized return percentage
            max_drawdown_pct: Maximum drawdown as percentage
        
        Returns:
            Calmar ratio
        """
        if max_drawdown_pct == 0:
            return float('inf') if annualized_return > 0 else 0.0
        
        return annualized_return / max_drawdown_pct
    
    @staticmethod
    def calculate_sortino_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.02, target: float = 0.0) -> float:
        """
        Calculate annualized Sortino ratio using proper semi-deviation.

        Args:
            daily_returns: Series of daily fractional returns from build_daily_returns
            risk_free_rate: Annual risk-free rate (default 2%)
            target: Minimum acceptable daily return (default 0.0)

        Returns:
            Annualized Sortino ratio
        """
        if daily_returns.empty:
            return 0.0

        daily_rf = risk_free_rate / 252
        excess = daily_returns - daily_rf

        downside = np.minimum(daily_returns - target, 0)
        semi_dev = np.sqrt(np.mean(downside ** 2))

        if semi_dev == 0:
            return float('inf') if excess.mean() > 0 else 0.0

        return (excess.mean() / semi_dev) * np.sqrt(252)
