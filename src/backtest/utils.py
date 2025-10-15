
import pandas as pd


class BacktestUtils:
    @staticmethod
    def backtest_pair(df: pd.DataFrame, stock1:str, stock2:str, zscore_window:int, entry_threshold:float, capital: float)-> pd.DataFrame:
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

                # Stop loss: exit if z-score moves 1.5 units further away from mean
                # For LONG: entry is negative (e.g., -2.5), stop at -4.0 (entry - 1.5)
                # For SHORT: entry is positive (e.g., +2.5), stop at +4.0 (entry + 1.5)
                elif position['type'] == 'LONG' and zscore < (position['entry_zscore'] - 1.5):
                    exit_signal = True
                    exit_reason = 'Stop Loss'
                elif position['type'] == 'SHORT' and zscore > (position['entry_zscore'] + 1.5):
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

                    total_pnl = stock1_pnl + stock2_pnl
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
                        'PnL ($)': total_pnl,
                        'PnL (%)': pnl_percent,
                        'Exit Reason': exit_reason
                    })

                    position = None  # Reset position
        
        return pd.DataFrame(trades)
    
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

        # Calculate percentage drawdown relative to INITIAL CAPITAL (not peak)
        max_drawdown_pct = (max_drawdown_dollars / initial_capital) * 100 if initial_capital > 0 else 0.0

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
    def calculate_sharpe_ratio(pnl_series: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio from P&L series
        
        Args:
            pnl_series: Series of P&L values from individual trades
            risk_free_rate: Annual risk-free rate (default 2%)
        
        Returns:
            Sharpe ratio
        """
        if pnl_series.empty or pnl_series.std() == 0:
            return 0.0
        
        # Convert to daily returns (assuming trades are roughly daily frequency)
        daily_rf_rate = risk_free_rate / 252  # 252 trading days per year
        excess_returns = pnl_series - daily_rf_rate
        
        return excess_returns.mean() / pnl_series.std() if pnl_series.std() > 0 else 0.0
    
    @staticmethod
    def calculate_volatility(pnl_series: pd.Series, annualized: bool = True) -> float:
        """
        Calculate volatility of P&L series
        
        Args:
            pnl_series: Series of P&L values
            annualized: Whether to annualize the volatility
        
        Returns:
            Volatility (standard deviation)
        """
        if pnl_series.empty:
            return 0.0
        
        volatility = pnl_series.std()
        if annualized:
            volatility *= (252 ** 0.5)  # Annualize assuming 252 trading days
        
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
    def calculate_sortino_ratio(pnl_series: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sortino ratio (uses downside deviation instead of total volatility)
        
        Args:
            pnl_series: Series of P&L values
            risk_free_rate: Annual risk-free rate
        
        Returns:
            Sortino ratio
        """
        if pnl_series.empty:
            return 0.0
        
        daily_rf_rate = risk_free_rate / 252
        excess_returns = pnl_series - daily_rf_rate
        
        # Calculate downside deviation (only negative returns)
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return float('inf') if excess_returns.mean() > 0 else 0.0
        
        downside_deviation = downside_returns.std()
        
        return excess_returns.mean() / downside_deviation if downside_deviation > 0 else 0.0
