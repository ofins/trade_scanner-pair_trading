
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
                if zscore <= -entry_threshold:
                    # LONG spread entry
                    # Stock 1 is undervalued relative to Stock 2
                    # Action: BUY stock1, SELL stock2
                    position = {
                        'type': 'LONG',
                        'entry_date': current_date,
                        'entry_zscore': zscore,
                        'stock1_price': df[stock1].iloc[i],
                        'stock2_price': df[stock2].iloc[i],
                        'hedge_ratio': df['Hedge_Ratio'].iloc[i]
                    }
                elif zscore >= entry_threshold:
                    # SHORT spread entry
                    # Stock 1 is overvalued relative to Stock 2
                    # Action: SELL stock1, BUY stock2
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

                # Stop loss (z-score moves 75% further away)
                elif position['type'] == 'LONG' and zscore < position['entry_zscore'] * 1.75:
                    exit_signal = True
                    exit_reason = 'Stop Loss'
                elif position['type'] == 'SHORT' and zscore > position['entry_zscore'] * 1.75:
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
                        # LONG spread: BUY stock1, SELL stock2
                        stock1_pnl = stock1_shares * (exit_stock1_price - position['stock1_price'])
                        stock2_pnl = stock2_shares * (position['stock2_price'] - exit_stock2_price)
                    else:
                        # SHORT spread: SELL stock1, BUY stock2
                        stock1_pnl = stock1_shares * (position['stock1_price'] - exit_stock1_price)
                        stock2_pnl = stock2_shares * (exit_stock2_price - position['stock2_price'])

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
    
    @staticmethod
    def calculate_max_drawdown(pnl_series: pd.Series) -> float:
        """
        Calculate the maximum drawdown from a series of P&L values.
        
        Args:
            pnl_series: Series of P&L values from individual trades
            
        Returns:
            Maximum drawdown as a positive value (e.g., 500.0 for $500 drawdown)
        """
        if pnl_series.empty:
            return 0.0
        
        # Calculate cumulative P&L starting from 0
        cumulative_pnl = pnl_series.cumsum()
        
        # Calculate running maximum (peak values)
        running_max = cumulative_pnl.expanding().max()
        
        # Calculate drawdown at each point (peak - current value)
        drawdown = running_max - cumulative_pnl
        
        # Return the maximum drawdown as a positive value
        max_drawdown = drawdown.max()
        
        return max_drawdown if not pd.isna(max_drawdown) else 0.0