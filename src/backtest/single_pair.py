
from datetime import datetime
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backtest.utils import BacktestUtils
from common.utils import CommonUtils
from scanner.utils import PairScannerUtils

class SinglePairBacktest:
    def __init__(self, ticker1:str, ticker2:str, period:str, zscore_window:int, entry_threshold:float, capital:float, should_plot_results:bool = False):
        self.ticker1 = ticker1
        self.ticker2 = ticker2
        self.period = period
        self.zscore_window = zscore_window
        self.entry_threshold = entry_threshold
        self.capital = capital
        self.should_plot_results = should_plot_results

    def run_test(self) -> dict:
        df = CommonUtils.fetch_data([self.ticker1, self.ticker2], period=self.period, interval="1d")
        if df is None:
            return
        
        stats = PairScannerUtils.calculate_spread_stats(
            df[self.ticker1],
            df[self.ticker2],
            zscore_window=self.zscore_window,
            zscore_entry_threshold=self.entry_threshold
        )

        df['Spread'] = stats.get('spread_series', pd.Series(dtype=float))
        df['ZScore'] = stats.get('rolling_zscore_series', pd.Series(dtype=float))
        df['Hedge_Ratio'] = stats.get('hedge_ratio', 1.0)

        if not stats:
            print("No stats calculated.")
            return
        
        print(stats)

        trades_df = BacktestUtils.backtest_pair(
            df,
            self.ticker1,
            self.ticker2,
            zscore_window=self.zscore_window,
            entry_threshold=self.entry_threshold,
            capital=self.capital
        )

        if trades_df is None:
            print("No trades executed.")
            return

        if self.should_plot_results:
            self.plot_results(df, trades_df)

        return self.generate_results(df, trades_df)

    def plot_results(self, df, trades_df) -> None:
        """Plot z-score chart and trade markers"""
        print(f"\n📊 Generating visualization...")

        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        # Plot 1: Price comparison (normalized) with BUY/SELL arrows
        ax1 = axes[0]
        norm1 = df[self.ticker1] / df[self.ticker1].iloc[0] * 100
        norm2 = df[self.ticker2] / df[self.ticker2].iloc[0] * 100
        ax1.plot(df.index, norm1, label=self.ticker1, linewidth=2, color='blue')
        ax1.plot(df.index, norm2, label=self.ticker2, linewidth=2, color='orange')

        # Add trade markers on price chart
        if not trades_df.empty:
            for _, trade in trades_df.iterrows():
                entry_date = trade['Entry Date']
                exit_date = trade['Exit Date']

                # Get normalized prices at entry and exit
                entry_price1 = norm1.loc[entry_date] if entry_date in norm1.index else None
                entry_price2 = norm2.loc[entry_date] if entry_date in norm2.index else None
                exit_price1 = norm1.loc[exit_date] if exit_date in norm1.index else None
                exit_price2 = norm2.loc[exit_date] if exit_date in norm2.index else None

                if trade['Position'] == 'LONG':
                    # LONG spread: BUY stock2 (orange), SELL stock1 (blue)
                    if entry_price1 is not None:
                        ax1.annotate('SELL', xy=(entry_date, entry_price1),
                                   xytext=(0, 15), textcoords='offset points',
                                   ha='center', fontsize=8, fontweight='bold',
                                   color='red',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7),
                                   arrowprops=dict(arrowstyle='->', color='red', lw=2))
                    if entry_price2 is not None:
                        ax1.annotate('BUY', xy=(entry_date, entry_price2),
                                   xytext=(0, -15), textcoords='offset points',
                                   ha='center', fontsize=8, fontweight='bold',
                                   color='green',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                                   arrowprops=dict(arrowstyle='->', color='green', lw=2))

                    # Exit markers (opposite)
                    if exit_price1 is not None:
                        ax1.scatter(exit_date, exit_price1, color='darkred', marker='x', s=100, zorder=10, linewidths=3)
                    if exit_price2 is not None:
                        ax1.scatter(exit_date, exit_price2, color='darkgreen', marker='x', s=100, zorder=10, linewidths=3)

                else:  # SHORT
                    # SHORT spread: SELL stock2 (orange), BUY stock1 (blue)
                    if entry_price1 is not None:
                        ax1.annotate('BUY', xy=(entry_date, entry_price1),
                                   xytext=(0, 15), textcoords='offset points',
                                   ha='center', fontsize=8, fontweight='bold',
                                   color='green',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                                   arrowprops=dict(arrowstyle='->', color='green', lw=2))
                    if entry_price2 is not None:
                        ax1.annotate('SELL', xy=(entry_date, entry_price2),
                                   xytext=(0, -15), textcoords='offset points',
                                   ha='center', fontsize=8, fontweight='bold',
                                   color='red',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7),
                                   arrowprops=dict(arrowstyle='->', color='red', lw=2))

                    # Exit markers
                    if exit_price1 is not None:
                        ax1.scatter(exit_date, exit_price1, color='darkgreen', marker='x', s=100, zorder=10, linewidths=3)
                    if exit_price2 is not None:
                        ax1.scatter(exit_date, exit_price2, color='darkred', marker='x', s=100, zorder=10, linewidths=3)

        ax1.set_title(f'{self.ticker1} vs {self.ticker2} - Normalized Prices with Trade Signals', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Normalized Price (Base=100)')
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Z-Score with entry/exit markers and annotations
        ax2 = axes[1]
        ax2.plot(df.index, df['ZScore'], label='Z-Score', color='purple', linewidth=2)
        ax2.axhline(y=self.entry_threshold, color='red', linestyle='--', label=f'SHORT Entry (+{self.entry_threshold})', alpha=0.7, linewidth=2)
        ax2.axhline(y=-self.entry_threshold, color='green', linestyle='--', label=f'LONG Entry (-{self.entry_threshold})', alpha=0.7, linewidth=2)
        ax2.axhline(y=0, color='black', linestyle='-', label='Mean (Exit Target)', alpha=0.7, linewidth=2)

        # Mark trade entries and exits with detailed annotations
        if not trades_df.empty:
            for idx, trade in trades_df.iterrows():
                if trade['Position'] == 'LONG':
                    # LONG spread entry
                    color = 'green'
                    marker = '^'
                    label_text = f"LONG\nBuy {self.ticker2}\nSell {self.ticker1}"
                else:
                    # SHORT spread entry
                    color = 'red'
                    marker = 'v'
                    label_text = f"SHORT\nSell {self.ticker2}\nBuy {self.ticker1}"

                # Entry marker
                ax2.scatter(trade['Entry Date'], trade['Entry ZScore'], color=color, marker=marker,
                           s=150, zorder=5, alpha=0.8, edgecolors='black', linewidths=1.5)

                # Add text annotation for entry
                ax2.annotate(label_text, xy=(trade['Entry Date'], trade['Entry ZScore']),
                           xytext=(10, 10 if trade['Position'] == 'SHORT' else -10),
                           textcoords='offset points',
                           fontsize=7, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.6, edgecolor='black'),
                           arrowprops=dict(arrowstyle='->', color='black', lw=1))

                # Exit marker
                exit_color = 'darkgreen' if trade['Win'] else 'darkred'
                exit_marker = 'X'
                ax2.scatter(trade['Exit Date'], trade['Exit ZScore'], color=exit_color, marker=exit_marker,
                           s=150, zorder=5, linewidths=2, edgecolors='black')

                # Exit annotation with P&L
                exit_label = f"{'WIN' if trade['Win'] else 'LOSS'}\n${trade['PnL ($)']:.0f}"
                ax2.annotate(exit_label, xy=(trade['Exit Date'], trade['Exit ZScore']),
                           xytext=(10, -10),
                           textcoords='offset points',
                           fontsize=7, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen' if trade['Win'] else 'lightcoral',
                                   alpha=0.7, edgecolor='black'))

        ax2.set_title('Z-Score with Trade Signals & Stock Actions', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Z-Score', fontsize=11)
        ax2.legend(loc='upper left', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.fill_between(df.index, self.entry_threshold, 5, alpha=0.1, color='red', label='SHORT Zone')
        ax2.fill_between(df.index, -self.entry_threshold, -5, alpha=0.1, color='green', label='LONG Zone')

        # Plot 3: Cumulative P&L
        ax3 = axes[2]
        if not trades_df.empty:
            # Create cumulative P&L starting from 0
            cumulative_pnl = [0] + trades_df['PnL ($)'].cumsum().tolist()
            dates = [trades_df['Entry Date'].iloc[0]] + trades_df['Exit Date'].tolist()
            
            ax3.plot(dates, cumulative_pnl, marker='o', linewidth=2, markersize=5, color='purple')
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax3.fill_between(dates, cumulative_pnl, 0,
                            where=[x >= 0 for x in cumulative_pnl], alpha=0.3, color='green', interpolate=True)
            ax3.fill_between(dates, cumulative_pnl, 0,
                            where=[x < 0 for x in cumulative_pnl], alpha=0.3, color='red', interpolate=True)

        ax3.set_title('Cumulative P&L', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Cumulative P&L ($)')
        ax3.grid(True, alpha=0.3)

        # Align x-axis scales across all subplots
        x_min, x_max = df.index.min(), df.index.max()
        ax1.set_xlim(x_min, x_max)
        ax2.set_xlim(x_min, x_max)
        ax3.set_xlim(x_min, x_max)

        # Share x-axis formatting
        for ax in [ax1, ax2, ax3]:
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        # Create directory structure for saving images
        base_folder = "__reports__"
        today_folder = datetime.now().strftime("%Y-%m-%d")
        save_path = os.path.join(base_folder, today_folder)
        
        # Create directories if they don't exist
        os.makedirs(save_path, exist_ok=True)
        
        filename = f'{self.ticker1}_{self.ticker2}_{datetime.now().strftime("%Y%m%d")}_single_pair.png'
        full_filepath = os.path.join(save_path, filename)
        
        plt.savefig(full_filepath, dpi=150, bbox_inches='tight')
        print(f"   ✓ Chart saved as '{full_filepath}'")
        plt.close()

    def generate_results(self, df: pd.DataFrame, trades_df: pd.DataFrame) -> dict:
        """ Generate results that can be used by other modules to produce multi pair reports."""

        max_drawdown, max_drawdown_pct = BacktestUtils.calculate_max_drawdown(trades_df['PnL ($)']) if not trades_df.empty else (0, 0)
        annualized_return = BacktestUtils.calculate_annualized_return(trades_df['PnL ($)'].sum(), self.capital, df.index) if not trades_df.empty else 0

        results = {
            'Ticker1': self.ticker1,
            'Ticker2': self.ticker2,
            'Total Trades': len(trades_df),
            'Winning Trades': trades_df['Win'].sum(),
            'Losing Trades': len(trades_df) - trades_df['Win'].sum(),
            'Average trade duration (days)': (trades_df['Days held'].mean() if not trades_df.empty else 0),
            'Win Rate (%)': (trades_df['Win'].sum() / len(trades_df) * 100) if len(trades_df) > 0 else 0,
            'Profit factor': BacktestUtils.calculate_profit_factor(trades_df) if not trades_df.empty else 0,
            'Total PnL ($)': trades_df['PnL ($)'].sum(),
            'Compound Annual Growth Rate (%)': BacktestUtils.calculate_cagr(trades_df['PnL ($)'].sum(), self.capital, df.index) if not trades_df.empty else 0,
            'Annualized Return (%)': annualized_return,
            'Average PnL per Trade ($)': (trades_df['PnL ($)'].mean() if not trades_df.empty else 0),
            'Max Drawdown ($)': max_drawdown,
            'Max Drawdown (%)': max_drawdown_pct,
            'Hedge Ratio': df['Hedge_Ratio'].iloc[-1] if 'Hedge_Ratio' in df.columns else None,
            'Final Z-Score': df['ZScore'].iloc[-1] if 'ZScore' in df.columns else None,
            'Sharpe ratio': BacktestUtils.calculate_sharpe_ratio(trades_df['PnL ($)']) if not trades_df.empty else 0,
            'Sortino ratio': BacktestUtils.calculate_sortino_ratio(trades_df['PnL ($)']) if not trades_df.empty else 0,
            'Calmar ratio': BacktestUtils.calculate_calmar_ratio(annualized_return, max_drawdown_pct) if not trades_df.empty else 0,
            'Volatility': BacktestUtils.calculate_volatility(trades_df['PnL ($)'], annualized=True) if not trades_df.empty else 0,
        }
        return results

    def main(self):
        self.run_test()


if __name__ == "__main__":
    # If run directly, ask for input.
    ticker1 = input("Enter first ticker: ").strip().upper()
    ticker2 = input("Enter second ticker: ").strip().upper()
    try:
        period = int(input("Enter backtest period in years (e.g., 2): ").strip() or "2")
    except:
        period = 2

    backtest = SinglePairBacktest(ticker1, ticker2, period=f"{period}y", zscore_window=60, entry_threshold=2.0, capital=2500, should_plot_results=True)
    backtest.main()