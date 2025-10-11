import os
import sys
import pandas as pd
from tqdm import tqdm

from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backtest.single_pair import SinglePairBacktest
from common.utils import CommonUtils


class BacktestPairTrading:
    def __init__(self):
        pass

    def get_top_pairs(self, n:int = 50):
        # Build the correct path from the project root
        project_root = os.path.join(os.path.dirname(__file__), '..', '..')
        today_folder = datetime.now().strftime("%Y-%m-%d")
        filename = f"pairs_trading_results_{datetime.now().strftime('%Y%m%d')}.xlsx"
        file_path = os.path.join(project_root, "__reports__", today_folder, filename)
        
        pairs_df = CommonUtils.read_xlsx(file_path)
        top_pairs = pairs_df.head(n).copy()

        # Vectorized swap - no loop needed
        mask = top_pairs['Optimal_Direction'] == 2
        top_pairs.loc[mask, ['Stock_1', 'Stock_2']] = top_pairs.loc[mask, ['Stock_2', 'Stock_1']].values

        return top_pairs
    
    def run_backtests(self, pairs: pd.DataFrame) -> list[dict]:
        results = []
        pbar = tqdm(total=len(pairs), desc="Running backtests")

        for _, row in pairs.iterrows():
            stock1 = row['Stock_1']
            stock2 = row['Stock_2']
            pair_name = f"{stock1}-{stock2}"

            pbar.set_description(f"Backtesting {pair_name}")

            try:
                pair_tester = SinglePairBacktest(stock1, stock2, period="2y",
                    zscore_window=60,
                    entry_threshold=2.0,
                    capital=2500,
                    should_plot_results=False
                )
                result = pair_tester.run_test()
                if result:
                    results.append(result)
                pbar.update(1)
            except Exception as e:
                print(f"Error backtesting {pair_name}: {e}")
                continue

        pbar.close()
        return results

    def main(self):
        top_pairs = self.get_top_pairs()
        
        results = self.run_backtests(top_pairs)

        if results:
            CommonUtils.save_to_xlsx(results, "backtest_pair_trading_results", base_folder="__reports__")
        print(f"\n🏁 Completed backtests for {len(results)} pairs.")

if __name__ == "__main__":
    backtest = BacktestPairTrading()
    backtest.main()

