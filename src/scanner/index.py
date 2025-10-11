import numpy as np
import pandas as pd
import yfinance as yf
import json
import os
import sys
from statsmodels.tsa.stattools import coint, adfuller

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.utils import CommonUtils
from utils import PairScannerUtils

class PairsScanner: 
    def __init__(self, min_correlation: float = 0.7, zscore_window: int = 60, zscore_entry_threshold: float = 2.0):
        self.min_correlation = min_correlation
        self.zscore_window = zscore_window
        self.zscore_entry_threshold = zscore_entry_threshold
    
    def fetch_stocks_by_sector(self) -> dict[str, list[str]]:
        json_path = os.path.join(os.path.dirname(__file__), "../constants/stock_by_sectors.json")
        with open(json_path, 'r') as f:
            data = json.load(f)
            print(f"Loaded sectors: {list(data.keys())} with counts: {[len(v) for v in data.values()]}")
            return data
    

    def find_cointegrated_pairs(self, data: pd.DataFrame, sector_name: str) -> list[dict]:
        if data.empty or data.shape[1] < 2:
            print(f"⚠️ Not enough data to find cointegrated pairs in sector {sector_name}.")
            return []
        
        n = data.shape[1]
        tickers = data.columns.tolist()

        print(f" Calculating correlations for {sector_name}...")
        corr_matrix = data.corr()

        pairs_to_test: list[tuple[str, str]] = []

        # Identify highly correlated pairs 
        for i in range(n):
            for j in range(i+1,n):
                if abs(corr_matrix.iloc[i,j]) >= self.min_correlation:
                    pairs_to_test.append((tickers[i], tickers[j]))

        if not pairs_to_test:
            print(f"⚠️ No highly correlated pairs found in sector {sector_name}.")
            return []
        
        print(f" Testing {len(pairs_to_test)} pairs in BOTH directions")

        results: list[dict] = []
        tested = 0

        for ticker1, ticker2 in pairs_to_test:
            try:
                # return only rows where both tickers have data
                pair_data = data[[ticker1, ticker2]].dropna()

                # Skip pairs with insufficient data
                if len(pair_data) < 100:
                    continue

                # TEST DIRECTION 1: ticker1 -> ticker2
                score1, pvalue1, _ = coint(pair_data[ticker1], pair_data[ticker2])
                tested += 1

                # TEST DIRECTION 2: ticker2 -> ticker1
                score2, pvalue2, _ = coint(pair_data[ticker2], pair_data[ticker1])
                tested += 1

                if pvalue1 < 0.10 or pvalue2 < 0.10:
                # Use the direction with the lower p-value
                    if pvalue1 < pvalue2:
                        # DIRECTION 1 is better
                        stock_y, stock_x = ticker1, ticker2
                        best_pvalue = pvalue1
                        best_score = score1
                        direction = 1
                    else:
                        # DIRECTION 2 is better
                        stock_y, stock_x = ticker2, ticker1
                        best_pvalue = pvalue2
                        best_score = score2
                        direction = 2


                    spread_stats = PairScannerUtils.calculate_spread_stats(pair_data[stock_x], pair_data[stock_y], zscore_window=self.zscore_window, zscore_entry_threshold=self.zscore_entry_threshold)
                    spread = pair_data[stock_y] - spread_stats['hedge_ratio'] * pair_data[stock_x]
                    spread_adf_pvalue = PairScannerUtils.test_stationarity(spread)


                    result = {
                            'Sector': sector_name,
                            'Stock_1': stock_x,  # X variable (independent)
                            'Stock_2': stock_y,  # Y variable (dependent)
                            'Correlation': corr_matrix.loc[stock_x, stock_y],
                            'Coint_PValue': best_pvalue,
                            'Coint_Score': best_score,
                            'Optimal_Direction': direction,  # Track which direction was better
                            'Alt_PValue': pvalue2 if direction == 1 else pvalue1,  # Store alternative p-value
                            'Spread_ADF_PValue': spread_adf_pvalue,
                            'Hedge_Ratio': spread_stats['hedge_ratio'],
                            'Spread_Mean': spread_stats['spread_mean'],
                            'Spread_Std': spread_stats['spread_std'],
                            'Current_ZScore': spread_stats['current_zscore'],
                            'Half_Life': spread_stats['halflife'],
                            'Stock1_Price': pair_data[stock_x].iloc[-1],
                            'Stock2_Price': pair_data[stock_y].iloc[-1],
                            'ZScore_Mean': spread_stats['zscore_mean'],
                            'ZScore_Std': spread_stats['zscore_std'],
                            'ZScore_Min': spread_stats['zscore_min'],
                            'ZScore_Max': spread_stats['zscore_max'],
                            'Reversion_Rate': spread_stats['reversion_rate']
                        }
                    
                    results.append(result)
            except Exception as e:
                continue
        print(f"  Tested {tested} pairs, found {len(results)} cointegrated pairs in sector {sector_name}.")
        return results
        
    def run_scanner(self, sectors: dict[str, list[str]]):
        all_results = []

        for sector, tickers in sectors.items():
            print(f"Scanning sector: {sector} with {len(tickers)} tickers")

            data = CommonUtils.fetch_data(tickers, period="2y", interval="1d")
            print(f" Data fetched for {sector}, shape: {data.shape}")

            if data.empty or data.shape[1] < 2:
                print(f"  ⚠️  Insufficient data for {sector}")
                continue
            
            # Find cointegrated pairs
            sector_results = self.find_cointegrated_pairs(data, sector)
            all_results.extend(sector_results)
        
        return all_results
        
    def main(self):
        sectors = self.fetch_stocks_by_sector()
        results = self.run_scanner(sectors)
        CommonUtils.save_to_xlsx(results, filename="pairs_trading_results", base_folder="__reports__")
        
            
if __name__ == "__main__":
    scanner = PairsScanner()
    scanner.main()
    
