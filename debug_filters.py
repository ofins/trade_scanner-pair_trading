"""
Debug script: traces exactly which filter eliminates each correlated pair.
Run from project root: python debug_filters.py
"""
import numpy as np
import pandas as pd
import json
import os
import sys
from statsmodels.tsa.stattools import coint
from statsmodels.stats.multitest import multipletests

sys.path.append('src')
from common.utils import CommonUtils
from scanner.utils import PairScannerUtils

# --- Config (matches PairsScanner defaults) ---
MIN_CORRELATION      = 0.7
MAX_COINT_PVALUE     = 0.05
STRICT_COINT_PVALUE  = 0.02
ZSCORE_WINDOW        = 60
ZSCORE_ENTRY         = 2.0
MIN_HALFLIFE         = 5
MAX_HALFLIFE         = 30
MIN_REVERSION_RATE   = 60.0
MIN_ZERO_CROSSINGS   = 7
MAX_ZSCORE_ABS       = 3.5

def load_tickers():
    path = os.path.join('src', 'constants', 'stock_by_sectors.json')
    with open(path) as f:
        return json.load(f)

def main():
    sectors = load_tickers()
    sector_name = 'Technology'
    tickers = sectors[sector_name]

    print(f"Fetching data for {len(tickers)} tickers...")
    data = CommonUtils.fetch_data(tickers, period="2y", interval="1d")
    print(f"Data shape: {data.shape}\n")

    split_point = len(data) // 2
    training_data = data.iloc[:split_point].copy()
    print(f"Training: {training_data.index[0].date()} → {training_data.index[-1].date()} ({len(training_data)} days)")
    print(f"Full:     {data.index[0].date()} → {data.index[-1].date()} ({len(data)} days)\n")

    n = training_data.shape[1]
    tickers_list = training_data.columns.tolist()

    log_returns = np.log(training_data / training_data.shift(1)).iloc[1:]
    corr_matrix = log_returns.corr()

    pairs_to_test = []
    for i in range(n):
        for j in range(i+1, n):
            if abs(corr_matrix.iloc[i, j]) >= MIN_CORRELATION:
                pairs_to_test.append((tickers_list[i], tickers_list[j]))

    print(f"Correlated pairs (≥{MIN_CORRELATION}): {len(pairs_to_test)}\n")
    print(f"{'Pair':<20} {'CointP':>8} {'StrictP':>8} {'ADF':>8} {'HL':>7} {'Hurst':>7} {'ZX':>5} {'RevR%':>7} {'CurrZ':>7}  Result")
    print("-" * 105)

    filter_counts = {
        'coint_pvalue':    0,
        'strict_pvalue':   0,
        'adf_pvalue':      0,
        'half_life':       0,
        'hurst':           0,
        'zero_crossings':  0,
        'reversion_rate':  0,
        'current_zscore':  0,
        'fdr_correction':  0,
        'passed':          0,
        'exception':       0,
    }

    pre_fdr_results = []

    for ticker1, ticker2 in pairs_to_test:
        try:
            pair_data = training_data[[ticker1, ticker2]].dropna()
            if len(pair_data) < 100:
                continue

            score1, pvalue1, _ = coint(pair_data[ticker1], pair_data[ticker2])
            score2, pvalue2, _ = coint(pair_data[ticker2], pair_data[ticker1])

            best_p = min(pvalue1, pvalue2)
            stock_y, stock_x = (ticker1, ticker2) if pvalue1 < pvalue2 else (ticker2, ticker1)
            best_pvalue = pvalue1 if pvalue1 < pvalue2 else pvalue2
            alt_pvalue  = pvalue2 if pvalue1 < pvalue2 else pvalue1

            tag = None

            if best_pvalue > MAX_COINT_PVALUE:
                tag = f"FAIL coint>{MAX_COINT_PVALUE}"
                filter_counts['coint_pvalue'] += 1
            elif best_pvalue > STRICT_COINT_PVALUE:
                tag = f"FAIL strict>{STRICT_COINT_PVALUE}"
                filter_counts['strict_pvalue'] += 1
            else:
                # Run spread stats
                spread_stats = PairScannerUtils.calculate_spread_stats(
                    pair_data[stock_x], pair_data[stock_y],
                    zscore_window=ZSCORE_WINDOW,
                    zscore_entry_threshold=ZSCORE_ENTRY
                )
                adf_p     = PairScannerUtils.test_stationarity(spread_stats['spread_series'])
                hl        = spread_stats['halflife']
                hurst     = PairScannerUtils.calculate_hurst_exponent(spread_stats['spread_series'])
                rolling_z = spread_stats['rolling_zscore_series']
                zx        = int(((rolling_z.shift(1) * rolling_z) < 0).sum())
                rev_rate  = spread_stats.get('reversion_rate')
                curr_z    = spread_stats['current_zscore']

                hl_str      = f"{hl:.1f}" if hl is not None else "None"
                rev_str     = f"{rev_rate:.1f}" if rev_rate is not None else "None"
                curr_z_str  = f"{curr_z:.2f}" if not np.isnan(curr_z) else "NaN"

                if adf_p > 0.05:
                    tag = f"FAIL ADF={adf_p:.3f}"
                    filter_counts['adf_pvalue'] += 1
                elif hl is None or hl < MIN_HALFLIFE or hl > MAX_HALFLIFE:
                    tag = f"FAIL HL={hl_str}"
                    filter_counts['half_life'] += 1
                elif hurst >= 0.5:
                    tag = f"FAIL Hurst={hurst:.3f}"
                    filter_counts['hurst'] += 1
                elif zx < MIN_ZERO_CROSSINGS:
                    tag = f"FAIL ZX={zx}"
                    filter_counts['zero_crossings'] += 1
                elif rev_rate is None or rev_rate < MIN_REVERSION_RATE:
                    tag = f"FAIL RevR={rev_str}"
                    filter_counts['reversion_rate'] += 1
                elif abs(curr_z) > MAX_ZSCORE_ABS:
                    tag = f"FAIL CurrZ={curr_z_str}"
                    filter_counts['current_zscore'] += 1
                else:
                    tag = "PRE-FDR PASS"
                    filter_counts['passed'] += 1
                    pre_fdr_results.append({
                        'pair': f"{stock_x}/{stock_y}",
                        'pvalue': best_pvalue,
                        'adf': adf_p,
                        'hl': hl,
                        'hurst': hurst,
                        'zx': zx,
                        'rev_rate': rev_rate,
                        'curr_z': curr_z,
                    })

                print(f"{stock_x}/{stock_y:<20} {best_pvalue:>8.4f} {alt_pvalue:>8.4f} {adf_p:>8.4f} {hl_str:>7} {hurst:>7.3f} {zx:>5} {rev_str:>7} {curr_z_str:>7}  {tag}")
                continue

            # Pairs that failed coint/strict don't have spread stats
            corr = corr_matrix.loc[ticker1, ticker2]
            print(f"{ticker1}/{ticker2:<20} {best_pvalue:>8.4f} {'--':>8} {'--':>8} {'--':>7} {'--':>7} {'--':>5} {'--':>7} {'--':>7}  {tag}")

        except Exception as e:
            filter_counts['exception'] += 1
            print(f"{ticker1}/{ticker2:<20} {'ERR':>8}  exception: {e}")

    # FDR correction on pre-FDR passes
    if pre_fdr_results:
        pvalues = [r['pvalue'] for r in pre_fdr_results]
        reject, _, _, _ = multipletests(pvalues, alpha=0.05, method='fdr_bh')
        survivors = [r for r, keep in zip(pre_fdr_results, reject) if keep]
        fdr_removed = len(pre_fdr_results) - len(survivors)
        filter_counts['fdr_correction'] = fdr_removed
        filter_counts['passed'] = len(survivors)

    print("\n" + "=" * 60)
    print("FILTER SUMMARY")
    print("=" * 60)
    for k, v in filter_counts.items():
        print(f"  {k:<22}: {v}")

if __name__ == "__main__":
    main()
