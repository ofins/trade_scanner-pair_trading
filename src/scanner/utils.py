from datetime import datetime, timedelta
from io import StringIO
import os
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from statsmodels.tsa.stattools import coint, adfuller

class PairScannerUtils:

    @staticmethod
    def find_all_sp500_tickers() -> list[str]:
        """
        Fetch all tickers that are part of the S&P 500 from Wikipedia.
        """

        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "Mozilla/5.0"}

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        htmlBuffer = StringIO(response.text)

        sp500_table = pd.read_html(htmlBuffer)
        sp500_tickers = sp500_table[0]['Symbol'].tolist()
        return sp500_tickers

    @staticmethod
    def filter_large_cap_tickers(
        tickers: list[str], 
        min_market_cap: int = 10_000_000_000,
        batch_size: int = 50
    ) -> list[str]:
        """
        Filter tickers to only include those with market cap > min_market_cap.
        Uses batch fetching for better performance.
        """
        large_cap_tickers = []

        print(f"🔍 Checking {len(tickers)} tickers in batches of {batch_size}...")

        # Process tickers in batches
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            tickers_str = " ".join(batch)
            tickers_data = yf.Tickers(tickers_str)

            for symbol, ticker_obj in tickers_data.tickers.items():
                try:
                    info = ticker_obj.info
                    market_cap = info.get("marketCap", 0)
                    quote_type = info.get("quoteType", "")
                    
                    if market_cap > min_market_cap and quote_type != "ETF":
                        large_cap_tickers.append(symbol)
                except Exception:
                    continue  # skip errors silently

            print(f"✅ Processed batch {i // batch_size + 1} — total large caps so far: {len(large_cap_tickers)}")

        print(f"\n🏁 Found {len(large_cap_tickers)} tickers with market cap > ${min_market_cap/1e9:.1f}B")
        return large_cap_tickers

    """ Formulas relevant to pairs trading """

    @staticmethod
    def calculate_spread_stats(stock_x: pd.Series, stock_y: pd.Series, zscore_window: int, zscore_entry_threshold: float = 2.0, use_rolling_beta: bool = False, beta_lookback: int = 126) -> dict:
        """
        Calculate comprehensive spread statistics for pairs trading.

        Args:
            stock_x: Price series for first stock
            stock_y: Price series for second stock
            zscore_window: Window size for z-score calculation (typically 60 days)
            zscore_entry_threshold: Threshold for entry signals
            use_rolling_beta: If True, use rolling beta. If False, use static beta from all data (recommended).
            beta_lookback: Lookback period for beta calculation when use_rolling_beta=True

        Note: For most use cases, static beta (use_rolling_beta=False) is recommended as it provides
        a stable hedge ratio while the z-score window captures mean reversion signals.
        """
        try:
            if use_rolling_beta:
                # Calculate rolling beta and spread with separate beta window
                rolling_beta_series = pd.Series(index=stock_x.index, dtype=float)
                rolling_intercept_series = pd.Series(index=stock_x.index, dtype=float)
                spread = pd.Series(index=stock_x.index, dtype=float)

                # Use beta_lookback for beta calculation (separate from z-score window)
                for i in range(beta_lookback, len(stock_x)):
                    # Use longer window for stable beta estimation
                    window_x = stock_x.iloc[i-beta_lookback:i].values.reshape(-1, 1)
                    window_y = stock_y.iloc[i-beta_lookback:i].values

                    X_with_intercept = np.column_stack([window_x, np.ones(len(window_x))])
                    beta, intercept = np.linalg.lstsq(X_with_intercept, window_y, rcond=None)[0]

                    rolling_beta_series.iloc[i] = beta
                    rolling_intercept_series.iloc[i] = intercept
                    spread.iloc[i] = stock_y.iloc[i] - beta * stock_x.iloc[i] - intercept

                # Use the most recent beta as the "current" hedge ratio
                beta = rolling_beta_series.iloc[-1]
            else:
                # Static beta calculated from all data (RECOMMENDED - default behavior)
                X = stock_x.values.reshape(-1, 1)
                y = stock_y.values
                X = np.column_stack([X, np.ones(len(X))])
                beta, intercept = np.linalg.lstsq(X, y, rcond=None)[0]

                # Calculate spread using static beta
                spread = stock_y - beta * stock_x - intercept

            # Calculate rolling z-score (using specified window)
            rolling_mean = spread.rolling(window=zscore_window).mean()
            rolling_std = spread.rolling(window=zscore_window).std()
            rolling_zscore = (spread - rolling_mean) / rolling_std

            # Current z-score (last value)
            current_zscore = rolling_zscore.iloc[-1] if not rolling_zscore.empty else 0

            # Calculate full-period z-score for comparison
            full_zscore = (spread - spread.mean()) / spread.std()

            # Calculate half-life of mean reversion (scalar for current)
            halflife = PairScannerUtils.calculate_half_life(spread)

            # Calculate rolling Hurst exponent
            rolling_hurst = pd.Series(index=spread.index, dtype=float)
            for i in range(zscore_window, len(spread)):
                window_spread = spread.iloc[i-zscore_window:i]
                rolling_hurst.iloc[i] = PairScannerUtils.calculate_hurst_exponent(window_spread)

            # Calculate rolling Half-Life
            rolling_halflife = pd.Series(index=spread.index, dtype=float)
            for i in range(zscore_window, len(spread)):
                window_spread = spread.iloc[i-zscore_window:i]
                hl = PairScannerUtils.calculate_half_life(window_spread)
                rolling_halflife.iloc[i] = hl if hl and 0 < hl < 365 else np.nan

            # Calculate rolling ADF p-value
            rolling_adf = pd.Series(index=spread.index, dtype=float)
            for i in range(zscore_window, len(spread)):
                window_spread = spread.iloc[i-zscore_window:i]
                rolling_adf.iloc[i] = PairScannerUtils.test_stationarity(window_spread)

            # Calculate historical z-score statistics
            zscore_mean = rolling_zscore.dropna().mean()
            zscore_std = rolling_zscore.dropna().std()
            zscore_min = rolling_zscore.dropna().min()
            zscore_max = rolling_zscore.dropna().max()

            # Count mean reversion success rate
            # How often does the spread cross zero after hitting entry threshold?
            crosses = 0
            successes = 0
            for i in range(zscore_window, len(rolling_zscore) - 10):
                z = rolling_zscore.iloc[i]
                if abs(z) >= zscore_entry_threshold and not np.isnan(z):
                    crosses += 1
                    # Check if it reverts to mean within next 20 days
                    future_z = rolling_zscore.iloc[i+1:i+21]
                    if len(future_z) > 0:
                        # Success if z-score crosses zero or gets close to it
                        if (z > 0 and future_z.min() < 0.5) or (z < 0 and future_z.max() > -0.5):
                            successes += 1

            reversion_rate = (successes / crosses * 100) if crosses > 0 else None

            return {
                'hedge_ratio': beta,
                'spread': spread,
                'spread_mean': spread.mean(),
                'spread_std': spread.std(),
                'current_zscore': current_zscore,
                'full_period_zscore': full_zscore.iloc[-1] if not full_zscore.empty else 0,
                'halflife': halflife if halflife and 0 < halflife < 365 else None,
                'zscore_mean': zscore_mean,
                'zscore_std': zscore_std,
                'zscore_min': zscore_min,
                'zscore_max': zscore_max,
                'reversion_rate': reversion_rate,
                'spread_series': spread,
                'rolling_zscore_series': rolling_zscore,
                'hurst': rolling_hurst,
                'halflife_series': rolling_halflife,
                'adf_pvalue': rolling_adf
            }
        except Exception as e:
            print(f"Error processing pair {stock_x}-{stock_y}: {e}")
            return {
                'hedge_ratio': 1.0,
                'spread': pd.Series(dtype=float),
                'spread_mean': 0,
                'spread_std': 1,
                'current_zscore': 0,
                'full_period_zscore': 0,
                'halflife': None,
                'zscore_mean': 0,
                'zscore_std': 1,
                'zscore_min': 0,
                'zscore_max': 0,
                'reversion_rate': None,
                'spread_series': pd.Series(dtype=float),
                'rolling_zscore_series': pd.Series(dtype=float),
                'hurst': pd.Series(dtype=float),
                'halflife_series': pd.Series(dtype=float),
                'adf_pvalue': pd.Series(dtype=float)
            }
        
    @staticmethod
    def calculate_half_life(spread: pd.Series) -> float:
        """ Calculate half-life of mean reversion for a given spread series  """
        halflife = None
        try:
            lagged_spread = spread.shift(1).iloc[1:]
            delta_spread = spread.diff().iloc[1:]

            valid_idx = ~(lagged_spread.isna() | delta_spread.isna())
            lagged_spread = lagged_spread[valid_idx]
            delta_spread = delta_spread[valid_idx]

            if len(lagged_spread) > 10:
                X_hl = lagged_spread.values.reshape(-1, 1)
                y_hl = delta_spread.values
                theta = np.linalg.lstsq(X_hl, y_hl, rcond=None)[0][0]
                halflife = -np.log(2) / theta if theta < 0 else None
            else:
                halflife = None
        except:
            halflife = None
        return halflife if halflife and halflife > 0 else None

    @staticmethod
    def test_stationarity(series: pd.Series) -> float:
        """Test if a series is stationary using ADF test"""
        try:
            result = adfuller(series.dropna(), autolag='AIC')
            return result[1]  # Return p-value
        except:
            return 1.0  # Return high p-value if test fails
    
    @staticmethod
    def calculate_hurst_exponent(series: pd.Series) -> float:
        """
        Calculate Hurst Exponent using simplified variance method.
        For spread series, measures mean reversion tendency.
        H < 0.5: Mean reverting (good for pairs trading)
        H = 0.5: Random walk
        H > 0.5: Trending/persistent
        """
        try:
            # Remove NaN and ensure we have enough data
            series = series.dropna()
            # Require at least 20 data points for meaningful calculation
            if len(series) < 20:
                return 0.5

            # Demean the series (critical for spreads!)
            series = series - series.mean()

            # Use simplified variance method
            # For different time lags, calculate variance of differences
            # Adjust max lag based on series length
            max_lag = min(len(series) // 4, 100)
            if max_lag < 3:
                return 0.5

            lags = range(2, max_lag)

            variances = []
            valid_lags = []

            for lag in lags:
                # Calculate differences at this lag
                diffs = series.diff(lag).dropna()

                if len(diffs) < 5:
                    continue

                var = np.var(diffs)
                if var > 0:
                    variances.append(var)
                    valid_lags.append(lag)

            # Need at least 5 valid variance points for regression
            if len(variances) < 5:
                return 0.5

            # Fit log(variance) vs log(lag): variance ~ lag^(2H)
            # So: log(variance) = 2H * log(lag) + constant
            # Therefore: H = slope / 2
            poly = np.polyfit(np.log(valid_lags), np.log(variances), 1)
            hurst = poly[0] / 2.0  # Divide by 2 because variance scales as lag^(2H)

            # Bound to reasonable range
            hurst = max(0.0, min(1.0, hurst))

            return hurst
        except Exception as e:
            return 0.5  # Return neutral value if calculation fails

if __name__ == "__main__":
    tickers = PairScannerUtils.find_all_sp500_tickers()