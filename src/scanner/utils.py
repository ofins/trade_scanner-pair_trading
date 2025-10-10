from datetime import datetime, timedelta
from io import StringIO
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
    
    """ Fetching data """

    @staticmethod
    def fetch_data(tickers: list[str], period:str = "5y", interval: str = "1d") -> pd.DataFrame:
        """ Fetch historical price data for given tickers """
        print(f"  Fetching data for {len(tickers)} tickers...")

        try:
            tickers_str = ' '.join(tickers)
            print(f"Downloading...")

            raw_data = yf.download(
                tickers_str,
                period=period,
                interval=interval,
                progress=False,
                group_by='ticker' if len(tickers) > 1 else None,
                threads=True,
                repair=True
            )

            if raw_data.empty:
                print(" No data downloaded.")
                return pd.DataFrame()
            
            # Extract price data
            data_dict = PairScannerUtils._extract_prices(raw_data, tickers)

            if not data_dict:
                print("     No valid ticker data extracted.")
                return pd.DataFrame()
            
            # Build and validate timeframe
            df = pd.DataFrame(data_dict).ffill().dropna()

            min_required_points = 264 # ~1 year of trading days

            if len(df) < min_required_points:
                print(f"    Insufficient data points: {len(df) < {min_required_points}}")
                return pd.DataFrame()
            
            print(f"    Successfully loaded: {len(data_dict)} tickers")
            return df

        except Exception as e:
            print(f"Error fetching data: {e}")

    @staticmethod
    def _extract_prices(raw_data: pd.DataFrame | None, tickers: list[str]) -> dict:
        data_dict = {}

        if len(tickers) == 1:
            # Single ticker
            price_col = PairScannerUtils._get_price_column(raw_data)
            if price_col is not None:
                data_dict[tickers[0]] = price_col

        else:
            # Multiple tickers
            for ticker in tickers:
                price_col = PairScannerUtils._get_ticker_price(raw_data, ticker)
                if price_col is not None and len(price_col.dropna()) > 200: # at least ~200 data points
                    data_dict[ticker] = price_col
        
        return data_dict

    @staticmethod
    def _get_price_column(data: pd.DataFrame) -> pd.Series | None:
        """Get price column from simple column structure"""
        for col_name in ['Adj Close', 'Close']:
            if col_name in data.columns:
                return data[col_name]
        return None

    @staticmethod
    def _get_ticker_price(data: pd.DataFrame, ticker: list[str]) -> pd.Series | None:
        """Get price column for specific ticker from MultiIndex"""
        for col_name in ['Adj Close', 'Close']:
            if (ticker, col_name) in data.columns:
                return data[(ticker, col_name)]
        return None

    """ Formulas relevant to pairs trading """

    @staticmethod
    def calculate_spread_stats(stock_x: pd.Series, stock_y: pd.Series, zscore_window: int, zscore_entry_threshold: float = 2.0) -> dict:
        """ Calculate hedge ratio using OLS regression"""
        try:
            X = stock_x.values.reshape(-1, 1) # 2D array for sklearn
            y = stock_y.values
            X = np.column_stack([X, np.ones(len(X))])
            beta, intercept = np.linalg.lstsq(X, y, rcond=None)[0]

            # Calculate spread
            spread = stock_y - beta * stock_x - intercept

            # Calculate rolling z-score (using specified window)
            rolling_mean = spread.rolling(window=zscore_window).mean()
            rolling_std = spread.rolling(window=zscore_window).std()
            rolling_zscore = (spread - rolling_mean) / rolling_std

            # Current z-score (last value)
            current_zscore = rolling_zscore.iloc[-1] if not rolling_zscore.empty else 0

            # Calculate full-period z-score for comparison
            full_zscore = (spread - spread.mean()) / spread.std()

            halflife = PairScannerUtils.calculate_half_life(spread)

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
                'rolling_zscore_series': rolling_zscore
            }
        except Exception as e:
            print(f"Error processing pair {stock_x}-{stock_y}: {e}")  # Add this line
            return {
                'hedge_ratio': 1.0,
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
                'spread_series': None,
                'rolling_zscore_series': None
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


if __name__ == "__main__":
    tickers = PairScannerUtils.find_all_sp500_tickers()