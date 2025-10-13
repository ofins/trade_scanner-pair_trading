
from datetime import datetime
import os
import pandas as pd
import yfinance as yf


class CommonUtils:
    @staticmethod
    def save_to_xlsx(data: list[dict], filename: str, base_folder:str):
        """ Save results to Excel file """
        if not data:
            print("No data to save.")
            return
        
        # Create directory structure for saving Excel files
        today_folder = datetime.now().strftime("%Y-%m-%d")
        save_path = os.path.join(base_folder, today_folder)
        
        # Create directories if they don't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Remove .xlsx extension if already present, then add timestamped version
        base_filename = filename.replace('.xlsx', '')
        timestamped_filename = f'{base_filename}_{datetime.now().strftime("%Y%m%d")}.xlsx'
        full_filepath = os.path.join(save_path, timestamped_filename)

        df = pd.DataFrame(data)
        df.to_excel(full_filepath, index=False)
        print(f"Results saved to {full_filepath}")

    @staticmethod
    def read_xlsx(filepath: str) -> pd.DataFrame:
        """ Read Excel file and return DataFrame """
        try:
            df = pd.read_excel(filepath, index_col=0, parse_dates=True)
            print(f"Loaded data from {filepath} with shape {df.shape}")
            return df
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return pd.DataFrame()
        

    """ Fetching data """

    @staticmethod
    def fetch_data(tickers: list[str], period: str = None, start: str = None, end: str = None, interval: str = "1d") -> pd.DataFrame:
        """
        Fetch historical price data for given tickers

        Args:
            tickers: List of ticker symbols
            period: Period to download (e.g., "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
            start: Start date string (YYYY-MM-DD) - used with end parameter
            end: End date string (YYYY-MM-DD) - used with start parameter
            interval: Data interval (e.g., "1d", "1h", "1m")

        Note: Either period OR (start and end) should be provided, not both
        """
        print(f"  Fetching data for {len(tickers)} tickers...")

        try:
            tickers_str = ' '.join(tickers)
            print(f"Downloading...")

            # Build download parameters based on what's provided
            download_params = {
                'tickers': tickers_str,
                'interval': interval,
                'progress': False,
                'group_by': 'ticker' if len(tickers) > 1 else None,
                'threads': True,
                'repair': True
            }

            # Add either period or start/end
            if start is not None and end is not None:
                download_params['start'] = start
                download_params['end'] = end
            elif period is not None:
                download_params['period'] = period
            else:
                # Default to 1y if nothing specified
                download_params['period'] = '1y'

            raw_data = yf.download(**download_params)

            # Explicitly close any connections from yfinance
            import gc
            gc.collect()

            if raw_data.empty:
                print(" No data downloaded.")
                return pd.DataFrame()
            
            # Extract price data
            data_dict = CommonUtils._extract_prices(raw_data, tickers)

            if not data_dict:
                print("     No valid ticker data extracted.")
                return pd.DataFrame()
            
            # Build and validate timeframe
            df = pd.DataFrame(data_dict).ffill().dropna()

            min_required_points = 132 # ~1/2 year of trading days

            if len(df) < min_required_points:
                print(f"    Insufficient data points: {len(df)} < {min_required_points}")
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
            price_col = CommonUtils._get_price_column(raw_data)
            if price_col is not None:
                data_dict[tickers[0]] = price_col

        else:
            # Multiple tickers
            for ticker in tickers:
                price_col = CommonUtils._get_ticker_price(raw_data, ticker)
                if price_col is not None and len(price_col.dropna()) > 126: 
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
    def _get_ticker_price(data: pd.DataFrame, ticker: str) -> pd.Series | None:
        """Get price column for specific ticker from MultiIndex"""
        for col_name in ['Adj Close', 'Close']:
            if (ticker, col_name) in data.columns:
                return data[(ticker, col_name)]
        return None
