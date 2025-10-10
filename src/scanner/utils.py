from ast import List
from io import StringIO
import pandas as pd
import requests
import yfinance as yf

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

if __name__ == "__main__":
    tickers = PairScannerUtils.find_all_sp500_tickers()