import pandas as pd
import json
import requests
from io import StringIO

def get_sp500_sector_json() -> dict[str, list[str]]:
    # Fetch current S&P 500 components from Wikipedia
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    html = StringIO(resp.text)
    sp500 = pd.read_html(html)[0]

    # The table has columns: “Symbol”, “Security”, “GICS Sector”, etc.
    # Group symbols by their “GICS Sector”
    grouped = sp500.groupby("GICS Sector")["Symbol"].apply(list).to_dict()

    # Optionally reorder or rename sectors to match your domain names
    # Example: rename “Information Technology” → “Technology”
    mapping = {
        "Information Technology": "Technology",
        "Health Care": "Healthcare",
        "Consumer Discretionary": "Consumer",
        # add more remaps here as needed
    }
    result = {}
    for sector, symbols in grouped.items():
        key = mapping.get(sector, sector)
        result[key] = sorted(symbols)

    return result

if __name__ == "__main__":
    sector_json = get_sp500_sector_json()
    # Pretty print / save
    print(json.dumps(sector_json, indent=2))
    # Optionally, save to file
    with open("sp500_by_sector.json", "w") as f:
        json.dump(sector_json, f, indent=2)
