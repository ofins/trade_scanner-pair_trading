# /project:scan

Run the pair scanner across S&P 500 sectors to identify cointegrated pair candidates.

## Steps

1. Read `src/scanner/index.py` to understand current configuration (sectors, date range, filters).
2. Ask the user which sectors to scan, or default to all sectors in `src/constants/stock_by_sectors.json`.
3. Confirm the date range to use for the scan.
4. Run `python src/scanner/index.py` and monitor output.
5. Report the number of pairs found per sector and the output file path in `__reports__/`.
6. Summarize the top 10 pairs by cointegration p-value with their key metrics (half-life, Hurst, correlation).

## Notes

- Walk-forward split is applied automatically (first 50% train, full dataset for validation).
- Results are saved as a timestamped Excel file in `__reports__/`.
- If no pairs are found for a sector, that is expected — most sectors produce 0–5 candidates.
