# Code Style

## General

- Python 3.8+ compatible code only.
- Use class-based structure: one class per module, matching the existing pattern (`PairsScanner`, `BacktestUtils`, etc.).
- Keep statistical logic in `utils.py` files; orchestration logic in `index.py` files.
- Do not add type annotations unless the existing file already uses them.
- Do not add docstrings to functions that didn't have them before.

## Naming

- Classes: `PascalCase`
- Methods and variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- DataFrames: suffix with `_df` (e.g., `prices_df`, `pairs_df`)
- Statistical results: use the full term (e.g., `adf_pvalue`, `hurst_exponent`, `half_life`)

## Data Handling

- Always validate data length: minimum 132 trading days (~6 months) before processing.
- Use `yfinance` for all market data — do not introduce other data sources without discussion.
- Store outputs in `__reports__/` with timestamped filenames.
- Use `CommonUtils.save_to_xlsx()` and `CommonUtils.read_xlsx()` for file I/O — do not write custom Excel code.

## Performance

- Include `gc.collect()` after processing each sector — memory spikes without it.
- Avoid storing entire price histories in memory for all sectors simultaneously.
- Use vectorized pandas/numpy operations; avoid Python-level loops over rows.

## Error Handling

- Use `try/except` around yfinance calls — API can be flaky.
- Print informative messages when a pair is skipped and why (filter that failed).
- Do not silently swallow exceptions.
