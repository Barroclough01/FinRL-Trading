# Trading Calendar and Data Freshness

The local weekly workflow uses NYSE sessions from `pandas_market_calendars`.
Calendar logic is shared by price refresh, paper-run date resolution, metrics,
and offline RL valuation.

## Run-date rule

`latest_trading_day_on_or_before(date)` returns the most recent NYSE session on
or before the supplied date. Therefore:

- a normal Friday run uses Friday when the exchange was open;
- a Saturday or Sunday catch-up uses Friday;
- an exchange holiday uses the prior NYSE session;
- an explicit `--date` on `run_paper_trading.py` is parsed as supplied, while
  freshness validation resolves the expected session on or before it.

The source of this local behavior is `refresh_fmp_daily.py`. The broader library
also contains `src/data/trading_calendar.py`; do not assume its generic data
fetching behavior overrides the weekly wrapper.

## Refresh rule

`refresh_fmp_daily.py` builds its required symbol set from each configured
Adaptive Rotation YAML, including asset groups, excess-return benchmark, and
fallback symbols. SPY and QQQ are always added.

Despite the script and directory names, the current refresh provider is
yfinance. Downloads use adjusted OHLCV with an exclusive end date, so the
implementation requests one day beyond the target session.

The refresh appends missing rows, deduplicates by date, sorts ascending, and
fails when a required live/benchmark symbol remains stale. RL-only symbols can
be classified as optional stale because they do not affect broker orders; the
offline simulator applies its own active-symbol freshness check.

## Cache contracts

Every `data/fmp_daily/<SYMBOL>_daily.csv` used by this workflow must contain:

```text
date,open,high,low,close,volume
```

Dates must parse cleanly, close values must be usable, and the last row must
represent the terminal cached session. A middle gap does not imply terminal
staleness if later rows resume. Do not silently forward-fill a missing terminal
close.

The Adaptive Rotation loader and offline RL tracker both read these CSVs. The
separate SQLite `price_data` table can be updated from them with
`sync_rl_price_data.py`; it is not automatically identical to the CSV cache.

## Inspection commands

```bash
finrl-env/bin/python refresh_fmp_daily.py --help
finrl-env/bin/python refresh_fmp_daily.py --dry-run
finrl-env/bin/python sync_rl_price_data.py --dry-run
```

`refresh_fmp_daily.py --dry-run` still contacts yfinance but does not write CSV
rows. `sync_rl_price_data.py --dry-run` reads local CSV and SQLite state without
inserting rows.

To inspect terminal CSV dates locally:

```bash
finrl-env/bin/python - <<'PY'
from pathlib import Path
import pandas as pd

for path in sorted(Path('data/fmp_daily').glob('*_daily.csv')):
    dates = pd.read_csv(path, usecols=['date'])['date']
    print(path.stem.removesuffix('_daily'), dates.iloc[-1])
PY
```

## Failure handling

- Missing/corrupt required CSV: stop before paper execution and repair the
  cache from an authoritative provider response.
- Empty provider response with an already-current cache: no new row is needed.
- Empty provider response with a stale required cache: fail the refresh.
- Stale RL-only cache during shared refresh: record the warning, then let the
  offline RL freshness policy determine whether the current RL snapshot can run.
- Verified inactive RL symbol: update `rl_inactive_symbols.json` only after
  confirming inactive/non-tradable status and the last usable close.

Do not convert a temporary provider outage or historical gap into an inactive
symbol policy.
