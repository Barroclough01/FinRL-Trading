#!/usr/bin/env python3
"""
refresh_fmp_daily.py
Weekly refresh of fmp_daily/ OHLCV CSVs for all symbols in the Adaptive Rotation YAML.

- Reads symbol list from AR YAML config (all asset_groups)
- Resolves weekends/holidays to the latest US trading session on or before today
- For each symbol, appends any missing trading days through that session
- Idempotent: safe to run multiple times — will not duplicate rows
- Logs a summary of what was updated / skipped / failed

Usage:
    python refresh_fmp_daily.py [--config PATH] [--dry-run] [--force]

    --config   Path to AR YAML (default: AdaptiveRotationConf_v1.2.2.yaml)
    --dry-run  Show what would be fetched without writing anything
    --force    Explicitly run on a non-trading day (same catch-up target)
"""

import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal
import requests
import yaml
import yfinance as yf
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
DEFAULT_CONFIG = SCRIPT_DIR / "src/strategies/AdaptiveRotationConf_v1.2.2.yaml"
FMP_DAILY_DIR = SCRIPT_DIR / "data/fmp_daily"
OHLCV_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
REQUIRED_BENCHMARK_SYMBOLS = ("SPY", "QQQ")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_api_key() -> str:
    """No longer used — kept for compatibility. yfinance needs no API key."""
    return ""


def load_symbols_from_yaml(config_path: Path) -> list[str]:
    """Extract all ticker symbols from asset_groups in the AR YAML."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    symbols = []
    asset_groups = cfg.get("asset_groups", {})
    for group_name, group_cfg in asset_groups.items():
        tickers = group_cfg.get("symbols", [])
        for t in tickers:
            ticker = str(t).strip().upper()
            if ticker and ticker not in symbols:
                symbols.append(ticker)

    # Also pull benchmark if present
    benchmark = cfg.get("benchmark", {}).get("excess_return_benchmark")
    if benchmark and str(benchmark).upper() not in symbols:
        symbols.append(str(benchmark).upper())

    # Also pull fallback symbols if present
    fallback = cfg.get("portfolio", {}).get("fallback", {}).get("symbols", [])
    for t in fallback:
        ticker = str(t).strip().upper()
        if ticker and ticker not in symbols:
            symbols.append(ticker)

    for ticker in REQUIRED_BENCHMARK_SYMBOLS:
        if ticker not in symbols:
            symbols.append(ticker)

    return sorted(symbols)


def is_trading_day(check_date: date) -> bool:
    """Return True if check_date is a valid NYSE trading day."""
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(
        start_date=check_date.strftime("%Y-%m-%d"),
        end_date=check_date.strftime("%Y-%m-%d"),
    )
    return not schedule.empty


def latest_trading_day_on_or_before(check_date: date) -> date:
    """Return the latest NYSE session date on or before ``check_date``."""
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(
        start_date=(check_date - timedelta(days=10)).strftime("%Y-%m-%d"),
        end_date=check_date.strftime("%Y-%m-%d"),
    )
    if schedule.empty:
        raise ValueError(f"No NYSE trading session found on or before {check_date}")
    return schedule.index[-1].date()


def get_refresh_target_date(check_date: date) -> date:
    """Return the latest NYSE session on or before the supplied date."""
    return latest_trading_day_on_or_before(check_date)


def empty_fetch_is_stale(
    last_date: date | None, expected_latest_date: date
) -> bool:
    """Return whether an empty provider response leaves local history stale."""
    return last_date is None or last_date < expected_latest_date


def stale_data_severity(ticker: str, required_symbols: set[str]) -> str:
    """Make stale live inputs fatal while leaving RL-only inputs as warnings."""
    return "failed" if ticker in required_symbols else "optional_stale"


def get_last_csv_date(csv_path: Path) -> date | None:
    """Return the most recent date in an existing CSV, or None if missing/empty."""
    if not csv_path.exists():
        return None

    # Check if file is completely empty (0 bytes)
    if csv_path.stat().st_size == 0:
        raise ValueError(f"File is empty (0 bytes) at path: {csv_path}")

    try:
        # Try reading only the header first to see if 'date' column is present
        header_df = pd.read_csv(csv_path, nrows=0)
        if "date" not in header_df.columns:
            raise ValueError(
                f"CSV is missing required 'date' column at path: {csv_path}"
            )

        # Pandas accepts these lists, but its stubs exclude ordinary strings.
        df = pd.read_csv(  # ty: ignore[no-matching-overload]
            csv_path,
            usecols=["date"],
            parse_dates=["date"],
        )
        if df.empty:
            raise ValueError(f"CSV is empty (no data rows) at path: {csv_path}")

        max_val = df["date"].max()
        if pd.isnull(max_val):
            raise ValueError(
                f"CSV has null values in 'date' column at path: {csv_path}"
            )

        return max_val.date()
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise exc
        raise ValueError(
            f"Corrupt CSV or unreadable file: {exc} at path: {csv_path}"
        ) from exc


def fetch_fmp_daily(
    ticker: str, from_date: date, to_date: date, api_key: str
) -> pd.DataFrame:
    """
    Fetch EOD OHLCV data from yfinance for a single ticker.
    Returns columns: date, open, high, low, close, volume — sorted ascending.
    """
    raw = yf.download(
        ticker,
        start=from_date.strftime("%Y-%m-%d"),
        end=(to_date + timedelta(days=1)).strftime(
            "%Y-%m-%d"
        ),  # yfinance end is exclusive
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        return pd.DataFrame(
            {column: pd.Series(dtype="object") for column in OHLCV_COLUMNS}
        )

    # Flatten MultiIndex columns if present (yfinance quirk with single ticker)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    raw.index.name = "date"
    raw = raw.reset_index()
    raw["date"] = pd.to_datetime(raw["date"]).dt.date
    df = raw[OHLCV_COLUMNS].sort_values("date").reset_index(drop=True)
    return df


def append_new_rows(csv_path: Path, new_rows: pd.DataFrame, dry_run: bool) -> int:
    """
    Append new_rows to csv_path, deduplicating on date.
    Returns number of rows actually written.
    """
    if new_rows.empty:
        return 0

    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        existing["date"] = pd.to_datetime(existing["date"]).dt.date
        existing_dates = existing["date"]
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        existing_dates = pd.Series([], dtype="object")
        combined = new_rows.copy()

    combined = (
        combined.drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    n_new = len(new_rows[~new_rows["date"].isin(existing_dates)])

    if not dry_run:
        FMP_DAILY_DIR.mkdir(parents=True, exist_ok=True)
        combined.to_csv(csv_path, index=False)

    return n_new


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Refresh fmp_daily/ OHLCV CSVs from FMP API"
    )
    parser.add_argument(
        "--config",
        type=Path,
        action="append",
        dest="configs",
        help=(
            "Path to AR YAML config (repeatable; defaults to configs from "
            "APCA_ACCOUNTS)"
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be fetched, no writes"
    )
    parser.add_argument(
        "--force", action="store_true", help="Skip market calendar check"
    )
    args = parser.parse_args()

    today = date.today()
    target_date = get_refresh_target_date(today)
    mode = " [DRY RUN]" if args.dry_run else ""
    print(f"=== refresh_fmp_daily.py  [{today}]{mode} ===\n")

    # A missed Friday task may start when the computer next becomes available.
    # Catch up through Friday instead of treating a weekend start as a no-op.
    if not is_trading_day(today):
        print(
            f"Today ({today}) is not a NYSE trading day; "
            f"refreshing through {target_date}.\n"
        )
    elif args.force:
        print("--force: refreshing the current trading session\n")

    # --- Load symbols from all configs ---
    # If no --config flags passed, auto-discover from APCA_ACCOUNTS env
    if not args.configs:
        load_dotenv(SCRIPT_DIR / ".env")
        accounts_str = os.getenv("APCA_ACCOUNTS", "").strip()
        if accounts_str:
            discovered = []
            for name in [a.strip() for a in accounts_str.split(",") if a.strip()]:
                cfg_val = os.getenv(f"APCA_{name}_CONFIG", "").strip()
                if cfg_val:
                    discovered.append(SCRIPT_DIR / cfg_val)
            args.configs = discovered if discovered else [DEFAULT_CONFIG]
        else:
            args.configs = [DEFAULT_CONFIG]

    all_symbols = []
    required_symbols = set()
    for cfg_path in args.configs:
        if not cfg_path.exists():
            print(f"ERROR: Config not found: {cfg_path}", file=sys.stderr)
            sys.exit(1)
        syms = load_symbols_from_yaml(cfg_path)
        print(f"Symbols from {cfg_path.name}: {syms}")
        for s in syms:
            required_symbols.add(s)
            if s not in all_symbols:
                all_symbols.append(s)

    # Auto-discover symbols from RL weights file (results/drl_weight.csv) if it exists
    drl_weight_path = SCRIPT_DIR / "results/drl_weight.csv"
    if drl_weight_path.exists():
        try:
            drl_df = pd.read_csv(drl_weight_path)
            rl_symbols = []
            if "gvkey" in drl_df.columns:
                # Long format: gvkey contains tickers
                rl_symbols = drl_df["gvkey"].dropna().unique().tolist()
            elif "date" in drl_df.columns:
                # Wide format: columns except date are tickers
                rl_symbols = [col for col in drl_df.columns if col != "date"]

            print(f"Found {len(rl_symbols)} symbols in DRL weights file.")
            for t in rl_symbols:
                ticker = str(t).strip().upper()
                if ticker and ticker not in all_symbols:
                    all_symbols.append(ticker)
        except Exception as e:
            print(f"WARNING: Could not parse RL symbols from {drl_weight_path}: {e}")

    symbols = sorted(all_symbols)
    print("\nTotal unique symbols across all configs and DRL weights: ")
    print(f"{len(symbols)} symbols\n")

    api_key = load_api_key()

    # --- Refresh loop ---
    results = {
        "updated": [],
        "up_to_date": [],
        "failed": [],
        "optional_stale": [],
        "new_file": [],
    }

    for ticker in symbols:
        csv_path = FMP_DAILY_DIR / f"{ticker}_daily.csv"
        try:
            last_date = get_last_csv_date(csv_path)
        except Exception as e:
            print(f"ERROR: Failed to read existing data for {ticker} - {e}")
            results["failed"].append((ticker, f"Unreadable existing CSV: {e}"))
            continue

        if last_date is None:
            # New ticker — fetch full history (5 years back)
            from_date = target_date - timedelta(days=5 * 365)
            status_prefix = "[NEW]"
        elif last_date >= target_date:
            print(f"  {ticker:6s} — already up to date (last: {last_date})")
            results["up_to_date"].append(ticker)
            continue
        else:
            from_date = last_date + timedelta(days=1)
            status_prefix = "[UPD]"

        print(
            f"  {ticker:6s} — fetching {from_date} → {target_date} ...",
            end=" ",
            flush=True,
        )

        try:
            new_rows = fetch_fmp_daily(ticker, from_date, target_date, api_key)

            if new_rows.empty:
                if empty_fetch_is_stale(last_date, target_date):
                    message = (
                        "No data returned while local history is stale "
                        f"(last: {last_date or 'missing'}, expected: "
                        f"{target_date})"
                    )
                    severity = stale_data_severity(ticker, required_symbols)
                    if severity == "failed":
                        print(f"FAILED: {message}")
                        results["failed"].append((ticker, message))
                    else:
                        print(f"WARNING (RL-only): {message}")
                        results["optional_stale"].append((ticker, message))
                else:
                    print("no data returned; local history is current")
                    results["up_to_date"].append(ticker)
                continue

            # Deduplicate and write
            if csv_path.exists():
                existing = pd.read_csv(csv_path)
                existing_dates = set(pd.to_datetime(existing["date"]).dt.date)
            else:
                existing_dates = set()

            truly_new = new_rows[~new_rows["date"].isin(existing_dates)]
            n_new = len(truly_new)

            if n_new == 0:
                print("no new rows after dedup")
                results["up_to_date"].append(ticker)
                continue

            if not args.dry_run:
                FMP_DAILY_DIR.mkdir(parents=True, exist_ok=True)
                if csv_path.exists():
                    existing_df = pd.read_csv(csv_path, parse_dates=["date"])
                    existing_df["date"] = pd.to_datetime(existing_df["date"]).dt.date
                    combined = pd.concat([existing_df, truly_new], ignore_index=True)
                else:
                    combined = truly_new.copy()
                combined = (
                    combined.drop_duplicates(subset=["date"])
                    .sort_values("date")
                    .reset_index(drop=True)
                )
                combined.to_csv(csv_path, index=False)

            dry_run_suffix = " [dry run]" if args.dry_run else ""
            last_new_date = new_rows["date"].max()
            print(
                f"{status_prefix} +{n_new} rows (last: {last_new_date}){dry_run_suffix}"
            )

            if last_date is None:
                results["new_file"].append(ticker)
            else:
                results["updated"].append(ticker)

        except requests.HTTPError as e:
            print(f"HTTP ERROR {e.response.status_code}")
            results["failed"].append((ticker, str(e)))
        except Exception as e:
            print(f"ERROR: {e}")
            results["failed"].append((ticker, str(e)))

    # --- Summary ---
    print(f"\n{'=' * 50}")
    print(f"Summary through {target_date}:")
    print(f"  New files created : {len(results['new_file'])}  {results['new_file']}")
    print(f"  Updated           : {len(results['updated'])}  {results['updated']}")
    print(f"  Already up to date: {len(results['up_to_date'])}")
    print(f"  RL-only stale     : {len(results['optional_stale'])}")
    for ticker, err in results["optional_stale"]:
        print(f"    {ticker}: {err}")
    print(f"  Failed            : {len(results['failed'])}")
    for ticker, err in results["failed"]:
        print(f"    {ticker}: {err}")

    if results["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
