#!/usr/bin/env python3
"""Fetch SP500 fundamental data and store to local database."""

import argparse
import os
import sys

# Allow running as standalone script: python3 src/data/fetch_and_store_fundamentals.py
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, "src"))

import pandas as pd
from data.data_fetcher import fetch_fundamental_data, fetch_sp500_tickers
from data.data_store import get_data_store


def main():
    parser = argparse.ArgumentParser(description="Fetch & store SP500 fundamentals")
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2026-04-01")
    parser.add_argument("--limit", type=int, default=10000, help="Universe cap")
    parser.add_argument("--preferred-source", default="FMP")
    parser.add_argument("--output-csv", default=None, help="Also save to CSV")
    args = parser.parse_args()

    # 1. Get SP500 tickers
    print("Fetching SP500 universe ...")
    tickers = fetch_sp500_tickers(preferred_source=args.preferred_source)
    if tickers is None or len(tickers) == 0:
        raise ValueError("Failed to fetch SP500 tickers")
    if args.limit > 0:
        tickers = tickers.head(args.limit)
    print(f"Universe: {len(tickers)} tickers")

    # 2. Fetch fundamental data
    print(f"Fetching fundamentals {args.start_date} ~ {args.end_date} ...")
    df = fetch_fundamental_data(tickers, args.start_date, args.end_date,
                                preferred_source=args.preferred_source)
    print(f"Fetched {len(df)} fundamental records")

    if df.empty:
        print("No data fetched, exiting.")
        return

    # 3. Store to database
    store = get_data_store()
    n = store.save_fundamental_data(df)
    print(f"Saved {n} records to {store.db_path}")

    # 4. Optionally save CSV
    if args.output_csv:
        out = args.output_csv
        if not os.path.isabs(out):
            out = os.path.join(project_root, out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        df.to_csv(out, index=False)
        print(f"CSV saved to {out}")

    # 5. Summary
    print(f"\nSummary:")
    print(f"  Tickers: {df['tic'].nunique()}")
    print(f"  Date range: {df['datadate'].min()} ~ {df['datadate'].max()}")
    print(f"  Total records: {len(df)}")
    print(f"  Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
