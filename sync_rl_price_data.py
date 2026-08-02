#!/usr/bin/env python3
"""Sync cached yfinance OHLCV files into the RL SQLite price store."""

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "finrl_trading.db"
DEFAULT_CSV_DIR = PROJECT_ROOT / "data" / "fmp_daily"
REQUIRED_COLUMNS = {"date", "open", "high", "low", "close", "volume"}


def load_new_rows(csv_path: Path, last_date: str | None) -> list[tuple]:
    """Return validated rows newer than the database date for one ticker."""
    frame = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(f"{csv_path} is missing required columns: {joined}")

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    numeric_columns = ["open", "high", "low", "close", "volume"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "close"])
    if last_date:
        frame = frame[frame["date"] > pd.Timestamp(last_date)]

    ticker = csv_path.name.removesuffix("_daily.csv").upper()
    rows = []
    for row in frame.to_dict(orient="records"):
        close = float(row["close"])
        rows.append(
            (
                ticker,
                row["date"].strftime("%Y-%m-%d"),
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                close,
                close,
                float(row["volume"]),
            )
        )
    return rows


def sync_prices(db_path: Path, csv_dir: Path, dry_run: bool) -> tuple[int, int]:
    """Insert newer cached OHLCV data and return updated ticker and row counts."""
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite price store not found: {db_path}")
    csv_paths = sorted(csv_dir.glob("*_daily.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No cached OHLCV files found in: {csv_dir}")

    inserted_rows = 0
    updated_tickers = 0
    with sqlite3.connect(db_path) as conn:
        for csv_path in csv_paths:
            ticker = csv_path.name.removesuffix("_daily.csv").upper()
            last_row = conn.execute(
                "SELECT MAX(date) FROM price_data WHERE ticker = ?", (ticker,)
            ).fetchone()
            last_date = last_row[0] if last_row else None
            rows = load_new_rows(csv_path, last_date)
            if not rows:
                continue
            updated_tickers += 1
            inserted_rows += len(rows)
            if not dry_run:
                conn.executemany(
                    """
                    INSERT INTO price_data
                    (ticker, date, open, high, low, close, adj_close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(ticker, date) DO UPDATE SET
                        open = excluded.open,
                        high = excluded.high,
                        low = excluded.low,
                        close = excluded.close,
                        adj_close = excluded.adj_close,
                        volume = excluded.volume
                    """,
                    rows,
                )

    return updated_tickers, inserted_rows


def main() -> None:
    """Run the cached-price sync and print its coverage summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tickers, rows = sync_prices(args.db, args.csv_dir, args.dry_run)
    action = "Would insert" if args.dry_run else "Inserted"
    print(f"{action} {rows} rows across {tickers} tickers.")

    with sqlite3.connect(args.db) as conn:
        first_date, last_date, ticker_count = conn.execute(
            "SELECT MIN(date), MAX(date), COUNT(DISTINCT ticker) FROM price_data"
        ).fetchone()
    print(
        f"RL price coverage: {first_date} to {last_date} across {ticker_count} tickers."
    )


if __name__ == "__main__":
    main()
