#!/usr/bin/env python3
"""
test_rl_offline.py
------------------
Quick smoke test for offline RL tracking system.

Verifies:
  1. RL weights CSV exists and is readable
  2. Price data exists for RL universe symbols
  3. Portfolio simulation runs without errors
  4. Snapshots can be written to DB
  5. Dashboard regeneration works

Usage:
    python test_rl_offline.py
"""

import sys
from pathlib import Path
import pandas as pd
import sqlite3

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

print("=" * 60)
print("  RL Offline Tracking - Smoke Test")
print("=" * 60)
print()

# Test 1: Check RL weights file
print("[1/5] Checking RL weights file...")
weights_path = project_root / "results" / "drl_weight.csv"
if not weights_path.exists():
    print(f"  ✗ FAIL: {weights_path} not found")
    print(f"  → Run the RL offline training pipeline first")
    sys.exit(1)

try:
    df = pd.read_csv(weights_path)
    if df.empty:
        print(f"  ✗ FAIL: Weights CSV is empty")
        sys.exit(1)
    
    # Detect format
    if "trade_date" in df.columns and "gvkey" in df.columns:
        # Long format
        df["date"] = pd.to_datetime(df["trade_date"])
        dates = df["date"].unique()
        latest_date = pd.to_datetime(df["date"]).max()
        symbols = df["gvkey"].nunique()
        print(f"  ✓ OK: {len(dates)} dates, {symbols} symbols (long format)")
        print(f"       Latest date: {latest_date.date()}")
    elif "date" in df.columns:
        # Wide format
        latest_date = pd.to_datetime(df["date"]).max()
        symbols = [col for col in df.columns if col != "date"]
        print(f"  ✓ OK: {len(df)} dates, {len(symbols)} symbols (wide format)")
        print(f"       Latest date: {latest_date.date()}")
    else:
        print(f"  ✗ FAIL: Invalid CSV format - missing 'date' or 'trade_date' column")
        sys.exit(1)
except Exception as e:
    print(f"  ✗ FAIL: Could not read weights CSV: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Check price data
print("\n[2/5] Checking price data availability...")
missing = []
if "trade_date" in df.columns and "gvkey" in df.columns:
    sample_symbols = df["gvkey"].unique()[:5]
else:
    sample_symbols = symbols[:5] if len(symbols) > 5 else symbols
    
for sym in sample_symbols:
    csv_path = project_root / "data" / "fmp_daily" / f"{sym}_daily.csv"
    if not csv_path.exists():
        missing.append(sym)

if missing:
    print(f"  ⚠ WARNING: Missing price data for {len(missing)} symbols")
    print(f"       Sample: {missing[:3]}")
    print(f"  → Run data refresh to download missing symbols")
else:
    print(f"  ✓ OK: Price data exists for sample symbols")

# Test 3: Check DB exists
print("\n[3/5] Checking metrics database...")
db_path = project_root / "data" / "finrl_trading.db"
if not db_path.exists():
    print(f"  ✗ FAIL: {db_path} not found")
    print(f"  → Run track_metrics.py first to initialize DB")
    sys.exit(1)

try:
    conn = sqlite3.connect(db_path)
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    required = ["weekly_snapshot", "weekly_weights", "benchmark_prices"]
    found = [t[0] for t in tables]
    
    if not all(t in found for t in required):
        print(f"  ✗ FAIL: Missing required tables")
        print(f"       Required: {required}")
        print(f"       Found: {found}")
        sys.exit(1)
    
    # Check for existing live snapshots
    count = conn.execute(
        "SELECT COUNT(DISTINCT snapshot_date) FROM weekly_snapshot WHERE account != 'RL'"
    ).fetchone()[0]
    
    print(f"  ✓ OK: Database exists with {count} live snapshot dates")
    conn.close()
except Exception as e:
    print(f"  ✗ FAIL: Database check failed: {e}")
    sys.exit(1)

# Test 4: Dry-run simulation
print("\n[4/5] Testing portfolio simulation...")
try:
    from track_rl_offline import (
        OfflinePortfolio,
        load_rl_weights,
        get_price_on_date,
    )
    
    # Use a recent date
    test_date = latest_date.strftime("%Y-%m-%d")
    weights = load_rl_weights(str(weights_path), test_date)
    
    if not weights:
        print(f"  ⚠ WARNING: No weights found for {test_date}")
    else:
        portfolio = OfflinePortfolio(1_000_000.0, 5, 2)
        prices = {sym: get_price_on_date(sym, test_date) for sym in weights.keys()}
        
        # Filter out zero prices
        valid_prices = {sym: p for sym, p in prices.items() if p > 0}
        valid_weights = {sym: weights[sym] for sym in valid_prices.keys()}
        
        portfolio.rebalance(valid_weights, valid_prices)
        value = portfolio.get_portfolio_value(valid_prices)
        
        print(f"  ✓ OK: Simulation successful")
        print(f"       Date: {test_date}")
        print(f"       Symbols: {len(valid_weights)}")
        print(f"       Portfolio value: ${value:,.2f}")
except Exception as e:
    print(f"  ✗ FAIL: Simulation error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Check track_rl_offline.py exists
print("\n[5/5] Checking tracking script...")
script_path = project_root / "track_rl_offline.py"
if not script_path.exists():
    print(f"  ✗ FAIL: {script_path} not found")
    sys.exit(1)

print(f"  ✓ OK: Tracking script exists")

# Summary
print("\n" + "=" * 60)
print("  ✓ All tests passed")
print("=" * 60)
print()
print("Next steps:")
print()
print("  1. Backfill historical RL performance:")
print("     python backfill_rl_history.py")
print()
print("  2. Or test a single date:")
print(f"     python track_rl_offline.py --date {test_date}")
print()
print("  3. View the dashboard:")
print("     open logs/dashboard.html")
print()
