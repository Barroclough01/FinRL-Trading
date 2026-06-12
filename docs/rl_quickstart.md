# RL Offline Tracking - Quick Start Guide

## Overview

Since you can only create two Alpaca paper trading accounts, the RL strategy is tracked **offline** through simulation while FinRL and AR run live. All three strategies appear side-by-side in the same dashboard and metrics.

## Setup Summary

### What Was Added

1. **`track_rl_offline.py`** - Portfolio simulation engine
   - Reads RL weights from `results/drl_weight.csv`
   - Simulates rebalancing with realistic transaction costs
   - Injects synthetic snapshots into metrics DB

2. **`backfill_rl_history.py`** - One-time historical backfill
   - Simulates all past weeks
   - Populates complete RL performance history

3. **`test_rl_offline.py`** - Validation script
   - Checks setup and data availability
   - Verifies simulation works correctly

4. **Auto-integration** - Modified `run_paper_trading.py`
   - Automatically runs RL offline tracking after live metrics
   - No manual steps needed for weekly runs

5. **Documentation** - `docs/offline_rl_tracking.md`
   - Complete technical documentation
   - Architecture, simulation details, limitations

### Configuration Changes

**`.env` file:**
- Removed `RL` from `APCA_ACCOUNTS` (only `FinRL,AR` remain live)
- Removed placeholder `APCA_RL_*` credentials (not needed)

## Quick Start

### 1. Verify Setup

```bash
python test_rl_offline.py
```

This checks:
- RL weights CSV exists
- Price data is available
- Database is initialized
- Simulation runs without errors

### 2. Backfill Historical Performance

```bash
python backfill_rl_history.py
```

This will:
- Simulate RL portfolio for all past weeks
- Match the same dates as live FinRL and AR accounts
- Inject synthetic snapshots into the DB
- Regenerate dashboard with all 3 strategies

**Expected output:**
```
[1/2] Running RL offline backfill...
RL snapshot: 2025-10-24 | value=$1,023,456.78 | weekly=+2.35% | cum=+2.35%
RL snapshot: 2025-10-31 | value=$1,034,567.89 | weekly=+1.08% | cum=+3.46%
...
RL backfill complete: 15 snapshots

[2/2] Regenerating dashboard with RL history...
Dashboard written to logs/dashboard.html
Saved comparison metrics CSV to logs/comparison_metrics_latest.csv

✓ RL historical backfill complete
```

### 3. View Results

**Dashboard (HTML):**
```bash
open logs/dashboard.html
```

Shows:
- Summary cards for all 3 accounts (FinRL, AR, RL)
- Cumulative return chart (all vs SPY)
- Weekly return chart
- Current positions tables

**Metrics (CSV):**
```bash
cat logs/comparison_metrics_latest.csv
```

Shows side-by-side comparison:
- Weekly/cumulative returns
- Sharpe ratio, volatility, max drawdown
- Beta to SPY/QQQ
- Turnover, weight drift, hit rate
- Weeks in fallback

### 4. Weekly Paper Trading

Normal weekly execution now handles everything:

```bash
python run_paper_trading.py --date 2026-06-13
```

This automatically:
1. Executes live trades for FinRL account
2. Executes live trades for AR account
3. Records live metrics to DB
4. **Simulates RL portfolio for the same date**
5. Regenerates dashboard with all 3 strategies

## How It Works

### Data Flow

```
Weekly Paper Trading Run
    ↓
├─ FinRL Account → Alpaca (live)
├─ AR Account    → Alpaca (live)
    ↓
track_metrics.py
    ↓
├─ Fetch live positions from Alpaca
├─ Calculate returns, metrics
├─ Record to weekly_snapshot table
    ↓
track_rl_offline.py (auto-triggered)
    ↓
├─ Read latest RL weights from results/drl_weight.csv
├─ Fetch historical prices from data/fmp_daily/*.csv
├─ Simulate portfolio rebalancing
│   ├─ Apply transaction costs (5 bps)
│   ├─ Apply slippage (2 bps)
│   └─ Update cash and positions
├─ Record to same weekly_snapshot table (account='RL')
    ↓
Dashboard Generation
    ↓
├─ Load all 3 accounts from DB
├─ Calculate comparison metrics
├─ Generate HTML dashboard
└─ Export CSV metrics
```

### Cost Model

The RL simulation uses **conservative** assumptions:

- **Transaction cost**: 5 bps (0.05%) per trade
- **Slippage**: 2 bps (0.02%) per execution
- **Total round-trip cost**: ~14 bps

This is more realistic than live paper trading (which has ~0 costs) and ensures fair comparison.

## Maintenance

### Daily/Weekly Runs

Just use the normal paper trading command:

```bash
python run_paper_trading.py
```

RL tracking happens automatically.

### Regenerating RL Weights

If you retrain the RL model and update `results/drl_weight.csv`:

```bash
# Backfill from scratch
python track_rl_offline.py --backfill

# Or update specific date
python track_rl_offline.py --date 2026-06-13

# Regenerate dashboard
python track_metrics.py --report-only
```

### Troubleshooting

**Problem**: RL snapshots not appearing in dashboard

```bash
# Check if RL snapshots exist in DB
sqlite3 data/finrl_trading.db \
  "SELECT snapshot_date, portfolio_value FROM weekly_snapshot WHERE account='RL' ORDER BY snapshot_date DESC LIMIT 5;"
```

**Problem**: "No RL weights available"

```bash
# Check weights file
head results/drl_weight.csv

# Verify date coverage
python -c "import pandas as pd; df=pd.read_csv('results/drl_weight.csv'); print(f'First: {df.date.min()}, Last: {df.date.max()}')"
```

**Problem**: "Price data not found"

```bash
# Refresh price data for missing symbols
python refresh_fmp_daily.py --symbols AAPL MSFT GOOGL ...
```

## Files Reference

| File | Purpose |
|:-----|:--------|
| `track_rl_offline.py` | Portfolio simulation engine |
| `backfill_rl_history.py` | Historical backfill helper |
| `test_rl_offline.py` | Setup validation |
| `run_paper_trading.py` | Main weekly execution (auto-triggers RL) |
| `track_metrics.py` | Dashboard generation |
| `results/drl_weight.csv` | RL target weights (input) |
| `data/finrl_trading.db` | Metrics database (output) |
| `logs/dashboard.html` | Visual dashboard (output) |
| `logs/comparison_metrics_latest.csv` | Metrics export (output) |
| `docs/offline_rl_tracking.md` | Detailed documentation |

## Next Steps

1. **Verify setup**: `python test_rl_offline.py`
2. **Backfill history**: `python backfill_rl_history.py`
3. **View dashboard**: `open logs/dashboard.html`
4. **Check metrics**: `cat logs/comparison_metrics_latest.csv`

That's it! RL is now fully integrated into your paper trading comparison system, even without a third Alpaca account.
