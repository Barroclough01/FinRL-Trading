# RL Offline Tracking - Setup Status

## Current Status: ✓ Infrastructure Ready, ⚠ Weights Update Needed

### What's Working

✅ **Offline tracking infrastructure**
- `track_rl_offline.py` - Portfolio simulation engine
- `backfill_rl_history.py` - Historical backfill helper
- `test_rl_offline.py` - Validation (all tests passed)
- Auto-integration in `run_paper_trading.py`

✅ **Data format support**
- Long format CSV (trade_date, gvkey, weights) ✓
- Wide format CSV (date, AAPL, MSFT, ...) ✓

✅ **Database integration**
- SQLite schema created
- Dashboard generation working
- Metrics comparison ready

### Issue: Date Mismatch

**Problem:**
- Live paper trading dates: `2026-05-10` to `2026-06-12` (5 snapshots)
- RL weights available: `2018-12-03` to `2020-11-30` (444 dates)
- **No overlap** → Cannot compare RL vs live strategies

**Current RL weights:**
```
File: results/drl_weight.csv
Format: Long (trade_date, gvkey, weights)
Dates: 444 trading days
Date range: 2018-12-03 to 2020-11-30
Symbols: 151 unique tickers
Latest weight date: 2020-11-30
```

**Live trading dates:**
```
2026-05-10
2026-05-15
2026-05-22
2026-05-29
2026-06-12
```

### Solutions

#### Option 1: Backfill 2020 Historical Data (Quick Test)

Test the offline tracking system with existing weights:

```bash
# Backfill 2018-2020 RL performance
python track_rl_offline.py --backfill

# Generate dashboard (will show RL for 2018-2020, separate from live 2026 data)
python track_metrics.py --report-only
```

This will populate the database with RL snapshots for 2018-2020, but they won't appear in the same comparison charts as your 2026 live data.

**Use case:** Validate that the offline tracking system works correctly before regenerating weights.

#### Option 2: Regenerate RL Weights for 2025-2026 (Full Integration)

Re-run your RL training pipeline to generate weights for recent dates:

```bash
# Step 1: Re-train RL model for 2025-2026
python src/strategies/run_rl_offline_pipeline.py \
    --start-date 2025-01-01 \
    --end-date 2026-06-30

# Step 2: Verify new weights exist
head results/drl_weight.csv
tail results/drl_weight.csv

# Step 3: Backfill RL tracking for 2025-2026
python track_rl_offline.py --backfill

# Step 4: View integrated dashboard
python track_metrics.py --report-only
open logs/dashboard.html
```

**Result:** RL strategy appears alongside FinRL and AR in the same charts, metrics, and comparison tables.

### Recommended Workflow

**Phase 1: Validate (5 minutes)**
```bash
# Test with existing 2020 data to ensure system works
python track_rl_offline.py --date 2020-11-30
python track_metrics.py --report-only
```

**Phase 2: Generate New Weights (depends on your RL pipeline)**
```bash
# Re-run RL training for 2025-2026
# (Use your existing RL training script/pipeline)
```

**Phase 3: Backfill and Compare (5 minutes)**
```bash
# Once new weights exist for 2025-2026
python backfill_rl_history.py
open logs/dashboard.html
cat logs/comparison_metrics_latest.csv
```

### Files to Update

If you regenerate RL weights, ensure the CSV format is one of:

**Long format (current):**
```csv
,trade_date,gvkey,weights
0,2026-05-10 00:00:00,AAPL,0.05
1,2026-05-10 00:00:00,MSFT,0.05
```

**Wide format (alternative):**
```csv
date,AAPL,MSFT,GOOGL,...
2026-05-10,0.05,0.05,0.03,...
```

The loader automatically detects and handles both formats.

### Next Steps

1. **Decide:** Historical backfill (2020) or regenerate weights (2026)?

2. **If historical backfill:**
   ```bash
   python track_rl_offline.py --backfill
   ```

3. **If regenerating weights:**
   - Re-run RL training for 2025-2026
   - Verify `results/drl_weight.csv` has recent dates
   - Run `python test_rl_offline.py` to validate
   - Run `python backfill_rl_history.py` to populate

4. **Weekly automation (after weights are updated):**
   ```bash
   python run_paper_trading.py  # Auto-runs RL tracking
   ```

### Contact / Support

If you need help:
1. Check `docs/offline_rl_tracking.md` for technical details
2. Check `docs/rl_quickstart.md` for step-by-step guide
3. Run `python test_rl_offline.py` to diagnose issues

---

**Status as of:** 2026-06-12  
**RL weights date range:** 2018-12-03 to 2020-11-30  
**Live trading date range:** 2026-05-10 to 2026-06-12  
**Action required:** Regenerate RL weights for 2025-2026 to enable live comparison
