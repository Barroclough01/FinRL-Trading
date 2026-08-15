# Offline RL Strategy Tracking

This document explains how the RL (Reinforcement Learning) strategy is tracked offline and compared against the live AR (Adaptive Rotation) and FinRL (ML Rolling Selection) strategies.

## Why Offline?

Since we can only create two Alpaca paper trading accounts, the RL strategy runs as an **offline simulation** rather than live trading. This approach:

- Maintains realistic comparison by simulating the same transaction costs and slippage as live accounts
- Enables side-by-side performance tracking in the same dashboard
- Preserves the same weekly snapshot cadence as live strategies
- Uses actual market prices from the same data source

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Live Paper Trading (run_paper_trading.py)                  │
├─────────────────────────────────────────────────────────────┤
│  1. FinRL account → Alpaca execution                        │
│  2. AR account    → Alpaca execution                        │
│  3. track_metrics.py → Record live snapshots to SQLite      │
│  4. track_rl_offline.py → Simulate RL + inject to SQLite   │
│  5. Dashboard regeneration with all 3 strategies            │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. `track_rl_offline.py`

Main simulation engine that:

- Reads RL target weights from `results/drl_weight.csv`
- Fetches historical prices for each symbol
- Simulates portfolio rebalancing with realistic costs:
  - Transaction cost: 0.05% (5 bps) per trade
  - Slippage: 0.02% (2 bps) per execution
- Maintains cash balance and position tracking
- Records weekly snapshots to the same SQLite DB as live accounts

**Usage:**

```bash
# Simulate single date
python track_rl_offline.py --date 2026-06-12

# Backfill all historical dates
python track_rl_offline.py --backfill
```

### 2. `backfill_rl_history.py`

One-time helper to populate all historical RL performance:

```bash
python backfill_rl_history.py
```

This:
1. Identifies all existing weekly snapshot dates from live accounts
2. Simulates the RL portfolio for each historical week
3. Injects synthetic snapshots into the DB
4. Regenerates the dashboard with complete RL history

### 3. Integration with Paper Trading

The `run_paper_trading.py` script automatically runs RL offline tracking after collecting live metrics:

```python
# After live metrics collection
python track_rl_offline.py --date {run_date}
```

## Simulation Details

### Portfolio Initialization

- Starting capital: $1,000,000 (same as Alpaca paper accounts)
- Initial positions: None (100% cash)
- Persistent state: Positions and cash carry forward week-to-week

### Rebalancing Logic

For each weekly snapshot:

1. **Load target weights** from `results/drl_weight.csv` (latest on or before snapshot date)
2. **Fetch prices** for all symbols from `data/fmp_daily/*.csv`
3. **Inactive-symbol phase**: Apply `src/strategies/rl_inactive_symbols.json`
   - Liquidate confirmed inactive holdings at their last cached close
   - Apply the normal transaction-cost and slippage assumptions
   - Retain proceeds as cash rather than reallocating them
   - Preserve source target weights with zero actual weight for drift reporting
4. **Sell phase**: Close or reduce positions not in target
   - Calculate sell value
   - Apply slippage (2 bps against trader)
   - Apply transaction cost (5 bps)
   - Credit net proceeds to cash
5. **Buy phase**: Open or add to target positions
   - Calculate buy value
   - Apply slippage (2 bps against trader)
   - Apply transaction cost (5 bps)
   - Cap to available cash
   - Deduct total cost from cash
6. **Record snapshot** to SQLite with positions, weights, and returns

### Cost Model

- **Transaction cost**: 5 bps (0.05%) of trade value
  - Models broker commissions + SEC fees + exchange fees
- **Slippage**: 2 bps (0.02%) of trade value
  - Models market impact and bid-ask spread
- **Total round-trip cost**: ~14 bps (0.14%)
  - Sell: 5 bps + 2 bps = 7 bps
  - Buy: 5 bps + 2 bps = 7 bps

This is conservative compared to typical retail paper trading which has ~0 explicit costs, making the RL comparison more realistic.

## Data Flow

```
results/drl_weight.csv (RL weights)
          ↓
   [Read latest weights for date]
          ↓
data/fmp_daily/*.csv (price data)
          ↓
   [Simulate rebalancing]
          ↓
data/finrl_trading.db (SQLite)
  - weekly_snapshot (account='RL')
  - weekly_weights (account='RL')
          ↓
   [Dashboard generation]
          ↓
logs/dashboard.html (all 3 strategies)
logs/comparison_metrics_latest.csv
```

## Metrics Comparison

The dashboard and metrics files include all three strategies:

- **FinRL**: Live paper trading (ML rolling selection)
- **AR**: Live paper trading (adaptive rotation)
- **RL**: Offline simulation (DRL portfolio allocation)

All metrics are calculated identically:

- Weekly return
- Cumulative return
- Volatility (annualized)
- Sharpe ratio
- Max drawdown
- Beta to SPY/QQQ
- Tracking error
- Hit rate
- Turnover
- Weight drift

## Limitations

### What's Realistic

- Transaction costs and slippage
- Sequential execution (sell before buy)
- Cash constraint (can't buy more than available)
- Price data from same source as live strategies
- Same snapshot frequency (weekly)

### What's Not Realistic

- **Perfect execution**: Assumes all orders fill at close price
- **No partial fills**: All trades execute fully
- **Explicit inactive-symbol handling only**: Confirmed inactive symbols are
  maintained in `src/strategies/rl_inactive_symbols.json`; new corporate actions
  still require verification and a policy update
- **No market impact beyond slippage**: Large orders don't move price
- **No overnight gaps**: Uses daily close-to-close prices
- **No opportunity cost**: Cash earns 0% (should be ~4-5% money market rate)

These limitations make the RL simulation slightly **pessimistic** (transaction costs are modeled, but not cash drag) and slightly **optimistic** (perfect fills, no rejections).

## Maintenance

### Weekly Paper Trading Run

Automatically handled by `run_paper_trading.py`:

```bash
python run_paper_trading.py --date 2026-06-12
```

This will:
1. Execute live trades for FinRL and AR accounts
2. Record live snapshots to DB
3. Simulate RL portfolio for the same date
4. Regenerate dashboard with all 3 strategies

### Historical Backfill

If you add new historical data or want to recalculate RL performance:

```bash
python backfill_rl_history.py
```

### Manual RL Update

If you regenerate `results/drl_weight.csv` with new RL weights:

```bash
# Backfill from first date in weights file
python track_rl_offline.py --backfill

# Or update single date
python track_rl_offline.py --date 2026-06-12

# Regenerate dashboard
python track_metrics.py --report-only
```

## Validation

To verify RL tracking accuracy:

```bash
# Check RL snapshots exist
sqlite3 data/finrl_trading.db "SELECT snapshot_date, portfolio_value, cumulative_return FROM weekly_snapshot WHERE account='RL' ORDER BY snapshot_date DESC LIMIT 10;"

# Compare total transaction costs
sqlite3 data/finrl_trading.db "SELECT SUM(cash) FROM weekly_snapshot WHERE account='RL';"

# View dashboard
open logs/dashboard.html

# Check metrics CSV
cat logs/comparison_metrics_latest.csv | grep RL
```

## Future Improvements

Possible enhancements for more realistic simulation:

1. **Cash interest**: Model risk-free rate on uninvested cash
2. **Partial fills**: Simulate liquidity constraints for small-cap stocks
3. **Intraday volatility**: Use OHLC data instead of just close
4. **Market impact model**: Scale slippage with position size
5. **Order rejection**: Model scenarios where positions can't be opened/closed
6. **Tax drag**: Simulate capital gains taxes on realized profits

These are not currently implemented to keep the simulation simple and reproducible.
