# Offline RL Tracking

The `RL` line in the weekly comparison is a local portfolio simulation. It does
not own an Alpaca account and does not submit broker orders.

## Inputs and outputs

| Artifact | Purpose | Lifecycle |
| --- | --- | --- |
| `results/drl_weight.csv` | Target weights produced by offline research | Generated, ignored |
| `data/fmp_daily/*_daily.csv` | Adjusted daily OHLCV used for valuation | Runtime cache, ignored |
| `src/strategies/rl_inactive_symbols.json` | Verified inactive-symbol policy | Durable, tracked |
| `data/finrl_trading.db` | RL snapshots, weights, and shared metrics | Runtime authority, ignored |
| `logs/dashboard.html` | Human-readable comparison | Generated, ignored |
| `logs/comparison_metrics_latest.csv` | Latest tabular comparison | Generated, ignored |

`track_rl_offline.py` accepts both weight formats:

- long: `trade_date,gvkey,weights`
- wide: `date,<ticker>,<ticker>,...`

For a snapshot date it chooses the latest target-weight row on or before that
date. Returns are computed from the prior chronological RL snapshot. Shared
performance comparisons must use dates present for both paper accounts and RL.

## Simulation contract

The simulator starts with $1,000,000 cash and carries positions and cash across
weekly snapshots. At each snapshot it:

1. loads the latest eligible target weights;
2. applies verified inactive-symbol policy entries that are effective by the
   snapshot date;
3. values active holdings and targets using the expected NYSE session close;
4. sells reductions before buying increases;
5. applies 5 bps transaction cost and 2 bps adverse slippage per trade;
6. caps purchases at available cash;
7. records target and actual weights, positions, cash, return, turnover, and
   cost data in the shared SQLite database;
8. regenerates the dashboard and comparison exports.

The implementation uses floating quantities. It assumes complete close-price
execution after costs. Cash earns no interest.

## Freshness and inactive symbols

An active holding or positive target must have a cached close for the expected
NYSE session. A missing or terminally stale active cache raises an actionable
error. A temporary historical gap followed by newer cached rows is not terminal
staleness and must not invalidate older snapshots retroactively.

Only independently verified inactive/non-tradable symbols belong in
`src/strategies/rl_inactive_symbols.json`. The policy may liquidate an existing
holding at its last cached close on or after the configured effective date. It
retains proceeds as cash and preserves the source target with zero actual weight
for drift reporting. The effective date is an operational policy date; it must
not be described as a corporate-action date unless a primary source establishes
that fact.

## Weekly integration

After a non-dry paper run, `run_paper_trading.py` calls:

```bash
finrl-env/bin/python track_rl_offline.py --date YYYY-MM-DD
```

The RL subprocess is currently warning-level in the parent orchestrator. A
failure is logged but does not by itself force the parent command to exit
nonzero. Confirm that the expected RL snapshot exists before calling the
three-way comparison current.

`run_paper_trading.py --dry-run` does not invoke offline RL tracking.

## Non-ordering inspection

These commands do not submit orders or record broker/database snapshots:

```bash
finrl-env/bin/python track_rl_offline.py --help
finrl-env/bin/python track_metrics.py --report-only --date YYYY-MM-DD
finrl-env/bin/python sync_rl_price_data.py --dry-run
```

Example database check:

```bash
finrl-env/bin/python - <<'PY'
import sqlite3

with sqlite3.connect('data/finrl_trading.db') as conn:
    print(conn.execute('PRAGMA integrity_check').fetchone())
    print(conn.execute(
        "SELECT account, MAX(snapshot_date), COUNT(*) "
        "FROM weekly_snapshot GROUP BY account ORDER BY account"
    ).fetchall())
PY
```

`track_metrics.py --report-only` does rewrite generated dashboard/metric files.

## Rebuild workflow

Rebuilding is a state-changing maintenance operation. It can replace simulated
history and derived results.

1. Stop if broker/order state is unresolved and the intended comparison dates
   are unclear.
2. Copy `data/finrl_trading.db` to a dated backup and verify the backup exists.
3. Verify the weight file's format, date range, symbol normalization, and the
   shared FinRL/AR comparison calendar.
4. Verify terminal cache dates for active RL symbols. Investigate stale symbols
   before changing the inactive policy.
5. Run the focused RL tests.
6. Run the interactive helper:

   ```bash
   finrl-env/bin/python backfill_rl_history.py
   ```

7. Verify database integrity, per-account date ranges, shared dates, generated
   metrics, and current parity artifacts.

Do not add RL-only dates to make a chart look complete. Do not overwrite a
known-good backup during a repair.

## Research and promotion boundary

`src/strategies/run_rl_offline_pipeline.py` trains/evaluates models and writes an
offline gate report. Its contract and thresholds live in
`rl_contract.json` and `rl_acceptance_gate.json`. A passing offline gate is a
research result. It does not authorize adding RL to `APCA_ACCOUNTS`, assigning
credentials, submitting paper orders, or using real capital.

## Interpretation limits

The offline line differs from Alpaca paper accounts because it models fixed
costs and slippage but assumes complete close-price fills, no rejections, no
partial fills, no extra market impact, and no cash yield. It is useful for a
consistent candidate comparison, not proof of executable performance. Always
report the shared observation count and these assumptions with performance.
