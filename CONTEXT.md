# Local Operating Context

This file describes the user-specific FinRL-Trading workflow in this checkout.
The root `README.md` describes the upstream FinRL-X project. Start here for
local maintenance and use `docs/README.md` to find deeper references.

Last documentation verification: 2026-09-07. This date records when paths and
behavior were checked; it is not a claim that the scheduler, broker, market
data, or generated metrics are currently healthy.

## Purpose

The project is collecting comparable weekly evidence for three strategies:

| Label | Strategy | Execution boundary |
| --- | --- | --- |
| `FinRL` | ML-enhanced Adaptive Rotation | Alpaca paper account |
| `AR` | Baseline Adaptive Rotation | Alpaca paper account |
| `RL` | DRL-produced target weights | Local offline simulation only |

The current goal is reliable evidence collection. The project is not ready for
real capital, and the small number of weekly observations does not support a
strategy winner or a promotion decision.

## Checkout and environment

- Repository: `/home/paxto/stock-trading/FinRL-Trading` in Ubuntu WSL
- Python environment: `finrl-env`
- Git: run Git inside WSL
- Secrets: `.env` (ignored; never print or commit it)
- Paper account list: `APCA_ACCOUNTS=FinRL,AR`
- Per-account strategy paths: `APCA_<NAME>_CONFIG`
- Closed-market policy: `MARKET_CLOSED_ACTION=next_open`

Use the environment explicitly:

```bash
cd /home/paxto/stock-trading/FinRL-Trading
finrl-env/bin/python run_paper_trading.py --help
```

## Weekly workflow

The registered Windows task is `FinRL Paper Trading`. It is scheduled for
Friday at 18:15 local time, wakes the computer, and starts when available. Its
wrapper is outside the repository:

```text
C:\Users\paxto\stock-trading\run_paper_trading.ps1
```

The wrapper performs two WSL commands:

1. `refresh_fmp_daily.py` refreshes cached OHLCV data through the latest NYSE
   session. Its name is historical; it currently fetches from yfinance.
2. `run_paper_trading.py` generates each account's target weights, validates
   inputs and broker state, submits paper rebalances, captures metrics, runs the
   offline RL simulation, performs sanity/parity checks, and writes artifacts.

If the computer starts the task on a weekend, both commands resolve the run to
Friday's latest NYSE session. Friday-evening `DAY` orders are expected to remain
open until the next regular session. Persisted broker order IDs and later broker
status are the basis for reconciliation; never resubmit merely because an order
was still open after the Friday run.

The wrapper returns the trading exit code first and then a failed refresh exit
code. A successful scheduled-task result therefore means both commands returned
zero. It does not prove that queued orders later filled or that external state
has remained healthy.

## Data and artifact map

| Path | Role | Authority / lifecycle |
| --- | --- | --- |
| `src/strategies/AdaptiveRotationConf_v1.2.2.yaml` | FinRL account strategy config | Durable, tracked |
| `src/strategies/AdaptiveRotationConf_baseline.yaml` | AR account strategy config | Durable, tracked |
| `src/strategies/rl_contract.json` | Offline RL research contract | Durable, tracked |
| `src/strategies/rl_acceptance_gate.json` | Offline evaluation thresholds | Durable, tracked |
| `src/strategies/rl_inactive_symbols.json` | Verified inactive-symbol policy | Durable, tracked |
| `results/drl_weight.csv` | RL target-weight input | Generated, ignored |
| `data/fmp_daily/*_daily.csv` | Local adjusted OHLCV cache | Runtime data, ignored |
| `data/finrl_trading.db` | Snapshot, decision, weight, and price store | Operational authority, ignored |
| `logs/strategy_decisions.jsonl` | Append-only decision mirror | Generated audit evidence, ignored |
| `logs/execution_YYYY-MM-DD.json` | Per-run submitted-order record | Generated audit evidence, ignored |
| `logs/parity_check_YYYY-MM-DD.json` | Replay/execution/database comparison | Generated audit evidence, ignored |
| `logs/comparison_metrics_YYYY-MM-DD.json` | Dated comparison export | Generated summary, ignored |
| `logs/comparison_metrics_latest.csv` | Latest tabular comparison | Generated summary, ignored |
| `logs/dashboard.html` | Human-readable comparison | Generated summary, ignored |
| Windows scheduler logs | Wrapper/task result | Runtime evidence outside repo |

Before rebuilding `data/finrl_trading.db` or RL history, create a dated backup
outside the active database path and verify it exists. Existing large database
backups are operational evidence; do not delete or replace them during routine
maintenance.

## Comparison invariants

- Use only shared chronological dates when comparing FinRL, AR, and RL.
- SPY and QQQ are required benchmarks and refresh inputs.
- Target weights record the strategy decision. Actual weights record observed
  paper positions or simulated holdings. Preserve their union so unfilled or
  skipped assets remain visible.
- Enabled fallback with an empty symbol list represents defensive cash.
- Position-quantity parity is exact within numeric tolerance. Weight differences
  caused by distinct valuation timestamps are informational.
- Active RL holdings and targets require a fresh terminal cached close. A
  verified inactive holding may be liquidated at its last cached close under
  `rl_inactive_symbols.json`; proceeds remain cash and source target weight is
  retained for drift reporting.
- Offline RL assumes close-price execution, no partial fills or rejections, 5
  bps transaction cost per trade, 2 bps slippage per side, and zero cash yield.
  Those assumptions differ from Alpaca paper execution and must be disclosed in
  any performance interpretation.

## Safe orientation

These commands are read-only with respect to broker orders:

```bash
git status --short --branch
finrl-env/bin/python run_paper_trading.py --dry-run --date YYYY-MM-DD
finrl-env/bin/python refresh_fmp_daily.py --dry-run
finrl-env/bin/python sync_rl_price_data.py --dry-run
finrl-env/bin/python track_metrics.py --report-only --date YYYY-MM-DD
finrl-env/bin/python -m pytest
```

`run_paper_trading.py` without `--dry-run` submits paper orders.
`track_rl_offline.py`, `backfill_rl_history.py`, non-report metrics runs, price
syncs, and non-dry-run refreshes write local operational state. The offline RL
pipeline can also train models and replace generated results.

## Current health checks

For a fresh status report, verify rather than repeat this file's date:

1. Windows task configuration, last result, next run, and wrapper contents.
2. Tail of the Windows task log and `logs/paper_trading_cron.log`.
3. Latest SQLite rows by account and database `PRAGMA integrity_check`.
4. Latest execution and parity reports, including unresolved broker orders.
5. Terminal dates for required live and benchmark CSVs; review RL-only stale
   symbols separately from live inputs.
6. Current Alpaca paper account positions and persisted order IDs when broker
   reconciliation is part of the request.
7. Shared-date metrics and observation count before interpreting performance.

Detailed commands and failure semantics are in `docs/operations.md` and
`docs/offline_rl_tracking.md`.
