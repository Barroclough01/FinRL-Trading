# Operations Guide

This guide covers the local weekly paper-comparison workflow. Read
`../CONTEXT.md` first for boundaries and artifact ownership.

## Safety model

`FinRL` and `AR` use Alpaca paper accounts. `RL` is local simulation only.
Nothing in this guide authorizes a real-money order or RL broker activation.

The supported preview of account targets is:

```bash
finrl-env/bin/python run_paper_trading.py --dry-run --date YYYY-MM-DD
```

Dry run generates strategy output but skips live metrics, offline RL tracking,
and order submission. It is a signal and pre-trade preview, not a full rehearsal
of the weekly tail.

## Scheduled path

The Windows task `FinRL Paper Trading` invokes
`C:\Users\paxto\stock-trading\run_paper_trading.ps1`. That untracked wrapper:

1. enters the WSL checkout and activates `finrl-env`;
2. runs `refresh_fmp_daily.py`;
3. runs `run_paper_trading.py`;
4. propagates failures to Task Scheduler.

Inspect the registered task and wrapper before changing scheduler documentation.
They are outside Git and can drift independently.

## Command behavior

| Command | Reads external state | Writes local state | Can submit orders |
| --- | --- | --- | --- |
| `refresh_fmp_daily.py --dry-run` | yfinance | No | No |
| `refresh_fmp_daily.py` | yfinance | OHLCV CSV cache | No |
| `run_paper_trading.py --dry-run` | Strategy inputs | Logs/audit files | No |
| `run_paper_trading.py` | Alpaca paper + market cache | Orders, DB, logs, reports | Paper only |
| `track_metrics.py --report-only` | SQLite/cache | Dashboard and exports | No |
| `track_metrics.py` | Alpaca paper + cache | SQLite, dashboard, exports | No |
| `track_rl_offline.py` | RL weights/cache/SQLite | Simulated SQLite rows, reports | No |
| `sync_rl_price_data.py --dry-run` | CSV cache + SQLite | No | No |
| `sync_rl_price_data.py` | CSV cache + SQLite | SQLite price rows | No |
| `run_rl_offline_pipeline.py` | Research inputs | Models/results/gate report | No |

`backfill_rl_history.py` is interactive and rewrites RL comparison history. Back
up the database first. Do not use it as a routine health-check command.

## Weekly run semantics

When no date is supplied, the refresh and orchestrator resolve to the latest
NYSE session on or before today. A missed Friday task that starts Saturday or
Sunday therefore uses Friday's session.

With `MARKET_CLOSED_ACTION=next_open`, closed-market orders use ordinary `DAY`
time in force and queue for the next regular session. An accepted/open order is
pending, not filled and not failed. Reconcile its persisted broker ID later; do
not blindly resubmit it.

The orchestrator runs each configured account sequentially. On non-dry runs it
then records/regenerates live metrics, invokes offline RL tracking, runs sanity
checks, writes parity information, and optionally sends a webhook. Offline RL
tracking is currently invoked as a warning-level subprocess: its failure is
logged but is not itself appended to the main error list. A status review must
therefore verify the current RL row and logs rather than infer RL success from a
zero orchestrator exit alone.

## Data freshness

`refresh_fmp_daily.py` reads every configured account's Adaptive Rotation YAML,
adds its benchmark/fallback symbols, and always includes SPY and QQQ. The
script's historical `fmp` name does not describe its current provider; it uses
yfinance adjusted daily OHLCV.

Live target symbols and benchmarks must have a close for the expected NYSE
session. RL-only symbols can be reported as optional stale during the shared
refresh because they cannot affect broker orders. `track_rl_offline.py` applies
its own stricter rule to active RL holdings and targets.

## Failure triage

1. Read the Windows task result and wrapper log.
2. Read the final error in `logs/paper_trading_cron.log` and the dated Python
   log.
3. Identify which stage failed: refresh, target generation, pre-trade check,
   paper execution, metrics, offline RL, or post-run sanity/parity.
4. Verify authoritative state before retrying. For orders, query Alpaca by the
   persisted ID. For snapshots and decisions, query SQLite. For market data,
   inspect the terminal CSV date.
5. Retry only the failed idempotent stage when its contract supports that. Do
   not rerun the full paper orchestrator while order status is unresolved.

Common interpretations:

- A queued Friday `DAY` order is expected until the next regular session.
- A refresh warning for an RL-only symbol is not a live-order data failure.
- A terminally stale active RL symbol must be refreshed or explicitly verified
  inactive; an earlier gap followed by newer rows is historical.
- Dashboard weight drift can reflect valuation time. Quantity parity and target
  persistence establish the hard database check.
- A dashboard is a generated view. Use SQLite and broker receipts to resolve a
  disagreement.

## Verification after changes

Documentation-only work:

```bash
finrl-env/bin/python run_paper_trading.py --help
finrl-env/bin/python refresh_fmp_daily.py --help
finrl-env/bin/python track_rl_offline.py --help
finrl-env/bin/python track_metrics.py --help
git diff --check
```

Behavior changes require targeted tests plus the full configured suite:

```bash
finrl-env/bin/python -m pytest
finrl-env/bin/ruff check <touched-python-files>
finrl-env/bin/ty check <touched-python-files>
```

Do not install new verification tools solely for a maintenance pass.
