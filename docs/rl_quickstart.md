# Offline RL Command Reference

Read [`offline_rl_tracking.md`](offline_rl_tracking.md) before changing RL
history. RL is local simulation only.

## Inspect

```bash
cd /home/paxto/stock-trading/FinRL-Trading
finrl-env/bin/python track_rl_offline.py --help
finrl-env/bin/python sync_rl_price_data.py --dry-run
finrl-env/bin/python track_metrics.py --report-only --date YYYY-MM-DD
```

`--report-only` avoids a broker snapshot but regenerates dashboard and metric
files. To inspect database dates and integrity without any writes, use the query
in `offline_rl_tracking.md`.

## Simulate one shared date

The next command writes or replaces an RL snapshot and derived reports. Use it
only after confirming the date belongs to the FinRL/AR comparison calendar and
active symbol caches are fresh:

```bash
finrl-env/bin/python track_rl_offline.py --date YYYY-MM-DD
```

## Rebuild history

Back up `data/finrl_trading.db`, verify the backup, and review weights, shared
dates, and inactive-symbol policy first:

```bash
finrl-env/bin/python backfill_rl_history.py
```

The helper is interactive. It backfills only the dates selected by the current
implementation and regenerates the dashboard.

## Train or evaluate models

The following is a research workflow, not routine weekly maintenance:

```bash
finrl-env/bin/python src/strategies/run_rl_offline_pipeline.py --help
```

Training can replace generated model/results artifacts. `--skip-train` evaluates
existing summaries, but a passing gate still does not authorize broker use.

## Validate

```bash
finrl-env/bin/python -m pytest tests/test_rl_contract.py \
  tests/test_rl_candidate_strategy.py tests/test_rl_inactive_policy.py
git diff --check
```

For a complete project check, run `finrl-env/bin/python -m pytest`.
