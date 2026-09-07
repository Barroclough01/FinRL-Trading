# Project Instructions

This checkout runs a bounded comparison of two Alpaca paper accounts and one
offline RL simulation. Preserve comparison integrity and operational safety
before adding strategies, retraining models, or changing execution behavior.

## Start here

1. Read `CONTEXT.md` for the local system boundary and `docs/README.md` for the
   documentation map.
2. Inspect `git status --short --branch` with Git inside WSL. The checkout is
   WSL-owned; Windows Git can report misleading ownership errors.
3. For an operational status request, inspect the current scheduler state,
   wrapper and logs, SQLite records, generated metrics, cached market data, and
   broker state as needed. Dated documentation is historical evidence, not a
   health check.
4. Use `finrl-env/bin/python` or activate `finrl-env`; do not assume bare
   `python` points to the project environment.

## Non-negotiable boundaries

- `FinRL` and `AR` are Alpaca paper accounts. They may submit paper orders.
- `RL` is an offline simulation. `track_rl_offline.py` writes simulated
  snapshots to SQLite and never submits Alpaca orders.
- Do not activate RL for broker execution, add another strategy, retrain a
  model, or promote anything toward real capital without an explicit request.
- Never place real-money orders. Verify that broker URLs and credentials refer
  to paper accounts before any non-dry-run maintenance.
- Friday-evening `DAY` orders intentionally queue for the next regular session.
  Do not add a market-hours-only rejection. `MARKET_CLOSED_ACTION=next_open` is
  the established behavior.
- Preserve target weights separately from actual weights. A missing fill must
  remain visible as `actual_weight=0.0`; do not normalize it away.
- Empty fallback symbols in an enabled Adaptive Rotation fallback mean cash,
  not SPY, QQQ, or another synthetic exposure.
- Compare strategies only on shared chronological dates. Keep SPY and QQQ data
  current, and describe small observation counts as preliminary.
- Verified inactive RL symbols are handled only through
  `src/strategies/rl_inactive_symbols.json`. Do not infer a corporate action or
  backfill a policy entry from a stale file alone.

## Runtime topology and authority

- `run_paper_trading.py` is the weekly orchestrator. A live run can refresh
  metrics, invoke offline RL tracking, write audit artifacts, and submit paper
  orders. `--dry-run` suppresses order submission and does not run the complete
  metrics/offline-RL tail.
- `refresh_fmp_daily.py` refreshes the local `data/fmp_daily/` cache with
  yfinance despite its historical filename. SPY and QQQ are mandatory refresh
  symbols.
- `track_metrics.py` records Alpaca snapshots and builds the HTML/CSV/JSON
  comparison artifacts.
- `track_rl_offline.py` consumes `results/drl_weight.csv`, local cached closes,
  and the inactive-symbol policy. Active terminally stale data is fatal; an
  historical gap that later resumes is not.
- `src/strategies/rl_contract.json` and `rl_acceptance_gate.json` describe the
  offline training/evaluation contract. Passing that gate does not authorize
  paper or live execution.
- The tracked source and configuration are durable. `.env`, `data/`, `logs/`,
  `results/`, model files, caches, databases, and local wrapper output are
  runtime state unless a tracked fixture explicitly says otherwise.
- The Windows Task Scheduler wrapper lives outside this repository at
  `C:\Users\paxto\stock-trading\run_paper_trading.ps1`. Inspect it and the
  registered task read-only when diagnosing scheduling. Do not overwrite it as
  part of repository documentation work.
- The SQLite database and broker are operational authorities for recorded
  snapshots and order state. Generated dashboards summarize them; they do not
  override them.

## Safe and mutating commands

Commands that do not submit broker orders include:

```bash
git status --short --branch
finrl-env/bin/python run_paper_trading.py --help
finrl-env/bin/python run_paper_trading.py --dry-run --date YYYY-MM-DD
finrl-env/bin/python refresh_fmp_daily.py --dry-run
finrl-env/bin/python sync_rl_price_data.py --dry-run
finrl-env/bin/python track_metrics.py --report-only --date YYYY-MM-DD
```

Treat these as state-changing even though they are paper/offline operations:

- `run_paper_trading.py` without `--dry-run` can submit Alpaca paper orders.
- `refresh_fmp_daily.py` without `--dry-run` writes cached market data.
- `track_metrics.py` without `--report-only` writes broker snapshots.
- `track_rl_offline.py` and `backfill_rl_history.py` rewrite simulated database
  history; back up the database before a rebuild.
- `sync_rl_price_data.py` without `--dry-run` writes SQLite price rows.
- `run_rl_offline_pipeline.py` can train models and replace generated results.

## Change and verification discipline

- Keep routine experiment check-ins read-only unless implementation is
  explicitly requested.
- Do not build or update Graphify artifacts unless the user explicitly asks.
- Preserve unrelated changes, ignored operational data, databases, logs,
  wrappers outside the repository, and secrets.
- Production-code refactoring is out of scope for documentation maintenance.
- Use `uv run` only when it resolves to this project's configured environment;
  the checked-in and known-good path is `finrl-env/bin/python`.
- Run targeted tests for touched behavior, then the full test suite when code or
  executable configuration changes. Run Ruff and Ty on relevant Python files.
- For documentation-only changes, validate Markdown links, referenced paths,
  command help, UTF-8 encoding, `git diff --check`, and the final diff.

## AI Wiki research context

The canonical project bridge is available at either host path:

- Windows/Codex: `C:\Users\paxto\ai-wiki\wiki\projects\finrl-trading.md`
- WSL: `/mnt/c/Users/paxto/ai-wiki/wiki/projects/finrl-trading.md`

Consult it before proposing changes to strategy architecture, evaluation
metrics, benchmark handling, RL scope, data sources, or paper execution. Follow
only relevant links. The repository's current data, weights, configuration, and
metrics remain authoritative. Separate sourced findings, inference, and
recommendations, and do not update the wiki unless the task authorizes it.
