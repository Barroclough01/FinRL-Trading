# Paper Trading to Production and RL Roadmap

Last updated: 2026-06-07

This document tracks the path from the current dual-account paper comparison to
either production trading or a reinforcement learning strategy layer.

## Current State

The project is currently in a paper shadow-trading phase.

- `FinRL` account: ML-enhanced adaptive rotation config.
- `AR` account: baseline adaptive rotation config.
- Benchmarks: SPY in the dashboard, QQQ inside the adaptive rotation group
  strength calculation.
- Weekly execution: `run_paper_trading.py`.
- Metrics and dashboard: `track_metrics.py`, `data/finrl_trading.db`,
  `logs/dashboard.html`.
- RL code exists, but it is not integrated into the weekly comparison or
  execution path.

The next few months should be treated as a system validation and evidence
collection period, not as enough live data to train RL. Weekly observations are
valuable for execution quality, slippage, drift, and live-vs-backtest parity.

## Guiding Decisions

1. Keep the current paper comparison running while improving observability.
2. Do not move to production until pre-trade risk checks, reconciliation, and
   kill-switch behavior are explicit and tested.
3. Do not make RL the primary strategy first. RL should first become another
   target-weight strategy that can run through the same paper/live execution
   contract as AR and FinRL.
4. Prefer durable weekly artifacts over dashboard-only analysis. Each run should
   leave data that can be replayed, audited, and used for later model evaluation.
5. Compare strategies on the same dates, capital assumptions, benchmark windows,
   and execution assumptions.

## Near-Term Work: Paper Comparison Foundation

### 1. Strategy Decision Records (Completed)

Add a normalized decision record for every account and run date.

**Status: Completed (2026-05-31); first successful live persistence pending next weekly run**

Record at minimum:

- run date
- account name
- strategy/config path
- config hash
- regime state
- active groups
- ranked groups and group metrics
- fallback status and fallback reason
- target weights
- pre-trade positions
- order plan
- submitted orders
- filled orders
- post-trade positions
- cash and equity
- benchmark snapshot

Storage implemented:

- SQLite table: `strategy_decisions`
- JSONL mirror: `logs/strategy_decisions.jsonl`

The SQLite table is better for reports. The JSONL mirror is better for agent
inspection and append-only debugging. Fully integrated into `run_paper_trading.py` and covered by unit tests.

### 2. Weekly Comparison Metrics (Completed)

Add a durable weekly metrics artifact alongside the HTML dashboard.

**Status: Completed (2026-05-31)**

Suggested output:

- `logs/comparison_metrics_YYYY-MM-DD.json`
- `logs/comparison_metrics_latest.csv`

Metrics:

- cumulative return
- weekly return
- volatility (annualized)
- Sharpe ratio (annualized)
- max drawdown
- turnover
- cash exposure (average cash weight)
- hit rate
- beta to SPY and QQQ
- tracking error vs SPY and QQQ
- weeks in fallback
- realized vs target weight drift

Fully integrated into `track_metrics.py` and covered by unit tests.

### 3. AR Fallback Diagnostics

The AR baseline can enter fallback because every active group fails the positive
excess return filter vs QQQ. That is different from missing data or invalid
groups.

Every AR run should explicitly log one fallback reason:

- `no_valid_groups`
- `all_groups_negative_excess_return`
- `risk_off_cash_floor`
- `fast_risk_off`
- `stop_loss_or_cooldown`
- `missing_or_stale_data`
- `order_or_execution_failure`

This lets later analysis distinguish strategy behavior from data or execution
failures.

### 4. Live vs Replay Parity

For each weekly paper run, replay the same date locally using the same input
data and config. Compare:

- replay target weights
- submitted target weights
- filled actual weights
- dashboard recorded weights

Any mismatch should be investigated before production.

### 5. Data Freshness Checks

Before order generation, validate:

- every required ticker has a recent close
- SPY and QQQ benchmark data are present
- no close values are null, zero, or stale
- all target weights are finite
- target weights sum within tolerance
- account equity and cash are readable
- market calendar status is known

Failures should stop the run before order submission.

## Production Readiness Work

Production mode requires operational safeguards, not just promising paper
returns.

Required before real capital:

- hard pre-trade risk gate
- post-trade broker reconciliation
- kill switch
- order idempotency
- dry-run/live parity
- alerting on failures
- explicit production config
- tests for order sizing and risk checks

### Pre-Trade Risk Gate

Block orders if any rule fails:

- max single-name weight
- max bucket/sector exposure
- max turnover
- max order notional
- max cash usage
- max daily loss
- max drawdown pause
- stale data
- missing benchmark
- unexpectedly large drift from current holdings

### Reconciliation

After execution, reconcile:

- target weights
- open orders
- filled orders
- actual positions
- cash
- equity
- unfilled or rejected orders

Write a reconciliation report to `logs/reconciliation_YYYY-MM-DD.json`.

### Kill Switch

Add a file or environment flag that blocks order submission, for example:

```text
TRADING_DISABLED=true
```

The script may still refresh data and produce dry-run plans, but it must not
submit orders while disabled.

## RL Readiness Work

RL should share the same target-weight contract as the existing strategies.

### Required RL Contract

Define these before training is considered meaningful:

- observation space
- action space
- reward function
- transaction cost model
- turnover penalty
- drawdown penalty
- train/eval split
- walk-forward evaluation windows
- model artifact metadata
- acceptance gate

### Observation Space

Candidate fields:

- recent returns
- volatility
- covariance features
- regime state
- group strength metrics
- benchmark state
- fundamental features
- current holdings
- cash weight
- realized drift

### Action Space

The action should be target portfolio weights, not direct broker orders.

This keeps RL compatible with:

- backtests
- paper trading
- production trading
- risk overlays
- broker execution

### Reward Function

The reward should penalize behavior that would be unacceptable in production.

Candidate reward:

```text
portfolio_return
- turnover_penalty
- drawdown_penalty
- volatility_penalty
- concentration_penalty
```

Avoid using raw return alone. It encourages excessive risk and churn.

### Evaluation Baselines

RL must beat these before promotion:

- SPY
- QQQ
- equal-weight selected universe
- AR baseline
- ML-enhanced AR

### Acceptance Gate

The existing `src/strategies/rl_acceptance_gate.json` is a seed. Expand it to
include:

- minimum number of walk-forward windows
- minimum Sharpe
- maximum drawdown
- minimum annualized return
- maximum turnover
- maximum concentration
- minimum improvement vs SPY and QQQ
- stability across multiple random seeds

## Suggested Build Order

1. **Add strategy decision records.** (Completed: SQLite and JSONL mirror implemented and tested)
2. **Add weekly comparison metrics artifacts.** (Completed: JSON and CSV files generated side-by-side)
3. **Add explicit AR fallback diagnostics.** (Completed: Fallback reasons parsed and logged to decision records)
4. **Add pre-trade validation.** (Completed: 9-point validation gate implemented and tested)
5. **Add post-trade reconciliation.** (Completed: Discrepancy checks, alert flags, and JSON reports implemented and tested)
6. **Add a live-vs-replay parity check.** (Completed: `run_parity_checks()` in `run_paper_trading.py`, reports at `logs/parity_check_YYYY-MM-DD.json`)
7. **Add production kill switch.** (Completed: `TRADING_DISABLED=true` env var or `.kill_switch` file forces dry-run)
8. **Harden tests around order sizing, metrics, and risk gates.** (Completed: 17 passing tests in `tests/test_weekly_workflow.py`)
9. **Standardize RL observation/action/reward contracts.**
10. **Integrate RL as an offline candidate strategy.**
11. **Add RL as a third paper account only after it passes offline gates.**
12. **Consider production only after several months of clean paper evidence.**

## Operational Notes (2026-06-07)

The 2026-06-05 weekly run exposed two bugs in newly shipped foundation code:

1. **Timestamp serialization** — Alpaca order responses include pandas `Timestamp`
   fields (`submitted_at`, `filled_at`). `save_strategy_decision()` SQLite inserts
   called `json.dumps()` without `default=str`, crashing after orders were placed.
   Fixed: all `json.dumps()` calls in `save_strategy_decision()` now use
   `default=str`.

2. **Negative cash validation** — Pre-trade rule 9 rejected accounts with
   `cash < 0`, but Alpaca margin/paper accounts routinely show small negative cash
   when fully invested (AR had -$383.70 with $988K equity). Historical snapshots
   confirm both accounts often carry negative cash. Fixed: only block when
   `equity <= 0`; log a warning for negative cash.

Next weekly run should produce the first complete decision records, reconciliation
reports, and updated metrics artifacts.

## Path Forward (as of 2026-06-07)

### Phase: Paper evidence collection (current)

The foundation build order (items 1–8) is complete in code. The next step is
operational validation: prove the weekly workflow produces complete artifacts on
a live run before starting production-readiness or RL work.

### Immediate (this week)

1. **Validate the next Friday live run end-to-end.**
   Confirm decision records, reconciliation reports, weekly snapshots, dashboard,
   and comparison metrics all update after a successful (or partially successful)
   execution.

2. **Resolve FinRL June 5 pending orders.**
   Four orders (SATS/MCHP/ORCL sells, ON buy) were submitted before the
   decision-record crash. Confirm fills or cancel before the next rebalance.

3. **Keep the comparison running unchanged.**
   Do not alter AR fallback logic or swap benchmarks mid-experiment. Collect
   weeks of FinRL vs AR vs SPY evidence.

4. **Monitor, don't build.**
   Resist RL integration or production work until several clean weekly runs
   establish operational confidence.

### Short-term engineering (after one clean weekly run)

| Priority | Work | Rationale |
| --- | --- | --- |
| 1 | Metrics on partial/total failure | **Completed (2026-06-07):** `run_metrics_tracker()` always runs after live sessions; falls back to `--report-only` if full snapshot fails. |
| 2 | Hard pre-trade risk gate | Extend validation with max turnover, drawdown pause, max order notional, drift-from-holdings checks. |
| 3 | Order idempotency | Prevent duplicate submissions on retry. |
| 4 | Alerting | Webhook returned 403 on June 5; failures need a reliable notification channel. |
| 5 | Explicit production config | Separate prod credentials and YAML from paper comparison accounts. |

Items 1 (metrics resilience) ships in the same session as this doc update.
Items 2–5 are the next development sprint after operational validation.

### Medium-term (roadmap items 9–12)

1. Standardize RL observation/action/reward contracts.
2. Integrate RL as an offline candidate (offline gate failed Sharpe 0.75 vs 0.80
   on 2026-05-17).
3. Add RL as a third paper account only after offline gate passes.
4. Consider production only after months of clean paper evidence.

### Calendar / strategy (not code)

| When | Task |
| --- | --- |
| ~July 2026 | Quarterly ML refresh: `ml_bucket_selection.py --mixed-vintage` → `update_adaptive_rotation_symbols.py` |
| Ongoing | Track whether AR exits fallback (mega-caps vs QQQ trend filter) |
| Open | Decide if QQQ is too demanding a benchmark for the AR baseline |

### Explicitly deferred

- RL training on live weekly data (too few observations).
- Production capital (operational safeguards incomplete).
- Changing AR fallback behavior (breaks baseline comparison).

## Open Questions

- Should AR compare group strength against QQQ, SPY, or a blended benchmark?
- Should fallback allocate to broad ETFs or cash when groups fail the trend
  filter?
- What maximum turnover is acceptable for production?
- What drawdown or weekly loss should trigger an automatic pause?
- Should RL allocate across the current selected stocks only, or across all
  bucket candidates?
- Should the live production strategy be the best single strategy, or an
  ensemble of FinRL, AR, and future RL weights?

