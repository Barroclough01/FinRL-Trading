# Decision Log

This file records project decisions that should be easy to revisit later.

## 2026-06-07: Fix June 5 Run Failures (Timestamp + Negative Cash)

Decision:

Fix two bugs exposed by the 2026-06-05 weekly run: pandas Timestamp serialization
in strategy decision records, and overly strict negative-cash pre-trade validation.

Reasoning:

The FinRL account placed orders successfully but crashed when saving decision
records because Alpaca order responses include `submitted_at`/`filled_at` as pandas
Timestamps. The SQLite insert path called `json.dumps()` without `default=str`
(only the JSONL mirror had it).

The AR account was blocked because pre-trade rule 9 rejected `cash < 0`. Alpaca
margin/paper accounts routinely show small negative cash when fully invested —
historical weekly snapshots show both accounts with negative cash (-$47 to -$469)
during successful runs. Equity was healthy at $988K.

Implementation details:

- Added `default=str` to all `json.dumps()` calls in `save_strategy_decision()`.
- Changed pre-trade rule 9 to only block when `equity <= 0`; log a warning for
  negative cash instead of failing.
- Added 3 tests: Timestamp serialization, negative-cash pass, zero-equity fail.
- Test suite now 17 passing.

## 2026-05-31: Treat Current Period as Evidence Collection

Decision:

Continue running the FinRL vs AR vs SPY comparison for the next few months, but
treat it as paper-trading system validation rather than enough data to train RL.

Reasoning:

Weekly live observations are too few for RL training, but they are useful for
checking execution, fill quality, strategy drift, data freshness, and dashboard
metrics.

Follow-up:

Add durable weekly artifacts so future analysis does not depend on screenshots
or dashboard state.

## 2026-05-31: Preserve Target Weights as the Strategy Contract

Decision:

Production strategies and future RL strategies should emit target portfolio
weights. Broker orders remain downstream of strategy logic.

Reasoning:

The repo already follows a weight-centric architecture. Keeping this contract
lets AR, FinRL, and RL share backtesting, risk overlays, paper trading, and live
execution.

Follow-up:

Formalize a strategy decision record that stores target weights, actual weights,
orders, fills, and reconciliation data.

## 2026-05-31: RL Should Be Added as a Candidate Strategy First

Decision:

Do not replace the current paper strategies with RL. First integrate RL as an
offline candidate, then as a third paper account only after it passes acceptance
gates.

Reasoning:

RL needs a defined observation space, action space, reward, train/eval split,
transaction cost model, and promotion gate before its results are meaningful.

Follow-up:

Standardize the RL contract and expand `src/strategies/rl_acceptance_gate.json`.

## 2026-05-31: Production Requires Operational Safeguards

Decision:

Do not move from paper to real capital until pre-trade validation, risk gates,
post-trade reconciliation, idempotency, and a kill switch are implemented.

Reasoning:

A profitable strategy can still fail operationally. Production readiness means
the system can fail safely and produce clear evidence of what happened.

Follow-up:

Add pre-trade and post-trade reports under `logs/`, plus tests for order sizing,
risk checks, stale data handling, and metrics calculations.

## 2026-05-31: Diagnose AR Fallback Explicitly

Decision:

When AR enters fallback, log a concrete reason instead of treating fallback as a
single opaque outcome.

Reasoning:

Fallback due to all groups underperforming QQQ is different from fallback due to
missing data, risk-off regime, stop-loss, or execution failure.

Follow-up:

Add fallback reason values such as `no_valid_groups`,
`all_groups_negative_excess_return`, `fast_risk_off`,
`stop_loss_or_cooldown`, and `missing_or_stale_data`.

## 2026-05-31: Completion of Foundation Stabilization Tasks

Decision:

Successfully implemented and completed all eight priorities of the Foundation Stabilization Plan.

Reasoning:

To build a reliable, observable, and testable codebase before moving on to production readiness and offline RL improvements.

Implementation details:
- Created structured JSON output contract (`--json-output PATH`) in strategy runner.
- Replaced fragile stdout regex parsing with robust JSON parsing in the paper trading script.
- Made metrics tracking failures fatal, causing track_metrics.py to exit nonzero and aborting the paper trading workflow.
- Introduced a pre-trade validation gate (checking weight shapes, freshness, benchmarks, calendar, and Alpaca cash/equity status) and logged validation output to `logs/pre_trade_validation_YYYY-MM-DD.json`.
- Added a robust unit test suite (`tests/test_weekly_workflow.py`) with 9 tests covering all failure paths.
- Made corrupt or empty local CSV file detection explicit with ticker and path in data refresh.
- Re-enabled RL training pipeline dependency checks.
- Clarified exception detector rule status in audit logs and deactivated persistence checks when history is missing.
- Added project-level tooling config (`pyproject.toml`) for pytest and ruff.

## 2026-05-31: Design and Implementation of Strategy Decision Records

Decision:

Adopted a dual-storage model (SQLite and JSONL mirror) to capture complete, high-fidelity audit trails for strategy executions.

Reasoning:

To enable deep post-trade analysis, historical replays, and offline RL training. SQLite provides easy querying and report generation, while append-only JSONL provides robust, inspection-friendly mirrors.

Implementation details:
- Created SQLite table `strategy_decisions` to store run metadata, strategy state, target weights, pre/post-trade positions, order plans, submitted orders, cash/equity, and benchmark snapshots.
- Created JSONL mirror at `logs/strategy_decisions.jsonl` to record the same data per line.
- Fully integrated the recording logic into `run_paper_trading.py`.
- Added robust fallback reason detection by inspecting audit files.
- Added MD5-based config file hashing for config tracking.
- Covered with comprehensive pytest verification.

## 2026-05-31: Design and Implementation of Weekly Comparison Metrics

Decision:

Implemented a robust metrics engine to calculate and save comprehensive annualized performance and risk metrics side-by-side for all accounts and benchmarks.

Reasoning:

To provide durable, mathematically precise comparison metrics alongside the visual HTML dashboard. These metrics are crucial for evaluating strategy performance (e.g. Sharpe ratio, max drawdown, tracking error, beta, turnover, and realized vs target weight drift) over walk-forward windows.

Implementation details:
- Created the metrics calculator function `calculate_comparison_metrics(conn, run_date)` in `track_metrics.py`.
- Implemented standard formulas for cumulative return, annualized volatility, annualized Sharpe ratio (assuming 2% risk-free rate), max drawdown, and hit rate.
- Implemented portfolio turnover and realized vs target weight drift calculations by querying historical data from `weekly_weights`.
- Implemented relative benchmark metrics including Beta ($\beta$) and Tracking Error vs SPY and QQQ.
- Integrated the outputs into `track_metrics.py` to write `logs/comparison_metrics_YYYY-MM-DD.json` and a flat `logs/comparison_metrics_latest.csv` side-by-side.
- Verified mathematically and logically with comprehensive pytest unit tests.

## 2026-05-31: Design and Implementation of Post-Trade Reconciliation

Decision:

Adopted a structured post-trade reconciliation workflow to check for weight drift, unexpected holdings, missing assets, and failed orders after every weekly execution run.

Reasoning:

To ensure robust execution quality and operational safety before moving from paper trading to live production capital. Post-trade reconciliation guarantees that the broker's actual portfolio matches the strategy's target weights, and flags any execution anomalies.

Implementation details:
- Created `reconcile_post_trade(run_date, account_name, record)` in `run_paper_trading.py` to compare target weights with realized weights and flag weight drift exceeding a 2.0% tolerance threshold.
- Added alert flags for missing target assets, unexpected holdings, and failed/rejected orders.
- Created `save_reconciliation_report(run_date, account_name, recon_result)` to consolidate and save reports to `logs/reconciliation_YYYY-MM-DD.json`.
- Integrated reconciliation checks at the very end of `run_account` in `run_paper_trading.py`.
- Verified execution quality and alert flagging with comprehensive pytest unit tests.

