# Decision Log

This file records project decisions that should be easy to revisit later.

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

