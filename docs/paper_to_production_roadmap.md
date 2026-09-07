# Paper Comparison and Promotion Roadmap

Last policy review: 2026-09-07

This project is in paper/offline evidence collection. The roadmap defines what
must be learned before a later promotion discussion. It does not authorize
paper-account changes, RL activation, or real-money trading.

## Current system

- `FinRL`: Alpaca paper account using the ML-enhanced Adaptive Rotation config.
- `AR`: Alpaca paper account using the baseline Adaptive Rotation config.
- `RL`: offline simulation using generated target weights and local closes.
- SPY and QQQ: required benchmark series.
- Weekly cadence: Friday evening, with closed-market `DAY` orders queued for
  the next regular session.
- Audit contract: strategy decisions, target and actual weights, execution
  records, parity reports, SQLite snapshots, and dated metrics.

The foundation work once described as planned in this repository is now
implemented: structured strategy output, pre-trade validation, decision
records, post-trade reconciliation, comparison metrics, parity checks, kill
switches, shared-date offline RL tracking, weekend session resolution, and
terminal data-freshness checks. Their presence does not prove each weekly run
is healthy; verify current runtime artifacts.

## Present objective

Keep the three-way design unchanged while collecting enough comparable evidence
to evaluate:

- scheduler and refresh reliability;
- order acceptance, later fill status, and target-to-actual drift;
- strategy determinism and replay parity;
- fallback frequency and cash exposure;
- turnover and cost sensitivity;
- benchmark-relative performance on shared dates;
- offline RL sensitivity to simulated execution assumptions.

Operational integrity defects take priority over strategy expansion. A normal
week should leave enough evidence to explain what was decided, what was
submitted, what actually happened, and how the comparison was calculated.

## Stay in the current phase while

- observation counts remain small;
- orders, fills, or reconciliation require manual interpretation;
- required market-data freshness is inconsistent;
- replay, decision, or database parity is not stable;
- offline RL results change materially under reasonable cost, date-alignment,
  or inactive-symbol assumptions;
- the paper workflow still needs regular code repair to complete.

No calendar date by itself ends this phase.

## Gate for an RL paper-account proposal

Before discussing RL broker execution, require all of the following:

1. The offline contract and acceptance gate pass across the required
   walk-forward windows, with artifacts that identify model, seed, data window,
   configuration, and metrics.
2. Offline weights pass the same target-weight, concentration, turnover, data,
   and risk contracts used by the paper strategies.
3. Shared-date shadow evidence is stable under reasonable execution and
   cash-yield sensitivity checks.
4. The integration has an explicit paper-only account/config design, idempotent
   order handling, reconciliation, kill switch, and rollback plan.
5. The user explicitly approves changing RL from offline simulation to paper
   execution.

A passing offline gate is one input to this decision. It is not sufficient
authorization or evidence by itself.

## Gate for any real-capital proposal

Real capital remains out of scope. A future proposal would require explicit
user approval after, at minimum:

- a substantial paper history across varied regimes;
- stable scheduled operations and alerting;
- verified broker reconciliation and duplicate-order prevention;
- independent risk limits, exposure caps, and tested kill/rollback behavior;
- realistic cost, liquidity, gap, rejection, and partial-fill analysis;
- a documented capital limit and staged rollout;
- a review of tax, account, and operational consequences.

No repository setting, passing test, offline metric, paper result, or prior
approval substitutes for that real-money decision.

## Maintenance priorities

1. Keep required market data and benchmarks fresh.
2. Reconcile persisted paper order IDs before rerunning a failed or ambiguous
   weekly operation.
3. Preserve shared dates and target-versus-actual evidence.
4. Investigate operational failures with the narrowest corrective change.
5. Continue observation after reliability is restored.
6. Revisit strategy/model changes only when accumulated evidence identifies a
   specific limitation worth testing.

Current health and performance belong in SQLite, broker state, logs, and dated
generated metrics. Do not copy fast-changing figures into this roadmap.
