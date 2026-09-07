# Documentation Map

Start with [`../CONTEXT.md`](../CONTEXT.md). It explains the local three-way
comparison, execution boundaries, and artifact authority. This directory then
separates current operating guidance from dated history and design records.

## Current operating guidance

- [`operations.md`](operations.md): scheduler and wrapper behavior, command
  side effects, weekly failure semantics, and verification.
- [`offline_rl_tracking.md`](offline_rl_tracking.md): offline RL inputs,
  simulation rules, freshness policy, rebuild workflow, and limitations.
- [`rl_quickstart.md`](rl_quickstart.md): short, safe command reference for
  inspecting or regenerating offline RL outputs.
- [`trading_calendar_guide.md`](trading_calendar_guide.md): NYSE session-date
  resolution and cache-freshness rules used by the local workflow.
- [`paper_to_production_roadmap.md`](paper_to_production_roadmap.md): current
  evidence-collection stage and promotion gates. It is policy, not permission
  to trade or deploy.
- [`../ML_STOCK_SELECTION.md`](../ML_STOCK_SELECTION.md): ML bucket-selection
  data contract and point-in-time modeling rules. Reverify dated counts and
  cutoffs before a new run.
- [`../examples/README.md`](../examples/README.md): upstream notebook index.

## Dated records

- [`decision_log.md`](decision_log.md): adopted decisions and rationale. Later
  entries supersede conflicting earlier entries.
- [`weekly_comparison_journal.md`](weekly_comparison_journal.md): historical
  review snapshots and templates. Never use its last row as current health.

## Completed design plans

These files explain why the present workflow has its current artifacts. Their
implementation checklists are historical and should not be treated as open
work without confirming current code and state:

- [`foundation_stabilization_plan.md`](foundation_stabilization_plan.md)
- [`post_trade_reconciliation_plan.md`](post_trade_reconciliation_plan.md)
- [`strategy_decision_records_plan.md`](strategy_decision_records_plan.md)
- [`weekly_comparison_metrics_plan.md`](weekly_comparison_metrics_plan.md)

## Documentation rules

- Put stable boundaries and orientation in `AGENTS.md` and `CONTEXT.md`.
- Put operational procedures here and decisions in `decision_log.md`.
- Keep current performance, broker state, scheduler results, and data freshness
  in generated/runtime artifacts rather than hand-maintained prose.
- Date any observed state and name its source. A document verification date is
  not evidence that an external system is still healthy.
- Use repository-relative paths for tracked files. Explicitly label paths that
  live outside the repository.
