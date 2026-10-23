# Missing-return comparison repair — October 23, 2026

Status: Chief-reviewed and authorized for scoped commit/push. Baseline: `2b6d0da` on
`codex/paper-comparison-reliability` in `C:\Users\paxto\FinRL-Trading`.

## Change and existing contract

The public `calculate_comparison_metrics()` grouped all account snapshots but
created its dated-return mapping only for finite, non-NULL observations. With
two or more benchmark-matched dates, an account with only NULL returns therefore
raised `KeyError` before the existing insufficient-pair guard could run.

The account pairing loop now uses an empty mapping when that account has no
valid returns. Missing observations never enter benchmark pairs or the shared
comparison. The existing fewer-than-two-valid-pairs behavior is retained:
beta and tracking error keep their initialized zero defaults. These defaults
mean insufficient data; they are not evidence of measured zero exposure or
measured zero tracking error. A missing required account still yields shared
comparison status `unavailable`, no as-of date, and empty shared weekly returns.
Other accounts continue to calculate normally.

Standalone account statistics still use the pre-existing NULL-to-zero
conversion. This is an interpretation limit, not a newly adopted missing-data
policy. Changing that convention or adding explicit insufficient-data fields
would require a separate reporting/metric contract decision. This repair does
not change return intervals, calendar selection, risk estimators, strategies,
the preceding beta normalization repair, or report schemas.

## Verification

- Reconstructed the baseline function in memory without changing repository
  files: the new tests reproduced three `KeyError: 'FinRL'` failures (all-NULL
  two-date and three-date cases, plus export); four partial/normal cases passed.
- Seven new public-calculator tests cover all-NULL accounts, mixed healthy
  accounts, one valid pair at an early or latest date, sparse two-pair and normal
  three-pair beta/tracking error, and JSON/CSV report output. Shared comparison
  dates and returns are checked against actual observations.
- Focused missing-return, beta, chronology and reporting-provenance checks:
  **34 passed**, with RuntimeWarnings treated as errors.
- Full configured `tests/` suite: **234 passed** in 19.67 seconds, also treating
  RuntimeWarnings as errors.
- Ty on `track_metrics.py` and the new test file: passed. Ruff on the new tests:
  passed. Production Ruff's **32 findings** match baseline by diagnostic code,
  message and source text; none introduced. `git diff --check`: passed.

Native Windows Python lacks Alpaca dependencies. Tests used the existing WSL
`/home/paxto/stock-trading/FinRL-Trading/finrl-env/bin/python` with asserted import
binding to `/mnt/c/Users/paxto/FinRL-Trading/track_metrics.py`, disposable working
and output directories, bytecode disabled, and dotenv disabled before imports.
No packages were installed. Synthetic databases and mocked external boundaries
were used; no broker/provider calls, operational metrics regeneration, runtime
source synchronization, deployment, activation, scheduler/configuration changes,
backup, or historical output rewrite was performed. The three pre-existing
untracked readiness/canary paths remain untouched.

This is source-level evidence only; it establishes no current operational or
strategy-comparison health.

## Routing documentation correction

Chief review found stale `AGENTS.md` startup instructions calling the native
development checkout WSL-owned. On October 23, native Git reported top-level
`C:/Users/paxto/FinRL-Trading` and Git directory
`C:/Users/paxto/FinRL-Trading/.git`. A separate read-only WSL Git invocation
reported runtime top-level `/home/paxto/stock-trading/FinRL-Trading` and Git
directory `/home/paxto/stock-trading/FinRL-Trading/.git`. These confirm distinct
repository metadata locations and agree with the current AI Wiki project bridge
and Chief's current project-discovery routing.

`AGENTS.md` now names both routes, directs Git to the respective host, and
requires verified source binding, disposable outputs and dotenv suppression
when using the existing WSL interpreter against native development sources.
Runtime rules and authorization limits remain intact. Older dated context was
not broadly rewritten. This check does not establish deployed runtime revision,
runtime/source equality, or operational health. Final documentation validation:
`git diff --check` and review of the scoped instructions/report diff.

The Chief reviewed the calculator change, seven tests, missing-data conventions,
routing correction and this report, then passed the four-file publication gate.
Runtime synchronization and deployment remain outside this authorization.
