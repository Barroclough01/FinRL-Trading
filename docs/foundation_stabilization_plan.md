# Foundation Stabilization Plan

Last updated: 2026-05-31
Status: **COMPLETED (2026-05-31)**

This document lists the immediate fixes to make before building larger
production-readiness or RL features. The goal is to make the current paper
trading system reliable, observable, and testable.

## Why This Comes First

The repo currently runs, and tracked Python files compile, but the most important
weekly workflow still depends on fragile operational assumptions:

- strategy weights are parsed from console output (Fixed: Structured JSON output contract introduced)
- metrics can fail per account without failing the process (Fixed: Collected snapshot failures and exit nonzero)
- pre-trade validation is incomplete (Fixed: Added 9-point validation gate logging to logs/)
- there is no project-level test suite (Fixed: Added robust pytest suite under tests/)
- some data refresh failures are treated like missing/new files (Fixed: Explicit empty/corrupt CSV validation)
- RL dependency checks are disabled (Fixed: Restored fail-fast check)

Fixing these does not change the strategy thesis. It makes the system safer to
iterate on.

## Priority 1: Replace Stdout Weight Parsing

Current issue:

`run_paper_trading.py` shells out to
`src/strategies/run_adaptive_rotation_strategy.py` and regex-parses printed
portfolio lines to recover target weights.

Risk:

Any formatting change in the strategy CLI can silently break live order
generation.

Desired fix:

Add a structured output path.

Options:

1. Add `--json-output PATH` to `run_adaptive_rotation_strategy.py`.
2. Add a direct Python function that returns weights and audit data, then call
   it from `run_paper_trading.py`.

Recommended:

Start with `--json-output PATH` because it is small, explicit, and preserves the
existing script boundary. Later, refactor to direct imports if useful.

Acceptance criteria:

- `run_paper_trading.py` no longer parses human-readable stdout.
- Strategy output includes target weights, cash weight, regime state, active
  groups, ranked groups, fallback status, and audit file path.
- If the JSON file is missing or malformed, the run fails before order planning.

## Priority 2: Make Metrics Failures Fatal

Current issue:

`track_metrics.py` logs account snapshot failures but continues to print reports
and regenerate the dashboard.

Risk:

The dashboard can look fresh while one account is missing or stale.

Desired fix:

Collect per-account snapshot failures and exit nonzero after attempting all
accounts.

Acceptance criteria:

- Missing Alpaca snapshot for any configured account causes `track_metrics.py`
  to exit nonzero.
- `run_paper_trading.py` treats metrics failure as a run failure.
- Dashboard generation does not mask missing weekly rows.

## Priority 3: Add Pre-Trade Validation

Current issue:

Order planning happens before a full validation of data freshness, weight shape,
and benchmark availability.

Risk:

Bad inputs can reach order planning or broker submission.

Desired fix:

Add a validation gate before dry-run order planning in `run_paper_trading.py`.

Validate:

- target weights are nonempty
- all weights are finite numbers
- no negative weights unless shorting is explicitly supported
- sum of target weights is within tolerance
- no single symbol exceeds configured maximum weight
- all required symbol prices are fresh
- SPY and QQQ benchmark data are present
- market calendar status is known
- account cash/equity can be read

Acceptance criteria:

- Validation failures stop the run before order planning.
- Error messages name the failed rule and suggested fix.
- Validation output is saved to `logs/pre_trade_validation_YYYY-MM-DD.json`.

## Priority 4: Add Focused Tests

Current issue:

There is no visible project test suite for the weekly trading workflow.

Risk:

Small changes to metrics, order sizing, or strategy output can break paper
trading without being caught locally.

Desired fix:

Add a small `tests/` tree focused on behavior, not implementation details.

Initial tests:

- structured strategy output can be parsed and validated
- invalid target weights fail validation
- dashboard SPY series aligns to weekly snapshot dates
- metrics fail when an account snapshot fails
- post-run sanity checks detect missing rows
- corrupted CSV date column fails data refresh with a clear error

Acceptance criteria:

- `pytest -q` runs locally.
- Tests do not require real Alpaca credentials.
- Broker/network boundaries are mocked.
- At least the changed foundation code is covered.

## Priority 5: Make Data Refresh Errors Explicit

Current issue:

`refresh_fmp_daily.py` returns `None` when an existing CSV cannot be read, making
corrupt files look like missing files.

Risk:

A corrupt local price file can trigger a full-history rewrite or hide data
quality problems.

Desired fix:

Fail fast when an existing CSV is unreadable or missing required columns.

Acceptance criteria:

- Missing file still means new symbol.
- Empty file, corrupt CSV, or missing `date` column produces a clear failure.
- The failing ticker and file path are included in the error.

## Priority 6: Re-Enable RL Dependency Check

Current issue:

`run_rl_offline_pipeline.py` has `_check_required_modules()` commented out.

Risk:

The RL pipeline can start and fail later with less actionable errors.

Desired fix:

Restore fail-fast dependency checking before training.

Acceptance criteria:

- Missing RL dependencies produce one actionable install message.
- `--skip-train` still checks dependencies only if evaluation needs them.

## Priority 7: Clarify Exception Signal Status

Current issue:

`adaptive_rotation_engine.py` notes that historical Z-scores are not maintained
yet, but exception detection still builds a one-point series from the latest
score.

Risk:

Exception behavior may look more complete than it is.

Desired fix:

Either implement the intended historical Z-score tracking or explicitly disable
exception behavior that depends on unavailable history.

Acceptance criteria:

- Audit logs show whether exception detection is fully enabled or disabled.
- No incomplete exception signal silently influences target weights.

## Priority 8: Add Minimal Tooling Config

Current issue:

The repo has `requirements.txt` and `setup.py`, but no project-level lint,
format, or test config.

Desired fix:

Add minimal tooling without large refactors.

Suggested baseline:

- `pyproject.toml` with pytest and ruff config
- `tests/` directory
- optional pre-commit or `prek` config once the first tests exist

Acceptance criteria:

- `ruff check` has a known baseline.
- `ruff format --check` has a known baseline.
- `pytest -q` has at least the foundation tests.

## Suggested Implementation Order

1. Structured strategy output.
2. Metrics fail-fast behavior.
3. Pre-trade validation.
4. Initial tests for the above.
5. Data refresh error handling.
6. RL dependency check.
7. Exception signal clarification.
8. Tooling config and pre-commit/prek hooks.

## Definition of Stable Enough

The foundation is stable enough for roadmap work when:

- weekly runs fail loudly on missing data, missing metrics, or malformed weights
- every submitted order can be traced back to a structured strategy decision
- dashboard data cannot silently omit an account
- basic tests cover the weekly workflow's failure paths
- dry-run, paper, and replay paths use the same target-weight contract

