# Beta normalization correction — October 23, 2026

Status: Chief-reviewed and authorized for scoped commit/push. Baseline: `e693dae` on
`codex/paper-comparison-reliability` in `C:\Users\paxto\FinRL-Trading`.
The WSL runtime has not been synchronized or deployed.

## Change

`calculate_comparison_metrics()` divided sample covariance (`np.cov` defaults
to `ddof=1`) by population variance (`np.var` defaults to `ddof=0`). For `n`
matched observations this inflated beta by `n / (n - 1)`. Three identical
nonconstant returns therefore produced beta 1.5 instead of 1.0.

Covariance now explicitly uses `ddof=0`, matching the existing variance. This
changes the active matched-date account calculation and the two reference-series
paths in its nested statistics helper. The helper's reference-series paths
currently have no callers. Date pairing, tracking error, reporting schema,
short-sample guards, and the existing zero-beta convention for constant
benchmarks are preserved.

## Verification

- Before the fix, synthetic public-calculator checks reproduced true slope 1
  reported as 2.0, 1.5, and 1.25 with two, three, and five observations.
- Eleven new tests cover identical series, an intercept, scale 2, slope -1,
  constant benchmark/account returns, zero or one valid pair, unchanged
  volatility/tracking error, and corrected JSON/CSV values.
- Focused beta, chronology, and provenance checks: **27 passed**, with runtime
  warnings treated as errors.
- Full configured `tests/` suite: **227 passed** in 24.60 seconds.
- Ty on `track_metrics.py` and `tests/test_metrics_beta.py`: passed.
- Ruff on the new tests: passed. The 32 findings in `track_metrics.py` are
  identical to baseline by diagnostic code, message, and source line; none
  introduced. `git diff --check`: passed.

Windows Python lacked Alpaca dependencies. Checks used the existing WSL
`finrl-env/bin/python` to read the Windows development source, with disposable
working/output directories, bytecode disabled, and dotenv loading disabled.
No packages were installed. Tests used synthetic databases and mocked external
boundaries; no operational metric regeneration, broker calls, backup/restore,
runtime source sync, scheduler changes, training, or activation was performed.
The three pre-existing untracked readiness/canary paths were preserved.

## Limits and separate findings

This correction does not establish valid strategy comparisons. Same-date return
endpoints can still represent unequal intervals, and the broader evaluation
limitations remain. No historical metrics were regenerated; existing reports
retain their originally calculated beta values.

A separate synthetic check exposed a pre-existing `KeyError` at
`account_returns[account]` when an account has only NULL returns and at least two
matched benchmark dates. That missing-data issue is reported separately and is
outside this normalization-only change.

The Chief reviewed the calculation diff, eleven observable tests, and this report,
then approved scoped publication. Runtime deployment remains outside this task's
authorization.
