# Invalid metric evidence and export repair — October 23, 2026

Status: Chief-reviewed and authorized for scoped commit/push. Baseline:
`a10c312fa722aabc0728930cc810c3e390452566` on
`codex/paper-comparison-reliability`, native development checkout
`C:\Users\paxto\FinRL-Trading`.

## Behavior

The public calculator previously divided by an initial zero portfolio value,
crashed on NULL NAV, and could retain infinite account returns in standalone
statistics. The exporter accepted bare JSON `Infinity`/`NaN` and wrote JSON
before discovering CSV serialization errors. The report-only entry point also
wrote the dashboard before discovering invalid comparison evidence.

The calculator now refuses an initial zero return basis with an explicit
`ValueError` naming account, source date and `portfolio_value`. Public
`record_snapshot()` still accepts and retains a legitimate empty zero-equity
account. Reporting refusal concerns undefined cumulative percentage return; it
does not label the capture a broker failure, rewrite the snapshot, choose a
replacement statistic, or drop the account. A positive basis can still end at
zero (defined cumulative return -100%) or pass through zero. The pre-existing
cash-weight zero sentinel for NAV <= 0 is unchanged.

Loaded required NAV/cash values must be finite numbers. Loaded returns,
cumulative returns, weights and benchmark closes reject nonfinite or nonnumeric
values with account/benchmark, date, field and symbol context where available.
Finite inputs that overflow NumPy statistics receive contextual calculation
errors; calculated statistics are checked for finite output. Existing missing
and nonpositive benchmark-close handling, NULL-return conversion for standalone
statistics, missing-pair zero defaults, and NULL-weight conventions remain.

This is a calculation/export boundary, not universal raw-input quarantine.
Capture and database schema are unchanged. SQLite can normalize Python NaN to
NULL; loaded optional NULL returns/weights therefore retain the existing
sentinels. A future unavailable/missing-data representation policy (F7) still
requires a separate decision. No calendar, annualization, risk-estimator,
strategy, return-interval or report-schema policy was changed.

Export recursively checks numeric report contents and uses
`json.dumps(..., allow_nan=False)`. Both JSON and CSV are completely serialized
in memory before either destination opens. Invalid numeric evidence,
unsupported JSON types or malformed CSV structure therefore leave last-good
JSON/CSV bytes unchanged, or leave both files absent. Calculation validation
also precedes CLI/dashboard generation in `main()`, preserving the dashboard
when source evidence is refused. This guarantee covers invalid evidence and
serialization; arbitrary disk/I/O failures or process crashes during successful
publication are not a multi-file transaction and remain a separate concern.

## Verification

- Baseline reconstruction from native Git reproduced **14 failing cases** and
  **2 passing later-zero cases**: initial zero division, unsupported/nonfinite
  returns, finite-input overflow, JSON-only/partially overwritten exports after
  serialization failure, and dashboard publication before calculation failure.
  This baseline run deliberately used normal warning behavior.
- **56 new behavior/error-path cases** pass. They cover contextual required and
  optional input refusal, weights and benchmarks, derived overflow, unchanged
  positive-basis histories reaching/passing through zero, public zero capture,
  strict JSON/CSV round trip, nested nonfinite export fields, serialization
  failures, and existing/absent last-good report sets.
- Full configured `tests/` suite: **290 passed**, with RuntimeWarnings treated
  as errors. All generated test outputs use disposable directories.
- Reused the prior independent synthetic oracle against the final native
  source: **320 seeded scenarios / 20,919 scalar assertions**, **2,048
  exhaustive missing-observation masks / 26,624 scalar assertions**, and
  **5 oracle self-checks** pass. These are scenarios/assertions, not additional
  pytest test counts. The seeded suite includes 40 JSON/CSV round trips. Its
  former zero/NULL/infinite boundary failures now produce contextual refusal.
- Ty passes on production and new test files. Ruff passes on the new tests;
  production's **32 findings** match baseline by code, message and source text,
  with zero introduced findings. Native `git diff --check` passes.

The existing WSL interpreter
`/home/paxto/stock-trading/FinRL-Trading/finrl-env/bin/python` imported asserted
native development sources from `/mnt/c/Users/paxto/FinRL-Trading`. Dotenv loading
and bytecode were disabled; credentials were removed from the validation process
and real outbound socket connections blocked. No dependencies were installed.
Disposable reproduction/check scripts, outputs, source binding/hash and the
strict completion receipt are under
`C:\Users\paxto\AppData\Local\Temp\finrl-invalid-metrics-20261023`.

No broker/provider/notification calls, operational data or metrics regeneration,
runtime synchronization/deployment, backfills, training, activation, schedules,
credential changes or backup operations were performed. The three pre-existing
untracked readiness/canary paths remain outside the diff. Their preservation is
based on no writes to those paths and unchanged status, not a prior byte-level
hash attestation. This is development-source evidence, not current operational
or strategy performance evidence.

The Chief independently checked the final source hash, full-suite log, oracle
and missing-observation receipts, and zero-introduced-Ruff comparison, then
passed the three-file publication gate. Runtime synchronization/deployment
remain outside this authorization.
