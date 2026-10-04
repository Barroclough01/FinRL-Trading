# Open-order rebalance guard validation

Verified 2026-10-04 against baseline
`02d966d7524350a1dbfdb1c25a3889da72657a68` on
`codex/paper-comparison-reliability`. This record covers source behavior and
runtime preservation, not current broker health or later fill reconciliation.

## Scoped behavior

The live rebalance previously canceled all open orders before planning. It now
reads open orders for the resolved account and raises the existing `ValueError`
on a nonempty list, unreadable state or malformed response. No automatic cancel
or replacement occurs. The existing orchestrator retains its final failure
audit/parity sequence and does not replace the strategy decision on this hold.
Preview/closed-market skip remain plans; clear-account next-open submissions
still use DAY. No strategy, target weights, metrics chronology or final-outcome
schema changed.

## Isolated checks

All Python checks used disposable tracked-source copies at
`/home/paxto/.cache/finrl-execution-safety-20261004/{baseline,candidate}`, with
the existing `/home/paxto/stock-trading/FinRL-Trading/finrl-env/bin/python`.
No test ran in either operational checkout. The import barrier disabled dotenv,
removed APCA/ALPACA and webhook environment entries, and prohibited real socket,
HTTP-request and subprocess boundaries. Broker behavior used fake methods only;
an explicit barrier probe passed. Existing test mocks override only their fake
boundaries. No credential values were read or printed.

| Check | Result |
| --- | --- |
| New regression tests on unchanged baseline | 17 failed, 3 passed; baseline cancels/submits instead of holding |
| Candidate guard/final-outcome/GET-retry tests | 49 passed |
| Unchanged baseline full configured suite | 116 passed |
| Candidate full configured suite | 136 passed |
| Final guard tests after message formatting | 20 passed |
| New test file Ruff and Ty | Both pass |
| Touched legacy manager/workflow Ruff | 79 baseline, 78 candidate; counter comparison by file/rule/message confirms no introduced finding |
| Touched legacy manager/workflow Ty | 18 baseline and candidate; identical diagnostic rule/message multiset |
| Critical Ruff E9/F63/F7/F82 on touched Python | Pass |
| Touched Markdown UTF-8/local links; diff check | Pass |

Coverage includes accepted, partially filled and pending-cancel orders in open,
next-open and OPG modes; default-account resolution; failed/malformed reads;
preview and closed skip; clear-account DAY behavior; and actual account runner
hold propagation through the final failed outcome without decision replacement.
Legacy diagnostics are outside this scope and were reproduced on the baseline.
Installed Ruff/Ty are `/home/paxto/.local/bin/{ruff,ty}`; they are not present in
`finrl-env/bin`. Ty used `--python` pointing at the configured environment.

Commands were run from each disposable copy, with `PYTHONDONTWRITEBYTECODE=1`
and `PYTHONPATH` pointing at the external barrier directory:

```bash
finrl-env/bin/python -B -m pytest -p no:cacheprovider
finrl-env/bin/python -B -m pytest -p no:cacheprovider \
  tests/test_rebalance_pending_orders.py tests/test_final_outcome.py \
  tests/test_alpaca_read_retries.py
ruff check tests/test_rebalance_pending_orders.py
ty check --python <configured-finrl-python> tests/test_rebalance_pending_orders.py
ruff check --output-format=json src/trading/alpaca_manager.py tests/test_weekly_workflow.py
ty check --python <configured-finrl-python> src/trading/alpaca_manager.py tests/test_weekly_workflow.py
ruff check --select E9,F63,F7,F82 src/trading/alpaca_manager.py \
  tests/test_rebalance_pending_orders.py tests/test_weekly_workflow.py
```

Here `finrl-env/bin/python` in the command display denotes the absolute existing
configured interpreter above; disposable copies do not contain an environment.

## Preservation and release boundary

Before/after-tests inventories retained relative paths, byte counts, nanosecond
modification times and SHA-256 for native and WSL `data`, `logs`, `results`,
`models`, `trained_models`, `cache` and `.env` where present. They contain 15
native runtime files and 665 WSL runtime files, including operational SQLite
databases and present sidecars. The external scheduler wrapper hash/mtime was
included. All before/after values match. No test touched runtime history.

Detailed outputs, barriers and inventories are retained outside the operational
repositories at `/home/paxto/.cache/finrl-execution-safety-20261004/`:
`before.json`, `after-tests.json`, `baseline-regression.txt`,
`baseline-full.txt`, `candidate-full.txt`, `candidate-targeted.txt`,
`candidate-final-targeted.txt`, `barriers.txt`, baseline/candidate Ruff JSON and
Ty text. A final inventory is required after publication/source synchronization.

Only scoped source/tests/docs may be committed and pushed. Synchronization of
the existing scheduled WSL checkout must verify no active workflow process,
clean Git status, same branch and fast-forward relation, then use WSL-native
Git with `--ff-only`. No operational command, authenticated broker read,
order/cancellation, journal activation, history repair, schedule or credential
change is authorized in this release.

## Limits and pending decision

This closes destructive pending-order replacement only. No-open-order state
does not prove the session was never executed. Lost acceptance, process death
before decision persistence, concurrent writers and different account aliases
can still bypass full session-level repeat safety. Prior same-date execution
summaries can be overwritten by a refused rerun. The order list is a broker
observation, not a transaction lock.

The [durable recovery proposal](execution_recovery_proposal.md) is reviewable and
pending separate approval. It recommends a separate journal, broker-account
session identity, WSL single-writer lock, pre-POST client IDs, immutable attempt
evidence and broker-read-only recovery, with default-disabled activation and
automatic resume deferred. No such new interface or model is implemented here.
