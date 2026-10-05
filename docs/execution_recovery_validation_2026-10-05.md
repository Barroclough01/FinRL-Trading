# Durable paper recovery source validation

Approved contract: [2026-10-04 decision](execution_recovery_proposal.md).
Validated 2026-10-05 against clean baseline
`8569d25cccc4337272e46c0dbbff562e5c4ea56c` on
`codex/paper-comparison-reliability`. Source publication and WSL source
synchronization are authorized; operational activation remains excluded.

## Results and isolation

The full configured suite passes: **136 baseline tests; 192 candidate tests**.
The 56 new acceptance tests exercise journal/crash/recovery and actual weekly
integration. Earlier full candidate run had two existing fixture-path seams;
the audit-path helper was corrected to preserve the original Path lookup and
the final full run passes. A cross-CWD disabled-output regression also confirms
target files remain rooted in the project directory. These are compatibility
corrections, not relaxation of the tests or financial gate.

All verification ran in disposable tracked-source copies at
`/home/paxto/.cache/finrl-full-recovery-20261004/{baseline,candidate}`, using the
existing scheduled checkout's `finrl-env/bin/python` as interpreter only.
No tests ran in either operational checkout. An external import barrier disables
dotenv, removes broker/webhook credentials and blocks real socket, HTTP-request
and subprocess calls. Fake method/request adapters override only their tested
boundaries. Real fork/SIGKILL exercises disposable child processes with fake
broker acceptance, never a production process or network action.

| Check | Result |
| --- | --- |
| Full baseline/candidate pytest, no cache writes | 136 / 192 passed |
| New module, CLI and acceptance tests Ruff/Ty | Pass |
| Touched legacy source Ty with configured interpreter | Baseline and candidate clean |
| Touched legacy source full Ruff | 85 baseline / 85 candidate; no introduced diagnostic signatures |
| Critical Ruff E9/F63/F7/F82 | Pass |
| Markdown UTF-8/local references and final diff check | Pass |
| Barrier probe | dotenv disabled; credentials absent; socket/request/subprocess blocking verified |
| Actual operational activation state | Native/WSL process and both `.env` gates unset; wrapper assignment absent |
| Before/after protected runtime inventories | Exact match: 15 native files, 665 WSL files, external wrapper |

Ruff/Ty are the existing `/home/paxto/.local/bin/` tools, with no installation
or dependency added. Full Ruff's legacy findings are reproduced on unchanged
source and outside this repair; critical checks and all new files are clean.

## Approved acceptance mapping

Names below refer to `tests/test_execution_journal.py` (journal) and
`tests/test_journal_integration.py` (integration). Existing pending-order and
final-outcome suites remain part of the full configured run.

| Approved case | Observable proof |
| --- | --- |
| First clear Friday run | `test_intent_and_unknown_commit_precede_the_only_post` opens a second SQLite reader during fake submission and observes committed exact payload/unknown state; integration first-run case preserves DAY and routes its actual audit targets |
| Prior pending/partial/inherited orders | Existing pending-order matrix plus integration failed-account case: no cancel/replacement; prior IDs remain |
| Lost acceptance | Crash-boundary accepted case discovers the same client ID via GET; no second POST; interrupted remaining work stays held |
| Process interruption around intents/POST/receipt/export | Five crash-boundary cases freeze/mark/accept/observe/complete and reopen; actual `test_sigkill_after_fake_acceptance_releases_lock_and_recovers_same_id` kills the writer after fsynced fake acceptance but before receipt, then verifies SQLite integrity, same unknown intent, GET recovery and no resume |
| Completed terminal same-session repeat | Accepted/partial/filled/canceled/rejected/expired matrix observes receipts with one session/one intent and no replacement; integration repeat preserves original decision DB bytes and all prior attempt files |
| Concurrent writers/aliases | Actual forked contender cannot acquire the account lock; SIGKILL releases the owner lock; alias observation shares the canonical session rather than duplicating it |
| Alias account/endpoint retarget | Binding and exact-endpoint tests fail before another session/event mutation or transport |
| Partial execution across accounts | Mixed prior/new integration only snapshots/parity-persists new AR, does not repeat FinRL and skips offline-history regeneration; failed/unknown-account cases retain final failure |
| Changed targets/config/external positions | Frozen-input and binding matrix refuse; expected quantities include cumulative legitimate fills and unexplained position drift holds without POST |
| Read/not-found/unknown/corrupt/disk/receipt errors | Recovery matrix, SQLite read-only write failures, schema/corruption cases and actual strict clock/asset/price/account helper cases stop submission/phases; post-sell zero/nonfinite value cases hold |
| Known pending/partial DAY | Receipt matrix retains filled quantity/price and pending classification; real enabled parity pending case suppresses `ok` with unchanged comparison DB |
| Audit/recovery repeat | Original acceptance remains in timestamped events after later partial/terminal receipts; immutable attempt files cannot overwrite; first/repeat integration preserves prior files and DB; real mismatch fails through final parity without history writes |
| Offline/native/default-disabled | Unset/false/empty gate uses existing contract without journal; enabled native/wrong checkout/endpoint refuses; existing RL/chronology/parity suites still pass |

The real enabled parity test covers both pending validity and a replay mismatch:
the mismatch returns exit 1 and `failed`; recovery-only observations never send
`ok`. All-held identity/corrupt/binding/config cases invoke no metrics/RL/sanity
history tail. Current immutable target selection is verified against a deliberately
stale legacy execution log; only the selected fresh account receives current targets.

Original roadmap fill-trail acceptance is also covered: timestamped append-only
accepted/partial/filled/canceled/rejected observations are obtained by the separate
GET-only command using existing broker/client IDs. Original receipts remain;
no submit/cancel or comparison-history rewrite is part of that command. No legacy
receipt/history migration or operational observation was performed.

## Independent review and retained evidence

Chief and its read-only reviewer independently inspected financial control flow,
identity/locks, error holds, exact requests, history suppression and final gates.
The Chief additionally reproduced transport behavior below `requests.request`
using installed Requests and a fake BaseAdapter: legacy 307/308 redirects send
two POSTs; durable redirects send one and are refused. There was no network.
The owner suite also verifies durable redirect refusal, GET retry preservation
and genuine SIGKILL recovery. Independent selected redirect/SIGKILL tests passed.

Retained evidence directory:
`/home/paxto/.cache/finrl-full-recovery-20261004/`. Files include:
`baseline-full.txt`, `candidate-final-full.txt` (192 passed),
`new-final-ruff.txt`, `new-final-ty.txt`, `critical-final-ruff.txt`,
`final-diagnostics.txt` (85/85 unchanged), baseline/candidate Ruff JSON and
Ty text, `barriers-verified.txt`, `activation-gate.json`, `before.json`,
`after-tests.json`, and Chief's `chief-transport-proof.json`/proof script.
Final source/remote relation and after-release inventory are retained as
`release.json` and `final.json` after the source release. Git history identifies
the scoped source commit; no operational journal was initialized by validation.

Protected manifests contain paths, sizes, nanosecond modification times and
SHA-256 for native and WSL data/logs/results/models/caches/`.env`, including
databases and present sidecars, plus the external scheduler wrapper. Source
sync must remain a clean WSL-native fast-forward with no active workflow and
the activation gate confirmed unset/false. No authenticated broker call,
actual paper/real order, cancellation, scheduler/credential change, history
remediation or operational entrypoint was used.

## Activation and residual limits

The [operations contract](execution_recovery.md) documents separate activation.
Verify journal and immutable-attempt backup coverage before activation; the
existing Kopia utility predates these paths and was not changed. Local locks
coordinate this host, not independent manual/cross-host broker writers. Fake
process death validates local durability; physical storage must honor SQLite/OS
flushes for power-loss durability. Cash/valuation/settlement movements are
observed without a full cash/fee attribution model. There is no automatic resume,
legacy-history migration or exactly-once broker claim.
