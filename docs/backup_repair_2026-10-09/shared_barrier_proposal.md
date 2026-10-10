# Separate decision: production journal capture barrier

Historical proposal, now superseded by the separately authorized October 10
[shared capture implementation](../shared_capture_2026-10-10/report.md).
Paper activation remains off; this does not authorize a paper canary.

Add one WSL-wide OS flock at `data/execution_capture.lock` in the authoritative
scheduled checkout. Durable execution and recovery acquire shared ownership,
nonblocking, **before attempt creation or journal initialization** and retain it
through every journal mutation and final immutable evidence write. Backup acquires
exclusive ownership, nonblocking, before checking required artifacts and holds it
through both SQLite exports, attempt-tree inventory/copy and reference validation.
Conflicts return an actionable hold and never create an attempt or publish a snapshot.
Leave existing per-account locks inside this global barrier. New account aliases
use the same global path, so enumeration is unnecessary. Lock ordering is always
global, then account; OS process exit releases ownership, never delete lock files.

Concrete integration points: `src/trading/execution_journal.py` provides the small
context manager; `run_paper_trading.py` wraps the enabled main path before its
existing main-level `new_attempt` and the callable `execute_account` enabled route
before fallback attempt creation; `recover_paper_execution.py:recover_account`
wraps its enabled route before `new_attempt`, keeping journal close/final evidence
inside ownership. Verify all `new_attempt`, `ExecutionJournal` and `write_evidence`
call sites; prevent nested exclusive ownership, and require explicit ownership
for future production writer entry points. Tests may use synthetic roots under
their existing isolation, without changing the production host/gate restrictions.

The native backup helper delegates the guarded export/copy interval to one WSL
process so its descriptor stays alive through capture; hash/Kopia stages operate
on the finished disposable staging tree afterward. No process-name scan or list of
account locks substitutes for this lock. Marker `tools/journal-required.json` is
recorded before any separately approved activation; thereafter a missing journal,
missing references or invalid contracts always hold. Do not infer readiness from
an empty database or restore old pre-journal state over newer intents.

Required observable tests: writer-before-lock attempt creation prevented; writer
already holding the global lock blocks capture; capture blocks a concurrent writer
and a newly appearing account before either creates an attempt; process interruption
releases the kernel lock but preserves unknown/interrupted evidence; held/missing
reference/corrupt journal refuses snapshot publication; actual isolated Kopia restore
retains both database/attempt histories and explicit holds. No broker network or
primary writes in tests. Review lock lifetime through error and cleanup paths.

This coordinates cooperating WSL writers in this checkout. Independent programs,
other hosts and manual raw SQLite/filesystem edits remain outside the guarantee.
No source changes, task/gate/environment changes, canary, broker GET/order/cancel,
credential change, VPS or persistent paper activation are authorized by this proposal.
