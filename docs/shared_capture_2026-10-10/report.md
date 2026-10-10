# Shared execution/capture barrier, October 10

Implementation and isolated proof are ready for Chief/peer release gates.
Nothing has been committed, pushed, synchronized to WSL or installed by this
integration yet. Paper activation remains OFF. This is the separately approved
successor to the helper-only repair; it creates no primary journal, attempt or
capture lock during verification and invokes no broker/ordinary production run.

## Behavior and exact guarantee

Enabled scheduled WSL main, direct `run_account` and direct `recover_account`
acquire shared nonblocking `data/execution_capture.lock` ownership before attempt
creation, journal initialization or broker identity reads. Main retains ownership
through its metrics/offline-RL/parity/final-write tail and cleanup. Account locks
stay inside global ownership. Production low-level attempt/journal/evidence/conn
operations require explicit ownership; disposable low-level fixtures retain
their established isolation. The host/checkout gate and paper-only identity
contract remain strict. Default-disabled/legacy execution creates no new lock.

All main subprocesses inherit the shared descriptor. The RL report descendant
validates canonical lock inode, actual shared lease in `/proc/self/fdinfo` and
WSL host before propagating it. Transient descriptor transport does not grant
execution authority or change persistent environment/gate settings. Cleanup
closes descriptors without `LOCK_UN`, so surviving descendants retain ownership
after parent/intermediate-child termination until their final descriptor closes.
No process enumeration substitutes for this lease.

Native backup uses one WSL `capture_runtime.py` process and one exclusive
descriptor through comparison/journal exports, attempt copy/inventory, contract
generation and restore/reference validation. Hash/Kopia stages use the completed
staging tree. Required journal evidence is selected and checked inside ownership.
An initially absent journal observed during capture makes the journal-required
marker monotonic before publication. Native journal-era state, unsafe/invalid
markers, unavailable sources and invalid required evidence hold.

The guarantee is a common journal/attempt cutoff among cooperating durable WSL
writers. Each SQLite export is independently transaction-consistent. Standalone
comparison/backfill/price-sync tools, manual edits, independent same-user programs
and other hosts are outside this protocol; no single cross-DB/project-wide cutoff
is claimed against them. Ordinary logs/results are not atomic snapshot artifacts.
Contention creates no execution attempt/journal and publishes no snapshot;
existing ordinary import-time diagnostic logging and disposable backup staging
are outside that claim. Lock files are never removed; no automatic retry/resume.

Valid unknown/interrupted intents, missing receipts, abandoned attempts and
legitimately incomplete external evidence are preserved as explicit holds in
`coordinated_wsl_journal`, with `execution_authorized:false`. Missing referenced
attempts, database corruption, reference mismatch and malformed/truncated required
JSON refuse a complete snapshot and require manual review. Every interruption
is not guaranteed to yield a complete recoverable snapshot. No evidence or receipt
is synthesized and restore never permits automatic execution or recovery.

## Verification

- [Final configured suite](evidence/full-suite-final.txt): **216 passed**, including
  24 added global-barrier cases in a complete tracked-source disposable WSL copy,
  fake `.env` and isolated runtime paths. No primary project module was invoked.
  Proof source: [launcher](proof.py), [tests](../../tests/test_capture_barrier.py).
- Actual Landlock ABI3 rejects writes to primary WSL DB/native runner; seccomp
  rejects socket creation. All test descendants inherit these restrictions.
  Allowlist is task scratch plus `/dev` runtime devices. First harness refused
  pytest's `/dev/null` open; this test-only correction is retained in
  [initial output](evidence/targeted.txt). No host/global sandbox is claimed.
- Contended main/direct-account/recovery and new aliases create no attempt/journal
  or account lock. Nested error/final-write paths retain ownership; capture
  checkpoints prove exclusion at ownership, both exports, attempt copy and final
  validation. Stale/wrong-inode/unlocked inherited descriptors reject.
- [Actual crash receipt](evidence/crash-proof.json): real fixture journal records
  an `attempted_unknown` intent before SIGKILL of parent and intermediate child.
  The surviving grandchild blocks capture until its final evidence write and
  exit. Unknown intent and interrupted session remain explicit capture holds.
- [Final captures](evidence/capture-proof-final.json) use the exact candidate
  helper bytes; [actual Kopia proof](evidence/kopia-proof.json) creates snapshots
  `122c1cd4af606154cbfe1285aed6e5db` and
  `0fcccce3f68df771e52e41338d04ecdb`. V1/V2 and repeated V1 restore verify
  6/5/6 payload hashes, both DB histories, empty abandoned/interrupted directories,
  overwrite/deletion history and explicit holds. Existing destinations reject;
  all 12 earlier snapshots survive (14 now). Retention remains 3 latest/7 daily/
  4 weekly/3 monthly; no policy/delete/expiry operation occurred.
- [Ruff](evidence/ruff-final.txt) passes new/changed support files. Existing runner
  findings remain exactly baseline, with
  [zero introduced diagnostics](evidence/ruff-baseline-comparison.json).
  [Ty](evidence/ty-final.txt) passes operational source, new tests, capture and
  contract helper using installed Ty 0.0.32 found at its explicit WSL path.
  Windows backup helper has four `msvcrt` member diagnostics even with Windows
  platform override; the [same baseline](evidence/ty-windows-baseline.txt) has
  exactly those four diagnostics. No tool was installed.
- [Final bytes](evidence/final-byte-validation.json) bind nine runtime/test files
  to exact tested source-copy bytes and canonical LF hashes. Three proof scripts
  also parse successfully. Kopia's four code-helper identities still match;
  README was updated afterward only for current capability/limitations.
  Git normalizes Python to LF; tested/live helper physical hashes are separate.
- [Preservation](evidence/preservation-comparison.json): fresh 16:50–17:05 UTC
  inventories retain all 18 native/267 WSL runtime entries, source HEADs, gates,
  task XML/state, wrapper, restricted credential/config hashes and backup/restore
  receipts. External installed helpers remain unchanged pending release.
  Supplementary post-test probes explicitly confirm primary capture lock,
  journals/attempt roots and external journal-required marker are absent. The
  original before-inventory helper did not enumerate capture-lock absence; the
  kernel negative controls enforce no test-process primary writes throughout.

## Release plan and remaining activation boundary

After both gates, publish only this scope on native current branch
`codex/paper-comparison-reliability`, preserving the unrelated untracked readiness
document. Verify exact live origin commit and upstream 0/0. Check scheduled WSL
source is clean, default-off and has no active conflicting operational process;
fetch and fast-forward source only from existing `5ec847c` through the verified
native commit. Verify tracked canonical identities and ignored runtime inventories;
never copy native fake credentials/runtime into WSL.

Only after authoritative WSL writer source is current, acquire the existing native
backup run lock, preserve exact installed originals in a new dated helper-history
directory, and install reviewed `backup_finrl.py`, `capture_runtime.py`,
`recovery_contract.py` and README. Leave unchanged restore helper, repository
config, keys, retention, scheduler, wrapper, receipts and all older histories.
Verify exact physical installed hashes equal final tested bytes, canonical
published relation and continued gate/runtime preservation. No ordinary backup
or broker workflow is used to make a green receipt. No VPS deployment.

Activation still needs a separately approved paper-only activation/canary
procedure, deliberate journal-required marker, canonical paper identities,
review of unresolved/legacy evidence and isolated activation prerequisites.
This integration neither activates the gate nor authorizes orders, cancellation,
primary recovery, scheduling changes or real-money actions.
