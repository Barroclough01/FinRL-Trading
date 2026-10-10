# Backup-only repair: reviewed helpers installed, journal capture held

This report records the earlier helper-only release. Its journal-capture hold is
superseded by the separately authorized October 10
[shared capture integration](../shared_capture_2026-10-10/report.md); historical
proofs and original helper copies remain retained.

Continued October 10 from the October 9 retained draft. This is a **partial recovery
readiness repair**, not new operational journal coverage. Activation remains off.
Only backup helper candidates, README, disposable tests/proofs and this proposal
were changed. No primary workflow/module invocation, broker request, order/cancel,
primary journal initialization/restore, scheduler/service/environment/retention/
credential/history cleanup or memory change occurred during verification.
Chief and independent peer release gates subsequently passed. Three reviewed
helpers and README were installed under the existing backup run lock; exact live
identity and preserved originals are recorded in
[`installation.json`](evidence/installation.json). Git publication is separately
verified against the native current branch after scoped diff/secrets review.

## Candidate behavior

[`helpers/backup_finrl.py`](helpers/backup_finrl.py) preserves the ordinary
comparison export and fails closed before production publication for journal-era
indicators or the monotonic journal-required marker. Both authoritative WSL and
native source paths are checked; missing paths, symlinks and malformed markers hold.
Pre-journal absence observations are labeled and do not assert global quiescence.
The marker must also be deliberately recorded before a future approved activation;
a created/deleted journal entirely between backup runs cannot be observed later.

[`helpers/restore_latest.py`](helpers/restore_latest.py) selects complete snapshots
only whose source matches the existing production `tools/staging/finrl-runtime`.
Synthetic snapshot sources cannot become default production restore candidates.
It restores only into a new directory and delegates exact inventory/hash, DB and
recovery-contract validation to [`recovery_contract.py`](helpers/recovery_contract.py).
Old snapshots restore as legacy_pre_journal. Every contract explicitly denies
execution authority; journal/attempt references without a contract fail.

Journal verification covers schema types/PK/NOT NULL/defaults/unique/FK declarations,
foreign-key rows, SQLite integrity/version, stable session/client IDs and target
hashes, receipt payload/status/quantity/price consistency, immutable evidence hashes
and event/attempt references. Missing receipts, unknown intents, interrupted sessions,
abandoned attempts and incomplete **per-session/per-attempt** evidence remain holds.
No receipt or evidence is synthesized to turn a hold into completed execution.

## Actual proof and limits

[`capture-v1.json`](evidence/capture-v1.json),
[`capture-v2.json`](evidence/capture-v2.json) and
[`cases.json`](evidence/cases.json) record kernel Landlock ABI3 write restrictions,
denied O_WRONLY opens of the primary WSL DB/native runner and seccomp socket denial.
Controlled fixture writers use one nonblocking global flock before attempt creation;
a concurrent writer exits 23 while capture retains the lock through export/copy/
inventory/verification. This proves the controlled fixture protocol, not exclusion
of independent same-user writers or existing production coordination.

The first isolation setup on DrvFS refused even allowed disposable writes. Fixtures
were moved to WSL-native `/home/paxto/.cache/finrl-backup-proof-20261010`;
the correction changed neither operational source nor runtime. Landlock restricts
the test process and descendants, not all processes on the host. Installed Windows
Kopia separately receives only the completed disposable payload; no operational
source imports/invocations are used. Its repository configuration remains the
existing local filesystem repository; network denial applies to synthetic tests,
not a claimed new Windows sandbox for Kopia.

[`kopia-proof.json`](evidence/kopia-proof.json) and raw
[`v1 snapshot`](evidence/kopia-snapshot-v1.json)/
[`v2 snapshot`](evidence/kopia-snapshot-v2.json) record actual installed Kopia 0.23.1
snapshot/restore into new task-owned paths. V1 restores six manifest payload files
and V2 five. Overwritten contents and the deleted-only-in-v2 file recover correctly;
V1 restores again after V2 exists. Existing destination is refused. All ten prior
snapshot IDs survive; the two synthetic snapshots are additional and remain retained.
Old proof versions/fixtures from setup corrections are also preserved.

[`legacy-kopia-proof.json`](evidence/legacy-kopia-proof.json) records actual snapshot
`6c41793089ccdc0bfbbfc05ce81c4b3b` restored into a new disposable directory: 387
file hashes and comparison DB integrity pass, classified legacy_pre_journal with
execution_authorized false. The ordinary production backup was not invoked.

[`final-byte-validation.json`](evidence/final-byte-validation.json) is authoritative
for final reviewed helper hashes and verifies both synthetic copies, repeated V1
and legacy restore after the final helper edits. Earlier kopia-proof hashes describe
the snapshot-time draft; final edits are formatting/unused-import removal and the
additional unsafe marker refusal. The same final bytes passed the isolated cases,
syntax compilation of all seven Python files and installed Ruff with configured
rules. Ty is absent from the configured WSL environment and no dependency installed.
No full operational project suite was run for these standalone external helpers.

[`effective production retention`](evidence/retention-effective-production.json)
and [`synthetic retention`](evidence/retention-effective-synthetic.json) each show
3 latest, 7 daily, 4 weekly and 3 monthly. No policy/expiry/delete operation ran.
Full raw before/after inventories are retained under evidence.

## Preservation and staged deployment

[`preservation-comparison.json`](evidence/preservation-comparison.json) compares
fresh October 10 native/WSL runtime inventories, source HEAD, gates, task XML/state,
external helpers/README/config/restricted key hashes/wrapper and backup/restore
receipts. All are unchanged during this drill. The before state includes naturally
advanced October 9 weekly legacy evidence and October 10 catch-up backup receipt;
it is not forced back to the October 9 readiness inventory. Dated source probes
are not broker/fill health checks. Existing primary journal/attempt paths remain absent.

After both gates pass, preserve exact installed originals in a new dated external
directory, then install only the three reviewed helper files and replace the external
README with the reviewed candidate. Verify live SHA256 equals final-byte-validation,
retain all old helper copies/snapshots/receipts and confirm task/config/wrapper/gates
unchanged. Do not invoke production backup to make a green receipt. Existing scheduler
will use the reviewed bytes on its next normal run. Record this file installation
separately from Git publication; it is not a Git deployment.

Commit/push only this scoped repair directory on the existing current native branch
after release. Leave the previously untracked readiness proposal alone. Verify the
remote branch resolves to the new commit and local upstream relation is 0/0. WSL
source synchronization is unnecessary: operational source is unchanged. No VPS.

Existing repository attributes normalize published Python blobs to LF. Tested
working copies and the installed external helpers retain the recorded CRLF bytes;
`final-byte-validation.json` and `installation.json` identify those exact bytes.
Git normalization changes line endings, not executable content. Exact installed
originals remain in the external dated history directory; no attributes/config
were changed to bypass the established repository policy.

## Residual decision

Production journal captures are deliberately refused. The current writer creates
attempts before account locking; new accounts can appear during lock enumeration.
Neither existing account locks, process-name observations nor the synthetic fixture
proof repairs this. The exact minimal next decision is
[`shared_barrier_proposal.md`](shared_barrier_proposal.md): add a cooperating WSL
global barrier before every attempt/journal write and use the same exclusive barrier
through backup export/inventory, with race/error-path tests. This requires separate
operational-source authorization before considering a paper-only canary.
