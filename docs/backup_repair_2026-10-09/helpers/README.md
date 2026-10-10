# Free local FinRL backups

Existing local encrypted Kopia repository, installed CLI and daily Windows task
are unchanged. This protects accidental deletion/overwrite on this laptop; it
does not protect laptop/disk loss. No subscription, cloud service or broker API.

## Included and held

The daily helper exports the scheduled WSL comparison database with SQLite's
read-only source/native backup and integrity check. It copies WSL logs (excluding
execution_attempts), selected CSV/JSON/HTML/PNG results and root reports, and native
Windows logs. Model files, bulk caches, credentials and other projects are excluded.
Comparison DB export and other ordinary files are not one atomic project snapshot.

**Production journal-era capture is held.** Any journal or sidecar, nonempty
attempt/lock tree, unsafe journal/attempt/lock symlink, missing checkout, or existing
`tools/journal-required.json` stops before publishing a snapshot. First observation
creates a monotonic marker; later deletion of source artifacts does not clear it.
Invalid/unreadable markers also fail closed. Never delete the marker to bypass a
hold. Before any future activation, explicitly record this marker under the
separately approved activation procedure; observation only at backup time cannot
detect a journal created and deleted between runs. No helper supports automatic
production journal capture yet. Existing account locks cannot cover the current
pre-lock attempt creation and new-account race.

`RECOVERY.json` labels new successful production snapshots `pre_journal`, with
`execution_authorized:false`. Absence observations do not prove global quiescence.
The journal-era guard deliberately stops automatic backups after activation until
shared coordination is separately implemented and reviewed. Previous snapshots
remain available. Check the dated `last-run.json` status and task result.

## Restore and evidence

`Restore Latest.cmd` restores the latest complete production-source snapshot into
a NEW timestamped directory under `Restored/`; existing destinations are refused.
It checks exact file inventory/SHA256 and comparison DB integrity. Old snapshots
without RECOVERY.json are labeled `legacy_pre_journal` and never authorize execution.
Journal artifacts without a contract, forged execution authority, unknown contracts,
unsafe paths and mismatched hashes fail verification.

The separate disposable proof uses `sealed_synthetic_journal`: both DBs, exact
journal schema constraints/version/foreign keys, stable IDs/target hashes,
receipt identity/payload/quantities, event/attempt references and immutable evidence
hashes are validated. Unknown/interrupted intents, missing receipts, abandoned
attempts and incomplete per-attempt evidence remain explicit holds after restore.
Restore success means preserved evidence, never execution approval or automatic
resume. Synthetic capture uses a fixture-wide cooperating flock before any fixture
writer creates an attempt, held through export/inventory. Landlock protects test
processes against primary writes and seccomp denies sockets. These controlled
fixture actors do not prove current production coordination or exclude unrelated
same-user writers. Production journal capture remains refused.

## Existing operation and maintenance

`Paxton FinRL Local Backup` runs daily at 20:00 Eastern while signed in, starts
missed runs when available, and does not wake the host. Existing Ubuntu/WSL Python
environment and pinned Kopia CLI are required. At proof time effective retention
is 3 latest, 7 daily, 4 weekly and 3 monthly. No retention policy was changed or
snapshot deleted/expired. Less than 10 GiB free stops staging.

`Backup Now.cmd` invokes the ordinary production backup and remains outside this
repair's verification scope. `last-restore.json` records the most recent successful
ordinary restore. The disposable drill uses task-owned receipts instead. Do not
edit Kopia repository internals or overwrite running data with a restore. A future
incident recovery must review later broker/client IDs and newer evidence first.

The existing restricted `tools/recovery-key.txt` is passed only through Kopia's
child environment, never command arguments or logs. Store a separate safe copy;
its off-device preservation was not verified here. Review receipt/disk space monthly
and do a disposable restore drill every three months. Scheduler, credentials,
gate, service settings and retention were not changed by this repair.
