# Free local FinRL backups

Existing local encrypted Kopia repository, installed CLI and daily Windows task
are unchanged. This protects accidental deletion/overwrite on this laptop; it
does not protect laptop/disk loss. No subscription, cloud service or broker API.

## Included and held

The daily helper exports the scheduled WSL comparison database with SQLite's
read-only source/native backup and integrity check. It copies WSL ordinary logs,
coordinated journal/attempt evidence when required, selected CSV/JSON/HTML/PNG
results and root reports, and native
Windows logs. Model files, bulk caches, credentials and other projects are excluded.
Comparison DB export and other ordinary files are not one atomic project snapshot.

**October 10 shared-lock integration supersedes the initial capture hold.** The
scheduled WSL durable execution/recovery entry points now acquire shared ownership
of `data/execution_capture.lock` before attempt or journal creation, inside the
existing default-disabled gate. Account locks remain inside this global barrier.
Backup acquires exclusive ownership in one WSL process through comparison/journal
SQLite exports, immutable attempt copy/inventory and contract verification. New
account aliases need no lock enumeration. Contention holds without creating an
attempt/journal or publishing a snapshot; existing ordinary diagnostic logging
can occur before the execution barrier. Never unlink lock files or automatically
retry/resume a held run.

The marker stays monotonic. Any journal or sidecar, nonempty
attempt/lock tree, unsafe journal/attempt/lock symlink, missing checkout, or existing
`tools/journal-required.json` requires a valid journal capture. First observation
creates a monotonic marker; later deletion of source artifacts does not clear it.
Invalid/unreadable markers and native journal-era state fail closed. Never delete
the marker to bypass a hold. Before any future activation, explicitly record this marker under the
separately approved activation procedure; observation only at backup time cannot
detect a journal created and deleted between runs. A journal first observed inside
the exclusive capture also records the marker before snapshot publication.

`RECOVERY.json` labels snapshots `pre_journal` or `coordinated_wsl_journal`, always
with `execution_authorized:false`. The guarantee is a journal/attempt cutoff among
cooperating scheduled WSL durable writers; each database export is independently
transaction-consistent. Standalone comparison/backfill/price-sync tools, manual
edits, independent programs and other hosts are outside this protocol. It does
not promise one cross-database cutoff against those writers or atomic ordinary
logs/results. Previous snapshots remain available. Check dated receipts.

## Restore and evidence

`Restore Latest.cmd` restores the latest complete production-source snapshot into
a NEW timestamped directory under `Restored/`; existing destinations are refused.
It checks exact file inventory/SHA256 and comparison DB integrity. Old snapshots
without RECOVERY.json are labeled `legacy_pre_journal` and never authorize execution.
Journal artifacts without a contract, forged execution authority, unknown contracts,
unsafe paths and mismatched hashes fail verification.

Coordinated snapshots and the retained earlier `sealed_synthetic_journal` proof
validate both DBs, journal schema constraints/version/foreign keys, stable IDs/
target hashes, receipt identity/payload/quantities, event/attempt references and
immutable evidence hashes. Unknown/interrupted intents, missing receipts, abandoned
attempts and incomplete per-attempt evidence remain explicit holds after restore.
Restore success means preserved evidence, never execution approval or automatic
resume. Valid unknown/interrupted rows and legitimately incomplete evidence are
preserved with explicit holds. Missing referenced attempts, corrupt databases,
mismatched references and malformed/truncated required JSON refuse a complete
snapshot and require manual review. Every mid-write crash is not guaranteed to
produce a valid complete backup.

Strategy/metrics/RL children inherit the parent's shared descriptor. The RL report
descendant validates canonical inode identity, the actual shared lease and WSL
host before passing it onward. Parent/child cleanup closes descriptors without
unlocking the surviving descendant's open-file description. This transient
descriptor transport grants no execution authority and changes no persistent
environment or activation gate. Disposable tests prove parent/intermediate-child
termination cannot let capture precede the surviving grandchild's final write.
Kernel Landlock denies test-process primary writes; seccomp denies sockets.

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
