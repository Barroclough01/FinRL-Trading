# Durable paper execution and receipt recovery

Approved 2026-10-04, implemented in source and **disabled by default**.
`PAPER_EXECUTION_JOURNAL_ENABLED` unset, empty or `false` retains the existing
guarded workflow. Only explicit `true` enables the new path; invalid values fail.
Source publication does not initialize an operational journal or activate orders.
No automatic resume, order replacement, cancellation or additional schedule exists.

## Authority and identity

The enabled path requires the established WSL operating system and exact
scheduled checkout `/home/paxto/stock-trading/FinRL-Trading`. Native Windows and
the WSL-mounted native development checkout cannot execute/recover this path.
The broker endpoint must equal `https://paper-api.alpaca.markets`; the account's
GET response must supply its canonical ID. RL remains offline.

`data/paper_execution_journal.sqlite3` is independent of the historical
comparison database. SQLite schema version 1 contains execution sessions,
order intents, append-only timestamped events and persistent account-alias
bindings. The binding table prevents a familiar alias from silently changing
accounts. Sessions are unique by endpoint/broker ID/NYSE signal date. Config,
canonical decimal targets and algorithm version are frozen; changed inputs hold.
Existing same-date legacy decisions/audits block cold adoption rather than being
silently converted into fresh authority.

An account-ID OS lock under `data/execution_locks/` spans execution or recovery.
Database uniqueness also applies. Another local writer fails; process death
releases the OS lock but never authorizes resubmission. These locks coordinate
cooperating writers on this host only. They do not lock the broker against
manual trades or another machine.

## Submission and interruptions

Each sell/buy phase freezes exact requests before its first POST. Quantities are
canonical decimal text; deterministic `fr-<40 hex chars>` client IDs survive
process restarts. SQLite uses synchronous FULL transactions and commits
`attempted_unknown` before each POST. No POST retry or HTTP redirect is allowed.
Receipt identity, quantity, type, time in force, status and fills are validated
before updating current receipt state; each observation remains an immutable
event with a UTC timestamp. Rejects, unknown responses and write failures stop
later orders/phases. Partial fills retain filled quantity/price separately.

Buy requests are frozen after the existing post-sell state/cash calculation.
Known pending Friday sells retain DAY queuing and existing buying-power constraints.
Unknown sells prevent advancing to buys. Enabled execution refuses unreadable or
malformed market clocks, asset eligibility/fractionability, prices, portfolio
values and buying power; it never interprets unavailable data as inactive assets,
closed market, cash or an invented price. Verified inactive assets still follow
the existing exclusion rules.

A process can die after committing unknown intent but before actually sending.
It can also die after acceptance but before recording the receipt. Both cases
remain conservative holds. Lost response recovery may discover the accepted
order, but an interrupted session still does not resume remaining phases or
unattempted intents. Completed same-date reruns observe receipts rather than
creating new orders, including after terminal fills/cancellations.

## Broker-read-only recovery

After separate operational authorization and activation, the explicit interface is:

```bash
finrl-env/bin/python recover_paper_execution.py --date YYYY-MM-DD --account FinRL
```

The date is the original signal session, including a Friday whose DAY orders
queued for the next session. The command only GETs canonical identity, existing
orders by broker/client ID and current account/positions. It appends local
receipts and immutable attempt evidence; it never POSTs, DELETEs, resumes or
rewrites comparison history. Missing journal/session, not-found/read failures,
changed identities, mismatched payloads, unknown outcomes or unexplained
position drift return a hold/nonzero result. Current position quantities must
equal frozen pre-run quantities plus legitimate cumulative journal fills.
Cash/equity/buying power are recorded as observations; valuation and settlement
changes are not converted into permission to resume. The command has no resume
flag and no success webhook.

## Audits and weekly history

Enabled invocations get unique `logs/execution_attempts/<uuid>/` directories.
Targets, validation, execution, decisions, reconciliation, parity and receipt
observations reference the attempt/session and cannot overwrite earlier attempts.
Decision exports are exclusive and fsynced. Strategy audit suffixes also include
the attempt ID. Original decisions/receipts remain evidence even when later
export/metrics writes fail.

Metrics receive the current immutable execution audit explicitly and record only
new-session accounts. Missing/date-mismatched audit targets fail. Repeated accounts
are excluded from snapshot and parity persistence; an all-held or recovery-only
run skips metrics/offline-RL/sanity history mutation. A mixed prior/new run may
snapshot only new accounts and skips the offline-RL history tail. Replay/parity
checks and immutable audit writes still gate the final result. Known pending
orders remain unreconciled and suppress `ok`; failures still produce `failed`.
Recovery-only observations never send `ok`, even after all orders become terminal.
Legacy default-disabled behavior and chronological metrics remain unchanged.

## Activation prerequisites and limits

Activation is a separate operational decision. Before enabling, verify paper
account/alias identities, supported WSL host/checkout, no competing writer,
startup date and any legacy same-session evidence. Verify backup coverage for
both the execution journal (using a consistent SQLite snapshot) and immutable
attempt directories, independently of the comparison DB. The existing Kopia
utility predates these paths; this release does not change its configuration or
schedule and does not claim coverage. Include lock-directory handling in recovery
procedures; never delete lock files while writers may be active.

The guarantee is durable local intent with conservative holds, not exactly-once
broker execution. SQLite/OS fsync relies on storage honoring flushes; fake crash
tests demonstrate process-death recovery, not every power/disk failure. Broker
receipts can be unavailable or lag current positions, producing safe false holds.
External cash changes are observed, not fully attributed to fees/settlement.
Cross-host coordination, automatic resume and legacy-history migration are outside
this contract. Native/WSL operational gate remains disabled in this source release;
no authenticated recovery, actual paper order or cancellation was used to validate it.
# October 10 capture coordination

Enabled scheduled WSL execution/recovery now acquires shared ownership of
`data/execution_capture.lock` before attempt/journal creation and retains it
through final writes and cleanup. Account locks are inside this global barrier.
The backup helper uses exclusive nonblocking ownership through both transaction-
consistent SQLite exports, attempt inventory/copy and recovery validation. A
conflict holds and requires explicit review/retry; never unlink the lock inode.
Children and the RL reporting descendant retain inherited shared ownership after
parent termination. Descriptor transport never grants execution authority.

This covers cooperating durable writers and the enabled main's comparison tail,
not standalone comparison/backfill/price-sync tools, manual edits or other hosts.
Valid unresolved evidence is captured with holds; corrupt/missing/mismatched or
truncated required evidence refuses a complete snapshot. See the dated
[implementation proof](shared_capture_2026-10-10/report.md). Activation remains
off and still needs the separate approved activation procedure and monotonic
journal-required marker. This source/backup integration authorizes no broker run.
