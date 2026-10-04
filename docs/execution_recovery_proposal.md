# Paper execution recovery decision

Prepared 2026-10-04. Proposal only for the durable contract below. No journal,
new CLI, automatic recovery, scheduler or broker action has been implemented.

## Recommendation and choices

Approve a separate execution journal and broker-read-only recovery path, with
single-writer account locking and frozen order intents written before submission.
Keep uncertain submissions held for human review; do not automatically resubmit
after a lost response. This adds a consequential data model and interface, so it
requires a separate user decision under Chief of Staff's AGENTS.md.

| Choice | Benefit | Remaining cost or risk |
| --- | --- | --- |
| Keep the implemented open-order guard only | Small existing-interface repair; pending orders survive a rerun | Completed duplicate runs, concurrent reads, lost acceptance and overwritten evidence remain possible |
| Separate journal and conservative recovery (recommended) | Preserve intent and attempts across process failure; read receipts without replacing orders | New durable tables, locking and recovery interface; unknown outcomes may need manual investigation |
| Journal with automatic resume/resubmission | Less manual handling | Larger execution policy decision; a missing receipt does not establish non-acceptance; defer |

## Evidence from the current source

- `src/trading/alpaca_manager.py::execute_portfolio_rebalance` previously called
  `cancel_all_orders` before planning every live rebalance, including `next_open`.
  Its cancellation failure was swallowed. Existing positions could therefore
  generate replacement orders while earlier orders were queued or partially filled.
- The implemented repair replaces that call with `get_orders(status='open')`
  for the resolved account. A nonempty list, failed read or invalid list response
  raises `ValueError` before planning/submission. It is deliberately conservative
  for unrelated orders and account aliases: no existing order is replacement authority.
- `OrderRequest` and `place_order` have no caller-supplied client order identity.
  `place_orders_batch` catches submission exceptions and creates an empty-ID
  failed response. A response lost after acceptance cannot be distinguished from
  no acceptance by that response alone. Existing GET retries do not retry POST.
- `run_paper_trading.py::run_account` saves the decision after both execution
  phases and post-account reads. Process death before that write can lose the
  broker IDs from durable local evidence. SQLite uses one decision per
  date/account with `INSERT OR REPLACE`; the JSONL mirror is written afterward.
- `main` runs accounts sequentially and retains final audit/parity failure gates.
  It opens `logs/execution_<date>.json` in write mode. A refused same-date rerun
  can overwrite an earlier summary even though the old decision remains.
- Sell and buy requests are calculated in separate phases; buys are recalculated
  after sells using fresh account state. A durable plan must represent those
  phases rather than pretending the dry-run preview freezes final buy quantities.

Depth: relevant executor, submission/read helpers, weekly orchestration,
decision persistence and final outcome were inspected; isolated fake-broker
tests reproduced the issue. No authenticated broker reads, orders/cancellations,
operational reruns or comparison-history remediation were performed. Current
broker state and later fills remain unchecked.

## Recommended contract for approval

1. **Scope and identity.** Support only the existing two paper accounts and the
   established WSL execution host. Resolve account aliases to the broker account
   ID and verified paper endpoint before execution. Key a session by paper
   endpoint, broker account ID and resolved NYSE signal date, with config hash,
   canonical target hash and algorithm version immutable. Another alias does
   not create another session. A changed target/config for the same key holds
   for review; it never silently opens a replacement revision. RL stays offline.
2. **Independent durable authority.** Add ignored
   `data/paper_execution_journal.sqlite3`, independent of comparison/history
   rebuilding. Proposed tables: `execution_sessions` (unique account/session key,
   frozen targets/config, pre-execution account snapshot, phase state),
   `order_intents` (session, phase, sequence, exact payload, unique client ID,
   attempt state, broker ID and latest receipt), and append-only
   `execution_events` (transition, timestamp, attempt/receipt/error evidence).
   Store decimal quantities as canonical text. Commit intent and event records
   before each POST; a failed commit means zero submissions. Use SQLite durable
   transaction settings, schema version checks and fail-closed corruption handling.
   Do not store credentials. Do not migrate or repair comparison history.
3. **One writer.** Acquire an OS lock in the shared WSL runtime journal directory,
   keyed by verified broker account identity, before any session mutation or
   submission. Keep it throughout execution/recovery and also enforce database
   uniqueness. A second local writer fails without submission. Lock release after
   process death permits receipt recovery, never automatic repeat execution.
   Native Windows stays a development checkout; another execution host would
   require a separate coordination design. External/manual account changes remain
   possible and must cause resume validation to hold, not be ignored.
4. **Order identity and submission.** Persist a deterministic client ID such as
   `fr-<40 hex chars>` derived from the immutable account/session/phase/sequence
   identity. Enforce uniqueness and compare the exact payload on replay. Mark
   `attempted_unknown` durably immediately before POST. Submit each intent once;
   persist broker acceptance and receipts afterward. Timeout, connection loss,
   unclassified response or receipt-write failure holds further submissions.
   A client ID enables lookup; it does not justify an automatic retry. A crash
   between committing `attempted_unknown` and actually sending can therefore
   require human review, which is the conservative tradeoff.
5. **Execution phases.** Freeze validated targets and pre-run account evidence
   first; freeze sell intents before their first POST. Freeze buy intents only
   after the existing post-sell account/buying-power calculation, before their
   first POST. Pending known Friday sells retain current behavior and cash
   constraints; unknown sell outcomes prevent advancing the buy phase. No
   automatic cancel/replace, changed payload or new intent on a resumed phase.
6. **Repeat/recovery behavior.** An existing session never enters fresh order
   execution by default, even when no orders remain open. A proposed
   `recover_paper_execution.py --date YYYY-MM-DD --account NAME` reads orders by
   persisted broker ID or client ID and appends receipts locally, without POST
   or DELETE. Not-found, unavailable reads, mismatched payload/account or ambiguous
   acceptance remain held. No recovery schedule is created. An explicit future
   resume interface for provably unattempted intents is deferred: the first
   implementation leaves them held rather than resuming orders automatically.
7. **Receipts and evidence.** Preserve submitted, pending, partial fills and
   terminal outcomes separately, retaining requested/filled quantity and price.
   Append observations; do not rewrite comparison history or equate an accepted
   order with a fill. Give each invocation a unique attempt directory for audit
   outputs; dated/latest summaries may reference immutable attempts but cannot
   erase their evidence. A decision export references its session/attempt, and
   repeated exports cannot replace the original order evidence.
8. **Final result and rollout.** Preserve the existing final audit/parity gate:
   known pending DAY orders are valid but unreconciled and suppress `ok`; unknown
   acceptance, read failures, intent write failures and mismatches fail. Completed
   sessions produce receipt observation rather than new trades. Test with fake
   brokers and disposable databases only. Publish source using the existing
   clean fast-forward workflow. Proposed integration is default-disabled via
   `PAPER_EXECUTION_JOURNAL_ENABLED=false` (unset also means disabled), so source
   synchronization alone leaves the existing guarded workflow in use. Enabling
   this new gate, initializing the operational journal and any authenticated
   recovery require separately authorized operational scope; do not edit `.env`
   or enable the gate during the source release. No new dependencies,
   scheduler, credentials, strategies or real-money authority.

Alpaca documents caller-defined client order IDs and lookup by that identity.
The current create-order reference allows IDs up to 128 characters, so the
proposed 43-character format is within that limit. Its DAY definition also
documents after-close orders queuing for the following trading day.
[Working with orders](https://docs.alpaca.markets/us/docs/working-with-orders),
[create order](https://docs.alpaca.markets/us/reference/postorder),
[lookup by client ID](https://docs.alpaca.markets/us/reference/getorderbyclientorderid).
These sources support identity/lookup, not an exactly-once guarantee; the
conservative local policy above is a design recommendation.

## Acceptance cases for the proposed contract

| Case | Required observable result |
| --- | --- |
| First clear-account Friday run | Durable session/intents before each fake POST; unchanged DAY payloads and targets |
| Existing open, partial or inherited orders | No cancel/POST; actionable account/order hold |
| Response lost after fake acceptance | Intent survives with the same client ID; GET recovery records one accepted order; no second POST |
| Crash before/after intent commit, before POST, after POST, before receipt/export | Recoverable evidence or conservative unknown hold; zero unrecorded submissions |
| Same-session rerun after all orders fill or cancel | No new POST, no replacement intent; new receipt observation only |
| Concurrent same-account processes or different aliases | Exactly one local writer, unique canonical session; loser submits nothing |
| Account alias points to a different broker account or endpoint | Identity mismatch fails before mutation/submission |
| Partial execution across accounts | Completed account does not repeat; unfinished account evidence is explicit; final comparison failure remains visible |
| Changed targets/config or external positions on repeat | Hold without new payload/session revision or automatic replacement |
| Unknown/read failure/not-found/corrupt journal/disk-full | Fail with account/session/path context; no subsequent orders or success notification |
| Known pending/partial DAY receipts | Preserve partial quantities and pending status; unreconciled, no `ok` |
| Recovery/export repeat | Immutable prior attempts remain; original intent/IDs retained; no historical metrics rewrite |
| Offline RL or native developer process | No Alpaca order execution; existing chronology/parity tests still pass |

Approval question: **Approve implementing this separate paper execution journal,
canonical account/session identity, WSL writer lock, pre-POST durable client IDs,
immutable attempt evidence and broker-read-only recovery contract, with every
uncertain or unattempted interrupted submission held and automatic resume deferred?**
Approval would authorize source/test/docs implementation, verified publication
and established source synchronization, not orders, cancellation, runtime journal
activation, authenticated recovery, history changes or additional schedules.
