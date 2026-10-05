"""Conservative paper execution intent and GET-only receipt recovery."""

import hashlib
import json
import os
import platform
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

GATE = "PAPER_EXECUTION_JOURNAL_ENABLED"
PAPER_ENDPOINT = "https://paper-api.alpaca.markets"
SCHEMA_VERSION = 1
ALGORITHM_VERSION = "phased-rebalance-v1"
PENDING = frozenset(
    {
        "new",
        "accepted",
        "pending_new",
        "partially_filled",
        "pending_cancel",
        "pending_replace",
        "accepted_for_bidding",
        "done_for_day",
        "stopped",
        "suspended",
        "calculated",
    }
)
TERMINAL = frozenset({"filled", "canceled", "expired", "rejected"})


def enabled() -> bool:
    value = os.getenv(GATE, "false").strip().lower()
    if value not in {"true", "false", ""}:
        raise ValueError(f"{GATE} must be true or false")
    return value == "true"


def require_execution_host(root: Path | None = None) -> None:
    if os.name != "posix" or "microsoft" not in platform.release().lower():
        raise ValueError("Journal execution/recovery requires the established WSL host")
    if root is not None and root.resolve() != Path(
        "/home/paxto/stock-trading/FinRL-Trading"
    ):
        raise ValueError(
            "Journal execution/recovery requires the scheduled WSL checkout"
        )


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def decimal_text(value: Any) -> str:
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError(f"Nonfinite decimal: {value}")
    return format(result.normalize(), "f")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def identity(manager: Any, alias: str) -> tuple[str, str]:
    account = manager._get_account(alias)
    endpoint = account.base_url.rstrip("/")
    if endpoint != PAPER_ENDPOINT:
        raise ValueError(f"{alias}: journal requires the verified paper endpoint")
    info = manager._api_request(
        "GET", "/v2/account", account=account, allow_redirects=False
    )
    broker_id = info.get("id") if isinstance(info, dict) else None
    if not isinstance(broker_id, str) or not broker_id.strip():
        raise ValueError(f"{alias}: missing canonical broker account ID")
    return endpoint, broker_id


@contextmanager
def account_lock(root: Path, endpoint: str, broker_id: str):
    require_execution_host(root)
    import fcntl

    directory = root / "data" / "execution_locks"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest([endpoint, broker_id])}.lock"
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(
                f"Account {broker_id}: execution writer already active"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def new_attempt(root: Path) -> Path:
    path = root / "logs" / "execution_attempts" / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_evidence(path: Path, value: Any) -> None:
    """Exclusive, fsynced evidence; never truncate an earlier attempt."""
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, default=str, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


class ExecutionJournal:
    def __init__(self, root: Path, *, create: bool = True):
        self.path = root / "data" / "paper_execution_journal.sqlite3"
        self.connection: sqlite3.Connection | None = None
        if not create and not self.path.is_file():
            raise ValueError(f"Recovery journal missing: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.connection = sqlite3.connect(self.path, timeout=5)
            self.connection.row_factory = sqlite3.Row
            conn = self.connection
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute("PRAGMA journal_mode=DELETE")
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            if version == 0 and not tables and create:
                conn.executescript("""
                    BEGIN IMMEDIATE;
                    CREATE TABLE account_bindings (
                        alias TEXT PRIMARY KEY, endpoint TEXT NOT NULL,
                        broker_id TEXT NOT NULL
                    );
                    CREATE TABLE execution_sessions (
                        session_id TEXT PRIMARY KEY, endpoint TEXT NOT NULL,
                        broker_id TEXT NOT NULL, signal_date TEXT NOT NULL,
                        config_hash TEXT NOT NULL, target_hash TEXT NOT NULL,
                        targets TEXT NOT NULL, algorithm TEXT NOT NULL,
                        snapshot TEXT NOT NULL, state TEXT NOT NULL,
                        phases TEXT NOT NULL DEFAULT '[]',
                        UNIQUE(endpoint, broker_id, signal_date)
                    );
                    CREATE TABLE order_intents (
                        client_id TEXT PRIMARY KEY, session_id TEXT NOT NULL
                          REFERENCES execution_sessions(session_id),
                        phase TEXT NOT NULL, sequence INTEGER NOT NULL,
                        payload TEXT NOT NULL, state TEXT NOT NULL,
                        broker_id TEXT, receipt TEXT,
                        UNIQUE(session_id, phase, sequence)
                    );
                    CREATE TABLE execution_events (
                        id INTEGER PRIMARY KEY, session_id TEXT NOT NULL
                          REFERENCES execution_sessions(session_id),
                        attempt_id TEXT NOT NULL, observed_at TEXT NOT NULL,
                        kind TEXT NOT NULL, evidence TEXT NOT NULL
                    );
                    PRAGMA user_version=1;
                    COMMIT;
                """)
            elif version != SCHEMA_VERSION:
                raise ValueError(f"Unsupported journal schema version {version}")
            if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("Execution journal integrity check failed")
            # Verify the schema before any broker mutation, even if version is forged.
            for table in (
                "account_bindings",
                "execution_sessions",
                "order_intents",
                "execution_events",
            ):
                conn.execute(f"SELECT * FROM {table} LIMIT 0")
        except Exception as exc:
            self.close()
            raise ValueError(
                f"Cannot open execution journal {self.path}: {exc}"
            ) from exc

    @property
    def conn(self) -> sqlite3.Connection:
        if self.connection is None:
            raise ValueError(f"Journal closed: {self.path}")
        return self.connection

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def open_session(
        self,
        *,
        endpoint: str,
        broker_id: str,
        alias: str,
        day: str,
        config_hash: str,
        targets: dict,
        snapshot: dict,
        attempt: Path,
        recovery: bool = False,
    ):
        session_id = digest([endpoint, broker_id, day])
        normalized_targets = {s: decimal_text(w) for s, w in targets.items()}
        target_hash = digest(normalized_targets)
        conn = self.conn
        with conn:
            conn.execute("BEGIN IMMEDIATE")
            binding = conn.execute(
                "SELECT endpoint, broker_id FROM account_bindings WHERE alias=?",
                (alias,),
            ).fetchone()
            if binding and tuple(binding) != (endpoint, broker_id):
                raise ValueError(f"{alias}: broker account/endpoint binding changed")
            row = conn.execute(
                "SELECT * FROM execution_sessions WHERE session_id=?", (session_id,)
            ).fetchone()
            if recovery and row is None:
                raise ValueError(f"{alias} {day}: execution session missing")
            if row is not None and row["algorithm"] != ALGORITHM_VERSION:
                raise ValueError(
                    f"{alias} {day}: unsupported frozen execution algorithm"
                )
            if (
                row is not None
                and not recovery
                and (
                    row["config_hash"] != config_hash
                    or row["target_hash"] != target_hash
                    or row["algorithm"] != ALGORITHM_VERSION
                )
            ):
                raise ValueError(f"{alias} {day}: frozen session inputs changed")
            conn.execute(
                "INSERT OR IGNORE INTO account_bindings VALUES (?, ?, ?)",
                (alias, endpoint, broker_id),
            )
            if row is None:
                if not config_hash:
                    raise ValueError(f"{alias} {day}: missing config hash")
                conn.execute(
                    "INSERT INTO execution_sessions (session_id, endpoint, broker_id, "
                    "signal_date, config_hash, target_hash, targets, algorithm, "
                    "snapshot, state) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'started')",
                    (
                        session_id,
                        endpoint,
                        broker_id,
                        day,
                        config_hash,
                        target_hash,
                        canonical(normalized_targets),
                        ALGORITHM_VERSION,
                        canonical(snapshot),
                    ),
                )
            session = ExecutionSession(self, session_id, attempt, row is not None)
            session.event(
                "recovery_started" if row else "session_started",
                {
                    "alias": alias,
                    "signal_date": day,
                    "endpoint": endpoint,
                    "broker_account_id": broker_id,
                },
            )
        return session


class ExecutionSession:
    def __init__(
        self, journal: ExecutionJournal, session_id: str, attempt: Path, repeated: bool
    ):
        self.journal = journal
        self.session_id = session_id
        self.attempt = attempt
        self.repeated = repeated

    def event(self, kind: str, evidence: Any) -> None:
        self.journal.conn.execute(
            "INSERT INTO execution_events (session_id, attempt_id, observed_at, "
            "kind, evidence) VALUES (?, ?, ?, ?, ?)",
            (
                self.session_id,
                self.attempt.name,
                datetime.now(timezone.utc).isoformat(),
                kind,
                canonical(evidence),
            ),
        )

    def freeze(self, phase: str, payloads: list[dict]) -> list[dict]:
        if self.repeated:
            raise ValueError("Existing sessions cannot resume order execution")
        conn = self.journal.conn
        with conn:
            row = conn.execute(
                "SELECT phases FROM execution_sessions WHERE session_id=?",
                (self.session_id,),
            ).fetchone()
            phases = json.loads(row[0])
            if phase in phases or (phase == "buy" and "sell" not in phases):
                raise ValueError(f"Invalid or repeated execution phase: {phase}")
            frozen = []
            for sequence, source in enumerate(sorted(payloads, key=canonical)):
                payload = dict(source)
                client_id = "fr-" + digest([self.session_id, phase, sequence])[:40]
                payload["client_order_id"] = client_id
                conn.execute(
                    "INSERT INTO order_intents (client_id, session_id, phase, "
                    "sequence, payload, state) VALUES (?, ?, ?, ?, ?, 'planned')",
                    (client_id, self.session_id, phase, sequence, canonical(payload)),
                )
                frozen.append(payload)
            conn.execute(
                "UPDATE execution_sessions SET phases=? WHERE session_id=?",
                (canonical(phases + [phase]), self.session_id),
            )
            self.event("phase_frozen", {"phase": phase, "payloads": frozen})
        return frozen

    @staticmethod
    def validate_receipt(payload: dict, receipt: dict) -> str:
        if not isinstance(receipt, dict) or not receipt.get("id"):
            raise ValueError("Missing broker receipt/ID")
        for key in ("symbol", "side", "type", "time_in_force", "client_order_id"):
            if receipt.get(key) != payload[key]:
                raise ValueError(f"Broker receipt mismatch: {key}")
        if decimal_text(receipt.get("qty")) != payload["qty"]:
            raise ValueError("Broker receipt quantity mismatch")
        if receipt.get("extended_hours", False) != payload["extended_hours"]:
            raise ValueError("Broker receipt extended_hours mismatch")
        status = receipt.get("status", "")
        if status not in PENDING | TERMINAL:
            raise ValueError(f"Unknown broker status: {status}")
        filled = Decimal(decimal_text(receipt.get("filled_qty", "0")))
        qty = Decimal(payload["qty"])
        if not 0 <= filled <= qty or (status == "filled" and filled != qty):
            raise ValueError("Invalid filled quantity")
        if filled > 0 and Decimal(decimal_text(receipt.get("filled_avg_price"))) <= 0:
            raise ValueError("Missing/invalid filled price")
        return status

    def observe(self, payload: dict, receipt: dict) -> None:
        status = self.validate_receipt(payload, receipt)
        conn = self.journal.conn
        with conn:
            prior = conn.execute(
                "SELECT broker_id FROM order_intents WHERE client_id=?",
                (payload["client_order_id"],),
            ).fetchone()
            if prior[0] and prior[0] != receipt["id"]:
                raise ValueError("Broker order ID changed")
            conn.execute(
                "UPDATE order_intents SET state=?, broker_id=?, receipt=? "
                "WHERE client_id=?",
                (status, receipt["id"], canonical(receipt), payload["client_order_id"]),
            )
            self.event("receipt_observed", receipt)

    def submit_batch(self, phase: str, payloads: list[dict], transport) -> list[dict]:
        frozen = self.freeze(phase, payloads)
        receipts = []
        for payload in frozen:
            with self.journal.conn:
                updated = self.journal.conn.execute(
                    "UPDATE order_intents SET state='attempted_unknown' "
                    "WHERE client_id=? AND state='planned'",
                    (payload["client_order_id"],),
                )
                if updated.rowcount != 1:
                    raise ValueError("Order intent already attempted")
                self.event("submission_attempted", payload)
            # Never retry POST; failure leaves the committed unknown intent intact.
            receipt = transport(payload)
            self.observe(payload, receipt)
            receipts.append(receipt)
            if receipt["status"] == "rejected":
                raise ValueError("Broker rejected order; further submissions held")
        return receipts

    def finish(self) -> None:
        with self.journal.conn:
            row = self.journal.conn.execute(
                "SELECT phases FROM execution_sessions WHERE session_id=?",
                (self.session_id,),
            ).fetchone()
            if json.loads(row[0]) != ["sell", "buy"]:
                raise ValueError("Incomplete execution phases")
            self.journal.conn.execute(
                "UPDATE execution_sessions SET state='submitted' WHERE session_id=?",
                (self.session_id,),
            )
            self.event("submission_complete", {})

    def recover(self, manager: Any, alias: str) -> dict:
        failures = []
        receipts = []
        intents = self.journal.conn.execute(
            "SELECT * FROM order_intents WHERE session_id=? ORDER BY phase, sequence",
            (self.session_id,),
        ).fetchall()
        account = manager._get_account(alias)
        for intent in intents:
            if intent["state"] == "planned":
                failures.append(f"Unattempted intent held: {intent['client_id']}")
                continue
            payload = json.loads(intent["payload"])
            try:
                if intent["broker_id"]:
                    receipt = manager._api_request(
                        "GET",
                        f"/v2/orders/{intent['broker_id']}",
                        account=account,
                        allow_redirects=False,
                    )
                else:
                    receipt = manager._api_request(
                        "GET",
                        "/v2/orders:by_client_order_id",
                        params={"client_order_id": intent["client_id"]},
                        account=account,
                        allow_redirects=False,
                    )
                self.observe(payload, receipt)
                receipts.append(receipt)
                if receipt["status"] == "rejected":
                    failures.append(f"Rejected order: {intent['client_id']}")
            except Exception as exc:
                failures.append(f"Receipt held {intent['client_id']}: {exc}")
                with self.journal.conn:
                    self.event(
                        "receipt_read_failed",
                        {"client_id": intent["client_id"], "error": str(exc)},
                    )
        state = self.journal.conn.execute(
            "SELECT state, snapshot FROM execution_sessions WHERE session_id=?",
            (self.session_id,),
        ).fetchone()
        if state[0] != "submitted":
            failures.append("Interrupted session held; automatic resume is disabled")
        if len(receipts) == len(intents):
            try:
                snapshot = json.loads(state[1])
                expected = {
                    p["symbol"]: Decimal(decimal_text(p["qty"]))
                    for p in snapshot["positions"]
                }
                for receipt in receipts:
                    symbol = receipt["symbol"]
                    filled = Decimal(decimal_text(receipt.get("filled_qty", "0")))
                    expected[symbol] = expected.get(symbol, Decimal(0)) + (
                        filled if receipt["side"] == "buy" else -filled
                    )
                positions = manager._api_request(
                    "GET", "/v2/positions", account=account, allow_redirects=False
                )
                if not isinstance(positions, list):
                    raise ValueError("Invalid current positions response")
                actual = {
                    p["symbol"]: Decimal(decimal_text(p["qty"])) for p in positions
                }
                if any(
                    abs(expected.get(s, Decimal(0)) - actual.get(s, Decimal(0)))
                    > Decimal("0.00000001")
                    for s in set(expected) | set(actual)
                ):
                    raise ValueError("Unexplained external position drift")
                # Value/cash moves can reflect prices and settlement; record them,
                # never use them as permission to resume an interrupted phase.
                info = manager._api_request(
                    "GET", "/v2/account", account=account, allow_redirects=False
                )
                identity_row = self.journal.conn.execute(
                    "SELECT broker_id FROM execution_sessions WHERE session_id=?",
                    (self.session_id,),
                ).fetchone()
                if info.get("id") != identity_row[0]:
                    raise ValueError("Current broker account identity changed")
                with self.journal.conn:
                    self.event(
                        "account_observed",
                        {
                            "positions": positions,
                            "account": {
                                k: info.get(k)
                                for k in ("id", "cash", "equity", "buying_power")
                            },
                        },
                    )
            except Exception as exc:
                failures.append(f"Account observation held: {exc}")
        result = {
            "session_id": self.session_id,
            "attempt_id": self.attempt.name,
            "broker_read_only": True,
            "orders": receipts,
            "failures": failures,
            "execution_pending": any(r["status"] in PENDING for r in receipts),
            "reconciled_successfully": not failures
            and bool(receipts)
            and all(r["status"] == "filled" for r in receipts),
        }
        write_evidence(self.attempt / f"recovery_{self.session_id}.json", result)
        return result
