"""Validate exported recovery data without importing the operational project.

Validation does not establish capture consistency. A caller must enforce an OS
write barrier over its source through SQLite export and attempt inventory.
The scheduled helper deliberately refuses journal-era production capture.
"""

import hashlib
import json
import sqlite3
from decimal import Decimal
from pathlib import Path

JOURNAL = "wsl/database/paper_execution_journal.sqlite3"
ATTEMPTS = "wsl/logs/execution_attempts"
CONTRACT = "RECOVERY.json"
TABLES = {
    "account_bindings": ["alias", "endpoint", "broker_id"],
    "execution_sessions": [
        "session_id",
        "endpoint",
        "broker_id",
        "signal_date",
        "config_hash",
        "target_hash",
        "targets",
        "algorithm",
        "snapshot",
        "state",
        "phases",
    ],
    "order_intents": [
        "client_id",
        "session_id",
        "phase",
        "sequence",
        "payload",
        "state",
        "broker_id",
        "receipt",
    ],
    "execution_events": [
        "id",
        "session_id",
        "attempt_id",
        "observed_at",
        "kind",
        "evidence",
    ],
}
SCHEMA = """
CREATE TABLE account_bindings (alias TEXT PRIMARY KEY, endpoint TEXT NOT NULL,
broker_id TEXT NOT NULL);
CREATE TABLE execution_sessions (session_id TEXT PRIMARY KEY, endpoint TEXT NOT
NULL, broker_id TEXT NOT NULL, signal_date TEXT NOT NULL, config_hash TEXT NOT
NULL, target_hash TEXT NOT NULL, targets TEXT NOT NULL, algorithm TEXT NOT NULL,
snapshot TEXT NOT NULL, state TEXT NOT NULL, phases TEXT NOT NULL DEFAULT '[]',
UNIQUE(endpoint, broker_id, signal_date));
CREATE TABLE order_intents (client_id TEXT PRIMARY KEY, session_id TEXT NOT NULL
REFERENCES execution_sessions(session_id), phase TEXT NOT NULL, sequence INTEGER
NOT NULL, payload TEXT NOT NULL, state TEXT NOT NULL, broker_id TEXT, receipt
TEXT, UNIQUE(session_id, phase, sequence));
CREATE TABLE execution_events (id INTEGER PRIMARY KEY, session_id TEXT NOT NULL
REFERENCES execution_sessions(session_id), attempt_id TEXT NOT NULL, observed_at
TEXT NOT NULL, kind TEXT NOT NULL, evidence TEXT NOT NULL);
PRAGMA user_version=1;
"""


def schema_signature(conn, table):
    columns = [tuple(r) for r in conn.execute(f"PRAGMA table_info({table})")]
    foreign = sorted(
        tuple(r) for r in conn.execute(f"PRAGMA foreign_key_list({table})")
    )
    unique = sorted(
        sorted(r[2] for r in conn.execute(f"PRAGMA index_info({index[1]})"))
        for index in conn.execute(f"PRAGMA index_list({table})")
        if index[2]
    )
    return columns, foreign, unique


def hash_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def integrity(path):
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Missing or unsafe database: {path.name}")
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        if conn.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise ValueError(f"Database integrity failed: {path.name}")


def journal_summary(stage):
    stage = Path(stage)
    db = stage / JOURNAL
    integrity(db)
    root = stage / ATTEMPTS
    directories = (
        sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []
    )
    if root.exists():
        for item in root.rglob("*"):
            if item.is_symlink():
                raise ValueError("Attempt symlink is forbidden")
        if any(not p.is_dir() for p in root.iterdir()):
            raise ValueError("Unexpected attempt root file")
    held = []
    evidence_refs = []
    with sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        if conn.execute("PRAGMA user_version").fetchone()[0] != 1:
            raise ValueError("Unsupported journal schema")
        with sqlite3.connect(":memory:") as reference:
            reference.executescript(SCHEMA)
            for table in TABLES:
                if schema_signature(conn, table) != schema_signature(reference, table):
                    raise ValueError(f"Unexpected journal schema: {table}")
        if conn.execute("PRAGMA foreign_key_check").fetchall():
            raise ValueError("Journal foreign key violation")
        rows = {
            table: [dict(r) for r in conn.execute(f"SELECT * FROM {table} ORDER BY 1")]
            for table in TABLES
        }
    sessions = {s["session_id"]: s for s in rows["execution_sessions"]}
    for session in sessions.values():
        if session["session_id"] != digest(
            [session["endpoint"], session["broker_id"], session["signal_date"]]
        ):
            raise ValueError("Unstable session identity")
        if session["target_hash"] != digest(json.loads(session["targets"])):
            raise ValueError("Frozen target hash mismatch")
        for key in ("snapshot", "phases"):
            json.loads(session[key])
        if session["state"] != "submitted":
            held.append(f"interrupted_session:{session['session_id']}")
    for binding in rows["account_bindings"]:
        if not any(
            s["endpoint"] == binding["endpoint"]
            and s["broker_id"] == binding["broker_id"]
            for s in sessions.values()
        ):
            raise ValueError("Account binding has no session")
    for intent in rows["order_intents"]:
        payload = json.loads(intent["payload"])
        expected = (
            "fr-"
            + digest([intent["session_id"], intent["phase"], intent["sequence"]])[:40]
        )
        if (
            intent["client_id"] != expected
            or payload.get("client_order_id") != expected
        ):
            raise ValueError("Unstable client identity")
        if intent["state"] != "filled":
            held.append(f"intent_{intent['state']}:{intent['client_id']}")
        if not intent["receipt"]:
            held.append(f"missing_receipt:{intent['client_id']}")
        if intent["receipt"]:
            receipt = json.loads(intent["receipt"])
            if (
                receipt.get("client_order_id") != expected
                or receipt.get("id") != intent["broker_id"]
            ):
                raise ValueError("Receipt identity mismatch")
            if receipt.get("status") != intent["state"] or any(
                receipt.get(key) != payload.get(key)
                for key in ("symbol", "side", "type", "time_in_force")
            ):
                raise ValueError("Receipt state or payload mismatch")
            qty = Decimal(str(receipt.get("qty")))
            filled = Decimal(str(receipt.get("filled_qty", "0")))
            if (
                not qty.is_finite()
                or not filled.is_finite()
                or qty <= 0
                or qty != Decimal(str(payload.get("qty")))
                or not 0 <= filled <= qty
                or receipt.get("extended_hours", False) != payload.get("extended_hours")
            ):
                raise ValueError("Receipt quantity or extended_hours mismatch")
            if intent["state"] == "filled" and filled != qty:
                raise ValueError("Filled receipt quantity mismatch")
            if filled > 0:
                price = Decimal(str(receipt.get("filled_avg_price")))
                if not price.is_finite() or price <= 0:
                    raise ValueError("Invalid filled price")
    referenced = set()
    for event in rows["execution_events"]:
        attempt = event["attempt_id"]
        if not attempt or Path(attempt).name != attempt or attempt in (".", ".."):
            raise ValueError("Unsafe attempt reference")
        if attempt not in directories:
            raise ValueError(f"Missing referenced attempt: {attempt}")
        referenced.add(attempt)
        json.loads(event["evidence"])
    for name in directories:
        attempt = root / name
        if name not in referenced:
            held.append(f"abandoned_attempt:{name}")
        for path in attempt.rglob("*.json"):
            value = json.loads(path.read_text(encoding="utf-8"))
            if path.name.startswith("decision_"):
                sid = value.get("execution_session_id")
                aid = value.get("execution_attempt_id")
            elif path.name.startswith("recovery_"):
                sid = value.get("session_id")
                aid = value.get("attempt_id")
            else:
                continue
            if (
                sid not in sessions
                or aid != name
                or not any(
                    e["session_id"] == sid and e["attempt_id"] == name
                    for e in rows["execution_events"]
                )
            ):
                raise ValueError(f"Immutable evidence reference mismatch: {path.name}")
            evidence_refs.append(
                [sid, aid, path.relative_to(stage).as_posix(), hash_file(path)]
            )
    for sid, aid in sorted(
        {
            (event["session_id"], event["attempt_id"])
            for event in rows["execution_events"]
        }
    ):
        if not any(ref[0] == sid and ref[1] == aid for ref in evidence_refs):
            held.append(f"incomplete_external_evidence:{sid}:{aid}")
    for sid in sessions:
        if not any(event["session_id"] == sid for event in rows["execution_events"]):
            held.append(f"missing_event_history:{sid}")
    return {
        "schema_version": 1,
        "database_sha256": hash_file(db),
        "row_counts": {k: len(v) for k, v in rows.items()},
        "rows_sha256": digest(rows),
        "session_ids": sorted(sessions),
        "client_ids": sorted(i["client_id"] for i in rows["order_intents"]),
        "attempt_directories": directories,
        "evidence_refs": sorted(evidence_refs),
        "holds": sorted(held),
        "execution_authorized": False,
    }


def verify_restore(dest):
    dest = Path(dest)
    expected = json.loads((dest / "SHA256.json").read_text(encoding="utf-8"))
    actual = {}
    for item in dest.rglob("*"):
        if item.is_symlink():
            raise ValueError("Restored symlink is forbidden")
        if item.is_file() and item.relative_to(dest).as_posix() != "SHA256.json":
            actual[item.relative_to(dest).as_posix()] = hash_file(item)
    if actual != expected:
        raise ValueError(
            "Restored file inventory or hashes do not match the snapshot manifest"
        )
    integrity(dest / "wsl/database/finrl_trading.db")
    contract = dest / CONTRACT
    if not contract.exists():
        if (dest / JOURNAL).exists() or (dest / ATTEMPTS).exists():
            raise ValueError("Journal artifacts without recovery contract")
        state = {"coverage": "legacy_pre_journal", "execution_authorized": False}
    else:
        state = json.loads(contract.read_text(encoding="utf-8"))
        if state.get("format_version") != 1:
            raise ValueError("Unsupported recovery contract")
        if state.get("execution_authorized") is not False:
            raise ValueError("Restored snapshot cannot authorize execution")
        if state.get("coverage") == "pre_journal":
            if (dest / JOURNAL).exists() or (dest / ATTEMPTS).exists():
                raise ValueError(
                    "Unexpected journal-era artifacts in pre-journal snapshot"
                )
        elif state.get("coverage") == "sealed_synthetic_journal":
            if journal_summary(dest) != state.get("journal"):
                raise ValueError("Restored journal/reference summary mismatch")
            if state.get("execution_authorized") is not False:
                raise ValueError("Synthetic snapshot cannot authorize execution")
        else:
            raise ValueError(
                "Unsupported recovery coverage; production journal capture remains held"
            )
    return {
        "files_verified": len(actual),
        "database_integrity": "ok",
        "recovery": state,
    }
