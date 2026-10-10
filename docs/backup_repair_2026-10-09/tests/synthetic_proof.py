# ruff: noqa: E402
"""Disposable proof only, with Linux kernel write and network barriers."""

import argparse
import ctypes
import errno
import fcntl
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "helpers"))
from recovery_contract import (
    ATTEMPTS,
    CONTRACT,
    JOURNAL,
    SCHEMA,
    canonical,
    digest,
    hash_file,
    journal_summary,
    verify_restore,
)


def barriers(allowed):
    """Landlock ABI3 limits all filesystem writes; seccomp denies socket syscalls."""
    libc = ctypes.CDLL(None, use_errno=True)

    def checked(value):
        if value < 0:
            raise OSError(ctypes.get_errno(), "Kernel barrier setup failed")
        return value

    if checked(libc.syscall(444, 0, 0, 1)) < 3:
        raise RuntimeError("Landlock ABI3 required")
    access = sum(1 << i for i in (1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14))

    class Ruleset(ctypes.Structure):
        _fields_ = [("access", ctypes.c_uint64)]

    class PathRule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("access", ctypes.c_uint64), ("parent_fd", ctypes.c_int)]

    attr = Ruleset(access)
    fd = checked(libc.syscall(444, ctypes.byref(attr), ctypes.sizeof(attr), 0))
    for path in allowed:
        parent = os.open(path, os.O_PATH | os.O_CLOEXEC)
        rule = PathRule(access, parent)
        checked(libc.syscall(445, fd, 1, ctypes.byref(rule), 0))
        os.close(parent)
    checked(libc.prctl(38, 1, 0, 0, 0))  # no_new_privs
    checked(libc.syscall(446, fd, 0))
    os.close(fd)

    class Filter(ctypes.Structure):
        _fields_ = [
            ("code", ctypes.c_ushort),
            ("jt", ctypes.c_ubyte),
            ("jf", ctypes.c_ubyte),
            ("k", ctypes.c_uint),
        ]

    class Program(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(Filter))]

    instructions = [
        Filter(0x20, 0, 0, 4),
        Filter(0x15, 1, 0, 0xC000003E),
        Filter(0x06, 0, 0, 0x80000000),
        Filter(0x20, 0, 0, 0),
    ]
    for number in (41, 42, 43, 44, 45, 46, 47, 49, 50, 53, 288):
        instructions.extend(
            (Filter(0x15, 0, 1, number), Filter(0x06, 0, 0, 0x00050000 | errno.EPERM))
        )
    instructions.append(Filter(0x06, 0, 0, 0x7FFF0000))
    array = (Filter * len(instructions))(*instructions)
    program = Program(len(array), array)
    checked(libc.prctl(22, 2, ctypes.byref(program), 0, 0))
    denied = []
    for path in (
        Path("/home/paxto/stock-trading/FinRL-Trading/data/finrl_trading.db"),
        Path("/mnt/c/Users/paxto/FinRL-Trading/run_paper_trading.py"),
    ):
        try:
            handle = os.open(path, os.O_WRONLY)
        except PermissionError:
            denied.append(str(path))
        else:
            os.close(handle)
            raise RuntimeError("Primary write barrier is ineffective")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except PermissionError:
        pass
    else:
        sock.close()
        raise RuntimeError("Network barrier is ineffective")
    return {
        "landlock_abi": 3,
        "primary_write_denied": denied,
        "network_socket_denied": True,
        "write_allowed_only": [str(p) for p in allowed],
    }


def make_fixture(root, version):
    root.mkdir(parents=True)
    (root / "data").mkdir()
    with sqlite3.connect(root / "data/finrl_trading.db") as db:
        db.execute("CREATE TABLE synthetic(value TEXT)")
        db.execute("INSERT INTO synthetic VALUES (?)", (str(version),))
    endpoint = "https://paper-api.alpaca.markets"
    sid = digest([endpoint, "synthetic-account", "2026-10-09"])
    cid = "fr-" + digest([sid, "buy", 0])[:40]
    payload = {
        "client_order_id": cid,
        "symbol": "SYNTH",
        "qty": "2",
        "side": "buy",
        "type": "market",
        "time_in_force": "day",
        "extended_hours": False,
    }
    receipt = (
        {
            **payload,
            "id": "synthetic-order",
            "status": "filled",
            "filled_qty": "2",
            "filled_avg_price": "10",
        }
        if version == 2
        else None
    )
    with sqlite3.connect(root / "data/paper_execution_journal.sqlite3") as db:
        db.executescript(SCHEMA)
        db.execute(
            "INSERT INTO account_bindings VALUES (?,?,?)",
            ("FinRL", endpoint, "synthetic-account"),
        )
        db.execute(
            "INSERT INTO execution_sessions VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                sid,
                endpoint,
                "synthetic-account",
                "2026-10-09",
                "synthetic-config",
                digest({"SYNTH": "0.5"}),
                canonical({"SYNTH": "0.5"}),
                "phased-rebalance-v1",
                "{}",
                "submitted" if version == 2 else "started",
                '["sell","buy"]',
            ),
        )
        db.execute(
            "INSERT INTO order_intents VALUES (?,?,?,?,?,?,?,?)",
            (
                cid,
                sid,
                "buy",
                0,
                canonical(payload),
                "filled" if receipt else "attempted_unknown",
                "synthetic-order" if receipt else None,
                canonical(receipt) if receipt else None,
            ),
        )
        for aid in ("attempt-main", "attempt-interrupted"):
            db.execute(
                "INSERT INTO execution_events(session_id,attempt_id,observed_"
                "at,kind,evidence) VALUES (?,?,?,?,?)",
                (sid, aid, "2026-10-09T00:00:00Z", "synthetic", "{}"),
            )
    attempts = root / "logs/execution_attempts"
    for aid in ("attempt-main", "attempt-interrupted", "attempt-abandoned"):
        (attempts / aid).mkdir(parents=True)
    (attempts / "attempt-main/decision_FinRL.json").write_text(
        canonical(
            {
                "execution_session_id": sid,
                "execution_attempt_id": "attempt-main",
                "version": version,
            }
        )
    )
    (root / "overwrite.txt").write_text(f"version-{version}")
    if version == 1:
        (root / "delete-me.txt").write_text("version-one-only")
    (root / "fixture-barrier.lock").touch()


def capture(source, stage):
    # Source is outside Landlock writable directories for the entire export/copy.
    try:
        fd = os.open(source / "data/paper_execution_journal.sqlite3", os.O_WRONLY)
    except PermissionError:
        pass
    else:
        os.close(fd)
        raise RuntimeError("Synthetic source is writable during capture")
    cutoff = datetime.now(timezone.utc).isoformat()
    (stage / "wsl/database").mkdir(parents=True)
    for name in ("finrl_trading.db", "paper_execution_journal.sqlite3"):
        with sqlite3.connect(
            (source / "data" / name).as_uri() + "?mode=ro", uri=True
        ) as src:
            with sqlite3.connect(stage / "wsl/database" / name) as dst:
                src.backup(dst)
    shutil.copytree(source / "logs/execution_attempts", stage / ATTEMPTS)
    for name in ("overwrite.txt", "delete-me.txt"):
        if (source / name).exists():
            shutil.copy2(source / name, stage / name)
    contract = {
        "format_version": 1,
        "coverage": "sealed_synthetic_journal",
        "execution_authorized": False,
        "cutoff_utc": cutoff,
        "capture_barrier": (
            "Cooperating fixture global flock through export and inventory; "
            "Landlock process isolation + seccomp; synthetic only"
        ),
        "journal": journal_summary(stage),
    }
    (stage / CONTRACT).write_text(json.dumps(contract, indent=2))
    seal_manifest(stage)
    return verify_restore(stage)


def seal_manifest(stage):
    manifest = {
        p.relative_to(stage).as_posix(): hash_file(p)
        for p in stage.rglob("*")
        if p.is_file() and p.name != "SHA256.json"
    }
    (stage / "SHA256.json").write_text(json.dumps(manifest, indent=2))


def cases(valid, scratch):
    results = []

    def case(name, change, expected):
        copy = scratch / name
        shutil.copytree(valid, copy)
        change(copy)
        try:
            value = journal_summary(copy)
        except Exception as exc:
            if expected != "failure":
                raise
            results.append({"case": name, "result": "rejected", "error": str(exc)})
        else:
            if expected == "failure" or not any(
                expected in hold for hold in value["holds"]
            ):
                raise AssertionError(f"{name} missing expected hold {expected}")
            results.append({"case": name, "result": "held", "holds": value["holds"]})

    def sql(stage, statement):
        with sqlite3.connect(stage / JOURNAL) as db:
            db.execute(statement)

    case("missing-journal", lambda p: (p / JOURNAL).unlink(), "failure")
    case(
        "corrupt-journal", lambda p: (p / JOURNAL).write_bytes(b"not sqlite"), "failure"
    )
    case(
        "missing-reference",
        lambda p: shutil.rmtree(p / ATTEMPTS / "attempt-main"),
        "failure",
    )
    case(
        "missing-receipt",
        lambda p: sql(p, "UPDATE order_intents SET receipt=NULL"),
        "missing_receipt",
    )
    case(
        "unknown-intent",
        lambda p: sql(
            p,
            "UPDATE order_intents SET "
            "state='attempted_unknown',receipt=NULL,broker_id=NULL",
        ),
        "intent_attempted_unknown",
    )
    case(
        "interrupted-session",
        lambda p: sql(p, "UPDATE execution_sessions SET state='started'"),
        "interrupted_session",
    )
    case(
        "target-mismatch",
        lambda p: sql(p, "UPDATE execution_sessions SET targets='{}'"),
        "failure",
    )
    case(
        "client-mismatch",
        lambda p: sql(p, "UPDATE order_intents SET client_id='bad'"),
        "failure",
    )
    case(
        "foreign-key-violation",
        lambda p: sql(p, "UPDATE execution_events SET session_id='orphan'"),
        "failure",
    )
    case(
        "evidence-reference-mismatch",
        lambda p: (p / ATTEMPTS / "attempt-main/decision_FinRL.json").write_text(
            '{"execution_session_id":"wrong","execution_attempt_id":"attempt-main"}'
        ),
        "failure",
    )
    case(
        "receipt-state-mismatch",
        lambda p: sql(p, "UPDATE order_intents SET state='rejected'"),
        "failure",
    )

    def remove_fk(stage):
        with sqlite3.connect(stage / JOURNAL) as db:
            db.executescript(
                "ALTER TABLE execution_events RENAME TO wrong_events; "
                "CREATE TABLE execution_events (id INTEGER PRIMARY KEY, "
                "session_id TEXT NOT NULL, attempt_id TEXT NOT NULL, "
                "observed_at TEXT NOT NULL, kind TEXT NOT NULL, evidence TEXT"
                " NOT NULL); "
                "INSERT INTO execution_events SELECT * FROM wrong_events; "
                "DROP TABLE wrong_events;"
            )

    case("removed-fk-same-columns", remove_fk, "failure")
    # Old manifests have no RECOVERY.json and must retain pre-journal classification.
    legacy = scratch / "legacy"
    shutil.copytree(valid, legacy)
    (legacy / JOURNAL).unlink()
    shutil.rmtree(legacy / ATTEMPTS)
    (legacy / CONTRACT).unlink()
    seal_manifest(legacy)
    assert verify_restore(legacy)["recovery"]["coverage"] == "legacy_pre_journal"
    results.append(
        {
            "case": "legacy-pre-journal",
            "result": "verified",
            "execution_authorized": False,
        }
    )
    summary = journal_summary(valid)
    assert any("attempt-interrupted" in hold for hold in summary["holds"])
    assert any("abandoned_attempt" in hold for hold in summary["holds"])
    results.append(
        {
            "case": "partial-and-abandoned-attempt",
            "result": "held",
            "holds": summary["holds"],
        }
    )
    # Backup guard imports no operational module; Windows lock module stub unused.
    sys.modules["msvcrt"] = types.ModuleType("msvcrt")
    import backup_finrl as backup

    backup.ROOT = scratch / "guard"
    backup.ROOT.mkdir()
    backup.WSL_ROOT = scratch / "source-wsl"
    backup.WINDOWS_ROOT = scratch / "source-native"
    for root in (backup.WSL_ROOT, backup.WINDOWS_ROOT):
        (root / "data").mkdir(parents=True)
    assert backup.require_pre_journal()["coverage"] == "pre_journal"
    journal = backup.WSL_ROOT / "data/paper_execution_journal.sqlite3"
    journal.write_bytes(b"corrupt-still-observed")
    for label in ("journal-present", "journal-deleted-after-observation"):
        try:
            backup.require_pre_journal()
        except RuntimeError as exc:
            results.append({"case": label, "result": "held", "error": str(exc)})
        else:
            raise AssertionError("Production guard failed open")
        if journal.exists():
            journal.unlink()
    marker = backup.ROOT / "journal-required.json"
    marker.write_text('{"format_version":1,"journal_required":false}')
    try:
        backup.require_pre_journal()
    except RuntimeError:
        results.append({"case": "malformed-marker", "result": "held"})
    else:
        raise AssertionError("Malformed marker accepted")
    marker.unlink()
    marker.symlink_to(backup.ROOT / "nonexistent-marker-target")
    try:
        backup.require_pre_journal()
    except RuntimeError:
        results.append({"case": "dangling-marker-symlink", "result": "held"})
    else:
        raise AssertionError("Dangling marker accepted")
    marker.unlink()
    for source in (backup.WSL_ROOT, backup.WINDOWS_ROOT):
        for kind in ("sidecar", "attempt", "lock", "dangling-journal"):
            if kind == "sidecar":
                indicator = source / "data/paper_execution_journal.sqlite3-wal"
                indicator.touch()
            elif kind == "dangling-journal":
                indicator = source / "data/paper_execution_journal.sqlite3"
                indicator.symlink_to(source / "not-there")
            else:
                directory = source / (
                    "logs/execution_attempts"
                    if kind == "attempt"
                    else "data/execution_locks"
                )
                directory.mkdir(parents=True, exist_ok=True)
                indicator = directory / "synthetic-indicator"
                indicator.touch()
            try:
                backup.require_pre_journal()
            except RuntimeError:
                results.append({"case": f"{source.name}-{kind}", "result": "held"})
            else:
                raise AssertionError("Journal-era indicator accepted")
            indicator.unlink()
            marker.unlink()
    backup.WSL_ROOT = scratch / "missing-checkout"
    try:
        backup.require_pre_journal()
    except RuntimeError:
        results.append({"case": "missing-checkout", "result": "held"})
    else:
        raise AssertionError("Missing checkout accepted")
    for name, mutation, remanifest in (
        ("journal-without-contract", lambda p: (p / CONTRACT).unlink(), True),
        (
            "forged-pre-journal-authorization",
            lambda p: (p / CONTRACT).write_text(
                '{"format_version":1,"coverage":"pre_journal","execution_authorized":true}'
            ),
            True,
        ),
        (
            "manifest-hash-mismatch",
            lambda p: (p / "overwrite.txt").write_text("tampered"),
            False,
        ),
    ):
        copy = scratch / name
        shutil.copytree(valid, copy)
        mutation(copy)
        if remanifest:
            seal_manifest(copy)
        try:
            verify_restore(copy)
        except ValueError as exc:
            results.append({"case": name, "result": "rejected", "error": str(exc)})
        else:
            raise AssertionError(f"{name} accepted")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=("generate", "capture", "test", "verify", "writer")
    )
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--version", type=int, default=1)
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=True)
    barrier = barriers([args.scratch])
    if args.mode == "generate":
        make_fixture(args.scratch / f"source-v{args.version}", args.version)
        outcome = {"generated": args.version}
    elif args.mode == "capture":
        with (args.source / "fixture-barrier.lock").open("rb") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Every controlled fixture writer locks before creating accounts/attempts.
            writer = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    __file__,
                    "writer",
                    "--scratch",
                    str(args.scratch),
                    "--source",
                    str(args.source),
                ],
                capture_output=True,
                text=True,
            )
            if (
                writer.returncode != 23
                or "fixture writer blocked before attempt creation" not in writer.stdout
            ):
                raise RuntimeError(
                    f"Concurrent fixture writer did not hold: {writer.stderr}"
                )
            outcome = capture(args.source, args.scratch / "payload")
            outcome["writer_conflict"] = {
                "exit": writer.returncode,
                "message": writer.stdout.strip(),
            }
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    elif args.mode == "writer":
        with (args.source / "fixture-barrier.lock").open("rb") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print("fixture writer blocked before attempt creation")
                raise SystemExit(23)
            raise RuntimeError("Writer unexpectedly acquired capture barrier")
    elif args.mode == "test":
        outcome = cases(args.source, args.scratch)
    else:
        outcome = verify_restore(args.source)
    print(json.dumps({"barrier": barrier, "outcome": outcome}, indent=2))


if __name__ == "__main__":
    main()
