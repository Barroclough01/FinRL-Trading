"""WSL-only coordinated export. No broker or operational source imports."""

import fcntl
import importlib.util
import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path


def load_contract(path):
    spec = importlib.util.spec_from_file_location("recovery_contract", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Capture contract helper unavailable: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe(path):
    if path.is_symlink():
        raise ValueError(f"Capture held: source symlink {path}")


def copy_files(source, dest, extensions=None, exclude=()):
    safe(source)
    if not source.is_dir():
        raise ValueError(f"Capture held: required directory missing {source}")

    def fail_walk(error):
        raise error

    for current, dirs, names in os.walk(source, followlinks=False, onerror=fail_walk):
        current = Path(current)
        for name in dirs + names:
            safe(current / name)
        dirs[:] = [name for name in dirs if name not in exclude]
        for name in names:
            item = current / name
            if extensions and item.suffix.lower() not in extensions:
                continue
            target = dest / item.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def export(source, dest, contract):
    safe(source)
    if not source.is_file():
        raise ValueError(f"Capture held: required database missing {source.name}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True) as src:
        with sqlite3.connect(dest) as dst:
            src.backup(dst)
    contract.integrity(dest)


def capture(source, stage, contract, required=False, checkpoint=None):
    """Exactly one FD lifetime. Checkpoints are disposable-test callbacks only."""
    source, stage = Path(source), Path(stage)
    for path in (source, source / "data", source / "logs"):
        safe(path)
        if not path.is_dir():
            raise ValueError(f"Capture held: required source missing {path}")
    fd = os.open(
        source / "data/execution_capture.lock",
        os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
        0o600,
    )
    with os.fdopen(fd, "a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(
                "Capture held: execution/recovery writer active; "
                "inspect execution evidence before an explicit retry"
            ) from exc
        # No staging/attempt/journal creation before ownership succeeds.
        journal = source / "data/paper_execution_journal.sqlite3"
        attempts = source / "logs/execution_attempts"
        locks = source / "data/execution_locks"
        indicators = [journal, attempts, locks] + [
            Path(str(journal) + suffix) for suffix in ("-wal", "-shm", "-journal")
        ]
        for path in indicators:
            safe(path)
        observed = any(
            p.exists() and (not p.is_dir() or any(p.iterdir())) for p in indicators
        )
        required = required or observed
        if required and not journal.is_file():
            raise ValueError("Capture held: journal-required database missing")
        stage.mkdir(parents=True, exist_ok=True)
        cutoff = datetime.now(timezone.utc).isoformat()
        if checkpoint:
            checkpoint("owned")
        export(
            source / "data/finrl_trading.db",
            stage / "wsl/database/finrl_trading.db",
            contract,
        )
        if checkpoint:
            checkpoint("comparison_exported")
        copy_files(source / "logs", stage / "wsl/logs", exclude=("execution_attempts",))
        copy_files(
            source / "results",
            stage / "wsl/results",
            {".csv", ".json", ".html", ".png"},
        )
        for item in source.iterdir():
            safe(item)
            if item.is_file() and item.suffix.lower() in {".csv", ".json", ".html"}:
                target = stage / "wsl/root-reports" / item.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)
        state = {
            "format_version": 1,
            "execution_authorized": False,
            "coverage": "pre_journal",
            "cutoff_utc": cutoff,
            "capture_barrier": "WSL execution_capture.lock exclusive flock",
            "consistency": "journal/attempt cooperating-writer cutoff; each DB "
            "transaction-consistent; standalone/manual writers excluded",
        }
        if required:
            export(journal, stage / contract.JOURNAL, contract)
            if checkpoint:
                checkpoint("journal_exported")
            if attempts.exists():
                copy_files(attempts, stage / contract.ATTEMPTS)
                # Preserve empty abandoned/interrupted directories too.
                for directory in attempts.rglob("*"):
                    if directory.is_dir():
                        (
                            stage / contract.ATTEMPTS / directory.relative_to(attempts)
                        ).mkdir(parents=True, exist_ok=True)
            if checkpoint:
                checkpoint("attempts_copied")
            state["coverage"] = "coordinated_wsl_journal"
            state["journal"] = contract.journal_summary(stage)
        (stage / contract.CONTRACT).write_text(
            json.dumps(state, indent=2), encoding="utf-8"
        )
        manifest = {
            p.relative_to(stage).as_posix(): contract.hash_file(p)
            for p in stage.rglob("*")
            if p.is_file() and p.name != "SHA256.json"
        }
        (stage / "SHA256.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        result = contract.verify_restore(stage)
        if checkpoint:
            checkpoint("verified")
        return result


if __name__ == "__main__":
    print(
        json.dumps(
            capture(
                sys.argv[1],
                sys.argv[2],
                load_contract(sys.argv[3]),
                sys.argv[4] == "true",
            ),
            indent=2,
        )
    )
