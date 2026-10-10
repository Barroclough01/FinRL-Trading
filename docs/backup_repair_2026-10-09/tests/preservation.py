"""Read-only, non-secret runtime inventory; never imports operational source."""

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


root = (
    Path("/home/paxto/stock-trading/FinRL-Trading")
    if os.name == "posix"
    else Path("C:/Users/paxto/FinRL-Trading")
)
files = [root / ".env", root / "data/finrl_trading.db"]
for directory in ("logs", "data/execution_locks", "logs/execution_attempts"):
    if (root / directory).exists():
        files.extend(p for p in (root / directory).rglob("*") if p.is_file())
files.extend(
    root / "data" / name
    for name in (
        "paper_execution_journal.sqlite3",
        "paper_execution_journal.sqlite3-wal",
        "paper_execution_journal.sqlite3-shm",
        "paper_execution_journal.sqlite3-journal",
    )
)
inventory = {
    str(p.relative_to(root)): {
        "exists": p.exists(),
        "sha256": sha(p) if p.is_file() else None,
        "bytes": p.stat().st_size if p.exists() else None,
    }
    for p in set(files)
}
result = {
    "observed_utc": datetime.now(timezone.utc).isoformat(),
    "root": str(root),
    "inventory": inventory,
    "head": subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip(),
    "branch": subprocess.check_output(
        ["git", "-C", str(root), "branch", "--show-current"], text=True
    ).strip(),
    "gate_process": os.getenv("PAPER_EXECUTION_JOURNAL_ENABLED"),
}
if os.name == "posix":
    env = root / ".env"
    result["gate_file"] = [
        line.strip().split("=", 1)[1]
        for line in env.read_text().splitlines()
        if line.strip().startswith("PAPER_EXECUTION_JOURNAL_ENABLED=")
    ]
    result["matching_processes"] = []
    names = (
        "run_paper_trading.py",
        "recover_paper_execution.py",
        "track_metrics.py",
        "track_rl_offline.py",
        "refresh_fmp_daily.py",
        "backup_finrl.py",
    )
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            command = (proc / "cmdline").read_bytes().split(b"\0")
            matching = [
                name
                for name in names
                if any(
                    Path(arg.decode(errors="ignore")).name == name for arg in command
                )
            ]
            if matching:
                result["matching_processes"].append(
                    {"pid": proc.name, "scripts": matching}
                )
        except (FileNotFoundError, PermissionError):
            pass
print(json.dumps(result, indent=2, sort_keys=True))
