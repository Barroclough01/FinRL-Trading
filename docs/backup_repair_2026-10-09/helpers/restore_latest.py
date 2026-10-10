"""Restore the latest FinRL snapshot into a NEW directory and verify every file."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from backup_finrl import ROOT, SOURCE, kopia
from recovery_contract import verify_restore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    snapshots = json.loads(kopia("snapshot", "list", "--json"))
    complete = [
        s
        for s in snapshots
        if s.get("source", {}).get("path", "").lower() == str(SOURCE).lower()
        and not s.get("incomplete")
        and not s.get("incompleteReason")
        and s.get("stats", {}).get("errorCount", 0) == 0
        and s.get("rootEntry", {}).get("summ", {}).get("numFailed", 0) == 0
    ]
    if not complete:
        raise RuntimeError("No complete FinRL snapshot exists")
    latest = max(complete, key=lambda s: s["startTime"])
    dest = args.destination or (
        Path(r"C:\Users\paxto\PaxtonBackups\Restored")
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    if dest.exists():
        raise RuntimeError(
            "Restore destination already exists; choose a NEW empty path"
        )
    kopia(
        "snapshot",
        "restore",
        latest["rootEntry"]["obj"],
        dest,
        "--no-overwrite-files",
        "--no-overwrite-directories",
        "--no-overwrite-symlinks",
    )
    validation = verify_restore(dest)
    report = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "status": "success",
        "snapshot_id": latest["id"],
        **validation,
        "destination": str(dest),
    }
    (ROOT / "last-restore.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Restore verification failed: {exc}", file=sys.stderr)
        sys.exit(1)
