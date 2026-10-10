# ruff: noqa: E402
"Installed Kopia, existing repository, disposable payload only; no production"

" backup."

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
TOOLS = Path("C:/Users/paxto/PaxtonBackups/tools")
SCRATCH = Path("C:/Users/paxto/.codex/tmp/finrl-backup-proof-20261010")
SCRATCH.mkdir(parents=True, exist_ok=False)
SOURCE = SCRATCH / "synthetic-runtime"
EVIDENCE = BASE / "evidence"
sys.path.insert(0, str(BASE / "helpers"))
import restore_latest
from recovery_contract import hash_file, verify_restore


def kopia(*args):
    env = os.environ.copy()
    env["KOPIA_PASSWORD"] = (TOOLS / "recovery-key.txt").read_text().strip()
    result = subprocess.run(
        [
            str(TOOLS / "kopia.exe"),
            "--config-file",
            str(TOOLS / "repository.config"),
            "--no-progress",
            *map(str, args),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(f"Kopia exited {result.returncode}: {result.stderr[-1500:]}")
    return result.stdout


def save(name, value):
    (EVIDENCE / name).write_text(json.dumps(value, indent=2), encoding="utf-8")


before = json.loads(kopia("snapshot", "list", "--json"))
save("kopia-inventory-before.json", before)
save("retention-global.json", json.loads(kopia("policy", "show", "--global", "--json")))
receipts = []
restore_latest.ROOT = SCRATCH
restore_latest.SOURCE = SOURCE
restore_latest.kopia = kopia
for version in (1, 2):
    sealed = (
        Path(r"\\wsl.localhost\Ubuntu\home\paxto\.cache\finrl-backup-proof-20261010")
        / f"capture-v{version}-reviewed/payload"
    )
    if not SOURCE.exists():
        shutil.copytree(sealed, SOURCE)
    else:
        # Model overwritten/deleted evidence in the SAME disposable snapshot source.
        for path in sorted(SOURCE.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            rel = path.relative_to(SOURCE)
            if not (sealed / rel).exists():
                if path.is_dir():
                    path.rmdir()
                else:
                    path.unlink()
        shutil.copytree(sealed, SOURCE, dirs_exist_ok=True)
    verify_restore(SOURCE)
    snap = json.loads(
        kopia("snapshot", "create", SOURCE, "--json", "--fail-fast", "--force-hash=100")
    )
    if (
        snap.get("incomplete")
        or snap.get("incompleteReason")
        or snap["rootEntry"].get("summ", {}).get("numFailed", 0)
    ):
        raise RuntimeError("Synthetic snapshot incomplete")
    save(f"kopia-snapshot-v{version}.json", snap)
    dest = SCRATCH / f"restored-v{version}"
    sys.argv = ["restore_latest.py", "--destination", str(dest)]
    restore_latest.main()
    validation = verify_restore(dest)
    assert (dest / "overwrite.txt").read_text() == f"version-{version}"
    assert (dest / "delete-me.txt").exists() == (version == 1)
    receipts.append(
        {
            "version": version,
            "snapshot_id": snap["id"],
            "destination": str(dest),
            "validation": validation,
            "manifest_sha256": hash_file(dest / "SHA256.json"),
        }
    )
    try:
        restore_latest.main()
    except RuntimeError as exc:
        assert "already exists" in str(exc)
    else:
        raise RuntimeError("Restore allowed an existing destination")

# Retrieve the older version again after v2 exists; no snapshot deletion/expiry.
older = SCRATCH / "restored-v1-after-v2"
kopia(
    "snapshot",
    "restore",
    json.loads((EVIDENCE / "kopia-snapshot-v1.json").read_text())["rootEntry"]["obj"],
    older,
    "--no-overwrite-files",
    "--no-overwrite-directories",
    "--no-overwrite-symlinks",
)
assert (
    verify_restore(older)["recovery"]["journal"]
    == receipts[0]["validation"]["recovery"]["journal"]
)
after = json.loads(kopia("snapshot", "list", "--json"))
save("kopia-inventory-after.json", after)
assert {s["id"] for s in before}.issubset({s["id"] for s in after})
save(
    "retention-effective-synthetic.json",
    json.loads(kopia("policy", "show", SOURCE, "--json")),
)
save(
    "kopia-proof.json",
    {
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "receipts": receipts,
        "existing_destination_refused": True,
        "older_version_recovered_after_new_version": True,
        "all_prior_snapshot_ids_preserved": True,
        "helper_hashes": {
            p.name: hash_file(p) for p in (BASE / "helpers").glob("*") if p.is_file()
        },
        "scratch": str(SCRATCH),
        "kopia_version": kopia("--version").strip(),
    },
)
print(
    "Synthetic Kopia snapshot/restore proof passed; two versions and prior "
    "snapshots preserved"
)
