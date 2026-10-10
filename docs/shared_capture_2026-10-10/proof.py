"""Run source-copy verification behind actual primary-write/network barriers."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

NATIVE = Path("/mnt/c/Users/paxto/FinRL-Trading")
SCRATCH = Path("/home/paxto/.cache/finrl-shared-capture-20261010")
COPY = SCRATCH / "source"


def setup():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    COPY.mkdir(exist_ok=False)
    files = subprocess.check_output(["git", "-C", str(NATIVE), "ls-files"], text=True)
    for rel in files.splitlines():
        src, dest = NATIVE / rel, COPY / rel
        if src.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
    for rel in (
        "tests/test_capture_barrier.py",
        "docs/backup_repair_2026-10-09/helpers/capture_runtime.py",
    ):
        dest = COPY / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(NATIVE / rel, dest)
    (COPY / ".env").write_text(
        "APCA_ACCOUNTS=FinRL,AR\nAPCA_API_KEY_ID=fake\nAPCA_API_SECRET_KEY=fake\n"
        "PAPER_EXECUTION_JOURNAL_ENABLED=false\n",
        encoding="utf-8",
    )
    (SCRATCH / "temp").mkdir()
    return {
        "source_copy": str(COPY),
        "tracked_files": len(files.splitlines()),
        "fake_credentials": True,
        "operational_runtime_copied": False,
    }


def verify():
    os.environ["TMPDIR"] = str(SCRATCH / "temp")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    import tempfile

    tempfile.tempdir = str(SCRATCH / "temp")
    os.chdir(COPY)
    sys.path.insert(0, str(COPY))
    sys.path.insert(0, str(COPY / "docs/backup_repair_2026-10-09/tests"))
    from synthetic_proof import barriers

    print(json.dumps({"barriers": barriers([SCRATCH, Path("/dev")])}), flush=True)
    import pytest

    return pytest.main(sys.argv[2:] + ["--basetemp", str(SCRATCH / "temp/pytest")])


if __name__ == "__main__":
    if sys.argv[1] == "setup":
        print(json.dumps(setup(), indent=2))
    else:
        raise SystemExit(verify())
