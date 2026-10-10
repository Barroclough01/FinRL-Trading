# ruff: noqa: E402
"""Restore an existing legacy snapshot only into new disposable task scratch."""

import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "helpers"))
import backup_finrl
import restore_latest

tools = Path("C:/Users/paxto/PaxtonBackups/tools")
backup_finrl.ROOT = tools
backup_finrl.KOPIA = tools / "kopia.exe"
backup_finrl.CONFIG = tools / "repository.config"
scratch = Path("C:/Users/paxto/.codex/tmp/finrl-backup-proof-20261010")
restore_latest.ROOT = scratch
restore_latest.SOURCE = tools / "staging/finrl-runtime"
destination = scratch / "restored-legacy-pre-journal"
sys.argv = ["restore_latest.py", "--destination", str(destination)]
restore_latest.main()
receipt = json.loads((scratch / "last-restore.json").read_text())
assert receipt["recovery"]["coverage"] == "legacy_pre_journal"
assert receipt["recovery"]["execution_authorized"] is False
(BASE / "evidence/legacy-kopia-proof.json").write_text(json.dumps(receipt, indent=2))
