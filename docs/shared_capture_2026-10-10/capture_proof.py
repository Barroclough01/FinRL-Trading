"""Permanent disposable captures using exact candidate helper bytes."""

import importlib.util
import json
import sys
from pathlib import Path

SCRATCH = Path("/home/paxto/.cache/finrl-shared-capture-20261010")
SOURCE = SCRATCH / "source"
sys.path.insert(0, str(SOURCE / "docs/backup_repair_2026-10-09/tests"))
from synthetic_proof import barriers, make_fixture  # noqa: E402

HELPERS = SOURCE / "docs/backup_repair_2026-10-09/helpers"
spec = importlib.util.spec_from_file_location(
    "capture_runtime", HELPERS / "capture_runtime.py"
)
assert spec is not None and spec.loader is not None
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)
contract = capture.load_contract(HELPERS / "recovery_contract.py")
barrier = barriers([SCRATCH, Path("/dev")])
outcomes = []
for version in (1, 2):
    source = SCRATCH / f"journal-source-v{version}-final"
    make_fixture(source, version)
    (source / "results").mkdir()
    (source / "overwrite.json").write_text(json.dumps({"version": version}))
    if version == 1:
        (source / "delete-me.json").write_text('{"version_one_only":true}')
    stage = SCRATCH / f"capture-v{version}-final"
    result = capture.capture(source, stage, contract)
    outcomes.append(
        {
            "version": version,
            "source": str(source),
            "stage": str(stage),
            "result": result,
        }
    )
print(json.dumps({"barrier": barrier, "outcomes": outcomes}, indent=2))
