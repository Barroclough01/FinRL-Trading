"""Observable global contention, entrypoint and killed-parent child lifetime."""

import fcntl
import importlib.util
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
import types
from pathlib import Path
from unittest.mock import Mock

import pytest

import recover_paper_execution as recovery
import run_paper_trading as runner
from src.trading import execution_journal as journal

HELPERS = Path(__file__).resolve().parents[1] / "docs/backup_repair_2026-10-09/helpers"
spec = importlib.util.spec_from_file_location(
    "capture_runtime", HELPERS / "capture_runtime.py"
)
assert spec is not None and spec.loader is not None
capture_runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture_runtime)
contract = capture_runtime.load_contract(HELPERS / "recovery_contract.py")


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(journal, "require_execution_host", lambda *a: None)
    monkeypatch.setattr(journal, "EXECUTION_ROOT", tmp_path)
    monkeypatch.setenv(journal.GATE, "true")
    (tmp_path / "data").mkdir()
    (tmp_path / "results").mkdir()
    with sqlite3.connect(tmp_path / "data/finrl_trading.db") as db:
        db.execute("CREATE TABLE fixture(value TEXT)")
    return tmp_path


def exclusive(root):
    handle = (root / "data/execution_capture.lock").open("a+b")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def assert_no_intents(root):
    assert not (root / "logs/execution_attempts").exists()
    assert not (root / "data/paper_execution_journal.sqlite3").exists()
    assert not (root / "data/execution_locks").exists()


@pytest.mark.parametrize("alias", ["FinRL", "new-never-enumerated-account"])
def test_capture_blocks_direct_account_before_any_broker_or_artifacts(
    root, monkeypatch, alias
):
    core = Mock()
    monkeypatch.setattr(runner, "_run_journal_account", core)
    with exclusive(root):
        with pytest.raises(ValueError, match="backup capture owns"):
            runner.run_account({"name": alias}, "2026-10-09", False)
    core.assert_not_called()
    assert_no_intents(root)


def test_capture_blocks_recovery_before_identity_or_artifacts(root):
    manager = Mock()
    with exclusive(root):
        with pytest.raises(ValueError, match="backup capture owns"):
            recovery.recover_account(manager, "new-account", "2026-10-09", root)
    manager.assert_not_called()
    assert manager.mock_calls == []
    assert_no_intents(root)


def test_capture_blocks_main_before_attempt_and_tail(root, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["runner", "--date", "2026-10-09"])
    monkeypatch.setattr(runner, "resolve_run_date", lambda day: day)
    tail = Mock()
    monkeypatch.setattr(runner, "_main_run", tail)
    with exclusive(root):
        with pytest.raises(ValueError, match="backup capture owns"):
            runner.main()
    tail.assert_not_called()
    assert runner._ATTEMPT_DIR is None
    assert_no_intents(root)


def test_production_lowlevel_without_owner_fails_before_artifacts(root):
    for call in (
        lambda: journal.new_attempt(root),
        lambda: journal.ExecutionJournal(root),
        lambda: journal.write_evidence(root / "logs/new.json", {}),
    ):
        with pytest.raises(ValueError, match="ownership required"):
            call()
    assert_no_intents(root)


def test_open_journal_cannot_be_used_after_owner_closes(root):
    with journal.capture_writer(root):
        store = journal.ExecutionJournal(root)
    try:
        with pytest.raises(ValueError, match="ownership required"):
            _ = store.conn
    finally:
        store.close()


@pytest.mark.parametrize("descriptor", ["stale", "wrong-inode", "unlocked"])
def test_inherited_descriptor_cannot_be_spoofed(root, monkeypatch, descriptor):
    lockpath = root / "data/execution_capture.lock"
    lockpath.touch()
    file = root / "wrong" if descriptor == "wrong-inode" else lockpath
    with file.open("a+b") as handle:
        fd = -1 if descriptor == "stale" else handle.fileno()
        monkeypatch.setenv("FINRL_CAPTURE_DESCRIPTOR", str(fd))
        with pytest.raises(ValueError, match="Inherited capture descriptor held"):
            journal.capture_subprocess_kwargs()


def test_nested_writer_retains_outer_lock_through_error_and_final_write(root):
    with journal.capture_writer(root):
        attempt = journal.new_attempt(root)
        with pytest.raises(RuntimeError):
            with journal.capture_writer(root):
                raise RuntimeError("nested failure")
        with pytest.raises(ValueError, match="writer active"):
            capture_runtime.capture(root, root / "blocked", contract)
        assert not (root / "blocked").exists()
        journal.write_evidence(attempt / "final.json", {"interrupted": True})
    with exclusive(root):
        pass
    assert (attempt / "final.json").exists()


def test_main_keeps_owner_through_tail_cleanup(root, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["runner", "--date", "2026-10-09"])
    monkeypatch.setattr(runner, "resolve_run_date", lambda day: day)

    def tail(args):
        journal.require_capture_ownership(root)
        with pytest.raises(ValueError, match="writer active"):
            capture_runtime.capture(root, root / "blocked", contract)
        assert runner._ATTEMPT_DIR is not None
        journal.write_evidence(runner._ATTEMPT_DIR / "final.json", {})
        raise SystemExit(7)

    monkeypatch.setattr(runner, "_main_run", tail)
    with pytest.raises(SystemExit, match="7"):
        runner.main()
    assert runner._ATTEMPT_DIR is None
    with exclusive(root):
        pass


def test_disabled_direct_legacy_does_not_initialize_global_lock(root, monkeypatch):
    monkeypatch.setenv(journal.GATE, "false")
    legacy = Mock(return_value={"legacy": True})
    monkeypatch.setattr(runner, "_run_account", legacy)
    assert runner.run_account({"name": "FinRL"}, "2026-10-09", False) == {
        "legacy": True
    }
    assert not (root / "data/execution_capture.lock").exists()
    assert_no_intents(root)


def wait_for(path):
    deadline = time.monotonic() + 10
    while not path.exists():
        if time.monotonic() > deadline:
            raise AssertionError(f"Fixture did not create {path}")
        time.sleep(0.02)


def test_killed_parent_child_inherits_lock_until_final_evidence(root):
    script = root / "parent.py"
    (root / "grandchild.py").write_text("""import sys,time
from pathlib import Path
root = Path(sys.argv[1])
(root / "grandchild-ready").touch()
while not (root / "release-grandchild").exists(): time.sleep(0.02)
Path(sys.argv[2]).write_text('{"interrupted":true}')
(root / "child-done").touch()
""")
    (root / "child.py").write_text("""import sys,time,subprocess,os
from pathlib import Path
from src.trading import execution_journal as j
root = Path(sys.argv[1])
j.EXECUTION_ROOT = root
j.require_execution_host = lambda *a: None
subprocess.Popen([sys.executable, '-B', str(root/'grandchild.py'),
                  str(root), sys.argv[2]],
    **j.capture_subprocess_kwargs())
(root / 'child-ready').write_text(str(os.getpid()))
while not (root / 'release-child').exists(): time.sleep(0.02)
""")
    code = """import sys
from pathlib import Path
from src.trading import execution_journal as j
import run_paper_trading as r
root = Path(sys.argv[1])
j.require_execution_host = lambda *a: None
j.EXECUTION_ROOT = root
with j.capture_writer(root):
    attempt = j.new_attempt(root)
    store = j.ExecutionJournal(root)
    one = store.open_session(endpoint=j.PAPER_ENDPOINT, broker_id="fixture",
        alias="New", day="2026-10-09", config_hash="fixture", targets={},
        snapshot={}, attempt=attempt)
    one.submit_batch("sell", [], lambda payload: {})
    def unknown(payload):
        raise RuntimeError("fixture transport interruption")
    try:
        one.submit_batch("buy", [{"symbol":"A", "qty":"1", "side":"buy",
            "type":"market", "time_in_force":"day", "extended_hours":False}], unknown)
    except RuntimeError:
        pass
    store.close()
    r._run_subprocess([sys.executable, '-B', str(root/'child.py'), str(root),
                      str(attempt/'child-final.json')], check=True)
"""
    script.write_text(code)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(runner.__file__).resolve().parent)
    with (root / "child-output.txt").open("w") as out:
        parent = subprocess.Popen(
            [sys.executable, "-B", script, str(root)], env=env, stdout=out, stderr=out
        )
        try:
            wait_for(root / "child-ready")
            wait_for(root / "grandchild-ready")
            parent.send_signal(signal.SIGKILL)
            parent.wait(timeout=10)
            os.kill(int((root / "child-ready").read_text()), signal.SIGKILL)
            with pytest.raises(ValueError, match="writer active"):
                capture_runtime.capture(root, root / "premature", contract)
            assert not (root / "premature").exists()
            (root / "release-grandchild").touch()
            wait_for(root / "child-done")
            # Last descriptor closes after child-done; bounded polling is test only.
            deadline = time.monotonic() + 3
            while True:
                try:
                    handle = exclusive(root)
                    handle.close()
                    break
                except BlockingIOError:
                    assert time.monotonic() < deadline
                    time.sleep(0.02)
            with sqlite3.connect(root / "data/paper_execution_journal.sqlite3") as db:
                assert (
                    db.execute("SELECT state FROM execution_sessions").fetchone()[0]
                    == "started"
                )
            assert list((root / "logs/execution_attempts").rglob("child-final.json"))
            captured = capture_runtime.capture(root, root / "after-crash", contract)
            holds = captured["recovery"]["journal"]["holds"]
            assert any("intent_attempted_unknown" in hold for hold in holds)
            (root / "crash-proof.json").write_text(json.dumps(captured, indent=2))
        finally:
            (root / "release-child").touch()
            (root / "release-grandchild").touch()
            if parent.poll() is None:
                parent.kill()
                parent.wait(timeout=10)


def synthetic_source(root, version=1):
    spec = importlib.util.spec_from_file_location(
        "old_fixture", HELPERS.parent / "tests/synthetic_proof.py"
    )
    assert spec is not None and spec.loader is not None
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    source = root / "fixture-source"
    fixture.make_fixture(source, version)
    (source / "results").mkdir()
    return source


def test_capture_descriptor_covers_every_export_copy_and_validation(root, monkeypatch):
    source = synthetic_source(root)
    monkeypatch.setattr(runner, "project_root", str(source))
    checkpoints = []

    def check(name):
        checkpoints.append(name)
        for alias in ("FinRL", "brand-new-account"):
            with pytest.raises(ValueError, match="backup capture owns"):
                runner.run_account({"name": alias}, "2026-10-09", False)

    before = sorted(
        p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file()
    )
    result = capture_runtime.capture(
        source, root / "captured", contract, checkpoint=check
    )
    after = sorted(
        p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file()
    )
    assert set(after) - set(before) == {"data/execution_capture.lock"}
    assert checkpoints == [
        "owned",
        "comparison_exported",
        "journal_exported",
        "attempts_copied",
        "verified",
    ]
    assert result["recovery"]["execution_authorized"] is False
    assert any(
        "intent_attempted_unknown" in h for h in result["recovery"]["journal"]["holds"]
    )
    assert any(
        "interrupted_session" in h for h in result["recovery"]["journal"]["holds"]
    )
    assert (root / "captured" / contract.ATTEMPTS / "attempt-abandoned").is_dir()


@pytest.mark.parametrize(
    "defect", ["missing", "corrupt", "reference", "required", "partial-json"]
)
def test_invalid_required_capture_refuses_complete_contract(root, defect):
    source = synthetic_source(root)
    db = source / "data/paper_execution_journal.sqlite3"
    if defect in {"missing", "required"}:
        db.unlink()
    elif defect == "corrupt":
        db.write_bytes(b"corrupt")
    elif defect == "partial-json":
        (source / "logs/execution_attempts/attempt-main/partial.json").write_text("{")
    else:
        import shutil

        shutil.rmtree(source / "logs/execution_attempts/attempt-main")
    stage = root / "invalid"
    with pytest.raises((ValueError, sqlite3.DatabaseError)):
        capture_runtime.capture(source, stage, contract, required=defect == "required")
    assert not (stage / "SHA256.json").exists()
    assert not (stage / contract.CONTRACT).exists()


def test_unreadable_walk_is_an_actionable_failure(root, monkeypatch):
    source = synthetic_source(root)

    def broken_walk(*args, **kwargs):
        kwargs["onerror"](PermissionError("fixture unreadable directory"))
        yield

    monkeypatch.setattr(capture_runtime.os, "walk", broken_walk)
    with pytest.raises(PermissionError, match="unreadable"):
        capture_runtime.capture(source, root / "unreadable", contract)
    assert not (root / "unreadable/SHA256.json").exists()


def backup_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "msvcrt", types.ModuleType("msvcrt"))
    monkeypatch.setitem(sys.modules, "recovery_contract", contract)
    spec = importlib.util.spec_from_file_location(
        "backup_candidate", HELPERS / "backup_finrl.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_journal_appearing_after_initial_probe_makes_marker_monotonic(
    root, monkeypatch
):
    backup = backup_module(monkeypatch)
    backup.ROOT = root
    backup.WSL_ROOT = root / "wsl"
    backup.WINDOWS_ROOT = root / "native"
    backup.WSL_ROOT.mkdir()
    backup.WINDOWS_ROOT.mkdir()
    assert backup.capture_requirement() is False
    source = synthetic_source(root)
    captured = capture_runtime.capture(source, root / "race-captured", contract)
    backup.retain_capture_requirement(captured)
    assert backup.capture_requirement() is True
    assert (
        json.loads((root / "journal-required.json").read_text())["journal_required"]
        is True
    )
    # Deleted source evidence never relaxes marker requirement; next capture holds.
    with pytest.raises(ValueError, match="journal-required database missing"):
        capture_runtime.capture(root, root / "deleted-journal", contract, required=True)


@pytest.mark.parametrize(
    "content", ["not-json", "{}", '{"format_version":1,"journal_required":false}']
)
def test_invalid_marker_holds(root, monkeypatch, content):
    backup = backup_module(monkeypatch)
    backup.ROOT = backup.WSL_ROOT = backup.WINDOWS_ROOT = root
    (root / "journal-required.json").write_text(content)
    with pytest.raises((ValueError, RuntimeError)):
        backup.capture_requirement()
