"""Local FinRL evidence backup. Never invokes project runners or broker APIs."""

import json
import msvcrt
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KOPIA = ROOT / "kopia.exe"
CONFIG = ROOT / "repository.config"
STAGING = ROOT / "staging"
SOURCE = STAGING / "finrl-runtime"
WSL_ROOT = Path(r"\\wsl.localhost\Ubuntu\home\paxto\stock-trading\FinRL-Trading")
WINDOWS_ROOT = Path(r"C:\Users\paxto\FinRL-Trading")


def capture_requirement():
    """Monotonic marker; native journal writers are outside the WSL protocol."""
    marker = ROOT / "journal-required.json"
    if marker.is_symlink() or (marker.exists() and not marker.is_file()):
        raise RuntimeError("Unsafe journal-required marker; backup held")
    for source in (WSL_ROOT, WINDOWS_ROOT):
        if not source.is_dir() or source.is_symlink():
            raise RuntimeError(
                "Required source checkout unavailable/unsafe; backup held"
            )
    if marker.exists():
        state = json.loads(marker.read_text(encoding="utf-8"))
        if (
            state.get("journal_required") is not True
            or state.get("format_version") != 1
        ):
            raise RuntimeError("Invalid journal-required marker; backup held")

    def observed(source):
        for rel in (
            "data/paper_execution_journal.sqlite3",
            "data/paper_execution_journal.sqlite3-wal",
            "data/paper_execution_journal.sqlite3-shm",
            "data/paper_execution_journal.sqlite3-journal",
            "logs/execution_attempts",
            "data/execution_locks",
        ):
            path = source / rel
            if path.is_symlink() or (
                path.exists() and (not path.is_dir() or any(path.iterdir()))
            ):
                return True
        return False

    native = observed(WINDOWS_ROOT)
    seen = observed(WSL_ROOT) or native
    if seen and not marker.exists():
        with marker.open("x", encoding="utf-8") as handle:
            json.dump(
                {
                    "format_version": 1,
                    "journal_required": True,
                    "observed_utc": datetime.now(timezone.utc).isoformat(),
                },
                handle,
            )
            handle.flush()
            os.fsync(handle.fileno())
    if native:
        raise RuntimeError(
            "Native journal-era state is outside WSL capture; backup held"
        )
    return marker.exists()


def retain_capture_requirement(capture):
    if capture["recovery"]["coverage"] != "coordinated_wsl_journal":
        return
    marker = ROOT / "journal-required.json"
    if not marker.exists():
        with marker.open("x", encoding="utf-8") as handle:
            json.dump(
                {
                    "format_version": 1,
                    "journal_required": True,
                    "observed_utc": datetime.now(timezone.utc).isoformat(),
                },
                handle,
            )
            handle.flush()
            os.fsync(handle.fileno())
    # Validate existing markers again; never publish with an unsafe/invalid marker.
    capture_requirement()


def run(args, timeout=600, env=None):
    result = subprocess.run(
        [str(x) for x in args], capture_output=True, text=True, timeout=timeout, env=env
    )
    if result.returncode:
        raise RuntimeError(
            f"{Path(str(args[0])).name} exited {result.returncode}: "
            f"{result.stderr[-1800:]}"
        )
    return result.stdout


def kopia(*args):
    # Noninteractive subprocesses cannot rely on Kopia's interactive password fallback.
    # The recovery key is restricted to this Windows user and SYSTEM, never
    # logged or passed in argv.
    env = os.environ.copy()
    env["KOPIA_PASSWORD"] = (
        (ROOT / "recovery-key.txt").read_text(encoding="utf-8").strip()
    )
    return run([KOPIA, "--config-file", CONFIG, "--no-progress", *args], env=env)


def remove_stage(path):
    resolved = path.resolve()
    if resolved.parent != STAGING.resolve() or path.is_symlink():
        raise RuntimeError(f"Refusing to remove unexpected staging directory: {path}")
    shutil.rmtree(path)


def copy_files(src, dest, extensions=None, exclude_dirs=()):
    if not src.is_dir():
        raise RuntimeError(f"Required backup source is unavailable: {src}")

    def fail_walk(error):
        raise error

    for current, dirs, names in os.walk(src, followlinks=False, onerror=fail_walk):
        current = Path(current)
        if any((current / d).is_symlink() for d in dirs):
            raise RuntimeError(f"Unexpected symlink in backup source: {current}")
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for name in names:
            item = current / name
            if item.is_symlink():
                raise RuntimeError(f"Unexpected symlink in backup source: {item}")
            if extensions is not None and item.suffix.lower() not in extensions:
                continue
            target = dest / item.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def backup():
    required = capture_requirement()
    if shutil.disk_usage(ROOT).free < 10 * 1024**3:
        raise RuntimeError(
            "Less than 10 GiB free; backup stopped before staging. Review "
            "local backup storage."
        )
    STAGING.mkdir(exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="run-", dir=STAGING))
    try:
        copy_files(WINDOWS_ROOT / "logs", stage / "windows/logs")
        # One WSL process owns the exclusive FD through exports, attempt copy,
        # inventory and reference verification. No operational project imports.
        linux_stage = "/mnt/c/" + stage.as_posix()[3:]
        helper = ROOT / "capture_runtime.py"
        contract_helper = ROOT / "recovery_contract.py"
        capture = json.loads(
            run(
                [
                    "wsl.exe",
                    "-d",
                    "Ubuntu",
                    "--exec",
                    "/home/paxto/stock-trading/FinRL-Trading/finrl-env/bin/python",
                    "-B",
                    "-c",
                    helper.read_text(encoding="utf-8"),
                    "/home/paxto/stock-trading/FinRL-Trading",
                    linux_stage,
                    "/mnt/c/" + contract_helper.as_posix()[3:],
                    "true" if required else "false",
                ]
            )
        )
        retain_capture_requirement(capture)
        files = sorted(p for p in stage.rglob("*") if p.is_file())
        if SOURCE.exists():
            remove_stage(SOURCE)
        stage.rename(SOURCE)
        snapshot = json.loads(
            kopia(
                "snapshot",
                "create",
                SOURCE,
                "--json",
                "--fail-fast",
                "--force-hash=100",
            )
        )
        if (
            snapshot.get("incomplete")
            or snapshot.get("incompleteReason")
            or snapshot.get("rootEntry", {}).get("summ", {}).get("numFailed", 0)
        ):
            raise RuntimeError(
                "Kopia produced an incomplete snapshot; previous backups "
                "remain available"
            )
        report = {
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "status": "success",
            "files": len(files),
            "bytes": sum(p.stat().st_size for p in SOURCE.rglob("*") if p.is_file()),
            "snapshot": snapshot,
        }
        return report
    finally:
        if stage.exists():
            remove_stage(stage)


def main():
    ROOT.mkdir(exist_ok=True)
    with (ROOT / "run.lock").open("a+b") as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise RuntimeError("Another backup is already running")
        try:
            report = backup()
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
    temp = ROOT / "last-run.json.tmp"
    temp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temp.replace(ROOT / "last-run.json")
    print(json.dumps({k: v for k, v in report.items() if k != "snapshot"}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        failure = {
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "status": "failed",
            "error": str(exc),
        }
        (ROOT / "last-run.json").write_text(
            json.dumps(failure, indent=2), encoding="utf-8"
        )
        print(json.dumps(failure), file=sys.stderr)
        sys.exit(1)
