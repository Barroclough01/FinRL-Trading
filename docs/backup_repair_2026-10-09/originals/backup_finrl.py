"""Local FinRL evidence backup. Never invokes project runners or broker APIs."""
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
KOPIA = ROOT / 'kopia.exe'
CONFIG = ROOT / 'repository.config'
STAGING = ROOT / 'staging'
SOURCE = STAGING / 'finrl-runtime'
WSL_ROOT = Path(r'\\wsl.localhost\Ubuntu\home\paxto\stock-trading\FinRL-Trading')
WINDOWS_ROOT = Path(r'C:\Users\paxto\FinRL-Trading')

def run(args, timeout=600, env=None):
    result = subprocess.run([str(x) for x in args], capture_output=True, text=True, timeout=timeout, env=env)
    if result.returncode:
        raise RuntimeError(f'{Path(str(args[0])).name} exited {result.returncode}: {result.stderr[-1800:]}')
    return result.stdout

def kopia(*args):
    # Noninteractive subprocesses cannot rely on Kopia's interactive password fallback.
    # The recovery key is restricted to this Windows user and SYSTEM, never logged or passed in argv.
    env = os.environ.copy()
    env['KOPIA_PASSWORD'] = (ROOT / 'recovery-key.txt').read_text(encoding='utf-8').strip()
    return run([KOPIA, '--config-file', CONFIG, '--no-progress', *args], env=env)

def remove_stage(path):
    resolved = path.resolve()
    if resolved.parent != STAGING.resolve() or path.is_symlink():
        raise RuntimeError(f'Refusing to remove unexpected staging directory: {path}')
    shutil.rmtree(path)

def copy_files(src, dest, extensions=None):
    if not src.is_dir():
        raise RuntimeError(f'Required backup source is unavailable: {src}')
    def fail_walk(error):
        raise error
    for current, dirs, names in os.walk(src, followlinks=False, onerror=fail_walk):
        current = Path(current)
        if any((current / d).is_symlink() for d in dirs):
            raise RuntimeError(f'Unexpected symlink in backup source: {current}')
        for name in names:
            item = current / name
            if item.is_symlink():
                raise RuntimeError(f'Unexpected symlink in backup source: {item}')
            if extensions is not None and item.suffix.lower() not in extensions:
                continue
            target = dest / item.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)

EXPORT_SQLITE = '''import sqlite3, sys
from pathlib import Path
source = Path(sys.argv[1])
with sqlite3.connect(source.as_uri() + '?mode=ro', uri=True, timeout=30) as src:
    with sqlite3.connect(sys.argv[2]) as dst:
        src.backup(dst, pages=4096, sleep=0.1)
        result = dst.execute('PRAGMA integrity_check').fetchall()
        if result != [('ok',)]:
            raise RuntimeError('Exported database integrity check failed')
print('SQLite export integrity: ok')
'''

def backup():
    if shutil.disk_usage(ROOT).free < 10 * 1024**3:
        raise RuntimeError('Less than 10 GiB free; backup stopped before staging. Review local backup storage.')
    STAGING.mkdir(exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='run-', dir=STAGING))
    try:
        copy_files(WSL_ROOT / 'logs', stage / 'wsl/logs')
        copy_files(WSL_ROOT / 'results', stage / 'wsl/results', {'.csv', '.json', '.html', '.png'})
        copy_files(WINDOWS_ROOT / 'logs', stage / 'windows/logs')
        for item in WSL_ROOT.iterdir():
            if item.is_file() and not item.is_symlink() and item.suffix.lower() in {'.csv', '.json', '.html'}:
                target = stage / 'wsl/root-reports' / item.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)
        db = stage / 'wsl/database/finrl_trading.db'
        db.parent.mkdir(parents=True)
        linux_dest = '/mnt/c/' + db.as_posix()[3:]
        run(['wsl.exe', '-d', 'Ubuntu', '--exec',
             '/home/paxto/stock-trading/FinRL-Trading/finrl-env/bin/python',
             '-c', EXPORT_SQLITE,
             '/home/paxto/stock-trading/FinRL-Trading/data/finrl_trading.db', linux_dest])
        files = sorted(p for p in stage.rglob('*') if p.is_file())
        manifest = {}
        for p in files:
            with p.open('rb') as handle:
                manifest[p.relative_to(stage).as_posix()] = hashlib.file_digest(handle, 'sha256').hexdigest()
        # Each component is captured during this run; only the DB is a transaction-consistent snapshot.
        (stage / 'SHA256.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        if SOURCE.exists():
            remove_stage(SOURCE)
        stage.rename(SOURCE)
        snapshot = json.loads(kopia('snapshot', 'create', SOURCE, '--json', '--fail-fast', '--force-hash=100'))
        if (snapshot.get('incomplete') or snapshot.get('incompleteReason')
                or snapshot.get('rootEntry', {}).get('summ', {}).get('numFailed', 0)):
            raise RuntimeError('Kopia produced an incomplete snapshot; previous backups remain available')
        report = {'time_utc': datetime.now(timezone.utc).isoformat(), 'status': 'success',
                  'files': len(files), 'bytes': sum(p.stat().st_size for p in SOURCE.rglob('*') if p.is_file()),
                  'snapshot': snapshot}
        return report
    finally:
        if stage.exists():
            remove_stage(stage)

def main():
    ROOT.mkdir(exist_ok=True)
    with (ROOT / 'run.lock').open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise RuntimeError('Another backup is already running')
        try:
            report = backup()
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
    temp = ROOT / 'last-run.json.tmp'
    temp.write_text(json.dumps(report, indent=2), encoding='utf-8')
    temp.replace(ROOT / 'last-run.json')
    print(json.dumps({k: v for k, v in report.items() if k != 'snapshot'}))

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        failure = {'time_utc': datetime.now(timezone.utc).isoformat(), 'status': 'failed', 'error': str(exc)}
        (ROOT / 'last-run.json').write_text(json.dumps(failure, indent=2), encoding='utf-8')
        print(json.dumps(failure), file=sys.stderr)
        sys.exit(1)
