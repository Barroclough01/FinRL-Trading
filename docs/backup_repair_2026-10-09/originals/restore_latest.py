"""Restore the latest FinRL snapshot into a NEW directory and verify every file."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from backup_finrl import ROOT, SOURCE, kopia

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path)
    args = parser.parse_args()
    snapshots = json.loads(kopia('snapshot', 'list', '--json'))
    complete = [s for s in snapshots
                if s.get('source', {}).get('path', '').lower() == str(SOURCE).lower()
                and not s.get('incomplete') and not s.get('incompleteReason')
                and s.get('stats', {}).get('errorCount', 0) == 0
                and s.get('rootEntry', {}).get('summ', {}).get('numFailed', 0) == 0]
    if not complete:
        raise RuntimeError('No complete FinRL snapshot exists')
    latest = max(complete, key=lambda s: s['startTime'])
    dest = args.destination or (Path(r'C:\Users\paxto\PaxtonBackups\Restored') /
                               datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    if dest.exists():
        raise RuntimeError('Restore destination already exists; choose a NEW empty path')
    kopia('snapshot', 'restore', latest['rootEntry']['obj'], dest,
          '--no-overwrite-files', '--no-overwrite-directories', '--no-overwrite-symlinks')
    expected = json.loads((dest / 'SHA256.json').read_text(encoding='utf-8'))
    actual = {}
    for item in dest.rglob('*'):
        if item.is_file() and item.relative_to(dest).as_posix() != 'SHA256.json':
            with item.open('rb') as handle:
                actual[item.relative_to(dest).as_posix()] = hashlib.file_digest(handle, 'sha256').hexdigest()
    if actual != expected:
        raise RuntimeError('Restored file inventory or hashes do not match the snapshot manifest')
    db = dest / 'wsl/database/finrl_trading.db'
    with sqlite3.connect(db.as_uri() + '?mode=ro', uri=True) as connection:
        if connection.execute('PRAGMA integrity_check').fetchall() != [('ok',)]:
            raise RuntimeError('Restored SQLite database integrity check failed')
    report = {'time_utc': datetime.now(timezone.utc).isoformat(), 'status': 'success',
              'snapshot_id': latest['id'], 'files_verified': len(actual),
              'database_integrity': 'ok', 'destination': str(dest)}
    (ROOT / 'last-restore.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'Restore verification failed: {exc}', file=sys.stderr)
        sys.exit(1)
