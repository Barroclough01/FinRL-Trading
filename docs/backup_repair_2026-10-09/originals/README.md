# Free local FinRL backups

This setup has no software subscription, API use, or cloud-storage charge.
Everything stays on this laptop. It protects against accidentally overwritten
or deleted files; it does not protect against loss or failure of this laptop.

## Included

- FinRL's scheduled Ubuntu checkout: logs; results ending in `.csv`, `.json`,
  `.html`, or `.png`; root CSV/JSON/HTML files; and a SQLite native backup of
  `data/finrl_trading.db` that passes an integrity check before snapshotting.
- Logs from the native Windows FinRL development checkout.

Model pickle files, bulk market-price caches, credentials, other projects,
personal documents/photos, and VPS files are outside this initial scope.
The database is consistent at export time. Other files are captured during the
same backup run, not as one atomic snapshot of the entire project.

## Automatic backup

Windows Task Scheduler task: **Paxton FinRL Local Backup**.
Runs daily at 8 PM Eastern when signed in. Missed runs start when Windows can
run them again. It does not wake the laptop. No Codex subscription or open Codex
window is needed. Ubuntu/WSL and the existing FinRL Python environment are needed.

Retains 3 latest, 7 daily, 4 weekly, and 3 monthly snapshots, sharing unchanged
data between versions. A run refuses to start if less than 10 GiB is free.
Errors stop the backup and set a nonzero task result; older snapshots survive.

## Run or restore manually

- Open `tools/Backup Now.cmd` to make a fresh backup.
- Open `tools/Restore Latest.cmd` to restore into a NEW timestamped directory
  under `Restored/`, then verify every file hash and database integrity.
- Read `tools/last-run.json` for the latest run result and UTC time, or
  `tools/last-restore.json` for the latest successful recovery drill.
- Repository: `kopia/`. Do not edit or delete its internal files.

Restore copies never overwrite the running project. Review recovered data before
using it to replace anything. Old snapshots can be inspected with:

```powershell
& 'C:\Users\paxto\PaxtonBackups\tools\kopia.exe' --config-file 'C:\Users\paxto\PaxtonBackups\tools\repository.config' snapshot list
```

## Recovery key and maintenance

`tools/recovery-key.txt` contains the encryption password. Access is restricted
to Paxton's Windows account and SYSTEM. The helpers pass it privately to Kopia
through the child process environment, never in command arguments or logs.
Save this key separately in a password manager or another safe location.

Once a month, check `tools/last-run.json` and remaining disk space. Every three
months, run `Restore Latest.cmd`. Review stable Kopia releases periodically;
the scheduled task uses the pinned CLI in `tools`, independent of GUI updates.

To pause: disable **Paxton FinRL Local Backup** in Windows Task Scheduler.
To add disk-failure protection without a subscription, use an existing external
drive and separately configure/test the backup destination there.
