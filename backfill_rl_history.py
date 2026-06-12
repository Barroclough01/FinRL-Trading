#!/usr/bin/env python3
"""
backfill_rl_history.py
----------------------
One-time backfill of RL strategy performance for historical comparison.

This script:
  1. Identifies all existing weekly snapshot dates from live accounts
  2. Simulates the RL portfolio for each historical week
  3. Injects synthetic snapshots into the DB
  4. Regenerates the dashboard with complete RL history

Usage:
    python backfill_rl_history.py
"""

import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).parent

print("=" * 60)
print("  RL Strategy Historical Backfill")
print("=" * 60)
print()
print("This will simulate RL portfolio performance for all")
print("historical weeks and inject synthetic snapshots into")
print("the metrics database for comparison.")
print()

response = input("Continue? [y/N]: ")
if response.lower() != "y":
    print("Aborted.")
    sys.exit(0)

# Run backfill
print("\n[1/2] Running RL offline backfill...")
proc = subprocess.run(
    [sys.executable, "track_rl_offline.py", "--backfill"],
    cwd=project_root,
)

if proc.returncode != 0:
    print("\nERROR: RL backfill failed")
    sys.exit(1)

# Regenerate dashboard
print("\n[2/2] Regenerating dashboard with RL history...")
proc = subprocess.run(
    [sys.executable, "track_metrics.py", "--report-only"],
    cwd=project_root,
)

if proc.returncode != 0:
    print("\nWARNING: Dashboard regeneration failed")
    sys.exit(1)

print("\n" + "=" * 60)
print("  ✓ RL historical backfill complete")
print("=" * 60)
print()
print(f"Dashboard: {project_root / 'logs' / 'dashboard.html'}")
print(f"Metrics:   {project_root / 'logs' / 'comparison_metrics_latest.csv'}")
print()
