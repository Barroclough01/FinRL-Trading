#!/usr/bin/env python3
"""
patch_dashboard_cash.py
-----------------------
Fixes the FinRL (and AR) position percentage display in track_metrics.py.

Bug: generate_html_dashboard() never queries the `cash` column from
     weekly_snapshot, so the positions table only sums to ~70-90% with
     no cash row shown.

Fix:
  1. Add `cash` to the SELECT query
  2. Pass it through to the JS data structure
  3. Render a synthetic "CASH" row in the positions table

Usage (from ~/stock-trading/FinRL-Trading in WSL):
    python3 patch_dashboard_cash.py
    python3 patch_dashboard_cash.py --dry-run   # print diff only
"""

import argparse
import re
import shutil
from pathlib import Path
from datetime import datetime

TARGET = Path("track_metrics.py")


def apply_patch(dry_run: bool = False) -> None:
    if not TARGET.exists():
        raise FileNotFoundError(f"{TARGET} not found. Run from project root.")

    src = TARGET.read_text(encoding="utf-8")
    original = src

    # -------------------------------------------------------------------------
    # Patch 1: Add `cash` to the SELECT inside generate_html_dashboard
    # -------------------------------------------------------------------------
    old_select = (
        "        SELECT snapshot_date, account, portfolio_value, weekly_return,\n"
        "               cumulative_return, spy_weekly_return, spy_cumulative_return, positions_json\n"
        "        FROM weekly_snapshot\n"
        "        ORDER BY snapshot_date ASC, account"
    )
    new_select = (
        "        SELECT snapshot_date, account, portfolio_value, weekly_return,\n"
        "               cumulative_return, spy_weekly_return, spy_cumulative_return, positions_json,\n"
        "               cash\n"
        "        FROM weekly_snapshot\n"
        "        ORDER BY snapshot_date ASC, account"
    )
    if old_select not in src:
        print("SKIP Patch 1: SELECT already patched or pattern not found")
    else:
        src = src.replace(old_select, new_select)
        print("OK   Patch 1: Added `cash` to SELECT")

    # -------------------------------------------------------------------------
    # Patch 2: Unpack `cash` from the row tuple + pass to JS data
    # -------------------------------------------------------------------------
    old_unpack = (
        "        snap_date, account, value, wkly, cum, spy_wkly, spy_cum, pos_json = r\n"
        "        if account not in accounts_data:\n"
        "            accounts_data[account] = []\n"
        "        accounts_data[account].append(\n"
        "            {\n"
        '                "date": snap_date,\n'
        '                "value": value,\n'
        '                "weekly": wkly,\n'
        '                "cumulative": cum,\n'
        '                "spy_weekly": spy_wkly,\n'
        '                "spy_cumulative": spy_cum,\n'
        '                "positions": json.loads(pos_json) if pos_json else [],\n'
        "            }\n"
        "        )"
    )
    new_unpack = (
        "        snap_date, account, value, wkly, cum, spy_wkly, spy_cum, pos_json, cash_val = r\n"
        "        if account not in accounts_data:\n"
        "            accounts_data[account] = []\n"
        "        accounts_data[account].append(\n"
        "            {\n"
        '                "date": snap_date,\n'
        '                "value": value,\n'
        '                "cash": cash_val or 0.0,\n'
        '                "weekly": wkly,\n'
        '                "cumulative": cum,\n'
        '                "spy_weekly": spy_wkly,\n'
        '                "spy_cumulative": spy_cum,\n'
        '                "positions": json.loads(pos_json) if pos_json else [],\n'
        "            }\n"
        "        )"
    )
    if old_unpack not in src:
        print("SKIP Patch 2: row unpack already patched or pattern not found")
    else:
        src = src.replace(old_unpack, new_unpack)
        print("OK   Patch 2: Added cash_val to row unpack + JS data")

    # -------------------------------------------------------------------------
    # Patch 3: Also pass `cash` through latest_js (latestData in JS)
    # The `latest` dict comes from accounts_data[acct][-1], so cash is already
    # included after Patch 2. No extra change needed here.
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Patch 4: JS — render CASH row in the positions table
    #
    # Original JS snippet (inside the positions-section forEach):
    #   const rows = positions.sort(...)...map(p => { ... }).join('');
    #   posSection.innerHTML += `...<tbody>${rows || ...}</tbody>...`;
    #
    # We inject a cash row AFTER the equity positions.
    # -------------------------------------------------------------------------
    old_js_tbody = '      <tbody>${{rows || \'<tr><td colspan="4" style="color:var(--muted)">No positions</td></tr>\'}}</tbody>'
    new_js_tbody = (
        "      <tbody>\n"
        "        ${{rows}}\n"
        "        ${{(() => {{\n"
        "          const cashVal = d.cash || 0;\n"
        "          if (cashVal <= 0) return '';\n"
        "          const cashW = cashVal / d.value * 100;\n"
        '          return `<tr style="opacity:0.6">\n'
        "            <td>CASH</td>\n"
        "            <td>$${{cashVal.toLocaleString('en-US', {{minimumFractionDigits:2, maximumFractionDigits:2}})}}</td>\n"
        "            <td>\n"
        "              ${{cashW.toFixed(1)}}%\n"
        '              <div class="weight-bar"><div class="weight-fill" style="width:${{Math.min(cashW*2,100)}}%;background:var(--muted)"></div></div>\n'
        "            </td>\n"
        '            <td class="neu">—</td>\n'
        "          </tr>`;\n"
        "        }})()\n"
        "        }}\n"
        "        ${{!rows && cashVal <= 0 ? '<tr><td colspan=\"4\" style=\"color:var(--muted)\">No positions</td></tr>' : ''}}\n"
        "      </tbody>"
    )

    if old_js_tbody not in src:
        print("SKIP Patch 4: JS tbody already patched or pattern not found")
    else:
        src = src.replace(old_js_tbody, new_js_tbody)
        print("OK   Patch 4: Added CASH row to positions table JS")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    if src == original:
        print("\nNo changes made — all patches already applied or patterns not found.")
        return

    if dry_run:
        print("\n--- DRY RUN: diff preview ---")
        orig_lines = original.splitlines()
        new_lines = src.splitlines()
        import difflib

        for line in list(
            difflib.unified_diff(
                orig_lines,
                new_lines,
                fromfile="track_metrics.py (original)",
                tofile="track_metrics.py (patched)",
                lineterm="",
            )
        )[:80]:
            print(line)
        print("\nDry run complete — no files written.")
        return

    # Backup
    backup = TARGET.with_suffix(f".py.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    shutil.copy2(TARGET, backup)
    print(f"\nBackup: {backup}")

    TARGET.write_text(src, encoding="utf-8")
    print(f"Patched: {TARGET}")
    print("\nRegenerate dashboard:")
    print("  python3 track_metrics.py --report-only")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    apply_patch(dry_run=args.dry_run)
