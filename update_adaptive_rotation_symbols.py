#!/usr/bin/env python3
"""
update_adaptive_rotation_symbols.py
------------------------------------
Quarterly integration script: reads the latest ML bucket predictions CSV and
patches the asset_groups symbol lists in the Adaptive Rotation YAML config.

Usage:
    python update_adaptive_rotation_symbols.py \
        --predictions data/sp500_ml_bucket_predictions_<timestamp>.csv \
        --config src/strategies/AdaptiveRotationConf_v1.2.1.yaml \
        [--top-n 5] \
        [--dry-run]

The script:
  1. Reads the ML predictions CSV (output of ml_bucket_selection.py)
  2. Selects top-N stocks per bucket ranked by predicted_return
  3. Patches the four asset_groups symbol lists in the YAML
  4. Writes a timestamped backup of the original YAML before overwriting
  5. Prints a diff summary of what changed

Adaptive Rotation still controls all regime logic, weighting, and stop-loss.
This script only updates which symbols are in each group's candidate pool.
"""

import argparse
import os
import re
import shutil
import sys
from datetime import datetime
from glob import glob

import pandas as pd

# ---------------------------------------------------------------------------
# Bucket -> YAML group key mapping
# ---------------------------------------------------------------------------
BUCKET_TO_GROUP = {
    "growth_tech": "group_a_growth_tech",
    "cyclical": "group_b_cyclical",
    "real_assets": "group_c_real_assets",
    "defensive": "group_d_defensive",
}

# Fallback symbols used when a bucket has no ML picks (preserves strategy safety)
FALLBACK_SYMBOLS = {
    "growth_tech": ["AAPL", "MSFT", "NVDA", "META", "AMZN"],
    "cyclical": ["JPM", "GS", "HD", "CAT", "UNP"],
    "real_assets": ["XOM", "CVX", "COP", "FCX", "GLD"],
    "defensive": ["JNJ", "PG", "KO", "UNH", "PEP"],
}


def find_latest_predictions(data_dir: str) -> str:
    """Return path to the most recent ml_bucket_predictions CSV in data_dir."""
    pattern = os.path.join(data_dir, "sp500_ml_bucket_predictions_*.csv")
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"No ml_bucket_predictions CSV found in {data_dir}")
    return files[-1]


def load_predictions(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"tic", "bucket", "predicted_return"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Predictions CSV missing columns: {missing}")
    return df


def get_top_picks(df: pd.DataFrame, top_n: int) -> dict[str, list[str]]:
    """Return {bucket: [ticker, ...]} with top-N per bucket by predicted_return."""
    picks = {}
    for bucket in BUCKET_TO_GROUP:
        bdf = df[df["bucket"] == bucket].sort_values(
            "predicted_return", ascending=False
        )
        tickers = bdf["tic"].head(top_n).tolist()
        if not tickers:
            print(f"  WARNING: no picks for {bucket}, using fallback symbols")
            tickers = FALLBACK_SYMBOLS[bucket]
        picks[bucket] = tickers
    return picks


def patch_yaml(yaml_path: str, picks: dict[str, list[str]], dry_run: bool) -> str:
    """
    Patch asset_groups symbol lists in the YAML line by line.
    Finds each group block, locates its symbols: section, replaces the ticker lines.
    Robust to any indentation width.
    """
    with open(yaml_path, "r") as f:
        lines = f.readlines()

    original = "".join(lines)

    for bucket, group_key in BUCKET_TO_GROUP.items():
        tickers = picks[bucket]

        # Find the line index of this group key
        group_line = None
        for i, line in enumerate(lines):
            if line.strip().startswith(group_key + ":"):
                group_line = i
                break

        if group_line is None:
            print(f"  WARNING: could not find group key '{group_key}' — skipping")
            continue

        # Find `symbols:` line after group_line
        symbols_line = None
        for i in range(group_line + 1, min(group_line + 10, len(lines))):
            if lines[i].strip() == "symbols:":
                symbols_line = i
                break

        if symbols_line is None:
            print(f"  WARNING: could not find symbols: under '{group_key}' — skipping")
            continue

        # Detect indentation from the first existing ticker line
        indent = "      "  # default 6 spaces
        if symbols_line + 1 < len(lines):
            next_line = lines[symbols_line + 1]
            m = re.match(r"^(\s+)-\s+\S+", next_line)
            if m:
                indent = m.group(1)

        # Find the range of ticker lines (lines starting with indent + "- ")
        start = symbols_line + 1
        end = start
        while end < len(lines) and re.match(
            rf"^{re.escape(indent)}-\s+\S+", lines[end]
        ):
            end += 1

        # Build replacement lines — quote tickers that YAML would misparse as booleans
        YAML_BOOL_TICKERS = {"ON", "NO", "YES", "TRUE", "FALSE", "NULL", "OFF"}
        new_ticker_lines = [
            f'{indent}- "{t}"\n'
            if t.upper() in YAML_BOOL_TICKERS
            else f"{indent}- {t}\n"
            for t in tickers
        ]

        # Splice in
        lines = lines[:start] + new_ticker_lines + lines[end:]

    patched = "".join(lines)
    return patched, original


def write_output(yaml_path: str, patched_text: str, dry_run: bool) -> str:
    """Backup original and write patched YAML. Returns backup path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = yaml_path.replace(".yaml", f"_backup_{ts}.yaml")

    if dry_run:
        print(f"\n[dry-run] Would write backup to: {backup_path}")
        print(f"[dry-run] Would overwrite: {yaml_path}")
        return backup_path

    shutil.copy2(yaml_path, backup_path)
    with open(yaml_path, "w") as f:
        f.write(patched_text)
    return backup_path


def print_diff_summary(picks: dict[str, list[str]], original_text: str) -> None:
    """Print a human-readable summary of old vs new symbols per group."""
    print("\n" + "=" * 60)
    print("  Symbol Update Summary")
    print("=" * 60)

    orig_lines = original_text.splitlines()
    for bucket, group_key in BUCKET_TO_GROUP.items():
        # Extract old symbols line by line
        old_tickers = []
        group_line = next(
            (
                i
                for i, l in enumerate(orig_lines)
                if l.strip().startswith(group_key + ":")
            ),
            None,
        )
        if group_line is not None:
            sym_line = next(
                (
                    i
                    for i in range(
                        group_line + 1, min(group_line + 10, len(orig_lines))
                    )
                    if orig_lines[i].strip() == "symbols:"
                ),
                None,
            )
            if sym_line is not None:
                for l in orig_lines[sym_line + 1 :]:
                    m = re.match(r"^\s+-\s+(\S+)", l)
                    if m:
                        old_tickers.append(m.group(1))
                    else:
                        break

        new_tickers = picks[bucket]
        added = [t for t in new_tickers if t not in old_tickers]
        removed = [t for t in old_tickers if t not in new_tickers]
        kept = [t for t in new_tickers if t in old_tickers]

        print(f"\n  {bucket} ({group_key})")
        print(f"    New pool : {', '.join(new_tickers)}")
        if added:
            print(f"    Added    : {', '.join(added)}")
        if removed:
            print(f"    Removed  : {', '.join(removed)}")
        if kept:
            print(f"    Kept     : {', '.join(kept)}")


def main():
    parser = argparse.ArgumentParser(
        description="Patch Adaptive Rotation YAML with latest ML bucket picks"
    )
    parser.add_argument(
        "--predictions",
        default=None,
        help="Path to ml_bucket_predictions CSV. Defaults to latest in --data-dir.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory to search for latest predictions CSV (default: data/)",
    )
    parser.add_argument(
        "--config",
        default="src/strategies/AdaptiveRotationConf_v1.2.2.yaml",
        help="Path to Adaptive Rotation YAML config",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of ML picks to include per bucket (default: 5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without writing any files",
    )
    args = parser.parse_args()

    # Resolve predictions CSV
    if args.predictions:
        csv_path = args.predictions
    else:
        csv_path = find_latest_predictions(args.data_dir)
    print(f"Predictions : {csv_path}")
    print(f"Config      : {args.config}")
    print(f"Top-N       : {args.top_n} per bucket")
    print(f"Dry-run     : {args.dry_run}")

    if not os.path.exists(csv_path):
        print(f"ERROR: predictions file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.config):
        print(f"ERROR: config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    # Load and process
    df = load_predictions(csv_path)
    print(f"\nLoaded {len(df)} predictions across {df['bucket'].nunique()} buckets")

    picks = get_top_picks(df, args.top_n)

    # Print picks before patching
    print("\nTop picks per bucket:")
    for bucket, tickers in picks.items():
        print(f"  {bucket:15s}: {', '.join(tickers)}")

    # Patch YAML
    patched_text, original_text = patch_yaml(args.config, picks, dry_run=args.dry_run)

    # Diff summary
    print_diff_summary(picks, original_text)

    # Write
    backup_path = write_output(args.config, patched_text, dry_run=args.dry_run)

    if not args.dry_run:
        print(f"\nBackup saved : {backup_path}")
        print(f"Config updated: {args.config}")
        print(
            "\nReady to run: ./deploy.sh --strategy adaptive_rotation --mode backtest ..."
        )
    else:
        print("\n[dry-run] No files written.")


if __name__ == "__main__":
    main()
