#!/usr/bin/env python3
"""
run_paper_trading.py
---------------------
Daily paper trading script for the Adaptive Rotation + ML strategy.

Workflow:
  1. Run Adaptive Rotation for today to get target weights
  2. Load existing Alpaca paper account state
  3. Execute rebalance via TradeExecutor.execute_portfolio_rebalance()
  4. Print execution summary

Usage:
    # Dry-run (no orders submitted)
    python run_paper_trading.py --dry-run

    # Live paper trading
    python run_paper_trading.py

    # Override date (for testing)
    python run_paper_trading.py --date 2026-04-25 --dry-run

Schedule this via cron on weekly rebalance days (e.g. every Friday at market open):
    0 14 * * 5 cd ~/stock-trading/FinRL-Trading && source finrl-env/bin/activate && python run_paper_trading.py >> logs/paper_trading.log 2>&1
"""

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------------
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, "src"))

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"logs/paper_trading_{date.today().isoformat()}.log"),
    ],
)
logger = logging.getLogger(__name__)


def get_ar_weights(config_path: str, run_date: str) -> dict[str, float]:
    """
    Run Adaptive Rotation strategy for run_date and return target weights dict.
    Returns {ticker: weight} e.g. {"DOW": 0.2143, "LYB": 0.2143, ...}
    """
    import subprocess, json, re

    logger.info(f"Running Adaptive Rotation for date: {run_date}")

    result = subprocess.run(
        [
            sys.executable,
            "src/strategies/run_adaptive_rotation_strategy.py",
            "--config", config_path,
            "--date", run_date,
        ],
        capture_output=True, text=True, cwd=project_root
    )

    if result.returncode != 0:
        logger.error(f"AR strategy failed:\n{result.stderr}")
        raise RuntimeError("Adaptive Rotation strategy run failed")

    output = result.stdout
    logger.debug(f"AR output:\n{output}")

    # Parse weights from output lines like:  "  DOW     :  21.43%"
    weights = {}
    in_portfolio = False
    for line in output.splitlines():
        if "Target Portfolio" in line:
            in_portfolio = True
            continue
        if in_portfolio:
            m = re.match(r"\s+(\S+)\s*:\s*([\d.]+)%", line)
            if m:
                ticker = m.group(1).strip('"')  # strip quotes from e.g. "ON"
                weight = float(m.group(2)) / 100.0
                weights[ticker] = weight
            elif line.strip() == "" or line.startswith("Audit"):
                break

    if not weights:
        raise ValueError("Could not parse any weights from AR strategy output")

    total = sum(weights.values())
    logger.info(f"AR target weights ({len(weights)} assets, total={total:.1%}):")
    for tic, w in sorted(weights.items(), key=lambda x: -x[1]):
        logger.info(f"  {tic:8s}: {w:.2%}")

    return weights


def get_executor():
    """Build TradeExecutor from environment credentials."""
    # trade_executor uses 'from src.trading.alpaca_manager import ...'
    # so project_root (not src/) must be on sys.path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    src_path = os.path.join(project_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from src.trading.trade_executor import create_trade_executor_from_env
    return create_trade_executor_from_env()


def print_execution_summary(result: dict) -> None:
    """Print a clean summary of the execution result dict from alpaca_manager."""
    print("\n" + "=" * 60)
    print("  Execution Summary")
    print("=" * 60)
    print(f"  Orders placed : {result.get('orders_placed', 0)}")
    print(f"  Orders failed : {result.get('orders_failed', 0)}")
    print(f"  Market open   : {result.get('market_open', '?')}")
    print(f"  TIF used      : {result.get('used_time_in_force', '?')}")

    orders = result.get("orders", [])
    if orders:
        print(f"\n  Orders ({len(orders)}):")
        for o in orders:
            status = o.get("status", "?")
            side   = o.get("side", "?")
            symbol = o.get("symbol", "?")
            qty    = o.get("qty", o.get("notional", "?"))
            print(f"    [{status:6s}] {side:4s} {symbol:8s}  qty={qty}")
    print("=" * 60)


def dry_run_summary(weights: dict[str, float]) -> None:
    """Print what would be traded without submitting orders."""
    print("\n" + "=" * 60)
    print("  [DRY RUN] Target Weights — No Orders Submitted")
    print("=" * 60)
    for tic, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"  {tic:8s}: {w:.2%}")
    print(f"\n  Total invested: {sum(weights.values()):.2%}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run AR + ML paper trading")
    parser.add_argument(
        "--config",
        default="src/strategies/AdaptiveRotationConf_v1.2.2.yaml",
        help="Path to Adaptive Rotation YAML config"
    )
    parser.add_argument(
        "--date", default=date.today().isoformat(),
        help="Trading date (default: today)"
    )
    parser.add_argument(
        "--account", default=None,
        help="Alpaca account name (default: first account in config)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print target weights without submitting orders"
    )
    args = parser.parse_args()

    logger.info(f"{'[DRY RUN] ' if args.dry_run else ''}Paper trading run: {args.date}")

    # Step 1: Get target weights from AR strategy
    weights = get_ar_weights(args.config, args.date)

    if args.dry_run:
        dry_run_summary(weights)
        return

    # Step 2: Execute rebalance via Alpaca
    logger.info("Connecting to Alpaca paper trading account...")
    executor = get_executor()

    # Dry-run plan first to check market status
    logger.info("Generating order plan (dry-run)...")
    plan = executor.alpaca.execute_portfolio_rebalance(
        target_weights=weights,
        account_name=args.account,
        dry_run=True,
    )
    market_open = plan.get("market_open", False)
    plan_sells = len(plan.get("orders_plan", {}).get("sell", []))
    plan_buys  = len(plan.get("orders_plan", {}).get("buy", []))
    logger.info(f"Order plan: {plan_sells} sells, {plan_buys} buys | Market open: {market_open}")

    use_opg = os.getenv("USE_OPG", "false").lower() == "true"

    if market_open:
        logger.info("Market is open — submitting orders now")
        rebalance_result = executor.alpaca.execute_portfolio_rebalance(
            target_weights=weights,
            account_name=args.account,
        )
    elif use_opg:
        logger.info("Market closed — submitting as OPG (executes at tomorrow's open)")
        rebalance_result = executor.alpaca.execute_portfolio_rebalance(
            target_weights=weights,
            account_name=args.account,
            market_closed_action="opg",
        )
    else:
        logger.info("Market closed and USE_OPG not set — skipping submission")
        logger.info("Set USE_OPG=true in .env to submit OPG orders for next open")
        return

    # Step 3: Print summary
    print_execution_summary(rebalance_result)

    # Step 4: Save result to log
    log_path = f"logs/execution_{args.date}.json"
    try:
        with open(log_path, "w") as f:
            json.dump({
                "date": args.date,
                "target_weights": weights,
                "market_open": market_open,
                "orders_placed": rebalance_result.get("orders_placed", 0),
                "orders_failed": rebalance_result.get("orders_failed", 0),
            }, f, indent=2, default=str)
        logger.info(f"Execution log saved: {log_path}")
    except Exception as e:
        logger.warning(f"Could not save execution log: {e}")


if __name__ == "__main__":
    main()