#!/usr/bin/env python3
"""
run_paper_trading.py
---------------------
Weekly paper trading script for the Adaptive Rotation + ML strategy.
Supports multiple Alpaca accounts for live strategy comparison.

Workflow (per account):
  1. Run Adaptive Rotation for today using the account's config to get target weights
  2. Connect to the account's Alpaca paper account
  3. Execute rebalance via TradeExecutor.execute_portfolio_rebalance()
  4. Save execution log

Multi-account mode (default):
  Reads APCA_ACCOUNTS from .env and runs each account's strategy sequentially.
  Account configs are read from APCA_<NAME>_API_KEY, APCA_<NAME>_API_SECRET, etc.
  Strategy config per account is read from APCA_<NAME>_CONFIG (falls back to default).

  Example .env:
    APCA_ACCOUNTS=paper1,paper2
    APCA_PAPER1_API_KEY=...
    APCA_PAPER1_API_SECRET=...
    APCA_PAPER1_CONFIG=src/strategies/AdaptiveRotationConf_v1.2.2.yaml

    APCA_PAPER2_API_KEY=...
    APCA_PAPER2_API_SECRET=...
    APCA_PAPER2_CONFIG=src/strategies/AdaptiveRotationConf_baseline.yaml

Single-account mode (--account flag):
  Runs only the specified account.

Usage:
    python run_paper_trading.py [--dry-run] [--date YYYY-MM-DD] [--account paper1]
"""

import argparse
import json
import logging
import os
import sys
from datetime import date
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

# ---------------------------------------------------------------------------
# Default config (single-account fallback)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = "src/strategies/AdaptiveRotationConf_v1.2.2.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_ar_weights(config_path: str, run_date: str) -> dict[str, float]:
    """
    Run Adaptive Rotation strategy for run_date and return target weights dict.
    Returns {ticker: weight} e.g. {"DOW": 0.2143, "LYB": 0.2143, ...}
    """
    import subprocess
    import re

    logger.info(f"Running Adaptive Rotation for date: {run_date}")

    result = subprocess.run(
        [
            sys.executable,
            "src/strategies/run_adaptive_rotation_strategy.py",
            "--config",
            config_path,
            "--date",
            run_date,
        ],
        capture_output=True,
        text=True,
        cwd=project_root,
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


def load_accounts_from_env() -> list[dict]:
    """
    Read multi-account config from environment variables.
    Returns list of dicts with keys: name, api_key, api_secret, base_url, config
    """
    accounts_str = os.getenv("APCA_ACCOUNTS", "").strip()

    if not accounts_str:
        # Fall back to single legacy account
        return [
            {
                "name": "default",
                "api_key": os.getenv("APCA_API_KEY", ""),
                "api_secret": os.getenv("APCA_SECRET_KEY", ""),
                "base_url": os.getenv(
                    "APCA_BASE_URL", "https://paper-api.alpaca.markets"
                ),
                "config": DEFAULT_CONFIG,
            }
        ]

    accounts = []
    for name in [a.strip() for a in accounts_str.split(",") if a.strip()]:
        prefix = f"APCA_{name}"  # preserve case to match .env exactly
        api_key = os.getenv(f"{prefix}_API_KEY", "")
        api_secret = os.getenv(f"{prefix}_API_SECRET", "")
        base_url = os.getenv(
            f"{prefix}_BASE_URL", "https://paper-api.alpaca.markets"
        ).rstrip("/")
        config = os.getenv(f"{prefix}_CONFIG", DEFAULT_CONFIG)

        if not api_key or not api_secret:
            logger.warning(f"Account '{name}' missing API credentials — skipping")
            continue

        accounts.append(
            {
                "name": name,
                "api_key": api_key,
                "api_secret": api_secret,
                "base_url": base_url,
                "config": config,
            }
        )

    if not accounts:
        raise ValueError("No valid Alpaca accounts found in environment variables")

    return accounts


def get_executor_for_account(account: dict):
    """Build a TradeExecutor scoped to a single Alpaca account."""
    from src.trading.alpaca_manager import AlpacaAccount, AlpacaManager
    from src.trading.trade_executor import TradeExecutor

    alpaca_account = AlpacaAccount(
        name=account["name"],
        api_key=account["api_key"],
        api_secret=account["api_secret"],
        base_url=account["base_url"],
    )
    manager = AlpacaManager([alpaca_account])
    return TradeExecutor(manager)


def print_execution_summary(result: dict, account_name: str) -> None:
    """Print a clean summary of the execution result."""
    print("\n" + "=" * 60)
    print(f"  Execution Summary — {account_name}")
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
            side = o.get("side", "?")
            symbol = o.get("symbol", "?")
            qty = o.get("qty", o.get("notional", "?"))
            print(f"    [{status:6s}] {side:4s} {symbol:8s}  qty={qty}")
    print("=" * 60)


def dry_run_summary(weights: dict[str, float], account_name: str) -> None:
    """Print what would be traded without submitting orders."""
    print("\n" + "=" * 60)
    print(f"  [DRY RUN] Target Weights — {account_name} — No Orders Submitted")
    print("=" * 60)
    for tic, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"  {tic:8s}: {w:.2%}")
    print(f"\n  Total invested: {sum(weights.values()):.2%}")
    print("=" * 60)


def run_account(account: dict, run_date: str, dry_run: bool) -> dict:
    """
    Run the full paper trading cycle for a single account.
    Returns execution result dict for logging.
    """
    name = account["name"]
    config = account["config"]

    logger.info(f"{'=' * 50}")
    logger.info(f"Account: {name} | Config: {config}")
    logger.info(f"{'=' * 50}")

    # Step 1: Get target weights
    weights = get_ar_weights(config, run_date)

    if dry_run:
        dry_run_summary(weights, name)
        return {"account": name, "dry_run": True, "weights": weights}

    # Step 2: Connect and execute
    logger.info(f"Connecting to Alpaca account: {name}")
    executor = get_executor_for_account(account)

    # Dry-run plan first to check market status
    logger.info("Generating order plan (dry-run)...")
    plan = executor.alpaca.execute_portfolio_rebalance(
        target_weights=weights,
        account_name=name,
        dry_run=True,
    )
    market_open = plan.get("market_open", False)
    plan_sells = len(plan.get("orders_plan", {}).get("sell", []))
    plan_buys = len(plan.get("orders_plan", {}).get("buy", []))
    logger.info(
        f"Order plan: {plan_sells} sells, {plan_buys} buys | Market open: {market_open}"
    )

    use_opg = os.getenv("USE_OPG", "false").lower() == "true"

    if market_open:
        logger.info("Market is open — submitting orders now")
        rebalance_result = executor.alpaca.execute_portfolio_rebalance(
            target_weights=weights,
            account_name=name,
        )
    elif use_opg:
        logger.info("Market closed — submitting as OPG (executes at next open)")
        rebalance_result = executor.alpaca.execute_portfolio_rebalance(
            target_weights=weights,
            account_name=name,
            market_closed_action="opg",
        )
    else:
        logger.info("Market closed and USE_OPG not set — skipping submission")
        return {"account": name, "skipped": True, "weights": weights}

    print_execution_summary(rebalance_result, name)

    return {
        "account": name,
        "config": config,
        "date": run_date,
        "target_weights": weights,
        "market_open": market_open,
        "orders_placed": rebalance_result.get("orders_placed", 0),
        "orders_failed": rebalance_result.get("orders_failed", 0),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Run AR paper trading (single or multi-account)"
    )
    parser.add_argument(
        "--date", default=date.today().isoformat(), help="Trading date (default: today)"
    )
    parser.add_argument(
        "--account",
        default=None,
        help="Run only this account name (default: all accounts in APCA_ACCOUNTS)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print target weights without submitting orders",
    )
    args = parser.parse_args()

    logger.info(f"Paper trading run: {args.date}")

    # Load all accounts
    all_accounts = load_accounts_from_env()

    # Filter to single account if requested
    if args.account:
        accounts = [a for a in all_accounts if a["name"] == args.account]
        if not accounts:
            logger.error(f"Account '{args.account}' not found in APCA_ACCOUNTS")
            sys.exit(1)
    else:
        accounts = all_accounts

    logger.info(f"Running {len(accounts)} account(s): {[a['name'] for a in accounts]}")

    # Run each account sequentially
    results = []
    errors = []
    for account in accounts:
        try:
            result = run_account(account, args.date, args.dry_run)
            results.append(result)
        except Exception as e:
            logger.error(f"Account '{account['name']}' failed: {e}", exc_info=True)
            errors.append({"account": account["name"], "error": str(e)})

    # Save combined execution log
    if not args.dry_run:
        log_path = f"logs/execution_{args.date}.json"
        try:
            with open(log_path, "w") as f:
                json.dump(
                    {
                        "date": args.date,
                        "accounts": results,
                        "errors": errors,
                    },
                    f,
                    indent=2,
                    default=str,
                )
            logger.info(f"Execution log saved: {log_path}")
        except Exception as e:
            logger.warning(f"Could not save execution log: {e}")

    # Run metrics tracker
    if not args.dry_run and results:
        logger.info("Running metrics tracker...")
        import subprocess

        subprocess.run(
            [sys.executable, "track_metrics.py", "--date", args.date], cwd=project_root
        )

    if errors:
        logger.error(
            f"{len(errors)} account(s) failed: {[e['account'] for e in errors]}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
