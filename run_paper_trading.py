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
import math
import os
import sqlite3
import sys
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
import pandas as pd

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
    import json

    logger.info(f"Running Adaptive Rotation for date: {run_date}")

    config_name = Path(config_path).stem
    json_output_path = os.path.join(
        project_root, "logs", f"target_weights_{config_name}_{run_date}.json"
    )

    # Ensure output directory exists
    Path(json_output_path).parent.mkdir(parents=True, exist_ok=True)

    # Delete existing JSON file if it exists to avoid reading stale results
    if os.path.exists(json_output_path):
        try:
            os.remove(json_output_path)
        except Exception as e:
            logger.warning(f"Could not remove stale JSON output file: {e}")

    result = subprocess.run(
        [
            sys.executable,
            "src/strategies/run_adaptive_rotation_strategy.py",
            "--config",
            config_path,
            "--date",
            run_date,
            "--json-output",
            json_output_path,
        ],
        capture_output=True,
        text=True,
        cwd=project_root,
    )

    if result.returncode != 0:
        logger.error(f"AR strategy failed:\n{result.stderr}")
        raise RuntimeError("Adaptive Rotation strategy run failed")

    # Verify JSON file exists and is readable
    if not os.path.exists(json_output_path):
        raise FileNotFoundError(
            f"Strategy JSON output file missing: {json_output_path}"
        )

    try:
        with open(json_output_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Strategy JSON output is malformed: {exc}") from exc

    # Validate keys in the JSON output
    required_keys = [
        "target_weights",
        "cash_weight",
        "regime_state",
        "active_groups",
        "ranked_groups",
        "fallback_status",
        "audit_file_path",
    ]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Strategy JSON output is missing required key: {key}")

    weights = data["target_weights"]

    # Log some structure details
    logger.info(
        f"Strategy run details: Regime={data['regime_state']}, Fallback={data['fallback_status']}, Audit={data['audit_file_path']}"
    )

    total = sum(weights.values())
    logger.info(f"AR target weights ({len(weights)} assets, total={total:.1%}):")
    for tic, w in sorted(weights.items(), key=lambda x: -x[1]):
        logger.info(f"  {tic:8s}: {w:.2%}")

    return weights


def validate_pre_trade(
    account: dict, run_date: str, target_weights: dict[str, float], executor
) -> tuple[bool, str | None, str | None, str | None]:
    """
    Run pre-trade validation checks for an account.
    Returns (is_valid, failed_rule, error_message, suggested_fix).
    """
    run_date_parsed = pd.to_datetime(run_date).date()

    # 1. target weights are nonempty
    if not target_weights:
        return (
            False,
            "target weights are nonempty",
            "Target weights dictionary is empty.",
            "Ensure the strategy run completed successfully and generated weights.",
        )

    # 2. all weights are finite numbers
    for sym, w in target_weights.items():
        if not math.isfinite(w):
            return (
                False,
                "all weights are finite numbers",
                f"Weight for {sym} is not a finite number: {w}",
                "Check the strategy code and data inputs for NaN or Inf values.",
            )

    # 3. no negative weights unless shorting is explicitly supported
    # (shorting is not supported in this strategy baseline)
    for sym, w in target_weights.items():
        if w < 0:
            return (
                False,
                "no negative weights unless shorting is explicitly supported",
                f"Negative weight found for {sym}: {w}",
                "Shorting is not supported. Ensure all strategy target weights are non-negative.",
            )

    # 4. sum of target weights is within tolerance
    total_weight = sum(target_weights.values())
    if total_weight > 1.0001:
        return (
            False,
            "sum of target weights is within tolerance",
            f"Sum of target weights ({total_weight:.4%}) exceeds 100% plus tolerance.",
            "Check the strategy portfolio construction and normalization logic.",
        )

    # 5. no single symbol exceeds configured maximum weight
    # We use a default limit of 50% for any single position
    max_single_weight = 0.5
    for sym, w in target_weights.items():
        if w > max_single_weight:
            return (
                False,
                "no single symbol exceeds configured maximum weight",
                f"Weight for {sym} ({w:.2%}) exceeds maximum allowed weight ({max_single_weight:.2%}).",
                "Verify strategy allocation limits and group constraints.",
            )

    # 6. all required symbol prices are fresh
    for sym, w in target_weights.items():
        if w > 0:
            csv_path = Path("data/fmp_daily") / f"{sym}_daily.csv"
            if not csv_path.exists():
                return (
                    False,
                    "all required symbol prices are fresh",
                    f"Price CSV missing for required symbol {sym}: {csv_path}",
                    f"Run data backfill or refresh for {sym}.",
                )
            try:
                df = pd.read_csv(csv_path)
                if df.empty or "date" not in df.columns:
                    return (
                        False,
                        "all required symbol prices are fresh",
                        f"Price CSV for {sym} is empty or missing 'date' column.",
                        f"Re-refresh price data for {sym}.",
                    )
                last_date_str = df["date"].iloc[-1]
                last_date = pd.to_datetime(last_date_str).date()
                if (run_date_parsed - last_date).days > 10:
                    return (
                        False,
                        "all required symbol prices are fresh",
                        f"Price data for {sym} is stale. Last date: {last_date_str}, Run date: {run_date}",
                        f"Run data refresh for {sym}.",
                    )
            except Exception as e:
                return (
                    False,
                    "all required symbol prices are fresh",
                    f"Failed to read price data for {sym}: {e}",
                    f"Check file integrity of {csv_path}.",
                )

    # 7. SPY and QQQ benchmark data are present
    for bench in ["SPY", "QQQ"]:
        csv_path = Path("data/fmp_daily") / f"{bench}_daily.csv"
        if not csv_path.exists():
            return (
                False,
                "SPY and QQQ benchmark data are present",
                f"Benchmark CSV missing for {bench}: {csv_path}",
                f"Run data refresh for {bench}.",
            )
        try:
            df = pd.read_csv(csv_path)
            if df.empty or "date" not in df.columns:
                return (
                    False,
                    "SPY and QQQ benchmark data are present",
                    f"Benchmark CSV for {bench} is empty or invalid.",
                    f"Re-refresh {bench} data.",
                )
            last_date_str = df["date"].iloc[-1]
            last_date = pd.to_datetime(last_date_str).date()
            if (run_date_parsed - last_date).days > 10:
                return (
                    False,
                    "SPY and QQQ benchmark data are present",
                    f"Benchmark data for {bench} is stale. Last date: {last_date_str}",
                    f"Run data refresh for {bench}.",
                )
        except Exception as e:
            return (
                False,
                "SPY and QQQ benchmark data are present",
                f"Failed to read benchmark data for {bench}: {e}",
                f"Check file integrity of {csv_path}.",
            )

    # 8. market calendar status is known
    from src.data.trading_calendar import is_trading_day

    try:
        _ = is_trading_day(run_date)
    except Exception as e:
        return (
            False,
            "market calendar status is known",
            f"Market calendar status is unknown or query failed: {e}",
            "Ensure calendar libraries are installed and configured properly.",
        )

    # 9. account cash/equity can be read
    try:
        info = executor.alpaca.get_account_info(account_name=account["name"])
        cash = float(info.get("cash", 0))
        equity = float(info.get("equity", 0))
        if cash < 0 or equity <= 0:
            return (
                False,
                "account cash/equity can be read",
                f"Account {account['name']} has invalid cash (${cash:,.2f}) or equity (${equity:,.2f}).",
                "Check Alpaca account status and credentials.",
            )
    except Exception as e:
        return (
            False,
            "account cash/equity can be read",
            f"Failed to read Alpaca account info for {account['name']}: {e}",
            "Check your Alpaca API credentials, network connection, or Alpaca API status.",
        )

    return True, None, None, None


def save_validation_result(
    run_date: str,
    account_name: str,
    valid: bool,
    failed_rule: str | None,
    error_msg: str | None,
    suggested_fix: str | None,
) -> None:
    """Save validation status to logs/pre_trade_validation_YYYY-MM-DD.json"""
    from datetime import datetime

    log_path = Path(f"logs/pre_trade_validation_{run_date}.json")

    # Read existing log if it exists
    data = {}
    if log_path.exists():
        try:
            with open(log_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Could not read existing pre-trade validation log: {e}")

    # Add/update this account's validation result
    data[account_name] = {
        "status": "passed" if valid else "failed",
        "failed_rule": failed_rule,
        "error_message": error_msg,
        "suggested_fix": suggested_fix,
        "timestamp": datetime.now().isoformat(),
    }

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Pre-trade validation log saved to {log_path}")
    except Exception as e:
        logger.warning(f"Could not save pre-trade validation log: {e}")


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


def notify_status(webhook_url: str | None, payload: dict) -> None:
    """Send run status to a generic webhook endpoint if configured."""
    if not webhook_url:
        return

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            logger.info(
                "Webhook notification sent (status=%s)", getattr(resp, "status", "?")
            )
    except (urllib.error.URLError, TimeoutError) as exc:
        logger.warning("Webhook notification failed: %s", exc)


def run_post_run_sanity_checks(
    run_date: str, accounts: list[dict], results: list[dict], errors: list[dict]
) -> list[str]:
    """
    Validate that key artifacts exist and contain expected rows for each account.
    Returns a list of failure messages; empty means all checks passed.
    """
    failures: list[str] = []
    account_names = {a["name"] for a in accounts}
    result_names = {r.get("account") for r in results}

    if errors:
        failures.append(
            f"execution reported account errors: {[e['account'] for e in errors]}"
        )

    missing_accounts = sorted(account_names - result_names)
    if missing_accounts:
        failures.append(f"missing execution results for account(s): {missing_accounts}")

    execution_log = Path(f"logs/execution_{run_date}.json")
    if not execution_log.exists():
        failures.append(f"missing execution log: {execution_log}")

    dashboard = Path("logs/dashboard.html")
    if not dashboard.exists():
        failures.append("missing dashboard output: logs/dashboard.html")

    db_path = Path("data/finrl_trading.db")
    if not db_path.exists():
        failures.append("missing SQLite DB: data/finrl_trading.db")
        return failures

    try:
        conn = sqlite3.connect(db_path)
        for account in sorted(account_names):
            snap = conn.execute(
                """
                SELECT COUNT(*) FROM weekly_snapshot
                WHERE snapshot_date = ? AND account = ?
                """,
                (run_date, account),
            ).fetchone()
            if not snap or snap[0] < 1:
                failures.append(
                    f"weekly_snapshot missing for account={account}, date={run_date}"
                )

            weights = conn.execute(
                """
                SELECT COUNT(*) FROM weekly_weights
                WHERE snapshot_date = ? AND account = ?
                """,
                (run_date, account),
            ).fetchone()
            if not weights or weights[0] < 1:
                failures.append(
                    f"weekly_weights missing for account={account}, date={run_date}"
                )
        conn.close()
    except sqlite3.Error as exc:
        failures.append(f"DB sanity check failed: {exc}")

    return failures


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

    # Step 2: Connect and execute
    logger.info(f"Connecting to Alpaca account: {name}")
    executor = get_executor_for_account(account)

    # Run pre-trade validation gate
    logger.info("Running pre-trade validation gate...")
    valid, failed_rule, error_msg, suggested_fix = validate_pre_trade(
        account, run_date, weights, executor
    )
    save_validation_result(run_date, name, valid, failed_rule, error_msg, suggested_fix)

    if not valid:
        logger.error(f"Pre-trade validation FAILED for {name}!")
        logger.error(f"  Failed Rule: {failed_rule}")
        logger.error(f"  Error Message: {error_msg}")
        logger.error(f"  Suggested Fix: {suggested_fix}")
        raise ValueError(f"Pre-trade validation failed: {error_msg}")

    logger.info("Pre-trade validation passed successfully.")

    if dry_run:
        dry_run_summary(weights, name)
        return {"account": name, "dry_run": True, "weights": weights}

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
    parser.add_argument(
        "--notify-webhook",
        default=os.getenv("PAPER_TRADING_WEBHOOK_URL", "").strip(),
        help="Webhook URL for success/failure notifications "
        "(default: PAPER_TRADING_WEBHOOK_URL env)",
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

        metrics_proc = subprocess.run(
            [sys.executable, "track_metrics.py", "--date", args.date], cwd=project_root
        )
        if metrics_proc.returncode != 0:
            errors.append(
                {
                    "account": "metrics",
                    "error": f"track_metrics.py failed with return code {metrics_proc.returncode}",
                }
            )

        failures = run_post_run_sanity_checks(args.date, accounts, results, errors)
        if failures:
            for failure in failures:
                logger.error("Sanity check failed: %s", failure)
            notify_status(
                args.notify_webhook,
                {
                    "status": "failed",
                    "date": args.date,
                    "accounts": sorted([a["name"] for a in accounts]),
                    "errors": errors,
                    "sanity_failures": failures,
                },
            )
            sys.exit(1)

        logger.info("Post-run sanity checks passed")
        notify_status(
            args.notify_webhook,
            {
                "status": "ok",
                "date": args.date,
                "accounts": sorted([a["name"] for a in accounts]),
                "orders_failed": {
                    r["account"]: r.get("orders_failed", 0) for r in results
                },
            },
        )

    if errors:
        logger.error(
            f"{len(errors)} account(s) failed: {[e['account'] for e in errors]}"
        )
        notify_status(
            args.notify_webhook,
            {
                "status": "failed",
                "date": args.date,
                "accounts": sorted([a["name"] for a in accounts]),
                "errors": errors,
            },
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
