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


def get_ar_weights(
    config_path: str,
    run_date: str,
    is_replay: bool = False,
    account_name: str | None = None,
) -> dict[str, float]:
    """
    Run Adaptive Rotation strategy for run_date and return target weights dict.
    Returns {ticker: weight} e.g. {"DOW": 0.2143, "LYB": 0.2143, ...}
    """
    import subprocess
    import json

    logger.info(
        f"Running Adaptive Rotation for date: {run_date} (is_replay={is_replay})"
    )

    config_name = Path(config_path).stem
    suffix = "_replay" if is_replay else ""
    json_output_path = os.path.join(
        project_root, "logs", f"target_weights_{config_name}_{run_date}{suffix}.json"
    )

    # Ensure output directory exists
    Path(json_output_path).parent.mkdir(parents=True, exist_ok=True)

    # Delete existing JSON file if it exists to avoid reading stale results
    if os.path.exists(json_output_path):
        try:
            os.remove(json_output_path)
        except Exception as e:
            logger.warning(f"Could not remove stale JSON output file: {e}")

    cmd = [
        sys.executable,
        "src/strategies/run_adaptive_rotation_strategy.py",
        "--config",
        config_path,
        "--date",
        run_date,
        "--json-output",
        json_output_path,
    ]
    if account_name:
        cmd.extend(["--audit-suffix", account_name])

    result = subprocess.run(
        cmd,
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
        # Negative cash is normal on Alpaca margin/paper accounts (unsettled
        # proceeds, fully-invested portfolios). Only block when equity is zero
        # or negative, which indicates a broken or unreadable account.
        if equity <= 0:
            return (
                False,
                "account cash/equity can be read",
                f"Account {account['name']} has invalid equity (${equity:,.2f}).",
                "Check Alpaca account status and credentials.",
            )
        if cash < 0:
            logger.warning(
                f"Account {account['name']} has negative cash (${cash:,.2f}); "
                "this is normal for margin/unsettled paper accounts."
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


def compute_file_hash(path: Path) -> str:
    """Compute MD5 hash of a file."""
    import hashlib

    try:
        return hashlib.md5(path.read_bytes()).hexdigest()
    except Exception:
        return ""


def save_strategy_decision(run_date: str, account_name: str, record: dict) -> None:
    """Save normalized decision record to SQLite and JSONL mirror."""
    db_path = Path("data/finrl_trading.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. SQLite Save
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS strategy_decisions (
                id                     INTEGER PRIMARY KEY AUTOINCREMENT,
                run_date               TEXT NOT NULL,
                account_name           TEXT NOT NULL,
                config_path            TEXT,
                config_hash            TEXT,
                regime_state           TEXT,
                active_groups          TEXT, -- JSON array
                ranked_groups          TEXT, -- JSON array
                fallback_status        INTEGER DEFAULT 0,
                fallback_reason        TEXT,
                target_weights         TEXT, -- JSON object
                pre_trade_positions    TEXT, -- JSON array
                order_plan             TEXT, -- JSON object
                submitted_orders       TEXT, -- JSON array
                filled_orders          TEXT, -- JSON array
                post_trade_positions   TEXT, -- JSON array
                cash                   REAL,
                equity                 REAL,
                benchmark_snapshot     TEXT, -- JSON object
                created_at             TEXT DEFAULT (datetime('now')),
                UNIQUE(run_date, account_name)
            );
        """)
        conn.commit()

        # Insert or replace
        conn.execute(
            """
            INSERT OR REPLACE INTO strategy_decisions (
                run_date, account_name, config_path, config_hash, regime_state,
                active_groups, ranked_groups, fallback_status, fallback_reason,
                target_weights, pre_trade_positions, order_plan, submitted_orders,
                filled_orders, post_trade_positions, cash, equity, benchmark_snapshot
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                run_date,
                account_name,
                record.get("config_path"),
                record.get("config_hash"),
                record.get("regime_state"),
                json.dumps(record.get("active_groups", []), default=str),
                json.dumps(record.get("ranked_groups", []), default=str),
                1 if record.get("fallback_status") else 0,
                record.get("fallback_reason"),
                json.dumps(record.get("target_weights", {}), default=str),
                json.dumps(record.get("pre_trade_positions", []), default=str),
                json.dumps(record.get("order_plan", {}), default=str),
                json.dumps(record.get("submitted_orders", []), default=str),
                json.dumps(record.get("filled_orders", []), default=str),
                json.dumps(record.get("post_trade_positions", []), default=str),
                record.get("cash"),
                record.get("equity"),
                json.dumps(record.get("benchmark_snapshot", {}), default=str),
            ),
        )
        conn.commit()
        conn.close()
        logger.info(
            f"Saved strategy decision to SQLite for account={account_name}, date={run_date}"
        )
    except sqlite3.Error as exc:
        logger.error(
            f"Failed to save strategy decision to SQLite: {exc}", exc_info=True
        )

    # 2. JSONL Mirror Append
    jsonl_path = Path("logs/strategy_decisions.jsonl")
    try:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        record_serialized = {
            "run_date": run_date,
            "account_name": account_name,
            **record,
        }
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(record_serialized, default=str) + "\n")
        logger.info(f"Appended strategy decision to JSONL mirror: {jsonl_path}")
    except Exception as e:
        logger.error(
            f"Failed to append strategy decision to JSONL mirror: {e}", exc_info=True
        )


def reconcile_post_trade(run_date: str, account_name: str, record: dict) -> dict:
    """Run post-trade reconciliation checks for an account and return the result."""
    target_weights = record.get("target_weights", {})
    post_trade_positions = record.get("post_trade_positions", [])
    submitted_orders = record.get("submitted_orders", [])
    equity = record.get("equity", 0.0)
    cash = record.get("cash", 0.0)

    alerts = []
    comparison = []

    # Map actual positions for fast lookup
    actual_weights = {}
    for pos in post_trade_positions:
        sym = pos.get("symbol")
        mv = pos.get("market_value", 0.0)
        act_w = mv / equity if equity > 0 else 0.0
        actual_weights[sym] = act_w

    # Union of all symbols in target and actual
    all_symbols = set(target_weights.keys()) | set(actual_weights.keys())

    for sym in all_symbols:
        tgt_w = target_weights.get(sym, 0.0)
        act_w = actual_weights.get(sym, 0.0)
        drift = abs(act_w - tgt_w)

        comparison.append(
            {
                "symbol": sym,
                "target_weight": tgt_w,
                "actual_weight": act_w,
                "drift": drift,
            }
        )

        # Alert checks
        if tgt_w > 0.0 and act_w == 0.0:
            alerts.append(
                f"Target asset {sym} is missing from actual portfolio positions."
            )
        elif tgt_w == 0.0 and act_w > 0.01:  # allow tiny fractional dust
            alerts.append(
                f"Unexpected holding: {sym} has actual weight {act_w:.2%} but target weight is 0.0%."
            )
        elif drift > 0.02:
            alerts.append(
                f"Weight drift for {sym} ({drift:.2%}) exceeds tolerance threshold of 2.0%."
            )

    # Orders summary
    submitted_count = len(submitted_orders)
    failed_count = sum(
        1 for o in submitted_orders if o.get("status") in ["rejected", "failed"]
    )
    filled_count = sum(1 for o in submitted_orders if o.get("status") == "filled")
    open_count = sum(
        1
        for o in submitted_orders
        if o.get("status") not in ["filled", "rejected", "failed", "canceled"]
    )

    if failed_count > 0:
        alerts.append(
            f"{failed_count} submitted order(s) failed or were rejected by the broker."
        )

    reconciled_successfully = len(alerts) == 0

    return {
        "reconciled_successfully": reconciled_successfully,
        "discrepancies_found": not reconciled_successfully,
        "alerts": alerts,
        "target_vs_actual_weights": comparison,
        "orders_summary": {
            "submitted": submitted_count,
            "filled": filled_count,
            "open": open_count,
            "failed_or_rejected": failed_count,
        },
        "cash": cash,
        "equity": equity,
    }


def save_reconciliation_report(
    run_date: str, account_name: str, recon_result: dict
) -> None:
    """Save the post-trade reconciliation report to logs/reconciliation_YYYY-MM-DD.json."""
    log_path = Path(f"logs/reconciliation_{run_date}.json")

    # Read existing report if it exists
    data = {}
    if log_path.exists():
        try:
            with open(log_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Could not read existing reconciliation report: {e}")

    # Add/update this account's reconciliation result
    if "date" not in data:
        data["date"] = run_date
    if "accounts" not in data:
        data["accounts"] = {}

    data["accounts"][account_name] = recon_result

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Reconciliation report saved to {log_path}")
    except Exception as e:
        logger.warning(f"Could not save reconciliation report: {e}")


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


def format_webhook_body(webhook_url: str, payload: dict) -> bytes:
    """Format notification payload for Discord or generic JSON webhooks."""
    if "discord.com/api/webhooks" in webhook_url or "discordapp.com/api/webhooks" in webhook_url:
        status = str(payload.get("status", "unknown")).upper()
        run_date = payload.get("date", "")
        accounts = payload.get("accounts", [])
        lines = [
            f"**Paper Trading {status}** — {run_date}",
            f"Accounts: {', '.join(accounts) if accounts else 'n/a'}",
        ]
        if payload.get("orders_failed"):
            failed = ", ".join(
                f"{acct}={count}"
                for acct, count in payload["orders_failed"].items()
                if count
            )
            if failed:
                lines.append(f"Orders failed: {failed}")
        for err in payload.get("errors", []):
            acct = err.get("account", "?")
            msg = str(err.get("error", ""))[:500]
            lines.append(f"• **{acct}**: {msg}")
        for failure in payload.get("sanity_failures", []):
            lines.append(f"• Sanity: {str(failure)[:500]}")
        content = "\n".join(lines)
        if len(content) > 1900:
            content = content[:1900] + "…"
        return json.dumps({"content": content}).encode("utf-8")

    return json.dumps(payload).encode("utf-8")


def notify_status(webhook_url: str | None, payload: dict) -> None:
    """Send run status to a generic webhook endpoint if configured."""
    if not webhook_url:
        return

    body = format_webhook_body(webhook_url, payload)
    req = urllib.request.Request(
        webhook_url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "FinRL-Trading-PaperBot/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            logger.info(
                "Webhook notification sent (status=%s)", getattr(resp, "status", "?")
            )
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
        except Exception:
            pass
        logger.warning(
            "Webhook notification failed: HTTP %s %s", exc.code, detail or exc.reason
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


def run_metrics_tracker(run_date: str, project_root: Path) -> tuple[bool, str | None]:
    """
    Run track_metrics.py after a live paper trading session.

    Always attempts a full snapshot run first. If that fails (e.g. one or more
    account snapshots could not be recorded), falls back to --report-only so the
    dashboard and comparison metrics are still regenerated from existing DB data.

    Returns (success, error_message).
    """
    import subprocess

    logger.info("Running metrics tracker (full snapshot)...")
    full_proc = subprocess.run(
        [sys.executable, "track_metrics.py", "--date", run_date],
        cwd=project_root,
    )
    if full_proc.returncode == 0:
        logger.info("Metrics tracker completed successfully.")
        return True, None

    logger.warning(
        "Full metrics run failed (exit %s); refreshing dashboard from existing DB...",
        full_proc.returncode,
    )
    report_proc = subprocess.run(
        [sys.executable, "track_metrics.py", "--report-only", "--date", run_date],
        cwd=project_root,
    )
    if report_proc.returncode == 0:
        logger.info(
            "Dashboard and comparison metrics regenerated via --report-only fallback."
        )
        return True, (
            f"Full metrics snapshot failed (exit {full_proc.returncode}); "
            "dashboard refreshed from existing data only."
        )

    return False, (
        f"track_metrics.py failed: full run exit {full_proc.returncode}, "
        f"report-only exit {report_proc.returncode}"
    )


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
    weights = get_ar_weights(config, run_date, account_name=name)

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

    # Fetch pre-trade positions
    try:
        pre_trade_positions = executor.alpaca.get_positions(account_name=name)
    except Exception as e:
        logger.warning(f"Could not fetch pre-trade positions: {e}")
        pre_trade_positions = []

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

    skipped = False
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
        skipped = True
        rebalance_result = {"orders": [], "orders_placed": 0, "orders_failed": 0}

    if not skipped:
        print_execution_summary(rebalance_result, name)

    # Fetch post-trade positions, cash, and equity
    try:
        post_trade_positions = executor.alpaca.get_positions(account_name=name)
        post_info = executor.alpaca.get_account_info(account_name=name)
        cash = float(post_info.get("cash", 0))
        equity = float(post_info.get("equity", 0))
    except Exception as e:
        logger.warning(f"Could not fetch post-trade account details: {e}")
        post_trade_positions = []
        cash = 0.0
        equity = 0.0

    # Load strategy output JSON to get metadata
    config_name = Path(config).stem
    json_output_path = os.path.join(
        project_root, "logs", f"target_weights_{config_name}_{run_date}.json"
    )
    strategy_meta = {}
    if os.path.exists(json_output_path):
        try:
            with open(json_output_path, "r") as f:
                strategy_meta = json.load(f)
        except Exception as e:
            logger.warning(f"Could not read strategy JSON output: {e}")

    # Determine fallback reason if fallback triggered
    fallback_status = strategy_meta.get("fallback_status", False)
    fallback_reason = None
    if fallback_status:
        fallback_reason = "all_groups_negative_excess_return"  # default
        try:
            audit_path = Path(strategy_meta.get("audit_file_path", ""))
            if audit_path.exists():
                with open(audit_path, "r") as af:
                    audit_data = json.load(af)
                regime = audit_data.get("regime", {})
                if regime.get("fast_risk_off", {}).get("is_active"):
                    fallback_reason = "fast_risk_off"
                elif regime.get("effective", {}).get("state") == "risk_off":
                    fallback_reason = "risk_off_cash_floor"
                else:
                    metrics = audit_data.get("group_strength", {}).get("metrics", {})
                    if metrics:
                        if all(m.get("excess_return", 0) < 0 for m in metrics.values()):
                            fallback_reason = "all_groups_negative_excess_return"
                        else:
                            fallback_reason = "no_valid_groups"
        except Exception as e:
            logger.warning(f"Could not parse fallback reason from audit log: {e}")

    # Fetch benchmark snapshot
    benchmark_snapshot = {}
    for bench in ["SPY", "QQQ"]:
        try:
            csv_path = Path("data/fmp_daily") / f"{bench}_daily.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                if not df.empty:
                    benchmark_snapshot[f"{bench.lower()}_close"] = float(
                        df["close"].iloc[-1]
                    )
                    benchmark_snapshot[f"{bench.lower()}_date"] = str(
                        df["date"].iloc[-1]
                    )
        except Exception as e:
            logger.warning(f"Could not read benchmark snapshot for {bench}: {e}")

    # Compile the decision record
    record = {
        "config_path": config,
        "config_hash": compute_file_hash(Path(config)),
        "regime_state": strategy_meta.get("regime_state"),
        "active_groups": strategy_meta.get("active_groups", []),
        "ranked_groups": strategy_meta.get("ranked_groups", []),
        "fallback_status": fallback_status,
        "fallback_reason": fallback_reason,
        "target_weights": weights,
        "pre_trade_positions": pre_trade_positions,
        "order_plan": plan.get("orders_plan", {}),
        "submitted_orders": rebalance_result.get("orders", []),
        "filled_orders": [],
        "post_trade_positions": post_trade_positions,
        "cash": cash,
        "equity": equity,
        "benchmark_snapshot": benchmark_snapshot,
    }

    # Save to SQLite and JSONL mirror
    save_strategy_decision(run_date, name, record)

    # Run post-trade reconciliation and save report
    logger.info("Running post-trade reconciliation checks...")
    recon_result = reconcile_post_trade(run_date, name, record)
    save_reconciliation_report(run_date, name, recon_result)

    if skipped:
        return {"account": name, "skipped": True, "weights": weights}

    return {
        "account": name,
        "config": config,
        "date": run_date,
        "target_weights": weights,
        "market_open": market_open,
        "orders_placed": rebalance_result.get("orders_placed", 0),
        "orders_failed": rebalance_result.get("orders_failed", 0),
    }


def run_parity_checks(
    run_date: str, accounts: list[dict], results: list[dict], dry_run: bool
) -> None:
    """
    Perform live-vs-replay parity checks for all accounts run on run_date.
    Saves a consolidated report to logs/parity_check_YYYY-MM-DD.json and
    updates the SQLite strategy_decisions table with the results.
    """
    logger.info("Starting live-vs-replay parity checks...")
    parity_report = {"date": run_date, "dry_run": dry_run, "accounts": {}}

    db_path = Path("data/finrl_trading.db")

    for account in accounts:
        name = account["name"]
        config = account["config"]
        logger.info(f"Running parity check for account: {name}")

        # 1. Get submitted target weights
        res_entry = next((r for r in results if r.get("account") == name), None)
        submitted_weights = {}
        if res_entry:
            submitted_weights = res_entry.get("target_weights", {})

        if not submitted_weights and db_path.exists():
            try:
                conn = sqlite3.connect(db_path)
                row = conn.execute(
                    "SELECT target_weights FROM strategy_decisions WHERE run_date = ? AND account_name = ?",
                    (run_date, name),
                ).fetchone()
                if row and row[0]:
                    submitted_weights = json.loads(row[0])
                conn.close()
            except Exception as e:
                logger.warning(f"Could not load submitted weights from DB: {e}")

        # 2. Get replay target weights
        replay_weights = {}
        determinism_ok = True
        determinism_msg = "OK"
        replay_vs_submitted_mae = 0.0
        try:
            replay_weights = get_ar_weights(
                config, run_date, is_replay=True, account_name=name
            )

            all_syms = set(submitted_weights.keys()) | set(replay_weights.keys())
            diffs = []
            for sym in all_syms:
                sub_w = submitted_weights.get(sym, 0.0)
                rep_w = replay_weights.get(sym, 0.0)
                diffs.append(abs(sub_w - rep_w))
                if abs(sub_w - rep_w) > 1e-5:
                    determinism_ok = False
            replay_vs_submitted_mae = float(sum(diffs) / len(diffs)) if diffs else 0.0
        except Exception as e:
            logger.error(f"Failed to generate replay weights for {name}: {e}")
            determinism_ok = False
            determinism_msg = f"Error during replay generation: {e}"
            replay_vs_submitted_mae = 1.0

        # 3. Get filled actual weights
        filled_weights = {}
        execution_ok = True
        submitted_vs_actual_mae = 0.0
        if not dry_run:
            try:
                if db_path.exists():
                    conn = sqlite3.connect(db_path)
                    row = conn.execute(
                        "SELECT post_trade_positions, equity FROM strategy_decisions WHERE run_date = ? AND account_name = ?",
                        (run_date, name),
                    ).fetchone()
                    if row and row[0]:
                        post_trade_positions = json.loads(row[0])
                        equity = float(row[1]) if row[1] else 0.0
                        for pos in post_trade_positions:
                            sym = pos.get("symbol")
                            mv = pos.get("market_value", 0.0)
                            act_w = mv / equity if equity > 0 else 0.0
                            filled_weights[sym] = act_w
                    conn.close()

                if filled_weights:
                    all_syms = set(submitted_weights.keys()) | set(
                        filled_weights.keys()
                    )
                    diffs = []
                    for sym in all_syms:
                        sub_w = submitted_weights.get(sym, 0.0)
                        fil_w = filled_weights.get(sym, 0.0)
                        diffs.append(abs(sub_w - fil_w))
                        if abs(sub_w - fil_w) > 0.02:  # 2% drift tolerance
                            execution_ok = False
                    submitted_vs_actual_mae = (
                        float(sum(diffs) / len(diffs)) if diffs else 0.0
                    )
                else:
                    # If we expected trades but got none, check if target weights are 100% cash
                    if submitted_weights and all(
                        w == 0.0 for w in submitted_weights.values()
                    ):
                        execution_ok = True
                    else:
                        execution_ok = False
            except Exception as e:
                logger.warning(f"Could not load filled weights from DB: {e}")
                execution_ok = False
        else:
            execution_ok = True

        # 4. Get dashboard weights
        dashboard_target_weights = {}
        dashboard_actual_weights = {}
        dashboard_ok = True
        submitted_vs_dashboard_target_mae = 0.0
        actual_vs_dashboard_actual_mae = 0.0
        if not dry_run:
            try:
                if db_path.exists():
                    conn = sqlite3.connect(db_path)
                    rows = conn.execute(
                        "SELECT symbol, target_weight, actual_weight FROM weekly_weights WHERE snapshot_date = ? AND account = ?",
                        (run_date, name),
                    ).fetchall()
                    for r in rows:
                        sym, tgt_w, act_w = r
                        dashboard_target_weights[sym] = (
                            tgt_w if tgt_w is not None else 0.0
                        )
                        dashboard_actual_weights[sym] = (
                            act_w if act_w is not None else 0.0
                        )
                    conn.close()

                if dashboard_target_weights or dashboard_actual_weights:
                    all_syms = set(submitted_weights.keys()) | set(
                        dashboard_target_weights.keys()
                    )
                    diffs_tgt = []
                    for sym in all_syms:
                        sub_w = submitted_weights.get(sym, 0.0)
                        dash_tgt_w = dashboard_target_weights.get(sym, 0.0)
                        diffs_tgt.append(abs(sub_w - dash_tgt_w))
                        if abs(sub_w - dash_tgt_w) > 1e-5:
                            dashboard_ok = False
                    submitted_vs_dashboard_target_mae = (
                        float(sum(diffs_tgt) / len(diffs_tgt)) if diffs_tgt else 0.0
                    )

                    all_syms = set(filled_weights.keys()) | set(
                        dashboard_actual_weights.keys()
                    )
                    diffs_act = []
                    for sym in all_syms:
                        fil_w = filled_weights.get(sym, 0.0)
                        dash_act_w = dashboard_actual_weights.get(sym, 0.0)
                        diffs_act.append(abs(fil_w - dash_act_w))
                        if abs(fil_w - dash_act_w) > 1e-5:
                            dashboard_ok = False
                    actual_vs_dashboard_actual_mae = (
                        float(sum(diffs_act) / len(diffs_act)) if diffs_act else 0.0
                    )
                else:
                    dashboard_ok = False
            except Exception as e:
                logger.warning(f"Could not load dashboard weights from DB: {e}")
                dashboard_ok = False
        else:
            dashboard_ok = True

        # 5. Compile account parity status
        mismatches = []
        if not determinism_ok:
            mismatches.append(
                f"Determinism mismatch (submitted vs replay): {determinism_msg}"
            )
        if not execution_ok and not dry_run:
            mismatches.append(
                "Execution drift mismatch (submitted vs filled actual weights > 2.0%)"
            )
        if not dashboard_ok and not dry_run:
            mismatches.append(
                "Dashboard database mismatch (submitted/actual vs weekly_weights table)"
            )

        reconciled_successfully = len(mismatches) == 0

        acc_parity = {
            "reconciled_successfully": reconciled_successfully,
            "determinism_ok": determinism_ok,
            "execution_ok": execution_ok,
            "dashboard_ok": dashboard_ok,
            "mismatches": mismatches,
            "metrics": {
                "replay_vs_submitted_mae": replay_vs_submitted_mae,
                "submitted_vs_actual_mae": submitted_vs_actual_mae,
                "submitted_vs_dashboard_target_mae": submitted_vs_dashboard_target_mae,
                "actual_vs_dashboard_actual_mae": actual_vs_dashboard_actual_mae,
            },
            "details": {
                "replay_weights": replay_weights,
                "submitted_weights": submitted_weights,
                "filled_weights": filled_weights,
                "dashboard_target_weights": dashboard_target_weights,
                "dashboard_actual_weights": dashboard_actual_weights,
            },
        }

        parity_report["accounts"][name] = acc_parity

        # 6. Update SQLite strategy_decisions table with parity_check JSON
        try:
            if db_path.exists():
                conn = sqlite3.connect(db_path)
                try:
                    conn.execute(
                        "ALTER TABLE strategy_decisions ADD COLUMN parity_check TEXT"
                    )
                    conn.commit()
                except sqlite3.OperationalError:
                    pass

                conn.execute(
                    "UPDATE strategy_decisions SET parity_check = ? WHERE run_date = ? AND account_name = ?",
                    (json.dumps(acc_parity), run_date, name),
                )
                conn.commit()
                conn.close()
                logger.info(
                    f"Updated SQLite strategy_decisions parity_check for {name}"
                )
        except Exception as e:
            logger.error(
                f"Failed to update strategy_decisions parity_check in SQLite: {e}"
            )

    # Save consolidated JSON report
    report_path = Path(f"logs/parity_check_{run_date}.json")
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(parity_report, f, indent=2)
        logger.info(f"Saved consolidated parity check report to {report_path}")
    except Exception as e:
        logger.error(f"Failed to save parity check report: {e}")


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

    # Check Production Kill Switch
    trading_disabled_env = os.getenv("TRADING_DISABLED", "false").lower() == "true"
    kill_switch_file = Path(".kill_switch")
    kill_switch_active = trading_disabled_env or kill_switch_file.exists()

    if kill_switch_active:
        logger.warning("!!!" + "=" * 50 + "!!!")
        logger.warning("!!! PRODUCTION KILL SWITCH IS ACTIVE !!!")
        if trading_disabled_env:
            logger.warning("!!! TRADING_DISABLED=true is set in the environment.")
        if kill_switch_file.exists():
            logger.warning("!!! .kill_switch file exists in the directory.")
        logger.warning(
            "!!! FORCING DRY-RUN MODE. NO ORDERS WILL BE SUBMITTED TO ALPACA."
        )
        logger.warning("!!!" + "=" * 50 + "!!!")
        args.dry_run = True

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

    # Run metrics tracker (always on live runs, even if all accounts failed)
    metrics_warning = None
    if not args.dry_run:
        metrics_ok, metrics_error = run_metrics_tracker(args.date, project_root)
        if not metrics_ok:
            errors.append({"account": "metrics", "error": metrics_error})
        elif metrics_error:
            metrics_warning = metrics_error
            logger.warning(metrics_error)

    if not args.dry_run and results:
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

    # Run parity checks (for both dry-run and live executions)
    try:
        run_parity_checks(args.date, accounts, results, args.dry_run)
    except Exception as e:
        logger.error(f"Parity checks failed to run: {e}", exc_info=True)

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
