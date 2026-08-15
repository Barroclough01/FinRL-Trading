#!/usr/bin/env python3
"""
track_rl_offline.py
-------------------
Synthetic tracking for RL strategy using offline backtest simulation.

Since we can't create a third Alpaca account, this script:
  1. Reads RL target weights from results/drl_weight.csv
  2. Simulates portfolio rebalancing offline using historical prices
  3. Injects the simulated snapshots into the same SQLite DB as live accounts
  4. Enables direct comparison in the dashboard alongside live FinRL and AR accounts

The simulation uses the same starting capital as paper accounts and applies
realistic transaction costs and market execution.

Usage:
    python track_rl_offline.py --date YYYY-MM-DD
    python track_rl_offline.py --backfill  # Simulate all historical weeks
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, "src"))

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

DB_PATH = Path("data/finrl_trading.db")
INACTIVE_SYMBOLS_PATH = Path("src/strategies/rl_inactive_symbols.json")
STARTING_CAPITAL = 1_000_000.0
TRANSACTION_COST_BPS = 5  # 0.05% per trade
SLIPPAGE_BPS = 2  # 0.02% execution slippage


# ---------------------------------------------------------------------------
# Price data fetcher
# ---------------------------------------------------------------------------


def load_price_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load historical price data from local CSV cache."""
    csv_path = Path("data/fmp_daily") / f"{symbol}_daily.csv"
    if not csv_path.exists():
        logger.warning(f"Price data not found: {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
    df = df.sort_values("date").reset_index(drop=True)
    return df


def get_price_on_date(symbol: str, target_date: str) -> float:
    """Get close price for symbol on or before target_date."""
    df = load_price_data(symbol, "2020-01-01", target_date)
    if df.empty:
        return 0.0
    return float(df.iloc[-1]["close"])


# ---------------------------------------------------------------------------
# RL weights reader
# ---------------------------------------------------------------------------


def load_rl_weights(weights_path: str, target_date: str) -> dict:
    """Load RL target weights from CSV on or before target_date.

    Supports two formats:
    1. Wide format: date,AAPL,MSFT,GOOGL,...
    2. Long format: trade_date,gvkey,weights (one row per symbol-date)
    """
    if not Path(weights_path).exists():
        logger.error(f"RL weights file not found: {weights_path}")
        return {}

    df = pd.read_csv(weights_path)

    # Detect format based on columns
    if "trade_date" in df.columns and "gvkey" in df.columns and "weights" in df.columns:
        # Long format: trade_date, gvkey, weights
        df["date"] = pd.to_datetime(df["trade_date"])
        df = df[["date", "gvkey", "weights"]].copy()
        df = df[df["date"] <= target_date].copy()

        if df.empty:
            logger.warning(f"No RL weights available on or before {target_date}")
            return {}

        # Get latest date
        latest_date = df["date"].max()
        latest_df = df[df["date"] == latest_date].copy()

        # Build weights dict
        weights = {}
        for _, row in latest_df.iterrows():
            symbol = str(row["gvkey"])
            weight = float(row["weights"])
            if weight > 0:
                weights[symbol] = weight

        return weights

    elif "date" in df.columns:
        # Wide format: date, AAPL, MSFT, ...
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= target_date].copy()

        if df.empty:
            logger.warning(f"No RL weights available on or before {target_date}")
            return {}

        latest_row = df.sort_values("date").iloc[-1]
        weights = {}
        for col in df.columns:
            if col != "date":
                w = latest_row[col]
                if pd.notna(w) and w != 0:
                    weights[col] = float(w)

        return weights

    else:
        logger.error(
            "Unrecognized CSV format. Expected columns: 'date' or "
            "'trade_date,gvkey,weights'"
        )
        return {}


def load_inactive_symbols(policy_path: str | Path) -> dict[str, dict]:
    """Load explicitly verified inactive-symbol handling metadata."""
    path = Path(policy_path)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as policy_file:
        payload = json.load(policy_file)
    symbols = payload.get("symbols", {})
    if not isinstance(symbols, dict):
        raise ValueError("Inactive-symbol policy must contain a symbols object")
    return {str(symbol).upper(): details for symbol, details in symbols.items()}


# ---------------------------------------------------------------------------
# Portfolio simulator
# ---------------------------------------------------------------------------


class OfflinePortfolio:
    """Simulate portfolio rebalancing with transaction costs."""

    def __init__(self, starting_cash: float, tx_cost_bps: float, slippage_bps: float):
        self.cash = starting_cash
        self.positions = {}  # symbol -> qty
        self.tx_cost_bps = tx_cost_bps
        self.slippage_bps = slippage_bps
        self.total_tx_cost = 0.0

    def get_portfolio_value(self, prices: dict) -> float:
        """Calculate total portfolio value."""
        holdings = sum(
            self.positions.get(sym, 0) * prices.get(sym, 0) for sym in self.positions
        )
        return self.cash + holdings

    def liquidate_position(self, symbol: str, price: float) -> float:
        """Sell an inactive holding at its last available close and return net cash."""
        quantity = self.positions.get(symbol, 0.0)
        if quantity <= 0:
            return 0.0
        if price <= 0:
            raise ValueError(f"Cannot liquidate {symbol} without a positive price")

        gross_proceeds = quantity * price
        slippage = gross_proceeds * (self.slippage_bps / 10000)
        transaction_cost = gross_proceeds * (self.tx_cost_bps / 10000)
        net_proceeds = gross_proceeds - slippage - transaction_cost
        self.cash += net_proceeds
        self.total_tx_cost += transaction_cost
        del self.positions[symbol]
        return net_proceeds

    def rebalance(self, target_weights: dict, prices: dict) -> None:
        """Rebalance portfolio to target weights."""
        portfolio_value = self.get_portfolio_value(prices)
        if portfolio_value <= 0:
            return

        # Phase 1: Sell positions not in target or overweight
        for symbol in list(self.positions.keys()):
            current_qty = self.positions[symbol]
            price = prices.get(symbol, 0)
            if price == 0 or current_qty == 0:
                continue

            target_weight = target_weights.get(symbol, 0)
            target_value = portfolio_value * target_weight
            target_qty = target_value / price

            if target_qty < current_qty:
                sell_qty = current_qty - target_qty
                sell_value = sell_qty * price
                # Apply slippage and transaction cost
                slippage = sell_value * (self.slippage_bps / 10000)
                tx_cost = sell_value * (self.tx_cost_bps / 10000)
                net_proceeds = sell_value - slippage - tx_cost

                self.cash += net_proceeds
                self.positions[symbol] = target_qty
                self.total_tx_cost += tx_cost

                if target_qty == 0:
                    del self.positions[symbol]

        # Phase 2: Buy new positions or add to underweight
        for symbol, target_weight in target_weights.items():
            if target_weight == 0:
                continue

            price = prices.get(symbol, 0)
            if price == 0:
                continue

            current_qty = self.positions.get(symbol, 0)
            target_value = portfolio_value * target_weight
            target_qty = target_value / price

            if target_qty > current_qty:
                buy_qty = target_qty - current_qty
                buy_value = buy_qty * price
                slippage = buy_value * (self.slippage_bps / 10000)
                tx_cost = buy_value * (self.tx_cost_bps / 10000)
                total_cost = buy_value + slippage + tx_cost

                # Cap to available cash
                if total_cost > self.cash:
                    buy_qty = self.cash / (
                        price * (1 + (self.slippage_bps + self.tx_cost_bps) / 10000)
                    )
                    buy_value = buy_qty * price
                    slippage = buy_value * (self.slippage_bps / 10000)
                    tx_cost = buy_value * (self.tx_cost_bps / 10000)
                    total_cost = buy_value + slippage + tx_cost

                if buy_qty > 0:
                    self.cash -= total_cost
                    self.positions[symbol] = self.positions.get(symbol, 0) + buy_qty
                    self.total_tx_cost += tx_cost

    def get_positions_list(self, prices: dict) -> list:
        """Return positions as list of dicts."""
        portfolio_value = self.get_portfolio_value(prices)
        pos_list = []
        for symbol, qty in self.positions.items():
            price = prices.get(symbol, 0)
            market_value = qty * price
            actual_weight = market_value / portfolio_value if portfolio_value > 0 else 0
            pos_list.append(
                {
                    "symbol": symbol,
                    "qty": qty,
                    "market_value": market_value,
                    "actual_weight": actual_weight,
                    "avg_cost": price,  # Simplified (use price as proxy)
                    "unrealized_pl": 0.0,  # Not tracking cost basis for simplicity
                    "unrealized_plpc": 0.0,
                }
            )
        return pos_list


def apply_inactive_symbol_policy(
    portfolio: OfflinePortfolio,
    source_target_weights: dict[str, float],
    snapshot_date: str,
    inactive_symbols: dict[str, dict],
) -> tuple[dict[str, float], list[dict]]:
    """Remove inactive targets and liquidate held positions into cash."""
    effective_target_weights = dict(source_target_weights)
    liquidation_events = []

    for symbol, details in inactive_symbols.items():
        effective_date = details.get("effective_date")
        if not effective_date or snapshot_date < str(effective_date):
            continue

        source_weight = effective_target_weights.pop(symbol, 0.0)
        held_quantity = portfolio.positions.get(symbol, 0.0)
        if held_quantity <= 0:
            continue

        price = get_price_on_date(symbol, snapshot_date)
        if price <= 0:
            raise RuntimeError(
                f"Inactive RL holding {symbol} cannot be liquidated: "
                "no positive cached price is available"
            )
        net_proceeds = portfolio.liquidate_position(symbol, price)
        event = {
            "symbol": symbol,
            "snapshot_date": snapshot_date,
            "quantity": held_quantity,
            "liquidation_price": price,
            "net_proceeds": net_proceeds,
            "source_target_weight": source_weight,
            "reason": details.get("reason"),
        }
        liquidation_events.append(event)
        logger.warning(
            "RL inactive-symbol liquidation: %s qty=%.6f price=%.4f "
            "net=$%.2f; proceeds retained as cash",
            symbol,
            held_quantity,
            price,
            net_proceeds,
        )

    return effective_target_weights, liquidation_events


# ---------------------------------------------------------------------------
# Snapshot recorder
# ---------------------------------------------------------------------------


def record_rl_snapshot(
    conn: sqlite3.Connection,
    snapshot_date: str,
    portfolio: OfflinePortfolio,
    prices: dict,
    target_weights: dict,
    benchmark: dict,
) -> None:
    """Record RL synthetic snapshot to DB."""
    portfolio_value = portfolio.get_portfolio_value(prices)
    cash = portfolio.cash
    equity = portfolio_value  # For synthetic account, equity == portfolio_value

    # Calculate weekly return
    prev = conn.execute(
        """SELECT portfolio_value FROM weekly_snapshot
           WHERE account='RL' AND snapshot_date < ?
           ORDER BY snapshot_date DESC LIMIT 1""",
        (snapshot_date,),
    ).fetchone()

    weekly_return = (
        ((portfolio_value - prev[0]) / prev[0]) if prev and prev[0] > 0 else 0.0
    )

    # Calculate cumulative return
    first = conn.execute(
        """SELECT portfolio_value FROM weekly_snapshot
           WHERE account='RL' AND snapshot_date < ?
           ORDER BY snapshot_date ASC LIMIT 1""",
        (snapshot_date,),
    ).fetchone()

    cumulative_return = (
        ((portfolio_value - first[0]) / first[0]) if first and first[0] > 0 else 0.0
    )

    spy_weekly_return = benchmark.get("spy_weekly_return")
    spy_cum = benchmark.get("spy_cumulative_return")

    positions = portfolio.get_positions_list(prices)

    conn.execute(
        """
        INSERT OR REPLACE INTO weekly_snapshot
        (snapshot_date, account, config, portfolio_value, cash, equity,
         weekly_return, cumulative_return, spy_weekly_return,
         spy_cumulative_return, positions_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """,
        (
            snapshot_date,
            "RL",
            "results/drl_weight.csv",
            portfolio_value,
            cash,
            equity,
            weekly_return,
            cumulative_return,
            spy_weekly_return,
            spy_cum,
            json.dumps(positions),
        ),
    )

    # Replace the whole RL weight set so rebuilt snapshots cannot retain stale rows.
    conn.execute(
        "DELETE FROM weekly_weights WHERE snapshot_date=? AND account='RL'",
        (snapshot_date,),
    )
    positions_by_symbol = {position["symbol"]: position for position in positions}
    all_symbols = set(target_weights) | set(positions_by_symbol)
    for symbol in sorted(all_symbols):
        position = positions_by_symbol.get(symbol, {})
        conn.execute(
            """
            INSERT OR REPLACE INTO weekly_weights
            (snapshot_date, account, symbol, target_weight, actual_weight, market_value)
            VALUES (?,?,?,?,?,?)
        """,
            (
                snapshot_date,
                "RL",
                symbol,
                target_weights.get(symbol, 0.0),
                position.get("actual_weight", 0.0),
                position.get("market_value", 0.0),
            ),
        )

    conn.commit()
    logger.info(
        f"RL snapshot: {snapshot_date} | "
        f"value=${portfolio_value:,.2f} | "
        f"weekly={weekly_return:+.2%} | cum={cumulative_return:+.2%}"
    )


# ---------------------------------------------------------------------------
# Benchmark fetcher
# ---------------------------------------------------------------------------


def fetch_spy_data(target_date: str) -> dict:
    """Fetch SPY data for target date from DB or CSV."""
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT spy_close FROM benchmark_prices WHERE price_date = ?", (target_date,)
    ).fetchone()
    conn.close()

    if row:
        return {"spy_close": row[0]}

    # Fallback to loading from CSV
    spy_price = get_price_on_date("SPY", target_date)
    return {"spy_close": spy_price} if spy_price > 0 else {}


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------


def simulate_rl_tracking(
    weights_path: str,
    snapshot_date: str,
    conn: sqlite3.Connection,
    portfolio: OfflinePortfolio | None = None,
    inactive_symbols: dict[str, dict] | None = None,
) -> OfflinePortfolio:
    """Simulate RL portfolio for a single date and record snapshot."""

    # Initialize portfolio if first run
    if portfolio is None:
        portfolio = OfflinePortfolio(
            STARTING_CAPITAL, TRANSACTION_COST_BPS, SLIPPAGE_BPS
        )

    # Load RL target weights
    source_target_weights = load_rl_weights(weights_path, snapshot_date)
    if not source_target_weights:
        logger.warning(
            f"No RL weights for {snapshot_date} — holding previous positions"
        )
        source_target_weights = {}

    if inactive_symbols is None:
        inactive_symbols = load_inactive_symbols(INACTIVE_SYMBOLS_PATH)
    target_weights, _ = apply_inactive_symbol_policy(
        portfolio,
        source_target_weights,
        snapshot_date,
        inactive_symbols,
    )

    # Get current prices for all symbols
    all_symbols = set(target_weights.keys()) | set(portfolio.positions.keys())
    prices = {sym: get_price_on_date(sym, snapshot_date) for sym in all_symbols}

    # Rebalance portfolio
    portfolio.rebalance(target_weights, prices)

    # Fetch benchmark data
    benchmark = fetch_spy_data(snapshot_date)

    # Add SPY cumulative return
    conn_db = sqlite3.connect(DB_PATH)
    first_spy = conn_db.execute(
        "SELECT spy_close FROM benchmark_prices ORDER BY price_date ASC LIMIT 1"
    ).fetchone()
    if first_spy and benchmark.get("spy_close"):
        benchmark["spy_cumulative_return"] = (
            benchmark["spy_close"] - first_spy[0]
        ) / first_spy[0]
    conn_db.close()

    # Record snapshot
    record_rl_snapshot(
        conn, snapshot_date, portfolio, prices, source_target_weights, benchmark
    )

    return portfolio


def backfill_rl_history(weights_path: str, conn: sqlite3.Connection) -> None:
    """Backfill all historical weekly snapshots for RL strategy."""
    # Keep RL observations aligned to dates shared by the live comparison.
    conn.execute(
        """DELETE FROM weekly_weights
           WHERE account='RL' AND snapshot_date NOT IN (
               SELECT DISTINCT snapshot_date FROM weekly_snapshot
               WHERE account != 'RL'
           )"""
    )
    conn.execute(
        """DELETE FROM weekly_snapshot
           WHERE account='RL' AND snapshot_date NOT IN (
               SELECT DISTINCT snapshot_date FROM weekly_snapshot
               WHERE account != 'RL'
           )"""
    )

    rows = conn.execute(
        """SELECT DISTINCT snapshot_date FROM weekly_snapshot
           WHERE account != 'RL' ORDER BY snapshot_date ASC"""
    ).fetchall()

    if not rows:
        logger.warning("No existing snapshots found — cannot backfill RL")
        return

    dates = [r[0] for r in rows]
    portfolio = None

    for snapshot_date in dates:
        logger.info(f"Backfilling RL for {snapshot_date}...")
        portfolio = simulate_rl_tracking(weights_path, snapshot_date, conn, portfolio)

    logger.info(f"RL backfill complete: {len(dates)} snapshots")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Track RL strategy offline with synthetic snapshots"
    )
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Snapshot date (default: today)",
    )
    parser.add_argument(
        "--weights-path",
        default="results/drl_weight.csv",
        help="Path to RL weights CSV",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Backfill all historical snapshots",
    )
    args = parser.parse_args()

    if not DB_PATH.exists():
        logger.error(f"Database not found: {DB_PATH}")
        logger.error("Run track_metrics.py first to initialize the database")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)

    if args.backfill:
        backfill_rl_history(args.weights_path, conn)
    else:
        # Get last recorded portfolio state
        last_snapshot = conn.execute(
            """SELECT positions_json, cash FROM weekly_snapshot
               WHERE account='RL' ORDER BY snapshot_date DESC LIMIT 1"""
        ).fetchone()

        portfolio = None
        if last_snapshot:
            positions_json, cash = last_snapshot
            portfolio = OfflinePortfolio(cash, TRANSACTION_COST_BPS, SLIPPAGE_BPS)
            positions = json.loads(positions_json)
            for pos in positions:
                portfolio.positions[pos["symbol"]] = pos["qty"]

        simulate_rl_tracking(args.weights_path, args.date, conn, portfolio)

    conn.close()
    logger.info("RL offline tracking complete")

    # Regenerate dashboard to include the new/updated RL data points
    logger.info("Regenerating dashboard to include RL data...")
    try:
        import subprocess

        cmd = [sys.executable, "track_metrics.py", "--report-only", "--date", args.date]
        subprocess.run(cmd, cwd=project_root)
        logger.info("Dashboard regeneration complete")
    except Exception as e:
        logger.warning(f"Could not regenerate dashboard: {e}")


if __name__ == "__main__":
    main()
