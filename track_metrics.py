#!/usr/bin/env python3
"""
track_metrics.py
-----------------
Records weekly performance snapshots for all paper trading accounts
and generates a CLI summary + HTML dashboard.

Workflow:
  1. For each account, fetch portfolio value + positions from Alpaca
  2. Fetch SPY weekly close as benchmark
  3. Write snapshot to finrl_trading.db (weekly_snapshot, weekly_weights tables)
  4. Print CLI summary
  5. Generate HTML dashboard at logs/dashboard.html

Usage:
    python track_metrics.py               # Record snapshot + generate dashboard
    python track_metrics.py --report-only # Generate dashboard from existing data only
    python track_metrics.py --date YYYY-MM-DD  # Override date (for testing)
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, "src"))

from dotenv import load_dotenv

load_dotenv()

import yfinance as yf

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

DB_PATH = Path("data/finrl_trading.db")
DASHBOARD_PATH = Path("logs/dashboard.html")
STARTING_CAPITAL = 1_000_000.0  # Alpaca paper account starting value


# ---------------------------------------------------------------------------
# DB setup
# ---------------------------------------------------------------------------


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS weekly_snapshot (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_date TEXT NOT NULL,
            account       TEXT NOT NULL,
            config        TEXT,
            portfolio_value REAL,
            cash          REAL,
            equity        REAL,
            weekly_return REAL,
            cumulative_return REAL,
            spy_weekly_return REAL,
            spy_cumulative_return REAL,
            positions_json TEXT,
            created_at    TEXT DEFAULT (datetime('now')),
            UNIQUE(snapshot_date, account)
        );

        CREATE TABLE IF NOT EXISTS weekly_weights (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_date TEXT NOT NULL,
            account       TEXT NOT NULL,
            symbol        TEXT NOT NULL,
            target_weight REAL,
            actual_weight REAL,
            market_value  REAL,
            UNIQUE(snapshot_date, account, symbol)
        );

        CREATE TABLE IF NOT EXISTS benchmark_prices (
            price_date TEXT PRIMARY KEY,
            spy_close  REAL,
            qqq_close  REAL
        );
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------


def load_accounts_from_env() -> list[dict]:
    accounts_str = os.getenv("APCA_ACCOUNTS", "").strip()
    if not accounts_str:
        return []
    accounts = []
    for name in [a.strip() for a in accounts_str.split(",") if a.strip()]:
        prefix = f"APCA_{name}"
        api_key = os.getenv(f"{prefix}_API_KEY", "")
        api_secret = os.getenv(f"{prefix}_API_SECRET", "")
        base_url = os.getenv(
            f"{prefix}_BASE_URL", "https://paper-api.alpaca.markets"
        ).rstrip("/")
        config = os.getenv(f"{prefix}_CONFIG", "")
        if api_key and api_secret:
            accounts.append(
                {
                    "name": name,
                    "api_key": api_key,
                    "api_secret": api_secret,
                    "base_url": base_url,
                    "config": config,
                }
            )
    return accounts


def get_alpaca_snapshot(account: dict) -> dict:
    """Fetch portfolio value and positions from Alpaca."""
    from src.trading.alpaca_manager import AlpacaAccount, AlpacaManager

    acc = AlpacaAccount(
        name=account["name"],
        api_key=account["api_key"],
        api_secret=account["api_secret"],
        base_url=account["base_url"],
    )
    mgr = AlpacaManager([acc])

    info = mgr.get_account_info(account_name=account["name"])
    positions = mgr.get_positions(account_name=account["name"])

    portfolio_value = float(info.get("portfolio_value", 0))
    cash = float(info.get("cash", 0))
    equity = float(info.get("equity", 0))

    pos_list = []
    for p in positions:
        market_value = float(p.get("market_value", 0))
        actual_weight = market_value / portfolio_value if portfolio_value else 0
        pos_list.append(
            {
                "symbol": p.get("symbol"),
                "qty": float(p.get("qty", 0)),
                "market_value": market_value,
                "actual_weight": actual_weight,
                "avg_cost": float(p.get("avg_entry_price", 0)),
                "unrealized_pl": float(p.get("unrealized_pl", 0)),
                "unrealized_plpc": float(p.get("unrealized_plpc", 0)),
            }
        )

    return {
        "portfolio_value": portfolio_value,
        "cash": cash,
        "equity": equity,
        "positions": pos_list,
    }


def fetch_benchmark_prices(as_of: date) -> dict:
    """Fetch SPY and QQQ close prices for the given date (and prior week)."""
    start = (as_of - timedelta(days=10)).strftime("%Y-%m-%d")
    end = (as_of + timedelta(days=1)).strftime("%Y-%m-%d")

    df = yf.download(
        ["SPY", "QQQ"], start=start, end=end, auto_adjust=True, progress=False
    )

    if df.empty:
        return {}

    close = df["Close"] if "Close" in df else df
    close.index = close.index.date

    # Get the most recent available date up to as_of
    available = [d for d in close.index if d <= as_of]
    if not available:
        return {}

    latest = max(available)
    prev = max([d for d in available if d < latest], default=None)

    result = {
        "date": latest,
        "spy_close": float(close.loc[latest, "SPY"]),
        "qqq_close": float(close.loc[latest, "QQQ"]),
    }
    if prev:
        result["spy_prev"] = float(close.loc[prev, "SPY"])
        result["qqq_prev"] = float(close.loc[prev, "QQQ"])
        result["spy_weekly_return"] = (
            result["spy_close"] - result["spy_prev"]
        ) / result["spy_prev"]

    return result


# ---------------------------------------------------------------------------
# Snapshot recording
# ---------------------------------------------------------------------------


def record_snapshot(
    conn: sqlite3.Connection,
    snapshot_date: str,
    account: dict,
    alpaca: dict,
    benchmark: dict,
    target_weights: dict,
) -> None:
    """Write weekly snapshot and weights to DB."""

    portfolio_value = alpaca["portfolio_value"]

    # Weekly return: 0.0 on first snapshot, otherwise vs prior week
    prev = conn.execute(
        "SELECT portfolio_value FROM weekly_snapshot WHERE account=? ORDER BY snapshot_date DESC LIMIT 1",
        (account["name"],),
    ).fetchone()

    if prev:
        weekly_return = (portfolio_value - prev[0]) / prev[0] if prev[0] else 0.0
    else:
        weekly_return = 0.0  # first snapshot — no prior week

    # Cumulative return: 0.0 on first snapshot, otherwise vs first recorded value
    first = conn.execute(
        "SELECT portfolio_value FROM weekly_snapshot WHERE account=? ORDER BY snapshot_date ASC LIMIT 1",
        (account["name"],),
    ).fetchone()

    if first:
        cumulative_return = (portfolio_value - first[0]) / first[0] if first[0] else 0.0
    else:
        cumulative_return = 0.0  # first snapshot

    spy_weekly_return = benchmark.get("spy_weekly_return", None)

    # SPY cumulative: first benchmark entry
    first_spy = conn.execute(
        "SELECT spy_close FROM benchmark_prices ORDER BY price_date ASC LIMIT 1"
    ).fetchone()
    spy_cum = None
    if first_spy and benchmark.get("spy_close"):
        spy_cum = (benchmark["spy_close"] - first_spy[0]) / first_spy[0]

    conn.execute(
        """
        INSERT OR REPLACE INTO weekly_snapshot
        (snapshot_date, account, config, portfolio_value, cash, equity,
         weekly_return, cumulative_return, spy_weekly_return, spy_cumulative_return, positions_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """,
        (
            snapshot_date,
            account["name"],
            account.get("config"),
            portfolio_value,
            alpaca["cash"],
            alpaca["equity"],
            weekly_return,
            cumulative_return,
            spy_weekly_return,
            spy_cum,
            json.dumps(alpaca["positions"]),
        ),
    )

    # Weekly weights
    for pos in alpaca["positions"]:
        sym = pos["symbol"]
        conn.execute(
            """
            INSERT OR REPLACE INTO weekly_weights
            (snapshot_date, account, symbol, target_weight, actual_weight, market_value)
            VALUES (?,?,?,?,?,?)
        """,
            (
                snapshot_date,
                account["name"],
                sym,
                target_weights.get(sym),
                pos["actual_weight"],
                pos["market_value"],
            ),
        )

    # Benchmark prices
    if benchmark.get("spy_close"):
        conn.execute(
            """
            INSERT OR REPLACE INTO benchmark_prices (price_date, spy_close, qqq_close)
            VALUES (?,?,?)
        """,
            (
                str(benchmark["date"]),
                benchmark["spy_close"],
                benchmark.get("qqq_close"),
            ),
        )

    conn.commit()
    logger.info(
        f"  Snapshot saved: {account['name']} | "
        f"value=${portfolio_value:,.2f} | "
        f"weekly={weekly_return:+.2%} | cum={cumulative_return:+.2%}"
    )


# ---------------------------------------------------------------------------
# CLI report
# ---------------------------------------------------------------------------


def print_cli_report(conn: sqlite3.Connection) -> None:
    rows = conn.execute("""
        SELECT snapshot_date, account, portfolio_value, weekly_return,
               cumulative_return, spy_weekly_return, spy_cumulative_return
        FROM weekly_snapshot
        ORDER BY snapshot_date DESC, account
    """).fetchall()

    if not rows:
        print("No snapshots recorded yet.")
        return

    print("\n" + "=" * 72)
    print(f"  PAPER TRADING PERFORMANCE REPORT  —  {date.today()}")
    print("=" * 72)

    # Latest snapshot per account
    seen = {}
    for r in rows:
        if r[1] not in seen:
            seen[r[1]] = r

    print(
        f"\n{'Account':<12} {'Value':>14} {'Weekly':>10} {'Cumulative':>12} {'SPY Wkly':>10} {'vs SPY':>10}"
    )
    print("-" * 72)
    for acct, r in seen.items():
        snap_date, account, value, wkly, cum, spy_wkly, spy_cum = r
        vs_spy = (cum - spy_cum) if (cum is not None and spy_cum is not None) else None
        print(
            f"  {account:<10} ${value:>13,.2f} {wkly:>+9.2%} {cum:>+11.2%} "
            f"{spy_wkly if spy_wkly else 0:>+9.2%} "
            f"{vs_spy if vs_spy else 0:>+9.2%}"
        )

    print("\n  Weekly History:")
    print(
        f"  {'Date':<12} {'Account':<12} {'Value':>14} {'Weekly':>10} {'Cumulative':>12}"
    )
    print("  " + "-" * 62)
    for r in rows[:20]:  # last 20 entries
        snap_date, account, value, wkly, cum, spy_wkly, spy_cum = r
        print(
            f"  {snap_date:<12} {account:<12} ${value:>13,.2f} {wkly:>+9.2%} {cum:>+11.2%}"
        )

    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# HTML dashboard
# ---------------------------------------------------------------------------


def generate_html_dashboard(conn: sqlite3.Connection, output_path: Path) -> None:
    rows = conn.execute("""
        SELECT snapshot_date, account, portfolio_value, weekly_return,
               cumulative_return, spy_weekly_return, spy_cumulative_return, positions_json
        FROM weekly_snapshot
        ORDER BY snapshot_date ASC, account
    """).fetchall()

    # Build per-account time series
    accounts_data = {}
    spy_by_date = {}
    for r in rows:
        snap_date, account, value, wkly, cum, spy_wkly, spy_cum, pos_json = r
        if account not in accounts_data:
            accounts_data[account] = []
        accounts_data[account].append(
            {
                "date": snap_date,
                "value": value,
                "weekly": wkly,
                "cumulative": cum,
                "spy_weekly": spy_wkly,
                "spy_cumulative": spy_cum,
                "positions": json.loads(pos_json) if pos_json else [],
            }
        )
        if snap_date not in spy_by_date:
            spy_by_date[snap_date] = {
                "date": snap_date,
                "weekly": spy_wkly,
                "cumulative": spy_cum,
            }

    # Latest snapshot per account for summary cards
    latest = {acct: data[-1] for acct, data in accounts_data.items()}

    spy_series = [spy_by_date[d] for d in sorted(spy_by_date)]

    dates_js = json.dumps(sorted(set(r[0] for r in rows)))
    accounts_js = json.dumps(accounts_data)
    spy_series_js = json.dumps(spy_series)
    latest_js = json.dumps(latest)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Politic-Trader — Performance Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg:       #0a0c0f;
    --surface:  #111318;
    --border:   #1e2128;
    --accent1:  #00e5ff;
    --accent2:  #ff6b35;
    --accent3:  #7fff6b;
    --text:     #e8eaf0;
    --muted:    #5a6070;
    --positive: #4cffaa;
    --negative: #ff4d6d;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    line-height: 1.6;
    min-height: 100vh;
  }}

  header {{
    border-bottom: 1px solid var(--border);
    padding: 28px 40px 24px;
    display: flex;
    align-items: baseline;
    gap: 20px;
  }}

  header h1 {{
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: var(--accent1);
    text-transform: uppercase;
  }}

  header .sub {{
    color: var(--muted);
    font-size: 11px;
    letter-spacing: 0.1em;
  }}

  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1px;
    border: 1px solid var(--border);
    margin: 32px 40px 0;
    background: var(--border);
  }}

  .card {{
    background: var(--surface);
    padding: 24px 28px;
  }}

  .card-label {{
    font-size: 10px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 10px;
  }}

  .card-account {{
    font-family: 'Syne', sans-serif;
    font-size: 18px;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 16px;
  }}

  .card-value {{
    font-size: 28px;
    font-weight: 500;
    letter-spacing: -1px;
    margin-bottom: 12px;
  }}

  .stats-row {{
    display: flex;
    gap: 24px;
    margin-top: 8px;
  }}

  .stat {{
    display: flex;
    flex-direction: column;
    gap: 2px;
  }}

  .stat-label {{
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--muted);
  }}

  .stat-value {{
    font-size: 14px;
    font-weight: 500;
  }}

  .pos {{ color: var(--positive); }}
  .neg {{ color: var(--negative); }}
  .neu {{ color: var(--muted); }}

  .chart-section {{
    margin: 32px 40px 0;
  }}

  .section-title {{
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: var(--muted);
    margin-bottom: 16px;
  }}

  .chart-wrap {{
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 24px;
    position: relative;
    height: 320px;
  }}

  .two-col {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1px;
    background: var(--border);
    margin: 32px 40px 0;
    border: 1px solid var(--border);
  }}

  .positions-table {{
    background: var(--surface);
    padding: 24px;
    overflow-x: auto;
  }}

  table {{
    width: 100%;
    border-collapse: collapse;
  }}

  th {{
    text-align: left;
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--muted);
    padding: 0 12px 10px 0;
    border-bottom: 1px solid var(--border);
    font-weight: 400;
  }}

  td {{
    padding: 8px 12px 8px 0;
    border-bottom: 1px solid var(--border);
    font-size: 12px;
  }}

  tr:last-child td {{ border-bottom: none; }}

  .weight-bar {{
    height: 3px;
    background: var(--border);
    border-radius: 2px;
    margin-top: 4px;
  }}

  .weight-fill {{
    height: 100%;
    border-radius: 2px;
    background: var(--accent1);
  }}

  footer {{
    margin: 48px 40px 32px;
    color: var(--muted);
    font-size: 10px;
    letter-spacing: 0.08em;
  }}
</style>
</head>
<body>

<header>
  <h1>Politic-Trader</h1>
  <span class="sub">Paper Trading Performance Dashboard — Generated {date.today()}</span>
</header>

<div class="grid" id="summary-cards"></div>

<div class="chart-section">
  <div class="section-title">Cumulative Return</div>
  <div class="chart-wrap">
    <canvas id="cumReturnChart"></canvas>
  </div>
</div>

<div class="chart-section">
  <div class="section-title">Weekly Return</div>
  <div class="chart-wrap">
    <canvas id="weeklyReturnChart"></canvas>
  </div>
</div>

<div class="two-col" id="positions-section"></div>

<footer>
  Auto-generated by track_metrics.py &nbsp;·&nbsp; Data source: Alpaca Paper Trading + yfinance
</footer>

<script>
const accountsData = {accounts_js};
const spySeries    = {spy_series_js};
const latestData   = {latest_js};
const COLORS = {{
  FinRL: '#00e5ff',
  AR:    '#ff6b35',
  SPY:   '#5a6070',
}};
const DEFAULT_COLORS = ['#00e5ff','#ff6b35','#7fff6b','#c77dff','#ffbe0b'];

// --- Summary cards ---
const cardsEl = document.getElementById('summary-cards');
Object.entries(latestData).forEach(([acct, d], i) => {{
  const color = COLORS[acct] || DEFAULT_COLORS[i % DEFAULT_COLORS.length];
  const cum   = d.cumulative ?? 0;
  const wkly  = d.weekly ?? 0;
  const spyCum = d.spy_cumulative ?? 0;
  const vsspy = cum - spyCum;
  cardsEl.innerHTML += `
    <div class="card">
      <div class="card-label">Account</div>
      <div class="card-account" style="color:${{color}}">${{acct}}</div>
      <div class="card-value">$${{d.value.toLocaleString('en-US', {{minimumFractionDigits:2, maximumFractionDigits:2}})}}</div>
      <div class="stats-row">
        <div class="stat">
          <span class="stat-label">This Week</span>
          <span class="stat-value ${{wkly>=0?'pos':'neg'}}">${{wkly>=0?'+':''}}${{(wkly*100).toFixed(2)}}%</span>
        </div>
        <div class="stat">
          <span class="stat-label">Cumulative</span>
          <span class="stat-value ${{cum>=0?'pos':'neg'}}">${{cum>=0?'+':''}}${{(cum*100).toFixed(2)}}%</span>
        </div>
        <div class="stat">
          <span class="stat-label">vs SPY</span>
          <span class="stat-value ${{vsspy>=0?'pos':'neg'}}">${{vsspy>=0?'+':''}}${{(vsspy*100).toFixed(2)}}%</span>
        </div>
      </div>
    </div>`;
}});

// --- Cumulative return chart ---
function buildDatasets(key, includeSpy=true) {{
  const datasets = [];
  Object.entries(accountsData).forEach(([acct, data], i) => {{
    const color = COLORS[acct] || DEFAULT_COLORS[i % DEFAULT_COLORS.length];
    datasets.push({{
      label: acct,
      data: data.map(d => ({{x: d.date, y: d[key] != null ? +(d[key]*100).toFixed(3) : null}})),
      borderColor: color,
      backgroundColor: color + '18',
      borderWidth: 2,
      pointRadius: 3,
      pointHoverRadius: 5,
      tension: 0.3,
      fill: false,
    }});
  }});
  if (includeSpy && spySeries.length > 1) {{
    datasets.push({{
      label: 'SPY',
      data: spySeries.map(d => ({{x: d.date, y: d[key] != null ? +(d[key]*100).toFixed(3) : null}})),
      borderColor: COLORS.SPY,
      borderDash: [4, 4],
      borderWidth: 1.5,
      pointRadius: 0,
      tension: 0.3,
      fill: false,
    }});
  }}
  return datasets;
}}

const chartDefaults = {{
  type: 'line',
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{
        labels: {{ color: '#5a6070', font: {{ family: 'DM Mono', size: 11 }}, boxWidth: 20 }}
      }},
      tooltip: {{
        backgroundColor: '#111318',
        borderColor: '#1e2128',
        borderWidth: 1,
        titleColor: '#e8eaf0',
        bodyColor: '#5a6070',
        titleFont: {{ family: 'DM Mono' }},
        bodyFont:  {{ family: 'DM Mono' }},
        callbacks: {{ label: ctx => ` ${{ctx.dataset.label}}: ${{ctx.parsed.y >= 0 ? '+' : ''}}${{ctx.parsed.y.toFixed(2)}}%` }}
      }}
    }},
    scales: {{
      x: {{ type: 'category', ticks: {{ color: '#5a6070', font: {{ family: 'DM Mono', size: 10 }} }}, grid: {{ color: '#1e2128' }} }},
      y: {{ ticks: {{ color: '#5a6070', font: {{ family: 'DM Mono', size: 10 }}, callback: v => v+'%' }}, grid: {{ color: '#1e2128' }} }}
    }}
  }}
}};

new Chart(document.getElementById('cumReturnChart'), {{
  ...chartDefaults,
  data: {{ datasets: buildDatasets('cumulative', true) }}
}});

new Chart(document.getElementById('weeklyReturnChart'), {{
  ...chartDefaults,
  data: {{ datasets: buildDatasets('weekly', true) }}
}});

// --- Positions tables ---
const posSection = document.getElementById('positions-section');
Object.entries(latestData).forEach(([acct, d], i) => {{
  const color = COLORS[acct] || DEFAULT_COLORS[i % DEFAULT_COLORS.length];
  const positions = d.positions || [];
  const rows = positions.sort((a,b) => b.actual_weight - a.actual_weight).map(p => {{
    const plClass = p.unrealized_pl >= 0 ? 'pos' : 'neg';
    return `<tr>
      <td>${{p.symbol}}</td>
      <td>$${{p.market_value.toLocaleString('en-US', {{maximumFractionDigits:0}})}}</td>
      <td>
        ${{(p.actual_weight*100).toFixed(1)}}%
        <div class="weight-bar"><div class="weight-fill" style="width:${{Math.min(p.actual_weight*100*2,100)}}%;background:${{color}}"></div></div>
      </td>
      <td class="${{plClass}}">${{p.unrealized_pl >= 0 ? '+' : ''}}$${{p.unrealized_pl.toLocaleString('en-US', {{maximumFractionDigits:0}})}}</td>
    </tr>`;
  }}).join('');
  posSection.innerHTML += `
    <div class="positions-table">
      <div class="section-title" style="margin-bottom:16px;color:${{color}}">${{acct}} — Current Positions</div>
      <table>
        <thead><tr><th>Symbol</th><th>Value</th><th>Weight</th><th>Unreal. P&L</th></tr></thead>
        <tbody>${{rows || '<tr><td colspan="4" style="color:var(--muted)">No positions</td></tr>'}}</tbody>
      </table>
    </div>`;
}});
</script>
</body>
</html>"""

    output_path.write_text(html)
    logger.info(f"Dashboard written to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Record metrics snapshot and generate dashboard"
    )
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Snapshot date (default: today)",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Skip recording, just regenerate dashboard",
    )
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    if not args.report_only:
        accounts = load_accounts_from_env()
        benchmark = fetch_benchmark_prices(date.fromisoformat(args.date))

        if benchmark.get("spy_close"):
            logger.info(
                f"Benchmark: SPY={benchmark['spy_close']:.2f} | "
                f"weekly={benchmark.get('spy_weekly_return', 0):+.2%}"
            )

        # Load latest execution log for target weights
        exec_log_path = Path(f"logs/execution_{args.date}.json")
        target_weights_by_account = {}
        if exec_log_path.exists():
            with open(exec_log_path) as f:
                exec_log = json.load(f)
            for entry in exec_log.get("accounts", []):
                target_weights_by_account[entry["account"]] = entry.get(
                    "target_weights", {}
                )

        failed_accounts = []
        for account in accounts:
            logger.info(f"Fetching snapshot: {account['name']}")
            try:
                alpaca = get_alpaca_snapshot(account)
                target_weights = target_weights_by_account.get(account["name"], {})
                record_snapshot(
                    conn, args.date, account, alpaca, benchmark, target_weights
                )
            except Exception as e:
                logger.error(
                    f"Failed to record snapshot for {account['name']}: {e}",
                    exc_info=True,
                )
                failed_accounts.append((account["name"], str(e)))

    print_cli_report(conn)
    generate_html_dashboard(conn, DASHBOARD_PATH)
    conn.close()

    if not args.report_only and failed_accounts:
        logger.error(
            f"Metrics collection failed for {len(failed_accounts)} account(s):"
        )
        for name, err in failed_accounts:
            logger.error(f"  {name}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
