# Session Context — FinRL-X + Qlib Trading Stack

## What we've built so far

Two separate environments, both working:

### 1. Qlib (Windows, native Python / uv)

- Installed from source in `C:\Users\paxto\stock-trading\qlib` with a uv venv at `qlib-env`
- S&P 500 (~490 tickers) downloaded via yfinance, normalized, and dumped to Qlib binary format at `~/.qlib/qlib_data/us_data`
- Successfully ran LightGBM + Alpha158 benchmark end-to-end via `qrun` with a custom YAML (US data, SPY benchmark, 2010-2024 date range)
- MLflow tracking working — results stored in `mlruns/` inside the qlib repo
- Next Qlib step: set up RD-Agent in WSL for LLM-driven automated factor discovery

---

### 2. FinRL-X / FinRL-Trading (WSL, Ubuntu)

Repo cloned to `~/stock-trading/FinRL-Trading` with uv venv at `finrl-env`.

#### Alpaca Paper Trading — Two Accounts

| Account | Purpose | Config |
|---------|---------|--------|
| `FinRL` | ML-enhanced AR (live) | `AdaptiveRotationConf_v1.2.2.yaml` |
| `AR`    | Baseline AR (shadow comparison) | `AdaptiveRotationConf_baseline.yaml` |

Both accounts funded with $1,000,000 paper capital. Credentials stored in `.env` under `APCA_FinRL_*` and `APCA_AR_*`. `APCA_ACCOUNTS=FinRL,AR`.

---

### Three Strategy Layers

#### Layer 1 — Adaptive Rotation (working, live paper trading)

- Rules-based macro regime strategy (risk-on / neutral / risk-off / fast risk-off)
- Rotates across Growth Tech, Real Assets, Defensive, Cyclical asset groups
- Weekly rebalance + daily stop-loss / fast risk-off monitoring
- **ML-enhanced config (FinRL account):** ML-picked symbols per bucket, updated quarterly
- **Baseline config (AR account):** Original hardcoded symbols (AAPL, MSFT, NVDA, JPM, XOM, JNJ, etc.)
- Backtest 2020-2024 with original symbols: 32.9% annualized, Sharpe 1.32, max drawdown -23% vs SPY's -32%
- Run via: `python src/strategies/run_adaptive_rotation_strategy.py --config <yaml> --date YYYY-MM-DD`

#### Layer 2 — ML Bucket Selection (working, integrated)

- `src/strategies/ml_bucket_selection.py` — trains RF, XGBoost, LightGBM, HistGBM, ExtraTrees, Ridge, and Stacking ensemble per sector bucket
- Four buckets: `growth_tech`, `cyclical`, `real_assets`, `defensive`
- Uses 24 fundamental features + 7 momentum features + sector dummies
- Point-in-time S&P 500 membership filtering (no survivorship bias)
- Fundamental data in `data/finrl_trading.db` (SQLite) + `data/fundamental_data_full.csv`
- Mixed-vintage mode fixed: Q4 2025 fallback correctly populates late-filing buckets
- Latest run (2026-04-26, `--mixed-vintage`, 503 stocks across all 4 buckets):
  - `growth_tech` [Ridge]: SATS(+12.0%), MCHP(+9.5%), ON(+8.2%), ORCL(+8.2%), TXN(+6.9%)
  - `cyclical` [Ridge]: IVZ(+8.1%), ODFL(+6.2%), MAS(+6.2%), TROW(+6.2%), HON(+6.1%)
  - `real_assets` [Stacking]: LYB(+4.9%), FCX(+4.6%), ALB(+3.3%), DOW(+3.0%), SLB(+3.0%)
  - `defensive` [Stacking]: ADM(+4.2%), INCY(+3.4%), DVA(+2.8%), UHS(+2.5%), ZTS(+2.4%)
- Run via: `python src/strategies/ml_bucket_selection.py --mixed-vintage`

#### Layer 3 — RL model (present in repo, not yet run)

- `src/strategies/rl_model.py` — A2C, PPO, DDPG agents via FinRL + StableBaselines3
- Uses `StockPortfolioEnv` from original FinRL library
- Not yet integrated or tested

---

### Integration Layer (all working, live)

#### `refresh_fmp_daily.py` — Weekly price data refresh
- Auto-discovers all configs from `APCA_ACCOUNTS` in `.env`
- Fetches OHLCV data via yfinance (FMP deprecated their free tier endpoints)
- 52 unique symbols across both configs (25 ML config + 33 baseline, deduplicated)
- Checks NYSE calendar via `pandas_market_calendars` — skips silently on non-trading days
- Idempotent: deduplicates on date before writing
- CSV format: `data/fmp_daily/<TICKER>_daily.csv` (date, open, high, low, close, volume)
- Run via: `python refresh_fmp_daily.py [--force] [--dry-run] [--config PATH]`

#### `update_adaptive_rotation_symbols.py` — Quarterly ML→AR symbol patcher
- Reads ML predictions CSV and patches `asset_groups` symbol lists in AR YAML config
- Top-5 per bucket, handles YAML boolean tickers (`ON` → `"ON"`), timestamped backup before overwrite
- Run via: `python update_adaptive_rotation_symbols.py --top-n 5 [--dry-run]`

#### `run_paper_trading.py` — Weekly execution script (dual-account)
1. Loads all accounts from `APCA_ACCOUNTS` in `.env`
2. For each account: runs AR strategy for today → gets target weights
3. Connects to that account's Alpaca paper account
4. Generates dry-run order plan, checks market open/closed
5. Submits rebalance orders (`MARKET_CLOSED_ACTION=next_open` queues Friday-evening DAY orders for the next regular session)
6. Logs execution to `logs/execution_YYYY-MM-DD.json`
7. After all accounts: automatically runs `track_metrics.py`
- Run via: `python run_paper_trading.py [--dry-run] [--date YYYY-MM-DD] [--account FinRL]`

#### `track_metrics.py` — Performance metrics tracker
- Records weekly snapshot per account to `data/finrl_trading.db`
- Tables: `weekly_snapshot`, `weekly_weights`, `benchmark_prices`
- Fetches SPY/QQQ benchmark via yfinance
- Reads target weights from `logs/execution_YYYY-MM-DD.json`
- Prints CLI performance report (cumulative return, weekly return, vs SPY, vs each other)
- Generates HTML dashboard at `logs/dashboard.html`
- Latest snapshot 2026-05-29: FinRL=$1,139,861.42 (+4.46% cum), AR=$1,010,839.29 (+1.08% cum)
- Run via: `python track_metrics.py [--report-only] [--date YYYY-MM-DD]`

#### `run_paper_trading.ps1` — Windows PowerShell wrapper
- Called by Windows Task Scheduler every Friday at 9:25am ET
- Step 1: Runs `refresh_fmp_daily.py` (price refresh)
- Step 2: Runs `run_paper_trading.py` (strategy + orders + metrics)
- Logs to `logs/paper_trading_cron.log` (WSL) and `logs/task_scheduler_YYYY-MM-DD.log` (Windows)
- Located at `C:\Users\paxto\stock-trading\run_paper_trading.ps1`

---

## Key File Locations (WSL)

```
~/stock-trading/FinRL-Trading/
├── .env                                          # Multi-account Alpaca credentials + MARKET_CLOSED_ACTION=next_open
├── deploy.sh                                     # Main entry point (updated to v1.2.2)
├── refresh_fmp_daily.py                          # Weekly price refresh (yfinance, 52 symbols)
├── run_paper_trading.py                          # Weekly dual-account paper trading script
├── track_metrics.py                              # Performance metrics + HTML dashboard
├── update_adaptive_rotation_symbols.py           # Quarterly ML→AR symbol patcher
├── data/
│   ├── finrl_trading.db                          # SQLite: price_data, fundamental_data,
│   │                                             #   weekly_snapshot, weekly_weights, benchmark_prices
│   ├── fundamental_data_full.csv                 # Pre-built fundamental dataset (22,909 rows, 715 tickers)
│   ├── sp500_historical_constituents.csv
│   ├── fmp_daily/                                # OHLCV CSVs for all 52 symbols (yfinance)
│   │   └── <TICKER>_daily.csv                    # date, open, high, low, close, volume
│   └── sp500_ml_bucket_predictions_20260426_135742.csv
├── logs/
│   ├── paper_trading_cron.log                    # WSL-side execution log
│   ├── execution_YYYY-MM-DD.json                 # Per-run order log (both accounts)
│   ├── dashboard.html                            # HTML performance dashboard
│   └── paper_trading_YYYY-MM-DD.log              # Per-run detailed log
└── src/strategies/
    ├── AdaptiveRotationConf_v1.2.2.yaml          # ML-enhanced AR config (FinRL account)
    ├── AdaptiveRotationConf_baseline.yaml        # Original AR config (AR account)
    ├── ml_bucket_selection.py                    # ML fundamental factor model
    ├── ml_strategy.py                            # ML + min-variance portfolio construction
    └── rl_model.py                               # RL agents (not yet run)
```

## Key File Locations (Windows)

```
C:\Users\paxto\stock-trading\
├── run_paper_trading.ps1                         # Task Scheduler PowerShell wrapper
└── logs\                                         # Windows-side Task Scheduler logs
C:\Users\paxto\stock-trading\qlib\
├── examples\benchmarks\LightGBM\workflow_config_lightgbm_Alpha158.yaml
├── mlruns\
~/.qlib/qlib_data/us_data/
```

## Environment Activation

```bash
# WSL / FinRL-X
cd ~/stock-trading/FinRL-Trading
source finrl-env/bin/activate

# Windows / Qlib (PowerShell)
cd C:\Users\paxto\stock-trading\qlib
.\qlib-env\Scripts\activate
```

## Known Issues / Gotchas

- Tickers that are YAML booleans (`ON`, `NO`, `YES`) must be quoted — `update_adaptive_rotation_symbols.py` handles this automatically
- `APCA_BASE_URL` in `.env` must include `/v2` suffix: `https://paper-api.alpaca.markets/v2`
- Multi-account `.env` keys are case-sensitive: `APCA_FinRL_API_KEY` not `APCA_FINRL_API_KEY`
- Friday-evening DAY orders are queued by Alpaca for the next regular session; this is the expected weekly execution protocol
- Stale live-strategy or benchmark prices fail the refresh; stale RL-only symbols are reported as warnings because RL remains offline
- Confirmed inactive RL holdings are liquidated at their last cached close and retained as cash according to `src/strategies/rl_inactive_symbols.json`
- The quarterly ML workflow treats its Excel dashboard as optional; CSV outputs remain authoritative when `openpyxl` is unavailable
- FMP API deprecated all legacy endpoints post-Aug 2025; free tier only allows SPY on stable endpoints — yfinance used instead
- Task Scheduler wake-from-sleep requires laptop in sleep (not shutdown) + "Allow wake timers" enabled in Windows power settings
- AR baseline account (AR) may show fallback positions (SPY/QQQ/IAU/XLU/XLV) for first few weeks while regime signals warm up on new symbol CSVs
- `Target weights sum 1.000200 > 1` warning in AR strategy is a floating point artifact — normalizes correctly, benign

## Weekly Run Checklist

1. **Friday evening**: Task Scheduler runs after the computer is available
2. **Friday evening**: signals and closing-price metrics are recorded; DAY orders queue for the next regular session
3. **Monday after market open**: check Alpaca for both accounts (FinRL + AR) to confirm fills
4. **Friday or weekend**: `explorer.exe logs/dashboard.html` to view updated performance dashboard
5. **Quarterly (~July 2026)**: re-run `ml_bucket_selection.py --mixed-vintage` → `update_adaptive_rotation_symbols.py` → download any new symbols

## Current Roadmap Phase

Paper shadow-trading / evidence collection (see `docs/paper_to_production_roadmap.md`).

**Completed foundation work (2026-05-31):**
- Structured JSON strategy output, pre-trade validation, post-trade reconciliation
- Strategy decision records (SQLite + JSONL), weekly comparison metrics
- Live-vs-replay parity checks, production kill switch (`TRADING_DISABLED` / `.kill_switch`)
- Test suite: 17 tests in `tests/test_weekly_workflow.py`

**Paper comparison status (through 2026-05-29):**
- FinRL: +4.46% cumulative (ML-picked tech, higher turnover)
- AR: +1.08% cumulative (still in fallback — `all_groups_negative_excess_return`)
- SPY: +2.30%, QQQ: +3.43%

**Known incident (2026-06-05):** Both accounts failed on the first run after foundation work shipped. FinRL orders were placed but decision-record save crashed on pandas Timestamp serialization; AR was blocked by an overly strict negative-cash validation rule. Both bugs fixed 2026-06-07.

## Immediate Next Steps (in priority order)

See `docs/paper_to_production_roadmap.md` → **Path Forward** for the full plan.

1. **VERIFY**: Next Friday run completes end-to-end — decision records, reconciliation, metrics
2. **RESOLVE**: FinRL June 5 pending orders (confirm fills or cancel before next rebalance)
3. **MONITOR**: Keep FinRL vs AR comparison unchanged; collect evidence through June/July
4. **WATCH**: AR baseline — still in fallback; mega-cap groups underperform QQQ trend filter
5. **NEXT CODE SPRINT** (after one clean run): hard risk gate, order idempotency, alerting
6. **QUARTERLY (~July 2026)**: Re-run `ml_bucket_selection.py --mixed-vintage` → `update_adaptive_rotation_symbols.py`
7. **DEFERRED**: RL integration, production capital, AR fallback changes
