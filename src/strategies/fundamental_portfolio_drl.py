import warnings
import argparse

warnings.filterwarnings("ignore")
import contextlib
import hashlib
import io
import os
import sqlite3

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import pandas as pd
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt import expected_returns
from datetime import datetime
from pandas.tseries.offsets import BDay

# Try to import gymnasium instead of gym for compatibility
import gymnasium as gym


from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.preprocessor.preprocessors import data_split
from finrl import config
import random
import torch

import time
from rl_model import run_models

# ==== ADD：Temp directory ====
# CACHE_DIR = "./cache"
# CKPT_DIR  = "./checkpoints"  # For readability; training saved by rl_model.py
RESULTS_DIR = "./results"
CACHE_DIR = f"{RESULTS_DIR}/rl_cache"
# os.makedirs(CACHE_DIR, exist_ok=True)
# os.makedirs(CKPT_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


# ==== ADD：Deterministic & Random Seed ====
def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # CuDNN Deterministic: Same input, same output (slightly sacrifice speed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


# ==== ADD：Fix for crash at end of vec env ====
def _safe_DRL_prediction(model, environment, deterministic=True):
    """
    Run a test episode and ALWAYS return (account_value_df, actions_df),
    even if the vec env ends early.
    """
    test_env, test_obs = environment.get_sb_env()
    test_env.reset()
    n_steps = len(environment.df.index.unique())
    max_steps = max(n_steps - 1, 0)

    account_memory = None
    actions_memory = None

    for i in range(max_steps):
        action, _ = model.predict(test_obs, deterministic=deterministic)
        test_obs, rewards, dones, info = test_env.step(action)

        # Fetch before the terminal auto-reset clears the portfolio memories.
        if (i == max_steps - 1) or dones[0]:
            account_memory = test_env.env_method("save_asset_memory")
            actions_memory = test_env.env_method("save_action_memory")
            if dones[0] and getattr(environment, "verbose", 0):
                print("hit end!")
            break

    # Fallback: if for any reason memories weren't fetched in the loop
    if account_memory is None:
        account_memory = test_env.env_method("save_asset_memory")
    if actions_memory is None:
        actions_memory = test_env.env_method("save_action_memory")

    # env_method returns list-of-envs; take the first
    return account_memory[0], actions_memory[0]


# Apply the patch
DRLAgent.DRL_prediction = staticmethod(_safe_DRL_prediction)


# ==== ADD：Atomic Write ====
def atomic_to_csv(df: pd.DataFrame, path: str, index: bool | None = None):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=(True if index is None else index))
    os.replace(tmp, path)


# def atomic_to_parquet(df: pd.DataFrame, path: str, index: bool = False):
#    tmp = path + ".tmp"
#    df.to_parquet(tmp, index=index)
#    os.replace(tmp, path)


def atomic_write_json(obj: dict, path: str):
    import json

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


# ==== ADD：Progress Tracking ====
import json

PROGRESS_PATH = f"{RESULTS_DIR}/progress.json"


def load_progress():
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "last_idx": -1,
        "last_trade_date": None,
        "df_dict": None,
    }  # add df_dict field


def save_progress(idx, trade_date, df_dict):  # add df_dict parameter
    atomic_write_json(
        {
            "last_idx": idx,
            "last_trade_date": str(trade_date.date()),
            "df_dict": df_dict,  # save df_dict state
        },
        PROGRESS_PATH,
    )
    print(f"Saved progress to {PROGRESS_PATH}")


"""


import hashlib

def _hash_list(values) -> str:
    s = ",".join(map(str, sorted(list(values))))
    return hashlib.md5(s.encode()).hexdigest()[:10]

def load_or_build_fe_features(df_src: pd.DataFrame,
                              p1_stock: pd.Series,
                              earliest_date: pd.Timestamp,
                              end_exclusive: pd.Timestamp) -> pd.DataFrame:

#    Only cache FeatureEngineer.preprocess_data() output (without cov_list/return_list).
#    key is determined by (earliest_date, end_exclusive, stock set hash).

    key = f"{earliest_date.date()}_{end_exclusive.date()}_{_hash_list(p1_stock)}"
    feat_path = f"{CACHE_DIR}/fe_{key}.parquet"

    if os.path.exists(feat_path):
        print(f"Loading cached FE features from {feat_path}")
        return pd.read_parquet(feat_path)

    # —— Original logic: slice + FE preprocess ——
    df_ = df_src[df_src['tic'].isin(p1_stock) &
                 (df_src['date'] >= earliest_date) &
                 (df_src['date'] < end_exclusive)]
    if df_.empty:
        return df_

    fe = FeatureEngineer(use_technical_indicator=True,
                         use_turbulence=False,
                         user_defined_feature=False)
    df_ = fe.preprocess_data(df_)
    df_ = df_.sort_values(['date', 'tic'], ignore_index=True)
    # Keep the factorized index for lookback later
    df_.index = df_.date.factorize()[0]

    # Cache FE output (cov_list/return_list still calculated as before)
    atomic_to_parquet(df_, feat_path, index=False)
    print(f"Cached FE features to {feat_path}")
    return df_
"""


def check_per_date_stock_coverage(df_, stock_dim):
    stock_counts = df_.groupby("date")["tic"].nunique()
    invalid_dates = stock_counts[stock_counts != stock_dim]
    if not invalid_dates.empty:
        print("[WARNING] Found dates with missing stocks:")
        print(invalid_dates)
        return df_[df_["date"].isin(stock_counts[stock_counts == stock_dim].index)]
    return df_


def zscore_normalize_indicators(
    df: pd.DataFrame, indicators: list[str]
) -> pd.DataFrame:
    """
    Global Z-score normalization for technical indicators: --》 improve RL performance
      x' = (x - mean_all) / std_all
    - Only applies to columns listed in `indicators`
    - Safely handles inf/NaN; zero-variance columns become 0
    """
    df = df.copy()
    for col in indicators:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        mu = vals.mean(skipna=True)
        sigma = vals.std(skipna=True)
        if sigma and sigma > 0:
            df[col] = (vals - mu) / sigma
        else:
            # if no variance (or all NaN), set to 0 to avoid NaN spillover
            df[col] = 0.0
    return df


def compute_and_save_performance(
    df_daily_return: pd.DataFrame,
    df_actions: pd.DataFrame,
    out_prefix: str = "backtest",
    results_dir: str = "results",
    rf_annual: float = 0.0,
    trading_days: int = 252,
) -> pd.DataFrame:
    """
    Calculate backtest performance metrics and save them to files:
      - Total return, annualized return, annualized volatility, Sharpe ratio, max drawdown
      - Daily weight sum (check if ≈ 1)
      - Turnover (total and average daily)

    Saves:
      - <results_dir>/{out_prefix}_summary.csv: single-line summary table
      - <results_dir>/{out_prefix}_equity_curve.csv: equity curve and drawdown
      - <results_dir>/{out_prefix}_turnover.csv: daily turnover
      - <results_dir>/{out_prefix}_weights_sum.csv: daily sum of portfolio weights

    Parameters
    ----------
    df_daily_return : DataFrame
        Must contain 'daily_return' column; optionally a 'date' column (otherwise index is used).
    df_actions : DataFrame
        Index is date, columns are stock weights.
    out_prefix : str
        Prefix for saved files.
    results_dir : str
        Output directory.
    rf_annual : float
        Annualized risk-free rate for Sharpe calculation (e.g., 0.02 for 2%).
    trading_days : int
        Number of trading days used for annualization (default 252).

    Returns
    -------
    summary_df : DataFrame
        One-row DataFrame with key metrics.
    """
    if not isinstance(df_daily_return, pd.DataFrame) or df_daily_return.empty:
        raise ValueError("df_daily_return is None or empty")
    if "daily_return" not in df_daily_return.columns:
        raise ValueError("df_daily_return must contain 'daily_return'")
    if "date" in df_daily_return.columns:
        df_daily_return = df_daily_return.sort_values("date").set_index("date")
    import os

    os.makedirs(results_dir, exist_ok=True)

    # --- Prepare daily returns ---
    dr = df_daily_return.copy()
    if "date" in dr.columns:
        dr = dr.sort_values("date")
        dr.set_index("date", inplace=True)
    dr = dr[["daily_return"]].dropna()

    # --- Equity curve & drawdown ---
    equity = (1.0 + dr["daily_return"]).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_drawdown = drawdown.min() if len(drawdown) > 0 else np.nan

    # --- Annualized return, volatility, Sharpe ---
    n = len(dr)
    if n > 0:
        total_return = equity.iloc[-1] - 1.0
        ann_return = (equity.iloc[-1]) ** (trading_days / n) - 1.0
        ann_vol = dr["daily_return"].std() * np.sqrt(trading_days)
        rf_daily = (1.0 + rf_annual) ** (1.0 / trading_days) - 1.0
        excess_daily = dr["daily_return"] - rf_daily
        ann_excess_ret = excess_daily.mean() * trading_days
        sharpe = ann_excess_ret / ann_vol if ann_vol and ann_vol > 0 else np.nan
    else:
        total_return = ann_return = ann_vol = sharpe = np.nan

    # --- Weight sum check ---
    if df_actions is not None and not df_actions.empty:
        weights_sum = df_actions.sum(axis=1)
        weights_sum.to_frame("weights_sum").to_csv(
            os.path.join(results_dir, f"{out_prefix}_weights_sum.csv")
        )
        weights_sum_min = float(weights_sum.min())
        weights_sum_max = float(weights_sum.max())
        weights_sum_mean = float(weights_sum.mean())
    else:
        weights_sum_min = weights_sum_max = weights_sum_mean = np.nan

    # --- Turnover calculation ---
    # turnover_t = sum(|w_t - w_{t-1}|) / 2
    if df_actions is not None and len(df_actions) > 1:
        actions_sorted = df_actions.copy().sort_index()
        dw = actions_sorted.diff().abs()
        turnover_series = dw.sum(axis=1) / 2.0
        turnover_series = turnover_series.dropna()
        turnover_series.to_frame("turnover").to_csv(
            os.path.join(results_dir, f"{out_prefix}_turnover.csv")
        )
        total_turnover = float(turnover_series.sum())
        avg_daily_turnover = float(turnover_series.mean())
    else:
        total_turnover = avg_daily_turnover = np.nan

    # --- Save equity curve & drawdown ---
    eq_df = pd.DataFrame(
        {
            "equity": equity,
            "drawdown": drawdown,
            "daily_return": dr["daily_return"],
        }
    )
    eq_df.to_csv(os.path.join(results_dir, f"{out_prefix}_equity_curve.csv"))

    # --- Summary table ---
    summary = {
        "n_days": n,
        "total_return": float(total_return) if pd.notna(total_return) else np.nan,
        "annual_return": float(ann_return) if pd.notna(ann_return) else np.nan,
        "annual_vol": float(ann_vol) if pd.notna(ann_vol) else np.nan,
        "sharpe": float(sharpe) if pd.notna(sharpe) else np.nan,
        "max_drawdown": float(max_drawdown) if pd.notna(max_drawdown) else np.nan,
        "weights_sum_min": weights_sum_min,
        "weights_sum_mean": weights_sum_mean,
        "weights_sum_max": weights_sum_max,
        "total_turnover": total_turnover,
        "avg_daily_turnover": avg_daily_turnover,
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(
        os.path.join(results_dir, f"{out_prefix}_summary.csv"), index=False
    )
    return summary_df


def load_price_data_from_db(db_path: str = "./data/finrl_trading.db") -> pd.DataFrame:
    """Load OHLCV history from SQLite and normalize to RL schema."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Price DB not found: {db_path}")

    query = """
        SELECT ticker, date, open, high, low, close, adj_close, volume
        FROM price_data
        WHERE date IS NOT NULL
          AND close IS NOT NULL
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(query, conn)

    if df.empty:
        raise ValueError("price_data table is empty in finrl_trading.db")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "ticker"])
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df = df[df["ticker"] != ""]

    df["adjcp"] = np.where(df["adj_close"].notna(), df["adj_close"], df["close"])
    df["gvkey"] = df["ticker"]
    df["tic"] = df["ticker"]
    df["volume"] = df["volume"].fillna(0.0)

    out = df[["date", "open", "close", "high", "low", "adjcp", "volume", "gvkey", "tic"]]
    out = (
        out.drop_duplicates(["tic", "date"])
        .sort_values(["date", "tic"])
        .reset_index(drop=True)
    )
    return out


def build_quarterly_trade_dates(df: pd.DataFrame) -> list[pd.Timestamp]:
    """Use first available trading day of Mar/Jun/Sep/Dec as rebalance dates."""
    dates = pd.Series(pd.to_datetime(df["date"]).dropna().unique()).sort_values()
    by_month = pd.DataFrame({"date": dates})
    by_month["year"] = by_month["date"].dt.year
    by_month["month"] = by_month["date"].dt.month
    by_month = by_month[by_month["month"].isin([3, 6, 9, 12])]
    first_dates = by_month.groupby(["year", "month"], as_index=False)["date"].min()
    return [pd.Timestamp(d) for d in first_dates["date"].sort_values().tolist()]


def _stable_ticker_hash(tickers: list[str]) -> str:
    joined = ",".join(sorted(tickers))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def _window_cache_path(
    current_trade_date: pd.Timestamp,
    next_trade_date: pd.Timestamp,
    stable_tics: list[str],
    lookback: int,
) -> str:
    ticker_hash = _stable_ticker_hash(stable_tics)
    current = current_trade_date.strftime("%Y%m%d")
    next_date = next_trade_date.strftime("%Y%m%d")
    filename = f"window_v2_{current}_{next_date}_{lookback}_{ticker_hash}.pkl"
    return os.path.join(CACHE_DIR, filename)


def build_or_load_window_data(
    df: pd.DataFrame,
    stable_tics: list[str],
    current_trade_date: pd.Timestamp,
    next_trade_date: pd.Timestamp,
    earliest_date: pd.Timestamp,
    lookback: int,
    quiet: bool,
) -> pd.DataFrame:
    """Build or load the feature/covariance frame for one RL training window."""
    cache_path = _window_cache_path(
        current_trade_date=current_trade_date,
        next_trade_date=next_trade_date,
        stable_tics=stable_tics,
        lookback=lookback,
    )
    if os.path.exists(cache_path):
        if not quiet:
            print(f"Loading cached RL window data: {cache_path}")
        return pd.read_pickle(cache_path)

    df_window = df[
        df["tic"].isin(stable_tics)
        & (df["date"] >= earliest_date)
        & (df["date"] < next_trade_date)
    ].copy()
    if df_window.empty:
        return df_window

    fe = FeatureEngineer(
        use_technical_indicator=True,
        use_turbulence=False,
        user_defined_feature=False,
    )
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            df_window = fe.preprocess_data(df_window)
    else:
        df_window = fe.preprocess_data(df_window)
    df_window = df_window.sort_values(["date", "tic"], ignore_index=True)

    unique_dates = pd.Index(pd.to_datetime(df_window["date"]).drop_duplicates())
    price_pivot = df_window.pivot_table(index="date", columns="tic", values="close")
    price_pivot = price_pivot.reindex(columns=stable_tics).sort_index()
    returns_pivot = price_pivot.pct_change().replace([np.inf, -np.inf], np.nan)
    returns_pivot = returns_pivot.fillna(0.0)

    cov_rows = []
    for pos in range(lookback, len(unique_dates)):
        return_lookback = returns_pivot.iloc[pos - lookback + 1 : pos + 1].copy()
        cov_rows.append(
            {
                "date": unique_dates[pos],
                "cov_list": return_lookback.cov().values,
                "return_list": return_lookback,
            }
        )

    df_cov = pd.DataFrame(cov_rows)
    df_window = df_window.merge(df_cov, on="date")
    df_window = df_window.sort_values(["date", "tic"]).reset_index(drop=True)
    df_window.to_pickle(cache_path)
    if not quiet:
        print(f"Cached RL window data: {cache_path}")
    return df_window


def main():
    parser = argparse.ArgumentParser(description="Offline DRL portfolio training/backtest")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", default="a2c,ppo,ddpg", help="Comma-separated models: a2c,ppo,ddpg")
    parser.add_argument("--a2c-timesteps", type=int, default=50000)
    parser.add_argument("--ppo-timesteps", type=int, default=80000)
    parser.add_argument("--ddpg-timesteps", type=int, default=50000)
    parser.add_argument("--max-windows", type=int, default=0, help="0 means all windows")
    parser.add_argument("--max-universe", type=int, default=0, help="0 means no cap; otherwise cap stable universe size")
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Retry from this quarterly-window index instead of saved progress.",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce debug logs")
    args = parser.parse_args()

    # read price data

    set_global_seed(args.seed)
    selected_models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    allowed_models = {"a2c", "ppo", "ddpg"}
    selected_models = [m for m in selected_models if m in allowed_models]
    if not selected_models:
        raise ValueError("No valid models selected. Use --models a2c,ppo,ddpg")

    print("Loading price data from SQLite...")
    df = load_price_data_from_db("./data/finrl_trading.db")
    print(f"Price data loaded: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Number of unique stocks: {len(df['tic'].unique())}")

    trade_date = build_quarterly_trade_dates(df)
    if len(trade_date) < 2:
        raise ValueError("Not enough quarterly trade dates in price data to run RL pipeline")
    print(f"Quarterly trade dates generated: {len(trade_date)}")
    print(f"First/last trade dates: {trade_date[0]} -> {trade_date[-1]}")
    global_min_date = pd.to_datetime(df["date"]).min()

    prog = load_progress()
    start_idx = (
        args.start_index
        if args.start_index is not None
        else max(1, prog.get("last_idx", -1) + 1)
    )
    if start_idx < 1:
        raise ValueError("--start-index must be at least 1")
    df_dict = prog.get("df_dict")
    if not isinstance(df_dict, dict):
        df_dict = {"trade_date": [], "gvkey": [], "weights": []}
    else:
        for key in ("trade_date", "gvkey", "weights"):
            if key not in df_dict or not isinstance(df_dict[key], list):
                df_dict[key] = []

    # 1 year
    # testing_window = pd.Timedelta(np.timedelta64(1,'Y'))
    testing_window = pd.Timedelta(days=365)  # 1 year --
    # max_rolling_window = pd.Timedelta(np.timedelta64(10, 'Y'))
    max_rolling_window = pd.Timedelta(
        days=1095
    )  # 10 years -->change from 10year to 3 year

    print(f"Number of trade dates used (should be ~31): {len(trade_date)}")
    # ==== ADD：Progress Tracking ====
    # prog = load_progress()
    # start_idx = max(1, prog.get("last_idx", -1) + 1)

    attempted_windows = 0
    for idx in range(start_idx, len(trade_date)):
        if args.max_windows > 0 and attempted_windows >= args.max_windows:
            print(f"Reached --max-windows={args.max_windows}, stopping loop")
            break
        current_trade_date = trade_date[idx - 1]
        min_required_date = current_trade_date - max_rolling_window
        if global_min_date > min_required_date:
            print(
                f"Skipping {current_trade_date}: not enough rolling history "
                f"(need <= {min_required_date.date()}, have {global_min_date.date()})"
            )
            save_progress(idx, trade_date[idx], df_dict)
            continue
        # for idx in range(1, len(trade_date)):
        #    current_trade_date = trade_date[idx-1]
        #
        # Build universe from symbols tradable on current trade date
        day_slice = df[df["date"] == current_trade_date][["tic"]].drop_duplicates()
        if day_slice.empty:
            print(f"Warning: no symbols at trade date {current_trade_date}. Skipping...")
            continue
        p1_stock = day_slice["tic"].astype(str).tolist()

        earliest_date = current_trade_date - max_rolling_window

        df_ = df[
            df["tic"].isin(p1_stock)
            & (df["date"] >= earliest_date)
            & (df["date"] < trade_date[idx])
        ]
        print(f"Processing trade date {idx}: {current_trade_date}")
        print(f"Data shape: {df_.shape}")

        if df_.empty:
            print(f"Warning: No data for trade date {current_trade_date}. Skipping...")
            save_progress(idx, trade_date[idx], df_dict)
            continue

        if len(pd.to_datetime(df_["date"]).drop_duplicates()) < 252:
            print(
                f"Skipping {current_trade_date}: insufficient lookback days for covariance "
                f"(need >=252, have {len(pd.to_datetime(df_['date']).drop_duplicates())})"
            )
            save_progress(idx, trade_date[idx], df_dict)
            continue

        # Keep a stable universe: symbols must exist on every date in this window.
        n_dates_window = df_["date"].nunique()
        per_tic_counts = df_.groupby("tic")["date"].nunique()
        stable_tics = sorted(
            per_tic_counts[per_tic_counts == n_dates_window].index.astype(str).tolist()
        )
        if len(stable_tics) < 20:
            print(
                f"Skipping {current_trade_date}: insufficient stable-universe symbols "
                f"(need >=20, have {len(stable_tics)})"
            )
            save_progress(idx, trade_date[idx], df_dict)
            continue
        if args.max_universe > 0 and len(stable_tics) > args.max_universe:
            stable_tics = stable_tics[: args.max_universe]
        df_ = df_[df_["tic"].astype(str).isin(stable_tics)].copy()

        lookback = 252
        df_ = build_or_load_window_data(
            df=df,
            stable_tics=stable_tics,
            current_trade_date=current_trade_date,
            next_trade_date=trade_date[idx],
            earliest_date=earliest_date,
            lookback=lookback,
            quiet=args.quiet,
        )
        if df_.empty:
            print(f"Warning: No prepared data for trade date {current_trade_date}. Skipping...")
            save_progress(idx, trade_date[idx], df_dict)
            continue

        # Stabilize again after FE/cov merge: keep only dates with complete universe,
        # then keep only symbols present in all remaining dates.
        expected_dim = len(stable_tics)
        per_date_counts = df_.groupby("date")["tic"].nunique()
        valid_dates = per_date_counts[per_date_counts == expected_dim].index
        df_ = df_[df_["date"].isin(valid_dates)].copy()
        if df_.empty:
            print(f"Skipping {current_trade_date}: empty after post-FE stabilization")
            save_progress(idx, trade_date[idx], df_dict)
            continue

        tics_by_date = (
            df_.groupby("date")["tic"].apply(lambda s: set(s.astype(str))).tolist()
        )
        stable_tics_post = sorted(set.intersection(*tics_by_date)) if tics_by_date else []
        if len(stable_tics_post) < 20:
            print(
                f"Skipping {current_trade_date}: insufficient stable symbols after FE "
                f"(need >=20, have {len(stable_tics_post)})"
            )
            save_progress(idx, trade_date[idx], df_dict)
            continue
        df_ = df_[df_["tic"].astype(str).isin(stable_tics_post)].copy()
        df_ = df_.sort_values(["date", "tic"], ignore_index=True)

        stock_dimension = len(stable_tics_post)
        # FinRL StockPortfolioEnv builds state as cov_matrix + indicator rows:
        # shape = (stock_dim + len(indicators), stock_dim).
        state_space = stock_dimension
        env_kwargs = {
            "hmax": 100,
            "initial_amount": 1000000,
            "transaction_cost_pct": 0.001,
            "state_space": state_space,
            "stock_dim": stock_dimension,
            "tech_indicator_list": config.INDICATORS,
            "action_space": stock_dimension,
            "reward_scaling": 1e-4,
        }

        # Skip early/invalid windows with insufficient data for RL split.
        train_start = current_trade_date - max_rolling_window
        train_end = current_trade_date - testing_window
        test_start = current_trade_date - testing_window
        test_end = current_trade_date
        train_rows = df_[(df_["date"] >= train_start) & (df_["date"] < train_end)]
        test_rows = df_[(df_["date"] >= test_start) & (df_["date"] < test_end)]
        if train_rows.empty or test_rows.empty:
            print(
                f"Skipping {current_trade_date}: insufficient train/test rows "
                f"(train={len(train_rows)}, test={len(test_rows)})"
            )
            save_progress(idx, trade_date[idx], df_dict)
            continue

        attempted_windows += 1
        try:
            # before calling run_models, rename column name
            if not args.quiet:
                print(f"=== DEBUG: Before run_models ===")
                print(f"Before rename - df_ columns: {list(df_.columns)}")
                print(f"Before rename - df_ has 'date' column: {'date' in df_.columns}")
                print(
                    f"Before rename - df_ has 'datadate' column: {'datadate' in df_.columns}"
                )
                print(f"Before rename - df_ shape: {df_.shape}")
                print(f"Before rename - df_ sample data:")
                print(df_.head(2))

            # df_ = df_.rename(columns={'date': 'datadate'})
            if not args.quiet:
                print(f"=== DEBUG: After rename ===")
                print(f"After rename - df_ columns: {list(df_.columns)}")
                print(f"After rename - df_ has 'date' column: {'date' in df_.columns}")
                print(
                    f"After rename - df_ has 'datadate' column: {'datadate' in df_.columns}"
                )
                print(f"=== DEBUG: Calling run_models ===")
                print(f"Calling run_models with date_column='date'")
            # print(f"Stock count used in training: {len(df_.tic.unique())}")
            # print(f"Stock list: {df_.tic.unique()}")
            df_ = check_per_date_stock_coverage(df_, stock_dimension)
            # move td3 and sac model td3_model,sac_model,
            try:
                a2c_model, ppo_model, ddpg_model, best_model = run_models(
                    df_,
                    "date",
                    current_trade_date,
                    env_kwargs,
                    testing_window,
                    max_rolling_window,
                    model_names=selected_models,
                    timesteps={
                        "a2c": args.a2c_timesteps,
                        "ppo": args.ppo_timesteps,
                        "ddpg": args.ddpg_timesteps,
                    },
                    verbose=0 if args.quiet else 1,
                )
                if not args.quiet:
                    print(f"=== DEBUG: run_models completed successfully ===")
            except Exception as run_models_error:
                print(f"=== DEBUG: run_models failed ===")
                print(f"Error in run_models: {str(run_models_error)}")
                print(f"Error type: {type(run_models_error)}")
                import traceback

                print(f"Traceback:")
                # traceback.print_exc()
                raise run_models_error

            # now df_ has 'datadate' column, use it directly
            if not args.quiet:
                print(f"=== DEBUG: Trading data ===")
                print(f"Before data_split - df_ columns: {list(df_.columns)}")
                print(f"Before data_split - df_ shape: {df_.shape}")
                print(f"current_trade_date: {current_trade_date}")
                print(f"trade_date[idx]: {trade_date[idx]}")

            trade = data_split(df_, current_trade_date, trade_date[idx])
            if not args.quiet:
                print(f"=== DEBUG: After trade date data_split ===")
                print(
                    f"After data_split - trade shape: {trade.shape if hasattr(trade, 'shape') else 'No shape'}"
                )
                print(
                    f"After data_split - trade columns: {list(trade.columns) if hasattr(trade, 'columns') else 'No columns'}"
                )
                print(f"After data_split - trade type: {type(trade)}")

            # print(f"=== DEBUG: Before StockPortfolioEnv ===")
            e_trade_gym = StockPortfolioEnv(df=trade, **env_kwargs)
            if not args.quiet:
                print(f"=== DEBUG: StockPortfolioEnv created successfully ===")

            # print(f"=== DEBUG: Before DRL_prediction ===")
            # print(f"Predicting with model: A2C")
            # using best model as call back test , if best model is null , then select a2c model
            if not args.quiet:
                print("=== DEBUG: Before DRL_prediction ===")
            # ==== ADD: Use best_model if available; fallback to A2C if best_model is None ====
            model_for_backtest = best_model if best_model is not None else a2c_model
            # print(f"best model is {best_model}")
            if model_for_backtest is None:
                raise RuntimeError(
                    "No model available for backtesting (best_model and a2c_model are both None)."
                )

            model_name = type(model_for_backtest).__name__
            if not args.quiet:
                print(f"trade date Predicting with model: {model_name}")

            df_daily_return, df_actions = DRLAgent.DRL_prediction(
                model=model_for_backtest, environment=e_trade_gym
            )
            #        df_daily_return, df_actions = DRLAgent.DRL_prediction(
            #        model=a2c_model, environment=e_trade_gym
            #        )
            if not args.quiet:
                print(f"tradedate  df_daily_return.shape: {df_daily_return.shape}")
                print(f"tradedatedf_actions.shape: {df_actions.shape}")
                print(f"=== DEBUG: DRL_prediction completed successfully ===")
                print(
                    f"df_actions shape: {df_actions.shape if hasattr(df_actions, 'shape') else 'No shape'}"
                )
                print(
                    f"df_actions columns: {list(df_actions.columns) if hasattr(df_actions, 'columns') else 'No columns'}"
                )

            # weight accumulation
            for i in range(len(df_actions)):
                for j in df_actions.columns:
                    df_dict["trade_date"].append(df_actions.index[i])
                    df_dict["gvkey"].append(j)
                    df_dict["weights"].append(df_actions.loc[df_actions.index[i], j])

            out_prefix = f"bt_{current_trade_date.strftime('%Y%m%d')}_{trade_date[idx].strftime('%Y%m%d')}"
            if not args.quiet:
                print(f"[PERF CALL] calling compute_and_save_performance for {out_prefix}")

            try:
                summary_df = compute_and_save_performance(
                    df_daily_return=df_daily_return,
                    df_actions=df_actions,
                    out_prefix=out_prefix,
                    results_dir="results",
                    rf_annual=0.02,
                    trading_days=252,
                )
                print(summary_df)
                print(
                    f"[PERF DONE] compute_and_save_performance finished for {out_prefix}"
                )
            except Exception as perf_err:
                print(f"[PERF ERROR] {out_prefix}: {perf_err}")

        except Exception as e:
            print(
                f"[PERF SKIP] compute_and_save_performance skipped due to error in {current_trade_date}"
            )
            print(f"Error processing trade date {current_trade_date}: {str(e)}")

            # Add detailed debugging information for array dimension mismatch
            if "array dimensions" in str(e) and "concatenation axis" in str(e):
                print("\n=== ARRAY DIMENSION DEBUG INFO ===")
                print(f"Current trade date: {current_trade_date}")
                print(f"DataFrame shape before run_models: {df_.shape}")
                print(f"Number of unique stocks: {len(df_.tic.unique())}")
                # make sure right column name
                if "date" in df_.columns:
                    print(f"Number of unique dates: {len(df_.date.unique())}")
                elif "datadate" in df_.columns:
                    print(f"Number of unique dates: {len(df_.datadate.unique())}")

                # Check data structure
                print(f"\nDataFrame columns: {list(df_.columns)}")
                print(f"DataFrame dtypes: {df_.dtypes}")

                # Check for any NaN values
                print(f"\nNaN values in DataFrame:")
                print(df_.isnull().sum())

                # Check data distribution by stock
                stock_counts = df_.groupby("tic").size()
                print(f"\nData points per stock:")
                print(f"Min: {stock_counts.min()}")
                print(f"Max: {stock_counts.max()}")
                print(f"Mean: {stock_counts.mean():.2f}")
                print(f"Stocks with < 252 data points: {(stock_counts < 252).sum()}")

                # Check date range for each stock
                print(f"\nDate range analysis:")
                for tic in df_.tic.unique()[:5]:  # Show first 5 stocks
                    stock_data = df_[df_.tic == tic]
                    # make sure right column name
                    date_col = "date" if "date" in stock_data.columns else "datadate"
                    print(
                        f"Stock {tic}: {stock_data[date_col].min()} to {stock_data[date_col].max()} ({len(stock_data)} records)"
                    )

                # Check if there are any stocks with insufficient data
                insufficient_stocks = stock_counts[stock_counts < 252]
                if len(insufficient_stocks) > 0:
                    print(f"\nStocks with insufficient data (< 252 records):")
                    print(insufficient_stocks.head(10))
        finally:
            save_progress(idx, trade_date[idx], df_dict)  #  save df_dict

    # save the accumulated weights data after the loop
    df_rl = pd.DataFrame(df_dict)
    df_rl.to_csv("./results/drl_weight.csv")
    print("DRL weights saved to drl_weight.csv")

    # add debug info at the end of the file
    print(f"\nDebug: df_dict contents:")
    print(f"  trade_date entries: {len(df_dict['trade_date'])}")
    print(f"  gvkey entries: {len(df_dict['gvkey'])}")
    print(f"  weights entries: {len(df_dict['weights'])}")

    if len(df_dict["trade_date"]) == 0:
        print("Warning: No data was processed. df_dict is empty.")
        print("This could be due to:")
        print("1. All trade dates were skipped")
        print("2. No data available for the selected stocks")
        print("3. Errors in the DRL model training/prediction")
    else:
        # df_rl = pd.DataFrame(df_dict)
        # df_rl.to_csv("drl_weight.csv")
        # print(f"DRL weights saved to drl_weight.csv")
        print(f"Final DataFrame shape: {df_rl.shape}")


if __name__ == "__main__":
    # Windows/Colab/多数环境都推荐 spawn；SB3 的 SubprocVecEnv 也默认用 spawn
    import multiprocessing as mp
    from multiprocessing import freeze_support

    freeze_support()  # 不是必须，但对某些环境友好

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # 已设置过启动方式会抛 RuntimeError，忽略即可
        pass

    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted. Cleaning up child processes...")
        import multiprocessing as mp
        for child in mp.active_children():
            child.terminate()
            child.join(timeout=1)
        print("Interrupted cleanly.")
