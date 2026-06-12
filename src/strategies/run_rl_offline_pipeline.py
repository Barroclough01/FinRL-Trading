#!/usr/bin/env python3
"""Run offline RL train/eval and enforce a promotion gate."""

import argparse
import glob
import importlib
import json
import os
import subprocess
import sys
from datetime import date
from math import isfinite
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.strategies.rl_contract import load_rl_contract

DEFAULT_GATE_PATH = PROJECT_ROOT / "src/strategies/rl_acceptance_gate.json"
DEFAULT_RL_CONTRACT_PATH = PROJECT_ROOT / "src/strategies/rl_contract.json"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"
REQUIRED_MODULES = {
    "finrl": "finrl",
    "pypfopt": "PyPortfolioOpt",
}


def _load_gate(gate_path: Path) -> dict:
    with gate_path.open() as f:
        return json.load(f)


def _run_training(
    seed: int,
    models: str,
    a2c_timesteps: int,
    ppo_timesteps: int,
    ddpg_timesteps: int,
    max_windows: int,
    max_universe: int,
    quiet: bool,
) -> None:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    cmd = [
        sys.executable,
        "src/strategies/fundamental_portfolio_drl.py",
        "--seed",
        str(seed),
        "--models",
        models,
        "--a2c-timesteps",
        str(a2c_timesteps),
        "--ppo-timesteps",
        str(ppo_timesteps),
        "--ddpg-timesteps",
        str(ddpg_timesteps),
        "--max-windows",
        str(max_windows),
        "--max-universe",
        str(max_universe),
    ]
    if quiet:
        cmd.append("--quiet")
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Training script failed with return code {proc.returncode}: {' '.join(cmd)}"
        )


def _check_required_modules() -> None:
    missing: list[str] = []
    for module_name, package_name in REQUIRED_MODULES.items():
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            missing.append(package_name)
    if missing:
        joined = " ".join(sorted(missing))
        raise RuntimeError(
            "Missing Python dependencies for RL pipeline: "
            f"{', '.join(sorted(missing))}. Install with: uv pip install {joined}"
        )


def _candidate_results_dirs(results_dir: Path) -> list[Path]:
    return [
        results_dir,
        PROJECT_ROOT / "src/strategies/results",
        PROJECT_ROOT / "src/strategies/output/results",
    ]


def _load_summaries(results_dir: Path) -> tuple[pd.DataFrame, Path]:
    summary_paths: list[str] = []
    selected_dir: Path | None = None
    for candidate in _candidate_results_dirs(results_dir):
        paths = sorted(glob.glob(str(candidate / "bt_*_summary.csv")))
        if paths:
            summary_paths = paths
            selected_dir = candidate
            break

    if not summary_paths or selected_dir is None:
        searched = ", ".join(str(p) for p in _candidate_results_dirs(results_dir))
        raise FileNotFoundError(
            "No backtest summaries found. Searched: "
            f"{searched}. Run without --skip-train to generate summaries first."
        )
    frames = [pd.read_csv(path) for path in summary_paths]
    return pd.concat(frames, ignore_index=True), selected_dir


def _safe_mean(series: pd.Series) -> float | None:
    mean = pd.to_numeric(series, errors="coerce").dropna().mean()
    if pd.isna(mean) or not isfinite(float(mean)):
        return None
    return float(mean)


def _evaluate_gate(summary_df: pd.DataFrame, gate: dict) -> tuple[bool, dict]:
    min_metric_days = int(gate.get("min_metric_days", 20))
    summary_df = summary_df.copy()
    summary_df["n_days"] = pd.to_numeric(summary_df["n_days"], errors="coerce")
    metric_df = summary_df[summary_df["n_days"] >= min_metric_days]

    metrics = {
        "windows": int(len(summary_df)),
        "metric_windows": int(len(metric_df)),
        "skipped_windows_too_short": int(len(summary_df) - len(metric_df)),
        "min_metric_days": min_metric_days,
        "avg_sharpe": _safe_mean(metric_df["sharpe"]),
        "avg_max_drawdown": _safe_mean(metric_df["max_drawdown"]),
        "avg_annual_return": _safe_mean(metric_df["annual_return"]),
        "avg_daily_turnover": _safe_mean(metric_df["avg_daily_turnover"]),
    }
    enough_windows = metrics["metric_windows"] >= int(gate["min_windows"])
    if not enough_windows:
        checks = {
            "min_windows": False,
            "min_avg_sharpe": None,
            "max_avg_drawdown": None,
            "min_avg_annual_return": None,
            "max_avg_daily_turnover": None,
        }
        return False, {
            "status": "insufficient_data",
            "reason": (
                f"Need {gate['min_windows']} windows with at least "
                f"{min_metric_days} trading days; found {metrics['metric_windows']}."
            ),
            "metrics": metrics,
            "checks": checks,
            "gate": gate,
        }

    checks = {
        "min_windows": True,
        "min_avg_sharpe": (
            metrics["avg_sharpe"] is not None
            and metrics["avg_sharpe"] >= float(gate["min_avg_sharpe"])
        ),
        "max_avg_drawdown": (
            metrics["avg_max_drawdown"] is not None
            and metrics["avg_max_drawdown"] >= float(gate["max_avg_drawdown"])
        ),
        "min_avg_annual_return": (
            metrics["avg_annual_return"] is not None
            and metrics["avg_annual_return"] >= float(gate["min_avg_annual_return"])
        ),
        "max_avg_daily_turnover": (
            metrics["avg_daily_turnover"] is not None
            and metrics["avg_daily_turnover"] <= float(gate["max_avg_daily_turnover"])
        ),
    }
    return all(checks.values()), {
        "status": "passed" if all(checks.values()) else "failed",
        "metrics": metrics,
        "checks": checks,
        "gate": gate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline RL train/eval + acceptance gate"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and only evaluate existing results/*.csv summaries",
    )
    parser.add_argument(
        "--gate-config",
        type=Path,
        default=DEFAULT_GATE_PATH,
        help="Path to RL acceptance gate JSON",
    )
    parser.add_argument(
        "--rl-contract",
        type=Path,
        default=DEFAULT_RL_CONTRACT_PATH,
        help="Path to the RL contract JSON document",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing bt_*_summary.csv outputs",
    )
    parser.add_argument("--models", default="a2c", help="Comma-separated: a2c,ppo,ddpg")
    parser.add_argument("--a2c-timesteps", type=int, default=5000)
    parser.add_argument("--ppo-timesteps", type=int, default=10000)
    parser.add_argument("--ddpg-timesteps", type=int, default=5000)
    parser.add_argument("--max-windows", type=int, default=1)
    parser.add_argument("--max-universe", type=int, default=150)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not args.skip_train:
        _check_required_modules()
        _run_training(
            seed=args.seed,
            models=args.models,
            a2c_timesteps=args.a2c_timesteps,
            ppo_timesteps=args.ppo_timesteps,
            ddpg_timesteps=args.ddpg_timesteps,
            max_windows=args.max_windows,
            max_universe=args.max_universe,
            quiet=args.quiet,
        )

    gate = _load_gate(args.gate_config)
    rl_contract = load_rl_contract(args.rl_contract)
    summary_df, source_dir = _load_summaries(args.results_dir)
    passed, evaluation = _evaluate_gate(summary_df, gate)

    DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    report_path = DEFAULT_LOG_DIR / f"rl_offline_gate_{date.today().isoformat()}.json"
    payload = {
        "date": date.today().isoformat(),
        "source_results_dir": str(source_dir),
        "rl_contract": rl_contract,
        **evaluation,
    }
    with report_path.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"Gate report written: {report_path}")
    print(json.dumps(payload["metrics"], indent=2))
    if payload["status"] == "failed":
        print("RL acceptance gate failed")
        sys.exit(1)
    if payload["status"] == "insufficient_data":
        print(f"RL acceptance gate has insufficient data: {payload['reason']}")
        return
    print("RL acceptance gate passed")


if __name__ == "__main__":
    main()
