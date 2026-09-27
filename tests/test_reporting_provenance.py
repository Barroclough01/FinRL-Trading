import csv
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

import run_paper_trading
import track_metrics
from src.strategies import run_adaptive_rotation_strategy as strategy_runner


def comparison_db():
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE weekly_snapshot (
            snapshot_date TEXT, account TEXT, portfolio_value REAL, cash REAL,
            weekly_return REAL, cumulative_return REAL, spy_weekly_return REAL,
            spy_cumulative_return REAL
        );
        CREATE TABLE weekly_weights (
            snapshot_date TEXT, account TEXT, symbol TEXT,
            actual_weight REAL, target_weight REAL
        );
        CREATE TABLE benchmark_prices (
            price_date TEXT, spy_close REAL, qqq_close REAL
        );
    """)
    for day, account, ret in (
        ("2026-09-18", "FinRL", 0.01),
        ("2026-09-18", "AR", 0.02),
        ("2026-09-18", "RL", 0.03),
        ("2026-09-25", "FinRL", 0.04),
        ("2026-09-25", "AR", 0.05),
    ):
        conn.execute(
            "INSERT INTO weekly_snapshot VALUES (?, ?, 100, 10, ?, 0, NULL, NULL)",
            (day, account, ret),
        )
    for day, spy, qqq in (
        ("2026-09-11", 100, 200),
        ("2026-09-18", 101, 202),
        ("2026-09-24", None, None),
        ("2026-09-25", 103, 206),
    ):
        conn.execute("INSERT INTO benchmark_prices VALUES (?, ?, ?)", (day, spy, qqq))
    return conn


def test_mixed_dates_preserve_series_but_compare_last_shared_date(
    tmp_path, monkeypatch
):
    conn = comparison_db()
    metrics = track_metrics.calculate_comparison_metrics(conn, "2026-09-25")
    assert metrics["accounts"]["FinRL"]["as_of_date"] == "2026-09-25"
    assert metrics["accounts"]["RL"]["as_of_date"] == "2026-09-18"
    assert metrics["benchmarks"]["QQQ"]["as_of_date"] == "2026-09-25"
    assert metrics["comparison"] == {
        "status": "historical",
        "as_of_date": "2026-09-18",
        "weekly_returns": {
            "FinRL": 0.01,
            "AR": 0.02,
            "RL": 0.03,
            "SPY": pytest.approx(0.01),
            "QQQ": pytest.approx(0.01),
        },
    }
    monkeypatch.chdir(tmp_path)
    track_metrics.save_comparison_metrics(metrics, "2026-09-25")
    saved = json.loads(
        (tmp_path / "logs/comparison_metrics_2026-09-25.json").read_text()
    )
    assert saved["comparison"]["as_of_date"] == "2026-09-18"
    with (tmp_path / "logs/comparison_metrics_latest.csv").open(newline="") as handle:
        rows = {row["account"]: row for row in csv.DictReader(handle)}
    assert rows["FinRL"]["weekly_return"] == "0.04"
    assert rows["FinRL"]["comparison_status"] == "historical"
    assert rows["FinRL"]["comparison_weekly_return"] == "0.01"
    assert rows["RL"]["as_of_date"] == "2026-09-18"
    assert rows["QQQ"]["comparison_as_of_date"] == "2026-09-18"
    conn.close()


def test_comparison_unavailable_without_a_shared_benchmark_date():
    conn = comparison_db()
    conn.execute("DELETE FROM benchmark_prices WHERE price_date = '2026-09-18'")
    metrics = track_metrics.calculate_comparison_metrics(conn, "2026-09-25")
    assert metrics["comparison"] == {
        "status": "unavailable",
        "as_of_date": None,
        "weekly_returns": {},
    }
    conn.close()


def test_comparison_unavailable_without_snapshots():
    conn = comparison_db()
    conn.execute("DELETE FROM weekly_snapshot")
    metrics = track_metrics.calculate_comparison_metrics(conn, "2026-09-25")
    assert metrics["comparison"]["status"] == "unavailable"
    assert metrics["comparison"]["as_of_date"] is None
    conn.close()


def test_comparison_becomes_current_only_with_all_five_returns():
    conn = comparison_db()
    conn.execute(
        "INSERT INTO weekly_snapshot VALUES (?, ?, 100, 10, ?, 0, NULL, NULL)",
        ("2026-09-25", "RL", 0.06),
    )
    metrics = track_metrics.calculate_comparison_metrics(conn, "2026-09-25")
    assert metrics["comparison"]["status"] == "current"
    assert metrics["comparison"]["as_of_date"] == "2026-09-25"
    assert metrics["comparison"]["weekly_returns"]["RL"] == 0.06
    conn.close()


def test_unrelated_account_does_not_narrow_shared_calendar():
    conn = comparison_db()
    conn.execute(
        "INSERT INTO weekly_snapshot VALUES (?, ?, 100, 10, ?, 0, NULL, NULL)",
        ("2026-09-25", "other", 0.09),
    )
    metrics = track_metrics.calculate_comparison_metrics(conn, "2026-09-25")
    assert metrics["comparison"]["as_of_date"] == "2026-09-18"
    assert "other" not in metrics["comparison"]["weekly_returns"]
    conn.close()


def test_paper_account_names_reach_strategy_runner(monkeypatch):
    target = Mock(return_value={"A": 1.0})
    monkeypatch.setattr(run_paper_trading, "get_target_weights", target)
    monkeypatch.setattr(run_paper_trading, "get_executor_for_account", Mock())
    monkeypatch.setattr(run_paper_trading, "allows_cash_fallback", lambda _: False)
    monkeypatch.setattr(
        run_paper_trading,
        "validate_pre_trade",
        lambda *a, **k: (True, None, None, None),
    )
    monkeypatch.setattr(run_paper_trading, "save_validation_result", Mock())
    monkeypatch.setattr(run_paper_trading, "dry_run_summary", Mock())
    for name in ("FinRL", "AR"):
        run_paper_trading.run_account(
            {"name": name, "config": f"{name}.yaml"}, "2026-09-25", dry_run=True
        )
    assert [call.kwargs["account_name"] for call in target.call_args_list] == [
        "FinRL",
        "AR",
    ]


def test_account_audit_references_contain_their_own_targets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class Preprocessor:
        def __init__(self, config):
            pass

        def load_and_prepare(self, data_dir=None):
            pass

        def get_data_as_of(self, day):
            return {
                "SPY": pd.DataFrame(
                    {"close": [100]}, index=pd.DatetimeIndex([day])
                )
            }

    class Audit:
        group_strength = {}
        portfolio = {}

        def __init__(self, target):
            self.target = target

        def to_json(self, path):
            Path(path).write_text(json.dumps({"portfolio": {"weights": self.target}}))

    class Engine:
        def __init__(self, config, data_preprocessor):
            self.config = config

        def get_config(self):
            return SimpleNamespace(paths=SimpleNamespace(audit_dir="audit"))

        def run(self, price_data, as_of_date):
            target = {"FIN": 1.0} if self.config == "FinRL.yaml" else {"BASE": 1.0}
            weights = SimpleNamespace(
                weights=target,
                cash_weight=0.0,
                regime_state="risk_on",
                active_groups=[],
                get_invested_weight=lambda: 1.0,
            )
            return weights, Audit(target)

    monkeypatch.setattr(strategy_runner, "DataPreprocessor", Preprocessor)
    monkeypatch.setattr(strategy_runner, "AdaptiveRotationEngine", Engine)
    monkeypatch.setattr(
        "src.strategies.adaptive_rotation.config_loader.load_config",
        lambda _: object(),
    )
    for name, expected in (("FinRL", {"FIN": 1.0}), ("AR", {"BASE": 1.0})):
        output = tmp_path / f"{name}.json"
        strategy_runner.run_single_date(
            f"{name}.yaml",
            "2026-09-25",
            json_output_path=str(output),
            audit_suffix=name,
        )
        reference = json.loads(output.read_text())["audit_file_path"]
        assert Path(reference).name == f"audit_2026-09-25_{name}.json"
        assert (
            json.loads(Path(reference).read_text())["portfolio"]["weights"] == expected
        )
