"""Missing returns stay absent from benchmark pairs and shared comparisons."""

import csv
import json
import sqlite3
from datetime import date, timedelta

import numpy as np
import pytest

import track_metrics


@pytest.fixture
def calculate():
    def run(account_returns):
        with sqlite3.connect(":memory:") as conn:
            track_metrics.init_db(conn)
            start = date(2026, 1, 2)
            closes = {"SPY": 100.0, "QQQ": 100.0}
            conn.execute(
                "INSERT INTO benchmark_prices VALUES (?, ?, ?)",
                (start.isoformat(), closes["SPY"], closes["QQQ"]),
            )
            for i, account_return in enumerate(account_returns):
                day = (start + timedelta(weeks=i + 1)).isoformat()
                benchmark_return = [0.125, -0.25, 0.5][i]
                closes["SPY"] *= 1 + benchmark_return
                closes["QQQ"] *= 1 - benchmark_return
                conn.execute(
                    "INSERT INTO benchmark_prices VALUES (?, ?, ?)",
                    (day, closes["SPY"], closes["QQQ"]),
                )
                for account in ("FinRL", "AR", "RL"):
                    value = account_return if account == "FinRL" else benchmark_return
                    conn.execute(
                        "INSERT INTO weekly_snapshot "
                        "(snapshot_date, account, portfolio_value, cash, "
                        "weekly_return, cumulative_return) "
                        "VALUES (?, ?, 100, 10, ?, 0)",
                        (day, account, value),
                    )
            return track_metrics.calculate_comparison_metrics(conn, day)

    return run


@pytest.mark.parametrize("count", [2, 3])
def test_all_null_returns_keep_insufficient_pair_defaults(calculate, count):
    metrics = calculate([None] * count)
    stats = metrics["accounts"]["FinRL"]
    # These are existing insufficient-data defaults, not measured zero exposure.
    for symbol in ("spy", "qqq"):
        assert stats[f"beta_{symbol}"] == 0.0
        assert stats[f"tracking_error_{symbol}"] == 0.0
    assert metrics["comparison"] == {
        "status": "unavailable", "as_of_date": None, "weekly_returns": {}
    }
    # Other accounts must continue to compute from their observed returns.
    assert metrics["accounts"]["AR"]["beta_spy"] == pytest.approx(1.0)
    assert metrics["accounts"]["RL"]["beta_qqq"] == pytest.approx(-1.0)


@pytest.mark.parametrize("returns", [[None, None, 0.5], [0.125, None, None]])
def test_one_valid_pair_keeps_defaults_and_only_real_comparison_date(
    calculate, returns
):
    metrics = calculate(returns)
    stats = metrics["accounts"]["FinRL"]
    for symbol in ("spy", "qqq"):
        assert stats[f"beta_{symbol}"] == 0.0
        assert stats[f"tracking_error_{symbol}"] == 0.0
    observed_index = next(i for i, value in enumerate(returns) if value is not None)
    expected_date = (
        date(2026, 1, 2) + timedelta(weeks=observed_index + 1)
    ).isoformat()
    comparison = metrics["comparison"]
    assert comparison["as_of_date"] == expected_date
    assert comparison["status"] == (
        "current" if expected_date == metrics["date"] else "historical"
    )
    assert comparison["weekly_returns"]["FinRL"] == returns[observed_index]


@pytest.mark.parametrize("returns", [[0.25, None, 1.0], [0.25, -0.5, 1.0]])
def test_partial_and_normal_returns_use_only_observed_pairs(calculate, returns):
    metrics = calculate(returns)
    stats = metrics["accounts"]["FinRL"]
    observed = [value for value in returns if value is not None]
    assert stats["beta_spy"] == pytest.approx(2.0)
    assert stats["beta_qqq"] == pytest.approx(-2.0)
    assert stats["tracking_error_spy"] == pytest.approx(
        np.std([value / 2 for value in observed]) * np.sqrt(52)
    )
    assert stats["tracking_error_qqq"] == pytest.approx(
        np.std([value * 1.5 for value in observed]) * np.sqrt(52)
    )
    assert metrics["comparison"]["weekly_returns"]["FinRL"] == 1.0


def test_missing_returns_export_without_inventing_shared_observations(
    calculate, tmp_path
):
    metrics = calculate([None, None, None])
    track_metrics.save_comparison_metrics(metrics, metrics["date"])
    saved = json.loads(
        (tmp_path / f"logs/comparison_metrics_{metrics['date']}.json").read_text()
    )
    assert saved == metrics
    with (tmp_path / "logs/comparison_metrics_latest.csv").open(newline="") as handle:
        rows = {row["account"]: row for row in csv.DictReader(handle)}
    assert set(rows) == {"FinRL", "AR", "RL", "SPY", "QQQ"}
    for row in rows.values():
        assert row["comparison_status"] == "unavailable"
        assert row["comparison_as_of_date"] == ""
        assert row["comparison_weekly_return"] == ""
    for symbol in ("spy", "qqq"):
        assert float(rows["FinRL"][f"beta_{symbol}"]) == 0.0
        assert float(rows["FinRL"][f"tracking_error_{symbol}"]) == 0.0
