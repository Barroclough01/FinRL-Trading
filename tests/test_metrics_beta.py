"""Beta normalization through the public calculator and disposable reports."""

import csv
import json
import sqlite3
from datetime import date, timedelta

import numpy as np
import pytest

import track_metrics


def calculate_beta_metrics(account_returns, benchmark_returns):
    """Build synthetic observations without reading runtime or broker data."""
    with sqlite3.connect(":memory:") as conn:
        track_metrics.init_db(conn)
        start = date(2026, 1, 2)
        close = 100.0
        conn.execute(
            "INSERT INTO benchmark_prices VALUES (?, ?, ?)",
            (start.isoformat(), close, close),
        )
        for i, (account_return, benchmark_return) in enumerate(
            zip(account_returns, benchmark_returns), start=1
        ):
            day = (start + timedelta(weeks=i)).isoformat()
            close *= 1 + benchmark_return
            conn.execute(
                "INSERT INTO benchmark_prices VALUES (?, ?, ?)", (day, close, close)
            )
            conn.execute(
                "INSERT INTO weekly_snapshot "
                "(snapshot_date, account, portfolio_value, cash, weekly_return, "
                "cumulative_return) VALUES (?, 'FinRL', 100, 10, ?, 0)",
                (day, account_return),
            )
        return track_metrics.calculate_comparison_metrics(conn, day)


@pytest.mark.parametrize(
    "benchmark_returns,scale",
    [
        ([0.125, -0.25], 1.0),
        ([0.125, -0.25, 0.5], 1.0),
        ([0.125, -0.25, 0.5, 0.0, -0.125], 1.0),
        ([0.125, -0.25, 0.5], 2.0),
        ([0.125, -0.25, 0.5], -1.0),
    ],
)
def test_beta_is_regression_slope_without_sample_inflation(benchmark_returns, scale):
    # An intercept must not affect beta; expected slopes come from the fixture,
    # independently of NumPy covariance/variance implementations.
    account_returns = [scale * r + 0.03125 for r in benchmark_returns]
    metrics = calculate_beta_metrics(account_returns, benchmark_returns)
    stats = metrics["accounts"]["FinRL"]
    for symbol in ("spy", "qqq"):
        assert stats[f"beta_{symbol}"] == pytest.approx(scale)
        expected_error = np.std(np.subtract(account_returns, benchmark_returns))
        assert stats[f"tracking_error_{symbol}"] == pytest.approx(
            expected_error * np.sqrt(52), abs=1e-12
        )


@pytest.mark.parametrize(
    "account_returns,benchmark_returns",
    [
        ([0.125, -0.25, 0.5], [0.0, 0.0, 0.0]),
        ([0.0, 0.0, 0.0], [0.125, -0.25, 0.5]),
        ([0.125], [0.125]),
        ([None, 0.125], [0.125, -0.25]),
        ([None], [0.125]),
    ],
)
def test_constant_or_insufficient_pairs_keep_zero_beta(
    account_returns, benchmark_returns
):
    metrics = calculate_beta_metrics(account_returns, benchmark_returns)
    stats = metrics["accounts"]["FinRL"]
    assert stats["beta_spy"] == 0.0
    assert stats["beta_qqq"] == 0.0


def test_corrected_beta_reaches_json_and_csv_without_changing_other_metrics(tmp_path):
    returns = [0.125, -0.25, 0.5]
    metrics = calculate_beta_metrics(returns, returns)
    stats = metrics["accounts"]["FinRL"]
    assert stats["weekly_return"] == 0.5
    assert stats["cash_exposure"] == pytest.approx(0.1)
    assert stats["volatility"] == pytest.approx(np.std(returns) * np.sqrt(52))
    assert stats["tracking_error_spy"] == 0.0
    assert stats["tracking_error_qqq"] == 0.0
    track_metrics.save_comparison_metrics(metrics, metrics["date"])
    saved = json.loads(
        (tmp_path / f"logs/comparison_metrics_{metrics['date']}.json").read_text()
    )
    with (tmp_path / "logs/comparison_metrics_latest.csv").open(newline="") as handle:
        rows = {row["account"]: row for row in csv.DictReader(handle)}
    for symbol in ("spy", "qqq"):
        assert saved["accounts"]["FinRL"][f"beta_{symbol}"] == pytest.approx(1.0)
        assert float(rows["FinRL"][f"beta_{symbol}"]) == pytest.approx(1.0)
