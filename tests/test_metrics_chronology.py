"""Chronological capture and report cutoffs using disposable SQLite fixtures."""

import json
import sqlite3

import pytest

import track_metrics


@pytest.fixture
def database(tmp_path):
    conn = sqlite3.connect(tmp_path / "metrics.db")
    track_metrics.init_db(conn)
    conn.execute("""
        CREATE TABLE strategy_decisions (
            run_date TEXT, account_name TEXT, fallback_status INTEGER,
            submitted_orders TEXT
        )
    """)
    yield conn
    conn.close()


def capture(conn, day, value, *, account="FinRL", spy_close=None):
    track_metrics.record_snapshot(
        conn,
        day,
        {"name": account, "config": "fixture.yaml"},
        {
            "portfolio_value": value,
            "cash": value * 0.6,
            "equity": value,
            "positions": [
                {"symbol": "A", "actual_weight": 0.4, "market_value": value * 0.4}
            ],
        },
        {"date": day, "spy_close": spy_close, "qqq_close": spy_close},
        {"A": 0.4},
    )


def returns(conn, day):
    return conn.execute(
        "SELECT weekly_return, cumulative_return, spy_weekly_return, "
        "spy_cumulative_return FROM weekly_snapshot "
        "WHERE account='FinRL' AND snapshot_date=?",
        (day,),
    ).fetchone()


def test_identical_capture_retains_ten_percent_return(database):
    capture(database, "2026-05-01", 100, spy_close=100)
    capture(database, "2026-05-08", 110, spy_close=110)
    before = returns(database, "2026-05-08")
    capture(database, "2026-05-08", 110, spy_close=110)
    assert before == pytest.approx((0.1, 0.1, 0.1, 0.1))
    assert returns(database, "2026-05-08") == before


def test_capture_uses_strictly_previous_account_observation(database):
    capture(database, "2026-05-01", 100)
    capture(database, "2026-05-08", 900)
    capture(database, "2026-05-15", 500)
    capture(database, "2026-05-07", 2000, account="AR")
    capture(database, "2026-05-08", 110)
    assert returns(database, "2026-05-08")[:2] == pytest.approx((0.1, 0.1))


@pytest.mark.parametrize(
    "day,value,expected",
    [
        ("2026-04-24", 90, 0.0),
        ("2026-05-08", 110, 0.1),
    ],
)
def test_historical_insertion_ignores_and_preserves_later_observations(
    database, day, value, expected
):
    capture(database, "2026-05-01", 100, spy_close=100)
    capture(database, "2026-05-15", 500, spy_close=500)
    database.execute(
        "INSERT INTO strategy_decisions VALUES (?, ?, ?, ?)",
        ("2026-05-15", "FinRL", 1, json.dumps([{"id": "observed-order"}])),
    )
    observed = database.execute(
        "SELECT * FROM weekly_snapshot WHERE snapshot_date='2026-05-15'"
    ).fetchall()
    decisions = database.execute("SELECT * FROM strategy_decisions").fetchall()
    capture(database, day, value, spy_close=value)
    actual = returns(database, day)
    assert actual[:2] == pytest.approx((expected, expected))
    assert actual[3] == pytest.approx(expected)
    assert (
        database.execute(
            "SELECT * FROM weekly_snapshot WHERE snapshot_date='2026-05-15'"
        ).fetchall()
        == observed
    )
    assert database.execute("SELECT * FROM strategy_decisions").fetchall() == decisions


def test_first_capture_is_zero_and_repeatable_even_when_valuation_changes(database):
    capture(database, "2026-05-01", 100, spy_close=100)
    capture(database, "2026-05-01", 120, spy_close=120)
    assert returns(database, "2026-05-01") == (0.0, 0.0, None, 0.0)


@pytest.mark.parametrize("future_source", ["weights", "decisions", "all"])
def test_future_rows_cannot_change_an_older_comparison(database, future_source):
    capture(database, "2026-05-01", 100, spy_close=100)
    capture(database, "2026-05-08", 110, spy_close=110)
    database.execute(
        "INSERT INTO strategy_decisions VALUES ('2026-05-08', 'FinRL', 0, '[]')"
    )
    before = track_metrics.calculate_comparison_metrics(database, "2026-05-08")
    if future_source in ("weights", "all"):
        database.execute(
            "INSERT INTO weekly_weights "
            "(snapshot_date, account, symbol, actual_weight, target_weight) "
            "VALUES ('2026-05-15', 'FinRL', 'FUTURE', 1.0, 0.0)"
        )
    if future_source in ("decisions", "all"):
        database.execute(
            "INSERT INTO strategy_decisions "
            "VALUES ('2026-05-15', 'FinRL', 1, '[{\"id\":\"future-order\"}]')"
        )
    if future_source == "all":
        capture(database, "2026-05-15", 500, spy_close=500)
    after = track_metrics.calculate_comparison_metrics(database, "2026-05-08")
    assert after == before


def test_future_only_decision_does_not_disable_historical_fallback_inference(database):
    capture(database, "2026-05-01", 100)
    database.execute("DELETE FROM weekly_weights")
    for symbol in ("SPY", "A", "B", "C", "D"):
        database.execute(
            "INSERT INTO weekly_weights "
            "(snapshot_date, account, symbol, actual_weight, target_weight) "
            "VALUES ('2026-05-01', 'FinRL', ?, 0.2, 0.2)",
            (symbol,),
        )
    before = track_metrics.calculate_comparison_metrics(database, "2026-05-01")
    assert before["accounts"]["FinRL"]["weeks_in_fallback"] == 1
    database.execute(
        "INSERT INTO strategy_decisions VALUES ('2026-05-08', 'FinRL', 0, '[]')"
    )
    assert track_metrics.calculate_comparison_metrics(database, "2026-05-01") == before
