"""Invalid evidence refuses publication without inventing replacement statistics."""

import copy
import csv
import json
import sqlite3
import sys

import pytest

import track_metrics

DAY = "2026-01-02"


@pytest.fixture
def database():
    with sqlite3.connect(":memory:") as conn:
        track_metrics.init_db(conn)
        conn.execute(
            "INSERT INTO weekly_snapshot "
            "(snapshot_date,account,portfolio_value,cash,weekly_return) "
            "VALUES (?, 'FinRL', 100, 10, 0)",
            (DAY,),
        )
        yield conn


def test_zero_equity_capture_is_retained_but_undefined_metrics_are_refused(database):
    track_metrics.record_snapshot(
        database,
        DAY,
        {"name": "FinRL"},
        {"portfolio_value": 0, "cash": 0, "equity": 0, "positions": []},
        {},
        {},
    )
    before = database.execute("SELECT * FROM weekly_snapshot").fetchall()
    with pytest.raises(ValueError, match="Undefined cumulative_return") as error:
        track_metrics.calculate_comparison_metrics(database, DAY)
    assert "account=FinRL" in str(error.value)
    assert f"date={DAY}" in str(error.value)
    assert "portfolio_value=0" in str(error.value)
    assert database.execute("SELECT * FROM weekly_snapshot").fetchall() == before


@pytest.mark.parametrize("values,expected", [([0], -1.0), ([0, 120], 0.2)])
def test_positive_basis_can_reach_or_pass_through_zero(database, values, expected):
    for index, value in enumerate(values, start=3):
        database.execute(
            "INSERT INTO weekly_snapshot "
            "(snapshot_date,account,portfolio_value,cash,weekly_return) "
            "VALUES (?, 'FinRL', ?, 0, 0)",
            (f"2026-01-{index:02d}", value),
        )
    metrics = track_metrics.calculate_comparison_metrics(database, "2026-01-05")
    stats = metrics["accounts"]["FinRL"]
    assert stats["cumulative_return"] == pytest.approx(expected)
    assert stats["max_drawdown"] == -1.0
    # Preserve the existing zero cash-weight sentinel on nonpositive NAV rows.
    assert stats["cash_exposure"] == pytest.approx(0.1 / (len(values) + 1))


@pytest.mark.parametrize("field", ["portfolio_value", "cash"])
@pytest.mark.parametrize("value", [None, float("inf"), -float("inf"), "bad"])
def test_invalid_required_snapshot_number_has_account_date_field(
    database, field, value
):
    database.execute(f"UPDATE weekly_snapshot SET {field}=?", (value,))
    with pytest.raises(ValueError, match="Invalid metric input") as error:
        track_metrics.calculate_comparison_metrics(database, DAY)
    assert all(text in str(error.value) for text in ["account=FinRL", DAY, field])


@pytest.mark.parametrize("field", ["weekly_return", "cumulative_return"])
@pytest.mark.parametrize("value", [float("inf"), -float("inf"), "bad"])
def test_invalid_optional_snapshot_number_is_not_silently_omitted(
    database, field, value
):
    database.execute(f"UPDATE weekly_snapshot SET {field}=?", (value,))
    with pytest.raises(ValueError, match="Invalid metric input") as error:
        track_metrics.calculate_comparison_metrics(database, DAY)
    assert all(text in str(error.value) for text in ["account=FinRL", DAY, field])


@pytest.mark.parametrize("field", ["actual_weight", "target_weight"])
@pytest.mark.parametrize("value", [float("inf"), -float("inf"), "bad"])
def test_invalid_weight_has_account_date_symbol_field(database, field, value):
    database.execute(
        "INSERT INTO weekly_weights "
        "(snapshot_date,account,symbol,actual_weight,target_weight) "
        "VALUES (?, 'FinRL', 'A', 0.5, 0.5)",
        (DAY,),
    )
    database.execute(f"UPDATE weekly_weights SET {field}=?", (value,))
    with pytest.raises(ValueError, match="Invalid metric input") as error:
        track_metrics.calculate_comparison_metrics(database, DAY)
    assert all(
        text in str(error.value) for text in ["account=FinRL", DAY, "symbol=A", field]
    )


@pytest.mark.parametrize("symbol", ["spy", "qqq"])
@pytest.mark.parametrize("value", [float("inf"), -float("inf"), "bad"])
def test_nonfinite_benchmark_is_refused_without_changing_missing_conventions(
    database, symbol, value
):
    database.execute("INSERT INTO benchmark_prices VALUES (?, 100, 100)", (DAY,))
    database.execute(f"UPDATE benchmark_prices SET {symbol}_close=?", (value,))
    with pytest.raises(ValueError, match="Invalid metric input") as error:
        track_metrics.calculate_comparison_metrics(database, DAY)
    assert all(text in str(error.value) for text in [symbol.upper(), DAY, "close"])


def test_derived_benchmark_overflow_has_source_context(database):
    database.executemany(
        "INSERT INTO benchmark_prices VALUES (?, ?, 100)",
        [(DAY, 1e-300), ("2026-01-03", 1e300)],
    )
    with pytest.raises(ValueError, match="benchmark=SPY date=2026-01-03 weekly_return"):
        track_metrics.calculate_comparison_metrics(database, "2026-01-03")


def test_finite_input_statistical_overflow_is_contextualized(database):
    database.execute("UPDATE weekly_snapshot SET weekly_return=1e308")
    database.execute(
        "INSERT INTO weekly_snapshot "
        "(snapshot_date,account,portfolio_value,cash,weekly_return) "
        "VALUES ('2026-01-03', 'FinRL', 100, 10, -1e308)"
    )
    with pytest.raises(ValueError, match="account=FinRL date=2026-01-03"):
        track_metrics.calculate_comparison_metrics(database, "2026-01-03")


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("location", ["statistic", "shared", "extra"])
def test_invalid_export_preserves_complete_last_good_set(
    database, tmp_path, existing, value, location
):
    metrics = track_metrics.calculate_comparison_metrics(database, DAY)
    json_path = tmp_path / f"logs/comparison_metrics_{DAY}.json"
    csv_path = tmp_path / "logs/comparison_metrics_latest.csv"
    if existing:
        track_metrics.save_comparison_metrics(metrics, DAY)
    before = {
        path: path.read_bytes() for path in (json_path, csv_path) if path.exists()
    }
    invalid = copy.deepcopy(metrics)
    if location == "statistic":
        invalid["accounts"]["FinRL"]["weekly_return"] = value
    elif location == "shared":
        invalid["comparison"]["weekly_returns"]["FinRL"] = value
    else:
        invalid["extra"] = [{"hidden": value}]
    with pytest.raises(ValueError, match="report date=2026-01-02"):
        track_metrics.save_comparison_metrics(invalid, DAY)
    after = {path: path.read_bytes() for path in (json_path, csv_path) if path.exists()}
    assert after == before
    assert not list((tmp_path / "logs").glob("*.tmp"))


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("malformed", ["csv_structure", "json_type"])
def test_serialization_failure_cannot_publish_only_json(
    database, tmp_path, existing, malformed
):
    metrics = track_metrics.calculate_comparison_metrics(database, DAY)
    paths = [
        tmp_path / f"logs/comparison_metrics_{DAY}.json",
        tmp_path / "logs/comparison_metrics_latest.csv",
    ]
    if existing:
        track_metrics.save_comparison_metrics(metrics, DAY)
    before = {path: path.read_bytes() for path in paths if path.exists()}
    if malformed == "csv_structure":
        metrics["accounts"]["FinRL"] = []
    else:
        metrics["extra"] = object()
    with pytest.raises((AttributeError, TypeError)):
        track_metrics.save_comparison_metrics(metrics, DAY)
    assert {path: path.read_bytes() for path in paths if path.exists()} == before


def test_valid_export_remains_strict_json_and_preserves_csv(database, tmp_path):
    metrics = track_metrics.calculate_comparison_metrics(database, DAY)
    track_metrics.save_comparison_metrics(metrics, DAY)

    def reject_constant(value):
        pytest.fail(f"Nonstandard JSON constant: {value}")

    saved = json.loads(
        (tmp_path / f"logs/comparison_metrics_{DAY}.json").read_text(),
        parse_constant=reject_constant,
    )
    assert saved == metrics
    with (tmp_path / "logs/comparison_metrics_latest.csv").open(newline="") as handle:
        rows = {row["account"]: row for row in csv.DictReader(handle)}
    assert set(rows) == {"FinRL", "SPY", "QQQ"}
    assert float(rows["FinRL"]["cash_exposure"]) == 0.1


@pytest.mark.parametrize("existing", [False, True])
def test_invalid_report_only_run_preserves_dashboard_and_reports(
    tmp_path, monkeypatch, existing, caplog
):
    database_path = tmp_path / "source.db"
    with sqlite3.connect(database_path) as conn:
        track_metrics.init_db(conn)
        track_metrics.record_snapshot(
            conn,
            DAY,
            {"name": "FinRL"},
            {"portfolio_value": 0, "cash": 0, "equity": 0, "positions": []},
            {},
            {},
        )
    paths = [
        tmp_path / "logs/dashboard.html",
        tmp_path / f"logs/comparison_metrics_{DAY}.json",
        tmp_path / "logs/comparison_metrics_latest.csv",
    ]
    if existing:
        for path in paths:
            path.write_text("last good evidence")
    before = {path: path.read_bytes() for path in paths if path.exists()}
    monkeypatch.setattr(track_metrics, "DB_PATH", database_path)
    monkeypatch.setattr(
        sys, "argv", ["track_metrics.py", "--report-only", "--date", DAY]
    )
    with pytest.raises(SystemExit) as error:
        track_metrics.main()
    assert error.value.code == 1
    assert "Undefined cumulative_return" in caplog.text
    assert "Failed to record snapshot" not in caplog.text
    assert {path: path.read_bytes() for path in paths if path.exists()} == before
