import json
import math
import os
import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Ensure project root is in path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import src.data.trading_calendar
from run_paper_trading import (
    get_ar_weights,
    get_rl_candidate_weights,
    get_target_weights,
    reconcile_post_trade,
    run_post_run_sanity_checks,
    validate_pre_trade,
)
import track_metrics
from src.strategies.rl_candidate_strategy import load_rl_candidate_weights
from refresh_fmp_daily import get_last_csv_date


# ---------------------------------------------------------------------------
# Priority 4 Tests
# ---------------------------------------------------------------------------


@patch("subprocess.run")
def test_structured_strategy_output_parsing(mock_run):
    """Test that structured strategy output can be parsed and validated."""
    dummy_json = {
        "target_weights": {"SATS": 0.25, "MCHP": 0.25, "ON": 0.25, "ORCL": 0.25},
        "cash_weight": 0.0,
        "regime_state": "risk_on",
        "active_groups": ["group_a_growth_tech"],
        "ranked_groups": ["group_a_growth_tech"],
        "fallback_status": False,
        "audit_file_path": "/dummy/audit.json",
    }

    def side_effect(*args, **kwargs):
        cmd = args[0]
        json_path_idx = cmd.index("--json-output") + 1
        json_path = cmd[json_path_idx]
        with open(json_path, "w") as f:
            json.dump(dummy_json, f)
        res = MagicMock()
        res.returncode = 0
        return res

    mock_run.side_effect = side_effect

    weights = get_ar_weights(
        "src/strategies/AdaptiveRotationConf_v1.2.2.yaml", "2026-05-31"
    )
    assert weights == {"SATS": 0.25, "MCHP": 0.25, "ON": 0.25, "ORCL": 0.25}


@patch("run_paper_trading.get_rl_candidate_weights")
def test_get_target_weights_prefers_rl_candidate_csv(mock_get_rl_candidate_weights):
    """The paper-trading helper should route CSV RL exports through the candidate path."""
    mock_get_rl_candidate_weights.return_value = {"A": 0.6, "B": 0.4}

    weights = get_target_weights("results/drl_weight.csv", "2026-05-31")

    assert weights == {"A": 0.6, "B": 0.4}
    mock_get_rl_candidate_weights.assert_called_once_with(
        "results/drl_weight.csv", "2026-05-31"
    )


@patch("subprocess.run")
def test_rl_candidate_weights_dispatch(mock_run):
    """RL candidate weights should be readable through the paper-trading helper."""
    dummy_json = {
        "target_weights": {"A": 0.6, "B": 0.4},
        "cash_weight": 0.0,
        "regime_state": "rl_candidate",
        "active_groups": [],
        "ranked_groups": [],
        "fallback_status": False,
        "audit_file_path": "/dummy/audit.json",
    }

    def side_effect(*args, **kwargs):
        cmd = args[0]
        json_path_idx = cmd.index("--json-output") + 1
        json_path = cmd[json_path_idx]
        Path(json_path).write_text(json.dumps(dummy_json), encoding="utf-8")
        res = MagicMock()
        res.returncode = 0
        return res

    mock_run.side_effect = side_effect

    weights = get_rl_candidate_weights("results/drl_weight.csv", "2026-05-31")

    assert weights == {"A": 0.6, "B": 0.4}


def test_load_rl_candidate_weights_filters_unavailable_symbols(tmp_path):
    """RL candidate weights should only keep symbols with local price data."""
    weights_path = tmp_path / "rl_weights.csv"
    weights_path.write_text(
        "trade_date,gvkey,weights\n2026-06-12,A,0.50\n2026-06-12,AAPL,0.50\n",
        encoding="utf-8",
    )
    (tmp_path / "AAPL_daily.csv").write_text(
        "date,close\n2026-06-12,1\n", encoding="utf-8"
    )

    weights = load_rl_candidate_weights(str(weights_path), data_dir=tmp_path)

    assert weights == {"AAPL": 1.0}


def test_invalid_target_weights_fail_validation():
    """Test that invalid target weights fail validation."""
    account = {"name": "test_acc", "config": "dummy_config.yaml"}
    executor = MagicMock()

    # 1. Empty weights
    valid, failed_rule, error_msg, suggested_fix = validate_pre_trade(
        account, "2026-05-31", {}, executor
    )
    assert not valid
    assert failed_rule == "target weights are nonempty"
    assert "empty" in error_msg
    assert suggested_fix is not None

    # 2. Non-finite weights
    valid, failed_rule, error_msg, _ = validate_pre_trade(
        account, "2026-05-31", {"AAPL": float("nan")}, executor
    )
    assert not valid
    assert failed_rule == "all weights are finite numbers"

    # 3. Negative weights
    valid, failed_rule, error_msg, _ = validate_pre_trade(
        account, "2026-05-31", {"AAPL": -0.15}, executor
    )
    assert not valid
    assert failed_rule == "no negative weights unless shorting is explicitly supported"

    # 4. Sum exceeding tolerance
    valid, failed_rule, error_msg, _ = validate_pre_trade(
        account, "2026-05-31", {"AAPL": 0.6, "MSFT": 0.5}, executor
    )
    assert not valid
    assert failed_rule == "sum of target weights is within tolerance"

    # 5. Single symbol exceeding max weight
    valid, failed_rule, error_msg, _ = validate_pre_trade(
        account, "2026-05-31", {"AAPL": 0.6}, executor
    )
    assert not valid
    assert failed_rule == "no single symbol exceeds configured maximum weight"


@patch("pathlib.Path.exists")
@patch("pandas.read_csv")
@patch("src.data.trading_calendar.is_trading_day")
def test_negative_cash_passes_when_equity_positive(
    mock_is_trading_day, mock_read_csv, mock_path_exists
):
    """Negative cash on a margin paper account should not block validation."""
    account = {"name": "AR", "config": "dummy_config.yaml"}
    executor = MagicMock()
    executor.alpaca.get_account_info.return_value = {
        "cash": -383.70,
        "equity": 988233.87,
    }
    mock_is_trading_day.return_value = True
    mock_path_exists.return_value = True
    mock_read_csv.return_value = pd.DataFrame(
        {"date": ["2026-06-05"], "close": [500.0]}
    )

    valid, failed_rule, error_msg, _ = validate_pre_trade(
        account, "2026-06-05", {"SPY": 0.14, "QQQ": 0.14}, executor
    )
    assert valid
    assert failed_rule is None
    assert error_msg is None


@patch("pathlib.Path.exists")
@patch("pandas.read_csv")
@patch("src.data.trading_calendar.is_trading_day")
def test_zero_equity_fails_validation(
    mock_is_trading_day, mock_read_csv, mock_path_exists
):
    """Zero or negative equity should still block validation."""
    account = {"name": "AR", "config": "dummy_config.yaml"}
    executor = MagicMock()
    executor.alpaca.get_account_info.return_value = {"cash": 0, "equity": 0}
    mock_is_trading_day.return_value = True
    mock_path_exists.return_value = True
    mock_read_csv.return_value = pd.DataFrame(
        {"date": ["2026-06-05"], "close": [500.0]}
    )

    valid, failed_rule, error_msg, _ = validate_pre_trade(
        account, "2026-06-05", {"SPY": 0.14}, executor
    )
    assert not valid
    assert failed_rule == "account cash/equity can be read"
    assert "invalid equity" in error_msg


@patch("pathlib.Path.exists")
@patch("pandas.read_csv")
@patch("src.data.trading_calendar.is_trading_day")
def test_stale_prices_fail_validation(
    mock_is_trading_day, mock_read_csv, mock_path_exists
):
    """Test that stale prices fail validation checks."""
    account = {"name": "test_acc", "config": "dummy_config.yaml"}
    executor = MagicMock()
    executor.alpaca.get_account_info.return_value = {"cash": 100000, "equity": 100000}
    mock_is_trading_day.return_value = True

    # Mock files exist
    mock_path_exists.return_value = True

    # Mock stale data (last date is 2026-05-01, but run date is 2026-05-31)
    mock_read_csv.return_value = pd.DataFrame(
        {"date": ["2026-05-01"], "close": [150.0]}
    )

    valid, failed_rule, error_msg, _ = validate_pre_trade(
        account, "2026-05-31", {"AAPL": 0.4}, executor
    )
    assert not valid
    assert failed_rule == "all required symbol prices are fresh"
    assert "stale" in error_msg


@patch("track_metrics.load_accounts_from_env")
@patch("track_metrics.get_alpaca_snapshot")
@patch("track_metrics.fetch_benchmark_prices")
@patch("sqlite3.connect")
def test_metrics_fail_when_account_snapshot_fails(
    mock_connect, mock_fetch_bench, mock_get_snapshot, mock_load_accounts
):
    """Test that metrics collection fails (exits nonzero) when an account snapshot fails."""
    mock_load_accounts.return_value = [{"name": "test_acc", "config": "dummy.yaml"}]
    mock_get_snapshot.side_effect = Exception("Alpaca connection failed")
    mock_fetch_bench.return_value = {"spy_close": 400.0, "date": "2026-05-31"}

    with patch("sys.argv", ["track_metrics.py"]):
        with patch("sys.exit") as mock_exit:
            track_metrics.main()
            mock_exit.assert_called_once_with(1)


@patch("sqlite3.connect")
@patch("pathlib.Path.exists")
def test_post_run_sanity_checks_detect_missing_rows(mock_exists, mock_connect):
    """Test that post-run sanity checks detect missing rows in weekly_snapshot/weights."""
    mock_exists.return_value = True

    # Mock DB to return 0 rows
    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn
    mock_conn.execute.return_value.fetchone.return_value = (0,)

    accounts = [{"name": "test_acc"}]
    results = [{"account": "test_acc"}]
    errors = []

    failures = run_post_run_sanity_checks("2026-05-31", accounts, results, errors)
    assert len(failures) > 0
    assert any("weekly_snapshot missing" in f for f in failures)


def test_reconcile_post_trade_accepts_string_numeric_fields():
    """String-valued equity/market_value should be coerced to floats during reconciliation."""
    record = {
        "equity": "1000.0",
        "cash": "100.0",
        "target_weights": {"AAPL": 1.0},
        "post_trade_positions": [{"symbol": "AAPL", "market_value": "1000.0"}],
        "submitted_orders": [],
    }

    result = reconcile_post_trade("2026-06-12", "test_acc", record)

    assert result["reconciled_successfully"] is True
    assert result["target_vs_actual_weights"][0]["actual_weight"] == 1.0


def test_corrupted_csv_fails_refresh(tmp_path):
    """Test that empty, corrupt, or missing 'date' column CSVs produce a clear failure in get_last_csv_date."""
    # 1. Test empty file (0 bytes)
    empty_file = tmp_path / "empty_daily.csv"
    empty_file.touch()
    with pytest.raises(ValueError, match="empty.*0 bytes"):
        get_last_csv_date(empty_file)

    # 2. Test missing 'date' column
    missing_date_file = tmp_path / "missing_date_daily.csv"
    with open(missing_date_file, "w") as f:
        f.write("open,high,low,close,volume\n1,2,3,4,100\n")
    with pytest.raises(ValueError, match="missing required 'date' column"):
        get_last_csv_date(missing_date_file)

    # 3. Test corrupt CSV
    corrupt_file = tmp_path / "corrupt_daily.csv"
    with open(corrupt_file, "w") as f:
        f.write("date,open,high\n2026-05-31,1,2,3,4,5,6,7\n")
    with pytest.raises(ValueError, match="Corrupt CSV or unreadable file"):
        get_last_csv_date(corrupt_file)


def test_dashboard_spy_series_aligns_to_weekly_dates():
    """Test that generate_html_dashboard aligns SPY series and accounts correctly."""
    # Mock SQL connection
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = [
        ("2026-05-15", "test_acc", 1000000.0, 0.0, 0.0, 0.01, 0.01, "[]"),
        ("2026-05-22", "test_acc", 1010000.0, 0.01, 0.01, -0.005, 0.005, "[]"),
    ]

    output_path = MagicMock()
    track_metrics.generate_html_dashboard(conn, output_path)

    # Verify write_text was called
    assert output_path.write_text.called
    html_content = output_path.write_text.call_args[0][0]
    assert "test_acc" in html_content
    assert "2026-05-15" in html_content
    assert "2026-05-22" in html_content


def test_rl_dependency_check():
    """Test that RL pipeline dependency check fails with clear actionable message when modules are missing."""
    from src.strategies.run_rl_offline_pipeline import _check_required_modules

    with patch("importlib.import_module") as mock_import:
        # Simulate missing finrl
        mock_import.side_effect = ModuleNotFoundError()

        with pytest.raises(RuntimeError) as exc_info:
            _check_required_modules()

        assert "Missing Python dependencies for RL pipeline" in str(exc_info.value)
        assert "uv pip install" in str(exc_info.value)


def test_exception_detection_clarified_status():
    """Test that exception detection result contains clarified status for persistence and strong signal rules."""
    from src.strategies.adaptive_rotation.exception_framework import ExceptionDetector

    # Create detector
    detector = ExceptionDetector(
        z_threshold=2.5,
        lookback_weeks=4,
        min_trigger_count=2,
        strong_signal_enabled=True,
    )

    # Run with 1-point series (no history)
    import pandas as pd

    asset_zscores = {"SATS": pd.Series([1.5], index=[pd.Timestamp("2026-05-31")])}

    result = detector.detect_exceptions(asset_zscores, pd.Timestamp("2026-05-31"))
    assert result.persistence_rule_status == "disabled_due_to_missing_history"
    assert result.strong_signal_rule_status == "enabled"


def test_strategy_decision_records(tmp_path):
    """Test that strategy decision records can be written and read from SQLite and JSONL mirror."""
    from run_paper_trading import save_strategy_decision

    # Mock database path and JSONL mirror path
    db_file = tmp_path / "finrl_trading.db"
    jsonl_file = tmp_path / "strategy_decisions.jsonl"

    dummy_record = {
        "config_path": "dummy_config.yaml",
        "config_hash": "abcdef123456",
        "regime_state": "risk_on",
        "active_groups": ["group_a"],
        "ranked_groups": ["group_a", "group_b"],
        "fallback_status": False,
        "fallback_reason": None,
        "target_weights": {"AAPL": 0.5, "MSFT": 0.5},
        "pre_trade_positions": [],
        "order_plan": {"sell": [], "buy": []},
        "submitted_orders": [],
        "filled_orders": [],
        "post_trade_positions": [],
        "cash": 100000.0,
        "equity": 100000.0,
        "benchmark_snapshot": {"spy_close": 500.0},
    }

    with patch("run_paper_trading.Path") as mock_path:
        # Side effect to return our tmp_path files
        def path_side_effect(*args, **kwargs):
            val = str(args[0])
            if "finrl_trading.db" in val:
                return db_file
            if "strategy_decisions.jsonl" in val:
                return jsonl_file
            return Path(*args, **kwargs)

        mock_path.side_effect = path_side_effect

        # Save decision
        save_strategy_decision("2026-05-31", "test_acc", dummy_record)

        # Verify JSONL mirror
        assert jsonl_file.exists()
        with open(jsonl_file) as f:
            lines = f.readlines()
        assert len(lines) == 1
        saved_json = json.loads(lines[0])
        assert saved_json["account_name"] == "test_acc"
        assert saved_json["config_hash"] == "abcdef123456"
        assert saved_json["target_weights"] == {"AAPL": 0.5, "MSFT": 0.5}

        # Verify SQLite DB
        import sqlite3

        conn = sqlite3.connect(db_file)
        row = conn.execute(
            "SELECT run_date, account_name, config_hash, target_weights, cash FROM strategy_decisions"
        ).fetchone()
        assert row is not None
        assert row[0] == "2026-05-31"
        assert row[1] == "test_acc"
        assert row[2] == "abcdef123456"
        assert json.loads(row[3]) == {"AAPL": 0.5, "MSFT": 0.5}
        assert row[4] == 100000.0
        conn.close()


def test_strategy_decision_records_with_timestamps(tmp_path):
    """Test that pandas Timestamps in order records serialize to SQLite."""
    from run_paper_trading import save_strategy_decision

    db_file = tmp_path / "finrl_trading.db"
    jsonl_file = tmp_path / "strategy_decisions.jsonl"

    dummy_record = {
        "config_path": "dummy_config.yaml",
        "config_hash": "abcdef123456",
        "regime_state": "fast_risk_off",
        "active_groups": ["group_a"],
        "ranked_groups": ["group_a"],
        "fallback_status": False,
        "fallback_reason": None,
        "target_weights": {"SATS": 0.2333},
        "pre_trade_positions": [],
        "order_plan": {"sell": [], "buy": []},
        "submitted_orders": [
            {
                "order_id": "abc-123",
                "status": "accepted",
                "symbol": "SATS",
                "quantity": 751.06,
                "side": "sell",
                "submitted_at": pd.Timestamp("2026-06-05 18:08:39"),
                "filled_at": None,
            }
        ],
        "filled_orders": [],
        "post_trade_positions": [],
        "cash": -468.93,
        "equity": 1139861.42,
        "benchmark_snapshot": {"spy_close": 500.0},
    }

    with patch("run_paper_trading.Path") as mock_path:

        def path_side_effect(*args, **kwargs):
            val = str(args[0])
            if "finrl_trading.db" in val:
                return db_file
            if "strategy_decisions.jsonl" in val:
                return jsonl_file
            return Path(*args, **kwargs)

        mock_path.side_effect = path_side_effect

        save_strategy_decision("2026-06-05", "FinRL", dummy_record)

        conn = sqlite3.connect(db_file)
        row = conn.execute(
            "SELECT submitted_orders FROM strategy_decisions WHERE account_name = 'FinRL'"
        ).fetchone()
        assert row is not None
        orders = json.loads(row[0])
        assert len(orders) == 1
        assert orders[0]["symbol"] == "SATS"
        assert "2026-06-05" in orders[0]["submitted_at"]
        conn.close()


def test_weekly_comparison_metrics(tmp_path):
    """Test that weekly comparison metrics can be calculated and saved correctly."""
    from track_metrics import calculate_comparison_metrics, save_comparison_metrics

    # 1. Create a dummy SQLite DB with snapshot and weights data
    db_file = tmp_path / "finrl_trading.db"
    conn = sqlite3.connect(db_file)

    # Create tables
    conn.executescript("""
        CREATE TABLE weekly_snapshot (
            snapshot_date TEXT,
            account       TEXT,
            portfolio_value REAL,
            cash          REAL,
            weekly_return REAL,
            cumulative_return REAL,
            spy_weekly_return REAL,
            spy_cumulative_return REAL,
            UNIQUE(snapshot_date, account)
        );
        CREATE TABLE weekly_weights (
            snapshot_date TEXT,
            account       TEXT,
            symbol        TEXT,
            actual_weight REAL,
            target_weight REAL
        );
        CREATE TABLE benchmark_prices (
            price_date TEXT,
            spy_close  REAL,
            qqq_close  REAL
        );
    """)

    # Insert snapshot rows (2 weeks of data)
    conn.execute(
        "INSERT INTO weekly_snapshot VALUES ('2026-05-15', 'test_acc', 100000.0, 10000.0, 0.0, 0.0, 0.005, 0.0)"
    )
    conn.execute(
        "INSERT INTO weekly_snapshot VALUES ('2026-05-22', 'test_acc', 101000.0, 10100.0, 0.01, 0.01, 0.006, 0.005)"
    )

    # Insert weights rows
    conn.execute(
        "INSERT INTO weekly_weights VALUES ('2026-05-15', 'test_acc', 'AAPL', 0.5, 0.5)"
    )
    conn.execute(
        "INSERT INTO weekly_weights VALUES ('2026-05-22', 'test_acc', 'AAPL', 0.48, 0.5)"
    )

    # Insert benchmark prices (for QQQ returns calculation)
    conn.execute("INSERT INTO benchmark_prices VALUES ('2026-05-14', 400.0, 300.0)")
    conn.execute("INSERT INTO benchmark_prices VALUES ('2026-05-22', 402.4, 303.0)")

    conn.commit()

    # Calculate metrics
    metrics = calculate_comparison_metrics(conn, "2026-05-22")
    conn.close()

    assert metrics is not None
    assert "date" in metrics
    assert metrics["date"] == "2026-05-22"
    assert "test_acc" in metrics["accounts"]

    acc_stats = metrics["accounts"]["test_acc"]
    assert acc_stats["weekly_return"] == pytest.approx(0.01)
    assert acc_stats["cumulative_return"] == pytest.approx(0.01)
    assert acc_stats["cash_exposure"] == pytest.approx(0.10)  # 10% cash exposure
    assert acc_stats["weight_drift"] == pytest.approx(0.01)  # average of 0.0 and 0.02
    assert acc_stats["turnover"] == pytest.approx(
        0.02
    )  # |0.48 - 0.5| = 0.02 (from week 1 to week 2)

    # Save files using patch
    json_file = tmp_path / "comparison_metrics_2026-05-22.json"
    csv_file = tmp_path / "comparison_metrics_latest.csv"

    with patch("track_metrics.Path") as mock_path:

        def path_side_effect(*args, **kwargs):
            val = str(args[0])
            if "comparison_metrics_2026-05-22.json" in val:
                return json_file
            if "comparison_metrics_latest.csv" in val:
                return csv_file
            return Path(*args, **kwargs)

        mock_path.side_effect = path_side_effect

        save_comparison_metrics(metrics, "2026-05-22")

        assert json_file.exists()
        assert csv_file.exists()

        # Verify CSV content
        with open(csv_file) as f:
            lines = f.readlines()
        assert len(lines) > 1
        assert "test_acc" in lines[1]


def test_post_trade_reconciliation(tmp_path):
    """Test that post-trade reconciliation checks can identify discrepancies and save reports correctly."""
    from run_paper_trading import reconcile_post_trade, save_reconciliation_report

    recon_log_file = tmp_path / "reconciliation_2026-05-31.json"

    # 1. Test clean reconciliation (no discrepancies)
    clean_record = {
        "target_weights": {"AAPL": 0.5, "MSFT": 0.5},
        "post_trade_positions": [
            {"symbol": "AAPL", "market_value": 50000.0},
            {"symbol": "MSFT", "market_value": 50000.0},
        ],
        "submitted_orders": [
            {"symbol": "AAPL", "status": "filled"},
            {"symbol": "MSFT", "status": "filled"},
        ],
        "equity": 100000.0,
        "cash": 0.0,
    }

    result = reconcile_post_trade("2026-05-31", "clean_acc", clean_record)
    assert result["reconciled_successfully"]
    assert not result["discrepancies_found"]
    assert len(result["alerts"]) == 0
    assert result["orders_summary"]["submitted"] == 2
    assert result["orders_summary"]["failed_or_rejected"] == 0

    # 2. Test reconciliation with discrepancies (missing asset, unexpected holding, weight drift, failed order)
    discrepant_record = {
        "target_weights": {"AAPL": 0.5, "MSFT": 0.5},
        "post_trade_positions": [
            {
                "symbol": "AAPL",
                "market_value": 40000.0,
            },  # Drift = |0.4 - 0.5| = 0.10 (exceeds 2% threshold)
            {"symbol": "GOOGL", "market_value": 10000.0},  # Unexpected holding
            # MSFT is missing
        ],
        "submitted_orders": [
            {"symbol": "AAPL", "status": "filled"},
            {"symbol": "MSFT", "status": "rejected"},  # Failed order
        ],
        "equity": 100000.0,
        "cash": 50000.0,
    }

    result_fail = reconcile_post_trade("2026-05-31", "fail_acc", discrepant_record)
    assert not result_fail["reconciled_successfully"]
    assert result_fail["discrepancies_found"]
    assert len(result_fail["alerts"]) > 0

    # Check specific alerts
    alerts_str = " ".join(result_fail["alerts"])
    assert "missing" in alerts_str
    assert "Unexpected holding: GOOGL" in alerts_str
    assert "Weight drift for AAPL" in alerts_str
    assert "failed or were rejected" in alerts_str

    # Save report using patch
    with patch("run_paper_trading.Path") as mock_path:

        def path_side_effect(*args, **kwargs):
            val = str(args[0])
            if "reconciliation_2026-05-31.json" in val:
                return recon_log_file
            return Path(*args, **kwargs)

        mock_path.side_effect = path_side_effect

        save_reconciliation_report("2026-05-31", "fail_acc", result_fail)

        assert recon_log_file.exists()
        with open(recon_log_file) as f:
            saved_data = json.load(f)
        assert "fail_acc" in saved_data["accounts"]
        assert saved_data["accounts"]["fail_acc"]["discrepancies_found"]


@patch("subprocess.run")
def test_run_metrics_tracker_full_success(mock_subprocess_run):
    """Full metrics run success should not invoke report-only fallback."""
    from run_paper_trading import run_metrics_tracker

    mock_subprocess_run.return_value = MagicMock(returncode=0)

    ok, err = run_metrics_tracker("2026-06-06", project_root)

    assert ok
    assert err is None
    assert mock_subprocess_run.call_count == 1
    assert mock_subprocess_run.call_args_list[0][0][0] == [
        sys.executable,
        "track_metrics.py",
        "--date",
        "2026-06-06",
    ]


@patch("subprocess.run")
def test_run_metrics_tracker_report_only_fallback(mock_subprocess_run):
    """Failed full metrics run should fall back to --report-only."""
    from run_paper_trading import run_metrics_tracker

    mock_subprocess_run.side_effect = [
        MagicMock(returncode=1),
        MagicMock(returncode=0),
    ]

    ok, err = run_metrics_tracker("2026-06-06", project_root)

    assert ok
    assert err is not None
    assert "report-only" in err.lower() or "existing data" in err.lower()
    assert mock_subprocess_run.call_count == 2
    assert mock_subprocess_run.call_args_list[1][0][0] == [
        sys.executable,
        "track_metrics.py",
        "--report-only",
        "--date",
        "2026-06-06",
    ]


def test_discord_webhook_body_format():
    """Discord webhooks require a content field, not arbitrary JSON."""
    from run_paper_trading import format_webhook_body

    body = format_webhook_body(
        "https://discord.com/api/webhooks/123/token",
        {
            "status": "failed",
            "date": "2026-06-06",
            "accounts": ["FinRL", "AR"],
            "errors": [{"account": "AR", "error": "validation failed"}],
        },
    )
    payload = json.loads(body.decode("utf-8"))
    assert "content" in payload
    assert "Paper Trading FAILED" in payload["content"]
    assert "FinRL" in payload["content"]
    assert "AR" in payload["content"]


def test_generic_webhook_body_format():
    """Non-Discord webhooks should receive the raw JSON payload."""
    from run_paper_trading import format_webhook_body

    original = {"status": "ok", "date": "2026-06-06", "accounts": ["FinRL"]}
    body = format_webhook_body("https://example.com/hook", original)
    assert json.loads(body.decode("utf-8")) == original


@patch("subprocess.run")
def test_get_ar_weights_passes_audit_suffix(mock_run):
    """Account name should be passed as --audit-suffix to avoid audit file collisions."""
    dummy_json = {
        "target_weights": {"SPY": 0.5},
        "cash_weight": 0.5,
        "regime_state": "risk_on",
        "active_groups": [],
        "ranked_groups": [],
        "fallback_status": True,
        "audit_file_path": "/dummy/audit_2026-06-06_AR.json",
    }

    def side_effect(*args, **kwargs):
        cmd = kwargs.get("args") or args[0]
        assert "--audit-suffix" in cmd
        assert "AR" in cmd
        json_path_idx = cmd.index("--json-output") + 1
        with open(cmd[json_path_idx], "w") as f:
            json.dump(dummy_json, f)
        res = MagicMock()
        res.returncode = 0
        return res

    mock_run.side_effect = side_effect

    weights = get_ar_weights(
        "src/strategies/AdaptiveRotationConf_baseline.yaml",
        "2026-06-06",
        account_name="AR",
    )
    assert weights == {"SPY": 0.5}


@patch("subprocess.run")
def test_run_metrics_tracker_both_fail(mock_subprocess_run):
    """Metrics should fail only when both full and report-only runs fail."""
    from run_paper_trading import run_metrics_tracker

    mock_subprocess_run.side_effect = [
        MagicMock(returncode=1),
        MagicMock(returncode=1),
    ]

    ok, err = run_metrics_tracker("2026-06-06", project_root)

    assert not ok
    assert err is not None
    assert mock_subprocess_run.call_count == 2


@patch("run_paper_trading.run_metrics_tracker")
@patch("run_paper_trading.run_parity_checks")
@patch("run_paper_trading.notify_status")
@patch("run_paper_trading.load_accounts_from_env")
@patch("run_paper_trading.run_account")
def test_metrics_runs_when_all_accounts_fail(
    mock_run_account,
    mock_load,
    mock_notify,
    mock_parity,
    mock_metrics,
):
    """Metrics tracker should run even when every account fails."""
    mock_load.return_value = [
        {"name": "FinRL", "config": "dummy1.yaml"},
        {"name": "AR", "config": "dummy2.yaml"},
    ]
    mock_run_account.side_effect = RuntimeError("account failed")
    mock_metrics.return_value = (True, None)

    with patch("sys.argv", ["run_paper_trading.py", "--date", "2026-06-06"]):
        with patch("sys.exit") as mock_exit:
            from run_paper_trading import main

            main()

            mock_metrics.assert_called_once()
            mock_exit.assert_called_once_with(1)


@patch("os.getenv")
@patch("run_paper_trading.load_accounts_from_env")
@patch("run_paper_trading.run_account")
def test_production_kill_switch(mock_run_account, mock_load, mock_getenv):
    """Test that the production kill switch forces dry-run mode."""

    # Simulate TRADING_DISABLED=true
    def getenv_side_effect(key, default=None):
        if key == "TRADING_DISABLED":
            return "true"
        return default

    mock_getenv.side_effect = getenv_side_effect

    mock_load.return_value = [{"name": "test_acc", "config": "dummy.yaml"}]
    mock_run_account.return_value = {"account": "test_acc", "dry_run": True}

    # We call main with patched sys.argv, and --dry-run not set originally
    with patch("sys.argv", ["run_paper_trading.py", "--date", "2026-05-31"]):
        from run_paper_trading import main

        main()

        # Verify that run_account was called with dry_run = True
        mock_run_account.assert_called_once_with(
            {"name": "test_acc", "config": "dummy.yaml"},
            "2026-05-31",
            True,  # dry_run should be True!
        )


@patch("run_paper_trading.get_ar_weights")
@patch("pathlib.Path.exists")
def test_live_vs_replay_parity(mock_exists, mock_get_ar_weights, tmp_path):
    """Test that live-vs-replay parity checks correctly identify discrepancies."""
    from run_paper_trading import run_parity_checks

    # Mock files
    db_file = tmp_path / "finrl_trading.db"
    report_file = tmp_path / "parity_check_2026-05-31.json"

    # Mock exists to return True
    def exists_side_effect(*args, **kwargs):
        return True

    mock_exists.side_effect = exists_side_effect

    # Create SQLite tables and insert mock data
    conn = sqlite3.connect(db_file)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS strategy_decisions (
            id                     INTEGER PRIMARY KEY AUTOINCREMENT,
            run_date               TEXT NOT NULL,
            account_name           TEXT NOT NULL,
            target_weights         TEXT,
            post_trade_positions   TEXT,
            equity                 REAL,
            parity_check           TEXT
        );
        CREATE TABLE IF NOT EXISTS weekly_weights (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_date TEXT NOT NULL,
            account       TEXT NOT NULL,
            symbol        TEXT NOT NULL,
            target_weight REAL,
            actual_weight REAL
        );
    """)

    # Case 1: Perfect parity
    conn.execute(
        "INSERT INTO strategy_decisions (run_date, account_name, target_weights, post_trade_positions, equity) "
        "VALUES ('2026-05-31', 'perfect_acc', '{\"AAPL\": 0.5, \"MSFT\": 0.5}', "
        '\'[{"symbol": "AAPL", "market_value": 50000.0}, {"symbol": "MSFT", "market_value": 50000.0}]\', 100000.0)'
    )
    conn.execute(
        "INSERT INTO weekly_weights (snapshot_date, account, symbol, target_weight, actual_weight) VALUES ('2026-05-31', 'perfect_acc', 'AAPL', 0.5, 0.5)"
    )
    conn.execute(
        "INSERT INTO weekly_weights (snapshot_date, account, symbol, target_weight, actual_weight) VALUES ('2026-05-31', 'perfect_acc', 'MSFT', 0.5, 0.5)"
    )

    # Case 2: Determinism mismatch
    conn.execute(
        "INSERT INTO strategy_decisions (run_date, account_name, target_weights, post_trade_positions, equity) "
        "VALUES ('2026-05-31', 'mismatch_acc', '{\"AAPL\": 0.5, \"MSFT\": 0.5}', "
        '\'[{"symbol": "AAPL", "market_value": 50000.0}, {"symbol": "MSFT", "market_value": 50000.0}]\', 100000.0)'
    )
    conn.execute(
        "INSERT INTO weekly_weights (snapshot_date, account, symbol, target_weight, actual_weight) VALUES ('2026-05-31', 'mismatch_acc', 'AAPL', 0.5, 0.5)"
    )
    conn.execute(
        "INSERT INTO weekly_weights (snapshot_date, account, symbol, target_weight, actual_weight) VALUES ('2026-05-31', 'mismatch_acc', 'MSFT', 0.5, 0.5)"
    )

    conn.commit()
    conn.close()

    # Mock get_ar_weights: return perfect match for perfect_acc, mismatched for mismatch_acc
    def get_ar_weights_side_effect(
        config, run_date, is_replay=False, account_name=None
    ):
        if account_name == "perfect_acc":
            return {"AAPL": 0.5, "MSFT": 0.5}
        return {
            "AAPL": 0.4,
            "MSFT": 0.6,
        }  # different weights -> determinism mismatch!

    mock_get_ar_weights.side_effect = get_ar_weights_side_effect

    accounts = [
        {"name": "perfect_acc", "config": "perfect_config.yaml"},
        {"name": "mismatch_acc", "config": "mismatch_config.yaml"},
    ]
    results = [
        {"account": "perfect_acc", "target_weights": {"AAPL": 0.5, "MSFT": 0.5}},
        {"account": "mismatch_acc", "target_weights": {"AAPL": 0.5, "MSFT": 0.5}},
    ]

    with patch("run_paper_trading.Path") as mock_path:

        def path_side_effect(*args, **kwargs):
            val = str(args[0])
            if "finrl_trading.db" in val:
                return db_file
            if "parity_check_2026-05-31.json" in val:
                return report_file
            return Path(*args, **kwargs)

        mock_path.side_effect = path_side_effect

        run_parity_checks("2026-05-31", accounts, results, dry_run=False)

        # Verify JSON report exists
        assert report_file.exists()
        with open(report_file) as f:
            report_data = json.load(f)

        assert "perfect_acc" in report_data["accounts"]
        assert "mismatch_acc" in report_data["accounts"]

        assert report_data["accounts"]["perfect_acc"]["reconciled_successfully"]
        assert not report_data["accounts"]["mismatch_acc"]["reconciled_successfully"]
        assert (
            "Determinism mismatch"
            in report_data["accounts"]["mismatch_acc"]["mismatches"][0]
        )

        # Verify SQLite strategy_decisions was updated with parity_check JSON
        conn = sqlite3.connect(db_file)
        rows = conn.execute(
            "SELECT account_name, parity_check FROM strategy_decisions"
        ).fetchall()
        conn.close()

        assert len(rows) == 2
        for name, parity_json_str in rows:
            assert parity_json_str is not None
            parity_json = json.loads(parity_json_str)
            if name == "perfect_acc":
                assert parity_json["reconciled_successfully"]
            else:
                assert not parity_json["reconciled_successfully"]


@patch("src.trading.alpaca_manager.AlpacaManager._is_market_open")
@patch("src.trading.alpaca_manager.AlpacaManager.get_positions")
@patch("src.trading.alpaca_manager.AlpacaManager.get_portfolio_value")
@patch("src.trading.alpaca_manager.AlpacaManager._is_symbol_tradable")
@patch("src.trading.alpaca_manager.AlpacaManager._is_symbol_fractionable")
@patch("src.trading.alpaca_manager.AlpacaManager.get_account_info")
@patch("src.trading.alpaca_manager.AlpacaManager._get_latest_price")
def test_alpaca_manager_normalization_with_skipped_assets(
    mock_get_price,
    mock_get_acct_info,
    mock_is_fractionable,
    mock_is_tradable,
    mock_get_portfolio_val,
    mock_get_positions,
    mock_is_market_open,
):
    """Test that AlpacaManager normalizes remaining weights when some assets are skipped/non-tradable."""
    from src.trading.alpaca_manager import AlpacaManager, AlpacaAccount

    # Create manager
    acc = AlpacaAccount(name="test_acc", api_key="key", api_secret="sec")
    manager = AlpacaManager([acc])

    # Mock dependencies
    mock_is_market_open.return_value = True
    mock_get_portfolio_val.return_value = 100000.0
    mock_get_positions.return_value = []
    mock_get_acct_info.return_value = {"buying_power": "100000.0"}
    mock_get_price.return_value = 100.0
    mock_is_fractionable.return_value = True

    # SATS is non-tradable, TXN and MCHP are tradable
    def is_tradable_side_effect(symbol):
        return symbol in ["TXN", "MCHP"]

    mock_is_tradable.side_effect = is_tradable_side_effect

    target_weights = {"SATS": 0.3333, "TXN": 0.3333, "MCHP": 0.3333}

    # Execute rebalance plan (dry_run=True so no actual orders are placed)
    result = manager.execute_portfolio_rebalance(
        target_weights=target_weights, account_name="test_acc", dry_run=True
    )

    # The remaining target weights (TXN, MCHP) should be scaled up to sum to min(1.0, original_sum) = 0.9999
    # TXN target weight should be: 0.3333 * (0.9999 / 0.6666) = 0.50
    # MCHP target weight should be: 0.3333 * (0.9999 / 0.6666) = 0.50
    res_weights = result["target_weights"]
    assert "SATS" not in res_weights
    assert res_weights["TXN"] == pytest.approx(0.50, abs=1e-4)
    assert res_weights["MCHP"] == pytest.approx(0.50, abs=1e-4)
