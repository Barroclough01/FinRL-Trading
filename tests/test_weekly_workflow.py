import json
import math
import os
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
    run_post_run_sanity_checks,
    validate_pre_trade,
)
import track_metrics
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
