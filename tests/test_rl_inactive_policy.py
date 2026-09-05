import sqlite3
from unittest.mock import patch

import pandas as pd
import pytest

import track_metrics
from src.strategies.ml_bucket_selection import write_optional_excel_dashboard
from track_rl_offline import (
    OfflinePortfolio,
    apply_inactive_symbol_policy,
    get_fresh_price_on_date,
    record_rl_snapshot,
)


def test_inactive_holding_is_liquidated_to_cash_without_renormalizing():
    portfolio = OfflinePortfolio(100.0, tx_cost_bps=5, slippage_bps=2)
    portfolio.positions["BK"] = 10.0
    source_targets = {"BK": 0.2, "AAPL": 0.8}
    policy = {
        "BK": {
            "effective_date": "2026-07-03",
            "reason": "confirmed inactive",
        }
    }

    with patch("track_rl_offline.get_price_on_date", return_value=50.0):
        effective_targets, events = apply_inactive_symbol_policy(
            portfolio, source_targets, "2026-07-03", policy
        )

    assert effective_targets == {"AAPL": 0.8}
    assert "BK" not in portfolio.positions
    assert portfolio.cash == pytest.approx(599.65)
    assert events[0]["source_target_weight"] == pytest.approx(0.2)


def test_inactive_holding_without_cached_price_fails_safely():
    portfolio = OfflinePortfolio(100.0, tx_cost_bps=5, slippage_bps=2)
    portfolio.positions["DAY"] = 10.0
    policy = {"DAY": {"effective_date": "2025-08-30"}}

    with (
        patch("track_rl_offline.get_price_on_date", return_value=0.0),
        pytest.raises(RuntimeError, match="cannot be liquidated"),
    ):
        apply_inactive_symbol_policy(
            portfolio, {"DAY": 0.1, "AAPL": 0.9}, "2026-05-10", policy
        )

    assert portfolio.positions["DAY"] == 10.0
    assert portfolio.cash == 100.0


def test_active_symbol_with_stale_cached_price_fails_safely():
    stale_prices = pd.DataFrame(
        [{"date": pd.Timestamp("2026-08-24"), "close": 68.14}]
    )

    with (
        patch("track_rl_offline.load_price_data", return_value=stale_prices),
        pytest.raises(RuntimeError, match=r"AVB.*last close 2026-08-24.*2026-09-04"),
    ):
        get_fresh_price_on_date("AVB", "2026-09-04")


def test_active_symbol_accepts_latest_session_on_weekend():
    friday_prices = pd.DataFrame(
        [{"date": pd.Timestamp("2026-08-14"), "close": 100.0}]
    )

    with patch("track_rl_offline.load_price_data", return_value=friday_prices):
        assert get_fresh_price_on_date("AAPL", "2026-08-15") == 100.0


def test_avb_policy_liquidates_at_last_verified_close_after_cutoff():
    portfolio = OfflinePortfolio(100.0, tx_cost_bps=0, slippage_bps=0)
    portfolio.positions["AVB"] = 10.0
    policy = {
        "AVB": {
            "effective_date": "2026-08-25",
            "last_price_date": "2026-08-24",
            "reason": "verified inactive",
        }
    }

    with patch("track_rl_offline.get_price_on_date", return_value=68.14):
        targets, events = apply_inactive_symbol_policy(
            portfolio, {"AVB": 0.1, "AAPL": 0.9}, "2026-08-29", policy
        )

    assert targets == {"AAPL": 0.9}
    assert portfolio.cash == pytest.approx(781.4)
    assert events[0]["liquidation_price"] == pytest.approx(68.14)


def test_inactive_source_target_is_preserved_as_zero_actual_weight(tmp_path):
    database = sqlite3.connect(tmp_path / "metrics.db")
    track_metrics.init_db(database)
    portfolio = OfflinePortfolio(1_000.0, tx_cost_bps=0, slippage_bps=0)

    record_rl_snapshot(
        database,
        "2026-08-14",
        portfolio,
        prices={},
        target_weights={"BK": 0.1},
        benchmark={},
    )

    target, actual, market_value = database.execute(
        "SELECT target_weight, actual_weight, market_value "
        "FROM weekly_weights WHERE account='RL' AND symbol='BK'"
    ).fetchone()
    database.close()

    assert target == pytest.approx(0.1)
    assert actual == 0.0
    assert market_value == 0.0


def test_optional_excel_dashboard_skips_missing_openpyxl(capsys):
    missing_dependency = ModuleNotFoundError("No module named 'openpyxl'")
    missing_dependency.name = "openpyxl"
    predictions = pd.DataFrame([{"rank_best": 1, "tic": "AAPL"}])

    with patch(
        "src.strategies.ml_bucket_selection.pd.ExcelWriter",
        side_effect=missing_dependency,
    ):
        written = write_optional_excel_dashboard(
            "ignored.xlsx", predictions, [], []
        )

    assert not written
    assert "skipping optional Excel dashboard" in capsys.readouterr().out


def test_optional_excel_dashboard_reraises_other_missing_dependency():
    missing_dependency = ModuleNotFoundError("No module named 'other_package'")
    missing_dependency.name = "other_package"

    with (
        patch(
            "src.strategies.ml_bucket_selection.pd.ExcelWriter",
            side_effect=missing_dependency,
        ),
        pytest.raises(ModuleNotFoundError, match="other_package"),
    ):
        write_optional_excel_dashboard(
            "ignored.xlsx",
            pd.DataFrame([{"rank_best": 1}]),
            [],
            [],
        )
