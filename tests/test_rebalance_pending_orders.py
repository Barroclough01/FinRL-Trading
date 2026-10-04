"""Existing broker orders must survive a repeated rebalance attempt."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import run_paper_trading as runner
from src.trading.alpaca_manager import AlpacaAccount, AlpacaManager


@pytest.fixture
def manager(monkeypatch):
    monkeypatch.setattr("src.trading.alpaca_manager.load_dotenv", lambda: False)
    broker = AlpacaManager([AlpacaAccount("FinRL", "fake", "fake-secret")])
    broker._assets_loaded = True
    monkeypatch.setattr(broker, "_is_market_open", Mock(return_value=True))
    monkeypatch.setattr(broker, "get_orders", Mock(return_value=[]))
    monkeypatch.setattr(broker, "cancel_all_orders", Mock(return_value=1))
    monkeypatch.setattr(broker, "get_positions", Mock(return_value=[]))
    monkeypatch.setattr(broker, "get_portfolio_value", Mock(return_value=100_000.0))
    monkeypatch.setattr(
        broker, "get_account_info", Mock(return_value={"buying_power": "100000"})
    )
    monkeypatch.setattr(broker, "_is_symbol_tradable", Mock(return_value=True))
    monkeypatch.setattr(broker, "_is_symbol_fractionable", Mock(return_value=True))
    monkeypatch.setattr(broker, "_get_latest_price", Mock(return_value=100.0))
    monkeypatch.setattr(broker, "place_orders_batch", Mock(
        side_effect=lambda orders, account: [
            SimpleNamespace(order_id="fake-order", status="accepted")
            for order in orders
        ]
    ))
    return broker


@pytest.mark.parametrize("status", ["accepted", "partially_filled", "pending_cancel"])
@pytest.mark.parametrize("mode", ["open", "next_open", "opg"])
def test_prior_open_orders_hold_before_cancellation_or_submission(
    manager, status, mode
):
    manager._is_market_open.return_value = mode == "open"
    manager.get_orders.return_value = [{"id": "prior-order", "status": status}]
    with pytest.raises(ValueError, match="FinRL.*prior-order"):
        manager.execute_portfolio_rebalance(
            {"A": 0.5}, "FinRL", market_closed_action="skip" if mode == "open" else mode
        )
    manager.get_orders.assert_called_once_with(status="open", account_name="FinRL")
    manager.cancel_all_orders.assert_not_called()
    manager.place_orders_batch.assert_not_called()
    manager.get_positions.assert_not_called()


@pytest.mark.parametrize("response", [None, {}, "", {"error": "unavailable"}])
def test_unusable_open_order_response_fails_closed(manager, response):
    manager.get_orders.return_value = response
    with pytest.raises(ValueError, match="open orders"):
        manager.execute_portfolio_rebalance({"A": 0.5}, "FinRL")
    manager.cancel_all_orders.assert_not_called()
    manager.place_orders_batch.assert_not_called()


def test_failed_open_order_read_fails_closed_with_account_context(manager):
    manager.get_orders.side_effect = RuntimeError("fake broker unavailable")
    with pytest.raises(ValueError, match="FinRL.*fake broker unavailable"):
        manager.execute_portfolio_rebalance({"A": 0.5}, "FinRL")
    manager.cancel_all_orders.assert_not_called()
    manager.place_orders_batch.assert_not_called()


def test_default_account_is_resolved_for_open_order_gate(manager):
    manager.get_orders.return_value = [{"id": "inherited-order"}]
    with pytest.raises(ValueError, match="FinRL.*inherited-order"):
        manager.execute_portfolio_rebalance({"A": 0.5})
    manager.get_orders.assert_called_once_with(status="open", account_name="FinRL")
    manager.place_orders_batch.assert_not_called()


@pytest.mark.parametrize(
    "dry_run,market_open", [(True, True), (True, False), (False, False)]
)
def test_preview_or_closed_skip_does_not_read_cancel_or_submit(
    manager, dry_run, market_open
):
    manager._is_market_open.return_value = market_open
    result = manager.execute_portfolio_rebalance({"A": 0.5}, "FinRL", dry_run=dry_run)
    assert result["orders_placed"] == 0
    assert result["orders_plan"]["buy"]
    manager.get_orders.assert_not_called()
    manager.cancel_all_orders.assert_not_called()
    manager.place_orders_batch.assert_not_called()


def test_clear_account_still_queues_day_orders_without_cancellation(manager):
    manager._is_market_open.return_value = False
    result = manager.execute_portfolio_rebalance(
        {"A": 0.5}, "FinRL", market_closed_action="next_open"
    )
    assert result["orders_placed"] == 1
    assert result["orders"][0]["status"] == "accepted"
    assert result["used_time_in_force"] == "day"
    assert manager.place_orders_batch.call_args.args[0][0].time_in_force == "day"
    manager.cancel_all_orders.assert_not_called()


def test_hold_reaches_final_failure_without_replacing_decision(manager, monkeypatch):
    account = {"name": "FinRL", "config": "fake.yaml"}
    manager.get_orders.return_value = [{"id": "prior-order", "status": "accepted"}]
    manager._is_market_open.return_value = False
    monkeypatch.setenv("MARKET_CLOSED_ACTION", "next_open")
    monkeypatch.delenv("TRADING_DISABLED", raising=False)
    monkeypatch.setattr("sys.argv", ["run_paper_trading.py", "--date", "2026-10-02"])
    monkeypatch.setattr(runner, "load_accounts_from_env", lambda: [account])
    monkeypatch.setattr(runner, "get_target_weights", lambda *a, **k: {"A": 0.5})
    monkeypatch.setattr(
        runner, "get_executor_for_account", lambda a: SimpleNamespace(alpaca=manager)
    )
    monkeypatch.setattr(
        runner, "validate_pre_trade", lambda *a, **k: (True, None, None, None)
    )
    monkeypatch.setattr(runner, "allows_cash_fallback", lambda *a: False)
    decision = Mock()
    monkeypatch.setattr(runner, "save_strategy_decision", decision)
    monkeypatch.setattr(runner, "run_metrics_tracker", lambda *a: (True, None))
    monkeypatch.setattr(runner, "run_rl_tracker", lambda *a: (True, None))
    monkeypatch.setattr(runner, "run_post_run_sanity_checks", lambda *a: [])
    parity = Mock(return_value={"accounts": {}})
    monkeypatch.setattr(runner, "run_parity_checks", parity)
    notification = Mock()
    monkeypatch.setattr(runner, "notify_status", notification)
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 1
    parity.assert_called_once()
    decision.assert_not_called()
    manager.cancel_all_orders.assert_not_called()
    manager.place_orders_batch.assert_not_called()
    assert notification.call_count == 1
    outcome = notification.call_args.args[1]
    assert outcome["status"] == "failed"
    assert "prior-order" in outcome["errors"][0]["error"]
