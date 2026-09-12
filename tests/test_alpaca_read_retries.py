from unittest.mock import MagicMock, call, patch

import pytest
import requests

from src.trading.alpaca_manager import AlpacaAccount, AlpacaManager, OrderRequest


@pytest.fixture
def manager():
    account = AlpacaAccount(name="paper", api_key="key", api_secret="secret")
    return AlpacaManager([account])


def response(status_code, payload):
    result = MagicMock()
    result.status_code = status_code
    result.headers = {"content-type": "application/json"}
    result.json.return_value = payload
    result.text = str(payload)
    return result


@patch("src.trading.alpaca_manager.time.sleep")
@patch("src.trading.alpaca_manager.requests.request")
def test_transient_read_failure_retries_then_succeeds(
    mock_request, mock_sleep, manager
):
    mock_request.side_effect = [
        requests.exceptions.ConnectionError("temporary DNS resolution failure"),
        response(200, {"equity": "100000"}),
    ]

    result = manager.get_account_info()

    assert result == {"equity": "100000"}
    assert mock_request.call_count == 2
    mock_sleep.assert_called_once_with(0.5)


@patch("src.trading.alpaca_manager.time.sleep")
@patch("src.trading.alpaca_manager.requests.request")
def test_transient_http_read_failure_retries_then_succeeds(
    mock_request, mock_sleep, manager
):
    mock_request.side_effect = [
        response(503, {"message": "temporarily unavailable"}),
        response(200, [{"symbol": "SPY"}]),
    ]

    result = manager.get_positions()

    assert result == [{"symbol": "SPY"}]
    assert mock_request.call_count == 2
    mock_sleep.assert_called_once_with(0.5)


@patch("src.trading.alpaca_manager.time.sleep")
@patch("src.trading.alpaca_manager.requests.request")
def test_transient_read_failure_exhausts_bounded_retries(
    mock_request, mock_sleep, manager
):
    mock_request.side_effect = requests.exceptions.Timeout("broker timed out")

    with pytest.raises(RuntimeError, match="failed after 3 attempts.*broker timed out"):
        manager.get_account_info()

    assert mock_request.call_count == 3
    assert mock_sleep.call_args_list == [call(0.5), call(1.0)]


@patch("src.trading.alpaca_manager.time.sleep")
@patch("src.trading.alpaca_manager.requests.request")
def test_permanent_read_error_is_not_retried(mock_request, mock_sleep, manager):
    mock_request.return_value = response(401, {"message": "unauthorized"})

    with pytest.raises(RuntimeError, match="Alpaca API error 401"):
        manager.get_account_info()

    mock_request.assert_called_once()
    mock_sleep.assert_not_called()


@patch("src.trading.alpaca_manager.time.sleep")
@patch("src.trading.alpaca_manager.requests.request")
def test_order_write_failure_is_not_retried(mock_request, mock_sleep, manager):
    mock_request.side_effect = requests.exceptions.ConnectionError(
        "connection dropped during submission"
    )

    with pytest.raises(RuntimeError, match="connection dropped during submission"):
        manager.place_order(OrderRequest(symbol="SPY", quantity=1, side="buy"))

    mock_request.assert_called_once()
    mock_sleep.assert_not_called()
