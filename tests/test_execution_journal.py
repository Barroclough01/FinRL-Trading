"""Crash, identity, concurrency and receipt recovery without broker mutations."""

import json
import os
import signal
import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import recover_paper_execution as recovery
import run_paper_trading as runner
from src.trading import execution_journal as journal
from src.trading.alpaca_manager import AlpacaAccount, AlpacaManager

DAY = "2026-10-02"
PAYLOAD = {
    "symbol": "A",
    "qty": "1",
    "side": "buy",
    "type": "market",
    "time_in_force": "day",
    "extended_hours": False,
}


def receipt(payload, status="accepted", filled="0"):
    return {
        **payload,
        "id": "broker-" + payload["client_order_id"],
        "status": status,
        "filled_qty": filled,
        "filled_avg_price": "100" if filled != "0" else None,
        "submitted_at": "2026-10-02T22:00:00Z",
    }


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(journal, "require_execution_host", lambda *a: None)
    monkeypatch.setenv(journal.GATE, "true")
    instance = journal.ExecutionJournal(tmp_path)
    yield instance
    instance.close()


def session(
    store,
    *,
    alias="FinRL",
    broker_id="account-1",
    targets=None,
    config_hash="config-1",
    day=DAY,
    recovery_only=False,
):
    return store.open_session(
        endpoint=journal.PAPER_ENDPOINT,
        broker_id=broker_id,
        alias=alias,
        day=day,
        config_hash=config_hash,
        targets=targets or {"A": 0.5},
        snapshot={"positions": [], "account": {"id": broker_id}},
        attempt=journal.new_attempt(store.path.parent.parent),
        recovery=recovery_only,
    )


def fake_manager(store, *, status="accepted", filled="0", positions=None):
    account = AlpacaAccount("FinRL", "fake", "fake-secret")
    manager = SimpleNamespace(_get_account=lambda *a: account)
    calls = []

    def api(method, path, **kwargs):
        calls.append((method, path))
        assert method == "GET"
        assert kwargs["allow_redirects"] is False
        if path == "/v2/account":
            return {
                "id": "account-1",
                "cash": "100",
                "equity": "1000",
                "buying_power": "100",
            }
        if path == "/v2/positions":
            return (
                positions
                if positions is not None
                else ([{"symbol": "A", "qty": filled}] if filled != "0" else [])
            )
        intent = store.conn.execute(
            "SELECT payload FROM order_intents LIMIT 1"
        ).fetchone()
        return receipt(json.loads(intent[0]), status, filled)

    manager._api_request = Mock(side_effect=api)
    return manager, calls


def execute(one, *, status="accepted", filled="0"):
    one.submit_batch("sell", [], lambda p: receipt(p))
    result = one.submit_batch("buy", [PAYLOAD], lambda p: receipt(p, status, filled))
    one.finish()
    return result


def test_intent_and_unknown_commit_precede_the_only_post(store):
    one = session(store)
    one.submit_batch("sell", [], Mock())

    def transport(payload):
        with sqlite3.connect(store.path) as independent:
            row = independent.execute(
                "SELECT state, payload FROM order_intents"
            ).fetchone()
            assert row[0] == "attempted_unknown"
            assert json.loads(row[1]) == payload
        assert len(payload["client_order_id"]) == 43
        return receipt(payload)

    post = Mock(side_effect=transport)
    one.submit_batch("buy", [PAYLOAD], post)
    one.finish()
    post.assert_called_once()


@pytest.mark.parametrize(
    "stage", ["frozen", "marked", "accepted", "receipt", "complete"]
)
def test_crash_boundaries_never_resume_or_resubmit(store, stage):
    one = session(store)
    one.submit_batch("sell", [], Mock())
    post = Mock(side_effect=lambda p: receipt(p))
    if stage == "frozen":
        one.freeze("buy", [PAYLOAD])
    elif stage == "marked":
        post.side_effect = KeyboardInterrupt("process interrupted before transport")
        with pytest.raises(KeyboardInterrupt):
            one.submit_batch("buy", [PAYLOAD], post)
    elif stage == "accepted":

        def lost(payload):
            receipt(payload)  # fake broker accepted; response is then lost
            raise TimeoutError("response lost after acceptance")

        post.side_effect = lost
        with pytest.raises(TimeoutError):
            one.submit_batch("buy", [PAYLOAD], post)
    else:
        one.submit_batch("buy", [PAYLOAD], post)
        if stage == "complete":
            one.finish()
    store.close()
    reopened = journal.ExecutionJournal(store.path.parent.parent, create=False)
    try:
        repeat = session(reopened, recovery_only=True)
        assert repeat.repeated
        with pytest.raises(ValueError, match="cannot resume"):
            repeat.submit_batch("buy", [PAYLOAD], post)
        manager, calls = fake_manager(reopened)
        result = repeat.recover(manager, "FinRL")
        assert all(method == "GET" for method, _ in calls)
        assert bool(result["failures"]) == (stage != "complete")
        if stage == "accepted":
            assert any(path == "/v2/orders:by_client_order_id" for _, path in calls)
            assert (
                reopened.conn.execute("SELECT state FROM order_intents").fetchone()[0]
                == "accepted"
            )
    finally:
        reopened.close()


@pytest.mark.parametrize(
    "status,filled",
    [
        ("accepted", "0"),
        ("partially_filled", "0.5"),
        ("filled", "1"),
        ("canceled", "0"),
        ("rejected", "0"),
        ("expired", "0"),
    ],
)
def test_repeat_observes_append_only_status_and_fill_evidence(store, status, filled):
    one = session(store)
    execute(one)
    before = store.conn.execute("SELECT count(*) FROM execution_events").fetchone()[0]
    repeated = session(store, alias="same-account-alias", recovery_only=True)
    manager, calls = fake_manager(store, status=status, filled=filled)
    result = repeated.recover(manager, "FinRL")
    assert all(m == "GET" for m, _ in calls)
    assert result["execution_pending"] == (status in journal.PENDING)
    assert result["reconciled_successfully"] == (status == "filled")
    assert (
        store.conn.execute("SELECT count(*) FROM execution_sessions").fetchone()[0] == 1
    )
    assert store.conn.execute("SELECT count(*) FROM order_intents").fetchone()[0] == 1
    assert (
        store.conn.execute("SELECT count(*) FROM execution_events").fetchone()[0]
        > before
    )
    prior_receipts = [
        json.loads(row[0])
        for row in store.conn.execute(
            "SELECT evidence FROM execution_events WHERE kind='receipt_observed'"
        )
    ]
    assert prior_receipts[0]["status"] == "accepted"
    assert prior_receipts[-1]["status"] == status
    assert prior_receipts[-1]["filled_qty"] == filled


@pytest.mark.parametrize("change", ["config", "targets", "account", "endpoint"])
def test_binding_and_frozen_inputs_hold_before_new_session_mutation(store, change):
    session(store)
    before = store.conn.execute("SELECT count(*) FROM execution_events").fetchone()[0]
    args = (
        {"config_hash": "changed"}
        if change == "config"
        else (
            {"targets": {"A": 0.4}} if change == "targets" else {"broker_id": "other"}
        )
    )
    if change == "endpoint":
        with pytest.raises(ValueError, match="binding changed"):
            store.open_session(
                endpoint="other",
                broker_id="account-1",
                alias="FinRL",
                day=DAY,
                config_hash="config-1",
                targets={},
                snapshot={},
                attempt=journal.new_attempt(store.path.parent.parent),
            )
    else:
        with pytest.raises(ValueError, match="changed"):
            session(store, **args)
    assert (
        store.conn.execute("SELECT count(*) FROM execution_sessions").fetchone()[0] == 1
    )
    assert (
        store.conn.execute("SELECT count(*) FROM execution_events").fetchone()[0]
        == before
    )


def test_external_position_drift_holds_with_no_post(store):
    execute(session(store))
    repeat = session(store, recovery_only=True)
    manager, _ = fake_manager(
        store, status="filled", filled="1", positions=[{"symbol": "A", "qty": "2"}]
    )
    result = repeat.recover(manager, "FinRL")
    assert "external position drift" in str(result["failures"])
    assert not result["reconciled_successfully"]


@pytest.mark.parametrize("bad", ["read", "not_found", "mismatch", "unknown"])
def test_recovery_read_and_receipt_failures_hold(store, bad):
    execute(session(store))
    repeat = session(store, recovery_only=True)
    manager, _ = fake_manager(
        store, status="unrecognized" if bad == "unknown" else "accepted"
    )
    if bad in {"read", "not_found"}:
        manager._api_request.side_effect = RuntimeError(bad)
    elif bad == "mismatch":
        manager._api_request.return_value = {"id": "wrong"}
        manager._api_request.side_effect = None
    assert repeat.recover(manager, "FinRL")["failures"]


def test_disk_failure_before_intent_commit_means_zero_posts(store):
    one = session(store)
    store.conn.execute("PRAGMA query_only=ON")
    post = Mock()
    with pytest.raises(sqlite3.OperationalError):
        one.submit_batch("sell", [PAYLOAD], post)
    post.assert_not_called()


def test_receipt_write_failure_stops_remaining_orders_and_buy_phase(store):
    one = session(store)
    one.submit_batch("sell", [], Mock())
    second = {**PAYLOAD, "symbol": "B"}

    def transport(payload):
        store.conn.execute("PRAGMA query_only=ON")
        return receipt(payload)

    post = Mock(side_effect=transport)
    with pytest.raises(sqlite3.OperationalError):
        one.submit_batch("buy", [PAYLOAD, second], post)
    post.assert_called_once()


def test_corrupt_and_unknown_schema_refuse_before_submission(tmp_path):
    path = tmp_path / "data" / "paper_execution_journal.sqlite3"
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(b"corrupt database")
    with pytest.raises(ValueError, match="Cannot open execution journal"):
        journal.ExecutionJournal(tmp_path)
    path.unlink()  # disposable fixture only
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA user_version=99")
    with pytest.raises(ValueError, match="Unsupported journal schema"):
        journal.ExecutionJournal(tmp_path)


def test_os_lock_blocks_a_second_process_and_releases_after_death(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(journal, "require_execution_host", lambda *a: None)
    read_fd, write_fd = os.pipe()
    with journal.account_lock(tmp_path, journal.PAPER_ENDPOINT, "account-1"):
        pid = os.fork()
        if pid == 0:
            os.close(read_fd)
            try:
                with journal.account_lock(
                    tmp_path, journal.PAPER_ENDPOINT, "account-1"
                ):
                    os.write(write_fd, b"unexpected")
            except ValueError:
                os.write(write_fd, b"blocked")
            os._exit(0)
        os.close(write_fd)
        assert os.read(read_fd, 32) == b"blocked"
        os.close(read_fd)
        os.waitpid(pid, 0)
    with journal.account_lock(tmp_path, journal.PAPER_ENDPOINT, "account-1"):
        pass


def test_sigkill_after_fake_acceptance_releases_lock_and_recovers_same_id(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(journal, "require_execution_host", lambda *a: None)
    accepted_path = tmp_path / "fake_broker_acceptance.json"
    child = os.fork()
    if child == 0:
        with journal.account_lock(tmp_path, journal.PAPER_ENDPOINT, "account-1"):
            child_store = journal.ExecutionJournal(tmp_path)
            one = session(child_store)
            one.submit_batch("sell", [], Mock())

            def kill_after_acceptance(payload):
                journal.write_evidence(accepted_path, receipt(payload))
                os.kill(os.getpid(), signal.SIGKILL)

            one.submit_batch("buy", [PAYLOAD], kill_after_acceptance)
        os._exit(99)
    _, status = os.waitpid(child, 0)
    assert os.WIFSIGNALED(status) and os.WTERMSIG(status) == signal.SIGKILL
    with journal.account_lock(tmp_path, journal.PAPER_ENDPOINT, "account-1"):
        reopened = journal.ExecutionJournal(tmp_path, create=False)
        try:
            assert reopened.conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            intent = reopened.conn.execute(
                "SELECT state, client_id FROM order_intents"
            ).fetchone()
            assert intent[0] == "attempted_unknown"
            accepted = json.loads(accepted_path.read_text())
            assert accepted["client_order_id"] == intent[1]
            repeat = session(reopened, recovery_only=True)
            manager, calls = fake_manager(reopened)
            result = repeat.recover(manager, "FinRL")
            assert result["orders"][0]["client_order_id"] == intent[1]
            assert result["failures"]  # interrupted phase never resumes
            assert all(method == "GET" for method, _ in calls)
        finally:
            reopened.close()


@pytest.mark.parametrize("value", [None, "false", ""])
def test_disabled_gate_uses_existing_contract_without_journal(
    value, monkeypatch, tmp_path
):
    if value is None:
        monkeypatch.delenv(journal.GATE, raising=False)
    else:
        monkeypatch.setenv(journal.GATE, value)
    core = Mock(return_value={"disabled": True})
    monkeypatch.setattr(runner, "_run_account", core)
    assert runner.run_account({"name": "FinRL"}, DAY, False) == {"disabled": True}
    assert not (tmp_path / "data/paper_execution_journal.sqlite3").exists()


def test_exact_endpoint_and_native_host_fail_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(journal.platform, "release", lambda: "native-windows")
    with pytest.raises(ValueError, match="WSL"):
        journal.require_execution_host(tmp_path)
    manager = SimpleNamespace(
        _get_account=lambda a: AlpacaAccount(
            "FinRL", "fake", "fake", "https://paper.evil.example"
        ),
        _api_request=Mock(),
    )
    with pytest.raises(ValueError, match="verified paper endpoint"):
        journal.identity(manager, "FinRL")
    manager._api_request.assert_not_called()


def test_get_only_command_recovers_existing_ids(store, monkeypatch):
    execute(session(store))
    manager, calls = fake_manager(store)
    result = recovery.recover_account(manager, "FinRL", DAY, store.path.parent.parent)
    assert not result["failures"]
    assert result["broker_read_only"]
    assert all(method == "GET" for method, _ in calls)


@pytest.mark.parametrize("redirect", [307, 308])
def test_durable_transport_never_follows_post_redirects(monkeypatch, redirect):
    monkeypatch.setattr("src.trading.alpaca_manager.load_dotenv", lambda: False)
    manager = AlpacaManager([AlpacaAccount("FinRL", "fake", "fake")])
    response = SimpleNamespace(status_code=redirect)
    request = Mock(return_value=response)
    monkeypatch.setattr("src.trading.alpaca_manager.requests.request", request)
    with pytest.raises(RuntimeError, match="redirect refused"):
        manager._api_request(
            "POST", "/v2/orders", json_body=PAYLOAD, allow_redirects=False
        )
    request.assert_called_once()
    assert request.call_args.kwargs["allow_redirects"] is False
