"""Journal phase safety and history boundaries in the real weekly orchestration."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import run_paper_trading as runner
import track_metrics
from src.trading import execution_journal as journal
from src.trading.alpaca_manager import AlpacaAccount, AlpacaManager

DAY = "2026-10-02"


@pytest.fixture
def environment(monkeypatch, tmp_path):
    monkeypatch.setenv(journal.GATE, "true")
    monkeypatch.delenv("TRADING_DISABLED", raising=False)
    monkeypatch.setattr(journal, "require_execution_host", lambda *a: None)
    config = tmp_path / "fixture.yaml"
    config.write_text("fixture: true")
    accounts = [{"name": name, "config": str(config)} for name in ("FinRL", "AR")]
    monkeypatch.setattr("sys.argv", ["run_paper_trading.py", "--date", DAY])
    monkeypatch.setattr(runner, "load_accounts_from_env", lambda: accounts)
    monkeypatch.setattr(runner, "get_target_weights", lambda *a, **k: {"A": 0.5})
    monkeypatch.setattr(
        runner, "validate_pre_trade", lambda *a, **k: (True, None, None, None)
    )
    monkeypatch.setattr(runner, "allows_cash_fallback", lambda *a: False)
    metrics = Mock(return_value=(True, None))
    rl = Mock(return_value=(True, None))
    sanity = Mock(return_value=[])
    real_parity = runner.run_parity_checks
    parity = Mock(
        return_value={
            "accounts": {name: {"execution_pending": True} for name in ("FinRL", "AR")}
        }
    )
    notify = Mock()
    monkeypatch.setattr(runner, "run_metrics_tracker", metrics)
    monkeypatch.setattr(runner, "run_rl_tracker", rl)
    monkeypatch.setattr(runner, "run_post_run_sanity_checks", sanity)
    monkeypatch.setattr(runner, "run_parity_checks", parity)
    monkeypatch.setattr(runner, "notify_status", notify)
    return SimpleNamespace(
        accounts=accounts,
        metrics=metrics,
        rl=rl,
        sanity=sanity,
        parity=parity,
        real_parity=real_parity,
        notify=notify,
        root=tmp_path,
    )


def manager(monkeypatch, name, *, pending=False, lost=False, bad_cash=False):
    broker = AlpacaManager([AlpacaAccount(name, "fake", "fake-secret")])
    broker._assets_loaded = True
    monkeypatch.setattr(broker, "_is_market_open", Mock(return_value=False))
    monkeypatch.setattr(
        broker,
        "get_orders",
        Mock(
            return_value=(
                [{"id": "prior", "status": "partially_filled"}] if pending else []
            )
        ),
    )
    monkeypatch.setattr(broker, "get_positions", Mock(return_value=[]))
    monkeypatch.setattr(broker, "get_portfolio_value", Mock(return_value=100000.0))
    info = {
        "id": name + "-id",
        "cash": "100000",
        "equity": "100000",
        "buying_power": "100000",
    }
    monkeypatch.setattr(broker, "get_account_info", Mock(return_value=info))
    monkeypatch.setattr(broker, "_is_symbol_tradable", Mock(return_value=True))
    monkeypatch.setattr(broker, "_is_symbol_fractionable", Mock(return_value=True))
    monkeypatch.setattr(broker, "_get_latest_price", Mock(return_value=100.0))
    monkeypatch.setattr(broker, "cancel_all_orders", Mock())
    receipts = {}

    def api(method, path, **kwargs):
        if path == "/v2/account":
            return info
        if path == "/v2/positions":
            return []
        if method == "POST":
            p = kwargs["json_body"]
            assert kwargs["allow_redirects"] is False
            r = {
                **p,
                "id": "order-" + p["client_order_id"],
                "status": "accepted",
                "filled_qty": "0",
                "filled_avg_price": None,
                "submitted_at": "2026-10-02T22:00:00Z",
            }
            receipts[r["id"]] = r
            if lost:
                raise TimeoutError("acceptance response lost")
            return r
        return receipts[path.rsplit("/", 1)[-1]]

    monkeypatch.setattr(broker, "_api_request", Mock(side_effect=api))
    if bad_cash:

        def read_cash(*a, **k):
            raise RuntimeError("buying-power read failed")

        monkeypatch.setattr(broker, "get_account_info", Mock(side_effect=read_cash))
    return broker


def test_first_day_run_and_repeat_preserve_prior_decision_and_audit(
    environment, monkeypatch
):
    monkeypatch.setenv("MARKET_CLOSED_ACTION", "next_open")
    brokers = {a["name"]: manager(monkeypatch, a["name"]) for a in environment.accounts}
    monkeypatch.setattr(
        runner,
        "get_executor_for_account",
        lambda a: SimpleNamespace(alpaca=brokers[a["name"]]),
    )
    runner.main()
    first_audits = {
        p: p.read_bytes()
        for p in environment.root.glob("logs/execution_attempts/*/*.json")
    }
    db = environment.root / "data/finrl_trading.db"
    before_db = db.read_bytes()
    first_posts = [
        sum(c.args[0] == "POST" for c in b._api_request.call_args_list)
        for b in brokers.values()
    ]
    assert first_posts == [1, 1]
    assert environment.metrics.call_args.kwargs["accounts"] == ["AR", "FinRL"]
    assert (
        environment.metrics.call_args.kwargs["execution_log"].parent.parent.name
        == "execution_attempts"
    )
    environment.metrics.reset_mock()
    environment.rl.reset_mock()
    environment.sanity.reset_mock()
    environment.notify.reset_mock()
    runner.main()
    assert db.read_bytes() == before_db
    assert all(p.read_bytes() == content for p, content in first_audits.items())
    assert [
        sum(c.args[0] == "POST" for c in b._api_request.call_args_list)
        for b in brokers.values()
    ] == first_posts
    environment.metrics.assert_not_called()
    environment.rl.assert_not_called()
    environment.sanity.assert_not_called()
    assert environment.parity.call_args.kwargs["persist_accounts"] == set()
    assert (
        environment.notify.call_args.args[1]["status"] != "ok"
        if environment.notify.called
        else True
    )


@pytest.mark.parametrize("failure", ["identity", "corrupt", "binding", "config"])
def test_all_held_preflight_never_invokes_history_tail(
    environment, monkeypatch, failure
):
    environment.accounts[:] = environment.accounts[:1]
    broker = manager(monkeypatch, "FinRL")
    monkeypatch.setattr(
        runner, "get_executor_for_account", lambda a: SimpleNamespace(alpaca=broker)
    )
    if failure == "identity":
        broker._api_request.side_effect = lambda *a, **k: {}
    elif failure == "corrupt":
        path = environment.root / "data/paper_execution_journal.sqlite3"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"corrupt")
    else:
        store = journal.ExecutionJournal(environment.root)
        try:
            store.open_session(
                endpoint=journal.PAPER_ENDPOINT,
                broker_id="other-id" if failure == "binding" else "FinRL-id",
                alias="FinRL",
                day=DAY,
                config_hash="original-config",
                targets={"A": 0.5},
                snapshot={"positions": []},
                attempt=journal.new_attempt(environment.root),
            )
        finally:
            store.close()
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 1
    environment.metrics.assert_not_called()
    environment.rl.assert_not_called()
    environment.sanity.assert_not_called()
    assert environment.parity.call_args.kwargs["persist_accounts"] == set()
    assert environment.notify.call_args.args[1]["status"] == "failed"


def test_mixed_prior_and_new_only_snapshots_and_persists_new_account(
    environment, monkeypatch
):
    monkeypatch.setenv("MARKET_CLOSED_ACTION", "next_open")
    original = environment.accounts[:]
    brokers = {a["name"]: manager(monkeypatch, a["name"]) for a in original}
    monkeypatch.setattr(
        runner,
        "get_executor_for_account",
        lambda a: SimpleNamespace(alpaca=brokers[a["name"]]),
    )
    environment.accounts[:] = original[:1]
    runner.main()
    first_posts = sum(
        c.args[0] == "POST" for c in brokers["FinRL"]._api_request.call_args_list
    )
    environment.metrics.reset_mock()
    environment.rl.reset_mock()
    environment.accounts[:] = original
    runner.main()
    assert environment.metrics.call_args.kwargs["accounts"] == ["AR"]
    assert environment.parity.call_args.kwargs["persist_accounts"] == {"AR"}
    environment.rl.assert_not_called()
    assert (
        sum(c.args[0] == "POST" for c in brokers["FinRL"]._api_request.call_args_list)
        == first_posts
    )


@pytest.mark.parametrize("pending,lost", [(True, False), (False, True)])
def test_failed_account_holds_existing_or_unknown_orders(
    environment, monkeypatch, pending, lost
):
    monkeypatch.setenv("MARKET_CLOSED_ACTION", "next_open")
    broker = manager(monkeypatch, "FinRL", pending=pending, lost=lost)
    environment.accounts[:] = environment.accounts[:1]
    monkeypatch.setattr(
        runner, "get_executor_for_account", lambda a: SimpleNamespace(alpaca=broker)
    )
    with pytest.raises(SystemExit):
        runner.main()
    broker.cancel_all_orders.assert_not_called()
    posts = [c for c in broker._api_request.call_args_list if c.args[0] == "POST"]
    assert len(posts) == (1 if lost else 0)
    environment.metrics.reset_mock()
    environment.rl.reset_mock()
    with pytest.raises(SystemExit):
        runner.main()
    assert len(
        [c for c in broker._api_request.call_args_list if c.args[0] == "POST"]
    ) == len(posts)
    environment.metrics.assert_not_called()
    environment.rl.assert_not_called()


def test_metrics_reads_current_immutable_targets_and_excludes_prior_accounts(
    monkeypatch, tmp_path
):
    audit = tmp_path / "current.json"
    audit.write_text(
        json.dumps(
            {"date": DAY, "accounts": [{"account": "AR", "target_weights": {"A": 0.4}}]}
        )
    )
    Path(f"logs/execution_{DAY}.json").write_text(
        json.dumps({"accounts": [{"account": "AR", "target_weights": {"OLD": 1.0}}]})
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "track_metrics.py",
            "--date",
            DAY,
            "--account",
            "AR",
            "--execution-log",
            str(audit),
        ],
    )
    monkeypatch.setattr(track_metrics, "DB_PATH", str(tmp_path / "metrics.db"))
    monkeypatch.setattr(
        track_metrics,
        "load_accounts_from_env",
        lambda: [{"name": "FinRL"}, {"name": "AR"}],
    )
    monkeypatch.setattr(track_metrics, "fetch_benchmark_prices", lambda *a: {})
    monkeypatch.setattr(track_metrics, "get_alpaca_snapshot", Mock(return_value={}))
    record = Mock()
    monkeypatch.setattr(track_metrics, "record_snapshot", record)
    monkeypatch.setattr(track_metrics, "print_cli_report", Mock())
    monkeypatch.setattr(track_metrics, "generate_html_dashboard", Mock())
    monkeypatch.setattr(track_metrics, "calculate_comparison_metrics", lambda *a: {})
    monkeypatch.setattr(track_metrics, "save_comparison_metrics", Mock())
    track_metrics.main()
    record.assert_called_once()
    assert record.call_args.args[2]["name"] == "AR"
    assert record.call_args.args[-1] == {"A": 0.4}


@pytest.mark.parametrize("mismatch", [False, True])
def test_enabled_repeat_runs_real_final_parity_read_only_and_never_ok(
    environment, monkeypatch, mismatch
):
    import hashlib
    import sqlite3

    from test_final_outcome import prepare_parity

    environment.accounts[:] = environment.accounts[:1]
    # Existing original snapshots/decision are the comparison authority.
    prepare_parity(pending=True)
    broker = manager(monkeypatch, "FinRL")
    monkeypatch.setattr(
        runner, "get_executor_for_account", lambda a: SimpleNamespace(alpaca=broker)
    )
    store = journal.ExecutionJournal(environment.root)
    try:
        one = store.open_session(
            endpoint=journal.PAPER_ENDPOINT,
            broker_id="FinRL-id",
            alias="FinRL",
            day=DAY,
            config_hash=hashlib.sha256(
                Path(environment.accounts[0]["config"]).read_bytes()
            ).hexdigest(),
            targets={"A": 0.5},
            snapshot={"positions": []},
            attempt=journal.new_attempt(environment.root),
        )
        one.submit_batch("sell", [], Mock())
        one.submit_batch(
            "buy",
            [
                {
                    "symbol": "A",
                    "qty": "1",
                    "side": "buy",
                    "type": "market",
                    "time_in_force": "day",
                    "extended_hours": False,
                }
            ],
            lambda p: broker._api_request(
                "POST", "/v2/orders", json_body=p, allow_redirects=False
            ),
        )
        one.finish()
    finally:
        store.close()
    # Fixture helper uses its original date; move only disposable fixture rows.
    db = environment.root / "data/finrl_trading.db"
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE strategy_decisions SET run_date=?", (DAY,))
        conn.execute("UPDATE weekly_snapshot SET snapshot_date=?", (DAY,))
        conn.execute("UPDATE weekly_weights SET snapshot_date=?", (DAY,))
        if mismatch:
            conn.execute(
                "UPDATE strategy_decisions SET target_weights=?",
                (json.dumps({"A": 0.4}),),
            )
    before = db.read_bytes()
    monkeypatch.setattr(runner, "run_parity_checks", environment.real_parity)
    if mismatch:
        monkeypatch.setattr(
            runner,
            "get_target_weights",
            lambda *a, **k: {"A": 0.4 if k.get("is_replay") else 0.5},
        )
        with pytest.raises(SystemExit) as exc:
            runner.main()
        assert exc.value.code == 1
        assert environment.notify.call_args.args[1]["status"] == "failed"
    else:
        runner.main()
        environment.notify.assert_not_called()
    assert db.read_bytes() == before
    environment.metrics.assert_not_called()
    environment.rl.assert_not_called()


@pytest.mark.parametrize(
    "failure", ["price", "cash", "missing_cash", "post_value", "post_nan"]
)
def test_durable_phase_read_failure_never_fabricates_orders(
    environment, monkeypatch, failure
):
    broker = manager(monkeypatch, "FinRL")
    store = journal.ExecutionJournal(environment.root)
    try:
        one = store.open_session(
            endpoint=journal.PAPER_ENDPOINT,
            broker_id="FinRL-id",
            alias="FinRL",
            day=DAY,
            config_hash="config",
            targets={"A": 0.5},
            snapshot={"positions": []},
            attempt=journal.new_attempt(environment.root),
        )
        if failure == "price":
            broker._get_latest_price.return_value = None
        elif failure == "cash":
            broker.get_account_info.side_effect = RuntimeError("read failed")
        elif failure.startswith("post_"):
            broker.get_portfolio_value.side_effect = [
                100000.0,
                float("nan") if failure == "post_nan" else 0.0,
            ]
        else:
            broker.get_account_info.return_value = {}
        with pytest.raises((ValueError, RuntimeError)):
            broker.execute_portfolio_rebalance(
                {"A": 0.5},
                "FinRL",
                market_closed_action="next_open",
                execution_session=one,
            )
        assert not [
            c for c in broker._api_request.call_args_list if c.args[0] == "POST"
        ]
        assert store.conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        store.close()


def test_unknown_sell_acceptance_stops_before_buy_intents(environment, monkeypatch):
    broker = manager(monkeypatch, "FinRL", lost=True)
    broker.get_positions.return_value = [
        {"symbol": "OLD", "qty": "10", "market_value": "1000"}
    ]
    store = journal.ExecutionJournal(environment.root)
    try:
        one = store.open_session(
            endpoint=journal.PAPER_ENDPOINT,
            broker_id="FinRL-id",
            alias="FinRL",
            day=DAY,
            config_hash="config",
            targets={"A": 0.5},
            snapshot={"positions": broker.get_positions()},
            attempt=journal.new_attempt(environment.root),
        )
        with pytest.raises(TimeoutError):
            broker.execute_portfolio_rebalance(
                {"A": 0.5},
                "FinRL",
                market_closed_action="next_open",
                execution_session=one,
            )
        assert [
            r[0] for r in store.conn.execute("SELECT phase FROM order_intents")
        ] == ["sell"]
        assert (
            len([c for c in broker._api_request.call_args_list if c.args[0] == "POST"])
            == 1
        )
    finally:
        store.close()


@pytest.mark.parametrize(
    "failure",
    ["bulk", "asset_read", "asset_malformed", "clock_read", "clock_malformed"],
)
def test_required_helper_reads_hold_before_any_replacement(
    environment, monkeypatch, failure
):
    broker = manager(monkeypatch, "FinRL")
    broker.get_positions.return_value = [
        {"symbol": "A", "qty": "10", "market_value": "1000"}
    ]
    if failure.startswith("clock"):
        monkeypatch.setattr(
            broker, "_is_market_open", AlpacaManager._is_market_open.__get__(broker)
        )
    else:
        monkeypatch.setattr(
            broker,
            "_is_symbol_tradable",
            AlpacaManager._is_symbol_tradable.__get__(broker),
        )
        monkeypatch.setattr(
            broker,
            "_is_symbol_fractionable",
            AlpacaManager._is_symbol_fractionable.__get__(broker),
        )
        broker._assets_loaded = failure != "bulk"

    def failed_read(method, path, **kwargs):
        assert method == "GET"
        if failure.endswith("malformed"):
            return {"symbol": "A"}
        raise RuntimeError("required read unavailable")

    broker._api_request.side_effect = failed_read
    store = journal.ExecutionJournal(environment.root)
    try:
        one = store.open_session(
            endpoint=journal.PAPER_ENDPOINT,
            broker_id="FinRL-id",
            alias="FinRL",
            day=DAY,
            config_hash="config",
            targets={"A": 0.5},
            snapshot={"positions": broker.get_positions()},
            attempt=journal.new_attempt(environment.root),
        )
        with pytest.raises(ValueError, match="metadata|clock"):
            broker.execute_portfolio_rebalance(
                {"A": 0.5},
                "FinRL",
                market_closed_action="next_open",
                execution_session=one,
            )
        assert not store.conn.execute("SELECT 1 FROM order_intents").fetchone()
        assert all(c.args[0] == "GET" for c in broker._api_request.call_args_list)
        broker.cancel_all_orders.assert_not_called()
    finally:
        store.close()


def test_disabled_strategy_output_stays_rooted_when_caller_cwd_differs(
    monkeypatch, tmp_path
):
    monkeypatch.setenv(journal.GATE, "false")
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.chdir(outside)

    def strategy(cmd, **kwargs):
        output = Path(cmd[cmd.index("--json-output") + 1])
        assert output.parent == tmp_path / "logs"
        output.write_text(
            json.dumps(
                {
                    "target_weights": {"A": 0.5},
                    "cash_weight": 0.5,
                    "regime_state": "risk_on",
                    "active_groups": [],
                    "ranked_groups": [],
                    "fallback_status": False,
                    "audit_file_path": "fixture-audit.json",
                }
            )
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", strategy)
    assert runner.get_ar_weights("fixture.yaml", DAY) == {"A": 0.5}
