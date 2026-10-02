"""Required checks and audit writes gate the single final outcome."""

import builtins
import json
import sqlite3
from pathlib import Path
from unittest.mock import Mock

import pytest

import run_paper_trading as runner
import track_metrics

DAY = "2026-09-25"
ACCOUNT = {"name": "FinRL", "config": "fixture.yaml"}
TARGET = {"A": 0.5}


@pytest.fixture
def main_environment(monkeypatch):
    monkeypatch.delenv("TRADING_DISABLED", raising=False)
    monkeypatch.setattr("sys.argv", ["run_paper_trading.py", "--date", DAY])
    monkeypatch.setattr(runner, "load_accounts_from_env", lambda: [ACCOUNT])
    result = {"account": "FinRL", "target_weights": TARGET, "orders_failed": 0}
    monkeypatch.setattr(runner, "run_account", Mock(return_value=result))
    monkeypatch.setattr(runner, "run_metrics_tracker", lambda *a: (True, None))
    monkeypatch.setattr(runner, "run_rl_tracker", lambda *a: (True, None))
    monkeypatch.setattr(runner, "run_post_run_sanity_checks", lambda *a: [])
    notify = Mock()
    monkeypatch.setattr(runner, "notify_status", notify)
    monkeypatch.setattr(runner, "get_target_weights", Mock(return_value=TARGET))
    return notify


def prepare_parity(*, pending=False, quantity_mismatch=False):
    positions = (
        []
        if pending
        else [{"symbol": "A", "qty": 1.0, "market_value": 50.0, "actual_weight": 0.5}]
    )
    record = {
        "target_weights": TARGET,
        "post_trade_positions": positions,
        "equity": 100,
        "cash": 50,
        "submitted_orders": [
            {"id": "paper-order", "status": "accepted", "time_in_force": "day"}
        ]
        if pending
        else [],
    }
    runner.save_strategy_decision(DAY, "FinRL", record)
    conn = sqlite3.connect("data/finrl_trading.db")
    track_metrics.init_db(conn)
    snapshot_positions = [dict(pos) for pos in positions]
    if quantity_mismatch:
        snapshot_positions[0]["qty"] = 2.0
    track_metrics.record_snapshot(
        conn,
        DAY,
        ACCOUNT,
        {
            "portfolio_value": 100,
            "equity": 100,
            "cash": 50,
            "positions": snapshot_positions,
        },
        {},
        TARGET,
    )
    conn.close()


def assert_failed(notify, expected):
    assert notify.call_count == 1
    payload = notify.call_args.args[1]
    assert payload["status"] == "failed"
    assert expected in json.dumps(payload)


def test_parity_exception_precedes_the_only_failure_notification(
    main_environment, monkeypatch
):
    def fail_parity(*args):
        assert not main_environment.called
        raise RuntimeError("replay unavailable: inspect price fixture")

    monkeypatch.setattr(runner, "run_parity_checks", fail_parity)
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 1
    assert_failed(main_environment, "replay unavailable")


@pytest.mark.parametrize("failure", ["replay", "quantity", "check_exception"])
def test_real_parity_failures_are_actionable_and_gate_success(
    main_environment, monkeypatch, failure
):
    prepare_parity(quantity_mismatch=failure == "quantity")
    if failure == "replay":
        monkeypatch.setattr(runner, "get_target_weights", lambda *a, **k: {"A": 0.4})
    elif failure == "check_exception":
        monkeypatch.setattr(
            runner, "get_target_weights", Mock(side_effect=ValueError("stale price"))
        )
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 1
    assert_failed(main_environment, "FinRL")
    report = json.loads(Path(f"logs/parity_check_{DAY}.json").read_text())
    assert not report["accounts"]["FinRL"]["reconciled_successfully"]
    if failure == "check_exception":
        assert "stale price" in json.dumps(report)


def test_pending_day_order_with_no_positions_is_valid_but_not_reconciled(
    main_environment, caplog
):
    prepare_parity(pending=True)
    runner.main()
    main_environment.assert_not_called()
    report = json.loads(Path(f"logs/parity_check_{DAY}.json").read_text())
    account = report["accounts"]["FinRL"]
    assert account["execution_pending"]
    assert account["execution_ok"]
    assert account["dashboard_ok"]
    assert not account["reconciled_successfully"]
    assert "pending" in caplog.text.lower()


def test_pending_does_not_mask_a_real_parity_mismatch(main_environment, monkeypatch):
    prepare_parity(pending=True)
    monkeypatch.setattr(runner, "get_target_weights", lambda *a, **k: {"A": 0.4})
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 1
    assert_failed(main_environment, "Determinism")


def test_success_is_notified_only_after_required_parity(main_environment, monkeypatch):
    prepare_parity()
    original = runner.run_parity_checks

    def parity(*args):
        assert not main_environment.called
        return original(*args)

    monkeypatch.setattr(runner, "run_parity_checks", parity)
    runner.main()
    assert main_environment.call_count == 1
    assert main_environment.call_args.args[1]["status"] == "ok"


def deny_write(monkeypatch, filename):
    original = builtins.open

    def guarded(path, mode="r", *args, **kwargs):
        if Path(path).name == filename and ("w" in mode or "a" in mode):
            raise OSError(f"disk full for {filename}")
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded)


@pytest.mark.parametrize(
    "filename,stage",
    [
        (f"execution_{DAY}.json", "execution"),
        (f"parity_check_{DAY}.json", "parity"),
        (f"pre_trade_validation_{DAY}.json", "validation"),
        (f"reconciliation_{DAY}.json", "reconciliation"),
        ("strategy_decisions.jsonl", "decision"),
    ],
)
def test_required_audit_write_failure_gates_final_outcome(
    main_environment, monkeypatch, filename, stage
):
    prepare_parity()
    deny_write(monkeypatch, filename)
    if stage in ("validation", "reconciliation", "decision"):

        def account(*args):
            if stage == "validation":
                runner.save_validation_result(DAY, "FinRL", True, None, None, None)
            elif stage == "reconciliation":
                runner.save_reconciliation_report(DAY, "FinRL", {})
            else:
                runner.save_strategy_decision(DAY, "FinRL", {"target_weights": TARGET})
            return {"account": "FinRL", "target_weights": TARGET}

        monkeypatch.setattr(runner, "run_account", account)
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 1
    assert_failed(main_environment, filename)


def test_required_decision_db_failure_still_attempts_jsonl_mirror(monkeypatch):
    Path("data").mkdir()
    monkeypatch.setattr(
        runner.sqlite3,
        "connect",
        Mock(side_effect=sqlite3.OperationalError("read only")),
    )
    with pytest.raises(RuntimeError, match="FinRL.*read only"):
        runner.save_strategy_decision(DAY, "FinRL", {"submitted_orders": [{"id": "x"}]})
    mirror = json.loads(Path("logs/strategy_decisions.jsonl").read_text())
    assert mirror["submitted_orders"] == [{"id": "x"}]


def test_required_parity_db_write_failure_gates_final_outcome(
    main_environment, monkeypatch
):
    prepare_parity()
    original = runner.sqlite3.connect

    def connect(*args, **kwargs):
        real = original(*args, **kwargs)
        wrapped = Mock(wraps=real)

        def execute(sql, *parameters):
            if sql.startswith("UPDATE strategy_decisions"):
                raise sqlite3.OperationalError("read only parity database")
            return real.execute(sql, *parameters)

        wrapped.execute.side_effect = execute
        return wrapped

    monkeypatch.setattr(runner.sqlite3, "connect", connect)
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 1
    assert_failed(main_environment, "read only parity database")


def test_sanity_exception_is_a_single_final_failure(main_environment, monkeypatch):
    prepare_parity()
    monkeypatch.setattr(
        runner,
        "run_post_run_sanity_checks",
        Mock(side_effect=OSError("cannot read DB")),
    )
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 1
    assert_failed(main_environment, "cannot read DB")


def test_pending_and_rejected_orders_cannot_be_reported_as_success(main_environment):
    prepare_parity(pending=True)
    conn = sqlite3.connect("data/finrl_trading.db")
    conn.execute(
        "UPDATE strategy_decisions SET submitted_orders=?",
        (json.dumps([{"status": "accepted"}, {"status": "rejected"}]),),
    )
    conn.commit()
    conn.close()
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 1
    assert_failed(main_environment, "rejected/failed")


def test_malformed_execution_evidence_fails_with_the_read_error(main_environment):
    prepare_parity()
    conn = sqlite3.connect("data/finrl_trading.db")
    conn.execute("UPDATE strategy_decisions SET post_trade_positions='invalid-json'")
    conn.commit()
    conn.close()
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 1
    assert_failed(main_environment, "Cannot read execution evidence")


def test_pending_orders_never_claim_completed_reconciliation():
    result = runner.reconcile_post_trade(
        DAY,
        "FinRL",
        {
            "target_weights": TARGET,
            "post_trade_positions": [{"symbol": "A", "market_value": 50}],
            "equity": 100,
            "submitted_orders": [{"status": "accepted", "time_in_force": "day"}],
        },
    )
    assert result["orders_summary"]["open"] == 1
    assert not result["reconciled_successfully"]
    assert not result["discrepancies_found"]


def test_validation_audit_failure_stops_before_submission(monkeypatch):
    executor = Mock()
    monkeypatch.setattr(runner, "get_target_weights", lambda *a, **k: TARGET)
    monkeypatch.setattr(runner, "get_executor_for_account", lambda *a: executor)
    monkeypatch.setattr(runner, "allows_cash_fallback", lambda *a: False)
    monkeypatch.setattr(
        runner, "validate_pre_trade", lambda *a, **k: (True, None, None, None)
    )
    deny_write(monkeypatch, f"pre_trade_validation_{DAY}.json")
    with pytest.raises(RuntimeError, match="required validation audit"):
        runner.run_account(ACCOUNT, DAY, dry_run=False)
    executor.alpaca.execute_portfolio_rebalance.assert_not_called()


def test_dry_run_parity_uses_preview_weights_and_preserves_decision_db(monkeypatch):
    prepare_parity()
    db_path = Path("data/finrl_trading.db")
    before = db_path.read_bytes()
    monkeypatch.setattr(runner, "get_target_weights", lambda *a, **k: TARGET)
    report = runner.run_parity_checks(
        DAY, [ACCOUNT], [{"account": "FinRL", "weights": TARGET}], dry_run=True
    )
    assert report["accounts"]["FinRL"]["reconciled_successfully"]
    assert db_path.read_bytes() == before


@pytest.mark.parametrize("stage", ["run_metrics_tracker", "run_rl_tracker"])
def test_required_tracker_exception_becomes_final_failure(
    main_environment, monkeypatch, stage
):
    prepare_parity()
    monkeypatch.setattr(
        runner, stage, Mock(side_effect=OSError("required tracker cannot start"))
    )
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 1
    assert_failed(main_environment, "required tracker cannot start")


def test_missing_parity_database_is_a_final_failure(main_environment):
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 1
    assert_failed(main_environment, "data/finrl_trading.db")


def test_failed_submission_count_cannot_be_notified_as_ok(
    main_environment, monkeypatch
):
    prepare_parity()
    monkeypatch.setattr(
        runner,
        "run_account",
        Mock(
            return_value={
                "account": "FinRL",
                "target_weights": TARGET,
                "orders_failed": 1,
            }
        ),
    )
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 1
    assert_failed(main_environment, "submission(s) failed")
