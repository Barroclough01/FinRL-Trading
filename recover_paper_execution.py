#!/usr/bin/env python3
"""Explicit GET-only paper receipt recovery; never resume or cancel orders."""

import argparse
from pathlib import Path

from src.trading import execution_journal as journal


def recover_account(manager, account_name: str, run_date: str, root: Path) -> dict:
    journal.require_execution_host(root)
    if not journal.enabled():
        raise ValueError("Recovery integration is disabled")
    if account_name.upper() == "RL":
        raise ValueError("RL remains offline")
    manager._journal_transport = True
    endpoint, broker_id = journal.identity(manager, account_name)
    attempt = journal.new_attempt(root)
    with journal.account_lock(root, endpoint, broker_id):
        store = journal.ExecutionJournal(root, create=False)
        try:
            session = store.open_session(
                endpoint=endpoint,
                broker_id=broker_id,
                alias=account_name,
                day=run_date,
                config_hash="",
                targets={},
                snapshot={},
                attempt=attempt,
                recovery=True,
            )
            return session.recover(manager, account_name)
        finally:
            store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="Original NYSE signal date")
    parser.add_argument(
        "--account", required=True, help="Configured paper account alias"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    journal.require_execution_host(root)
    from dotenv import load_dotenv

    load_dotenv(root / ".env")
    if not journal.enabled():
        parser.error(f"Recovery is disabled; {journal.GATE} must be explicitly enabled")
    if args.account.upper() == "RL":
        parser.error("RL remains offline")
    # Import only after the host/gate checks; --help needs no credentials or runtime.
    from run_paper_trading import (
        get_executor_for_account,
        load_accounts_from_env,
        resolve_run_date,
    )

    accounts = [a for a in load_accounts_from_env() if a["name"] == args.account]
    if len(accounts) != 1:
        parser.error("Account alias must resolve to exactly one configured account")
    day = resolve_run_date(args.date)
    if day != args.date:
        parser.error("Recovery date must equal the original resolved signal date")
    try:
        result = recover_account(
            get_executor_for_account(accounts[0]).alpaca, args.account, day, root
        )
    except Exception as exc:
        raise SystemExit(f"Recovery held for {args.account} {day}: {exc}") from exc
    print(
        f"Receipt observation {result['attempt_id']}: "
        f"pending={result['execution_pending']}; failures={result['failures']}"
    )
    if result["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
