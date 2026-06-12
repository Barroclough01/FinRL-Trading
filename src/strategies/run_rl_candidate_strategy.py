#!/usr/bin/env python3
"""Run the RL candidate strategy using the existing target-weight contract."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.strategies.base_strategy import StrategyConfig
from src.strategies.rl_candidate_strategy import RLCandidateStrategy
from src.strategies.rl_contract import DEFAULT_RL_CONTRACT_PATH, load_rl_contract


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate RL candidate target weights in the same contract as AR/FinRL"
    )
    parser.add_argument(
        "--weights-path",
        default="results/drl_weight.csv",
        help="Path to the RL candidate weight export (default: results/drl_weight.csv)",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Optional decision date to select the latest RL weights on or before this date.",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional path to write structured JSON output in the same format as the AR strategy.",
    )
    args = parser.parse_args()

    strategy = RLCandidateStrategy(
        StrategyConfig(name="rl_candidate"),
        weights_path=args.weights_path,
    )
    result = strategy.generate_weights({}, target_date=args.date)

    weights = dict(zip(result.weights["symbol"], result.weights["weight"]))
    total = float(sum(weights.values()))
    print(f"RL candidate weights ({len(weights)} symbols, total={total:.2%})")
    for symbol, weight in sorted(
        weights.items(), key=lambda item: item[1], reverse=True
    ):
        print(f"  {symbol:8s}: {weight:.2%}")

    if args.json_output:
        out_path = Path(args.json_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "target_weights": weights,
            "cash_weight": 0.0,
            "regime_state": "rl_candidate",
            "active_groups": [],
            "ranked_groups": [],
            "fallback_status": False,
            "audit_file_path": str(out_path.resolve()),
            "rl_contract": load_rl_contract(DEFAULT_RL_CONTRACT_PATH),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Structured JSON output saved to: {out_path}")


if __name__ == "__main__":
    main()
