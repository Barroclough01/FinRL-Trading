import pandas as pd

from src.strategies.rl_candidate_strategy import (
    RLCandidateStrategy,
    load_rl_candidate_weights,
)
from src.strategies.base_strategy import StrategyConfig


def test_load_rl_candidate_weights_uses_latest_date(tmp_path):
    sample = tmp_path / "drl_weight.csv"
    pd.DataFrame(
        [
            {"trade_date": "2024-01-05", "gvkey": "A", "weights": 0.4},
            {"trade_date": "2024-01-05", "gvkey": "B", "weights": 0.6},
            {"trade_date": "2024-01-12", "gvkey": "C", "weights": 1.0},
        ]
    ).to_csv(sample, index=False)

    weights = load_rl_candidate_weights(sample, target_date="2024-01-12")

    assert list(weights.keys()) == ["C"]
    assert weights["C"] == 1.0


def test_rl_candidate_strategy_returns_target_weight_contract(tmp_path):
    sample = tmp_path / "drl_weight.csv"
    pd.DataFrame(
        [
            {"trade_date": "2024-01-05", "gvkey": "A", "weights": 0.2},
            {"trade_date": "2024-01-05", "gvkey": "B", "weights": 0.8},
        ]
    ).to_csv(sample, index=False)

    strategy = RLCandidateStrategy(
        StrategyConfig(name="rl_candidate"), weights_path=sample
    )
    result = strategy.generate_weights({}, target_date="2024-01-05")

    assert result.strategy_name == "rl_candidate"
    assert list(result.weights.columns) == ["symbol", "weight"]
    assert result.weights["weight"].sum() == 1.0
    assert set(result.metadata["symbols"]) == {"A", "B"}
