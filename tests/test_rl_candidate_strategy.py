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

    # Create dummy daily file for C so it doesn't get skipped
    (tmp_path / "C_daily.csv").touch()

    weights = load_rl_candidate_weights(
        sample, target_date="2024-01-12", data_dir=tmp_path
    )

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

    # Create dummy daily files for A and B so they don't get skipped
    (tmp_path / "A_daily.csv").touch()
    (tmp_path / "B_daily.csv").touch()

    strategy = RLCandidateStrategy(
        StrategyConfig(name="rl_candidate"), weights_path=sample, data_dir=tmp_path
    )
    result = strategy.generate_weights({}, target_date="2024-01-05")

    assert result.strategy_name == "rl_candidate"
    assert list(result.weights.columns) == ["symbol", "weight"]
    assert result.weights["weight"].sum() == 1.0
    assert set(result.metadata["symbols"]) == {"A", "B"}
