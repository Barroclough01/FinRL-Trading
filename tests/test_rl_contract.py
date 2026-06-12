import json
from pathlib import Path

from src.strategies.rl_contract import DEFAULT_RL_CONTRACT_PATH, load_rl_contract


def test_rl_contract_file_exists_and_contains_expected_sections():
    contract = load_rl_contract(DEFAULT_RL_CONTRACT_PATH)

    assert isinstance(contract, dict)
    assert contract["contract_version"] == 1
    assert contract["action_space"] == "target_portfolio_weights"
    assert "observation_space" in contract
    assert contract["reward_function"]["name"] == "risk_adjusted_weighted_return"
    assert contract["transaction_cost_model"]["name"] == "linear_pct_notional"


def test_rl_contract_matches_documented_fields():
    contract_path = Path("src/strategies/rl_contract.json")
    data = json.loads(contract_path.read_text())

    assert set(data.keys()) >= {
        "contract_version",
        "observation_space",
        "action_space",
        "reward_function",
        "transaction_cost_model",
        "turnover_penalty",
        "drawdown_penalty",
        "train_eval_split",
        "walk_forward_windows",
    }
    assert isinstance(data["observation_space"], list)
    assert isinstance(data["reward_function"]["penalties"], dict)
