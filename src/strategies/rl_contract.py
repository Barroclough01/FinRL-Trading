from __future__ import annotations

import json
from pathlib import Path

DEFAULT_RL_CONTRACT_PATH = Path(__file__).with_name("rl_contract.json")


def load_rl_contract(path: str | Path = DEFAULT_RL_CONTRACT_PATH) -> dict:
    """Load the RL contract metadata used by the offline RL pipeline."""
    contract_path = Path(path)
    with contract_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_rl_contract(
    contract: dict, path: str | Path = DEFAULT_RL_CONTRACT_PATH
) -> None:
    """Persist the RL contract metadata to disk."""
    contract_path = Path(path)
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    with contract_path.open("w", encoding="utf-8") as handle:
        json.dump(contract, handle, indent=2)
        handle.write("\n")
