from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.strategies.base_strategy import BaseStrategy, StrategyConfig, StrategyResult


def load_rl_candidate_weights(
    weights_path,
    target_date: str | None = None,
    data_dir: str | None = None,
) -> dict[str, float]:
    """Load the latest RL-generated target weights for a given date.

    Only keeps symbols that have local price data available in ``data_dir`` so
    the paper-trading validation path can execute without failing on missing
    CSV inputs.
    """
    frame = pd.read_csv(weights_path)
    if frame.empty:
        return {}

    frame = frame.copy()
    if "trade_date" in frame.columns:
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
        if target_date is not None:
            target_ts = pd.Timestamp(target_date)
            frame = frame[frame["trade_date"] <= target_ts]
        if frame.empty:
            return {}
        latest_date = frame["trade_date"].max()
        frame = frame[frame["trade_date"] == latest_date]

    weights = {}
    data_dir = Path(data_dir) if data_dir is not None else Path("data/fmp_daily")
    data_dir.mkdir(parents=True, exist_ok=True)

    for _, row in frame.iterrows():
        symbol = str(
            row.get("gvkey") or row.get("tic") or row.get("symbol") or ""
        ).strip()
        if not symbol:
            continue

        csv_path = data_dir / f"{symbol}_daily.csv"
        if not csv_path.exists():
            continue

        value = float(row.get("weights", 0.0) or 0.0)
        weights[symbol] = value

    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    return weights


class RLCandidateStrategy(BaseStrategy):
    """Small adapter that turns RL output into the same target-weight contract as AR/FinRL."""

    def __init__(
        self,
        config: StrategyConfig,
        weights_path: str | None = None,
        data_dir: str | None = None,
    ):
        super().__init__(config)
        self.weights_path = weights_path or "results/drl_weight.csv"
        self.data_dir = data_dir or "data/fmp_daily"

    def generate_weights(
        self, data: dict, target_date: str | None = None
    ) -> StrategyResult:
        weights = load_rl_candidate_weights(
            self.weights_path,
            target_date=target_date,
            data_dir=self.data_dir,
        )
        frame = pd.DataFrame(
            [(symbol, weight) for symbol, weight in sorted(weights.items())],
            columns=["symbol", "weight"],
        )
        return StrategyResult(
            strategy_name=self.config.name,
            weights=frame,
            metadata={
                "source": self.weights_path,
                "target_date": target_date,
                "symbols": list(weights.keys()),
                "weight_sum": float(frame["weight"].sum()),
            },
        )
