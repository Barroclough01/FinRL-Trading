"""
Base Strategy Module
===================

Base classes and interfaces for trading strategies:
- Abstract strategy class
- Strategy configuration
- Common strategy utilities
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """Configuration for trading strategies."""
    name: str
    description: str = ""
    parameters: Dict[str, Any] = None
    risk_limits: Dict[str, float] = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.risk_limits is None:
            self.risk_limits = {
                'max_weight': 0.1,  # Maximum weight per stock
                'max_sector_weight': 0.3,  # Maximum weight per sector
                'max_turnover': 0.5  # Maximum portfolio turnover
            }


@dataclass
class StrategyResult:
    """Result from strategy execution."""
    strategy_name: str
    weights: pd.DataFrame
    metadata: Dict[str, Any] = None
    execution_time: datetime = None

    def __post_init__(self):
        if self.execution_time is None:
            self.execution_time = datetime.now()
        if self.metadata is None:
            self.metadata = {}


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    def __init__(self, config: StrategyConfig):
        """
        Initialize strategy.

        Args:
            config: Strategy configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def generate_weights(self, data: Dict[str, pd.DataFrame],
                        **kwargs) -> StrategyResult:
        """
        Generate portfolio weights based on input data.

        Args:
            data: Dictionary containing relevant data (prices, fundamentals, etc.)
            **kwargs: Additional strategy-specific parameters

        Returns:
            StrategyResult with generated weights
        """
        pass

    def validate_weights(self, weights: pd.DataFrame) -> bool:
        """
        Validate generated weights against risk limits.

        Args:
            weights: Portfolio weights DataFrame

        Returns:
            True if weights are valid, False otherwise
        """
        try:
            # Check for negative weights (short selling not allowed)
            if (weights['weight'] < 0).any():
                self.logger.warning("Negative weights detected")
                return False

            # Check maximum weight per stock
            max_weight = self.config.risk_limits.get('max_weight', 0.1)
            if (weights['weight'] > max_weight).any():
                self.logger.warning(f"Weights exceed maximum allowed weight of {max_weight}")
                return False

            # Check sector concentration if sector data available
            if 'sector' in weights.columns:
                sector_weights = weights.groupby('sector')['weight'].sum()
                max_sector_weight = self.config.risk_limits.get('max_sector_weight', 0.3)
                if (sector_weights > max_sector_weight).any():
                    self.logger.warning(f"Sector weights exceed maximum allowed weight of {max_sector_weight}")
                    return False

            # Check total weight sum (should be close to 1)
            total_weight = weights['weight'].sum()
            if not np.isclose(total_weight, 1.0, atol=1e-6):
                self.logger.warning(f"Total weight sum is {total_weight}, should be close to 1.0")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating weights: {e}")
            return False

    def normalize_weights(self, weights: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize weights to sum to 1.0.

        Args:
            weights: Portfolio weights DataFrame

        Returns:
            Normalized weights DataFrame
        """
        total_weight = weights['weight'].sum()
        if total_weight > 0:
            weights = weights.copy()
            weights['weight'] = weights['weight'] / total_weight
        return weights

    def apply_risk_limits(self, weights: pd.DataFrame) -> pd.DataFrame:
        """
        Apply risk limits to weights.

        Args:
            weights: Portfolio weights DataFrame

        Returns:
            Risk-adjusted weights DataFrame
        """
        weights = weights.copy()

        # Apply maximum weight per stock
        max_weight = self.config.risk_limits.get('max_weight', 0.1)
        weights['weight'] = np.minimum(weights['weight'], max_weight)

        # Re-normalize after applying limits
        weights = self.normalize_weights(weights)

        return weights

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information."""
        return {
            'name': self.config.name,
            'description': self.config.description,
            'parameters': self.config.parameters,
            'risk_limits': self.config.risk_limits,
            'strategy_type': self.__class__.__name__
        }


class EqualWeightStrategy(BaseStrategy):
    """Equal weight strategy - assigns equal weights to all stocks."""

    def generate_weights(self, data: Dict[str, pd.DataFrame],
                        date: str = None, **kwargs) -> StrategyResult:
        """
        Generate equal weights for all available stocks.

        Args:
            data: Dictionary containing stock data
            date: Target date for weight generation
            **kwargs: Additional parameters

        Returns:
            StrategyResult with equal weights
        """
        self.logger.info("Generating equal weights")

        # Get stock universe
        if 'prices' in data:
            stocks = data['prices']
        elif 'fundamentals' in data:
            stocks = data['fundamentals']
        else:
            raise ValueError("No stock data provided")

        # Filter by date if specified
        if date and 'datadate' in stocks.columns:
            target_date = pd.to_datetime(date)
            stocks = stocks[pd.to_datetime(stocks['datadate']) <= target_date]

        # Get unique stocks
        if 'gvkey' in stocks.columns:
            unique_stocks = stocks['gvkey'].unique()
        elif 'ticker' in stocks.columns:
            unique_stocks = stocks['ticker'].unique()
        else:
            raise ValueError("No stock identifier column found")

        # Generate equal weights
        n_stocks = len(unique_stocks)
        if n_stocks == 0:
            self.logger.warning("No stocks available for weighting")
            weights_df = pd.DataFrame(columns=['gvkey', 'weight'])
        else:
            weight = 1.0 / n_stocks
            weights_df = pd.DataFrame({
                'gvkey': unique_stocks,
                'weight': [weight] * n_stocks
            })

        # Add date column if provided
        if date:
            weights_df['date'] = date

        result = StrategyResult(
            strategy_name=self.config.name,
            weights=weights_df,
            metadata={
                'n_stocks': n_stocks,
                'equal_weight': weight if n_stocks > 0 else 0
            }
        )

        self.logger.info(f"Generated equal weights for {n_stocks} stocks")
        return result


class MarketCapWeightStrategy(BaseStrategy):
    """Market capitalization weighted strategy."""

    def generate_weights(self, data: Dict[str, pd.DataFrame],
                        date: str = None, **kwargs) -> StrategyResult:
        """
        Generate market cap weighted portfolio.

        Args:
            data: Dictionary containing stock data with market cap info
            date: Target date for weight generation
            **kwargs: Additional parameters

        Returns:
            StrategyResult with market cap weights
        """
        self.logger.info("Generating market cap weights")

        if 'fundamentals' not in data:
            raise ValueError("Fundamental data required for market cap weighting")

        stocks = data['fundamentals']

        # Filter by date if specified
        if date and 'datadate' in stocks.columns:
            target_date = pd.to_datetime(date)
            stocks = stocks[pd.to_datetime(stocks['datadate']) <= target_date]

        # Check for market cap column
        market_cap_col = None
        for col in ['market_cap', 'marketcap', 'cap']:
            if col in stocks.columns:
                market_cap_col = col
                break

        if market_cap_col is None:
            self.logger.warning("No market cap column found, using equal weights")
            equal_strategy = EqualWeightStrategy(self.config)
            return equal_strategy.generate_weights(data, date, **kwargs)

        # Filter out stocks with invalid market cap
        valid_stocks = stocks[stocks[market_cap_col] > 0].copy()

        if len(valid_stocks) == 0:
            self.logger.warning("No stocks with valid market cap found")
            weights_df = pd.DataFrame(columns=['gvkey', 'weight'])
        else:
            # Calculate weights
            total_cap = valid_stocks[market_cap_col].sum()
            valid_stocks['weight'] = valid_stocks[market_cap_col] / total_cap

            weights_df = valid_stocks[['gvkey', 'weight']].copy()

        # Add date column if provided
        if date:
            weights_df['date'] = date

        result = StrategyResult(
            strategy_name=self.config.name,
            weights=weights_df,
            metadata={
                'n_stocks': len(weights_df),
                'total_market_cap': total_cap if len(valid_stocks) > 0 else 0
            }
        )

        self.logger.info(f"Generated market cap weights for {len(weights_df)} stocks")
        return result


# Strategy registry
STRATEGY_REGISTRY = {
    'equal_weight': EqualWeightStrategy,
    'market_cap_weight': MarketCapWeightStrategy,
}


def create_strategy(strategy_type: str, config: StrategyConfig) -> BaseStrategy:
    """
    Factory function to create strategy instances.

    Args:
        strategy_type: Type of strategy to create
        config: Strategy configuration

    Returns:
        Strategy instance

    Raises:
        ValueError: If strategy type is not registered
    """
    if strategy_type not in STRATEGY_REGISTRY:
        available = list(STRATEGY_REGISTRY.keys())
        raise ValueError(f"Unknown strategy type: {strategy_type}. Available: {available}")

    strategy_class = STRATEGY_REGISTRY[strategy_type]
    return strategy_class(config)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create sample data
    sample_data = {
        'fundamentals': pd.DataFrame({
            'gvkey': ['001', '002', '003'],
            'datadate': ['2023-01-01', '2023-01-01', '2023-01-01'],
            'market_cap': [1000000, 2000000, 3000000]
        })
    }

    # Test equal weight strategy
    config = StrategyConfig(
        name="Equal Weight Test",
        description="Test equal weight strategy"
    )

    strategy = EqualWeightStrategy(config)
    result = strategy.generate_weights(sample_data)

    print(f"Strategy: {result.strategy_name}")
    print(f"Weights:\n{result.weights}")
    print(f"Metadata: {result.metadata}")
