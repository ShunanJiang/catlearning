"""Momentum-based trading strategy."""

import pandas as pd
import numpy as np
from typing import Dict, Any

from .base_strategy import BaseStrategy, Signal
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class MomentumStrategy(BaseStrategy):
    """
    Momentum trading strategy based on price trends and indicators.

    Buys when momentum is strong and positive, sells when momentum weakens.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize momentum strategy.

        Args:
            config: Strategy configuration with:
                - lookback_period: Period for momentum calculation
                - entry_threshold: Threshold for entry signal
                - exit_threshold: Threshold for exit signal
        """
        super().__init__("Momentum", config)
        self.lookback_period = config.get('lookback_period', 20)
        self.entry_threshold = config.get('entry_threshold', 0.02)
        self.exit_threshold = config.get('exit_threshold', -0.01)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate momentum-based trading signals.

        Strategy logic:
        - Buy when: price > SMA and momentum > entry_threshold and RSI < 70
        - Sell when: price < SMA or momentum < exit_threshold or RSI > 70

        Args:
            data: Market data with technical indicators

        Returns:
            Series of signals
        """
        signals = pd.Series(0, index=data.index)

        # Calculate momentum if not already in data
        if f'momentum_{self.lookback_period}' not in data.columns:
            momentum = data['close'] / data['close'].shift(self.lookback_period) - 1
        else:
            momentum = data[f'momentum_{self.lookback_period}']

        # Get other indicators
        price = data['close']
        sma = data.get('sma_50', data['close'].rolling(50).mean())
        rsi = data.get('rsi', 50)  # Default to neutral if not available

        # Buy conditions
        buy_conditions = (
            (price > sma) &
            (momentum > self.entry_threshold) &
            (rsi < 70)
        )

        # Sell conditions
        sell_conditions = (
            (price < sma) |
            (momentum < self.exit_threshold) |
            (rsi > 70)
        )

        # Generate signals
        signals[buy_conditions] = 1
        signals[sell_conditions] = -1

        # Log signal statistics
        buy_count = (signals == 1).sum()
        sell_count = (signals == -1).sum()
        logger.info(f"Momentum strategy generated {buy_count} buy and {sell_count} sell signals")

        return signals


class DualMomentumStrategy(BaseStrategy):
    """
    Dual momentum strategy combining absolute and relative momentum.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize dual momentum strategy."""
        super().__init__("DualMomentum", config)
        self.lookback_period = config.get('lookback_period', 60)
        self.benchmark_symbol = config.get('benchmark_symbol', 'SPY')

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate dual momentum signals.

        Args:
            data: Market data (should include benchmark data if available)

        Returns:
            Series of signals
        """
        signals = pd.Series(0, index=data.index)

        # Absolute momentum
        returns = data['close'].pct_change(self.lookback_period)

        # Relative momentum (if benchmark data available)
        # For now, use simple absolute momentum
        # In practice, you'd compare with benchmark returns

        # Buy when positive momentum, sell when negative
        signals[returns > 0] = 1
        signals[returns < 0] = -1

        logger.info(f"Dual momentum generated {(signals == 1).sum()} buy signals")

        return signals


class TrendFollowingStrategy(BaseStrategy):
    """
    Trend following strategy using moving average crossovers.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize trend following strategy."""
        super().__init__("TrendFollowing", config)
        self.fast_period = config.get('fast_period', 20)
        self.slow_period = config.get('slow_period', 50)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trend following signals based on MA crossovers.

        Args:
            data: Market data

        Returns:
            Series of signals
        """
        signals = pd.Series(0, index=data.index)

        # Calculate moving averages
        fast_ma = data['close'].rolling(self.fast_period).mean()
        slow_ma = data['close'].rolling(self.slow_period).mean()

        # Golden cross (buy) and death cross (sell)
        signals[fast_ma > slow_ma] = 1
        signals[fast_ma < slow_ma] = -1

        # Only signal on crossovers (change in signal)
        signal_changes = signals.diff()

        logger.info(f"Trend following generated {(signal_changes == 2).sum()} crossover signals")

        return signals
