"""Mean reversion trading strategy."""

import pandas as pd
import numpy as np
from typing import Dict, Any

from .base_strategy import BaseStrategy, Signal
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy using Bollinger Bands and statistical measures.

    Buys when price is oversold (below lower band), sells when overbought (above upper band).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize mean reversion strategy.

        Args:
            config: Strategy configuration with:
                - lookback_period: Period for mean calculation
                - entry_std: Standard deviations for entry
                - exit_std: Standard deviations for exit
        """
        super().__init__("MeanReversion", config)
        self.lookback_period = config.get('lookback_period', 30)
        self.entry_std = config.get('entry_std', 2.0)
        self.exit_std = config.get('exit_std', 0.5)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate mean reversion signals.

        Strategy logic:
        - Buy when price < mean - entry_std * std (oversold)
        - Sell when price > mean + entry_std * std (overbought)
        - Exit when price returns to mean ± exit_std * std

        Args:
            data: Market data

        Returns:
            Series of signals
        """
        signals = pd.Series(0, index=data.index)

        # Calculate mean and standard deviation
        price = data['close']
        mean = price.rolling(self.lookback_period).mean()
        std = price.rolling(self.lookback_period).std()

        # Calculate bands
        upper_entry = mean + (self.entry_std * std)
        lower_entry = mean - (self.entry_std * std)
        upper_exit = mean + (self.exit_std * std)
        lower_exit = mean - (self.exit_std * std)

        # Use Bollinger Bands if available
        if 'bb_upper' in data.columns and 'bb_lower' in data.columns:
            upper_entry = data['bb_upper']
            lower_entry = data['bb_lower']
            mean = data['bb_middle']

        # Generate signals
        # Buy when oversold
        buy_conditions = price < lower_entry

        # Sell when overbought
        sell_conditions = price > upper_entry

        signals[buy_conditions] = 1
        signals[sell_conditions] = -1

        # Log statistics
        buy_count = (signals == 1).sum()
        sell_count = (signals == -1).sum()
        logger.info(f"Mean reversion generated {buy_count} buy and {sell_count} sell signals")

        return signals


class ZScoreReversionStrategy(BaseStrategy):
    """
    Mean reversion using Z-score normalization.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize Z-score reversion strategy."""
        super().__init__("ZScoreReversion", config)
        self.lookback_period = config.get('lookback_period', 30)
        self.entry_zscore = config.get('entry_zscore', 2.0)
        self.exit_zscore = config.get('exit_zscore', 0.0)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate signals based on Z-score.

        Args:
            data: Market data

        Returns:
            Series of signals
        """
        signals = pd.Series(0, index=data.index)

        # Calculate Z-score
        price = data['close']
        mean = price.rolling(self.lookback_period).mean()
        std = price.rolling(self.lookback_period).std()
        zscore = (price - mean) / std

        # Generate signals
        # Buy when Z-score < -entry_zscore (oversold)
        signals[zscore < -self.entry_zscore] = 1

        # Sell when Z-score > entry_zscore (overbought)
        signals[zscore > self.entry_zscore] = -1

        # Exit when Z-score returns to neutral
        # This would require position tracking in practice

        logger.info(f"Z-score reversion generated {(signals != 0).sum()} signals")

        return signals


class PairsReversionStrategy(BaseStrategy):
    """
    Pairs trading mean reversion strategy.

    Trades based on the spread between two correlated assets.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize pairs reversion strategy."""
        super().__init__("PairsReversion", config)
        self.lookback_period = config.get('lookback_period', 60)
        self.entry_threshold = config.get('entry_threshold', 2.0)
        self.exit_threshold = config.get('exit_threshold', 0.5)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate pairs trading signals.

        Note: This requires data for two assets.
        For single asset, we'll use a simplified version.

        Args:
            data: Market data

        Returns:
            Series of signals
        """
        signals = pd.Series(0, index=data.index)

        # For demonstration, use price deviation from moving average
        # In practice, you'd calculate spread between two assets
        price = data['close']
        ma = price.rolling(self.lookback_period).mean()
        spread = price - ma
        spread_std = spread.rolling(self.lookback_period).std()

        # Normalize spread
        normalized_spread = spread / spread_std

        # Generate signals
        signals[normalized_spread < -self.entry_threshold] = 1
        signals[normalized_spread > self.entry_threshold] = -1

        logger.info(f"Pairs reversion generated {(signals != 0).sum()} signals")

        return signals


class RSIMeanReversionStrategy(BaseStrategy):
    """
    Mean reversion using RSI indicator.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize RSI mean reversion strategy."""
        super().__init__("RSIMeanReversion", config)
        self.oversold_threshold = config.get('oversold_threshold', 30)
        self.overbought_threshold = config.get('overbought_threshold', 70)
        self.exit_threshold = config.get('exit_threshold', 50)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate RSI-based mean reversion signals.

        Args:
            data: Market data with RSI

        Returns:
            Series of signals
        """
        signals = pd.Series(0, index=data.index)

        if 'rsi' not in data.columns:
            logger.warning("RSI not found in data, cannot generate signals")
            return signals

        rsi = data['rsi']

        # Buy when oversold
        signals[rsi < self.oversold_threshold] = 1

        # Sell when overbought
        signals[rsi > self.overbought_threshold] = -1

        buy_count = (signals == 1).sum()
        sell_count = (signals == -1).sum()
        logger.info(f"RSI mean reversion generated {buy_count} buy and {sell_count} sell signals")

        return signals
