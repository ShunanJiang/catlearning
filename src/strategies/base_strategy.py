"""Base strategy class for all trading strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class Signal(Enum):
    """Trading signal types."""
    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class TradeSignal:
    """Trade signal with metadata."""
    timestamp: pd.Timestamp
    signal: Signal
    price: float
    confidence: float = 1.0
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseStrategy(ABC):
    """Base class for all trading strategies."""

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize strategy.

        Args:
            name: Strategy name
            config: Strategy configuration
        """
        self.name = name
        self.config = config
        self.signals_history = []
        self.positions_history = []
        self.current_position = 0  # -1: short, 0: neutral, 1: long

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on the strategy logic.

        Args:
            data: Market data DataFrame with OHLCV and features

        Returns:
            Series of signals (1: buy, -1: sell, 0: hold)
        """
        pass

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data before generating signals.

        Args:
            data: Raw market data

        Returns:
            Preprocessed data
        """
        # Default implementation - can be overridden
        return data.copy()

    def calculate_position_size(
        self,
        signal: Signal,
        capital: float,
        price: float,
        risk_params: Dict[str, Any]
    ) -> float:
        """
        Calculate position size based on signal and risk parameters.

        Args:
            signal: Trading signal
            capital: Available capital
            price: Current price
            risk_params: Risk management parameters

        Returns:
            Number of shares to trade
        """
        if signal == Signal.HOLD:
            return 0

        max_position_size = risk_params.get('max_position_size', 0.3)
        max_capital = capital * max_position_size

        # Basic position sizing - can be overridden for more sophisticated methods
        shares = int(max_capital / price)

        return shares

    def update_position(self, signal: Signal, timestamp: pd.Timestamp):
        """Update current position based on signal."""
        if signal == Signal.BUY:
            self.current_position = 1
        elif signal == Signal.SELL:
            self.current_position = -1
        else:
            # For HOLD, we might want to close position in some strategies
            pass

        self.positions_history.append({
            'timestamp': timestamp,
            'position': self.current_position
        })

    def reset(self):
        """Reset strategy state."""
        self.signals_history = []
        self.positions_history = []
        self.current_position = 0

    def get_performance_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate strategy performance metrics.

        Args:
            returns: Series of strategy returns

        Returns:
            Dictionary of performance metrics
        """
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1

        # Sharpe ratio
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        sortino = returns.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0

        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Win rate
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(returns)
        }

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"
