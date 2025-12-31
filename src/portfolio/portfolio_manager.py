"""Portfolio management and orchestration."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..strategies.base_strategy import BaseStrategy
from ..risk.risk_manager import RiskManager, Position
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class PortfolioManager:
    """
    Manages portfolio allocation across multiple strategies.
    """

    def __init__(
        self,
        initial_capital: float,
        risk_manager: RiskManager,
        config: Dict[str, Any]
    ):
        """
        Initialize portfolio manager.

        Args:
            initial_capital: Starting capital
            risk_manager: Risk management system
            config: Portfolio configuration
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.risk_manager = risk_manager
        self.config = config

        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_allocations: Dict[str, float] = {}
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Dict[str, Any]] = []
        self.equity_history: List[Dict[str, Any]] = []

    def add_strategy(
        self,
        strategy: BaseStrategy,
        allocation: float = 1.0
    ):
        """
        Add a strategy to the portfolio.

        Args:
            strategy: Trading strategy
            allocation: Capital allocation (0.0 to 1.0)
        """
        if allocation < 0 or allocation > 1:
            raise ValueError("Allocation must be between 0 and 1")

        self.strategies[strategy.name] = strategy
        self.strategy_allocations[strategy.name] = allocation

        logger.info(f"Added strategy {strategy.name} with {allocation:.1%} allocation")

    def remove_strategy(self, strategy_name: str):
        """Remove a strategy from the portfolio."""
        if strategy_name in self.strategies:
            del self.strategies[strategy_name]
            del self.strategy_allocations[strategy_name]
            logger.info(f"Removed strategy {strategy_name}")

    def rebalance_allocations(self, new_allocations: Dict[str, float]):
        """
        Rebalance strategy allocations.

        Args:
            new_allocations: New allocation weights
        """
        # Validate allocations sum to ~1.0
        total = sum(new_allocations.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Allocations sum to {total:.2f}, normalizing")
            new_allocations = {k: v/total for k, v in new_allocations.items()}

        self.strategy_allocations.update(new_allocations)
        logger.info(f"Rebalanced allocations: {new_allocations}")

    def execute_strategies(
        self,
        data: pd.DataFrame,
        symbol: str = "QQQ"
    ) -> Dict[str, Any]:
        """
        Execute all strategies and generate trading signals.

        Args:
            data: Market data
            symbol: Trading symbol

        Returns:
            Dictionary with signals from all strategies
        """
        all_signals = {}

        for strategy_name, strategy in self.strategies.items():
            try:
                signals = strategy.generate_signals(data)
                all_signals[strategy_name] = signals
                logger.info(f"{strategy_name}: Generated {(signals != 0).sum()} signals")
            except Exception as e:
                logger.error(f"Error executing {strategy_name}: {e}")
                all_signals[strategy_name] = pd.Series(0, index=data.index)

        return all_signals

    def aggregate_signals(
        self,
        signals: Dict[str, pd.Series],
        method: str = 'weighted_average'
    ) -> pd.Series:
        """
        Aggregate signals from multiple strategies.

        Args:
            signals: Dictionary of signals from each strategy
            method: Aggregation method ('weighted_average', 'majority_vote', 'unanimous')

        Returns:
            Aggregated signal series
        """
        if not signals:
            return pd.Series()

        # Get common index
        indices = [s.index for s in signals.values()]
        common_index = indices[0]
        for idx in indices[1:]:
            common_index = common_index.intersection(idx)

        if method == 'weighted_average':
            # Weighted average of signals
            aggregated = pd.Series(0.0, index=common_index)

            for strategy_name, signal in signals.items():
                weight = self.strategy_allocations.get(strategy_name, 0)
                aligned_signal = signal.reindex(common_index, fill_value=0)
                aggregated += aligned_signal * weight

            # Convert to discrete signals
            aggregated = aggregated.apply(
                lambda x: 1 if x > 0.5 else (-1 if x < -0.5 else 0)
            )

        elif method == 'majority_vote':
            # Majority voting
            aggregated = pd.Series(0, index=common_index)

            for idx in common_index:
                votes = [signals[s].loc[idx] for s in signals.keys() if idx in signals[s].index]
                if votes:
                    # Count votes
                    buy_votes = sum(1 for v in votes if v > 0)
                    sell_votes = sum(1 for v in votes if v < 0)

                    if buy_votes > sell_votes:
                        aggregated.loc[idx] = 1
                    elif sell_votes > buy_votes:
                        aggregated.loc[idx] = -1

        elif method == 'unanimous':
            # All strategies must agree
            aggregated = pd.Series(0, index=common_index)

            for idx in common_index:
                votes = [signals[s].loc[idx] for s in signals.keys() if idx in signals[s].index]
                if votes and all(v > 0 for v in votes):
                    aggregated.loc[idx] = 1
                elif votes and all(v < 0 for v in votes):
                    aggregated.loc[idx] = -1

        else:
            logger.warning(f"Unknown aggregation method: {method}, using first strategy")
            aggregated = list(signals.values())[0]

        return aggregated

    def calculate_portfolio_metrics(self) -> Dict[str, float]:
        """Calculate current portfolio metrics."""
        if not self.equity_history:
            return {}

        equity_df = pd.DataFrame(self.equity_history)
        equity_series = equity_df['equity']

        returns = equity_series.pct_change().dropna()

        # Calculate metrics
        total_return = (equity_series.iloc[-1] / self.initial_capital) - 1
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        running_max = equity_series.cummax()
        drawdown = (equity_series - running_max) / running_max
        max_drawdown = drawdown.min()

        return {
            'current_capital': self.capital,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'num_positions': len(self.positions),
            'num_trades': len(self.trade_history)
        }

    def record_trade(
        self,
        symbol: str,
        action: str,
        shares: int,
        price: float,
        timestamp: pd.Timestamp
    ):
        """Record a trade."""
        trade = {
            'timestamp': timestamp,
            'symbol': symbol,
            'action': action,
            'shares': shares,
            'price': price,
            'value': shares * price
        }

        self.trade_history.append(trade)
        logger.info(f"Recorded trade: {action} {shares} {symbol} @ ${price:.2f}")

    def record_equity(self, timestamp: pd.Timestamp, equity: float):
        """Record equity value."""
        self.equity_history.append({
            'timestamp': timestamp,
            'equity': equity
        })

    def get_equity_curve(self) -> pd.Series:
        """Get portfolio equity curve."""
        if not self.equity_history:
            return pd.Series()

        df = pd.DataFrame(self.equity_history)
        return df.set_index('timestamp')['equity']

    def get_trade_log(self) -> pd.DataFrame:
        """Get trade history as DataFrame."""
        if not self.trade_history:
            return pd.DataFrame()

        return pd.DataFrame(self.trade_history)

    def reset(self):
        """Reset portfolio to initial state."""
        self.capital = self.initial_capital
        self.positions = {}
        self.trade_history = []
        self.equity_history = []

        for strategy in self.strategies.values():
            strategy.reset()

        logger.info("Portfolio reset to initial state")
