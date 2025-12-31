"""Backtesting engine for trading strategies."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from ..strategies.base_strategy import BaseStrategy, Signal
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class Trade:
    """Represents a single trade."""
    entry_date: pd.Timestamp
    exit_date: Optional[pd.Timestamp] = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    shares: float = 0.0
    direction: int = 1  # 1: long, -1: short
    pnl: float = 0.0
    return_pct: float = 0.0
    commission: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    strategy_name: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_trade_return: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    positions: pd.Series = field(default_factory=pd.Series)
    signals: pd.Series = field(default_factory=pd.Series)
    metrics: Dict[str, float] = field(default_factory=dict)


class BacktestEngine:
    """
    Backtesting engine for trading strategies.

    Simulates strategy execution on historical data with realistic constraints.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize backtest engine.

        Args:
            config: Backtesting configuration
        """
        self.config = config
        self.initial_capital = config.get('initial_capital', 100000)
        self.commission = config.get('commission', 0.001)
        self.slippage = config.get('slippage', 0.0005)

    def run(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        risk_params: Optional[Dict[str, Any]] = None
    ) -> BacktestResult:
        """
        Run backtest for a strategy.

        Args:
            strategy: Trading strategy to backtest
            data: Historical market data
            risk_params: Risk management parameters

        Returns:
            BacktestResult with performance metrics
        """
        logger.info(f"Starting backtest for {strategy.name}")

        if risk_params is None:
            risk_params = self.config.get('risk', {})

        # Generate signals
        signals = strategy.generate_signals(data)

        # Initialize tracking variables
        capital = self.initial_capital
        positions = pd.Series(0, index=data.index)
        equity_curve = pd.Series(self.initial_capital, index=data.index)
        trades = []
        current_trade: Optional[Trade] = None

        # Simulate trading
        for i, (timestamp, row) in enumerate(data.iterrows()):
            signal = signals.loc[timestamp] if timestamp in signals.index else 0

            # Current position
            current_position = positions.iloc[i-1] if i > 0 else 0
            price = row['close']

            # Handle trade exits and entries
            if current_position != 0 and current_trade is not None:
                # Check for exit conditions
                should_exit = False

                # Exit on opposite signal
                if (current_position > 0 and signal == -1) or \
                   (current_position < 0 and signal == 1):
                    should_exit = True

                # Stop loss check
                if 'stop_loss' in risk_params:
                    stop_loss = risk_params['stop_loss']
                    entry_price = current_trade.entry_price
                    if current_position > 0:  # Long position
                        if price <= entry_price * (1 - stop_loss):
                            should_exit = True
                    else:  # Short position
                        if price >= entry_price * (1 + stop_loss):
                            should_exit = True

                # Take profit check
                if 'take_profit' in risk_params:
                    take_profit = risk_params['take_profit']
                    entry_price = current_trade.entry_price
                    if current_position > 0:  # Long position
                        if price >= entry_price * (1 + take_profit):
                            should_exit = True
                    else:  # Short position
                        if price <= entry_price * (1 - take_profit):
                            should_exit = True

                if should_exit:
                    # Close position
                    exit_price = price * (1 - self.slippage if current_position > 0 else 1 + self.slippage)
                    pnl = (exit_price - current_trade.entry_price) * current_trade.shares * current_trade.direction
                    commission_cost = abs(exit_price * current_trade.shares * self.commission)
                    pnl -= commission_cost

                    capital += pnl
                    current_trade.exit_date = timestamp
                    current_trade.exit_price = exit_price
                    current_trade.pnl = pnl
                    current_trade.return_pct = pnl / (current_trade.entry_price * abs(current_trade.shares))

                    trades.append(current_trade)
                    current_position = 0
                    current_trade = None

            # Enter new position
            if current_position == 0 and signal != 0:
                # Calculate position size
                shares = strategy.calculate_position_size(
                    Signal(signal),
                    capital,
                    price,
                    risk_params
                )

                if shares > 0:
                    # Entry with slippage
                    entry_price = price * (1 + self.slippage if signal == 1 else 1 - self.slippage)
                    commission_cost = entry_price * shares * self.commission

                    # Check if we have enough capital
                    required_capital = entry_price * shares + commission_cost
                    if required_capital <= capital:
                        capital -= commission_cost
                        current_position = shares if signal == 1 else -shares
                        current_trade = Trade(
                            entry_date=timestamp,
                            entry_price=entry_price,
                            shares=shares,
                            direction=signal,
                            commission=commission_cost
                        )

            # Update positions and equity
            positions.iloc[i] = current_position

            # Calculate equity
            if current_position != 0 and current_trade is not None:
                unrealized_pnl = (price - current_trade.entry_price) * abs(current_position) * current_trade.direction
                equity_curve.iloc[i] = capital + unrealized_pnl
            else:
                equity_curve.iloc[i] = capital

        # Close any open position at end
        if current_trade is not None:
            final_price = data['close'].iloc[-1]
            pnl = (final_price - current_trade.entry_price) * current_trade.shares * current_trade.direction
            commission_cost = abs(final_price * current_trade.shares * self.commission)
            pnl -= commission_cost

            current_trade.exit_date = data.index[-1]
            current_trade.exit_price = final_price
            current_trade.pnl = pnl
            current_trade.return_pct = pnl / (current_trade.entry_price * abs(current_trade.shares))
            trades.append(current_trade)
            capital += pnl

        # Calculate performance metrics
        metrics = self._calculate_metrics(equity_curve, trades)

        result = BacktestResult(
            strategy_name=strategy.name,
            start_date=data.index[0],
            end_date=data.index[-1],
            initial_capital=self.initial_capital,
            final_capital=equity_curve.iloc[-1],
            trades=trades,
            equity_curve=equity_curve,
            positions=positions,
            signals=signals,
            **metrics
        )

        logger.info(f"Backtest complete: {result.num_trades} trades, "
                   f"{result.total_return:.2%} return, "
                   f"Sharpe: {result.sharpe_ratio:.2f}")

        return result

    def _calculate_metrics(
        self,
        equity_curve: pd.Series,
        trades: List[Trade]
    ) -> Dict[str, float]:
        """Calculate performance metrics."""
        # Returns
        returns = equity_curve.pct_change().dropna()
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

        # Annualized return
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Sharpe ratio
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        sortino_ratio = returns.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0

        # Maximum drawdown
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Trade statistics
        num_trades = len(trades)
        if num_trades > 0:
            trade_returns = [t.return_pct for t in trades if t.exit_date is not None]
            winning_trades = [t for t in trades if t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl < 0]

            win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0
            avg_trade_return = np.mean(trade_returns) if trade_returns else 0

            # Profit factor
            gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
            gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        else:
            win_rate = 0
            avg_trade_return = 0
            profit_factor = 0

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'num_trades': num_trades,
            'avg_trade_return': avg_trade_return,
            'metrics': {
                'total_trades': num_trades,
                'avg_return': avg_trade_return,
                'win_rate': win_rate
            }
        }

    def walk_forward_analysis(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        train_size: int = 252,
        test_size: int = 63,
        risk_params: Optional[Dict[str, Any]] = None
    ) -> List[BacktestResult]:
        """
        Perform walk-forward analysis.

        Args:
            strategy: Trading strategy
            data: Historical data
            train_size: Training window size (days)
            test_size: Testing window size (days)
            risk_params: Risk parameters

        Returns:
            List of backtest results for each window
        """
        logger.info("Starting walk-forward analysis")

        results = []
        window_start = 0

        while window_start + train_size + test_size <= len(data):
            # Split data
            train_end = window_start + train_size
            test_end = train_end + test_size

            train_data = data.iloc[window_start:train_end]
            test_data = data.iloc[train_end:test_end]

            # Train strategy (if applicable)
            if hasattr(strategy, 'train'):
                strategy.train(train_data)

            # Backtest on test data
            result = self.run(strategy, test_data, risk_params)
            results.append(result)

            logger.info(f"Window {len(results)}: {result.total_return:.2%} return")

            # Move window
            window_start += test_size

        logger.info(f"Walk-forward analysis complete: {len(results)} windows")

        return results
