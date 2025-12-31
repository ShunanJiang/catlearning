"""Risk management system for portfolio and position management."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    shares: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    position_value: float


class PositionSizer:
    """
    Position sizing algorithms for risk management.
    """

    @staticmethod
    def fixed_fractional(
        capital: float,
        price: float,
        risk_fraction: float = 0.02
    ) -> int:
        """
        Fixed fractional position sizing.

        Args:
            capital: Available capital
            price: Entry price
            risk_fraction: Fraction of capital to risk per trade

        Returns:
            Number of shares
        """
        risk_amount = capital * risk_fraction
        shares = int(risk_amount / price)
        return shares

    @staticmethod
    def kelly_criterion(
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        capital: float,
        price: float
    ) -> int:
        """
        Kelly Criterion position sizing.

        Args:
            win_rate: Historical win rate
            avg_win: Average winning trade
            avg_loss: Average losing trade (positive number)
            capital: Available capital
            price: Entry price

        Returns:
            Number of shares
        """
        if avg_loss == 0 or win_rate >= 1:
            return 0

        # Kelly fraction
        win_loss_ratio = avg_win / avg_loss
        kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio

        # Conservative Kelly (use half Kelly)
        kelly_fraction = max(0, min(kelly_fraction * 0.5, 0.25))  # Cap at 25%

        position_value = capital * kelly_fraction
        shares = int(position_value / price)

        return shares

    @staticmethod
    def volatility_based(
        capital: float,
        price: float,
        volatility: float,
        target_volatility: float = 0.15
    ) -> int:
        """
        Volatility-based position sizing.

        Args:
            capital: Available capital
            price: Entry price
            volatility: Asset volatility
            target_volatility: Target portfolio volatility

        Returns:
            Number of shares
        """
        if volatility == 0:
            return 0

        # Calculate position size to achieve target volatility
        position_value = capital * (target_volatility / volatility)
        shares = int(position_value / price)

        return shares

    @staticmethod
    def max_position_size(
        capital: float,
        price: float,
        max_fraction: float = 0.3
    ) -> int:
        """
        Maximum position size as fraction of capital.

        Args:
            capital: Available capital
            price: Entry price
            max_fraction: Maximum fraction of capital for position

        Returns:
            Number of shares
        """
        max_value = capital * max_fraction
        shares = int(max_value / price)
        return shares


class RiskManager:
    """
    Comprehensive risk management system.

    Handles position sizing, risk limits, and portfolio constraints.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize risk manager.

        Args:
            config: Risk management configuration
        """
        self.config = config
        self.max_position_size = config.get('max_position_size', 0.3)
        self.max_drawdown = config.get('max_drawdown', 0.15)
        self.stop_loss = config.get('stop_loss', 0.05)
        self.take_profit = config.get('take_profit', 0.15)
        self.max_leverage = config.get('max_leverage', 1.0)

        self.position_sizer = PositionSizer()
        self.positions: Dict[str, Position] = {}
        self.peak_equity = 0.0

    def check_risk_limits(
        self,
        symbol: str,
        shares: int,
        price: float,
        capital: float
    ) -> Tuple[bool, str]:
        """
        Check if trade meets risk limits.

        Args:
            symbol: Trading symbol
            shares: Number of shares
            price: Entry price
            capital: Current capital

        Returns:
            (allowed, reason) tuple
        """
        # Check position size limit
        position_value = shares * price
        position_fraction = position_value / capital

        if position_fraction > self.max_position_size:
            return False, f"Position size {position_fraction:.2%} exceeds limit {self.max_position_size:.2%}"

        # Check leverage limit
        total_position_value = sum(p.position_value for p in self.positions.values())
        new_total = total_position_value + position_value
        leverage = new_total / capital

        if leverage > self.max_leverage:
            return False, f"Leverage {leverage:.2f} exceeds limit {self.max_leverage:.2f}"

        # Check if we have enough capital
        if position_value > capital:
            return False, "Insufficient capital"

        return True, "OK"

    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        capital: float,
        volatility: float,
        method: str = 'fixed_fractional',
        **kwargs
    ) -> int:
        """
        Calculate position size using specified method.

        Args:
            symbol: Trading symbol
            price: Entry price
            capital: Available capital
            volatility: Asset volatility
            method: Sizing method ('fixed_fractional', 'kelly', 'volatility', 'max')
            **kwargs: Additional parameters for sizing method

        Returns:
            Number of shares
        """
        if method == 'fixed_fractional':
            risk_fraction = kwargs.get('risk_fraction', 0.02)
            shares = self.position_sizer.fixed_fractional(capital, price, risk_fraction)

        elif method == 'kelly':
            win_rate = kwargs.get('win_rate', 0.5)
            avg_win = kwargs.get('avg_win', 0.1)
            avg_loss = kwargs.get('avg_loss', 0.05)
            shares = self.position_sizer.kelly_criterion(
                win_rate, avg_win, avg_loss, capital, price
            )

        elif method == 'volatility':
            target_vol = kwargs.get('target_volatility', 0.15)
            shares = self.position_sizer.volatility_based(
                capital, price, volatility, target_vol
            )

        elif method == 'max':
            shares = self.position_sizer.max_position_size(
                capital, price, self.max_position_size
            )

        else:
            logger.warning(f"Unknown sizing method: {method}, using fixed_fractional")
            shares = self.position_sizer.fixed_fractional(capital, price, 0.02)

        # Apply risk limits
        allowed, reason = self.check_risk_limits(symbol, shares, price, capital)
        if not allowed:
            logger.warning(f"Position size reduced due to: {reason}")
            # Reduce shares to meet limits
            shares = int(shares * 0.5)

        return max(0, shares)

    def update_position(
        self,
        symbol: str,
        shares: float,
        entry_price: float,
        current_price: float
    ):
        """Update or create position."""
        unrealized_pnl = (current_price - entry_price) * shares
        position_value = current_price * abs(shares)

        self.positions[symbol] = Position(
            symbol=symbol,
            shares=shares,
            entry_price=entry_price,
            current_price=current_price,
            unrealized_pnl=unrealized_pnl,
            position_value=position_value
        )

    def close_position(self, symbol: str):
        """Close position."""
        if symbol in self.positions:
            del self.positions[symbol]

    def check_stop_loss(
        self,
        symbol: str,
        current_price: float
    ) -> bool:
        """
        Check if stop loss is triggered.

        Args:
            symbol: Trading symbol
            current_price: Current price

        Returns:
            True if stop loss triggered
        """
        if symbol not in self.positions:
            return False

        position = self.positions[symbol]
        entry_price = position.entry_price

        # Long position
        if position.shares > 0:
            loss_pct = (current_price - entry_price) / entry_price
            if loss_pct <= -self.stop_loss:
                logger.info(f"Stop loss triggered for {symbol}: {loss_pct:.2%}")
                return True

        # Short position
        else:
            loss_pct = (entry_price - current_price) / entry_price
            if loss_pct <= -self.stop_loss:
                logger.info(f"Stop loss triggered for {symbol}: {loss_pct:.2%}")
                return True

        return False

    def check_take_profit(
        self,
        symbol: str,
        current_price: float
    ) -> bool:
        """
        Check if take profit is triggered.

        Args:
            symbol: Trading symbol
            current_price: Current price

        Returns:
            True if take profit triggered
        """
        if symbol not in self.positions:
            return False

        position = self.positions[symbol]
        entry_price = position.entry_price

        # Long position
        if position.shares > 0:
            profit_pct = (current_price - entry_price) / entry_price
            if profit_pct >= self.take_profit:
                logger.info(f"Take profit triggered for {symbol}: {profit_pct:.2%}")
                return True

        # Short position
        else:
            profit_pct = (entry_price - current_price) / entry_price
            if profit_pct >= self.take_profit:
                logger.info(f"Take profit triggered for {symbol}: {profit_pct:.2%}")
                return True

        return False

    def check_drawdown_limit(self, current_equity: float) -> bool:
        """
        Check if maximum drawdown limit is exceeded.

        Args:
            current_equity: Current portfolio equity

        Returns:
            True if drawdown limit exceeded
        """
        self.peak_equity = max(self.peak_equity, current_equity)

        if self.peak_equity == 0:
            return False

        drawdown = (self.peak_equity - current_equity) / self.peak_equity

        if drawdown >= self.max_drawdown:
            logger.warning(f"Max drawdown limit exceeded: {drawdown:.2%}")
            return True

        return False

    def get_portfolio_metrics(self) -> Dict[str, float]:
        """Get current portfolio risk metrics."""
        if not self.positions:
            return {
                'total_positions': 0,
                'total_value': 0,
                'total_pnl': 0,
                'largest_position': 0
            }

        total_value = sum(p.position_value for p in self.positions.values())
        total_pnl = sum(p.unrealized_pnl for p in self.positions.values())
        largest_position = max(p.position_value for p in self.positions.values()) if self.positions else 0

        return {
            'total_positions': len(self.positions),
            'total_value': total_value,
            'total_pnl': total_pnl,
            'largest_position': largest_position
        }
