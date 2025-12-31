"""Options trading strategies."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from scipy.stats import norm

from ..strategies.base_strategy import BaseStrategy
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class OptionsStrategy(BaseStrategy):
    """Base class for options trading strategies."""

    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize options strategy."""
        super().__init__(name, config)
        self.delta_target = config.get('delta_target', 0.30)
        self.dte_range = config.get('dte_range', [30, 45])
        self.min_premium = config.get('min_premium', 0.01)

    def calculate_option_greeks(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        risk_free_rate: float = 0.05,
        option_type: str = 'call'
    ) -> Dict[str, float]:
        """
        Calculate Black-Scholes Greeks for an option.

        Args:
            spot: Current stock price
            strike: Option strike price
            time_to_expiry: Time to expiration in years
            volatility: Implied volatility
            risk_free_rate: Risk-free interest rate
            option_type: 'call' or 'put'

        Returns:
            Dictionary with Greeks (delta, gamma, theta, vega, rho)
        """
        if time_to_expiry <= 0:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}

        # Black-Scholes parameters
        d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry) / \
             (volatility * np.sqrt(time_to_expiry))
        d2 = d1 - volatility * np.sqrt(time_to_expiry)

        # Delta
        if option_type.lower() == 'call':
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1

        # Gamma (same for calls and puts)
        gamma = norm.pdf(d1) / (spot * volatility * np.sqrt(time_to_expiry))

        # Theta
        if option_type.lower() == 'call':
            theta = (-spot * norm.pdf(d1) * volatility / (2 * np.sqrt(time_to_expiry)) -
                    risk_free_rate * strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)) / 365
        else:
            theta = (-spot * norm.pdf(d1) * volatility / (2 * np.sqrt(time_to_expiry)) +
                    risk_free_rate * strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2)) / 365

        # Vega (same for calls and puts)
        vega = spot * norm.pdf(d1) * np.sqrt(time_to_expiry) / 100

        # Rho
        if option_type.lower() == 'call':
            rho = strike * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2) / 100
        else:
            rho = -strike * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) / 100

        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }

    def filter_options_chain(
        self,
        options_df: pd.DataFrame,
        spot_price: float,
        target_delta: Optional[float] = None,
        dte_range: Optional[Tuple[int, int]] = None
    ) -> pd.DataFrame:
        """
        Filter options chain based on criteria.

        Args:
            options_df: Options chain DataFrame
            spot_price: Current stock price
            target_delta: Target delta value
            dte_range: (min_dte, max_dte) range for expiration

        Returns:
            Filtered DataFrame
        """
        filtered = options_df.copy()

        # Filter by DTE if specified
        if dte_range:
            min_dte, max_dte = dte_range
            if 'expiration' in filtered.columns:
                filtered['dte'] = (pd.to_datetime(filtered['expiration']) - datetime.now()).dt.days
                filtered = filtered[(filtered['dte'] >= min_dte) & (filtered['dte'] <= max_dte)]

        # Filter by liquidity
        if 'volume' in filtered.columns:
            filtered = filtered[filtered['volume'] > 0]

        # Filter by bid-ask spread (if available)
        if 'bid' in filtered.columns and 'ask' in filtered.columns:
            filtered['spread'] = (filtered['ask'] - filtered['bid']) / filtered['ask']
            filtered = filtered[filtered['spread'] < 0.10]  # Less than 10% spread

        return filtered


class CoveredCallStrategy(OptionsStrategy):
    """
    Covered call strategy - sell call options against long stock position.

    Income generation strategy with limited upside.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize covered call strategy."""
        super().__init__("CoveredCall", config)
        self.target_return = config.get('target_return', 0.02)  # 2% target monthly return

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate covered call signals.

        Signals indicate when to sell calls against stock position.

        Args:
            data: Market data

        Returns:
            Series of signals
        """
        signals = pd.Series(0, index=data.index)

        # Covered call is typically executed when:
        # 1. We have a long stock position
        # 2. Market is neutral to slightly bullish
        # 3. Implied volatility is elevated

        # Check for neutral to bullish conditions
        if 'rsi' in data.columns:
            # Sell calls when RSI is in mid-range (neutral)
            neutral_conditions = (data['rsi'] >= 40) & (data['rsi'] <= 60)
            signals[neutral_conditions] = 1

        # Also consider when price is above moving average
        if 'sma_20' in data.columns:
            bullish_conditions = data['close'] > data['sma_20']
            signals[bullish_conditions] = 1

        logger.info(f"Covered call generated {(signals == 1).sum()} sell call signals")

        return signals

    def select_strike(
        self,
        options_chain: pd.DataFrame,
        spot_price: float,
        volatility: float
    ) -> Optional[pd.Series]:
        """
        Select optimal strike for covered call.

        Args:
            options_chain: Calls options chain
            spot_price: Current stock price
            volatility: Current volatility

        Returns:
            Selected option contract
        """
        # Filter options
        filtered = self.filter_options_chain(
            options_chain,
            spot_price,
            dte_range=self.dte_range
        )

        if filtered.empty:
            logger.warning("No suitable options found")
            return None

        # Calculate Greeks for each option
        for idx, row in filtered.iterrows():
            tte = row['dte'] / 365 if 'dte' in row else 0.1
            greeks = self.calculate_option_greeks(
                spot_price,
                row['strike'],
                tte,
                volatility,
                option_type='call'
            )
            filtered.loc[idx, 'delta'] = greeks['delta']

        # Select option closest to target delta
        filtered['delta_diff'] = abs(filtered['delta'] - self.delta_target)
        selected = filtered.loc[filtered['delta_diff'].idxmin()]

        logger.info(f"Selected call strike: {selected['strike']} with delta {selected['delta']:.2f}")

        return selected


class CashSecuredPutStrategy(OptionsStrategy):
    """
    Cash-secured put strategy - sell put options with cash backing.

    Income generation with obligation to buy stock at strike.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize cash-secured put strategy."""
        super().__init__("CashSecuredPut", config)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate cash-secured put signals.

        Sell puts when:
        - Want to own stock at lower price
        - Market is bullish
        - Support levels nearby

        Args:
            data: Market data

        Returns:
            Series of signals
        """
        signals = pd.Series(0, index=data.index)

        # Sell puts in bullish conditions
        if 'sma_50' in data.columns:
            bullish = data['close'] > data['sma_50']
            signals[bullish] = 1

        # Also when price is consolidating (low volatility)
        if 'volatility_20' in data.columns:
            low_vol = data['volatility_20'] < data['volatility_20'].quantile(0.3)
            signals[low_vol] = 1

        logger.info(f"Cash-secured put generated {(signals == 1).sum()} sell put signals")

        return signals

    def select_strike(
        self,
        options_chain: pd.DataFrame,
        spot_price: float,
        volatility: float,
        support_level: Optional[float] = None
    ) -> Optional[pd.Series]:
        """
        Select optimal strike for cash-secured put.

        Args:
            options_chain: Puts options chain
            spot_price: Current stock price
            volatility: Current volatility
            support_level: Support price level

        Returns:
            Selected option contract
        """
        # Filter options
        filtered = self.filter_options_chain(
            options_chain,
            spot_price,
            dte_range=self.dte_range
        )

        if filtered.empty:
            return None

        # Calculate Greeks
        for idx, row in filtered.iterrows():
            tte = row['dte'] / 365 if 'dte' in row else 0.1
            greeks = self.calculate_option_greeks(
                spot_price,
                row['strike'],
                tte,
                volatility,
                option_type='put'
            )
            filtered.loc[idx, 'delta'] = abs(greeks['delta'])  # Use absolute value for puts

        # Select option closest to target delta
        filtered['delta_diff'] = abs(filtered['delta'] - self.delta_target)
        selected = filtered.loc[filtered['delta_diff'].idxmin()]

        logger.info(f"Selected put strike: {selected['strike']} with delta {selected['delta']:.2f}")

        return selected


class ProtectivePutStrategy(OptionsStrategy):
    """
    Protective put strategy - buy put options to hedge long stock position.

    Downside protection with limited loss.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize protective put strategy."""
        super().__init__("ProtectivePut", config)
        self.protection_delta = config.get('protection_delta', 0.30)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate protective put signals.

        Buy puts when:
        - Market shows signs of weakness
        - High volatility
        - Breaking support levels

        Args:
            data: Market data

        Returns:
            Series of signals
        """
        signals = pd.Series(0, index=data.index)

        # Buy protection in bearish conditions
        if 'sma_50' in data.columns:
            bearish = data['close'] < data['sma_50']
            signals[bearish] = 1

        # Buy when volatility is rising
        if 'volatility_20' in data.columns:
            vol_change = data['volatility_20'].pct_change()
            rising_vol = vol_change > 0.1
            signals[rising_vol] = 1

        # Buy when RSI shows weakness
        if 'rsi' in data.columns:
            weak = data['rsi'] < 40
            signals[weak] = 1

        logger.info(f"Protective put generated {(signals == 1).sum()} buy put signals")

        return signals


class IronCondorStrategy(OptionsStrategy):
    """
    Iron Condor strategy - sell OTM call and put spreads.

    Neutral strategy profiting from low volatility.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize iron condor strategy."""
        super().__init__("IronCondor", config)
        self.wing_width = config.get('wing_width', 5)  # Strike width

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate iron condor signals.

        Execute when:
        - Low volatility expected
        - Range-bound market
        - High IV rank

        Args:
            data: Market data

        Returns:
            Series of signals
        """
        signals = pd.Series(0, index=data.index)

        # Execute in low volatility, range-bound conditions
        if 'volatility_20' in data.columns and 'bb_width' in data.columns:
            low_vol = data['volatility_20'] < data['volatility_20'].quantile(0.4)
            narrow_range = data['bb_width'] < data['bb_width'].quantile(0.4)
            signals[low_vol & narrow_range] = 1

        logger.info(f"Iron condor generated {(signals == 1).sum()} signals")

        return signals
