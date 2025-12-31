"""Market analysis agent using Claude Agent SDK."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from anthropic import Anthropic

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class MarketAnalyzerAgent:
    """
    Claude-powered market analysis agent.

    Analyzes market conditions, identifies regimes, and provides insights.
    """

    def __init__(self, api_key: str, config: Dict[str, Any]):
        """
        Initialize market analyzer agent.

        Args:
            api_key: Anthropic API key
            config: Agent configuration
        """
        self.client = Anthropic(api_key=api_key)
        self.config = config
        self.model = config.get('model', 'claude-sonnet-4-5-20250929')
        self.max_tokens = config.get('max_tokens', 4096)
        self.temperature = config.get('temperature', 0.7)

    def analyze_market_regime(
        self,
        data: pd.DataFrame,
        symbol: str = "QQQ"
    ) -> Dict[str, Any]:
        """
        Analyze current market regime using Claude.

        Args:
            data: Market data with technical indicators
            symbol: Trading symbol

        Returns:
            Dictionary with market regime analysis
        """
        logger.info(f"Analyzing market regime for {symbol}")

        # Prepare market summary
        recent_data = data.tail(60)  # Last 60 days
        summary = self._prepare_market_summary(recent_data, symbol)

        # Create prompt for Claude
        prompt = f"""Analyze the following market data for {symbol} and provide a comprehensive market regime analysis.

Market Data Summary:
{summary}

Please provide:
1. Current Market Regime: (Trending/Ranging/Volatile/Calm)
2. Trend Direction: (Bullish/Bearish/Neutral)
3. Volatility Assessment: (High/Medium/Low)
4. Key Support/Resistance Levels
5. Trading Recommendations based on current conditions
6. Risk Factors to monitor

Format your response as a structured analysis."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            analysis_text = response.content[0].text

            # Parse and structure the response
            analysis = self._parse_market_analysis(analysis_text, recent_data)

            logger.info(f"Market regime: {analysis.get('regime', 'Unknown')}")

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing market: {e}")
            return self._get_default_analysis(recent_data)

    def _prepare_market_summary(
        self,
        data: pd.DataFrame,
        symbol: str
    ) -> str:
        """Prepare concise market summary for Claude."""
        latest = data.iloc[-1]
        last_week = data.tail(5)
        last_month = data.tail(20)

        summary = f"""
Symbol: {symbol}
Current Price: ${latest['close']:.2f}

Price Action:
- 1-Day Change: {latest.get('returns', 0) * 100:.2f}%
- 5-Day Change: {(latest['close'] / data.iloc[-6]['close'] - 1) * 100:.2f}%
- 20-Day Change: {(latest['close'] / data.iloc[-21]['close'] - 1) * 100:.2f}%

Technical Indicators:
- RSI(14): {latest.get('rsi', 50):.2f}
- MACD: {latest.get('macd', 0):.4f}
- Volatility(20d): {latest.get('volatility_20', 0) * 100:.2f}%
- SMA(20): ${latest.get('sma_20', 0):.2f}
- SMA(50): ${latest.get('sma_50', 0):.2f}
- Bollinger Band Position: {((latest['close'] - latest.get('bb_lower', 0)) / (latest.get('bb_upper', 0) - latest.get('bb_lower', 0))):.2%}

Volume Analysis:
- Current Volume: {latest.get('volume', 0):,.0f}
- Avg Volume(20d): {data.tail(20)['volume'].mean():,.0f}
- Volume Ratio: {latest.get('volume_ratio', 1):.2f}

Recent Highs/Lows:
- 20-Day High: ${last_month['high'].max():.2f}
- 20-Day Low: ${last_month['low'].min():.2f}
- Distance from High: {(latest['close'] / last_month['high'].max() - 1) * 100:.2f}%
- Distance from Low: {(latest['close'] / last_month['low'].min() - 1) * 100:.2f}%
"""
        return summary

    def _parse_market_analysis(
        self,
        analysis_text: str,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Parse Claude's analysis into structured format."""
        # Extract key information
        regime = "Unknown"
        trend = "Neutral"
        volatility = "Medium"

        # Simple keyword-based parsing
        text_lower = analysis_text.lower()

        # Regime detection
        if "trending" in text_lower:
            regime = "Trending"
        elif "ranging" in text_lower or "range-bound" in text_lower:
            regime = "Ranging"
        elif "volatile" in text_lower:
            regime = "Volatile"

        # Trend detection
        if "bullish" in text_lower:
            trend = "Bullish"
        elif "bearish" in text_lower:
            trend = "Bearish"

        # Volatility detection
        if "high volatility" in text_lower or "very volatile" in text_lower:
            volatility = "High"
        elif "low volatility" in text_lower or "calm" in text_lower:
            volatility = "Low"

        return {
            'regime': regime,
            'trend': trend,
            'volatility': volatility,
            'analysis_text': analysis_text,
            'timestamp': data.index[-1],
            'current_price': data.iloc[-1]['close']
        }

    def _get_default_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get default analysis based on technical indicators."""
        latest = data.iloc[-1]

        # Simple regime detection
        volatility = latest.get('volatility_20', 0.15)
        rsi = latest.get('rsi', 50)

        regime = "Volatile" if volatility > 0.25 else "Calm"
        trend = "Bullish" if rsi > 55 else ("Bearish" if rsi < 45 else "Neutral")

        return {
            'regime': regime,
            'trend': trend,
            'volatility': "High" if volatility > 0.25 else "Low",
            'analysis_text': "Default technical analysis",
            'timestamp': data.index[-1],
            'current_price': latest['close']
        }

    def identify_opportunities(
        self,
        data: pd.DataFrame,
        strategies: List[str],
        symbol: str = "QQQ"
    ) -> Dict[str, Any]:
        """
        Identify trading opportunities using Claude.

        Args:
            data: Market data
            strategies: List of available strategies
            symbol: Trading symbol

        Returns:
            Dictionary with opportunity analysis
        """
        logger.info(f"Identifying opportunities for {symbol}")

        market_summary = self._prepare_market_summary(data.tail(60), symbol)

        prompt = f"""Based on the following market data for {symbol}, identify the best trading opportunities.

{market_summary}

Available Strategies:
{', '.join(strategies)}

Please analyze:
1. Which strategies are most suitable for current market conditions?
2. Specific entry/exit points
3. Risk/reward assessment
4. Time horizon recommendation
5. Confidence level (1-10)

Provide actionable insights for trading decisions."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            opportunities = {
                'analysis': response.content[0].text,
                'timestamp': data.index[-1],
                'symbol': symbol,
                'strategies_analyzed': strategies
            }

            return opportunities

        except Exception as e:
            logger.error(f"Error identifying opportunities: {e}")
            return {
                'analysis': "Error in analysis",
                'timestamp': data.index[-1],
                'symbol': symbol
            }

    def generate_market_commentary(
        self,
        data: pd.DataFrame,
        symbol: str = "QQQ"
    ) -> str:
        """
        Generate natural language market commentary.

        Args:
            data: Market data
            symbol: Trading symbol

        Returns:
            Market commentary string
        """
        logger.info(f"Generating market commentary for {symbol}")

        summary = self._prepare_market_summary(data.tail(30), symbol)

        prompt = f"""Generate a concise market commentary for {symbol} based on this data:

{summary}

Write a brief 2-3 paragraph commentary suitable for a daily market report.
Focus on: recent price action, technical setup, and outlook for the next few days."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0.7,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            return response.content[0].text

        except Exception as e:
            logger.error(f"Error generating commentary: {e}")
            return f"Market commentary unavailable due to error: {e}"
