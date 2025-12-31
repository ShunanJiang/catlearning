"""Risk assessment agent using Claude Agent SDK."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from anthropic import Anthropic

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class RiskAssessorAgent:
    """
    Claude-powered risk assessment agent.

    Evaluates risk and provides risk management recommendations.
    """

    def __init__(self, api_key: str, config: Dict[str, Any]):
        """
        Initialize risk assessor agent.

        Args:
            api_key: Anthropic API key
            config: Agent configuration
        """
        self.client = Anthropic(api_key=api_key)
        self.config = config
        self.model = config.get('model', 'claude-sonnet-4-5-20250929')
        self.max_tokens = config.get('max_tokens', 4096)

    def assess_trade_risk(
        self,
        symbol: str,
        proposed_trade: Dict[str, Any],
        portfolio_state: Dict[str, Any],
        market_conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Assess risk of a proposed trade.

        Args:
            symbol: Trading symbol
            proposed_trade: Trade details (direction, size, price, etc.)
            portfolio_state: Current portfolio state
            market_conditions: Current market conditions

        Returns:
            Risk assessment with recommendation
        """
        logger.info(f"Assessing risk for proposed {symbol} trade")

        prompt = f"""Assess the risk of the following proposed trade.

Trade Details:
- Symbol: {symbol}
- Direction: {proposed_trade.get('direction', 'Unknown')}
- Size: {proposed_trade.get('shares', 0)} shares
- Entry Price: ${proposed_trade.get('price', 0):.2f}
- Position Value: ${proposed_trade.get('value', 0):,.2f}

Portfolio State:
- Current Capital: ${portfolio_state.get('capital', 0):,.2f}
- Existing Positions: {portfolio_state.get('num_positions', 0)}
- Current Drawdown: {portfolio_state.get('drawdown', 0):.2%}
- Portfolio Beta: {portfolio_state.get('beta', 1.0):.2f}

Market Conditions:
- Regime: {market_conditions.get('regime', 'Unknown')}
- Volatility: {market_conditions.get('volatility', 'Medium')}
- Trend: {market_conditions.get('trend', 'Neutral')}

Risk Parameters:
- Max Position Size: 30% of capital
- Max Drawdown Limit: 15%
- Stop Loss: 5%

Provide:
1. Risk Level (Low/Medium/High)
2. Key Risk Factors
3. Suggested Position Size Adjustment (if needed)
4. Recommendation (Approve/Reject/Modify)
5. Alternative Approaches (if rejecting)

Be specific and quantitative where possible."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=0.4,  # Lower temperature for risk assessment
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            assessment_text = response.content[0].text

            # Parse assessment
            assessment = self._parse_risk_assessment(assessment_text)
            assessment['full_analysis'] = assessment_text
            assessment['symbol'] = symbol
            assessment['timestamp'] = pd.Timestamp.now()

            logger.info(f"Risk assessment: {assessment.get('risk_level', 'Unknown')}, "
                       f"Recommendation: {assessment.get('recommendation', 'Unknown')}")

            return assessment

        except Exception as e:
            logger.error(f"Error assessing trade risk: {e}")
            return {
                'risk_level': 'High',
                'recommendation': 'Reject',
                'reasoning': f"Error in assessment: {e}",
                'symbol': symbol
            }

    def _parse_risk_assessment(self, text: str) -> Dict[str, Any]:
        """Parse risk assessment from Claude's response."""
        text_lower = text.lower()

        # Detect risk level
        if "low risk" in text_lower or "risk level: low" in text_lower:
            risk_level = "Low"
        elif "high risk" in text_lower or "risk level: high" in text_lower:
            risk_level = "High"
        else:
            risk_level = "Medium"

        # Detect recommendation
        if "approve" in text_lower and "not" not in text_lower.split("approve")[0][-20:]:
            recommendation = "Approve"
        elif "reject" in text_lower:
            recommendation = "Reject"
        elif "modify" in text_lower or "adjust" in text_lower:
            recommendation = "Modify"
        else:
            recommendation = "Review"

        return {
            'risk_level': risk_level,
            'recommendation': recommendation,
            'reasoning': text
        }

    def portfolio_stress_test(
        self,
        portfolio: Dict[str, Any],
        scenarios: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Perform portfolio stress test using Claude.

        Args:
            portfolio: Current portfolio state
            scenarios: List of stress scenarios to test

        Returns:
            Stress test results and recommendations
        """
        logger.info("Performing portfolio stress test")

        scenario_desc = "\n".join([
            f"- {s.get('name', 'Scenario')}: {s.get('description', '')}"
            for s in scenarios
        ])

        prompt = f"""Perform a stress test on the following portfolio.

Portfolio Composition:
- Total Value: ${portfolio.get('total_value', 0):,.2f}
- Number of Positions: {portfolio.get('num_positions', 0)}
- Largest Position: {portfolio.get('largest_position_pct', 0):.1%}
- Current Drawdown: {portfolio.get('current_drawdown', 0):.2%}
- Beta: {portfolio.get('beta', 1.0):.2f}
- Leverage: {portfolio.get('leverage', 1.0):.2f}x

Stress Scenarios to Consider:
{scenario_desc}

Additional Scenarios:
- Market crash (-20% in one week)
- Volatility spike (VIX > 40)
- Interest rate shock (+2% in one month)
- Liquidity crisis (spreads widen 5x)

Analyze:
1. Portfolio resilience to each scenario
2. Estimated losses in worst-case scenarios
3. Weak points in current allocation
4. Hedging recommendations
5. Position adjustments to reduce risk

Provide quantitative estimates where possible."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=0.5,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            return {
                'stress_test_results': response.content[0].text,
                'timestamp': pd.Timestamp.now(),
                'portfolio_value': portfolio.get('total_value', 0)
            }

        except Exception as e:
            logger.error(f"Error in stress test: {e}")
            return {
                'stress_test_results': f"Error: {e}",
                'timestamp': pd.Timestamp.now()
            }

    def evaluate_correlation_risk(
        self,
        positions: Dict[str, Dict[str, Any]],
        correlation_matrix: Optional[pd.DataFrame] = None
    ) -> str:
        """
        Evaluate correlation risk in portfolio.

        Args:
            positions: Dictionary of current positions
            correlation_matrix: Correlation matrix of assets

        Returns:
            Correlation risk analysis
        """
        logger.info("Evaluating correlation risk")

        positions_desc = "\n".join([
            f"- {symbol}: {details.get('shares', 0)} shares, "
            f"Value: ${details.get('value', 0):,.2f}"
            for symbol, details in positions.items()
        ])

        corr_desc = ""
        if correlation_matrix is not None:
            corr_desc = f"\nCorrelation Matrix:\n{correlation_matrix.to_string()}"

        prompt = f"""Analyze correlation risk in this portfolio.

Current Positions:
{positions_desc}
{corr_desc}

Evaluate:
1. Concentration risk from correlated positions
2. Diversification effectiveness
3. Potential for cascade effects during market stress
4. Recommended position adjustments
5. Hedging strategies to reduce correlation risk

Focus on practical risk mitigation strategies."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                temperature=0.5,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            return response.content[0].text

        except Exception as e:
            logger.error(f"Error evaluating correlation risk: {e}")
            return f"Correlation risk evaluation error: {e}"

    def analyze_tail_risk(
        self,
        returns: pd.Series,
        var_95: float,
        cvar_95: float
    ) -> Dict[str, Any]:
        """
        Analyze tail risk using Claude.

        Args:
            returns: Return series
            var_95: Value at Risk at 95% confidence
            cvar_95: Conditional VaR at 95%

        Returns:
            Tail risk analysis
        """
        logger.info("Analyzing tail risk")

        # Calculate additional statistics
        worst_returns = returns.nsmallest(10)
        worst_desc = "\n".join([
            f"- {date.date()}: {ret:.2%}"
            for date, ret in worst_returns.items()
        ])

        prompt = f"""Analyze tail risk based on the following statistics.

Risk Metrics:
- VaR (95%): {var_95:.2%}
- CVaR (95%): {cvar_95:.2%}
- Worst Return: {returns.min():.2%}
- Skewness: {returns.skew():.2f}
- Kurtosis: {returns.kurtosis():.2f}

10 Worst Days:
{worst_desc}

Evaluate:
1. Severity of tail risk
2. Fat-tail characteristics
3. Likelihood of extreme losses
4. Adequacy of current risk limits
5. Tail risk hedging recommendations

Provide specific, actionable insights."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                temperature=0.5,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            return {
                'tail_risk_analysis': response.content[0].text,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'timestamp': pd.Timestamp.now()
            }

        except Exception as e:
            logger.error(f"Error analyzing tail risk: {e}")
            return {
                'tail_risk_analysis': f"Error: {e}",
                'var_95': var_95,
                'cvar_95': cvar_95
            }
