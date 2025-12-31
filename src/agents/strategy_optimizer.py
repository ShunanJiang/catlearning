"""Strategy optimization agent using Claude Agent SDK."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from anthropic import Anthropic
import json

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class StrategyOptimizerAgent:
    """
    Claude-powered strategy optimization agent.

    Optimizes strategy parameters and suggests improvements.
    """

    def __init__(self, api_key: str, config: Dict[str, Any]):
        """
        Initialize strategy optimizer agent.

        Args:
            api_key: Anthropic API key
            config: Agent configuration
        """
        self.client = Anthropic(api_key=api_key)
        self.config = config
        self.model = config.get('model', 'claude-sonnet-4-5-20250929')
        self.max_tokens = config.get('max_tokens', 4096)

    def optimize_parameters(
        self,
        strategy_name: str,
        current_params: Dict[str, Any],
        performance_metrics: Dict[str, float],
        market_conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters using Claude.

        Args:
            strategy_name: Name of the strategy
            current_params: Current parameter values
            performance_metrics: Recent performance metrics
            market_conditions: Current market conditions

        Returns:
            Dictionary with optimized parameters and reasoning
        """
        logger.info(f"Optimizing parameters for {strategy_name}")

        prompt = f"""Optimize the parameters for a {strategy_name} trading strategy.

Current Parameters:
{json.dumps(current_params, indent=2)}

Recent Performance:
- Sharpe Ratio: {performance_metrics.get('sharpe_ratio', 0):.2f}
- Win Rate: {performance_metrics.get('win_rate', 0):.2%}
- Max Drawdown: {performance_metrics.get('max_drawdown', 0):.2%}
- Total Return: {performance_metrics.get('total_return', 0):.2%}

Market Conditions:
- Regime: {market_conditions.get('regime', 'Unknown')}
- Trend: {market_conditions.get('trend', 'Neutral')}
- Volatility: {market_conditions.get('volatility', 'Medium')}

Task:
1. Analyze the current performance
2. Suggest parameter adjustments to improve risk-adjusted returns
3. Consider current market conditions
4. Provide specific parameter values
5. Explain the reasoning for each change

Format response as JSON with:
{{
  "suggested_params": {{"param_name": value, ...}},
  "reasoning": "explanation",
  "expected_improvement": "description"
}}"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=0.5,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            # Try to parse JSON response
            content = response.content[0].text
            optimization = self._parse_optimization_response(content)

            logger.info(f"Parameter optimization complete for {strategy_name}")

            return optimization

        except Exception as e:
            logger.error(f"Error optimizing parameters: {e}")
            return {
                'suggested_params': current_params,
                'reasoning': f"Error in optimization: {e}",
                'expected_improvement': "None"
            }

    def _parse_optimization_response(self, content: str) -> Dict[str, Any]:
        """Parse Claude's optimization response."""
        try:
            # Try to extract JSON from response
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1

            if start_idx >= 0 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # Fallback: return text as reasoning
                return {
                    'suggested_params': {},
                    'reasoning': content,
                    'expected_improvement': "See reasoning"
                }
        except Exception as e:
            logger.warning(f"Could not parse optimization response: {e}")
            return {
                'suggested_params': {},
                'reasoning': content,
                'expected_improvement': "Manual review needed"
            }

    def suggest_strategy_combinations(
        self,
        available_strategies: List[str],
        market_regime: str,
        performance_history: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Suggest optimal strategy combinations using Claude.

        Args:
            available_strategies: List of available strategies
            market_regime: Current market regime
            performance_history: Historical performance of each strategy

        Returns:
            Dictionary with strategy combination recommendations
        """
        logger.info("Generating strategy combination suggestions")

        # Prepare performance summary
        perf_summary = "\n".join([
            f"{strategy}: Sharpe={metrics.get('sharpe_ratio', 0):.2f}, "
            f"Return={metrics.get('total_return', 0):.2%}"
            for strategy, metrics in performance_history.items()
        ])

        prompt = f"""Suggest optimal strategy combinations for the current market regime.

Available Strategies:
{', '.join(available_strategies)}

Historical Performance:
{perf_summary}

Current Market Regime: {market_regime}

Task:
1. Recommend 2-3 strategy combinations
2. Suggest allocation weights for each strategy
3. Explain why these combinations work well in the current regime
4. Identify potential risks and mitigation

Provide specific, actionable recommendations."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=0.6,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            suggestions = {
                'recommendations': response.content[0].text,
                'market_regime': market_regime,
                'timestamp': pd.Timestamp.now()
            }

            return suggestions

        except Exception as e:
            logger.error(f"Error generating combinations: {e}")
            return {
                'recommendations': "Error generating recommendations",
                'market_regime': market_regime,
                'error': str(e)
            }

    def evaluate_backtest_results(
        self,
        strategy_name: str,
        results_summary: Dict[str, Any],
        equity_curve_stats: Dict[str, float]
    ) -> str:
        """
        Evaluate backtest results and provide insights.

        Args:
            strategy_name: Strategy name
            results_summary: Summary of backtest results
            equity_curve_stats: Statistics about equity curve

        Returns:
            Evaluation text
        """
        logger.info(f"Evaluating backtest results for {strategy_name}")

        prompt = f"""Evaluate the backtest results for {strategy_name} strategy.

Results Summary:
- Total Return: {results_summary.get('total_return', 0):.2%}
- Annualized Return: {results_summary.get('annualized_return', 0):.2%}
- Sharpe Ratio: {results_summary.get('sharpe_ratio', 0):.2f}
- Sortino Ratio: {results_summary.get('sortino_ratio', 0):.2f}
- Max Drawdown: {results_summary.get('max_drawdown', 0):.2%}
- Win Rate: {results_summary.get('win_rate', 0):.2%}
- Profit Factor: {results_summary.get('profit_factor', 0):.2f}
- Number of Trades: {results_summary.get('num_trades', 0)}

Equity Curve Statistics:
- Mean Daily Return: {equity_curve_stats.get('mean_return', 0):.4f}
- Volatility: {equity_curve_stats.get('volatility', 0):.4f}
- Best Day: {equity_curve_stats.get('best_day', 0):.2%}
- Worst Day: {equity_curve_stats.get('worst_day', 0):.2%}

Provide:
1. Overall assessment of the strategy's performance
2. Strengths and weaknesses
3. Comparison to typical benchmarks (e.g., Sharpe > 1.0 is good)
4. Suggestions for improvement
5. Risk assessment
6. Recommendation (deploy/refine/reject)"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=0.6,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            return response.content[0].text

        except Exception as e:
            logger.error(f"Error evaluating results: {e}")
            return f"Evaluation error: {e}"

    def generate_feature_ideas(
        self,
        current_features: List[str],
        strategy_type: str,
        target_asset: str = "QQQ"
    ) -> List[str]:
        """
        Generate ideas for new features to improve strategy.

        Args:
            current_features: Currently used features
            strategy_type: Type of strategy (momentum, mean_reversion, ml)
            target_asset: Target trading asset

        Returns:
            List of suggested new features
        """
        logger.info(f"Generating feature ideas for {strategy_type} strategy")

        prompt = f"""Suggest new features to improve a {strategy_type} trading strategy for {target_asset}.

Current Features:
{', '.join(current_features)}

Task:
Generate 5-10 new feature ideas that could improve the strategy.
Focus on:
- Technical indicators not currently used
- Market microstructure features
- Inter-market relationships
- Sentiment indicators
- Volume-based features

For each feature, briefly explain why it might be valuable.
Format as a numbered list."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                temperature=0.7,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            # Parse suggestions from response
            content = response.content[0].text
            suggestions = [
                line.strip() for line in content.split('\n')
                if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('-'))
            ]

            return suggestions

        except Exception as e:
            logger.error(f"Error generating features: {e}")
            return []
