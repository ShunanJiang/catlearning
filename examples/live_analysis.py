"""
Live market analysis example.

This example shows how to use Claude AI agents for:
1. Real-time market regime detection
2. Trading opportunity identification
3. Risk assessment
4. Market commentary generation
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.trading_agent import QuantTradingAgent
from src.config import load_config
import pandas as pd


def main():
    """Run live market analysis."""
    print("="*80)
    print("Live Market Analysis with Claude AI")
    print("="*80)

    # Initialize agent
    config = load_config("config.yaml")
    agent = QuantTradingAgent(config)

    if not agent.market_analyzer:
        print("\nERROR: Claude API key not configured!")
        print("Please set ANTHROPIC_API_KEY in .env file")
        return

    # Fetch recent data
    print("\nFetching recent market data...")
    data = agent.fetch_data("QQQ")

    print(f"Analyzing {len(data)} days of market data")
    print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"Current price: ${data.iloc[-1]['close']:.2f}")

    # Market regime analysis
    print("\n" + "="*80)
    print("MARKET REGIME ANALYSIS")
    print("="*80)

    try:
        regime_analysis = agent.market_analyzer.analyze_market_regime(data, "QQQ")
        print(f"\nRegime: {regime_analysis.get('regime', 'Unknown')}")
        print(f"Trend: {regime_analysis.get('trend', 'Neutral')}")
        print(f"Volatility: {regime_analysis.get('volatility', 'Medium')}")
        print(f"\nDetailed Analysis:")
        print(regime_analysis.get('analysis_text', 'N/A'))
    except Exception as e:
        print(f"Error in regime analysis: {e}")

    # Trading opportunities
    print("\n" + "="*80)
    print("TRADING OPPORTUNITIES")
    print("="*80)

    try:
        strategy_names = list(agent.portfolio_manager.strategies.keys())
        opportunities = agent.market_analyzer.identify_opportunities(
            data, strategy_names, "QQQ"
        )
        print(f"\n{opportunities.get('analysis', 'N/A')}")
    except Exception as e:
        print(f"Error identifying opportunities: {e}")

    # Market commentary
    print("\n" + "="*80)
    print("MARKET COMMENTARY")
    print("="*80)

    try:
        commentary = agent.market_analyzer.generate_market_commentary(data, "QQQ")
        print(f"\n{commentary}")
    except Exception as e:
        print(f"Error generating commentary: {e}")

    # Risk assessment
    if agent.risk_assessor:
        print("\n" + "="*80)
        print("RISK ASSESSMENT")
        print("="*80)

        try:
            # Create a sample proposed trade
            current_price = data.iloc[-1]['close']
            proposed_trade = {
                'direction': 'Long',
                'shares': 100,
                'price': current_price,
                'value': current_price * 100
            }

            portfolio_state = {
                'capital': 100000,
                'num_positions': 0,
                'drawdown': 0,
                'beta': 1.0
            }

            market_conditions = regime_analysis

            risk_assessment = agent.risk_assessor.assess_trade_risk(
                symbol="QQQ",
                proposed_trade=proposed_trade,
                portfolio_state=portfolio_state,
                market_conditions=market_conditions
            )

            print(f"\nRisk Level: {risk_assessment.get('risk_level', 'Unknown')}")
            print(f"Recommendation: {risk_assessment.get('recommendation', 'Unknown')}")
            print(f"\nReasoning:")
            print(risk_assessment.get('reasoning', 'N/A'))
        except Exception as e:
            print(f"Error in risk assessment: {e}")

    # Generate full report
    print("\n" + "="*80)
    print("Generating comprehensive market report...")
    report_path = "./results/QQQ/live_market_report.md"
    agent.generate_market_report("QQQ", output_path=report_path)
    print(f"Report saved to {report_path}")

    print("\n" + "="*80)
    print("Live analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
