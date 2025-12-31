"""
Basic backtesting example for QQQ trading.

This example shows how to:
1. Initialize the trading agent
2. Fetch historical data
3. Run backtests
4. Display results
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.trading_agent import QuantTradingAgent
from src.config import load_config


def main():
    """Run basic backtest."""
    print("="*80)
    print("QQQ Quantitative Trading Agent - Basic Backtest")
    print("="*80)

    # Load configuration
    print("\n1. Loading configuration...")
    config = load_config("config.yaml")

    # Initialize trading agent
    print("2. Initializing trading agent...")
    agent = QuantTradingAgent(config)

    # Run backtest
    print("3. Running backtest for QQQ...")
    print("   This may take a few minutes...")
    results = agent.backtest(
        symbol="QQQ",
        start_date="2020-01-01",
        end_date=None,  # Use current date
        save_results=True
    )

    # Display results
    print("\n4. Backtest Results:")
    agent.display_results(results)

    # Generate market report
    print("\n5. Generating market analysis report...")
    report = agent.generate_market_report(
        symbol="QQQ",
        output_path="./results/QQQ/market_report.md"
    )
    print(f"\nMarket Report Preview:\n{report[:500]}...")

    print("\n" + "="*80)
    print("Backtest complete! Results saved to ./results/QQQ/")
    print("="*80)


if __name__ == "__main__":
    main()
