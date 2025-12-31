"""
Multi-strategy ensemble example.

This example demonstrates:
1. Running multiple strategies simultaneously
2. Comparing strategy performance
3. Portfolio-level analysis
4. Strategy optimization with Claude
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.trading_agent import QuantTradingAgent
from src.config import load_config
from src.backtesting.performance import PerformanceAnalyzer
import matplotlib.pyplot as plt


def main():
    """Run multi-strategy backtest."""
    print("="*80)
    print("Multi-Strategy Ensemble Backtesting")
    print("="*80)

    # Initialize agent
    config = load_config("config.yaml")
    agent = QuantTradingAgent(config)

    # Fetch data
    print("\nFetching market data...")
    data = agent.fetch_data("QQQ", start_date="2020-01-01")

    # Run backtests
    print("\nRunning backtests for all strategies...")
    results = agent.backtest("QQQ", start_date="2020-01-01", save_results=True)

    # Compare strategies
    print("\n" + "="*80)
    print("STRATEGY COMPARISON")
    print("="*80)

    comparison_data = []
    for strategy_name, result in results.items():
        comparison_data.append({
            'Strategy': strategy_name,
            'Total Return': f"{result.total_return:.2%}",
            'Sharpe Ratio': f"{result.sharpe_ratio:.2f}",
            'Max Drawdown': f"{result.max_drawdown:.2%}",
            'Win Rate': f"{result.win_rate:.2%}",
            'Trades': result.num_trades
        })

    # Print comparison table
    print(f"\n{'Strategy':<20} {'Return':<12} {'Sharpe':<10} {'Max DD':<12} {'Win Rate':<10} {'Trades':<8}")
    print("-"*80)
    for data in comparison_data:
        print(f"{data['Strategy']:<20} {data['Total Return']:<12} {data['Sharpe Ratio']:<10} "
              f"{data['Max Drawdown']:<12} {data['Win Rate']:<10} {data['Trades']:<8}")

    # Find best strategy
    best_strategy = max(results.items(), key=lambda x: x[1].sharpe_ratio)
    print(f"\nBest Strategy (by Sharpe Ratio): {best_strategy[0]}")

    # Optimize strategies with Claude
    if agent.strategy_optimizer:
        print("\n" + "="*80)
        print("STRATEGY OPTIMIZATION (Claude AI)")
        print("="*80)
        try:
            agent.optimize_strategies(data)
        except Exception as e:
            print(f"Optimization error: {e}")

    print("\n" + "="*80)
    print("Multi-strategy analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
