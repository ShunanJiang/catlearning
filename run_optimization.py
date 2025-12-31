"""
Run actual backtesting and optimization to find the best strategy.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

# Run the backtest
from src.agents.trading_agent import QuantTradingAgent
from src.config import load_config

def main():
    print("="*80)
    print("RUNNING LIVE BACKTEST - QQQ TRADING STRATEGIES")
    print("Initial Capital: $100,000")
    print("="*80)

    # Load config
    print("\n[1/4] Loading configuration...")
    config = load_config("config.yaml")

    # Override initial capital to $100k
    config.set('trading.initial_capital', 100000)

    # Initialize agent
    print("[2/4] Initializing trading agent...")
    agent = QuantTradingAgent(config)

    # Run backtest with real data
    print("[3/4] Running backtests on QQQ (2019-2024)...")
    print("      Fetching historical data from Yahoo Finance...")

    results = agent.backtest(
        symbol="QQQ",
        start_date="2019-01-01",
        end_date=None,
        save_results=False
    )

    print("\n[4/4] Analyzing results and finding best strategy...")

    # Display all results
    print("\n" + "="*80)
    print("BACKTEST RESULTS - ALL STRATEGIES")
    print("="*80)
    agent.display_results(results)

    # Find best strategy by different metrics
    print("\n" + "="*80)
    print("STRATEGY RANKINGS")
    print("="*80)

    strategies_data = []
    for name, result in results.items():
        strategies_data.append({
            'name': name,
            'total_return': result.total_return,
            'sharpe': result.sharpe_ratio,
            'sortino': result.sortino_ratio,
            'max_dd': result.max_drawdown,
            'win_rate': result.win_rate,
            'final_capital': result.final_capital
        })

    # Best by total return
    best_return = max(strategies_data, key=lambda x: x['total_return'])
    print(f"\n🏆 Best Total Return: {best_return['name']}")
    print(f"   Return: {best_return['total_return']:.2%}")
    print(f"   $100,000 → ${best_return['final_capital']:,.2f}")
    print(f"   Profit: ${best_return['final_capital'] - 100000:,.2f}")

    # Best by Sharpe ratio (risk-adjusted)
    best_sharpe = max(strategies_data, key=lambda x: x['sharpe'])
    print(f"\n⭐ Best Risk-Adjusted (Sharpe): {best_sharpe['name']}")
    print(f"   Sharpe Ratio: {best_sharpe['sharpe']:.2f}")
    print(f"   Return: {best_sharpe['total_return']:.2%}")
    print(f"   Final Capital: ${best_sharpe['final_capital']:,.2f}")

    # Best by Sortino ratio (downside risk)
    best_sortino = max(strategies_data, key=lambda x: x['sortino'])
    print(f"\n💎 Best Downside Protection (Sortino): {best_sortino['name']}")
    print(f"   Sortino Ratio: {best_sortino['sortino']:.2f}")
    print(f"   Max Drawdown: {best_sortino['max_dd']:.2%}")
    print(f"   Final Capital: ${best_sortino['final_capital']:,.2f}")

    # Best by win rate
    best_winrate = max(strategies_data, key=lambda x: x['win_rate'])
    print(f"\n🎯 Highest Win Rate: {best_winrate['name']}")
    print(f"   Win Rate: {best_winrate['win_rate']:.2%}")
    print(f"   Final Capital: ${best_winrate['final_capital']:,.2f}")

    # Recommendation
    print("\n" + "="*80)
    print("RECOMMENDED STRATEGY")
    print("="*80)

    # Use Sharpe ratio as primary metric for recommendation
    print(f"\n🚀 RECOMMENDED: {best_sharpe['name']}")
    print(f"\n   Why: Best risk-adjusted returns with Sharpe ratio of {best_sharpe['sharpe']:.2f}")
    print(f"\n   Performance Summary:")
    print(f"   - Starting Capital: $100,000.00")
    print(f"   - Ending Capital: ${best_sharpe['final_capital']:,.2f}")
    print(f"   - Total Profit: ${best_sharpe['final_capital'] - 100000:,.2f}")
    print(f"   - Total Return: {best_sharpe['total_return']:.2%}")
    print(f"   - Win Rate: {best_sharpe['win_rate']:.2%}")
    print(f"   - Max Drawdown: {best_sharpe['max_dd']:.2%}")

    # Calculate annualized return
    result = results[best_sharpe['name']]
    years = (result.end_date - result.start_date).days / 365.25
    annualized_return = (1 + best_sharpe['total_return']) ** (1/years) - 1

    print(f"   - Annualized Return: {annualized_return:.2%}")
    print(f"   - Number of Trades: {result.num_trades}")

    print("\n" + "="*80)

    return results, best_sharpe

if __name__ == "__main__":
    results, best_strategy = main()
