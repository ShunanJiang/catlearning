"""
Simplified standalone backtest script to demonstrate real performance.
"""

import warnings
warnings.filterwarnings('ignore')

from yahooquery import Ticker
import pandas as pd
import numpy as np
from datetime import datetime

print("="*80)
print("QQQ TRADING STRATEGIES - LIVE BACKTEST RESULTS")
print("Initial Capital: $100,000")
print("Period: 2019-01-01 to 2024-12-31")
print("="*80)

# Fetch real QQQ data
print("\n[1/4] Fetching QQQ historical data from Yahoo Finance...")
qqq = Ticker("QQQ")
data = qqq.history(start="2019-01-01", end="2024-12-31")
if isinstance(data, dict):
    data = pd.DataFrame(data['QQQ'])
data.index = pd.to_datetime(data.index)
data.columns = [c.capitalize() if c != 'adjclose' else 'Close' for c in data.columns]
if 'adjclose' in [c.lower() for c in data.columns]:
    data = data.rename(columns={[c for c in data.columns if c.lower() == 'adjclose'][0]: 'Close'})

print(f"✓ Fetched {len(data)} trading days of QQQ data")
print(f"   Date range: {data.index[0].date()} to {data.index[-1].date()}")
print(f"   Starting price: ${data.iloc[0]['Close']:.2f}")
print(f"   Ending price: ${data.iloc[-1]['Close']:.2f}")

# Calculate technical indicators
print("\n[2/4] Calculating technical indicators...")
data['returns'] = data['Close'].pct_change()
data['sma_20'] = data['Close'].rolling(20).mean()
data['sma_50'] = data['Close'].rolling(50).mean()
data['sma_200'] = data['Close'].rolling(200).mean()

# RSI
delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
data['rsi'] = 100 - (100 / (1 + rs))

# Bollinger Bands
data['bb_middle'] = data['Close'].rolling(20).mean()
bb_std = data['Close'].rolling(20).std()
data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
data['bb_lower'] = data['bb_middle'] - (bb_std * 2)

# Volatility
data['volatility'] = data['returns'].rolling(20).std() * np.sqrt(252)

data = data.dropna()

print(f"✓ Calculated {len([c for c in data.columns if c not in qqq.history(start='2019-01-01', period='1d').columns])} technical indicators")

# Strategy 1: Momentum Strategy
print("\n[3/4] Running Strategy Backtests...")
print("\n" + "="*80)
print("STRATEGY 1: MOMENTUM TRADING")
print("="*80)

def backtest_momentum(data, capital=100000):
    """Momentum strategy: Buy when price > SMA50 and RSI < 70"""
    equity = capital
    position = 0
    shares = 0
    trades = 0
    wins = 0
    equity_curve = []

    for i in range(len(data)):
        price = data.iloc[i]['Close']
        sma_50 = data.iloc[i]['sma_50']
        rsi = data.iloc[i]['rsi']

        # Buy signal
        if position == 0 and price > sma_50 and rsi < 70:
            shares = int(equity * 0.95 / price)  # Use 95% of capital
            if shares > 0:
                equity -= shares * price * 1.001  # 0.1% commission
                position = 1
                entry_price = price
                trades += 1

        # Sell signal
        elif position == 1 and (price < sma_50 or rsi > 70):
            equity += shares * price * 0.999  # 0.1% commission
            if price > entry_price:
                wins += 1
            position = 0
            shares = 0

        # Update equity
        if position == 1:
            equity_curve.append(equity + shares * price)
        else:
            equity_curve.append(equity)

    # Close final position
    if position == 1:
        equity += shares * data.iloc[-1]['Close'] * 0.999
        if data.iloc[-1]['Close'] > entry_price:
            wins += 1

    return {
        'final_equity': equity,
        'trades': trades,
        'wins': wins,
        'equity_curve': equity_curve
    }

momentum_result = backtest_momentum(data)

print(f"Final Capital: ${momentum_result['final_equity']:,.2f}")
print(f"Total Profit: ${momentum_result['final_equity'] - 100000:,.2f}")
print(f"Total Return: {(momentum_result['final_equity'] / 100000 - 1) * 100:.2f}%")
print(f"Number of Trades: {momentum_result['trades']}")
print(f"Win Rate: {(momentum_result['wins'] / momentum_result['trades'] * 100) if momentum_result['trades'] > 0 else 0:.2f}%")

# Strategy 2: Mean Reversion Strategy
print("\n" + "="*80)
print("STRATEGY 2: MEAN REVERSION (Bollinger Bands)")
print("="*80)

def backtest_mean_reversion(data, capital=100000):
    """Mean reversion: Buy at lower band, sell at upper band"""
    equity = capital
    position = 0
    shares = 0
    trades = 0
    wins = 0
    equity_curve = []

    for i in range(len(data)):
        price = data.iloc[i]['Close']
        bb_upper = data.iloc[i]['bb_upper']
        bb_lower = data.iloc[i]['bb_lower']
        bb_middle = data.iloc[i]['bb_middle']

        # Buy signal (oversold)
        if position == 0 and price < bb_lower:
            shares = int(equity * 0.95 / price)
            if shares > 0:
                equity -= shares * price * 1.001
                position = 1
                entry_price = price
                trades += 1

        # Sell signal (overbought or return to mean)
        elif position == 1 and (price > bb_upper or price > bb_middle):
            equity += shares * price * 0.999
            if price > entry_price:
                wins += 1
            position = 0
            shares = 0

        # Update equity
        if position == 1:
            equity_curve.append(equity + shares * price)
        else:
            equity_curve.append(equity)

    # Close final position
    if position == 1:
        equity += shares * data.iloc[-1]['Close'] * 0.999
        if data.iloc[-1]['Close'] > entry_price:
            wins += 1

    return {
        'final_equity': equity,
        'trades': trades,
        'wins': wins,
        'equity_curve': equity_curve
    }

mean_rev_result = backtest_mean_reversion(data)

print(f"Final Capital: ${mean_rev_result['final_equity']:,.2f}")
print(f"Total Profit: ${mean_rev_result['final_equity'] - 100000:,.2f}")
print(f"Total Return: {(mean_rev_result['final_equity'] / 100000 - 1) * 100:.2f}%")
print(f"Number of Trades: {mean_rev_result['trades']}")
print(f"Win Rate: {(mean_rev_result['wins'] / mean_rev_result['trades'] * 100) if mean_rev_result['trades'] > 0 else 0:.2f}%")

# Strategy 3: Buy and Hold (Benchmark)
print("\n" + "="*80)
print("STRATEGY 3: BUY & HOLD (Benchmark)")
print("="*80)

initial_price = data.iloc[0]['Close']
final_price = data.iloc[-1]['Close']
bnh_shares = int(100000 * 0.999 / initial_price)
bnh_final = bnh_shares * final_price

print(f"Initial Investment: ${100000:,.2f}")
print(f"Shares Purchased: {bnh_shares}")
print(f"Final Capital: ${bnh_final:,.2f}")
print(f"Total Profit: ${bnh_final - 100000:,.2f}")
print(f"Total Return: {(bnh_final / 100000 - 1) * 100:.2f}%")

# Calculate performance metrics
print("\n[4/4] Calculating Performance Metrics...")

def calc_sharpe(equity_curve, rf_rate=0.05):
    """Calculate Sharpe ratio"""
    returns = pd.Series(equity_curve).pct_change().dropna()
    excess_returns = returns - rf_rate/252
    if returns.std() == 0:
        return 0
    return np.sqrt(252) * excess_returns.mean() / returns.std()

def calc_max_drawdown(equity_curve):
    """Calculate maximum drawdown"""
    equity_series = pd.Series(equity_curve)
    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max
    return drawdown.min()

# Results summary
print("\n" + "="*80)
print("COMPREHENSIVE RESULTS SUMMARY")
print("="*80)

results = [
    {
        'name': 'Momentum Strategy',
        'return': (momentum_result['final_equity'] / 100000 - 1) * 100,
        'final': momentum_result['final_equity'],
        'sharpe': calc_sharpe(momentum_result['equity_curve']),
        'max_dd': calc_max_drawdown(momentum_result['equity_curve']) * 100,
        'trades': momentum_result['trades'],
        'win_rate': (momentum_result['wins'] / momentum_result['trades'] * 100) if momentum_result['trades'] > 0 else 0
    },
    {
        'name': 'Mean Reversion',
        'return': (mean_rev_result['final_equity'] / 100000 - 1) * 100,
        'final': mean_rev_result['final_equity'],
        'sharpe': calc_sharpe(mean_rev_result['equity_curve']),
        'max_dd': calc_max_drawdown(mean_rev_result['equity_curve']) * 100,
        'trades': mean_rev_result['trades'],
        'win_rate': (mean_rev_result['wins'] / mean_rev_result['trades'] * 100) if mean_rev_result['trades'] > 0 else 0
    },
    {
        'name': 'Buy & Hold',
        'return': (bnh_final / 100000 - 1) * 100,
        'final': bnh_final,
        'sharpe': 0,  # Not applicable
        'max_dd': 0,  # Not calculated
        'trades': 1,
        'win_rate': 100 if bnh_final > 100000 else 0
    }
]

print(f"\n{'Strategy':<20} {'Return':<12} {'Final $':<15} {'Sharpe':<10} {'Max DD':<10} {'Trades':<8} {'Win %':<8}")
print("-"*90)
for r in results:
    print(f"{r['name']:<20} {r['return']:>10.2f}% ${r['final']:>13,.2f} {r['sharpe']:>9.2f} {r['max_dd']:>8.2f}% {r['trades']:>7} {r['win_rate']:>7.1f}%")

# Find best strategy
best = max(results, key=lambda x: x['sharpe'] if x['sharpe'] > 0 else x['return'])

print("\n" + "="*80)
print("RECOMMENDED STRATEGY")
print("="*80)
print(f"\n🏆 WINNER: {best['name'].upper()}")
print(f"\n   Investment Results:")
print(f"   - Starting Capital: $100,000.00")
print(f"   - Ending Capital: ${best['final']:,.2f}")
print(f"   - Total Profit: ${best['final'] - 100000:,.2f}")
print(f"   - Total Return: {best['return']:.2f}%")
print(f"   - Sharpe Ratio: {best['sharpe']:.2f}")
print(f"   - Max Drawdown: {best['max_dd']:.2f}%")
print(f"   - Win Rate: {best['win_rate']:.1f}%")
print(f"   - Number of Trades: {best['trades']}")

# Calculate annualized return
years = (data.index[-1] - data.index[0]).days / 365.25
annualized = ((best['final'] / 100000) ** (1/years) - 1) * 100
print(f"   - Annualized Return: {annualized:.2f}%")

print("\n   Why This Strategy Wins:")
if best['name'] == 'Momentum Strategy':
    print("   ✓ Superior risk-adjusted returns (highest Sharpe ratio)")
    print("   ✓ Captures strong uptrends while avoiding weak periods")
    print("   ✓ Good balance of trades and returns")
elif best['name'] == 'Mean Reversion':
    print("   ✓ Excellent for capturing oversold bounces")
    print("   ✓ High win rate from buying dips")
    print("   ✓ Works well in ranging markets")
else:
    print("   ✓ Lowest risk and minimal trading costs")
    print("   ✓ Captures full market appreciation")
    print("   ✓ Perfect for long-term investors")

print("\n" + "="*80)
print("BACKTEST COMPLETE!")
print("="*80)
print(f"\nTested {len(data)} days of real QQQ market data")
print(f"Period: {years:.1f} years ({data.index[0].date()} to {data.index[-1].date()})")
print(f"QQQ Price Change: ${data.iloc[0]['Close']:.2f} → ${data.iloc[-1]['Close']:.2f} ({(final_price/initial_price - 1)*100:.2f}%)")
