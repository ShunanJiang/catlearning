# Quick Start Guide

Get started with the QQQ Quantitative Trading Agent in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Anthropic API key (for AI features)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd catlearning
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** If you encounter issues installing `ta-lib`, you may need to install it separately:

- **macOS:** `brew install ta-lib`
- **Ubuntu/Debian:** `sudo apt-get install ta-lib`
- **Windows:** Download from [here](https://github.com/mrjbq7/ta-lib#dependencies)

Alternatively, the framework will work without ta-lib using pandas-ta as fallback.

### 4. Configure API Keys

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```bash
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here  # Optional
```

**Get API Keys:**
- **Anthropic (Claude):** https://console.anthropic.com/
- **Alpha Vantage (Optional):** https://www.alphavantage.co/support/#api-key

## Running Your First Backtest

### Basic Example

```bash
python examples/basic_backtest.py
```

This will:
- Load configuration from `config.yaml`
- Fetch historical QQQ data
- Run backtests for all enabled strategies
- Save results to `./results/QQQ/`
- Display performance metrics

### Expected Output

```
================================================================================
QQQ Quantitative Trading Agent - Basic Backtest
================================================================================

1. Loading configuration...
2. Initializing trading agent...
3. Running backtest for QQQ...

Momentum - Return: 15.23%, Sharpe: 1.45, Trades: 45
MeanReversion - Return: 12.87%, Sharpe: 1.32, Trades: 52
MLHybrid - Return: 18.94%, Sharpe: 1.67, Trades: 38

================================================================================
Backtest complete! Results saved to ./results/QQQ/
================================================================================
```

## Usage Examples

### 1. Multi-Strategy Backtesting

```bash
python examples/multi_strategy.py
```

Compares performance across all strategies and provides Claude AI optimization suggestions.

### 2. Live Market Analysis

```bash
python examples/live_analysis.py
```

Generates real-time market analysis using Claude AI:
- Market regime detection
- Trading opportunities
- Risk assessment
- Market commentary

### 3. Options Trading

```bash
python examples/options_trading.py
```

Demonstrates options strategies:
- Covered calls
- Cash-secured puts
- Greeks calculation
- Options chain analysis

## Customizing Configuration

Edit `config.yaml` to customize:

```yaml
# Adjust trading parameters
trading:
  initial_capital: 100000
  commission: 0.001

# Enable/disable strategies
strategies:
  enabled:
    - momentum
    - mean_reversion
    - ml_hybrid

# Adjust risk management
risk:
  max_position_size: 0.3
  max_drawdown: 0.15
  stop_loss: 0.05
```

## Understanding Results

After running a backtest, check `./results/QQQ/` for:

- `{strategy}_report.html` - Detailed performance report
- `{strategy}_equity.html` - Interactive equity curve
- `{strategy}_dashboard.html` - Comprehensive analytics dashboard
- `market_report.md` - AI-generated market analysis

## Using Programmatically

```python
from src.agents.trading_agent import QuantTradingAgent
from src.config import load_config

# Initialize
config = load_config("config.yaml")
agent = QuantTradingAgent(config)

# Fetch data
data = agent.fetch_data("QQQ", start_date="2020-01-01")

# Run backtest
results = agent.backtest("QQQ")

# Get market analysis (requires Claude API key)
report = agent.generate_market_report("QQQ")
print(report)
```

## Key Features

✅ **Multiple Strategies**: Momentum, Mean Reversion, ML-based, Options
✅ **Claude AI Integration**: Market analysis, strategy optimization, risk assessment
✅ **Comprehensive Backtesting**: Walk-forward analysis, realistic slippage/commission
✅ **Risk Management**: Position sizing, stop-loss, drawdown limits
✅ **Options Support**: Covered calls, protective puts, Greeks calculation
✅ **Performance Analytics**: Sharpe, Sortino, drawdown, win rate, and more
✅ **Interactive Visualizations**: Equity curves, drawdowns, returns distributions

## Troubleshooting

### Import Errors

If you get import errors, ensure you're in the project root and virtual environment is activated:

```bash
cd /path/to/catlearning
source venv/bin/activate
python examples/basic_backtest.py
```

### API Key Issues

If Claude AI features don't work:
1. Check `.env` file has `ANTHROPIC_API_KEY` set
2. Verify API key is valid at https://console.anthropic.com/
3. Check you have API credits available

### Data Fetching Issues

If Yahoo Finance data fails:
- Try again (sometimes rate-limited)
- Check internet connection
- Verify symbol is correct (e.g., "QQQ" not "qqq")

### Missing Dependencies

If you get module not found errors:

```bash
pip install -r requirements.txt --upgrade
```

## Next Steps

1. **Explore Strategies**: Check `src/strategies/` to understand how each strategy works
2. **Customize Parameters**: Modify `config.yaml` to optimize for your preferences
3. **Add New Strategies**: Inherit from `BaseStrategy` to create custom strategies
4. **Live Trading**: Integrate with a broker API (not included - for safety)
5. **Advanced Features**: Check `README.md` for detailed documentation

## Getting Help

- **Documentation**: See `README.md` for comprehensive docs
- **Examples**: Check `examples/` directory for more use cases
- **Code**: All source code in `src/` is well-commented

## Important Disclaimer

⚠️ **This software is for educational and research purposes only.**

- Do not use for actual trading without thorough testing
- Past performance does not guarantee future results
- Trading involves risk of loss
- Always paper trade before risking real capital
- Consult a financial advisor before making investment decisions

## License

MIT License - See LICENSE file for details

---

Happy Trading! 🚀📈
