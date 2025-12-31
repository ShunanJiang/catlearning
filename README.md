# QQQ Quantitative Trading Agent

A comprehensive quantitative trading framework powered by Claude Agent SDK for developing and backtesting trading strategies for QQQ and related derivatives.

## Features

- **Multi-Strategy Framework**: Support for momentum, mean reversion, ML-based, and options strategies
- **Claude Agent Integration**: AI-powered market analysis, strategy optimization, and risk assessment
- **Comprehensive Backtesting**: Walk-forward analysis, performance metrics, and visualization
- **Derivatives Support**: Options strategies including covered calls, protective puts, and spreads
- **Risk Management**: Position sizing, stop-loss, drawdown limits, and portfolio optimization
- **Modular Architecture**: Reusable components for data collection, strategies, and analysis

## Project Structure

```
catlearning/
├── src/
│   ├── agents/          # Claude Agent SDK implementations
│   ├── data/            # Data collection and management
│   ├── strategies/      # Trading strategies
│   ├── backtesting/     # Backtesting engine
│   ├── risk/            # Risk management
│   ├── derivatives/     # Options and derivatives
│   ├── portfolio/       # Portfolio management
│   └── utils/           # Utilities and helpers
├── data/                # Historical data storage
├── results/             # Backtesting results
├── tests/               # Unit tests
├── examples/            # Example scripts
├── config.yaml          # Configuration
├── requirements.txt     # Dependencies
└── README.md
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your API keys
```

## Quick Start

```python
from src.agents.trading_agent import QuantTradingAgent
from src.config import load_config

# Load configuration
config = load_config("config.yaml")

# Initialize agent
agent = QuantTradingAgent(config)

# Run backtest
results = agent.backtest()

# Display results
agent.display_results(results)

# Generate report
agent.generate_report(results, output_path="./results/report.html")
```

## Configuration

Edit `config.yaml` to customize:
- Trading parameters (capital, commission, slippage)
- Data sources and date ranges
- Strategy selection and parameters
- Risk management rules
- Claude agent settings

## Strategies

### 1. Momentum Strategy
- Trend following based on price momentum
- Configurable lookback periods and thresholds

### 2. Mean Reversion Strategy
- Statistical arbitrage using Bollinger Bands
- Entry/exit based on standard deviation

### 3. ML Hybrid Strategy
- Machine learning-based signal generation
- Feature engineering with Claude Agent
- Adaptive model retraining

### 4. Options Income Strategy
- Covered calls and cash-secured puts
- Delta-neutral strategies
- Premium collection optimization

## Claude Agent SDK Integration

### Market Analyzer Agent
- Analyzes market conditions and regime detection
- Provides contextual insights for strategy selection

### Strategy Optimizer Agent
- Optimizes strategy parameters
- Discovers new trading opportunities

### Risk Assessor Agent
- Real-time risk evaluation
- Portfolio stress testing

### Signal Interpreter Agent
- Validates trading signals
- Provides confidence scores

## Usage Examples

See `examples/` directory for detailed examples:
- `basic_backtest.py` - Simple backtesting example
- `multi_strategy.py` - Multiple strategies with ensemble
- `options_trading.py` - Options strategies
- `live_analysis.py` - Real-time market analysis
- `claude_optimization.py` - AI-powered optimization

## Performance Metrics

The framework tracks comprehensive metrics:
- Returns (total, annualized, rolling)
- Risk-adjusted returns (Sharpe, Sortino, Calmar)
- Drawdown analysis
- Win rate and profit factor
- Market exposure (alpha, beta)

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_strategies.py

# Run with coverage
pytest --cov=src tests/
```

## Contributing

This is a modular framework designed for extensibility. To add new strategies:

1. Inherit from `BaseStrategy` class
2. Implement `generate_signals()` method
3. Add configuration to `config.yaml`
4. Add tests in `tests/`

## License

MIT License

## Disclaimer

This software is for educational and research purposes only. Do not use for actual trading without thorough testing and risk assessment. Past performance does not guarantee future results.
