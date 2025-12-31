"""Main trading agent orchestrator integrating all components."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path

from ..config import Config
from ..data.data_manager import DataManager
from ..strategies.momentum_strategy import MomentumStrategy
from ..strategies.mean_reversion_strategy import MeanReversionStrategy
from ..strategies.ml_strategy import MLHybridStrategy
from ..derivatives.options_strategies import CoveredCallStrategy, CashSecuredPutStrategy
from ..backtesting.backtest_engine import BacktestEngine, BacktestResult
from ..backtesting.performance import PerformanceAnalyzer
from ..risk.risk_manager import RiskManager
from ..portfolio.portfolio_manager import PortfolioManager
from .market_analyzer import MarketAnalyzerAgent
from .strategy_optimizer import StrategyOptimizerAgent
from .risk_assessor import RiskAssessorAgent
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class QuantTradingAgent:
    """
    Main quantitative trading agent orchestrating all components.

    Integrates:
    - Data collection
    - Strategy execution
    - Risk management
    - Portfolio management
    - Claude AI agents for analysis and optimization
    - Backtesting and performance analysis
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the trading agent.

        Args:
            config: Configuration object (loads from config.yaml if None)
        """
        if config is None:
            config = Config("config.yaml")

        self.config = config
        logger.info("Initializing Quant Trading Agent")

        # Initialize data manager
        self.data_manager = DataManager(config.data_config)

        # Initialize risk manager
        self.risk_manager = RiskManager(config.risk)

        # Initialize portfolio manager
        self.portfolio_manager = PortfolioManager(
            initial_capital=config.trading.get('initial_capital', 100000),
            risk_manager=self.risk_manager,
            config=config.trading
        )

        # Initialize backtest engine
        self.backtest_engine = BacktestEngine(config.trading)

        # Initialize Claude agents
        self._initialize_claude_agents()

        # Initialize strategies
        self._initialize_strategies()

        logger.info("Quant Trading Agent initialized successfully")

    def _initialize_claude_agents(self):
        """Initialize Claude AI agents."""
        claude_config = self.config.claude
        api_key = claude_config.get('api_key')

        if not api_key:
            logger.warning("Claude API key not found, AI features will be disabled")
            self.market_analyzer = None
            self.strategy_optimizer = None
            self.risk_assessor = None
            return

        agents_config = claude_config.get('agents', {})

        # Market analyzer
        if agents_config.get('market_analyzer', {}).get('enabled', True):
            self.market_analyzer = MarketAnalyzerAgent(api_key, claude_config)
            logger.info("Market Analyzer Agent initialized")
        else:
            self.market_analyzer = None

        # Strategy optimizer
        if agents_config.get('strategy_optimizer', {}).get('enabled', True):
            self.strategy_optimizer = StrategyOptimizerAgent(api_key, claude_config)
            logger.info("Strategy Optimizer Agent initialized")
        else:
            self.strategy_optimizer = None

        # Risk assessor
        if agents_config.get('risk_assessor', {}).get('enabled', True):
            self.risk_assessor = RiskAssessorAgent(api_key, claude_config)
            logger.info("Risk Assessor Agent initialized")
        else:
            self.risk_assessor = None

    def _initialize_strategies(self):
        """Initialize trading strategies based on configuration."""
        strategies_config = self.config.strategies
        enabled_strategies = strategies_config.get('enabled', [])

        logger.info(f"Initializing strategies: {enabled_strategies}")

        # Momentum strategy
        if 'momentum' in enabled_strategies:
            momentum_config = strategies_config.get('momentum', {})
            strategy = MomentumStrategy(momentum_config)
            self.portfolio_manager.add_strategy(strategy, allocation=0.25)
            logger.info("Momentum strategy added")

        # Mean reversion strategy
        if 'mean_reversion' in enabled_strategies:
            mr_config = strategies_config.get('mean_reversion', {})
            strategy = MeanReversionStrategy(mr_config)
            self.portfolio_manager.add_strategy(strategy, allocation=0.25)
            logger.info("Mean reversion strategy added")

        # ML hybrid strategy
        if 'ml_hybrid' in enabled_strategies:
            ml_config = strategies_config.get('ml_hybrid', {})
            strategy = MLHybridStrategy(ml_config)
            self.portfolio_manager.add_strategy(strategy, allocation=0.30)
            logger.info("ML hybrid strategy added")

        # Options strategies
        if 'options_income' in enabled_strategies:
            options_config = strategies_config.get('options_income', {})
            if 'covered_call' in options_config.get('strategies', []):
                strategy = CoveredCallStrategy(options_config)
                self.portfolio_manager.add_strategy(strategy, allocation=0.10)
                logger.info("Covered call strategy added")

            if 'cash_secured_put' in options_config.get('strategies', []):
                strategy = CashSecuredPutStrategy(options_config)
                self.portfolio_manager.add_strategy(strategy, allocation=0.10)
                logger.info("Cash secured put strategy added")

    def fetch_data(
        self,
        symbol: str = "QQQ",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch and prepare market data.

        Args:
            symbol: Trading symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Prepared market data with features
        """
        logger.info(f"Fetching data for {symbol}")

        # Use config defaults if not specified
        if start_date is None:
            start_date = self.config.data_config.get('start_date', '2019-01-01')
        if end_date is None:
            end_date = self.config.data_config.get('end_date')

        # Fetch data
        data = self.data_manager.get_market_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=self.config.data_config.get('interval', '1d')
        )

        logger.info(f"Fetched {len(data)} rows of data for {symbol}")

        return data

    def backtest(
        self,
        symbol: str = "QQQ",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        save_results: bool = True
    ) -> Dict[str, BacktestResult]:
        """
        Run backtest for all strategies.

        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            save_results: Save results to disk

        Returns:
            Dictionary of backtest results for each strategy
        """
        logger.info(f"Starting backtest for {symbol}")

        # Fetch data
        data = self.fetch_data(symbol, start_date, end_date)

        # Analyze market if Claude agent available
        if self.market_analyzer:
            try:
                market_analysis = self.market_analyzer.analyze_market_regime(data, symbol)
                logger.info(f"Market regime: {market_analysis.get('regime', 'Unknown')}")
            except Exception as e:
                logger.warning(f"Market analysis failed: {e}")

        # Run backtest for each strategy
        results = {}

        for strategy_name, strategy in self.portfolio_manager.strategies.items():
            logger.info(f"Backtesting {strategy_name}...")

            try:
                # Train ML models if applicable
                if hasattr(strategy, 'train'):
                    train_size = int(len(data) * 0.8)
                    train_data = data.iloc[:train_size]
                    strategy.train(train_data)

                # Run backtest
                result = self.backtest_engine.run(
                    strategy=strategy,
                    data=data,
                    risk_params=self.config.risk
                )

                results[strategy_name] = result

                logger.info(f"{strategy_name} - Return: {result.total_return:.2%}, "
                          f"Sharpe: {result.sharpe_ratio:.2f}, "
                          f"Trades: {result.num_trades}")

                # Get Claude evaluation if available
                if self.strategy_optimizer:
                    try:
                        evaluation = self.strategy_optimizer.evaluate_backtest_results(
                            strategy_name=strategy_name,
                            results_summary={
                                'total_return': result.total_return,
                                'sharpe_ratio': result.sharpe_ratio,
                                'max_drawdown': result.max_drawdown,
                                'win_rate': result.win_rate,
                                'num_trades': result.num_trades
                            },
                            equity_curve_stats={
                                'mean_return': result.equity_curve.pct_change().mean(),
                                'volatility': result.equity_curve.pct_change().std()
                            }
                        )
                        logger.info(f"Claude evaluation for {strategy_name}:\n{evaluation[:200]}...")
                    except Exception as e:
                        logger.warning(f"Strategy evaluation failed: {e}")

            except Exception as e:
                logger.error(f"Error backtesting {strategy_name}: {e}")

        # Save results if requested
        if save_results:
            self._save_backtest_results(results, symbol)

        return results

    def _save_backtest_results(
        self,
        results: Dict[str, BacktestResult],
        symbol: str
    ):
        """Save backtest results to disk."""
        results_dir = Path("results") / symbol
        results_dir.mkdir(parents=True, exist_ok=True)

        for strategy_name, result in results.items():
            # Save performance report
            analyzer = PerformanceAnalyzer(result)

            # Generate HTML report
            report_path = results_dir / f"{strategy_name}_report.html"
            analyzer.generate_report(output_path=str(report_path))

            # Save equity curve plot
            plot_path = results_dir / f"{strategy_name}_equity.html"
            analyzer.plot_equity_curve(save_path=str(plot_path))

            # Save comprehensive dashboard
            dashboard_path = results_dir / f"{strategy_name}_dashboard.html"
            analyzer.plot_comprehensive_dashboard(save_path=str(dashboard_path))

            logger.info(f"Saved results for {strategy_name} to {results_dir}")

    def optimize_strategies(self, market_data: pd.DataFrame):
        """
        Optimize strategy parameters using Claude.

        Args:
            market_data: Current market data
        """
        if not self.strategy_optimizer:
            logger.warning("Strategy optimizer not available")
            return

        logger.info("Optimizing strategy parameters")

        # Get market conditions
        market_conditions = {}
        if self.market_analyzer:
            try:
                analysis = self.market_analyzer.analyze_market_regime(market_data)
                market_conditions = {
                    'regime': analysis.get('regime', 'Unknown'),
                    'trend': analysis.get('trend', 'Neutral'),
                    'volatility': analysis.get('volatility', 'Medium')
                }
            except Exception as e:
                logger.warning(f"Market analysis failed: {e}")

        # Optimize each strategy
        for strategy_name, strategy in self.portfolio_manager.strategies.items():
            try:
                # Get current parameters
                current_params = strategy.config

                # Get recent performance (mock for now)
                performance_metrics = {
                    'sharpe_ratio': 1.0,
                    'win_rate': 0.55,
                    'max_drawdown': -0.10,
                    'total_return': 0.15
                }

                # Get optimization suggestions
                optimization = self.strategy_optimizer.optimize_parameters(
                    strategy_name=strategy_name,
                    current_params=current_params,
                    performance_metrics=performance_metrics,
                    market_conditions=market_conditions
                )

                logger.info(f"Optimization for {strategy_name}:")
                logger.info(f"Reasoning: {optimization.get('reasoning', 'N/A')[:200]}...")

            except Exception as e:
                logger.error(f"Error optimizing {strategy_name}: {e}")

    def generate_market_report(
        self,
        symbol: str = "QQQ",
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive market analysis report.

        Args:
            symbol: Trading symbol
            output_path: Path to save report

        Returns:
            Report text
        """
        logger.info(f"Generating market report for {symbol}")

        # Fetch recent data
        data = self.fetch_data(symbol)

        if not self.market_analyzer:
            logger.warning("Market analyzer not available")
            return "Market analysis not available - Claude API key missing"

        try:
            # Get market analysis
            regime_analysis = self.market_analyzer.analyze_market_regime(data, symbol)

            # Get market commentary
            commentary = self.market_analyzer.generate_market_commentary(data, symbol)

            # Get opportunities
            strategy_names = list(self.portfolio_manager.strategies.keys())
            opportunities = self.market_analyzer.identify_opportunities(
                data, strategy_names, symbol
            )

            # Compile report
            report = f"""
# Market Analysis Report for {symbol}
Generated: {pd.Timestamp.now()}

## Market Regime Analysis
{regime_analysis.get('analysis_text', 'N/A')}

## Market Commentary
{commentary}

## Trading Opportunities
{opportunities.get('analysis', 'N/A')}

## Current Market Data
- Price: ${data.iloc[-1]['close']:.2f}
- RSI: {data.iloc[-1].get('rsi', 0):.2f}
- Volatility: {data.iloc[-1].get('volatility_20', 0) * 100:.2f}%
- Regime: {regime_analysis.get('regime', 'Unknown')}
- Trend: {regime_analysis.get('trend', 'Neutral')}
"""

            if output_path:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    f.write(report)
                logger.info(f"Report saved to {output_path}")

            return report

        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return f"Error generating report: {e}"

    def display_results(self, results: Dict[str, BacktestResult]):
        """
        Display backtest results summary.

        Args:
            results: Dictionary of backtest results
        """
        print("\n" + "="*80)
        print("BACKTEST RESULTS SUMMARY")
        print("="*80)

        for strategy_name, result in results.items():
            print(f"\n{strategy_name}:")
            print(f"  Period: {result.start_date.date()} to {result.end_date.date()}")
            print(f"  Total Return: {result.total_return:.2%}")
            print(f"  Annualized Return: {result.annualized_return:.2%}")
            print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
            print(f"  Sortino Ratio: {result.sortino_ratio:.2f}")
            print(f"  Max Drawdown: {result.max_drawdown:.2%}")
            print(f"  Win Rate: {result.win_rate:.2%}")
            print(f"  Profit Factor: {result.profit_factor:.2f}")
            print(f"  Number of Trades: {result.num_trades}")
            print(f"  Final Capital: ${result.final_capital:,.2f}")

        print("\n" + "="*80)
