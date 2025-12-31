"""Performance analysis and visualization."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Optional
from pathlib import Path

from .backtest_engine import BacktestResult
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class PerformanceAnalyzer:
    """Analyze and visualize backtest performance."""

    def __init__(self, results: BacktestResult):
        """
        Initialize performance analyzer.

        Args:
            results: Backtest results to analyze
        """
        self.results = results

    def generate_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.

        Args:
            output_path: Path to save HTML report

        Returns:
            Dictionary with report data
        """
        logger.info("Generating performance report")

        report = {
            'summary': self._get_summary(),
            'returns_analysis': self._analyze_returns(),
            'risk_metrics': self._calculate_risk_metrics(),
            'trade_analysis': self._analyze_trades(),
        }

        if output_path:
            self._save_html_report(report, output_path)

        return report

    def _get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            'Strategy': self.results.strategy_name,
            'Period': f"{self.results.start_date.date()} to {self.results.end_date.date()}",
            'Initial Capital': f"${self.results.initial_capital:,.2f}",
            'Final Capital': f"${self.results.final_capital:,.2f}",
            'Total Return': f"{self.results.total_return:.2%}",
            'Annualized Return': f"{self.results.annualized_return:.2%}",
            'Sharpe Ratio': f"{self.results.sharpe_ratio:.2f}",
            'Sortino Ratio': f"{self.results.sortino_ratio:.2f}",
            'Max Drawdown': f"{self.results.max_drawdown:.2%}",
            'Calmar Ratio': f"{self.results.calmar_ratio:.2f}",
            'Win Rate': f"{self.results.win_rate:.2%}",
            'Profit Factor': f"{self.results.profit_factor:.2f}",
            'Total Trades': self.results.num_trades
        }

    def _analyze_returns(self) -> Dict[str, Any]:
        """Analyze returns distribution."""
        returns = self.results.equity_curve.pct_change().dropna()

        return {
            'mean_daily_return': returns.mean(),
            'std_daily_return': returns.std(),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'best_day': returns.max(),
            'worst_day': returns.min(),
            'positive_days': (returns > 0).sum() / len(returns),
            'negative_days': (returns < 0).sum() / len(returns)
        }

    def _calculate_risk_metrics(self) -> Dict[str, Any]:
        """Calculate additional risk metrics."""
        equity = self.results.equity_curve
        returns = equity.pct_change().dropna()

        # Value at Risk (VaR)
        var_95 = returns.quantile(0.05)
        var_99 = returns.quantile(0.01)

        # Conditional VaR (CVaR/Expected Shortfall)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()

        # Ulcer Index (measure of downside volatility)
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        ulcer_index = np.sqrt((drawdown ** 2).mean())

        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'ulcer_index': ulcer_index
        }

    def _analyze_trades(self) -> Dict[str, Any]:
        """Analyze individual trades."""
        if not self.results.trades:
            return {}

        trades = self.results.trades
        trade_returns = [t.return_pct for t in trades if t.exit_date is not None]

        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]

        return {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'avg_win': np.mean([t.pnl for t in winning_trades]) if winning_trades else 0,
            'avg_loss': np.mean([t.pnl for t in losing_trades]) if losing_trades else 0,
            'largest_win': max([t.pnl for t in winning_trades]) if winning_trades else 0,
            'largest_loss': min([t.pnl for t in losing_trades]) if losing_trades else 0,
            'avg_trade_duration': self._calculate_avg_duration(trades),
            'win_loss_ratio': len(winning_trades) / len(losing_trades) if losing_trades else 0
        }

    def _calculate_avg_duration(self, trades: List) -> float:
        """Calculate average trade duration in days."""
        durations = []
        for trade in trades:
            if trade.exit_date:
                duration = (trade.exit_date - trade.entry_date).days
                durations.append(duration)
        return np.mean(durations) if durations else 0

    def plot_equity_curve(self, save_path: Optional[str] = None):
        """Plot equity curve."""
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=self.results.equity_curve.index,
            y=self.results.equity_curve.values,
            mode='lines',
            name='Equity',
            line=dict(color='blue', width=2)
        ))

        fig.update_layout(
            title=f'{self.results.strategy_name} - Equity Curve',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            hovermode='x unified',
            template='plotly_white'
        )

        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()

        return fig

    def plot_drawdown(self, save_path: Optional[str] = None):
        """Plot drawdown curve."""
        equity = self.results.equity_curve
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values * 100,
            fill='tozeroy',
            name='Drawdown',
            line=dict(color='red', width=1)
        ))

        fig.update_layout(
            title=f'{self.results.strategy_name} - Drawdown',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            hovermode='x unified',
            template='plotly_white'
        )

        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()

        return fig

    def plot_returns_distribution(self, save_path: Optional[str] = None):
        """Plot returns distribution."""
        returns = self.results.equity_curve.pct_change().dropna() * 100

        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=returns.values,
            nbinsx=50,
            name='Returns',
            marker=dict(color='blue', line=dict(color='black', width=1))
        ))

        fig.update_layout(
            title=f'{self.results.strategy_name} - Returns Distribution',
            xaxis_title='Daily Return (%)',
            yaxis_title='Frequency',
            template='plotly_white',
            showlegend=False
        )

        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()

        return fig

    def plot_monthly_returns(self, save_path: Optional[str] = None):
        """Plot monthly returns heatmap."""
        returns = self.results.equity_curve.pct_change()
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)

        # Create pivot table for heatmap
        monthly_returns_df = pd.DataFrame({
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month,
            'return': monthly_returns.values * 100
        })

        pivot = monthly_returns_df.pivot(index='month', columns='year', values='return')

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            colorscale='RdYlGn',
            zmid=0,
            text=np.round(pivot.values, 2),
            texttemplate='%{text}%',
            textfont={"size": 10}
        ))

        fig.update_layout(
            title=f'{self.results.strategy_name} - Monthly Returns (%)',
            xaxis_title='Year',
            yaxis_title='Month',
            template='plotly_white'
        )

        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()

        return fig

    def plot_comprehensive_dashboard(self, save_path: Optional[str] = None):
        """Create comprehensive dashboard with multiple plots."""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Equity Curve',
                'Drawdown',
                'Returns Distribution',
                'Monthly Returns',
                'Trade Analysis',
                'Rolling Sharpe'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "histogram"}, {"type": "heatmap"}],
                [{"type": "bar"}, {"type": "scatter"}]
            ]
        )

        # Equity curve
        fig.add_trace(
            go.Scatter(x=self.results.equity_curve.index,
                      y=self.results.equity_curve.values,
                      name='Equity',
                      line=dict(color='blue')),
            row=1, col=1
        )

        # Drawdown
        equity = self.results.equity_curve
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        fig.add_trace(
            go.Scatter(x=drawdown.index,
                      y=drawdown.values * 100,
                      fill='tozeroy',
                      name='Drawdown',
                      line=dict(color='red')),
            row=1, col=2
        )

        # Returns distribution
        returns = equity.pct_change().dropna() * 100
        fig.add_trace(
            go.Histogram(x=returns.values, nbinsx=50, name='Returns'),
            row=2, col=1
        )

        # Rolling Sharpe
        rolling_sharpe = returns.rolling(60).mean() / returns.rolling(60).std() * np.sqrt(252)
        fig.add_trace(
            go.Scatter(x=rolling_sharpe.index,
                      y=rolling_sharpe.values,
                      name='Rolling Sharpe',
                      line=dict(color='green')),
            row=3, col=2
        )

        fig.update_layout(
            height=1200,
            title_text=f"{self.results.strategy_name} - Performance Dashboard",
            showlegend=False,
            template='plotly_white'
        )

        if save_path:
            fig.write_html(save_path)
            logger.info(f"Dashboard saved to {save_path}")
        else:
            fig.show()

        return fig

    def _save_html_report(self, report: Dict[str, Any], output_path: str):
        """Save HTML report."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        html_content = f"""
        <html>
        <head>
            <title>{self.results.strategy_name} Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                .metric {{ font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>{self.results.strategy_name} - Performance Report</h1>

            <h2>Summary</h2>
            <table>
                {self._dict_to_table(report['summary'])}
            </table>

            <h2>Returns Analysis</h2>
            <table>
                {self._dict_to_table(report['returns_analysis'])}
            </table>

            <h2>Risk Metrics</h2>
            <table>
                {self._dict_to_table(report['risk_metrics'])}
            </table>

            <h2>Trade Analysis</h2>
            <table>
                {self._dict_to_table(report['trade_analysis'])}
            </table>
        </body>
        </html>
        """

        with open(output_path, 'w') as f:
            f.write(html_content)

        logger.info(f"Report saved to {output_path}")

    def _dict_to_table(self, data: Dict[str, Any]) -> str:
        """Convert dictionary to HTML table rows."""
        rows = ""
        for key, value in data.items():
            if isinstance(value, float):
                value = f"{value:.4f}"
            rows += f"<tr><td class='metric'>{key}</td><td>{value}</td></tr>"
        return rows
