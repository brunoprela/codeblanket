export const performanceMetricsForTrading = {
  title: 'Performance Metrics for Trading',
  slug: 'performance-metrics-for-trading',
  description:
    'Master calculating and interpreting trading performance metrics including returns, risk measures, and risk-adjusted performance indicators',
  content: `
# Performance Metrics for Trading

## Introduction

Performance metrics are how you measure trading success. Understanding them deeply is critical for evaluating strategies, comparing approaches, and making informed decisions.

**What you'll learn:**
- Return metrics (CAGR, total return, log returns)
- Risk metrics (volatility, max drawdown, VaR, CVaR)
- Risk-adjusted returns (Sharpe, Sortino, Calmar, Information Ratio)
- Trading-specific metrics (win rate, profit factor, expectancy)
- Python implementation of all metrics

## Return Metrics

\`\`\`python
import pandas as pd
import numpy as np
from typing import Union

class ReturnMetrics:
    """Calculate various return metrics"""
    
    @staticmethod
    def total_return(returns: pd.Series) -> float:
        """Total return over period"""
        return (1 + returns).prod() - 1
    
    @staticmethod
    def cagr(returns: pd.Series, periods_per_year: int = 252) -> float:
        """Compound Annual Growth Rate"""
        total = (1 + returns).prod()
        n_periods = len(returns)
        years = n_periods / periods_per_year
        return (total ** (1/years)) - 1 if years > 0 else 0
    
    @staticmethod
    def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
        """Annualized return"""
        return returns.mean() * periods_per_year

## Risk Metrics

\`\`\`python
class RiskMetrics:
    """Calculate risk metrics"""
    
    @staticmethod
    def volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
        """Annualized volatility"""
        return returns.std() * np.sqrt(periods_per_year)
    
    @staticmethod
    def max_drawdown(returns: pd.Series) -> float:
        """Maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    @staticmethod
    def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
        """Value at Risk"""
        return returns.quantile(1 - confidence)

## Risk-Adjusted Returns

\`\`\`python
class RiskAdjustedMetrics:
    """Risk-adjusted performance metrics"""
    
    @staticmethod
    def sharpe_ratio(returns: pd.Series, rf: float = 0.02, 
                     periods_per_year: int = 252) -> float:
        """Sharpe Ratio"""
        excess = returns - rf / periods_per_year
        return np.sqrt(periods_per_year) * (excess.mean() / returns.std()) if returns.std() > 0 else 0
    
    @staticmethod
    def sortino_ratio(returns: pd.Series, rf: float = 0.02,
                      periods_per_year: int = 252) -> float:
        """Sortino Ratio - uses downside deviation"""
        excess = returns - rf / periods_per_year
        downside = returns[returns < 0].std()
        return np.sqrt(periods_per_year) * (excess.mean() / downside) if downside > 0 else 0
    
    @staticmethod
    def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
        """Calmar Ratio = Annual Return / Max Drawdown"""
        annual_return = returns.mean() * periods_per_year
        max_dd = RiskMetrics.max_drawdown(returns)
        return abs(annual_return / max_dd) if max_dd != 0 else 0
\`\`\`

## Trading-Specific Metrics

\`\`\`python
class TradingMetrics:
    """Metrics specific to trading"""
    
    @staticmethod
    def win_rate(trades: pd.DataFrame) -> float:
        """Percentage of winning trades"""
        if len(trades) == 0:
            return 0
        wins = len(trades[trades['pnl'] > 0])
        return wins / len(trades)
    
    @staticmethod
    def profit_factor(trades: pd.DataFrame) -> float:
        """Gross profits / Gross losses"""
        profits = trades[trades['pnl'] > 0]['pnl'].sum()
        losses = abs(trades[trades['pnl'] < 0]['pnl'].sum())
        return profits / losses if losses != 0 else np.inf
    
    @staticmethod
    def expectancy(trades: pd.DataFrame) -> float:
        """Average $ per trade"""
        return trades['pnl'].mean() if len(trades) > 0 else 0
\`\`\`

## Complete Performance Calculator

\`\`\`python
class PerformanceAnalyzer:
    """Comprehensive performance analysis"""
    
    def __init__(self, returns: pd.Series, trades: pd.DataFrame = None):
        self.returns = returns
        self.trades = trades
    
    def calculate_all_metrics(self) -> dict:
        """Calculate all performance metrics"""
        return {
            # Returns
            'total_return': ReturnMetrics.total_return(self.returns),
            'cagr': ReturnMetrics.cagr(self.returns),
            
            # Risk
            'volatility': RiskMetrics.volatility(self.returns),
            'max_drawdown': RiskMetrics.max_drawdown(self.returns),
            'var_95': RiskMetrics.value_at_risk(self.returns),
            
            # Risk-adjusted
            'sharpe_ratio': RiskAdjustedMetrics.sharpe_ratio(self.returns),
            'sortino_ratio': RiskAdjustedMetrics.sortino_ratio(self.returns),
            'calmar_ratio': RiskAdjustedMetrics.calmar_ratio(self.returns),
            
            # Trading metrics
            'win_rate': TradingMetrics.win_rate(self.trades) if self.trades is not None else None,
            'profit_factor': TradingMetrics.profit_factor(self.trades) if self.trades is not None else None,
            'expectancy': TradingMetrics.expectancy(self.trades) if self.trades is not None else None
        }
\`\`\`

## Summary

Key metrics for evaluating trading strategies:
1. **Returns**: CAGR, total return
2. **Risk**: Volatility, max drawdown, VaR
3. **Risk-adjusted**: Sharpe, Sortino, Calmar
4. **Trading**: Win rate, profit factor, expectancy

Always evaluate strategies on multiple metrics, not just returns.
`,
  exercises: [
    {
      prompt:
        'Calculate Sharpe ratio for a strategy with varying risk-free rates and compare results.',
      solution:
        '// Implement Sharpe calculation with different rf rates, show impact on ratio',
    },
    {
      prompt:
        'Build a performance comparison tool that ranks multiple strategies across all metrics.',
      solution:
        '// Create ranking system, normalize metrics, visualize results',
    },
  ],
};
