import { Content } from '@/lib/types';

const benchmarkComparison: Content = {
  title: 'Benchmark Comparison & Attribution Analysis',
  description:
    'Master benchmark selection, risk-adjusted performance comparison, attribution analysis, and information ratio calculation for professional strategy evaluation',
  sections: [
    {
      title: 'Choosing and Comparing to Benchmarks',
      content: `
# Benchmark Comparison & Attribution Analysis

A strategy's performance is meaningless without proper benchmark comparison. Is a 15% annual return good? It depends—if the S&P 500 returned 25%, you underperformed.

## Why Benchmarks Matter

**Real Example - Long Short Equity Fund**: A hedge fund reported 18% annual returns with 12% volatility (Sharpe ~1.5) and marketed themselves as "exceptional performers." 

**Reality Check**: During the same period:
- S&P 500: 22% return, 15% volatility (Sharpe 1.47)
- Market-neutral benchmark: 8% return, 5% volatility (Sharpe 1.6)

The fund simply held levered long equities and charged 2/20 fees for beta exposure. They added no alpha.

## Benchmark Selection Principles

### 1. **Match Strategy Characteristics**

\`\`\`python
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum

class StrategyType(Enum):
    """Strategy types"""
    LONG_ONLY_EQUITY = "long_only_equity"
    MARKET_NEUTRAL = "market_neutral"
    LONG_SHORT = "long_short"
    SECTOR_SPECIFIC = "sector_specific"
    MULTI_ASSET = "multi_asset"
    ABSOLUTE_RETURN = "absolute_return"

@dataclass
class BenchmarkSpec:
    """Benchmark specification"""
    name: str
    description: str
    ticker: str
    suitable_for: List[StrategyType]
    
# Common benchmarks
BENCHMARKS = {
    'sp500': BenchmarkSpec(
        name="S&P 500",
        description="Large-cap US equities",
        ticker="SPY",
        suitable_for=[StrategyType.LONG_ONLY_EQUITY]
    ),
    'spy_market_neutral': BenchmarkSpec(
        name="Market Neutral Index",
        description="Zero beta, absolute return",
        ticker="HFRXEMN",  # HFRI Equity Market Neutral Index
        suitable_for=[StrategyType.MARKET_NEUTRAL]
    ),
    'mixed_60_40': BenchmarkSpec(
        name="60/40 Portfolio",
        description="60% equities, 40% bonds",
        ticker="AOR",
        suitable_for=[StrategyType.MULTI_ASSET]
    )
}

def select_benchmark(
    strategy_type: StrategyType,
    strategy_returns: pd.Series
) -> str:
    """
    Select appropriate benchmark for strategy
    
    Args:
        strategy_type: Type of strategy
        strategy_returns: Strategy returns
        
    Returns:
        Benchmark identifier
    """
    # Find suitable benchmarks
    suitable = [
        key for key, spec in BENCHMARKS.items()
        if strategy_type in spec.suitable_for
    ]
    
    if not suitable:
        raise ValueError(f"No benchmark found for {strategy_type}")
    
    # For multiple options, select based on correlation
    if len(suitable) == 1:
        return suitable[0]
    
    # Return first suitable benchmark
    # In production: Load benchmark data and select highest correlation
    return suitable[0]


class BenchmarkComparator:
    """
    Compare strategy performance to benchmarks
    """
    
    def __init__(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float = 0.02
    ):
        """
        Initialize comparator
        
        Args:
            strategy_returns: Strategy returns
            benchmark_returns: Benchmark returns
            risk_free_rate: Risk-free rate (annual)
        """
        # Align returns
        common_idx = strategy_returns.index.intersection(benchmark_returns.index)
        self.strategy_returns = strategy_returns.loc[common_idx]
        self.benchmark_returns = benchmark_returns.loc[common_idx]
        self.risk_free_rate = risk_free_rate
        self.rf_daily = risk_free_rate / 252
    
    def compare_performance(self) -> Dict:
        """
        Comprehensive performance comparison
        
        Returns:
            Dictionary of comparison metrics
        """
        strategy_metrics = self._calculate_metrics(self.strategy_returns)
        benchmark_metrics = self._calculate_metrics(self.benchmark_returns)
        
        # Calculate relative metrics
        alpha, beta = self._calculate_alpha_beta()
        information_ratio = self._calculate_information_ratio()
        tracking_error = self._calculate_tracking_error()
        
        comparison = {
            'strategy': strategy_metrics,
            'benchmark': benchmark_metrics,
            'relative': {
                'alpha_annual': alpha,
                'beta': beta,
                'information_ratio': information_ratio,
                'tracking_error_annual': tracking_error,
                'excess_return_annual': (
                    strategy_metrics['total_return_annual'] -
                    benchmark_metrics['total_return_annual']
                ),
                'sharpe_difference': (
                    strategy_metrics['sharpe'] -
                    benchmark_metrics['sharpe']
                )
            }
        }
        
        return comparison
    
    def _calculate_metrics(self, returns: pd.Series) -> Dict:
        """Calculate performance metrics"""
        total_return = (1 + returns).prod() - 1
        n_years = len(returns) / 252
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        volatility = returns.std() * np.sqrt(252)
        sharpe = (annual_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        win_rate = (returns > 0).mean()
        
        return {
            'total_return': total_return,
            'total_return_annual': annual_return,
            'volatility_annual': volatility,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate
        }
    
    def _calculate_alpha_beta(self) -> tuple:
        """
        Calculate Jensen's Alpha and Beta
        
        Returns:
            (alpha, beta) tuple
        """
        # Run regression: R_strategy - Rf = alpha + beta * (R_benchmark - Rf)
        strategy_excess = self.strategy_returns - self.rf_daily
        benchmark_excess = self.benchmark_returns - self.rf_daily
        
        # Calculate beta
        covariance = np.cov(strategy_excess, benchmark_excess)[0, 1]
        benchmark_variance = np.var(benchmark_excess)
        
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        # Calculate alpha
        alpha_daily = strategy_excess.mean() - beta * benchmark_excess.mean()
        alpha_annual = alpha_daily * 252
        
        return alpha_annual, beta
    
    def _calculate_information_ratio(self) -> float:
        """
        Calculate Information Ratio
        
        IR = (Strategy Return - Benchmark Return) / Tracking Error
        """
        excess_returns = self.strategy_returns - self.benchmark_returns
        
        excess_return_annual = excess_returns.mean() * 252
        tracking_error = excess_returns.std() * np.sqrt(252)
        
        ir = excess_return_annual / tracking_error if tracking_error > 0 else 0
        
        return ir
    
    def _calculate_tracking_error(self) -> float:
        """Calculate tracking error (volatility of excess returns)"""
        excess_returns = self.strategy_returns - self.benchmark_returns
        tracking_error = excess_returns.std() * np.sqrt(252)
        return tracking_error
    
    def plot_comparison(self):
        """Plot strategy vs benchmark performance"""
        import matplotlib.pyplot as plt
        
        strategy_cumulative = (1 + self.strategy_returns).cumprod()
        benchmark_cumulative = (1 + self.benchmark_returns).cumprod()
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Cumulative returns
        axes[0].plot(
            strategy_cumulative.index,
            strategy_cumulative.values,
            label='Strategy',
            linewidth=2
        )
        axes[0].plot(
            benchmark_cumulative.index,
            benchmark_cumulative.values,
            label='Benchmark',
            linewidth=2,
            alpha=0.7
        )
        axes[0].set_title('Cumulative Returns')
        axes[0].set_ylabel('Cumulative Return')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Excess returns
        excess_returns = self.strategy_returns - self.benchmark_returns
        excess_cumulative = (1 + excess_returns).cumprod()
        
        axes[1].plot(
            excess_cumulative.index,
            excess_cumulative.values,
            label='Excess Return',
            color='green',
            linewidth=2
        )
        axes[1].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        axes[1].fill_between(
            excess_cumulative.index,
            1.0,
            excess_cumulative.values,
            where=(excess_cumulative.values >= 1.0),
            color='green',
            alpha=0.3
        )
        axes[1].fill_between(
            excess_cumulative.index,
            1.0,
            excess_cumulative.values,
            where=(excess_cumulative.values < 1.0),
            color='red',
            alpha=0.3
        )
        axes[1].set_title('Excess Returns (Strategy - Benchmark)')
        axes[1].set_ylabel('Cumulative Excess Return')
        axes[1].set_xlabel('Date')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('benchmark_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self) -> str:
        """Generate text report"""
        comparison = self.compare_performance()
        
        report = []
        report.append("="*80)
        report.append("BENCHMARK COMPARISON REPORT")
        report.append("="*80)
        
        report.append("\\nSTRATEGY PERFORMANCE:")
        for key, value in comparison['strategy'].items():
            report.append(f"  {key}: {value:.4f}")
        
        report.append("\\nBENCHMARK PERFORMANCE:")
        for key, value in comparison['benchmark'].items():
            report.append(f"  {key}: {value:.4f}")
        
        report.append("\\nRELATIVE PERFORMANCE:")
        for key, value in comparison['relative'].items():
            report.append(f"  {key}: {value:.4f}")
        
        # Interpretation
        report.append("\\nINTERPRETATION:")
        
        alpha = comparison['relative']['alpha_annual']
        beta = comparison['relative']['beta']
        ir = comparison['relative']['information_ratio']
        
        if alpha > 0:
            report.append(f"  ✓ Positive alpha ({alpha:.2%}): Strategy adds value above benchmark")
        else:
            report.append(f"  ✗ Negative alpha ({alpha:.2%}): Strategy underperforms benchmark")
        
        if 0.8 <= beta <= 1.2:
            report.append(f"  • Beta ({beta:.2f}): Similar market exposure to benchmark")
        elif beta < 0.8:
            report.append(f"  • Beta ({beta:.2f}): Lower market exposure (defensive)")
        else:
            report.append(f"  • Beta ({beta:.2f}): Higher market exposure (aggressive)")
        
        if ir > 0.5:
            report.append(f"  ✓ Information Ratio ({ir:.2f}): Excellent risk-adjusted outperformance")
        elif ir > 0:
            report.append(f"  • Information Ratio ({ir:.2f}): Modest outperformance")
        else:
            report.append(f"  ✗ Information Ratio ({ir:.2f}): Underperformance")
        
        report.append("="*80)
        
        return "\\n".join(report)


# Example usage
if __name__ == "__main__":
    # Generate sample returns
    np.random.seed(42)
    dates = pd.date_range('2018-01-01', '2023-12-31', freq='D')
    
    # Strategy: 12% annual return, 18% vol
    strategy_returns = pd.Series(
        np.random.randn(len(dates)) * 0.18 / np.sqrt(252) + 0.12 / 252,
        index=dates
    )
    
    # Benchmark (SPY): 10% annual return, 16% vol
    benchmark_returns = pd.Series(
        np.random.randn(len(dates)) * 0.16 / np.sqrt(252) + 0.10 / 252,
        index=dates
    )
    
    # Compare
    comparator = BenchmarkComparator(
        strategy_returns=strategy_returns,
        benchmark_returns=benchmark_returns,
        risk_free_rate=0.02
    )
    
    # Generate report
    report = comparator.generate_report()
    print(report)
    
    # Plot comparison
    comparator.plot_comparison()
    print("\\n✓ Comparison chart saved to benchmark_comparison.png")
\`\`\`

## Attribution Analysis

Attribution analysis decomposes returns into their sources.

\`\`\`python
class AttributionAnalyzer:
    """
    Performance attribution analysis
    """
    
    def __init__(
        self,
        portfolio_returns: pd.Series,
        holdings: pd.DataFrame,  # Date x Asset positions
        asset_returns: pd.DataFrame  # Date x Asset returns
    ):
        self.portfolio_returns = portfolio_returns
        self.holdings = holdings
        self.asset_returns = asset_returns
    
    def brinson_attribution(
        self,
        benchmark_weights: pd.DataFrame
    ) -> Dict:
        """
        Brinson attribution model
        
        Decomposes returns into:
        - Allocation effect
        - Selection effect
        - Interaction effect
        """
        # Align data
        common_dates = (
            self.holdings.index
            .intersection(self.asset_returns.index)
            .intersection(benchmark_weights.index)
        )
        
        holdings = self.holdings.loc[common_dates]
        returns = self.asset_returns.loc[common_dates]
        bm_weights = benchmark_weights.loc[common_dates]
        
        # Calculate effects
        allocation_effect = (
            (holdings - bm_weights) * returns.mean(axis=0)
        ).sum(axis=1)
        
        selection_effect = (
            bm_weights * (returns - returns.mean(axis=0))
        ).sum(axis=1)
        
        interaction_effect = (
            (holdings - bm_weights) * (returns - returns.mean(axis=0))
        ).sum(axis=1)
        
        return {
            'allocation': allocation_effect.sum(),
            'selection': selection_effect.sum(),
            'interaction': interaction_effect.sum(),
            'total': allocation_effect.sum() + selection_effect.sum() + interaction_effect.sum()
        }


# Example
def example_attribution():
    """Example attribution analysis"""
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    
    # Portfolio holdings (3 assets)
    holdings = pd.DataFrame({
        'AAPL': 0.4,
        'GOOGL': 0.3,
        'MSFT': 0.3
    }, index=dates)
    
    # Asset returns
    asset_returns = pd.DataFrame({
        'AAPL': np.random.randn(len(dates)) * 0.02 + 0.001,
        'GOOGL': np.random.randn(len(dates)) * 0.025 + 0.0008,
        'MSFT': np.random.randn(len(dates)) * 0.018 + 0.0012
    }, index=dates)
    
    # Benchmark weights (equal weight)
    benchmark_weights = pd.DataFrame({
        'AAPL': 0.33,
        'GOOGL': 0.33,
        'MSFT': 0.34
    }, index=dates)
    
    # Portfolio returns
    portfolio_returns = (holdings * asset_returns).sum(axis=1)
    
    # Attribution
    analyzer = AttributionAnalyzer(
        portfolio_returns=portfolio_returns,
        holdings=holdings,
        asset_returns=asset_returns
    )
    
    attribution = analyzer.brinson_attribution(benchmark_weights)
    
    print("\\nPERFORMANCE ATTRIBUTION")
    print("="*50)
    for key, value in attribution.items():
        print(f"{key.capitalize()}: {value:.4f}")

if __name__ == "__main__":
    example_attribution()
\`\`\`

## Key Metrics

### Information Ratio
- Measures risk-adjusted excess return over benchmark
- IR = (Strategy Return - Benchmark Return) / Tracking Error
- IR > 0.5: Good, IR > 1.0: Excellent

### Tracking Error
- Volatility of excess returns
- Low TE: Close to benchmark
- High TE: Differentiated strategy

### Alpha/Beta
- Alpha: Return not explained by benchmark exposure
- Beta: Sensitivity to benchmark movements

## Production Checklist

- [ ] Appropriate benchmark selected
- [ ] Alpha and beta calculated
- [ ] Information ratio computed
- [ ] Tracking error analyzed
- [ ] Attribution analysis performed
- [ ] Results interpreted and documented
`,
    },
  ],
};

export default benchmarkComparison;
