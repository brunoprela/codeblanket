import { Content } from '@/lib/types';

const monteCarloSimulation: Content = {
  title: 'Monte Carlo Simulation for Trading',
  description:
    'Master Monte Carlo techniques for strategy validation, randomizing trade sequences, bootstrap resampling, confidence intervals, and stress testing for robust trading systems',
  sections: [
    {
      title: 'Introduction to Monte Carlo Methods in Trading',
      content: `
# Introduction to Monte Carlo Methods in Trading

Monte Carlo simulation is a powerful technique for assessing the robustness and reliability of trading strategies by introducing randomness into the backtesting process.

## The Problem with Single-Path Backtesting

**Case Study**: A prop trading firm developed a momentum strategy that showed a Sharpe ratio of 2.3 over 5 years of backtesting. However, when they ran the same trades in a different order (same trades, just reshuffled), the Sharpe ratio dropped to 0.9. This massive variance indicated the strategy's success was highly dependent on the specific sequence of trades—a warning sign of fragility.

### Why Single-Path Tests Are Insufficient

1. **Path Dependency**: Performance depends on the specific sequence of events
2. **Luck vs Skill**: Hard to distinguish between genuine edge and fortunate timing
3. **Overconfidence**: Single number gives false sense of certainty
4. **Tail Risk**: Doesn't reveal worst-case scenarios

## Monte Carlo Simulation Solution

Monte Carlo methods address these issues by:

1. **Generating Multiple Scenarios**: Run thousands of variations
2. **Measuring Uncertainty**: Create confidence intervals for metrics
3. **Stress Testing**: Identify worst-case outcomes
4. **Robustness Testing**: Ensure strategy works across scenarios

### Types of Monte Carlo Simulations for Trading

\`\`\`python
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MonteCarloMethod(Enum):
    """Types of Monte Carlo simulation methods"""
    TRADE_RANDOMIZATION = "trade_randomization"  # Shuffle trade order
    BOOTSTRAP = "bootstrap"  # Resample with replacement
    PARAMETRIC = "parametric"  # Simulate from fitted distribution
    BLOCK_BOOTSTRAP = "block_bootstrap"  # Preserve time series structure

@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation"""
    method: MonteCarloMethod
    num_simulations: int
    metric_name: str
    simulated_values: np.ndarray
    original_value: float
    confidence_intervals: Dict[float, Tuple[float, float]]
    percentile_rank: float  # Where original falls in distribution
    
    def __repr__(self) -> str:
        ci_95 = self.confidence_intervals.get(0.95, (0, 0))
        return (
            f"MonteCarloResult({self.method.value}, "
            f"original={self.original_value:.3f}, "
            f"95% CI=[{ci_95[0]:.3f}, {ci_95[1]:.3f}], "
            f"percentile={self.percentile_rank:.1%})"
        )

class MonteCarloSimulator:
    """
    Monte Carlo simulation framework for trading strategy validation
    """
    
    def __init__(
        self,
        num_simulations: int = 10000,
        random_seed: Optional[int] = None,
        n_jobs: int = -1
    ):
        """
        Initialize Monte Carlo simulator
        
        Args:
            num_simulations: Number of Monte Carlo runs
            random_seed: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 for all CPUs)
        """
        self.num_simulations = num_simulations
        self.random_seed = random_seed
        self.n_jobs = n_jobs
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def randomize_trades(
        self,
        trades: pd.DataFrame,
        preserve_structure: bool = False
    ) -> pd.DataFrame:
        """
        Randomize the order of trades
        
        Args:
            trades: DataFrame with trade data
            preserve_structure: If True, preserve monthly/weekly structure
            
        Returns:
            Randomized trades DataFrame
        """
        if preserve_structure:
            # Randomize within time blocks (e.g., within each month)
            trades['month'] = pd.to_datetime(trades.index).to_period('M')
            
            randomized_parts = []
            for month, group in trades.groupby('month'):
                shuffled = group.sample(frac=1.0)
                randomized_parts.append(shuffled)
            
            randomized = pd.concat(randomized_parts)
            randomized = randomized.drop('month', axis=1)
        else:
            # Complete randomization
            randomized = trades.sample(frac=1.0)
        
        return randomized
    
    def bootstrap_resample(
        self,
        returns: np.ndarray,
        block_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Bootstrap resample returns
        
        Args:
            returns: Array of returns
            block_size: If provided, use block bootstrap to preserve autocorrelation
            
        Returns:
            Resampled returns array
        """
        n = len(returns)
        
        if block_size is None:
            # Standard bootstrap (with replacement)
            indices = np.random.randint(0, n, size=n)
            return returns[indices]
        else:
            # Block bootstrap
            num_blocks = int(np.ceil(n / block_size))
            resampled = []
            
            for _ in range(num_blocks):
                start_idx = np.random.randint(0, n - block_size + 1)
                block = returns[start_idx:start_idx + block_size]
                resampled.append(block)
            
            resampled = np.concatenate(resampled)[:n]
            return resampled
    
    def calculate_metric(
        self,
        returns: np.ndarray,
        metric: str = 'sharpe_ratio'
    ) -> float:
        """
        Calculate performance metric
        
        Args:
            returns: Array of returns
            metric: Metric to calculate
            
        Returns:
            Metric value
        """
        if len(returns) == 0:
            return 0.0
        
        if metric == 'sharpe_ratio':
            if returns.std() == 0:
                return 0.0
            return (returns.mean() / returns.std()) * np.sqrt(252)
        
        elif metric == 'total_return':
            return np.sum(returns)
        
        elif metric == 'max_drawdown':
            cumulative = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = cumulative - running_max
            return np.min(drawdown)
        
        elif metric == 'win_rate':
            return (returns > 0).mean()
        
        elif metric == 'sortino_ratio':
            downside_returns = returns[returns < 0]
            if len(downside_returns) == 0 or downside_returns.std() == 0:
                return 0.0
            return (returns.mean() / downside_returns.std()) * np.sqrt(252)
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def run_trade_randomization(
        self,
        trades: pd.DataFrame,
        metric: str = 'sharpe_ratio',
        preserve_structure: bool = False
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation by randomizing trade order
        
        Args:
            trades: DataFrame with trade returns
            metric: Metric to analyze
            preserve_structure: Preserve time structure in randomization
            
        Returns:
            MonteCarloResult object
        """
        logger.info(
            f"Running trade randomization MC ({self.num_simulations} simulations)"
        )
        
        # Calculate original metric
        original_returns = trades['returns'].values
        original_value = self.calculate_metric(original_returns, metric)
        
        # Run simulations
        simulated_values = np.zeros(self.num_simulations)
        
        for i in range(self.num_simulations):
            # Randomize trades
            randomized_trades = self.randomize_trades(
                trades,
                preserve_structure=preserve_structure
            )
            
            # Calculate metric
            randomized_returns = randomized_trades['returns'].values
            simulated_values[i] = self.calculate_metric(randomized_returns, metric)
            
            if (i + 1) % 1000 == 0:
                logger.debug(f"Completed {i + 1}/{self.num_simulations} simulations")
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(simulated_values)
        
        # Calculate percentile rank
        percentile_rank = stats.percentileofscore(simulated_values, original_value) / 100
        
        return MonteCarloResult(
            method=MonteCarloMethod.TRADE_RANDOMIZATION,
            num_simulations=self.num_simulations,
            metric_name=metric,
            simulated_values=simulated_values,
            original_value=original_value,
            confidence_intervals=confidence_intervals,
            percentile_rank=percentile_rank
        )
    
    def run_bootstrap(
        self,
        returns: np.ndarray,
        metric: str = 'sharpe_ratio',
        block_size: Optional[int] = None
    ) -> MonteCarloResult:
        """
        Run bootstrap Monte Carlo simulation
        
        Args:
            returns: Array of returns
            metric: Metric to analyze
            block_size: Block size for block bootstrap
            
        Returns:
            MonteCarloResult object
        """
        logger.info(
            f"Running bootstrap MC ({self.num_simulations} simulations)"
        )
        
        # Calculate original metric
        original_value = self.calculate_metric(returns, metric)
        
        # Run simulations
        simulated_values = np.zeros(self.num_simulations)
        
        for i in range(self.num_simulations):
            # Bootstrap resample
            resampled_returns = self.bootstrap_resample(returns, block_size)
            
            # Calculate metric
            simulated_values[i] = self.calculate_metric(resampled_returns, metric)
            
            if (i + 1) % 1000 == 0:
                logger.debug(f"Completed {i + 1}/{self.num_simulations} simulations")
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(simulated_values)
        
        # Calculate percentile rank
        percentile_rank = stats.percentileofscore(simulated_values, original_value) / 100
        
        method = (MonteCarloMethod.BLOCK_BOOTSTRAP if block_size 
                 else MonteCarloMethod.BOOTSTRAP)
        
        return MonteCarloResult(
            method=method,
            num_simulations=self.num_simulations,
            metric_name=metric,
            simulated_values=simulated_values,
            original_value=original_value,
            confidence_intervals=confidence_intervals,
            percentile_rank=percentile_rank
        )
    
    def run_parametric_simulation(
        self,
        returns: np.ndarray,
        metric: str = 'sharpe_ratio',
        distribution: str = 'normal'
    ) -> MonteCarloResult:
        """
        Run parametric Monte Carlo simulation
        
        Fits distribution to returns and simulates from it
        
        Args:
            returns: Array of returns
            metric: Metric to analyze
            distribution: Distribution to use ('normal', 't', 'skewnorm')
            
        Returns:
            MonteCarloResult object
        """
        logger.info(
            f"Running parametric MC ({self.num_simulations} simulations)"
        )
        
        # Calculate original metric
        original_value = self.calculate_metric(returns, metric)
        
        # Fit distribution
        n = len(returns)
        
        if distribution == 'normal':
            mu, sigma = returns.mean(), returns.std()
            
            def sample_func():
                return np.random.normal(mu, sigma, n)
        
        elif distribution == 't':
            # Fit Student's t-distribution
            params = stats.t.fit(returns)
            
            def sample_func():
                return stats.t.rvs(*params, size=n)
        
        elif distribution == 'skewnorm':
            # Fit skewed normal distribution
            params = stats.skewnorm.fit(returns)
            
            def sample_func():
                return stats.skewnorm.rvs(*params, size=n)
        
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        # Run simulations
        simulated_values = np.zeros(self.num_simulations)
        
        for i in range(self.num_simulations):
            # Sample from fitted distribution
            simulated_returns = sample_func()
            
            # Calculate metric
            simulated_values[i] = self.calculate_metric(simulated_returns, metric)
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(simulated_values)
        
        # Calculate percentile rank
        percentile_rank = stats.percentileofscore(simulated_values, original_value) / 100
        
        return MonteCarloResult(
            method=MonteCarloMethod.PARAMETRIC,
            num_simulations=self.num_simulations,
            metric_name=metric,
            simulated_values=simulated_values,
            original_value=original_value,
            confidence_intervals=confidence_intervals,
            percentile_rank=percentile_rank
        )
    
    def _calculate_confidence_intervals(
        self,
        values: np.ndarray
    ) -> Dict[float, Tuple[float, float]]:
        """
        Calculate confidence intervals
        
        Args:
            values: Array of simulated values
            
        Returns:
            Dictionary of confidence intervals
        """
        confidence_levels = [0.90, 0.95, 0.99]
        intervals = {}
        
        for conf_level in confidence_levels:
            alpha = 1 - conf_level
            lower = np.percentile(values, alpha / 2 * 100)
            upper = np.percentile(values, (1 - alpha / 2) * 100)
            intervals[conf_level] = (lower, upper)
        
        return intervals
    
    def plot_distribution(
        self,
        result: MonteCarloResult,
        save_path: Optional[str] = None
    ):
        """
        Plot distribution of simulated values
        
        Args:
            result: MonteCarloResult to plot
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(
            result.simulated_values,
            bins=50,
            alpha=0.7,
            edgecolor='black',
            density=True
        )
        axes[0].axvline(
            result.original_value,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Original: {result.original_value:.3f}'
        )
        
        # Add confidence interval
        ci_95 = result.confidence_intervals[0.95]
        axes[0].axvline(ci_95[0], color='green', linestyle=':', alpha=0.7)
        axes[0].axvline(ci_95[1], color='green', linestyle=':', alpha=0.7, 
                       label=f'95% CI: [{ci_95[0]:.3f}, {ci_95[1]:.3f}]')
        
        axes[0].set_xlabel(result.metric_name.replace('_', ' ').title())
        axes[0].set_ylabel('Density')
        axes[0].set_title(f'Distribution of {result.metric_name}\\n{result.method.value}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Q-Q plot
        stats.probplot(result.simulated_values, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot (Normal)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()


# Example usage
if __name__ == "__main__":
    # Simulate trading returns
    np.random.seed(42)
    n_trades = 252  # 1 year of daily trades
    
    # Generate returns with some positive drift (strategy edge)
    returns = np.random.normal(0.001, 0.02, n_trades)  # 0.1% mean, 2% std
    
    trades = pd.DataFrame({
        'returns': returns,
        'timestamp': pd.date_range('2023-01-01', periods=n_trades, freq='D')
    })
    trades.set_index('timestamp', inplace=True)
    
    # Initialize simulator
    mc_sim = MonteCarloSimulator(num_simulations=10000, random_seed=42)
    
    # Run trade randomization
    result_randomization = mc_sim.run_trade_randomization(
        trades,
        metric='sharpe_ratio',
        preserve_structure=False
    )
    
    print("\\n" + "="*80)
    print("MONTE CARLO SIMULATION RESULTS - Trade Randomization")
    print("="*80)
    print(f"Original Sharpe Ratio: {result_randomization.original_value:.3f}")
    print(f"Mean Simulated Sharpe: {result_randomization.simulated_values.mean():.3f}")
    print(f"Std Simulated Sharpe: {result_randomization.simulated_values.std():.3f}")
    print(f"\\nConfidence Intervals:")
    for conf_level, (lower, upper) in result_randomization.confidence_intervals.items():
        print(f"  {conf_level:.0%}: [{lower:.3f}, {upper:.3f}]")
    print(f"\\nPercentile Rank: {result_randomization.percentile_rank:.1%}")
    
    if result_randomization.percentile_rank > 0.95:
        print("✓ Strategy performs better than 95% of random permutations")
    elif result_randomization.percentile_rank > 0.75:
        print("⚠ Strategy shows some edge but not statistically strong")
    else:
        print("✗ Strategy performance may be due to luck")
    
    # Run bootstrap
    result_bootstrap = mc_sim.run_bootstrap(
        returns,
        metric='sharpe_ratio',
        block_size=None
    )
    
    print("\\n" + "="*80)
    print("MONTE CARLO SIMULATION RESULTS - Bootstrap")
    print("="*80)
    print(result_bootstrap)
    
    # Plot distributions
    mc_sim.plot_distribution(result_randomization, save_path='/tmp/mc_randomization.png')
\`\`\`

## Key Concepts

### 1. **Trade Randomization**
- Tests if trade sequence matters
- Reveals path dependency
- Identifies luck vs skill

### 2. **Bootstrap Resampling**
- Quantifies uncertainty
- Creates confidence intervals
- No distributional assumptions

### 3. **Stress Testing**
- Worst-case scenarios
- Tail risk assessment
- Drawdown analysis

## Production Considerations

Monte Carlo simulations are computationally expensive but essential for:
- Strategy validation
- Risk assessment
- Investor reporting
- Regulatory compliance

# Advanced Monte Carlo Techniques

## Stress Testing and Scenario Analysis

\`\`\`python
class StressTestSimulator:
    """
    Stress testing using Monte Carlo methods
    """
    
    def __init__(self, base_returns: np.ndarray):
        self.base_returns = base_returns
        self.base_metrics = self._calculate_all_metrics(base_returns)
    
    def stress_test_volatility(
        self,
        volatility_multipliers: List[float] = [1.5, 2.0, 3.0]
    ) -> pd.DataFrame:
        """
        Stress test strategy under different volatility regimes
        
        Args:
            volatility_multipliers: Volatility scaling factors
            
        Returns:
            DataFrame with stress test results
        """
        results = []
        
        for mult in volatility_multipliers:
            # Scale returns (preserving mean)
            mean_return = self.base_returns.mean()
            stressed_returns = mean_return + (self.base_returns - mean_return) * mult
            
            # Calculate metrics under stress
            stressed_metrics = self._calculate_all_metrics(stressed_returns)
            
            results.append({
                'volatility_multiplier': mult,
                **stressed_metrics,
                'sharpe_change_pct': (
                    (stressed_metrics['sharpe_ratio'] - self.base_metrics['sharpe_ratio']) /
                    self.base_metrics['sharpe_ratio'] * 100
                )
            })
        
        return pd.DataFrame(results)
    
    def stress_test_drawdowns(
        self,
        insert_crashes: List[Tuple[float, int]] = [(-0.10, 1), (-0.20, 1)]
    ) -> pd.DataFrame:
        """
        Stress test by inserting market crashes
        
        Args:
            insert_crashes: List of (crash_size, num_days) tuples
            
        Returns:
            DataFrame with results
        """
        results = []
        
        for crash_size, num_days in insert_crashes:
            # Insert crash into random position
            stressed_returns = self.base_returns.copy()
            crash_start = np.random.randint(0, len(stressed_returns) - num_days)
            
            # Distribute crash over multiple days
            crash_per_day = crash_size / num_days
            stressed_returns[crash_start:crash_start + num_days] += crash_per_day
            
            # Calculate metrics
            stressed_metrics = self._calculate_all_metrics(stressed_returns)
            
            results.append({
                'crash_size': crash_size,
                'crash_days': num_days,
                **stressed_metrics
            })
        
        return pd.DataFrame(results)
    
    def monte_carlo_worst_case(
        self,
        num_simulations: int = 10000,
        percentile: float = 0.05
    ) -> Dict[str, float]:
        """
        Find worst-case scenarios using Monte Carlo
        
        Args:
            num_simulations: Number of simulations
            percentile: Percentile for worst case (e.g., 0.05 for bottom 5%)
            
        Returns:
            Dict of worst-case metrics
        """
        all_sharpes = []
        all_drawdowns = []
        all_returns = []
        
        for _ in range(num_simulations):
            # Bootstrap resample
            resampled = np.random.choice(
                self.base_returns,
                size=len(self.base_returns),
                replace=True
            )
            
            metrics = self._calculate_all_metrics(resampled)
            all_sharpes.append(metrics['sharpe_ratio'])
            all_drawdowns.append(metrics['max_drawdown'])
            all_returns.append(metrics['total_return'])
        
        # Find worst-case (low percentile)
        worst_case = {
            'worst_sharpe': np.percentile(all_sharpes, percentile * 100),
            'worst_drawdown': np.percentile(all_drawdowns, percentile * 100),
            'worst_return': np.percentile(all_returns, percentile * 100),
            'median_sharpe': np.median(all_sharpes),
            'median_drawdown': np.median(all_drawdowns),
            'median_return': np.median(all_returns)
        }
        
        return worst_case
    
    def _calculate_all_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive metrics"""
        # Sharpe ratio
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        # Max drawdown
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_dd = np.min(drawdown)
        
        # Other metrics
        total_return = np.sum(returns)
        win_rate = (returns > 0).mean()
        
        return {
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'total_return': total_return,
            'win_rate': win_rate,
            'volatility': returns.std() * np.sqrt(252)
        }


class ConfidenceIntervalAnalyzer:
    """
    Analyze confidence intervals for strategy metrics
    """
    
    def __init__(self, mc_results: List[MonteCarloResult]):
        self.results = mc_results
    
    def compare_confidence_intervals(self) -> pd.DataFrame:
        """
        Compare confidence intervals across different metrics
        
        Returns:
            DataFrame with CI comparison
        """
        comparison = []
        
        for result in self.results:
            for conf_level, (lower, upper) in result.confidence_intervals.items():
                comparison.append({
                    'metric': result.metric_name,
                    'method': result.method.value,
                    'confidence_level': conf_level,
                    'original_value': result.original_value,
                    'ci_lower': lower,
                    'ci_upper': upper,
                    'ci_width': upper - lower,
                    'ci_width_pct': (upper - lower) / abs(result.original_value) * 100 
                                    if result.original_value != 0 else np.inf
                })
        
        return pd.DataFrame(comparison)
    
    def assess_statistical_significance(
        self,
        null_value: float = 0.0,
        confidence_level: float = 0.95
    ) -> Dict[str, bool]:
        """
        Assess if metrics are statistically significant
        
        Args:
            null_value: Null hypothesis value (e.g., 0 for no edge)
            confidence_level: Confidence level for test
            
        Returns:
            Dictionary of significance results
        """
        significance = {}
        
        for result in self.results:
            ci = result.confidence_intervals.get(confidence_level, (0, 0))
            
            # If null value is outside CI, reject null hypothesis
            is_significant = null_value < ci[0] or null_value > ci[1]
            
            significance[result.metric_name] = is_significant
        
        return significance


# Example: Comprehensive Monte Carlo Analysis
def run_comprehensive_mc_analysis(
    trades: pd.DataFrame,
    returns: np.ndarray
) -> Dict[str, Any]:
    """
    Run comprehensive Monte Carlo analysis
    
    Args:
        trades: Trade data
        returns: Return series
        
    Returns:
        Comprehensive analysis results
    """
    mc_sim = MonteCarloSimulator(num_simulations=10000, random_seed=42)
    
    # 1. Trade randomization
    result_randomization = mc_sim.run_trade_randomization(
        trades,
        metric='sharpe_ratio'
    )
    
    # 2. Bootstrap
    result_bootstrap = mc_sim.run_bootstrap(
        returns,
        metric='sharpe_ratio'
    )
    
    # 3. Block bootstrap (preserve autocorrelation)
    result_block_bootstrap = mc_sim.run_bootstrap(
        returns,
        metric='sharpe_ratio',
        block_size=20  # 20-day blocks
    )
    
    # 4. Stress testing
    stress_tester = StressTestSimulator(returns)
    
    volatility_stress = stress_tester.stress_test_volatility()
    drawdown_stress = stress_tester.stress_test_drawdowns()
    worst_case = stress_tester.monte_carlo_worst_case()
    
    # 5. Confidence interval analysis
    ci_analyzer = ConfidenceIntervalAnalyzer([
        result_randomization,
        result_bootstrap,
        result_block_bootstrap
    ])
    
    ci_comparison = ci_analyzer.compare_confidence_intervals()
    significance = ci_analyzer.assess_statistical_significance(null_value=0)
    
    return {
        'randomization_result': result_randomization,
        'bootstrap_result': result_bootstrap,
        'block_bootstrap_result': result_block_bootstrap,
        'volatility_stress': volatility_stress,
        'drawdown_stress': drawdown_stress,
        'worst_case_scenarios': worst_case,
        'ci_comparison': ci_comparison,
        'statistical_significance': significance
    }


if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_trades = 252
    returns = np.random.normal(0.001, 0.02, n_trades)
    
    trades = pd.DataFrame({
        'returns': returns,
        'timestamp': pd.date_range('2023-01-01', periods=n_trades, freq='D')
    })
    trades.set_index('timestamp', inplace=True)
    
    # Run comprehensive analysis
    analysis = run_comprehensive_mc_analysis(trades, returns)
    
    print("\\n" + "="*80)
    print("COMPREHENSIVE MONTE CARLO ANALYSIS")
    print("="*80)
    
    print("\\nStatistical Significance:")
    for metric, is_sig in analysis['statistical_significance'].items():
        status = "✓ Significant" if is_sig else "✗ Not significant"
        print(f"  {metric}: {status}")
    
    print("\\nWorst-Case Scenarios (5th percentile):")
    for key, value in analysis['worst_case_scenarios'].items():
        print(f"  {key}: {value:.3f}")
    
    print("\\nVolatility Stress Test:")
    print(analysis['volatility_stress'].to_string(index=False))
\`\`\`

## Summary

Monte Carlo simulation provides:
1. **Confidence intervals** for all metrics
2. **Robustness testing** across scenarios
3. **Stress testing** for extreme conditions
4. **Statistical significance** assessment

Essential for professional strategy validation and risk management.
`,
    },
  ],
};

export default monteCarloSimulation;
