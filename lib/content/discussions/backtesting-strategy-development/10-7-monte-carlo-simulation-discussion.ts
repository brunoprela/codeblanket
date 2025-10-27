import { Content } from '@/lib/types';

const monteCarloSimulationDiscussion: Content = {
  title: 'Monte Carlo Simulation - Discussion Questions',
  description:
    'Deep-dive discussion questions on Monte Carlo methods, bootstrap resampling, stress testing, and confidence interval analysis for trading strategies',
  sections: [
    {
      title: 'Discussion Questions',
      content: `
# Discussion Questions: Monte Carlo Simulation for Trading

## Question 1: Designing a Monte Carlo Framework for Strategy Validation

**Scenario**: You're building a Monte Carlo simulation framework that will be used by 20+ quantitative researchers to validate their trading strategies before deployment. The strategies span multiple asset classes (equities, futures, FX, options) and frequencies (HFT to position trading). Researchers have complained that existing tools are too slow, give inconsistent results, and don't handle edge cases well.

Your task is to design a production-grade Monte Carlo framework that:
1. Supports multiple simulation methods (trade randomization, bootstrap, parametric)
2. Runs efficiently (< 5 minutes for 10,000 simulations)
3. Provides comprehensive statistical analysis
4. Handles edge cases (low trade counts, non-normal distributions, autocorrelation)
5. Generates publication-ready reports for investors

**Questions to Address**:
- What simulation methods would you implement and when should each be used?
- How would you optimize for computational performance?
- What statistical tests and visualizations would you include?
- How would you handle edge cases and validate results?
- What best practices would you enforce?

### Comprehensive Answer

A production-grade Monte Carlo framework requires careful consideration of statistical rigor, computational efficiency, and user experience. Here's a comprehensive design:

#### Architecture and Implementation

\`\`\`python
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimulationMethod(Enum):
    """Supported Monte Carlo methods"""
    TRADE_SHUFFLE = "trade_shuffle"  # Randomize trade order
    BOOTSTRAP = "bootstrap"  # Standard bootstrap
    BLOCK_BOOTSTRAP = "block_bootstrap"  # Preserve autocorrelation
    PARAMETRIC = "parametric"  # Fit and sample from distribution
    STRATIFIED = "stratified"  # Stratified resampling
    
class EdgeCaseHandler:
    """Handle edge cases in Monte Carlo simulation"""
    
    @staticmethod
    def check_sufficient_data(
        returns: np.ndarray,
        min_trades: int = 30
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if there's sufficient data for reliable MC simulation
        
        Args:
            returns: Return series
            min_trades: Minimum number of trades required
            
        Returns:
            Tuple of (is_sufficient, warning_message)
        """
        n = len(returns)
        
        if n < min_trades:
            return False, (
                f"Insufficient data: {n} trades < minimum {min_trades}. "
                "Results may not be statistically reliable."
            )
        
        if n < 100:
            return True, (
                f"Limited data: {n} trades. Consider using bootstrap instead of "
                "trade randomization for better stability."
            )
        
        return True, None
    
    @staticmethod
    def check_distribution_assumptions(
        returns: np.ndarray
    ) -> Dict[str, Any]:
        """
        Check distributional properties of returns
        
        Returns:
            Dictionary with distribution statistics and warnings
        """
        # Normality test
        _, p_value_norm = stats.shapiro(returns[:5000])  # Shapiro-Wilk limited to 5000
        is_normal = p_value_norm > 0.05
        
        # Skewness and kurtosis
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Autocorrelation
        autocorr_lag1 = pd.Series(returns).autocorr(lag=1)
        
        warnings_list = []
        
        if not is_normal:
            warnings_list.append(
                "Returns are not normally distributed. "
                "Consider using bootstrap or fitted distribution (t, skew-normal)."
            )
        
        if abs(skewness) > 1:
            warnings_list.append(
                f"High skewness detected ({skewness:.2f}). "
                "Parametric simulation should use skewed distribution."
            )
        
        if kurtosis > 3:
            warnings_list.append(
                f"Fat tails detected (kurtosis={kurtosis:.2f}). "
                "Consider using Student's t-distribution."
            )
        
        if abs(autocorr_lag1) > 0.2:
            warnings_list.append(
                f"Significant autocorrelation detected ({autocorr_lag1:.2f}). "
                "Use block bootstrap to preserve time series structure."
            )
        
        return {
            'is_normal': is_normal,
            'p_value_normality': p_value_norm,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'autocorrelation_lag1': autocorr_lag1,
            'warnings': warnings_list,
            'recommended_method': EdgeCaseHandler._recommend_method(
                is_normal, skewness, kurtosis, autocorr_lag1
            )
        }
    
    @staticmethod
    def _recommend_method(
        is_normal: bool,
        skewness: float,
        kurtosis: float,
        autocorr: float
    ) -> SimulationMethod:
        """Recommend simulation method based on data characteristics"""
        
        # Strong autocorrelation -> block bootstrap
        if abs(autocorr) > 0.3:
            return SimulationMethod.BLOCK_BOOTSTRAP
        
        # Non-normal with fat tails -> parametric with t-dist
        if not is_normal and kurtosis > 3:
            return SimulationMethod.PARAMETRIC
        
        # Default to standard bootstrap
        return SimulationMethod.BOOTSTRAP


class ProductionMonteCarloFramework:
    """
    Production-grade Monte Carlo simulation framework
    """
    
    def __init__(
        self,
        num_simulations: int = 10000,
        confidence_levels: List[float] = [0.90, 0.95, 0.99],
        random_seed: Optional[int] = None,
        n_jobs: int = -1,
        verbose: bool = True
    ):
        """
        Initialize framework
        
        Args:
            num_simulations: Number of MC simulations
            confidence_levels: Confidence levels for intervals
            random_seed: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 for all CPUs)
            verbose: Show progress bars
        """
        self.num_simulations = num_simulations
        self.confidence_levels = confidence_levels
        self.random_seed = random_seed
        self.n_jobs = n_jobs if n_jobs != -1 else None
        self.verbose = verbose
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.edge_case_handler = EdgeCaseHandler()
    
    def run_comprehensive_analysis(
        self,
        returns: np.ndarray,
        metrics: List[str] = ['sharpe_ratio', 'max_drawdown', 'win_rate'],
        auto_select_method: bool = True,
        method: Optional[SimulationMethod] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive Monte Carlo analysis
        
        Args:
            returns: Return series
            metrics: Metrics to analyze
            auto_select_method: Automatically select best method
            method: Specific method to use (if not auto)
            
        Returns:
            Comprehensive results dictionary
        """
        logger.info("Starting comprehensive Monte Carlo analysis")
        
        # Edge case detection
        sufficient, warning = self.edge_case_handler.check_sufficient_data(returns)
        if not sufficient:
            raise ValueError(warning)
        if warning:
            logger.warning(warning)
        
        # Distribution analysis
        dist_analysis = self.edge_case_handler.check_distribution_assumptions(returns)
        
        for warn in dist_analysis['warnings']:
            logger.warning(warn)
        
        # Select method
        if auto_select_method:
            selected_method = dist_analysis['recommended_method']
            logger.info(f"Auto-selected method: {selected_method.value}")
        else:
            selected_method = method or SimulationMethod.BOOTSTRAP
        
        # Run simulations for each metric
        results = {}
        
        for metric_name in metrics:
            logger.info(f"Running simulations for {metric_name}")
            
            metric_result = self._run_single_metric(
                returns,
                metric_name,
                selected_method
            )
            
            results[metric_name] = metric_result
        
        # Generate summary report
        summary = self._generate_summary_report(
            results,
            dist_analysis,
            selected_method
        )
        
        return {
            'results': results,
            'distribution_analysis': dist_analysis,
            'method_used': selected_method,
            'summary_report': summary,
            'metadata': {
                'num_simulations': self.num_simulations,
                'num_observations': len(returns),
                'confidence_levels': self.confidence_levels
            }
        }
    
    def _run_single_metric(
        self,
        returns: np.ndarray,
        metric_name: str,
        method: SimulationMethod
    ) -> Dict[str, Any]:
        """Run MC simulation for a single metric"""
        
        # Calculate original metric
        original_value = self._calculate_metric(returns, metric_name)
        
        # Parallel simulation
        simulated_values = self._parallel_simulate(
            returns,
            metric_name,
            method
        )
        
        # Calculate statistics
        mean_sim = np.mean(simulated_values)
        std_sim = np.std(simulated_values)
        
        # Confidence intervals
        confidence_intervals = {}
        for conf_level in self.confidence_levels:
            alpha = 1 - conf_level
            lower = np.percentile(simulated_values, alpha / 2 * 100)
            upper = np.percentile(simulated_values, (1 - alpha / 2) * 100)
            confidence_intervals[conf_level] = (lower, upper)
        
        # Percentile rank
        percentile_rank = stats.percentileofscore(
            simulated_values, original_value
        ) / 100
        
        # Statistical significance (is original value outside 95% CI?)
        ci_95 = confidence_intervals[0.95]
        is_significant = original_value < ci_95[0] or original_value > ci_95[1]
        
        return {
            'original_value': original_value,
            'mean_simulated': mean_sim,
            'std_simulated': std_sim,
            'simulated_values': simulated_values,
            'confidence_intervals': confidence_intervals,
            'percentile_rank': percentile_rank,
            'is_significant_95': is_significant,
            'p_value': min(percentile_rank, 1 - percentile_rank) * 2  # Two-tailed
        }
    
    def _parallel_simulate(
        self,
        returns: np.ndarray,
        metric_name: str,
        method: SimulationMethod
    ) -> np.ndarray:
        """Run simulations in parallel"""
        
        # Determine chunk size for parallel processing
        chunk_size = max(100, self.num_simulations // (self.n_jobs or 8))
        
        simulated_values = []
        
        # Use tqdm for progress bar if verbose
        iterator = range(self.num_simulations)
        if self.verbose:
            iterator = tqdm(iterator, desc=f"MC Simulation ({metric_name})")
        
        # For true parallelization, we'd use ProcessPoolExecutor
        # Simplified here for demonstration
        for _ in iterator:
            # Resample based on method
            if method == SimulationMethod.BOOTSTRAP:
                resampled = np.random.choice(returns, size=len(returns), replace=True)
            elif method == SimulationMethod.BLOCK_BOOTSTRAP:
                resampled = self._block_bootstrap(returns, block_size=20)
            elif method == SimulationMethod.TRADE_SHUFFLE:
                resampled = np.random.permutation(returns)
            elif method == SimulationMethod.PARAMETRIC:
                resampled = self._parametric_sample(returns)
            else:
                resampled = returns
            
            # Calculate metric
            value = self._calculate_metric(resampled, metric_name)
            simulated_values.append(value)
        
        return np.array(simulated_values)
    
    def _block_bootstrap(self, returns: np.ndarray, block_size: int) -> np.ndarray:
        """Block bootstrap resampling"""
        n = len(returns)
        num_blocks = int(np.ceil(n / block_size))
        resampled = []
        
        for _ in range(num_blocks):
            start_idx = np.random.randint(0, n - block_size + 1)
            block = returns[start_idx:start_idx + block_size]
            resampled.append(block)
        
        resampled = np.concatenate(resampled)[:n]
        return resampled
    
    def _parametric_sample(self, returns: np.ndarray) -> np.ndarray:
        """Sample from fitted distribution"""
        # Fit Student's t-distribution
        params = stats.t.fit(returns)
        return stats.t.rvs(*params, size=len(returns))
    
    def _calculate_metric(self, returns: np.ndarray, metric_name: str) -> float:
        """Calculate performance metric"""
        if len(returns) == 0:
            return 0.0
        
        if metric_name == 'sharpe_ratio':
            if returns.std() == 0:
                return 0.0
            return (returns.mean() / returns.std()) * np.sqrt(252)
        
        elif metric_name == 'max_drawdown':
            cumulative = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = cumulative - running_max
            return np.min(drawdown)
        
        elif metric_name == 'win_rate':
            return (returns > 0).mean()
        
        elif metric_name == 'total_return':
            return np.sum(returns)
        
        else:
            raise ValueError(f"Unknown metric: {metric_name}")
    
    def _generate_summary_report(
        self,
        results: Dict[str, Dict],
        dist_analysis: Dict,
        method: SimulationMethod
    ) -> str:
        """Generate human-readable summary report"""
        
        report = []
        report.append("="*80)
        report.append("MONTE CARLO SIMULATION REPORT")
        report.append("="*80)
        report.append("")
        
        report.append(f"Method Used: {method.value}")
        report.append(f"Number of Simulations: {self.num_simulations:,}")
        report.append("")
        
        report.append("Data Characteristics:")
        report.append(f"  Normality: {'Yes' if dist_analysis['is_normal'] else 'No'}")
        report.append(f"  Skewness: {dist_analysis['skewness']:.3f}")
        report.append(f"  Kurtosis: {dist_analysis['kurtosis']:.3f}")
        report.append(f"  Autocorrelation (lag 1): {dist_analysis['autocorrelation_lag1']:.3f}")
        report.append("")
        
        report.append("Results by Metric:")
        report.append("")
        
        for metric_name, metric_result in results.items():
            report.append(f"  {metric_name.replace('_', ' ').title()}:")
            report.append(f"    Original Value: {metric_result['original_value']:.4f}")
            report.append(f"    Mean (Simulated): {metric_result['mean_simulated']:.4f}")
            report.append(f"    Std (Simulated): {metric_result['std_simulated']:.4f}")
            report.append(f"    Percentile Rank: {metric_result['percentile_rank']:.1%}")
            report.append(f"    P-value: {metric_result['p_value']:.4f}")
            
            ci_95 = metric_result['confidence_intervals'][0.95]
            report.append(f"    95% CI: [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")
            
            # Assessment
            if metric_result['percentile_rank'] > 0.95:
                report.append("    ✓ Strong performance (top 5%)")
            elif metric_result['percentile_rank'] > 0.75:
                report.append("    ⚠ Above average performance")
            elif metric_result['percentile_rank'] > 0.25:
                report.append("    ≈ Average performance")
            else:
                report.append("    ✗ Below average performance")
            
            report.append("")
        
        report.append("="*80)
        
        return "\\n".join(report)
    
    def plot_results(
        self,
        analysis_results: Dict[str, Any],
        save_path: Optional[str] = None
    ):
        """Generate comprehensive visualization"""
        
        results = analysis_results['results']
        n_metrics = len(results)
        
        fig, axes = plt.subplots(n_metrics, 2, figsize=(14, 5 * n_metrics))
        
        if n_metrics == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (metric_name, metric_result) in enumerate(results.items()):
            # Histogram
            axes[idx, 0].hist(
                metric_result['simulated_values'],
                bins=50,
                alpha=0.7,
                edgecolor='black',
                density=True
            )
            axes[idx, 0].axvline(
                metric_result['original_value'],
                color='red',
                linestyle='--',
                linewidth=2,
                label=f'Original: {metric_result["original_value"]:.3f}'
            )
            
            ci_95 = metric_result['confidence_intervals'][0.95]
            axes[idx, 0].axvline(ci_95[0], color='green', linestyle=':', alpha=0.7)
            axes[idx, 0].axvline(ci_95[1], color='green', linestyle=':', alpha=0.7,
                               label=f'95% CI')
            
            axes[idx, 0].set_xlabel(metric_name.replace('_', ' ').title())
            axes[idx, 0].set_ylabel('Density')
            axes[idx, 0].set_title(f'{metric_name} Distribution')
            axes[idx, 0].legend()
            axes[idx, 0].grid(True, alpha=0.3)
            
            # Q-Q Plot
            stats.probplot(metric_result['simulated_values'], dist="norm", plot=axes[idx, 1])
            axes[idx, 1].set_title(f'{metric_name} Q-Q Plot')
            axes[idx, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Plots saved to {save_path}")
        else:
            plt.show()


# Example usage
if __name__ == "__main__":
    # Simulate trading returns
    np.random.seed(42)
    n_trades = 252
    
    # Returns with skew and fat tails
    returns = stats.t.rvs(df=5, loc=0.001, scale=0.02, size=n_trades)
    
    # Initialize framework
    framework = ProductionMonteCarloFramework(
        num_simulations=10000,
        random_seed=42,
        verbose=True
    )
    
    # Run comprehensive analysis
    analysis = framework.run_comprehensive_analysis(
        returns,
        metrics=['sharpe_ratio', 'max_drawdown', 'win_rate'],
        auto_select_method=True
    )
    
    # Print report
    print(analysis['summary_report'])
    
    # Generate plots
    framework.plot_results(analysis, save_path='/tmp/mc_analysis.png')
\`\`\`

#### Key Design Decisions

**1. Method Selection Logic**:
- **Trade Randomization**: For strategies with sufficient trades (>100), tests path dependency
- **Bootstrap**: Default for most cases, no distributional assumptions
- **Block Bootstrap**: When autocorrelation detected (|ρ| > 0.2), preserves time series structure
- **Parametric**: When distribution is well-characterized (normal, t, skew-normal)

**2. Performance Optimization**:
- Parallel processing using ProcessPoolExecutor
- Vectorized operations with NumPy
- Efficient resampling algorithms
- Progress bars for user feedback
- Caching of intermediate results

**3. Edge Case Handling**:
- Minimum data requirements (30+ trades)
- Distribution diagnostics (normality tests, skew, kurtosis)
- Autocorrelation detection
- Automatic method recommendation
- Comprehensive warnings

**4. Statistical Rigor**:
- Multiple confidence levels (90%, 95%, 99%)
- P-value calculation
- Percentile ranking
- Distribution diagnostics
- Q-Q plots for normality assessment

**5. User Experience**:
- Automatic method selection with clear rationale
- Human-readable summary reports
- Publication-quality visualizations
- Comprehensive warnings and recommendations
- Reproducibility via random seeds

#### Best Practices Enforced

1. **Always check data sufficiency** before running simulations
2. **Use block bootstrap** for autocorrelated returns
3. **Run 10,000+ simulations** for stable estimates
4. **Report confidence intervals** along with point estimates
5. **Test multiple metrics** not just Sharpe ratio
6. **Document assumptions** and method selection
7. **Visualize distributions** don't just report numbers
8. **Set random seeds** for reproducibility

This framework would serve as the foundation for professional strategy validation across a quantitative research team.

---

## Question 2: Interpreting Divergent Monte Carlo Results

**Scenario**: You've run three different Monte Carlo methods on the same trading strategy and received conflicting results:

1. **Trade Randomization**: Original Sharpe 1.8, ranks at 85th percentile (good)
2. **Standard Bootstrap**: 95% CI for Sharpe is [0.4, 2.9], very wide
3. **Block Bootstrap (20-day blocks)**: 95% CI for Sharpe is [1.2, 2.3], narrower

The strategy is a daily momentum strategy with 250 trades over 1 year. Returns show significant positive autocorrelation (ρ=0.35).

**Task**:
1. Explain why the three methods give different results
2. Which result should you trust most and why?
3. What does this tell you about the strategy?
4. How would you report these results to stakeholders?

### Comprehensive Answer

The divergent results reveal important information about the strategy's characteristics and the appropriateness of each Monte Carlo method.

#### Why the Methods Differ

**Trade Randomization (85th percentile)**:
- Tests if the specific sequence of wins/losses matters
- 85th percentile means strategy performs better than 85% of random orderings
- Indicates genuine edge, not just lucky timing
- However, doesn't account for estimation uncertainty in the Sharpe ratio itself

**Standard Bootstrap (CI: [0.4, 2.9])**:
- Very wide interval indicates high uncertainty
- Standard bootstrap treats returns as independent
- With autocorrelation (ρ=0.35), this violates the independence assumption
- Underestimates actual confidence by breaking momentum structure
- Result: artificially inflated uncertainty

**Block Bootstrap (CI: [1.2, 2.3])**:
- Narrower interval because it preserves the momentum structure
- By keeping 20-day blocks intact, maintains autocorrelation
- More appropriate for this strategy
- Still wider than point estimate, reflecting true uncertainty

#### Which Result to Trust

**Primary**: Block bootstrap CI [1.2, 2.3]
**Supporting**: Trade randomization percentile (85th)

**Rationale**:

\`\`\`python
def analyze_monte_carlo_divergence(
    returns: np.ndarray,
    autocorr: float = 0.35
) -> Dict[str, Any]:
    """
    Analyze why Monte Carlo methods diverge
    
    Args:
        returns: Strategy returns
        autocorr: Measured autocorrelation
        
    Returns:
        Analysis of divergence
    """
    # Calculate effective N for autocorrelated series
    # Effective N is less than actual N when autocorrelation exists
    n_actual = len(returns)
    
    # Formula: n_eff = n_actual * (1 - ρ) / (1 + ρ)
    n_effective = n_actual * (1 - autocorr) / (1 + autocorr)
    
    # This explains why bootstrap CI is so wide
    # It uses n_actual, but effective information content is only n_effective
    
    # Standard error scales with sqrt(n_effective) not sqrt(n_actual)
    se_ratio = np.sqrt(n_actual / n_effective)
    
    print(f"Actual observations: {n_actual}")
    print(f"Effective observations: {n_effective:.0f}")
    print(f"SE inflation factor: {se_ratio:.2f}x")
    print(f"\\nThis explains why standard bootstrap CI is {se_ratio:.2f}x wider")
    
    return {
        'n_actual': n_actual,
        'n_effective': n_effective,
        'se_inflation': se_ratio,
        'interpretation': (
            "Block bootstrap correctly accounts for autocorrelation by "
            "preserving dependencies. Standard bootstrap incorrectly treats "
            "observations as independent, inflating uncertainty."
        )
    }

# For the scenario
analysis = analyze_monte_carlo_divergence(
    returns=np.random.randn(250),  # 250 trades
    autocorr=0.35
)
\`\`\`

Output:
\`\`\`
Actual observations: 250
Effective observations: 120
SE inflation factor: 1.44x

This explains why standard bootstrap CI is 1.44x wider
\`\`\`

#### What This Tells Us About the Strategy

1. **Genuine Edge**: Trade randomization (85th percentile) confirms the strategy has real alpha, not just lucky sequencing

2. **Momentum-Based**: High autocorrelation (ρ=0.35) indicates strategy profits from momentum/persistence
   - Winning streaks are real, not random
   - Past returns predict future returns (short-term)

3. **Confidence Range**: True Sharpe is likely between 1.2 and 2.3
   - Even worst case (1.2) is acceptable
   - Best case (2.3) is excellent
   - Original estimate (1.8) is near midpoint

4. **Statistical Significance**: Lower bound of 1.2 is well above zero
   - Strategy is statistically significant at 95% confidence
   - Robust even in worst-case resampling scenarios

5. **Method Sensitivity**: Wide divergence in results highlights importance of proper methodology
   - Using wrong method (standard bootstrap) would lead to underestimating strategy quality
   - Block bootstrap essential for momentum strategies

#### Reporting to Stakeholders

\`\`\`
STRATEGY VALIDATION REPORT
Monte Carlo Analysis for [Strategy Name]

EXECUTIVE SUMMARY:
The strategy demonstrates robust performance with statistical significance at 
the 95% confidence level. Monte Carlo analysis using appropriate methods 
(block bootstrap) confirms the strategy has a genuine edge.

KEY FINDINGS:

1. Performance Estimate:
   - Point Estimate: Sharpe Ratio 1.8
   - 95% Confidence Interval: [1.2, 2.3]
   - Interpretation: True Sharpe likely between 1.2 and 2.3

2. Statistical Robustness:
   - Strategy ranks at 85th percentile vs random orderings
   - Performance is NOT due to lucky timing
   - Genuine alpha detected

3. Strategy Characteristics:
   - Exhibits positive momentum (autocorrelation = 0.35)
   - Profits from short-term persistence in returns
   - Block bootstrap used to preserve momentum structure

4. Risk Considerations:
   - Even in worst-case simulation scenarios, Sharpe remains > 1.0
   - Strategy maintains edge across various market conditions
   - Confidence interval width reflects true uncertainty

RECOMMENDATION:
✓ Strategy approved for deployment
Rationale: Statistically significant edge with acceptable worst-case performance

METHODOLOGY NOTE:
We used block bootstrap (20-day blocks) rather than standard bootstrap because 
the strategy exhibits significant autocorrelation. This preserves the momentum 
structure and provides accurate uncertainty estimates. Standard bootstrap would 
have artificially inflated uncertainty by 1.4x.

RISK DISCLOSURE:
Past performance does not guarantee future results. Monte Carlo simulations 
are based on historical return distributions and may not capture regime changes 
or unprecedented market events.
\`\`\`

#### Technical Deep Dive for Quant Team

For internal documentation, provide more technical detail:

\`\`\`python
class MonteCarloComparison:
    """Compare different MC methods and explain divergence"""
    
    def __init__(self, returns: np.ndarray):
        self.returns = returns
        self.n = len(returns)
        self.autocorr = pd.Series(returns).autocorr(lag=1)
    
    def compare_methods(self) -> pd.DataFrame:
        """Generate comparison table"""
        
        results = []
        
        # Trade randomization
        trade_rand_result = self._trade_randomization()
        results.append({
            'method': 'Trade Randomization',
            'tests_for': 'Path dependency',
            'percentile_rank': trade_rand_result,
            'ci_width': 'N/A',
            'appropriate_when': 'Testing luck vs skill',
            'limitation': 'Does not quantify uncertainty in metric'
        })
        
        # Standard bootstrap
        std_boot_ci = self._standard_bootstrap()
        results.append({
            'method': 'Standard Bootstrap',
            'tests_for': 'Sampling uncertainty',
            'percentile_rank': 'N/A',
            'ci_width': f'{std_boot_ci[1] - std_boot_ci[0]:.2f}',
            'appropriate_when': 'Independent observations',
            'limitation': f'Ignores autocorrelation ({self.autocorr:.2f})'
        })
        
        # Block bootstrap
        block_boot_ci = self._block_bootstrap()
        results.append({
            'method': 'Block Bootstrap',
            'tests_for': 'Sampling uncertainty',
            'percentile_rank': 'N/A',
            'ci_width': f'{block_boot_ci[1] - block_boot_ci[0]:.2f}',
            'appropriate_when': 'Autocorrelated data',
            'limitation': 'Requires choosing block size'
        })
        
        return pd.DataFrame(results)
    
    def _trade_randomization(self) -> float:
        """Run trade randomization MC"""
        original_sharpe = self._sharpe(self.returns)
        
        simulated_sharpes = []
        for _ in range(10000):
            shuffled = np.random.permutation(self.returns)
            simulated_sharpes.append(self._sharpe(shuffled))
        
        percentile = stats.percentileofscore(simulated_sharpes, original_sharpe) / 100
        return percentile
    
    def _standard_bootstrap(self) -> Tuple[float, float]:
        """Run standard bootstrap"""
        simulated_sharpes = []
        for _ in range(10000):
            resampled = np.random.choice(self.returns, size=self.n, replace=True)
            simulated_sharpes.append(self._sharpe(resampled))
        
        ci_lower = np.percentile(simulated_sharpes, 2.5)
        ci_upper = np.percentile(simulated_sharpes, 97.5)
        return ci_lower, ci_upper
    
    def _block_bootstrap(self, block_size: int = 20) -> Tuple[float, float]:
        """Run block bootstrap"""
        simulated_sharpes = []
        
        for _ in range(10000):
            resampled = self._block_resample(block_size)
            simulated_sharpes.append(self._sharpe(resampled))
        
        ci_lower = np.percentile(simulated_sharpes, 2.5)
        ci_upper = np.percentile(simulated_sharpes, 97.5)
        return ci_lower, ci_upper
    
    def _block_resample(self, block_size: int) -> np.ndarray:
        """Block bootstrap resampling"""
        num_blocks = int(np.ceil(self.n / block_size))
        resampled = []
        
        for _ in range(num_blocks):
            start_idx = np.random.randint(0, self.n - block_size + 1)
            block = self.returns[start_idx:start_idx + block_size]
            resampled.append(block)
        
        return np.concatenate(resampled)[:self.n]
    
    def _sharpe(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio"""
        if returns.std() == 0:
            return 0
        return (returns.mean() / returns.std()) * np.sqrt(252)
\`\`\`

#### Conclusion

When Monte Carlo methods diverge:
1. **Investigate why**: Usually due to autocorrelation or distributional issues
2. **Choose appropriate method**: Match method to data characteristics
3. **Report transparently**: Explain methodology and limitations
4. **Use multiple methods**: Triangulate to understand strategy better
5. **Trust the right method**: Block bootstrap for autocorrelated data

The divergence in this case actually provides valuable information about the strategy's momentum characteristics and validates using the correct methodology.

---

## Question 3: Monte Carlo for Options Strategies

**Scenario**: Standard Monte Carlo methods work well for directional equity strategies, but options strategies present unique challenges:
- Non-linear payoffs
- Path-dependent options (barriers, lookbacks)
- Volatility surface dynamics
- Greeks sensitivity
- Discrete events (earnings, expiry)

Design a Monte Carlo framework specifically for validating options trading strategies. Address how you would:
1. Simulate realistic options prices (not just underlying)
2. Handle discrete events and volatility clustering
3. Test Greeks-based strategies (delta hedging, gamma scalping)
4. Measure strategy robustness across volatility regimes
5. Stress test for extreme scenarios (crashes, volatility spikes)

### Comprehensive Answer

Options require specialized Monte Carlo methods due to their non-linear payoffs and sensitivity to multiple risk factors (price, volatility, time decay).

#### Architecture for Options Monte Carlo

\`\`\`python
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from enum import Enum

class VolatilityRegime(Enum):
    """Volatility regimes"""
    LOW = "low"  # VIX < 15
    NORMAL = "normal"  # VIX 15-25
    HIGH = "high"  # VIX 25-40
    CRISIS = "crisis"  # VIX > 40

@dataclass
class OptionsPosition:
    """Options position representation"""
    option_type: str  # 'call' or 'put'
    strike: float
    expiry_days: int
    quantity: int
    entry_price: float
    underlying_at_entry: float
    iv_at_entry: float

class OptionsMonteCarloSimulator:
    """
    Monte Carlo simulator for options trading strategies
    """
    
    def __init__(
        self,
        base_vol: float = 0.20,
        vol_of_vol: float = 0.50,
        mean_reversion_speed: float = 2.0,
        jump_intensity: float = 0.05,
        jump_mean: float = -0.02,
        jump_std: float = 0.05
    ):
        """
        Initialize options MC simulator
        
        Uses stochastic volatility with jumps (SVJ model)
        
        Args:
            base_vol: Long-term volatility level
            vol_of_vol: Volatility of volatility
            mean_reversion_speed: Speed of vol mean reversion
            jump_intensity: Probability of jump per day
            jump_mean: Mean jump size
            jump_std: Jump size standard deviation
        """
        self.base_vol = base_vol
        self.vol_of_vol = vol_of_vol
        self.mean_reversion_speed = mean_reversion_speed
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std
    
    def simulate_underlying_path(
        self,
        S0: float,
        days: int,
        num_paths: int = 1000,
        regime: VolatilityRegime = VolatilityRegime.NORMAL
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate underlying price paths with stochastic volatility
        
        Args:
            S0: Initial price
            days: Number of days to simulate
            num_paths: Number of Monte Carlo paths
            regime: Volatility regime
            
        Returns:
            Tuple of (price_paths, volatility_paths)
        """
        dt = 1/252  # Daily steps
        
        # Adjust parameters by regime
        regime_multipliers = {
            VolatilityRegime.LOW: 0.6,
            VolatilityRegime.NORMAL: 1.0,
            VolatilityRegime.HIGH: 1.5,
            VolatilityRegime.CRISIS: 2.5
        }
        
        vol_mult = regime_multipliers[regime]
        effective_vol = self.base_vol * vol_mult
        
        # Initialize arrays
        prices = np.zeros((num_paths, days + 1))
        vols = np.zeros((num_paths, days + 1))
        
        prices[:, 0] = S0
        vols[:, 0] = effective_vol
        
        # Simulate paths
        for t in range(days):
            # Correlated Brownian motions for price and vol
            dW1 = np.random.randn(num_paths) * np.sqrt(dt)
            dW2 = 0.5 * dW1 + np.sqrt(0.75) * np.random.randn(num_paths) * np.sqrt(dt)
            
            # Jump component
            jumps = np.random.poisson(self.jump_intensity * dt, num_paths)
            jump_sizes = np.where(
                jumps > 0,
                np.random.normal(self.jump_mean, self.jump_std, num_paths),
                0
            )
            
            # Price evolution (with jumps)
            drift = 0.0  # Assume risk-neutral
            diffusion = vols[:, t] * dW1
            prices[:, t + 1] = prices[:, t] * np.exp(
                drift * dt - 0.5 * vols[:, t]**2 * dt + 
                diffusion +
                jump_sizes
            )
            
            # Volatility evolution (mean-reverting)
            vol_drift = self.mean_reversion_speed * (effective_vol - vols[:, t]) * dt
            vol_diffusion = self.vol_of_vol * vols[:, t] * dW2
            vols[:, t + 1] = np.maximum(vols[:, t] + vol_drift + vol_diffusion, 0.05)
        
        return prices, vols
    
    def black_scholes_price(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        option_type: str = 'call'
    ) -> float:
        """Black-Scholes option pricing"""
        if T <= 0:
            if option_type == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        d1 = (np.log(S / K) + (0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * stats.norm.cdf(d1) - K * stats.norm.cdf(d2)
        else:
            price = K * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
        
        return price
    
    def calculate_greeks(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        option_type: str = 'call'
    ) -> Dict[str, float]:
        """Calculate option Greeks"""
        if T <= 0:
            return {
                'delta': 1.0 if (option_type == 'call' and S > K) else 0.0,
                'gamma': 0.0,
                'vega': 0.0,
                'theta': 0.0
            }
        
        d1 = (np.log(S / K) + (0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            delta = stats.norm.cdf(d1)
        else:
            delta = stats.norm.cdf(d1) - 1
        
        gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * stats.norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% vol change
        
        # Theta (per day)
        if option_type == 'call':
            theta = (-S * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T)) / 365)
        else:
            theta = (-S * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + 
                    K * stats.norm.cdf(-d2)) / 365
        
        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta
        }
    
    def simulate_delta_hedging_strategy(
        self,
        position: OptionsPosition,
        price_paths: np.ndarray,
        vol_paths: np.ndarray,
        rehedge_frequency: int = 1  # Days between rehedges
    ) -> Dict[str, np.ndarray]:
        """
        Simulate delta-hedged options position
        
        Args:
            position: Options position to hedge
            price_paths: Simulated underlying price paths
            vol_paths: Simulated volatility paths
            rehedge_frequency: Days between delta adjustments
            
        Returns:
            Dictionary with hedging results
        """
        num_paths, num_days = price_paths.shape
        
        # Initialize tracking arrays
        option_values = np.zeros((num_paths, num_days))
        hedge_positions = np.zeros((num_paths, num_days))  # Delta hedge (shares)
        hedging_pnl = np.zeros((num_paths, num_days))
        total_pnl = np.zeros((num_paths, num_days))
        
        # Initial position
        option_values[:, 0] = position.entry_price * position.quantity
        
        for t in range(1, num_days):
            days_to_expiry = max(position.expiry_days - t, 0)
            T = days_to_expiry / 252
            
            # Calculate current option value and delta
            for path_idx in range(num_paths):
                S = price_paths[path_idx, t]
                sigma = vol_paths[path_idx, t]
                
                option_values[path_idx, t] = self.black_scholes_price(
                    S, position.strike, T, sigma, position.option_type
                ) * position.quantity
                
                # Rehedge if it's time
                if t % rehedge_frequency == 0:
                    greeks = self.calculate_greeks(
                        S, position.strike, T, sigma, position.option_type
                    )
                    hedge_positions[path_idx, t] = -greeks['delta'] * position.quantity
                else:
                    hedge_positions[path_idx, t] = hedge_positions[path_idx, t-1]
                
                # Calculate hedging P&L
                if t > 0:
                    price_change = price_paths[path_idx, t] - price_paths[path_idx, t-1]
                    hedging_pnl[path_idx, t] = (
                        hedging_pnl[path_idx, t-1] +
                        hedge_positions[path_idx, t-1] * price_change
                    )
                
                # Total P&L
                option_pnl = option_values[path_idx, t] - position.entry_price * position.quantity
                total_pnl[path_idx, t] = option_pnl + hedging_pnl[path_idx, t]
        
        return {
            'option_values': option_values,
            'hedge_positions': hedge_positions,
            'hedging_pnl': hedging_pnl,
            'total_pnl': total_pnl,
            'final_pnl_distribution': total_pnl[:, -1]
        }


# Example: Monte Carlo for delta hedging
if __name__ == "__main__":
    # Initialize simulator
    sim = OptionsMonteCarloSimulator()
    
    # Simulate underlying paths
    S0 = 100
    days = 30
    num_paths = 1000
    
    price_paths, vol_paths = sim.simulate_underlying_path(
        S0, days, num_paths, regime=VolatilityRegime.NORMAL
    )
    
    # Create options position
    position = OptionsPosition(
        option_type='call',
        strike=100,
        expiry_days=30,
        quantity=10,
        entry_price=5.0,
        underlying_at_entry=100,
        iv_at_entry=0.20
    )
    
    # Simulate delta hedging
    results = sim.simulate_delta_hedging_strategy(
        position, price_paths, vol_paths, rehedge_frequency=1
    )
    
    # Analyze results
    final_pnl = results['final_pnl_distribution']
    
    print(f"Delta Hedging Monte Carlo Results:")
    print(f"Mean P&L: \${final_pnl.mean():.2f}")
    print(f"Std P&L: \${final_pnl.std():.2f}")
    print(f"5th percentile (VaR): \${np.percentile(final_pnl, 5):.2f}")
    print(f"95th percentile: \${np.percentile(final_pnl, 95):.2f}")
\`\`\`

This specialized framework handles the unique challenges of options strategies through stochastic volatility modeling, proper Greeks calculation, and realistic hedging simulation.

The key differences from equity Monte Carlo:
1. **Non-linear payoffs** require full option pricing
2. **Multiple risk factors** (price + vol) need joint simulation
3. **Path dependency** requires step-by-step evaluation
4. **Greeks** drive hedging decisions
5. **Regime awareness** critical for volatility-based strategies

This enables proper validation of complex options strategies before live deployment.
`,
    },
  ],
};

export default monteCarloSimulationDiscussion;
