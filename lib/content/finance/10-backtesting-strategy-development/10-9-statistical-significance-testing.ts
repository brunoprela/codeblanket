import { Content } from '@/lib/types';

const statisticalSignificanceTesting: Content = {
  title: 'Statistical Significance Testing',
  description:
    'Master hypothesis testing for trading strategies, p-values, confidence intervals, permutation tests, and multiple testing corrections for robust statistical validation',
  sections: [
    {
      title: 'Hypothesis Testing for Trading Strategies',
      content: `
# Hypothesis Testing for Trading Strategies

Statistical significance testing answers the critical question: "Is my strategy's performance due to genuine edge or just luck?"

## The Fundamental Question

**Case Study**: A trader develops a strategy showing 15% annual returns over 3 years. Is this:
- A genuine trading edge worth $millions?
- Random luck that will disappear?
- Something in between?

Statistical hypothesis testing provides a rigorous framework to answer this question.

### The Null Hypothesis

**H₀ (Null Hypothesis)**: The strategy has no edge—observed returns are due to random chance
**H₁ (Alternative Hypothesis)**: The strategy has genuine edge

The goal is to reject H₀ with high confidence.

## Understanding P-Values

A p-value is the probability of observing results at least as extreme as what we saw, assuming the null hypothesis is true.

**Example**: 
- Strategy Sharpe ratio: 1.5
- P-value: 0.03

**Interpretation**: If the strategy actually had no edge, there's only a 3% chance we'd see a Sharpe ratio of 1.5 or better purely by luck. This suggests (but doesn't prove) genuine edge.

## Statistical Testing Framework

\`\`\`python
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
import warnings

@dataclass
class HypothesisTestResult:
    """Results from hypothesis test"""
    test_name: str
    test_statistic: float
    p_value: float
    is_significant: bool
    confidence_level: float
    interpretation: str
    
class StatisticalSignificanceTester:
    """
    Statistical significance testing for trading strategies
    """
    
    def __init__(
        self,
        returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None,
        confidence_level: float = 0.95
    ):
        """
        Initialize tester
        
        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns (optional)
            confidence_level: Confidence level (default 95%)
        """
        self.returns = returns
        self.benchmark_returns = benchmark_returns
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
        # Calculate basic statistics
        self.n = len(returns)
        self.mean_return = np.mean(returns)
        self.std_return = np.std(returns, ddof=1)
        self.sharpe = (self.mean_return / self.std_return) * np.sqrt(252) if self.std_return > 0 else 0
    
    def test_mean_return(
        self,
        null_mean: float = 0.0
    ) -> HypothesisTestResult:
        """
        Test if mean return is significantly different from null value
        
        H0: mean_return = null_mean
        H1: mean_return ≠ null_mean
        
        Args:
            null_mean: Null hypothesis mean (typically 0)
            
        Returns:
            HypothesisTestResult
        """
        # T-test for mean
        t_statistic = (self.mean_return - null_mean) / (self.std_return / np.sqrt(self.n))
        
        # Two-tailed p-value
        p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), self.n - 1))
        
        is_significant = p_value < self.alpha
        
        interpretation = (
            f"Mean return ({self.mean_return:.4f}) is "
            f"{'significantly' if is_significant else 'not significantly'} "
            f"different from {null_mean} (p={p_value:.4f})"
        )
        
        return HypothesisTestResult(
            test_name="t-test for mean return",
            test_statistic=t_statistic,
            p_value=p_value,
            is_significant=is_significant,
            confidence_level=self.confidence_level,
            interpretation=interpretation
        )
    
    def test_sharpe_ratio(
        self,
        null_sharpe: float = 0.0
    ) -> HypothesisTestResult:
        """
        Test if Sharpe ratio is significantly different from null value
        
        Uses Jobson-Korkie test for Sharpe ratio
        
        Args:
            null_sharpe: Null hypothesis Sharpe ratio
            
        Returns:
            HypothesisTestResult
        """
        # Sharpe ratio standard error
        # SE(SR) ≈ sqrt((1 + SR²/2) / T)
        sharpe_se = np.sqrt((1 + self.sharpe**2 / 2) / self.n)
        
        # Z-test
        z_statistic = (self.sharpe - null_sharpe) / sharpe_se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_statistic)))
        
        is_significant = p_value < self.alpha
        
        interpretation = (
            f"Sharpe ratio ({self.sharpe:.3f}) is "
            f"{'significantly' if is_significant else 'not significantly'} "
            f"different from {null_sharpe} (p={p_value:.4f})"
        )
        
        return HypothesisTestResult(
            test_name="Sharpe ratio significance test",
            test_statistic=z_statistic,
            p_value=p_value,
            is_significant=is_significant,
            confidence_level=self.confidence_level,
            interpretation=interpretation
        )
    
    def test_vs_benchmark(self) -> HypothesisTestResult:
        """
        Test if strategy outperforms benchmark
        
        H0: strategy_return = benchmark_return
        H1: strategy_return > benchmark_return
        
        Returns:
            HypothesisTestResult
        """
        if self.benchmark_returns is None:
            raise ValueError("Benchmark returns required for comparison test")
        
        # Paired t-test (strategies compared on same time periods)
        excess_returns = self.returns - self.benchmark_returns
        
        t_statistic, p_value = stats.ttest_1samp(excess_returns, 0)
        
        # One-tailed test (we care if strategy > benchmark)
        p_value = p_value / 2 if t_statistic > 0 else 1 - p_value / 2
        
        is_significant = p_value < self.alpha
        
        mean_excess = np.mean(excess_returns)
        
        interpretation = (
            f"Strategy {'significantly outperforms' if is_significant else 'does not significantly outperform'} "
            f"benchmark by {mean_excess:.4f} per period (p={p_value:.4f})"
        )
        
        return HypothesisTestResult(
            test_name="Strategy vs Benchmark t-test",
            test_statistic=t_statistic,
            p_value=p_value,
            is_significant=is_significant,
            confidence_level=self.confidence_level,
            interpretation=interpretation
        )
    
    def confidence_interval_sharpe(self) -> Tuple[float, float]:
        """
        Calculate confidence interval for Sharpe ratio
        
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        # Standard error of Sharpe ratio
        sharpe_se = np.sqrt((1 + self.sharpe**2 / 2) / self.n)
        
        # Z-score for confidence level
        z_score = stats.norm.ppf(1 - self.alpha / 2)
        
        lower = self.sharpe - z_score * sharpe_se
        upper = self.sharpe + z_score * sharpe_se
        
        return (lower, upper)
    
    def confidence_interval_mean(self) -> Tuple[float, float]:
        """
        Calculate confidence interval for mean return
        
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        # T-distribution for small samples
        t_score = stats.t.ppf(1 - self.alpha / 2, self.n - 1)
        
        se = self.std_return / np.sqrt(self.n)
        
        lower = self.mean_return - t_score * se
        upper = self.mean_return + t_score * se
        
        return (lower, upper)
    
    def permutation_test(
        self,
        n_permutations: int = 10000,
        test_statistic: str = 'sharpe'
    ) -> HypothesisTestResult:
        """
        Non-parametric permutation test
        
        Tests if observed performance could have occurred by chance by
        randomly shuffling returns and recalculating statistic
        
        Args:
            n_permutations: Number of random permutations
            test_statistic: Statistic to test ('sharpe', 'mean', 'sortino')
            
        Returns:
            HypothesisTestResult
        """
        # Calculate observed statistic
        if test_statistic == 'sharpe':
            observed = self.sharpe
        elif test_statistic == 'mean':
            observed = self.mean_return
        elif test_statistic == 'sortino':
            downside_std = np.std(self.returns[self.returns < 0], ddof=1)
            observed = (self.mean_return / downside_std) * np.sqrt(252) if downside_std > 0 else 0
        else:
            raise ValueError(f"Unknown test statistic: {test_statistic}")
        
        # Generate null distribution
        null_distribution = np.zeros(n_permutations)
        
        for i in range(n_permutations):
            # Randomly shuffle returns
            shuffled = np.random.permutation(self.returns)
            
            # Calculate statistic
            if test_statistic == 'sharpe':
                stat = (shuffled.mean() / shuffled.std()) * np.sqrt(252) if shuffled.std() > 0 else 0
            elif test_statistic == 'mean':
                stat = shuffled.mean()
            elif test_statistic == 'sortino':
                downside = shuffled[shuffled < 0]
                downside_std = np.std(downside, ddof=1) if len(downside) > 0 else 1
                stat = (shuffled.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else 0
            
            null_distribution[i] = stat
        
        # Calculate p-value (proportion of permutations >= observed)
        p_value = (null_distribution >= observed).mean()
        
        is_significant = p_value < self.alpha
        
        interpretation = (
            f"Permutation test: {test_statistic}={observed:.3f} is "
            f"{'significantly' if is_significant else 'not significantly'} "
            f"better than random (p={p_value:.4f}, {n_permutations} permutations)"
        )
        
        return HypothesisTestResult(
            test_name=f"Permutation test ({test_statistic})",
            test_statistic=observed,
            p_value=p_value,
            is_significant=is_significant,
            confidence_level=self.confidence_level,
            interpretation=interpretation
        )
    
    def comprehensive_report(self) -> Dict[str, HypothesisTestResult]:
        """
        Generate comprehensive statistical significance report
        
        Returns:
            Dictionary of all test results
        """
        results = {}
        
        # Test 1: Mean return
        results['mean_return'] = self.test_mean_return()
        
        # Test 2: Sharpe ratio
        results['sharpe_ratio'] = self.test_sharpe_ratio()
        
        # Test 3: Benchmark comparison (if available)
        if self.benchmark_returns is not None:
            results['vs_benchmark'] = self.test_vs_benchmark()
        
        # Test 4: Permutation test
        results['permutation'] = self.permutation_test(n_permutations=10000)
        
        return results
    
    def print_report(self):
        """Print comprehensive report"""
        print("\\n" + "="*80)
        print("STATISTICAL SIGNIFICANCE REPORT")
        print("="*80)
        print(f"Sample size: {self.n} observations")
        print(f"Mean return: {self.mean_return:.4f}")
        print(f"Std return: {self.std_return:.4f}")
        print(f"Sharpe ratio: {self.sharpe:.3f}")
        print("")
        
        # Confidence intervals
        ci_mean = self.confidence_interval_mean()
        ci_sharpe = self.confidence_interval_sharpe()
        
        print(f"Confidence Intervals ({self.confidence_level:.0%}):")
        print(f"  Mean return: [{ci_mean[0]:.4f}, {ci_mean[1]:.4f}]")
        print(f"  Sharpe ratio: [{ci_sharpe[0]:.3f}, {ci_sharpe[1]:.3f}]")
        print("")
        
        # Run all tests
        results = self.comprehensive_report()
        
        print("Hypothesis Tests:")
        for test_name, result in results.items():
            status = "✓ SIGNIFICANT" if result.is_significant else "✗ NOT SIGNIFICANT"
            print(f"\\n  {result.test_name}:")
            print(f"    {status} (p={result.p_value:.4f})")
            print(f"    {result.interpretation}")
        
        # Overall assessment
        significant_count = sum(1 for r in results.values() if r.is_significant)
        total_tests = len(results)
        
        print("\\n" + "="*80)
        print(f"Overall: {significant_count}/{total_tests} tests show statistical significance")
        
        if significant_count == total_tests:
            print("✓ STRONG EVIDENCE of genuine trading edge")
        elif significant_count >= total_tests / 2:
            print("⚠ MODERATE EVIDENCE of trading edge")
        else:
            print("✗ INSUFFICIENT EVIDENCE of trading edge")
        print("="*80 + "\\n")


class MultipleTestingCorrection:
    """
    Correct for multiple hypothesis testing
    
    When testing many strategies, some will appear significant by chance
    """
    
    def __init__(self, p_values: List[float], method: str = 'bonferroni'):
        """
        Args:
            p_values: List of p-values from multiple tests
            method: Correction method ('bonferroni', 'holm', 'fdr')
        """
        self.p_values = np.array(p_values)
        self.n_tests = len(p_values)
        self.method = method
    
    def bonferroni_correction(self, alpha: float = 0.05) -> np.ndarray:
        """
        Bonferroni correction (most conservative)
        
        Adjusted α = α / n_tests
        
        Args:
            alpha: Family-wise error rate
            
        Returns:
            Array of boolean (True if significant after correction)
        """
        adjusted_alpha = alpha / self.n_tests
        return self.p_values < adjusted_alpha
    
    def holm_correction(self, alpha: float = 0.05) -> np.ndarray:
        """
        Holm-Bonferroni correction (less conservative than Bonferroni)
        
        Args:
            alpha: Family-wise error rate
            
        Returns:
            Array of boolean (True if significant after correction)
        """
        # Sort p-values
        sorted_indices = np.argsort(self.p_values)
        sorted_pvalues = self.p_values[sorted_indices]
        
        # Holm procedure
        significant = np.zeros(self.n_tests, dtype=bool)
        
        for i, (idx, p) in enumerate(zip(sorted_indices, sorted_pvalues)):
            adjusted_alpha = alpha / (self.n_tests - i)
            if p < adjusted_alpha:
                significant[idx] = True
            else:
                # Once we fail to reject, all subsequent tests fail
                break
        
        return significant
    
    def fdr_correction(self, alpha: float = 0.05) -> np.ndarray:
        """
        False Discovery Rate (FDR) - Benjamini-Hochberg procedure
        
        Controls expected proportion of false discoveries
        Less conservative than Bonferroni, appropriate for exploratory analysis
        
        Args:
            alpha: False discovery rate
            
        Returns:
            Array of boolean (True if significant after correction)
        """
        # Sort p-values
        sorted_indices = np.argsort(self.p_values)
        sorted_pvalues = self.p_values[sorted_indices]
        
        # Benjamini-Hochberg procedure
        significant = np.zeros(self.n_tests, dtype=bool)
        
        # Find largest i where p(i) <= (i/m) * α
        for i in range(self.n_tests - 1, -1, -1):
            threshold = (i + 1) / self.n_tests * alpha
            if sorted_pvalues[i] <= threshold:
                # Reject all hypotheses from 1 to i
                significant[sorted_indices[:i+1]] = True
                break
        
        return significant
    
    def apply_correction(self, alpha: float = 0.05) -> Dict[str, np.ndarray]:
        """
        Apply selected correction method
        
        Returns:
            Dictionary with correction results
        """
        if self.method == 'bonferroni':
            significant = self.bonferroni_correction(alpha)
            adjusted_alpha = alpha / self.n_tests
        elif self.method == 'holm':
            significant = self.holm_correction(alpha)
            adjusted_alpha = alpha / self.n_tests  # First test
        elif self.method == 'fdr':
            significant = self.fdr_correction(alpha)
            adjusted_alpha = alpha  # FDR doesn't have single adjusted alpha
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return {
            'significant': significant,
            'n_significant': significant.sum(),
            'n_tests': self.n_tests,
            'method': self.method,
            'original_alpha': alpha,
            'adjusted_alpha': adjusted_alpha,
            'false_positive_rate': (self.p_values < alpha).sum() / self.n_tests
        }


# Example usage
if __name__ == "__main__":
    # Simulate strategy returns
    np.random.seed(42)
    n_days = 252
    
    # Strategy with small positive edge
    returns = np.random.normal(0.001, 0.02, n_days)  # 0.1% daily mean, 2% std
    
    # Benchmark returns
    benchmark_returns = np.random.normal(0.0005, 0.015, n_days)
    
    # Initialize tester
    tester = StatisticalSignificanceTester(
        returns=returns,
        benchmark_returns=benchmark_returns,
        confidence_level=0.95
    )
    
    # Generate report
    tester.print_report()
    
    # Example: Multiple testing correction
    print("\\n" + "="*80)
    print("MULTIPLE TESTING CORRECTION EXAMPLE")
    print("="*80)
    
    # Simulate testing 100 strategies
    p_values = np.random.uniform(0, 1, 100)
    # Add a few genuinely significant results
    p_values[:5] = [0.001, 0.005, 0.01, 0.02, 0.03]
    
    corrector = MultipleTestingCorrection(p_values, method='bonferroni')
    results = corrector.apply_correction(alpha=0.05)
    
    print(f"Tested {results['n_tests']} strategies")
    print(f"Uncorrected: {(p_values < 0.05).sum()} significant (α=0.05)")
    print(f"After {results['method']} correction: {results['n_significant']} significant")
    print(f"Adjusted α: {results['adjusted_alpha']:.6f}")
\`\`\`

## Key Concepts

### 1. **Type I and Type II Errors**
- **Type I Error (α)**: False positive—rejecting true null hypothesis
- **Type II Error (β)**: False negative—failing to reject false null hypothesis
- **Power (1-β)**: Probability of detecting real effect

### 2. **P-Value Misinterpretations**
Common mistakes:
- ❌ P-value is probability strategy has no edge
- ❌ P-value is probability of replicating result
- ✓ P-value is probability of observing data this extreme if null is true

### 3. **Multiple Testing Problem**
Test 100 strategies at α=0.05 → expect 5 false positives
Solution: Bonferroni, Holm, or FDR correction

### 4. **Statistical vs Economic Significance**
- Statistical significance: Results unlikely due to chance
- Economic significance: Profits after costs make strategy worthwhile
- Need both!

## Production Best Practices

1. **Always report p-values** alongside performance metrics
2. **Use multiple tests** (parametric and non-parametric)
3. **Correct for multiple testing** when screening strategies
4. **Report confidence intervals** not just point estimates
5. **Consider economic significance** not just statistical

Statistical significance is necessary but not sufficient—still need out-of-sample validation, transaction cost modeling, and live testing.
`,
    },
  ],
};

export default statisticalSignificanceTesting;
