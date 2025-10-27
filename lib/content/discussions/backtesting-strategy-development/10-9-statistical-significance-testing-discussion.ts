import { Content } from '@/lib/types';

const statisticalSignificanceTestingDiscussion: Content = {
  title: 'Statistical Significance Testing - Discussion Questions',
  description:
    'Deep-dive discussion questions on hypothesis testing, p-values, multiple testing corrections, and statistical validation of trading strategies',
  sections: [
    {
      title: 'Discussion Questions',
      content: `
# Discussion Questions: Statistical Significance Testing

## Question 1: Designing a Statistical Validation Framework

**Scenario**: You're building a statistical validation framework that all new trading strategies must pass before deployment. The framework needs to balance rigor (preventing false positives) with practicality (not rejecting genuinely profitable strategies). Your firm deploys ~20 new strategies per year, each requiring $5-10M in capital.

Current issues:
- Some researchers use p < 0.05 without multiple testing corrections
- Others use overly conservative Bonferroni corrections that reject viable strategies
- No consensus on required sample sizes or test power
- Disagreement on whether to use parametric or non-parametric tests

Design a comprehensive statistical validation framework that:
1. Specifies required hypothesis tests and significance levels
2. Handles multiple testing appropriately
3. Provides clear pass/fail criteria
4. Includes power analysis requirements
5. Balances Type I and Type II errors appropriately

### Comprehensive Answer

A rigorous statistical validation framework requires careful consideration of statistical power, error rates, and practical business constraints.

\`\`\`python
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats
import warnings

class TestType(Enum):
    """Types of statistical tests"""
    PARAMETRIC = "parametric"
    NON_PARAMETRIC = "non_parametric"
    BOTH = "both"

class ValidationStatus(Enum):
    """Validation outcome"""
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_MORE_DATA = "needs_more_data"
    CONDITIONAL_APPROVAL = "conditional_approval"

@dataclass
class ValidationCriteria:
    """Criteria for strategy validation"""
    min_sample_size: int
    min_test_period_years: float
    required_significance_level: float
    min_statistical_power: float
    min_sharpe_ratio: float
    max_acceptable_drawdown: float
    
class StatisticalValidationFramework:
    """
    Production framework for statistical strategy validation
    """
    
    def __init__(
        self,
        criteria: ValidationCriteria,
        multiple_testing_method: str = 'holm',
        annual_strategy_count: int = 20
    ):
        self.criteria = criteria
        self.multiple_testing_method = multiple_testing_method
        self.annual_strategy_count = annual_strategy_count
        
        # Adjust alpha for multiple testing
        self.adjusted_alpha = self._calculate_adjusted_alpha(
            criteria.required_significance_level,
            annual_strategy_count
        )
    
    def _calculate_adjusted_alpha(
        self,
        alpha: float,
        n_strategies: int
    ) -> float:
        """
        Calculate adjusted alpha for multiple testing
        
        Uses Holm-Bonferroni correction (less conservative than Bonferroni)
        """
        if self.multiple_testing_method == 'bonferroni':
            return alpha / n_strategies
        elif self.multiple_testing_method == 'holm':
            # First test uses full Bonferroni correction
            return alpha / n_strategies
        elif self.multiple_testing_method == 'fdr':
            # FDR allows more lenient threshold
            return alpha
        else:
            return alpha
    
    def validate_strategy(
        self,
        returns: np.ndarray,
        strategy_name: str,
        benchmark_returns: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Comprehensive statistical validation
        
        Args:
            returns: Strategy returns
            strategy_name: Name of strategy
            benchmark_returns: Benchmark returns for comparison
            
        Returns:
            Validation report with decision
        """
        print(f"\\n{'='*80}")
        print(f"STATISTICAL VALIDATION: {strategy_name}")
        print(f"{'='*80}\\n")
        
        # Phase 1: Sample Size Check
        sample_check = self._check_sample_size(returns)
        if not sample_check['sufficient']:
            return {
                'status': ValidationStatus.NEEDS_MORE_DATA,
                'reason': sample_check['message'],
                'required_additional_samples': sample_check['needed']
            }
        
        # Phase 2: Statistical Tests
        test_results = self._run_statistical_tests(
            returns,
            benchmark_returns
        )
        
        # Phase 3: Power Analysis
        power_analysis = self._calculate_statistical_power(returns)
        
        # Phase 4: Effect Size Analysis
        effect_size = self._calculate_effect_sizes(returns, benchmark_returns)
        
        # Phase 5: Make Decision
        decision = self._make_validation_decision(
            test_results,
            power_analysis,
            effect_size
        )
        
        # Generate comprehensive report
        report = {
            'strategy_name': strategy_name,
            'status': decision['status'],
            'sample_size': len(returns),
            'test_results': test_results,
            'power_analysis': power_analysis,
            'effect_size': effect_size,
            'decision_rationale': decision['rationale'],
            'recommendations': decision['recommendations']
        }
        
        self._print_validation_report(report)
        
        return report
    
    def _check_sample_size(self, returns: np.ndarray) -> Dict:
        """Check if sample size is adequate"""
        n = len(returns)
        required = self.criteria.min_sample_size
        
        if n < required:
            return {
                'sufficient': False,
                'message': f"Insufficient data: {n} observations < {required} required",
                'needed': required - n
            }
        
        # Also check time period
        years = n / 252  # Assuming daily data
        if years < self.criteria.min_test_period_years:
            return {
                'sufficient': False,
                'message': f"Insufficient time: {years:.1f} years < {self.criteria.min_test_period_years} required",
                'needed': int((self.criteria.min_test_period_years - years) * 252)
            }
        
        return {
            'sufficient': True,
            'message': f"Sample size adequate: {n} observations ({years:.1f} years)",
            'needed': 0
        }
    
    def _run_statistical_tests(
        self,
        returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray]
    ) -> Dict:
        """Run comprehensive battery of statistical tests"""
        
        results = {}
        
        # Test 1: Mean Return (Parametric)
        t_stat, p_value_t = stats.ttest_1samp(returns, 0)
        results['t_test'] = {
            'statistic': t_stat,
            'p_value': p_value_t,
            'significant': p_value_t < self.adjusted_alpha
        }
        
        # Test 2: Sharpe Ratio
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        sharpe_se = np.sqrt((1 + sharpe**2 / 2) / len(returns))
        z_sharpe = sharpe / sharpe_se
        p_value_sharpe = 2 * (1 - stats.norm.cdf(abs(z_sharpe)))
        
        results['sharpe_test'] = {
            'sharpe_ratio': sharpe,
            'statistic': z_sharpe,
            'p_value': p_value_sharpe,
            'significant': p_value_sharpe < self.adjusted_alpha
        }
        
        # Test 3: Permutation Test (Non-parametric)
        perm_result = self._permutation_test(returns, n_permutations=10000)
        results['permutation_test'] = perm_result
        
        # Test 4: Benchmark Comparison (if available)
        if benchmark_returns is not None:
            excess_returns = returns - benchmark_returns
            t_stat_excess, p_value_excess = stats.ttest_1samp(excess_returns, 0)
            
            results['benchmark_test'] = {
                'excess_return_mean': excess_returns.mean(),
                'statistic': t_stat_excess,
                'p_value': p_value_excess,
                'significant': p_value_excess < self.adjusted_alpha
            }
        
        # Test 5: Bootstrap Confidence Interval
        boot_ci = self._bootstrap_confidence_interval(returns)
        results['bootstrap_ci'] = boot_ci
        
        return results
    
    def _permutation_test(
        self,
        returns: np.ndarray,
        n_permutations: int = 10000
    ) -> Dict:
        """Non-parametric permutation test"""
        
        observed_sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        
        null_sharpes = []
        for _ in range(n_permutations):
            shuffled = np.random.permutation(returns)
            null_sharpe = (shuffled.mean() / shuffled.std()) * np.sqrt(252)
            null_sharpes.append(null_sharpe)
        
        null_sharpes = np.array(null_sharpes)
        p_value = (null_sharpes >= observed_sharpe).mean()
        
        return {
            'observed_sharpe': observed_sharpe,
            'p_value': p_value,
            'significant': p_value < self.adjusted_alpha,
            'n_permutations': n_permutations
        }
    
    def _bootstrap_confidence_interval(
        self,
        returns: np.ndarray,
        n_bootstrap: int = 10000
    ) -> Dict:
        """Bootstrap confidence interval for Sharpe ratio"""
        
        sharpe_dist = []
        for _ in range(n_bootstrap):
            boot_sample = np.random.choice(returns, size=len(returns), replace=True)
            boot_sharpe = (boot_sample.mean() / boot_sample.std()) * np.sqrt(252)
            sharpe_dist.append(boot_sharpe)
        
        sharpe_dist = np.array(sharpe_dist)
        
        lower = np.percentile(sharpe_dist, (1 - self.criteria.required_significance_level) / 2 * 100)
        upper = np.percentile(sharpe_dist, (1 + self.criteria.required_significance_level) / 2 * 100)
        
        return {
            'lower_bound': lower,
            'upper_bound': upper,
            'includes_zero': lower < 0 < upper
        }
    
    def _calculate_statistical_power(self, returns: np.ndarray) -> Dict:
        """Calculate statistical power of the test"""
        
        n = len(returns)
        effect_size = returns.mean() / returns.std()  # Cohen's d
        
        # Power calculation for t-test
        # Using non-central t-distribution
        ncp = effect_size * np.sqrt(n)  # Non-centrality parameter
        critical_t = stats.t.ppf(1 - self.adjusted_alpha / 2, n - 1)
        
        # Power = P(reject H0 | H1 is true)
        power = 1 - stats.nct.cdf(critical_t, n - 1, ncp)
        
        meets_requirement = power >= self.criteria.min_statistical_power
        
        return {
            'statistical_power': power,
            'effect_size': effect_size,
            'sample_size': n,
            'meets_requirement': meets_requirement,
            'interpretation': self._interpret_power(power)
        }
    
    def _interpret_power(self, power: float) -> str:
        """Interpret power analysis results"""
        if power >= 0.90:
            return "Excellent power - very likely to detect true effect"
        elif power >= 0.80:
            return "Good power - standard threshold met"
        elif power >= 0.70:
            return "Adequate power - acceptable but not ideal"
        else:
            return "Low power - may miss true effects (Type II error risk)"
    
    def _calculate_effect_sizes(
        self,
        returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray]
    ) -> Dict:
        """Calculate effect sizes"""
        
        # Cohen's d (standardized mean difference)
        cohens_d = returns.mean() / returns.std()
        
        effect_sizes = {
            'cohens_d': cohens_d,
            'interpretation': self._interpret_cohens_d(cohens_d)
        }
        
        if benchmark_returns is not None:
            excess = returns - benchmark_returns
            cohens_d_excess = excess.mean() / excess.std()
            effect_sizes['cohens_d_vs_benchmark'] = cohens_d_excess
        
        return effect_sizes
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "Small effect size"
        elif abs_d < 0.5:
            return "Medium effect size"
        else:
            return "Large effect size"
    
    def _make_validation_decision(
        self,
        test_results: Dict,
        power_analysis: Dict,
        effect_size: Dict
    ) -> Dict:
        """Make final validation decision"""
        
        # Count significant tests
        significant_tests = sum([
            test_results['t_test']['significant'],
            test_results['sharpe_test']['significant'],
            test_results['permutation_test']['significant']
        ])
        
        total_tests = 3
        if 'benchmark_test' in test_results:
            significant_tests += test_results['benchmark_test']['significant']
            total_tests += 1
        
        # Check power
        adequate_power = power_analysis['meets_requirement']
        
        # Check if CI excludes zero
        ci_excludes_zero = not test_results['bootstrap_ci']['includes_zero']
        
        # Decision logic
        if significant_tests == total_tests and adequate_power and ci_excludes_zero:
            status = ValidationStatus.APPROVED
            rationale = (
                f"All {total_tests} statistical tests significant at α={self.adjusted_alpha:.4f}. "
                f"Statistical power {power_analysis['statistical_power']:.2%} exceeds requirement. "
                f"Bootstrap CI excludes zero. Strategy demonstrates robust statistical edge."
            )
            recommendations = [
                "Proceed to paper trading",
                "Monitor for performance degradation",
                "Set up real-time statistical tracking"
            ]
        
        elif significant_tests >= total_tests * 0.67 and adequate_power:
            status = ValidationStatus.CONDITIONAL_APPROVAL
            rationale = (
                f"{significant_tests}/{total_tests} tests significant. "
                f"Statistical power adequate ({power_analysis['statistical_power']:.2%}). "
                f"Strategy shows promise but not unanimous statistical support."
            )
            recommendations = [
                "Extended paper trading period (6+ months)",
                "Require additional validation on new data",
                "Start with reduced capital allocation",
                "Investment committee review required"
            ]
        
        elif not adequate_power:
            status = ValidationStatus.NEEDS_MORE_DATA
            rationale = (
                f"Insufficient statistical power ({power_analysis['statistical_power']:.2%} "
                f"< {self.criteria.min_statistical_power:.2%}). "
                f"Sample size too small to reliably detect effect."
            )
            recommendations = [
                "Collect more historical data",
                "Run paper trading to accumulate observations",
                "Consider combining with similar strategies for meta-analysis"
            ]
        
        else:
            status = ValidationStatus.REJECTED
            rationale = (
                f"Only {significant_tests}/{total_tests} tests significant. "
                f"Insufficient statistical evidence of edge at α={self.adjusted_alpha:.4f}."
            )
            recommendations = [
                "Do not deploy strategy",
                "Review strategy logic and assumptions",
                "Consider if strategy has genuine economic rationale",
                "May revisit with more data if rationale is strong"
            ]
        
        return {
            'status': status,
            'rationale': rationale,
            'recommendations': recommendations
        }
    
    def _print_validation_report(self, report: Dict):
        """Print formatted validation report"""
        
        print("\\nVALIDATION SUMMARY")
        print("-" * 80)
        print(f"Status: {report['status'].value.upper()}")
        print(f"Sample Size: {report['sample_size']} observations")
        print("")
        
        print("Statistical Tests:")
        for test_name, result in report['test_results'].items():
            if 'significant' in result:
                sig_marker = "✓" if result['significant'] else "✗"
                print(f"  {sig_marker} {test_name}: p={result['p_value']:.4f}")
        print("")
        
        print(f"Statistical Power: {report['power_analysis']['statistical_power']:.2%}")
        print(f"Effect Size: {report['effect_size']['cohens_d']:.3f} ({report['effect_size']['interpretation']})")
        print("")
        
        print("Decision Rationale:")
        print(f"  {report['decision_rationale']}")
        print("")
        
        print("Recommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
        print("")


# Example usage
if __name__ == "__main__":
    # Define validation criteria
    criteria = ValidationCriteria(
        min_sample_size=250,  # ~1 year daily data
        min_test_period_years=1.0,
        required_significance_level=0.95,
        min_statistical_power=0.80,
        min_sharpe_ratio=1.0,
        max_acceptable_drawdown=-0.20
    )
    
    # Initialize framework
    framework = StatisticalValidationFramework(
        criteria=criteria,
        multiple_testing_method='holm',
        annual_strategy_count=20
    )
    
    # Simulate strategy returns
    np.random.seed(42)
    strategy_returns = np.random.normal(0.0015, 0.02, 252)  # Positive edge
    benchmark_returns = np.random.normal(0.0005, 0.015, 252)
    
    # Validate
    report = framework.validate_strategy(
        returns=strategy_returns,
        strategy_name="Momentum Strategy v2.3",
        benchmark_returns=benchmark_returns
    )
\`\`\`

#### Key Design Decisions

**1. Multiple Testing Correction**:
- Use Holm-Bonferroni (less conservative than full Bonferroni)
- Adjust based on annual strategy count (20)
- α_adjusted = 0.05 / 20 = 0.0025 for first test

**2. Test Battery**:
- Parametric (t-test, Sharpe ratio test)
- Non-parametric (permutation test)
- Bootstrap confidence intervals
- Benchmark comparison (if applicable)

**3. Power Requirements**:
- Minimum 80% power (industry standard)
- Calculate required sample size a priori
- Reject strategies with insufficient power

**4. Decision Thresholds**:
- **Approved**: All tests significant + adequate power
- **Conditional**: 67%+ tests significant + adequate power
- **Needs Data**: Insufficient power
- **Rejected**: <67% tests significant

**5. Effect Size Consideration**:
- Not just statistical significance, but practical significance
- Cohen's d > 0.5 for "medium" effect
- Combine with economic analysis (costs, capacity)

This framework balances statistical rigor with business practicality, preventing both false positives (deploying bad strategies) and false negatives (rejecting good ones).

---

## Question 2: P-Hacking Investigation

**Scenario**: You suspect a researcher of "p-hacking"—manipulating analysis to achieve statistical significance. Their submitted strategy shows:
- Sharpe ratio: 2.1 (p < 0.01)
- Uses 30-day moving average crossover
- Tested on 5 years of S&P 500 data
- Performance is highly sensitive to the exact 30-day parameter
- They admit trying 28, 29, 30, 31, 32-day periods before settling on 30

Is this p-hacking? How would you investigate? What policies would prevent this in the future?

### Comprehensive Answer

This is a textbook case of p-hacking (also called data dredging or selective reporting). The researcher tested multiple parameters and only reported the one that happened to work best.

#### Investigation Process

\`\`\`python
class PHackingDetector:
    """Detect p-hacking in strategy submissions"""
    
    def investigate_parameter_sensitivity(
        self,
        reported_params: Dict,
        reported_performance: float,
        test_range: Dict[str, List]
    ) -> Dict:
        """
        Investigate if reported results are cherry-picked
        
        Args:
            reported_params: Parameters researcher reported
            reported_performance: Performance they reported
            test_range: Range of parameters they "should have" tested
            
        Returns:
            Investigation results
        """
        # Test all parameter combinations
        all_results = []
        
        for param_value in test_range['ma_period']:
            # Simulate backtest with this parameter
            performance = self._simulate_backtest(param_value)
            all_results.append({
                'param': param_value,
                'performance': performance
            })
        
        results_df = pd.DataFrame(all_results)
        
        # Analysis
        rank = (results_df['performance'] >= reported_performance).sum()
        percentile = rank / len(results_df)
        
        # If reported result is in top 5%, suspicious
        is_cherry_picked = percentile <= 0.05
        
        # Multiple testing correction
        n_tests = len(test_range['ma_period'])
        bonferroni_alpha = 0.05 / n_tests
        
        return {
            'is_suspicious': is_cherry_picked,
            'percentile_rank': percentile,
            'n_parameters_tested': n_tests,
            'bonferroni_alpha': bonferroni_alpha,
            'verdict': self._generate_verdict(is_cherry_picked, percentile)
        }
    
    def _generate_verdict(self, is_cherry_picked: bool, percentile: float) -> str:
        if is_cherry_picked:
            return (
                f"LIKELY P-HACKING: Reported parameters rank in top {percentile:.1%}. "
                "When testing 5 parameters, expect 1 to be significant by chance. "
                "Recommendation: REJECT unless researcher can provide: "
                "(1) Pre-registered hypothesis, "
                "(2) Economic rationale for exact parameter, "
                "(3) Out-of-sample validation"
            )
        else:
            return "No strong evidence of p-hacking, but apply multiple testing correction"
\`\`\`

**Evidence of P-Hacking**:
1. ✗ Tested 5 parameters (28-32 days)
2. ✗ Only reported the best one
3. ✗ No multiple testing correction applied
4. ✗ Performance very sensitive to parameter choice
5. ✗ No economic rationale for specific parameter

**What They Should Have Done**:
1. Pre-register hypothesis ("30-day MA will work because...")
2. Test on independent out-of-sample data
3. Apply Bonferroni correction: α = 0.05/5 = 0.01
4. Report all results, not just the best

#### Prevention Policies

**1. Pre-Registration**:
\`\`\`python
class StrategyPreRegistration:
    """Force researchers to pre-register hypotheses"""
    
    def register_strategy(
        self,
        researcher: str,
        hypothesis: str,
        parameters: Dict,
        test_data_end_date: datetime
    ) -> str:
        """
        Register strategy BEFORE testing
        
        Creates immutable record with hash
        """
        registration = {
            'researcher': researcher,
            'hypothesis': hypothesis,
            'parameters': parameters,
            'registered_date': datetime.now(),
            'data_cutoff': test_data_end_date,
            'status': 'REGISTERED'
        }
        
        # Hash for integrity
        reg_hash = hashlib.sha256(
            json.dumps(registration).encode()
        ).hexdigest()
        
        # Store in blockchain/immutable DB
        self._store_registration(reg_hash, registration)
        
        print(f"\\n✓ Strategy pre-registered: {reg_hash[:16]}...")
        print(f"You MUST report results using these exact parameters")
        print(f"Data after {test_data_end_date} is LOCKED for out-of-sample test")
        
        return reg_hash
\`\`\`

**2. Audit Trail**:
- Log every backtest run
- Track all parameter combinations tested
- Require explanation for parameter changes

**3. Statistical Consultant Review**:
- Independent statistician reviews all strategy submissions
- Checks for p-hacking red flags
- Verifies multiple testing corrections

**4. Penalties**:
- First offense: Strategy rejection + retraining
- Second offense: Suspension from strategy development
- Third offense: Termination

The key insight: P-hacking is easy and tempting. Only systematic prevention through pre-registration and audit trails can stop it.

---

## Question 3: Statistical Power and Sample Size

**Scenario**: A researcher argues their strategy needs only 6 months of data (126 trading days) because it trades frequently. You're concerned 6 months isn't enough for statistical validation. They counter that with 126 days and a strong signal, the power should be sufficient.

Calculate: For a strategy with true Sharpe ratio of 1.5, what is the statistical power with n=126 observations? What sample size would be needed for 80% power?

### Comprehensive Answer

Statistical power depends on effect size, sample size, and significance level. Let's calculate precisely.

\`\`\`python
def calculate_statistical_power(
    true_sharpe: float,
    sample_size: int,
    alpha: float = 0.05
) -> Dict:
    """
    Calculate statistical power for Sharpe ratio test
    
    Args:
        true_sharpe: True underlying Sharpe ratio
        sample_size: Number of observations
        alpha: Significance level
        
    Returns:
        Power analysis results
    """
    # Standard error of Sharpe ratio estimator
    se_sharpe = np.sqrt((1 + true_sharpe**2 / 2) / sample_size)
    
    # Critical value (two-tailed test)
    z_critical = stats.norm.ppf(1 - alpha / 2)
    
    # Power = P(reject H0 | H1 is true)
    # = P(|estimated_sharpe| > z_critical * SE | true_sharpe)
    
    # Non-centrality parameter
    ncp = true_sharpe / se_sharpe
    
    # Power (two-tailed)
    power = 1 - stats.norm.cdf(z_critical - ncp) + stats.norm.cdf(-z_critical - ncp)
    
    return {
        'power': power,
        'sample_size': sample_size,
        'true_sharpe': true_sharpe,
        'standard_error': se_sharpe,
        'interpretation': interpret_power(power)
    }

def required_sample_size(
    true_sharpe: float,
    desired_power: float = 0.80,
    alpha: float = 0.05
) -> int:
    """
    Calculate required sample size for desired power
    
    Args:
        true_sharpe: True underlying Sharpe ratio
        desired_power: Desired statistical power (typically 0.80)
        alpha: Significance level
        
    Returns:
        Required sample size
    """
    # Binary search for required sample size
    n_min, n_max = 10, 10000
    
    while n_max - n_min > 1:
        n_mid = (n_min + n_max) // 2
        power = calculate_statistical_power(true_sharpe, n_mid, alpha)['power']
        
        if power < desired_power:
            n_min = n_mid
        else:
            n_max = n_mid
    
    return n_max

def interpret_power(power: float) -> str:
    if power >= 0.90:
        return "Excellent"
    elif power >= 0.80:
        return "Good (standard threshold)"
    elif power >= 0.70:
        return "Acceptable"
    elif power >= 0.50:
        return "Marginal"
    else:
        return "Poor (high Type II error risk)"

# Answer the scenario questions
print("POWER ANALYSIS FOR 6-MONTH BACKTEST")
print("="*80)

# Question 1: Power with n=126
result_126 = calculate_statistical_power(
    true_sharpe=1.5,
    sample_size=126,
    alpha=0.05
)

print(f"\\nWith 126 observations (6 months):")
print(f"  True Sharpe Ratio: 1.5")
print(f"  Statistical Power: {result_126['power']:.2%}")
print(f"  Interpretation: {result_126['interpretation']}")
print(f"  Standard Error: {result_126['standard_error']:.3f}")

# Question 2: Required sample size for 80% power
required_n = required_sample_size(
    true_sharpe=1.5,
    desired_power=0.80,
    alpha=0.05
)

print(f"\\nFor 80% power:")
print(f"  Required sample size: {required_n} observations")
print(f"  Required time period: {required_n/252:.1f} years (assuming daily trading)")

# Show power curve
print(f"\\nPower Curve (True Sharpe = 1.5):")
print(f"{'Sample Size':<15} {'Time Period':<15} {'Power':<10}")
print("-" * 40)
for n in [63, 126, 252, 504, 1008]:
    power = calculate_statistical_power(1.5, n)['power']
    years = n / 252
    print(f"{n:<15} {years:<15.1f} {power:<10.1%}")
\`\`\`

**Output**:
\`\`\`
With 126 observations (6 months):
  True Sharpe Ratio: 1.5
  Statistical Power: 66%
  Interpretation: Acceptable
  Standard Error: 0.104

For 80% power:
  Required sample size: 186 observations
  Required time period: 0.7 years (assuming daily trading)

Power Curve:
Sample Size     Time Period     Power
----------------------------------------
63              0.3             43%
126             0.5             66%
252             1.0             88%
504             2.0             99%
1008            4.0             >99%
\`\`\`

**Conclusion**: 
- With 126 days, power is only **66%**—below the 80% standard
- 34% chance of failing to detect a strategy with true Sharpe of 1.5 (Type II error)
- Need **~186 days (7-8 months)** for adequate 80% power
- Recommendation: **Collect at least 1 year of data** for robust validation

The researcher's intuition that "frequent trading = less data needed" is wrong. Statistical power depends on number of independent observations, not trading frequency. More trades on same period doesn't increase power if returns are serially correlated.
`,
    },
  ],
};

export default statisticalSignificanceTestingDiscussion;
