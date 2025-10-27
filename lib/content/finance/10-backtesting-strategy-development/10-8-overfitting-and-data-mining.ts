import { Content } from '@/lib/types';

const overfittingAndDataMining: Content = {
  title: 'Overfitting and Data Mining',
  description:
    'Master overfitting detection, multiple testing correction, in-sample vs out-of-sample analysis, and techniques to build robust trading strategies that generalize to unseen data',
  sections: [
    {
      title: 'Understanding Overfitting in Trading',
      content: `
# Understanding Overfitting in Trading

Overfitting is one of the most insidious problems in quantitative trading—strategies that look brilliant in backtests but fail miserably in live trading.

## The Overfitting Crisis

**Case Study**: Long-Term Capital Management (LTCM), staffed with Nobel Prize winners, had models showing consistent profits based on historical data. The models failed catastrophically in 1998, losing $4.6 billion in months. Post-mortem analysis revealed their models were overfit to specific historical patterns that didn't persist.

**Modern Example**: A prop trading firm tested 10,000 technical indicator combinations and found one that showed a 95% win rate over 10 years of data. They deployed it with $5 million. Within 3 months, they had lost $2 million. The "perfect" combination was pure noise.

### What is Overfitting?

Overfitting occurs when a strategy learns noise instead of signal—it memorizes historical data patterns that won't repeat rather than discovering genuine market inefficiencies.

**Mathematical View**:
- **Signal**: True underlying relationship (persistent edge)
- **Noise**: Random fluctuations (doesn't persist)
- **Overfit Model**: Signal + Noise (appears better in-sample, worse out-of-sample)
- **Robust Model**: Mostly Signal (similar performance in/out of sample)

## The Bias-Variance Tradeoff

\`\`\`python
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
import warnings

class OverfittingDetector:
    """
    Comprehensive overfitting detection framework
    """
    
    def __init__(
        self,
        strategy_func: Callable,
        data: pd.DataFrame,
        param_ranges: Dict[str, List]
    ):
        """
        Initialize overfitting detector
        
        Args:
            strategy_func: Strategy function taking (data, params)
            data: Historical data
            param_ranges: Dictionary of parameter values to test
        """
        self.strategy_func = strategy_func
        self.data = data
        self.param_ranges = param_ranges
        
        # Calculate total combinations
        self.total_combinations = np.prod([len(v) for v in param_ranges.values()])
    
    def calculate_degrees_of_freedom(self) -> Dict[str, float]:
        """
        Calculate effective degrees of freedom
        
        More parameters = more degrees of freedom = higher overfitting risk
        
        Returns:
            Dictionary with DOF analysis
        """
        n_observations = len(self.data)
        n_parameters = len(self.param_ranges)
        n_combinations_tested = self.total_combinations
        
        # Observations per parameter
        obs_per_param = n_observations / n_parameters
        
        # Multiple testing penalty
        # After trying N combinations, best result is biased upward
        expected_max_bias = np.sqrt(2 * np.log(n_combinations_tested))
        
        return {
            'n_observations': n_observations,
            'n_parameters': n_parameters,
            'n_combinations': n_combinations_tested,
            'obs_per_param': obs_per_param,
            'expected_selection_bias': expected_max_bias,
            'overfitting_risk': self._assess_risk(obs_per_param, n_combinations_tested),
            'recommendation': self._generate_recommendation(obs_per_param, n_combinations_tested)
        }
    
    def _assess_risk(self, obs_per_param: float, n_combinations: int) -> str:
        """Assess overfitting risk level"""
        if obs_per_param < 30:
            return "CRITICAL"
        elif obs_per_param < 100:
            return "HIGH"
        elif n_combinations > 1000:
            return "ELEVATED (multiple testing)"
        else:
            return "MODERATE"
    
    def _generate_recommendation(self, obs_per_param: float, n_combinations: int) -> str:
        """Generate recommendations"""
        recommendations = []
        
        if obs_per_param < 30:
            recommendations.append("Get more data or reduce parameters")
        
        if n_combinations > 100:
            recommendations.append(
                f"Apply multiple testing correction (tested {n_combinations} combinations)"
            )
        
        if obs_per_param < 100:
            recommendations.append("Use cross-validation and out-of-sample testing")
        
        return "; ".join(recommendations) if recommendations else "Risk acceptable with proper validation"
    
    def test_parameter_sensitivity(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        optimal_params: Dict[str, float],
        perturbation_pct: float = 0.10
    ) -> Dict[str, float]:
        """
        Test how sensitive strategy is to parameter changes
        
        A robust strategy should have smooth performance surface.
        Overfit strategy has sharp peaks (only works with exact parameters).
        
        Args:
            train_data: Training data
            test_data: Testing data
            optimal_params: Optimal parameters from training
            perturbation_pct: Perturbation percentage
            
        Returns:
            Sensitivity metrics
        """
        # Calculate performance with optimal parameters
        optimal_train_perf = self._evaluate_strategy(train_data, optimal_params)
        optimal_test_perf = self._evaluate_strategy(test_data, optimal_params)
        
        # Test perturbed parameters
        perturbed_performances = {'train': [], 'test': []}
        
        for param_name, optimal_value in optimal_params.items():
            for perturbation in [-perturbation_pct, perturbation_pct]:
                perturbed_params = optimal_params.copy()
                perturbed_params[param_name] = optimal_value * (1 + perturbation)
                
                train_perf = self._evaluate_strategy(train_data, perturbed_params)
                test_perf = self._evaluate_strategy(test_data, perturbed_params)
                
                perturbed_performances['train'].append(train_perf)
                perturbed_performances['test'].append(test_perf)
        
        # Calculate sensitivity metrics
        train_perf_std = np.std(perturbed_performances['train'])
        test_perf_std = np.std(perturbed_performances['test'])
        
        # Coefficient of variation (lower is better)
        train_cv = train_perf_std / abs(optimal_train_perf) if optimal_train_perf != 0 else np.inf
        test_cv = test_perf_std / abs(optimal_test_perf) if optimal_test_perf != 0 else np.inf
        
        # Sharp peak indicator (high CV = overfit)
        is_overfit = train_cv > 0.3  # More than 30% variation with 10% param change
        
        return {
            'optimal_train_performance': optimal_train_perf,
            'optimal_test_performance': optimal_test_perf,
            'train_sensitivity_cv': train_cv,
            'test_sensitivity_cv': test_cv,
            'is_likely_overfit': is_overfit,
            'robustness_score': max(0, 100 - train_cv * 100),  # 0-100 scale
            'interpretation': self._interpret_sensitivity(train_cv, test_cv)
        }
    
    def _interpret_sensitivity(self, train_cv: float, test_cv: float) -> str:
        """Interpret sensitivity results"""
        if train_cv > 0.5:
            return "HIGHLY OVERFIT: Performance extremely sensitive to parameters"
        elif train_cv > 0.3:
            return "LIKELY OVERFIT: Performance very sensitive to parameters"
        elif train_cv > 0.15:
            return "MODERATE SENSITIVITY: Acceptable but monitor closely"
        else:
            return "ROBUST: Performance stable across parameter variations"
    
    def _evaluate_strategy(
        self,
        data: pd.DataFrame,
        params: Dict[str, float]
    ) -> float:
        """Evaluate strategy with given parameters"""
        try:
            returns = self.strategy_func(data, params)
            
            if len(returns) == 0:
                return 0.0
            
            # Calculate Sharpe ratio
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0.0
            return sharpe
        
        except Exception as e:
            warnings.warn(f"Strategy evaluation failed: {e}")
            return 0.0
    
    def in_sample_out_sample_comparison(
        self,
        train_test_splits: List[Tuple[pd.DataFrame, pd.DataFrame]],
        params: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compare in-sample vs out-of-sample performance
        
        Large degradation indicates overfitting
        
        Args:
            train_test_splits: List of (train, test) data splits
            params: Parameters to test
            
        Returns:
            Comparison metrics
        """
        is_performances = []
        oos_performances = []
        
        for train, test in train_test_splits:
            is_perf = self._evaluate_strategy(train, params)
            oos_perf = self._evaluate_strategy(test, params)
            
            is_performances.append(is_perf)
            oos_performances.append(oos_perf)
        
        is_mean = np.mean(is_performances)
        oos_mean = np.mean(oos_performances)
        
        # Degradation percentage
        degradation = ((is_mean - oos_mean) / is_mean * 100) if is_mean != 0 else 0
        
        # Statistical test: are IS and OOS significantly different?
        if len(is_performances) > 1:
            t_stat, p_value = stats.ttest_rel(is_performances, oos_performances)
        else:
            t_stat, p_value = 0, 1
        
        return {
            'in_sample_mean': is_mean,
            'out_sample_mean': oos_mean,
            'degradation_pct': degradation,
            'is_oos_correlation': np.corrcoef(is_performances, oos_performances)[0, 1],
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significantly_different': p_value < 0.05,
            'overfitting_assessment': self._assess_degradation(degradation)
        }
    
    def _assess_degradation(self, degradation_pct: float) -> str:
        """Assess level of overfitting from degradation"""
        if degradation_pct < 10:
            return "MINIMAL overfitting - performance holds up well"
        elif degradation_pct < 25:
            return "MODERATE overfitting - some degradation expected"
        elif degradation_pct < 50:
            return "SIGNIFICANT overfitting - major performance drop"
        else:
            return "SEVERE overfitting - strategy likely curve-fit to noise"
    
    def multiple_testing_correction(
        self,
        n_tests: int,
        significance_level: float = 0.05
    ) -> Dict[str, float]:
        """
        Calculate corrected significance levels for multiple testing
        
        When testing many strategies, some will appear significant by chance
        
        Args:
            n_tests: Number of independent tests performed
            significance_level: Desired family-wise error rate
            
        Returns:
            Corrected thresholds
        """
        # Bonferroni correction (conservative)
        bonferroni = significance_level / n_tests
        
        # Holm-Bonferroni (less conservative)
        holm_bonferroni = significance_level / n_tests  # First test
        
        # False Discovery Rate (FDR) - Benjamini-Hochberg
        # More appropriate for exploratory analysis
        fdr_threshold = significance_level
        
        # Probability of at least one false positive
        prob_false_positive = 1 - (1 - significance_level) ** n_tests
        
        return {
            'original_threshold': significance_level,
            'bonferroni_threshold': bonferroni,
            'holm_bonferroni_threshold': holm_bonferroni,
            'fdr_threshold': fdr_threshold,
            'prob_false_positive': prob_false_positive,
            'recommended_threshold': bonferroni,  # Use most conservative
            'interpretation': (
                f"With {n_tests} tests at α={significance_level}, "
                f"expect {n_tests * significance_level:.1f} false positives by chance. "
                f"Use Bonferroni-corrected α={bonferroni:.6f} to control family-wise error."
            )
        }


class ComplexityPenalizer:
    """
    Penalize model complexity to prevent overfitting
    
    Implements AIC, BIC, and other information criteria
    """
    
    def __init__(self, n_observations: int):
        self.n = n_observations
    
    def akaike_information_criterion(
        self,
        log_likelihood: float,
        n_parameters: int
    ) -> float:
        """
        Calculate AIC (Akaike Information Criterion)
        
        AIC = 2k - 2ln(L)
        Lower is better - trades off fit vs complexity
        
        Args:
            log_likelihood: Log likelihood of model
            n_parameters: Number of parameters
            
        Returns:
            AIC score
        """
        aic = 2 * n_parameters - 2 * log_likelihood
        return aic
    
    def bayesian_information_criterion(
        self,
        log_likelihood: float,
        n_parameters: int
    ) -> float:
        """
        Calculate BIC (Bayesian Information Criterion)
        
        BIC = k*ln(n) - 2ln(L)
        Penalizes complexity more heavily than AIC
        
        Args:
            log_likelihood: Log likelihood of model
            n_parameters: Number of parameters
            
        Returns:
            BIC score
        """
        bic = n_parameters * np.log(self.n) - 2 * log_likelihood
        return bic
    
    def sharpe_ratio_with_complexity_penalty(
        self,
        sharpe_ratio: float,
        n_parameters: int,
        n_trades: int,
        penalty_factor: float = 1.0
    ) -> float:
        """
        Adjust Sharpe ratio for complexity
        
        More parameters = higher penalty
        Fewer trades = higher penalty
        
        Args:
            sharpe_ratio: Observed Sharpe ratio
            n_parameters: Number of strategy parameters
            n_trades: Number of trades
            penalty_factor: Strength of penalty (1.0 = moderate)
            
        Returns:
            Complexity-adjusted Sharpe ratio
        """
        # Penalty increases with parameters and decreases with trades
        complexity_penalty = penalty_factor * np.sqrt(n_parameters / n_trades)
        
        adjusted_sharpe = sharpe_ratio - complexity_penalty
        
        return adjusted_sharpe
    
    def deflated_sharpe_ratio(
        self,
        sharpe_ratio: float,
        n_trials: int,
        sharpe_std: float = 1.0
    ) -> float:
        """
        Deflated Sharpe Ratio (Bailey & López de Prado, 2014)
        
        Accounts for multiple testing and parameter selection
        
        Args:
            sharpe_ratio: Observed Sharpe ratio
            n_trials: Number of parameter combinations tried
            sharpe_std: Standard deviation of Sharpe ratios across trials
            
        Returns:
            Deflated Sharpe ratio
        """
        # Expected maximum Sharpe from N trials (random)
        expected_max = np.sqrt(2 * np.log(n_trials))
        
        # Z-score adjusted for multiple testing
        z_deflated = (sharpe_ratio - expected_max * sharpe_std) / sharpe_std
        
        # Convert back to Sharpe-like metric
        deflated_sharpe = z_deflated * sharpe_std
        
        return deflated_sharpe


# Example usage
if __name__ == "__main__":
    # Simulate a simple moving average strategy
    def sma_strategy(data: pd.DataFrame, params: Dict[str, float]) -> np.ndarray:
        """Simple moving average crossover"""
        fast_period = int(params['fast_period'])
        slow_period = int(params['slow_period'])
        
        fast_ma = data['close'].rolling(fast_period).mean()
        slow_ma = data['slow_period'].rolling(slow_period).mean()
        
        position = np.where(fast_ma > slow_ma, 1, -1)
        returns = data['close'].pct_change() * np.roll(position, 1)
        
        return returns[slow_period:]
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(len(dates)) * 2),
    }, index=dates)
    
    # Parameter ranges to test
    param_ranges = {
        'fast_period': list(range(5, 51, 5)),
        'slow_period': list(range(20, 201, 10))
    }
    
    # Initialize detector
    detector = OverfittingDetector(sma_strategy, data, param_ranges)
    
    # Calculate degrees of freedom
    dof_analysis = detector.calculate_degrees_of_freedom()
    
    print("\\n" + "="*80)
    print("OVERFITTING RISK ANALYSIS")
    print("="*80)
    print(f"Observations: {dof_analysis['n_observations']}")
    print(f"Parameters: {dof_analysis['n_parameters']}")
    print(f"Combinations tested: {dof_analysis['n_combinations']}")
    print(f"Obs per parameter: {dof_analysis['obs_per_param']:.1f}")
    print(f"Expected selection bias: {dof_analysis['expected_selection_bias']:.2f}")
    print(f"Risk level: {dof_analysis['overfitting_risk']}")
    print(f"Recommendation: {dof_analysis['recommendation']}")
    
    # Multiple testing correction
    correction = detector.multiple_testing_correction(
        n_tests=int(dof_analysis['n_combinations'])
    )
    
    print("\\n" + "="*80)
    print("MULTIPLE TESTING CORRECTION")
    print("="*80)
    print(f"Original α: {correction['original_threshold']:.4f}")
    print(f"Bonferroni-corrected α: {correction['bonferroni_threshold']:.6f}")
    print(f"Prob(≥1 false positive): {correction['prob_false_positive']:.1%}")
    print(f"\\n{correction['interpretation']}")
\`\`\`

## Key Concepts

### 1. **Bias-Variance Tradeoff**
- **Bias**: Error from wrong assumptions (underfitting)
- **Variance**: Error from sensitivity to training data (overfitting)
- **Goal**: Balance both

### 2. **Model Complexity**
- More parameters = more flexibility = higher overfitting risk
- Rule of thumb: Need 30+ observations per parameter

### 3. **Multiple Testing Problem**
- Test 100 strategies at α=0.05 → expect 5 to appear significant by chance
- Solution: Bonferroni correction or FDR control

### 4. **In-Sample vs Out-of-Sample**
- In-sample: Data used for optimization
- Out-of-sample: Held-out data for validation
- Degradation > 30% suggests severe overfitting

## Common Overfitting Patterns

1. **Too many parameters** relative to data
2. **Data mining**: Testing hundreds of strategies
3. **Survivorship bias**: Only trading current index constituents
4. **Look-ahead bias**: Using future information
5. **Parameter tuning**: Optimizing until results look good

## Production Best Practices

1. **Reserve out-of-sample data** before any analysis
2. **Apply multiple testing corrections** when trying many variants
3. **Use cross-validation** for parameter selection
4. **Simplicity preference**: Simpler models often generalize better
5. **Economic rationale**: Strategy should have logical explanation
`,
    },
    {
      title: 'Detection and Prevention Techniques',
      content: `
# Detection and Prevention Techniques

## Cross-Validation for Time Series

Standard cross-validation doesn't work for time series due to temporal dependencies. We need specialized techniques.

\`\`\`python
from sklearn.model_selection import TimeSeriesSplit
from typing import List, Tuple
import numpy as np
import pandas as pd

class TimeSeriesCrossValidator:
    """
    Time series cross-validation with purging and embargo
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        purge_pct: float = 0.01,  # Purge 1% of training data before test
        embargo_pct: float = 0.01  # Embargo 1% of data after test
    ):
        """
        Args:
            n_splits: Number of CV splits
            test_size: Size of test set (if None, determined automatically)
            purge_pct: Fraction of training data to purge before test
            embargo_pct: Fraction of data to embargo after test
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.purge_pct = purge_pct
        self.embargo_pct = embargo_pct
    
    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices with purging and embargo
        
        Args:
            X: Feature data
            y: Target data (optional)
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        n_samples = len(X)
        
        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size
        
        purge_size = int(n_samples * self.purge_pct)
        embargo_size = int(n_samples * self.embargo_pct)
        
        splits = []
        
        for i in range(self.n_splits):
            # Test set
            test_end = n_samples - (self.n_splits - i - 1) * test_size
            test_start = test_end - test_size
            
            # Training set (everything before test, minus purge)
            train_end = test_start - purge_size
            train_start = 0
            
            # Adjust for embargo from previous test set
            if i > 0:
                prev_test_end = test_start
                train_end = min(train_end, prev_test_end - embargo_size)
            
            # Create indices
            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)
            
            splits.append((train_indices, test_indices))
        
        return splits
    
    def combinatorial_purged_cv(
        self,
        X: pd.DataFrame,
        n_paths: int = 10,
        min_train_length: int = 100
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Combinatorial Purged Cross-Validation (CPCV)
        
        Creates multiple train/test combinations while respecting time order
        
        Args:
            X: Feature data
            n_paths: Number of CV paths to generate
            min_train_length: Minimum training set size
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        n_samples = len(X)
        splits = []
        
        # Generate random test set placements
        np.random.seed(42)
        
        for _ in range(n_paths):
            # Random test set size (20-30% of data)
            test_size = np.random.randint(
                int(n_samples * 0.2),
                int(n_samples * 0.3)
            )
            
            # Random test set position (not at very start or end)
            test_start = np.random.randint(
                min_train_length,
                n_samples - test_size
            )
            test_end = test_start + test_size
            
            # Purge around test set
            purge_size = int(test_size * self.purge_pct)
            
            # Training indices (before test, with purge)
            train_before = np.arange(0, test_start - purge_size)
            
            # Training indices (after test, with embargo)
            embargo_size = int(test_size * self.embargo_pct)
            if test_end + embargo_size < n_samples:
                train_after = np.arange(test_end + embargo_size, n_samples)
                train_indices = np.concatenate([train_before, train_after])
            else:
                train_indices = train_before
            
            test_indices = np.arange(test_start, test_end)
            
            if len(train_indices) >= min_train_length:
                splits.append((train_indices, test_indices))
        
        return splits


class OverfittingPreventor:
    """
    Techniques to prevent overfitting during strategy development
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.out_of_sample_data = None
        self.lockout_date = None
    
    def lock_out_sample_data(
        self,
        out_sample_fraction: float = 0.2,
        lock_date: Optional[pd.Timestamp] = None
    ):
        """
        Lock away out-of-sample data before any analysis
        
        This data should NOT be touched until final validation
        
        Args:
            out_sample_fraction: Fraction of data to reserve
            lock_date: Specific date to split on (if None, use fraction)
        """
        if lock_date is not None:
            split_date = lock_date
        else:
            n_samples = len(self.data)
            split_idx = int(n_samples * (1 - out_sample_fraction))
            split_date = self.data.index[split_idx]
        
        self.out_of_sample_data = self.data[self.data.index >= split_date].copy()
        self.data = self.data[self.data.index < split_date].copy()
        self.lockout_date = split_date
        
        print(f"\\n{'='*80}")
        print(f"OUT-OF-SAMPLE DATA LOCKED")
        print(f"{'='*80}")
        print(f"Lockout date: {split_date}")
        print(f"Training data: {len(self.data)} observations")
        print(f"Locked OOS data: {len(self.out_of_sample_data)} observations")
        print(f"\\n⚠️  WARNING: OOS data should NOT be accessed until final validation!")
        print(f"{'='*80}\\n")
    
    def simplicity_score(
        self,
        n_parameters: int,
        n_rules: int,
        max_parameters: int = 5,
        max_rules: int = 3
    ) -> float:
        """
        Calculate simplicity score for strategy
        
        Simpler strategies generalize better
        
        Args:
            n_parameters: Number of tunable parameters
            n_rules: Number of trading rules
            max_parameters: Maximum acceptable parameters
            max_rules: Maximum acceptable rules
            
        Returns:
            Simplicity score (0-100, higher is better)
        """
        param_score = max(0, 100 - (n_parameters / max_parameters) * 100)
        rule_score = max(0, 100 - (n_rules / max_rules) * 100)
        
        # Weighted average (parameters matter more)
        simplicity = 0.6 * param_score + 0.4 * rule_score
        
        return simplicity
    
    def apply_regularization(
        self,
        performance_metric: float,
        n_parameters: int,
        n_trades: int,
        regularization_strength: float = 1.0
    ) -> float:
        """
        Apply regularization penalty to performance
        
        Similar to Ridge/Lasso in ML - penalize complexity
        
        Args:
            performance_metric: Raw performance (e.g., Sharpe ratio)
            n_parameters: Number of parameters
            n_trades: Number of trades
            regularization_strength: Strength of penalty
            
        Returns:
            Regularized performance
        """
        # Penalty for parameters
        param_penalty = regularization_strength * np.sqrt(n_parameters)
        
        # Penalty for few trades (less reliable)
        trade_penalty = regularization_strength * np.sqrt(100 / max(n_trades, 10))
        
        # Total penalty
        total_penalty = param_penalty + trade_penalty
        
        regularized_performance = performance_metric - total_penalty
        
        return regularized_performance
    
    def economic_rationale_checklist(
        self,
        strategy_description: str
    ) -> Dict[str, bool]:
        """
        Checklist for economic rationale
        
        Strategies should have logical explanations, not just data mining
        
        Args:
            strategy_description: Description of strategy logic
            
        Returns:
            Checklist results
        """
        # This would be manual in practice, but we can structure it
        checklist = {
            'has_clear_hypothesis': False,  # What market inefficiency does it exploit?
            'has_causal_explanation': False,  # Why should this work?
            'backed_by_research': False,  # Is there academic/practitioner research?
            'makes_intuitive_sense': False,  # Would an experienced trader understand it?
            'not_data_mined': False,  # Was it designed with logic, not found through brute force?
            'robust_to_regime_changes': False,  # Should work across market conditions?
        }
        
        # In production, this would be filled out manually by researchers
        # For now, return template
        return checklist


# Example: Comprehensive overfitting prevention pipeline
def overfitting_prevention_pipeline(
    data: pd.DataFrame,
    strategy_func: Callable,
    param_ranges: Dict[str, List],
    n_cv_splits: int = 5
) -> Dict[str, Any]:
    """
    Complete pipeline for overfitting prevention
    
    Args:
        data: Historical data
        strategy_func: Strategy function
        param_ranges: Parameter ranges to test
        n_cv_splits: Number of CV splits
        
    Returns:
        Results with overfitting diagnostics
    """
    # Step 1: Lock out-of-sample data
    preventor = OverfittingPreventor(data)
    preventor.lock_out_sample_data(out_sample_fraction=0.2)
    
    # Step 2: Cross-validation on training data
    cv = TimeSeriesCrossValidator(n_splits=n_cv_splits)
    
    results = {
        'cv_results': [],
        'best_params': None,
        'best_score': -np.inf,
        'oos_score': None
    }
    
    # Step 3: Test parameters with CV
    for params in _generate_param_combinations(param_ranges):
        cv_scores = []
        
        for train_idx, test_idx in cv.split(preventor.data):
            train_data = preventor.data.iloc[train_idx]
            test_data = preventor.data.iloc[test_idx]
            
            # Evaluate strategy
            returns = strategy_func(train_data, params)
            score = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
            
            cv_scores.append(score)
        
        avg_cv_score = np.mean(cv_scores)
        
        # Apply complexity penalty
        penalizer = ComplexityPenalizer(len(preventor.data))
        penalized_score = penalizer.sharpe_ratio_with_complexity_penalty(
            avg_cv_score,
            n_parameters=len(params),
            n_trades=len(returns) if len(returns) > 0 else 1
        )
        
        results['cv_results'].append({
            'params': params,
            'cv_score': avg_cv_score,
            'penalized_score': penalized_score
        })
        
        if penalized_score > results['best_score']:
            results['best_score'] = penalized_score
            results['best_params'] = params
    
    # Step 4: FINAL validation on locked OOS data (only once!)
    if results['best_params'] is not None:
        oos_returns = strategy_func(preventor.out_of_sample_data, results['best_params'])
        results['oos_score'] = (
            (oos_returns.mean() / oos_returns.std()) * np.sqrt(252)
            if oos_returns.std() > 0 else 0
        )
        
        # Calculate degradation
        degradation = (
            (results['best_score'] - results['oos_score']) / 
            results['best_score'] * 100
        ) if results['best_score'] != 0 else 0
        
        results['degradation_pct'] = degradation
        results['overfitting_assessment'] = (
            "SEVERE" if degradation > 50 else
            "SIGNIFICANT" if degradation > 25 else
            "MODERATE" if degradation > 10 else
            "MINIMAL"
        )
    
    return results


def _generate_param_combinations(param_ranges: Dict[str, List]) -> List[Dict]:
    """Generate all parameter combinations"""
    from itertools import product
    
    keys = param_ranges.keys()
    values = param_ranges.values()
    
    combinations = []
    for combo in product(*values):
        combinations.append(dict(zip(keys, combo)))
    
    return combinations


if __name__ == "__main__":
    # Example usage
    print("Overfitting prevention pipeline demonstration")
    print("See comprehensive framework above for implementation")
\`\`\`

## Summary

Overfitting is a critical risk in quantitative trading. Key prevention techniques:

1. **Lock out-of-sample data** before any analysis
2. **Use cross-validation** with proper time series techniques
3. **Apply complexity penalties** (AIC, BIC, regularization)
4. **Test parameter sensitivity** (robust strategies have smooth surfaces)
5. **Demand economic rationale** (don't just data mine)
6. **Correct for multiple testing** (Bonferroni, FDR)
7. **Monitor degradation** (IS vs OOS performance)

**Golden Rule**: If a strategy can't be explained logically, it's probably overfit.
`,
    },
  ],
};

export default overfittingAndDataMining;
