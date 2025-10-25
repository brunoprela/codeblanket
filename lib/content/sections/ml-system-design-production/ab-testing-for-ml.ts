export const abTestingForML = {
  title: 'A/B Testing for ML',
  id: 'ab-testing-for-ml',
  content: `
# A/B Testing for ML

## Introduction

**"In God we trust. All others must bring data."** - W. Edwards Deming

You've built a new model version that looks better in offline metrics. But will it perform better in production? **A/B testing** is how you answer this question scientifically.

**Why A/B Test ML Models**:
- Offline metrics ≠ online performance
- User behavior may differ from training data
- Business impact matters more than accuracy
- Safe deployment with gradual rollout
- Compare multiple model versions

This section covers designing and running A/B tests for machine learning systems, with emphasis on statistical rigor and practical implementation.

### A/B Testing Process

\`\`\`
Design → Split Traffic → Collect Data → Analyze → Decision
   ↓          ↓             ↓            ↓          ↓
Hypothesis  Random      Metrics      Stats    Deploy/Rollback
 & Sample   Assignment   Logging    Significance
  Size
\`\`\`

By the end of this section, you'll understand:
- Experimental design for ML
- Traffic splitting strategies
- Statistical significance testing
- Multi-armed bandits for adaptive testing
- Practical A/B testing implementation

---

## Experimental Design

### Hypothesis and Metrics

\`\`\`python
"""
A/B Test Design Framework
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

class MetricType(Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    GUARDRAIL = "guardrail"

@dataclass
class Metric:
    """Metric definition"""
    name: str
    type: MetricType
    description: str
    direction: str  # 'increase' or 'decrease'
    minimum_detectable_effect: float  # MDE as percentage

@dataclass
class Experiment:
    """Experiment design"""
    name: str
    hypothesis: str
    variants: List[str]
    metrics: List[Metric]
    traffic_allocation: Dict[str, float]
    duration_days: int
    sample_size_per_variant: int


# Example: Trading model A/B test
trading_experiment = Experiment(
    name="new_model_v2_test",
    hypothesis="New XGBoost model (v2.0) will improve Sharpe ratio by 15% vs current RandomForest (v1.0)",
    variants=["control_v1", "treatment_v2"],
    metrics=[
        Metric(
            name="sharpe_ratio",
            type=MetricType.PRIMARY,
            description="Risk-adjusted returns",
            direction="increase",
            minimum_detectable_effect=0.15  # 15% increase
        ),
        Metric(
            name="max_drawdown",
            type=MetricType.GUARDRAIL,
            description="Maximum drawdown",
            direction="decrease",
            minimum_detectable_effect=0.10  # Don't worsen by >10%
        ),
        Metric(
            name="win_rate",
            type=MetricType.SECONDARY,
            description="Percentage of winning trades",
            direction="increase",
            minimum_detectable_effect=0.05  # 5% increase
        ),
        Metric(
            name="prediction_latency",
            type=MetricType.GUARDRAIL,
            description="Prediction latency (ms)",
            direction="decrease",
            minimum_detectable_effect=0.20  # Don't worsen by >20%
        )
    ],
    traffic_allocation={
        "control_v1": 0.5,
        "treatment_v2": 0.5
    },
    duration_days=30,
    sample_size_per_variant=10000
)

print(f"Experiment: {trading_experiment.name}")
print(f"Hypothesis: {trading_experiment.hypothesis}")
print(f"\\nPrimary Metric: {trading_experiment.metrics[0].name}")
print(f"MDE: {trading_experiment.metrics[0].minimum_detectable_effect*100}%")
\`\`\`

### Sample Size Calculation

\`\`\`python
"""
Calculate Required Sample Size
"""

import numpy as np
from scipy import stats
from typing import Tuple

def calculate_sample_size(
    baseline_mean: float,
    minimum_detectable_effect: float,
    baseline_std: float,
    alpha: float = 0.05,
    power: float = 0.8
) -> int:
    """
    Calculate required sample size per variant
    
    Args:
        baseline_mean: Current metric value
        minimum_detectable_effect: MDE as proportion (e.g., 0.15 for 15%)
        baseline_std: Standard deviation of metric
        alpha: Significance level (Type I error rate)
        power: Statistical power (1 - Type II error rate)
    
    Returns:
        Required sample size per variant
    """
    # Effect size (Cohen\'s d)
    effect_size = (baseline_mean * minimum_detectable_effect) / baseline_std
    
    # Z-scores for alpha and power
    z_alpha = stats.norm.ppf(1 - alpha/2)  # Two-tailed
    z_beta = stats.norm.ppf (power)
    
    # Sample size formula
    n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
    
    return int (np.ceil (n))


def calculate_experiment_duration(
    sample_size_per_variant: int,
    daily_traffic: int,
    num_variants: int = 2
) -> int:
    """
    Calculate experiment duration in days
    """
    total_sample_needed = sample_size_per_variant * num_variants
    days = total_sample_needed / daily_traffic
    
    return int (np.ceil (days))


# Example: Trading model test
baseline_sharpe = 1.5
mde = 0.15  # Want to detect 15% improvement
baseline_std = 0.3

sample_size = calculate_sample_size(
    baseline_mean=baseline_sharpe,
    minimum_detectable_effect=mde,
    baseline_std=baseline_std,
    alpha=0.05,
    power=0.8
)

print(f"\\n=== Sample Size Calculation ===")
print(f"Baseline Sharpe Ratio: {baseline_sharpe}")
print(f"MDE: {mde*100}% ({baseline_sharpe * mde:.2f})")
print(f"Baseline Std: {baseline_std}")
print(f"\\nRequired sample size per variant: {sample_size:,}")

# If we have 1000 trades per day
daily_trades = 1000
duration = calculate_experiment_duration (sample_size, daily_trades, num_variants=2)

print(f"\\nWith {daily_trades} trades/day:")
print(f"Experiment duration: {duration} days")
\`\`\`

---

## Traffic Splitting

### Random Assignment

\`\`\`python
"""
Traffic Splitting Implementation
"""

import hashlib
from typing import Dict, Optional
import random

class TrafficSplitter:
    """
    Consistent traffic splitting for A/B tests
    """
    
    def __init__(self, variants: Dict[str, float], salt: str = "experiment_1"):
        """
        Args:
            variants: Dict of variant_name -> traffic_percentage
            salt: Experiment salt for consistent hashing
        """
        self.variants = variants
        self.salt = salt
        
        # Validate percentages sum to 1.0
        total = sum (variants.values())
        if abs (total - 1.0) > 0.001:
            raise ValueError (f"Traffic percentages must sum to 1.0, got {total}")
    
    def assign_variant (self, user_id: str) -> str:
        """
        Consistently assign user to variant
        
        Uses consistent hashing to ensure:
        - Same user always gets same variant
        - Distribution matches specified percentages
        """
        # Hash user_id with salt
        hash_input = f"{self.salt}:{user_id}".encode()
        hash_value = int (hashlib.md5(hash_input).hexdigest(), 16)
        
        # Convert to 0-1 range
        bucket = (hash_value % 10000) / 10000
        
        # Assign to variant based on bucket
        cumulative = 0.0
        for variant, percentage in self.variants.items():
            cumulative += percentage
            if bucket < cumulative:
                return variant
        
        # Fallback (should never reach here)
        return list (self.variants.keys())[0]


# Example usage
splitter = TrafficSplitter(
    variants={
        "control": 0.5,
        "treatment": 0.5
    },
    salt="trading_model_test_2024"
)

# Simulate 10,000 users
assignments = {}
for i in range(10000):
    user_id = f"user_{i}"
    variant = splitter.assign_variant (user_id)
    assignments[variant] = assignments.get (variant, 0) + 1

print("\\n=== Traffic Split Results ===")
for variant, count in assignments.items():
    percentage = count / 10000 * 100
    print(f"{variant}: {count:,} ({percentage:.1f}%)")

# Verify consistency
user_id = "user_123"
variant1 = splitter.assign_variant (user_id)
variant2 = splitter.assign_variant (user_id)
print(f"\\nConsistency check for {user_id}: {variant1} == {variant2}: {variant1 == variant2}")
\`\`\`

### Gradual Rollout

\`\`\`python
"""
Gradual Rollout Strategy
"""

from datetime import datetime, timedelta

class GradualRollout:
    """
    Gradually increase treatment traffic
    """
    
    def __init__(
        self,
        start_treatment_pct: float = 0.05,
        end_treatment_pct: float = 0.5,
        duration_days: int = 7
    ):
        self.start_pct = start_treatment_pct
        self.end_pct = end_treatment_pct
        self.duration_days = duration_days
        self.start_date = None
    
    def start (self):
        """Start rollout"""
        self.start_date = datetime.now()
        print(f"✓ Rollout started: {self.start_pct*100}% → {self.end_pct*100}% over {self.duration_days} days")
    
    def get_current_allocation (self) -> Dict[str, float]:
        """
        Get current traffic allocation
        """
        if self.start_date is None:
            return {"control": 1.0, "treatment": 0.0}
        
        # Calculate days since start
        days_elapsed = (datetime.now() - self.start_date).days
        
        if days_elapsed >= self.duration_days:
            # Rollout complete
            treatment_pct = self.end_pct
        else:
            # Linear increase
            progress = days_elapsed / self.duration_days
            treatment_pct = self.start_pct + (self.end_pct - self.start_pct) * progress
        
        control_pct = 1.0 - treatment_pct
        
        return {
            "control": control_pct,
            "treatment": treatment_pct
        }


# Example
rollout = GradualRollout(
    start_treatment_pct=0.05,  # Start with 5%
    end_treatment_pct=0.5,     # Ramp to 50%
    duration_days=7
)

rollout.start()

# Simulate rollout over 7 days
print("\\n=== Gradual Rollout Schedule ===")
for day in range(8):
    rollout.start_date = datetime.now() - timedelta (days=7-day)
    allocation = rollout.get_current_allocation()
    print(f"Day {day}: Control={allocation['control']*100:.1f}%, Treatment={allocation['treatment']*100:.1f}%")
\`\`\`

---

## Metrics Collection

### Experiment Logger

\`\`\`python
"""
Log Experiment Data
"""

import pandas as pd
from typing import Dict, Any
import json

class ExperimentLogger:
    """
    Log experiment data for analysis
    """
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.data = []
    
    def log_event(
        self,
        user_id: str,
        variant: str,
        metrics: Dict[str, float],
        metadata: Dict[str, Any] = None
    ):
        """
        Log experiment event
        """
        event = {
            'timestamp': pd.Timestamp.now(),
            'experiment': self.experiment_name,
            'user_id': user_id,
            'variant': variant,
            **metrics,
            'metadata': json.dumps (metadata or {})
        }
        
        self.data.append (event)
    
    def get_dataframe (self) -> pd.DataFrame:
        """
        Get data as DataFrame
        """
        return pd.DataFrame (self.data)
    
    def save (self, path: str):
        """
        Save to file
        """
        df = self.get_dataframe()
        df.to_csv (path, index=False)
        print(f"✓ Saved {len (df)} events to {path}")


# Example usage
logger = ExperimentLogger("trading_model_test")

# Simulate experiment data
for i in range(1000):
    user_id = f"user_{i}"
    variant = "control" if i % 2 == 0 else "treatment"
    
    # Simulate metrics (treatment slightly better)
    if variant == "control":
        sharpe = np.random.normal(1.5, 0.3)
        win_rate = np.random.normal(0.55, 0.05)
    else:
        sharpe = np.random.normal(1.65, 0.3)  # 10% better
        win_rate = np.random.normal(0.57, 0.05)  # 3.6% better
    
    logger.log_event(
        user_id=user_id,
        variant=variant,
        metrics={
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'num_trades': np.random.randint(50, 200)
        },
        metadata={'model_version': variant}
    )

# Get data
df = logger.get_dataframe()
print(f"\\nLogged {len (df)} events")
print(f"\\nSample data:")
print(df[['variant', 'sharpe_ratio', 'win_rate']].head())
\`\`\`

---

## Statistical Analysis

### A/B Test Analysis

\`\`\`python
"""
Statistical Analysis of A/B Test
"""

from scipy import stats
import numpy as np
import pandas as pd
from typing import Dict, Tuple

class ABTestAnalyzer:
    """
    Analyze A/B test results
    """
    
    def __init__(self, data: pd.DataFrame, metric: str, alpha: float = 0.05):
        """
        Args:
            data: DataFrame with 'variant' and metric columns
            metric: Metric to analyze
            alpha: Significance level
        """
        self.data = data
        self.metric = metric
        self.alpha = alpha
    
    def split_variants (self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split data into control and treatment
        """
        control = self.data[self.data['variant'] == 'control'][self.metric].values
        treatment = self.data[self.data['variant'] == 'treatment'][self.metric].values
        
        return control, treatment
    
    def ttest (self) -> Dict:
        """
        Two-sample t-test
        """
        control, treatment = self.split_variants()
        
        # T-test
        statistic, pvalue = stats.ttest_ind (treatment, control)
        
        # Effect size (Cohen\'s d)
        pooled_std = np.sqrt(
            ((len (control) - 1) * np.var (control) + 
             (len (treatment) - 1) * np.var (treatment)) /
            (len (control) + len (treatment) - 2)
        )
        cohens_d = (np.mean (treatment) - np.mean (control)) / pooled_std
        
        # Confidence interval for difference
        se = pooled_std * np.sqrt(1/len (control) + 1/len (treatment))
        ci_lower = (np.mean (treatment) - np.mean (control)) - 1.96 * se
        ci_upper = (np.mean (treatment) - np.mean (control)) + 1.96 * se
        
        return {
            'test': 't-test',
            'control_mean': np.mean (control),
            'control_std': np.std (control),
            'treatment_mean': np.mean (treatment),
            'treatment_std': np.std (treatment),
            'difference': np.mean (treatment) - np.mean (control),
            'relative_difference_pct': (np.mean (treatment) - np.mean (control)) / np.mean (control) * 100,
            'statistic': statistic,
            'pvalue': pvalue,
            'significant': pvalue < self.alpha,
            'cohens_d': cohens_d,
            'ci_95': (ci_lower, ci_upper)
        }
    
    def mann_whitney (self) -> Dict:
        """
        Mann-Whitney U test (non-parametric)
        
        Use when data is not normally distributed
        """
        control, treatment = self.split_variants()
        
        statistic, pvalue = stats.mannwhitneyu(
            treatment, control,
            alternative='two-sided'
        )
        
        return {
            'test': 'Mann-Whitney U',
            'control_median': np.median (control),
            'treatment_median': np.median (treatment),
            'statistic': statistic,
            'pvalue': pvalue,
            'significant': pvalue < self.alpha
        }
    
    def bootstrap_confidence_interval(
        self,
        n_bootstrap: int = 10000,
        confidence: float = 0.95
    ) -> Dict:
        """
        Bootstrap confidence interval for difference
        """
        control, treatment = self.split_variants()
        
        # Bootstrap
        differences = []
        for _ in range (n_bootstrap):
            # Resample
            control_sample = np.random.choice (control, size=len (control), replace=True)
            treatment_sample = np.random.choice (treatment, size=len (treatment), replace=True)
            
            # Calculate difference
            diff = np.mean (treatment_sample) - np.mean (control_sample)
            differences.append (diff)
        
        # Confidence interval
        alpha = 1 - confidence
        ci_lower = np.percentile (differences, alpha/2 * 100)
        ci_upper = np.percentile (differences, (1 - alpha/2) * 100)
        
        return {
            'test': 'Bootstrap',
            'mean_difference': np.mean (differences),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'significant': not (ci_lower < 0 < ci_upper)  # CI doesn't contain 0
        }
    
    def analyze (self) -> Dict:
        """
        Run complete analysis
        """
        print(f"\\n{'='*60}")
        print(f"A/B Test Analysis: {self.metric}")
        print(f"{'='*60}")
        
        # Basic stats
        control, treatment = self.split_variants()
        
        print(f"\\nSample Sizes:")
        print(f"  Control: {len (control):,}")
        print(f"  Treatment: {len (treatment):,}")
        
        # T-test
        ttest_results = self.ttest()
        
        print(f"\\nMetric Values:")
        print(f"  Control: {ttest_results['control_mean']:.4f} ± {ttest_results['control_std']:.4f}")
        print(f"  Treatment: {ttest_results['treatment_mean']:.4f} ± {ttest_results['treatment_std']:.4f}")
        print(f"  Difference: {ttest_results['difference']:.4f} ({ttest_results['relative_difference_pct']:.2f}%)")
        
        print(f"\\nT-Test:")
        print(f"  p-value: {ttest_results['pvalue']:.4f}")
        print(f"  Significant: {ttest_results['significant']} (α={self.alpha})")
        print(f"  Effect size (Cohen\'s d): {ttest_results['cohens_d']:.4f}")
        print(f"  95% CI: [{ttest_results['ci_95'][0]:.4f}, {ttest_results['ci_95'][1]:.4f}]")
        
        # Bootstrap
        bootstrap_results = self.bootstrap_confidence_interval()
        
        print(f"\\nBootstrap (10,000 samples):")
        print(f"  Mean difference: {bootstrap_results['mean_difference']:.4f}")
        print(f"  95% CI: [{bootstrap_results['ci_lower']:.4f}, {bootstrap_results['ci_upper']:.4f}]")
        print(f"  Significant: {bootstrap_results['significant']}")
        
        # Decision
        if ttest_results['significant'] and ttest_results['difference'] > 0:
            decision = "✅ LAUNCH TREATMENT"
            reason = f"Treatment is significantly better (p={ttest_results['pvalue']:.4f})"
        elif ttest_results['significant'] and ttest_results['difference'] < 0:
            decision = "❌ KEEP CONTROL"
            reason = f"Treatment is significantly worse (p={ttest_results['pvalue']:.4f})"
        else:
            decision = "⏸️  NO SIGNIFICANT DIFFERENCE"
            reason = f"Not significant (p={ttest_results['pvalue']:.4f} > {self.alpha})"
        
        print(f"\\n{'='*60}")
        print(f"Decision: {decision}")
        print(f"Reason: {reason}")
        print(f"{'='*60}")
        
        return {
            'ttest': ttest_results,
            'bootstrap': bootstrap_results,
            'decision': decision,
            'reason': reason
        }


# Example analysis
analyzer = ABTestAnalyzer (df, metric='sharpe_ratio', alpha=0.05)
results = analyzer.analyze()
\`\`\`

### Multiple Testing Correction

\`\`\`python
"""
Correct for Multiple Testing
"""

from statsmodels.stats.multitest import multipletests

def bonferroni_correction (pvalues: list, alpha: float = 0.05) -> Dict:
    """
    Bonferroni correction for multiple testing
    
    Adjusted alpha = alpha / number_of_tests
    """
    n_tests = len (pvalues)
    adjusted_alpha = alpha / n_tests
    
    significant = [p < adjusted_alpha for p in pvalues]
    
    return {
        'method': 'Bonferroni',
        'alpha': alpha,
        'adjusted_alpha': adjusted_alpha,
        'significant': significant,
        'num_significant': sum (significant)
    }


def benjamini_hochberg (pvalues: list, alpha: float = 0.05) -> Dict:
    """
    Benjamini-Hochberg FDR correction
    
    Less conservative than Bonferroni
    """
    reject, pvals_corrected, _, _ = multipletests(
        pvalues,
        alpha=alpha,
        method='fdr_bh'
    )
    
    return {
        'method': 'Benjamini-Hochberg',
        'alpha': alpha,
        'corrected_pvalues': pvals_corrected,
        'significant': reject,
        'num_significant': sum (reject)
    }


# Example: Test multiple metrics
metrics = ['sharpe_ratio', 'win_rate', 'max_drawdown']
pvalues = []

for metric in metrics:
    analyzer = ABTestAnalyzer (df, metric=metric)
    results = analyzer.ttest()
    pvalues.append (results['pvalue'])

print("\\n=== Multiple Testing Correction ===")
print(f"Metrics tested: {len (metrics)}")
print(f"\\nRaw p-values:")
for metric, pval in zip (metrics, pvalues):
    print(f"  {metric}: {pval:.4f}")

# Bonferroni
bonf = bonferroni_correction (pvalues, alpha=0.05)
print(f"\\nBonferroni correction:")
print(f"  Adjusted α: {bonf['adjusted_alpha']:.4f}")
print(f"  Significant: {bonf['num_significant']}/{len (metrics)}")

# Benjamini-Hochberg
bh = benjamini_hochberg (pvalues, alpha=0.05)
print(f"\\nBenjamini-Hochberg:")
print(f"  Significant: {bh['num_significant']}/{len (metrics)}")
\`\`\`

---

## Multi-Armed Bandits

### Epsilon-Greedy

\`\`\`python
"""
Multi-Armed Bandit for Adaptive Testing
"""

import numpy as np

class EpsilonGreedyBandit:
    """
    Epsilon-Greedy Multi-Armed Bandit
    
    Balances exploration and exploitation
    """
    
    def __init__(self, variants: list, epsilon: float = 0.1):
        self.variants = variants
        self.epsilon = epsilon
        
        # Track performance
        self.counts = {v: 0 for v in variants}
        self.rewards = {v: [] for v in variants}
    
    def select_variant (self) -> str:
        """
        Select variant using epsilon-greedy
        """
        # Explore with probability epsilon
        if np.random.random() < self.epsilon:
            return np.random.choice (self.variants)
        
        # Exploit: choose best variant
        avg_rewards = {
            v: np.mean (rewards) if rewards else 0.0
            for v, rewards in self.rewards.items()
        }
        
        best_variant = max (avg_rewards, key=avg_rewards.get)
        
        return best_variant
    
    def update (self, variant: str, reward: float):
        """
        Update with observed reward
        """
        self.counts[variant] += 1
        self.rewards[variant].append (reward)
    
    def get_stats (self) -> Dict:
        """
        Get current statistics
        """
        stats = {}
        
        for variant in self.variants:
            rewards = self.rewards[variant]
            
            stats[variant] = {
                'count': self.counts[variant],
                'mean_reward': np.mean (rewards) if rewards else 0.0,
                'total_reward': sum (rewards)
            }
        
        return stats


# Simulate bandit
bandit = EpsilonGreedyBandit(
    variants=['model_v1', 'model_v2', 'model_v3'],
    epsilon=0.1
)

# True reward distributions (unknown to bandit)
true_rewards = {
    'model_v1': 0.50,  # Average reward
    'model_v2': 0.55,  # Best model
    'model_v3': 0.45   # Worst model
}

# Run simulation
n_rounds = 1000

for i in range (n_rounds):
    # Select variant
    variant = bandit.select_variant()
    
    # Simulate reward (Bernoulli)
    reward = 1 if np.random.random() < true_rewards[variant] else 0
    
    # Update bandit
    bandit.update (variant, reward)

# Results
print("\\n=== Multi-Armed Bandit Results ===")
print(f"Rounds: {n_rounds}")
print(f"Epsilon: {bandit.epsilon}\\n")

stats = bandit.get_stats()
for variant, stat in stats.items():
    print(f"{variant}:")
    print(f"  Pulls: {stat['count']} ({stat['count']/n_rounds*100:.1f}%)")
    print(f"  Mean reward: {stat['mean_reward']:.4f}")
    print(f"  True reward: {true_rewards[variant]:.4f}")
    print()

# Best variant
best = max (stats.items(), key=lambda x: x[1]['mean_reward'])[0]
print(f"Best variant: {best}")
\`\`\`

### Thompson Sampling

\`\`\`python
"""
Thompson Sampling (Bayesian Bandit)
"""

from scipy.stats import beta

class ThompsonSamplingBandit:
    """
    Thompson Sampling for binary rewards
    
    More sophisticated than epsilon-greedy
    """
    
    def __init__(self, variants: list):
        self.variants = variants
        
        # Beta distribution parameters (start with uniform prior)
        self.alpha = {v: 1 for v in variants}  # Successes + 1
        self.beta = {v: 1 for v in variants}   # Failures + 1
    
    def select_variant (self) -> str:
        """
        Select variant using Thompson Sampling
        """
        # Sample from each variant's posterior
        samples = {
            v: np.random.beta (self.alpha[v], self.beta[v])
            for v in self.variants
        }
        
        # Select variant with highest sample
        best_variant = max (samples, key=samples.get)
        
        return best_variant
    
    def update (self, variant: str, reward: float):
        """
        Update posterior with observed reward
        """
        if reward > 0:
            self.alpha[variant] += 1
        else:
            self.beta[variant] += 1
    
    def get_stats (self) -> Dict:
        """
        Get posterior statistics
        """
        stats = {}
        
        for variant in self.variants:
            alpha = self.alpha[variant]
            beta_param = self.beta[variant]
            
            # Posterior mean = alpha / (alpha + beta)
            mean = alpha / (alpha + beta_param)
            
            stats[variant] = {
                'successes': alpha - 1,
                'failures': beta_param - 1,
                'mean': mean,
                'ci_95': beta.interval(0.95, alpha, beta_param)
            }
        
        return stats


# Run Thompson Sampling
ts_bandit = ThompsonSamplingBandit(['model_v1', 'model_v2', 'model_v3'])

for i in range(1000):
    variant = ts_bandit.select_variant()
    reward = 1 if np.random.random() < true_rewards[variant] else 0
    ts_bandit.update (variant, reward)

# Results
print("\\n=== Thompson Sampling Results ===")
stats = ts_bandit.get_stats()

for variant, stat in stats.items():
    print(f"{variant}:")
    print(f"  Successes: {stat['successes']}")
    print(f"  Failures: {stat['failures']}")
    print(f"  Posterior mean: {stat['mean']:.4f}")
    print(f"  95% CI: [{stat['ci_95'][0]:.4f}, {stat['ci_95'][1]:.4f}]")
    print(f"  True reward: {true_rewards[variant]:.4f}")
    print()
\`\`\`

---

## Key Takeaways

1. **Experimental Design**: Clear hypothesis, metrics, sample size
2. **Traffic Splitting**: Consistent hashing for user assignment
3. **Statistical Analysis**: T-tests, bootstrap, effect sizes
4. **Multiple Testing**: Bonferroni or FDR correction
5. **Adaptive Testing**: Multi-armed bandits for efficient exploration
6. **Gradual Rollout**: Start small (5%), ramp up gradually

**Trading-Specific**:
- Use Sharpe ratio as primary metric
- Test on paper trading first
- Monitor max drawdown closely (guardrail metric)
- Consider sequential testing for faster decisions

**Next Steps**: With A/B testing covered, we'll dive into scalability and performance optimization.
`,
};
