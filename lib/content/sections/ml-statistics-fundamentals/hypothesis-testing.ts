/**
 * Hypothesis Testing Section
 */

export const hypothesistestingSection = {
  id: 'hypothesis-testing',
  title: 'Hypothesis Testing',
  content: `# Hypothesis Testing

## Introduction

Hypothesis testing is a formal framework for making decisions based on data. It\'s everywhere in machine learning:

- **A/B Testing**: Is the new model better than the old one?
- **Feature Selection**: Does adding this feature significantly improve performance?
- **Model Comparison**: Are two models statistically different in accuracy?
- **Deployment Decisions**: Is the model significantly better than random/baseline?
- **Statistical Significance**: Is an observed difference real or just random chance?

The fundamental question: **"Is what we observe likely due to chance, or does it reflect a real effect?"**

## The Hypothesis Testing Framework

### Five Steps of Hypothesis Testing

1. **State the hypotheses** (H₀ and H₁)
2. **Choose a significance level** (α, typically 0.05)
3. **Calculate the test statistic**4. **Determine the p-value**5. **Make a decision** (reject or fail to reject H₀)

### Null and Alternative Hypotheses

**Null Hypothesis (H₀)**: The "status quo" or "no effect" hypothesis
- Example: "The new model has the same accuracy as the baseline"
- We assume this is true unless evidence suggests otherwise

**Alternative Hypothesis (H₁ or Hₐ)**: What we're trying to find evidence for
- Example: "The new model has different accuracy than the baseline"

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

np.random.seed(42)

print("=== Hypothesis Testing Framework ===")
print("Example: Is our model better than random guessing?")
print()
print("H₀ (Null): accuracy = 0.50 (random)")
print("H₁ (Alternative): accuracy > 0.50 (better than random)")
print()
print("We'll collect data and decide whether to reject H₀")
\`\`\`

## Type I and Type II Errors

### The Four Outcomes

|                | H₀ is True        | H₀ is False       |
|----------------|-------------------|-------------------|
| **Reject H₀**  | Type I Error (α)  | Correct Decision  |
| **Fail to Reject H₀** | Correct Decision  | Type II Error (β) |

**Type I Error (False Positive)**: Rejecting H₀ when it's actually true
- α = P(Type I Error) = Significance level
- Example: Concluding model is better when it's actually not

**Type II Error (False Negative)**: Failing to reject H₀ when it's actually false
- β = P(Type II Error)
- Power = 1 - β = P(Correctly rejecting false H₀)

\`\`\`python
def visualize_errors():
    """Visualize Type I and Type II errors"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Type I Error
    x = np.linspace(-4, 4, 1000)
    y_null = stats.norm.pdf (x, 0, 1)
    
    ax = axes[0]
    ax.plot (x, y_null, 'b-', linewidth=2, label='H₀ distribution')
    ax.fill_between (x[x > 1.96], 0, y_null[x > 1.96], alpha=0.3, color='red', 
                     label='Type I Error region (α=0.05)')
    ax.axvline(1.96, color='r', linestyle='--', linewidth=2)
    ax.set_title('Type I Error: Reject H₀ when H₀ is True', fontweight='bold')
    ax.set_xlabel('Test Statistic')
    ax.set_ylabel('Probability Density')
    ax.legend()
    ax.text(2.5, 0.15, f'α = 0.05', fontsize=12, color='red')
    
    # Type II Error
    y_alt = stats.norm.pdf (x, 2, 1)
    
    ax = axes[1]
    ax.plot (x, y_null, 'b--', linewidth=2, label='H₀ distribution', alpha=0.5)
    ax.plot (x, y_alt, 'g-', linewidth=2, label='H₁ distribution (true)')
    ax.fill_between (x[x < 1.96], 0, y_alt[x < 1.96], alpha=0.3, color='orange',
                     label='Type II Error region (β)')
    ax.fill_between (x[x >= 1.96], 0, y_alt[x >= 1.96], alpha=0.3, color='green',
                     label='Power (1-β)')
    ax.axvline(1.96, color='r', linestyle='--', linewidth=2, label='Decision threshold')
    ax.set_title('Type II Error: Fail to Reject H₀ when H₁ is True', fontweight='bold')
    ax.set_xlabel('Test Statistic')
    ax.set_ylabel('Probability Density')
    ax.legend (loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('hypothesis_testing_errors.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Type I Error (α): False alarm - saying there's an effect when there isn't")
    print("Type II Error (β): Miss - failing to detect a real effect")
    print("Power (1-β): Probability of correctly detecting a real effect")

visualize_errors()
\`\`\`

### Error Trade-offs

\`\`\`python
print("\\n=== Error Trade-offs ===")
print("• Lower α (stricter) → Fewer false positives, but more false negatives")
print("• Higher α (lenient) → Fewer false negatives, but more false positives")
print()
print("Common α levels:")
print("  α = 0.05 (5%): Standard in most sciences")
print("  α = 0.01 (1%): More conservative, fewer false positives")
print("  α = 0.10 (10%): More lenient, exploratory analysis")
print()
print("In ML:")
print("  • Medical diagnosis: Prefer low α (avoid false positives)")
print("  • Exploratory feature selection: Can use higher α")
\`\`\`

## P-Values

The **p-value** is the probability of observing data at least as extreme as what we got, assuming H₀ is true.

**Interpretation**:
- p < α: Reject H₀ (statistically significant)
- p ≥ α: Fail to reject H₀ (not statistically significant)

**What p-values ARE NOT**:
- ✗ NOT the probability that H₀ is true
- ✗ NOT the probability that the result is due to chance
- ✗ NOT the effect size or practical importance

\`\`\`python
def demonstrate_p_value():
    """Demonstrate p-value calculation and interpretation"""
    
    # Scenario: Testing if a coin is fair
    # H₀: p = 0.5 (fair coin)
    # H₁: p ≠ 0.5 (biased coin)
    
    n_flips = 100
    observed_heads = 60  # Observed 60 heads in 100 flips
    
    # Under H₀, expected 50 heads
    # Calculate p-value using binomial test
    p_value = stats.binom_test (observed_heads, n_flips, 0.5, alternative='two-sided')
    
    print("=== P-Value Example ===")
    print(f"Observed: {observed_heads} heads in {n_flips} flips")
    print(f"Expected under H₀: {n_flips * 0.5:.0f} heads")
    print(f"p-value: {p_value:.4f}")
    print()
    
    if p_value < 0.05:
        print(f"Decision: Reject H₀ (p < 0.05)")
        print("Conclusion: Evidence suggests coin is biased")
    else:
        print(f"Decision: Fail to reject H₀ (p ≥ 0.05)")
        print("Conclusion: Insufficient evidence that coin is biased")
    
    # Visualize
    x = np.arange(0, n_flips + 1)
    pmf = stats.binom.pmf (x, n_flips, 0.5)
    
    fig, ax = plt.subplots (figsize=(12, 6))
    ax.bar (x, pmf, alpha=0.7, edgecolor='black', label='Binomial distribution under H₀')
    
    # Highlight observed value and more extreme
    extreme_mask = (x <= (n_flips - observed_heads)) | (x >= observed_heads)
    ax.bar (x[extreme_mask], pmf[extreme_mask], color='red', alpha=0.7, 
           label=f'Values as or more extreme (p-value region)')
    
    ax.axvline (observed_heads, color='green', linestyle='--', linewidth=2, 
               label=f'Observed: {observed_heads}')
    ax.axvline (n_flips * 0.5, color='blue', linestyle='--', linewidth=2,
               label=f'Expected: {n_flips * 0.5:.0f}')
    
    ax.set_xlabel('Number of Heads', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title (f'P-Value Visualization (p = {p_value:.4f})', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid (axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('p_value_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

demonstrate_p_value()
\`\`\`

## One-Tailed vs Two-Tailed Tests

**Two-tailed test**: Testing for any difference (≠)
- H₀: μ = μ₀
- H₁: μ ≠ μ₀
- Use when direction doesn't matter

**One-tailed test**: Testing for a specific direction (> or <)
- H₀: μ ≤ μ₀ vs H₁: μ > μ₀ (right-tailed)
- H₀: μ ≥ μ₀ vs H₁: μ < μ₀ (left-tailed)
- Use when you only care about one direction

\`\`\`python
def compare_one_two_tailed():
    """Compare one-tailed and two-tailed tests"""
    
    # Sample data: model accuracies
    baseline_accuracy = 0.75
    new_model_accuracies = np.random.normal(0.78, 0.05, 30)  # Slightly better
    
    # Two-tailed test: Is accuracy different?
    t_stat, p_two = stats.ttest_1samp (new_model_accuracies, baseline_accuracy)
    
    # One-tailed test: Is accuracy greater?
    p_one = p_two / 2 if t_stat > 0 else 1 - (p_two / 2)
    
    print("=== One-Tailed vs Two-Tailed Test ===")
    print(f"Baseline accuracy: {baseline_accuracy:.2f}")
    print(f"New model mean accuracy: {new_model_accuracies.mean():.4f}")
    print()
    print("Two-Tailed Test (H₁: accuracy ≠ 0.75):")
    print(f"  p-value: {p_two:.4f}")
    print(f"  Result: {'Significant' if p_two < 0.05 else 'Not significant'} at α=0.05")
    print()
    print("One-Tailed Test (H₁: accuracy > 0.75):")
    print(f"  p-value: {p_one:.4f}")
    print(f"  Result: {'Significant' if p_one < 0.05 else 'Not significant'} at α=0.05")
    print()
    print("One-tailed test has more power when direction is known!")

compare_one_two_tailed()
\`\`\`

## Statistical Power

**Power** = P(Rejecting H₀ | H₁ is true) = 1 - β

Higher power means:
- Better ability to detect real effects
- Fewer false negatives (Type II errors)

**Factors affecting power**:
1. **Sample size** (larger → more power)
2. **Effect size** (larger → more power)
3. **Significance level α** (higher → more power, but more Type I errors)
4. **Variability** (lower → more power)

\`\`\`python
def power_analysis_demo():
    """Demonstrate power analysis"""
    
    from statsmodels.stats.power import ttest_power
    
    # Calculate power for different sample sizes
    effect_size = 0.5  # Cohen\'s d (medium effect)
    alpha = 0.05
    sample_sizes = [10, 20, 30, 50, 100, 200, 500]
    
    powers = [ttest_power (effect_size, n, alpha, alternative='two-sided') 
              for n in sample_sizes]
    
    # Plot
    fig, ax = plt.subplots (figsize=(10, 6))
    ax.plot (sample_sizes, powers, 'o-', linewidth=2, markersize=8)
    ax.axhline(0.8, color='r', linestyle='--', linewidth=2, 
               label='Target power: 0.80')
    ax.set_xlabel('Sample Size', fontsize=12)
    ax.set_ylabel('Statistical Power', fontsize=12)
    ax.set_title (f'Power Analysis (Effect Size = {effect_size}, α = {alpha})', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Annotate required sample size for 80% power
    target_n = next (n for n, p in zip (sample_sizes, powers) if p >= 0.8)
    ax.plot (target_n, 0.8, 'ro', markersize=12)
    ax.annotate (f'n ≈ {target_n} needed\\nfor 80% power', 
                xy=(target_n, 0.8), xytext=(target_n + 50, 0.7),
                fontsize=10, arrowprops=dict (arrowstyle='->', color='red'))
    
    plt.tight_layout()
    plt.savefig('power_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("=== Power Analysis ===")
    print(f"Effect size (Cohen\'s d): {effect_size}")
    print(f"Significance level: {alpha}")
    print()
    print("Sample Size    Power")
    print("-" * 25)
    for n, power in zip (sample_sizes, powers):
        print(f"{n:6d}        {power:.3f}")
    print()
    print(f"Rule of thumb: Aim for 80% power")
    print(f"Need n ≈ {target_n} for this effect size")

power_analysis_demo()
\`\`\`

## ML Application: A/B Testing Models

\`\`\`python
def model_ab_test():
    """A/B test: Is new model better than baseline?"""
    
    # Simulate model performances
    np.random.seed(42)
    
    n_samples = 100
    baseline_accuracies = np.random.binomial(1, 0.75, n_samples)  # 75% baseline
    new_model_accuracies = np.random.binomial(1, 0.78, n_samples)  # 78% new model
    
    baseline_acc = baseline_accuracies.mean()
    new_acc = new_model_accuracies.mean()
    
    # Hypothesis test: Are accuracies different?
    # H₀: accuracy_new = accuracy_baseline
    # H₁: accuracy_new > accuracy_baseline (one-tailed)
    
    # Using proportions test
    from statsmodels.stats.proportion import proportions_ztest
    
    count = np.array([new_model_accuracies.sum(), baseline_accuracies.sum()])
    nobs = np.array([n_samples, n_samples])
    
    z_stat, p_value = proportions_ztest (count, nobs, alternative='larger')
    
    print("=== Model A/B Test ===")
    print(f"Baseline model accuracy: {baseline_acc:.2%}")
    print(f"New model accuracy: {new_acc:.2%}")
    print(f"Absolute improvement: {new_acc - baseline_acc:.2%}")
    print(f"Relative improvement: {((new_acc / baseline_acc) - 1):.1%}")
    print()
    print(f"H₀: new_accuracy ≤ baseline_accuracy")
    print(f"H₁: new_accuracy > baseline_accuracy")
    print()
    print(f"Z-statistic: {z_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    print()
    
    if p_value < 0.05:
        print("✓ Decision: Reject H₀ (p < 0.05)")
        print("✓ Conclusion: New model is significantly better!")
        print("✓ Recommendation: Deploy new model")
    else:
        print("✗ Decision: Fail to reject H₀ (p ≥ 0.05)")
        print("✗ Conclusion: Improvement is not statistically significant")
        print("✗ Recommendation: Keep baseline model or collect more data")
    
    # Visualize
    fig, ax = plt.subplots (figsize=(10, 6))
    
    models = ['Baseline', 'New Model']
    accuracies = [baseline_acc, new_acc]
    colors = ['gray', 'green' if p_value < 0.05 else 'orange']
    
    bars = ax.bar (models, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add error bars (confidence intervals)
    for i, (acc, n) in enumerate([(baseline_acc, n_samples), (new_acc, n_samples)]):
        se = np.sqrt (acc * (1 - acc) / n)
        ci = 1.96 * se
        ax.errorbar (i, acc, yerr=ci, fmt='none', color='black', capsize=10, linewidth=2)
    
    # Add value labels
    for bar, acc in zip (bars, accuracies):
        height = bar.get_height()
        ax.text (bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title (f'Model A/B Test\\n (p = {p_value:.4f}, {"Significant" if p_value < 0.05 else "Not Significant"})',
                 fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid (axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_ab_test.png', dpi=300, bbox_inches='tight')
    plt.show()

model_ab_test()
\`\`\`

## Common Pitfalls

### 1. P-Hacking (Multiple Testing)

\`\`\`python
def demonstrate_p_hacking():
    """Show how multiple testing inflates false positive rate"""
    
    np.random.seed(42)
    n_tests = 20
    false_positives = 0
    
    print("=== P-Hacking Demonstration ===")
    print(f"Testing {n_tests} features with NO real effect")
    print("H₀ is true for all features (all noise)")
    print()
    
    for i in range (n_tests):
        # Generate noise data (no real effect)
        group1 = np.random.normal(0, 1, 30)
        group2 = np.random.normal(0, 1, 30)
        
        _, p = stats.ttest_ind (group1, group2)
        
        if p < 0.05:
            false_positives += 1
            print(f"Test {i+1}: p = {p:.4f} ⚠️ FALSE POSITIVE!")
    
    print()
    print(f"False positives: {false_positives}/{n_tests} ({false_positives/n_tests:.1%})")
    print(f"Expected: ~{0.05 * n_tests:.0f} (α = 0.05)")
    print()
    print("Solution: Bonferroni correction")
    print(f"Adjusted α = 0.05/{n_tests} = {0.05/n_tests:.4f}")
    print("Or use False Discovery Rate (FDR) control")

demonstrate_p_hacking()
\`\`\`

### 2. Confusing Statistical and Practical Significance

\`\`\`python
# Statistical significance ≠ Practical importance
n_huge = 100000
improvement = 0.001  # 0.1% improvement

baseline = np.random.binomial(1, 0.75, n_huge)
new_model = np.random.binomial(1, 0.75 + improvement, n_huge)

_, p_value = proportions_ztest(
    [new_model.sum(), baseline.sum()],
    [n_huge, n_huge],
    alternative='two-sided'
)

print("\\n=== Statistical vs Practical Significance ===")
print(f"Sample size: {n_huge:,}")
print(f"Baseline: {baseline.mean():.4f}")
print(f"New model: {new_model.mean():.4f}")
print(f"Improvement: {new_model.mean() - baseline.mean():.4f} ({improvement:.1%})")
print(f"P-value: {p_value:.6f}")
print()
print("Result: Statistically significant (p < 0.05) ✓")
print("But: 0.1% improvement - is it worth the cost?")
print()
print("Always ask: Is the effect size practically important?")
\`\`\`

## Key Takeaways

1. **Hypothesis testing**: Framework for making decisions under uncertainty
2. **Null hypothesis (H₀)**: Assumption we're trying to reject
3. **P-value**: Probability of data if H₀ is true (NOT probability H₀ is true)
4. **Significance level (α)**: Threshold for Type I error, typically 0.05
5. **Type I error**: False positive (rejecting true H₀)
6. **Type II error**: False negative (failing to reject false H₀)
7. **Power**: Probability of detecting real effects = 1 - β
8. **Sample size**: Larger samples give more power
9. **Multiple testing**: Adjust significance level to avoid false positives
10. **Practical significance**: Statistical significance ≠ practical importance

## Connection to Machine Learning

- **Model comparison**: Test if new model significantly outperforms baseline
- **Feature selection**: Test if features significantly contribute
- **A/B testing**: Deploy model A or B based on statistical tests
- **Early stopping**: Stop training if validation improvement not significant
- **Hyperparameter tuning**: Test if configuration significantly improves performance
- **Deployment decisions**: Only deploy if significantly better than threshold
- **Monitoring**: Detect if model performance significantly degrades

Hypothesis testing turns "this model seems better" into "this model is statistically significantly better with p<0.001" - essential for rigorous ML development!
`,
};
