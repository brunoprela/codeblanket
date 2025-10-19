/**
 * Statistical Inference Fundamentals Section
 */

export const statisticalinferenceSection = {
  id: 'statistical-inference',
  title: 'Statistical Inference Fundamentals',
  content: `# Statistical Inference Fundamentals

## Introduction

Statistical inference is the process of drawing conclusions about a population based on a sample. This is fundamental to machine learning because:

- **We never have all the data**: Models train on samples, not entire populations
- **Predictions are estimates**: ML predictions come with uncertainty
- **Generalization is key**: We care about performance on unseen data, not just training data
- **Decision making**: Confidence intervals guide whether to deploy a model
- **A/B testing**: Statistical inference determines if model improvements are real

Every ML task involves inference:
- Training set → Population of all possible data
- Validation accuracy → True generalization error
- Model predictions → Expected outcomes

## Population vs Sample

### Definitions

**Population**: The entire group we want to understand
- All customers (past, present, and future)
- All possible transactions
- All images that could ever be classified

**Sample**: A subset of the population we actually observe
- Training data
- Test data
- Observed transactions

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

np.random.seed(42)

# True population (theoretical)
population_mean = 100
population_std = 15
population_size = 1_000_000

# Generate population (simulate)
population = np.random.normal(population_mean, population_std, population_size)

print("=== POPULATION (True Parameters) ===")
print(f"Mean: {population.mean():.4f}")
print(f"Std Dev: {population.std(ddof=0):.4f}")
print(f"Size: {len(population):,}")

# Take a sample
sample_size = 100
sample = np.random.choice(population, size=sample_size, replace=False)

print(f"\\n=== SAMPLE (Estimated Parameters) ===")
print(f"Mean: {sample.mean():.4f}")
print(f"Std Dev: {sample.std(ddof=1):.4f}")  # Use ddof=1 for sample
print(f"Size: {len(sample)}")

print(f"\\n=== ESTIMATION ERROR ===")
print(f"Mean error: {abs(sample.mean() - population.mean()):.4f}")
print(f"Std error: {abs(sample.std(ddof=1) - population.std(ddof=0)):.4f}")
\`\`\`

### Why Sampling?

1. **Impractical to measure everyone**: Can't survey all customers
2. **Impossible to know the future**: Can't train on future data
3. **Costly**: Measuring entire population is expensive
4. **Time-consuming**: Sampling is faster

**Key Insight**: Good samples can give accurate estimates of population parameters!

## Sampling Distributions

### The Distribution of Sample Means

If we take many samples and compute the mean of each, those means form a distribution:

\`\`\`python
def demonstrate_sampling_distribution():
    """Show how sample means form a normal distribution"""
    
    # Population
    population = np.random.exponential(scale=2, size=100000)  # Skewed!
    
    # Take many samples and compute means
    n_samples = 1000
    sample_size = 30
    sample_means = []
    
    for _ in range(n_samples):
        sample = np.random.choice(population, size=sample_size, replace=False)
        sample_means.append(sample.mean())
    
    sample_means = np.array(sample_means)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Original population (skewed)
    axes[0].hist(population, bins=50, density=True, alpha=0.7, edgecolor='black')
    axes[0].axvline(population.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {population.mean():.2f}')
    axes[0].set_title('Population Distribution\\n(Exponential - Skewed!)')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Density')
    axes[0].legend()
    
    # Distribution of sample means (normal!)
    axes[1].hist(sample_means, bins=50, density=True, alpha=0.7, edgecolor='black')
    axes[1].axvline(sample_means.mean(), color='r', linestyle='--', linewidth=2, 
                     label=f'Mean: {sample_means.mean():.2f}')
    axes[1].axvline(population.mean(), color='g', linestyle='--', linewidth=2,
                     label=f'True: {population.mean():.2f}')
    axes[1].set_title(f'Sampling Distribution of Means\\n(n={sample_size}, samples={n_samples})')
    axes[1].set_xlabel('Sample Mean')
    axes[1].set_ylabel('Density')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('sampling_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("CENTRAL LIMIT THEOREM in action!")
    print("• Population: Skewed (exponential)")
    print("• Sampling distribution: Normal (bell curve)")
    print(f"• Mean of sample means: {sample_means.mean():.4f}")
    print(f"• True population mean: {population.mean():.4f}")
    print(f"• Difference: {abs(sample_means.mean() - population.mean()):.4f}")

demonstrate_sampling_distribution()
\`\`\`

**Key Insight**: Even if the population is skewed, the distribution of sample means is approximately normal (Central Limit Theorem)!

## Standard Error

The **standard error** is the standard deviation of the sampling distribution.

### Standard Error of the Mean

\\[ SE = \\frac{\\sigma}{\\sqrt{n}} \\]

Where:
- σ = population standard deviation
- n = sample size

**Interpretation**: How much sample means vary from the true population mean.

\`\`\`python
def compute_standard_error(data, sample_sizes):
    """Demonstrate standard error decreases with sample size"""
    
    population = np.random.normal(100, 15, 100000)
    true_mean = population.mean()
    true_std = population.std(ddof=0)
    
    results = []
    
    for n in sample_sizes:
        # Take many samples of size n
        sample_means = []
        for _ in range(1000):
            sample = np.random.choice(population, size=n, replace=False)
            sample_means.append(sample.mean())
        
        sample_means = np.array(sample_means)
        
        # Empirical standard error
        empirical_se = sample_means.std(ddof=1)
        
        # Theoretical standard error
        theoretical_se = true_std / np.sqrt(n)
        
        results.append({
            'n': n,
            'empirical_se': empirical_se,
            'theoretical_se': theoretical_se,
            'mean_error': abs(sample_means.mean() - true_mean)
        })
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Standard error vs sample size
    ns = [r['n'] for r in results]
    empirical = [r['empirical_se'] for r in results]
    theoretical = [r['theoretical_se'] for r in results]
    
    axes[0].plot(ns, empirical, 'o-', linewidth=2, markersize=8, label='Empirical SE')
    axes[0].plot(ns, theoretical, 's--', linewidth=2, markersize=8, label='Theoretical SE')
    axes[0].set_xlabel('Sample Size (n)', fontsize=12)
    axes[0].set_ylabel('Standard Error', fontsize=12)
    axes[0].set_title('Standard Error Decreases with Sample Size')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log')
    
    # Mean estimation error
    errors = [r['mean_error'] for r in results]
    axes[1].plot(ns, errors, 'o-', linewidth=2, markersize=8, color='red')
    axes[1].set_xlabel('Sample Size (n)', fontsize=12)
    axes[1].set_ylabel('Mean Estimation Error', fontsize=12)
    axes[1].set_title('Accuracy Improves with More Data')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('standard_error_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("=== Standard Error Analysis ===")
    for r in results:
        print(f"n={r['n']:4d}: SE={r['empirical_se']:.4f}, Error={r['mean_error']:.4f}")
    
    print(f"\\nKey insight: To halve the SE, you need 4x the data!")
    print(f"SE ∝ 1/√n")

compute_standard_error(None, [10, 30, 100, 300, 1000, 3000])
\`\`\`

## Confidence Intervals

A **confidence interval** gives a range of plausible values for a parameter.

### 95% Confidence Interval for the Mean

\\[ CI = \\bar{x} \\pm z_{\\alpha/2} \\cdot SE \\]

Where:
- \\(\\bar{x}\\) = sample mean
- \\(z_{\\alpha/2}\\) = 1.96 for 95% confidence (from standard normal)
- SE = standard error

**Interpretation**: "We are 95% confident the true mean is in this range"

\`\`\`python
def compute_confidence_interval(sample, confidence=0.95):
    """Compute confidence interval for the mean"""
    
    n = len(sample)
    mean = np.mean(sample)
    se = stats.sem(sample)  # Standard error of the mean
    
    # For large samples, use z-score
    # For small samples (n < 30), use t-score
    if n >= 30:
        critical_value = stats.norm.ppf((1 + confidence) / 2)
        method = "Z"
    else:
        critical_value = stats.t.ppf((1 + confidence) / 2, df=n-1)
        method = "t"
    
    margin_of_error = critical_value * se
    ci_lower = mean - margin_of_error
    ci_upper = mean + margin_of_error
    
    return mean, ci_lower, ci_upper, method

# Example
np.random.seed(42)
population = np.random.normal(100, 15, 100000)
sample = np.random.choice(population, size=50, replace=False)

mean, ci_lower, ci_upper, method = compute_confidence_interval(sample, 0.95)

print("=== Confidence Interval ===")
print(f"Sample size: {len(sample)}")
print(f"Sample mean: {mean:.2f}")
print(f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
print(f"Method used: {method}-distribution")
print(f"True population mean: {population.mean():.2f}")
print(f"\\nIs true mean in CI? {ci_lower <= population.mean() <= ci_upper}")
\`\`\`

### Visualizing Confidence Intervals

\`\`\`python
def visualize_confidence_intervals(population, n_samples=20, sample_size=30):
    """Show that ~95% of CIs contain the true mean"""
    
    true_mean = population.mean()
    
    # Take many samples and compute CIs
    cis = []
    contains_true = []
    
    for _ in range(n_samples):
        sample = np.random.choice(population, size=sample_size, replace=False)
        mean, ci_lower, ci_upper, _ = compute_confidence_interval(sample, 0.95)
        cis.append((mean, ci_lower, ci_upper))
        contains_true.append(ci_lower <= true_mean <= ci_upper)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i, (mean, lower, upper) in enumerate(cis):
        color = 'blue' if contains_true[i] else 'red'
        ax.plot([lower, upper], [i, i], 'o-', linewidth=2, color=color, alpha=0.6)
        ax.plot(mean, i, 'o', markersize=8, color=color)
    
    # True mean line
    ax.axvline(true_mean, color='green', linestyle='--', linewidth=2, 
               label=f'True Mean: {true_mean:.2f}')
    
    # Count
    n_contain = sum(contains_true)
    percentage = (n_contain / n_samples) * 100
    
    ax.set_xlabel('Value', fontsize=12)
    ax.set_ylabel('Sample Number', fontsize=12)
    ax.set_title(f'95% Confidence Intervals\\n{n_contain}/{n_samples} ({percentage:.0f}%) contain true mean')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('confidence_intervals_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Expected to contain true mean: ~95%")
    print(f"Actually contain true mean: {percentage:.1f}%")
    print(f"\\nBlue CIs: contain true mean")
    print(f"Red CIs: do NOT contain true mean (expected ~5%)")

visualize_confidence_intervals(population, n_samples=20, sample_size=30)
\`\`\`

## Types of Estimation

### Point Estimation

A single value estimate:
- Sample mean estimates population mean
- Sample proportion estimates population proportion

\`\`\`python
# Point estimates
sample = np.random.choice(population, size=100, replace=False)

print("=== Point Estimates ===")
print(f"Population mean: {population.mean():.2f}")
print(f"Sample mean (point estimate): {sample.mean():.2f}")
print(f"Error: {abs(sample.mean() - population.mean()):.2f}")
\`\`\`

### Interval Estimation

A range of plausible values:
- Confidence intervals
- More informative than point estimates

\`\`\`python
# Interval estimate
mean, ci_lower, ci_upper, _ = compute_confidence_interval(sample, 0.95)

print(f"\\n=== Interval Estimate ===")
print(f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
print(f"Width: {ci_upper - ci_lower:.2f}")
print(f"Contains true mean? {ci_lower <= population.mean() <= ci_upper}")
\`\`\`

## Bootstrap Method

**Bootstrap**: Resample from your sample to estimate uncertainty.

\`\`\`python
def bootstrap_confidence_interval(data, n_bootstrap=10000, confidence=0.95):
    """Non-parametric bootstrap CI"""
    
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(bootstrap_sample.mean())
    
    bootstrap_means = np.array(bootstrap_means)
    
    # Percentile method
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_means, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
    
    return bootstrap_means, ci_lower, ci_upper

# Compare bootstrap vs traditional CI
sample = np.random.choice(population, size=50, replace=False)

# Traditional CI
mean, trad_lower, trad_upper, _ = compute_confidence_interval(sample, 0.95)

# Bootstrap CI
bootstrap_means, boot_lower, boot_upper = bootstrap_confidence_interval(sample, 10000, 0.95)

print("=== Bootstrap vs Traditional CI ===")
print(f"Traditional 95% CI: [{trad_lower:.2f}, {trad_upper:.2f}]")
print(f"Bootstrap 95% CI:   [{boot_lower:.2f}, {boot_upper:.2f}]")
print(f"\\nBootstrap advantages:")
print("• Works for any statistic (not just mean)")
print("• No distributional assumptions")
print("• Easy to implement")

# Visualize bootstrap distribution
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(bootstrap_means, bins=50, density=True, alpha=0.7, edgecolor='black')
ax.axvline(boot_lower, color='r', linestyle='--', linewidth=2, label=f'95% CI: [{boot_lower:.2f}, {boot_upper:.2f}]')
ax.axvline(boot_upper, color='r', linestyle='--', linewidth=2)
ax.axvline(sample.mean(), color='g', linestyle='-', linewidth=2, label=f'Sample mean: {sample.mean():.2f}')
ax.set_xlabel('Bootstrap Sample Mean', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Bootstrap Distribution of Sample Means')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('bootstrap_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
\`\`\`

## ML Application: Model Performance Inference

\`\`\`python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                            n_redundant=5, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Point estimate of accuracy
test_accuracy = model.score(X_test, y_test)

print("=== ML Model Performance Inference ===")
print(f"Test set accuracy (point estimate): {test_accuracy:.4f}")

# Confidence interval via cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=10)

mean_cv = cv_scores.mean()
se_cv = cv_scores.std() / np.sqrt(len(cv_scores))
ci_lower_cv = mean_cv - 1.96 * se_cv
ci_upper_cv = mean_cv + 1.96 * se_cv

print(f"\\nCross-validation accuracy: {mean_cv:.4f}")
print(f"95% CI for accuracy: [{ci_lower_cv:.4f}, {ci_upper_cv:.4f}]")

# Bootstrap confidence interval for test accuracy
def bootstrap_model_accuracy(model, X, y, n_bootstrap=1000):
    """Bootstrap CI for model accuracy"""
    
    bootstrap_accuracies = []
    
    for _ in range(n_bootstrap):
        # Resample test set
        indices = np.random.choice(len(X), size=len(X), replace=True)
        X_boot = X[indices]
        y_boot = y[indices]
        
        accuracy = model.score(X_boot, y_boot)
        bootstrap_accuracies.append(accuracy)
    
    bootstrap_accuracies = np.array(bootstrap_accuracies)
    ci_lower = np.percentile(bootstrap_accuracies, 2.5)
    ci_upper = np.percentile(bootstrap_accuracies, 97.5)
    
    return bootstrap_accuracies, ci_lower, ci_upper

boot_acc, boot_lower, boot_upper = bootstrap_model_accuracy(model, X_test, y_test, 1000)

print(f"\\nBootstrap 95% CI for test accuracy: [{boot_lower:.4f}, {boot_upper:.4f}]")
print(f"\\nInterpretation:")
print(f"• We're 95% confident the true accuracy is between {boot_lower:.1%} and {boot_upper:.1%}")
print(f"• The range is {(boot_upper - boot_lower):.1%} wide")
print(f"• More test data would narrow this interval")
\`\`\`

## Key Takeaways

1. **Sample → Population**: Use samples to infer population parameters
2. **Sampling Distribution**: Distribution of a statistic across many samples
3. **Standard Error**: Standard deviation of the sampling distribution; decreases with √n
4. **Confidence Intervals**: Range of plausible values; 95% CI contains true parameter 95% of the time
5. **Bootstrap**: Powerful non-parametric method for computing CIs
6. **Larger samples**: Reduce standard error and narrow confidence intervals
7. **ML applications**: Quantify uncertainty in model performance estimates

## Connection to Machine Learning

- **Training set**: Sample from population of all possible data
- **Validation accuracy**: Point estimate of true generalization error
- **Cross-validation**: Better estimate with confidence interval
- **Bootstrap**: Estimate uncertainty without distributional assumptions
- **Confidence intervals**: Communicate model performance with uncertainty
- **Deployment decisions**: Only deploy if lower bound of CI exceeds threshold
- **A/B testing**: Compare two models with statistical inference

Statistical inference transforms ML from "this model got 85% accuracy" to "we're 95% confident the true accuracy is between 83% and 87%" - much more actionable!
`,
};
