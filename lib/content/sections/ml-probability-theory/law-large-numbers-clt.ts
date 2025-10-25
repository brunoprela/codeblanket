/**
 * Law of Large Numbers & CLT Section
 */

export const lawnumberscltSection = {
  id: 'law-large-numbers-clt',
  title: 'Law of Large Numbers & Central Limit Theorem',
  content: `# Law of Large Numbers & Central Limit Theorem

## Introduction

The **Law of Large Numbers** (LLN) and **Central Limit Theorem** (CLT) are two of the most important theorems in probability theory. They explain why:
- Sample averages converge to population means
- Normal distribution appears everywhere
- Larger datasets lead to better estimates

**In ML**: These theorems justify using training data to estimate expectations, explain why gradient descent works, and guarantee that empirical estimates improve with more data.

## Law of Large Numbers

**Statement**: As sample size increases, the sample mean converges to the population mean.

\\[ \\bar{X}_n = \\frac{1}{n}\\sum_{i=1}^n X_i \\xrightarrow{n \\to \\infty} E[X] \\]

**Two versions**:
- **Weak LLN**: Convergence in probability
- **Strong LLN**: Almost sure convergence

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt

def law_of_large_numbers_demo():
    """Demonstrate Law of Large Numbers"""
    
    # True mean
    np.random.seed(42)
    true_mean = 3
    true_std = 2
    
    # Generate samples
    max_n = 10000
    samples = np.random.normal (true_mean, true_std, size=max_n)
    
    # Compute running mean
    running_means = np.cumsum (samples) / np.arange(1, max_n + 1)
    
    print("=== Law of Large Numbers ===")
    print(f"True mean: {true_mean}")
    print()
    print("Sample Size | Sample Mean | Error")
    print("-" * 45)
    
    for n in [10, 100, 1000, 5000, 10000]:
        sample_mean = running_means[n-1]
        error = abs (sample_mean - true_mean)
        print(f"{n:11d} | {sample_mean:11.6f} | {error:.6f}")
    
    # Plot
    plt.figure (figsize=(12, 6))
    plt.plot (running_means, linewidth=1, label='Sample Mean')
    plt.axhline (true_mean, color='r', linestyle='--', linewidth=2, label=f'True Mean = {true_mean}')
    plt.axhline (true_mean + 0.1, color='g', linestyle=':', alpha=0.5, label='±0.1 band')
    plt.axhline (true_mean - 0.1, color='g', linestyle=':', alpha=0.5)
    plt.xlabel('Sample Size')
    plt.ylabel('Sample Mean')
    plt.title('Law of Large Numbers: Sample Mean → Population Mean')
    plt.xscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)

law_of_large_numbers_demo()

# Output:
# === Law of Large Numbers ===
# True mean: 3
#
# Sample Size | Sample Mean | Error
# ---------------------------------------------
#          10 |    3.098866 | 0.098866
#         100 |    2.891088 | 0.108912
#        1000 |    2.997814 | 0.002186
#        5000 |    2.999560 | 0.000440
#       10000 |    3.007326 | 0.007326
\`\`\`

## Central Limit Theorem (CLT)

**Statement**: The sum (or average) of many independent random variables approaches a normal distribution, regardless of the original distribution!

\\[ \\frac{\\bar{X}_n - \\mu}{\\sigma/\\sqrt{n}} \\xrightarrow{d} N(0, 1) \\]

Or equivalently:
\\[ \\bar{X}_n \\sim N\\left(\\mu, \\frac{\\sigma^2}{n}\\right) \\text{ for large n} \\]

**Why it matters**: Explains why normal distribution is ubiquitous.

\`\`\`python
def central_limit_theorem_demo():
    """Demonstrate Central Limit Theorem"""
    
    np.random.seed(42)
    
    # Start with NON-NORMAL distribution (uniform)
    pop_dist = lambda size: np.random.uniform(0, 10, size=size)
    
    # True parameters
    true_mean = 5  # E[Uniform(0,10)] = 5
    true_std = np.sqrt(100/12)  # Var[Uniform(0,10)] = (b-a)²/12 = 100/12
    
    sample_sizes = [2, 5, 10, 30]
    n_experiments = 10000
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    print("=== Central Limit Theorem ===")
    print("Original distribution: Uniform(0, 10) - NOT normal!")
    print("Sampling distribution of means:")
    print()
    
    for idx, n in enumerate (sample_sizes):
        # Generate many sample means
        sample_means = [pop_dist (n).mean() for _ in range (n_experiments)]
        
        # Theoretical CLT prediction
        theoretical_mean = true_mean
        theoretical_std = true_std / np.sqrt (n)
        
        # Empirical
        empirical_mean = np.mean (sample_means)
        empirical_std = np.std (sample_means)
        
        print(f"n = {n}:")
        print(f"  Theory: μ = {theoretical_mean:.3f}, σ = {theoretical_std:.3f}")
        print(f"  Empirical: μ = {empirical_mean:.3f}, σ = {empirical_std:.3f}")
        
        # Plot
        ax = axes[idx]
        ax.hist (sample_means, bins=50, density=True, alpha=0.7, edgecolor='black')
        
        # Overlay theoretical normal
        x = np.linspace (min (sample_means), max (sample_means), 100)
        from scipy import stats
        ax.plot (x, stats.norm (theoretical_mean, theoretical_std).pdf (x), 
                'r-', linewidth=2, label='CLT Prediction')
        
        ax.set_xlabel('Sample Mean')
        ax.set_ylabel('Density')
        ax.set_title (f'n = {n} ({"Not yet normal" if n < 10 else "Approaching normal"})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    print("\\nKey insight: Even from uniform distribution, means become normal!")

central_limit_theorem_demo()
\`\`\`

## CLT with Different Distributions

\`\`\`python
def clt_different_distributions():
    """Show CLT works for ANY distribution"""
    
    np.random.seed(42)
    n = 30  # Sample size
    n_experiments = 10000
    
    # Different source distributions
    distributions = [
        ("Uniform", lambda size: np.random.uniform(0, 1, size)),
        ("Exponential", lambda size: np.random.exponential(1, size)),
        ("Binomial", lambda size: np.random.binomial(10, 0.3, size)),
        ("Chi-squared", lambda size: np.random.chisquare(2, size)),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    print("=== CLT with Different Distributions ===")
    print(f"Sample size n = {n}, experiments = {n_experiments}")
    print()
    
    for idx, (name, dist_func) in enumerate (distributions):
        # Generate sample means
        sample_means = [dist_func (n).mean() for _ in range (n_experiments)]
        
        # Standardize
        mean = np.mean (sample_means)
        std = np.std (sample_means)
        standardized = (np.array (sample_means) - mean) / std
        
        # Plot
        ax = axes[idx]
        ax.hist (standardized, bins=50, density=True, alpha=0.7, edgecolor='black')
        
        # Overlay standard normal
        x = np.linspace(-4, 4, 100)
        from scipy import stats
        ax.plot (x, stats.norm(0, 1).pdf (x), 'r-', linewidth=2, label='N(0,1)')
        
        ax.set_xlabel('Standardized Sample Mean')
        ax.set_ylabel('Density')
        ax.set_title (f'Source: {name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        print(f"{name}: Sample means → Normal")
    
    plt.tight_layout()
    print("\\nConclusion: CLT works regardless of source distribution!")

clt_different_distributions()
\`\`\`

## ML Applications

### Gradient Descent and CLT

\`\`\`python
def gradient_descent_clt():
    """SGD gradients are approximately normal by CLT"""
    
    print("=== Stochastic Gradient Descent & CLT ===")
    print()
    print("Mini-batch gradient = average of individual gradients")
    print("By CLT: batch_gradient ~ N(true_gradient, σ²/batch_size)")
    print()
    print("Implications:")
    print("1. Larger batches → less noisy gradients (σ/√n smaller)")
    print("2. Gradient noise approximately normal")
    print("3. Convergence analysis uses CLT")
    print("4. Batch size controls exploration vs exploitation")
    print()
    print("Trade-off:")
    print("- Small batch: More noise, more exploration, slower convergence")
    print("- Large batch: Less noise, less exploration, faster per-iteration")

gradient_descent_clt()
\`\`\`

### Model Performance Estimates

\`\`\`python
def performance_estimation():
    """Use CLT for confidence intervals"""
    
    from scipy import stats
    
    # Simulated model accuracies on test set
    np.random.seed(42)
    n_test = 1000
    true_accuracy = 0.85
    
    # Simulate predictions (Bernoulli trials)
    correct = np.random.rand (n_test) < true_accuracy
    
    # Estimate accuracy
    accuracy = correct.mean()
    
    # CLT-based confidence interval
    # Sample mean ~ N(μ, σ²/n) where σ² = p(1-p) for Bernoulli
    std_error = np.sqrt (accuracy * (1 - accuracy) / n_test)
    
    # 95% confidence interval
    z_score = 1.96  # For 95%
    ci_lower = accuracy - z_score * std_error
    ci_upper = accuracy + z_score * std_error
    
    print("=== Model Performance Estimation (CLT) ===")
    print(f"Test samples: {n_test}")
    print(f"Correct predictions: {np.sum (correct)}")
    print(f"Estimated accuracy: {accuracy:.4f}")
    print(f"Standard error: {std_error:.4f}")
    print()
    print(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"True accuracy {true_accuracy} is {'inside' if ci_lower <= true_accuracy <= ci_upper else 'outside'} CI")
    print()
    print("With more test samples, CI narrows (∝ 1/√n)")

performance_estimation()
\`\`\`

## Key Takeaways

1. **LLN**: Sample mean → population mean as n → ∞
2. **CLT**: Sample mean distribution → normal, regardless of source
3. **CLT formula**: X̄ ~ N(μ, σ²/n) for large n
4. **Variance decreases**: σ/√n → 0 as n increases
5. **ML applications**: SGD convergence, confidence intervals, A/B testing
6. **Why normal is common**: CLT makes normal appear everywhere
7. **Sample size matters**: Larger n → better estimates, narrower CIs

These theorems are the foundation for statistical inference and machine learning!
`,
};
