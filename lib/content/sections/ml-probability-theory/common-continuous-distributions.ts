/**
 * Common Continuous Distributions Section
 */

export const commoncontinuousdistributionsSection = {
  id: 'common-continuous-distributions',
  title: 'Common Continuous Distributions',
  content: `# Common Continuous Distributions

## Introduction

Continuous probability distributions model random variables that can take any value in an interval. They're fundamental to machine learning for:
- Regression problems (predicting continuous values)
- Neural network activations
- Bayesian inference
- Probability density estimation
- Uncertainty quantification

Unlike discrete distributions (PMF), continuous distributions use **Probability Density Functions (PDF)** where probabilities are computed as areas under curves.

## Uniform Distribution

The **uniform distribution** assigns equal probability density to all values in an interval [a, b].

**PDF**:
\\[ f (x) = \\begin{cases} \\frac{1}{b-a} & \\text{if } a \\leq x \\leq b \\\\ 0 & \\text{otherwise} \\end{cases} \\]

**Properties**:
- E[X] = (a + b) / 2
- Var(X) = (b - a)² / 12

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def uniform_demo():
    """Demonstrate Uniform distribution"""
    
    a, b = 2, 8  # Uniform on [2, 8]
    
    uniform = stats.uniform (loc=a, scale=b-a)
    
    print("=== Uniform Distribution ===")
    print(f"Parameters: a={a}, b={b}")
    print(f"E[X] = (a+b)/2 = {(a+b)/2}")
    print(f"Var(X) = (b-a)²/12 = {(b-a)**2/12:.4f}")
    
    # Sample
    np.random.seed(42)
    samples = uniform.rvs (size=10000)
    
    print(f"\\nEmpirical: E[X] = {samples.mean():.4f}, Var(X) = {samples.var():.4f}")
    
    # Plot
    x = np.linspace(0, 10, 1000)
    pdf = uniform.pdf (x)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # PDF
    ax1.plot (x, pdf, 'b-', linewidth=2, label='PDF')
    ax1.fill_between (x, pdf, alpha=0.3)
    ax1.set_xlabel('x')
    ax1.set_ylabel('Density')
    ax1.set_title (f'Uniform PDF (a={a}, b={b})')
    ax1.axvline (a, color='r', linestyle='--', label=f'a={a}')
    ax1.axvline (b, color='r', linestyle='--', label=f'b={b}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Histogram of samples
    ax2.hist (samples, bins=50, density=True, alpha=0.7, edgecolor='black')
    ax2.plot (x, pdf, 'r-', linewidth=2, label='Theoretical PDF')
    ax2.set_xlabel('x')
    ax2.set_ylabel('Density')
    ax2.set_title('Uniform Samples (n=10000)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    print("\\nML Applications:")
    print("- Random weight initialization (uniform)")
    print("- Random sampling in algorithms")
    print("- Uniformly distributed noise")

uniform_demo()
\`\`\`

## Exponential Distribution

Models the **time between events** in a Poisson process (time until next event).

**PDF**:
\\[ f (x) = \\lambda e^{-\\lambda x} \\text{ for } x \\geq 0 \\]

**Properties**:
- E[X] = 1/λ
- Var(X) = 1/λ²
- **Memoryless property** (like Geometric for discrete)

\`\`\`python
def exponential_demo():
    """Demonstrate Exponential distribution"""
    
    lambda_param = 2  # Rate parameter
    
    exponential = stats.expon (scale=1/lambda_param)
    
    print("=== Exponential Distribution ===")
    print(f"Parameter λ = {lambda_param}")
    print(f"E[X] = 1/λ = {1/lambda_param}")
    print(f"Var(X) = 1/λ² = {1/lambda_param**2}")
    
    # Sample
    np.random.seed(42)
    samples = exponential.rvs (size=10000)
    
    print(f"\\nEmpirical: E[X] = {samples.mean():.4f}, Var(X) = {samples.var():.4f}")
    
    # Plot
    x = np.linspace(0, 5, 1000)
    pdf = exponential.pdf (x)
    cdf = exponential.cdf (x)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # PDF
    ax1.plot (x, pdf, 'b-', linewidth=2, label='PDF')
    ax1.fill_between (x, pdf, alpha=0.3)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Density')
    ax1.set_title (f'Exponential PDF (λ={lambda_param})')
    ax1.axvline(1/lambda_param, color='r', linestyle='--', label=f'Mean = 1/λ = {1/lambda_param}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # CDF
    ax2.plot (x, cdf, 'r-', linewidth=2)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title('Exponential CDF')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0.632, color='g', linestyle='--', alpha=0.5, label='P(X ≤ 1/λ) ≈ 0.63')
    ax2.legend()
    
    plt.tight_layout()
    
    print("\\nMemoryless Property:")
    print("P(X > s+t | X > s) = P(X > t)")
    print("Waiting time doesn't depend on how long you've already waited!")
    
    print("\\nML Applications:")
    print("- Time until event (server response time)")
    print("- Survival analysis")
    print("- Reinforcement learning (time between events)")

exponential_demo()
\`\`\`

## Beta Distribution

Defined on interval [0, 1], commonly used for modeling **probabilities** themselves.

**PDF**:
\\[ f (x) = \\frac{x^{\\alpha-1}(1-x)^{\\beta-1}}{B(\\alpha,\\beta)} \\text{ for } 0 \\leq x \\leq 1 \\]

**Properties**:
- E[X] = α / (α + β)
- Very flexible shape with parameters α, β
- **Conjugate prior** for Binomial/Bernoulli

\`\`\`python
def beta_demo():
    """Demonstrate Beta distribution"""
    
    # Different parameter combinations
    params = [(0.5, 0.5), (2, 2), (2, 5), (5, 2)]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    print("=== Beta Distribution ===")
    print("PDF: f (x) = x^(α-1) * (1-x)^(β-1) / B(α,β)")
    print("\\nParameter effects:")
    
    x = np.linspace(0, 1, 1000)
    
    for i, (alpha, beta_param) in enumerate (params):
        beta_dist = stats.beta (alpha, beta_param)
        pdf = beta_dist.pdf (x)
        mean = alpha / (alpha + beta_param)
        
        # Plot
        ax = axes[i]
        ax.plot (x, pdf, 'b-', linewidth=2)
        ax.fill_between (x, pdf, alpha=0.3)
        ax.axvline (mean, color='r', linestyle='--', label=f'Mean = {mean:.3f}')
        ax.set_xlabel('x')
        ax.set_ylabel('Density')
        ax.set_title (f'Beta(α={alpha}, β={beta_param})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        print(f"  Beta({alpha}, {beta_param}): Mean = {mean:.3f}")
    
    plt.tight_layout()
    
    print("\\nShape interpretation:")
    print("- α, β < 1: U-shaped (extremes more likely)")
    print("- α = β = 1: Uniform")
    print("- α > 1, β > 1: Bell-shaped")
    print("- α > β: Skewed right (high values more likely)")
    print("- α < β: Skewed left (low values more likely)")
    
    print("\\nML Applications:")
    print("- Bayesian inference (conjugate prior for Bernoulli)")
    print("- Modeling probabilities/proportions")
    print("- A/B testing (updating conversion rate beliefs)")

beta_demo()
\`\`\`

## Gamma Distribution

Generalizes the exponential distribution, models **waiting time for k events**.

**PDF**:
\\[ f (x) = \\frac{\\beta^\\alpha x^{\\alpha-1} e^{-\\beta x}}{\\Gamma(\\alpha)} \\text{ for } x > 0 \\]

**Properties**:
- E[X] = α/β
- Var(X) = α/β²
- When α=1, reduces to Exponential(β)

\`\`\`python
def gamma_demo():
    """Demonstrate Gamma distribution"""
    
    # Different shapes
    shapes = [1, 2, 5, 10]
    rate = 2
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    print("=== Gamma Distribution ===")
    print(f"Rate β = {rate}")
    print("\\nShape parameter α effects:")
    
    x = np.linspace(0, 8, 1000)
    
    for alpha in shapes:
        gamma_dist = stats.gamma (alpha, scale=1/rate)
        pdf = gamma_dist.pdf (x)
        mean = alpha / rate
        
        # Plot PDF
        ax1.plot (x, pdf, linewidth=2, label=f'α={alpha}, Mean={mean}')
        
        print(f"  α={alpha:2d}: E[X] = {mean:.2f}, Var(X) = {alpha/rate**2:.2f}")
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('Density')
    ax1.set_title (f'Gamma PDF (β={rate})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Relationship to Exponential
    exponential = stats.expon (scale=1/rate)
    gamma_1 = stats.gamma(1, scale=1/rate)
    
    ax2.plot (x, exponential.pdf (x), 'b-', linewidth=2, label='Exponential(β=2)')
    ax2.plot (x, gamma_1.pdf (x), 'r--', linewidth=2, label='Gamma(α=1, β=2)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('Density')
    ax2.set_title('Gamma(α=1) = Exponential')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    print("\\nML Applications:")
    print("- Modeling waiting times")
    print("- Bayesian inference (conjugate prior for Poisson)")
    print("- Lifetime/survival analysis")

gamma_demo()
\`\`\`

## Chi-Squared Distribution

Special case of Gamma, used in **hypothesis testing** and **confidence intervals**.

**Definition**: Sum of squares of k independent standard normal variables

\\[ X = Z_1^2 + Z_2^2 + \\cdots + Z_k^2 \\text{ where } Z_i \\sim N(0,1) \\]

**Properties**:
- Parameter: k (degrees of freedom)
- E[X] = k
- Var(X) = 2k
- Chi-Squared (k) = Gamma (k/2, 1/2)

\`\`\`python
def chi_squared_demo():
    """Demonstrate Chi-Squared distribution"""
    
    # Different degrees of freedom
    dofs = [1, 2, 5, 10]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    print("=== Chi-Squared Distribution ===")
    print("Sum of k squared standard normals")
    print("\\nDegrees of freedom effects:")
    
    x = np.linspace(0, 20, 1000)
    
    for k in dofs:
        chi2 = stats.chi2(k)
        pdf = chi2.pdf (x)
        
        # Plot
        ax1.plot (x, pdf, linewidth=2, label=f'k={k}, E[X]={k}')
        
        print(f"  k={k:2d}: E[X] = {k}, Var(X) = {2*k}")
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('Density')
    ax1.set_title('Chi-Squared PDF')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Demonstration: sum of squared normals
    np.random.seed(42)
    k = 5
    n_samples = 10000
    
    # Generate k standard normals, square and sum
    samples = np.sum (np.random.randn (n_samples, k)**2, axis=1)
    
    # Compare with theoretical
    chi2_theoretical = stats.chi2(k)
    ax2.hist (samples, bins=50, density=True, alpha=0.7, edgecolor='black', label='Empirical')
    x_plot = np.linspace(0, 20, 1000)
    ax2.plot (x_plot, chi2_theoretical.pdf (x_plot), 'r-', linewidth=2, label='Theoretical χ²(5)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('Density')
    ax2.set_title('Chi-Squared from Sum of Squared Normals')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    print("\\nML Applications:")
    print("- Goodness-of-fit tests")
    print("- Independence tests")
    print("- Variance estimation")

chi_squared_demo()
\`\`\`

## Student\'s t-Distribution

Used for **small samples** and **hypothesis testing** when population variance is unknown.

**Properties**:
- Parameter: ν (degrees of freedom)
- Heavier tails than normal (more outliers)
- As ν → ∞, converges to standard normal

\`\`\`python
def t_distribution_demo():
    """Demonstrate Student's t-distribution"""
    
    # Different degrees of freedom
    dofs = [1, 2, 5, 30]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    print("=== Student's t-Distribution ===")
    print("Used for small samples (unknown population variance)")
    print("\\nDegrees of freedom effects:")
    
    x = np.linspace(-4, 4, 1000)
    normal = stats.norm(0, 1)
    
    # Plot t-distributions
    ax = axes[0]
    for nu in dofs:
        t_dist = stats.t (nu)
        pdf = t_dist.pdf (x)
        ax.plot (x, pdf, linewidth=2, label=f'ν={nu}')
        
        print(f"  ν={nu:2d}: {'Heavy tails' if nu < 10 else 'Approaching normal'}")
    
    # Add standard normal for comparison
    ax.plot (x, normal.pdf (x), 'k--', linewidth=2, label='Normal(0,1)')
    ax.set_xlabel('x')
    ax.set_ylabel('Density')
    ax.set_title('t-Distribution vs Normal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Tail comparison
    ax = axes[1]
    t_1 = stats.t(1)
    t_5 = stats.t(5)
    t_30 = stats.t(30)
    
    # Focus on tails
    x_tail = np.linspace(0, 4, 1000)
    ax.plot (x_tail, t_1.pdf (x_tail), linewidth=2, label='t(ν=1)')
    ax.plot (x_tail, t_5.pdf (x_tail), linewidth=2, label='t(ν=5)')
    ax.plot (x_tail, t_30.pdf (x_tail), linewidth=2, label='t(ν=30)')
    ax.plot (x_tail, normal.pdf (x_tail), 'k--', linewidth=2, label='Normal')
    ax.set_xlabel('x')
    ax.set_ylabel('Density')
    ax.set_title('Tail Comparison (Right Tail)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    print("\\nKey insight: t-distribution has heavier tails")
    print("More probability in extremes → more robust to outliers")
    
    print("\\nML Applications:")
    print("- Confidence intervals with small samples")
    print("- t-tests for comparing means")
    print("- Robust estimation")

t_distribution_demo()
\`\`\`

## Distribution Comparison Summary

\`\`\`python
def compare_continuous_distributions():
    """Summary comparison of continuous distributions"""
    
    print("=== Continuous Distribution Summary ===\\n")
    
    distributions = [
        ("Uniform", "U(a,b)", "Equal density on interval", "Random initialization"),
        ("Exponential", "Exp(λ)", "Time to next event", "Waiting times"),
        ("Beta", "Beta(α,β)", "Probabilities on [0,1]", "Bayesian priors"),
        ("Gamma", "Gamma(α,β)", "Time for k events", "Waiting times"),
        ("Chi-Squared", "χ²(k)", "Sum of squared normals", "Hypothesis tests"),
        ("t", "t(ν)", "Heavy-tailed normal", "Small sample inference"),
    ]
    
    print(f"{'Distribution':<15} {'Notation':<12} {'Description':<30} {'ML Use':<20}")
    print("-" * 85)
    for name, notation, desc, use in distributions:
        print(f"{name:<15} {notation:<12} {desc:<30} {use:<20}")
    
    print("\\nKey Relationships:")
    print("- Exponential = Gamma(α=1)")
    print("- Chi-Squared = Gamma (k/2, 1/2)")
    print("- t-distribution → Normal as ν → ∞")
    print("- Beta(1,1) = Uniform(0,1)")

compare_continuous_distributions()
\`\`\`

## Key Takeaways

1. **Uniform**: Constant density on interval, for random sampling
2. **Exponential**: Time between events, memoryless property
3. **Beta**: Models probabilities themselves, Bayesian prior
4. **Gamma**: Generalizes exponential, sum of waiting times
5. **Chi-Squared**: Sum of squared normals, hypothesis testing
6. **t-Distribution**: Heavy tails for small samples
7. **Choose wisely**: Match distribution to problem structure
8. **Continuous vs Discrete**: PDF vs PMF, areas vs points

Understanding these distributions is essential for statistical modeling and Bayesian inference in ML!
`,
};
