/**
 * Expectation & Variance Section
 */

export const expectationvarianceSection = {
  id: 'expectation-variance',
  title: 'Expectation & Variance',
  content: `# Expectation & Variance

## Introduction

**Expectation** (mean) and **variance** are the two most important summary statistics of a probability distribution. They characterize the "center" and "spread" of a distribution.

**In ML**: Model predictions are expectations, loss functions are expectations, and understanding variance is crucial for bias-variance tradeoff.

## Expectation (Expected Value)

The **expected value** E[X] is the average value of a random variable.

**Discrete**: \\[ E[X] = \\sum_{x} x \\cdot P(X=x) \\]

**Continuous**: \\[ E[X] = \\int_{-\\infty}^{\\infty} x \\cdot f_X(x) dx \\]

**Interpretation**: Long-run average if you repeat the experiment infinitely many times.

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def expectation_demo():
    """Demonstrate expectation"""
    
    # Discrete: Die roll
    outcomes = np.array([1, 2, 3, 4, 5, 6])
    probabilities = np.array([1/6] * 6)
    
    theoretical_mean = np.sum(outcomes * probabilities)
    
    print("=== Expectation ===")
    print("Fair die roll:")
    print(f"E[X] = Σ x·P(X=x) = {theoretical_mean}")
    
    # Simulate
    np.random.seed(42)
    sample_sizes = [10, 100, 1000, 10000, 100000]
    
    print("\\nLaw of Large Numbers in action:")
    print("Sample Size | Sample Mean | Error")
    print("-" * 40)
    
    for n in sample_sizes:
        rolls = np.random.choice(outcomes, size=n)
        sample_mean = rolls.mean()
        error = abs(sample_mean - theoretical_mean)
        print(f"{n:11d} | {sample_mean:11.4f} | {error:.6f}")
    
    # Continuous: Normal
    mu, sigma = 5, 2
    normal = stats.norm(mu, sigma)
    
    samples = normal.rvs(size=10000)
    
    print(f"\\nContinuous (Normal): E[X] = μ = {mu}")
    print(f"Empirical mean: {samples.mean():.4f}")

expectation_demo()
\`\`\`

## Properties of Expectation

### Linearity

\\[ E[aX + b] = aE[X] + b \\]
\\[ E[X + Y] = E[X] + E[Y] \\text{ (always, even if dependent!)} \\]

\`\`\`python
def expectation_properties():
    """Demonstrate expectation properties"""
    
    np.random.seed(42)
    X = np.random.normal(10, 2, size=10000)
    Y = np.random.normal(5, 1, size=10000)
    
    a, b = 3, 5
    
    print("=== Linearity of Expectation ===")
    print(f"E[X] = {X.mean():.4f}")
    print(f"E[Y] = {Y.mean():.4f}")
    print()
    
    # E[aX + b] = aE[X] + b
    Z1 = a*X + b
    print(f"E[{a}X + {b}] = {Z1.mean():.4f}")
    print(f"{a}·E[X] + {b} = {a*X.mean() + b:.4f}")
    print()
    
    # E[X + Y] = E[X] + E[Y]
    Z2 = X + Y
    print(f"E[X + Y] = {Z2.mean():.4f}")
    print(f"E[X] + E[Y] = {X.mean() + Y.mean():.4f}")
    print()
    print("Linearity holds even for dependent variables!")

expectation_properties()
\`\`\`

## Variance

**Variance** measures spread around the mean:

\\[ \\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2 \\]

**Standard deviation**: \\( \\sigma = \\sqrt{\\text{Var}(X)} \\)

**Discrete**: \\[ \\text{Var}(X) = \\sum_{x} (x - \\mu)^2 \\cdot P(X=x) \\]

**Continuous**: \\[ \\text{Var}(X) = \\int_{-\\infty}^{\\infty} (x - \\mu)^2 \\cdot f_X(x) dx \\]

\`\`\`python
def variance_demo():
    """Demonstrate variance"""
    
    np.random.seed(42)
    
    # Two distributions with same mean, different variance
    mean = 10
    samples1 = np.random.normal(mean, 1, size=10000)  # Small variance
    samples2 = np.random.normal(mean, 5, size=10000)  # Large variance
    
    print("=== Variance ===")
    print(f"Both distributions have E[X] = {mean}")
    print()
    print(f"Small spread: Var(X) = {samples1.var():.4f}, σ = {samples1.std():.4f}")
    print(f"Large spread: Var(X) = {samples2.var():.4f}, σ = {samples2.std():.4f}")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.hist(samples1, bins=50, density=True, alpha=0.7, label='σ=1')
    ax1.hist(samples2, bins=50, density=True, alpha=0.7, label='σ=5')
    ax1.axvline(mean, color='r', linestyle='--', linewidth=2, label='Mean')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Density')
    ax1.set_title('Same Mean, Different Variance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plots
    ax2.boxplot([samples1, samples2], labels=['σ=1', 'σ=5'])
    ax2.set_ylabel('Value')
    ax2.set_title('Spread Comparison')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()

variance_demo()
\`\`\`

## Properties of Variance

### Variance of Linear Transformation

\\[ \\text{Var}(aX + b) = a^2 \\text{Var}(X) \\]

Note: **Constant b doesn't affect variance** (just shifts the distribution).

### Variance of Sum (Independent)

For **independent** X and Y:
\\[ \\text{Var}(X + Y) = \\text{Var}(X) + \\text{Var}(Y) \\]

**Not true for dependent variables!**

\`\`\`python
def variance_properties():
    """Demonstrate variance properties"""
    
    np.random.seed(42)
    X = np.random.normal(0, 2, size=10000)
    
    a, b = 3, 5
    
    print("=== Properties of Variance ===")
    print(f"Var(X) = {X.var():.4f}")
    print()
    
    # Var(aX + b) = a²Var(X)
    Y = a*X + b
    print(f"Var({a}X + {b}) = {Y.var():.4f}")
    print(f"{a}²·Var(X) = {a**2 * X.var():.4f}")
    print(f"Note: Constant {b} doesn't affect variance")
    print()
    
    # Variance of sum (independent)
    X1 = np.random.normal(0, 2, size=10000)
    X2 = np.random.normal(0, 3, size=10000)
    Sum = X1 + X2
    
    print("Independent variables:")
    print(f"Var(X1) = {X1.var():.4f}")
    print(f"Var(X2) = {X2.var():.4f}")
    print(f"Var(X1 + X2) = {Sum.var():.4f}")
    print(f"Var(X1) + Var(X2) = {X1.var() + X2.var():.4f}")

variance_properties()
\`\`\`

## Covariance (Review)

\\[ \\text{Cov}(X, Y) = E[(X - E[X])(Y - E[Y])] \\]

**General variance formula**:
\\[ \\text{Var}(X + Y) = \\text{Var}(X) + \\text{Var}(Y) + 2\\text{Cov}(X,Y) \\]

\`\`\`python
def covariance_variance_demo():
    """Show relationship between covariance and variance"""
    
    np.random.seed(42)
    X = np.random.normal(0, 1, size=10000)
    
    # Positive correlation
    Y_pos = 0.8*X + np.random.normal(0, 0.5, size=10000)
    
    # Negative correlation
    Y_neg = -0.8*X + np.random.normal(0, 0.5, size=10000)
    
    print("=== Variance of Sum with Covariance ===")
    print("Var(X + Y) = Var(X) + Var(Y) + 2Cov(X,Y)")
    print()
    
    for Y, label in [(Y_pos, "Positive corr"), (Y_neg, "Negative corr")]:
        Sum = X + Y
        cov = np.cov(X, Y)[0,1]
        
        var_sum_empirical = Sum.var()
        var_sum_formula = X.var() + Y.var() + 2*cov
        
        print(f"{label}:")
        print(f"  Cov(X,Y) = {cov:.4f}")
        print(f"  Var(X+Y) empirical = {var_sum_empirical:.4f}")
        print(f"  Var(X) + Var(Y) + 2Cov(X,Y) = {var_sum_formula:.4f}")
        print()

covariance_variance_demo()
\`\`\`

## ML Applications

### Loss as Expectation

\`\`\`python
def loss_expectation_demo():
    """Loss functions are expectations"""
    
    print("=== Loss Functions as Expectations ===")
    print()
    print("Mean Squared Error (MSE):")
    print("  MSE = E[(y - ŷ)²]")
    print("  Expectation over the data distribution")
    print()
    print("Cross-Entropy Loss:")
    print("  L = -E[y log(ŷ)]")
    print("  Expectation of negative log-likelihood")
    print()
    print("Why this matters:")
    print("- We minimize expected loss, not loss on one sample")
    print("- Training set gives empirical estimate: L̂ = (1/n)Σ loss_i")
    print("- True risk: E_P(x,y)[loss]")
    print("- Empirical risk: (1/n)Σ loss(x_i, y_i)")

loss_expectation_demo()
\`\`\`

### Bias-Variance Tradeoff

Expected prediction error decomposes into:

\\[ E[(y - \\hat{y})^2] = \\text{Bias}^2 + \\text{Variance} + \\text{Irreducible Error} \\]

\`\`\`python
def bias_variance_demo():
    """Demonstrate bias-variance tradeoff"""
    
    print("=== Bias-Variance Tradeoff ===")
    print()
    print("Bias: How far predictions are from true values (systematic error)")
    print("Variance: How much predictions vary for different training sets")
    print()
    print("Simple model (underfitting):")
    print("  High bias, Low variance")
    print("  Consistently wrong")
    print()
    print("Complex model (overfitting):")
    print("  Low bias, High variance")
    print("  Fits training data well, varies wildly on test data")
    print()
    print("Goal: Balance bias and variance for minimum total error")

bias_variance_demo()
\`\`\`

## Key Takeaways

1. **Expectation E[X]**: Long-run average, center of distribution
2. **Variance Var(X)**: Spread around mean, E[(X-μ)²]
3. **Linearity of expectation**: E[X+Y] = E[X] + E[Y] always
4. **Variance NOT linear**: Var(X+Y) = Var(X) + Var(Y) only if independent
5. **Var(aX+b) = a²Var(X)**: Scaling squares variance, shift doesn't affect it
6. **ML losses**: Expectations over data distribution
7. **Bias-variance**: Fundamental tradeoff in model complexity

Expectation and variance are foundational to all of statistics and machine learning!
`,
};
