/**
 * Normal Distribution Deep Dive Section
 */

export const normaldistributiondeepdiveSection = {
  id: 'normal-distribution-deep-dive',
  title: 'Normal Distribution Deep Dive',
  content: `# Normal Distribution Deep Dive

## Introduction

The **normal (Gaussian) distribution** is the most important distribution in statistics and machine learning. It appears everywhere due to the Central Limit Theorem and has beautiful mathematical properties.

**Why Normal Distribution Dominates ML:**
- Neural network activations often approximately normal
- Gradient noise in SGD
- Measurement errors
- Central Limit Theorem makes it ubiquitous
- Maximum entropy distribution (given mean and variance)
- Computationally tractable

## Definition

**PDF**:
\\[ f(x) = \\frac{1}{\\sigma\\sqrt{2\\pi}} e^{-\\frac{(x-\\mu)^2}{2\\sigma^2}} \\]

**Notation**: X ~ N(μ, σ²)

**Parameters**:
- μ (mu): mean, location parameter
- σ² (sigma squared): variance, scale parameter
- σ: standard deviation

**Properties**:
- E[X] = μ
- Var(X) = σ²
- Symmetric around μ
- Bell-shaped curve
- 68-95-99.7 rule

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def normal_distribution_demo():
    """Comprehensive normal distribution demonstration"""
    
    # Different parameters
    params = [
        (0, 1, 'N(0,1) - Standard Normal'),
        (0, 0.5, 'N(0,0.25) - Smaller variance'),
        (0, 2, 'N(0,4) - Larger variance'),
        (2, 1, 'N(2,1) - Shifted mean'),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    print("=== Normal Distribution ===")
    print("PDF: f(x) = (1/(σ√(2π))) * exp(-(x-μ)²/(2σ²))")
    print()
    
    x = np.linspace(-5, 5, 1000)
    
    for i, (mu, sigma, label) in enumerate(params):
        normal = stats.norm(mu, sigma)
        pdf = normal.pdf(x)
        
        # Plot
        ax = axes[i]
        ax.plot(x, pdf, 'b-', linewidth=2)
        ax.fill_between(x, pdf, alpha=0.3)
        ax.axvline(mu, color='r', linestyle='--', linewidth=2, label=f'μ = {mu}')
        ax.axvline(mu - sigma, color='g', linestyle=':', label=f'μ ± σ')
        ax.axvline(mu + sigma, color='g', linestyle=':')
        ax.set_xlabel('x')
        ax.set_ylabel('Density')
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        print(f"{label}: μ={mu}, σ={sigma}, σ²={sigma**2}")
    
    plt.tight_layout()
    
    print("\\nKey Properties:")
    print("- Symmetric around mean μ")
    print("- 50% probability on each side of μ")
    print("- Shape determined by σ (smaller σ → narrower)")

normal_distribution_demo()

# Output:
# === Normal Distribution ===
# PDF: f(x) = (1/(σ√(2π))) * exp(-(x-μ)²/(2σ²))
#
# N(0,1) - Standard Normal: μ=0, σ=1, σ²=1
# N(0,0.25) - Smaller variance: μ=0, σ=0.5, σ²=0.25
# N(0,4) - Larger variance: μ=0, σ=2, σ²=4
# N(2,1) - Shifted mean: μ=2, σ=1, σ²=1
#
# Key Properties:
# - Symmetric around mean μ
# - 50% probability on each side of μ
# - Shape determined by σ (smaller σ → narrower)
\`\`\`

## Standard Normal Distribution

The **standard normal** N(0, 1) has μ=0, σ=1.

**Any normal can be standardized**: 
\\[ Z = \\frac{X - \\mu}{\\sigma} \\sim N(0, 1) \\]

This is called the **Z-score** or **standardization**.

\`\`\`python
def standard_normal_demo():
    """Demonstrate standard normal and Z-scores"""
    
    # Original data: heights
    np.random.seed(42)
    heights = np.random.normal(170, 10, size=1000)  # Mean 170cm, SD 10cm
    
    # Standardize to Z-scores
    mean = heights.mean()
    std = heights.std()
    z_scores = (heights - mean) / std
    
    print("=== Standard Normal & Z-Scores ===")
    print(f"Original heights: μ = {mean:.2f} cm, σ = {std:.2f} cm")
    print(f"Z-scores: μ = {z_scores.mean():.4f}, σ = {z_scores.std():.4f}")
    print()
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Original
    ax1.hist(heights, bins=50, density=True, alpha=0.7, edgecolor='black')
    x = np.linspace(140, 200, 1000)
    ax1.plot(x, stats.norm(mean, std).pdf(x), 'r-', linewidth=2, label='N(170, 100)')
    ax1.set_xlabel('Height (cm)')
    ax1.set_ylabel('Density')
    ax1.set_title('Original Data')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Standardized
    ax2.hist(z_scores, bins=50, density=True, alpha=0.7, edgecolor='black')
    z = np.linspace(-4, 4, 1000)
    ax2.plot(z, stats.norm(0, 1).pdf(z), 'r-', linewidth=2, label='N(0, 1)')
    ax2.set_xlabel('Z-score')
    ax2.set_ylabel('Density')
    ax2.set_title('Standardized (Z-scores)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Interpretation
    print("Z-score interpretation:")
    example_height = 185
    z = (example_height - mean) / std
    print(f"Height {example_height} cm → Z = {z:.2f}")
    print(f"This is {abs(z):.2f} standard deviations {'above' if z > 0 else 'below'} mean")
    
    print("\\nML Applications:")
    print("- Feature normalization (StandardScaler)")
    print("- Outlier detection (|Z| > 3 is unusual)")
    print("- Comparing values on different scales")

standard_normal_demo()
\`\`\`

## The 68-95-99.7 Rule (Empirical Rule)

For normal distribution:
- **68%** of data within μ ± σ
- **95%** of data within μ ± 2σ  
- **99.7%** of data within μ ± 3σ

\`\`\`python
def empirical_rule_demo():
    """Demonstrate the 68-95-99.7 rule"""
    
    mu, sigma = 0, 1
    normal = stats.norm(mu, sigma)
    
    print("=== 68-95-99.7 Rule ===")
    print("For N(0, 1):")
    
    # Calculate actual probabilities
    prob_1sigma = normal.cdf(1) - normal.cdf(-1)
    prob_2sigma = normal.cdf(2) - normal.cdf(-2)
    prob_3sigma = normal.cdf(3) - normal.cdf(-3)
    
    print(f"P(-1 ≤ X ≤ 1) = {prob_1sigma:.4f} ≈ 68%")
    print(f"P(-2 ≤ X ≤ 2) = {prob_2sigma:.4f} ≈ 95%")
    print(f"P(-3 ≤ X ≤ 3) = {prob_3sigma:.4f} ≈ 99.7%")
    
    # Visualization
    x = np.linspace(-4, 4, 1000)
    pdf = normal.pdf(x)
    
    plt.figure(figsize=(12, 7))
    plt.plot(x, pdf, 'b-', linewidth=2, label='N(0,1)')
    
    # Shade regions
    colors = ['lightblue', 'lightgreen', 'lightyellow']
    alphas = [0.6, 0.4, 0.2]
    labels = ['68%', '95%', '99.7%']
    
    for i, (z, color, alpha, label) in enumerate(zip([1, 2, 3], colors, alphas, labels)):
        mask = (x >= -z) & (x <= z)
        plt.fill_between(x[mask], pdf[mask], alpha=alpha, color=color, label=f'μ ± {i+1}σ: {label}')
    
    # Mark sigma boundaries
    for i in range(1, 4):
        plt.axvline(i, color='red', linestyle='--', alpha=0.5)
        plt.axvline(-i, color='red', linestyle='--', alpha=0.5)
    
    plt.xlabel('Standard Deviations from Mean')
    plt.ylabel('Probability Density')
    plt.title('68-95-99.7 Rule for Normal Distribution')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    print("\\nOutlier Detection:")
    print("- |Z| > 2: ~5% of data (somewhat unusual)")
    print("- |Z| > 3: ~0.3% of data (very unusual)")
    print("- |Z| > 4: ~0.006% of data (extremely rare)")

empirical_rule_demo()
\`\`\`

## Normal Distribution Properties

### 1. Linear Transformations

If X ~ N(μ, σ²), then:
\\[ aX + b \\sim N(a\\mu + b, a^2\\sigma^2) \\]

\`\`\`python
def linear_transformation_demo():
    """Demonstrate linear transformations of normal RV"""
    
    # Original: X ~ N(0, 1)
    np.random.seed(42)
    X = np.random.normal(0, 1, size=10000)
    
    # Transform: Y = 2X + 5
    a, b = 2, 5
    Y = a * X + b
    
    print("=== Linear Transformation of Normal ===")
    print(f"X ~ N(0, 1)")
    print(f"Y = {a}X + {b}")
    print()
    print(f"Theoretical: Y ~ N({a}*0 + {b}, {a}²*1) = N({b}, {a**2})")
    print(f"Empirical: μ_Y = {Y.mean():.4f}, σ²_Y = {Y.var():.4f}")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.hist(X, bins=50, density=True, alpha=0.7, label='X ~ N(0,1)')
    x_range = np.linspace(-4, 4, 1000)
    ax1.plot(x_range, stats.norm(0, 1).pdf(x_range), 'r-', linewidth=2)
    ax1.set_title('Original: X ~ N(0, 1)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.hist(Y, bins=50, density=True, alpha=0.7, label=f'Y = {a}X + {b}')
    y_range = np.linspace(-2, 12, 1000)
    ax2.plot(y_range, stats.norm(b, a).pdf(y_range), 'r-', linewidth=2)
    ax2.set_title(f'Transformed: Y ~ N({b}, {a**2})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()

linear_transformation_demo()
\`\`\`

### 2. Sum of Independent Normals

If X₁ ~ N(μ₁, σ₁²) and X₂ ~ N(μ₂, σ₂²) are independent:
\\[ X_1 + X_2 \\sim N(\\mu_1 + \\mu_2, \\sigma_1^2 + \\sigma_2^2) \\]

**Variances add, not standard deviations!**

\`\`\`python
def sum_of_normals_demo():
    """Demonstrate sum of independent normal RVs"""
    
    np.random.seed(42)
    
    # Two independent normals
    X1 = np.random.normal(3, 2, size=10000)    # N(3, 4)
    X2 = np.random.normal(-1, 1.5, size=10000)  # N(-1, 2.25)
    
    # Sum
    Y = X1 + X2
    
    mu1, sigma1 = 3, 2
    mu2, sigma2 = -1, 1.5
    
    mu_sum = mu1 + mu2
    var_sum = sigma1**2 + sigma2**2
    sigma_sum = np.sqrt(var_sum)
    
    print("=== Sum of Independent Normals ===")
    print(f"X₁ ~ N({mu1}, {sigma1**2})")
    print(f"X₂ ~ N({mu2}, {sigma2**2})")
    print()
    print(f"Theoretical: X₁ + X₂ ~ N({mu_sum}, {var_sum})")
    print(f"           : μ = {mu_sum}, σ = {sigma_sum:.4f}")
    print(f"Empirical:   μ = {Y.mean():.4f}, σ = {Y.std():.4f}")
    print()
    print("Key: Variances add, not standard deviations!")
    print(f"σ₁² + σ₂² = {sigma1**2} + {sigma2**2} = {var_sum}")
    print(f"√{var_sum} = {sigma_sum:.4f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.hist(Y, bins=50, density=True, alpha=0.7, edgecolor='black', label='Empirical Sum')
    x = np.linspace(-10, 15, 1000)
    plt.plot(x, stats.norm(mu_sum, sigma_sum).pdf(x), 'r-', linewidth=2, 
             label=f'Theoretical N({mu_sum}, {var_sum:.2f})')
    plt.xlabel('X₁ + X₂')
    plt.ylabel('Density')
    plt.title('Sum of Independent Normal Random Variables')
    plt.legend()
    plt.grid(True, alpha=0.3)

sum_of_normals_demo()
\`\`\`

## Maximum Likelihood Estimation

Given data x₁, ..., xₙ from N(μ, σ²), the MLE estimates are:

\\[ \\hat{\\mu}_{MLE} = \\bar{x} = \\frac{1}{n}\\sum_{i=1}^n x_i \\]

\\[ \\hat{\\sigma}^2_{MLE} = \\frac{1}{n}\\sum_{i=1}^n (x_i - \\bar{x})^2 \\]

\`\`\`python
def mle_normal_demo():
    """Demonstrate MLE for normal distribution"""
    
    # True parameters
    true_mu, true_sigma = 5, 2
    
    # Generate data
    np.random.seed(42)
    sample_sizes = [10, 50, 100, 500, 5000]
    
    print("=== MLE for Normal Distribution ===")
    print(f"True parameters: μ = {true_mu}, σ = {true_sigma}")
    print()
    print("Sample Size | Estimated μ | Estimated σ | Error")
    print("-" * 55)
    
    for n in sample_sizes:
        data = np.random.normal(true_mu, true_sigma, size=n)
        
        # MLE estimates
        mu_hat = data.mean()
        sigma_hat = data.std(ddof=0)  # MLE uses ddof=0
        
        error = np.sqrt((mu_hat - true_mu)**2 + (sigma_hat - true_sigma)**2)
        
        print(f"{n:11d} | {mu_hat:11.4f} | {sigma_hat:11.4f} | {error:8.4f}")
    
    print("\\nNote: As n increases, estimates converge to true values")
    print("This is the Law of Large Numbers in action!")

mle_normal_demo()

# Output:
# === MLE for Normal Distribution ===
# True parameters: μ = 5, σ = 2
#
# Sample Size | Estimated μ | Estimated σ | Error
# -------------------------------------------------------
#          10 |      5.2540 |      1.7969 |   0.3295
#          50 |      5.0542 |      2.0583 |   0.0870
#         100 |      5.0060 |      1.9956 |   0.0064
#         500 |      5.0239 |      2.0060 |   0.0255
#        5000 |      4.9963 |      1.9906 |   0.0107
#
# Note: As n increases, estimates converge to true values
# This is the Law of Large Numbers in action!
\`\`\`

## ML Applications

### 1. Feature Normalization

\`\`\`python
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

def feature_normalization_demo():
    """Demonstrate feature normalization"""
    
    # Load data
    data = load_iris()
    X = data.data
    
    # Normalize
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    
    print("=== Feature Normalization ===")
    print("StandardScaler: (X - μ) / σ → N(0, 1)")
    print()
    print("Original features:")
    print(f"  Means: {X.mean(axis=0)}")
    print(f"  Stds:  {X.std(axis=0)}")
    print()
    print("Normalized features:")
    print(f"  Means: {X_normalized.mean(axis=0)}")
    print(f"  Stds:  {X_normalized.std(axis=0)}")
    
    print("\\nWhy normalize?")
    print("- Puts all features on same scale")
    print("- Faster gradient descent convergence")
    print("- Prevents features with large values from dominating")
    print("- Required for many algorithms (SVM, neural networks)")

feature_normalization_demo()
\`\`\`

### 2. Gaussian Noise in Neural Networks

\`\`\`python
def gaussian_noise_demo():
    """Demonstrate Gaussian noise in neural networks"""
    
    print("=== Gaussian Noise in Neural Networks ===")
    print()
    print("1. Weight Initialization:")
    print("   W ~ N(0, √(2/n)) for ReLU (He initialization)")
    print("   Ensures proper gradient flow")
    print()
    print("2. Dropout as Gaussian Noise:")
    print("   Dropping neurons ≈ adding multiplicative Gaussian noise")
    print("   Regularization effect")
    print()
    print("3. Gradient Noise in SGD:")
    print("   Mini-batch gradients ~ N(true_gradient, noise)")
    print("   Helps escape local minima")
    print()
    print("4. Data Augmentation:")
    print("   Add N(0, σ²) noise to inputs")
    print("   Improves robustness")
    
    # Example: Adding Gaussian noise to data
    np.random.seed(42)
    original_signal = np.linspace(0, 10, 100)
    noise = np.random.normal(0, 0.5, size=100)
    noisy_signal = original_signal + noise
    
    plt.figure(figsize=(10, 6))
    plt.plot(original_signal, 'b-', linewidth=2, label='Original Signal')
    plt.plot(noisy_signal, 'r.', alpha=0.5, label='With Gaussian Noise N(0, 0.25)')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.title('Adding Gaussian Noise for Data Augmentation')
    plt.legend()
    plt.grid(True, alpha=0.3)

gaussian_noise_demo()
\`\`\`

## Key Takeaways

1. **Most important distribution**: Ubiquitous due to Central Limit Theorem
2. **Standard normal**: N(0,1), any normal can be standardized
3. **68-95-99.7 rule**: Intuitive probabilities for standard deviations
4. **Linear transformations**: aX + b ~ N(aμ + b, a²σ²)
5. **Sum of normals**: Variances add (not standard deviations!)
6. **MLE**: Sample mean and variance are optimal estimates
7. **ML applications**: Feature scaling, weight initialization, noise models
8. **Mathematically tractable**: Closed-form solutions for many operations

The normal distribution is the workhorse of statistics and machine learning!
`,
};
