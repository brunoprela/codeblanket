/**
 * Joint & Marginal Distributions Section
 */

export const jointmarginaldistributionsSection = {
  id: 'joint-marginal-distributions',
  title: 'Joint & Marginal Distributions',
  content: `# Joint & Marginal Distributions

## Introduction

When dealing with multiple random variables simultaneously, we need joint distributions. This is fundamental to ML where we model relationships between features and labels, or between multiple features.

**Key Concepts:**
- Joint distribution: P(X, Y) - probability of both variables
- Marginal distribution: P(X) from P(X, Y) 
- Conditional distribution: P(X|Y)
- Independence: P(X, Y) = P(X)P(Y)
- Covariance and correlation

## Joint Probability Distribution

For discrete RVs X and Y:
\\[ P(X=x, Y=y) = P(X=x \\cap Y=y) \\]

For continuous RVs:
\\[ f_{X,Y}(x,y) \\text{ (joint PDF)} \\]

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def joint_distribution_demo():
    """Demonstrate joint distribution"""
    
    # Create bivariate normal
    mean = [0, 0]
    cov = [[1, 0.8], [0.8, 1]]  # Correlated
    
    np.random.seed(42)
    samples = np.random.multivariate_normal(mean, cov, size=5000)
    X, Y = samples[:, 0], samples[:, 1]
    
    print("=== Joint Distribution ===")
    print("Bivariate Normal: X, Y ~ N(μ=[0,0], Σ=[[1,0.8],[0.8,1]])")
    print(f"Sample correlation: {np.corrcoef(X, Y)[0,1]:.3f}")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Scatter plot
    axes[0].scatter(X, Y, alpha=0.3, s=1)
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_title('Joint Distribution: (X, Y)')
    axes[0].grid(True, alpha=0.3)
    
    # 2D histogram
    axes[1].hist2d(X, Y, bins=50, cmap='Blues')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].set_title('2D Histogram (Joint Density)')
    
    # Contour plot
    from scipy.stats import multivariate_normal
    x_range = np.linspace(-3, 3, 100)
    y_range = np.linspace(-3, 3, 100)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    pos = np.dstack((X_grid, Y_grid))
    rv = multivariate_normal(mean, cov)
    axes[2].contour(X_grid, Y_grid, rv.pdf(pos), levels=10)
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    axes[2].set_title('Contour Plot (Joint PDF)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()

joint_distribution_demo()
\`\`\`

## Marginal Distributions

**Marginal distribution**: Distribution of one variable regardless of the other.

**Discrete**: \\( P(X=x) = \\sum_y P(X=x, Y=y) \\)

**Continuous**: \\( f_X(x) = \\int_{-\\infty}^{\\infty} f_{X,Y}(x,y) dy \\)

\`\`\`python
def marginal_distribution_demo():
    """Demonstrate marginal distributions"""
    
    # Joint distribution
    np.random.seed(42)
    mean = [2, -1]
    cov = [[2, 1], [1, 3]]
    samples = np.random.multivariate_normal(mean, cov, size=10000)
    X, Y = samples[:, 0], samples[:, 1]
    
    print("=== Marginal Distributions ===")
    print("From joint (X,Y), extract marginals P(X) and P(Y)")
    print()
    print(f"Theoretical marginals:")
    print(f"  X ~ N({mean[0]}, {cov[0][0]})")
    print(f"  Y ~ N({mean[1]}, {cov[1][1]})")
    print()
    print(f"Empirical:")
    print(f"  E[X] = {X.mean():.3f}, Var(X) = {X.var():.3f}")
    print(f"  E[Y] = {Y.mean():.3f}, Var(Y) = {Y.var():.3f}")
    
    # Plot
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Joint in center
    ax_joint = fig.add_subplot(gs[1:, :-1])
    ax_joint.scatter(X, Y, alpha=0.3, s=1)
    ax_joint.set_xlabel('X')
    ax_joint.set_ylabel('Y')
    ax_joint.set_title('Joint Distribution')
    
    # Marginal X on top
    ax_marg_x = fig.add_subplot(gs[0, :-1], sharex=ax_joint)
    ax_marg_x.hist(X, bins=50, density=True, alpha=0.7, edgecolor='black')
    x_range = np.linspace(X.min(), X.max(), 100)
    ax_marg_x.plot(x_range, stats.norm(mean[0], np.sqrt(cov[0][0])).pdf(x_range), 'r-', linewidth=2)
    ax_marg_x.set_ylabel('Density')
    ax_marg_x.set_title('Marginal P(X)')
    ax_marg_x.tick_params(labelbottom=False)
    
    # Marginal Y on right
    ax_marg_y = fig.add_subplot(gs[1:, -1], sharey=ax_joint)
    ax_marg_y.hist(Y, bins=50, density=True, alpha=0.7, edgecolor='black', orientation='horizontal')
    y_range = np.linspace(Y.min(), Y.max(), 100)
    ax_marg_y.plot(stats.norm(mean[1], np.sqrt(cov[1][1])).pdf(y_range), y_range, 'r-', linewidth=2)
    ax_marg_y.set_xlabel('Density')
    ax_marg_y.set_title('Marginal P(Y)')
    ax_marg_y.tick_params(labelleft=False)

marginal_distribution_demo()
\`\`\`

## Independence

X and Y are **independent** if:
\\[ P(X, Y) = P(X) \\times P(Y) \\text{ for all x, y} \\]

**Test for independence**: Check if knowing Y changes P(X).

\`\`\`python
def independence_demo():
    """Demonstrate independence vs dependence"""
    
    np.random.seed(42)
    n = 5000
    
    # Independent
    X_indep = np.random.normal(0, 1, n)
    Y_indep = np.random.normal(0, 1, n)
    
    # Dependent
    X_dep = np.random.normal(0, 1, n)
    Y_dep = 2*X_dep + np.random.normal(0, 0.5, n)
    
    print("=== Independence Test ===")
    print(f"Independent: Correlation = {np.corrcoef(X_indep, Y_indep)[0,1]:.3f}")
    print(f"Dependent: Correlation = {np.corrcoef(X_dep, Y_dep)[0,1]:.3f}")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.scatter(X_indep, Y_indep, alpha=0.3, s=1)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Independent: P(X,Y) = P(X)P(Y)')
    ax1.grid(True, alpha=0.3)
    
    ax2.scatter(X_dep, Y_dep, alpha=0.3, s=1)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Dependent: P(X,Y) ≠ P(X)P(Y)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()

independence_demo()
\`\`\`

## Covariance

Measures **linear relationship** between variables:

\\[ \\text{Cov}(X, Y) = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X]E[Y] \\]

**Properties**:
- Cov(X, Y) = 0 if independent (but not vice versa!)
- Cov(X, X) = Var(X)
- Units: product of X and Y units

\`\`\`python
def covariance_demo():
    """Demonstrate covariance"""
    
    np.random.seed(42)
    n = 1000
    
    # Different relationships
    X = np.random.normal(0, 1, n)
    Y_positive = 2*X + np.random.normal(0, 1, n)  # Positive relationship
    Y_negative = -2*X + np.random.normal(0, 1, n)  # Negative relationship
    Y_independent = np.random.normal(0, 1, n)  # Independent
    Y_nonlinear = X**2 + np.random.normal(0, 0.5, n)  # Nonlinear
    
    print("=== Covariance ===")
    print(f"Positive relationship: Cov(X,Y) = {np.cov(X, Y_positive)[0,1]:.3f}")
    print(f"Negative relationship: Cov(X,Y) = {np.cov(X, Y_negative)[0,1]:.3f}")
    print(f"Independent: Cov(X,Y) = {np.cov(X, Y_independent)[0,1]:.3f}")
    print(f"Nonlinear (X²): Cov(X,Y) = {np.cov(X, Y_nonlinear)[0,1]:.3f}")
    print()
    print("Note: Covariance detects linear relationships only!")
    print("Nonlinear relationship (X²) has near-zero covariance")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    data_sets = [
        (X, Y_positive, 'Positive Cov'),
        (X, Y_negative, 'Negative Cov'),
        (X, Y_independent, 'Zero Cov (Independent)'),
        (X, Y_nonlinear, 'Near-Zero Cov (Nonlinear)'),
    ]
    
    for ax, (x, y, title) in zip(axes, data_sets):
        ax.scatter(x, y, alpha=0.5, s=10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()

covariance_demo()
\`\`\`

## Correlation

**Correlation coefficient** (Pearson's r): Normalized covariance

\\[ \\rho_{X,Y} = \\frac{\\text{Cov}(X,Y)}{\\sigma_X \\sigma_Y} \\]

**Properties**:
- -1 ≤ ρ ≤ 1
- ρ = 1: Perfect positive linear relationship
- ρ = -1: Perfect negative linear relationship  
- ρ = 0: No linear relationship
- Scale-independent

\`\`\`python
def correlation_demo():
    """Demonstrate correlation"""
    
    np.random.seed(42)
    n = 1000
    X = np.random.normal(0, 1, n)
    
    # Different correlations
    correlations = [0.9, 0.5, 0, -0.5, -0.9]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    print("=== Correlation ===")
    print("ρ = Cov(X,Y) / (σ_X σ_Y)")
    print()
    
    for i, rho in enumerate(correlations):
        # Generate Y with specified correlation
        Y = rho * X + np.sqrt(1 - rho**2) * np.random.normal(0, 1, n)
        
        # Verify
        actual_corr = np.corrcoef(X, Y)[0, 1]
        
        # Plot
        ax = axes[i]
        ax.scatter(X, Y, alpha=0.3, s=5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'ρ = {rho:.1f} (actual: {actual_corr:.3f})')
        ax.grid(True, alpha=0.3)
        
        print(f"Target ρ = {rho:.1f}, Actual ρ = {actual_corr:.3f}")
    
    # Remove extra subplot
    fig.delaxes(axes[5])
    plt.tight_layout()

correlation_demo()
\`\`\`

## ML Applications

### Feature Correlation Analysis

\`\`\`python
from sklearn.datasets import load_boston
import seaborn as sns

def feature_correlation_ml():
    """Analyze feature correlations in ML"""
    
    # Load data (using diabetes as boston is deprecated)
    from sklearn.datasets import load_diabetes
    data = load_diabetes()
    X = data.data
    y = data.target
    
    # Compute correlation matrix
    import pandas as pd
    df = pd.DataFrame(X, columns=data.feature_names)
    df['target'] = y
    
    corr_matrix = df.corr()
    
    print("=== Feature Correlation in ML ===")
    print("Correlation with target:")
    print(corr_matrix['target'].sort_values(ascending=False))
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    
    print("\\nWhy this matters:")
    print("- Highly correlated features are redundant")
    print("- Can cause multicollinearity in linear models")
    print("- Feature selection: remove correlated features")
    print("- PCA automatically handles correlations")

feature_correlation_ml()
\`\`\`

## Key Takeaways

1. **Joint distribution**: P(X,Y) describes both variables together
2. **Marginal distribution**: P(X) from P(X,Y) by summing/integrating
3. **Independence**: P(X,Y) = P(X)P(Y)
4. **Covariance**: Measures linear relationship, unit-dependent
5. **Correlation**: Normalized covariance, -1 to 1
6. **Zero covariance ≠ independence**: Only for linear relationships
7. **ML application**: Feature correlation analysis critical

Understanding joint distributions is essential for multivariate machine learning!
`,
};
