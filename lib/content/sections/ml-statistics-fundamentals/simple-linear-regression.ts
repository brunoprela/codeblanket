/**
 * Simple Linear Regression Section
 */

export const simplelinearregressionSection = {
  id: 'simple-linear-regression',
  title: 'Simple Linear Regression',
  content: `# Simple Linear Regression

## Introduction

Linear regression is one of the most fundamental and widely used techniques in statistics and machine learning. It models the relationship between:
- **One predictor variable** (X, independent variable, feature)
- **One response variable** (Y, dependent variable, target)

As a linear relationship: \\( Y = \\beta_0 + \\beta_1 X + \\epsilon \\)

**Applications in ML**:
- Baseline model for comparison
- Feature importance assessment
- Understanding linear relationships
- Foundation for more complex models
- Interpretable predictions

## The Linear Model

\\[ y = \\beta_0 + \\beta_1 x + \\epsilon \\]

Where:
- \\(\\beta_0\\): Intercept (value when x=0)
- \\(\\beta_1\\): Slope (change in y per unit change in x)
- \\(\\epsilon\\): Error term (residual)

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

np.random.seed(42)

def visualize_linear_relationship():
    """Demonstrate simple linear regression"""
    
    # Generate data
    n = 100
    X = np.random.rand (n) * 10
    true_slope = 2.5
    true_intercept = 5
    noise = np.random.normal(0, 2, n)
    y = true_intercept + true_slope * X + noise
    
    # Fit regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
    
    # Predictions
    y_pred = intercept + slope * X
    residuals = y - y_pred
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Scatter with regression line
    axes[0].scatter(X, y, alpha=0.6, s=50, label='Data')
    axes[0].plot(X, y_pred, 'r-', linewidth=2, label=f'y = {intercept:.2f} + {slope:.2f}x')
    axes[0].plot(X, true_intercept + true_slope * X, 'g--', linewidth=2, 
                 label=f'True: y = {true_intercept} + {true_slope}x')
    axes[0].set_xlabel('X', fontsize=12)
    axes[0].set_ylabel('y', fontsize=12)
    axes[0].set_title (f'Linear Regression (R² = {r_value**2:.3f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Residuals vs fitted
    axes[1].scatter (y_pred, residuals, alpha=0.6, s=50)
    axes[1].axhline(0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Fitted Values', fontsize=12)
    axes[1].set_ylabel('Residuals', fontsize=12)
    axes[1].set_title('Residual Plot')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Residual histogram
    axes[2].hist (residuals, bins=20, edgecolor='black', alpha=0.7)
    axes[2].axvline(0, color='r', linestyle='--', linewidth=2)
    axes[2].set_xlabel('Residuals', fontsize=12)
    axes[2].set_ylabel('Frequency', fontsize=12)
    axes[2].set_title('Residual Distribution')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('simple_linear_regression.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("=== Simple Linear Regression ===")
    print(f"Estimated equation: y = {intercept:.4f} + {slope:.4f}x")
    print(f"True equation: y = {true_intercept} + {true_slope}x")
    print(f"\\nR² = {r_value**2:.4f} ({r_value**2*100:.1f}% variance explained)")
    print(f"RMSE = {np.sqrt (mean_squared_error (y, y_pred)):.4f}")
    print(f"p-value: {p_value:.6f} (highly significant)")

visualize_linear_relationship()
\`\`\`

## Ordinary Least Squares (OLS)

**Goal**: Minimize the sum of squared residuals

\\[ \\text{minimize} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2 = \\sum_{i=1}^{n} (y_i - \\beta_0 - \\beta_1 x_i)^2 \\]

**Closed-form solution**:

\\[ \\beta_1 = \\frac{\\sum (x_i - \\bar{x})(y_i - \\bar{y})}{\\sum (x_i - \\bar{x})^2} \\]

\\[ \\beta_0 = \\bar{y} - \\beta_1 \\bar{x} \\]

\`\`\`python
def ols_from_scratch(X, y):
    """Implement OLS regression from scratch"""
    
    # Calculate means
    x_mean = np.mean(X)
    y_mean = np.mean (y)
    
    # Calculate slope (beta_1)
    numerator = np.sum((X - x_mean) * (y - y_mean))
    denominator = np.sum((X - x_mean) ** 2)
    beta_1 = numerator / denominator
    
    # Calculate intercept (beta_0)
    beta_0 = y_mean - beta_1 * x_mean
    
    # Predictions
    y_pred = beta_0 + beta_1 * X
    
    # Residuals
    residuals = y - y_pred
    
    # Standard error of residuals
    n = len(X)
    rss = np.sum (residuals ** 2)
    sigma_squared = rss / (n - 2)  # degrees of freedom = n - 2
    
    # Standard errors of coefficients
    se_beta_1 = np.sqrt (sigma_squared / np.sum((X - x_mean) ** 2))
    se_beta_0 = np.sqrt (sigma_squared * (1/n + x_mean**2 / np.sum((X - x_mean) ** 2)))
    
    # t-statistics
    t_beta_1 = beta_1 / se_beta_1
    t_beta_0 = beta_0 / se_beta_0
    
    # p-values (two-tailed)
    p_beta_1 = 2 * (1 - stats.t.cdf (abs (t_beta_1), df=n-2))
    p_beta_0 = 2 * (1 - stats.t.cdf (abs (t_beta_0), df=n-2))
    
    # R-squared
    ss_total = np.sum((y - y_mean) ** 2)
    ss_residual = rss
    r_squared = 1 - (ss_residual / ss_total)
    
    return {
        'beta_0': beta_0,
        'beta_1': beta_1,
        'se_beta_0': se_beta_0,
        'se_beta_1': se_beta_1,
        't_beta_0': t_beta_0,
        't_beta_1': t_beta_1,
        'p_beta_0': p_beta_0,
        'p_beta_1': p_beta_1,
        'r_squared': r_squared,
        'residuals': residuals,
        'predictions': y_pred
    }

# Example
X = np.random.rand(100) * 10
y = 5 + 2.5 * X + np.random.normal(0, 2, 100)

results = ols_from_scratch(X, y)

print("=== OLS Results (From Scratch) ===")
print(f"Intercept (β₀): {results['beta_0']:.4f} (SE: {results['se_beta_0']:.4f}, p: {results['p_beta_0']:.6f})")
print(f"Slope (β₁): {results['beta_1']:.4f} (SE: {results['se_beta_1']:.4f}, p: {results['p_beta_1']:.6f})")
print(f"R²: {results['r_squared']:.4f}")

# Compare with scipy
slope_scipy, intercept_scipy, r_scipy, p_scipy, se_scipy = stats.linregress(X, y)
print(f"\\nScipy verification:")
print(f"Slope: {slope_scipy:.4f}, Intercept: {intercept_scipy:.4f}, R²: {r_scipy**2:.4f}")
\`\`\`

## R-Squared (Coefficient of Determination)

\\[ R^2 = 1 - \\frac{SS_{\\text{residual}}}{SS_{\\text{total}}} = 1 - \\frac{\\sum (y_i - \\hat{y}_i)^2}{\\sum (y_i - \\bar{y})^2} \\]

**Interpretation**:
- R² = 0: Model explains 0% of variance (useless)
- R² = 0.5: Model explains 50% of variance
- R² = 1: Model explains 100% of variance (perfect fit)

\`\`\`python
def demonstrate_r_squared():
    """Show different R² values"""
    
    n = 100
    X = np.random.rand (n) * 10
    
    # Create data with different noise levels
    scenarios = [
        ("High R² (0.95)", 2*X + np.random.normal(0, 0.5, n)),
        ("Medium R² (0.7)", 2*X + np.random.normal(0, 2, n)),
        ("Low R² (0.3)", 2*X + np.random.normal(0, 4, n)),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (title, y) in enumerate (scenarios):
        slope, intercept, r_value, _, _ = stats.linregress(X, y)
        y_pred = intercept + slope * X
        
        axes[idx].scatter(X, y, alpha=0.6, s=30)
        axes[idx].plot(X, y_pred, 'r-', linewidth=2)
        axes[idx].set_title (f'{title}\\nR² = {r_value**2:.3f}')
        axes[idx].set_xlabel('X')
        axes[idx].set_ylabel('y')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('r_squared_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

demonstrate_r_squared()
\`\`\`

## Inference for Regression

### Hypothesis Testing for Slope

**H₀**: β₁ = 0 (no relationship)
**H₁**: β₁ ≠ 0 (relationship exists)

\`\`\`python
def regression_inference():
    """Complete inference for regression"""
    
    # Generate data
    n = 50
    X = np.random.rand (n) * 10
    y = 5 + 2 * X + np.random.normal(0, 2, n)
    
    # Fit model
    slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
    
    # Confidence intervals
    alpha = 0.05
    t_crit = stats.t.ppf(1 - alpha/2, df=n-2)
    
    ci_slope = (slope - t_crit * std_err, slope + t_crit * std_err)
    
    print("=== Regression Inference ===")
    print(f"Sample size: n = {n}")
    print(f"\\nSlope (β₁):")
    print(f"  Estimate: {slope:.4f}")
    print(f"  Standard error: {std_err:.4f}")
    print(f"  95% CI: [{ci_slope[0]:.4f}, {ci_slope[1]:.4f}]")
    print(f"  t-statistic: {slope/std_err:.4f}")
    print(f"  p-value: {p_value:.6f}")
    print()
    
    if p_value < 0.05:
        print("✓ Slope is significantly different from zero (p < 0.05)")
        print("  → There is a significant linear relationship between X and y")
    else:
        print("✗ Slope is not significantly different from zero")
        print("  → No evidence of linear relationship")
    
    # Visualize confidence bands
    X_range = np.linspace(X.min(), X.max(), 100)
    y_pred = intercept + slope * X_range
    
    # Standard error of prediction
    se_pred = std_err * np.sqrt(1 + 1/n + (X_range - X.mean())**2 / np.sum((X - X.mean())**2))
    ci_lower = y_pred - t_crit * se_pred
    ci_upper = y_pred + t_crit * se_pred
    
    plt.figure (figsize=(10, 6))
    plt.scatter(X, y, alpha=0.6, s=50, label='Data')
    plt.plot(X_range, y_pred, 'r-', linewidth=2, label='Regression line')
    plt.fill_between(X_range, ci_lower, ci_upper, alpha=0.2, color='red', 
                      label='95% Prediction interval')
    plt.xlabel('X', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Regression with Prediction Interval', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('regression_inference.png', dpi=300, bbox_inches='tight')
    plt.show()

regression_inference()
\`\`\`

## Prediction vs Confidence Intervals

**Confidence Interval**: Uncertainty in the mean response
**Prediction Interval**: Uncertainty in individual predictions (wider!)

\`\`\`python
def compare_intervals():
    """Compare confidence and prediction intervals"""
    
    n = 30
    X = np.random.rand (n) * 10
    y = 5 + 2 * X + np.random.normal(0, 2, n)
    
    # Fit model
    slope, intercept, _, _, std_err = stats.linregress(X, y)
    
    # Prediction range
    X_new = np.linspace(X.min(), X.max(), 100)
    y_pred = intercept + slope * X_new
    
    # Residual standard error
    residuals = y - (intercept + slope * X)
    rss = np.sum (residuals ** 2)
    sigma = np.sqrt (rss / (n - 2))
    
    # Standard errors
    se_mean = sigma * np.sqrt(1/n + (X_new - X.mean())**2 / np.sum((X - X.mean())**2))
    se_pred = sigma * np.sqrt(1 + 1/n + (X_new - X.mean())**2 / np.sum((X - X.mean())**2))
    
    # Critical value
    t_crit = stats.t.ppf(0.975, df=n-2)
    
    # Intervals
    ci_lower = y_pred - t_crit * se_mean
    ci_upper = y_pred + t_crit * se_mean
    pi_lower = y_pred - t_crit * se_pred
    pi_upper = y_pred + t_crit * se_pred
    
    # Plot
    plt.figure (figsize=(12, 6))
    plt.scatter(X, y, alpha=0.6, s=50, label='Data', zorder=3)
    plt.plot(X_new, y_pred, 'r-', linewidth=2, label='Regression line', zorder=2)
    plt.fill_between(X_new, ci_lower, ci_upper, alpha=0.3, color='blue',
                      label='95% Confidence interval (for mean)', zorder=1)
    plt.fill_between(X_new, pi_lower, pi_upper, alpha=0.2, color='red',
                      label='95% Prediction interval (for individuals)', zorder=0)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Confidence vs Prediction Intervals', fontsize=14, fontweight='bold')
    plt.legend (fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('confidence_vs_prediction.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("=== Confidence vs Prediction Intervals ===")
    print("Confidence Interval (CI):")
    print("  • For the MEAN response at a given X")
    print("  • Narrower (less uncertainty)")
    print("  • Answers: "What\'s the average y for this X?"")
    print()
    print("Prediction Interval (PI):")
    print("  • For an INDIVIDUAL response at a given X")
    print("  • Wider (more uncertainty)")
    print("  • Answers: "What's a plausible y for this specific X?"")
    print()
    print("PI is always wider than CI because:")
    print("  PI uncertainty = CI uncertainty + Individual variation")

compare_intervals()
\`\`\`

## Residual Analysis

\`\`\`python
def comprehensive_residual_analysis(X, y):
    """Complete residual diagnostics"""
    
    # Fit model
    slope, intercept, r_value, _, _ = stats.linregress(X, y)
    y_pred = intercept + slope * X
    residuals = y - y_pred
    
    # Standardized residuals
    rss = np.sum (residuals ** 2)
    sigma = np.sqrt (rss / (len(X) - 2))
    standardized_residuals = residuals / sigma
    
    # Create diagnostic plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Residuals vs Fitted
    axes[0, 0].scatter (y_pred, residuals, alpha=0.6)
    axes[0, 0].axhline(0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted\\n(Check for patterns)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Q-Q Plot
    stats.probplot (residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot\\n(Check normality)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Scale-Location
    axes[1, 0].scatter (y_pred, np.sqrt (np.abs (standardized_residuals)), alpha=0.6)
    axes[1, 0].set_xlabel('Fitted Values')
    axes[1, 0].set_ylabel('√|Standardized Residuals|')
    axes[1, 0].set_title('Scale-Location\\n(Check homoscedasticity)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Residuals Histogram
    axes[1, 1].hist (residuals, bins=20, edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(0, color='r', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Residuals')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Residual Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('residual_diagnostics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Statistical tests
    print("=== Residual Diagnostics ===")
    print(f"R² = {r_value**2:.4f}")
    print(f"RMSE = {np.sqrt (np.mean (residuals**2)):.4f}")
    print()
    
    # Normality test
    _, p_norm = stats.shapiro (residuals)
    print(f"Shapiro-Wilk test (normality): p = {p_norm:.4f}")
    print(f"  {'✓ Residuals appear normal' if p_norm > 0.05 else '✗ Residuals not normal'}")
    print()
    
    # Heteroscedasticity (visual check)
    print("Visual checks:")
    print("  • Residuals vs Fitted: Should show random scatter (no pattern)")
    print("  • Q-Q Plot: Points should follow diagonal line")
    print("  • Scale-Location: Should show horizontal band")
    print("  • Histogram: Should be roughly bell-shaped")

# Example with good fit
X_good = np.random.rand(100) * 10
y_good = 5 + 2 * X_good + np.random.normal(0, 1, 100)
comprehensive_residual_analysis(X_good, y_good)
\`\`\`

## ML Application: Feature Importance

\`\`\`python
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

def assess_feature_importance_with_regression():
    """Use univariate regression to assess feature importance"""
    
    # Generate dataset with multiple features
    X, y = make_regression (n_samples=200, n_features=10, n_informative=5,
                            noise=10, random_state=42)
    
    feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit univariate regression for each feature
    results = []
    for i in range(X.shape[1]):
        slope, _, r_value, p_value, _ = stats.linregress(X_scaled[:, i], y)
        results.append({
            'Feature': feature_names[i],
            'Slope': slope,
            'R²': r_value**2,
            'p-value': p_value
        })
    
    results_df = pd.DataFrame (results).sort_values('R²', ascending=False)
    
    print("=== Feature Importance via Univariate Regression ===")
    print(results_df.to_string (index=False))
    
    # Visualize
    fig, ax = plt.subplots (figsize=(10, 6))
    colors = ['green' if p < 0.05 else 'gray' for p in results_df['p-value']]
    ax.barh (results_df['Feature'], results_df['R²'], color=colors, alpha=0.7)
    ax.set_xlabel('R² (Variance Explained)', fontsize=12)
    ax.set_title('Feature Importance (Univariate Regression)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance_regression.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\\nGreen bars: Statistically significant (p < 0.05)")
    print("Gray bars: Not significant")

assess_feature_importance_with_regression()
\`\`\`

## Key Takeaways

1. **Simple linear regression**: Models Y = β₀ + β₁X + ε
2. **OLS**: Minimizes sum of squared residuals
3. **R²**: Proportion of variance explained (0 to 1)
4. **Slope inference**: Test if β₁ ≠ 0 (relationship exists)
5. **Assumptions**: Linearity, independence, homoscedasticity, normality of residuals
6. **Confidence interval**: Uncertainty in mean response
7. **Prediction interval**: Uncertainty in individual predictions (wider)
8. **Residual analysis**: Check assumptions visually and statistically

## Assumptions to Check

1. **Linearity**: Relationship should be linear
2. **Independence**: Observations independent
3. **Homoscedasticity**: Constant variance of residuals
4. **Normality**: Residuals normally distributed
5. **No outliers**: Influential points can distort fit

## Connection to Machine Learning

- **Baseline model**: Always try linear regression first
- **Interpretability**: Coefficients have clear meaning
- **Feature selection**: Univariate regression assesses individual features
- **Regularization preview**: L2 (Ridge) extends to multiple features
- **Foundation**: Understanding regression → understanding GLMs, neural nets
- **Diagnostics**: Residual analysis applies to complex models too

Simple linear regression is the foundation - master it before moving to multiple regression and beyond!
`,
};
