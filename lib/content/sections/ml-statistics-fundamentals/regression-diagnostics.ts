/**
 * Regression Diagnostics Section
 */

export const regressiondiagnosticsSection = {
  id: 'regression-diagnostics',
  title: 'Regression Diagnostics',
  content: `# Regression Diagnostics

## Introduction

Regression diagnostics check if your model's assumptions hold. Violating assumptions leads to:
- Biased coefficient estimates
- Invalid hypothesis tests
- Poor predictions
- Misleading conclusions

**Key assumptions** (LINE):
1. **L**inearity: Relationship is linear
2. **I**ndependence: Observations are independent
3. **N**ormality: Residuals are normally distributed
4. **E**qual variance (Homoscedasticity): Constant error variance

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

np.random.seed(42)

def comprehensive_diagnostics(X, y):
    """Complete regression diagnostic suite"""
    
    # Fit model
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()
    
    # Get key values
    fitted = model.fittedvalues
    residuals = model.resid
    standardized_residuals = model.resid_pearson
    influence = model.get_influence()
    leverage = influence.hat_matrix_diag
    cooks_d = influence.cooks_distance[0]
    
    # Create diagnostic plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Residuals vs Fitted
    axes[0, 0].scatter(fitted, residuals, alpha=0.6)
    axes[0, 0].axhline(0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted\\n(Check linearity, homoscedasticity)')
    
    # 2. Q-Q Plot
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Normal Q-Q Plot\\n(Check normality)')
    
    # 3. Scale-Location
    axes[0, 2].scatter(fitted, np.sqrt(np.abs(standardized_residuals)), alpha=0.6)
    axes[0, 2].set_xlabel('Fitted Values')
    axes[0, 2].set_ylabel('√|Standardized Residuals|')
    axes[0, 2].set_title('Scale-Location\\n(Check homoscedasticity)')
    
    # 4. Residuals vs Leverage
    axes[1, 0].scatter(leverage, standardized_residuals, alpha=0.6)
    axes[1, 0].axhline(0, color='r', linestyle='--')
    axes[1, 0].set_xlabel('Leverage')
    axes[1, 0].set_ylabel('Standardized Residuals')
    axes[1, 0].set_title('Residuals vs Leverage\\n(Check influential points)')
    
    # 5. Cook's Distance
    axes[1, 1].stem(range(len(cooks_d)), cooks_d, markerfmt=',')
    axes[1, 1].axhline(4/len(y), color='r', linestyle='--', label='Threshold')
    axes[1, 1].set_xlabel('Observation')
    axes[1, 1].set_ylabel("Cook's Distance")
    axes[1, 1].set_title("Cook's Distance\\n(Identify influential points)")
    axes[1, 1].legend()
    
    # 6. Histogram of Residuals
    axes[1, 2].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 2].axvline(0, color='r', linestyle='--', linewidth=2)
    axes[1, 2].set_xlabel('Residuals')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Residual Histogram\\n(Check normality)')
    
    plt.tight_layout()
    plt.savefig('regression_diagnostics_full.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Statistical tests
    print("=== Regression Diagnostics ===")
    print(f"R² = {model.rsquared:.4f}")
    print(f"Adjusted R² = {model.rsquared_adj:.4f}")
    print()
    
    # Normality test
    _, p_shapiro = stats.shapiro(residuals)
    print(f"Shapiro-Wilk (normality): p = {p_shapiro:.4f}")
    print(f"  {'✓ Normal' if p_shapiro > 0.05 else '✗ Not normal'}")
    
    # Homoscedasticity test
    _, p_het, _, _ = het_breuschpagan(residuals, X_with_const)
    print(f"Breusch-Pagan (homoscedasticity): p = {p_het:.4f}")
    print(f"  {'✓ Homoscedastic' if p_het > 0.05 else '✗ Heteroscedastic'}")
    
    # Influential points
    n_influential = np.sum(cooks_d > 4/len(y))
    print(f"\\nInfluential points (Cook's D > 4/n): {n_influential}")
    
    return model

# Example with good model
X = np.random.randn(100, 2)
y = 2 + 3*X[:, 0] - 2*X[:, 1] + np.random.normal(0, 1, 100)
comprehensive_diagnostics(X, y)
\`\`\`

## Checking Assumptions

### 1. Linearity

\`\`\`python
def check_linearity():
    """Detect non-linearity"""
    
    X = np.random.rand(100) * 10
    y = 2*X**2 + np.random.normal(0, 5, 100)  # Non-linear!
    
    # Fit linear model (wrong!)
    model = sm.OLS(y, sm.add_constant(X)).fit()
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, alpha=0.6)
    plt.plot(X, model.predict(sm.add_constant(X)), 'r-', linewidth=2)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Data with Linear Fit (Wrong!)')
    
    plt.subplot(1, 2, 2)
    plt.scatter(model.fittedvalues, model.resid, alpha=0.6)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot Shows Curved Pattern!')
    
    plt.tight_layout()
    plt.savefig('nonlinearity_check.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("=== Non-Linearity Detection ===")
    print("Residual plot shows clear curved pattern")
    print("→ Relationship is non-linear!")
    print("\\nRemedies:")
    print("  • Transform X (e.g., X², log(X), √X)")
    print("  • Polynomial regression")
    print("  • Non-linear models (GAMs, splines)")

check_linearity()
\`\`\`

### 2. Independence

\`\`\`python
def check_independence():
    """Check for autocorrelation in residuals"""
    
    from statsmodels.stats.stattools import durbin_watson
    
    # Time series with autocorrelation
    n = 100
    errors = np.zeros(n)
    errors[0] = np.random.randn()
    for i in range(1, n):
        errors[i] = 0.7*errors[i-1] + np.random.randn()  # AR(1)
    
    X = np.arange(n)
    y = 2 + 0.5*X + errors
    
    model = sm.OLS(y, sm.add_constant(X)).fit()
    
    # Durbin-Watson test
    dw = durbin_watson(model.resid)
    
    print("=== Independence Check ===")
    print(f"Durbin-Watson statistic: {dw:.4f}")
    print("Interpretation:")
    print("  DW ≈ 2: No autocorrelation ✓")
    print("  DW < 2: Positive autocorrelation")
    print("  DW > 2: Negative autocorrelation")
    print()
    print(f"Result: {'Positive autocorrelation detected!' if dw < 1.5 else 'OK'}")
    
    # Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(model.resid, marker='o', linestyle='-', alpha=0.6)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Residuals')
    plt.title('Residuals Over Time (Shows Pattern!)')
    
    plt.subplot(1, 2, 2)
    plt.scatter(model.resid[:-1], model.resid[1:], alpha=0.6)
    plt.xlabel('Residual at t')
    plt.ylabel('Residual at t+1')
    plt.title('Lag Plot (Shows Correlation!)')
    
    plt.tight_layout()
    plt.savefig('independence_check.png', dpi=300, bbox_inches='tight')
    plt.show()

check_independence()
\`\`\`

### 3. Normality

\`\`\`python
def check_normality():
    """Multiple tests for normality"""
    
    # Normal residuals
    normal_resid = np.random.normal(0, 1, 100)
    
    # Non-normal (heavy-tailed)
    heavy_tail = np.random.standard_t(3, 100)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Normal
    stats.probplot(normal_resid, dist="norm", plot=axes[0, 0])
    axes[0, 0].set_title('Q-Q: Normal Residuals')
    
    axes[0, 1].hist(normal_resid, bins=20, edgecolor='black')
    axes[0, 1].set_title('Histogram: Normal')
    
    # Non-normal
    stats.probplot(heavy_tail, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q: Heavy-Tailed Residuals')
    
    axes[1, 1].hist(heavy_tail, bins=20, edgecolor='black')
    axes[1, 1].set_title('Histogram: Heavy-Tailed')
    
    plt.tight_layout()
    plt.savefig('normality_check.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Statistical tests
    _, p_normal = stats.shapiro(normal_resid)
    _, p_heavy = stats.shapiro(heavy_tail)
    
    print("=== Normality Tests ===")
    print(f"Normal residuals: p = {p_normal:.4f} ✓")
    print(f"Heavy-tailed: p = {p_heavy:.4f} ✗")

check_normality()
\`\`\`

### 4. Homoscedasticity

\`\`\`python
def check_homoscedasticity():
    """Detect heteroscedasticity"""
    
    n = 100
    X = np.random.rand(n) * 10
    errors = np.random.normal(0, 0.5 + 0.5*X, n)  # Variance increases with X!
    y = 2 + 3*X + errors
    
    model = sm.OLS(y, sm.add_constant(X)).fit()
    
    plt.figure(figsize=(10, 5))
    plt.scatter(model.fittedvalues, model.resid, alpha=0.6)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Heteroscedasticity: Funnel Shape!')
    plt.savefig('heteroscedasticity.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Breusch-Pagan test
    from statsmodels.stats.diagnostic import het_breuschpagan
    _, p_value, _, _ = het_breuschpagan(model.resid, sm.add_constant(X))
    
    print("=== Heteroscedasticity Test ===")
    print(f"Breusch-Pagan p-value: {p_value:.4f}")
    print(f"  {'✗ Heteroscedastic!' if p_value < 0.05 else '✓ Homoscedastic'}")
    print("\\nRemedies:")
    print("  • Transform Y (log, sqrt)")
    print("  • Weighted Least Squares (WLS)")
    print("  • Robust standard errors")

check_homoscedasticity()
\`\`\`

## Influential Points

\`\`\`python
def identify_influential_points():
    """Find and handle influential observations"""
    
    n = 50
    X = np.random.randn(n)
    y = 2 + 3*X + np.random.normal(0, 1, n)
    
    # Add influential outlier
    X = np.append(X, 3)
    y = np.append(y, 20)  # Way off the line!
    
    model = sm.OLS(y, sm.add_constant(X)).fit()
    influence = model.get_influence()
    
    # Metrics
    leverage = influence.hat_matrix_diag
    cooks_d = influence.cooks_distance[0]
    dffits = influence.dffits[0]
    
    # Identify problematic points
    high_leverage = leverage > 2 * 2 / len(X)  # threshold
    high_cooks = cooks_d > 4 / len(X)
    
    print("=== Influential Points ===")
    print(f"High leverage points: {np.sum(high_leverage)}")
    print(f"High Cook's D points: {np.sum(high_cooks)}")
    print()
    print("Most influential points:")
    top_5 = np.argsort(cooks_d)[-5:]
    for idx in top_5:
        print(f"  Obs {idx}: Cook's D = {cooks_d[idx]:.4f}, Leverage = {leverage[idx]:.4f}")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Data with outlier
    axes[0].scatter(X, y, c=cooks_d, s=100, cmap='Reds', alpha=0.6)
    axes[0].plot(X, model.predict(sm.add_constant(X)), 'b-', linewidth=2, label='With outlier')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('y')
    axes[0].set_title('Regression with Influential Point')
    axes[0].legend()
    
    # Cook's distance plot
    axes[1].stem(range(len(cooks_d)), cooks_d, markerfmt=',')
    axes[1].axhline(4/len(X), color='r', linestyle='--', label='Threshold')
    axes[1].set_xlabel('Observation')
    axes[1].set_ylabel("Cook's Distance")
    axes[1].set_title("Influential Points Detection")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('influential_points.png', dpi=300, bbox_inches='tight')
    plt.show()

identify_influential_points()
\`\`\`

## Remedial Measures

\`\`\`python
def apply_remedies():
    """Common fixes for assumption violations"""
    
    print("=== Remedial Measures ===")
    print()
    print("1. NON-LINEARITY:")
    print("   • Add polynomial terms: X, X², X³")
    print("   • Transform X: log(X), √X, 1/X")
    print("   • Use non-linear models: GAMs, splines")
    print()
    print("2. HETEROSCEDASTICITY:")
    print("   • Transform Y: log(Y), √Y")
    print("   • Weighted Least Squares (WLS)")
    print("   • Robust standard errors (HC3)")
    print()
    print("3. NON-NORMALITY:")
    print("   • Large n: CLT makes it less critical")
    print("   • Transform Y: Box-Cox transformation")
    print("   • Robust regression: Huber, RANSAC")
    print("   • Bootstrap confidence intervals")
    print()
    print("4. INFLUENTIAL POINTS:")
    print("   • Investigate: Data error? True outlier?")
    print("   • Remove if justified (document!)")
    print("   • Robust regression methods")
    print("   • Report sensitivity analysis")
    print()
    print("5. MULTICOLLINEARITY:")
    print("   • Remove correlated features")
    print("   • Ridge regression (L2 regularization)")
    print("   • Principal Component Analysis")

apply_remedies()
\`\`\`

## ML Application: Model Validation

\`\`\`python
def ml_model_validation():
    """Complete diagnostic workflow for ML"""
    
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    
    # Generate data
    X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit model
    model = sm.OLS(y_train, sm.add_constant(X_train)).fit()
    
    print("=== ML Model Validation Checklist ===")
    print()
    print("✓ 1. Check R² and Adjusted R²")
    print(f"   R² = {model.rsquared:.4f}, Adj R² = {model.rsquared_adj:.4f}")
    
    print()
    print("✓ 2. Check residual plots")
    print("   → No patterns in residuals vs fitted")
    
    print()
    print("✓ 3. Check normality (if n<100)")
    _, p_norm = stats.shapiro(model.resid)
    print(f"   Shapiro-Wilk p = {p_norm:.4f}")
    
    print()
    print("✓ 4. Check homoscedasticity")
    from statsmodels.stats.diagnostic import het_breuschpagan
    _, p_het, _, _ = het_breuschpagan(model.resid, sm.add_constant(X_train))
    print(f"   Breusch-Pagan p = {p_het:.4f}")
    
    print()
    print("✓ 5. Check multicollinearity (VIF)")
    print("   → All VIF < 10")
    
    print()
    print("✓ 6. Check influential points")
    influence = model.get_influence()
    n_influential = np.sum(influence.cooks_distance[0] > 4/len(y_train))
    print(f"   Influential points: {n_influential}")
    
    print()
    print("✓ 7. Validate on test set")
    y_pred = model.predict(sm.add_constant(X_test))
    test_r2 = 1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - y_test.mean())**2)
    print(f"   Test R² = {test_r2:.4f}")
    
    print()
    if test_r2 > 0.7 and abs(test_r2 - model.rsquared) < 0.1:
        print("✓ MODEL READY FOR DEPLOYMENT")
    else:
        print("⚠ MODEL NEEDS IMPROVEMENT")

ml_model_validation()
\`\`\`

## Key Takeaways

1. **Always check assumptions** before trusting results
2. **Residual plots** reveal most problems visually
3. **Q-Q plot**: Check normality of residuals
4. **Scale-Location**: Check homoscedasticity
5. **Cook's Distance**: Identify influential points
6. **Statistical tests**: Confirm visual checks
7. **Remedial measures**: Transformations, robust methods
8. **Validate on test set**: Final check of generalization

Diagnostics are not optional - they're essential for reliable inference and predictions!
`,
};
