/**
 * Correlation & Association Section
 */

export const correlationassociationSection = {
  id: 'correlation-association',
  title: 'Correlation & Association',
  content: `# Correlation & Association

## Introduction

Correlation measures the strength and direction of relationships between variables. In machine learning:

- **Feature selection**: Highly correlated features may be redundant
- **Multicollinearity**: Correlated predictors cause problems in regression
- **Feature engineering**: Correlation suggests useful interactions
- **Causality warning**: Correlation ≠ Causation!
- **Model interpretation**: Understanding feature relationships

Correlation is one of the most important yet most misunderstood concepts in data science.

## Pearson Correlation Coefficient

The **Pearson correlation** (r) measures linear relationships between continuous variables.

\\[ r = \\frac{\\sum(x_i - \\bar{x})(y_i - \\bar{y})}{\\sqrt{\\sum(x_i - \\bar{x})^2 \\sum(y_i - \\bar{y})^2}} \\]

**Properties**:
- Range: -1 to +1
- r = +1: Perfect positive linear relationship
- r = 0: No linear relationship  
- r = -1: Perfect negative linear relationship

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

np.random.seed(42)

def demonstrate_correlation():
    """Show different correlation strengths"""
    
    n = 100
    x = np.random.randn(n)
    
    # Create data with different correlations
    correlations = [0.95, 0.7, 0.4, 0, -0.4, -0.7, -0.95]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, target_r in enumerate(correlations):
        # Generate y with specific correlation to x
        y = target_r * x + np.sqrt(1 - target_r**2) * np.random.randn(n)
        
        # Calculate actual correlation
        actual_r, p_value = stats.pearsonr(x, y)
        
        # Plot
        axes[idx].scatter(x, y, alpha=0.6, s=30)
        axes[idx].set_title(f'r = {actual_r:.2f}\\n(p = {p_value:.4f})')
        axes[idx].set_xlabel('X')
        axes[idx].set_ylabel('Y')
        axes[idx].grid(True, alpha=0.3)
        
        # Add regression line
        slope, intercept = np.polyfit(x, y, 1)
        axes[idx].plot(x, slope*x + intercept, 'r--', linewidth=2)
    
    # Remove extra subplot
    fig.delaxes(axes[7])
    
    plt.suptitle('Pearson Correlation: Different Strengths', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_examples.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("=== Interpreting Pearson Correlation ===")
    print("|r| = 0.0 - 0.1: Negligible")
    print("|r| = 0.1 - 0.3: Weak")
    print("|r| = 0.3 - 0.5: Moderate")
    print("|r| = 0.5 - 0.7: Strong")
    print("|r| = 0.7 - 0.9: Very strong")
    print("|r| = 0.9 - 1.0: Extremely strong")

demonstrate_correlation()
\`\`\`

## Testing Correlation Significance

\`\`\`python
def test_correlation_significance():
    """Test if correlation is statistically significant"""
    
    # Example: Is feature correlated with target?
    n = 50
    feature = np.random.randn(n)
    target = 0.4 * feature + np.random.randn(n)  # r ≈ 0.4
    
    # Pearson correlation with p-value
    r, p_value = stats.pearsonr(feature, target)
    
    # Confidence interval (Fisher z-transformation)
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_ci = [z - 1.96*se, z + 1.96*se]
    r_ci = [np.tanh(z_ci[0]), np.tanh(z_ci[1])]
    
    print("=== Correlation Significance Test ===")
    print(f"Sample size: {n}")
    print(f"Pearson r: {r:.4f}")
    print(f"95% CI: [{r_ci[0]:.4f}, {r_ci[1]:.4f}]")
    print(f"P-value: {p_value:.6f}")
    print()
    
    if p_value < 0.05:
        print("✓ Correlation is statistically significant (p < 0.05)")
    else:
        print("✗ Correlation is not statistically significant")
    
    # Power analysis
    print(f"\\nNote: With n={n}, we have 80% power to detect |r| ≥ 0.38")
    print("Smaller correlations might be missed (Type II error)")

test_correlation_significance()
\`\`\`

## Spearman Rank Correlation

**Use case**: Monotonic relationships (not necessarily linear), ordinal data, or presence of outliers

\`\`\`python
def compare_pearson_spearman():
    """Compare Pearson and Spearman correlations"""
    
    n = 100
    x = np.linspace(0, 10, n)
    
    # Three scenarios
    scenarios = [
        ("Linear", x, 2*x + np.random.normal(0, 1, n)),
        ("Nonlinear (Quadratic)", x, x**2 + np.random.normal(0, 10, n)),
        ("With Outliers", 
         np.append(x[:95], [15, 16, 17, 18, 19]),
         np.append(2*x[:95] + np.random.normal(0, 1, 95), [50, 55, 60, 65, 70]))
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (title, x_data, y_data) in enumerate(scenarios):
        # Calculate both correlations
        r_pearson, _ = stats.pearsonr(x_data, y_data)
        r_spearman, _ = stats.spearmanr(x_data, y_data)
        
        # Plot
        axes[idx].scatter(x_data, y_data, alpha=0.6, s=30)
        axes[idx].set_title(f'{title}\\nPearson: {r_pearson:.3f}\\nSpearman: {r_spearman:.3f}')
        axes[idx].set_xlabel('X')
        axes[idx].set_ylabel('Y')
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('Pearson vs Spearman Correlation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('pearson_vs_spearman.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("=== Pearson vs Spearman ===")
    print("Pearson:")
    print("  • Measures LINEAR relationships")
    print("  • Sensitive to outliers")
    print("  • Assumes continuous data")
    print()
    print("Spearman:")
    print("  • Measures MONOTONIC relationships")
    print("  • Robust to outliers")
    print("  • Works with ordinal data")
    print("  • Based on ranks, not raw values")

compare_pearson_spearman()
\`\`\`

## Correlation is NOT Causation

\`\`\`python
def correlation_vs_causation():
    """Classic example: Ice cream sales and drowning deaths"""
    
    # Months
    months = np.arange(12)
    temperature = 20 + 10 * np.sin((months - 3) * np.pi / 6)  # Seasonal
    
    # Both driven by temperature (confounding variable)
    ice_cream_sales = 100 + 50 * (temperature - 20) + np.random.normal(0, 10, 12)
    drownings = 20 + 3 * (temperature - 20) + np.random.normal(0, 2, 12)
    
    # Spurious correlation!
    r, p = stats.pearsonr(ice_cream_sales, drownings)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Spurious correlation
    axes[0].scatter(ice_cream_sales, drownings, s=100, alpha=0.6)
    axes[0].set_xlabel('Ice Cream Sales', fontsize=12)
    axes[0].set_ylabel('Drowning Deaths', fontsize=12)
    axes[0].set_title(f'Spurious Correlation (r = {r:.3f})\\n"Ice cream causes drowning?"',
                      fontsize=12, fontweight='bold')
    slope, intercept = np.polyfit(ice_cream_sales, drownings, 1)
    axes[0].plot(ice_cream_sales, slope*ice_cream_sales + intercept, 'r--', linewidth=2)
    axes[0].grid(True, alpha=0.3)
    
    # True cause: temperature
    axes[1].plot(months, temperature, 'o-', label='Temperature', linewidth=2, markersize=8)
    axes[1].plot(months, ice_cream_sales/10, 's-', label='Ice Cream (scaled)', linewidth=2, markersize=8)
    axes[1].plot(months, drownings, '^-', label='Drownings', linewidth=2, markersize=8)
    axes[1].set_xlabel('Month', fontsize=12)
    axes[1].set_ylabel('Value', fontsize=12)
    axes[1].set_title('True Cause: Temperature (Confounding Variable)',
                      fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('correlation_causation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("=== Correlation ≠ Causation ===")
    print(f"Ice cream sales and drownings: r = {r:.3f} (p = {p:.4f})")
    print()
    print("Why they're correlated:")
    print("  Both increase in summer (high temperature)")
    print("  Temperature is a CONFOUNDING variable")
    print()
    print("Doesn't mean ice cream causes drowning!")
    print()
    print("To establish causation, you need:")
    print("  1. Temporal precedence (cause before effect)")
    print("  2. Covariation (correlation)")
    print("  3. No plausible alternatives (control confounders)")
    print("  4. Ideally: Randomized controlled experiment")

correlation_vs_causation()
\`\`\`

## Partial and Semi-Partial Correlation

**Partial correlation**: Correlation between X and Y, controlling for Z

\`\`\`python
from scipy import linalg

def partial_correlation(X, Y, Z):
    """Compute partial correlation between X and Y, controlling for Z"""
    # Residuals after regressing on Z
    beta_xz = np.dot(Z.T, X) / np.dot(Z.T, Z)
    residual_x = X - Z * beta_xz
    
    beta_yz = np.dot(Z.T, Y) / np.dot(Z.T, Z)
    residual_y = Y - Z * beta_yz
    
    # Correlation of residuals
    r_partial, _ = stats.pearsonr(residual_x, residual_y)
    return r_partial

def demonstrate_partial_correlation():
    """Show how confounders affect correlation"""
    
    n = 100
    Z = np.random.randn(n)  # Confounder (e.g., temperature)
    X = 0.7*Z + np.random.randn(n)  # Ice cream (depends on Z)
    Y = 0.7*Z + np.random.randn(n)  # Drowning (depends on Z)
    
    # Naive correlation (spurious!)
    r_naive, _ = stats.pearsonr(X, Y)
    
    # Partial correlation (controlling for Z)
    r_partial = partial_correlation(X, Y, Z)
    
    print("=== Partial Correlation ===")
    print(f"Naive correlation X-Y: {r_naive:.4f} (spurious!)")
    print(f"Partial correlation X-Y|Z: {r_partial:.4f} (true relationship)")
    print()
    print("Interpretation:")
    print(f"  • Without controlling for Z: r = {r_naive:.3f} (looks related)")
    print(f"  • Controlling for Z: r = {r_partial:.3f} (actually independent)")
    print("  • The naive correlation was driven by the confounder Z!")

demonstrate_partial_correlation()
\`\`\`

## Correlation Matrices and Heatmaps

\`\`\`python
def correlation_matrix_analysis():
    """Analyze correlation matrix for ML features"""
    
    # Simulate ML dataset
    n = 200
    data = pd.DataFrame({
        'age': np.random.randint(18, 70, n),
        'income': np.random.normal(50000, 20000, n),
        'credit_score': np.random.normal(700, 50, n),
        'debt': np.random.normal(10000, 5000, n),
        'savings': np.random.normal(20000, 10000, n),
    })
    
    # Add correlations
    data['income'] = 1000 * data['age'] + np.random.normal(0, 5000, n)
    data['savings'] = 0.3 * data['income'] - 0.5 * data['debt'] + np.random.normal(0, 5000, n)
    data['credit_score'] = 600 + 0.001 * data['income'] - 0.002 * data['debt'] + np.random.normal(0, 30, n)
    
    # Compute correlation matrix
    corr_matrix = data.corr()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='coolwarm', center=0, square=True,
                linewidths=1, cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1, ax=ax)
    
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Find high correlations
    print("=== Correlation Matrix Analysis ===")
    print("\\nHigh correlations (|r| > 0.7):")
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                print(f"  {corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}: {corr_val:.3f}")
    
    print("\\n⚠ Multicollinearity Warning:")
    print("  Highly correlated features may cause issues in linear models")
    print("  Consider: removing one, PCA, or regularization (Ridge/Lasso)")

correlation_matrix_analysis()
\`\`\`

## Multicollinearity Detection (VIF)

\`\`\`python
def calculate_vif(data):
    """Calculate Variance Inflation Factor for each feature"""
    from sklearn.linear_model import LinearRegression
    
    vif_data = pd.DataFrame()
    vif_data["Feature"] = data.columns
    vif_data["VIF"] = [
        1 / (1 - LinearRegression().fit(
            data.drop(columns=[col]), data[col]
        ).score(data.drop(columns=[col]), data[col]))
        if data.drop(columns=[col]).shape[1] > 0 else 1
        for col in data.columns
    ]
    
    return vif_data.sort_values('VIF', ascending=False)

# Example
np.random.seed(42)
n = 100
df = pd.DataFrame({
    'X1': np.random.randn(n),
    'X2': np.random.randn(n),
})
df['X3'] = 0.9 * df['X1'] + 0.1 * np.random.randn(n)  # Highly correlated with X1!

vif = calculate_vif(df)

print("=== Variance Inflation Factor (VIF) ===")
print(vif)
print()
print("Interpretation:")
print("  VIF = 1: Not correlated with other features")
print("  VIF = 1-5: Moderate correlation")
print("  VIF = 5-10: High correlation (concerning)")
print("  VIF > 10: Severe multicollinearity (problematic)")
print()
print("Rule of thumb: Drop features with VIF > 10")
\`\`\`

## Point-Biserial and Phi Coefficients

\`\`\`python
def point_biserial_example():
    """Correlation between continuous and binary variable"""
    
    # Binary: Gender (0=F, 1=M)
    # Continuous: Income
    n = 100
    gender = np.random.binomial(1, 0.5, n)
    income = np.where(gender == 1, 
                      np.random.normal(55000, 15000, n),
                      np.random.normal(45000, 15000, n))
    
    # Point-biserial correlation
    r_pb, p_value = stats.pointbiserialr(gender, income)
    
    print("=== Point-Biserial Correlation ===")
    print("(Continuous variable vs Binary variable)")
    print()
    print(f"Correlation: {r_pb:.4f}")
    print(f"P-value: {p_value:.4f}")
    print()
    print(f"Mean income (Female): \${income[gender == 0].mean():, .0f
}")
print(f"Mean income (Male): \${income[gender==1].mean():,.0f}")
print()

if p_value < 0.05:
    print("✓ Significant association between gender and income")
else:
print("✗ No significant association")

point_biserial_example()

def phi_coefficient_example():
"""Correlation between two binary variables"""
    
    # 2x2 contingency table
    # Rows: Passed test(0 = No, 1 = Yes)
    # Cols: Studied(0 = No, 1 = Yes)
table = np.array([[30, 10],   # Didn't study
[20, 40]])   # Studied
    
    # Phi coefficient
chi2, _, _, _ = stats.chi2_contingency(table, correction = False)
n = table.sum()
phi = np.sqrt(chi2 / n)

print("\\n=== Phi Coefficient ===")
print("(Binary variable vs Binary variable)")
print()
print("Contingency table:")
print(pd.DataFrame(table,
    index = ['No study', 'Studied'],
    columns = ['Failed', 'Passed']))
print()
print(f"Phi coefficient: {phi:.4f}")
print()
print("Interpretation:")
print(f"  Studying is {'positively' if phi > 0 else 'negatively'} associated with passing")

phi_coefficient_example()
\`\`\`

## ML Applications

\`\`\`python
def feature_selection_by_correlation():
    """Use correlation for feature selection"""
    
    from sklearn.datasets import make_classification
    from sklearn.feature_selection import SelectKBest, f_classif
    
    # Generate data
    X, y = make_classification(n_samples=200, n_features=20, 
                                n_informative=10, n_redundant=5,
                                random_state=42)
    
    # Correlation with target
    correlations = [abs(stats.pearsonr(X[:, i], y)[0]) for i in range(X.shape[1])]
    
    # Sort by correlation
    feature_importance = pd.DataFrame({
        'Feature': [f'X{i}' for i in range(X.shape[1])],
        'Correlation': correlations
    }).sort_values('Correlation', ascending=False)
    
    print("=== Feature Selection by Correlation ===")
    print("\\nTop 10 features by correlation with target:")
    print(feature_importance.head(10))
    
    # Visualize
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(10), feature_importance['Correlation'].head(10))
    ax.set_yticks(range(10))
    ax.set_yticklabels(feature_importance['Feature'].head(10))
    ax.set_xlabel('|Correlation with Target|', fontsize=12)
    ax.set_title('Feature Importance by Correlation', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()

feature_selection_by_correlation()
\`\`\`

## Key Takeaways

1. **Pearson (r)**: Measures linear relationships, continuous data
2. **Spearman (ρ)**: Measures monotonic relationships, robust to outliers
3. **Range**: All correlations are between -1 and +1
4. **Significance testing**: Use p-values, but also check effect size
5. **Correlation ≠ Causation**: Confounding variables create spurious correlations
6. **Partial correlation**: Controls for confounders
7. **Multicollinearity**: VIF > 10 indicates problematic correlation
8. **Feature selection**: Remove highly correlated features (redundant)
9. **Different types**: Point-biserial (continuous-binary), Phi (binary-binary)

## Connection to Machine Learning

- **Feature engineering**: Correlation suggests useful combinations
- **Feature selection**: Remove redundant (highly correlated) features
- **Multicollinearity**: Causes instability in linear models (use Ridge/Lasso)
- **Interpretability**: Correlated features make coefficients hard to interpret
- **Causality**: ML finds correlations, not causation - be careful with interpretations!
- **Monitoring**: Track correlation drift to detect distribution shifts

Always remember: Correlation finds associations, not causes. Use domain knowledge and experiments to establish causation!
`,
};
