/**
 * Bivariate Analysis Section
 */

export const bivariateanalysisSection = {
  id: 'bivariate-analysis',
  title: 'Bivariate Analysis',
  content: `# Bivariate Analysis

## Introduction

Bivariate analysis examines relationships between two variables - typically between features and the target variable, or between pairs of features. This is where you discover which features are predictive and how they relate to what you're trying to predict.

**Why Bivariate Analysis Matters**:
- **Feature Selection**: Identify which features correlate with target
- **Understand Relationships**: Linear, non-linear, monotonic, categorical
- **Detect Interactions**: Features that work together
- **Guide Feature Engineering**: Combine or transform based on relationships
- **Model Choice**: Relationship patterns suggest appropriate algorithms

## Continuous vs Continuous: Correlation Analysis

### Pearson Correlation

\`\`\`python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.datasets import fetch_california_housing

# Load data
housing = fetch_california_housing(as_frame=True)
df = housing.frame

print("=" * 70)
print("PEARSON CORRELATION ANALYSIS")
print("=" * 70)

# Calculate correlation with target
correlations = df.corr()['MedHouseVal'].sort_values(ascending=False)

print("\\nCorrelation with Target (MedHouseVal):\\n")
for feature, corr in correlations.items():
    if feature != 'MedHouseVal':
        strength = (
            "Very Strong" if abs(corr) >= 0.7 else
            "Strong" if abs(corr) >= 0.5 else
            "Moderate" if abs(corr) >= 0.3 else
            "Weak"
        )
        direction = "Positive" if corr > 0 else "Negative"
        print(f"  {feature:15s}: {corr:7.4f} ({direction:8s}, {strength})")

# Scatter plots with regression line
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
features = [f for f in df.columns if f != 'MedHouseVal']

for idx, feature in enumerate(features):
    row, col = divmod(idx, 4)
    ax = axes[row, col]
    
    # Scatter plot with sample (for performance)
    sample = df.sample(min(1000, len(df)), random_state=42)
    ax.scatter(sample[feature], sample['MedHouseVal'], alpha=0.5)
    
    # Regression line
    z = np.polyfit(sample[feature], sample['MedHouseVal'], 1)
    p = np.poly1d(z)
    ax.plot(sample[feature].sort_values(), 
            p(sample[feature].sort_values()), 
            "r--", linewidth=2)
    
    corr = df[feature].corr(df['MedHouseVal'])
    ax.set_title(f'{feature}\\nr = {corr:.3f}')
    ax.set_xlabel(feature)
    ax.set_ylabel('MedHouseVal')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Output:
# ======================================================================
# PEARSON CORRELATION ANALYSIS
# ======================================================================
# 
# Correlation with Target (MedHouseVal):
# 
#   MedInc         :  0.6880 (Positive, Strong)
#   AveRooms       :  0.1514 (Positive, Weak)
#   HouseAge       :  0.1058 (Positive, Weak)
#   ...
\`\`\`

### Correlation Significance Testing

\`\`\`python
def test_correlation_significance(df, feature, target, alpha=0.05):
    """Test if correlation is statistically significant"""
    
    # Calculate Pearson correlation
    corr, p_value = stats.pearsonr(df[feature], df[target])
    
    # Calculate confidence interval (Fisher's z-transformation)
    n = len(df)
    z = np.arctanh(corr)
    se = 1 / np.sqrt(n - 3)
    z_critical = stats.norm.ppf(1 - alpha/2)
    ci_lower = np.tanh(z - z_critical * se)
    ci_upper = np.tanh(z + z_critical * se)
    
    print(f"\\n{feature} vs {target}")
    print("-" * 60)
    print(f"Pearson r: {corr:.4f}")
    print(f"P-value: {p_value:.6f}")
    print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"Significant: {'✓ Yes' if p_value < alpha else '✗ No'} (α={alpha})")
    
    # Effect size interpretation (Cohen's guidelines)
    effect_size = (
        "Large" if abs(corr) >= 0.5 else
        "Medium" if abs(corr) >= 0.3 else
        "Small" if abs(corr) >= 0.1 else
        "Negligible"
    )
    print(f"Effect Size: {effect_size}")
    
    return corr, p_value

# Test significance for key features
for feature in ['MedInc', 'AveRooms', 'Latitude']:
    test_correlation_significance(df, feature, 'MedHouseVal')
\`\`\`

### Non-Parametric Correlations: Spearman and Kendall

\`\`\`python
def compare_correlation_methods(df, feature, target):
    """Compare Pearson, Spearman, and Kendall correlations"""
    
    # Pearson (linear relationships)
    pearson_r, pearson_p = stats.pearsonr(df[feature], df[target])
    
    # Spearman (monotonic relationships, rank-based)
    spearman_r, spearman_p = stats.spearmanr(df[feature], df[target])
    
    # Kendall (monotonic relationships, rank-based, robust)
    kendall_tau, kendall_p = stats.kendalltau(df[feature], df[target])
    
    print(f"\\nCORRELATION COMPARISON: {feature} vs {target}")
    print("=" * 70)
    print(f"\\n{'Method':<15} {'Coefficient':>12} {'P-value':>12} {'Interpretation'}")
    print("-" * 70)
    print(f"{'Pearson':<15} {pearson_r:12.4f} {pearson_p:12.6f}   Linear relationship")
    print(f"{'Spearman':<15} {spearman_r:12.4f} {spearman_p:12.6f}   Monotonic relationship")
    print(f"{'Kendall':<15} {kendall_tau:12.4f} {kendall_p:12.6f}   Rank correlation")
    
    # Interpretation
    if abs(pearson_r - spearman_r) < 0.1:
        print("\\n✓ Pearson ≈ Spearman: Linear relationship")
    else:
        print("\\n⚠️  Pearson ≠ Spearman: Non-linear but monotonic relationship")
    
    return {
        'pearson': (pearson_r, pearson_p),
        'spearman': (spearman_r, spearman_p),
        'kendall': (kendall_tau, kendall_p)
    }

# Compare methods
compare_correlation_methods(df, 'MedInc', 'MedHouseVal')
\`\`\`

## Continuous vs Categorical: Box Plots and Statistical Tests

### Visualizing Relationships with Box Plots

\`\`\`python
# Create categorical version of a continuous feature
df['IncomeBracket'] = pd.cut(df['MedInc'], 
                              bins=[0, 2, 4, 6, 15], 
                              labels=['Low', 'Medium', 'High', 'Very High'])

def analyze_continuous_vs_categorical(df, continuous_var, categorical_var):
    """Analyze relationship between continuous and categorical variables"""
    
    print(f"\\n{'='*70}")
    print(f"CONTINUOUS VS CATEGORICAL: {continuous_var} by {categorical_var}")
    print(f"{'='*70}")
    
    # Group statistics
    grouped = df.groupby(categorical_var)[continuous_var].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ])
    
    print("\\nGroup Statistics:\\n")
    print(grouped)
    
    # Visualizations
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Box plot
    df.boxplot(column=continuous_var, by=categorical_var, ax=axes[0])
    axes[0].set_title(f'{continuous_var} by {categorical_var}')
    axes[0].set_xlabel(categorical_var)
    axes[0].set_ylabel(continuous_var)
    plt.sca(axes[0])
    plt.xticks(rotation=45)
    
    # Violin plot
    parts = axes[1].violinplot([
        df[df[categorical_var] == cat][continuous_var].dropna()
        for cat in df[categorical_var].cat.categories
    ], positions=range(len(df[categorical_var].cat.categories)))
    axes[1].set_title(f'{continuous_var} by {categorical_var}\\n(Violin Plot)')
    axes[1].set_xlabel(categorical_var)
    axes[1].set_ylabel(continuous_var)
    axes[1].set_xticks(range(len(df[categorical_var].cat.categories)))
    axes[1].set_xticklabels(df[categorical_var].cat.categories, rotation=45)
    
    # Strip plot with means
    for idx, cat in enumerate(df[categorical_var].cat.categories):
        data = df[df[categorical_var] == cat][continuous_var]
        axes[2].scatter([idx] * len(data), data, alpha=0.3, s=10)
        axes[2].scatter(idx, data.mean(), color='red', s=200, marker='_', linewidths=3)
    axes[2].set_title(f'{continuous_var} by {categorical_var}\\n(Means in Red)')
    axes[2].set_xlabel(categorical_var)
    axes[2].set_ylabel(continuous_var)
    axes[2].set_xticks(range(len(df[categorical_var].cat.categories)))
    axes[2].set_xticklabels(df[categorical_var].cat.categories, rotation=45)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

analyze_continuous_vs_categorical(df, 'MedHouseVal', 'IncomeBracket')
\`\`\`

### Statistical Tests for Group Differences

\`\`\`python
def test_group_differences(df, continuous_var, categorical_var):
    """Test if groups have significantly different means"""
    
    print(f"\\n{'='*70}")
    print(f"STATISTICAL TESTS: {continuous_var} across {categorical_var}")
    print(f"{'='*70}")
    
    groups = [
        df[df[categorical_var] == cat][continuous_var].dropna()
        for cat in df[categorical_var].cat.categories
    ]
    
    # Check assumptions
    print("\\n1. NORMALITY TEST (Shapiro-Wilk per group):")
    for cat, group in zip(df[categorical_var].cat.categories, groups):
        if len(group) <= 5000:
            stat, p = stats.shapiro(group)
            print(f"   {cat}: p={p:.6f} ({'Normal' if p > 0.05 else 'Not Normal'})")
    
    print("\\n2. VARIANCE HOMOGENEITY TEST (Levene's Test):")
    stat, p = stats.levene(*groups)
    print(f"   Statistic: {stat:.4f}, P-value: {p:.6f}")
    print(f"   {'Equal variances' if p > 0.05 else 'Unequal variances'}")
    
    # ANOVA (parametric test)
    print("\\n3. ONE-WAY ANOVA:")
    f_stat, anova_p = stats.f_oneway(*groups)
    print(f"   F-statistic: {f_stat:.4f}")
    print(f"   P-value: {anova_p:.6f}")
    print(f"   Result: {'✓ Groups differ significantly' if anova_p < 0.05 else '✗ No significant difference'}")
    
    # Kruskal-Wallis (non-parametric alternative)
    print("\\n4. KRUSKAL-WALLIS TEST (Non-parametric):")
    h_stat, kw_p = stats.kruskal(*groups)
    print(f"   H-statistic: {h_stat:.4f}")
    print(f"   P-value: {kw_p:.6f}")
    print(f"   Result: {'✓ Groups differ significantly' if kw_p < 0.05 else '✗ No significant difference'}")
    
    # Effect size (Eta-squared for ANOVA)
    ss_between = sum(len(g) * (g.mean() - df[continuous_var].mean())**2 for g in groups)
    ss_total = sum((df[continuous_var] - df[continuous_var].mean())**2)
    eta_squared = ss_between / ss_total
    
    print(f"\\n5. EFFECT SIZE (Eta-squared): {eta_squared:.4f}")
    effect = (
        "Large" if eta_squared >= 0.14 else
        "Medium" if eta_squared >= 0.06 else
        "Small"
    )
    print(f"   Interpretation: {effect} effect")

test_group_differences(df, 'MedHouseVal', 'IncomeBracket')
\`\`\`

## Categorical vs Categorical: Contingency Tables

### Chi-Square Test of Independence

\`\`\`python
# Create another categorical variable
df['HouseAgeGroup'] = pd.cut(df['HouseAge'], 
                              bins=[0, 15, 30, 45, 60], 
                              labels=['New', 'Medium', 'Old', 'Very Old'])

def analyze_categorical_vs_categorical(df, cat_var1, cat_var2):
    """Analyze relationship between two categorical variables"""
    
    print(f"\\n{'='*70}")
    print(f"CATEGORICAL VS CATEGORICAL: {cat_var1} vs {cat_var2}")
    print(f"{'='*70}")
    
    # Contingency table
    contingency = pd.crosstab(df[cat_var1], df[cat_var2])
    
    print("\\nContingency Table (Counts):\\n")
    print(contingency)
    
    # Normalized (proportions)
    print("\\nRow Proportions:\\n")
    print(contingency.div(contingency.sum(axis=1), axis=0).round(3))
    
    # Chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    
    print(f"\\nChi-Square Test of Independence:")
    print(f"  Chi-square statistic: {chi2:.4f}")
    print(f"  Degrees of freedom: {dof}")
    print(f"  P-value: {p_value:.6f}")
    print(f"  Result: {'✓ Variables are dependent' if p_value < 0.05 else '✗ Variables are independent'}")
    
    # Cramér's V (effect size for chi-square)
    n = contingency.sum().sum()
    min_dim = min(contingency.shape[0], contingency.shape[1]) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim))
    
    print(f"\\nCramér's V (Effect Size): {cramers_v:.4f}")
    effect = (
        "Strong" if cramers_v >= 0.25 else
        "Moderate" if cramers_v >= 0.15 else
        "Weak"
    )
    print(f"  Interpretation: {effect} association")
    
    # Heatmap visualization
    plt.figure(figsize=(10, 6))
    sns.heatmap(contingency, annot=True, fmt='d', cmap='YlOrRd')
    plt.title(f'Contingency Table Heatmap: {cat_var1} vs {cat_var2}')
    plt.xlabel(cat_var2)
    plt.ylabel(cat_var1)
    plt.tight_layout()
    plt.show()

analyze_categorical_vs_categorical(df, 'IncomeBracket', 'HouseAgeGroup')
\`\`\`

## Key Takeaways

1. **Pearson correlation measures linear relationships between continuous variables**
2. **Spearman/Kendall correlations capture monotonic (non-linear) relationships**
3. **Correlation ≠ causation (always remember this!)**
4. **Box plots visualize distribution differences across categories**
5. **ANOVA tests if groups have different means (parametric)**
6. **Kruskal-Wallis is the non-parametric alternative to ANOVA**
7. **Chi-square test examines independence of categorical variables**
8. **Effect sizes (Eta-squared, Cramér's V) quantify strength of relationships**
9. **Statistical significance (p-value) tells if effect is real, not size**
10. **Always visualize before testing - see the relationship!**

## Connection to Machine Learning

- **Feature Selection**: High correlation with target → potentially important feature
- **Multicollinearity**: High correlation between features → consider removing one
- **Non-linear Relationships**: Suggest polynomial features or non-linear models
- **Categorical Relationships**: Inform interaction terms and feature combinations
- **Model Choice**: Linear relationships → linear models sufficient; Non-linear → tree-based/neural networks
- **Target Encoding**: For categorical features with strong relationship to continuous target
- **Feature Engineering**: Combine features that show interesting bivariate patterns

Bivariate analysis reveals which features to keep, transform, or combine for modeling.
`,
};
