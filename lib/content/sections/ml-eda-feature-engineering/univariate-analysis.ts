/**
 * Univariate Analysis Section
 */

export const univariateanalysisSection = {
  id: 'univariate-analysis',
  title: 'Univariate Analysis',
  content: `# Univariate Analysis

## Introduction

Univariate analysis examines each feature independently, one at a time, to understand its distribution, central tendency, spread, and anomalies. Before analyzing relationships between features, you must understand each feature individually.

**Why Univariate Analysis Matters**:
- **Understand distributions**: Normal, skewed, bimodal, uniform?
- **Detect anomalies**: Outliers, impossible values, data errors
- **Inform transformations**: Log, sqrt, scaling based on distribution shape
- **Feature selection**: Identify zero-variance or near-constant features
- **Set baselines**: Understand typical values before modeling

## Distribution Analysis

### Understanding Distributions

\`\`\`python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Generate sample data with different distributions
np.random.seed(42)
n = 1000

data = pd.DataFrame({
    'normal': np.random.normal(100, 15, n),
    'right_skewed': np.random.exponential(2, n),
    'left_skewed': 100 - np.random.exponential(2, n) ** 2,
    'bimodal': np.concatenate([
        np.random.normal(30, 5, n//2),
        np.random.normal(70, 5, n//2)
    ]),
    'uniform': np.random.uniform(0, 100, n),
})

print("=" * 70)
print("DISTRIBUTION ANALYSIS")
print("=" * 70)

for col in data.columns:
    print(f"\\n{col.upper()}:")
    print(f"  Mean: {data[col].mean():.2f}")
    print(f"  Median: {data[col].median():.2f}")
    print(f"  Std: {data[col].std():.2f}")
    print(f"  Skewness: {data[col].skew():.2f}")
    print(f"  Kurtosis: {data[col].kurtosis():.2f}")

# Visualize distributions
fig, axes = plt.subplots(2, 5, figsize=(20, 8))

for idx, col in enumerate (data.columns):
    # Histogram
    axes[0, idx].hist (data[col], bins=30, edgecolor='black', alpha=0.7)
    axes[0, idx].set_title (f'{col}\\nHistogram')
    axes[0, idx].set_xlabel('Value')
    axes[0, idx].set_ylabel('Frequency')
    
    # Box plot
    axes[1, idx].boxplot (data[col])
    axes[1, idx].set_title (f'{col}\\nBox Plot')
    axes[1, idx].set_ylabel('Value')

plt.tight_layout()
plt.show()

# Output:
# ======================================================================
# DISTRIBUTION ANALYSIS
# ======================================================================
# 
# NORMAL:
#   Mean: 99.87
#   Median: 100.02
#   Std: 14.91
#   Skewness: -0.06
#   Kurtosis: -0.04
# 
# RIGHT_SKEWED:
#   Mean: 1.99
#   Median: 1.38
#   Std: 2.01
#   Skewness: 2.01
#   Kurtosis: 6.42
\`\`\`

### Interpreting Distribution Shapes

\`\`\`python
def interpret_distribution (series, name):
    """Interpret distribution characteristics and recommend transformations"""
    
    mean = series.mean()
    median = series.median()
    skewness = series.skew()
    kurtosis = series.kurtosis()
    
    print(f"\\nDISTRIBUTION INTERPRETATION: {name}")
    print("=" * 70)
    
    # Skewness interpretation
    print(f"\\n📊 SKEWNESS: {skewness:.3f}")
    if abs (skewness) < 0.5:
        skew_type = "Approximately symmetric"
        transform = "No transformation needed"
    elif skewness > 0.5:
        skew_type = "Right-skewed (positive skew)"
        transform = "Consider log or sqrt transformation"
    else:
        skew_type = "Left-skewed (negative skew)"
        transform = "Consider square or exponential transformation"
    
    print(f"  Interpretation: {skew_type}")
    print(f"  Recommendation: {transform}")
    
    # Kurtosis interpretation
    print(f"\\n📈 KURTOSIS: {kurtosis:.3f}")
    if abs (kurtosis) < 1:
        kurt_type = "Mesokurtic (normal-like tails)"
    elif kurtosis > 1:
        kurt_type = "Leptokurtic (heavy tails, more outliers)"
    else:
        kurt_type = "Platykurtic (light tails, fewer outliers)"
    
    print(f"  Interpretation: {kurt_type}")
    
    # Mean vs Median
    print(f"\\n⚖️  CENTRAL TENDENCY:")
    print(f"  Mean: {mean:.3f}")
    print(f"  Median: {median:.3f}")
    print(f"  Difference: {abs (mean - median):.3f}")
    
    if abs (mean - median) / median < 0.01:
        print(f"  Interpretation: Mean ≈ Median → Symmetric distribution")
    elif mean > median:
        print(f"  Interpretation: Mean > Median → Right-skewed (outliers pull mean right)")
    else:
        print(f"  Interpretation: Mean < Median → Left-skewed (outliers pull mean left)")
    
    return {
        'skewness': skewness,
        'kurtosis': kurtosis,
        'transformation': transform
    }

# Analyze each distribution
for col in data.columns:
    interpret_distribution (data[col], col)
\`\`\`

## Central Tendency and Spread

### Comprehensive Summary Statistics

\`\`\`python
def comprehensive_statistics (series, name):
    """Calculate and display comprehensive statistics for a feature"""
    
    print(f"\\n{'='*70}")
    print(f"COMPREHENSIVE STATISTICS: {name}")
    print(f"{'='*70}")
    
    # Central Tendency
    print(f"\\n📍 CENTRAL TENDENCY:")
    print(f"  Mean:               {series.mean():.4f}")
    print(f"  Median:             {series.median():.4f}")
    print(f"  Mode:               {series.mode().values[0]:.4f}")
    print(f"  Trimmed Mean (10%): {stats.trim_mean (series, 0.1):.4f}")
    
    # Spread
    print(f"\\n📏 SPREAD:")
    print(f"  Std Deviation:      {series.std():.4f}")
    print(f"  Variance:           {series.var():.4f}")
    print(f"  Range:              {series.max() - series.min():.4f}")
    print(f"  IQR (Q3-Q1):        {series.quantile(0.75) - series.quantile(0.25):.4f}")
    print(f"  MAD (Mean Abs Dev): {np.mean (np.abs (series - series.mean())):.4f}")
    print(f"  Coefficient of Var: {series.std() / series.mean():.4f}")
    
    # Quantiles
    print(f"\\n📊 QUANTILES:")
    quantiles = [0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0]
    for q in quantiles:
        print(f"  {int (q*100):3d}%: {series.quantile (q):10.4f}")
    
    # Shape
    print(f"\\n🎨 SHAPE:")
    print(f"  Skewness:           {series.skew():.4f}")
    print(f"  Kurtosis:           {series.kurtosis():.4f}")
    
    # Data Quality
    print(f"\\n✅ DATA QUALITY:")
    print(f"  Count:              {series.count():,}")
    print(f"  Missing:            {series.isnull().sum():,} ({100*series.isnull().sum()/len (series):.2f}%)")
    print(f"  Unique:             {series.nunique():,}")
    print(f"  Zeros:              {(series == 0).sum():,}")
    print(f"  Negative:           {(series < 0).sum():,}")

# Example: Analyze California housing data
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing (as_frame=True)
df = housing.frame

# Analyze median income
comprehensive_statistics (df['MedInc'], 'Median Income')
\`\`\`

## Identifying Outliers

### Multiple Outlier Detection Methods

\`\`\`python
def detect_outliers (series, name):
    """Detect outliers using multiple methods"""
    
    print(f"\\n{'='*70}")
    print(f"OUTLIER DETECTION: {name}")
    print(f"{'='*70}")
    
    n = len (series)
    outliers_methods = {}
    
    # Method 1: IQR Method (most common)
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    iqr_outliers = series[(series < lower_bound) | (series > upper_bound)]
    outliers_methods['IQR Method'] = {
        'count': len (iqr_outliers),
        'percent': 100 * len (iqr_outliers) / n,
        'bounds': (lower_bound, upper_bound)
    }
    
    # Method 2: Z-Score Method
    z_scores = np.abs (stats.zscore (series))
    z_outliers = series[z_scores > 3]
    outliers_methods['Z-Score (>3)'] = {
        'count': len (z_outliers),
        'percent': 100 * len (z_outliers) / n,
        'threshold': 3
    }
    
    # Method 3: Modified Z-Score (using MAD)
    median = series.median()
    mad = np.median (np.abs (series - median))
    modified_z_scores = 0.6745 * (series - median) / mad
    mad_outliers = series[np.abs (modified_z_scores) > 3.5]
    outliers_methods['Modified Z-Score'] = {
        'count': len (mad_outliers),
        'percent': 100 * len (mad_outliers) / n,
        'threshold': 3.5
    }
    
    # Method 4: Percentile Method
    lower_percentile = series.quantile(0.01)
    upper_percentile = series.quantile(0.99)
    percentile_outliers = series[(series < lower_percentile) | (series > upper_percentile)]
    outliers_methods['Percentile (1%, 99%)'] = {
        'count': len (percentile_outliers),
        'percent': 100 * len (percentile_outliers) / n,
        'bounds': (lower_percentile, upper_percentile)
    }
    
    # Display results
    print(f"\\nTotal samples: {n:,}\\n")
    
    for method, results in outliers_methods.items():
        print(f"{method}:")
        print(f"  Outliers: {results['count']:,} ({results['percent']:.2f}%)")
        if 'bounds' in results:
            print(f"  Bounds: [{results['bounds'][0]:.2f}, {results['bounds'][1]:.2f}]")
        if 'threshold' in results:
            print(f"  Threshold: {results['threshold']}")
        print()
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Box plot
    axes[0].boxplot (series)
    axes[0].set_title('Box Plot')
    axes[0].set_ylabel (name)
    axes[0].axhline (lower_bound, color='r', linestyle='--', label='IQR bounds')
    axes[0].axhline (upper_bound, color='r', linestyle='--')
    axes[0].legend()
    
    # Histogram with outlier bounds
    axes[1].hist (series, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline (lower_bound, color='r', linestyle='--', label='IQR bounds')
    axes[1].axvline (upper_bound, color='r', linestyle='--')
    axes[1].set_title('Distribution with Outlier Bounds')
    axes[1].set_xlabel (name)
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    
    # Z-scores
    axes[2].scatter (range (len (series)), z_scores, alpha=0.5)
    axes[2].axhline(3, color='r', linestyle='--', label='Z=3 threshold')
    axes[2].set_title('Z-Scores')
    axes[2].set_xlabel('Index')
    axes[2].set_ylabel('|Z-Score|')
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()
    
    return outliers_methods

# Detect outliers in median income
outlier_analysis = detect_outliers (df['MedInc'], 'Median Income')
\`\`\`

## Normality Testing

### Statistical Tests for Normality

\`\`\`python
def test_normality (series, name):
    """Test if data follows a normal distribution using multiple tests"""
    
    print(f"\\n{'='*70}")
    print(f"NORMALITY TESTS: {name}")
    print(f"{'='*70}")
    
    # Visual tests
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Histogram with normal curve overlay
    axes[0].hist (series, bins=30, density=True, alpha=0.7, edgecolor='black')
    mu, sigma = series.mean(), series.std()
    x = np.linspace (series.min(), series.max(), 100)
    axes[0].plot (x, stats.norm.pdf (x, mu, sigma), 'r-', linewidth=2, label='Normal PDF')
    axes[0].set_title('Histogram vs Normal Distribution')
    axes[0].set_xlabel (name)
    axes[0].set_ylabel('Density')
    axes[0].legend()
    
    # Q-Q Plot
    stats.probplot (series, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot')
    axes[1].grid(True, alpha=0.3)
    
    # Box plot
    axes[2].boxplot (series)
    axes[2].set_title('Box Plot')
    axes[2].set_ylabel (name)
    
    plt.tight_layout()
    plt.show()
    
    # Statistical tests
    print(f"\\n📊 STATISTICAL TESTS:\\n")
    
    # Shapiro-Wilk Test (best for n < 5000)
    if len (series) <= 5000:
        shapiro_stat, shapiro_p = stats.shapiro (series)
        print(f"Shapiro-Wilk Test:")
        print(f"  Statistic: {shapiro_stat:.4f}")
        print(f"  P-value: {shapiro_p:.4f}")
        print(f"  Result: {'✓ Normal' if shapiro_p > 0.05 else '✗ Not Normal'} (α=0.05)")
        print()
    
    # Kolmogorov-Smirnov Test
    ks_stat, ks_p = stats.kstest (series, 'norm', args=(mu, sigma))
    print(f"Kolmogorov-Smirnov Test:")
    print(f"  Statistic: {ks_stat:.4f}")
    print(f"  P-value: {ks_p:.4f}")
    print(f"  Result: {'✓ Normal' if ks_p > 0.05 else '✗ Not Normal'} (α=0.05)")
    print()
    
    # Anderson-Darling Test
    anderson_result = stats.anderson (series, dist='norm')
    print(f"Anderson-Darling Test:")
    print(f"  Statistic: {anderson_result.statistic:.4f}")
    print(f"  Critical values: {anderson_result.critical_values}")
    print(f"  Significance levels: {anderson_result.significance_level}%")
    
    # D'Agostino-Pearson Test
    dagostino_stat, dagostino_p = stats.normaltest (series)
    print(f"\\nD'Agostino-Pearson Test:")
    print(f"  Statistic: {dagostino_stat:.4f}")
    print(f"  P-value: {dagostino_p:.4f}")
    print(f"  Result: {'✓ Normal' if dagostino_p > 0.05 else '✗ Not Normal'} (α=0.05)")
    
    # Summary
    print(f"\\n📝 SUMMARY:")
    skewness = series.skew()
    kurtosis = series.kurtosis()
    print(f"  Skewness: {skewness:.4f} ({'near 0 = symmetric' if abs (skewness) < 0.5 else 'skewed'})")
    print(f"  Kurtosis: {kurtosis:.4f} ({'near 0 = normal-like' if abs (kurtosis) < 1 else 'heavy/light tails'})")

# Test normality for median income
test_normality (df['MedInc'], 'Median Income')
\`\`\`

## Transformation Techniques

### Common Transformations for Different Distributions

\`\`\`python
def apply_transformations (series, name):
    """Apply common transformations and compare results"""
    
    print(f"\\n{'='*70}")
    print(f"TRANSFORMATION ANALYSIS: {name}")
    print(f"{'='*70}")
    
    # Original
    original = series
    
    # Transformations
    transformations = {
        'Original': original,
        'Log': np.log1p (original),  # log(1 + x) to handle zeros
        'Square Root': np.sqrt (original - original.min() + 1),
        'Cube Root': np.cbrt (original),
        'Box-Cox': stats.boxcox (original - original.min() + 1)[0],
        'Yeo-Johnson': stats.yeojohnson (original)[0],
    }
    
    # Compare skewness
    print(f"\\nSKEWNESS COMPARISON:\\n")
    skewness_results = []
    
    for trans_name, trans_data in transformations.items():
        skew = trans_data.skew()
        skewness_results.append((trans_name, skew))
        print(f"  {trans_name:15s}: {skew:7.4f}")
    
    # Recommend best transformation
    best_transform = min (skewness_results, key=lambda x: abs (x[1]))
    print(f"\\n✓ Best transformation: {best_transform[0]} (skewness closest to 0)")
    
    # Visualize transformations
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    
    for idx, (trans_name, trans_data) in enumerate (transformations.items()):
        axes[idx].hist (trans_data, bins=30, edgecolor='black', alpha=0.7)
        axes[idx].set_title (f'{trans_name}\\nSkewness: {trans_data.skew():.3f}')
        axes[idx].set_xlabel('Value')
        axes[idx].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    return transformations

# Apply transformations to right-skewed data
skewed_feature = df['MedInc'] ** 2  # Create more skewed version
transformations = apply_transformations (skewed_feature, 'Skewed Feature')
\`\`\`

## Zero-Variance and Near-Constant Features

### Identifying Uninformative Features

\`\`\`python
def identify_low_variance_features (df, threshold=0.01):
    """Identify features with low variance that provide little information"""
    
    print(f"\\n{'='*70}")
    print(f"LOW VARIANCE FEATURE DETECTION")
    print(f"{'='*70}")
    
    low_variance = []
    
    for col in df.select_dtypes (include=[np.number]).columns:
        # Calculate coefficient of variation (std / mean)
        if df[col].mean() != 0:
            cv = df[col].std() / abs (df[col].mean())
        else:
            cv = 0
        
        # Calculate unique value ratio
        unique_ratio = df[col].nunique() / len (df)
        
        # Check if nearly constant
        if cv < threshold or unique_ratio < 0.01:
            low_variance.append({
                'column': col,
                'cv': cv,
                'unique_ratio': unique_ratio,
                'unique_values': df[col].nunique(),
                'most_common': df[col].mode().values[0] if len (df[col].mode()) > 0 else None,
                'most_common_freq': df[col].value_counts().iloc[0] / len (df) if len (df[col]) > 0 else 0
            })
    
    if low_variance:
        print(f"\\nFound {len (low_variance)} low-variance features:\\n")
        for feat in low_variance:
            print(f"Column: {feat['column']}")
            print(f"  Coefficient of Variation: {feat['cv']:.6f}")
            print(f"  Unique Value Ratio: {feat['unique_ratio']:.4f}")
            print(f"  Unique Values: {feat['unique_values']}")
            print(f"  Most Common Value: {feat['most_common']}")
            print(f"  Most Common Frequency: {feat['most_common_freq']:.2%}")
            print()
        
        print(f"⚠️  Consider removing these features as they provide little information")
    else:
        print(f"\\n✓ No low-variance features found")
    
    return low_variance

# Identify low-variance features
low_var = identify_low_variance_features (df)
\`\`\`

## Key Takeaways

1. **Understand each feature individually before analyzing relationships**
2. **Distribution shape informs necessary transformations**
3. **Skewness indicates direction of outliers (left vs right)**
4. **Multiple outlier detection methods provide different perspectives**
5. **Normality tests guide transformation and model selection**
6. **Log transformation effective for right-skewed data**
7. **Box-Cox and Yeo-Johnson automatically find optimal transformations**
8. **Low-variance features provide little predictive value**
9. **Central tendency (mean vs median) reveals distribution symmetry**
10. **Always visualize distributions - numbers alone can mislead**

## Connection to Machine Learning

- **Distribution shape** affects model choice (parametric vs non-parametric)
- **Outliers** can severely impact linear models and distance-based algorithms
- **Skewed features** benefit from log/sqrt transformations for linear models
- **Normality** is assumed by many statistical tests and some ML algorithms
- **Feature variance** directly impacts feature importance in models
- **Transformation** can make relationships more linear and improve model performance
- **Understanding distributions** informs appropriate imputation strategies for missing data

Univariate analysis is the foundation - master this before moving to relationships between features.
`,
};
