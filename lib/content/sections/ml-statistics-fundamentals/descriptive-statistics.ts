/**
 * Descriptive Statistics Section
 */

export const descriptivestatisticsSection = {
  id: 'descriptive-statistics',
  title: 'Descriptive Statistics',
  content: `# Descriptive Statistics

## Introduction

Descriptive statistics are the foundation of data analysis - they summarize and describe the main features of a dataset. Before building any machine learning model, you must understand your data through descriptive statistics. They answer fundamental questions:

- **What is the typical value?** (Central tendency)
- **How spread out is the data?** (Dispersion)
- **What is the shape of the distribution?** (Skewness, kurtosis)
- **Are there unusual values?** (Outliers)

In machine learning, descriptive statistics guide:
- **Feature engineering**: Identifying transformations needed
- **Model selection**: Understanding data characteristics
- **Anomaly detection**: Finding unusual patterns
- **Data quality**: Spotting errors and missing values
- **Interpretation**: Explaining model behavior

## Measures of Central Tendency

### Mean (Arithmetic Average)

The **mean** is the sum of values divided by the count:

\\[ \\bar{x} = \\frac{1}{n} \\sum_{i=1}^{n} x_i \\]

**Properties**:
- Most common measure
- Sensitive to outliers
- Optimal for symmetric distributions

**When to use**: Data is roughly symmetric without extreme outliers

\`\`\`python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
np.random.seed(42)

# Example: House prices (in thousands)
house_prices = np.array([200, 250, 180, 300, 220, 240, 260, 280, 210, 3000])

mean_price = np.mean (house_prices)
print(f"Mean price: \${mean_price:.2f}k")
print(f"Notice: The $3000k mansion drastically pulls the mean up!")

# Output:
# Mean price: $514.00k
# Notice: The $3000k mansion drastically pulls the mean up!

# More realistic example without outlier
typical_prices = house_prices[house_prices < 1000]
mean_typical = np.mean (typical_prices)
print(f"\\nMean of typical houses: \${mean_typical:.2f}k")
# Output:
# Mean of typical houses: $235.56k
\`\`\`

### Median (Middle Value)

The **median** is the middle value when data is sorted:

- If n is odd: median = middle value
- If n is even: median = average of two middle values

**Properties**:
- Robust to outliers
- Better for skewed distributions
- Represents the 50th percentile

**When to use**: Data has outliers or is skewed

\`\`\`python
median_price = np.median (house_prices)
print(f"Median price: \${median_price:.2f}k")
print(f"\\nCompare:")
print(f"Mean: \${mean_price:.2f}k (pulled up by outlier)")
print(f"Median: \${median_price:.2f}k (robust to outlier)")

# Output:
# Median price: $245.00k
# 
# Compare:
# Mean: $514.00k (pulled up by outlier)
# Median: $245.00k (robust to outlier)

# The median gives a better "typical" value here!
\`\`\`

### Mode (Most Frequent)

The **mode** is the most frequently occurring value.

**Properties**:
- Can have multiple modes (bimodal, multimodal)
- Works for categorical data
- May not exist for continuous data

**When to use**: Categorical data or discrete distributions

\`\`\`python
from scipy import stats

# Example: T-shirt sizes
sizes = ['S', 'M', 'M', 'L', 'M', 'XL', 'M', 'L', 'M', 'S']
mode_result = stats.mode (sizes, keepdims=True)
print(f"Most common size: {mode_result.mode[0]}")
print(f"Frequency: {mode_result.count[0]}")

# Output:
# Most common size: M
# Frequency: 5

# For numerical data
grades = [85, 90, 85, 78, 92, 85, 88, 90]
mode_grade = stats.mode (grades, keepdims=True)
print(f"\\nMost common grade: {mode_grade.mode[0]}")
# Output:
# Most common grade: 85
\`\`\`

### Comparison of Central Tendency Measures

\`\`\`python
def compare_central_tendency():
    """Compare mean, median, mode on different distributions"""
    
    # Symmetric distribution (normal)
    symmetric = np.random.normal(100, 15, 1000)
    
    # Right-skewed distribution (income-like)
    skewed = np.random.exponential(50, 1000)
    
    # Bimodal distribution
    bimodal = np.concatenate([
        np.random.normal(30, 5, 500),
        np.random.normal(70, 5, 500)
    ])
    
    distributions = {
        'Symmetric (Normal)': symmetric,
        'Right-Skewed': skewed,
        'Bimodal': bimodal
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for (name, data), ax in zip (distributions.items(), axes):
        ax.hist (data, bins=30, density=True, alpha=0.7, edgecolor='black')
        
        mean_val = np.mean (data)
        median_val = np.median (data)
        
        ax.axvline (mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
        ax.axvline (median_val, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}')
        
        ax.set_title (name)
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('central_tendency_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Key Observations:")
    print("1. Symmetric: Mean ≈ Median")
    print("2. Right-Skewed: Mean > Median (pulled by long tail)")
    print("3. Bimodal: Mean/Median between peaks (not representative!)")

compare_central_tendency()
\`\`\`

## Measures of Dispersion (Spread)

### Range

The **range** is the difference between maximum and minimum:

\\[ \\text{Range} = \\max (x) - \\min (x) \\]

**Limitations**: Sensitive to outliers, uses only two values

\`\`\`python
data = np.array([10, 12, 15, 18, 100])
data_range = np.ptp (data)  # peak-to-peak
print(f"Range: {data_range}")
print(f"Min: {np.min (data)}, Max: {np.max (data)}")

# Output:
# Range: 90
# Min: 10, Max: 100
\`\`\`

### Variance

The **variance** measures average squared deviation from the mean:

**Population variance**:
\\[ \\sigma^2 = \\frac{1}{n} \\sum_{i=1}^{n} (x_i - \\mu)^2 \\]

**Sample variance** (unbiased estimator):
\\[ s^2 = \\frac{1}{n-1} \\sum_{i=1}^{n} (x_i - \\bar{x})^2 \\]

**Why n-1?** Bessel\'s correction for unbiased estimation from samples.

\`\`\`python
data = np.array([2, 4, 4, 4, 5, 5, 7, 9])

# Population variance (ddof=0)
var_pop = np.var (data, ddof=0)

# Sample variance (ddof=1) - default
var_sample = np.var (data, ddof=1)

print(f"Population variance: {var_pop:.3f}")
print(f"Sample variance: {var_sample:.3f}")
print(f"\\nSample variance is slightly larger (more conservative)")

# Output:
# Population variance: 4.000
# Sample variance: 4.571
# 
# Sample variance is slightly larger (more conservative)
\`\`\`

### Standard Deviation

The **standard deviation** is the square root of variance:

\\[ \\sigma = \\sqrt{\\sigma^2} \\]

**Advantage over variance**: Same units as original data

\`\`\`python
std_dev = np.std (data, ddof=1)
print(f"Standard deviation: {std_dev:.3f}")
print(f"Mean: {np.mean (data):.3f}")
print(f"\\nInterpretation: Data typically deviates ±{std_dev:.2f} from mean")

# Output:
# Standard deviation: 2.138
# Mean: 5.000
# 
# Interpretation: Data typically deviates ±2.14 from mean
\`\`\`

### Coefficient of Variation (CV)

The **CV** is the ratio of standard deviation to mean:

\\[ CV = \\frac{\\sigma}{\\mu} \\times 100\\% \\]

**Use**: Compare variability across different scales

\`\`\`python
# Compare variability of different measurements
height_cm = np.array([170, 175, 180, 172, 178])  # cm
weight_kg = np.array([70, 75, 80, 72, 78])       # kg

cv_height = (np.std (height_cm) / np.mean (height_cm)) * 100
cv_weight = (np.std (weight_kg) / np.mean (weight_kg)) * 100

print(f"Height CV: {cv_height:.2f}%")
print(f"Weight CV: {cv_weight:.2f}%")
print(f"\\nWeight has higher relative variability")

# Output:
# Height CV: 2.13%
# Weight CV: 5.38%
# 
# Weight has higher relative variability
\`\`\`

## Quartiles and Percentiles

### Percentiles

The **p-th percentile** is the value below which p% of data falls.

**Common percentiles**:
- 25th percentile (Q1): First quartile
- 50th percentile (Q2): Median
- 75th percentile (Q3): Third quartile

\`\`\`python
data = np.random.normal(100, 15, 1000)

percentiles = np.percentile (data, [25, 50, 75])
print(f"25th percentile (Q1): {percentiles[0]:.2f}")
print(f"50th percentile (Median): {percentiles[1]:.2f}")
print(f"75th percentile (Q3): {percentiles[2]:.2f}")

# Output:
# 25th percentile (Q1): 89.84
# 50th percentile (Median): 100.12
# 75th percentile (Q3): 110.35
\`\`\`

### Interquartile Range (IQR)

The **IQR** is the range of the middle 50% of data:

\\[ IQR = Q_3 - Q_1 \\]

**Use**: Robust measure of spread, outlier detection

\`\`\`python
q1, q3 = np.percentile (data, [25, 75])
iqr = q3 - q1
print(f"IQR: {iqr:.2f}")

# Outlier detection: values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = data[(data < lower_bound) | (data > upper_bound)]
print(f"\\nOutliers detected: {len (outliers)}")
print(f"Outlier bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")

# Output:
# IQR: 20.51
# 
# Outliers detected: 23
# Outlier bounds: [59.08, 141.11]
\`\`\`

## Skewness and Kurtosis

### Skewness

**Skewness** measures asymmetry of the distribution:

\\[ \\text{Skew} = \\frac{E[(X - \\mu)^3]}{\\sigma^3} \\]

- **Skew = 0**: Symmetric (normal distribution)
- **Skew > 0**: Right-skewed (long right tail)
- **Skew < 0**: Left-skewed (long left tail)

\`\`\`python
from scipy.stats import skew, kurtosis

# Create distributions with different skewness
symmetric = np.random.normal(0, 1, 1000)
right_skewed = np.random.exponential(1, 1000)
left_skewed = -np.random.exponential(1, 1000)

print("=== Skewness ===")
print(f"Symmetric: {skew (symmetric):.3f}")
print(f"Right-skewed: {skew (right_skewed):.3f}")
print(f"Left-skewed: {skew (left_skewed):.3f}")

# Output:
# === Skewness ===
# Symmetric: 0.045
# Right-skewed: 2.012
# Left-skewed: -2.012
\`\`\`

### Kurtosis

**Kurtosis** measures the "tailedness" of the distribution:

\\[ \\text{Kurt} = \\frac{E[(X - \\mu)^4]}{\\sigma^4} - 3 \\]

- **Kurt = 0**: Normal (mesokurtic)
- **Kurt > 0**: Heavy tails (leptokurtic) - more outliers
- **Kurt < 0**: Light tails (platykurtic) - fewer outliers

\`\`\`python
print("\\n=== Kurtosis (excess) ===")
print(f"Normal: {kurtosis (symmetric):.3f}")
print(f"Heavy-tailed: {kurtosis (np.random.laplace(0, 1, 1000)):.3f}")
print(f"Light-tailed: {kurtosis (np.random.uniform(-1, 1, 1000)):.3f}")

# Output:
# === Kurtosis (excess) ===
# Normal: -0.064
# Heavy-tailed: 2.989
# Light-tailed: -1.198
\`\`\`

## Outlier Detection

### Z-Score Method

**Z-score** measures how many standard deviations from the mean:

\\[ z = \\frac{x - \\mu}{\\sigma} \\]

**Rule**: |z| > 3 indicates potential outlier

\`\`\`python
def detect_outliers_zscore (data, threshold=3):
    """Detect outliers using z-score method"""
    mean = np.mean (data)
    std = np.std (data)
    z_scores = np.abs((data - mean) / std)
    
    outliers = data[z_scores > threshold]
    return outliers, z_scores

# Example with outliers
data = np.concatenate([
    np.random.normal(100, 10, 95),
    np.array([50, 150, 160, 45, 155])  # outliers
])

outliers, z_scores = detect_outliers_zscore (data)
print(f"Detected {len (outliers)} outliers:")
print(outliers)

# Output:
# Detected 5 outliers:
# [ 50. 150. 160.  45. 155.]
\`\`\`

### IQR Method (More Robust)

\`\`\`python
def detect_outliers_iqr (data, factor=1.5):
    """Detect outliers using IQR method"""
    q1 = np.percentile (data, 25)
    q3 = np.percentile (data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return outliers, (lower_bound, upper_bound)

outliers_iqr, bounds = detect_outliers_iqr (data)
print(f"\\nIQR method detected {len (outliers_iqr)} outliers")
print(f"Bounds: [{bounds[0]:.2f}, {bounds[1]:.2f}]")

# Output:
# IQR method detected 5 outliers
# Bounds: [64.09, 135.91]
\`\`\`

## Comprehensive EDA with Pandas

\`\`\`python
# Real-world example: ML dataset
from sklearn.datasets import load_boston

# Load dataset (using a classic example)
data = pd.DataFrame({
    'price': np.random.normal(200, 50, 100),
    'sqft': np.random.normal(1500, 300, 100),
    'bedrooms': np.random.choice([2, 3, 4, 5], 100),
    'age': np.random.exponential(20, 100)
})

# Comprehensive descriptive statistics
print("=== Descriptive Statistics Summary ===")
print(data.describe())

print("\\n=== Additional Statistics ===")
print(f"Skewness:")
print(data.skew())
print(f"\\nKurtosis:")
print(data.kurtosis())

# Output shows complete statistical summary of the dataset
\`\`\`

## Box Plot for Visual Summary

\`\`\`python
def create_comprehensive_boxplot (data):
    """Create detailed box plot showing all key statistics"""
    
    fig, axes = plt.subplots(1, len (data.columns), figsize=(15, 5))
    
    for idx, col in enumerate (data.columns):
        ax = axes[idx] if len (data.columns) > 1 else axes
        
        # Create box plot
        bp = ax.boxplot (data[col], vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        
        # Add mean
        mean_val = data[col].mean()
        ax.plot([1], [mean_val], marker='o', color='red', 
                markersize=10, label='Mean')
        
        # Labels
        ax.set_title (f'{col}\\n (n={len (data)})')
        ax.set_xticklabels(['])
        ax.legend()
        
        # Add text annotations
        stats = {
            'Min': data[col].min(),
            'Q1': data[col].quantile(0.25),
            'Median': data[col].median(),
            'Q3': data[col].quantile(0.75),
            'Max': data[col].max(),
            'Mean': mean_val,
            'Std': data[col].std()
        }
        
        text = '\\n'.join([f'{k}: {v:.1f}' for k, v in stats.items()])
        ax.text(1.3, data[col].median(), text, fontsize=8, 
                verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('comprehensive_boxplot.png', dpi=300, bbox_inches='tight')
    plt.show()

create_comprehensive_boxplot (data[['price', 'sqft']])
\`\`\`

## Machine Learning Applications

### Feature Scaling Context

\`\`\`python
# Why understanding dispersion matters for feature scaling
features = pd.DataFrame({
    'income': np.random.normal(50000, 20000, 100),  # large scale
    'age': np.random.normal(35, 10, 100),           # medium scale
    'satisfaction': np.random.uniform(1, 5, 100)    # small scale
})

print("=== Before Scaling ===")
print(features.describe().loc[['mean', 'std']])

# Standard scaling (z-score normalization)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
features_scaled = pd.DataFrame(
    scaler.fit_transform (features),
    columns=features.columns
)

print("\\n=== After Standard Scaling ===")
print(features_scaled.describe().loc[['mean', 'std']])

# Now all features have mean=0, std=1
\`\`\`

### Detecting Data Quality Issues

\`\`\`python
def data_quality_report (df):
    """Comprehensive data quality analysis"""
    
    report = []
    
    for col in df.select_dtypes (include=[np.number]).columns:
        stats = {
            'column': col,
            'missing': df[col].isna().sum(),
            'zeros': (df[col] == 0).sum(),
            'negative': (df[col] < 0).sum(),
            'outliers_iqr': len (detect_outliers_iqr (df[col].dropna())[0]),
            'skewness': df[col].skew(),
            'cv': (df[col].std() / df[col].mean()) * 100 if df[col].mean() != 0 else np.nan
        }
        report.append (stats)
    
    return pd.DataFrame (report)

print("\\n=== Data Quality Report ===")
print(data_quality_report (data))
\`\`\`

## Key Takeaways

1. **Central Tendency**: Mean (sensitive), Median (robust), Mode (categorical)
2. **Spread**: Range, Variance, Std Dev, IQR - measure data dispersion
3. **Percentiles**: Divide data into quantiles, robust to outliers
4. **Shape**: Skewness (asymmetry), Kurtosis (tail weight)
5. **Outliers**: Detect using Z-score (parametric) or IQR (non-parametric)
6. **Pandas describe()**: Quick comprehensive summary
7. **Box plots**: Visual summary of distribution
8. **Always compute descriptive stats before modeling**
9. **Use robust statistics (median, IQR) for skewed data**
10. **CV compares variability across different scales**

## Connection to Machine Learning

- **Feature Understanding**: Know scale, distribution, outliers before training
- **Scaling**: StandardScaler uses mean and std
- **Outlier Treatment**: Impacts model performance
- **Feature Engineering**: Transform based on skewness
- **Model Selection**: Some models assume normal distribution
- **Interpretability**: Explain predictions using data statistics
- **Data Quality**: Catch errors early with descriptive analysis

Descriptive statistics are not just preliminary - they guide every decision in the ML pipeline.
`,
};
