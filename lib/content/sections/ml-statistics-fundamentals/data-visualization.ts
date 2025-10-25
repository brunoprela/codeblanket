/**
 * Data Visualization Section
 */

export const datavisualizationSection = {
  id: 'data-visualization',
  title: 'Data Visualization',
  content: `# Data Visualization

## Introduction

"A picture is worth a thousand numbers." Data visualization is not just about making pretty charts - it's a critical tool for:

- **Understanding Data**: Patterns invisible in tables become obvious in plots
- **Detecting Anomalies**: Outliers, errors, and unusual patterns jump out visually
- **Communicating Insights**: Stakeholders understand visualizations better than statistics
- **Model Diagnostics**: Residual plots, learning curves reveal model behavior
- **Feature Engineering**: Visualizations suggest transformations and interactions

In machine learning, visualization happens at every stage:
- **EDA**: Understanding data before modeling
- **Feature Engineering**: Discovering relationships
- **Model Selection**: Comparing performance visually
- **Debugging**: Diagnosing what went wrong
- **Reporting**: Communicating results

## Matplotlib Fundamentals

### Basic Plotting

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Generate data
np.random.seed(42)
x = np.linspace(0, 10, 100)
y = 2 * x + 1 + np.random.normal(0, 1, 100)

# Create figure and axis
fig, ax = plt.subplots (figsize=(10, 6))

# Plot
ax.scatter (x, y, alpha=0.6, label='Data points')
ax.plot (x, 2*x + 1, 'r--', linewidth=2, label='True relationship')

# Labels and title
ax.set_xlabel('X variable', fontsize=12)
ax.set_ylabel('Y variable', fontsize=12)
ax.set_title('Simple Scatter Plot with True Relationship', fontsize=14, fontweight='bold')
ax.legend (loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('basic_scatter.png', dpi=300, bbox_inches='tight')
plt.show()

print("Key elements of a good plot:")
print("✓ Clear axis labels with units")
print("✓ Descriptive title")
print("✓ Legend explaining what's shown")
print("✓ Appropriate size and readability")
print("✓ Grid for easier reading (optional)")
\`\`\`

### Subplots for Multiple Visualizations

\`\`\`python
# Create a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Generate different datasets
data1 = np.random.normal(0, 1, 1000)
data2 = np.random.exponential(2, 1000)
x_scatter = np.random.rand(100)
y_scatter = 2 * x_scatter + np.random.normal(0, 0.1, 100)

# Plot 1: Histogram
axes[0, 0].hist (data1, bins=30, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Normal Distribution')
axes[0, 0].set_xlabel('Value')
axes[0, 0].set_ylabel('Frequency')

# Plot 2: Histogram
axes[0, 1].hist (data2, bins=30, edgecolor='black', alpha=0.7, color='orange')
axes[0, 1].set_title('Exponential Distribution')
axes[0, 1].set_xlabel('Value')
axes[0, 1].set_ylabel('Frequency')

# Plot 3: Scatter plot
axes[1, 0].scatter (x_scatter, y_scatter, alpha=0.6)
axes[1, 0].set_title('Correlation Example')
axes[1, 0].set_xlabel('X')
axes[1, 0].set_ylabel('Y')

# Plot 4: Line plot
axes[1, 1].plot (np.cumsum (np.random.randn(100)), linewidth=2)
axes[1, 1].set_title('Random Walk')
axes[1, 1].set_xlabel('Time')
axes[1, 1].set_ylabel('Position')

plt.tight_layout()
plt.savefig('subplots_example.png', dpi=300, bbox_inches='tight')
plt.show()
\`\`\`

## Univariate Visualizations

### Histograms and Density Plots

\`\`\`python
def plot_distribution_analysis (data, title="Distribution Analysis"):
    """Comprehensive univariate distribution visualization"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Histogram with KDE
    axes[0, 0].hist (data, bins=30, density=True, alpha=0.7, 
                     edgecolor='black', label='Histogram')
    
    # Add KDE (Kernel Density Estimation)
    from scipy.stats import gaussian_kde
    kde = gaussian_kde (data)
    x_range = np.linspace (data.min(), data.max(), 100)
    axes[0, 0].plot (x_range, kde (x_range), 'r-', linewidth=2, label='KDE')
    axes[0, 0].set_title('Histogram + Density')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()
    
    # 2. Box plot
    bp = axes[0, 1].boxplot (data, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    axes[0, 1].set_title('Box Plot')
    axes[0, 1].set_ylabel('Value')
    
    # Add statistics as text
    stats_text = f"Mean: {np.mean (data):.2f}\\n"
    stats_text += f"Median: {np.median (data):.2f}\\n"
    stats_text += f"Std: {np.std (data):.2f}\\n"
    stats_text += f"Skew: {pd.Series (data).skew():.2f}"
    axes[0, 1].text(1.2, np.median (data), stats_text, 
                     fontsize=10, verticalalignment='center')
    
    # 3. Cumulative Distribution Function (CDF)
    sorted_data = np.sort (data)
    cumulative = np.arange(1, len (sorted_data) + 1) / len (sorted_data)
    axes[1, 0].plot (sorted_data, cumulative, linewidth=2)
    axes[1, 0].set_title('Cumulative Distribution Function')
    axes[1, 0].set_xlabel('Value')
    axes[1, 0].set_ylabel('Cumulative Probability')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Q-Q Plot (Quantile-Quantile)
    from scipy import stats
    stats.probplot (data, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot (vs Normal)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle (title, fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# Example: Analyzing a feature
feature_data = np.random.exponential(2, 1000)
plot_distribution_analysis (feature_data, "Feature Distribution Analysis")
\`\`\`

### Violin Plots

\`\`\`python
# Violin plots combine box plot with KDE
fig, ax = plt.subplots (figsize=(10, 6))

# Generate data for different categories
categories = ['A', 'B', 'C', 'D']
data_by_category = [
    np.random.normal(0, 1, 100),
    np.random.normal(1, 1.5, 100),
    np.random.normal(-1, 0.8, 100),
    np.random.exponential(1, 100)
]

# Create violin plot
parts = ax.violinplot (data_by_category, positions=range (len (categories)),
                       showmeans=True, showmedians=True)

ax.set_xticks (range (len (categories)))
ax.set_xticklabels (categories)
ax.set_xlabel('Category')
ax.set_ylabel('Value')
ax.set_title('Violin Plot: Distribution by Category')
ax.grid (axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('violin_plot.png', dpi=300, bbox_inches='tight')
plt.show()

print("Violin plots show:")
print("• Full distribution shape (density)")
print("• Median and quartiles")
print("• Comparison across categories")
print("• More info than box plots alone")
\`\`\`

## Bivariate Visualizations

### Scatter Plots with Insights

\`\`\`python
def enhanced_scatter_plot (x, y, title="Correlation Analysis"):
    """Scatter plot with regression line and statistics"""
    
    fig, ax = plt.subplots (figsize=(10, 6))
    
    # Scatter plot with color based on density
    from scipy.stats import gaussian_kde
    xy = np.vstack([x, y])
    z = gaussian_kde (xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    
    scatter = ax.scatter (x, y, c=z, s=50, alpha=0.6, cmap='viridis',
                          edgecolors='black', linewidth=0.5)
    
    # Add regression line
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress (x, y)
    line_x = np.array([x.min(), x.max()])
    line_y = slope * line_x + intercept
    ax.plot (line_x, line_y, 'r--', linewidth=2, 
            label=f'y = {slope:.2f}x + {intercept:.2f}')
    
    # Add statistics
    from scipy.stats import pearsonr
    corr, _ = pearsonr (x, y)
    stats_text = f'Correlation: {corr:.3f}\\n'
    stats_text += f'R²: {r_value**2:.3f}\\n'
    stats_text += f'p-value: {p_value:.4f}'
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict (boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('X variable', fontsize=12)
    ax.set_ylabel('Y variable', fontsize=12)
    ax.set_title (title, fontsize=14, fontweight='bold')
    ax.legend (loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar (scatter, ax=ax)
    cbar.set_label('Density', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.savefig('enhanced_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()

# Example
x = np.random.rand(500) * 10
y = 2 * x + 1 + np.random.normal(0, 2, 500)
enhanced_scatter_plot (x, y, "Feature Correlation Analysis")
\`\`\`

### Hexbin Plots (For Dense Data)

\`\`\`python
# When you have too many points, hexbin is better than scatter
np.random.seed(42)
n = 10000
x = np.random.randn (n)
y = x + np.random.randn (n) * 0.5

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Regular scatter (overcrowded)
ax1.scatter (x, y, alpha=0.1, s=1)
ax1.set_title('Scatter Plot (Overcrowded)')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')

# Hexbin plot (shows density)
hb = ax2.hexbin (x, y, gridsize=30, cmap='YlOrRd')
ax2.set_title('Hexbin Plot (Density Visible)')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
cb = fig.colorbar (hb, ax=ax2)
cb.set_label('Counts')

plt.tight_layout()
plt.savefig('hexbin_vs_scatter.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"With {n:,} points, hexbin is much clearer!")
\`\`\`

## Multivariate Visualizations

### Correlation Heatmaps

\`\`\`python
def plot_correlation_matrix (data, title="Correlation Matrix"):
    """Beautiful correlation heatmap with insights"""
    
    # Compute correlation matrix
    corr = data.corr()
    
    # Create mask for upper triangle
    mask = np.triu (np.ones_like (corr, dtype=bool))
    
    # Set up the figure
    fig, ax = plt.subplots (figsize=(10, 8))
    
    # Draw heatmap
    sns.heatmap (corr, mask=mask, annot=True, fmt='.2f', 
                cmap='coolwarm', center=0, square=True,
                linewidths=1, cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1, ax=ax)
    
    ax.set_title (title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Find high correlations
    high_corr = []
    for i in range (len (corr.columns)):
        for j in range (i+1, len (corr.columns)):
            if abs (corr.iloc[i, j]) > 0.7:
                high_corr.append((
                    corr.columns[i], 
                    corr.columns[j], 
                    corr.iloc[i, j]
                ))
    
    if high_corr:
        print("\\nHigh correlations detected:")
        for var1, var2, corr_val in high_corr:
            print(f"  {var1} ↔ {var2}: {corr_val:.3f}")
        print("\\n⚠ Consider removing one feature from highly correlated pairs")

# Example with real-ish data
data = pd.DataFrame({
    'price': np.random.normal(200, 50, 100),
    'sqft': np.random.normal(1500, 300, 100),
    'bedrooms': np.random.choice([2, 3, 4, 5], 100),
    'bathrooms': np.random.choice([1, 2, 3], 100),
    'age': np.random.exponential(20, 100)
})

# Add correlated feature
data['price'] = 100 + 0.1 * data['sqft'] + 20 * data['bedrooms'] + np.random.normal(0, 20, 100)

plot_correlation_matrix (data, "Housing Features Correlation")
\`\`\`

### Pair Plots

\`\`\`python
def create_pairplot (data, target_col=None):
    """Create comprehensive pair plot"""
    
    if target_col:
        # Color by target variable
        g = sns.pairplot (data, hue=target_col, diag_kind='kde',
                         plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'k'},
                         corner=True)
    else:
        g = sns.pairplot (data, diag_kind='kde',
                         plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'k'},
                         corner=True)
    
    g.fig.suptitle('Pair Plot: Feature Relationships', y=1.01, fontsize=14, fontweight='bold')
    plt.savefig('pairplot.png', dpi=300, bbox_inches='tight')
    plt.show()

# Example with classification target
data_with_target = data.copy()
data_with_target['expensive'] = (data_with_target['price'] > data_with_target['price'].median()).astype (int)

create_pairplot (data_with_target[['sqft', 'bedrooms', 'age', 'expensive']], 'expensive')

print("Pair plots reveal:")
print("• All pairwise relationships")
print("• Clusters and separability")
print("• Feature distributions")
print("• Linear vs non-linear patterns")
\`\`\`

## Time Series Visualizations

\`\`\`python
def plot_time_series_analysis (dates, values, title="Time Series Analysis"):
    """Comprehensive time series visualization"""
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 1. Original series
    axes[0].plot (dates, values, linewidth=1.5)
    axes[0].set_title('Original Time Series')
    axes[0].set_ylabel('Value')
    axes[0].grid(True, alpha=0.3)
    
    # Add moving average
    window = 30
    ma = pd.Series (values).rolling (window=window).mean()
    axes[0].plot (dates, ma, 'r-', linewidth=2, label=f'{window}-day MA')
    axes[0].legend()
    
    # 2. Daily returns/changes
    returns = np.diff (values) / values[:-1] * 100
    axes[1].plot (dates[1:], returns, linewidth=1, alpha=0.7)
    axes[1].axhline (y=0, color='r', linestyle='--', linewidth=1)
    axes[1].set_title('Daily Returns (%)')
    axes[1].set_ylabel('Return (%)')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Rolling statistics
    rolling_mean = pd.Series (values).rolling (window=30).mean()
    rolling_std = pd.Series (values).rolling (window=30).std()
    
    axes[2].plot (dates, rolling_mean, label='Rolling Mean', linewidth=2)
    axes[2].fill_between (dates, 
                          rolling_mean - 2*rolling_std,
                          rolling_mean + 2*rolling_std,
                          alpha=0.3, label='±2 Std Dev')
    axes[2].set_title('Rolling Statistics (30-day window)')
    axes[2].set_xlabel('Date')
    axes[2].set_ylabel('Value')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle (title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('time_series_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# Example: Stock price-like data
dates = pd.date_range('2020-01-01', periods=365)
prices = 100 * np.exp (np.cumsum (np.random.randn(365) * 0.02))

plot_time_series_analysis (dates, prices, "Stock Price Analysis")
\`\`\`

## ML-Specific Visualizations

### Learning Curves

\`\`\`python
def plot_learning_curve (train_scores, val_scores, train_sizes):
    """Plot learning curve to diagnose bias/variance"""
    
    fig, ax = plt.subplots (figsize=(10, 6))
    
    # Calculate mean and std
    train_mean = np.mean (train_scores, axis=1)
    train_std = np.std (train_scores, axis=1)
    val_mean = np.mean (val_scores, axis=1)
    val_std = np.std (val_scores, axis=1)
    
    # Plot learning curves
    ax.plot (train_sizes, train_mean, 'o-', linewidth=2, 
            label='Training score', color='blue')
    ax.fill_between (train_sizes, train_mean - train_std, 
                     train_mean + train_std, alpha=0.2, color='blue')
    
    ax.plot (train_sizes, val_mean, 'o-', linewidth=2,
            label='Validation score', color='red')
    ax.fill_between (train_sizes, val_mean - val_std,
                     val_mean + val_std, alpha=0.2, color='red')
    
    # Add annotations
    gap = train_mean[-1] - val_mean[-1]
    if gap > 0.1:
        ax.text(0.5, 0.1, 'High variance (overfitting)\\nConsider: More data, regularization',
                transform=ax.transAxes, fontsize=10,
                bbox=dict (boxstyle='round', facecolor='yellow', alpha=0.5))
    elif val_mean[-1] < 0.7:
        ax.text(0.5, 0.1, 'High bias (underfitting)\\nConsider: More complex model, more features',
                transform=ax.transAxes, fontsize=10,
                bbox=dict (boxstyle='round', facecolor='yellow', alpha=0.5))
    
    ax.set_xlabel('Training Set Size', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Learning Curve: Diagnosing Model Performance', fontsize=14, fontweight='bold')
    ax.legend (loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('learning_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

# Example: Generate learning curve data
train_sizes = np.array([50, 100, 200, 400, 800, 1600])
train_scores = np.array([[0.9, 0.89, 0.91], [0.92, 0.91, 0.93], 
                          [0.94, 0.93, 0.95], [0.95, 0.94, 0.96],
                          [0.96, 0.95, 0.97], [0.97, 0.96, 0.98]])
val_scores = np.array([[0.75, 0.73, 0.77], [0.78, 0.76, 0.80],
                        [0.81, 0.79, 0.83], [0.83, 0.81, 0.85],
                        [0.84, 0.82, 0.86], [0.85, 0.83, 0.87]])

plot_learning_curve (train_scores, val_scores, train_sizes)
\`\`\`

### Feature Importance Plots

\`\`\`python
def plot_feature_importance (feature_names, importances, title="Feature Importance"):
    """Visualize feature importance from tree-based models"""
    
    # Sort by importance
    indices = np.argsort (importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    fig, ax = plt.subplots (figsize=(10, 6))
    
    # Create bar plot
    colors = plt.cm.viridis (sorted_importances / sorted_importances.max())
    bars = ax.barh (range (len (sorted_features)), sorted_importances, color=colors)
    
    ax.set_yticks (range (len (sorted_features)))
    ax.set_yticklabels (sorted_features)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title (title, fontsize=14, fontweight='bold')
    ax.invert_yaxis()  # Top to bottom
    
    # Add value labels
    for i, v in enumerate (sorted_importances):
        ax.text (v, i, f' {v:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Top 3 features:")
    for i in range (min(3, len (sorted_features))):
        print(f"  {i+1}. {sorted_features[i]}: {sorted_importances[i]:.3f}")

# Example
features = ['age', 'income', 'credit_score', 'debt_ratio', 'employment_length']
importances = np.array([0.15, 0.35, 0.30, 0.12, 0.08])

plot_feature_importance (features, importances, "Loan Default Prediction: Feature Importance")
\`\`\`

## Best Practices for ML Visualization

\`\`\`python
def comprehensive_eda_report (data):
    """Generate comprehensive EDA visualization report"""
    
    print("Generating comprehensive EDA report...")
    
    # 1. Distribution of all numeric features
    numeric_cols = data.select_dtypes (include=[np.number]).columns
    n_cols = len (numeric_cols)
    n_rows = (n_cols + 2) // 3
    
    fig, axes = plt.subplots (n_rows, 3, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_cols > 3 else [axes]
    
    for idx, col in enumerate (numeric_cols):
        axes[idx].hist (data[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
        axes[idx].set_title (f'{col}\\n (skew={data[col].skew():.2f})')
        axes[idx].set_xlabel('Value')
        axes[idx].set_ylabel('Frequency')
    
    # Remove empty subplots
    for idx in range (n_cols, len (axes)):
        fig.delaxes (axes[idx])
    
    plt.suptitle('Feature Distributions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('eda_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Correlation heatmap
    plot_correlation_matrix (data, "Feature Correlations")
    
    # 3. Missing value analysis
    missing = data.isnull().sum()
    missing = missing[missing > 0].sort_values (ascending=False)
    
    if len (missing) > 0:
        fig, ax = plt.subplots (figsize=(10, 6))
        missing.plot (kind='barh', ax=ax, color='coral')
        ax.set_xlabel('Missing Count')
        ax.set_title('Missing Values by Feature', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('missing_values.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    print("✓ EDA report complete!")
    print("  - Distribution plots saved")
    print("  - Correlation heatmap saved")
    if len (missing) > 0:
        print("  - Missing value plot saved")

# Example usage
comprehensive_eda_report (data)
\`\`\`

## Key Takeaways

1. **Histograms**: Show distribution shape, identify skewness and outliers
2. **Box plots**: Summarize distribution with quartiles, good for comparing groups
3. **Scatter plots**: Reveal relationships between two variables
4. **Heatmaps**: Display correlation matrices and detect multicollinearity
5. **Pair plots**: Show all pairwise relationships at once
6. **Time series plots**: Track changes over time, identify trends and seasonality
7. **Learning curves**: Diagnose overfitting (high variance) vs underfitting (high bias)
8. **Feature importance**: Identify which features drive predictions

## Visualization Guidelines

### Do\'s ✓ Always label axes with units
✓ Use descriptive titles
✓ Include legends when multiple series
✓ Choose appropriate chart for data type
✓ Use color meaningfully (not decoration)
✓ Make it readable (font size, resolution)
✓ Show uncertainty (error bars, confidence intervals)

### Don'ts
✗ Don't use 3D charts (hard to read)
✗ Don't use pie charts (bar charts are better)
✗ Don't omit axis labels
✗ Don't use too many colors
✗ Don't distort scales to mislead
✗ Don't use default figure sizes (too small)

## Connection to Machine Learning

- **EDA phase**: Visualize distributions, correlations, outliers
- **Feature engineering**: Plots suggest transformations
- **Model selection**: Learning curves show bias/variance
- **Hyperparameter tuning**: Visualize validation curves
- **Model interpretation**: Feature importance, SHAP plots
- **Debugging**: Residual plots, confusion matrices
- **Communication**: Present results to stakeholders

Master visualization and you'll understand your data - and models - much better!
`,
};
