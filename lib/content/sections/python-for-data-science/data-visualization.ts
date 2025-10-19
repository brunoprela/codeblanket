/**
 * Section: Data Visualization
 * Module: Python for Data Science
 *
 * Covers matplotlib fundamentals, seaborn for statistical plots, pandas plotting, customization, and publication-quality figures
 */

export const dataVisualization = {
  id: 'data-visualization',
  title: 'Data Visualization',
  content: `
# Data Visualization

## Introduction

Data visualization is crucial for understanding patterns, communicating insights, and detecting anomalies. Python offers powerful visualization libraries: matplotlib for fine control, seaborn for statistical plots, and pandas for quick exploration.

**Key Libraries:**
- **Matplotlib**: Low-level, full control
- **Seaborn**: Statistical visualizations, beautiful defaults
- **Pandas**: Quick plots directly from DataFrames
- **Plotly**: Interactive visualizations (bonus)

\`\`\`python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
\`\`\`

## Matplotlib Fundamentals

### Basic Plot Types

\`\`\`python
# Create sample data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Line plot
plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='sin(x)', linewidth=2)
plt.plot(x, y2, label='cos(x)', linewidth=2, linestyle='--')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Sine and Cosine')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(x, y1, alpha=0.5, s=50, c=y2, cmap='viridis')
plt.colorbar(label='cos(x)')
plt.xlabel('X')
plt.ylabel('sin(X)')
plt.title('Scatter Plot with Color Map')
plt.show()

# Bar plot
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 32]

plt.figure(figsize=(10, 6))
bars = plt.bar(categories, values, color=['red', 'blue', 'green', 'orange', 'purple'])
plt.xlabel('Category')
plt.ylabel('Value')
plt.title('Bar Chart')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height}',
             ha='center', va='bottom')
plt.show()

# Histogram
data = np.random.normal(100, 15, 1000)

plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, edgecolor='black', alpha=0.7)
plt.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.1f}')
plt.axvline(data.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {data.median():.1f}')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram with Mean and Median')
plt.legend()
plt.show()

# Box plot
data = [np.random.normal(100, 10, 100),
        np.random.normal(110, 15, 100),
        np.random.normal(105, 12, 100)]

plt.figure(figsize=(10, 6))
plt.boxplot(data, labels=['Group A', 'Group B', 'Group C'])
plt.ylabel('Value')
plt.title('Box Plot Comparison')
plt.grid(True, alpha=0.3)
plt.show()
\`\`\`

### Subplots

\`\`\`python
# Create 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Line
axes[0, 0].plot(x, y1)
axes[0, 0].set_title('Sine Wave')
axes[0, 0].set_xlabel('X')
axes[0, 0].set_ylabel('Y')

# Plot 2: Scatter
axes[0, 1].scatter(x, y2, alpha=0.5)
axes[0, 1].set_title('Cosine Scatter')
axes[0, 1].set_xlabel('X')
axes[0, 1].set_ylabel('Y')

# Plot 3: Histogram
axes[1, 0].hist(np.random.randn(1000), bins=30, edgecolor='black')
axes[1, 0].set_title('Histogram')
axes[1, 0].set_xlabel('Value')
axes[1, 0].set_ylabel('Frequency')

# Plot 4: Bar
axes[1, 1].bar(categories, values)
axes[1, 1].set_title('Bar Chart')
axes[1, 1].set_xlabel('Category')
axes[1, 1].set_ylabel('Value')

plt.tight_layout()
plt.show()

# Alternative: Different subplot sizes
fig = plt.figure(figsize=(12, 8))

# Top subplot spans full width
ax1 = plt.subplot(2, 2, (1, 2))  # Row 1, spans columns 1-2
ax1.plot(x, y1)
ax1.set_title('Full Width Plot')

# Bottom left
ax2 = plt.subplot(2, 2, 3)
ax2.scatter(x, y2)
ax2.set_title('Bottom Left')

# Bottom right
ax3 = plt.subplot(2, 2, 4)
ax3.hist(data, bins=20)
ax3.set_title('Bottom Right')

plt.tight_layout()
plt.show()
\`\`\`

### Customization

\`\`\`python
# Highly customized plot
fig, ax = plt.subplots(figsize=(12, 6))

# Plot data
ax.plot(x, y1, label='sin(x)', linewidth=2.5, color='#FF6B6B')
ax.plot(x, y2, label='cos(x)', linewidth=2.5, color='#4ECDC4')

# Fill between
ax.fill_between(x, y1, y2, alpha=0.2, color='gray')

# Customize spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)

# Customize ticks
ax.tick_params(labelsize=12, width=2)

# Labels and title
ax.set_xlabel('X Axis', fontsize=14, fontweight='bold')
ax.set_ylabel('Y Axis', fontsize=14, fontweight='bold')
ax.set_title('Custom Styled Plot', fontsize=16, fontweight='bold', pad=20)

# Legend
ax.legend(frameon=True, shadow=True, fontsize=12)

# Grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()
\`\`\`

## Seaborn Statistical Plots

### Distribution Plots

\`\`\`python
# Generate data
data = np.random.normal(100, 15, 1000)

# Distribution plot with KDE
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(data, bins=30, kde=True, color='skyblue')
plt.title('Histogram with KDE')

plt.subplot(1, 2, 2)
sns.kdeplot(data, fill=True, color='coral')
plt.title('KDE Plot')

plt.tight_layout()
plt.show()

# Multiple distributions
df = pd.DataFrame({
    'Group A': np.random.normal(100, 10, 500),
    'Group B': np.random.normal(110, 15, 500),
    'Group C': np.random.normal(105, 12, 500)
})

plt.figure(figsize=(12, 6))
for column in df.columns:
    sns.kdeplot(df[column], label=column, fill=True, alpha=0.5)
plt.title('Multiple Distributions')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.show()
\`\`\`

### Categorical Plots

\`\`\`python
# Sample data
df = pd.DataFrame({
    'Category': ['A', 'B', 'C', 'D'] * 50,
    'Value': np.concatenate([
        np.random.normal(20, 5, 50),
        np.random.normal(30, 7, 50),
        np.random.normal(25, 6, 50),
        np.random.normal(35, 8, 50)
    ]),
    'Group': np.random.choice(['X', 'Y'], 200)
})

# Box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Category', y='Value', hue='Group')
plt.title('Box Plot by Category and Group')
plt.show()

# Violin plot (box plot + KDE)
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='Category', y='Value', hue='Group', split=True)
plt.title('Violin Plot')
plt.show()

# Bar plot with error bars
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Category', y='Value', hue='Group', ci=95)
plt.title('Bar Plot with 95% Confidence Intervals')
plt.show()

# Point plot
plt.figure(figsize=(10, 6))
sns.pointplot(data=df, x='Category', y='Value', hue='Group')
plt.title('Point Plot')
plt.show()
\`\`\`

### Relationship Plots

\`\`\`python
# Scatter with regression
df = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100)
})
df['y'] = df['x'] * 2 + df['y'] * 0.5  # Add correlation

plt.figure(figsize=(10, 6))
sns.regplot(data=df, x='x', y='y', scatter_kws={'alpha':0.5})
plt.title('Scatter Plot with Regression Line')
plt.show()

# Joint plot (scatter + marginal distributions)
sns.jointplot(data=df, x='x', y='y', kind='reg', height=8)
plt.show()

# Pair plot (all combinations)
iris = sns.load_dataset('iris')
sns.pairplot(iris, hue='species', diag_kind='kde', height=2.5)
plt.show()
\`\`\`

### Heatmaps

\`\`\`python
# Correlation matrix
df = pd.DataFrame(np.random.randn(100, 5), columns=['A', 'B', 'C', 'D', 'E'])
corr = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap')
plt.show()

# Pivot table heatmap
flights = sns.load_dataset('flights')
flights_pivot = flights.pivot('month', 'year', 'passengers')

plt.figure(figsize=(12, 8))
sns.heatmap(flights_pivot, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Flight Passengers Over Time')
plt.show()
\`\`\`

## Pandas Built-in Plotting

\`\`\`python
# Create time series data
dates = pd.date_range('2024-01-01', periods=365, freq='D')
df = pd.DataFrame({
    'Sales': np.random.randn(365).cumsum() + 1000,
    'Costs': np.random.randn(365).cumsum() + 800,
    'Profit': np.random.randn(365).cumsum() + 200
}, index=dates)

# Line plot
df.plot(figsize=(12, 6), title='Time Series Data')
plt.ylabel('Amount ($)')
plt.show()

# Multiple subplots
df.plot(subplots=True, figsize=(12, 10), title='Time Series Subplots')
plt.show()

# Area plot
df[['Sales', 'Costs']].plot(kind='area', alpha=0.4, figsize=(12, 6))
plt.title('Area Plot')
plt.ylabel('Amount ($)')
plt.show()

# Bar plot
monthly = df.resample('ME').sum()
monthly.plot(kind='bar', figsize=(12, 6))
plt.title('Monthly Totals')
plt.ylabel('Amount ($)')
plt.xticks(rotation=45)
plt.show()

# Histogram
df.plot(kind='hist', bins=30, alpha=0.7, figsize=(12, 6))
plt.title('Distribution of Values')
plt.xlabel('Value')
plt.show()

# Box plot
df.plot(kind='box', figsize=(10, 6))
plt.title('Box Plot Comparison')
plt.ylabel('Amount ($)')
plt.show()

# Scatter plot
df.plot(kind='scatter', x='Sales', y='Profit', 
        c='Costs', cmap='viridis', s=50, figsize=(10, 6))
plt.title('Sales vs Profit (colored by Costs)')
plt.show()
\`\`\`

## Practical Examples

### Example 1: Stock Price Analysis

\`\`\`python
# Generate stock data
np.random.seed(42)
dates = pd.date_range('2023-01-01', '2024-12-31', freq='B')
returns = np.random.normal(0.0005, 0.02, len(dates))
price = 100 * (1 + returns).cumprod()
volume = np.random.randint(1000000, 10000000, len(dates))

df = pd.DataFrame({
    'Price': price,
    'Volume': volume
}, index=dates)

# Calculate technical indicators
df['SMA_20'] = df['Price'].rolling(20).mean()
df['SMA_50'] = df['Price'].rolling(50).mean()
df['Returns'] = df['Price'].pct_change()

# Create comprehensive visualization
fig = plt.figure(figsize=(14, 10))

# Price and moving averages
ax1 = plt.subplot(3, 1, 1)
ax1.plot(df.index, df['Price'], label='Price', linewidth=1.5)
ax1.plot(df.index, df['SMA_20'], label='SMA 20', linewidth=1.5, alpha=0.7)
ax1.plot(df.index, df['SMA_50'], label='SMA 50', linewidth=1.5, alpha=0.7)
ax1.set_ylabel('Price ($)')
ax1.set_title('Stock Price Analysis')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Volume
ax2 = plt.subplot(3, 1, 2)
ax2.bar(df.index, df['Volume'], alpha=0.5, color='gray')
ax2.set_ylabel('Volume')
ax2.set_title('Trading Volume')
ax2.grid(True, alpha=0.3)

# Returns distribution
ax3 = plt.subplot(3, 1, 3)
ax3.hist(df['Returns'].dropna(), bins=50, edgecolor='black', alpha=0.7)
ax3.axvline(df['Returns'].mean(), color='red', linestyle='--', 
            label=f'Mean: {df["Returns"].mean():.4f}')
ax3.set_xlabel('Daily Returns')
ax3.set_ylabel('Frequency')
ax3.set_title('Returns Distribution')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
\`\`\`

### Example 2: Sales Dashboard

\`\`\`python
# Generate sales data
np.random.seed(42)
dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
df = pd.DataFrame({
    'Date': dates,
    'Sales': np.random.uniform(1000, 5000, len(dates)),
    'Region': np.random.choice(['North', 'South', 'East', 'West'], len(dates)),
    'Product': np.random.choice(['A', 'B', 'C'], len(dates))
})

# Aggregate data
monthly = df.groupby(df['Date'].dt.to_period('M'))['Sales'].sum()
by_region = df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
by_product = df.groupby('Product')['Sales'].sum()

# Create dashboard
fig = plt.figure(figsize=(16, 10))

# Monthly trend
ax1 = plt.subplot(2, 3, (1, 3))
ax1.plot(monthly.index.astype(str), monthly.values, marker='o', linewidth=2)
ax1.set_xlabel('Month')
ax1.set_ylabel('Sales ($)')
ax1.set_title('Monthly Sales Trend', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.tick_params(rotation=45)

# Sales by region
ax2 = plt.subplot(2, 3, 4)
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
bars = ax2.bar(by_region.index, by_region.values, color=colors)
ax2.set_xlabel('Region')
ax2.set_ylabel('Total Sales ($)')
ax2.set_title('Sales by Region')
# Add values on bars
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'\${height:, .0f
}',
ha = 'center', va = 'bottom')

# Sales by product(pie chart)
ax3 = plt.subplot(2, 3, 5)
ax3.pie(by_product.values, labels = by_product.index, autopct = '%1.1f%%',
    startangle = 90, colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'])
ax3.set_title('Sales by Product')

# Daily distribution
ax4 = plt.subplot(2, 3, 6)
ax4.hist(df['Sales'], bins = 30, edgecolor = 'black', alpha = 0.7, color = '#4ECDC4')
ax4.set_xlabel('Daily Sales ($)')
ax4.set_ylabel('Frequency')
ax4.set_title('Sales Distribution')
ax4.axvline(df['Sales'].mean(), color = 'red', linestyle = '--', linewidth = 2)

plt.tight_layout()
plt.show()
\`\`\`

### Example 3: Correlation Analysis

\`\`\`python
# Generate correlated data
np.random.seed(42)
n = 200
df = pd.DataFrame({
    'Feature1': np.random.randn(n),
    'Feature2': np.random.randn(n),
    'Feature3': np.random.randn(n),
    'Feature4': np.random.randn(n),
    'Feature5': np.random.randn(n)
})

# Create correlations
df['Feature2'] = df['Feature1'] * 0.8 + df['Feature2'] * 0.4
df['Feature3'] = df['Feature1'] * -0.6 + df['Feature3'] * 0.5
df['Feature4'] = df['Feature2'] * 0.5 + df['Feature4'] * 0.7
df['Target'] = (df['Feature1'] * 2 + df['Feature2'] * 1.5 - 
                df['Feature3'] * 0.8 + np.random.randn(n) * 0.5)

# Correlation matrix
corr = df.corr()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Heatmap
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=1, ax=axes[0],
            cbar_kws={"shrink": 0.8})
axes[0].set_title('Correlation Heatmap', fontsize=14, fontweight='bold')

# Clustered heatmap
sns.clustermap(corr, annot=True, cmap='coolwarm', center=0,
               square=True, linewidths=1, figsize=(10, 10))
plt.show()

# Scatter matrix of top correlates with target
top_features = corr['Target'].abs().sort_values(ascending=False)[1:4].index

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, feature in enumerate(top_features):
    axes[idx].scatter(df[feature], df['Target'], alpha=0.5)
    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel('Target')
    axes[idx].set_title(f'Correlation: {corr.loc[feature, "Target"]:.3f}')
    # Add regression line
    z = np.polyfit(df[feature], df['Target'], 1)
    p = np.poly1d(z)
    axes[idx].plot(df[feature], p(df[feature]), "r--", alpha=0.8)

plt.tight_layout()
plt.show()
\`\`\`

## Customization and Styling

\`\`\`python
# Custom style
custom_style = {
    'axes.facecolor': '#F0F0F0',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 2,
    'axes.grid': True,
    'grid.color': 'white',
    'grid.linestyle': '-',
    'grid.linewidth': 1.5,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'font.size': 12,
    'figure.facecolor': 'white'
}

with plt.style.context(custom_style):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.random.randn(100).cumsum(), linewidth=2, color='#FF6B6B')
    ax.set_title('Custom Styled Plot', fontsize=16, fontweight='bold')
    ax.set_xlabel('X Axis', fontsize=14)
    ax.set_ylabel('Y Axis', fontsize=14)
    plt.show()

# Seaborn themes
for style in ['darkgrid', 'whitegrid', 'dark', 'white', 'ticks']:
    sns.set_style(style)
    plt.figure(figsize=(10, 4))
    plt.plot(np.random.randn(100).cumsum())
    plt.title(f'Seaborn Style: {style}')
    plt.show()
\`\`\`

## Key Takeaways

1. **Matplotlib**: Full control, publication-quality plots
2. **Seaborn**: Statistical visualizations with beautiful defaults
3. **Pandas**: Quick exploratory plots
4. **Subplots**: Combine multiple visualizations
5. **Customization**: Colors, styles, labels for clarity
6. **Context**: Choose right plot type for your data
7. **Storytelling**: Visualizations should communicate insights clearly

Effective visualization makes data accessible and insights actionable!
`,
};
