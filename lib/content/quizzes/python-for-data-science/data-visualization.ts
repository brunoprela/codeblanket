import { QuizQuestion } from '../../../types';

export const datavisualizationQuiz: QuizQuestion[] = [
  {
    id: 'data-visualization-dq-1',
    question:
      'Discuss the principles of effective data visualization. What makes a visualization good vs bad? Provide examples of common mistakes and how to avoid them.',
    sampleAnswer: `Effective data visualization is both an art and a science. Good visualizations reveal insights, while bad ones mislead or confuse.

**Principles of Effective Visualization:**

**1. Clarity and Purpose**

Every visualization should answer a specific question.

\`\`\`python
# BAD: Cluttered and unclear
plt.figure(figsize=(15, 10))
for i in range(50):
    plt.plot(np.random.randn(100).cumsum(), alpha=0.3)
plt.title('Data')  # What data? What question?
plt.show()

# GOOD: Clear purpose and clean design
plt.figure(figsize=(10, 6))
stock_prices = pd.DataFrame({
    'AAPL': apple_prices,
    'GOOGL': google_prices,
    'MSFT': microsoft_prices
}, index=dates)

stock_prices.plot(linewidth=2)
plt.title('Tech Stock Prices 2023-2024', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend(title='Stock')
plt.grid(True, alpha=0.3)
plt.show()
\`\`\`

**2. Choose Right Chart Type**

Different data requires different visualizations:

**Comparison:** Bar charts
\`\`\`python
# Comparing categories
sales_by_region = pd.Series({
    'North': 45000,
    'South': 38000,
    'East': 52000,
    'West': 41000
}).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
bars = plt.bar(sales_by_region.index, sales_by_region.values, 
               color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A',])
plt.ylabel('Sales ($)')
plt.title('Sales by Region')
for i, (region, value) in enumerate(sales_by_region.items()):
    plt.text(i, value, f'\${value:,}', ha='center', va='bottom')
plt.show()
            \`\`\`

**Trends over time:** Line charts
\`\`\`python
# Time series
dates = pd.date_range('2024-01-01', periods=365, freq='D')
df = pd.DataFrame({
    'sales': np.random.randn(365).cumsum() + 1000
}, index=dates)

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['sales',], linewidth=2)
plt.title('Daily Sales Trend 2024')
plt.xlabel('Date')
plt.ylabel('Sales ($)')
plt.grid(True, alpha=0.3)
plt.show()
\`\`\`

**Distribution:** Histograms or box plots
\`\`\`python
# Distribution of values
data = np.random.normal(100, 15, 1000)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.hist(data, bins=30, edgecolor='black')
ax1.set_title('Distribution (Histogram)')
ax1.set_xlabel('Value')
ax1.set_ylabel('Frequency')

ax2.boxplot([data], labels=['Data',])
ax2.set_title('Distribution (Box Plot)')
ax2.set_ylabel('Value')

plt.tight_layout()
plt.show()
\`\`\`

**Correlation:** Scatter plots
\`\`\`python
# Relationship between variables
x = np.random.randn(100)
y = x * 2 + np.random.randn(100) * 0.5

plt.figure(figsize=(8, 8))
plt.scatter(x, y, alpha=0.6)
plt.xlabel('Variable X')
plt.ylabel('Variable Y')
plt.title(f'Correlation: {np.corrcoef(x, y)[0,1]:.3f}')
plt.grid(True, alpha=0.3)
plt.show()
\`\`\`

**3. Avoid Common Mistakes**

**Mistake 1: 3D charts (usually misleading)**
\`\`\`python
# BAD: 3D pie chart (distorts perception)
# from mpl_toolkits.mplot3d import Axes3D
# Don't do this!

# GOOD: Simple 2D pie or bar
values = [30, 25, 20, 15, 10]
labels = ['A', 'B', 'C', 'D', 'E',]

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title('Clear 2D Pie Chart')

plt.subplot(1, 2, 2)
plt.bar(labels, values)
plt.title('Even Better: Bar Chart')
plt.ylabel('Value')

plt.tight_layout()
plt.show()
\`\`\`

**Mistake 2: Non-zero baseline**
\`\`\`python
# BAD: Doesn't start at zero (exaggerates differences)
data = [98, 99, 100, 101, 102]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.bar(range(len(data)), data)
ax1.set_ylim(95, 105)  # BAD: Doesn't include zero
ax1.set_title('Misleading: Exaggerated Differences')

ax2.bar(range(len(data)), data)
ax2.set_ylim(0, max(data) * 1.1)  # GOOD: Starts at zero
ax2.set_title('Honest: Shows True Scale')

plt.tight_layout()
plt.show()
\`\`\`

**Mistake 3: Too many colors**
\`\`\`python
# BAD: Rainbow colors with no meaning
# Good: Purposeful color scheme

# Use color to highlight important data
values = [23, 45, 67, 89, 34, 56, 78, 90, 12]
highlight_idx = 3  # 4th bar is important

colors = ['#CCCCCC',] * len(values)
colors[highlight_idx] = '#FF6B6B'  # Highlight one

plt.figure(figsize=(10, 6))
plt.bar(range(len(values)), values, color=colors)
plt.title('Using Color to Highlight')
plt.ylabel('Value')
plt.show()
\`\`\`

**Mistake 4: Unclear labels**
\`\`\`python
# BAD
plt.figure(figsize=(10, 6))
plt.plot([1, 2, 3, 4], [10, 20, 15, 25])
plt.title('Data')  # What data?
plt.show()

# GOOD
plt.figure(figsize=(10, 6))
plt.plot([1, 2, 3, 4], [10, 20, 15, 25], marker='o', linewidth=2)
plt.title('Quarterly Revenue Growth 2024', fontsize=14, fontweight='bold')
plt.xlabel('Quarter', fontsize=12)
plt.ylabel('Revenue ($ millions)', fontsize=12)
plt.xticks([1, 2, 3, 4], ['Q1', 'Q2', 'Q3', 'Q4',])
plt.grid(True, alpha=0.3)
plt.show()
\`\`\`

**4. Design for Your Audience**

**Technical audience:**
- Can handle complexity
- Include statistical details
- Show uncertainty

\`\`\`python
# Technical: Show confidence intervals
from scipy import stats

x = np.linspace(0, 10, 50)
y = 2 * x + np.random.randn(50) * 2

plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.6, label='Data')

# Regression line
slope, intercept = np.polyfit(x, y, 1)
plt.plot(x, slope * x + intercept, 'r-', linewidth=2, label='Fit')

# Confidence interval
predict_y = slope * x + intercept
ci = 1.96 * np.std(y - predict_y)
plt.fill_between(x, predict_y - ci, predict_y + ci, alpha=0.2, label='95% CI')

plt.xlabel('X Variable')
plt.ylabel('Y Variable')
plt.title('Linear Regression with Confidence Interval')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
\`\`\`

**Executive audience:**
- Keep it simple
- Focus on key takeaways
- Use annotations

\`\`\`python
# Executive: Clear story
monthly_revenue = pd.Series([10, 12, 11, 15, 18, 20, 19, 22, 25, 28, 30, 35])

plt.figure(figsize=(12, 6))
plt.plot(range(1, 13), monthly_revenue, marker='o', linewidth=3, markersize=8, color='#4ECDC4')

# Highlight key points
plt.annotate('Growth acceleration', 
             xy=(4, 15), xytext=(6, 12),
             arrowprops=dict(arrowstyle='->', color='red', lw=2),
             fontsize=12, color='red', fontweight='bold')

plt.axhline(monthly_revenue.mean(), color='gray', linestyle='--', 
            label=f'Average: \${monthly_revenue.mean(): .0f}M', linewidth=2)

plt.xlabel('Month', fontsize = 12)
plt.ylabel('Revenue ($ millions)', fontsize = 12)
plt.title('Revenue Growth - 200% Increase YoY', fontsize = 16, fontweight = 'bold')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',])
plt.legend(fontsize = 12)
plt.grid(True, alpha = 0.3)
plt.tight_layout()
plt.show()
            \`\`\`

**5. Accessibility**

- Colorblind-friendly palettes
- High contrast
- Clear labels

\`\`\`python
# Colorblind-friendly colors
colorblind_safe = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#CA9161',]

plt.figure(figsize=(10, 6))
for i, color in enumerate(colorblind_safe):
    plt.plot(np.random.randn(100).cumsum(), color=color, linewidth=2, label=f'Series {i+1}')
plt.title('Colorblind-Friendly Visualization')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
\`\`\`

**Best Practices Checklist:**

✅ Clear title explaining what the plot shows  
✅ Labeled axes with units  
✅ Appropriate chart type for data  
✅ Starts at zero for bar charts  
✅ Legend when multiple series  
✅ Grid for easier reading (subtle)  
✅ Consistent color scheme  
✅ Not cluttered (remove non-essential elements)  
✅ High contrast colors  
✅ Large enough font sizes  
✅ Source attribution when needed  

**Key Takeaway:**

Good visualization:
- Has a clear purpose
- Uses appropriate chart type
- Is honest (doesn't mislead)
- Is accessible (clear labels, good colors)
- Tells a story

Bad visualization:
- Misleads through scale manipulation
- Uses wrong chart type
- Is cluttered and confusing
- Relies on 3D when 2D is better
- Has poor color choices

Always ask: "What story am I telling?" and "Is this the clearest way to tell it?"`,
    keyPoints: [
      'Good visualizations have clear purpose, appropriate chart type, and minimize cognitive load',
      'Avoid 3D charts, dual y-axes, pie charts with many slices, and chartjunk',
      'Choose chart type based on data: bar for categories, line for time series, scatter for correlation',
      'Use consistent colors, direct labeling, and descriptive titles',
      'Common mistakes: truncated y-axis, too many colors, missing context',
    ],
  },
  {
    id: 'data-visualization-dq-2',
    question:
      'Explain how to create publication-quality figures in Python. Discuss figure sizing, resolution, fonts, colors, and export formats suitable for papers, presentations, and web.',
    sampleAnswer: `Creating publication-quality figures requires attention to technical details beyond basic plotting. Different venues (academic papers, presentations, web) have different requirements.

**1. Figure Sizing and Resolution**

**DPI (Dots Per Inch):**
- Screen: 72-96 DPI
- Print: 300-600 DPI minimum
- Journals: Often require 600 DPI for line art, 300 DPI for photos

\`\`\`python
# Configure for publication
import matplotlib.pyplot as plt
import matplotlib as mpl

# High-resolution figure
plt.figure(figsize=(8, 6), dpi=300)  # 2400x1800 pixels at 300 DPI
plt.plot([1, 2, 3], [1, 4, 9])
plt.title('High Resolution Plot')
plt.savefig('publication_fig.png', dpi=300, bbox_inches='tight')
plt.show()

# Vector graphics (infinitely scalable)
plt.figure(figsize=(8, 6))
plt.plot([1, 2, 3], [1, 4, 9])
plt.title('Vector Graphics')
plt.savefig('publication_fig.pdf', bbox_inches='tight')  # PDF is vector
plt.savefig('publication_fig.eps', bbox_inches='tight')  # EPS also vector
plt.show()
\`\`\`

**Aspect Ratios for Different Venues:**

\`\`\`python
# Academic paper (typically 3.5" or 7" width)
single_column = (3.5, 2.625)  # 4:3 ratio
double_column = (7, 5.25)     # 4:3 ratio

# Presentation (16:9 widescreen)
presentation = (10, 5.625)

# Poster
poster = (24, 18)

# Example: Single column figure
plt.figure(figsize=single_column, dpi=300)
x = np.linspace(0, 10, 100)
plt.plot(x, np.sin(x), linewidth=1.5)
plt.xlabel('X', fontsize=9)
plt.ylabel('sin(X)', fontsize=9)
plt.title('Single Column Figure', fontsize=10)
plt.tick_params(labelsize=8)
plt.tight_layout()
plt.savefig('single_column.pdf', dpi=300, bbox_inches='tight')
plt.show()
\`\`\`

**2. Fonts and Typography**

\`\`\`python
# Use LaTeX fonts for consistency with paper
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman',],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12
})

# Or use LaTeX directly (requires LaTeX installation)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman",],
})

fig, ax = plt.subplots(figsize=(6, 4))
x = np.linspace(0, 2*np.pi, 100)
ax.plot(x, np.sin(x), label=r'$y = \\sin(x)$')
ax.plot(x, np.cos(x), label=r'$y = \\cos(x)$')
ax.set_xlabel(r'$x$ (radians)')
ax.set_ylabel(r'$y$')
ax.set_title(r'Trigonometric Functions')
ax.legend()
plt.tight_layout()
plt.savefig('latex_fonts.pdf', dpi=300, bbox_inches='tight')
plt.show()
\`\`\`

**3. Color Schemes**

\`\`\`python
# Professional color schemes

# Grayscale-friendly (for black & white printing)
grayscale_friendly = {
    'colors': ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442',],
    'linestyles': ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
}

fig, ax = plt.subplots(figsize=(8, 5))
for i, (color, linestyle) in enumerate(zip(grayscale_friendly['colors',], 
                                            grayscale_friendly['linestyles',])):
    ax.plot(np.random.randn(100).cumsum(), 
            color=color, linestyle=linestyle, linewidth=2,
            label=f'Series {i+1}')
ax.legend()
ax.set_title('Grayscale-Friendly Colors')
plt.tight_layout()
plt.savefig('grayscale_friendly.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Colorblind-friendly (Okabe-Ito palette)
okabe_ito = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', 
             '#0072B2', '#D55E00', '#CC79A7', '#000000',]

fig, ax = plt.subplots(figsize=(8, 5))
for i, color in enumerate(okabe_ito):
    ax.plot(np.random.randn(100).cumsum(), color=color, 
            linewidth=2, label=f'Series {i+1}')
ax.legend(ncol=2)
ax.set_title('Colorblind-Friendly (Okabe-Ito)')
plt.tight_layout()
plt.savefig('colorblind_friendly.pdf', dpi=300, bbox_inches='tight')
plt.show()
\`\`\`

**4. Complete Publication Template**

\`\`\`python
def create_publication_figure(figsize=(7, 5), dpi=300):
    """
    Create a figure with publication-quality settings
    """
    # Set style
    plt.style.use('seaborn-v0_8-paper')
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Configure fonts
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'axes.linewidth': 1.0,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.5,
        'patch.linewidth': 1.0,
    })
    
    return fig, ax

# Example usage
fig, ax = create_publication_figure(figsize=(7, 5))

# Plot data
x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x), label='sin(x)', linewidth=2)
ax.plot(x, np.cos(x), label='cos(x)', linewidth=2)

# Customize
ax.set_xlabel('X (units)', fontsize=11)
ax.set_ylabel('Y (units)', fontsize=11)
ax.set_title('Publication Quality Figure', fontsize=12, fontweight='bold')
ax.legend(frameon=True, fancybox=False, shadow=False)
ax.grid(True, alpha=0.3, linestyle='--')

# Tight layout
plt.tight_layout()

# Save in multiple formats
plt.savefig('figure.pdf', dpi=300, bbox_inches='tight')  # Vector
plt.savefig('figure.png', dpi=300, bbox_inches='tight')  # Raster
plt.savefig('figure.svg', bbox_inches='tight')           # Web vector
plt.show()
\`\`\`

**5. Multi-Panel Figures**

\`\`\`python
def create_multipanel_figure():
    """Create publication-quality multi-panel figure"""
    fig = plt.figure(figsize=(10, 8), dpi=300)
    
    # Define grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Panel A
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.linspace(0, 10, 100)
    ax1.plot(x, np.sin(x), linewidth=2)
    ax1.set_xlabel('X')
    ax1.set_ylabel('sin(X)')
    ax1.set_title('(A) Time Series', loc='left', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Panel B
    ax2 = fig.add_subplot(gs[0, 1])
    data = np.random.randn(1000)
    ax2.hist(data, bins=30, edgecolor='black')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('(B) Distribution', loc='left', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Panel C (spans bottom)
    ax3 = fig.add_subplot(gs[1, :])
    categories = ['A', 'B', 'C', 'D', 'E',]
    values = [23, 45, 56, 78, 32]
    ax3.bar(categories, values, edgecolor='black')
    ax3.set_xlabel('Category')
    ax3.set_ylabel('Value')
    ax3.set_title('(C) Comparison', loc='left', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.savefig('multipanel.pdf', dpi=300, bbox_inches='tight')
    plt.show()

create_multipanel_figure()
\`\`\`

**6. Format-Specific Guidelines**

**For Academic Papers:**
\`\`\`python
# Journal requirements (example: Nature)
nature_style = {
    'figure.figsize': (3.5, 2.625),  # Single column
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'lines.linewidth': 1.0,
    'axes.linewidth': 0.5,
}

plt.rcParams.update(nature_style)

# Save as TIFF (some journals require)
plt.figure()
plt.plot([1, 2, 3], [1, 4, 9])
plt.savefig('figure.tiff', dpi=600, bbox_inches='tight', 
            pil_kwargs={'compression': 'tiff_lzw'})
\`\`\`

**For Presentations:**
\`\`\`python
# Presentation style (high contrast, large fonts)
presentation_style = {
    'figure.figsize': (10, 5.625),  # 16:9
    'font.size': 18,
    'axes.labelsize': 20,
    'axes.titlesize': 24,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'lines.linewidth': 3,
    'axes.linewidth': 2,
}

plt.rcParams.update(presentation_style)

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9], linewidth=4, color='#FF6B6B')
ax.set_xlabel('X Variable', fontweight='bold')
ax.set_ylabel('Y Variable', fontweight='bold')
ax.set_title('Clear Title for Presentation', fontweight='bold')
ax.grid(True, alpha=0.3, linewidth=1.5)

plt.savefig('presentation.png', dpi=150, bbox_inches='tight')  # Lower DPI for slides
\`\`\`

**For Web:**
\`\`\`python
# Web-optimized (SVG for interactive, PNG for static)
web_style = {
    'figure.figsize': (10, 6),
    'font.size': 12,
    'savefig.format': 'svg',
    'svg.fonttype': 'none',  # Don't convert text to paths
}

plt.rcParams.update(web_style)

plt.figure()
plt.plot([1, 2, 3], [1, 4, 9])
plt.savefig('web_figure.svg', bbox_inches='tight')  # Scalable
plt.savefig('web_figure.png', dpi=96, bbox_inches='tight')  # Screen resolution
\`\`\`

**7. Export Best Practices**

\`\`\`python
# Comprehensive export function
def save_figure(fig, basename, formats=['pdf', 'png', 'svg',], **kwargs):
    """
    Save figure in multiple formats
    
    Parameters:
    -----------
    fig : matplotlib figure
    basename : str (without extension)
    formats : list of formats
    kwargs : passed to savefig
    """
    default_kwargs = {
        'bbox_inches': 'tight',
        'dpi': 300,
        'transparent': False
    }
    default_kwargs.update(kwargs)
    
    for fmt in formats:
        filename = f"{basename}.{fmt}"
        fig.savefig(filename, **default_kwargs)
        print(f"Saved: {filename}")

# Usage
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([1, 2, 3], [1, 4, 9])
save_figure(fig, 'my_figure', formats=['pdf', 'png', 'svg',])
\`\`\`

**Checklist for Publication:**

✅ **Resolution**: 300+ DPI for print, vector (PDF/SVG) preferred  
✅ **Size**: Match journal requirements (typically 3.5" or 7" width)  
✅ **Fonts**: Consistent with paper, readable size (8-12pt)  
✅ **Colors**: Colorblind-friendly, works in grayscale  
✅ **Labels**: All axes labeled with units  
✅ **Legend**: Clear, not overlapping data  
✅ **Lines**: Thick enough (1.5-2pt)  
✅ **Grid**: Subtle if used  
✅ **Format**: PDF/EPS for vector, TIFF/PNG for raster  
✅ **File size**: Reasonable (<10 MB typically)  
✅ **Permissions**: Own all images/data shown  

**Key Takeaway:**

Publication quality requires:
- Appropriate resolution and format
- Consistent, readable fonts
- Colorblind-friendly colors
- Clear labels and legends
- Venue-specific sizing
- Multiple export formats

Always check target venue's specific requirements!`,
    keyPoints: [
      'Matplotlib provides fine-grained control with pyplot or OO API',
      'Seaborn adds statistical visualizations with better defaults built on matplotlib',
      'Plotly enables interactive plots with hover, zoom, and export capabilities',
      'Pandas .plot() offers quick visualization directly from DataFrames',
      'Choose based on needs: matplotlib for customization, seaborn for statistical, plotly for interactivity',
    ],
  },
  {
    id: 'data-visualization-dq-3',
    question:
      'Compare different plot types (line, bar, scatter, box, violin, heatmap) and provide decision criteria for choosing the right visualization for your data and question.',
    sampleAnswer: `Choosing the right plot type is crucial for effectively communicating insights. Different plots are suited for different data types and questions.

**Decision Framework:**

\`\`\`
What is your data?
├─ One continuous variable
│  ├─ Distribution → Histogram, KDE
│  └─ Single values → Dot plot
│
├─ Two continuous variables
│  ├─ Relationship → Scatter plot
│  ├─ Time series → Line plot
│  └─ Density → Hex bin, 2D KDE
│
├─ One continuous + one categorical
│  ├─ Compare groups → Box plot, Violin plot
│  ├─ Show all points → Strip plot, Swarm plot
│  └─ Summary stats → Bar plot (mean/sum)
│
├─ Two categorical + one continuous
│  └─ Compare across both → Heatmap, Grouped bar
│
└─ Many variables
   ├─ Correlations → Heatmap, Pair plot
   └─ Dimensionality → PCA plot
\`\`\`

**1. LINE PLOT**

**Best for:**
- Time series data
- Trends over continuous variable
- Comparing multiple series

\`\`\`python
# Example: Stock prices over time
dates = pd.date_range('2024-01-01', periods=365, freq='D')
stocks = pd.DataFrame({
    'AAPL': 100 * (1 + np.random.randn(365) * 0.02).cumprod(),
    'GOOGL': 100 * (1 + np.random.randn(365) * 0.02).cumprod(),
}, index=dates)

plt.figure(figsize=(12, 6))
plt.plot(stocks.index, stocks['AAPL',], label='AAPL', linewidth=2)
plt.plot(stocks.index, stocks['GOOGL',], label='GOOGL', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.title('Stock Prices Over Time')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
\`\`\`

**Use when:**
✅ X-axis is ordered (time, distance, etc.)  
✅ Showing trends and patterns  
✅ Comparing multiple time series  

**Avoid when:**
❌ X-axis is categorical (use bar instead)  
❌ Too many series (becomes cluttered)  
❌ Data points need emphasis (use scatter)  

**2. BAR PLOT**

**Best for:**
- Comparing categories
- Showing discrete counts/sums
- Rankings

\`\`\`python
# Example: Sales by region
regions = ['North', 'South', 'East', 'West',]
sales = [45000, 38000, 52000, 41000]

plt.figure(figsize=(10, 6))
bars = plt.bar(regions, sales, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A',])
plt.ylabel('Sales ($)')
plt.title('Sales by Region')

# Add values on bars
for i, (region, value) in enumerate(zip(regions, sales)):
    plt.text(i, value, f'\${value:,}', ha='center', va='bottom')
plt.show()

# Horizontal for long labels
categories = ['Category A', 'Category B with Long Name', 'Category C',]
values = [23, 45, 67]

plt.figure(figsize = (10, 6))
plt.barh(categories, values)
plt.xlabel('Value')
plt.title('Horizontal Bar Plot')
plt.show()

# Grouped bar for comparisons
quarters = ['Q1', 'Q2', 'Q3', 'Q4',]
product_a = [20, 25, 30, 35]
product_b = [15, 20, 28, 32]

x = np.arange(len(quarters))
width = 0.35

fig, ax = plt.subplots(figsize = (10, 6))
ax.bar(x - width / 2, product_a, width, label = 'Product A')
ax.bar(x + width / 2, product_b, width, label = 'Product B')
ax.set_xlabel('Quarter')
ax.set_ylabel('Sales')
ax.set_title('Sales by Product and Quarter')
ax.set_xticks(x)
ax.set_xticklabels(quarters)
ax.legend()
plt.show()
\`\`\`

**Use when:**
✅ Comparing categories  
✅ Showing counts or sums  
✅ Discrete data  

**Avoid when:**
❌ Too many categories (>15)  
❌ Baseline isn't zero (misleading)  
❌ Time series (use line instead)  

**3. SCATTER PLOT**

**Best for:**
- Relationship between two continuous variables
- Identifying correlations
- Showing clusters

\`\`\`python
# Example: Height vs Weight with correlation
n = 200
height = np.random.normal(170, 10, n)
weight = height * 0.8 + np.random.normal(0, 5, n)

plt.figure(figsize=(10, 8))
plt.scatter(height, weight, alpha=0.5, s=50)
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.title(f'Height vs Weight (r = {np.corrcoef(height, weight)[0,1]:.3f})')

# Add regression line
z = np.polyfit(height, weight, 1)
p = np.poly1d(z)
plt.plot(height, p(height), "r--", linewidth=2, label='Trend')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# With categorical color
categories = np.random.choice(['A', 'B', 'C',], n)
plt.figure(figsize=(10, 8))
for category in ['A', 'B', 'C',]:
    mask = categories == category
    plt.scatter(height[mask], weight[mask], label=category, alpha=0.6, s=50)
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.title('Height vs Weight by Category')
plt.legend()
plt.show()
\`\`\`

**Use when:**
✅ Two continuous variables  
✅ Looking for correlations  
✅ Identifying outliers  
✅ Showing clusters  

**Avoid when:**
❌ One variable is categorical (use box/violin)  
❌ Too many points (>10000) - use hexbin  
❌ Points overlap heavily - use alpha or 2D histogram  

**4. BOX PLOT**

**Best for:**
- Comparing distributions across categories
- Showing median, quartiles, outliers
- Identifying outliers

\`\`\`python
# Example: Salary by department
departments = ['Engineering', 'Sales', 'Marketing', 'HR',]
salaries = [
    np.random.normal(80000, 15000, 100),
    np.random.normal(70000, 12000, 100),
    np.random.normal(65000, 10000, 100),
    np.random.normal(60000, 8000, 100)
]

plt.figure(figsize=(10, 6))
box = plt.boxplot(salaries, labels=departments, patch_artist=True)
plt.ylabel('Salary ($)')
plt.title('Salary Distribution by Department')
plt.grid(True, alpha=0.3, axis='y')

# Color boxes
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A',]
for patch, color in zip(box['boxes',], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

plt.show()
\`\`\`

**Use when:**
✅ Comparing distributions across groups  
✅ Need to show median, quartiles  
✅ Identifying outliers important  
✅ Multiple groups (works well up to ~10)  

**Avoid when:**
❌ Distribution shape matters (use violin)  
❌ Need to show all data points (use strip/swarm)  
❌ Single distribution (use histogram)  

**5. VIOLIN PLOT**

**Best for:**
- Comparing distributions (more detail than box plot)
- Showing distribution shape (bimodal, skewed)
- Statistical comparisons

\`\`\`python
# Example: Test scores by class
classes = ['Class A', 'Class B', 'Class C',]
scores = [
    np.concatenate([np.random.normal(70, 10, 50), np.random.normal(90, 5, 50)]),  # Bimodal
    np.random.normal(80, 12, 100),  # Normal
    np.random.beta(5, 2, 100) * 50 + 50  # Skewed
]

df = pd.DataFrame({
    'Class': np.repeat(classes, 100),
    'Score': np.concatenate(scores)
})

plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='Class', y='Score')
plt.title('Test Score Distributions by Class')
plt.ylabel('Score')
plt.show()

# With inner box plot
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='Class', y='Score', inner='box')
plt.title('Violin Plot with Inner Box Plot')
plt.show()
\`\`\`

**Use when:**
✅ Distribution shape is important  
✅ Comparing groups (like box plot but richer)  
✅ Bimodal or complex distributions  

**Avoid when:**
❌ Small sample sizes (<20 per group)  
❌ Simple comparison (box plot simpler)  
❌ Too many groups (>6 gets cluttered)  

**6. HISTOGRAM**

**Best for:**
- Distribution of single continuous variable
- Identifying shape (normal, skewed, bimodal)
- Frequency analysis

\`\`\`python
# Example: Age distribution
ages = np.concatenate([
    np.random.normal(30, 5, 300),
    np.random.normal(55, 8, 200)
])

plt.figure(figsize=(10, 6))
plt.hist(ages, bins=30, edgecolor='black', alpha=0.7)
plt.axvline(ages.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {ages.mean():.1f}')
plt.axvline(np.median(ages), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(ages):.1f}')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')
plt.legend()
plt.show()
\`\`\`

**Use when:**
✅ Single continuous variable  
✅ Want to see distribution shape  
✅ Identifying patterns (normal, skewed, multimodal)  

**Avoid when:**
❌ Comparing many groups (use box/violin)  
❌ Need smooth density (use KDE)  
❌ Categorical data (use bar plot)  

**7. HEATMAP**

**Best for:**
- Correlation matrices
- Two categorical + one continuous
- Grid data

\`\`\`python
# Example: Correlation matrix
df = pd.DataFrame(np.random.randn(100, 5), columns=['A', 'B', 'C', 'D', 'E',])
df['B',] = df['A',] * 0.8 + df['B',] * 0.4
df['C',] = df['A',] * -0.6 + df['C',] * 0.5
corr = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()

# Example: Categorical grid
pivot_data = pd.DataFrame({
    'Month': ['Jan', 'Feb', 'Mar',] * 4,
    'Region': np.repeat(['North', 'South', 'East', 'West',], 3),
    'Sales': np.random.randint(1000, 5000, 12)
})
pivot = pivot_data.pivot('Region', 'Month', 'Sales')

plt.figure(figsize=(8, 6))
sns.heatmap(pivot, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Sales by Region and Month')
plt.show()
\`\`\`

**Use when:**
✅ Matrix/grid data  
✅ Correlation analysis  
✅ Two categorical variables + one continuous  

**Avoid when:**
❌ More than ~20x20 cells (unreadable)  
❌ Need precise value reading (use table)  
❌ Relationships between variables not grid-like  

**Decision Matrix:**

| Data Type | Question | Plot Type |
|-----------|----------|-----------|
| One continuous | Distribution? | Histogram, KDE |
| Two continuous | Correlation? | Scatter |
| Continuous over time | Trend? | Line |
| Continuous by category | Compare groups? | Box, Violin |
| Categories | Compare values? | Bar |
| Many variables | Correlations? | Heatmap |
| Two categorical + continuous | Compare across both? | Heatmap, Grouped bar |

**Key Takeaway:**

Choose plot type based on:
1. **Data types** (continuous vs categorical)
2. **Question** (distribution? correlation? comparison?)
3. **Audience** (technical vs general)
4. **Number of variables** (1, 2, or many)

When in doubt, try multiple types and see which tells the clearest story!`,
    keyPoints: [
      'Subplots share common scales and can be compared directly side-by-side',
      'Use fig, axes = plt.subplots() for grid creation and axes[i] for access',
      'sharex/sharey parameters synchronize axes across subplots',
      'Iterate through axes with axes.flat for efficient subplot population',
      'Adjust spacing with tight_layout() or subplots_adjust() to prevent overlap',
    ],
  },
];
