/**
 * Advanced Visualization Techniques Section
 */

export const advancedvisualizationSection = {
  id: 'advanced-visualization',
  title: 'Advanced Visualization Techniques',
  content: `# Advanced Visualization Techniques

## Introduction

Effective data visualization transforms complex patterns into intuitive insights. While basic plots (histograms, scatter plots) are essential, advanced visualizations reveal multidimensional patterns, geographical relationships, network structures, and temporal dynamics that simpler plots miss.

**Why Advanced Visualization Matters**:
- **Communicate Insights**: Stakeholders understand visuals better than numbers
- **Pattern Discovery**: Reveal hidden structures in complex data
- **Multidimensional Understanding**: Visualize more than 2-3 variables simultaneously
- **Interactive Exploration**: Dynamic plots enable deeper investigation
- **Publication Quality**: Professional visualizations for reports and papers

## Interactive Visualizations with Plotly

### Creating Interactive Plots

\`\`\`python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load data
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing (as_frame=True)
df = housing.frame

print("=" * 70)
print("INTERACTIVE VISUALIZATIONS WITH PLOTLY")
print("=" * 70)

# 1. Interactive Scatter Plot with hover information
fig = px.scatter(
    df.sample(2000, random_state=42),
    x='MedInc',
    y='MedHouseVal',
    color='AveRooms',
    size='Population',
    hover_data=['HouseAge', 'AveBedrms'],
    title='California Housing: Interactive Scatter Plot',
    labels={
        'MedInc': 'Median Income',
        'MedHouseVal': 'Median House Value',
        'AveRooms': 'Average Rooms'
    },
    color_continuous_scale='Viridis'
)

fig.update_layout(
    width=900,
    height=600,
    hovermode='closest'
)

# Save as HTML (can be opened in browser)
fig.write_html('interactive_scatter.html')
fig.show()

print("\\n✓ Interactive scatter plot created")
print("  - Hover for details")
print("  - Zoom and pan")
print("  - Click legend to toggle series")

# 2. Interactive 3D Scatter Plot
fig_3d = px.scatter_3d(
    df.sample(1000, random_state=42),
    x='Latitude',
    y='Longitude',
    z='MedHouseVal',
    color='MedInc',
    title='3D Geographic View of House Values',
    labels={
        'Latitude': 'Latitude',
        'Longitude': 'Longitude',
        'MedHouseVal': 'Median House Value',
        'MedInc': 'Median Income'
    }
)

fig_3d.update_layout (scene=dict(
    xaxis_title='Latitude',
    yaxis_title='Longitude',
    zaxis_title='House Value'
))

fig_3d.show()

# 3. Interactive Time Series (using synthetic time data)
dates = pd.date_range('2020-01-01', periods=len (df), freq='D')
df_time = df.copy()
df_time['Date'] = dates
df_time_agg = df_time.groupby (pd.Grouper (key='Date', freq='M'))['MedHouseVal'].mean().reset_index()

fig_time = px.line(
    df_time_agg,
    x='Date',
    y='MedHouseVal',
    title='House Values Over Time (Interactive)',
    labels={'MedHouseVal': 'Average Median House Value'}
)

fig_time.update_traces (mode='lines+markers')
fig_time.update_layout (hovermode='x unified')
fig_time.show()

print("\\n✓ Multiple interactive visualizations created")
\`\`\`

### Dashboard-Style Layouts

\`\`\`python
def create_interactive_dashboard (df):
    """Create a comprehensive interactive dashboard"""
    
    # Sample for performance
    df_sample = df.sample (min(2000, len (df)), random_state=42)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Distribution of House Values', 
                       'Income vs House Value',
                       'Geographic Distribution',
                       'Feature Correlations'),
        specs=[[{'type': 'histogram'}, {'type': 'scatter'}],
               [{'type': 'scatter'}, {'type': 'heatmap'}]]
    )
    
    # 1. Histogram
    fig.add_trace(
        go.Histogram (x=df_sample['MedHouseVal'], name='House Values',
                     marker_color='lightblue'),
        row=1, col=1
    )
    
    # 2. Scatter plot
    fig.add_trace(
        go.Scatter (x=df_sample['MedInc'], y=df_sample['MedHouseVal'],
                   mode='markers', name='Income vs Value',
                   marker=dict (size=5, opacity=0.6)),
        row=1, col=2
    )
    
    # 3. Geographic scatter
    fig.add_trace(
        go.Scatter (x=df_sample['Longitude'], y=df_sample['Latitude'],
                   mode='markers', name='Locations',
                   marker=dict (size=5, color=df_sample['MedHouseVal'],
                             colorscale='Viridis', showscale=True)),
        row=2, col=1
    )
    
    # 4. Correlation heatmap
    corr = df.corr()
    fig.add_trace(
        go.Heatmap (z=corr.values, x=corr.columns, y=corr.columns,
                   colorscale='RdBu', zmid=0),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="California Housing Dashboard",
        showlegend=False,
        height=800,
        width=1200
    )
    
    fig.show()
    
    return fig

# Create dashboard
dashboard = create_interactive_dashboard (df)
print("\\n✓ Interactive dashboard created")
\`\`\`

## Geographical Visualizations

### Mapping Data with Geospatial Context

\`\`\`python
def create_geographic_heatmap (df):
    """Create geographic heatmap showing house values by location"""
    
    print("\\nGEOGRAPHIC VISUALIZATION")
    print("=" * 70)
    
    # Create figure with multiple representations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Scatter plot colored by house value
    scatter = axes[0, 0].scatter(
        df['Longitude'], df['Latitude'],
        c=df['MedHouseVal'], s=10, alpha=0.6,
        cmap='YlOrRd', vmin=0, vmax=5
    )
    axes[0, 0].set_xlabel('Longitude')
    axes[0, 0].set_ylabel('Latitude')
    axes[0, 0].set_title('House Values by Location')
    plt.colorbar (scatter, ax=axes[0, 0], label='Median House Value ($100k)')
    
    # 2. Hexbin plot (aggregated heatmap)
    hexbin = axes[0, 1].hexbin(
        df['Longitude'], df['Latitude'],
        C=df['MedHouseVal'], gridsize=30,
        cmap='YlOrRd', reduce_C_function=np.mean
    )
    axes[0, 1].set_xlabel('Longitude')
    axes[0, 1].set_ylabel('Latitude')
    axes[0, 1].set_title('Average House Values (Hexbin)')
    plt.colorbar (hexbin, ax=axes[0, 1], label='Avg House Value')
    
    # 3. 2D Histogram
    hist = axes[1, 0].hist2d(
        df['Longitude'], df['Latitude'],
        bins=50, cmap='YlOrRd', weights=df['MedHouseVal']
    )
    axes[1, 0].set_xlabel('Longitude')
    axes[1, 0].set_ylabel('Latitude')
    axes[1, 0].set_title('Density-Weighted House Values')
    plt.colorbar (hist[3], ax=axes[1, 0], label='Total Value')
    
    # 4. Contour plot
    # Create grid
    from scipy.interpolate import griddata
    xi = np.linspace (df['Longitude'].min(), df['Longitude'].max(), 100)
    yi = np.linspace (df['Latitude'].min(), df['Latitude'].max(), 100)
    xi, yi = np.meshgrid (xi, yi)
    
    # Interpolate values
    zi = griddata(
        (df['Longitude'], df['Latitude']),
        df['MedHouseVal'],
        (xi, yi),
        method='cubic'
    )
    
    contour = axes[1, 1].contourf (xi, yi, zi, levels=15, cmap='YlOrRd')
    axes[1, 1].set_xlabel('Longitude')
    axes[1, 1].set_ylabel('Latitude')
    axes[1, 1].set_title('Interpolated House Value Contours')
    plt.colorbar (contour, ax=axes[1, 1], label='House Value')
    
    plt.tight_layout()
    plt.show()
    
    print("\\n✓ Geographic visualizations created")
    print("  Insight: Coastal areas (low longitude) show higher values")

create_geographic_heatmap (df)
\`\`\`

### Interactive Maps with Plotly

\`\`\`python
def create_interactive_map (df):
    """Create interactive choropleth-style map"""
    
    # Sample for performance
    df_sample = df.sample(5000, random_state=42)
    
    # Create density mapbox
    fig = px.density_mapbox(
        df_sample,
        lat='Latitude',
        lon='Longitude',
        z='MedHouseVal',
        radius=10,
        center=dict (lat=37, lon=-119),
        zoom=5,
        mapbox_style="open-street-map",
        title='Interactive California Housing Density Map',
        color_continuous_scale='Viridis',
        labels={'MedHouseVal': 'House Value'}
    )
    
    fig.update_layout (height=600, width=900)
    fig.show()
    
    print("\\n✓ Interactive map created (zoom, pan, hover)")

create_interactive_map (df)
\`\`\`

## Network Graphs

### Visualizing Relationships as Networks

\`\`\`python
def create_correlation_network (df, threshold=0.5):
    """Visualize correlations as a network graph"""
    
    import networkx as nx
    
    print("\\nCORRELATION NETWORK")
    print("=" * 70)
    
    # Calculate correlations
    corr = df.corr()
    
    # Create graph
    G = nx.Graph()
    
    # Add edges for correlations above threshold
    for i in range (len (corr.columns)):
        for j in range (i+1, len (corr.columns)):
            if abs (corr.iloc[i, j]) >= threshold:
                G.add_edge(
                    corr.columns[i],
                    corr.columns[j],
                    weight=abs (corr.iloc[i, j]),
                    sign=1 if corr.iloc[i, j] > 0 else -1
                )
    
    # Create visualization
    plt.figure (figsize=(12, 8))
    
    # Layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, node_size=3000, node_color='lightblue',
        edgecolors='black', linewidths=2
    )
    
    # Draw edges (color by sign, width by strength)
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    signs = [G[u][v]['sign'] for u, v in edges]
    colors = ['green' if s > 0 else 'red' for s in signs]
    widths = [w * 5 for w in weights]
    
    nx.draw_networkx_edges(
        G, pos, width=widths, edge_color=colors, alpha=0.6
    )
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    plt.title (f'Feature Correlation Network (|r| >= {threshold})', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    print(f"\\nNetwork Stats:")
    print(f"  Nodes (features): {G.number_of_nodes()}")
    print(f"  Edges (correlations): {G.number_of_edges()}")
    print(f"  Avg degree: {np.mean([d for n, d in G.degree()]):.2f}")
    
    # Most connected features
    degrees = dict(G.degree())
    most_connected = sorted (degrees.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"\\n  Most connected features:")
    for feature, degree in most_connected:
        print(f"    {feature}: {degree} connections")

create_correlation_network (df, threshold=0.3)
\`\`\`

## Animated Visualizations

### Time-Based Animations

\`\`\`python
def create_animated_visualization (df):
    """Create animated visualization showing changes over time"""
    
    # Create time-based grouping (synthetic temporal data)
    df_anim = df.copy()
    df_anim['Year'] = np.random.choice (range(2015, 2024), len (df))
    
    # Aggregate by year
    df_yearly = df_anim.groupby('Year').agg({
        'MedHouseVal': 'mean',
        'MedInc': 'mean',
        'HouseAge': 'mean',
        'Population': 'sum'
    }).reset_index()
    
    # Create animated scatter plot
    fig = px.scatter(
        df_anim.sample(2000, random_state=42),
        x='MedInc',
        y='MedHouseVal',
        animation_frame='Year',
        animation_group='Year',
        size='Population',
        color='HouseAge',
        hover_name='Year',
        size_max=50,
        range_x=[0, 15],
        range_y=[0, 5],
        title='House Values Over Time (Animated)',
        labels={
            'MedInc': 'Median Income',
            'MedHouseVal': 'Median House Value',
            'HouseAge': 'House Age'
        }
    )
    
    fig.update_layout (width=900, height=600)
    fig.show()
    
    print("\\n✓ Animated visualization created")
    print("  Click play button to see animation")

# Note: Requires plotly, uncomment to run
# create_animated_visualization (df)
\`\`\`

## Advanced Seaborn Techniques

### Complex Multi-Panel Visualizations

\`\`\`python
def create_comprehensive_seaborn_analysis (df):
    """Create publication-quality multi-panel analysis"""
    
    print("\\nCOMPREHENSIVE SEABORN ANALYSIS")
    print("=" * 70)
    
    # Set style
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    # Create figure with custom layout
    fig = plt.figure (figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Joint plot embedded
    ax1 = fig.add_subplot (gs[0:2, 0:2])
    df_sample = df.sample(2000, random_state=42)
    ax1.scatter (df_sample['MedInc'], df_sample['MedHouseVal'], 
                alpha=0.5, s=20)
    ax1.set_xlabel('Median Income')
    ax1.set_ylabel('Median House Value')
    ax1.set_title('Income vs House Value')
    
    # Add marginal distributions
    ax_top = fig.add_subplot (gs[0, 0:2])
    ax_top.hist (df_sample['MedInc'], bins=30, alpha=0.7, edgecolor='black')
    ax_top.set_xticks([])
    ax_top.set_ylabel('Frequency')
    
    ax_right = fig.add_subplot (gs[1, 2])
    ax_right.hist (df_sample['MedHouseVal'], bins=30, alpha=0.7, 
                  orientation='horizontal', edgecolor='black')
    ax_right.set_yticks([])
    ax_right.set_xlabel('Frequency')
    
    # 2. Violin plots
    ax2 = fig.add_subplot (gs[0, 2])
    df['IncomeCategory'] = pd.cut (df['MedInc'], bins=[0, 3, 6, 15],
                                   labels=['Low', 'Medium', 'High'])
    sns.violinplot (data=df.sample(1000), x='IncomeCategory', 
                   y='MedHouseVal', ax=ax2)
    ax2.set_title('House Value by Income Category')
    ax2.set_xlabel('Income Category')
    ax2.set_ylabel('House Value')
    
    # 3. Ridge plot style (multiple distributions)
    ax3 = fig.add_subplot (gs[2, :])
    categories = df['IncomeCategory'].unique()
    for i, cat in enumerate (categories):
        data = df[df['IncomeCategory'] == cat]['MedHouseVal']
        ax3.fill_between (np.linspace(0, 5, 100),
                        i, i + np.histogram (data, bins=100, range=(0, 5),
                                          density=True)[0],
                        alpha=0.6, label=cat)
    ax3.set_xlabel('Median House Value')
    ax3.set_ylabel('Income Category (stacked densities)')
    ax3.set_title('Distribution of House Values by Income Category')
    ax3.legend()
    
    plt.suptitle('Comprehensive California Housing Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.show()
    
    print("\\n✓ Comprehensive visualization created")

create_comprehensive_seaborn_analysis (df)
\`\`\`

## Key Takeaways

1. **Interactive plots enable deeper data exploration**
2. **Plotly provides browser-based interactivity (zoom, pan, hover)**
3. **Geographic visualizations reveal spatial patterns**
4. **Network graphs show relationship structures**
5. **Animations display temporal changes effectively**
6. **Multi-panel layouts tell complete stories**
7. **Color scales should be meaningful (diverging for correlations, sequential for values)**
8. **Always consider your audience (technical vs non-technical)**
9. **Publication-quality visualizations require attention to detail**
10. **Interactive dashboards enable stakeholder self-service exploration**

## Connection to Machine Learning

- **Interactive plots** help identify non-linear patterns requiring feature engineering
- **Geographic visualizations** reveal spatial features to create (distance to center, clustering)
- **Network graphs** identify feature groups for dimensionality reduction
- **Animations** show temporal patterns for time series features
- **Clear visualizations** communicate model insights to stakeholders
- **Dashboard-style layouts** enable monitoring of deployed model performance
- **Advanced visualizations** essential for explaining model predictions (SHAP, LIME)

Mastering visualization is as important as mastering models - it's how you communicate insights and drive decisions.
`,
};
