/**
 * Multivariate Analysis Section
 */

export const multivariateanalysisSection = {
  id: 'multivariate-analysis',
  title: 'Multivariate Analysis',
  content: `# Multivariate Analysis

## Introduction

Multivariate analysis examines relationships among three or more variables simultaneously. While univariate analyzes one variable and bivariate analyzes pairs, multivariate reveals complex interactions, patterns, and structures that emerge only when considering multiple variables together.

**Why Multivariate Analysis Matters**:
- **Feature Interactions**: Discover how features work together
- **Multicollinearity Detection**: Identify redundant features
- **Dimensionality Reduction**: Visualize high-dimensional data
- **Pattern Discovery**: Find hidden structures in data
- **Model Insight**: Understand complex relationships before modeling

## Correlation Matrix and Heatmaps

### Complete Correlation Analysis

\`\`\`python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

# Load data
housing = fetch_california_housing (as_frame=True)
df = housing.frame

print("=" * 70)
print("CORRELATION MATRIX ANALYSIS")
print("=" * 70)

# Calculate correlation matrix
corr_matrix = df.corr()

print("\\nCorrelation Matrix:\\n")
print(corr_matrix.round(3))

# Visualize with heatmap
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Standard heatmap
sns.heatmap (corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, ax=axes[0],
            cbar_kws={'label': 'Correlation'})
axes[0].set_title('Correlation Matrix Heatmap')

# Clustered heatmap (groups similar variables)
sns.clustermap (corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
               center=0, figsize=(10, 10))
plt.suptitle('Hierarchically Clustered Correlation Matrix', y=1.02)

plt.tight_layout()
plt.show()
\`\`\`

### Identifying Multicollinearity

\`\`\`python
def detect_multicollinearity (df, threshold=0.8):
    """Detect highly correlated feature pairs"""
    
    print(f"\\nMULTICOLLINEARITY DETECTION (threshold={threshold})")
    print("=" * 70)
    
    corr_matrix = df.corr().abs()
    
    # Find pairs with high correlation
    high_corr_pairs = []
    for i in range (len (corr_matrix.columns)):
        for j in range (i+1, len (corr_matrix.columns)):
            if corr_matrix.iloc[i, j] >= threshold:
                high_corr_pairs.append({
                    'feature_1': corr_matrix.columns[i],
                    'feature_2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
    
    if high_corr_pairs:
        print(f"\\nFound {len (high_corr_pairs)} highly correlated pairs:\\n")
        for pair in sorted (high_corr_pairs, key=lambda x: x['correlation'], reverse=True):
            print(f"  {pair['feature_1']} <-> {pair['feature_2']}: {pair['correlation']:.3f}")
        
        print(f"\\n⚠️  Consider removing one feature from each pair to reduce multicollinearity")
    else:
        print(f"\\n✓ No feature pairs with correlation >= {threshold}")
    
    return high_corr_pairs

# Detect multicollinearity
multicoll = detect_multicollinearity (df, threshold=0.7)
\`\`\`

## Pair Plots for Multiple Variables

\`\`\`python
# Select subset of features for pair plot
features_subset = ['MedInc', 'HouseAge', 'AveRooms', 'MedHouseVal']
df_subset = df[features_subset].sample(1000, random_state=42)

# Create pair plot
sns.pairplot (df_subset, diag_kind='kde', plot_kws={'alpha': 0.6})
plt.suptitle('Pair Plot: Feature Relationships', y=1.02)
plt.show()

# With target categorization for color coding
df['PriceCategory'] = pd.cut (df['MedHouseVal'], 
                               bins=[0, 2, 3.5, 10], 
                               labels=['Low', 'Medium', 'High'])
df_subset_cat = df[features_subset + ['PriceCategory']].sample(1000, random_state=42)

sns.pairplot (df_subset_cat, hue='PriceCategory', diag_kind='kde', 
             plot_kws={'alpha': 0.6}, palette='viridis')
plt.suptitle('Pair Plot by Price Category', y=1.02)
plt.show()
\`\`\`

## Principal Component Analysis (PCA) for Visualization

\`\`\`python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def perform_pca_analysis (df, n_components=None):
    """Perform PCA and visualize results"""
    
    print("\\n" + "=" * 70)
    print("PRINCIPAL COMPONENT ANALYSIS")
    print("=" * 70)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform (df.select_dtypes (include=[np.number]))
    
    # Perform PCA
    if n_components is None:
        n_components = min(X_scaled.shape[1], 10)
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Explained variance
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum (explained_var)
    
    print(f"\\nExplained Variance by Component:\\n")
    for i, (ev, cv) in enumerate (zip (explained_var, cumulative_var), 1):
        print(f"  PC{i}: {ev:.4f} ({ev*100:.2f}%) | Cumulative: {cv:.4f} ({cv*100:.2f}%)")
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Scree plot
    axes[0, 0].bar (range(1, len (explained_var)+1), explained_var)
    axes[0, 0].plot (range(1, len (explained_var)+1), explained_var, 'ro-')
    axes[0, 0].set_xlabel('Principal Component')
    axes[0, 0].set_ylabel('Explained Variance Ratio')
    axes[0, 0].set_title('Scree Plot')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Cumulative explained variance
    axes[0, 1].plot (range(1, len (cumulative_var)+1), cumulative_var, 'bo-')
    axes[0, 1].axhline (y=0.8, color='r', linestyle='--', label='80% threshold')
    axes[0, 1].axhline (y=0.9, color='g', linestyle='--', label='90% threshold')
    axes[0, 1].set_xlabel('Number of Components')
    axes[0, 1].set_ylabel('Cumulative Explained Variance')
    axes[0, 1].set_title('Cumulative Explained Variance')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 2D projection (PC1 vs PC2)
    axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
    axes[1, 0].set_xlabel (f'PC1 ({explained_var[0]*100:.1f}%)')
    axes[1, 0].set_ylabel (f'PC2 ({explained_var[1]*100:.1f}%)')
    axes[1, 0].set_title('2D PCA Projection')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Feature contributions to PC1 and PC2
    feature_names = df.select_dtypes (include=[np.number]).columns
    loadings = pca.components_[:2].T
    axes[1, 1].scatter (loadings[:, 0], loadings[:, 1])
    for i, feature in enumerate (feature_names):
        axes[1, 1].annotate (feature, (loadings[i, 0], loadings[i, 1]))
    axes[1, 1].set_xlabel('PC1 Loading')
    axes[1, 1].set_ylabel('PC2 Loading')
    axes[1, 1].set_title('Feature Contributions to PC1 and PC2')
    axes[1, 1].axhline (y=0, color='k', linestyle='-', linewidth=0.5)
    axes[1, 1].axvline (x=0, color='k', linestyle='-', linewidth=0.5)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return pca, X_pca

# Perform PCA
pca_model, X_pca = perform_pca_analysis (df.select_dtypes (include=[np.number]))
\`\`\`

## Key Takeaways

1. **Correlation matrices reveal pairwise relationships at a glance**2. **Heatmaps make correlation patterns visually obvious**3. **Multicollinearity (high feature correlation) can hurt linear models**4. **Pair plots visualize all pairwise relationships simultaneously**5. **PCA reduces dimensionality while preserving variance**6. **First few PCs typically capture 80-90% of variance**7. **PCA useful for visualization of high-dimensional data in 2D/3D**8. **Feature loadings show which original features contribute to each PC**9. **Always standardize features before PCA**10. **Multivariate patterns inform feature engineering and selection**

## Connection to Machine Learning

- **Multicollinearity** degrades linear model coefficients and interpretability
- **PCA** can be used as feature extraction technique
- **Correlation analysis** guides feature selection
- **Pair plots** reveal interaction terms to engineer
- **Dimensionality reduction** helps with visualization and some models
- **Highly correlated features** may be redundant - consider removing
- **PCA components** can replace original features in some cases

Multivariate analysis completes the EDA picture by revealing complex patterns among multiple variables.
`,
};
