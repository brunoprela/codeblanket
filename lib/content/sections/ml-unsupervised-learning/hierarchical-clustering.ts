/**
 * Section: Hierarchical Clustering
 * Module: Classical Machine Learning - Unsupervised Learning
 *
 * Comprehensive coverage of hierarchical clustering algorithms, dendrograms, and applications
 */

export const hierarchicalClustering = {
  id: 'hierarchical-clustering',
  title: 'Hierarchical Clustering',
  content: `
# Hierarchical Clustering

## Introduction

Hierarchical clustering is a powerful unsupervised learning technique that builds a hierarchy of clusters by iteratively merging or splitting them. Unlike K-Means, which requires specifying the number of clusters upfront, hierarchical clustering creates a tree-like structure (dendrogram) that shows relationships between clusters at all levels of granularity.

**Key Advantage**: Explore data at multiple scales without committing to a specific number of clusters

**Two Main Approaches**:
- **Agglomerative (Bottom-Up)**: Start with each point as its own cluster, merge closest pairs
- **Divisive (Top-Down)**: Start with all points in one cluster, recursively split

## Agglomerative vs Divisive

### Agglomerative Clustering (More Common)

**Algorithm**:
1. Start: Each data point is its own cluster (N clusters)
2. Find the two closest clusters
3. Merge them into one cluster
4. Repeat steps 2-3 until one cluster remains
5. Result: A hierarchy showing all merge operations

**Time Complexity**: O(n³) naive, O(n²log n) with optimizations

### Divisive Clustering (Less Common)

**Algorithm**:
1. Start: All data points in one cluster
2. Find the most heterogeneous cluster
3. Split it into two clusters
4. Repeat steps 2-3 until each point is its own cluster

**Computationally Expensive**: Less commonly used in practice

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform

# Generate sample data
np.random.seed(42)
X, y_true = make_blobs (n_samples=100, centers=4,
                       cluster_std=0.8, random_state=42)

print(f"Data shape: {X.shape}")
print(f"True number of clusters: {len (np.unique (y_true))}")

# Visualize original data
plt.figure (figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], alpha=0.6, s=50)
plt.title('Data to be Clustered')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
\`\`\`

## The Dendrogram

A dendrogram is a tree diagram showing the hierarchical relationship between clusters. The height of each merge represents the distance between the clusters being merged.

**Reading a Dendrogram**:
- **X-axis**: Data points or clusters
- **Y-axis**: Distance at which clusters merge
- **Horizontal lines**: Clusters
- **Vertical lines**: Merge operations
- **Height of merge**: Dissimilarity between clusters

\`\`\`python
# Create linkage matrix for dendrogram
# linkage performs hierarchical clustering
linkage_matrix = linkage(X, method='ward')

# Plot dendrogram
plt.figure (figsize=(14, 7))
dendrogram (linkage_matrix,
           truncate_mode='lastp',  # Show only last p merged clusters
           p=30,  # Show last 30 merges
           leaf_rotation=90,
           leaf_font_size=10,
           show_contracted=True)

plt.title('Hierarchical Clustering Dendrogram (Ward Linkage)', fontsize=14)
plt.xlabel('Sample Index or (Cluster Size)', fontsize=12)
plt.ylabel('Distance', fontsize=12)
plt.axhline (y=15, color='r', linestyle='--', linewidth=2, label='Cut at distance=15')
plt.legend()
plt.tight_layout()
plt.show()

# The dendrogram shows how clusters merge as we move up the tree
# Cutting at different heights gives different numbers of clusters
\`\`\`

## Linkage Methods

The **linkage criterion** determines how we measure distance between clusters. Different methods produce different clustering results.

### 1. Single Linkage (Minimum Linkage)

**Distance**: Minimum distance between any two points in different clusters

$$d(C_i, C_j) = \\min_{x \\in C_i, y \\in C_j} d (x, y)$$

**Characteristics**:
- Tends to create long, chain-like clusters
- Sensitive to noise and outliers
- Good for non-elliptical shapes
- Prone to "chaining" effect

\`\`\`python
# Single linkage
linkage_single = linkage(X, method='single')

hc_single = AgglomerativeClustering (n_clusters=4, linkage='single')
labels_single = hc_single.fit_predict(X)

plt.figure (figsize=(14, 5))

plt.subplot(1, 2, 1)
dendrogram (linkage_single, truncate_mode='lastp', p=30, no_labels=True)
plt.title('Single Linkage Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels_single, cmap='viridis', s=50, alpha=0.6)
plt.title('Single Linkage Clustering (K=4)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

print("Single Linkage tends to create elongated, chain-like clusters")
\`\`\`

### 2. Complete Linkage (Maximum Linkage)

**Distance**: Maximum distance between any two points in different clusters

$$d(C_i, C_j) = \\max_{x \\in C_i, y \\in C_j} d (x, y)$$

**Characteristics**:
- Tends to create compact, tight clusters
- Less sensitive to outliers than single linkage
- Prefers clusters of similar diameter
- More robust than single linkage

\`\`\`python
# Complete linkage
linkage_complete = linkage(X, method='complete')

hc_complete = AgglomerativeClustering (n_clusters=4, linkage='complete')
labels_complete = hc_complete.fit_predict(X)

plt.figure (figsize=(14, 5))

plt.subplot(1, 2, 1)
dendrogram (linkage_complete, truncate_mode='lastp', p=30, no_labels=True)
plt.title('Complete Linkage Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels_complete, cmap='viridis', s=50, alpha=0.6)
plt.title('Complete Linkage Clustering (K=4)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

print("Complete Linkage creates more compact, spherical clusters")
\`\`\`

### 3. Average Linkage (UPGMA)

**Distance**: Average distance between all pairs of points in different clusters

$$d(C_i, C_j) = \\frac{1}{|C_i| \\cdot |C_j|} \\sum_{x \\in C_i} \\sum_{y \\in C_j} d (x, y)$$

**Characteristics**:
- Compromise between single and complete
- Less extreme than single or complete
- Commonly used in practice
- Good default choice

\`\`\`python
# Average linkage
linkage_average = linkage(X, method='average')

hc_average = AgglomerativeClustering (n_clusters=4, linkage='average')
labels_average = hc_average.fit_predict(X)

plt.figure (figsize=(14, 5))

plt.subplot(1, 2, 1)
dendrogram (linkage_average, truncate_mode='lastp', p=30, no_labels=True)
plt.title('Average Linkage Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels_average, cmap='viridis', s=50, alpha=0.6)
plt.title('Average Linkage Clustering (K=4)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

print("Average Linkage balances between single and complete")
\`\`\`

### 4. Ward Linkage (Minimum Variance)

**Distance**: Increase in within-cluster variance when merging

$$d(C_i, C_j) = \\frac{|C_i| \\cdot |C_j|}{|C_i| + |C_j|} ||\\mu_i - \\mu_j||^2$$

**Characteristics**:
- Minimizes within-cluster variance (like K-Means objective)
- Creates clusters of similar size
- **Most commonly used** for general-purpose clustering
- Works only with Euclidean distance

\`\`\`python
# Ward linkage (most common)
linkage_ward = linkage(X, method='ward')

hc_ward = AgglomerativeClustering (n_clusters=4, linkage='ward')
labels_ward = hc_ward.fit_predict(X)

plt.figure (figsize=(14, 5))

plt.subplot(1, 2, 1)
dendrogram (linkage_ward, truncate_mode='lastp', p=30, no_labels=True)
plt.title('Ward Linkage Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels_ward, cmap='viridis', s=50, alpha=0.6)
plt.title('Ward Linkage Clustering (K=4)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

print("Ward Linkage minimizes variance, similar to K-Means objective")
\`\`\`

## Comparison of Linkage Methods

\`\`\`python
# Compare all linkage methods on same data
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

methods = ['single', 'complete', 'average', 'ward']
titles = ['Single Linkage', 'Complete Linkage', 'Average Linkage', 'Ward Linkage']

for idx, (method, title) in enumerate (zip (methods, titles)):
    hc = AgglomerativeClustering (n_clusters=4, linkage=method)
    labels = hc.fit_predict(X)

    axes[idx].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
    axes[idx].set_title (title, fontsize=14)
    axes[idx].set_xlabel('Feature 1')
    axes[idx].set_ylabel('Feature 2')

plt.tight_layout()
plt.show()

# Ward typically gives the best results for general-purpose clustering
\`\`\`

## Choosing the Number of Clusters

### Method 1: Visual Inspection of Dendrogram

Look for large gaps in the dendrogram - these suggest natural cutting points.

\`\`\`python
plt.figure (figsize=(14, 7))
dendrogram (linkage_ward)
plt.title('Full Dendrogram - Look for Large Vertical Gaps')
plt.xlabel('Sample Index')
plt.ylabel('Distance')

# Add horizontal lines at different cutting heights
for height, color in [(10, 'r'), (15, 'g'), (20, 'b')]:
    plt.axhline (y=height, color=color, linestyle='--',
                label=f'Cut at height={height}')

plt.legend()
plt.show()

# Cutting at height=15 gives 4 clusters (matches true number)
\`\`\`

### Method 2: Distance-Based Criterion

\`\`\`python
# Cut dendrogram at specific height
from scipy.cluster.hierarchy import fcluster

# Cut at height=15
clusters_at_15 = fcluster (linkage_ward, t=15, criterion='distance')
print(f"Clusters at height 15: {len (np.unique (clusters_at_15))}")

# Cut at height=10
clusters_at_10 = fcluster (linkage_ward, t=10, criterion='distance')
print(f"Clusters at height 10: {len (np.unique (clusters_at_10))}")

# Visualize different cuts
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(X[:, 0], X[:, 1], c=clusters_at_15, cmap='viridis', s=50, alpha=0.6)
axes[0].set_title (f'Cut at Height=15 ({len (np.unique (clusters_at_15))} clusters)')

axes[1].scatter(X[:, 0], X[:, 1], c=clusters_at_10, cmap='viridis', s=50, alpha=0.6)
axes[1].set_title (f'Cut at Height=10 ({len (np.unique (clusters_at_10))} clusters)')

plt.tight_layout()
plt.show()
\`\`\`

### Method 3: Inconsistency Method

Measures how inconsistent a merge is compared to nearby merges.

\`\`\`python
from scipy.cluster.hierarchy import inconsistent

# Calculate inconsistency
inconsistency = inconsistent (linkage_ward, d=2)

# High inconsistency suggests a good place to cut
print("Inconsistency statistics (last 10 merges):")
print(inconsistency[-10:])
\`\`\`

### Method 4: Silhouette Score

\`\`\`python
from sklearn.metrics import silhouette_score

# Try different numbers of clusters
silhouette_scores = []
K_range = range(2, 10)

for k in K_range:
    hc = AgglomerativeClustering (n_clusters=k, linkage='ward')
    labels = hc.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append (score)
    print(f"K={k}: Silhouette Score = {score:.3f}")

# Plot
plt.figure (figsize=(10, 6))
plt.plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('Silhouette Score', fontsize=12)
plt.title('Silhouette Analysis for Hierarchical Clustering', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xticks(K_range)
plt.show()

# Best K is where silhouette score is highest
best_k = K_range[np.argmax (silhouette_scores)]
print(f"\\nOptimal K based on silhouette score: {best_k}")
\`\`\`

## Distance Metrics

Hierarchical clustering can use any distance metric, making it very flexible.

\`\`\`python
from sklearn.metrics import pairwise_distances

# Common distance metrics
metrics = ['euclidean', 'manhattan', 'cosine', 'chebyshev']

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for idx, metric in enumerate (metrics):
    # Compute pairwise distances
    if metric == 'cosine':
        # Cosine distance requires non-Ward linkage
        linkage_mat = linkage(X, method='average', metric='cosine')
        hc = AgglomerativeClustering (n_clusters=4, linkage='average', metric='cosine')
    else:
        linkage_mat = linkage(X, method='ward', metric=metric)
        hc = AgglomerativeClustering (n_clusters=4, linkage='ward', metric=metric)

    labels = hc.fit_predict(X)

    axes[idx].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
    axes[idx].set_title (f'{metric.capitalize()} Distance', fontsize=14)
    axes[idx].set_xlabel('Feature 1')
    axes[idx].set_ylabel('Feature 2')

plt.tight_layout()
plt.show()
\`\`\`

## Comparison with K-Means

\`\`\`python
from sklearn.cluster import KMeans

# Compare hierarchical clustering with K-Means
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# K-Means
kmeans = KMeans (n_clusters=4, random_state=42, n_init=10)
labels_kmeans = kmeans.fit_predict(X)

axes[0].scatter(X[:, 0], X[:, 1], c=labels_kmeans, cmap='viridis', s=50, alpha=0.6)
axes[0].scatter (kmeans.cluster_centers_[:, 0],
               kmeans.cluster_centers_[:, 1],
               c='red', marker='X', s=200, edgecolors='black', linewidths=2)
axes[0].set_title('K-Means Clustering')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')

# Hierarchical
hc = AgglomerativeClustering (n_clusters=4, linkage='ward')
labels_hc = hc.fit_predict(X)

axes[1].scatter(X[:, 0], X[:, 1], c=labels_hc, cmap='viridis', s=50, alpha=0.6)
axes[1].set_title('Hierarchical Clustering (Ward)')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')

plt.tight_layout()
plt.show()

# Compare with evaluation metrics
from sklearn.metrics import adjusted_rand_score

ari = adjusted_rand_score (labels_kmeans, labels_hc)
print(f"Agreement between K-Means and Hierarchical: {ari:.3f}")
print(f"(1.0 = perfect agreement, 0 = random)")
\`\`\`

### Comparison Table

| Aspect | K-Means | Hierarchical |
|--------|---------|-------------|
| **K Required?** | ✅ Yes, upfront | ❌ No, cut later |
| **Speed** | Fast: O(nKt) | Slow: O(n²log n) |
| **Scalability** | Large datasets OK | Small to medium only |
| **Deterministic** | ❌ Random init | ✅ Yes |
| **Cluster Shape** | Spherical only | Any shape |
| **Distance Metric** | Euclidean | Any metric |
| **Visualization** | Scatter plot | Dendrogram |
| **Hierarchy** | No | Yes |

## Real-World Applications

### Application 1: Document Clustering

\`\`\`python
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents
documents = [
    "machine learning algorithms",
    "deep neural networks",
    "supervised learning methods",
    "stock market analysis",
    "financial trading strategies",
    "portfolio optimization",
    "natural language processing",
    "text classification models",
    "sentiment analysis techniques"
]

# Convert to TF-IDF features
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform (documents).toarray()

# Hierarchical clustering
linkage_docs = linkage(X_tfidf, method='ward')

# Plot dendrogram
plt.figure (figsize=(12, 6))
dendrogram (linkage_docs, labels=documents, leaf_rotation=45, leaf_font_size=10)
plt.title('Document Clustering Dendrogram')
plt.xlabel('Document')
plt.ylabel('Distance')
plt.tight_layout()
plt.show()

# Documents about similar topics cluster together
\`\`\`

### Application 2: Gene Expression Analysis

\`\`\`python
# Simulated gene expression data
# Rows = genes, Columns = experimental conditions
np.random.seed(42)
n_genes = 20
n_conditions = 5

gene_expression = np.random.randn (n_genes, n_conditions)

# Add structure: some genes have similar expression patterns
gene_expression[0:5] += 2  # Upregulated genes
gene_expression[10:15] -= 2  # Downregulated genes

gene_names = [f'Gene_{i}' for i in range (n_genes)]
condition_names = [f'Condition_{i}' for i in range (n_conditions)]

# Hierarchical clustering of genes
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

# Standardize
scaler = StandardScaler()
gene_expression_scaled = scaler.fit_transform (gene_expression)

# Cluster genes
linkage_genes = linkage (gene_expression_scaled, method='ward')

# Create heatmap with dendrogram
from scipy.cluster.hierarchy import dendrogram
import matplotlib.patches as mpatches

fig = plt.figure (figsize=(12, 8))

# Dendrogram
ax1 = fig.add_axes([0.05, 0.1, 0.2, 0.8])
dend = dendrogram (linkage_genes, orientation='left', no_labels=True)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title('Genes')

# Heatmap
ax2 = fig.add_axes([0.25, 0.1, 0.6, 0.8])
idx = dend['leaves']
gene_expression_ordered = gene_expression_scaled[idx, :]

im = ax2.imshow (gene_expression_ordered, aspect='auto', cmap='RdBu_r',
                vmin=-3, vmax=3)
ax2.set_xticks (range (n_conditions))
ax2.set_xticklabels (condition_names, rotation=45)
ax2.set_yticks (range (n_genes))
ax2.set_yticklabels([gene_names[i] for i in idx])
ax2.set_xlabel('Experimental Conditions')
ax2.set_title('Gene Expression Heatmap (Hierarchically Clustered)')

# Colorbar
ax3 = fig.add_axes([0.87, 0.1, 0.02, 0.8])
plt.colorbar (im, cax=ax3)

plt.show()

print("Genes with similar expression patterns cluster together")
print("This helps identify co-regulated genes")
\`\`\`

### Application 3: Customer Segmentation

\`\`\`python
import pandas as pd

# Customer data
np.random.seed(42)
n_customers = 50

customers = pd.DataFrame({
    'Recency': np.random.randint(1, 365, n_customers),  # Days since last purchase
    'Frequency': np.random.randint(1, 50, n_customers),  # Number of purchases
    'Monetary': np.random.randint(10, 5000, n_customers)  # Total amount spent
})

# Standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform (customers)

# Hierarchical clustering
linkage_customers = linkage(X_scaled, method='ward')

# Plot dendrogram
plt.figure (figsize=(14, 6))
dendrogram (linkage_customers, truncate_mode='lastp', p=20)
plt.title('Customer Segmentation Dendrogram')
plt.xlabel('Customer (or Cluster Size)')
plt.ylabel('Distance')
plt.axhline (y=8, color='r', linestyle='--', label='Cut at distance=8 (3 segments)')
plt.legend()
plt.show()

# Extract 3 customer segments
customers['Segment'] = fcluster (linkage_customers, t=3, criterion='maxclust')

# Analyze segments
print("\\nCustomer Segments:")
for segment in range(1, 4):
    segment_data = customers[customers['Segment'] == segment]
    print(f"\\nSegment {segment} (n={len (segment_data)}):")
    print(f"  Avg Recency: {segment_data['Recency'].mean():.1f} days")
    print(f"  Avg Frequency: {segment_data['Frequency'].mean():.1f} purchases")
    print(f"  Avg Monetary: \${segment_data['Monetary'].mean():.2f}")
\`\`\`

## Advantages of Hierarchical Clustering

✅ **No need to specify K upfront**: Explore at multiple granularities
✅ **Deterministic**: Same data always gives same result
✅ **Dendrogram provides insights**: Visual hierarchy of relationships
✅ **Any distance metric**: Euclidean, Manhattan, cosine, custom
✅ **Flexible cluster shapes**: Not limited to spherical clusters
✅ **Nested clusters**: Naturally handles hierarchical structure

## Limitations and Disadvantages

❌ **Computational complexity**: O(n²log n) - slow for large datasets
❌ **Memory intensive**: Requires storing distance matrix
❌ **Cannot undo merges**: Greedy algorithm, no backtracking
❌ **Sensitive to noise/outliers**: Early bad merges propagate
❌ **No global objective**: Unlike K-Means (minimizing WCSS)

## Practical Considerations

### When to Use Hierarchical Clustering

**✅ Use When**:
- Dataset is small to medium (< 10,000 samples)
- Want to explore data at multiple granularities
- Number of clusters unknown
- Need dendrogram visualization
- Have domain knowledge about hierarchy
- Deterministic results required

**❌ Avoid When**:
- Large datasets (use K-Means or Mini-Batch K-Means)
- Speed is critical
- Clear number of clusters known
- Memory constrained

### Scaling Features

\`\`\`python
# ALWAYS scale features for distance-based methods
from sklearn.preprocessing import StandardScaler

# Before scaling
linkage_unscaled = linkage(X, method='ward')

# After scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
linkage_scaled = linkage(X_scaled, method='ward')

# Scaling prevents features with large ranges from dominating
\`\`\`

### Handling Large Datasets

\`\`\`python
# For large datasets, use sampling or alternative methods
from sklearn.cluster import MiniBatchKMeans

# Option 1: Sample data first
sample_size = 1000
idx = np.random.choice (len(X), sample_size, replace=False)
X_sample = X[idx]

# Then apply hierarchical clustering
linkage_sample = linkage(X_sample, method='ward')

# Option 2: Use K-Means instead
# K-Means is much faster for large datasets
\`\`\`

## Summary

Hierarchical clustering is a powerful technique for discovering nested cluster structures without specifying the number of clusters upfront. The dendrogram provides intuitive visualization of relationships at all scales.

**Key Takeaways**:
- **Agglomerative**: Bottom-up merging (most common)
- **Linkage Methods**: Ward, Average, Complete, Single
- **Dendrogram**: Tree showing hierarchy
- **Cutting**: Choose K by cutting dendrogram
- **Best for**: Small datasets, exploratory analysis, hierarchical structures

**Next Steps**: For large datasets or when hierarchy isn't needed, consider K-Means or DBSCAN for better scalability.
`,
};
