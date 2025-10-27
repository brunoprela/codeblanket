/**
 * Section: Other Dimensionality Reduction Techniques
 * Module: Classical Machine Learning - Unsupervised Learning
 *
 * Comprehensive coverage of t-SNE, UMAP, and other advanced dimensionality reduction methods
 */

export const otherDimensionalityReduction = {
  id: 'other-dimensionality-reduction',
  title: 'Other Dimensionality Reduction Techniques',
  content: `
# Other Dimensionality Reduction Techniques

## Introduction

While PCA is powerful for linear dimensionality reduction, many real-world datasets have complex non-linear structures that PCA cannot capture. This section explores advanced techniques specifically designed for:

- **Non-linear manifolds**: Data lying on curved surfaces in high-D space
- **Visualization**: Preserving local neighborhood structure in 2D/3D
- **Complex relationships**: Capturing intricate patterns PCA misses

**Key Methods**:
- **t-SNE**: Visualization-focused, preserves local structure
- **UMAP**: Fast, scalable, preserves global and local structure
- **Autoencoders**: Deep learning-based, highly flexible
- **MDS, Isomap, LLE**: Classical manifold learning techniques

## t-SNE (t-Distributed Stochastic Neighbor Embedding)

### Introduction to t-SNE

t-SNE is a state-of-the-art technique for **visualizing** high-dimensional data. It excels at:
- Revealing clusters in high-dimensional data
- Preserving local neighborhood structure
- Creating beautiful, interpretable 2D/3D visualizations

**Core Idea**: Convert high-D distances to probabilities, then minimize difference between high-D and low-D probability distributions

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits, load_iris
from sklearn.preprocessing import StandardScaler
import time

# Load digits dataset
digits = load_digits()
X = digits.data
y = digits.target

print(f"Digits dataset: {X.shape}")
print(f"10 classes (0-9)")

# Visualize some digits
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate (axes.ravel()):
    ax.imshow(X[i].reshape(8, 8), cmap='gray')
    ax.set_title (f'Label: {y[i]}')
    ax.axis('off')
plt.suptitle('Sample Handwritten Digits')
plt.tight_layout()
plt.show()
\`\`\`

### Comparing PCA vs t-SNE

\`\`\`python
# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
from sklearn.decomposition import PCA

start_time = time.time()
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
pca_time = time.time() - start_time

# t-SNE
start_time = time.time()
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)
tsne_time = time.time() - start_time

# Visualize comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# PCA
scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', s=20, alpha=0.7)
axes[0].set_title (f'PCA (time: {pca_time:.2f}s)\\nVariance: {pca.explained_variance_ratio_.sum():.1%}')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
axes[0].grid(True, alpha=0.3)

# t-SNE
axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=20, alpha=0.7)
axes[1].set_title (f't-SNE (time: {tsne_time:.2f}s)\\nBetter cluster separation!')
axes[1].set_xlabel('t-SNE 1')
axes[1].set_ylabel('t-SNE 2')
axes[1].grid(True, alpha=0.3)

# Shared colorbar
fig.colorbar (scatter, ax=axes, ticks=range(10), label='Digit')

plt.tight_layout()
plt.show()

print("Observations:")
print("- PCA: Faster, but clusters overlap")
print("- t-SNE: Slower, but clusters are well-separated")
print("- t-SNE reveals structure PCA misses!")
\`\`\`

### How t-SNE Works

**Algorithm Steps**:

1. **High-Dimensional Similarities**: Convert distances to probabilities
   $$p_{j|i} = \\frac{\\exp(-||x_i - x_j||^2 / 2\\sigma_i^2)}{\\sum_{k \\neq i} \\exp(-||x_i - x_k||^2 / 2\\sigma_i^2)}$$

2. **Low-Dimensional Similarities**: Use Student\'s t-distribution
   $$q_{ij} = \\frac{(1 + ||y_i - y_j||^2)^{-1}}{\\sum_{k \\neq l} (1 + ||y_k - y_l||^2)^{-1}}$$

3. **Minimize KL Divergence**: Make \\(q_{ij}\\) match \\(p_{ij}\\)
   $$C = \\sum_i KL(P_i || Q_i) = \\sum_i \\sum_j p_{ij} \\log \\frac{p_{ij}}{q_{ij}}$$

4. **Gradient Descent**: Iteratively optimize point positions

### Key Parameters

#### Perplexity

Controls the balance between local and global structure (loosely: "number of neighbors to consider")

\`\`\`python
# Try different perplexity values
perplexity_values = [5, 30, 50, 100]

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for idx, perp in enumerate (perplexity_values):
    tsne_temp = TSNE(n_components=2, random_state=42,
                     perplexity=perp, n_iter=1000)
    X_tsne_temp = tsne_temp.fit_transform(X_scaled[:500])  # Use subset for speed

    axes[idx].scatter(X_tsne_temp[:, 0], X_tsne_temp[:, 1],
                     c=y[:500], cmap='tab10', s=30, alpha=0.7)
    axes[idx].set_title (f'Perplexity = {perp}')
    axes[idx].grid(True, alpha=0.3)

plt.suptitle('Effect of Perplexity on t-SNE', fontsize=14)
plt.tight_layout()
plt.show()

print("Perplexity guidelines:")
print("- Small (5-15): Focus on local structure, may create fragmented clusters")
print("- Medium (30-50): Balanced, DEFAULT and usually best")
print("- Large (100+): Focus on global structure, may merge distinct clusters")
print("- Rule of thumb: 5 to 50, depends on dataset size")
\`\`\`

#### Number of Iterations

\`\`\`python
# Try different iteration counts
n_iter_values = [250, 500, 1000, 2000]

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for idx, n_iter in enumerate (n_iter_values):
    tsne_temp = TSNE(n_components=2, random_state=42,
                     perplexity=30, n_iter=n_iter)
    X_tsne_temp = tsne_temp.fit_transform(X_scaled[:500])

    axes[idx].scatter(X_tsne_temp[:, 0], X_tsne_temp[:, 1],
                     c=y[:500], cmap='tab10', s=30, alpha=0.7)
    axes[idx].set_title (f'Iterations = {n_iter}')
    axes[idx].grid(True, alpha=0.3)

plt.suptitle('Effect of Iterations on t-SNE', fontsize=14)
plt.tight_layout()
plt.show()

print("More iterations = better convergence, but slower")
print("Default: 1000 iterations (usually sufficient)")
print("Increase if clusters look unformed")
\`\`\`

### Important Warnings About t-SNE

⚠️ **Critical Limitations**:

1. **Cluster sizes don't mean anything**: Expansion/contraction is arbitrary
2. **Distances between clusters don't mean anything**: Only local structure preserved
3. **Random initialization**: Different runs give different results
4. **Slow**: O(n²) - struggles with >10,000 samples
5. **No transform for new data**: Must re-run entire algorithm

\`\`\`python
# Same data, different random seeds
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for idx, seed in enumerate([42, 123, 456, 789]):
    tsne_temp = TSNE(n_components=2, random_state=seed, perplexity=30)
    X_tsne_temp = tsne_temp.fit_transform(X_scaled[:500])

    axes[idx].scatter(X_tsne_temp[:, 0], X_tsne_temp[:, 1],
                     c=y[:500], cmap='tab10', s=30, alpha=0.7)
    axes[idx].set_title (f'Random Seed = {seed}')
    axes[idx].grid(True, alpha=0.3)

plt.suptitle('t-SNE with Different Random Seeds', fontsize=14)
plt.tight_layout()
plt.show()

print("Same data, different layouts!")
print("But cluster structure (which points are together) is consistent")
\`\`\`

### Best Practices for t-SNE

\`\`\`python
# Recommended workflow
from sklearn.decomposition import PCA

# 1. Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Pre-reduce with PCA (if >50 dimensions)
if X_scaled.shape[1] > 50:
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X_scaled)
    print(f"Pre-reduced from {X_scaled.shape[1]}D to 50D with PCA")
else:
    X_pca = X_scaled

# 3. Apply t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=30,  # Start with 30
    n_iter=1000,  # At least 1000
    random_state=42,  # For reproducibility
    learning_rate='auto',  # Let sklearn choose
    init='pca'  # Initialize with PCA (faster convergence)
)

X_tsne = tsne.fit_transform(X_pca)

plt.figure (figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=30, alpha=0.7)
plt.colorbar (scatter, label='Class')
plt.title('t-SNE Visualization (Optimized)')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
\`\`\`

## UMAP (Uniform Manifold Approximation and Projection)

### Introduction to UMAP

UMAP is a modern alternative to t-SNE that addresses many of its limitations:

✅ **Faster**: Can handle millions of samples
✅ **Preserves global structure**: Better than t-SNE
✅ **Can transform new data**: Has .transform() method
✅ **More stable**: Less sensitive to parameters
✅ **General purpose**: Works for visualization AND downstream tasks

\`\`\`python
# Note: Requires umap-learn library
# pip install umap-learn

try:
    import umap

    # Apply UMAP
    start_time = time.time()
    umap_model = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
    X_umap = umap_model.fit_transform(X_scaled)
    umap_time = time.time() - start_time

    # Compare t-SNE vs UMAP
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # t-SNE
    axes[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=20, alpha=0.7)
    axes[0].set_title (f't-SNE (time: {tsne_time:.2f}s)')
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')
    axes[0].grid(True, alpha=0.3)

    # UMAP
    scatter = axes[1].scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10', s=20, alpha=0.7)
    axes[1].set_title (f'UMAP (time: {umap_time:.2f}s)\\nFaster & preserves global structure!')
    axes[1].set_xlabel('UMAP 1')
    axes[1].set_ylabel('UMAP 2')
    axes[1].grid(True, alpha=0.3)

    fig.colorbar (scatter, ax=axes, ticks=range(10), label='Digit')
    plt.tight_layout()
    plt.show()

    print(f"UMAP is {tsne_time/umap_time:.1f}x faster than t-SNE on this dataset")

except ImportError:
    print("UMAP not installed. Install with: pip install umap-learn")
    print("For now, we'll continue with other techniques...")
\`\`\`

### UMAP Parameters

\`\`\`python
try:
    # n_neighbors: controls local vs global structure
    n_neighbors_values = [5, 15, 50, 100]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()

    for idx, n_neighbors in enumerate (n_neighbors_values):
        umap_temp = umap.UMAP(n_components=2, random_state=42,
                             n_neighbors=n_neighbors)
        X_umap_temp = umap_temp.fit_transform(X_scaled[:500])

        axes[idx].scatter(X_umap_temp[:, 0], X_umap_temp[:, 1],
                         c=y[:500], cmap='tab10', s=30, alpha=0.7)
        axes[idx].set_title (f'n_neighbors = {n_neighbors}')
        axes[idx].grid(True, alpha=0.3)

    plt.suptitle('Effect of n_neighbors on UMAP', fontsize=14)
    plt.tight_layout()
    plt.show()

    print("n_neighbors guidelines:")
    print("- Small (5-15): Focus on local structure")
    print("- Medium (15-50): Balanced, DEFAULT")
    print("- Large (50+): Focus on global structure")

except NameError:
    print("UMAP not available")
\`\`\`

### UMAP Can Transform New Data!

Unlike t-SNE, UMAP can project new points into existing embedding:

\`\`\`python
try:
    from sklearn.model_selection import train_test_split

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    # Fit UMAP on training data
    umap_model = umap.UMAP(n_components=2, random_state=42)
    X_train_umap = umap_model.fit_transform(X_train)

    # Transform test data (no re-fitting!)
    X_test_umap = umap_model.transform(X_test)

    plt.figure (figsize=(12, 6))

    plt.scatter(X_train_umap[:, 0], X_train_umap[:, 1],
               c=y_train, cmap='tab10', s=30, alpha=0.5, label='Train')
    plt.scatter(X_test_umap[:, 0], X_test_umap[:, 1],
               c=y_test, cmap='tab10', s=80, alpha=0.8,
               marker='*', edgecolors='black', linewidths=1, label='Test')

    plt.title('UMAP: Train Data + Transformed Test Data')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("UMAP can transform new data - useful for production!")
    print("t-SNE cannot do this")

except NameError:
    print("UMAP not available")
\`\`\`

## Other Manifold Learning Techniques

### MDS (Multidimensional Scaling)

Preserves pairwise distances between points

\`\`\`python
from sklearn.manifold import MDS

# Apply MDS
mds = MDS(n_components=2, random_state=42)
X_mds = mds.fit_transform(X_scaled[:500])  # Use subset for speed

plt.figure (figsize=(10, 8))
plt.scatter(X_mds[:, 0], X_mds[:, 1], c=y[:500], cmap='tab10', s=30, alpha=0.7)
plt.title('MDS Visualization')
plt.xlabel('MDS 1')
plt.ylabel('MDS 2')
plt.colorbar (label='Digit')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("MDS: Preserves pairwise distances")
print("Good for: Distance-based analysis")
print("Slow: O(n²) memory and time")
\`\`\`

### Isomap

Non-linear dimensionality reduction based on geodesic distances

\`\`\`python
from sklearn.manifold import Isomap

# Apply Isomap
isomap = Isomap (n_components=2, n_neighbors=10)
X_isomap = isomap.fit_transform(X_scaled[:500])

plt.figure (figsize=(10, 8))
plt.scatter(X_isomap[:, 0], X_isomap[:, 1], c=y[:500], cmap='tab10', s=30, alpha=0.7)
plt.title('Isomap Visualization')
plt.xlabel('Isomap 1')
plt.ylabel('Isomap 2')
plt.colorbar (label='Digit')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("Isomap: Uses geodesic distances (along manifold)")
print("Good for: Data on curved manifolds")
print("Assumption: Data lies on a single connected manifold")
\`\`\`

### LLE (Locally Linear Embedding)

Preserves local linear structures

\`\`\`python
from sklearn.manifold import LocallyLinearEmbedding

# Apply LLE
lle = LocallyLinearEmbedding (n_components=2, n_neighbors=10, random_state=42)
X_lle = lle.fit_transform(X_scaled[:500])

plt.figure (figsize=(10, 8))
plt.scatter(X_lle[:, 0], X_lle[:, 1], c=y[:500], cmap='tab10', s=30, alpha=0.7)
plt.title('LLE Visualization')
plt.xlabel('LLE 1')
plt.ylabel('LLE 2')
plt.colorbar (label='Digit')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("LLE: Assumes data locally linear")
print("Good for: Smooth manifolds")
print("Sensitive to noise")
\`\`\`

## Comparison of All Methods

\`\`\`python
# Compare all methods on iris dataset (smaller for speed)
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

scaler_iris = StandardScaler()
X_iris_scaled = scaler_iris.fit_transform(X_iris)

methods = {
    'PCA': PCA(n_components=2),
    't-SNE': TSNE(n_components=2, random_state=42),
    'MDS': MDS(n_components=2, random_state=42),
    'Isomap': Isomap (n_components=2, n_neighbors=10),
    'LLE': LocallyLinearEmbedding (n_components=2, n_neighbors=10, random_state=42)
}

try:
    methods['UMAP'] = umap.UMAP(n_components=2, random_state=42)
except NameError:
    pass

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.ravel()

for idx, (name, method) in enumerate (methods.items()):
    start_time = time.time()
    X_transformed = method.fit_transform(X_iris_scaled)
    elapsed_time = time.time() - start_time

    axes[idx].scatter(X_transformed[:, 0], X_transformed[:, 1],
                     c=y_iris, cmap='viridis', s=50, alpha=0.7)
    axes[idx].set_title (f'{name}\\nTime: {elapsed_time:.3f}s')
    axes[idx].set_xlabel('Component 1')
    axes[idx].set_ylabel('Component 2')
    axes[idx].grid(True, alpha=0.3)

# Hide extra subplot if needed
if len (methods) < 6:
    axes[5].axis('off')

plt.suptitle('Comparison of Dimensionality Reduction Methods (Iris Dataset)', fontsize=14)
plt.tight_layout()
plt.show()
\`\`\`

## Method Selection Guide

| Method | Speed | Global Structure | Local Structure | New Data Transform | Use Case |
|--------|-------|-----------------|-----------------|-------------------|----------|
| **PCA** | ⚡⚡⚡ Fast | ✅ | ❌ | ✅ | Linear reduction, preprocessing |
| **t-SNE** | 🐌 Slow | ❌ | ✅✅ | ❌ | Visualization only |
| **UMAP** | ⚡⚡ Fast | ✅ | ✅ | ✅ | General purpose, best overall |
| **MDS** | 🐌 Slow | ✅✅ | ✅ | ❌ | Distance preservation |
| **Isomap** | 🐌 Slow | ✅ | ✅ | ⚠️ | Geodesic distances |
| **LLE** | 🐌 Slow | ❌ | ✅✅ | ❌ | Smooth manifolds |

### Decision Tree

\`\`\`
Need to transform new data?
├─ YES → PCA or UMAP
└─ NO
   ├─ Linear relationships? → PCA
   └─ Non-linear
      ├─ Just visualization? → t-SNE or UMAP
      ├─ Preserve distances? → MDS
      ├─ Geodesic distances? → Isomap
      └─ Local linear? → LLE
\`\`\`

## Real-World Applications

### Application 1: Single-Cell RNA Sequencing

\`\`\`python
# Simulated single-cell gene expression
# Cells × Genes (thousands of dimensions!)
np.random.seed(42)
n_cells = 500
n_genes = 2000

# Three cell types
X_cells = np.vstack([
    np.random.randn(200, n_genes) * 0.5 + np.random.randn(1, n_genes) * 2,  # Type A
    np.random.randn(150, n_genes) * 0.5 + np.random.randn(1, n_genes) * 2,  # Type B
    np.random.randn(150, n_genes) * 0.5 + np.random.randn(1, n_genes) * 2,  # Type C
])
y_cells = np.array([0]*200 + [1]*150 + [2]*150)

print(f"Single-cell data: {X_cells.shape}")
print(f"2000 genes measured per cell!")

# Scale
scaler_cells = StandardScaler()
X_cells_scaled = scaler_cells.fit_transform(X_cells)

# PCA preprocessing
pca_cells = PCA(n_components=50)
X_cells_pca = pca_cells.fit_transform(X_cells_scaled)

# t-SNE visualization
tsne_cells = TSNE(n_components=2, random_state=42, perplexity=30)
X_cells_tsne = tsne_cells.fit_transform(X_cells_pca)

plt.figure (figsize=(12, 8))
scatter = plt.scatter(X_cells_tsne[:, 0], X_cells_tsne[:, 1],
                     c=y_cells, cmap='viridis', s=30, alpha=0.7)
plt.title('Single-Cell RNA-seq: Cell Type Discovery\\n2000 genes → 50D (PCA) → 2D (t-SNE)')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.colorbar (scatter, ticks=[0, 1, 2], label='Cell Type')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("Dimensionality reduction reveals distinct cell populations!")
\`\`\`

### Application 2: Word Embeddings Visualization

\`\`\`python
# Simulated word embeddings (words in 100D space)
np.random.seed(42)

# Word categories
animals = np.random.randn(10, 100) + np.array([5, 5] + [0]*98)
fruits = np.random.randn(10, 100) + np.array([-5, 5] + [0]*98)
vehicles = np.random.randn(10, 100) + np.array([5, -5] + [0]*98)
colors = np.random.randn(10, 100) + np.array([-5, -5] + [0]*98)

X_words = np.vstack([animals, fruits, vehicles, colors])
y_words = np.array([0]*10 + [1]*10 + [2]*10 + [3]*10)
categories = ['Animals', 'Fruits', 'Vehicles', 'Colors']

# t-SNE visualization
tsne_words = TSNE(n_components=2, random_state=42, perplexity=15)
X_words_tsne = tsne_words.fit_transform(X_words)

plt.figure (figsize=(12, 8))
colors_map = ['red', 'green', 'blue', 'orange']
for i, (category, color) in enumerate (zip (categories, colors_map)):
    mask = y_words == i
    plt.scatter(X_words_tsne[mask, 0], X_words_tsne[mask, 1],
               c=color, label=category, s=100, alpha=0.7, edgecolors='black')

plt.title('Word Embeddings Visualization (100D → 2D)')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("Semantically similar words cluster together")
\`\`\`

## Best Practices Summary

1. **Always scale** features before dimensionality reduction
2. **PCA preprocessing**: For t-SNE/UMAP, first reduce to ~50D with PCA
3. **Use UMAP** for general purpose (unless you don't have the library)
4. **Use t-SNE** purely for visualization (beautiful plots!)
5. **Multiple random seeds**: Try different seeds for t-SNE/UMAP
6. **Validate clusters**: Use domain knowledge, not just visual appeal
7. **Don't over-interpret**: Cluster sizes/distances may be misleading
8. **Check parameters**: Try different perplexity/n_neighbors values

## Summary

**Key Takeaways**:
- **t-SNE**: Best for visualization, slow, can't transform new data
- **UMAP**: Faster, more versatile, can transform new data, preserves global structure
- **Classical methods** (MDS, Isomap, LLE): Specific use cases, generally slower
- **Always preprocess** with PCA for high-dimensional data (>50D)
- **Visualization ≠ Truth**: These are projections; always validate findings

**Recommended Workflow**:
1. Start with PCA (fast, interpretable)
2. If non-linear, try UMAP (versatile, fast)
3. For publication-quality plots, use t-SNE
4. Validate with domain knowledge and downstream tasks

**Next**: We'll explore anomaly detection techniques for finding outliers and unusual patterns in data.
`,
};
