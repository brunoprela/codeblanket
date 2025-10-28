/**
 * Section: K-Means Clustering
 * Module: Classical Machine Learning - Unsupervised Learning
 *
 * Comprehensive coverage of K-Means clustering algorithm, implementation, and optimization
 */

export const kMeansClustering = {
  id: 'k-means-clustering',
  title: 'K-Means Clustering',
  content: `
# K-Means Clustering

## Introduction

K-Means is one of the most popular and widely-used clustering algorithms in machine learning. It\'s simple, fast, and effective for discovering groups in data. The algorithm partitions data into K distinct, non-overlapping clusters where each data point belongs to the cluster with the nearest mean (centroid).

**Core Idea**: Minimize the within-cluster variance (distance from points to their cluster center)

## How K-Means Works

### Algorithm Steps

1. **Initialize**: Choose K random points as initial centroids
2. **Assignment**: Assign each point to nearest centroid
3. **Update**: Recalculate centroids as mean of assigned points
4. **Repeat**: Steps 2-3 until convergence (centroids stop moving)

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate sample data
X, y_true = make_blobs (n_samples=300, centers=4,
                       cluster_std=0.60, random_state=42)

# Visualize data
plt.scatter(X[:, 0], X[:, 1], alpha=0.6)
plt.title('Data to be Clustered')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
\`\`\`

### Mathematical Formulation

**Objective**: Minimize within-cluster sum of squares (WCSS)

$$J = \\sum_{i=1}^{K} \\sum_{x \\in C_i} ||x - \\mu_i||^2$$

Where:
- $K$ = number of clusters
- $C_i$ = set of points in cluster $i$
- $\\mu_i$ = centroid of cluster $i$
- $||x - \\mu_i||$ = Euclidean distance

## K-Means from Scratch

\`\`\`python
class KMeansFromScratch:
    def __init__(self, n_clusters=3, max_iters=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None

    def fit (self, X):
        """Fit K-Means to data"""
        np.random.seed (self.random_state)
        n_samples, n_features = X.shape

        # Initialize centroids randomly from data points
        idx = np.random.choice (n_samples, self.n_clusters, replace=False)
        self.centroids = X[idx]

        for iteration in range (self.max_iters):
            # Assignment step: assign points to nearest centroid
            distances = self._compute_distances(X)
            new_labels = np.argmin (distances, axis=1)

            # Check for convergence
            if iteration > 0 and np.array_equal (new_labels, self.labels_):
                print(f"Converged after {iteration} iterations")
                break

            self.labels_ = new_labels

            # Update step: recalculate centroids
            for k in range (self.n_clusters):
                if np.sum (self.labels_ == k) > 0:  # Avoid empty clusters
                    self.centroids[k] = X[self.labels_ == k].mean (axis=0)

        return self

    def _compute_distances (self, X):
        """Compute distances from all points to all centroids"""
        distances = np.zeros((X.shape[0], self.n_clusters))
        for k in range (self.n_clusters):
            distances[:, k] = np.linalg.norm(X - self.centroids[k], axis=1)
        return distances

    def predict (self, X):
        """Assign new points to nearest centroid"""
        distances = self._compute_distances(X)
        return np.argmin (distances, axis=1)

    def fit_predict (self, X):
        """Fit and return cluster labels"""
        self.fit(X)
        return self.labels_

# Test implementation
kmeans_scratch = KMeansFromScratch (n_clusters=4, random_state=42)
labels = kmeans_scratch.fit_predict(X)

# Visualize results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.scatter (kmeans_scratch.centroids[:, 0],
            kmeans_scratch.centroids[:, 1],
            c='red', marker='X', s=200, linewidths=2,
            edgecolors='black', label='Centroids')
plt.title('K-Means Clustering (From Scratch)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
\`\`\`

## Using scikit-learn K-Means

\`\`\`python
from sklearn.cluster import KMeans

# Create and fit model
kmeans = KMeans (n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

# Access results
print(f"Cluster centers:\\n{kmeans.cluster_centers_}")
print(f"Inertia (WCSS): {kmeans.inertia_:.2f}")
print(f"Number of iterations: {kmeans.n_iter_}")

# Visualize
plt.figure (figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.scatter (kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            c='red', marker='X', s=200, linewidths=2,
            edgecolors='black')
plt.title('K-Means Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.6)
plt.title('True Labels (for comparison)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()
\`\`\`

## Choosing K: The Elbow Method

One of the biggest challenges with K-Means is choosing the right number of clusters.

### Elbow Method

Plot WCSS (inertia) for different values of K and look for an "elbow" where the rate of decrease sharply changes.

\`\`\`python
# Calculate WCSS for different K values
wcss = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans (n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append (kmeans.inertia_)

# Plot elbow curve
plt.figure (figsize=(10, 6))
plt.plot(K_range, wcss, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=12)
plt.title('Elbow Method for Optimal K', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xticks(K_range)

# The "elbow" is around K=4 for our data
plt.axvline (x=4, color='r', linestyle='--', alpha=0.5, label='Optimal K=4')
plt.legend()
plt.show()

print("WCSS values:", wcss)
\`\`\`

### Silhouette Method

Measures how similar a point is to its own cluster compared to other clusters.

\`\`\`python
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.cm as cm

# Calculate silhouette scores for different K
silhouette_scores = []
K_range = range(2, 11)  # Silhouette requires at least 2 clusters

for k in K_range:
    kmeans = KMeans (n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append (score)
    print(f"K={k}: Silhouette Score = {score:.3f}")

# Plot silhouette scores
plt.figure (figsize=(10, 6))
plt.plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('Silhouette Score', fontsize=12)
plt.title('Silhouette Method for Optimal K', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xticks(K_range)
plt.axvline (x=4, color='r', linestyle='--', alpha=0.5, label='Optimal K=4')
plt.legend()
plt.show()
\`\`\`

### Silhouette Plot

Visualize silhouette coefficients for each sample.

\`\`\`python
def plot_silhouette(X, n_clusters):
    """Create silhouette plot for given number of clusters"""
    fig, ax = plt.subplots (figsize=(10, 7))

    # Fit K-Means
    kmeans = KMeans (n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # Calculate silhouette scores
    silhouette_avg = silhouette_score(X, labels)
    sample_silhouette_values = silhouette_samples(X, labels)

    y_lower = 10
    for i in range (n_clusters):
        # Get silhouette scores for cluster i
        cluster_silhouette_values = sample_silhouette_values[labels == i]
        cluster_silhouette_values.sort()

        size_cluster_i = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral (float (i) / n_clusters)
        ax.fill_betweenx (np.arange (y_lower, y_upper),
                         0, cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)

        # Label clusters
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str (i))
        y_lower = y_upper + 10

    ax.set_xlabel("Silhouette Coefficient", fontsize=12)
    ax.set_ylabel("Cluster Label", fontsize=12)
    ax.set_title (f"Silhouette Plot for K={n_clusters}\\n" +
                 f"Average Silhouette Score: {silhouette_avg:.3f}", fontsize=14)

    # Vertical line for average score
    ax.axvline (x=silhouette_avg, color="red", linestyle="--",
               label=f'Average: {silhouette_avg:.3f}')
    ax.legend()

    plt.tight_layout()
    plt.show()

# Plot for K=4
plot_silhouette(X, n_clusters=4)
\`\`\`

## Initialization Strategies

The initial placement of centroids significantly impacts the final result.

### Random Initialization Problem

\`\`\`python
# Show different results with different random initializations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i in range(6):
    kmeans = KMeans (n_clusters=4, random_state=i, n_init=1, max_iter=10)
    labels = kmeans.fit_predict(X)

    axes[i].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
    axes[i].scatter (kmeans.cluster_centers_[:, 0],
                   kmeans.cluster_centers_[:, 1],
                   c='red', marker='X', s=100)
    axes[i].set_title (f'Random State = {i}\\nInertia: {kmeans.inertia_:.0f}')

plt.tight_layout()
plt.show()

# Different initializations can give different results!
\`\`\`

### K-Means++ Initialization

Smart initialization that spreads initial centroids out, leading to better and more consistent results.

\`\`\`python
# Compare random vs K-Means++ initialization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Random initialization
kmeans_random = KMeans (n_clusters=4, init='random', n_init=1, random_state=42)
labels_random = kmeans_random.fit_predict(X)

axes[0].scatter(X[:, 0], X[:, 1], c=labels_random, cmap='viridis', alpha=0.6)
axes[0].scatter (kmeans_random.cluster_centers_[:, 0],
               kmeans_random.cluster_centers_[:, 1],
               c='red', marker='X', s=200)
axes[0].set_title (f'Random Init\\nInertia: {kmeans_random.inertia_:.0f}\\n' +
                  f'Iterations: {kmeans_random.n_iter_}')

# K-Means++ initialization (default in sklearn)
kmeans_pp = KMeans (n_clusters=4, init='k-means++', n_init=1, random_state=42)
labels_pp = kmeans_pp.fit_predict(X)

axes[1].scatter(X[:, 0], X[:, 1], c=labels_pp, cmap='viridis', alpha=0.6)
axes[1].scatter (kmeans_pp.cluster_centers_[:, 0],
               kmeans_pp.cluster_centers_[:, 1],
               c='red', marker='X', s=200)
axes[1].set_title (f'K-Means++ Init\\nInertia: {kmeans_pp.inertia_:.0f}\\n' +
                  f'Iterations: {kmeans_pp.n_iter_}')

plt.tight_layout()
plt.show()

# K-Means++ usually gives better results with fewer iterations
\`\`\`

## Real-World Applications

### Customer Segmentation

\`\`\`python
# E-commerce customer segmentation
customers = pd.DataFrame({
    'Annual_Income': [15, 16, 17, 18, 19, 20, 25, 30, 35, 40,
                      45, 50, 55, 60, 65, 70, 75, 80, 85, 90],
    'Spending_Score': [39, 81, 6, 77, 40, 76, 94, 3, 72, 14,
                       99, 15, 77, 13, 79, 35, 66, 29, 80, 87]
})

# Scale features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform (customers)

# Find optimal K
wcss = []
for k in range(1, 11):
    kmeans = KMeans (n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append (kmeans.inertia_)

# Cluster with K=5
kmeans = KMeans (n_clusters=5, random_state=42)
customers['Segment'] = kmeans.fit_predict(X_scaled)

# Visualize segments
plt.figure (figsize=(10, 6))
scatter = plt.scatter (customers['Annual_Income'],
                     customers['Spending_Score'],
                     c=customers['Segment'],
                     cmap='viridis', s=100, alpha=0.6)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segmentation')
plt.colorbar (scatter, label='Segment')
plt.show()

# Interpret segments
for segment in range(5):
    segment_data = customers[customers['Segment'] == segment]
    print(f"\\nSegment {segment}:")
    print(f"  Avg Income: \\$\{segment_data['Annual_Income'].mean():.1f}k")
    print(f"  Avg Spending Score: {segment_data['Spending_Score'].mean():.1f}")
    print(f"  Size: {len (segment_data)} customers")
\`\`\`

### Image Compression

\`\`\`python
from sklearn.cluster import MiniBatchKMeans
from skimage import io

# Load image
image = io.imread('image.jpg')  # RGB image
h, w, c = image.shape

# Reshape to (n_pixels, 3)
X_pixels = image.reshape(-1, 3)

print(f"Original: {len (np.unique(X_pixels, axis=0))} unique colors")

# Compress to 64 colors
n_colors = 64
kmeans = MiniBatchKMeans (n_clusters=n_colors, random_state=42, batch_size=1000)
labels = kmeans.fit_predict(X_pixels)
compressed = kmeans.cluster_centers_[labels].reshape (h, w, c).astype('uint8')

print(f"Compressed: {n_colors} colors")
print(f"Compression ratio: {len (np.unique(X_pixels, axis=0)) / n_colors:.1f}x")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].imshow (image)
axes[0].set_title('Original Image')
axes[0].axis('off')
axes[1].imshow (compressed)
axes[1].set_title (f'Compressed ({n_colors} colors)')
axes[1].axis('off')
plt.tight_layout()
plt.show()
\`\`\`

## Limitations and Assumptions

### 1. Assumes Spherical Clusters

\`\`\`python
# Generate non-spherical clusters
from sklearn.datasets import make_moons

X_moons, _ = make_moons (n_samples=200, noise=0.05, random_state=42)

kmeans = KMeans (n_clusters=2, random_state=42)
labels = kmeans.fit_predict(X_moons)

plt.scatter(X_moons[:, 0], X_moons[:, 1], c=labels, cmap='viridis')
plt.title('K-Means on Non-Spherical Data (Poor Results)')
plt.show()

# K-Means struggles with non-spherical clusters!
\`\`\`

### 2. Sensitive to Outliers

\`\`\`python
# Add outliers
X_with_outliers = np.vstack([X, [[10, 10], [10, -10], [-10, 10]]])

kmeans = KMeans (n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X_with_outliers)

plt.scatter(X_with_outliers[:, 0], X_with_outliers[:, 1],
           c=labels, cmap='viridis', alpha=0.6)
plt.scatter (kmeans.cluster_centers_[:, 0],
           kmeans.cluster_centers_[:, 1],
           c='red', marker='X', s=200)
plt.title('K-Means with Outliers (Centroids Shifted)')
plt.show()

# Outliers pull centroids away from true cluster centers
\`\`\`

### 3. Must Specify K in Advance

\`\`\`python
# No automatic way to determine K
# Must use elbow method or silhouette analysis
\`\`\`

### 4. Produces Equal-Sized Clusters (Tendency)

\`\`\`python
# K-Means tends to create clusters of similar sizes
# even when true clusters have different sizes
\`\`\`

## Mini-Batch K-Means

For large datasets, Mini-Batch K-Means is much faster with slightly less accuracy.

\`\`\`python
import time

# Generate large dataset
X_large, _ = make_blobs (n_samples=100000, centers=5, random_state=42)

# Regular K-Means
start = time.time()
kmeans = KMeans (n_clusters=5, random_state=42)
kmeans.fit(X_large)
time_kmeans = time.time() - start

# Mini-Batch K-Means
start = time.time()
mb_kmeans = MiniBatchKMeans (n_clusters=5, random_state=42, batch_size=1000)
mb_kmeans.fit(X_large)
time_mb = time.time() - start

print(f"K-Means: {time_kmeans:.2f}s, Inertia: {kmeans.inertia_:.0f}")
print(f"Mini-Batch: {time_mb:.2f}s, Inertia: {mb_kmeans.inertia_:.0f}")
print(f"Speedup: {time_kmeans/time_mb:.1f}x")
\`\`\`

## Best Practices

1. **Always scale features** - K-Means uses Euclidean distance
2. **Use K-Means++** initialization (default in sklearn)
3. **Run multiple times** (n_init parameter) to avoid local minima
4. **Use elbow/silhouette** to choose K
5. **Check assumptions** - spherical clusters, similar sizes
6. **Handle outliers** - Remove or use robust alternatives
7. **Use Mini-Batch** for large datasets (>10K samples)

## Summary

K-Means is a fast, simple, and effective clustering algorithm that works well when:
- Clusters are roughly spherical
- Clusters have similar sizes
- Number of clusters is known
- Data is not too noisy

Choose other algorithms (DBSCAN, hierarchical) when these assumptions don't hold.
`,
};
