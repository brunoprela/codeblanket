/**
 * Section: DBSCAN & Density-Based Clustering
 * Module: Classical Machine Learning - Unsupervised Learning
 *
 * Comprehensive coverage of DBSCAN and density-based clustering for arbitrary shapes and outlier detection
 */

export const dbscanDensityClustering = {
  id: 'dbscan-density-clustering',
  title: 'DBSCAN & Density-Based Clustering',
  content: `
# DBSCAN & Density-Based Clustering

## Introduction

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a powerful clustering algorithm that groups together points that are closely packed together, marking points in low-density regions as outliers. Unlike K-Means and Hierarchical clustering, DBSCAN can:

- Find arbitrarily shaped clusters
- Handle noise and outliers elegantly
- Doesn't require specifying the number of clusters upfront

**Core Idea**: Clusters are dense regions in space separated by regions of lower density

## How DBSCAN Works

### Key Concepts

**1. Epsilon (ε)**: Maximum distance between two points to be considered neighbors

**2. MinPts**: Minimum number of points to form a dense region

**3. Point Types**:
- **Core Point**: Has at least MinPts neighbors within ε distance
- **Border Point**: Has fewer than MinPts neighbors but is within ε of a core point
- **Noise Point**: Neither core nor border point

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Generate data with non-spherical shapes
X_moons, _ = make_moons (n_samples=300, noise=0.05, random_state=42)

# Visualize data
plt.figure (figsize=(10, 6))
plt.scatter(X_moons[:, 0], X_moons[:, 1], alpha=0.6, s=50)
plt.title('Data with Non-Spherical Clusters (Moons)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# This shape is perfect for DBSCAN, challenging for K-Means
\`\`\`

### Algorithm Steps

1. **For each point** not yet visited:
   - Mark it as visited
   - Find all points within ε distance (neighbors)
   - If neighbors < MinPts: mark as noise (for now)
   - If neighbors ≥ MinPts: start new cluster
     - Add all neighbors to cluster
     - Recursively visit each neighbor and expand cluster

2. **Result**: Clusters and noise points

\`\`\`python
# Apply DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X_moons)

# -1 indicates noise points
n_clusters = len (set (labels)) - (1 if -1 in labels else 0)
n_noise = list (labels).count(-1)

print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")
print(f"Cluster labels: {set (labels)}")

# Visualize results
plt.figure (figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_moons[:, 0], X_moons[:, 1], alpha=0.6, s=50)
plt.title('Original Data')

plt.subplot(1, 2, 2)
# Plot clusters
for label in set (labels):
    if label == -1:
        # Noise points in black
        mask = labels == label
        plt.scatter(X_moons[mask, 0], X_moons[mask, 1], 
                   c='black', marker='x', s=50, label='Noise', alpha=0.6)
    else:
        mask = labels == label
        plt.scatter(X_moons[mask, 0], X_moons[mask, 1], 
                   s=50, alpha=0.6, label=f'Cluster {label}')

plt.title (f'DBSCAN Results ({n_clusters} clusters, {n_noise} noise)')
plt.legend()
plt.tight_layout()
plt.show()

# DBSCAN perfectly identifies the two moon-shaped clusters!
\`\`\`

## Understanding Parameters

### Epsilon (ε): Neighborhood Radius

The ε parameter defines "how far" to look for neighbors.

\`\`\`python
# Try different epsilon values
eps_values = [0.1, 0.2, 0.3, 0.5]

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for idx, eps in enumerate (eps_values):
    dbscan = DBSCAN(eps=eps, min_samples=5)
    labels = dbscan.fit_predict(X_moons)
    
    n_clusters = len (set (labels)) - (1 if -1 in labels else 0)
    n_noise = list (labels).count(-1)
    
    # Plot
    for label in set (labels):
        mask = labels == label
        if label == -1:
            axes[idx].scatter(X_moons[mask, 0], X_moons[mask, 1], 
                            c='black', marker='x', s=30, alpha=0.6)
        else:
            axes[idx].scatter(X_moons[mask, 0], X_moons[mask, 1], s=30, alpha=0.6)
    
    axes[idx].set_title (f'eps={eps}\\nClusters: {n_clusters}, Noise: {n_noise}')
    axes[idx].set_xlabel('Feature 1')
    axes[idx].set_ylabel('Feature 2')

plt.tight_layout()
plt.show()

# Small eps → more clusters, more noise
# Large eps → fewer clusters, less noise, may merge distinct clusters
\`\`\`

### MinPts: Minimum Points for Core

The MinPts parameter defines "how many" neighbors needed to be a core point.

\`\`\`python
# Try different min_samples values
min_samples_values = [3, 5, 10, 20]

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for idx, min_samples in enumerate (min_samples_values):
    dbscan = DBSCAN(eps=0.3, min_samples=min_samples)
    labels = dbscan.fit_predict(X_moons)
    
    n_clusters = len (set (labels)) - (1 if -1 in labels else 0)
    n_noise = list (labels).count(-1)
    
    # Plot
    for label in set (labels):
        mask = labels == label
        if label == -1:
            axes[idx].scatter(X_moons[mask, 0], X_moons[mask, 1], 
                            c='black', marker='x', s=30, alpha=0.6)
        else:
            axes[idx].scatter(X_moons[mask, 0], X_moons[mask, 1], s=30, alpha=0.6)
    
    axes[idx].set_title (f'min_samples={min_samples}\\nClusters: {n_clusters}, Noise: {n_noise}')
    axes[idx].set_xlabel('Feature 1')
    axes[idx].set_ylabel('Feature 2')

plt.tight_layout()
plt.show()

# Small min_samples → more lenient, fewer noise points
# Large min_samples → more strict, more noise points
\`\`\`

## Choosing Parameters

### Rule of Thumb for MinPts

**General Guideline**: 
- MinPts ≥ dimensions + 1
- For 2D data: MinPts ≥ 3
- Common values: 4, 5, 10

**Practical**: 
- Start with MinPts = 5
- Increase for noisy data
- Decrease for sparse data

### Finding Optimal Epsilon: K-Distance Graph

Plot the distance to the k-th nearest neighbor (k = MinPts) for all points. The "elbow" in the graph suggests a good ε value.

\`\`\`python
from sklearn.neighbors import NearestNeighbors

# Calculate k-distances (k = min_samples)
min_samples = 5
neighbors = NearestNeighbors (n_neighbors=min_samples)
neighbors.fit(X_moons)
distances, indices = neighbors.kneighbors(X_moons)

# Sort distances to k-th nearest neighbor
k_distances = distances[:, -1]
k_distances_sorted = np.sort (k_distances)

# Plot k-distance graph
plt.figure (figsize=(10, 6))
plt.plot (k_distances_sorted, linewidth=2)
plt.xlabel('Points sorted by distance', fontsize=12)
plt.ylabel (f'{min_samples}-th Nearest Neighbor Distance', fontsize=12)
plt.title (f'K-Distance Graph (k={min_samples})', fontsize=14)
plt.grid(True, alpha=0.3)

# Look for the "elbow" - suggests good epsilon
plt.axhline (y=0.25, color='r', linestyle='--', linewidth=2, 
           label='Suggested eps ≈ 0.25 (elbow point)')
plt.legend()
plt.tight_layout()
plt.show()

print("The 'elbow' in the curve suggests a good epsilon value")
print("Points after the elbow are likely outliers")
\`\`\`

### Automated Parameter Selection

\`\`\`python
from sklearn.metrics import silhouette_score

# Grid search for best parameters
eps_range = np.arange(0.1, 0.6, 0.05)
min_samples_range = [3, 5, 7, 10]

best_score = -1
best_params = {}

for eps in eps_range:
    for min_samples in min_samples_range:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_moons)
        
        # Skip if only one cluster or all noise
        n_clusters = len (set (labels)) - (1 if -1 in labels else 0)
        if n_clusters < 2:
            continue
        
        # Calculate silhouette score (only for non-noise points)
        mask = labels != -1
        if sum (mask) > min_samples:
            score = silhouette_score(X_moons[mask], labels[mask])
            
            if score > best_score:
                best_score = score
                best_params = {'eps': eps, 'min_samples': min_samples}

print(f"Best parameters: {best_params}")
print(f"Best silhouette score: {best_score:.3f}")

# Apply best parameters
dbscan_best = DBSCAN(**best_params)
labels_best = dbscan_best.fit_predict(X_moons)

plt.figure (figsize=(10, 6))
for label in set (labels_best):
    mask = labels_best == label
    if label == -1:
        plt.scatter(X_moons[mask, 0], X_moons[mask, 1], 
                   c='black', marker='x', s=50, label='Noise')
    else:
        plt.scatter(X_moons[mask, 0], X_moons[mask, 1], s=50, label=f'Cluster {label}')

plt.title (f"DBSCAN with Optimized Parameters\\neps={best_params['eps']:.2f}, min_samples={best_params['min_samples']}")
plt.legend()
plt.show()
\`\`\`

## DBSCAN vs K-Means

Let\'s see why DBSCAN excels at non-spherical clusters:

\`\`\`python
from sklearn.cluster import KMeans

# Generate challenging datasets
datasets = [
    make_moons (n_samples=300, noise=0.05, random_state=42)[0],
    make_blobs (n_samples=300, centers=3, random_state=42)[0],
]

dataset_names = ['Moons (Non-spherical)', 'Blobs (Spherical)']

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

for row, (X, name) in enumerate (zip (datasets, dataset_names)):
    # Standardize
    X_scaled = StandardScaler().fit_transform(X)
    
    # Original data
    axes[row, 0].scatter(X_scaled[:, 0], X_scaled[:, 1], alpha=0.6, s=30)
    axes[row, 0].set_title (f'{name}\\n(Original Data)')
    
    # K-Means
    kmeans = KMeans (n_clusters=2 if row == 0 else 3, random_state=42)
    labels_km = kmeans.fit_predict(X_scaled)
    axes[row, 1].scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_km, cmap='viridis', alpha=0.6, s=30)
    axes[row, 1].scatter (kmeans.cluster_centers_[:, 0], 
                        kmeans.cluster_centers_[:, 1],
                        c='red', marker='X', s=200, edgecolors='black', linewidths=2)
    axes[row, 1].set_title('K-Means Clustering')
    
    # DBSCAN
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    labels_db = dbscan.fit_predict(X_scaled)
    for label in set (labels_db):
        mask = labels_db == label
        if label == -1:
            axes[row, 2].scatter(X_scaled[mask, 0], X_scaled[mask, 1], 
                               c='black', marker='x', s=30, alpha=0.6)
        else:
            axes[row, 2].scatter(X_scaled[mask, 0], X_scaled[mask, 1], 
                               cmap='viridis', alpha=0.6, s=30)
    axes[row, 2].set_title('DBSCAN Clustering')

plt.tight_layout()
plt.show()

print("Observations:")
print("- K-Means struggles with non-spherical clusters (moons)")
print("- DBSCAN handles both shapes well")
print("- DBSCAN identifies noise, K-Means assigns everything to a cluster")
\`\`\`

## Advantages of DBSCAN

✅ **Arbitrary Shapes**: Not limited to spherical clusters  
✅ **No K Required**: Number of clusters determined automatically  
✅ **Noise Detection**: Identifies outliers as noise  
✅ **Flexible Density**: Works with varying density (to some extent)  
✅ **Single Pass**: Doesn't require multiple iterations  

## Limitations and Disadvantages

❌ **Parameter Sensitivity**: Results highly dependent on ε and MinPts  
❌ **Varying Density**: Struggles with clusters of very different densities  
❌ **High Dimensions**: "Curse of dimensionality" - distances become less meaningful  
❌ **Border Points**: Assignment can be ambiguous  

### Problem: Varying Density Clusters

\`\`\`python
# Generate data with different densities
np.random.seed(42)
X_dense = np.random.randn(100, 2) * 0.3
X_sparse = np.random.randn(100, 2) * 1.5 + np.array([5, 5])
X_varying = np.vstack([X_dense, X_sparse])

# Try DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X_varying)

plt.figure (figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_varying[:, 0], X_varying[:, 1], alpha=0.6)
plt.title('Data with Varying Density')

plt.subplot(1, 2, 2)
for label in set (labels):
    mask = labels == label
    if label == -1:
        plt.scatter(X_varying[mask, 0], X_varying[mask, 1], 
                   c='black', marker='x', s=30, alpha=0.6, label='Noise')
    else:
        plt.scatter(X_varying[mask, 0], X_varying[mask, 1], 
                   s=30, alpha=0.6, label=f'Cluster {label}')

plt.title('DBSCAN Struggles with Varying Density')
plt.legend()
plt.tight_layout()
plt.show()

print("Single epsilon can't handle both dense and sparse regions well")
print("Consider HDBSCAN for varying density")
\`\`\`

## Real-World Applications

### Application 1: Geospatial Clustering

Finding hotspots in geographic data (crime, taxi pickups, etc.)

\`\`\`python
# Simulated GPS coordinates
np.random.seed(42)

# Three hotspots
hotspot1 = np.random.randn(50, 2) * 0.005 + np.array([40.7580, -73.9855])  # Times Square
hotspot2 = np.random.randn(40, 2) * 0.003 + np.array([40.7614, -73.9776])  # Central Park
hotspot3 = np.random.randn(30, 2) * 0.004 + np.array([40.7489, -73.9680])  # Empire State

# Some random noise (isolated incidents)
noise = np.random.randn(20, 2) * 0.01 + np.array([40.75, -73.98])

X_geo = np.vstack([hotspot1, hotspot2, hotspot3, noise])

# DBSCAN to find hotspots
dbscan = DBSCAN(eps=0.008, min_samples=5)
labels = dbscan.fit_predict(X_geo)

n_clusters = len (set (labels)) - (1 if -1 in labels else 0)
n_noise = list (labels).count(-1)

plt.figure (figsize=(10, 8))
for label in set (labels):
    mask = labels == label
    if label == -1:
        plt.scatter(X_geo[mask, 1], X_geo[mask, 0], 
                   c='gray', marker='.', s=30, alpha=0.3, label='Isolated')
    else:
        plt.scatter(X_geo[mask, 1], X_geo[mask, 0], 
                   s=50, alpha=0.7, label=f'Hotspot {label+1}')

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title (f'Crime Hotspot Detection\\n{n_clusters} hotspots, {n_noise} isolated incidents')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("DBSCAN identifies concentrated areas while filtering isolated incidents")
\`\`\`

### Application 2: Anomaly Detection in Network Traffic

\`\`\`python
# Simulated network traffic data
np.random.seed(42)

# Normal traffic
normal_traffic = np.random.randn(200, 2) * 0.5 + np.array([2, 3])

# Anomalous traffic (DDoS attack, port scanning, etc.)
anomalies = np.random.randn(20, 2) * 0.1 + np.array([6, 8])
anomalies = np.vstack([anomalies, np.random.randn(15, 2) * 0.1 + np.array([7, 2])])

X_traffic = np.vstack([normal_traffic, anomalies])

# Use DBSCAN for anomaly detection
dbscan = DBSCAN(eps=0.5, min_samples=10)
labels = dbscan.fit_predict(X_traffic)

# Noise points are potential anomalies
n_anomalies = list (labels).count(-1)

plt.figure (figsize=(12, 6))

# Plot normal vs anomalous
for label in set (labels):
    mask = labels == label
    if label == -1:
        plt.scatter(X_traffic[mask, 0], X_traffic[mask, 1], 
                   c='red', marker='x', s=100, linewidths=2,
                   label=f'Anomalies ({n_anomalies})', alpha=0.8)
    else:
        plt.scatter(X_traffic[mask, 0], X_traffic[mask, 1], 
                   c='blue', s=30, alpha=0.5, label=f'Normal Traffic')

plt.xlabel('Packets per second')
plt.ylabel('Bytes per packet')
plt.title('Network Traffic Anomaly Detection with DBSCAN')
plt.legend()
plt.tight_layout()
plt.show()

print(f"Detected {n_anomalies} anomalous traffic patterns")
print("These could be attacks, errors, or unusual behavior")
\`\`\`

### Application 3: Image Segmentation

\`\`\`python
from sklearn.datasets import load_sample_image

# Load image
china = load_sample_image("china.jpg")
china_small = china[::8, ::8]  # Downsample for speed

# Convert to 2D array of pixels (features = RGB + position)
h, w, c = china_small.shape
X_pixels = china_small.reshape(-1, 3).astype (float)

# Add spatial information (position matters!)
positions = np.array([[i, j] for i in range (h) for j in range (w)])
X_features = np.hstack([X_pixels / 255.0, positions / np.array([h, w]) * 2])

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

# DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

# Visualize segmentation
segmented = np.zeros_like (china_small)
for label in set (labels):
    if label != -1:  # Ignore noise
        mask = (labels == label).reshape (h, w)
        color = np.random.randint(0, 255, 3)
        segmented[mask] = color

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].imshow (china_small)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow (segmented)
axes[1].set_title (f'DBSCAN Segmentation ({len (set (labels))-1} segments)')
axes[1].axis('off')

plt.tight_layout()
plt.show()
\`\`\`

## HDBSCAN: Hierarchical DBSCAN

An improved version that handles varying density better.

\`\`\`python
# Note: Requires hdbscan library (pip install hdbscan)
try:
    import hdbscan
    
    # HDBSCAN on varying density data
    hdb = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3)
    labels_hdb = hdb.fit_predict(X_varying)
    
    plt.figure (figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    # DBSCAN result
    for label in set (labels):
        mask = labels == label
        if label == -1:
            plt.scatter(X_varying[mask, 0], X_varying[mask, 1], 
                       c='black', marker='x', s=30)
        else:
            plt.scatter(X_varying[mask, 0], X_varying[mask, 1], s=30)
    plt.title('DBSCAN (struggles with varying density)')
    
    plt.subplot(1, 2, 2)
    # HDBSCAN result
    for label in set (labels_hdb):
        mask = labels_hdb == label
        if label == -1:
            plt.scatter(X_varying[mask, 0], X_varying[mask, 1], 
                       c='black', marker='x', s=30)
        else:
            plt.scatter(X_varying[mask, 0], X_varying[mask, 1], s=30)
    plt.title('HDBSCAN (handles varying density better)')
    
    plt.tight_layout()
    plt.show()
    
except ImportError:
    print("HDBSCAN not installed. Install with: pip install hdbscan")
\`\`\`

## Best Practices

1. **Scale Features**: Distance-based, so scaling is crucial
2. **Start with k-distance graph**: Find good epsilon
3. **Try MinPts = 2×dimensions**: Good starting point
4. **Visualize Results**: Always check if clusters make sense
5. **Consider Domain Knowledge**: Use it to validate parameters
6. **Check Noise Points**: Do they really look like outliers?
7. **Try HDBSCAN**: For varying density or when parameters are hard to tune

## Summary

DBSCAN is ideal for:
- **Arbitrary shaped clusters**: Not limited to spherical
- **Noise detection**: Identifies outliers
- **Unknown number of clusters**: K not required
- **Geospatial data**: Natural fit for location-based clustering
- **Anomaly detection**: Noise points are potential anomalies

Avoid DBSCAN when:
- Clusters have very different densities
- High-dimensional data (consider dimensionality reduction first)
- Need precise control over number of clusters
- Parameters are hard to tune for your data

**Next**: For dimensionality reduction and visualization, we'll explore PCA in the next section.
`,
};
