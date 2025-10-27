/**
 * Section: Unsupervised Learning Overview
 * Module: Classical Machine Learning - Unsupervised Learning
 *
 * Comprehensive introduction to unsupervised learning, its types, applications, and evaluation
 */

export const unsupervisedLearningOverview = {
  id: 'unsupervised-learning-overview',
  title: 'Unsupervised Learning Overview',
  content: `
# Unsupervised Learning Overview

## Introduction

Unsupervised learning is a fundamental branch of machine learning where algorithms discover hidden patterns and structures in data **without labeled outputs**. Unlike supervised learning, where we train models on input-output pairs, unsupervised learning works with input data alone, making it ideal for exploration, dimensionality reduction, and finding natural groupings.

**Key Distinction**:
- **Supervised Learning**: Learn f: X → Y from labeled examples (X, Y)
- **Unsupervised Learning**: Learn structure from X alone (no Y labels)

## What is Unsupervised Learning?

Unsupervised learning algorithms analyze and cluster unlabeled datasets to discover hidden patterns without human intervention. The goal is to model the underlying structure or distribution in the data.

**Core Objective**: Find structure, patterns, or relationships in data without explicit guidance

### Why Unsupervised Learning?

1. **Labeled Data is Expensive**
   - Manual labeling requires time, money, and expertise
   - Many domains have abundant unlabeled data
   - Unsupervised methods can extract value from raw data

2. **Discover Hidden Structure**
   - Patterns humans might miss
   - Natural groupings in data
   - Underlying factors or dimensions

3. **Data Exploration**
   - Understand data before modeling
   - Feature extraction
   - Preprocessing for supervised learning

4. **Real-World Necessity**
   - Most data in the world is unlabeled
   - Continuous data streams
   - Discovery-oriented problems

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate unlabeled data with natural clusters
X, true_labels = make_blobs(
    n_samples=300,
    centers=3,
    n_features=2,
    random_state=42
)

# Visualize: We have data but no labels
plt.figure (figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.6)
plt.title('Unlabeled Data (What We See)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=true_labels, alpha=0.6, cmap='viridis')
plt.title('True Structure (Unknown to Algorithm)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar (label='True Cluster')

plt.tight_layout()
plt.show()

# Unsupervised learning tries to discover the structure on the right
# using only the information on the left!
\`\`\`

## Types of Unsupervised Learning

### 1. Clustering

**Goal**: Group similar data points together

**Applications**:
- Customer segmentation
- Image segmentation
- Document organization
- Gene sequence analysis

**Common Algorithms**:
- K-Means
- Hierarchical Clustering
- DBSCAN
- Gaussian Mixture Models (GMM)

\`\`\`python
from sklearn.cluster import KMeans

# Example: Customer segmentation
# Features: [Annual Income, Spending Score]
X_customers = np.array([
    [15, 39], [15, 81], [16, 6], [17, 77], [18, 40],
    [19, 76], [19, 94], [20, 3], [20, 72], [20, 14]
])

# Apply K-Means clustering
kmeans = KMeans (n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_customers)

print("Customer Segments:", clusters)
print("Cluster Centers:", kmeans.cluster_centers_)

# Visualize segments
plt.scatter(X_customers[:, 0], X_customers[:, 1],
            c=clusters, cmap='viridis', s=100, alpha=0.6)
plt.scatter (kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            c='red', marker='X', s=200, label='Centroids')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segmentation')
plt.legend()
plt.show()
\`\`\`

### 2. Dimensionality Reduction

**Goal**: Reduce number of features while preserving important information

**Applications**:
- Data visualization (3D → 2D)
- Feature extraction
- Noise reduction
- Compression

**Common Algorithms**:
- Principal Component Analysis (PCA)
- t-SNE
- UMAP
- Autoencoders

\`\`\`python
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

# Load high-dimensional data (64 features)
digits = load_digits()
X_digits = digits.data  # 1797 samples, 64 features (8x8 images)

print(f"Original dimensions: {X_digits.shape}")

# Reduce to 2 dimensions for visualization
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_digits)

print(f"Reduced dimensions: {X_reduced.shape}")
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")

# Visualize reduced data
plt.figure (figsize=(10, 8))
scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1],
                     c=digits.target, cmap='tab10', alpha=0.6)
plt.colorbar (scatter, label='Digit')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('64D Digit Data Reduced to 2D with PCA')
plt.show()
\`\`\`

### 3. Anomaly Detection

**Goal**: Identify unusual or rare data points

**Applications**:
- Fraud detection
- Network intrusion detection
- Manufacturing defects
- Health monitoring

**Common Algorithms**:
- Isolation Forest
- One-Class SVM
- Local Outlier Factor (LOF)
- Autoencoders

\`\`\`python
from sklearn.ensemble import IsolationForest

# Normal data with some outliers
np.random.seed(42)
X_normal = np.random.randn(100, 2) * 0.5
X_outliers = np.random.randn(10, 2) * 2 + np.array([3, 3])
X_mixed = np.vstack([X_normal, X_outliers])

# Detect anomalies
iso_forest = IsolationForest (contamination=0.1, random_state=42)
predictions = iso_forest.fit_predict(X_mixed)

# Visualize
plt.figure (figsize=(10, 6))
plt.scatter(X_mixed[predictions == 1, 0],
            X_mixed[predictions == 1, 1],
            c='blue', label='Normal', alpha=0.6)
plt.scatter(X_mixed[predictions == -1, 0],
            X_mixed[predictions == -1, 1],
            c='red', label='Anomaly', alpha=0.8, marker='x', s=100)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Anomaly Detection with Isolation Forest')
plt.legend()
plt.show()
\`\`\`

### 4. Association Rule Learning

**Goal**: Discover interesting relationships between variables

**Applications**:
- Market basket analysis
- Recommendation systems
- Web usage mining

**Common Algorithms**:
- Apriori
- FP-Growth
- ECLAT

\`\`\`python
# Example: Market basket analysis
# Transactions: What items are bought together?
transactions = [
    ['milk', 'bread', 'butter'],
    ['milk', 'bread'],
    ['milk', 'butter'],
    ['bread', 'butter'],
    ['milk', 'bread', 'butter', 'eggs']
]

# Simple association rule: milk → bread
milk_transactions = [t for t in transactions if 'milk' in t]
milk_and_bread = [t for t in milk_transactions if 'bread' in t]

support_milk = len (milk_transactions) / len (transactions)
confidence_milk_bread = len (milk_and_bread) / len (milk_transactions)

print(f"Support (milk): {support_milk:.2%}")
print(f"Confidence (milk → bread): {confidence_milk_bread:.2%}")
print("If customer buys milk, they buy bread " +
      f"{confidence_milk_bread:.0%} of the time")
\`\`\`

## Applications and Use Cases

### 1. Customer Segmentation

**Problem**: Understand different customer groups for targeted marketing

\`\`\`python
# E-commerce customer data
customer_data = {
    'recency': [1, 5, 10, 2, 50, 45, 3, 30, 25, 7],  # days since last purchase
    'frequency': [50, 30, 5, 40, 2, 3, 45, 10, 8, 35],  # number of purchases
    'monetary': [5000, 3000, 100, 4500, 50, 80, 4800, 500, 300, 3500]  # total spent
}

import pandas as pd
df_customers = pd.DataFrame (customer_data)

# Clustering reveals customer segments:
# - High-value frequent buyers
# - At-risk customers (high recency, low recent activity)
# - New potential high-value customers
\`\`\`

### 2. Image Compression

**Problem**: Reduce storage/bandwidth requirements

\`\`\`python
from sklearn.cluster import MiniBatchKMeans

# Load image
from skimage import data
image = data.astronaut()  # RGB image
h, w, c = image.shape

# Reshape to (n_pixels, 3)
X_pixels = image.reshape(-1, 3)

# K-means to find dominant colors
n_colors = 16
kmeans = MiniBatchKMeans (n_clusters=n_colors, random_state=42)
labels = kmeans.fit_predict(X_pixels)
centers = kmeans.cluster_centers_

# Reconstruct image with reduced colors
compressed = centers[labels].reshape (h, w, c)

print(f"Original: {X_pixels.shape[0]} unique colors")
print(f"Compressed: {n_colors} colors")
print(f"Compression ratio: {X_pixels.shape[0] / n_colors:.0f}x")
\`\`\`

### 3. Exploratory Data Analysis

**Problem**: Understand structure before modeling

\`\`\`python
# Discover natural groupings in data before supervised learning
from sklearn.datasets import load_wine

wine = load_wine()
X_wine = wine.data

# PCA for visualization
pca = PCA(n_components=2)
X_wine_2d = pca.fit_transform(X_wine)

# Reveals 3 distinct groups (matching 3 wine varieties)
# This validates our supervised learning approach
\`\`\`

## Evaluation Challenges

Unlike supervised learning, evaluating unsupervised learning is challenging because **there are no true labels** to compare against.

### Evaluation Strategies

**1. Internal Validation Metrics** (no ground truth needed)

\`\`\`python
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Silhouette Score: [-1, 1], higher is better
# Measures how similar objects are to their own cluster vs other clusters
silhouette = silhouette_score(X, clusters)
print(f"Silhouette Score: {silhouette:.3f}")

# Davies-Bouldin Index: [0, ∞), lower is better
# Ratio of within-cluster to between-cluster distances
db_index = davies_bouldin_score(X, clusters)
print(f"Davies-Bouldin Index: {db_index:.3f}")
\`\`\`

**2. External Validation Metrics** (when labels available for evaluation)

\`\`\`python
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Adjusted Rand Index: [-1, 1], 1 = perfect match
ari = adjusted_rand_score (true_labels, clusters)
print(f"Adjusted Rand Index: {ari:.3f}")

# Normalized Mutual Information: [0, 1], 1 = perfect agreement
nmi = normalized_mutual_info_score (true_labels, clusters)
print(f"Normalized Mutual Information: {nmi:.3f}")
\`\`\`

**3. Visual Inspection**

\`\`\`python
# Often the best evaluation is visual
# Do the clusters make sense?
# Do they align with domain knowledge?
\`\`\`

**4. Downstream Task Performance**

\`\`\`python
# Use unsupervised learning as preprocessing
# Evaluate on final supervised task
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Original features
scores_original = cross_val_score(LogisticRegression(), X, y, cv=5)

# PCA-reduced features
X_reduced = PCA(n_components=10).fit_transform(X)
scores_pca = cross_val_score(LogisticRegression(), X_reduced, y, cv=5)

print(f"Original: {scores_original.mean():.3f}")
print(f"After PCA: {scores_pca.mean():.3f}")
\`\`\`

## When to Use Unsupervised Learning

### ✅ Use Unsupervised Learning When:

1. **No labels available**
   - Can't afford labeling costs
   - Labeling is subjective
   - Continuous unlabeled data stream

2. **Exploratory analysis**
   - New dataset, unknown structure
   - Hypothesis generation
   - Feature understanding

3. **Preprocessing for supervised learning**
   - Dimensionality reduction
   - Feature extraction
   - Outlier removal

4. **Discovery-oriented problems**
   - Find customer segments
   - Detect anomalies
   - Discover patterns

### ❌ Avoid When:

1. **Clear prediction task with labels**
   - Use supervised learning instead
   - More accurate for prediction

2. **Need specific outputs**
   - Clustering won't give exact categories
   - PCA won't identify specific features

3. **Small datasets**
   - Patterns may not emerge
   - Statistical significance issues

## Comparison: Supervised vs Unsupervised Learning

\`\`\`python
import pandas as pd

comparison = pd.DataFrame({
    'Aspect': ['Training Data', 'Goal', 'Output', 'Accuracy',
               'Complexity', 'Use Cases'],
    'Supervised': ['Labeled (X, y)', 'Predict y from X',
                   'Specific predictions', 'Measurable',
                   'Simpler', 'Classification, Regression'],
    'Unsupervised': ['Unlabeled (X only)', 'Find structure in X',
                     'Patterns, groups', 'Harder to measure',
                     'More complex', 'Clustering, Dimensionality reduction']
})

print(comparison.to_string (index=False))
\`\`\`

## Python Libraries for Unsupervised Learning

\`\`\`python
# scikit-learn: Primary library
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, NMF, FastICA
from sklearn.ensemble import IsolationForest

# scipy: Hierarchical clustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Specialized libraries
# - umap-learn: UMAP for dimensionality reduction
# - hdbscan: Hierarchical DBSCAN
# - mlxtend: Association rule mining

# Example: Complete workflow
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Create unsupervised pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),  # Keep 95% variance
    ('clustering', KMeans (n_clusters=3))
])

# Fit and predict
clusters = pipeline.fit_predict(X)
\`\`\`

## Best Practices

1. **Always scale/normalize features**
   - Distance-based algorithms sensitive to scale
   - Standardize or min-max scale

2. **Try multiple algorithms**
   - Different algorithms for different structures
   - K-means for spherical clusters
   - DBSCAN for arbitrary shapes

3. **Validate with domain knowledge**
   - Do clusters make sense?
   - Consult domain experts

4. **Use multiple evaluation metrics**
   - No single metric is perfect
   - Combine internal metrics with visual inspection

5. **Consider interpretability**
   - Can you explain the results?
   - Are patterns actionable?

6. **Iterate and refine**
   - Unsupervised learning is exploratory
   - Try different parameters and methods

## Summary

Unsupervised learning is a powerful tool for discovering hidden patterns in unlabeled data. The main types are:

- **Clustering**: Group similar data points
- **Dimensionality Reduction**: Reduce features while preserving information
- **Anomaly Detection**: Find unusual patterns
- **Association Rules**: Discover relationships between variables

While evaluation is more challenging than supervised learning, unsupervised methods are essential for:
- Data exploration
- Preprocessing
- Working with unlabeled data
- Discovery-oriented problems

In the following sections, we'll dive deep into specific unsupervised learning algorithms, starting with clustering methods.
`,
};
