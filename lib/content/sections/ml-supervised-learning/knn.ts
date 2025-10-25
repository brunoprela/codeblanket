/**
 * k-Nearest Neighbors Section
 */

export const knnSection = {
  id: 'knn',
  title: 'k-Nearest Neighbors (kNN)',
  content: `# k-Nearest Neighbors (kNN)

## Introduction

k-Nearest Neighbors (kNN) is one of the simplest yet surprisingly effective machine learning algorithms. It\'s a **non-parametric, instance-based, lazy learning** algorithm that makes predictions based on the k most similar training examples.

**Key Characteristics:**
- **Non-parametric**: Makes no assumptions about data distribution
- **Instance-based**: Stores all training data (no training phase)
- **Lazy learning**: Computation happens at prediction time
- **Versatile**: Works for both classification and regression

**Applications:**
- Recommender systems (find similar users/items)
- Pattern recognition
- Medical diagnosis
- Credit rating
- Anomaly detection

## How kNN Works

### Classification

1. Store all training data
2. For a new point:
   - Calculate distance to all training points
   - Find k nearest neighbors
   - Take majority vote of their classes
   - Return the most common class

### Regression

Same as classification, but return the **average** (or weighted average) of the k nearest neighbors' values.

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Generate data
np.random.seed(42)
X, y = make_classification (n_samples=200, n_features=2, n_redundant=0,
                           n_informative=2, n_clusters_per_class=1, 
                           random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Visualize data
plt.figure (figsize=(10, 6))
plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], 
           c='blue', label='Class 0', s=50, alpha=0.7, edgecolors='k')
plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], 
           c='red', label='Class 1', s=50, alpha=0.7, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Training Data')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Train kNN classifier
k = 5
knn = KNeighborsClassifier (n_neighbors=k)
knn.fit(X_train, y_train)

# Make prediction for a new point
new_point = np.array([[0, 0]])
prediction = knn.predict (new_point)
probabilities = knn.predict_proba (new_point)

print(f"New point: {new_point[0]}")
print(f"Predicted class: {prediction[0]}")
print(f"Probabilities: Class 0: {probabilities[0][0]:.3f}, Class 1: {probabilities[0][1]:.3f}")

# Find nearest neighbors
distances, indices = knn.kneighbors (new_point, n_neighbors=k)
print(f"\\nNearest {k} neighbors:")
for i, (dist, idx) in enumerate (zip (distances[0], indices[0]), 1):
    print(f"  {i}. Distance: {dist:.3f}, Class: {y_train[idx]}")
\`\`\`

## Distance Metrics

The choice of distance metric is crucial for kNN performance.

### Euclidean Distance (L2)
\\[ d(\\mathbf{x}, \\mathbf{y}) = \\sqrt{\\sum_{i=1}^{n}(x_i - y_i)^2} \\]

Most common, works well for continuous features.

### Manhattan Distance (L1)
\\[ d(\\mathbf{x}, \\mathbf{y}) = \\sum_{i=1}^{n}|x_i - y_i| \\]

Better for high-dimensional data, less sensitive to outliers.

### Minkowski Distance (generalization)
\\[ d(\\mathbf{x}, \\mathbf{y}) = \\left(\\sum_{i=1}^{n}|x_i - y_i|^p\\right)^{1/p} \\]

- p=1: Manhattan
- p=2: Euclidean
- p=∞: Chebyshev (max difference)

### Cosine Similarity
\\[ \\text{similarity} = \\frac{\\mathbf{x} \\cdot \\mathbf{y}}{||\\mathbf{x}|| \\cdot ||\\mathbf{y}||} \\]

Good for text data and when direction matters more than magnitude.

\`\`\`python
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_similarity

# Compare distance metrics
point1 = np.array([[1, 2, 3]])
point2 = np.array([[4, 5, 6]])

euclidean = euclidean_distances (point1, point2)[0][0]
manhattan = manhattan_distances (point1, point2)[0][0]
cosine_sim = cosine_similarity (point1, point2)[0][0]

print("Distance Metrics:")
print(f"Euclidean: {euclidean:.3f}")
print(f"Manhattan: {manhattan:.3f}")
print(f"Cosine Similarity: {cosine_sim:.3f}")
print(f"Cosine Distance: {1 - cosine_sim:.3f}")

# Compare kNN with different metrics
metrics = ['euclidean', 'manhattan', 'chebyshev']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, metric in enumerate (metrics):
    knn_metric = KNeighborsClassifier (n_neighbors=5, metric=metric)
    knn_metric.fit(X_train, y_train)
    
    # Create mesh
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid (np.arange (x_min, x_max, h),
                         np.arange (y_min, y_max, h))
    
    Z = knn_metric.predict (np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape (xx.shape)
    
    axes[idx].contourf (xx, yy, Z, alpha=0.3, cmap='RdBu')
    axes[idx].scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], 
                     c='blue', s=50, alpha=0.7, edgecolors='k')
    axes[idx].scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], 
                     c='red', s=50, alpha=0.7, edgecolors='k')
    axes[idx].set_title (f'{metric.capitalize()} Distance')
    axes[idx].set_xlabel('Feature 1')
    axes[idx].set_ylabel('Feature 2')

plt.tight_layout()
plt.show()
\`\`\`

## Choosing k

The value of k significantly affects model performance.

**Small k (e.g., k=1)**:
- **Pros**: Captures fine details, low bias
- **Cons**: Sensitive to noise, high variance, overfitting

**Large k (e.g., k=n/2)**:
- **Pros**: Smooth boundaries, low variance
- **Cons**: May miss local patterns, high bias, underfitting

**Optimal k**: Use cross-validation!

\`\`\`python
from sklearn.model_selection import cross_val_score

# Try different k values
k_values = range(1, 51)
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier (n_neighbors=k)
    scores = cross_val_score (knn, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append (scores.mean())

# Plot results
plt.figure (figsize=(12, 6))
plt.plot (k_values, cv_scores, 'o-', linewidth=2)
plt.xlabel('k (number of neighbors)')
plt.ylabel('Cross-Validation Accuracy')
plt.title('kNN: Choosing Optimal k')
plt.grid(True, alpha=0.3)
plt.axvline (x=k_values[np.argmax (cv_scores)], color='r', 
           linestyle='--', label=f'Optimal k={k_values[np.argmax (cv_scores)]}')
plt.legend()
plt.show()

optimal_k = k_values[np.argmax (cv_scores)]
print(f"Optimal k: {optimal_k}")
print(f"Best CV Accuracy: {max (cv_scores):.4f}")
\`\`\`

## Weighted kNN

Give more influence to closer neighbors using distance weighting:

\\[ \\text{weight}_i = \\frac{1}{d_i + \\epsilon} \\]

\`\`\`python
# Compare uniform vs weighted
knn_uniform = KNeighborsClassifier (n_neighbors=15, weights='uniform')
knn_weighted = KNeighborsClassifier (n_neighbors=15, weights='distance')

knn_uniform.fit(X_train, y_train)
knn_weighted.fit(X_train, y_train)

score_uniform = knn_uniform.score(X_test, y_test)
score_weighted = knn_weighted.score(X_test, y_test)

print(f"Uniform weights accuracy: {score_uniform:.4f}")
print(f"Distance-weighted accuracy: {score_weighted:.4f}")
\`\`\`

## Feature Scaling is Critical!

kNN is **extremely sensitive** to feature scales because it uses distances.

\`\`\`python
# Demonstrate impact of scaling
X_unscaled = np.random.rand(100, 2)
X_unscaled[:, 0] *= 1000  # First feature in [0, 1000]
X_unscaled[:, 1] *= 1     # Second feature in [0, 1]
y_unscaled = (X_unscaled[:, 0] > 500).astype (int)

# Without scaling
knn_unscaled = KNeighborsClassifier (n_neighbors=5)
knn_unscaled.fit(X_unscaled, y_unscaled)

# With scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_unscaled)
knn_scaled = KNeighborsClassifier (n_neighbors=5)
knn_scaled.fit(X_scaled, y_unscaled)

print("\\nImpact of Feature Scaling:")
print(f"Without scaling - feature 1 dominates distances")
print(f"With scaling - both features contribute fairly")

# Calculate distances for a point
point = np.array([[600, 0.5]])
distances_unscaled = np.sqrt (np.sum((X_unscaled - point)**2, axis=1))
point_scaled = scaler.transform (point)
distances_scaled = np.sqrt (np.sum((X_scaled - point_scaled)**2, axis=1))

print(f"\\nDistance range without scaling: [{distances_unscaled.min():.1f}, {distances_unscaled.max():.1f}]")
print(f"Distance range with scaling: [{distances_scaled.min():.3f}, {distances_scaled.max():.3f}]")
\`\`\`

## Curse of Dimensionality

kNN suffers in high dimensions because:
- Distances become less meaningful
- All points become roughly equidistant
- Need exponentially more data

\`\`\`python
# Demonstrate curse of dimensionality
from sklearn.neighbors import NearestNeighbors

n_samples = 100
dimensions = [2, 10, 50, 100]

for d in dimensions:
    X_high = np.random.randn (n_samples, d)
    nn = NearestNeighbors (n_neighbors=2)
    nn.fit(X_high)
    
    distances, _ = nn.kneighbors(X_high)
    nearest_dist = distances[:, 1]  # Distance to nearest neighbor
    
    print(f"\\nDimensions: {d}")
    print(f"  Mean nearest neighbor distance: {nearest_dist.mean():.3f}")
    print(f"  Std of distances: {nearest_dist.std():.3f}")
    print(f"  Ratio (std/mean): {nearest_dist.std()/nearest_dist.mean():.3f}")

print("\\nAs dimensionality increases, distances become less discriminative!")
\`\`\`

## Computational Complexity

**Training**: O(1) - just store data
**Prediction**: O(n·d) - calculate distance to all n points with d dimensions

For large datasets, use:
- KD-Tree (good for low dimensions, d < 20)
- Ball Tree (better for high dimensions)
- Approximate methods (LSH, Annoy)

\`\`\`python
from sklearn.neighbors import KNeighborsClassifier
import time

# Compare algorithms
X_large = np.random.randn(10000, 10)
y_large = np.random.randint(0, 2, 10000)
X_test_large = np.random.randn(100, 10)

algorithms = ['brute', 'kd_tree', 'ball_tree']

for algo in algorithms:
    knn = KNeighborsClassifier (n_neighbors=10, algorithm=algo)
    
    # Train (actually just stores data)
    start = time.time()
    knn.fit(X_large, y_large)
    train_time = time.time() - start
    
    # Predict
    start = time.time()
    predictions = knn.predict(X_test_large)
    predict_time = time.time() - start
    
    print(f"{algo:10s}: Train={train_time:.4f}s, Predict={predict_time:.4f}s")
\`\`\`

## Real-World Example: Wine Classification

\`\`\`python
from sklearn.datasets import load_wine
from sklearn.metrics import classification_report, confusion_matrix

# Load wine dataset
wine = load_wine()
X_wine = wine.data
y_wine = wine.target

X_train, X_test, y_train, y_test = train_test_split(
    X_wine, y_wine, test_size=0.3, random_state=42, stratify=y_wine
)

# Scale features (CRITICAL for kNN!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Find optimal k
k_range = range(1, 31)
cv_scores = []

for k in k_range:
    knn = KNeighborsClassifier (n_neighbors=k)
    scores = cross_val_score (knn, X_train_scaled, y_train, cv=5)
    cv_scores.append (scores.mean())

optimal_k = k_range[np.argmax (cv_scores)]
print(f"Optimal k: {optimal_k}")

# Train final model
knn_final = KNeighborsClassifier (n_neighbors=optimal_k)
knn_final.fit(X_train_scaled, y_train)

# Evaluate
y_pred = knn_final.predict(X_test_scaled)
accuracy = np.mean (y_pred == y_test)

print(f"\\nTest Accuracy: {accuracy:.4f}")
print("\\nClassification Report:")
print(classification_report (y_test, y_pred, target_names=wine.target_names))
\`\`\`

## Advantages and Limitations

**Advantages:**
- Simple and intuitive
- No training phase
- Naturally handles multi-class
- No assumptions about data distribution
- Can capture complex decision boundaries
- Good for small datasets

**Limitations:**
- Slow prediction for large datasets
- Memory intensive (stores all data)
- Sensitive to irrelevant features
- Requires feature scaling
- Curse of dimensionality
- No model interpretation
- Poor with imbalanced classes

## When to Use kNN

**Good scenarios:**
- Small to medium datasets
- Low to moderate dimensions
- Non-linear decision boundaries
- Need probabilistic predictions
- Anomaly detection (find unusual points)

**Avoid when:**
- Very large datasets (slow prediction)
- High dimensions (curse of dimensionality)
- Need interpretable model
- Real-time predictions required
- Storage is constrained

## Summary

k-Nearest Neighbors is a simple yet powerful non-parametric algorithm:
- Makes predictions based on k nearest training examples
- Choice of k and distance metric matters
- **Feature scaling is critical**
- Suffers from curse of dimensionality
- No training but slow prediction
- Works for classification and regression

**Key Insight:** kNN assumes similar inputs have similar outputs!

Next: Naive Bayes for probabilistic classification!
`,
  codeExample: `# Complete kNN Implementation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, roc_curve, auc)

# Load breast cancer dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

print("="*70)
print("k-NEAREST NEIGHBORS: BREAST CANCER CLASSIFICATION")
print("="*70)
print(f"\\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Classes: {cancer.target_names}")
print(f"Class distribution: {np.bincount (y)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling (ESSENTIAL for kNN!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\\nTrain: {len(X_train)}, Test: {len(X_test)}")

# Find optimal hyperparameters
param_grid = {
    'n_neighbors': range(1, 31),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

grid_search = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)

print(f"\\nBest parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# Train final model
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test_scaled)
y_prob = best_knn.predict_proba(X_test_scaled)[:, 1]

# Evaluate
accuracy = accuracy_score (y_test, y_pred)
print(f"\\nTest Accuracy: {accuracy:.4f}")
print("\\nClassification Report:")
print(classification_report (y_test, y_pred, target_names=cancer.target_names))

# Visualizations
fig = plt.figure (figsize=(15, 10))

# 1. k vs Accuracy
ax1 = plt.subplot(2, 3, 1)
results = grid_search.cv_results_
k_values = param_grid['n_neighbors']
mean_scores = [results['mean_test_score'][i] for i in range(0, len (results['mean_test_score']), 4)]
plt.plot (k_values, mean_scores, 'o-', linewidth=2)
plt.xlabel('k (number of neighbors)')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Optimal k Selection')
plt.grid(True, alpha=0.3)

# 2. Confusion Matrix
ax2 = plt.subplot(2, 3, 2)
cm = confusion_matrix (y_test, y_pred)
import seaborn as sns
sns.heatmap (cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True')
plt.xlabel('Predicted')

# 3. ROC Curve
ax3 = plt.subplot(2, 3, 3)
fpr, tpr, _ = roc_curve (y_test, y_prob)
roc_auc = auc (fpr, tpr)
plt.plot (fpr, tpr, linewidth=2, label=f'AUC = {roc_auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"""
kNN achieved {accuracy:.1%} accuracy on breast cancer classification.
Best configuration: k={grid_search.best_params_['n_neighbors']}, 
                    weights={grid_search.best_params_['weights']}

Key insights:
- Feature scaling improved accuracy significantly
- Distance weighting helped with decision boundaries  
- Model makes interpretable predictions based on similar cases
""")
`,
};
