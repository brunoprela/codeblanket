/**
 * Principal Component Analysis (PCA) Section
 */

export const principalcomponentanalysisSection = {
  id: 'principal-component-analysis',
  title: 'Principal Component Analysis (PCA)',
  content: `
# Principal Component Analysis (PCA)

## Introduction

**Principal Component Analysis (PCA)** is one of the most important dimensionality reduction techniques in machine learning. It uses linear algebra to find the directions of maximum variance in high-dimensional data, enabling:
- Data compression
- Visualization
- Noise reduction
- Feature extraction
- Speeding up learning algorithms

**Key Idea**: Transform correlated features into uncorrelated principal components, ordered by importance.

## The PCA Problem

Given data **X** (n samples × d features), find **k** orthogonal directions that capture maximum variance.

**Mathematically**:
1. Center the data: **X** = **X** - mean(**X**)
2. Find covariance matrix: **C** = **XᵀX** / n
3. Compute eigenvectors and eigenvalues of **C**
4. Principal components = eigenvectors sorted by eigenvalue (descending)

**Result**: Project data onto top k principal components, reducing dimensions from d to k.

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

print("=== PCA: Basic Example ===")

# Load iris dataset
iris = load_iris()
X = iris.data  # 150 samples, 4 features
y = iris.target

print(f"Original data shape: {X.shape}")
print(f"Features: {iris.feature_names}")
print()

# Step 1: Standardize data (mean=0, std=1 for each feature)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Step 1: Standardized data")
print(f"Mean: {X_scaled.mean(axis=0)}")
print(f"Std: {X_scaled.std(axis=0)}")
print()

# Step 2: Compute covariance matrix
cov_matrix = np.cov(X_scaled.T)

print("Step 2: Covariance matrix:")
print(cov_matrix)
print()

# Step 3: Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort by eigenvalue (descending)
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("Step 3: Eigenvalues (variance explained by each PC):")
print(eigenvalues)
print()

print("Eigenvectors (principal components):")
print(eigenvectors)
print()

# Step 4: Explained variance ratio
total_variance = eigenvalues.sum()
explained_variance_ratio = eigenvalues / total_variance

print("Step 4: Explained variance ratio:")
for i, ratio in enumerate(explained_variance_ratio):
    print(f"  PC{i+1}: {ratio:.4f} ({ratio*100:.2f}%)")

cumulative_variance = np.cumsum(explained_variance_ratio)
print("\\nCumulative explained variance:")
for i, cum_var in enumerate(cumulative_variance):
    print(f"  First {i+1} PCs: {cum_var:.4f} ({cum_var*100:.2f}%)")
print()

# Step 5: Project onto first 2 principal components
X_pca = X_scaled @ eigenvectors[:, :2]

print(f"Step 5: Projected data shape: {X_pca.shape}")
print(f"Reduced from {X.shape[1]} to {X_pca.shape[1]} dimensions")
\`\`\`

## PCA via SVD

SVD provides a more numerically stable and efficient way to compute PCA.

For centered data **X** (n × d):

**X** = **UΣVᵀ**

**Principal components**: Columns of **V**
**Transformed data**: **XVₖ** = **UₖΣₖ**
**Variance**: σᵢ² / n = eigenvalue λᵢ

**Advantages**:
- No need to form covariance matrix (saves memory, more stable)
- More efficient for tall matrices (n >> d)
- Directly gives transformed data

\`\`\`python
print("\\n=== PCA via SVD ===")

# Center data
X_centered = X_scaled - X_scaled.mean(axis=0)

# SVD
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

print(f"U shape: {U.shape}")
print(f"S (singular values): {S}")
print(f"Vt shape: {Vt.shape}")
print()

# Principal components = rows of Vt (or columns of V)
V = Vt.T
print("Principal components (columns of V):")
print(V)
print()

# Eigenvalues from singular values
eigenvalues_svd = (S ** 2) / (len(X_centered) - 1)

print("Eigenvalues from SVD:")
print(eigenvalues_svd)
print()

print("Match eigenvalues from covariance:")
print(f"Equal: {np.allclose(eigenvalues_svd, eigenvalues)}")
print()

# Transformed data
X_pca_svd = X_centered @ V[:, :2]

print(f"Transformed data via SVD: {X_pca_svd.shape}")
print(f"Equal to previous method: {np.allclose(X_pca, X_pca_svd)}")
\`\`\`

## Choosing Number of Components

### Method 1: Explained Variance Threshold

Keep components until cumulative explained variance ≥ threshold (e.g., 95%).

\`\`\`python
print("\\n=== Method 1: Explained Variance Threshold ===")

threshold = 0.95
n_components = np.argmax(cumulative_variance >= threshold) + 1

print(f"Threshold: {threshold*100:.0f}%")
print(f"Components needed: {n_components}")
print(f"Actual variance explained: {cumulative_variance[n_components-1]:.4f}")
\`\`\`

### Method 2: Scree Plot (Elbow Method)

Plot eigenvalues and look for "elbow" where curve flattens.

\`\`\`python
print("\\n=== Method 2: Scree Plot ===")

# Would visualize in practice
print("Eigenvalues:")
for i, val in enumerate(eigenvalues):
    print(f"  PC{i+1}: {val:.4f}")

print("\\n→ Look for elbow in plot where additional components add little variance")
\`\`\`

### Method 3: Kaiser Criterion

Keep components with eigenvalue > 1 (for standardized data).

\`\`\`python
print("\\n=== Method 3: Kaiser Criterion ===")

n_components_kaiser = np.sum(eigenvalues > 1)

print(f"Components with eigenvalue > 1: {n_components_kaiser}")
print("Eigenvalues > 1:")
for i, val in enumerate(eigenvalues):
    if val > 1:
        print(f"  PC{i+1}: {val:.4f}")
\`\`\`

## Reconstructing Data from PCA

Project to k dimensions, then back to original space (lossy reconstruction).

\`\`\`python
print("\\n=== Data Reconstruction ===")

k = 2  # Use only first 2 components

# Project to k dimensions
X_reduced = X_centered @ V[:, :k]

# Reconstruct (back to d dimensions)
X_reconstructed = X_reduced @ V[:, :k].T

print(f"Original shape: {X_centered.shape}")
print(f"Reduced shape: {X_reduced.shape}")
print(f"Reconstructed shape: {X_reconstructed.shape}")
print()

# Reconstruction error
reconstruction_error = np.linalg.norm(X_centered - X_reconstructed, 'fro')**2 / X_centered.shape[0]

print(f"Reconstruction error (MSE): {reconstruction_error:.6f}")
print()

# This equals sum of discarded eigenvalues
expected_error = np.sum(eigenvalues[k:])
print(f"Expected error (sum of discarded eigenvalues): {expected_error:.6f}")
print(f"Match: {np.allclose(reconstruction_error, expected_error)}")
\`\`\`

## PCA with Scikit-learn

\`\`\`python
print("\\n=== PCA with Scikit-learn ===")

from sklearn.decomposition import PCA

# Create PCA object
pca = PCA(n_components=2)

# Fit and transform
X_pca_sklearn = pca.fit_transform(X_scaled)

print(f"Transformed data shape: {X_pca_sklearn.shape}")
print()

print("Principal components:")
print(pca.components_)
print()

print("Explained variance:")
print(pca.explained_variance_)
print()

print("Explained variance ratio:")
print(pca.explained_variance_ratio_)
print()

print(f"Cumulative variance: {pca.explained_variance_ratio_.sum():.4f}")
print()

# Inverse transform (reconstruction)
X_reconstructed_sklearn = pca.inverse_transform(X_pca_sklearn)

print(f"Reconstructed shape: {X_reconstructed_sklearn.shape}")
\`\`\`

## Applications in Machine Learning

### 1. Visualization

\`\`\`python
print("\\n=== Application: Visualization ===")

# Reduce high-dimensional data to 2D for plotting
pca_viz = PCA(n_components=2)
X_2d = pca_viz.fit_transform(X_scaled)

print(f"Original dimensions: {X.shape[1]}")
print(f"Reduced to: {X_2d.shape[1]} (for visualization)")
print(f"Variance preserved: {pca_viz.explained_variance_ratio_.sum():.2%}")
print()

# In practice, would create scatter plot with colors by class
print("Can now visualize 4D iris data in 2D scatter plot!")
\`\`\`

### 2. Speeding Up Learning

\`\`\`python
print("\\n=== Application: Speeding Up Learning ===")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# Train on original data
start = time.time()
clf_original = LogisticRegression(max_iter=1000, random_state=42)
clf_original.fit(X_train, y_train)
time_original = time.time() - start

y_pred_original = clf_original.predict(X_test)
acc_original = accuracy_score(y_test, y_pred_original)

print(f"Original data ({X.shape[1]} features):")
print(f"  Training time: {time_original:.4f}s")
print(f"  Accuracy: {acc_original:.4f}")
print()

# Train on PCA-reduced data
pca_fast = PCA(n_components=2)
X_train_pca = pca_fast.fit_transform(X_train)
X_test_pca = pca_fast.transform(X_test)

start = time.time()
clf_pca = LogisticRegression(max_iter=1000, random_state=42)
clf_pca.fit(X_train_pca, y_train)
time_pca = time.time() - start

y_pred_pca = clf_pca.predict(X_test_pca)
acc_pca = accuracy_score(y_test, y_pred_pca)

print(f"PCA data ({X_train_pca.shape[1]} features):")
print(f"  Training time: {time_pca:.4f}s")
print(f"  Accuracy: {acc_pca:.4f}")
print(f"  Speedup: {time_original/time_pca:.2f}x")
print(f"  Accuracy loss: {acc_original - acc_pca:.4f}")
\`\`\`

### 3. Noise Reduction

\`\`\`python
print("\\n=== Application: Noise Reduction ===")

# Add noise to data
np.random.seed(42)
X_noisy = X_scaled + np.random.normal(0, 0.5, X_scaled.shape)

print(f"Added Gaussian noise (std=0.5)")
print()

# Denoise using PCA (keep only top components)
pca_denoise = PCA(n_components=2)  # Keep top 2 (most signal)
X_denoised = pca_denoise.inverse_transform(
    pca_denoise.fit_transform(X_noisy)
)

# Compare errors
error_noisy = np.linalg.norm(X_scaled - X_noisy, 'fro') / np.sqrt(X_scaled.size)
error_denoised = np.linalg.norm(X_scaled - X_denoised, 'fro') / np.sqrt(X_scaled.size)

print(f"RMSE (noisy vs original): {error_noisy:.4f}")
print(f"RMSE (denoised vs original): {error_denoised:.4f}")
print(f"Noise reduction: {(1 - error_denoised/error_noisy)*100:.1f}%")
\`\`\`

### 4. Feature Extraction

\`\`\`python
print("\\n=== Application: Feature Extraction ===")

# Use PCA components as new features
pca_features = PCA(n_components=3)
X_new_features = pca_features.fit_transform(X_scaled)

print(f"Original features ({X.shape[1]}): {iris.feature_names}")
print()

print(f"New features ({X_new_features.shape[1]}): PC1, PC2, PC3")
print(f"These are uncorrelated and capture {pca_features.explained_variance_ratio_.sum():.1%} of variance")
print()

# Interpretation: what does PC1 represent?
print("PC1 loadings (contribution of each original feature):")
for i, feature in enumerate(iris.feature_names):
    print(f"  {feature}: {pca_features.components_[0, i]:.4f}")
\`\`\`

## Limitations and Considerations

### 1. Linearity

PCA finds linear combinations. For nonlinear structure, consider **Kernel PCA** or **t-SNE**.

\`\`\`python
print("\\n=== Limitation: Linearity ===")

# Generate nonlinear data (circle)
theta = np.linspace(0, 2*np.pi, 100)
X_circle = np.column_stack([np.cos(theta), np.sin(theta)])
X_circle += np.random.normal(0, 0.05, X_circle.shape)

# PCA fails to capture circular structure
pca_circle = PCA(n_components=1)
X_circle_pca = pca_circle.fit_transform(X_circle)

print("Circular data (inherently 1D structure)")
print(f"PCA variance explained with 1 component: {pca_circle.explained_variance_ratio_[0]:.2%}")
print("→ PCA requires 2 components for 1D circular manifold (inefficient)")
print("→ Use Kernel PCA or manifold learning for nonlinear structure")
\`\`\`

### 2. Scaling Sensitivity

PCA is sensitive to feature scales. Always standardize!

\`\`\`python
print("\\n=== Limitation: Scaling Sensitivity ===")

# Create data with different scales
X_unscaled = np.column_stack([
    np.random.randn(50) * 1,    # Feature 1: std = 1
    np.random.randn(50) * 100   # Feature 2: std = 100
])

# PCA without scaling
pca_unscaled = PCA(n_components=2)
pca_unscaled.fit(X_unscaled)

print("Unscaled data:")
print(f"  Feature 1 std: 1")
print(f"  Feature 2 std: 100")
print(f"  PC1 variance ratio: {pca_unscaled.explained_variance_ratio_[0]:.4f}")
print("  → PC1 dominated by high-variance feature!")
print()

# PCA with scaling
X_scaled_demo = StandardScaler().fit_transform(X_unscaled)
pca_scaled_demo = PCA(n_components=2)
pca_scaled_demo.fit(X_scaled_demo)

print("Scaled data:")
print(f"  PC1 variance ratio: {pca_scaled_demo.explained_variance_ratio_[0]:.4f}")
print("  → More balanced!")
\`\`\`

### 3. Interpretability

Principal components are linear combinations, sometimes hard to interpret.

\`\`\`python
print("\\n=== Limitation: Interpretability ===")

print("PC1 = 0.52*sepal_length + 0.37*sepal_width + ...")
print("→ Not always clear what this represents conceptually")
print("→ Trade-off: mathematical optimality vs human interpretability")
\`\`\`

## PCA vs Other Methods

\`\`\`python
print("\\n=== PCA vs Other Dimensionality Reduction Methods ===")

comparison = """
| Method     | Linear | Preserves | Use Case                          |
|------------|--------|-----------|-----------------------------------|
| PCA        | Yes    | Variance  | General purpose, fast, interpretable |
| t-SNE      | No     | Local structure | Visualization, nonlinear manifolds |
| UMAP       | No     | Global + local | Modern alternative to t-SNE |
| LDA        | Yes    | Class separation | Supervised dimensionality reduction |
| Kernel PCA | No     | Variance (in kernel space) | Nonlinear patterns |
| Autoencoders | No   | Reconstruction | Deep learning, very high dimensions |

**When to use PCA**:
- Data is (approximately) linear
- Want to preserve global variance
- Need fast, deterministic method
- Want interpretable components
- First step before trying complex methods

**When NOT to use PCA**:
- Data lies on nonlinear manifold (circle, Swiss roll, etc.)
- Need to preserve local neighborhood structure
- Supervised task (use LDA instead)
- Have very high dimensions (consider random projections, sparse PCA)
"""

print(comparison)
\`\`\`

## Summary

**PCA**: Projects data onto orthogonal directions of maximum variance.

**Mathematical Foundation**:
- Eigendecomposition of covariance matrix: **C** = **VΛVᵀ**
- Or SVD of data matrix: **X** = **UΣVᵀ**
- Principal components = eigenvectors of **C** = columns of **V**

**Key Steps**:
1. Center (and standardize) data
2. Compute covariance or use SVD
3. Find eigenvectors/eigenvalues (or singular vectors/values)
4. Sort by eigenvalue (descending)
5. Project onto top k components

**Choosing k**:
- Explained variance threshold (e.g., 95%)
- Scree plot (elbow method)
- Kaiser criterion (λ > 1)
- Cross-validation

**Applications**:
- **Visualization**: Reduce to 2D/3D
- **Speedup**: Fewer features → faster training
- **Noise reduction**: Keep signal, discard noise
- **Feature extraction**: Uncorrelated features
- **Data compression**: Approximate with fewer dimensions

**Limitations**:
- Linear only (use Kernel PCA for nonlinear)
- Sensitive to scaling (always standardize!)
- Components may be hard to interpret
- Assumes variance = importance (not always true)

**Best Practices**:
- Always standardize features first
- Check cumulative explained variance
- Validate on downstream task performance
- Consider nonlinear methods if PCA performs poorly

PCA is the workhorse of dimensionality reduction—fast, interpretable, and effective for many real-world problems!
`,
};
