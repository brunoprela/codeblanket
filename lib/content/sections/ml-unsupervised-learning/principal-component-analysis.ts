/**
 * Section: Principal Component Analysis (PCA)
 * Module: Classical Machine Learning - Unsupervised Learning
 *
 * Comprehensive coverage of PCA for dimensionality reduction and feature extraction
 */

export const principalComponentAnalysis = {
  id: 'principal-component-analysis',
  title: 'Principal Component Analysis (PCA)',
  content: `
# Principal Component Analysis (PCA)

## Introduction

Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional space while preserving as much variance as possible. It's one of the most widely used techniques in machine learning for:

- **Dimensionality Reduction**: Reduce features from thousands to tens
- **Visualization**: Project high-D data to 2D/3D for plotting
- **Noise Reduction**: Remove less important dimensions
- **Feature Extraction**: Create new meaningful features
- **Data Compression**: Reduce storage and computation

**Core Idea**: Find new axes (principal components) along which data varies the most

## The Mathematics Behind PCA

### What Are Principal Components?

Principal Components (PCs) are new orthogonal axes that:
1. **PC1**: Direction of maximum variance in the data
2. **PC2**: Direction of maximum remaining variance, orthogonal to PC1
3. **PC3**: Direction of maximum remaining variance, orthogonal to PC1 & PC2
4. And so on...

**Key Properties**:
- Linear combinations of original features
- Uncorrelated with each other (orthogonal)
- Ordered by variance explained

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_digits

# Load iris dataset (4 features)
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print(f"Original shape: {X.shape}")
print(f"Features: {feature_names}")
print(f"Classes: {target_names}")

# Visualize first two features
plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green']
for i, color, name in zip([0, 1, 2], colors, target_names):
    plt.scatter(X[y == i, 0], X[y == i, 1], 
               c=color, label=name, alpha=0.6, s=50)
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title('Iris Dataset: First Two Features')
plt.legend()
plt.show()

# Classes overlap significantly - PCA might help separate them
\`\`\`

### Mathematical Formulation

Given data matrix \\( X \\) (n samples × m features):

1. **Center the data**: \\( X_{centered} = X - \\mu \\)

2. **Compute covariance matrix**: 
$$\\Sigma = \\frac{1}{n-1} X_{centered}^T X_{centered}$$

3. **Eigenvalue decomposition**:
$$\\Sigma v_i = \\lambda_i v_i$$

Where:
- \\( v_i \\) = eigenvectors (principal components)
- \\( \\lambda_i \\) = eigenvalues (variance along PC_i)

4. **Project data**: \\( X_{PCA} = X_{centered} \\cdot V_k \\)

Where \\( V_k \\) contains the top k eigenvectors

\`\`\`python
# Manual PCA calculation
# Step 1: Center the data
X_centered = X - X.mean(axis=0)

# Step 2: Compute covariance matrix
cov_matrix = np.cov(X_centered.T)
print(f"Covariance matrix shape: {cov_matrix.shape}")

# Step 3: Eigenvalue decomposition
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort by eigenvalue (descending)
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print(f"\\nEigenvalues (variance): {eigenvalues}")
print(f"Eigenvectors shape: {eigenvectors.shape}")

# Step 4: Project to 2D
X_pca_manual = X_centered @ eigenvectors[:, :2]

print(f"\\nManual PCA result shape: {X_pca_manual.shape}")
\`\`\`

## Using Scikit-Learn PCA

\`\`\`python
# Standardize features first (important!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"\\nScikit-learn PCA result shape: {X_pca.shape}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.2%}")

# Visualize PCA projection
plt.figure(figsize=(10, 6))
for i, color, name in zip([0, 1, 2], colors, target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], 
               c=color, label=name, alpha=0.6, s=50)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('Iris Dataset: PCA Projection to 2D')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# PC1 and PC2 together explain ~97% of variance!
# Classes are now better separated
\`\`\`

## Choosing Number of Components

### Method 1: Explained Variance Ratio

\`\`\`python
# Fit PCA with all components
pca_full = PCA()
pca_full.fit(X_scaled)

# Plot explained variance
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(range(1, len(pca_full.explained_variance_ratio_) + 1),
        pca_full.explained_variance_ratio_,
        alpha=0.6, color='blue')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Variance Explained by Each PC')
plt.xticks(range(1, len(pca_full.explained_variance_ratio_) + 1))

plt.subplot(1, 2, 2)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
plt.plot(range(1, len(cumulative_variance) + 1), 
         cumulative_variance, 'go-', linewidth=2, markersize=8)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Variance Explained')
plt.axhline(y=0.95, color='r', linestyle='--', linewidth=2, label='95% threshold')
plt.axhline(y=0.99, color='orange', linestyle='--', linewidth=2, label='99% threshold')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xticks(range(1, len(cumulative_variance) + 1))

plt.tight_layout()
plt.show()

# Rule of thumb: keep enough components to explain 95-99% of variance
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Components needed for 95% variance: {n_components_95}")
\`\`\`

### Method 2: Elbow Method (Scree Plot)

\`\`\`python
# Scree plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca_full.explained_variance_) + 1),
         pca_full.explained_variance_, 'go-', linewidth=2, markersize=10)
plt.xlabel('Principal Component', fontsize=12)
plt.ylabel('Eigenvalue (Variance)', fontsize=12)
plt.title('Scree Plot - Look for the "Elbow"', fontsize=14)
plt.xticks(range(1, len(pca_full.explained_variance_) + 1))
plt.grid(True, alpha=0.3)
plt.show()

# Keep components before the "elbow" where eigenvalues drop off
\`\`\`

### Method 3: Kaiser Criterion

Keep components with eigenvalue > 1 (for standardized data)

\`\`\`python
n_components_kaiser = np.sum(pca_full.explained_variance_ > 1)
print(f"Kaiser criterion: keep {n_components_kaiser} components")
print(f"(eigenvalues > 1.0)")
\`\`\`

### Method 4: Cross-Validation

Choose number of components based on downstream task performance

\`\`\`python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Try different numbers of components
n_components_range = range(1, X.shape[1] + 1)
cv_scores = []

for n in n_components_range:
    pca_temp = PCA(n_components=n)
    X_temp = pca_temp.fit_transform(X_scaled)
    
    # Evaluate with logistic regression
    clf = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf, X_temp, y, cv=5)
    cv_scores.append(scores.mean())

# Plot
plt.figure(figsize=(10, 6))
plt.plot(n_components_range, cv_scores, 'go-', linewidth=2, markersize=8)
plt.xlabel('Number of Components')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Classification Accuracy vs Number of Components')
plt.grid(True, alpha=0.3)
plt.xticks(n_components_range)
plt.show()

best_n = n_components_range[np.argmax(cv_scores)]
print(f"Best number of components: {best_n}")
print(f"Best CV accuracy: {max(cv_scores):.3f}")
\`\`\`

## Interpreting Principal Components

\`\`\`python
# Component loadings (weights)
pca_interpret = PCA(n_components=2)
pca_interpret.fit(X_scaled)

# Loadings matrix
loadings = pca_interpret.components_.T * np.sqrt(pca_interpret.explained_variance_)

# Create loadings dataframe
import pandas as pd
loadings_df = pd.DataFrame(
    loadings,
    columns=['PC1', 'PC2'],
    index=feature_names
)

print("Component Loadings:")
print(loadings_df)
print("\\nLoadings show contribution of each original feature to each PC")

# Visualize loadings
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, pc in enumerate(['PC1', 'PC2']):
    loadings_df[pc].plot(kind='barh', ax=axes[idx], color='steelblue')
    axes[idx].set_title(f'{pc} Loadings')
    axes[idx].set_xlabel('Loading Value')
    axes[idx].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Interpretation:
# PC1: Dominated by petal features (length & width)
# PC2: Contrasts sepal width vs other features
\`\`\`

### Biplot: Data + Loadings Together

\`\`\`python
def biplot(X_pca, loadings, labels, feature_names):
    ''Create a biplot showing data and loadings''
    plt.figure(figsize=(12, 8))
    
    # Plot data points
    colors = ['red', 'blue', 'green']
    for i, color, name in zip([0, 1, 2], colors, target_names):
        mask = labels == i
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=color, label=name, alpha=0.5, s=50)
    
    # Plot loading vectors
    for i, feature in enumerate(feature_names):
        plt.arrow(0, 0, 
                 loadings[i, 0]*3, loadings[i, 1]*3,
                 head_width=0.1, head_length=0.1, 
                 fc='black', ec='black', linewidth=2)
        plt.text(loadings[i, 0]*3.2, loadings[i, 1]*3.2, 
                feature, fontsize=10, fontweight='bold')
    
    plt.xlabel(f'PC1')
    plt.ylabel(f'PC2')
    plt.title('PCA Biplot: Data Points + Feature Loadings')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.show()

biplot(X_pca, loadings, y, feature_names)

print("Arrows show how original features relate to principal components")
print("Longer arrows = stronger influence")
print("Parallel arrows = correlated features")
\`\`\`

## PCA for Visualization

### High-Dimensional Data: Digits Dataset

\`\`\`python
# Load digits dataset (64 features = 8x8 pixels)
digits = load_digits()
X_digits = digits.data
y_digits = digits.target

print(f"Digits shape: {X_digits.shape}")
print(f"Classes: {len(np.unique(y_digits))}")

# Visualize some digits
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(X_digits[i].reshape(8, 8), cmap='gray')
    ax.set_title(f'Label: {y_digits[i]}')
    ax.axis('off')
plt.suptitle('Sample Handwritten Digits (8x8 = 64 features)')
plt.tight_layout()
plt.show()

# Apply PCA
scaler_digits = StandardScaler()
X_digits_scaled = scaler_digits.fit_transform(X_digits)

pca_digits = PCA(n_components=2)
X_digits_pca = pca_digits.fit_transform(X_digits_scaled)

print(f"Reduced to: {X_digits_pca.shape}")
print(f"Variance explained: {pca_digits.explained_variance_ratio_.sum():.2%}")

# Visualize
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_digits_pca[:, 0], X_digits_pca[:, 1], 
                     c=y_digits, cmap='tab10', s=20, alpha=0.6)
plt.colorbar(scatter, ticks=range(10))
plt.xlabel(f'PC1 ({pca_digits.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca_digits.explained_variance_ratio_[1]:.1%})')
plt.title('Handwritten Digits: PCA from 64D to 2D')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 64 dimensions reduced to 2, and different digits cluster!
\`\`\`

### 3D Visualization

\`\`\`python
from mpl_toolkits.mplot3d import Axes3D

# PCA to 3D
pca_3d = PCA(n_components=3)
X_digits_pca_3d = pca_3d.fit_transform(X_digits_scaled)

# 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

for digit in range(10):
    mask = y_digits == digit
    ax.scatter(X_digits_pca_3d[mask, 0], 
              X_digits_pca_3d[mask, 1],
              X_digits_pca_3d[mask, 2],
              label=str(digit), s=20, alpha=0.6)

ax.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})')
ax.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})')
ax.set_title(f'3D PCA Projection ({pca_3d.explained_variance_ratio_.sum():.1%} variance)')
ax.legend(loc='upper right')
plt.tight_layout()
plt.show()
\`\`\`

## Inverse Transform: Reconstruction

PCA can compress and decompress data

\`\`\`python
# Original image
original_digit = X_digits[0].reshape(8, 8)

# Compress to different numbers of components
n_components_list = [2, 5, 10, 20, 40, 64]

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.ravel()

for idx, n_comp in enumerate(n_components_list):
    # Apply PCA
    pca_recon = PCA(n_components=n_comp)
    X_compressed = pca_recon.fit_transform(X_digits_scaled)
    
    # Reconstruct
    X_reconstructed = pca_recon.inverse_transform(X_compressed)
    X_reconstructed = scaler_digits.inverse_transform(X_reconstructed)
    
    # Show first digit
    reconstructed_digit = X_reconstructed[0].reshape(8, 8)
    
    # Calculate reconstruction error
    mse = np.mean((X_digits[0] - X_reconstructed[0])**2)
    
    axes[idx].imshow(reconstructed_digit, cmap='gray')
    axes[idx].set_title(f'{n_comp} components\\nMSE: {mse:.2f}')
    axes[idx].axis('off')

plt.suptitle('PCA Reconstruction with Different Components', fontsize=14)
plt.tight_layout()
plt.show()

# More components = better reconstruction
# But also more storage/computation
\`\`\`

## PCA for Noise Reduction

\`\`\`python
# Add noise to digits
noise = np.random.normal(0, 4, X_digits.shape)
X_noisy = X_digits + noise

# Original, noisy, and denoised
digit_idx = 5

# Denoise with PCA
pca_denoise = PCA(n_components=20)  # Keep only top 20 components
X_noisy_scaled = scaler_digits.fit_transform(X_noisy)
X_compressed = pca_denoise.fit_transform(X_noisy_scaled)
X_denoised_scaled = pca_denoise.inverse_transform(X_compressed)
X_denoised = scaler_digits.inverse_transform(X_denoised_scaled)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(X_digits[digit_idx].reshape(8, 8), cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(X_noisy[digit_idx].reshape(8, 8), cmap='gray')
axes[1].set_title('Noisy')
axes[1].axis('off')

axes[2].imshow(X_denoised[digit_idx].reshape(8, 8), cmap='gray')
axes[2].set_title('Denoised with PCA')
axes[2].axis('off')

plt.tight_layout()
plt.show()

# PCA removes noise by keeping only major patterns
\`\`\`

## PCA in Machine Learning Pipeline

### Feature Engineering with PCA

\`\`\`python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_digits, y_digits, test_size=0.3, random_state=42
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compare: with and without PCA
results = {}

# Without PCA
clf_raw = RandomForestClassifier(n_estimators=100, random_state=42)
clf_raw.fit(X_train_scaled, y_train)
y_pred_raw = clf_raw.predict(X_test_scaled)
results['Raw (64 features)'] = accuracy_score(y_test, y_pred_raw)

# With different PCA components
for n_comp in [5, 10, 20, 30]:
    pca = PCA(n_components=n_comp)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    clf_pca = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_pca.fit(X_train_pca, y_train)
    y_pred_pca = clf_pca.predict(X_test_pca)
    
    accuracy = accuracy_score(y_test, y_pred_pca)
    variance_explained = pca.explained_variance_ratio_.sum()
    results[f'PCA ({n_comp} comp, {variance_explained:.1%} var)'] = accuracy

# Plot results
plt.figure(figsize=(12, 6))
plt.bar(range(len(results)), list(results.values()), color='steelblue', alpha=0.7)
plt.xticks(range(len(results)), list(results.keys()), rotation=45, ha='right')
plt.ylabel('Test Accuracy')
plt.title('Classification Accuracy: Raw Features vs PCA')
plt.ylim([0.9, 1.0])
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

for name, acc in results.items():
    print(f"{name}: {acc:.3f}")

# PCA with 20+ components achieves similar accuracy with fewer features!
\`\`\`

## Practical Considerations

### When to Use PCA

✅ **Use PCA When**:
- High-dimensional data (many features)
- Features are correlated
- Need visualization of high-D data
- Want to reduce overfitting
- Need to speed up algorithms
- Want noise reduction
- Data is approximately linear

❌ **Avoid PCA When**:
- Features are already uncorrelated
- Non-linear relationships exist (consider Kernel PCA)
- Need interpretable features
- Data is sparse (PCA creates dense representations)
- Very few features already

### Scaling is Critical

\`\`\`python
# Compare PCA with and without scaling
pca_unscaled = PCA(n_components=2)
X_pca_unscaled = pca_unscaled.fit_transform(X)

pca_scaled = PCA(n_components=2)
X_pca_scaled = pca_scaled.fit_transform(X_scaled)

print("Without scaling:")
print(f"Variance explained: {pca_unscaled.explained_variance_ratio_}")
print("\\nWith scaling:")
print(f"Variance explained: {pca_scaled.explained_variance_ratio_}")

# Without scaling, features with large ranges dominate!
# ALWAYS scale before PCA
\`\`\`

### Handling Non-Linear Data

\`\`\`python
from sklearn.datasets import make_circles

# Non-linear data (circles)
X_circles, y_circles = make_circles(n_samples=400, factor=0.3, noise=0.05, random_state=42)

# Try linear PCA
pca_linear = PCA(n_components=2)
X_pca_linear = pca_linear.fit_transform(X_circles)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles, cmap='viridis', s=30)
axes[0].set_title('Original Non-Linear Data')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')

axes[1].scatter(X_pca_linear[:, 0], X_pca_linear[:, 1], c=y_circles, cmap='viridis', s=30)
axes[1].set_title('Linear PCA (fails to separate)')
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')

plt.tight_layout()
plt.show()

print("Linear PCA cannot handle non-linear structure")
print("Consider: Kernel PCA, t-SNE, UMAP for non-linear data")
\`\`\`

## Kernel PCA for Non-Linear Data

\`\`\`python
from sklearn.decomposition import KernelPCA

# Kernel PCA with RBF kernel
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_kpca = kpca.fit_transform(X_circles)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles, cmap='viridis', s=30)
plt.title('Original Non-Linear Data')

plt.subplot(1, 2, 2)
plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y_circles, cmap='viridis', s=30)
plt.title('Kernel PCA (RBF kernel)')
plt.xlabel('KPC1')
plt.ylabel('KPC2')

plt.tight_layout()
plt.show()

print("Kernel PCA can handle non-linear structure!")
\`\`\`

## Common Pitfalls

1. **Forgetting to Scale**: Different units → biased PCA
2. **Using Too Few Components**: Losing important information
3. **Interpreting PCs as Original Features**: They're combinations
4. **Applying to Categorical Data**: PCA assumes continuous, linear relationships
5. **Not Checking Variance Explained**: May be discarding too much
6. **Overfitting**: Using test data to choose n_components

## Real-World Applications

### Application 1: Image Compression

\`\`\`python
# Face image example
from sklearn.datasets import fetch_olivetti_faces

faces = fetch_olivetti_faces()
X_faces = faces.data
face_shape = (64, 64)

print(f"Original: {X_faces.shape}")

# One face
original_face = X_faces[0].reshape(face_shape)

# Compress with different components
n_components_list = [10, 50, 100, 200, 400]
compression_ratios = []

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

axes[0].imshow(original_face, cmap='gray')
axes[0].set_title('Original (4096 values)')
axes[0].axis('off')

for idx, n_comp in enumerate(n_components_list, 1):
    pca = PCA(n_components=n_comp)
    X_compressed = pca.fit_transform(X_faces)
    X_reconstructed = pca.inverse_transform(X_compressed)
    
    reconstructed_face = X_reconstructed[0].reshape(face_shape)
    
    # Compression ratio
    original_size = X_faces.shape[1]
    compressed_size = n_comp
    ratio = original_size / compressed_size
    compression_ratios.append(ratio)
    
    axes[idx].imshow(reconstructed_face, cmap='gray')
    axes[idx].set_title(f'{n_comp} components\\n{ratio:.1f}x compression')
    axes[idx].axis('off')

plt.tight_layout()
plt.show()
\`\`\`

### Application 2: Genomics

\`\`\`python
# Simulated gene expression data
# Samples (patients) × Genes (features)
np.random.seed(42)
n_samples = 100
n_genes = 5000  # Thousands of genes!

# Two groups: healthy vs disease
X_genes = np.vstack([
    np.random.randn(50, n_genes) + 0,  # Healthy
    np.random.randn(50, n_genes) + 0.5  # Disease
])
y_genes = np.array([0]*50 + [1]*50)

print(f"Gene expression data: {X_genes.shape}")
print(f"Features >> Samples (curse of dimensionality)")

# Apply PCA
scaler_genes = StandardScaler()
X_genes_scaled = scaler_genes.fit_transform(X_genes)

pca_genes = PCA(n_components=2)
X_genes_pca = pca_genes.fit_transform(X_genes_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(X_genes_pca[y_genes==0, 0], X_genes_pca[y_genes==0, 1], 
           c='blue', label='Healthy', alpha=0.6, s=50)
plt.scatter(X_genes_pca[y_genes==1, 0], X_genes_pca[y_genes==1, 1], 
           c='red', label='Disease', alpha=0.6, s=50)
plt.xlabel(f'PC1 ({pca_genes.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca_genes.explained_variance_ratio_[1]:.1%})')
plt.title('Gene Expression: 5000 Genes → 2D PCA Projection')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("PCA enables visualization and analysis of high-dimensional genomic data")
\`\`\`

## Summary

PCA is a fundamental technique for dimensionality reduction that:

**Key Concepts**:
- Transforms data to new uncorrelated axes (principal components)
- Orders components by variance explained
- Enables visualization, compression, and noise reduction

**When to Use**:
- High-dimensional data
- Correlated features
- Need for visualization
- Computational efficiency

**Best Practices**:
1. **Always scale** features first
2. Check **variance explained** to choose number of components
3. Use **Kernel PCA** for non-linear data
4. Interpret **loadings** to understand components
5. Validate on **downstream task** performance

**Limitations**:
- Assumes linear relationships
- Loses interpretability
- Sensitive to scaling
- May not preserve local structure

**Next**: We'll explore other dimensionality reduction techniques like t-SNE and UMAP that handle non-linear structures better for visualization.
`,
};
