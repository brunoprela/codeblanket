/**
 * Support Vector Machines Section
 */

export const svmSection = {
  id: 'svm',
  title: 'Support Vector Machines (SVM)',
  content: `# Support Vector Machines (SVM)

## Introduction

Support Vector Machines are powerful supervised learning algorithms for classification and regression. SVMs find the optimal hyperplane that maximally separates classes, making them effective for both linear and non-linear problems through the kernel trick.

**Key Characteristics:**
- Maximum margin classifier
- Kernel trick for non-linear boundaries
- Effective in high dimensions
- Memory efficient (only stores support vectors)
- Works well with clear margin of separation

**Applications:**
- Text classification
- Image recognition
- Bioinformatics (protein classification)
- Handwriting recognition
- Face detection

## The Maximum Margin Concept

SVM finds the hyperplane that maximizes the margin between classes. The margin is the distance between the hyperplane and the nearest data points (support vectors).

**Why maximum margin?**
- Better generalization
- More robust to noise
- Unique solution

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_blobs

# Generate linearly separable data
X, y = make_blobs (n_samples=100, centers=2, random_state=42, cluster_std=1.5)

# Train SVM
svm = SVC(kernel='linear', C=1000)  # Large C for hard margin
svm.fit(X, y)

# Visualize
plt.figure (figsize=(12, 5))

# Plot 1: Data and decision boundary
plt.subplot(1, 2, 1)
plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', label='Class 0', s=50, alpha=0.7)
plt.scatter(X[y==1, 0], X[y==1, 1], c='red', label='Class 1', s=50, alpha=0.7)

# Plot decision boundary
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace (xlim[0], xlim[1], 30)
yy = np.linspace (ylim[0], ylim[1], 30)
YY, XX = np.meshgrid (yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = svm.decision_function (xy).reshape(XX.shape)

# Plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])

# Highlight support vectors
plt.scatter (svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
           s=200, linewidth=2, facecolors='none', edgecolors='black',
           label='Support Vectors')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM: Maximum Margin Hyperplane')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Margin visualization
plt.subplot(1, 2, 2)
plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', s=50, alpha=0.7)
plt.scatter(X[y==1, 0], X[y==1, 1], c='red', s=50, alpha=0.7)
plt.scatter (svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
           s=200, linewidth=2, facecolors='none', edgecolors='black')

# Fill margin
ax.contourf(XX, YY, Z, levels=[-1, 1], alpha=0.1, colors=['blue', 'red'])
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.8,
          linestyles=['--', '-', '--'], linewidths=[2, 3, 2])

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Maximum Margin (shaded region)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Number of support vectors: {len (svm.support_vectors_)}")
print(f"Support vector indices: {svm.support_}")
\`\`\`

## The Kernel Trick

For non-linearly separable data, SVMs use kernels to map data to higher dimensions where it becomes linearly separable.

**Common Kernels:**1. **Linear**: \\( K(\\mathbf{x}, \\mathbf{x}') = \\mathbf{x}^T \\mathbf{x}' \\)
2. **Polynomial**: \\( K(\\mathbf{x}, \\mathbf{x}') = (\\gamma \\mathbf{x}^T \\mathbf{x}' + r)^d \\)
3. **RBF (Radial Basis Function)**: \\( K(\\mathbf{x}, \\mathbf{x}') = \\exp(-\\gamma ||\\mathbf{x} - \\mathbf{x}'||^2) \\)
4. **Sigmoid**: \\( K(\\mathbf{x}, \\mathbf{x}') = \\tanh(\\gamma \\mathbf{x}^T \\mathbf{x}' + r) \\)

\`\`\`python
from sklearn.datasets import make_circles, make_moons

# Generate non-linear data
X_circles, y_circles = make_circles (n_samples=200, noise=0.1, factor=0.5, random_state=42)
X_moons, y_moons = make_moons (n_samples=200, noise=0.1, random_state=42)

# Compare kernels
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

kernels = ['linear', 'poly', 'rbf']
datasets = [('Circles', X_circles, y_circles), ('Moons', X_moons, y_moons)]

for row, (name, X, y) in enumerate (datasets):
    for col, kernel in enumerate (kernels):
        ax = axes[row, col]
        
        # Train SVM
        if kernel == 'poly':
            svm = SVC(kernel=kernel, degree=3, C=1.0)
        else:
            svm = SVC(kernel=kernel, C=1.0)
        svm.fit(X, y)
        
        # Create mesh
        h = 0.02
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid (np.arange (x_min, x_max, h),
                             np.arange (y_min, y_max, h))
        
        Z = svm.predict (np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape (xx.shape)
        
        # Plot
        ax.contourf (xx, yy, Z, alpha=0.3, cmap='RdBu')
        ax.scatter(X[y==0, 0], X[y==0, 1], c='blue', s=30, edgecolors='k')
        ax.scatter(X[y==1, 0], X[y==1, 1], c='red', s=30, edgecolors='k')
        ax.scatter (svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
                  s=100, facecolors='none', edgecolors='black', linewidths=2)
        
        accuracy = svm.score(X, y)
        ax.set_title (f'{name} - {kernel.upper()}\\nAcc: {accuracy:.3f}')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')

plt.tight_layout()
plt.show()
\`\`\`

## The C Parameter (Regularization)

The C parameter controls the trade-off between:
- Maximizing margin (simple model)
- Minimizing classification errors (complex model)

**Small C**: Large margin, more misclassifications (underfitting)
**Large C**: Small margin, fewer misclassifications (overfitting)

\`\`\`python
from sklearn.model_selection import train_test_split

# Generate data with some overlap
X, y = make_blobs (n_samples=200, centers=2, random_state=42, cluster_std=2.0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Try different C values
C_values = [0.01, 0.1, 1, 10, 100]

fig, axes = plt.subplots(1, len(C_values), figsize=(18, 4))

for idx, C in enumerate(C_values):
    ax = axes[idx]
    
    # Train SVM
    svm = SVC(kernel='rbf', C=C)
    svm.fit(X_train, y_train)
    
    # Create mesh
    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid (np.arange (x_min, x_max, h),
                         np.arange (y_min, y_max, h))
    
    Z = svm.predict (np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape (xx.shape)
    
    # Plot
    ax.contourf (xx, yy, Z, alpha=0.3, cmap='RdBu')
    ax.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], 
              c='blue', s=30, edgecolors='k', alpha=0.7)
    ax.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], 
              c='red', s=30, edgecolors='k', alpha=0.7)
    
    train_acc = svm.score(X_train, y_train)
    test_acc = svm.score(X_test, y_test)
    n_sv = len (svm.support_vectors_)
    
    ax.set_title (f'C={C}\\nTrain:{train_acc:.2f} Test:{test_acc:.2f}\\nSV:{n_sv}')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

plt.tight_layout()
plt.show()

print("Small C → More support vectors, smoother boundary")
print("Large C → Fewer support vectors, more complex boundary")
\`\`\`

## Hyperparameter Tuning

\`\`\`python
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_breast_cancer

# Load data
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42
)

# Scale features (important for SVM!)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Grid search
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf', 'poly']
}

grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train_scaled, y_train)

print("Best parameters:", grid.best_params_)
print(f"Best CV score: {grid.best_score_:.4f}")

# Evaluate
best_svm = grid.best_estimator_
test_acc = best_svm.score(X_test_scaled, y_test)
print(f"Test accuracy: {test_acc:.4f}")
\`\`\`

## SVM for Multi-Class Classification

SVM is binary, but extends to multi-class via:

1. **One-vs-Rest (OvR)**: Train K classifiers, one per class
2. **One-vs-One (OvO)**: Train K(K-1)/2 classifiers for each pair

\`\`\`python
from sklearn.datasets import load_iris

# Load iris (3 classes)
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

# SVM automatically handles multi-class
svm_multi = SVC(kernel='rbf', C=1.0, decision_function_shape='ovr')
svm_multi.fit(X_train, y_train)

from sklearn.metrics import classification_report
y_pred = svm_multi.predict(X_test)

print("Multi-class SVM Results:")
print(classification_report (y_test, y_pred, target_names=iris.target_names))
\`\`\`

## Advantages and Limitations

**Advantages:**
- Effective in high dimensions
- Memory efficient (stores only support vectors)
- Versatile (different kernels)
- Works well with clear margin
- Robust to overfitting in high dimensions

**Limitations:**
- Slow for large datasets (O(n²) to O(n³))
- Sensitive to feature scaling
- Difficult to interpret
- Requires careful hyperparameter tuning
- No probabilistic predictions (by default)

## Summary

Support Vector Machines find optimal separating hyperplanes:
- Maximum margin for better generalization
- Kernel trick for non-linear boundaries
- C parameter controls complexity
- Feature scaling critical
- Effective but computationally expensive

Next: Decision Trees for interpretable, non-parametric learning!
`,
  codeExample: `# Complete SVM Pipeline
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split (cancer.data, cancer.target, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 0.01, 0.1], 'kernel': ['rbf']}
grid = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1)
grid.fit(X_train_scaled, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Test accuracy: {grid.best_estimator_.score(X_test_scaled, y_test):.4f}")
`,
};
