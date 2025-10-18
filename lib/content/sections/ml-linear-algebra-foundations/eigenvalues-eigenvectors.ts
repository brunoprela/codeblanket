/**
 * Eigenvalues & Eigenvectors Section
 */

export const eigenvalueseigenvectorsSection = {
  id: 'eigenvalues-eigenvectors',
  title: 'Eigenvalues & Eigenvectors',
  content: `
# Eigenvalues & Eigenvectors

## Introduction

**Eigenvalues** and **eigenvectors** are among the most important concepts in linear algebra and machine learning. They reveal the fundamental structure of linear transformations and appear throughout ML: in PCA, covariance matrices, graph analysis, Markov chains, and neural network analysis.

## Definition

For a square matrix **A** (n × n), a non-zero vector **v** is an **eigenvector** if:

**Av** = λ**v**

Where λ is a scalar called the **eigenvalue**.

**Geometric interpretation**: Matrix **A** stretches (or shrinks) **v** by factor λ, without changing direction.

**Key insight**: Most vectors change direction when multiplied by **A**, but eigenvectors only get scaled!

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt

print("=== Eigenvector Intuition ===")

# Simple 2×2 matrix
A = np.array([[3, 1],
              [0, 2]])

print("Matrix A:")
print(A)
print()

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)
print("\\nEigenvectors (as columns):")
print(eigenvectors)
print()

# Verify: Av = λv for each eigenvector
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lambda_i = eigenvalues[i]
    
    Av = A @ v
    lambda_v = lambda_i * v
    
    print(f"\\nEigenvector {i+1}: v = {v}")
    print(f"Eigenvalue {i+1}: λ = {lambda_i}")
    print(f"Av = {Av}")
    print(f"λv = {lambda_v}")
    print(f"Equal: {np.allclose(Av, lambda_v)}")
\`\`\`

## Computing Eigenvalues: Characteristic Equation

**Av** = λ**v**
**Av** - λ**v** = **0**
(**A** - λ**I**)**v** = **0**

For non-trivial solution (**v** ≠ **0**), (**A** - λ**I**) must be singular:

**det(A - λI) = 0**

This is the **characteristic equation**. Solving gives eigenvalues.

\`\`\`python
print("\\n=== Computing Eigenvalues Manually (2×2 case) ===")

A = np.array([[4, 2],
              [1, 3]])

print("Matrix A:")
print(A)
print()

# Characteristic equation: det(A - λI) = 0
# For 2×2: det([[4-λ, 2], [1, 3-λ]]) = 0
# (4-λ)(3-λ) - (2)(1) = 0
# 12 - 4λ - 3λ + λ² - 2 = 0
# λ² - 7λ + 10 = 0
# (λ - 5)(λ - 2) = 0
# λ = 5 or λ = 2

print("Characteristic equation: λ² - 7λ + 10 = 0")
print("Solutions: λ₁ = 5, λ₂ = 2")
print()

# Verify with NumPy
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"NumPy eigenvalues: {eigenvalues}")
print(f"Match: {np.allclose(sorted(eigenvalues), [2, 5])}")
\`\`\`

## Properties of Eigenvalues and Eigenvectors

### Property 1: Trace and Determinant

**Trace(A)** = sum of eigenvalues

**Det(A)** = product of eigenvalues

\`\`\`python
print("\\n=== Trace and Determinant ===")

A = np.array([[6, 2],
              [2, 3]])

eigenvalues = np.linalg.eigvals(A)

trace_A = np.trace(A)
sum_eigenvalues = np.sum(eigenvalues)

det_A = np.linalg.det(A)
prod_eigenvalues = np.prod(eigenvalues)

print("Matrix A:")
print(A)
print()
print(f"Eigenvalues: {eigenvalues}")
print()
print(f"Trace(A) = {trace_A}")
print(f"Sum of eigenvalues = {sum_eigenvalues}")
print(f"Equal: {np.allclose(trace_A, sum_eigenvalues)}")
print()
print(f"Det(A) = {det_A}")
print(f"Product of eigenvalues = {prod_eigenvalues}")
print(f"Equal: {np.allclose(det_A, prod_eigenvalues)}")
\`\`\`

### Property 2: Linear Independence

Eigenvectors corresponding to **distinct** eigenvalues are **linearly independent**.

\`\`\`python
print("\\n=== Linear Independence of Eigenvectors ===")

A = np.array([[5, 2],
              [2, 5]])

eigenvalues, eigenvectors = np.linalg.eig(A)

print("Matrix A:")
print(A)
print()
print(f"Eigenvalues: {eigenvalues}")
print(f"Distinct: {len(set(eigenvalues)) == len(eigenvalues)}")
print()

# Check linear independence via rank
rank = np.linalg.matrix_rank(eigenvectors)
print("Eigenvector matrix (columns are eigenvectors):")
print(eigenvectors)
print()
print(f"Rank: {rank}")
print(f"Number of eigenvectors: {eigenvectors.shape[1]}")
print(f"Linearly independent: {rank == eigenvectors.shape[1]}")
\`\`\`

### Property 3: Eigenvalues of Special Matrices

\`\`\`python
print("\\n=== Eigenvalues of Special Matrices ===")

# 1. Diagonal matrix: eigenvalues = diagonal entries
D = np.array([[3, 0, 0],
              [0, 7, 0],
              [0, 0, 5]])

eigenvalues_D = np.linalg.eigvals(D)
print("Diagonal matrix D:")
print(D)
print(f"Eigenvalues: {sorted(eigenvalues_D)}")
print(f"Diagonal entries: {[D[i,i] for i in range(3)]}")
print()

# 2. Symmetric matrix: all eigenvalues are real
S = np.array([[4, 2, 1],
              [2, 5, 3],
              [1, 3, 6]])

eigenvalues_S = np.linalg.eigvals(S)
print("Symmetric matrix S:")
print(S)
print(f"Eigenvalues: {eigenvalues_S}")
print(f"All real: {np.all(np.isreal(eigenvalues_S))}")
print()

# 3. Orthogonal matrix: |λ| = 1
# Rotation matrix (90 degrees)
theta = np.pi / 2
Q = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])

eigenvalues_Q = np.linalg.eigvals(Q)
print("Rotation matrix Q (90°):")
print(Q)
print(f"Eigenvalues: {eigenvalues_Q}")
print(f"Magnitudes: {np.abs(eigenvalues_Q)}")
print(f"All have |λ| = 1: {np.allclose(np.abs(eigenvalues_Q), 1)}")
\`\`\`

## Eigendecomposition (Diagonalization)

If **A** has n linearly independent eigenvectors, it can be **diagonalized**:

**A** = **PDP⁻¹**

Where:
- **P**: eigenvectors as columns
- **D**: diagonal matrix of eigenvalues
- **P⁻¹**: inverse of **P**

**Consequence**: **Aⁿ** = **PDⁿP⁻¹** (very efficient to compute)

\`\`\`python
print("\\n=== Eigendecomposition ===")

A = np.array([[5, 2],
              [2, 5]])

# Compute eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(A)

P = eigenvectors
D = np.diag(eigenvalues)
P_inv = np.linalg.inv(P)

print("Matrix A:")
print(A)
print()

print("P (eigenvectors as columns):")
print(P)
print()

print("D (diagonal of eigenvalues):")
print(D)
print()

# Reconstruct: A = PDP⁻¹
A_reconstructed = P @ D @ P_inv

print("A reconstructed (PDP⁻¹):")
print(A_reconstructed)
print()
print(f"Equal to original: {np.allclose(A, A_reconstructed)}")
print()

# Application: compute A^10 efficiently
A_10_direct = np.linalg.matrix_power(A, 10)
D_10 = np.diag(eigenvalues ** 10)
A_10_decomp = P @ D_10 @ P_inv

print("A^10 (via eigendecomposition):")
print(A_10_decomp)
print()
print(f"Equal to direct computation: {np.allclose(A_10_direct, A_10_decomp)}")
\`\`\`

## Symmetric Matrices and Orthogonal Eigenvectors

**Spectral Theorem**: Every **symmetric** matrix can be diagonalized by an **orthogonal** matrix:

**A** = **QΛQᵀ**

Where:
- **Q**: orthogonal matrix (columns are orthonormal eigenvectors)
- **Λ**: diagonal matrix of real eigenvalues
- **QᵀQ** = **QQᵀ** = **I**

\`\`\`python
print("\\n=== Spectral Theorem (Symmetric Matrices) ===")

A = np.array([[6, 2, 1],
              [2, 5, 2],
              [1, 2, 4]])

print("Symmetric matrix A:")
print(A)
print(f"Is symmetric: {np.allclose(A, A.T)}")
print()

# Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(A)

Q = eigenvectors
Lambda = np.diag(eigenvalues)

print("Q (eigenvectors as columns):")
print(Q)
print()

# Verify Q is orthogonal: QᵀQ = I
Q_T_Q = Q.T @ Q
print("QᵀQ:")
print(Q_T_Q)
print(f"Is identity: {np.allclose(Q_T_Q, np.eye(3))}")
print()

# Verify: A = QΛQᵀ
A_reconstructed = Q @ Lambda @ Q.T
print("A reconstructed (QΛQᵀ):")
print(A_reconstructed)
print(f"Equal to original: {np.allclose(A, A_reconstructed)}")
print()

# Eigenvalues are real
print(f"Eigenvalues: {eigenvalues}")
print(f"All real: {np.all(np.isreal(eigenvalues))}")
\`\`\`

## Applications in Machine Learning

### 1. Covariance Matrix Analysis

\`\`\`python
print("\\n=== Application: Covariance Matrix ===")

# Generate sample data
np.random.seed(42)
n_samples = 100

# Correlated 2D data
X = np.random.randn(n_samples, 2)
X[:, 1] = X[:, 0] + 0.5 * X[:, 1]  # Add correlation

# Covariance matrix
cov = np.cov(X.T)

print("Covariance matrix:")
print(cov)
print()

# Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(cov)

# Sort by eigenvalue (descending)
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print(f"Eigenvalues (variances along principal axes):")
print(eigenvalues)
print()

print("Eigenvectors (principal directions):")
print(eigenvectors)
print()

# Eigenvalues represent variance in principal directions
total_variance = np.sum(eigenvalues)
explained_variance_ratio = eigenvalues / total_variance

print("Explained variance ratio:")
for i, ratio in enumerate(explained_variance_ratio):
    print(f"  PC{i+1}: {ratio:.2%}")
\`\`\`

### 2. PageRank (Power Method)

\`\`\`python
print("\\n=== Application: PageRank (Simplified) ===")

# Transition matrix for web graph
# 4 pages: A, B, C, D
# Links: A→B, A→C, B→C, C→A, D→C

# Stochastic matrix (columns sum to 1)
P = np.array([
    [0,   0,   0.5, 0  ],   # A
    [0.5, 0,   0,   0  ],   # B
    [0.5, 1,   0,   1  ],   # C
    [0,   0,   0.5, 0  ]    # D
])

print("Transition matrix P:")
print(P)
print()

# Find dominant eigenvector (eigenvalue = 1)
eigenvalues, eigenvectors = np.linalg.eig(P)

# Find index of eigenvalue closest to 1
idx = np.argmin(np.abs(eigenvalues - 1))
dominant_eigenvalue = eigenvalues[idx]
page_rank = np.real(eigenvectors[:, idx])

# Normalize to probability distribution
page_rank = page_rank / np.sum(page_rank)

print(f"Dominant eigenvalue: {dominant_eigenvalue}")
print()
print("PageRank scores:")
pages = ['A', 'B', 'C', 'D']
for page, score in zip(pages, page_rank):
    print(f"  Page {page}: {score:.3f}")
\`\`\`

### 3. Principal Component Analysis (Preview)

\`\`\`python
print("\\n=== Application: PCA Preview ===")

# Generate 2D data with correlation
np.random.seed(42)
n = 50
theta = np.pi / 4
X_orig = np.random.randn(n, 2) @ np.diag([3, 1])

# Rotation
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])
X = (R @ X_orig.T).T

# Center data
X_centered = X - X.mean(axis=0)

# Covariance matrix
cov = np.cov(X_centered.T)

# Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(cov)

# Sort by eigenvalue
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("Eigenvalues (principal component variances):")
print(eigenvalues)
print()

# Project onto first principal component
PC1 = eigenvectors[:, 0]
X_projected = X_centered @ PC1.reshape(-1, 1)

print(f"First principal component direction: {PC1}")
print(f"Variance explained: {eigenvalues[0] / np.sum(eigenvalues):.2%}")
print()
print("PCA reduces dimensionality while preserving maximum variance!")
\`\`\`

### 4. Matrix Powers and Markov Chains

\`\`\`python
print("\\n=== Application: Markov Chain Convergence ===")

# Transition matrix for weather model
# States: Sunny, Rainy
P = np.array([[0.7, 0.4],
              [0.3, 0.6]])

print("Transition matrix P:")
print(P)
print()

# Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(P)

print(f"Eigenvalues: {eigenvalues}")
print()

# Steady-state distribution (eigenvector for λ=1)
idx = np.argmin(np.abs(eigenvalues - 1))
steady_state = np.real(eigenvectors[:, idx])
steady_state = steady_state / np.sum(steady_state)

print(f"Steady-state distribution: {steady_state}")
print("Long-run probabilities: {:.1%} Sunny, {:.1%} Rainy".format(
    steady_state[0], steady_state[1]))
print()

# Verify by computing P^100
P_100 = np.linalg.matrix_power(P, 100)
print("P^100 (each column converges to steady state):")
print(P_100)
\`\`\`

## Summary

**Eigenvalues & Eigenvectors**: **Av** = λ**v**
- Special vectors that only get scaled, not rotated

**Finding Eigenvalues**: Solve **det(A - λI) = 0** (characteristic equation)

**Properties**:
- **Trace(A)** = Σλᵢ
- **Det(A)** = Πλᵢ  
- Eigenvectors for distinct eigenvalues are linearly independent

**Eigendecomposition**: **A** = **PDP⁻¹**
- Efficient for computing **Aⁿ**

**Symmetric Matrices**: **A** = **QΛQᵀ** (Spectral Theorem)
- Real eigenvalues
- Orthonormal eigenvectors

**ML Applications**:
- **PCA**: Principal components = eigenvectors of covariance matrix
- **Covariance analysis**: Eigenvalues = variance in principal directions
- **PageRank**: Dominant eigenvector of transition matrix
- **Markov chains**: Steady state = eigenvector for λ = 1
- **Spectral clustering**: Eigenvectors of graph Laplacian
- **Neural network analysis**: Eigenvalues of Hessian → optimization landscape

**Why important in ML**:
1. **Dimensionality reduction**: PCA, Kernel PCA
2. **Data understanding**: Principal directions of variation
3. **Optimization**: Second-order methods (Newton)
4. **Stability analysis**: Eigenvalues of Jacobian/Hessian
5. **Graph algorithms**: Spectral methods
6. **Iterative algorithms**: Convergence rates depend on eigenvalues

Understanding eigenvalues and eigenvectors is essential for modern machine learning!
`,
};
