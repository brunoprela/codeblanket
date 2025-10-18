/**
 * Special Matrices Section
 */

export const specialmatricesSection = {
  id: 'special-matrices',
  title: 'Special Matrices',
  content: `
# Special Matrices

## Introduction

Certain types of matrices have special properties that make them particularly useful or efficient in machine learning. Understanding these matrices helps you recognize patterns, optimize computations, and gain insight into algorithm behavior.

## Diagonal Matrices

A **diagonal matrix** has non-zero elements only on the main diagonal (where row index = column index).

**D** = ⎡d₁  0   0   ...  0 ⎤
      ⎢0   d₂  0   ...  0 ⎥
      ⎢0   0   d₃  ...  0 ⎥
      ⎢...  ... ...  ... ...⎥
      ⎣0   0   0   ... dₙ⎦

**Properties**:
- Multiplication is very fast: O(n) instead of O(n³)
- **Dv** scales each component: (Dv)ᵢ = dᵢvᵢ
- Powers are trivial: **D**ᵏ has elements dᵢᵏ
- Determinant: det(**D**) = d₁d₂...dₙ (product of diagonal elements)
- Inverse: **D⁻¹** has elements 1/dᵢ (if all dᵢ ≠ 0)

\`\`\`python
import numpy as np

print("=== Diagonal Matrices ===")

# Create diagonal matrix
d = np.array([2, 3, 4])
D = np.diag(d)

print("Diagonal elements:", d)
print("\\nDiagonal matrix D:")
print(D)
print()

# Multiply by vector (fast scaling)
v = np.array([1, 1, 1])
Dv = D @ v

print(f"v: {v}")
print(f"Dv: {Dv}")
print("Effect: scales each component by diagonal element")
print()

# Matrix power (trivial for diagonal)
D2 = D @ D
D2_direct = np.diag(d**2)

print("D²:")
print(D2)
print("\\nD² (direct calculation):")
print(D2_direct)
print(f"Equal: {np.allclose(D2, D2_direct)}")
print()

# Inverse (if elements non-zero)
D_inv = np.linalg.inv(D)
D_inv_direct = np.diag(1/d)

print("D⁻¹:")
print(D_inv)
print("\\nD⁻¹ (direct 1/d):")
print(D_inv_direct)
\`\`\`

### Applications in ML

1. **Feature scaling**: Multiply by diagonal matrix to scale features
2. **Learning rate per parameter**: Diagonal preconditioning in optimization
3. **Covariance simplification**: Diagonal covariance assumes independent features

\`\`\`python
# Feature scaling with diagonal matrix
X = np.array([[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]])

# Scale factors for each feature
scale = np.array([0.1, 0.5, 1.0])
S = np.diag(scale)

X_scaled = X @ S  # Scale each column

print("\\n=== Feature Scaling ===")
print("Original X:")
print(X)
print("\\nScaling matrix S:")
print(S)
print("\\nScaled X:")
print(X_scaled)
\`\`\`

## Symmetric Matrices

A matrix **A** is **symmetric** if **A** = **Aᵀ** (equals its transpose).

aᵢⱼ = aⱼᵢ for all i, j

**Properties**:
- All eigenvalues are **real** numbers
- Eigenvectors corresponding to distinct eigenvalues are **orthogonal**
- Can be diagonalized with orthogonal eigenvectors
- Very important in optimization and statistics

\`\`\`python
print("\\n=== Symmetric Matrices ===")

# Create symmetric matrix
A_sym = np.array([[1, 2, 3],
                  [2, 4, 5],
                  [3, 5, 6]])

print("Symmetric matrix A:")
print(A_sym)
print()

# Verify symmetry
print("A transpose:")
print(A_sym.T)
print(f"\\nA == Aᵀ: {np.allclose(A_sym, A_sym.T)}")
print()

# Eigenvalues (will be real)
eigenvalues, eigenvectors = np.linalg.eig(A_sym)
print("Eigenvalues (all real):")
print(eigenvalues.real)
print(f"Imaginary parts (should be ~0): {eigenvalues.imag}")
\`\`\`

### Applications in ML

1. **Covariance matrices**: Always symmetric
2. **Hessian matrices**: Second derivatives in optimization
3. **Kernel matrices**: In SVM and kernel methods
4. **Graph Laplacians**: In graph neural networks

\`\`\`python
# Covariance matrix is always symmetric
print("\\n=== Covariance Matrix (Symmetric) ===")

# Generate random data
np.random.seed(42)
data = np.random.randn(100, 3)

# Compute covariance matrix
cov_matrix = np.cov(data.T)

print("Covariance matrix shape:", cov_matrix.shape)
print("Covariance matrix:")
print(cov_matrix.round(3))
print(f"\\nIs symmetric: {np.allclose(cov_matrix, cov_matrix.T)}")
\`\`\`

## Orthogonal Matrices

A square matrix **Q** is **orthogonal** if its columns are orthonormal (unit vectors that are mutually perpendicular).

**Q**ᵀ**Q** = **QQ**ᵀ = **I**

Equivalently: **Q**ᵀ = **Q⁻¹**

**Properties**:
- Preserves lengths: ||**Qv**|| = ||**v**||
- Preserves angles and dot products: (**Qu**) · (**Qv**) = **u** · **v**
- Determinant is ±1: det(**Q**) = ±1
- Inverse is just transpose (very cheap to compute!)

\`\`\`python
print("\\n=== Orthogonal Matrices ===")

# Example: Rotation matrix (orthogonal)
theta = np.pi / 4  # 45 degrees
Q = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)]
])

print("Rotation matrix Q (45°):")
print(Q)
print()

# Verify orthogonality: QᵀQ = I
QtQ = Q.T @ Q
print("QᵀQ:")
print(QtQ)
print(f"Is identity: {np.allclose(QtQ, np.eye(2))}")
print()

# Verify Q⁻¹ = Qᵀ
Q_inv = np.linalg.inv(Q)
print("Q⁻¹:")
print(Q_inv)
print("\\nQᵀ:")
print(Q.T)
print(f"Q⁻¹ == Qᵀ: {np.allclose(Q_inv, Q.T)}")
print()

# Verify length preservation
v = np.array([3, 4])
Qv = Q @ v

print(f"||v|| = {np.linalg.norm(v):.4f}")
print(f"||Qv|| = {np.linalg.norm(Qv):.4f}")
print("Length preserved!")
\`\`\`

### Applications in ML

1. **QR decomposition**: Orthogonal Q matrix
2. **SVD**: U and V matrices are orthogonal
3. **PCA**: Principal component directions are orthonormal
4. **Orthogonal initialization**: Some neural network weight initializations

## Triangular Matrices

**Upper triangular**: All elements below diagonal are zero
**Lower triangular**: All elements above diagonal are zero

\`\`\`python
print("\\n=== Triangular Matrices ===")

# Upper triangular
U = np.array([[1, 2, 3],
             [0, 4, 5],
             [0, 0, 6]])

# Lower triangular
L = np.array([[1, 0, 0],
             [2, 3, 0],
             [4, 5, 6]])

print("Upper triangular U:")
print(U)
print("\\nLower triangular L:")
print(L)
print()

# Determinant is product of diagonal elements
det_U = np.linalg.det(U)
det_U_diag = U[0,0] * U[1,1] * U[2,2]

print(f"det(U) = {det_U:.4f}")
print(f"Product of diagonal = {det_U_diag:.4f}")
\`\`\`

**Properties**:
- Determinant = product of diagonal elements
- Solving **Ux** = **b** is fast (back substitution)
- Solving **Lx** = **b** is fast (forward substitution)

### Applications in ML

1. **LU decomposition**: Factor **A** = **LU**
2. **Cholesky decomposition**: **A** = **LL**ᵀ for positive definite matrices
3. **Solving linear systems**: Much faster than generic methods

\`\`\`python
# Solving triangular system (fast!)
print("\\n=== Solving Triangular Systems ===")

b = np.array([14, 29, 66])

# Solve Ux = b using back substitution
x_U = np.linalg.solve(U, b)
print(f"Solving Ux = b:")
print(f"x = {x_U}")
print(f"Verification Ux = {U @ x_U}")

# Solve Lx = b using forward substitution  
x_L = np.linalg.solve(L, b)
print(f"\\nSolving Lx = b:")
print(f"x = {x_L}")
print(f"Verification Lx = {L @ x_L}")
\`\`\`

## Sparse Matrices

**Sparse matrices** have mostly zero elements. Instead of storing all elements, store only non-zero values and their positions.

\`\`\`python
from scipy import sparse

print("\\n=== Sparse Matrices ===")

# Create a large sparse matrix
n = 1000
# Most elements are zero
dense_matrix = np.eye(n)
dense_matrix[0, n-1] = 1
dense_matrix[n-1, 0] = 1

print(f"Dense matrix size: {dense_matrix.nbytes / 1024:.2f} KB")
print(f"Number of non-zeros: {np.count_nonzero(dense_matrix)}")
print()

# Convert to sparse (CSR format - Compressed Sparse Row)
sparse_matrix = sparse.csr_matrix(dense_matrix)

print(f"Sparse matrix size: {(sparse_matrix.data.nbytes + sparse_matrix.indices.nbytes + sparse_matrix.indptr.nbytes) / 1024:.2f} KB")
print(f"Space saving: {100 * (1 - sparse_matrix.data.nbytes / dense_matrix.nbytes):.1f}%")
print()

# Operations on sparse matrices
v = np.ones(n)
result_dense = dense_matrix @ v
result_sparse = sparse_matrix @ v

print(f"Results equal: {np.allclose(result_dense, result_sparse)}")
print(f"Result (first 5 elements): {result_sparse[:5]}")
\`\`\`

### Sparse Matrix Formats

\`\`\`python
print("\\n=== Sparse Matrix Formats ===")

# Small example for illustration
data_array = np.array([
    [0, 0, 3, 0],
    [1, 0, 0, 4],
    [0, 2, 0, 0]
])

print("Dense matrix:")
print(data_array)
print()

# COO (Coordinate) format - good for construction
coo = sparse.coo_matrix(data_array)
print("COO format:")
print(f"  Data: {coo.data}")
print(f"  Rows: {coo.row}")
print(f"  Cols: {coo.col}")
print()

# CSR (Compressed Sparse Row) - good for arithmetic and row slicing
csr = sparse.csr_matrix(data_array)
print("CSR format:")
print(f"  Data: {csr.data}")
print(f"  Indices: {csr.indices}")
print(f"  Indptr: {csr.indptr}")
print()

# CSC (Compressed Sparse Column) - good for column slicing
csc = sparse.csc_matrix(data_array)
print("CSC format:")
print(f"  Data: {csc.data}")
print(f"  Indices: {csc.indices}")
print(f"  Indptr: {csc.indptr}")
\`\`\`

### Applications in ML

1. **Text data**: Document-term matrices (mostly zeros)
2. **Recommender systems**: User-item interaction matrices
3. **Graph neural networks**: Adjacency matrices
4. **Feature engineering**: One-hot encoded categorical variables

\`\`\`python
# Example: Text document-term matrix
print("\\n=== Text Example (Sparse) ===")

from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "machine learning is great",
    "deep learning is powerful",
    "machine learning uses statistics",
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)

print(f"Document-term matrix shape: {X.shape}")
print(f"Sparsity: {100 * (1 - X.nnz / (X.shape[0] * X.shape[1])):.1f}% zeros")
print(f"Type: {type(X)}")
print("\\nMatrix (dense view):")
print(X.toarray())
print("\\nVocabulary:")
print(vectorizer.get_feature_names_out())
\`\`\`

## Positive Definite Matrices

A symmetric matrix **A** is **positive definite** if:
- **x**ᵀ**Ax** > 0 for all non-zero vectors **x**
- All eigenvalues are positive
- **A** = **BB**ᵀ for some matrix **B** (Cholesky decomposition)

\`\`\`python
print("\\n=== Positive Definite Matrices ===")

# Create positive definite matrix
B = np.random.randn(3, 3)
A_pd = B.T @ B  # This construction guarantees positive definite

print("Positive definite matrix A:")
print(A_pd)
print()

# Check eigenvalues (all should be positive)
eigenvalues = np.linalg.eigvalsh(A_pd)
print(f"Eigenvalues: {eigenvalues}")
print(f"All positive: {np.all(eigenvalues > 0)}")
print()

# Test definition: xᵀAx > 0
x = np.random.randn(3)
xAx = x.T @ A_pd @ x
print(f"For random x: xᵀAx = {xAx:.4f} > 0: {xAx > 0}")
\`\`\`

### Applications in ML

1. **Covariance matrices**: Always positive semi-definite (or positive definite if full rank)
2. **Kernel matrices**: In kernel methods (SVM, Gaussian processes)
3. **Hessian matrices**: At local minima in convex optimization
4. **Guarantee unique solution**: For **Ax** = **b**

## Comparing Special Matrices

\`\`\`python
print("\\n=== Matrix Properties Comparison ===")

properties = {
    'Diagonal': {'Fast multiply': True, 'Fast inverse': True, 'Real eigenvalues': True},
    'Symmetric': {'Fast multiply': False, 'Fast inverse': False, 'Real eigenvalues': True},
    'Orthogonal': {'Fast multiply': False, 'Fast inverse': True, 'Real eigenvalues': False},
    'Triangular': {'Fast multiply': False, 'Fast inverse': True, 'Real eigenvalues': True},
    'Sparse': {'Fast multiply': True, 'Fast inverse': False, 'Real eigenvalues': False},
}

import pandas as pd
df = pd.DataFrame(properties).T
print(df)
\`\`\`

## Summary

Special matrices optimize computation and provide theoretical guarantees:

1. **Diagonal**: Fastest operations, feature scaling
2. **Symmetric**: Real eigenvalues, covariance matrices, optimization
3. **Orthogonal**: Preserve lengths and angles, rotations, SVD/QR
4. **Triangular**: Fast system solving, LU/Cholesky decompositions
5. **Sparse**: Memory efficient, text/graph data
6. **Positive Definite**: Convex optimization, covariance, kernels

**Key Insights**:
- Recognize matrix structure to choose optimal algorithms
- Special matrices often guarantee certain properties (e.g., real eigenvalues)
- Sparse matrices are essential for large-scale ML
- Many ML matrices have special structure (covariance is symmetric, adjacency is sparse)

**Performance**:
- Diagonal matrix multiply: O(n) vs O(n³) for general matrices
- Orthogonal matrix inverse: O(n) (just transpose) vs O(n³)
- Sparse operations: O(nnz) where nnz << n²

Recognizing and exploiting special matrix structure is a key skill for efficient ML implementations!
`,
};
