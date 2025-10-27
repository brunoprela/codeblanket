/**
 * Matrix Decompositions Section
 */

export const matrixdecompositionsSection = {
  id: 'matrix-decompositions',
  title: 'Matrix Decompositions',
  content: `
# Matrix Decompositions

## Introduction

**Matrix decompositions** (or factorizations) express a matrix as a product of simpler matrices. These are fundamental to numerical linear algebra, enabling efficient computation and revealing matrix structure.

**Why decompose**?
1. **Numerical stability**: Solve systems more accurately
2. **Efficiency**: Faster computation for repeated operations
3. **Insight**: Reveal geometric/algebraic structure
4. **Applications**: Least squares, dimensionality reduction, data compression

We'll cover: **LU**, **QR**, **Cholesky**, and **SVD** decompositions.

## LU Decomposition

**LU decomposition** factors **A** into:

**A** = **LU**

Where:
- **L**: Lower triangular matrix (1s on diagonal)
- **U**: Upper triangular matrix

**Use case**: Solving **Ax** = **b** for multiple **b** efficiently.

\`\`\`python
import numpy as np
from scipy.linalg import lu

print("=== LU Decomposition ===")

A = np.array([[2, 1, 1],
              [4, 3, 3],
              [8, 7, 9]], dtype=float)

print("Matrix A:")
print(A)
print()

# Compute LU decomposition
P, L, U = lu(A)

print("L (lower triangular):")
print(L)
print()

print("U (upper triangular):")
print(U)
print()

print("P (permutation matrix):")
print(P)
print()

# Verify: PA = LU
PA = P @ A
LU_prod = L @ U

print("PA:")
print(PA)
print()

print("LU:")
print(LU_prod)
print()

print(f"PA = LU: {np.allclose(PA, LU_prod)}")
\`\`\`

### Solving Systems with LU

**Ax** = **b** becomes **LUx** = **b**

**Two steps**:
1. **Forward substitution**: Solve **Ly** = **b** for **y**2. **Back substitution**: Solve **Ux** = **y** for **x**

\`\`\`python
print("\\n=== Solving Ax = b with LU ===")

b1 = np.array([2, 4, 12])
b2 = np.array([1, 5, 15])

print(f"b1 = {b1}")
print(f"b2 = {b2}")
print()

# Solve for b1
from scipy.linalg import solve_triangular

# Step 1: Solve Ly = Pb
Pb1 = P @ b1
y1 = solve_triangular(L, Pb1, lower=True)

# Step 2: Solve Ux = y
x1 = solve_triangular(U, y1, lower=False)

print(f"Solution for b1: x1 = {x1}")
print(f"Verify Ax1 = b1: {np.allclose(A @ x1, b1)}")
print()

# Solve for b2 (reusing L and U!)
Pb2 = P @ b2
y2 = solve_triangular(L, Pb2, lower=True)
x2 = solve_triangular(U, y2, lower=False)

print(f"Solution for b2: x2 = {x2}")
print(f"Verify Ax2 = b2: {np.allclose(A @ x2, b2)}")
print()

print("✓ Reusing LU decomposition is much faster for multiple systems!")
\`\`\`

## QR Decomposition

**QR decomposition** factors **A** into:

**A** = **QR**

Where:
- **Q**: Orthogonal matrix (**QᵀQ** = **I**)
- **R**: Upper triangular matrix

**Use case**: Numerically stable solution of least squares problems.

\`\`\`python
print("\\n=== QR Decomposition ===")

A = np.array([[1, 1, 0],
              [1, 0, 1],
              [0, 1, 1]], dtype=float)

print("Matrix A:")
print(A)
print()

# Compute QR decomposition
Q, R = np.linalg.qr(A)

print("Q (orthogonal matrix):")
print(Q)
print()

print("R (upper triangular):")
print(R)
print()

# Verify: A = QR
QR_prod = Q @ R

print("QR:")
print(QR_prod)
print()

print(f"A = QR: {np.allclose(A, QR_prod)}")
print()

# Verify Q is orthogonal
Q_T_Q = Q.T @ Q

print("QᵀQ:")
print(Q_T_Q)
print(f"Is identity: {np.allclose(Q_T_Q, np.eye(3))}")
\`\`\`

### Solving Least Squares with QR

For overdetermined system **Ax** = **b** (more equations than unknowns):

**QRx** = **b**
**Rx** = **Qᵀb** (multiply both sides by **Qᵀ**)

Solve upper triangular system **Rx** = **Qᵀb** by back substitution.

\`\`\`python
print("\\n=== Least Squares with QR ===")

# Overdetermined system (4 equations, 2 unknowns)
A_over = np.array([[1, 1],
                   [1, 2],
                   [1, 3],
                   [1, 4]], dtype=float)

b_over = np.array([6, 5, 7, 10], dtype=float)

print("Overdetermined system:")
print(f"A shape: {A_over.shape}")
print(f"b shape: {b_over.shape}")
print()

# QR decomposition
Q_over, R_over = np.linalg.qr(A_over)

print(f"Q shape: {Q_over.shape}")
print(f"R shape: {R_over.shape}")
print()

# Solve Rx = Qᵀb
Q_T_b = Q_over.T @ b_over
x_qr = solve_triangular(R_over[:2, :], Q_T_b[:2], lower=False)

print(f"Least squares solution: x = {x_qr}")
print()

# Compare with np.linalg.lstsq
x_lstsq, _, _, _ = np.linalg.lstsq(A_over, b_over, rcond=None)

print(f"np.linalg.lstsq solution: {x_lstsq}")
print(f"Equal: {np.allclose (x_qr, x_lstsq)}")
print()

# Compute residual
residual = b_over - A_over @ x_qr
print(f"Residual norm: {np.linalg.norm (residual):.6f}")
\`\`\`

## Cholesky Decomposition

For **symmetric positive definite** matrix **A**:

**A** = **LLᵀ**

Where **L** is lower triangular with positive diagonal entries.

**Properties**:
- More efficient than LU (half the operations)
- Only works for positive definite matrices
- Commonly used for covariance matrices

\`\`\`python
print("\\n=== Cholesky Decomposition ===")

# Create a positive definite matrix
A_spd = np.array([[4, 2, 2],
                  [2, 5, 3],
                  [2, 3, 6]], dtype=float)

print("Symmetric positive definite matrix A:")
print(A_spd)
print()

# Check positive definiteness
eigenvalues = np.linalg.eigvals(A_spd)
print(f"Eigenvalues: {eigenvalues}")
print(f"All positive: {np.all (eigenvalues > 0)}")
print()

# Compute Cholesky decomposition
L_chol = np.linalg.cholesky(A_spd)

print("L (lower triangular):")
print(L_chol)
print()

# Verify: A = LLᵀ
L_L_T = L_chol @ L_chol.T

print("LLᵀ:")
print(L_L_T)
print()

print(f"A = LLᵀ: {np.allclose(A_spd, L_L_T)}")
\`\`\`

### Solving Systems with Cholesky

\`\`\`python
print("\\n=== Solving Ax = b with Cholesky ===")

b_chol = np.array([12, 15, 17])

print(f"b = {b_chol}")
print()

# Step 1: Solve Ly = b (forward substitution)
y_chol = solve_triangular(L_chol, b_chol, lower=True)

# Step 2: Solve Lᵀx = y (back substitution)
x_chol = solve_triangular(L_chol.T, y_chol, lower=False)

print(f"Solution: x = {x_chol}")
print(f"Verify Ax = b: {np.allclose(A_spd @ x_chol, b_chol)}")
print()

print("Cholesky is ~2x faster than LU for positive definite matrices!")
\`\`\`

## Singular Value Decomposition (SVD)

**SVD** is the most powerful decomposition. For any **A** (m × n):

**A** = **UΣVᵀ**

Where:
- **U**: m × m orthogonal matrix (left singular vectors)
- **Σ**: m × n diagonal matrix (singular values σ₁ ≥ σ₂ ≥ ... ≥ 0)
- **V**: n × n orthogonal matrix (right singular vectors)

**Key properties**:
- Works for ANY matrix (rectangular, singular, etc.)
- Reveals rank, null space, column space
- Optimal low-rank approximation
- Foundation of PCA

\`\`\`python
print("\\n=== Singular Value Decomposition (SVD) ===")

A_svd = np.array([[3, 1, 1],
                  [2, 1, 0],
                  [2, 0, 1],
                  [3, 1, 0]], dtype=float)

print("Matrix A (4×3):")
print(A_svd)
print()

# Compute SVD
U, S, Vt = np.linalg.svd(A_svd, full_matrices=True)

print(f"U shape: {U.shape}")
print(f"S (singular values): {S}")
print(f"Vt shape: {Vt.shape}")
print()

# Reconstruct Sigma as matrix
Sigma = np.zeros((A_svd.shape[0], A_svd.shape[1]))
Sigma[:S.shape[0], :S.shape[0]] = np.diag(S)

print("Σ (as matrix):")
print(Sigma)
print()

# Verify: A = UΣVᵀ
A_reconstructed = U @ Sigma @ Vt

print("Reconstructed A:")
print(A_reconstructed)
print()

print(f"A = UΣVᵀ: {np.allclose(A_svd, A_reconstructed)}")
print()

# Verify orthogonality
print(f"UᵀU = I: {np.allclose(U.T @ U, np.eye(U.shape[0]))}")
print(f"VVᵀ = I: {np.allclose(Vt @ Vt.T, np.eye(Vt.shape[0]))}")
\`\`\`

### Low-Rank Approximation with SVD

Keep only top **k** singular values for best rank-k approximation.

\`\`\`python
print("\\n=== Low-Rank Approximation ===")

# Create a larger matrix
np.random.seed(42)
A_large = np.random.randn(5, 5)

U_l, S_l, Vt_l = np.linalg.svd(A_large)

print("Singular values:")
print(S_l)
print()

# Rank-2 approximation
k = 2
Sigma_k = np.zeros((k, k))
Sigma_k[:k, :k] = np.diag(S_l[:k])

A_rank2 = U_l[:, :k] @ Sigma_k @ Vt_l[:k, :]

print(f"Original rank: {np.linalg.matrix_rank(A_large)}")
print(f"Approximation rank: {np.linalg.matrix_rank(A_rank2)}")
print()

# Error
error = np.linalg.norm(A_large - A_rank2, 'fro')
print(f"Frobenius norm error: {error:.6f}")
print(f"Error is σ₃² + σ₄² + σ₅²")
print(f"Computed: {np.sqrt (np.sum(S_l[k:]**2)):.6f}")
\`\`\`

### SVD for Pseudoinverse

For non-square or singular matrices, SVD computes Moore-Penrose pseudoinverse:

**A⁺** = **VΣ⁺Uᵀ**

Where **Σ⁺** has 1/σᵢ on diagonal (0 if σᵢ = 0).

\`\`\`python
print("\\n=== Pseudoinverse with SVD ===")

A_pseudo = np.array([[1, 2],
                     [3, 4],
                     [5, 6]])

print("Matrix A (3×2):")
print(A_pseudo)
print()

# SVD
U_p, S_p, Vt_p = np.linalg.svd(A_pseudo, full_matrices=False)

# Pseudoinverse: Σ⁺ = diag(1/σ₁, 1/σ₂)
Sigma_plus = np.diag(1 / S_p)

A_plus = Vt_p.T @ Sigma_plus @ U_p.T

print("A⁺ (pseudoinverse):")
print(A_plus)
print()

# Compare with np.linalg.pinv
A_plus_np = np.linalg.pinv(A_pseudo)

print("np.linalg.pinv:")
print(A_plus_np)
print()

print(f"Equal: {np.allclose(A_plus, A_plus_np)}")
\`\`\`

## Comparison of Decompositions

\`\`\`python
print("\\n=== Comparison of Decompositions ===")

comparison = """
| Decomposition | Form      | Requirements           | Use Case                        |
|---------------|-----------|------------------------|---------------------------------|
| LU            | A = LU    | Square, invertible*    | Solving Ax=b (multiple b)       |
| QR            | A = QR    | Any shape              | Least squares (stable)          |
| Cholesky      | A = LLᵀ   | Symmetric pos. def.    | Covariance matrices, fast solve |
| Eigendecomp   | A = PDP⁻¹ | Square, full rank*     | PCA, matrix powers              |
| SVD           | A = UΣVᵀ  | Any shape              | Dimensionality reduction, PCA   |

*With pivoting, LU works for all square matrices
*Not all matrices are diagonalizable

**Computational Cost** (n×n matrix):
- LU: O(n³) decomposition, O(n²) per solve
- QR: O(mn²) for m×n matrix
- Cholesky: O(n³/3) (half of LU)
- Eigen: O(n³)
- SVD: O(mn²) for m×n matrix (most expensive)

**Numerical Stability** (best to worst):
1. SVD (most stable)
2. QR
3. Cholesky (for SPD matrices)
4. LU with pivoting
5. LU without pivoting (least stable)
"""

print(comparison)
\`\`\`

## Applications in Machine Learning

### 1. Linear Regression via QR

\`\`\`python
print("\\n=== Application: Linear Regression ===")

from sklearn.datasets import make_regression

X_reg, y_reg = make_regression (n_samples=100, n_features=5, noise=10, random_state=42)

print(f"X shape: {X_reg.shape}")
print(f"y shape: {y_reg.shape}")
print()

# Add intercept column
X_reg_intercept = np.column_stack([np.ones(X_reg.shape[0]), X_reg])

# Solve using QR
Q_reg, R_reg = np.linalg.qr(X_reg_intercept)
coeffs_qr = solve_triangular(R_reg, Q_reg.T @ y_reg, lower=False)

print(f"Coefficients: {coeffs_qr}")
print()

# Compare with normal equations
coeffs_normal = np.linalg.inv(X_reg_intercept.T @ X_reg_intercept) @ X_reg_intercept.T @ y_reg

print(f"Normal equations: {coeffs_normal}")
print(f"QR more stable than normal equations: {np.allclose (coeffs_qr, coeffs_normal)}")
\`\`\`

### 2. Data Compression via SVD

\`\`\`python
print("\\n=== Application: Image Compression (Simulated) ===")

# Simulate grayscale image as matrix
np.random.seed(42)
image = np.random.rand(50, 50)

print(f"Original image shape: {image.shape}")
print(f"Original size: {image.size} values")
print()

# SVD
U_img, S_img, Vt_img = np.linalg.svd (image, full_matrices=False)

# Compress: keep only top k singular values
k_vals = [5, 10, 20]

for k in k_vals:
    compressed = U_img[:, :k] @ np.diag(S_img[:k]) @ Vt_img[:k, :]
    error = np.linalg.norm (image - compressed, 'fro') / np.linalg.norm (image, 'fro')
    
    # Storage: U[:, :k] + S[:k] + Vt[:k, :]
    storage = 50*k + k + k*50
    compression_ratio = image.size / storage
    
    print(f"Rank-{k} approximation:")
    print(f"  Relative error: {error:.4f}")
    print(f"  Storage: {storage} values")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    print()
\`\`\`

## Summary

**Matrix Decompositions** factor matrices into simpler forms:

**LU Decomposition**: **A** = **LU**
- Efficient for solving multiple systems
- Forward + back substitution

**QR Decomposition**: **A** = **QR**
- Numerically stable
- Best for least squares problems

**Cholesky Decomposition**: **A** = **LLᵀ**
- For symmetric positive definite matrices
- Fastest option (2× LU)
- Used for covariance matrices

**SVD**: **A** = **UΣVᵀ**
- Works for ANY matrix
- Optimal low-rank approximation
- Foundation of PCA, dimensionality reduction

**Choosing decomposition**:
- **Multiple solves, same A**: LU or Cholesky
- **Least squares**: QR or SVD
- **Positive definite**: Cholesky (fastest)
- **Dimensionality reduction**: SVD
- **Numerical stability critical**: QR or SVD

**ML Applications**:
- Linear regression: QR decomposition
- PCA: SVD of centered data matrix (or eigendecomposition of covariance)
- Recommender systems: SVD for matrix completion
- Image compression: Truncated SVD
- Numerical stability: QR/SVD over normal equations

Understanding these decompositions is essential for implementing robust ML algorithms!
`,
};
