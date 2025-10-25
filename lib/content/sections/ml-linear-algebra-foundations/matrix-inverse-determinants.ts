/**
 * Matrix Inverse & Determinants Section
 */

export const matrixinversedeterminantsSection = {
  id: 'matrix-inverse-determinants',
  title: 'Matrix Inverse & Determinants',
  content: `
# Matrix Inverse & Determinants

## Introduction

The **determinant** and **inverse** are fundamental concepts in linear algebra with deep geometric interpretations and critical applications in machine learning. Understanding them helps you solve systems of equations, analyze transformations, and diagnose numerical stability issues.

## Determinant

The **determinant** is a scalar value that encodes important properties of a square matrix.

### 2×2 Determinant

For a 2×2 matrix:

**A** = ⎡a  b⎤
      ⎣c  d⎦

det(**A**) = |**A**| = ad - bc

### 3×3 Determinant

For a 3×3 matrix, use cofactor expansion:

**A** = ⎡a  b  c⎤
      ⎢d  e  f⎥
      ⎣g  h  i⎦

det(**A**) = a (ei - fh) - b (di - fg) + c (dh - eg)

### General Case

For larger matrices, use recursive cofactor expansion or more efficient algorithms (LU decomposition).

\`\`\`python
import numpy as np

print("=== Determinants ===")

# 2x2 matrix
A_2x2 = np.array([[3, 2],
                  [1, 4]])

det_2x2 = np.linalg.det(A_2x2)
det_2x2_manual = A_2x2[0,0]*A_2x2[1,1] - A_2x2[0,1]*A_2x2[1,0]

print("2×2 Matrix A:")
print(A_2x2)
print(f"\\ndet(A) = {det_2x2:.4f}")
print(f"det(A) manual = {A_2x2[0,0]}*{A_2x2[1,1]} - {A_2x2[0,1]}*{A_2x2[1,0]} = {det_2x2_manual:.4f}")
print()

# 3x3 matrix
A_3x3 = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 10]])

det_3x3 = np.linalg.det(A_3x3)

print("3×3 Matrix A:")
print(A_3x3)
print(f"det(A) = {det_3x3:.4f}")
\`\`\`

### Properties of Determinants

\`\`\`python
print("\\n=== Determinant Properties ===")

A = np.array([[2, 1], [3, 4]])
B = np.array([[1, 2], [0, 3]])

print("Matrix A:")
print(A)
print(f"det(A) = {np.linalg.det(A):.4f}")
print()

print("Matrix B:")
print(B)
print(f"det(B) = {np.linalg.det(B):.4f}")
print()

# Property 1: det(AB) = det(A)det(B)
AB = A @ B
det_A = np.linalg.det(A)
det_B = np.linalg.det(B)
det_AB = np.linalg.det(AB)

print(f"det(A) * det(B) = {det_A:.4f} * {det_B:.4f} = {det_A * det_B:.4f}")
print(f"det(AB) = {det_AB:.4f}")
print(f"Equal: {np.isclose (det_A * det_B, det_AB)}")
print()

# Property 2: det(Aᵀ) = det(A)
det_A_T = np.linalg.det(A.T)
print(f"det(A) = {det_A:.4f}")
print(f"det(Aᵀ) = {det_A_T:.4f}")
print(f"Equal: {np.isclose (det_A, det_A_T)}")
print()

# Property 3: det (cA) = c^n det(A) for n×n matrix
c = 2
cA = c * A
det_cA = np.linalg.det (cA)
n = A.shape[0]
expected = c**n * det_A

print(f"For scalar c={c} and {n}×{n} matrix A:")
print(f"det (cA) = {det_cA:.4f}")
print(f"c^n * det(A) = {c}^{n} * {det_A:.4f} = {expected:.4f}")
print(f"Equal: {np.isclose (det_cA, expected)}")
\`\`\`

### Geometric Interpretation

The **absolute value** of the determinant represents:
- The **volume scaling factor** of the transformation
- In 2D: Area scaling factor
- In 3D: Volume scaling factor

**Sign** of determinant:
- Positive: Orientation preserved
- Negative: Orientation reversed (reflection)
- Zero: Collapses space (singular, not invertible)

\`\`\`python
import matplotlib.pyplot as plt

print("\\n=== Geometric Interpretation ===")

# Unit square
square = np.array([[0, 1, 1, 0, 0],
                   [0, 0, 1, 1, 0]])

# Transformation matrix
A_transform = np.array([[2, 0.5],
                        [0.5, 1.5]])

transformed_square = A_transform @ square

# Determinant = area scaling factor
det_transform = np.linalg.det(A_transform)
original_area = 1.0  # Unit square
transformed_area = det_transform * original_area

print("Transformation matrix:")
print(A_transform)
print(f"\\ndet(A) = {det_transform:.4f}")
print(f"Original area: {original_area}")
print(f"Transformed area: {transformed_area:.4f}")
print(f"Area scaled by factor of {det_transform:.4f}")

# Visualize
plt.figure (figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot (square[0, :], square[1, :], 'b-', linewidth=2)
plt.fill (square[0, :], square[1, :], alpha=0.3)
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.xlim(-0.5, 3)
plt.ylim(-0.5, 3)
plt.title (f'Original Square (Area = {original_area})')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(1, 2, 2)
plt.plot (transformed_square[0, :], transformed_square[1, :], 'r-', linewidth=2)
plt.fill (transformed_square[0, :], transformed_square[1, :], alpha=0.3, color='red')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.xlim(-0.5, 3)
plt.ylim(-0.5, 3)
plt.title (f'Transformed (Area = {transformed_area:.2f}, det = {det_transform:.2f})')
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()
plt.show()
\`\`\`

### Singular Matrices (Determinant = 0)

When det(**A**) = 0, the matrix is **singular** (not invertible):
- Collapses space to lower dimension
- Rows/columns are linearly dependent
- No unique solutions to **Ax** = **b**

\`\`\`python
print("\\n=== Singular Matrix ===")

# Singular matrix (second row is 2× first row)
A_singular = np.array([[1, 2],
                       [2, 4]])

det_singular = np.linalg.det(A_singular)

print("Singular matrix A:")
print(A_singular)
print(f"det(A) = {det_singular:.10f}")
print(f"Is singular: {np.isclose (det_singular, 0)}")
print()

# Transform unit square (collapses to line)
transformed_singular = A_singular @ square

plt.figure (figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot (square[0, :], square[1, :], 'b-', linewidth=2)
plt.fill (square[0, :], square[1, :], alpha=0.3)
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.xlim(-0.5, 5)
plt.ylim(-0.5, 5)
plt.title('Original Square')

plt.subplot(1, 2, 2)
plt.plot (transformed_singular[0, :], transformed_singular[1, :], 'r-', linewidth=3)
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.xlim(-0.5, 5)
plt.ylim(-0.5, 5)
plt.title('Collapsed to Line (det = 0)')

plt.tight_layout()
plt.show()

print("Note: The square collapsed to a line!")
\`\`\`

## Matrix Inverse

The **inverse** of a matrix **A** is **A⁻¹** such that:

**AA⁻¹** = **A⁻¹A** = **I**

### When Does an Inverse Exist?

A matrix is **invertible** (non-singular) if and only if:
- det(**A**) ≠ 0
- Columns are linearly independent
- Rows are linearly independent
- **A** has full rank

### 2×2 Inverse Formula

For 2×2 matrix:

**A** = ⎡a  b⎤
      ⎣c  d⎦

**A⁻¹** = (1/det(A)) ⎡d  -b⎤
                     ⎣-c   a⎦

\`\`\`python
print("\\n=== Matrix Inverse ===")

A = np.array([[3, 2],
              [1, 4]])

# Compute inverse
A_inv = np.linalg.inv(A)

print("Matrix A:")
print(A)
print("\\nInverse A⁻¹:")
print(A_inv)
print()

# Verify AA⁻¹ = I
I_check = A @ A_inv
print("A @ A⁻¹:")
print(I_check)
print(f"Is identity: {np.allclose(I_check, np.eye(2))}")
print()

# Manual 2×2 inverse
det_A = np.linalg.det(A)
A_inv_manual = (1/det_A) * np.array([[A[1,1], -A[0,1]],
                                     [-A[1,0], A[0,0]]])

print("A⁻¹ (manual calculation):")
print(A_inv_manual)
print(f"Matches numpy: {np.allclose(A_inv, A_inv_manual)}")
\`\`\`

### Properties of Inverse

\`\`\`python
print("\\n=== Inverse Properties ===")

A = np.array([[2, 1], [1, 3]])
B = np.array([[1, 2], [0, 1]])

A_inv = np.linalg.inv(A)
B_inv = np.linalg.inv(B)

# Property 1: (A⁻¹)⁻¹ = A
A_inv_inv = np.linalg.inv(A_inv)
print("(A⁻¹)⁻¹ = A:")
print(f"Equal: {np.allclose(A_inv_inv, A)}")
print()

# Property 2: (AB)⁻¹ = B⁻¹A⁻¹ (order reverses!)
AB = A @ B
AB_inv = np.linalg.inv(AB)
B_inv_A_inv = B_inv @ A_inv

print("(AB)⁻¹ = B⁻¹A⁻¹ (note the reversal!):")
print(f"Equal: {np.allclose(AB_inv, B_inv_A_inv)}")
print()

# Property 3: (Aᵀ)⁻¹ = (A⁻¹)ᵀ
A_T_inv = np.linalg.inv(A.T)
A_inv_T = A_inv.T

print("(Aᵀ)⁻¹ = (A⁻¹)ᵀ:")
print(f"Equal: {np.allclose(A_T_inv, A_inv_T)}")
print()

# Property 4: det(A⁻¹) = 1/det(A)
det_A = np.linalg.det(A)
det_A_inv = np.linalg.det(A_inv)

print(f"det(A) = {det_A:.4f}")
print(f"det(A⁻¹) = {det_A_inv:.4f}")
print(f"1/det(A) = {1/det_A:.4f}")
print(f"Equal: {np.isclose (det_A_inv, 1/det_A)}")
\`\`\`

### Geometric Interpretation

The inverse transformation **undoes** the original transformation:
- **A** transforms **x** to **y** = **Ax**
- **A⁻¹** transforms **y** back to **x** = **A⁻¹y**

\`\`\`python
print("\\n=== Inverse as Undo Operation ===")

# Original vector
x = np.array([1, 2])

# Transform
A = np.array([[2, 1], [0, 3]])
y = A @ x

# Inverse transform (undo)
A_inv = np.linalg.inv(A)
x_recovered = A_inv @ y

print(f"Original x: {x}")
print(f"Transformed y = Ax: {y}")
print(f"Recovered x = A⁻¹y: {x_recovered}")
print(f"Recovered correctly: {np.allclose (x, x_recovered)}")
\`\`\`

## Applications in Machine Learning

### 1. Solving Linear Systems

Instead of solving **Ax** = **b** directly, use **x** = **A⁻¹b**:

\`\`\`python
print("\\n=== Solving Linear Systems ===")

# System: Ax = b
A = np.array([[3, 2],
              [1, 4]])
b = np.array([7, 13])

# Method 1: Using inverse (not recommended in practice)
A_inv = np.linalg.inv(A)
x_inv = A_inv @ b

# Method 2: Using solver (more stable)
x_solve = np.linalg.solve(A, b)

print("System: Ax = b")
print("A:")
print(A)
print(f"b: {b}")
print()

print(f"Solution (using inverse): x = {x_inv}")
print(f"Solution (using solver): x = {x_solve}")
print(f"Verification Ax = {A @ x_solve}")
print()

print("Note: np.linalg.solve() is more numerically stable!")
\`\`\`

### 2. Computing Covariance Matrix Inverse

In Gaussian distributions and many ML algorithms:

\`\`\`python
print("\\n=== Covariance Matrix Inverse ===")

# Generate data
np.random.seed(42)
data = np.random.randn(100, 3)

# Covariance matrix
cov = np.cov (data.T)
cov_inv = np.linalg.inv (cov)

print("Covariance matrix Σ:")
print(cov.round(3))
print("\\nInverse covariance Σ⁻¹ (precision matrix):")
print(cov_inv.round(3))
print()

# Used in Mahalanobis distance
x = np.array([1, 0, -1])
mean = np.zeros(3)

# Mahalanobis distance: sqrt((x-μ)ᵀ Σ⁻¹ (x-μ))
diff = x - mean
mahal_dist = np.sqrt (diff @ cov_inv @ diff)

print(f"Point: {x}")
print(f"Mahalanobis distance: {mahal_dist:.4f}")
print("(Used in anomaly detection, Gaussian processes)")
\`\`\`

### 3. Pseudo-inverse for Non-square Matrices

For non-square or singular matrices, use the **Moore-Penrose pseudo-inverse**:

\`\`\`python
print("\\n=== Pseudo-inverse ===")

# Non-square matrix (more rows than columns)
A_rect = np.array([[1, 2],
                   [3, 4],
                   [5, 6]])

# Pseudo-inverse
A_pinv = np.linalg.pinv(A_rect)

print(f"A shape: {A_rect.shape}")
print("A:")
print(A_rect)
print(f"\\nA⁺ (pseudo-inverse) shape: {A_pinv.shape}")
print("A⁺:")
print(A_pinv)
print()

# Property: AA⁺A = A
AApinvA = A_rect @ A_pinv @ A_rect
print("AA⁺A = A:")
print(f"Equal: {np.allclose(AApinvA, A_rect)}")
print()

# Use in least squares
b = np.array([1, 2, 3])
x_ls = A_pinv @ b

print(f"\\nLeast squares solution x = A⁺b:")
print(f"x = {x_ls}")
print(f"Ax ≈ b: {A_rect @ x_ls}")
print(f"b: {b}")
print(f"Residual: {np.linalg.norm(A_rect @ x_ls - b):.6f}")
\`\`\`

## Numerical Considerations

### Condition Number

The **condition number** measures how sensitive a matrix is to numerical errors:

κ(**A**) = ||**A**|| · ||**A⁻¹**||

- κ = 1: Perfectly conditioned (e.g., identity matrix)
- κ large: Ill-conditioned (small changes → big errors)
- κ = ∞: Singular (not invertible)

\`\`\`python
print("\\n=== Condition Number ===")

# Well-conditioned matrix
A_good = np.eye(3)
cond_good = np.linalg.cond(A_good)

print("Well-conditioned (identity):")
print(f"κ(I) = {cond_good:.4f}")
print()

# Ill-conditioned matrix
A_bad = np.array([[1, 1],
                  [1, 1.0001]])
cond_bad = np.linalg.cond(A_bad)

print("Ill-conditioned:")
print(A_bad)
print(f"κ(A) = {cond_bad:.4f}")
print()

# Nearly singular
A_nearly_singular = np.array([[1, 1],
                              [1, 1.000001]])
cond_nearly_singular = np.linalg.cond(A_nearly_singular)

print("Nearly singular:")
print(f"κ(A) = {cond_nearly_singular:.4f}")
print("Very large condition number → numerical instability!")
\`\`\`

### Best Practices

\`\`\`python
print("\\n=== Best Practices ===")

# DON'T: Compute inverse explicitly
A = np.random.randn(100, 100)
b = np.random.randn(100)

# Slow and less stable
import time
start = time.time()
A_inv = np.linalg.inv(A)
x_bad = A_inv @ b
time_inv = time.time() - start

# DO: Use solver
start = time.time()
x_good = np.linalg.solve(A, b)
time_solve = time.time() - start

print(f"Using inverse: {time_inv:.6f}s")
print(f"Using solver: {time_solve:.6f}s")
print(f"Solver is {time_inv/time_solve:.2f}x faster")
print()
print(f"Solutions equal: {np.allclose (x_bad, x_good)}")
print()

print("Remember: Never compute A⁻¹ explicitly unless you need it!")
print("Use np.linalg.solve(A, b) instead of np.linalg.inv(A) @ b")
\`\`\`

## Summary

**Determinant**:
- Scalar that encodes matrix properties
- Geometric: Volume scaling factor
- det(**A**) = 0 ↔ singular (not invertible)
- det(**AB**) = det(**A**)·det(**B**)

**Inverse**:
- **AA⁻¹** = **I**
- Exists iff det(**A**) ≠ 0
- Geometric: Undoes transformation
- (**AB**)⁻¹ = **B⁻¹A⁻¹** (order reverses!)

**Applications in ML**:
- Solving linear systems
- Computing Mahalanobis distance
- Covariance inverse (precision matrix)
- Least squares (pseudo-inverse)

**Best Practices**:
- Don't compute inverse explicitly—use \`np.linalg.solve()\`
- Check condition number for numerical stability
- Use pseudo-inverse for non-square/singular matrices
- Be aware of computational cost: O(n³)

Understanding determinants and inverses is essential for linear algebra in ML!
`,
};
