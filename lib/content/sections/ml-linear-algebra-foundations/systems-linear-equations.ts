/**
 * Systems of Linear Equations Section
 */

export const systemslinearequationsSection = {
  id: 'systems-linear-equations',
  title: 'Systems of Linear Equations',
  content: `
# Systems of Linear Equations

## Introduction

A system of linear equations is a collection of linear equations involving the same set of variables. Solving these systems is fundamental to many machine learning algorithms, from linear regression to training neural networks.

**General form**:
a₁₁x₁ + a₁₂x₂ + ... + a₁ₙxₙ = b₁
a₂₁x₁ + a₂₂x₂ + ... + a₂ₙxₙ = b₂
...
aₘ₁x₁ + aₘ₂x₂ + ... + aₘₙxₙ = bₘ

**Matrix form**: **Ax** = **b**

Where:
- **A**: m × n coefficient matrix
- **x**: n × 1 solution vector  
- **b**: m × 1 constant vector

## Types of Systems

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt

print("=== Types of Linear Systems ===")

# Case 1: Unique solution (m = n, det(A) ≠ 0)
print("\\n1. UNIQUE SOLUTION (Square, non-singular)")
A1 = np.array([[2, 1], [1, 3]])
b1 = np.array([5, 7])
x1 = np.linalg.solve(A1, b1)

print(f"A: \\n{A1}")
print(f"b: {b1}")
print(f"Solution x: {x1}")
print(f"Verification Ax = {A1 @ x1}")
print(f"det(A) = {np.linalg.det(A1):.4f} ≠ 0 → unique solution")

# Case 2: No solution (inconsistent)
print("\\n2. NO SOLUTION (Inconsistent)")
A2 = np.array([[1, 2], [2, 4]])  # Singular
b2 = np.array([3, 7])  # Not in column space
print(f"A: \\n{A2}")
print(f"b: {b2}")
print(f"det(A) = {np.linalg.det(A2):.10f} ≈ 0")
print("Rows are linearly dependent but b is not in column space")
print("→ No solution exists")

# Case 3: Infinite solutions
print("\\n3. INFINITE SOLUTIONS")
A3 = np.array([[1, 2], [2, 4]])
b3 = np.array([3, 6])  # In column space (b3 = 2*b3[0])
print(f"A: \\n{A3}")
print(f"b: {b3}")
print("b is in the column space of A")
print("→ Infinitely many solutions")

# Case 4: Overdetermined (m > n)
print("\\n4. OVERDETERMINED (More equations than unknowns)")
A4 = np.array([[1, 2], [3, 4], [5, 6]])
b4 = np.array([1, 2, 3])
print(f"A shape: {A4.shape} (3 equations, 2 unknowns)")
print("Usually no exact solution → use least squares")
x4_ls = np.linalg.lstsq(A4, b4, rcond=None)[0]
print(f"Least squares solution: {x4_ls}")
print(f"Residual: {np.linalg.norm(A4 @ x4_ls - b4):.6f}")

# Case 5: Underdetermined (m < n)
print("\\n5. UNDERDETERMINED (More unknowns than equations)")
A5 = np.array([[1, 2, 3], [4, 5, 6]])
b5 = np.array([1, 2])
print(f"A shape: {A5.shape} (2 equations, 3 unknowns)")
print("Infinitely many solutions → find minimum norm solution")
x5_min = np.linalg.lstsq(A5, b5, rcond=None)[0]
print(f"Minimum norm solution: {x5_min}")
\`\`\`

## Gaussian Elimination

The fundamental algorithm for solving systems.

**Steps**:
1. **Forward elimination**: Convert to upper triangular form
2. **Back substitution**: Solve from bottom to top

\`\`\`python
print("\\n=== Gaussian Elimination ===")

def gaussian_elimination(A, b):
    """
    Solve Ax = b using Gaussian elimination
    Returns solution x
    """
    n = len (b)
    # Create augmented matrix [A|b]
    Ab = np.column_stack([A.astype (float), b.astype (float)])
    
    print("Augmented matrix [A|b]:")
    print(Ab)
    print()
    
    # Forward elimination
    for i in range (n):
        # Find pivot
        max_row = i + np.argmax (np.abs(Ab[i:, i]))
        Ab[[i, max_row]] = Ab[[max_row, i]]  # Swap rows
        
        # Eliminate below
        for j in range (i + 1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= factor * Ab[i, i:]
        
        print(f"After elimination step {i+1}:")
        print(Ab)
        print()
    
    # Back substitution
    x = np.zeros (n)
    for i in range (n-1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:n])) / Ab[i, i]
    
    return x

# Example
A = np.array([[2, 1, -1],
              [-3, -1, 2],
              [-2, 1, 2]], dtype=float)
b = np.array([8, -11, -3], dtype=float)

print("Solving system:")
print(f"A:\\n{A}")
print(f"b: {b}")
print()

x_ge = gaussian_elimination(A.copy(), b.copy())
print(f"\\nSolution: {x_ge}")
print(f"Verification Ax = {A @ x_ge}")
print(f"b = {b}")
print(f"Match: {np.allclose(A @ x_ge, b)}")
\`\`\`

## LU Decomposition

Factor **A** = **LU** where **L** is lower triangular and **U** is upper triangular.

**Advantages**:
- Solve multiple systems with same **A** efficiently
- Numerically stable with partial pivoting
- Foundation for many algorithms

\`\`\`python
from scipy.linalg import lu

print("\\n=== LU Decomposition ===")

A = np.array([[2, 1, 1],
              [4, 3, 3],
              [8, 7, 9]], dtype=float)

# Compute LU decomposition with partial pivoting
P, L, U = lu(A)

print("Original matrix A:")
print(A)
print("\\nPermutation matrix P:")
print(P)
print("\\nLower triangular L:")
print(L)
print("\\nUpper triangular U:")
print(U)
print()

# Verify PA = LU
print("Verification PA = LU:")
print(f"PA:\\n{P @ A}")
print(f"LU:\\n{L @ U}")
print(f"Equal: {np.allclose(P @ A, L @ U)}")
print()

# Solving Ax = b using LU
b = np.array([4, 10, 24])

# Step 1: Solve Ly = Pb (forward substitution)
Pb = P @ b
y = np.linalg.solve(L, Pb)

# Step 2: Solve Ux = y (back substitution)
x = np.linalg.solve(U, y)

print(f"Solution: {x}")
print(f"Verification Ax = {A @ x}")
print(f"b = {b}")
\`\`\`

## Least Squares Solutions

For overdetermined systems (m > n), find **x** that minimizes ||**Ax** - **b**||².

**Normal equations**: **AᵀAx** = **Aᵀb**

**Solution**: **x** = (**AᵀA**)⁻¹**Aᵀb**

\`\`\`python
print("\\n=== Least Squares ===")

# Overdetermined system: fit line to points
# y = mx + c
# [x₁ 1] [m]   [y₁]
# [x₂ 1] [c] = [y₂]
# [x₃ 1]       [y₃]
# ...

# Generate noisy data
np.random.seed(42)
x_data = np.linspace(0, 10, 20)
y_true = 2 * x_data + 1
y_data = y_true + np.random.randn(20) * 2

# Set up overdetermined system
A = np.column_stack([x_data, np.ones_like (x_data)])
b = y_data

print(f"Data points: {len (x_data)}")
print(f"Parameters to fit: 2 (slope and intercept)")
print(f"System shape: {A.shape}")
print()

# Method 1: Normal equations (not recommended)
x_normal = np.linalg.solve(A.T @ A, A.T @ b)

# Method 2: np.linalg.lstsq (recommended)
x_lstsq, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

# Method 3: Using pseudo-inverse
A_pinv = np.linalg.pinv(A)
x_pinv = A_pinv @ b

print("Solutions:")
print(f"Normal equations: {x_normal}")
print(f"np.linalg.lstsq: {x_lstsq}")
print(f"Pseudo-inverse: {x_pinv}")
print(f"True parameters: [2.0, 1.0]")
print()

print(f"Residual sum of squares: {residuals[0]:.4f}")
print(f"Residual norm: {np.linalg.norm(A @ x_lstsq - b):.4f}")

# Visualize
plt.figure (figsize=(10, 6))
plt.scatter (x_data, y_data, alpha=0.6, label='Data points')
plt.plot (x_data, y_true, 'g--', label='True line: y = 2x + 1')
plt.plot (x_data, A @ x_lstsq, 'r-', linewidth=2, label=f'Fitted: y = {x_lstsq[0]:.2f}x + {x_lstsq[1]:.2f}')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Least Squares Linear Regression')
plt.grid(True, alpha=0.3)
plt.show()
\`\`\`

## QR Decomposition for Least Squares

More numerically stable than normal equations.

**A** = **QR** where **Q** is orthogonal and **R** is upper triangular.

\`\`\`python
print("\\n=== QR Decomposition for Least Squares ===")

# Same data as before
Q, R = np.linalg.qr(A)

print("Q (orthogonal) shape:", Q.shape)
print("R (upper triangular) shape:", R.shape)
print()

# Verify QᵀQ = I
print("QᵀQ:")
print((Q.T @ Q).round(10))
print(f"Is identity: {np.allclose(Q.T @ Q, np.eye(Q.shape[1]))}")
print()

# Solve using QR: Ax = b → QRx = b → Rx = Qᵀb
Qtb = Q.T @ b
x_qr = np.linalg.solve(R, Qtb)

print(f"Solution using QR: {x_qr}")
print(f"Matches lstsq: {np.allclose (x_qr, x_lstsq)}")
print()

print("Advantages of QR over normal equations:")
print("1. Better numerical stability")
print("2. Avoids computing AᵀA (which squares condition number)")
print("3. Works well even when A is ill-conditioned")
\`\`\`

## Applications in Machine Learning

### 1. Linear Regression

\`\`\`python
print("\\n=== Linear Regression Application ===")

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Generate regression data
X, y = make_regression (n_samples=100, n_features=5, noise=10, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add intercept term
X_train_int = np.column_stack([X_train, np.ones (len(X_train))])
X_test_int = np.column_stack([X_test, np.ones (len(X_test))])

# Solve using least squares
weights = np.linalg.lstsq(X_train_int, y_train, rcond=None)[0]

# Predictions
y_pred = X_test_int @ weights

# Evaluate
mse = np.mean((y_pred - y_test)**2)
r2 = 1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean (y_test))**2)

print(f"Weights: {weights}")
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")
\`\`\`

### 2. Polynomial Regression

\`\`\`python
print("\\n=== Polynomial Regression ===")

# Generate non-linear data
x = np.linspace(0, 1, 50)
y = np.sin(2 * np.pi * x) + np.random.randn(50) * 0.1

# Create polynomial features [x, x², x³]
degree = 3
A_poly = np.column_stack([x**i for i in range (degree + 1)])

# Fit
coeffs = np.linalg.lstsq(A_poly, y, rcond=None)[0]

# Predict
y_pred = A_poly @ coeffs

# Plot
plt.figure (figsize=(10, 6))
plt.scatter (x, y, alpha=0.6, label='Data')
plt.plot (x, y_pred, 'r-', linewidth=2, label=f'Degree {degree} polynomial')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Polynomial Regression using Least Squares')
plt.grid(True, alpha=0.3)
plt.show()

print(f"Polynomial coefficients: {coeffs}")
\`\`\`

## Summary

**Key Methods**:
- **Gaussian Elimination**: Direct method for square systems
- **LU Decomposition**: Efficient for multiple systems with same **A**
- **Least Squares**: For overdetermined systems (more equations than unknowns)
- **QR Decomposition**: More stable than normal equations

**System Types**:
- **Square, non-singular** (m = n, det ≠ 0): Unique solution
- **Overdetermined** (m > n): Least squares solution
- **Underdetermined** (m < n): Minimum norm solution
- **Singular**: No solution or infinite solutions

**ML Applications**:
- Linear regression (least squares)
- Polynomial regression
- Ridge regression (regularized least squares)
- Neural network training (solving gradient equations)

**Best Practices**:
- Use \`np.linalg.solve()\` for square systems
- Use \`np.linalg.lstsq()\` for overdetermined systems
- Use QR decomposition for better numerical stability
- Check condition number for ill-conditioned systems
- Add regularization to prevent overfitting

Understanding how to solve linear systems is fundamental to implementing and debugging ML algorithms!
`,
};
