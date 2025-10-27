/**
 * Matrices Fundamentals Section
 */

export const matricesfundamentalsSection = {
  id: 'matrices-fundamentals',
  title: 'Matrices Fundamentals',
  content: `
# Matrices Fundamentals

## Introduction

A **matrix** is a rectangular array of numbers arranged in rows and columns. If vectors are the words of linear algebra, then matrices are the sentences—they enable us to express complex transformations, systems of equations, and entire datasets in a compact form.

In machine learning, matrices are everywhere:
- **Datasets**: Each row is a sample, each column is a feature
- **Weight matrices**: Connect layers in neural networks
- **Transformations**: Rotation, scaling, projection
- **Covariance**: Capture relationships between features

## Matrix Notation

### General Form

A matrix **A** with m rows and n columns is an **m × n matrix**:

**A** = ⎡a₁₁  a₁₂  ...  a₁ₙ⎤
      ⎢a₂₁  a₂₂  ...  a₂ₙ⎥
      ⎢ ...  ...  ...  ...⎥
      ⎣aₘ₁  aₘ₂  ...  aₘₙ⎦

- **m**: number of rows
- **n**: number of columns  
- **aᵢⱼ**: element in row i, column j

### Notation

- Bold capital letter: **A**, **B**, **X**
- Element: aᵢⱼ or Aᵢⱼ
- Size: **A** ∈ ℝᵐˣⁿ (m×n matrix of real numbers)

### Special Cases

- **Square matrix**: m = n (same number of rows and columns)
- **Row vector**: 1 × n matrix (single row)
- **Column vector**: m × 1 matrix (single column)
- **Scalar**: 1 × 1 matrix (single element)

## Matrix Indexing

In mathematics, indices typically start at 1:
- Rows: 1, 2, ..., m
- Columns: 1, 2, ..., n

In Python (NumPy), indices start at 0:
- Rows: 0, 1, ..., m-1
- Columns: 0, 1, ..., n-1

## Matrix Addition and Scalar Multiplication

### Matrix Addition

Two matrices can be added if they have the **same dimensions**:

**A** + **B** = [aᵢⱼ + bᵢⱼ]

Add corresponding elements.

**Properties**:
- Commutative: **A** + **B** = **B** + **A**
- Associative: (**A** + **B**) + **C** = **A** + (**B** + **C**)
- Identity: **A** + **0** = **A**
- Inverse: **A** + (-**A**) = **0**

### Scalar Multiplication

Multiply every element by a scalar c:

c**A** = [c·aᵢⱼ]

**Properties**:
- Distributive: c(**A** + **B**) = c**A** + c**B**
- Distributive: (c + d)**A** = c**A** + d**A**
- Associative: (cd)**A** = c (d**A**)
- Identity: 1**A** = **A**

## Matrix Multiplication

Matrix multiplication is more complex and incredibly important.

### Dimensions Must Be Compatible

To multiply **A** (m × n) by **B** (p × q):
- **Requirement**: n = p (columns of A = rows of B)
- **Result**: **C** = **AB** is m × q

The "inner dimensions" must match, and the result has the "outer dimensions."

### Definition

For **C** = **AB**, element cᵢⱼ is the dot product of row i of **A** and column j of **B**:

cᵢⱼ = Σₖ aᵢₖ bₖⱼ = aᵢ₁b₁ⱼ + aᵢ₂b₂ⱼ + ... + aᵢₙbₙⱼ

**Intuition**: Each element of the result is a dot product of a row from the first matrix and a column from the second matrix.

### Properties

1. **NOT commutative**: **AB** ≠ **BA** (in general)
2. **Associative**: (**AB**)**C** = **A**(**BC**)
3. **Distributive**: **A**(**B** + **C**) = **AB** + **AC**4. **Identity**: **AI** = **IA** = **A**

### Why Matrix Multiplication Works This Way

This definition might seem arbitrary, but it corresponds to **composing linear transformations** and **representing systems of linear equations**. It is precisely the right operation for these purposes.

## Identity Matrix

The **identity matrix** **I** is a square matrix with 1s on the diagonal and 0s elsewhere:

**I** = ⎡1  0  0⎤
      ⎢0  1  0⎥
      ⎣0  0  1⎦

**Property**: **AI** = **IA** = **A** (for compatible dimensions)

The identity matrix is the multiplicative identity—like multiplying by 1 for scalars.

## Matrix Transpose

The **transpose** of **A** (denoted **Aᵀ**) flips rows and columns:

If **A** is m × n, then **Aᵀ** is n × m

(Aᵀ)ᵢⱼ = Aⱼᵢ

**Properties**:
- (**Aᵀ**)ᵀ = **A**
- (**A** + **B**)ᵀ = **Aᵀ** + **Bᵀ**
- (**AB**)ᵀ = **Bᵀ Aᵀ** (note the reversal!)
- (c**A**)ᵀ = c**Aᵀ**

## Python Implementation

\`\`\`python
import numpy as np

# Creating matrices
A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

B = np.array([
    [7, 8],
    [9, 10],
    [11, 12]
])

print("Matrix A (2×3):")
print(A)
print(f"Shape: {A.shape}")
print()

print("Matrix B (3×2):")
print(B)
print(f"Shape: {B.shape}")
print()

# Accessing elements (0-indexed in Python)
print(f"Element A[0,0] (first row, first column): {A[0, 0]}")
print(f"Element A[1,2] (second row, third column): {A[1, 2]}")
print()

# Accessing rows and columns
print(f"First row of A: {A[0, :]}")
print(f"Second column of A: {A[:, 1]}")
print()

# Matrix dimensions
m, n = A.shape
print(f"A has {m} rows and {n} columns")
\`\`\`

\`\`\`python
# Matrix addition
C = np.array([[1, 2, 3], [4, 5, 6]])
D = np.array([[10, 20, 30], [40, 50, 60]])

print("=== Matrix Addition ===")
print("C:")
print(C)
print("\\nD:")
print(D)
print("\\nC + D:")
print(C + D)
print()

# Scalar multiplication
print("=== Scalar Multiplication ===")
print("C:")
print(C)
print("\\n3 * C:")
print(3 * C)
print()

# Element-wise multiplication (Hadamard product) - not standard matrix multiplication
print("=== Element-wise Multiplication ===")
print("C ⊙ D (element-wise):")
print(C * D)
print()
\`\`\`

\`\`\`python
# Matrix multiplication
print("=== Matrix Multiplication ===")
print("A (2×3):")
print(A)
print("\\nB (3×2):")
print(B)
print()

# Matrix multiplication: A @ B
C = A @ B  # Python 3.5+ operator
C_alt = np.dot(A, B)  # Alternative method
C_matmul = np.matmul(A, B)  # Another alternative

print("C = A @ B (2×2):")
print(C)
print()

# Verify all methods give same result
assert np.allclose(C, C_alt)
assert np.allclose(C, C_matmul)

# Show the computation for one element
print("Computing C[0,0]:")
print(f"C[0,0] = A[0,:] · B[:,0]")
print(f"       = {A[0,:]} · {B[:,0]}")
print(f"       = {A[0,0]}*{B[0,0]} + {A[0,1]}*{B[1,0]} + {A[0,2]}*{B[2,0]}")
print(f"       = {A[0,0]*B[0,0]} + {A[0,1]*B[1,0]} + {A[0,2]*B[2,0]}")
print(f"       = {C[0,0]}")
\`\`\`

\`\`\`python
# Matrix multiplication is NOT commutative
print("\\n=== Non-commutativity ===")
A_small = np.array([[1, 2], [3, 4]])
B_small = np.array([[5, 6], [7, 8]])

AB = A_small @ B_small
BA = B_small @ A_small

print("A:")
print(A_small)
print("\\nB:")
print(B_small)
print("\\nAB:")
print(AB)
print("\\nBA:")
print(BA)
print("\\nAB == BA:", np.allclose(AB, BA))
\`\`\`

\`\`\`python
# Identity matrix
print("\\n=== Identity Matrix ===")
I = np.eye(3)  # 3×3 identity matrix
print("Identity matrix I (3×3):")
print(I)
print()

# Matrix times identity
A_square = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("A:")
print(A_square)
print("\\nA @ I:")
print(A_square @ I)
print("\\nI @ A:")
print(I @ A_square)
print("\\nVerify A @ I == A:", np.allclose(A_square @ I, A_square))
\`\`\`

\`\`\`python
# Matrix transpose
print("\\n=== Matrix Transpose ===")
A_rect = np.array([[1, 2, 3], [4, 5, 6]])
print(f"A ({A_rect.shape[0]}×{A_rect.shape[1]}):")
print(A_rect)
print()

A_T = A_rect.T  # or np.transpose(A_rect)
print(f"Aᵀ ({A_T.shape[0]}×{A_T.shape[1]}):")
print(A_T)
print()

# Verify (Aᵀ)ᵀ = A
print("(Aᵀ)ᵀ == A:", np.allclose(A_T.T, A_rect))
print()

# Verify (AB)ᵀ = BᵀAᵀ
X = np.array([[1, 2], [3, 4], [5, 6]])
Y = np.array([[7, 8, 9], [10, 11, 12]])

XY = X @ Y
XY_T = XY.T
Y_T_X_T = Y.T @ X.T

print("Verify (XY)ᵀ = YᵀXᵀ:", np.allclose(XY_T, Y_T_X_T))
\`\`\`

## Matrices as Linear Transformations

A key insight: **matrix multiplication represents function composition**.

When we multiply a matrix **A** by a vector **x**, we transform **x**:

**y** = **Ax**

### Example: Rotation Matrix (2D)

Rotating a vector by angle θ:

**R**(θ) = ⎡cos(θ)  -sin(θ)⎤
          ⎣sin(θ)   cos(θ)⎦

\`\`\`python
# Rotation matrix example
def rotation_matrix_2d (theta):
    """Create 2D rotation matrix for angle theta (in radians)"""
    return np.array([
        [np.cos (theta), -np.sin (theta)],
        [np.sin (theta),  np.cos (theta)]
    ])

# Rotate vector [1, 0] by 90 degrees (π/2 radians)
v = np.array([1, 0])
theta = np.pi / 2  # 90 degrees

R = rotation_matrix_2d (theta)
v_rotated = R @ v

print("=== Rotation Example ===")
print(f"Original vector: {v}")
print(f"Rotation by {np.degrees (theta)}°")
print(f"Rotation matrix R:")
print(R)
print(f"Rotated vector: {v_rotated}")
print(f"Expected: [0, 1] (approximately)")
\`\`\`

\`\`\`python
# Visualize rotation
def plot_vector_transformation (v_original, v_transformed, title=""):
    """Plot original and transformed vectors"""
    plt.figure (figsize=(8, 8))
    plt.axhline (y=0, color='k', linewidth=0.5)
    plt.axvline (x=0, color='k', linewidth=0.5)
    plt.grid(True, alpha=0.3)
    
    # Plot vectors
    plt.quiver(0, 0, v_original[0], v_original[1], 
              angles='xy', scale_units='xy', scale=1, 
              color='blue', width=0.008, label='Original')
    plt.quiver(0, 0, v_transformed[0], v_transformed[1], 
              angles='xy', scale_units='xy', scale=1, 
              color='red', width=0.008, label='Transformed')
    
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title (title)
    plt.axis('equal')
    plt.show()

plot_vector_transformation (v, v_rotated, "90° Rotation")
\`\`\`

### Example: Scaling Matrix

Scaling by factors sₓ and sᵧ:

**S** = ⎡sₓ  0 ⎤
      ⎣0   sᵧ⎦

\`\`\`python
# Scaling matrix
S = np.array([
    [2, 0],  # Scale x by 2
    [0, 3]   # Scale y by 3
])

v = np.array([1, 1])
v_scaled = S @ v

print("\\n=== Scaling Example ===")
print(f"Original vector: {v}")
print(f"Scaling matrix S:")
print(S)
print(f"Scaled vector: {v_scaled}")

plot_vector_transformation (v, v_scaled, "Scaling: 2x in X, 3x in Y")
\`\`\`

## Matrices in Machine Learning

### Dataset Representation

\`\`\`python
# Dataset as matrix: rows = samples, columns = features
# Example: Housing data
# Features: [square_feet, bedrooms, bathrooms, age_years]
# Price is separate (target variable)

X = np.array([
    [2000, 3, 2, 10],  # House 1
    [1500, 2, 1, 15],  # House 2
    [2500, 4, 3, 5],   # House 3
    [1800, 3, 2, 8],   # House 4
    [2200, 3, 2.5, 12] # House 5
])

y = np.array([300000, 220000, 380000, 290000, 320000])  # Prices

print("=== Dataset Matrix ===")
print(f"X shape: {X.shape} (5 samples, 4 features)")
print("X (feature matrix):")
print(X)
print(f"\\ny shape: {y.shape}")
print(f"y (target): {y}")
\`\`\`

### Linear Regression with Matrices

Linear regression can be expressed compactly using matrices:

**y** = **Xw** + **b**

Where:
- **X**: m × n matrix of features (m samples, n features)
- **w**: n × 1 vector of weights  
- **b**: scalar bias (or m × 1 vector)
- **y**: m × 1 vector of predictions

\`\`\`python
# Simple linear regression using matrix operations
# Predict house prices from features

# Initialize random weights
np.random.seed(42)
w = np.random.randn(4)  # 4 weights for 4 features
b = np.random.randn()   # 1 bias

print("\\n=== Linear Regression (Matrix Form) ===")
print(f"Weights w: {w}")
print(f"Bias b: {b:.4f}")
print()

# Make predictions: y_pred = Xw + b
y_pred = X @ w + b

print("Predictions y_pred = Xw + b:")
print(y_pred)
print()
print("Actual prices y:")
print(y)
print()

# Compute error (Mean Squared Error)
mse = np.mean((y_pred - y)**2)
print(f"MSE (with random weights): {mse:.2f}")
\`\`\`

### Neural Network Layer

A fully connected layer transforms input **x** to output **y**:

**y** = **Wx** + **b**

Where **W** is the weight matrix.

\`\`\`python
# Neural network layer as matrix multiplication
# Input layer: 4 neurons
# Hidden layer: 3 neurons

input_size = 4
hidden_size = 3

# Initialize weights and bias
np.random.seed(42)
W = np.random.randn (hidden_size, input_size) * 0.1
b = np.zeros (hidden_size)

print("\\n=== Neural Network Layer ===")
print(f"Weight matrix W ({hidden_size}×{input_size}):")
print(W)
print(f"\\nBias vector b ({hidden_size},):")
print(b)
print()

# Forward pass for one sample
x = np.array([1.0, 2.0, 3.0, 4.0])  # Input features
z = W @ x + b  # Linear transformation
a = np.maximum(0, z)  # ReLU activation

print(f"Input x: {x}")
print(f"Linear output z = Wx + b: {z}")
print(f"Activated output a = ReLU(z): {a}")
\`\`\`

### Batch Processing

Process multiple samples simultaneously:

\`\`\`python
# Batch processing: multiple samples at once
batch_size = 5
X_batch = X  # Use our housing data (5 samples, 4 features)

# Forward pass for entire batch
Z_batch = X_batch @ W.T + b  # Note: W.T to match dimensions
A_batch = np.maximum(0, Z_batch)

print("\\n=== Batch Processing ===")
print(f"Batch input X ({X_batch.shape}):")
print(X_batch)
print(f"\\nBatch output A ({A_batch.shape}):")
print(A_batch)
print()
print("Each row is the output for one sample")
\`\`\`

## Common Matrix Shapes in ML

| Matrix | Shape | Description |
|--------|-------|-------------|
| **X** | m × n | Dataset: m samples, n features |
| **y** | m × 1 | Target values for m samples |
| **W** | k × n | Weights: k outputs, n inputs |
| **b** | k × 1 | Biases for k outputs |
| **θ** | n × 1 | Model parameters |

## Best Practices

1. **Check dimensions before matrix multiplication**: Use \`.shape\` liberally
2. **Use @ operator for clarity**: \`A @ B\` is clearer than \`np.dot(A, B)\` for matrices
3. **Vectorize operations**: Process entire batches, not individual samples in loops
4. **Transpose when necessary**: Match dimensions for multiplication
5. **Initialize sensibly**: Random initialization for neural networks, not zeros

\`\`\`python
# Good practice: Check shapes
def matrix_multiply_safe(A, B):
    """Safely multiply matrices with dimension checking"""
    if A.shape[1] != B.shape[0]:
        raise ValueError (f"Cannot multiply {A.shape} × {B.shape}: "
                        f"incompatible dimensions")
    return A @ B

# Test
try:
    result = matrix_multiply_safe(A, B)
    print("Multiplication successful!")
except ValueError as e:
    print(f"Error: {e}")
\`\`\`

## Summary

Matrices are the workhorse of machine learning:

- **Compact representation**: Entire datasets, transformations, and models
- **Efficient computation**: Matrix operations are highly optimized (BLAS/LAPACK)
- **Batch processing**: Process many samples simultaneously
- **Linear transformations**: Rotation, scaling, projection, neural network layers

**Key Operations**:
- Addition: Element-wise, same dimensions required
- Scalar multiplication: Scale all elements
- Matrix multiplication: Compose transformations, compute outputs
- Transpose: Flip rows and columns

**Applications**:
- Datasets (rows = samples, columns = features)
- Linear regression (**y** = **Xw** + **b**)
- Neural networks (each layer is matrix multiplication + activation)
- Image processing (images are matrices)

Master matrices, and you can express complex ML operations concisely and compute them efficiently!
`,
};
