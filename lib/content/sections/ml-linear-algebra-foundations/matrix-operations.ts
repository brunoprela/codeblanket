/**
 * Matrix Operations Section
 */

export const matrixoperationsSection = {
  id: 'matrix-operations',
  title: 'Matrix Operations',
  content: `
# Matrix Operations

## Introduction

Beyond basic addition and multiplication, matrices support advanced operations crucial for machine learning: matrix-vector multiplication, batch processing, broadcasting, and computing complex transformations. Understanding these operations deeply enables you to implement and optimize ML algorithms efficiently.

## Matrix-Vector Multiplication

One of the most fundamental operations: multiplying a matrix **A** (m × n) by a vector **x** (n × 1) produces a vector **y** (m × 1).

**y** = **Ax**

### Interpretation

Each element yᵢ is the dot product of row i of **A** with vector **x**:

yᵢ = **aᵢ** · **x** = Σⱼ aᵢⱼ xⱼ

This is exactly what happens in a neural network layer: the weight matrix multiplies the input vector.

### Geometric Interpretation

Matrix-vector multiplication is a **linear transformation**: it transforms vector **x** into vector **y** according to the transformation defined by **A**.

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt

print("=== Matrix-Vector Multiplication ===")

# Define matrix and vector
A = np.array([
    [2, 0],
    [0, 3]
])  # Scaling matrix

x = np.array([1, 1])

# Multiply
y = A @ x

print("Matrix A:")
print(A)
print(f"\\nVector x: {x}")
print(f"Result y = Ax: {y}")
print()

# Verify element-wise
print("Computing y manually:")
print(f"y[0] = A[0,:] · x = {A[0,:]} · {x} = {A[0,0]*x[0] + A[0,1]*x[1]}")
print(f"y[1] = A[1,:] · x = {A[1,:]} · {x} = {A[1,0]*x[0] + A[1,1]*x[1]}")
\`\`\`

\`\`\`python
# Visualize transformation
def visualize_matrix_vector_transform(A, x):
    """Visualize how matrix A transforms vector x"""
    y = A @ x
    
    plt.figure(figsize=(10, 5))
    
    # Plot original and transformed vector
    plt.subplot(1, 2, 1)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.grid(True, alpha=0.3)
    plt.quiver(0, 0, x[0], x[1], angles='xy', scale_units='xy', scale=1,
              color='blue', width=0.008, label='x (original)')
    plt.quiver(0, 0, y[0], y[1], angles='xy', scale_units='xy', scale=1,
              color='red', width=0.008, label='y = Ax (transformed)')
    plt.xlim(-0.5, 4)
    plt.ylim(-0.5, 4)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Matrix Transforms Vector')
    plt.axis('equal')
    
    # Show transformation of unit square
    plt.subplot(1, 2, 2)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.grid(True, alpha=0.3)
    
    # Unit square corners
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]).T
    transformed_square = A @ square
    
    plt.plot(square[0, :], square[1, :], 'b-', linewidth=2, label='Original square')
    plt.plot(transformed_square[0, :], transformed_square[1, :], 'r-', 
            linewidth=2, label='Transformed square')
    
    max_val = max(np.max(transformed_square), 2)
    plt.xlim(-0.5, max_val)
    plt.ylim(-0.5, max_val)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('How A Transforms Space')
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()

visualize_matrix_vector_transform(A, x)
\`\`\`

## Matrix-Matrix Multiplication Revisited

Let's understand matrix multiplication from the perspective of **column combinations** and **row transformations**.

### Column Perspective

**AB** can be viewed as **A** transforming each column of **B**:

If **B** = [**b₁** **b₂** ... **bₙ**], then
**AB** = [**Ab₁** **Ab₂** ... **Abₙ**]

Each column of the result is **A** times the corresponding column of **B**.

\`\`\`python
print("\\n=== Column Perspective ===")

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A:")
print(A)
print("\\nB:")
print(B)
print()

# Compute AB
AB = A @ B
print("AB:")
print(AB)
print()

# Verify column by column
print("Column perspective:")
print("First column of AB = A @ (first column of B)")
col1 = A @ B[:, 0]
print(f"  A @ {B[:, 0]} = {col1}")
print(f"  Matches AB[:, 0] = {AB[:, 0]}: {np.allclose(col1, AB[:, 0])}")
print()

print("Second column of AB = A @ (second column of B)")
col2 = A @ B[:, 1]
print(f"  A @ {B[:, 1]} = {col2}")
print(f"  Matches AB[:, 1] = {AB[:, 1]}: {np.allclose(col2, AB[:, 1])}")
\`\`\`

### Row Perspective

**AB** can also be viewed as each row of **A** forming a linear combination of rows of **B**.

\`\`\`python
print("\\n=== Row Perspective ===")
print("First row of AB = (first row of A) @ B")
row1 = A[0, :] @ B
print(f"  {A[0, :]} @ B = {row1}")
print(f"  Matches AB[0, :] = {AB[0, :]}: {np.allclose(row1, AB[0, :])}")
\`\`\`

## Batch Matrix-Vector Multiplication

In ML, we often process multiple vectors (a batch) simultaneously.

Given:
- **X**: (batch_size, n) - batch of input vectors
- **W**: (n, m) - weight matrix
- **Y** = **XW**: (batch_size, m) - batch of output vectors

Each row of **X** is multiplied by **W** to produce the corresponding row of **Y**.

\`\`\`python
print("\\n=== Batch Processing ===")

# Batch of 4 samples, each with 3 features
X = np.array([
    [1, 2, 3],  # Sample 1
    [4, 5, 6],  # Sample 2
    [7, 8, 9],  # Sample 3
    [2, 1, 0]   # Sample 4
])

# Weight matrix: 3 inputs → 2 outputs
W = np.array([
    [0.1, 0.4],
    [0.2, 0.5],
    [0.3, 0.6]
])

print(f"X shape: {X.shape} (4 samples, 3 features each)")
print(f"W shape: {W.shape} (3 inputs, 2 outputs)")
print()

# Batch multiplication
Y = X @ W

print(f"Y = X @ W, shape: {Y.shape} (4 samples, 2 outputs each)")
print("Y:")
print(Y)
print()

# Verify for first sample
print("Verification for first sample:")
y1 = X[0, :] @ W
print(f"X[0, :] @ W = {X[0, :]} @ W = {y1}")
print(f"Matches Y[0, :] = {Y[0, :]}: {np.allclose(y1, Y[0, :])}")
\`\`\`

## Matrix Powers

For square matrices, we can compute powers: **A²** = **AA**, **A³** = **AAA**, etc.

\`\`\`python
print("\\n=== Matrix Powers ===")

A = np.array([[1, 1], [0, 1]])
print("A:")
print(A)
print()

# A^2
A2 = A @ A
print("A² = A @ A:")
print(A2)
print()

# A^3
A3 = A2 @ A
print("A³:")
print(A3)
print()

# Using numpy's matrix_power
A4 = np.linalg.matrix_power(A, 4)
print("A⁴ (using np.linalg.matrix_power):")
print(A4)
\`\`\`

## Trace of a Matrix

The **trace** is the sum of diagonal elements:

tr(**A**) = Σᵢ aᵢᵢ = a₁₁ + a₂₂ + ... + aₙₙ

**Properties**:
- tr(**A** + **B**) = tr(**A**) + tr(**B**)
- tr(c**A**) = c·tr(**A**)
- tr(**AB**) = tr(**BA**) (even if **AB** ≠ **BA**!)
- tr(**A**ᵀ) = tr(**A**)

\`\`\`python
print("\\n=== Trace ===")

A = np.array([[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]])

trace_A = np.trace(A)
trace_manual = A[0,0] + A[1,1] + A[2,2]

print("A:")
print(A)
print(f"\\ntr(A) = {trace_A}")
print(f"tr(A) manual = {A[0,0]} + {A[1,1]} + {A[2,2]} = {trace_manual}")
print()

# Verify tr(AB) = tr(BA)
B = np.array([[1, 0, 2],
             [0, 1, 0],
             [2, 0, 1]])

AB = A @ B
BA = B @ A

print(f"tr(AB) = {np.trace(AB)}")
print(f"tr(BA) = {np.trace(BA)}")
print(f"Equal: {np.isclose(np.trace(AB), np.trace(BA))}")
\`\`\`

## Broadcasting in NumPy

**Broadcasting** allows operations on arrays of different shapes by automatically expanding dimensions.

### Rules

1. If arrays have different numbers of dimensions, pad the smaller one with 1s on the left
2. Arrays are compatible if dimensions are equal or one of them is 1
3. If compatible, broadcast the smaller dimension to match the larger

\`\`\`python
print("\\n=== Broadcasting ===")

# Example 1: Add vector to each row of matrix
X = np.array([[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]])

v = np.array([10, 20, 30])

print("X (3×3):")
print(X)
print(f"\\nv (3,): {v}")
print()

# Broadcasting: v is added to each row
X_plus_v = X + v
print("X + v (broadcasts v to each row):")
print(X_plus_v)
print()

# Example 2: Multiply matrix by column vector
col = np.array([[2], [3], [4]])  # 3×1
print(f"col shape {col.shape}:")
print(col)
print()

X_times_col = X * col  # Element-wise multiplication with broadcasting
print("X * col (broadcasts col to each column):")
print(X_times_col)
\`\`\`

\`\`\`python
# Common broadcasting patterns
print("\\n=== Common Broadcasting Patterns ===")

# 1. Add bias to each sample in batch
batch = np.random.randn(100, 5)  # 100 samples, 5 features
bias = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # 5 biases

batch_with_bias = batch + bias  # Bias broadcast to all samples
print(f"1. Batch {batch.shape} + bias {bias.shape} → {batch_with_bias.shape}")

# 2. Normalize each feature (across samples)
mean = batch.mean(axis=0)  # Mean of each feature
std = batch.std(axis=0)  # Std of each feature
batch_normalized = (batch - mean) / std
print(f"2. Batch normalization: {batch.shape} → {batch_normalized.shape}")

# 3. Outer product without explicit loops
a = np.array([1, 2, 3])
b = np.array([4, 5])
outer = a[:, np.newaxis] @ b[np.newaxis, :]
print(f"\\n3. Outer product: a{a.shape} ⊗ b{b.shape} → {outer.shape}")
print("Outer product:")
print(outer)
\`\`\`

## Element-wise Operations

Operations that work element-by-element (require same shape or broadcasting).

\`\`\`python
print("\\n=== Element-wise Operations ===")

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A:")
print(A)
print("\\nB:")
print(B)
print()

# Element-wise operations
print("A + B:")
print(A + B)

print("\\nA * B (Hadamard product):")
print(A * B)

print("\\nA / B:")
print(A / B)

print("\\nA ** 2 (element-wise square):")
print(A ** 2)

print("\\nnp.sqrt(A):")
print(np.sqrt(A))

print("\\nnp.exp(A):")
print(np.exp(A))
\`\`\`

## Matrix Operations in Neural Networks

Let's implement a complete forward pass through a neural network layer using pure matrix operations.

\`\`\`python
print("\\n=== Neural Network Forward Pass ===")

def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)

def forward_pass(X, W1, b1, W2, b2):
    """
    Two-layer neural network forward pass
    
    X: (batch_size, input_dim)
    W1: (input_dim, hidden_dim)
    b1: (hidden_dim,)
    W2: (hidden_dim, output_dim)
    b2: (output_dim,)
    """
    # Layer 1
    Z1 = X @ W1 + b1  # Linear transformation
    A1 = relu(Z1)  # Activation
    
    # Layer 2
    Z2 = A1 @ W2 + b2  # Linear transformation
    A2 = Z2  # No activation (linear output)
    
    return Z1, A1, Z2, A2

# Initialize network
np.random.seed(42)
input_dim = 4
hidden_dim = 3
output_dim = 2
batch_size = 5

W1 = np.random.randn(input_dim, hidden_dim) * 0.1
b1 = np.zeros(hidden_dim)
W2 = np.random.randn(hidden_dim, output_dim) * 0.1
b2 = np.zeros(output_dim)

# Input data
X = np.random.randn(batch_size, input_dim)

print(f"Input X: {X.shape}")
print(f"W1: {W1.shape}, b1: {b1.shape}")
print(f"W2: {W2.shape}, b2: {b2.shape}")
print()

# Forward pass
Z1, A1, Z2, A2 = forward_pass(X, W1, b1, W2, b2)

print(f"After layer 1: Z1 {Z1.shape}, A1 {A1.shape}")
print(f"After layer 2: Z2 {Z2.shape}, A2 {A2.shape}")
print(f"\\nFinal output A2:\\n{A2}")
\`\`\`

## Common Mistakes and Best Practices

\`\`\`python
print("\\n=== Common Mistakes ===")

# Mistake 1: Shape mismatches
A = np.array([[1, 2, 3]])  # Shape (1, 3)
B = np.array([[4], [5], [6]])  # Shape (3, 1)

print(f"A shape: {A.shape}")
print(f"B shape: {B.shape}")

try:
    result = A + B  # This will work due to broadcasting!
    print(f"A + B worked (broadcasting): shape {result.shape}")
    print(result)
except ValueError as e:
    print(f"Error: {e}")

print()

# Mistake 2: Confusing @ and *
X = np.array([[1, 2], [3, 4]])
Y = np.array([[5, 6], [7, 8]])

print("X @ Y (matrix multiplication):")
print(X @ Y)

print("\\nX * Y (element-wise, Hadamard):")
print(X * Y)

# Mistake 3: Forgetting to reshape
v = np.array([1, 2, 3])  # Shape (3,)
print(f"\\nv shape: {v.shape}")

v_col = v[:, np.newaxis]  # Shape (3, 1)
v_row = v[np.newaxis, :]  # Shape (1, 3)

print(f"v as column: {v_col.shape}")
print(f"v as row: {v_row.shape}")

outer_product = v_col @ v_row
print(f"\\nOuter product v_col @ v_row: shape {outer_product.shape}")
print(outer_product)
\`\`\`

## Performance Tips

\`\`\`python
import time

print("\\n=== Performance Comparison ===")

# Setup
n = 1000
A = np.random.randn(n, n)
B = np.random.randn(n, n)

# Vectorized (fast)
start = time.time()
C_fast = A @ B
time_fast = time.time() - start

# Loop-based (slow) - don't do this!
start = time.time()
C_slow = np.zeros((n, n))
for i in range(min(10, n)):  # Only 10 rows to save time
    for j in range(n):
        C_slow[i, j] = np.dot(A[i, :], B[:, j])
time_slow = time.time() - start
time_slow_extrapolated = time_slow * (n / 10)

print(f"Vectorized @ operator: {time_fast:.4f} seconds")
print(f"Loop-based (10 rows): {time_slow:.4f} seconds")
print(f"Loop-based (extrapolated full): ~{time_slow_extrapolated:.2f} seconds")
print(f"Speedup: ~{time_slow_extrapolated/time_fast:.0f}x faster")
\`\`\`

## Summary

Matrix operations are the computational engine of machine learning:

1. **Matrix-vector multiplication**: Transforms vectors (neural network layers)
2. **Matrix-matrix multiplication**: Composes transformations, processes batches
3. **Batch processing**: Process multiple samples simultaneously for efficiency
4. **Broadcasting**: Automatic dimension expansion for convenient operations
5. **Element-wise operations**: Apply functions to each element
6. **Trace**: Sum of diagonal elements, useful in loss functions

**Best Practices**:
- Always check shapes: use \`.shape\` liberally
- Use @ for matrix multiplication, * for element-wise
- Vectorize operations: avoid Python loops
- Leverage broadcasting for concise code
- Use proper reshaping when needed

**Performance**:
- Vectorized operations are 10-1000x faster than loops
- GPUs accelerate matrix operations even more
- Modern ML frameworks (PyTorch, TensorFlow) optimize these operations

Master these operations, and you can implement any neural network architecture efficiently!
`,
};
