/**
 * Tensor Operations in Deep Learning Section
 */

export const tensoroperationsSection = {
  id: 'tensor-operations',
  title: 'Tensor Operations in Deep Learning',
  content: `
# Tensor Operations in Deep Learning

## Introduction

**Tensors** are multi-dimensional arrays that generalize scalars (0D), vectors (1D), and matrices (2D) to higher dimensions. They are the fundamental data structure in deep learning frameworks like PyTorch and TensorFlow.

**Dimensionality**:
- **Scalar**: 0D tensor (single number)
- **Vector**: 1D tensor (array of numbers)
- **Matrix**: 2D tensor (table of numbers)
- **3D Tensor**: Batch of matrices or RGB image (width × height × channels)
- **4D Tensor**: Batch of images (batch × channels × height × width)
- **Higher**: Video, sequences, etc.

\`\`\`python
import numpy as np

print("=== Tensor Basics ===")

# Scalars (0D)
scalar = 42
print(f"Scalar (0D): {scalar}")
print(f"Shape: {np.array (scalar).shape}")
print()

# Vectors (1D)
vector = np.array([1, 2, 3, 4])
print(f"Vector (1D): {vector}")
print(f"Shape: {vector.shape}")
print()

# Matrices (2D)
matrix = np.array([[1, 2], [3, 4], [5, 6]])
print(f"Matrix (2D):\\n{matrix}")
print(f"Shape: {matrix.shape}")
print()

# 3D Tensor
tensor_3d = np.random.randn(2, 3, 4)  # 2 matrices, each 3×4
print(f"3D Tensor shape: {tensor_3d.shape}")
print(f"Interpretation: 2 samples, each 3×4")
print()

# 4D Tensor (typical for images)
tensor_4d = np.random.randn(8, 3, 32, 32)  # batch, channels, height, width
print(f"4D Tensor (images) shape: {tensor_4d.shape}")
print(f"Interpretation: batch of 8 images, 3 channels (RGB), 32×32 pixels")
\`\`\`

## Basic Tensor Operations

### Element-wise Operations

\`\`\`python
print("\\n=== Element-wise Operations ===")

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A:")
print(A)
print("\\nB:")
print(B)
print()

# Addition
print("A + B (element-wise):")
print(A + B)
print()

# Multiplication
print("A * B (element-wise, Hadamard product):")
print(A * B)
print()

# Exponentiation
print("A ** 2 (element-wise):")
print(A ** 2)
\`\`\`

### Reduction Operations

\`\`\`python
print("\\n=== Reduction Operations ===")

X = np.array([[1, 2, 3],
              [4, 5, 6]])

print("X:")
print(X)
print()

# Sum along different axes
print(f"Sum all: {np.sum(X)}")
print(f"Sum axis=0 (columns): {np.sum(X, axis=0)}")
print(f"Sum axis=1 (rows): {np.sum(X, axis=1)}")
print()

# Mean
print(f"Mean all: {np.mean(X)}")
print(f"Mean axis=0: {np.mean(X, axis=0)}")
print()

# Max/Min
print(f"Max: {np.max(X)}")
print(f"Argmax (index of max): {np.argmax(X)}")
\`\`\`

## Broadcasting

**Broadcasting** allows operations on arrays of different shapes by automatically expanding dimensions.

**Rules**:
1. Align shapes from right
2. Dimensions must be compatible (equal or one is 1)
3. Broadcast smaller dimension to match larger

\`\`\`python
print("\\n=== Broadcasting ===")

# Example 1: Vector + Scalar
v = np.array([1, 2, 3])
s = 10

result1 = v + s  # s broadcast to [10, 10, 10]
print(f"{v} + {s} = {result1}")
print()

# Example 2: Matrix + Vector (row)
M = np.array([[1, 2, 3],
              [4, 5, 6]])
v_row = np.array([10, 20, 30])

result2 = M + v_row  # v_row broadcast to each row
print("Matrix + row vector:")
print(f"M:\\n{M}")
print(f"v: {v_row}")
print(f"M + v:\\n{result2}")
print()

# Example 3: Matrix + Vector (column)
v_col = np.array([[100], [200]])  # Shape (2, 1)

result3 = M + v_col  # v_col broadcast to each column
print("Matrix + column vector:")
print(f"v_col:\\n{v_col}")
print(f"M + v_col:\\n{result3}")
\`\`\`

## Tensor Reshaping

\`\`\`python
print("\\n=== Tensor Reshaping ===")

X = np.arange(12)  # [0, 1, 2, ..., 11]
print(f"Original (1D): {X}")
print(f"Shape: {X.shape}")
print()

# Reshape to 2D
X_2d = X.reshape(3, 4)
print(f"Reshaped to (3, 4):\\n{X_2d}")
print()

# Reshape to 3D
X_3d = X.reshape(2, 2, 3)
print(f"Reshaped to (2, 2, 3):\\n{X_3d}")
print()

# Flatten
X_flat = X_3d.flatten()
print(f"Flattened: {X_flat}")
print()

# Transpose (swap axes)
X_T = X_2d.T
print(f"Transpose of (3, 4):\\n{X_T}")
print(f"Shape: {X_T.shape}")
\`\`\`

## Tensor Contraction (Einstein Summation)

Einstein summation (\`einsum\`) is a powerful notation for tensor operations.

\`\`\`python
print("\\n=== Einstein Summation ===")

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiplication: C[i,j] = Σₖ A[i,k] * B[k,j]
C_matmul = np.einsum('ik,kj->ij', A, B)
print("Matrix multiplication via einsum:")
print(f"A @ B =\\n{C_matmul}")
print(f"Verify: {np.allclose(C_matmul, A @ B)}")
print()

# Trace: sum of diagonal
trace = np.einsum('ii->', A)
print(f"Trace of A: {trace}")
print(f"Verify: {np.trace(A)}")
print()

# Outer product
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5])
outer = np.einsum('i,j->ij', v1, v2)
print(f"Outer product:\\n{outer}")
print(f"Verify: {np.allclose (outer, np.outer (v1, v2))}")
\`\`\`

## Batched Operations

Deep learning processes multiple samples simultaneously (batching).

\`\`\`python
print("\\n=== Batched Operations ===")

# Batch of samples
batch_size = 4
input_dim = 3
output_dim = 2

# Weight matrix
W = np.random.randn (input_dim, output_dim)
b = np.random.randn (output_dim)

# Batch of inputs
X_batch = np.random.randn (batch_size, input_dim)

print(f"Batch size: {batch_size}")
print(f"Input dim: {input_dim}")
print(f"Output dim: {output_dim}")
print()

# Batched matrix multiplication
Z = X_batch @ W + b  # Broadcasting bias

print(f"X_batch shape: {X_batch.shape}")
print(f"W shape: {W.shape}")
print(f"Z shape: {Z.shape}")
print()
print("Each row of Z is the transformation of corresponding row in X_batch")
\`\`\`

## Convolutional Operations

Convolution is a key operation in computer vision.

\`\`\`python
print("\\n=== Convolution (Simplified) ===")

from scipy.signal import correlate2d

# 5×5 image
image = np.array([
    [1, 2, 3, 4, 5],
    [5, 4, 3, 2, 1],
    [1, 2, 3, 4, 5],
    [5, 4, 3, 2, 1],
    [1, 2, 3, 4, 5]
])

# 3×3 edge detection kernel
kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

# Convolution (technically cross-correlation)
output = correlate2d (image, kernel, mode='valid')

print(f"Image shape: {image.shape}")
print(f"Kernel shape: {kernel.shape}")
print(f"Output shape: {output.shape}")
print()
print("Output (edge detected):")
print(output)
\`\`\`

## Applications in Deep Learning

### 1. Matrix Multiplication (Dense Layers)

\`\`\`python
print("\\n=== Application: Dense Layer ===")

batch_size = 32
input_features = 784  # e.g., 28×28 image flattened
hidden_units = 128

# Simulate layer
X = np.random.randn (batch_size, input_features)
W = np.random.randn (input_features, hidden_units) * 0.01
b = np.zeros (hidden_units)

# Forward pass
Z = X @ W + b
A = np.maximum(0, Z)  # ReLU

print(f"Input: {X.shape}")
print(f"Weights: {W.shape}")
print(f"Output (before activation): {Z.shape}")
print(f"Output (after ReLU): {A.shape}")
\`\`\`

### 2. Batch Normalization

\`\`\`python
print("\\n=== Application: Batch Normalization ===")

# Batch of activations
X_bn = np.random.randn(32, 10)  # 32 samples, 10 features

# Compute mean and std across batch
mean = np.mean(X_bn, axis=0, keepdims=True)
std = np.std(X_bn, axis=0, keepdims=True)

# Normalize
X_normalized = (X_bn - mean) / (std + 1e-8)

# Scale and shift (learnable parameters)
gamma = np.ones((1, 10))
beta = np.zeros((1, 10))
X_bn_out = gamma * X_normalized + beta

print(f"Input shape: {X_bn.shape}")
print(f"Mean shape: {mean.shape}")
print(f"Output shape: {X_bn_out.shape}")
print()
print(f"Output mean: {np.mean(X_bn_out, axis=0)}")  # ≈ 0
print(f"Output std: {np.std(X_bn_out, axis=0)}")    # ≈ 1
\`\`\`

### 3. Attention Mechanism

\`\`\`python
print("\\n=== Application: Attention Mechanism (Simplified) ===")

seq_len = 5
d_model = 8

# Query, Key, Value matrices
Q = np.random.randn (seq_len, d_model)
K = np.random.randn (seq_len, d_model)
V = np.random.randn (seq_len, d_model)

# Attention scores: QK^T / sqrt (d_model)
scores = (Q @ K.T) / np.sqrt (d_model)

# Softmax
def softmax (x):
    exp_x = np.exp (x - np.max (x, axis=-1, keepdims=True))
    return exp_x / np.sum (exp_x, axis=-1, keepdims=True)

attention_weights = softmax (scores)

# Weighted sum of values
output = attention_weights @ V

print(f"Q shape: {Q.shape}")
print(f"K shape: {K.shape}")
print(f"V shape: {V.shape}")
print(f"Attention weights shape: {attention_weights.shape}")
print(f"Output shape: {output.shape}")
print()
print("Attention weights (row i shows attention from token i to all tokens):")
print(attention_weights.round(2))
\`\`\`

## Memory Layout and Performance

\`\`\`python
print("\\n=== Memory Layout ===")

# Row-major (C-style) vs Column-major (Fortran-style)
A_row_major = np.array([[1, 2, 3], [4, 5, 6]], order='C')
A_col_major = np.array([[1, 2, 3], [4, 5, 6]], order='F')

print("Row-major (C-style): consecutive elements in same row are contiguous")
print("Column-major (F-style): consecutive elements in same column are contiguous")
print()

# Check strides (bytes to skip for each dimension)
print(f"Row-major strides: {A_row_major.strides}")
print(f"Column-major strides: {A_col_major.strides}")
print()

print("→ Access patterns matter for performance!")
print("→ Iterate along contiguous dimension for cache efficiency")
\`\`\`

## Summary

**Tensors**: Multi-dimensional arrays (generalize vectors/matrices)
- 0D: Scalar
- 1D: Vector
- 2D: Matrix
- 3D+: Higher-order tensors

**Key Operations**:
- **Element-wise**: +, *, exp, etc. (Hadamard)
- **Reduction**: sum, mean, max along axes
- **Broadcasting**: Automatic shape expansion
- **Reshaping**: Change dimensions without copying data
- **Contraction**: Einstein summation (einsum)
- **Batching**: Process multiple samples simultaneously

**Deep Learning Applications**:
- **Dense layers**: Batched matrix multiplication
- **Convolution**: Local connectivity for images
- **Batch normalization**: Normalize across batch
- **Attention**: Weighted aggregation of sequences
- **Memory layout**: Row vs column major affects performance

**Why Tensors in ML**:
1. **Batching**: Process multiple samples in parallel (GPU efficiency)
2. **High-dimensional data**: Images (4D), videos (5D), sequences
3. **Unified operations**: Same code for scalars, vectors, matrices, tensors
4. **Automatic differentiation**: Deep learning frameworks compute gradients
5. **Hardware acceleration**: GPUs/TPUs optimized for tensor operations

**Best Practices**:
- Use batching for parallel processing
- Leverage broadcasting to avoid loops
- Be mindful of shapes (debug tool: print tensor.shape)
- Use contiguous memory layouts when possible
- Prefer built-in operations over manual loops (vectorization)

Understanding tensor operations is essential for implementing and optimizing deep learning models!
`,
};
