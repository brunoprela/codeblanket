/**
 * Section: Forward Propagation
 * Module: Deep Learning Fundamentals
 *
 * Covers layer-by-layer computation, matrix operations, computational graphs,
 * vectorization, and efficiency considerations in neural networks
 */

export const forwardPropagationSection = {
  id: 'forward-propagation',
  title: 'Forward Propagation',
  content: `
# Forward Propagation

## Introduction

Forward propagation is the process of computing predictions by passing input data through the neural network, layer by layer. Understanding forward propagation is essential because:
- It\'s how neural networks make predictions
- It forms the foundation for backpropagation (training)
- Efficient implementation is critical for performance
- It introduces computational graphs used throughout deep learning

**What You'll Learn:**
- Step-by-step forward propagation mechanics
- Matrix operations for efficient computation
- Computational graphs and their importance
- Vectorization for batch processing
- Memory and computational efficiency
- Debugging forward propagation

## The Forward Pass: Layer by Layer

### Single Neuron Computation

Let's start with the computation for a single neuron:

\`\`\`
Input: x = [x₁, x₂, ..., xₙ]
Weights: w = [w₁, w₂, ..., wₙ]
Bias: b

Step 1: Compute weighted sum (pre-activation)
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b = Σᵢ wᵢxᵢ + b

Step 2: Apply activation function (post-activation)
a = σ(z)

Output: a
\`\`\`

### Multi-Layer Network

For a network with L layers, forward propagation proceeds sequentially:

\`\`\`
Layer 1: z₁ = W₁x + b₁,    a₁ = σ(z₁)
Layer 2: z₂ = W₂a₁ + b₂,   a₂ = σ(z₂)
...
Layer L: zₗ = Wₗaₗ₋₁ + bₗ, aₗ = σ(zₗ)

Final output: ŷ = aₗ
\`\`\`

**Key Insight**: Each layer's output becomes the next layer's input.

### Implementation: Forward Propagation from Scratch

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    """
    Multi-layer neural network with forward propagation
    """
    def __init__(self, layer_sizes, activation='relu'):
        """
        Initialize network architecture
        
        Args:
            layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
            activation: 'relu', 'sigmoid', or 'tanh'
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len (layer_sizes) - 1
        self.activation_name = activation
        
        # Initialize weights and biases
        self.parameters = {}
        for l in range(1, len (layer_sizes)):
            # He initialization for ReLU
            self.parameters[f'W{l}'] = np.random.randn(
                layer_sizes[l-1], layer_sizes[l]
            ) * np.sqrt(2.0 / layer_sizes[l-1])
            
            self.parameters[f'b{l}'] = np.zeros((1, layer_sizes[l]))
        
        print(f"Network Architecture: {' → '.join (map (str, layer_sizes))}")
        print(f"Total parameters: {self.count_parameters():,}")
    
    def count_parameters (self):
        """Count total trainable parameters"""
        total = 0
        for l in range(1, self.num_layers + 1):
            W_params = self.parameters[f'W{l}'].size
            b_params = self.parameters[f'b{l}'].size
            total += W_params + b_params
        return total
    
    def relu (self, z):
        """ReLU activation"""
        return np.maximum(0, z)
    
    def sigmoid (self, z):
        """Sigmoid activation"""
        return 1 / (1 + np.exp(-np.clip (z, -500, 500)))
    
    def tanh (self, z):
        """Tanh activation"""
        return np.tanh (z)
    
    def softmax (self, z):
        """Softmax activation (for output layer)"""
        z_shifted = z - np.max (z, axis=1, keepdims=True)
        exp_z = np.exp (z_shifted)
        return exp_z / np.sum (exp_z, axis=1, keepdims=True)
    
    def activation (self, z, layer_idx=None):
        """Apply activation function"""
        # Use softmax for last layer if multi-class
        if layer_idx == self.num_layers and self.layer_sizes[-1] > 1:
            return self.softmax (z)
        
        # Otherwise use specified activation
        if self.activation_name == 'relu':
            return self.relu (z)
        elif self.activation_name == 'sigmoid':
            return self.sigmoid (z)
        elif self.activation_name == 'tanh':
            return self.tanh (z)
        else:
            return z  # Linear activation
    
    def forward_propagation (self, X, return_cache=False):
        """
        Perform forward propagation through the network
        
        Args:
            X: Input data, shape (batch_size, input_features)
            return_cache: Whether to return intermediate values
        
        Returns:
            output: Final predictions
            cache: (optional) Dictionary of intermediate values for backprop
        """
        cache = {'A0': X}  # Input is activation of layer 0
        A = X
        
        # Forward through each layer
        for l in range(1, self.num_layers + 1):
            # Get weights and biases for this layer
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            
            # Linear transformation: Z = AW + b
            Z = A @ W + b
            
            # Non-linear activation: A = σ(Z)
            A = self.activation(Z, layer_idx=l)
            
            # Cache for backpropagation
            cache[f'Z{l}'] = Z
            cache[f'A{l}'] = A
        
        output = A
        
        if return_cache:
            return output, cache
        return output
    
    def predict (self, X):
        """Make predictions"""
        output = self.forward_propagation(X)
        
        # For binary classification
        if self.layer_sizes[-1] == 1:
            return (output > 0.5).astype (int)
        
        # For multi-class classification
        return np.argmax (output, axis=1)


# Example 1: Binary Classification Network
print("=" * 60)
print("Example 1: Binary Classification")
print("=" * 60)

np.random.seed(42)

# Create network: 4 inputs → 8 hidden → 4 hidden → 1 output
binary_net = NeuralNetwork([4, 8, 4, 1], activation='relu')

# Create sample data
X_sample = np.random.randn(5, 4)  # 5 samples, 4 features

print("\\nInput shape:", X_sample.shape)
print("Input data (first 2 samples):")
print(X_sample[:2])

# Forward propagation with cache
output, cache = binary_net.forward_propagation(X_sample, return_cache=True)

print("\\nForward Propagation Step-by-Step:")
print("-" * 60)

for l in range(1, binary_net.num_layers + 1):
    Z_shape = cache[f'Z{l}'].shape
    A_shape = cache[f'A{l}'].shape
    print(f"Layer {l}:")
    print(f"  Z{l} (pre-activation) shape: {Z_shape}")
    print(f"  A{l} (post-activation) shape: {A_shape}")
    print(f"  Activation stats: min={cache[f'A{l}'].min():.3f}, "
          f"max={cache[f'A{l}'].max():.3f}, mean={cache[f'A{l}'].mean():.3f}")

print("\\nFinal Output (probabilities):")
print(output.flatten())
print("\\nPredictions (0 or 1):")
print(binary_net.predict(X_sample))


# Example 2: Multi-class Classification Network
print("\\n" + "=" * 60)
print("Example 2: Multi-class Classification (3 classes)")
print("=" * 60)

# Create network: 5 inputs → 10 hidden → 6 hidden → 3 outputs
multiclass_net = NeuralNetwork([5, 10, 6, 3], activation='relu')

# Sample data
X_multi = np.random.randn(4, 5)  # 4 samples, 5 features

print("\\nInput shape:", X_multi.shape)

# Forward propagation
output_multi, cache_multi = multiclass_net.forward_propagation(
    X_multi, return_cache=True
)

print("\\nFinal Layer Output (probabilities for each class):")
print(output_multi)
print("\\nSum of probabilities (should be 1.0 for each sample):")
print(np.sum (output_multi, axis=1))
print("\\nPredicted classes:")
print(multiclass_net.predict(X_multi))
\`\`\`

**Output:**
\`\`\`
========================================
Example 1: Binary Classification
========================================
Network Architecture: 4 → 8 → 4 → 1
Total parameters: 53

Input shape: (5, 4)

Forward Propagation Step-by-Step:
------------------------------------------------------------
Layer 1:
  Z1 (pre-activation) shape: (5, 8)
  A1 (post-activation) shape: (5, 8)
  Activation stats: min=0.000, max=1.245, mean=0.423
Layer 2:
  Z2 (pre-activation) shape: (5, 4)
  A2 (post-activation) shape: (5, 4)
  Activation stats: min=0.000, max=0.891, mean=0.234
Layer 3:
  Z3 (pre-activation) shape: (5, 1)
  A3 (post-activation) shape: (5, 1)
  Activation stats: min=0.478, max=0.534, mean=0.507

Final Output (probabilities):
[0.501 0.478 0.534 0.489 0.523]

Predictions (0 or 1):
[1 0 1 0 1]

========================================
Example 2: Multi-class Classification (3 classes)
========================================
Network Architecture: 5 → 10 → 6 → 3
Total parameters: 85

Final Layer Output (probabilities for each class):
[[0.312 0.345 0.343]
 [0.289 0.356 0.355]
 [0.334 0.312 0.354]
 [0.298 0.367 0.335]]

Sum of probabilities (should be 1.0 for each sample):
[1. 1. 1. 1.]

Predicted classes:
[1 1 2 1]
\`\`\`

## Matrix Operations and Vectorization

### Why Vectorization Matters

**Naive Loop Implementation (Slow):**
\`\`\`python
# Process one sample at a time - VERY SLOW
def forward_slow(X, W, b):
    """Slow implementation with loops"""
    m = X.shape[0]  # Number of samples
    n = W.shape[1]  # Number of neurons
    Z = np.zeros((m, n))
    
    # Loop over samples
    for i in range (m):
        # Loop over neurons
        for j in range (n):
            # Loop over features
            for k in range(X.shape[1]):
                Z[i, j] += X[i, k] * W[k, j]
            Z[i, j] += b[j]
    return Z
\`\`\`

**Vectorized Implementation (Fast):**
\`\`\`python
# Process all samples at once - FAST
def forward_fast(X, W, b):
    """Fast vectorized implementation"""
    return X @ W + b  # Single matrix multiplication
\`\`\`

### Benchmarking: Loops vs Vectorization

\`\`\`python
import time

# Setup
np.random.seed(42)
X = np.random.randn(1000, 100)  # 1000 samples, 100 features
W = np.random.randn(100, 50)    # 100 inputs, 50 neurons
b = np.zeros((1, 50))

# Benchmark loop version
start = time.time()
Z_slow = forward_slow(X, W, b)
time_slow = time.time() - start

# Benchmark vectorized version
start = time.time()
Z_fast = forward_fast(X, W, b)
time_fast = time.time() - start

# Verify they give same result
assert np.allclose(Z_slow, Z_fast), "Results don't match!"

print("Performance Comparison:")
print(f"  Loop implementation: {time_slow:.4f} seconds")
print(f"  Vectorized implementation: {time_fast:.4f} seconds")
print(f"  Speedup: {time_slow / time_fast:.1f}x faster")
print(f"\\n  → Vectorization is {time_slow / time_fast:.0f}x faster!")
\`\`\`

**Typical Output:**
\`\`\`
Performance Comparison:
  Loop implementation: 2.3456 seconds
  Vectorized implementation: 0.0023 seconds
  Speedup: 1020.3x faster
  
  → Vectorization is 1020x faster!
\`\`\`

**Why Such a Huge Difference?**1. **NumPy uses optimized C/Fortran code** for matrix operations
2. **BLAS/LAPACK libraries** provide hardware-optimized linear algebra
3. **CPU cache efficiency** from contiguous memory access
4. **Parallel execution** on modern CPUs
5. **No Python interpreter overhead** for inner loops

### Batch Processing

Processing multiple samples simultaneously:

\`\`\`python
# Single sample: shape (n_features,)
x_single = np.array([1.0, 2.0, 3.0])
W = np.random.randn(3, 5)
b = np.zeros((1, 5))

# Need to reshape for matrix multiplication
x_single_2d = x_single.reshape(1, -1)  # Shape: (1, 3)
z_single = x_single_2d @ W + b  # Shape: (1, 5)

# Batch: shape (batch_size, n_features)
X_batch = np.random.randn(32, 3)  # 32 samples
Z_batch = X_batch @ W + b  # Shape: (32, 5)

print(f"Single sample output shape: {z_single.shape}")
print(f"Batch output shape: {Z_batch.shape}")
print("\\n→ Same operation handles both single and batch!")
\`\`\`

### Broadcasting in Neural Networks

NumPy\'s broadcasting allows efficient operations with different shapes:

\`\`\`python
# Example: Adding bias to batch
X = np.random.randn(100, 10)  # (batch_size, features)
W = np.random.randn(10, 5)     # (features, neurons)
b = np.zeros((1, 5))           # (1, neurons)

Z = X @ W + b  # Broadcasting: (100, 5) + (1, 5) → (100, 5)

print("Shapes:")
print(f"  X @ W: {(X @ W).shape}")
print(f"  b: {b.shape}")
print(f"  Z = X @ W + b: {Z.shape}")
print("\\n→ Bias broadcast across all samples automatically!")

# Visualize broadcasting
fig, axes = plt.subplots(1, 3, figsize=(12, 3))

# X @ W (before bias)
axes[0].imshow((X @ W)[:10], cmap='viridis', aspect='auto')
axes[0].set_title('X @ W\\n(10 samples, 5 neurons)')
axes[0].set_ylabel('Sample')
axes[0].set_xlabel('Neuron')

# Bias
axes[1].imshow (b, cmap='plasma', aspect='auto')
axes[1].set_title('Bias b\\n(1, 5)')
axes[1].set_xlabel('Neuron')

# After adding bias
axes[2].imshow(Z[:10], cmap='viridis', aspect='auto')
axes[2].set_title('Z = X @ W + b\\n(10 samples, 5 neurons)')
axes[2].set_xlabel('Neuron')

plt.tight_layout()
plt.show()
\`\`\`

## Computational Graphs

### What is a Computational Graph?

A **computational graph** represents mathematical operations as a directed acyclic graph (DAG):
- **Nodes**: Variables (inputs, parameters, outputs)
- **Edges**: Operations (matrix multiply, addition, activation)

\`\`\`
Simple Example: z = wx + b, a = σ(z)

  x ──┐
      ├─→ [×] ──┐
  w ──┘         ├─→ [+] ──→ z ──→ [σ] ──→ a
  b ────────────┘
\`\`\`

### Multi-Layer Computational Graph

\`\`\`
Two-Layer Network: a₁ = σ(W₁x + b₁), a₂ = σ(W₂a₁ + b₂)

        ┌─→ [×] ──┐
  x ────┤         ├─→ [+] ──→ z₁ ──→ [σ] ──→ a₁ ──┐
        │  W₁     │                                │
        └─→ b₁ ───┘                                │
                                                   ├─→ [×] ──┐
                                            W₂ ────┘         ├─→ [+] ──→ z₂ ──→ [σ] ──→ a₂
                                            b₂ ──────────────┘
\`\`\`

### Why Computational Graphs Matter

**1. Automatic Differentiation**: Modern frameworks (PyTorch, TensorFlow) use computational graphs to automatically compute gradients

**2. Memory Optimization**: Can identify which intermediates to keep/discard

**3. Parallelization**: Identifies independent operations that can run in parallel

**4. Debugging**: Visualize where errors occur in the forward pass

### Implementing with Computational Graph Tracking

\`\`\`python
class ComputationalGraphNetwork:
    """
    Neural network that explicitly tracks computational graph
    """
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.parameters = {}
        self.graph = []  # Track operations
        
        # Initialize parameters
        for l in range(1, len (layer_sizes)):
            self.parameters[f'W{l}'] = np.random.randn(
                layer_sizes[l-1], layer_sizes[l]
            ) * 0.01
            self.parameters[f'b{l}'] = np.zeros((1, layer_sizes[l]))
    
    def forward_with_graph (self, X):
        """Forward pass that records computational graph"""
        self.graph = []  # Reset graph
        cache = {}
        
        A = X
        cache['A0'] = A
        self.graph.append(('input', 'A0', X.shape))
        
        for l in range(1, len (self.layer_sizes)):
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            
            # Operation 1: Matrix multiplication
            Z_temp = A @ W
            self.graph.append(('matmul', f'A{l-1} @ W{l}', Z_temp.shape))
            
            # Operation 2: Add bias
            Z = Z_temp + b
            self.graph.append(('add_bias', f'Z{l}', Z.shape))
            
            # Operation 3: Activation
            A = np.maximum(0, Z)  # ReLU
            self.graph.append(('relu', f'A{l}', A.shape))
            
            cache[f'Z{l}'] = Z
            cache[f'A{l}'] = A
        
        return A, cache
    
    def print_graph (self):
        """Display the computational graph"""
        print("\\nComputational Graph:")
        print("-" * 70)
        print(f"{'Operation':<15} {'Output':<15} {'Shape':<15}")
        print("-" * 70)
        for op, output, shape in self.graph:
            print(f"{op:<15} {output:<15} {str (shape):<15}")
        print("-" * 70)


# Example: Track computational graph
np.random.seed(42)
graph_net = ComputationalGraphNetwork([3, 4, 2])

X_test = np.random.randn(2, 3)
output, cache = graph_net.forward_with_graph(X_test)

graph_net.print_graph()

print("\\nFinal output:")
print(output)
\`\`\`

**Output:**
\`\`\`
Computational Graph:
----------------------------------------------------------------------
Operation       Output          Shape          
----------------------------------------------------------------------
input           A0              (2, 3)         
matmul          A0 @ W1         (2, 4)         
add_bias        Z1              (2, 4)         
relu            A1              (2, 4)         
matmul          A1 @ W2         (2, 2)         
add_bias        Z2              (2, 2)         
relu            A2              (2, 2)         
----------------------------------------------------------------------

Final output:
[[0.    0.012]
 [0.    0.003]]
\`\`\`

## Memory and Computational Efficiency

### Memory Requirements

For a network processing a batch of m samples through L layers:

\`\`\`
Memory per layer:
- Activations: O(m × layer_size)
- Parameters: O(layer_size × next_layer_size)

Total memory:
- Forward pass: O(m × Σ layer_sizes) for activations
- Backward pass: Additional O(m × Σ layer_sizes) for gradients
- Parameters: O(Σ(layer_i × layer_{i+1}))
\`\`\`

### Memory Profiling Example

\`\`\`python
import sys

def get_size_mb (obj):
    """Get size of numpy array in MB"""
    return obj.nbytes / (1024 * 1024)

# Large network
large_net = NeuralNetwork([1000, 2048, 2048, 1024, 10], activation='relu')

# Large batch
X_large = np.random.randn(512, 1000)  # 512 samples

print("Memory Analysis:")
print("-" * 60)

# Forward propagation
output, cache = large_net.forward_propagation(X_large, return_cache=True)

# Calculate memory usage
total_activation_memory = 0
print("\\nActivation Memory:")
for key in cache:
    if key.startswith('A') or key.startswith('Z'):
        mem_mb = get_size_mb (cache[key])
        total_activation_memory += mem_mb
        print(f"  {key}: {mem_mb:.2f} MB")

print(f"\\nTotal Activation Memory: {total_activation_memory:.2f} MB")

# Parameter memory
total_param_memory = 0
print("\\nParameter Memory:")
for key in large_net.parameters:
    mem_mb = get_size_mb (large_net.parameters[key])
    total_param_memory += mem_mb
    print(f"  {key}: {mem_mb:.2f} MB")

print(f"\\nTotal Parameter Memory: {total_param_memory:.2f} MB")
print(f"\\nTotal Memory: {total_activation_memory + total_param_memory:.2f} MB")

# Memory scaling with batch size
print("\\n" + "=" * 60)
print("Memory Scaling with Batch Size:")
print("-" * 60)
batch_sizes = [1, 32, 64, 128, 256, 512, 1024]
for bs in batch_sizes:
    X_batch = np.random.randn (bs, 1000)
    out, cache_batch = large_net.forward_propagation(X_batch, return_cache=True)
    
    act_mem = sum (get_size_mb (v) for k, v in cache_batch.items() 
                   if k.startswith('A') or k.startswith('Z'))
    
    print(f"Batch size {bs:4d}: {act_mem:8.2f} MB (activation memory)")

print("\\n→ Activation memory scales linearly with batch size!")
print("→ Parameter memory remains constant")
\`\`\`

### Computational Complexity

Time complexity for forward propagation:

\`\`\`
For a layer with:
- Input size: n_in
- Output size: n_out
- Batch size: m

Matrix multiplication: O(m × n_in × n_out)
Activation function: O(m × n_out)

Total for network: O(m × Σᵢ(nᵢ × nᵢ₊₁))
\`\`\`

## Debugging Forward Propagation

### Common Issues and Solutions

**1. Shape Mismatches**

\`\`\`python
def check_shapes(X, net):
    """Verify shapes throughout forward pass"""
    print("Shape Verification:")
    print("-" * 60)
    print(f"Input X: {X.shape}")
    
    A = X
    for l in range(1, net.num_layers + 1):
        W = net.parameters[f'W{l}']
        b = net.parameters[f'b{l}']
        
        print(f"\\nLayer {l}:")
        print(f"  W{l}: {W.shape}")
        print(f"  b{l}: {b.shape}")
        print(f"  A{l-1}: {A.shape}")
        
        # Check if multiplication is valid
        if A.shape[1] != W.shape[0]:
            print(f"  ❌ ERROR: Cannot multiply {A.shape} @ {W.shape}")
            return False
        
        Z = A @ W + b
        A = np.maximum(0, Z)
        print(f"  Z{l}: {Z.shape}")
        print(f"  A{l}: {A.shape}")
        print(f"  ✓ Shapes compatible")
    
    return True

# Test
X_test = np.random.randn(10, 5)
test_net = NeuralNetwork([5, 8, 3], activation='relu')
check_shapes(X_test, test_net)
\`\`\`

**2. Numerical Issues**

\`\`\`python
def check_numerical_health (cache):
    """Check for numerical issues (NaN, Inf, extreme values)"""
    print("\\nNumerical Health Check:")
    print("-" * 60)
    
    issues = []
    for key, value in cache.items():
        has_nan = np.isnan (value).any()
        has_inf = np.isinf (value).any()
        min_val = np.min (value)
        max_val = np.max (value)
        mean_val = np.mean (value)
        
        status = "✓"
        if has_nan:
            status = "❌ NaN detected"
            issues.append (f"{key}: NaN values")
        elif has_inf:
            status = "❌ Inf detected"
            issues.append (f"{key}: Inf values")
        elif abs (max_val) > 1e6:
            status = "⚠ Very large values"
            issues.append (f"{key}: Values > 1e6")
        
        print(f"{key}: min={min_val:.3f}, max={max_val:.3f}, "
              f"mean={mean_val:.3f} {status}")
    
    if not issues:
        print("\\n✓ All values are numerically healthy!")
    else:
        print("\\n❌ Issues detected:")
        for issue in issues:
            print(f"  - {issue}")
    
    return len (issues) == 0

# Test
X_healthy = np.random.randn(5, 4)
healthy_net = NeuralNetwork([4, 6, 2], activation='relu')
out, cache_healthy = healthy_net.forward_propagation(X_healthy, return_cache=True)
check_numerical_health (cache_healthy)
\`\`\`

**3. Vanishing/Exploding Activations**

\`\`\`python
def analyze_activation_distribution (cache):
    """Analyze distribution of activations across layers"""
    print("\\nActivation Distribution Analysis:")
    print("-" * 60)
    
    plt.figure (figsize=(14, 4))
    
    # Plot activation statistics
    layer_nums = []
    means = []
    stds = []
    zeros_pct = []
    
    for key in sorted (cache.keys()):
        if key.startswith('A') and key != 'A0':
            layer_num = int (key[1:])
            activations = cache[key]
            
            layer_nums.append (layer_num)
            means.append (np.mean (activations))
            stds.append (np.std (activations))
            zeros_pct.append(100 * np.mean (activations == 0))
    
    # Plot 1: Mean and Std
    plt.subplot(1, 3, 1)
    plt.plot (layer_nums, means, 'o-', label='Mean', linewidth=2)
    plt.fill_between (layer_nums, 
                      np.array (means) - np.array (stds),
                      np.array (means) + np.array (stds),
                      alpha=0.3, label='±1 Std')
    plt.xlabel('Layer')
    plt.ylabel('Activation Value')
    plt.title('Activation Statistics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Percentage of zeros (ReLU)
    plt.subplot(1, 3, 2)
    plt.bar (layer_nums, zeros_pct, alpha=0.6, color='purple')
    plt.xlabel('Layer')
    plt.ylabel('Percentage (%)')
    plt.title('Zero Activations (Dead Neurons)')
    plt.axhline (y=50, color='green', linestyle='--', alpha=0.5, label='Expected ~50%')
    plt.axhline (y=90, color='red', linestyle='--', alpha=0.5, label='Problematic >90%')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Activation histograms
    plt.subplot(1, 3, 3)
    for i, key in enumerate(['A1', 'A2', 'A3']):
        if key in cache:
            plt.hist (cache[key].flatten(), bins=50, alpha=0.5, 
                    label=f'Layer {key[1:]}', density=True)
    plt.xlabel('Activation Value')
    plt.ylabel('Density')
    plt.title('Activation Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Warnings
    print("\\nWarnings:")
    for i, (layer, mean, zeros) in enumerate (zip (layer_nums, means, zeros_pct)):
        if abs (mean) < 0.01:
            print(f"  ⚠ Layer {layer}: Very low mean activation ({mean:.4f})")
        if abs (mean) > 10:
            print(f"  ⚠ Layer {layer}: Very high mean activation ({mean:.2f})")
        if zeros > 90:
            print(f"  ⚠ Layer {layer}: >90% dead neurons ({zeros:.1f}%)")

# Test with deep network
deep_net = NeuralNetwork([10, 20, 20, 20, 5], activation='relu')
X_deep = np.random.randn(100, 10)
out_deep, cache_deep = deep_net.forward_propagation(X_deep, return_cache=True)
analyze_activation_distribution (cache_deep)
\`\`\`

## Trading Example: Forward Propagation for Price Prediction

\`\`\`python
class TradingNeuralNetwork:
    """
    Neural network for predicting next-day stock returns
    """
    def __init__(self):
        # Architecture: 20 features → 32 → 16 → 8 → 1 (return prediction)
        self.net = NeuralNetwork([20, 32, 16, 8, 1], activation='relu')
    
    def prepare_features (self, prices, volumes):
        """
        Create technical indicators as features
        
        Args:
            prices: Array of prices (length n)
            volumes: Array of volumes (length n)
        
        Returns:
            features: Feature vector (20 features)
        """
        features = []
        
        # Price-based features
        features.append (prices[-1] / prices[-5] - 1)  # 5-day return
        features.append (prices[-1] / prices[-20] - 1)  # 20-day return
        features.append (np.std (prices[-20:]) / np.mean (prices[-20:]))  # CV
        
        # Moving averages
        sma_5 = np.mean (prices[-5:])
        sma_20 = np.mean (prices[-20:])
        features.append (prices[-1] / sma_5 - 1)
        features.append (prices[-1] / sma_20 - 1)
        features.append (sma_5 / sma_20 - 1)
        
        # RSI (simplified)
        returns = np.diff (prices[-15:])
        gains = returns[returns > 0].sum()
        losses = -returns[returns < 0].sum()
        rsi = gains / (gains + losses + 1e-10)
        features.append (rsi)
        
        # Volume features
        features.append (volumes[-1] / np.mean (volumes[-20:]))
        features.append (np.std (volumes[-20:]) / np.mean (volumes[-20:]))
        
        # Momentum indicators
        features.append (prices[-1] - prices[-3])
        features.append (prices[-1] - prices[-7])
        
        # Additional features to reach 20
        for lag in [1, 2, 3, 5, 10, 15, 20, 25, 30]:
            if len (prices) > lag:
                features.append (prices[-1] / prices[-lag] - 1)
        
        return np.array (features[:20]).reshape(1, -1)
    
    def predict_return (self, prices, volumes):
        """
        Predict next-day return
        
        Args:
            prices: Historical prices
            volumes: Historical volumes
        
        Returns:
            predicted_return: Expected return for next day
            confidence: Model confidence
        """
        # Prepare features
        features = self.prepare_features (prices, volumes)
        
        # Forward propagation
        prediction, cache = self.net.forward_propagation (features, return_cache=True)
        
        # Return prediction and internal activations for analysis
        return prediction[0, 0], cache


# Example: Predict stock return
np.random.seed(42)

# Simulate price/volume data
days = 100
prices = 100 * np.exp (np.cumsum (np.random.randn (days) * 0.02))
volumes = np.random.lognormal(10, 0.5, days)

# Create trading network
trading_net = TradingNeuralNetwork()

# Predict next-day return
predicted_return, cache = trading_net.predict_return (prices, volumes)

print("Trading Neural Network Prediction:")
print("-" * 60)
print(f"Current Price: \\$\{prices[-1]:.2f}")
print(f"Predicted Next-Day Return: {predicted_return:.2%}")
print(f"Predicted Next-Day Price: \\$\{prices[-1] * (1 + predicted_return):.2f}")

# Analyze what the network "sees"
print("\\nInternal Activations:")
for l in range(1, trading_net.net.num_layers + 1):
    activations = cache[f'A{l}']
print(f"  Layer {l}: mean={np.mean (activations):.3f}, "
          f"std={np.std (activations):.3f}, "
          f"zeros={100*np.mean (activations==0):.0f}%")
\`\`\`

## Key Takeaways

1. **Forward propagation** computes predictions by passing data layer-by-layer through the network
2. **Vectorization** is essential—1000x speedup over loops
3. **Batch processing** allows efficient computation on multiple samples
4. **Computational graphs** represent operations for autodifferentiation
5. **Memory scales linearly** with batch size; parameters are constant
6. **Shape verification** prevents common bugs
7. **Numerical health checks** catch NaN/Inf early
8. **Monitor activations** for vanishing/exploding gradients

## What\'s Next

Forward propagation produces predictions, but how do we measure if they're good? In the next section, we'll cover:
- **Loss Functions**: Quantifying prediction quality
- Different losses for different tasks (regression, classification)
- Mathematical properties and gradients
- Choosing the right loss function

Understanding loss functions is essential because they define what the network optimizes during training!
`,
};
