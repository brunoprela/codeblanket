import { QuizQuestion } from '../../../types';

export const forwardPropagationQuiz: QuizQuestion[] = [
  {
    id: 'forward-propagation-dq-1',
    question:
      'Explain why vectorization provides such dramatic speedups (often 100-1000x) over loop-based implementations in neural networks. What are the underlying hardware and software mechanisms that enable this performance?',
    sampleAnswer: `Vectorization achieves dramatic speedups through multiple complementary mechanisms at both hardware and software levels:

**1. Optimized Low-Level Implementation:**

NumPy\'s vectorized operations are implemented in C and Fortran, which are compiled languages orders of magnitude faster than interpreted Python:
- Python loops: Each iteration involves Python interpreter overhead
- Vectorized: Single call to compiled code, no per-element Python overhead
- Speedup factor: 10-100x just from avoiding interpretation

**2. BLAS/LAPACK Integration:**

NumPy links to highly optimized linear algebra libraries:
- BLAS (Basic Linear Algebra Subprograms): Optimized for specific CPU architectures
- LAPACK (Linear Algebra PACKage): Advanced linear algebra routines
- Vendors (Intel MKL, OpenBLAS, Apple Accelerate) optimize for their hardware
- These libraries use decades of optimization research

**3. SIMD (Single Instruction, Multiple Data):**

Modern CPUs can perform the same operation on multiple data elements simultaneously:
- SSE: Process 4 floats at once
- AVX: Process 8 floats at once
- AVX-512: Process 16 floats at once
- Speedup factor: 4-16x for floating point operations

Example:
\`\`\`python
# Loop: 4 operations sequentially
result[0] = a[0] * b[0]
result[1] = a[1] * b[1]
result[2] = a[2] * b[2]
result[3] = a[3] * b[3]

# SIMD: 1 operation processes all 4 simultaneously
result[0:4] = a[0:4] * b[0:4]  # Single CPU instruction!
\`\`\`

**4. CPU Cache Optimization:**

Vectorized code accesses memory in predictable, contiguous patterns:
- Cache Locality: Data loaded into fast L1/L2/L3 cache
- Prefetching: CPU predicts next memory access, loads ahead
- Loops with random access: Frequent cache misses (100x slower)
- Vectorized: Sequential access, cache hits (100x faster)

Memory hierarchy:
- L1 Cache: ~1 ns access, ~32 KB
- L2 Cache: ~4 ns access, ~256 KB
- L3 Cache: ~10 ns access, ~8 MB
- RAM: ~100 ns access, GB-scale
- Cache misses cost 100x more than cache hits!

**5. Parallelization:**

Modern libraries automatically parallelize large operations:
- Multi-threading across CPU cores
- Loop unrolling
- Instruction-level parallelism
- Speedup factor: Near-linear with core count (2-16x)

**6. Memory Layout Optimization:**

NumPy arrays are stored contiguously in memory:
- Python lists: Pointers to scattered objects (poor cache usage)
- NumPy arrays: Contiguous block of same-type data (excellent cache usage)
- Enables efficient memory bandwidth utilization

**7. Reduced Function Call Overhead:**

\`\`\`python
# Loops: N function calls
for i in range(N):
    result[i] = func (data[i])  # N Python function calls

# Vectorized: 1 function call
result = func (data)  # Single call processes all N elements
\`\`\`

**Concrete Example:**

\`\`\`python
# Matrix multiplication: C = A @ B
# A: (m, k), B: (k, n)
# Operations: m * k * n multiplications and additions

# Loop version (3 nested loops):
for i in range (m):
    for j in range (n):
        for k in range (k):
            C[i,j] += A[i,k] * B[k,j]
# - 3 nested loops: m*n*k interpreter cycles
# - Random memory access patterns
# - No SIMD utilization
# - No cache optimization

# Vectorized version:
C = A @ B
# - Single call to optimized BLAS routine
# - SIMD instructions process 8-16 elements at once
# - Blocked algorithms optimize cache usage
# - Multi-threaded across cores
# - Result: 100-1000x faster!
\`\`\`

**Practical Measurement:**

\`\`\`python
import numpy as np
import time

A = np.random.randn(1000, 1000)
B = np.random.randn(1000, 1000)

# Loop version
start = time.time()
C_loop = np.zeros((1000, 1000))
for i in range(1000):
    for j in range(1000):
        for k in range(1000):
            C_loop[i,j] += A[i,k] * B[k,j]
time_loop = time.time() - start

# Vectorized version
start = time.time()
C_vec = A @ B
time_vec = time.time() - start

print(f"Loop: {time_loop:.2f}s")
print(f"Vectorized: {time_vec:.4f}s")
print(f"Speedup: {time_loop / time_vec:.0f}x")
\`\`\`

Typical output: Speedup of 1000-5000x

**Why Each Mechanism Matters:**

| Mechanism | Typical Speedup | Why It Works |
|-----------|----------------|--------------|
| No interpreter overhead | 10-100x | Compiled C vs Python |
| SIMD instructions | 4-16x | Parallel arithmetic |
| Cache optimization | 10-100x | Fast L1/L2 vs slow RAM |
| Multi-threading | 2-8x | Parallel cores |
| Optimized algorithms | 2-10x | Decades of research |

**Total speedup**: Multiplicative effect of all mechanisms = 100-10000x

**Financial ML Implications:**

In trading, this matters enormously:
- Backtesting 10 years of data: 1 hour vs 1000 hours (41 days!)
- Real-time prediction: <1ms vs 1 second (miss the trade)
- Training iterations: Minutes vs days

**Best Practices:**

1. Always use vectorized NumPy operations
2. Avoid Python loops over large arrays
3. Profile code to find bottlenecks
4. Use batch processing (process 100s of samples at once)
5. Ensure data is contiguous in memory
6. Use \`@\` operator for matrix multiplication (not \`np.dot\` in loops)

**Conclusion:**

Vectorization achieves 100-1000x speedups through:
- Compiled code (10-100x)
- SIMD parallelism (4-16x)
- Cache optimization (10-100x)
- Multi-threading (2-8x)
- Optimized algorithms (2-10x)

These combine multiplicatively. In deep learning and financial ML, vectorization is not optional—it's essential for practical performance.`,
    keyPoints: [
      'NumPy uses compiled C/Fortran code, avoiding Python interpreter overhead (10-100x speedup)',
      'SIMD instructions process 4-16 elements simultaneously with single CPU instruction',
      'Cache optimization: Contiguous memory access is 100x faster than random access',
      'BLAS/LAPACK libraries provide hardware-optimized linear algebra (decades of research)',
      'Multi-threading automatically parallelizes operations across CPU cores',
      'Vectorization speedups are multiplicative: Total = 100-10000x',
      'In trading, vectorization means 1-hour backtests instead of days',
    ],
  },
  {
    id: 'forward-propagation-dq-2',
    question:
      'Describe the trade-offs between batch size in forward propagation. How does batch size affect memory usage, computational speed, and gradient estimates? What batch size would you choose for training a trading model and why?',
    sampleAnswer: `Batch size is a critical hyperparameter that affects memory, speed, gradient quality, and generalization. The optimal choice involves balancing multiple competing concerns:

**Memory Trade-offs:**

Memory usage scales linearly with batch size:

\`\`\`
Memory = Parameters + Activations + Gradients

For a layer with input n_in, output n_out, batch size m:
- Parameters: n_in × n_out (constant, independent of batch size)
- Activations: m × n_out (linear in m)
- Gradients: m × n_out (linear in m)

Total activation + gradient memory: O(m × Σ layer_sizes)
\`\`\`

Example calculation:
\`\`\`python
# Network: 1000 → 512 → 256 → 128 → 1
# Float32 (4 bytes per number)

batch_size = 32:
  Activations: 32 * (1000 + 512 + 256 + 128 + 1) * 4 = 243 KB

batch_size = 256:
  Activations: 256 * (1000 + 512 + 256 + 128 + 1) * 4 = 1.9 MB

batch_size = 1024:
  Activations: 1024 * (1000 + 512 + 256 + 128 + 1) * 4 = 7.8 MB

→ Memory increases linearly with batch size
\`\`\`

**Computational Speed Trade-offs:**

Larger batches are more efficient due to:
1. Better hardware utilization (GPUs)
2. Amortized function call overhead
3. Improved cache hit rates

\`\`\`python
# Measure throughput (samples/second)

Batch Size | Throughput | Efficiency
-----------|------------|------------
1          | 100 samp/s | 1x baseline
8          | 600 samp/s | 6x (75% efficient)
32         | 2000 samp/s| 20x (63% efficient)
128        | 6000 samp/s| 60x (47% efficient)
512        | 15000 samp/s| 150x (29% efficient)

→ Throughput increases sub-linearly
→ Returns diminish beyond batch_size ~128-256
\`\`\`

**Gradient Estimate Quality:**

Batch size affects gradient noise:

Small batches (1-32):
- High gradient noise
- Stochastic gradients (significant variance)
- Exploration of loss landscape
- Regularization effect (generalization benefit)
- May escape sharp minima (better generalization)

Large batches (256-1024+):
- Low gradient noise
- Near-deterministic gradients
- Faster convergence per epoch
- Can converge to sharp minima (worse generalization)
- Need more epochs for same performance

Mathematical relationship:
\`\`\`
Gradient variance ∝ 1 / batch_size

batch_size = 1: Maximum variance (pure SGD)
batch_size = 32: 5.6x less variance
batch_size = 256: 16x less variance
batch_size = N (full batch): Zero variance (GD)
\`\`\`

**Training Dynamics:**

\`\`\`python
# Typical training curves

Small batch (8-32):
- Loss: Noisy, fluctuates
- Convergence: Slower per epoch, but better final solution
- Generalization: Better (implicit regularization)
- Training time: More epochs needed

Large batch (256-1024):
- Loss: Smooth, steady decrease
- Convergence: Faster per epoch, may get stuck
- Generalization: Worse (overfits to training data)
- Training time: Fewer epochs, but may need learning rate tuning
\`\`\`

**For Trading Models: Practical Recommendations**

**Financial Data Characteristics:**
- Non-stationary (distributions change)
- High noise-to-signal ratio
- Limited data relative to market complexity
- Need for robust generalization

**Recommended Approach: Medium Batches (32-128)**

Reasoning:

1. **Batch Size 32-64 for Daily Trading:**
\`\`\`python
# Daily trading model
batch_size = 32  # ~1-2 months of trading days

Advantages:
- Enough samples for stable gradients
- Maintains some stochasticity for regularization
- Fits easily in memory (even on CPU)
- Good balance of speed vs generalization

# With 5000 training samples:
# batches_per_epoch = 5000 / 32 ≈ 156 batches
# Provides many weight updates per epoch
\`\`\`

2. **Batch Size 64-128 for Intraday Trading:**
\`\`\`python
# Intraday model (1-minute bars)
batch_size = 128  # ~2 hours of minute bars

# More data available (390 bars per day)
# Can use larger batches without sacrificing updates
\`\`\`

3. **Avoid Very Small Batches (<16):**
- Too noisy for financial data
- Training unstable
- Slow convergence

4. **Avoid Very Large Batches (>256):**
- Poor generalization (critical in finance)
- May overfit to training period
- Won't generalize to new market regimes

**Adaptive Batch Size Strategy:**

\`\`\`python
# Start small, increase gradually
training_schedule = {
    'epochs_0_50': 32,    # Early: Explore loss landscape
    'epochs_50_100': 64,  # Middle: Stable convergence
    'epochs_100_150': 32, # Late: Fine-tune with noise
}

# Or: Increase learning rate with batch size
batch_size = 64
learning_rate = 0.001 * (batch_size / 32)  # Linear scaling rule
\`\`\`

**Special Considerations for Trading:**

1. **Time-Series Dependency:**
\`\`\`python
# Don't randomly shuffle time series!
# Use sequential batches

# WRONG: Random batches break temporal dependencies
batches = random_shuffle (data)

# CORRECT: Sequential or walk-forward batches
batches = [data[i:i+32] for i in range(0, len (data), 32)]
\`\`\`

2. **Regime Changes:**
- Smaller batches adapt faster to regime changes
- Larger batches average across regimes (bad)
- Prefer batch_size = 32-64 for adaptability

3. **Walk-Forward Validation:**
\`\`\`python
# Retrain frequently with consistent batch size
retrain_frequency = '30D'  # Every 30 days
batch_size = 32  # Keep consistent
\`\`\`

**Empirical Testing:**

\`\`\`python
def experiment_with_batch_sizes():
    """Test different batch sizes"""
    batch_sizes = [8, 16, 32, 64, 128, 256]
    results = []
    
    for bs in batch_sizes:
        model = TrainingNetwork (batch_size=bs)
        train_loss, val_loss, sharpe = model.train (data)
        results.append({
            'batch_size': bs,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'sharpe': sharpe,  # Most important for trading!
        })
    
    # Plot results
    # Usually optimal around 32-64 for financial data
    return results
\`\`\`

**Summary Table:**

| Batch Size | Memory | Speed | Gradient Quality | Generalization | Trading Use Case |
|------------|--------|-------|------------------|----------------|------------------|
| 1-16 | Low | Slow | Very noisy | Best | Research only |
| 32-64 | Medium | Good | Balanced | Good | ✓ Daily trading |
| 128-256 | High | Fast | Smooth | Fair | Intraday (minute) |
| 512+ | Very high | Fastest | Too smooth | Poor | ✗ Avoid |

**Recommended for Trading:**
- **Daily models**: batch_size = 32
- **Intraday models**: batch_size = 64-128
- **Always test** on validation set (Sharpe ratio, not just loss)
- **Prioritize generalization** over training speed

**Final Recommendation:**

Start with batch_size = 32 for trading models:
- Good speed/memory trade-off
- Sufficient stochasticity for regularization
- Robust across market regimes
- Well-tested default in practice

Increase to 64 or 128 only if:
- You have abundant data (>100K samples)
- Training is too slow
- Validation performance improves

Never exceed 256 for financial applications—generalization matters more than training speed.`,
    keyPoints: [
      'Memory usage scales linearly with batch size: O(batch_size × Σ layer_sizes)',
      'Computational throughput increases sub-linearly - diminishing returns above batch_size=128-256',
      'Small batches: Noisy gradients, better generalization, implicit regularization',
      'Large batches: Smooth gradients, faster convergence, worse generalization (sharp minima)',
      'For daily trading: batch_size=32-64 provides best balance',
      'For intraday trading: batch_size=64-128 acceptable due to more data',
      'Financial data is non-stationary - prioritize generalization over training speed',
      'Never shuffle time series - use sequential batches to preserve temporal dependencies',
    ],
  },
  {
    id: 'forward-propagation-dq-3',
    question:
      'Computational graphs are fundamental to modern deep learning frameworks like PyTorch and TensorFlow. Explain how computational graphs enable automatic differentiation, and why tracking the forward pass is essential for backpropagation.',
    sampleAnswer: `Computational graphs are the core abstraction that makes modern deep learning possible. They enable automatic differentiation, which eliminates the need to manually compute gradients for complex neural networks:

**What is a Computational Graph?**

A directed acyclic graph (DAG) where:
- **Nodes**: Variables (inputs, parameters, intermediates, outputs)
- **Edges**: Operations (matrix multiply, add, activation functions)

Simple example:
\`\`\`
y = (x * w) + b

Graph:
  x ──┐
      ├─→ [multiply] ──→ temp ──┐
  w ──┘                          ├─→ [add] ──→ y
  b ─────────────────────────────┘
\`\`\`

Multi-layer network:
\`\`\`
a₁ = σ(W₁x + b₁)
a₂ = σ(W₂a₁ + b₂)

        ┌─→ [matmul] ──→ z₁ ──→ [σ] ──→ a₁ ──┐
  x, W₁─┤                                      │
        └─→ [add (b₁)]                           │
                                                ├─→ [matmul] ──→ z₂ ──→ [σ] ──→ a₂
                                         W₂, b₂─┘
\`\`\`

**How Computational Graphs Enable Autodiff:**

**1. Forward Pass: Build the Graph**

During forward propagation, the framework records:
- Every operation performed
- Input values to each operation
- Output values from each operation
- Operation type (for derivative computation)

\`\`\`python
# PyTorch example
import torch

# Create tensors with gradient tracking
x = torch.tensor([2.0], requires_grad=True)
w = torch.tensor([3.0], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)

# Forward pass builds computational graph
z = x * w      # Node: multiply operation
y = z + b      # Node: add operation

# Graph is built automatically:
# x ──→ [*] ──→ z ──→ [+] ──→ y
# w ──→       b ──→
\`\`\`

Each operation stores:
- Forward function: how to compute output
- Backward function: how to compute gradient
- Parent nodes: inputs to this operation
- Output value: for use in backward pass

**2. Backward Pass: Traverse in Reverse**

Given the graph, compute gradients using chain rule:

\`\`\`python
# Backward pass: Compute gradients
y.backward()  # dy/dy = 1.0 (gradient of output w.r.t. itself)

# Chain rule automatically applied:
# dy/dz = dy/dy * dy/dz = 1.0 * 1.0 = 1.0
# dy/db = dy/dy * dy/db = 1.0 * 1.0 = 1.0
# dy/dw = dy/dz * dz/dw = 1.0 * x = 2.0
# dy/dx = dy/dz * dz/dx = 1.0 * w = 3.0

print(f"dy/dx = {x.grad}")  # 3.0
print(f"dy/dw = {w.grad}")  # 2.0
print(f"dy/db = {b.grad}")  # 1.0
\`\`\`

**3. Chain Rule on Computational Graphs**

For any node n with parent p:
\`\`\`
∂Loss/∂p = ∂Loss/∂n * ∂n/∂p

Where:
- ∂Loss/∂n: Gradient from downstream (already computed)
- ∂n/∂p: Local gradient (from operation's backward function)
\`\`\`

**Why Tracking Forward Pass is Essential:**

**1. Store Intermediate Values:**

Gradient computation often requires forward pass values:

\`\`\`python
# Forward: a = σ(z) where σ(z) = 1 / (1 + e^(-z))
# Backward: ∂L/∂z = ∂L/∂a * ∂a/∂z
#                  = ∂L/∂a * a * (1 - a)
#                          ^^^^^^^^^^^^
#                          Need forward value!

# Without caching 'a', would need to recompute it
\`\`\`

Example:
\`\`\`python
class SigmoidNode:
    def forward (self, z):
        self.a = 1 / (1 + np.exp(-z))  # Cache for backward
        return self.a
    
    def backward (self, grad_output):
        # Need self.a from forward pass!
        grad_input = grad_output * self.a * (1 - self.a)
        return grad_input
\`\`\`

**2. Determine Computation Order:**

Graph structure determines backpropagation order:

\`\`\`
Forward (left to right):
  x → a → b → c → loss

Backward (right to left):
  loss → ∂loss/∂c → ∂loss/∂b → ∂loss/∂a → ∂loss/∂x

Must process in reverse topological order!
\`\`\`

**3. Handle Multiple Paths:**

When a variable is used multiple times, gradients accumulate:

\`\`\`python
x = torch.tensor([2.0], requires_grad=True)
y = x * x    # First use
z = x + y    # Second use
loss = z ** 2

# Computational graph:
#     ┌──→ [*] ──→ y ──┐
#  x ─┤                ├──→ [+] ──→ z ──→ [**2] ──→ loss
#     └────────────────┘

# x appears in two paths!
# ∂loss/∂x = (∂loss/∂x via y) + (∂loss/∂x via direct path)
#          = ... (automatically handled by graph)

loss.backward()
print(x.grad)  # Correctly accumulates gradients from both paths
\`\`\`

**4. Dynamic vs Static Graphs:**

**Static Graphs (TensorFlow 1.x):**
- Build graph once, execute many times
- Optimize graph before execution
- Faster, but less flexible

\`\`\`python
# TensorFlow 1.x style
x = tf.placeholder (tf.float32)
w = tf.Variable([3.0])
y = x * w

sess = tf.Session()
# Graph is fixed, just feed different x values
result = sess.run (y, feed_dict={x: 2.0})
\`\`\`

**Dynamic Graphs (PyTorch, TensorFlow 2.x):**
- Build graph on-the-fly during forward pass
- Different graph for each forward pass
- Flexible (control flow, debugging)

\`\`\`python
# PyTorch style
def forward (x, w):
    if x > 0:  # Control flow!
        return x * w
    else:
        return x * w * w

# Different graphs for different inputs
y1 = forward (torch.tensor([2.0]), torch.tensor([3.0]))  # Graph 1
y2 = forward (torch.tensor([-2.0]), torch.tensor([3.0])) # Graph 2
\`\`\`

**Complete Example: Manual Autodiff**

\`\`\`python
class ComputationNode:
    """Base class for computation graph nodes"""
    def __init__(self):
        self.inputs = []
        self.output = None
        self.grad = None
    
    def forward (self, *inputs):
        raise NotImplementedError
    
    def backward (self, grad_output):
        raise NotImplementedError


class MultiplyNode(ComputationNode):
    def forward (self, x, y):
        self.inputs = [x, y]
        self.output = x * y
        return self.output
    
    def backward (self, grad_output):
        x, y = self.inputs
        grad_x = grad_output * y  # ∂(xy)/∂x = y
        grad_y = grad_output * x  # ∂(xy)/∂y = x
        return grad_x, grad_y


class AddNode(ComputationNode):
    def forward (self, x, y):
        self.inputs = [x, y]
        self.output = x + y
        return self.output
    
    def backward (self, grad_output):
        # ∂(x+y)/∂x = 1, ∂(x+y)/∂y = 1
        return grad_output, grad_output


class SigmoidNode(ComputationNode):
    def forward (self, x):
        self.inputs = [x]
        self.output = 1 / (1 + np.exp(-x))
        return self.output
    
    def backward (self, grad_output):
        # ∂σ/∂x = σ * (1 - σ)
        sig = self.output
        grad_x = grad_output * sig * (1 - sig)
        return grad_x


# Build computation: y = σ(x * w + b)
x_val, w_val, b_val = 2.0, 3.0, 1.0

mul_node = MultiplyNode()
add_node = AddNode()
sig_node = SigmoidNode()

# Forward pass
z1 = mul_node.forward (x_val, w_val)  # x * w
z2 = add_node.forward (z1, b_val)      # + b
y = sig_node.forward (z2)               # σ(.)

print(f"Forward: y = {y:.4f}")

# Backward pass (traverse graph in reverse)
grad_y = 1.0  # dy/dy = 1

grad_z2 = sig_node.backward (grad_y)
grad_z1, grad_b = add_node.backward (grad_z2)
grad_x, grad_w = mul_node.backward (grad_z1)

print(f"Gradients:")
print(f"  dy/dx = {grad_x:.4f}")
print(f"  dy/dw = {grad_w:.4f}")
print(f"  dy/db = {grad_b:.4f}")
\`\`\`

**Benefits of Autodiff via Computational Graphs:**

1. **Correctness**: No manual derivative errors
2. **Flexibility**: Any differentiable operation works
3. **Efficiency**: Only compute needed gradients
4. **Modularity**: Compose complex functions from simple operations
5. **Debugging**: Inspect graph and gradients at any node

**Trading Application:**

\`\`\`python
# Complex trading loss function
def trading_loss (predictions, returns, positions, transaction_costs):
    # Computational graph automatically tracks all operations:
    pnl = predictions * returns * positions
    costs = transaction_costs * torch.abs (positions[1:] - positions[:-1])
    sharpe = pnl.mean() / (pnl.std() + 1e-6)
    loss = -sharpe + costs.sum()
    
    # Backward pass computes:
    # ∂loss/∂predictions automatically!
    # No manual calculus needed!
    return loss

# Framework handles all gradient computation
loss = trading_loss (model (features), returns, positions, costs)
loss.backward()  # All gradients computed automatically
optimizer.step()  # Update model parameters
\`\`\`

**Key Insights:**

1. **Computational graphs represent all operations as nodes**
2. **Forward pass builds graph and caches intermediate values**
3. **Backward pass traverses graph in reverse, applying chain rule**
4. **Each operation provides local gradient (∂output/∂input)**
5. **Gradients accumulate automatically when paths merge**
6. **Modern frameworks (PyTorch/TensorFlow) handle this automatically**

**Conclusion:**

Computational graphs are the "magic" behind modern deep learning. They enable:
- Automatic differentiation (no manual calculus)
- Efficient gradient computation
- Flexible model architectures
- Easy debugging and visualization

Without computational graphs, training deep neural networks would require manually deriving and implementing gradients for every model variant—an intractable task. The graph abstraction makes deep learning practical and accessible.`,
    keyPoints: [
      'Computational graphs represent operations as directed acyclic graphs (DAG)',
      'Forward pass builds graph and caches intermediate values needed for backprop',
      'Backward pass traverses graph in reverse, applying chain rule automatically',
      'Each operation stores forward function and backward function (local gradient)',
      'Caching forward values is essential - many gradients require forward outputs',
      'Gradients automatically accumulate when variables used in multiple paths',
      'Dynamic graphs (PyTorch) rebuild per forward pass, enabling control flow',
      'Autodiff eliminates manual calculus errors and enables rapid experimentation',
    ],
  },
];
