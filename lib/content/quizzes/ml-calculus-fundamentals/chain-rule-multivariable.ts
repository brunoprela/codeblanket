/**
 * Quiz questions for Chain Rule for Multiple Variables section
 */

export const chainrulemultivariableQuiz = [
  {
    id: 'chain-multi-disc-1',
    question:
      'Explain why backpropagation is efficient compared to naively computing all gradients. What is the computational complexity difference?',
    hint: 'Consider computing n gradients independently vs. backpropagation. Think about redundant computations.',
    sampleAnswer: `**Naive Gradient Computation vs. Backpropagation:**

**Naive Approach:**
Compute each gradient ∂L/∂θᵢ independently using the definition or numerical differentiation.

For n parameters:
- Each gradient: O(forward pass cost)
- Total: O(n × forward pass)
- For deep network with millions of parameters: prohibitively expensive

**Backpropagation Approach:**
Single backward pass computes ALL gradients simultaneously.

**Why It\'s Efficient:**1. **Shared Computations:**
   
   Consider network: x → f₁ → f₂ → f₃ → L
   
   Gradients share intermediate terms:
   - ∂L/∂θ₁ = ∂L/∂f₃ · ∂f₃/∂f₂ · ∂f₂/∂f₁ · ∂f₁/∂θ₁
   - ∂L/∂θ₂ = ∂L/∂f₃ · ∂f₃/∂f₂ · ∂f₂/∂θ₂
   
   Both use ∂L/∂f₃ and ∂f₃/∂f₂ - computed once, reused!

2. **Dynamic Programming:**
   
   Backpropagation is essentially dynamic programming:
   - Store intermediate gradients
   - Reuse them for earlier layers
   - No redundant computation

3. **Computational Complexity:**

   **Forward pass:** O(W) where W = number of weights
   
   **Backward pass:** O(W) - same as forward!
   
   **Total for all gradients:** O(W)
   
   vs.
   
   **Naive:** O(W × W) = O(W²)

4. **Concrete Example:**

   Network with L layers, each with n parameters:
   - Total parameters: N = L × n
   
   **Naive:**
   - Each gradient: evaluate network (O(N))
   - N gradients: O(N²)
   - For N = 10⁶: ~10¹² operations
   
   **Backpropagation:**
   - Forward: O(N)
   - Backward: O(N)
   - Total: O(N)
   - For N = 10⁶: ~10⁶ operations
   
   **Speedup:** O(N) → **Million times faster!**5. **Why This Works - Automatic Differentiation:**

   Backpropagation is reverse-mode automatic differentiation:
   
   **Forward mode:** Computes all derivatives of one output wrt all inputs
   - Efficient when: few inputs, many outputs
   - Complexity: O(inputs)
   
   **Reverse mode (backprop):** Computes all derivatives of one output wrt all inputs
   - Efficient when: many inputs, few outputs
   - Complexity: O(outputs)
   
   Neural networks: many parameters (inputs), one loss (output) → reverse mode optimal!

**Practical Impact:**

Without backpropagation's efficiency, training large neural networks would be impossible:
- GPT-3: 175 billion parameters
- Naive: 175B² ≈ 3×10²² operations per gradient step (centuries)
- Backprop: 175B operations (seconds)

The O(n) vs O(n²) difference is what enables deep learning at scale.`,
    keyPoints: [
      'Backprop computes all gradients in O(n) time vs naive O(n²)',
      'Shares intermediate computations via dynamic programming',
      'Reverse-mode autodiff optimal for many params, one loss',
      'Makes training billion-parameter models feasible',
      'Foundation of all modern deep learning frameworks',
    ],
  },
  {
    id: 'chain-multi-disc-2',
    question:
      'Describe how modern deep learning frameworks use computational graphs and automatic differentiation. How does this relate to the chain rule?',
    hint: "Consider PyTorch/TensorFlow\'s autograd, define-by-run vs static graphs, and gradient tape.",
    sampleAnswer: `**Computational Graphs and Automatic Differentiation:**

Modern frameworks (PyTorch, TensorFlow, JAX) automatically apply the chain rule through computational graphs.

**1. Computational Graph Structure:**

A computational graph represents function composition:
- **Nodes**: Operations or variables
- **Edges**: Data dependencies
- **Forward pass**: Evaluate nodes in topological order
- **Backward pass**: Propagate gradients in reverse order

Example: L = (W·x + b)²
\`\`\`
x, W, b → z1 = W·x → z2 = z1 + b → L = z2²
\`\`\`

**2. Two Paradigms:**

**Static Graphs (TensorFlow 1.x):**
\`\`\`python
# Define graph first
x = tf.placeholder()
W = tf.Variable()
z1 = tf.matmul(W, x)
loss = tf.reduce_sum (z1**2)

# Then execute
with tf.Session() as sess:
    result = sess.run (loss, feed_dict={x: data})
\`\`\`

Pros: Can optimize graph before execution
Cons: Less flexible, harder to debug

**Dynamic Graphs (PyTorch, TensorFlow 2.x):**
\`\`\`python
# Define and execute simultaneously
z1 = W @ x
loss = (z1**2).sum()
loss.backward()  # Automatic differentiation
\`\`\`

Pros: Pythonic, easy debugging, dynamic control flow
Cons: Less optimization opportunity

**3. How Automatic Differentiation Works:**

**Forward Pass (build graph):**
\`\`\`python
import torch

x = torch.tensor([1.0, 2.0], requires_grad=True)
W = torch.tensor([[1.0, 0.5], [0.5, 1.0]], requires_grad=True)

# Each operation creates a node
z1 = W @ x  # MatMul node
z2 = z1.sum()  # Sum node
loss = z2**2  # Pow node

# Graph: x, W → MatMul → Sum → Pow → loss
\`\`\`

**Backward Pass (apply chain rule):**
\`\`\`python
loss.backward()

# Internally, framework:
# 1. ∂loss/∂loss = 1
# 2. ∂loss/∂z2 = ∂loss/∂loss · ∂loss/∂z2 = 1 · 2z2
# 3. ∂loss/∂z1 = ∂loss/∂z2 · ∂z2/∂z1 = (2z2) · 1
# 4. ∂loss/∂W = ∂loss/∂z1 · ∂z1/∂W = (2z2) · x^T
# 5. ∂loss/∂x = ∂loss/∂z1 · ∂z1/∂x = (2z2) · W^T

print(x.grad)  # ∂loss/∂x
print(W.grad)  # ∂loss/∂W
\`\`\`

**4. Under the Hood:**

Each operation stores its local gradient function:

\`\`\`python
class MulOp:
    def forward (self, a, b):
        self.a = a
        self.b = b
        return a * b
    
    def backward (self, grad_output):
        # Chain rule: ∂L/∂a = ∂L/∂out · ∂out/∂a = grad_output · b
        grad_a = grad_output * self.b
        grad_b = grad_output * self.a
        return grad_a, grad_b
\`\`\`

**5. Gradient Tape (TensorFlow 2.x approach):**

\`\`\`python
import tensorflow as tf

with tf.GradientTape() as tape:
    x = tf.Variable([1.0, 2.0])
    y = x**2
    loss = tf.reduce_sum (y)

# Tape records all operations
gradients = tape.gradient (loss, x)
# Applies chain rule backward through recorded operations
\`\`\`

**6. Advanced Features:**

**Higher-Order Derivatives:**
\`\`\`python
x = torch.tensor(2.0, requires_grad=True)
y = x**3
dy_dx = torch.autograd.grad (y, x, create_graph=True)[0]
d2y_dx2 = torch.autograd.grad (dy_dx, x)[0]
print(f"d²y/dx² = {d2y_dx2}")  # Second derivative
\`\`\`

**Jacobian/Hessian:**
\`\`\`python
from torch.autograd.functional import jacobian, hessian

def f (x):
    return torch.sum (x**2)

x = torch.tensor([1.0, 2.0, 3.0])
J = jacobian (f, x)  # ∂f/∂x for vector f
H = hessian (f, x)   # ∂²f/∂x∂x
\`\`\`

**Gradient Accumulation:**
\`\`\`python
optimizer.zero_grad()
for batch in mini_batches:
    loss = model (batch)
    loss.backward()  # Accumulates gradients
optimizer.step()
\`\`\`

**7. Connection to Chain Rule:**

Every operation in the graph knows:
1. How to compute forward (output from inputs)
2. How to compute backward (gradient wrt inputs from gradient wrt output)

The framework:
1. Builds graph during forward pass
2. Traverses graph backward
3. At each node, applies local chain rule
4. Accumulates gradients through all paths

**Example - Why It's Powerful:**

\`\`\`python
# Complex computation graph
x = torch.randn(100, 10)
W1 = torch.randn(10, 50, requires_grad=True)
W2 = torch.randn(50, 20, requires_grad=True)

# Computation
h = torch.relu (x @ W1)
h = torch.dropout (h, 0.5, training=True)
y = torch.softmax (h @ W2, dim=1)
loss = -torch.log (y[range(100), labels]).mean()

# One line gets all gradients!
loss.backward()

# Framework applied chain rule through:
# log → softmax → matmul → dropout → relu → matmul
# All automatically, all correctly
\`\`\`

**Modern Innovations:**

- **JIT Compilation**: Optimize computational graph
- **Quantization**: Mixed precision training
- **Graph Optimization**: Fuse operations
- **Distributed Autograd**: Across multiple GPUs
- **Checkpointing**: Trade computation for memory

The beauty: as a user, you just write forward pass. The framework handles all chain rule complexity!`,
    keyPoints: [
      'Computational graphs represent function compositions',
      'Automatic differentiation applies chain rule automatically',
      'Dynamic graphs (PyTorch) build during execution',
      'Each operation stores local gradient function',
      'Backward pass traverses graph applying chain rule',
      'Enables complex models with one-line gradient computation',
    ],
  },
  {
    id: 'chain-multi-disc-3',
    question:
      'Explain the difference between forward-mode and reverse-mode automatic differentiation. Why is reverse-mode (backpropagation) preferred for neural networks?',
    hint: 'Consider a function f: ℝⁿ → ℝᵐ and the cost of computing all derivatives for different n and m.',
    sampleAnswer: `**Forward-Mode vs Reverse-Mode Automatic Differentiation:**

Both compute exact derivatives using the chain rule, but in different orders.

**1. Forward-Mode AD:**

**Strategy:** Propagate derivatives forward along with values.

**How it works:**
For y = f (x₁, ..., xₙ), compute ∂y/∂xᵢ by:
1. Set dx_i/dx_i = 1, dx_j/dx_i = 0 for j≠i
2. Propagate derivatives forward through operations
3. Result: dy/dx_i

**Example:** y = x₁·x₂ + sin (x₁)

Compute ∂y/∂x₁:
\`\`\`
Forward pass with derivatives:

x₁: value=2, derivative=1 (∂x₁/∂x₁=1)
x₂: value=3, derivative=0 (∂x₂/∂x₁=0)

v₁ = x₁·x₂: 
  value = 2·3 = 6
  derivative = 1·3 + 2·0 = 3  (product rule)

v₂ = sin (x₁):
  value = sin(2) ≈ 0.909
  derivative = cos(2)·1 ≈ -0.416  (chain rule)

y = v₁ + v₂:
  value = 6.909
  derivative = 3 + (-0.416) = 2.584 (∂y/∂x₁)
\`\`\`

**Complexity:** 
- One forward pass per input variable
- For n inputs: O(n × forward_cost)
- Efficient when n is small

**2. Reverse-Mode AD (Backpropagation):**

**Strategy:** Compute all derivatives in one backward pass.

**How it works:**
For L = f (x₁, ..., xₙ), compute all ∂L/∂xᵢ by:
1. Forward pass: Compute values, save intermediates
2. Set ∂L/∂L = 1
3. Propagate gradients backward
4. Result: All ∂L/∂xᵢ simultaneously

**Same example:** L = x₁·x₂ + sin (x₁)

\`\`\`
Forward pass (save values):
  x₁=2, x₂=3
  v₁ = 6
  v₂ = 0.909
  L = 6.909

Backward pass:
  ∂L/∂L = 1

  ∂L/∂v₁ = ∂L/∂L · ∂L/∂v₁ = 1 · 1 = 1
  ∂L/∂v₂ = ∂L/∂L · ∂L/∂v₂ = 1 · 1 = 1

  ∂L/∂x₁ = ∂L/∂v₁·∂v₁/∂x₁ + ∂L/∂v₂·∂v₂/∂x₁
          = 1·3 + 1·cos(2) = 2.584

  ∂L/∂x₂ = ∂L/∂v₁·∂v₁/∂x₂ = 1·2 = 2
\`\`\`

**Complexity:**
- One forward + one backward pass
- For n inputs: O(forward_cost + backward_cost) ≈ O(2×forward_cost)
- Efficient when n is large, one output

**3. Comparison Table:**

| Aspect | Forward-Mode | Reverse-Mode |
|--------|--------------|--------------|
| **Passes** | n forward passes | 1 forward + 1 backward |
| **Complexity** | O(n × C) | O(C) |
| **Best for** | f: ℝⁿ→ℝᵐ, n≪m | f: ℝⁿ→ℝᵐ, m≪n |
| **Memory** | Low | High (store intermediates) |
| **Example use** | Jacobian-vector products | Gradient of scalar loss |

where C = cost of evaluating f

**4. Why Reverse-Mode for Neural Networks?**

**Scenario:**
- Inputs: n = millions of parameters
- Output: m = 1 (scalar loss)

**Forward-Mode:**
- Need n forward passes
- One per parameter
- Compute ∂L/∂θ₁, ∂L/∂θ₂, ..., ∂L/∂θₙ separately
- Cost: O(n × forward_cost)
- For n = 10⁶: ~million forward passes!

**Reverse-Mode:**
- One forward + one backward pass
- Computes all ∂L/∂θᵢ simultaneously
- Cost: O(2 × forward_cost)
- Independent of n!

**Speedup:** n/2 = 500,000× faster for n=10⁶

**5. Practical Example:**

\`\`\`python
import torch

# Network with 1M parameters
n_params = 1_000_000
params = torch.randn (n_params, requires_grad=True)

def loss_fn (p):
    # Some complex computation
    return (p**2).sum()

# Reverse-mode (backprop): O(1)
loss = loss_fn (params)
loss.backward()  # All gradients in one backward pass
print(params.grad.shape)  # (1000000,) - all gradients!

# Forward-mode would need:
# for i in range (n_params):
#     compute ∂loss/∂params[i]  # 1M forward passes!
\`\`\`

**6. When to Use Each:**

**Use Forward-Mode when:**
- Few inputs, many outputs (computing Jacobian-vector products)
- Example: Sensitivity analysis (how outputs change with one input)
- Example: Optimal control (Jacobian of system dynamics)

**Use Reverse-Mode when:**
- Many inputs, few outputs (neural network training)
- Computing gradients of scalar objective
- Example: Any optimization problem with gradient descent

**7. Hybrid Approaches:**

Modern AD systems support both:

\`\`\`python
# JAX supports both modes
import jax

def f (x):
    return jax.numpy.sum (x**2)

x = jax.numpy.array([1.0, 2.0, 3.0])

# Reverse-mode (backprop)
grad_reverse = jax.grad (f)(x)

# Forward-mode
def forward_mode_grad (x):
    # Compute gradient via forward-mode
    # (JAX can do this with jax.jvp)
    pass
\`\`\`

**8. Memory Trade-offs:**

**Forward-Mode:**
- Low memory (only current values + derivatives)
- No need to store intermediate values

**Reverse-Mode:**
- High memory (must store all intermediate values)
- Proportional to network depth
- Solutions:
  - Gradient checkpointing (recompute instead of store)
  - Micro-batching (process smaller batches)

**Summary:**

Reverse-mode AD (backpropagation) dominates machine learning because:
1. Neural networks: many parameters (n→∞), one loss (m=1)
2. O(1) vs O(n) complexity
3. Makes billion-parameter models tractable
4. All modern frameworks implement reverse-mode

Forward-mode still useful for specific applications (sensitivity analysis, optimal control), but reverse-mode is the workhorse of deep learning.`,
    keyPoints: [
      'Forward-mode: O(n) passes, efficient for few inputs',
      'Reverse-mode: O(1) passes, efficient for many inputs',
      'Neural networks: n parameters, 1 loss → reverse-mode optimal',
      'Backprop is reverse-mode AD applied to neural networks',
      'Memory trade-off: reverse-mode stores intermediates',
      'Speedup for n=10⁶ parameters: ~500,000× faster',
    ],
  },
];
