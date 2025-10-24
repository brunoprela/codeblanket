import { QuizQuestion } from '../../../types';

export const backpropagationAlgorithmQuiz: QuizQuestion[] = [
  {
    id: 'backpropagation-algorithm-dq-1',
    question:
      'Walk through the mathematical derivation of backpropagation for a single layer z = Wx + b, a = σ(z), showing how the chain rule produces ∂L/∂W. Explain why we need to cache forward pass values.',
    sampleAnswer: `Backpropagation for a single layer demonstrates the power of the chain rule and why caching forward values is essential. Let's derive it step by step:

**Setup:**
- Input: x (shape: batch_size × n_in)
- Weights: W (shape: n_in × n_out)
- Bias: b (shape: 1 × n_out)
- Pre-activation: z = Wx + b
- Activation: a = σ(z)
- Loss: L (scalar)

**Goal:** Find ∂L/∂W, ∂L/∂b, and ∂L/∂x

**Step 1: Gradient w.r.t. Pre-activation (∂L/∂z)**

Assume we have ∂L/∂a from the next layer (or directly from loss if this is output layer).

Apply chain rule:
\\\`\\\`\\\`
∂L/∂z = ∂L/∂a · ∂a/∂z
\\\`\\\`\\\`

For element-wise operations:
\\\`\\\`\\\`
∂a/∂z = σ'(z)
\\\`\\\`\\\`

Therefore:
\\\`\\\`\\\`
∂L/∂z = ∂L/∂a ⊙ σ'(z)
\\\`\\\`\\\`

Where ⊙ denotes element-wise multiplication (Hadamard product).

**Why we need cached z:**
- σ'(z) depends on z from forward pass
- For sigmoid: σ'(z) = σ(z)(1 - σ(z)) needs σ(z) = a
- For ReLU: σ'(z) = 1 if z > 0 else 0 needs z directly
- Without caching, would need to recompute forward pass!

**Step 2: Gradient w.r.t. Weights (∂L/∂W)**

Now find ∂L/∂W using chain rule:
\\\`\\\`\\\`
∂L/∂W = ∂L/∂z · ∂z/∂W
\\\`\\\`\\\`

First, find ∂z/∂W. Since z = Wx + b:
\\\`\\\`\\\`
For z_ij = Σ_k W_kj x_k + b_j

∂z_ij/∂W_mn = x_m if j = n, else 0
\\\`\\\`\\\`

This means:
\\\`\\\`\\\`
∂z/∂W has shape (batch, n_out, n_in, n_out)
\\\`\\\`\\\`

But we don't compute this 4D tensor! Instead, use matrix calculus:

For batch of m samples:
\\\`\\\`\\\`
Z = XW + b  (shape: m × n_out)

where X is (m × n_in), W is (n_in × n_out)
\\\`\\\`\\\`

The gradient aggregates across batch:
\\\`\\\`\\\`
∂L/∂W = (1/m) X^T · (∂L/∂Z)
\\\`\\\`\\\`

**Derivation of matrix form:**

For a single sample:
\\\`\\\`\\\`
∂L/∂W_ij = ∂L/∂z_j · ∂z_j/∂W_ij
         = ∂L/∂z_j · x_i
\\\`\\\`\\\`

For batch (m samples):
\\\`\\\`\\\`
∂L/∂W_ij = (1/m) Σ_{sample k} ∂L/∂z_kj · x_ki
         = (1/m) Σ_k x_ki · ∂L/∂z_kj
         = (1/m) [X^T · ∂L/∂Z]_ij
\\\`\\\`\\\`

Therefore:
\\\`\\\`\\\`
∂L/∂W = (1/m) X^T @ (∂L/∂Z)
\\\`\\\`\\\`

Shape check:
- X^T: (n_in × m)
- ∂L/∂Z: (m × n_out)
- Product: (n_in × n_out) ✓ matches W shape!

**Why we need cached x:**
- Computing ∂L/∂W requires x from forward pass
- Without caching x, would need to store or recompute
- Memory-speed tradeoff

**Step 3: Gradient w.r.t. Bias (∂L/∂b)**

Since z = Wx + b and b is broadcast across batch:
\\\`\\\`\\\`
∂z_ij/∂b_j = 1
\\\`\\\`\\\`

Therefore:
\\\`\\\`\\\`
∂L/∂b_j = (1/m) Σ_i ∂L/∂z_ij
\\\`\\\`\\\`

In matrix form:
\\\`\\\`\\\`
∂L/∂b = (1/m) Σ_{samples} ∂L/∂Z
       = (1/m) · sum(∂L/∂Z, axis=0, keepdims=True)
\\\`\\\`\\\`

Shape check:
- ∂L/∂Z: (m × n_out)
- Sum over samples: (1 × n_out) ✓ matches b shape!

**Step 4: Gradient w.r.t. Input (∂L/∂x)**

To propagate gradients to previous layer:
\\\`\\\`\\\`
∂L/∂x = ∂L/∂z · ∂z/∂x
\\\`\\\`\\\`

Since z = Wx + b:
\\\`\\\`\\\`
∂z_j/∂x_i = W_ij
\\\`\\\`\\\`

In matrix form:
\\\`\\\`\\\`
∂L/∂X = (∂L/∂Z) @ W^T
\\\`\\\`\\\`

Shape check:
- ∂L/∂Z: (m × n_out)
- W^T: (n_out × n_in)
- Product: (m × n_in) ✓ matches X shape!

**Complete Backprop Algorithm for One Layer:**

\\\`\\\`\\\`python
# Forward pass (cache these!)
z = X @ W + b
a = activation(z)

# Backward pass
# 1. Gradient w.r.t. pre-activation
dL_dz = dL_da * activation_derivative(z)  # Needs cached z!

# 2. Gradient w.r.t. weights
dL_dW = (X.T @ dL_dz) / m  # Needs cached X!

# 3. Gradient w.r.t. bias
dL_db = np.sum(dL_dz, axis=0, keepdims=True) / m

# 4. Gradient to propagate backward
dL_dX = dL_dz @ W.T
\\\`\\\`\\\`

**Why Caching is Essential:**

1. **Activation Derivatives Need Forward Values:**
   - Sigmoid: σ'(z) = σ(z)(1 - σ(z)) needs a = σ(z)
   - Tanh: tanh'(z) = 1 - tanh²(z) needs a = tanh(z)
   - ReLU: requires z to check if z > 0

2. **Weight Gradients Need Input:**
   - ∂L/∂W = X^T @ (∂L/∂Z) needs X
   - Can't recompute X if it came from previous layer

3. **Efficiency:**
   - Caching uses O(batch_size × layer_size) memory
   - Alternative: Recompute forward (expensive)
   - Trade-off: Memory for speed

4. **Numerical Stability:**
   - Recomputation might give different values due to floating point errors
   - Cached values ensure consistency

**Memory Management:**

For very deep networks, caching all activations can exhaust memory. Solutions:

1. **Gradient Checkpointing:**
   - Cache only some layers (checkpoints)
   - Recompute between checkpoints during backward
   - Saves memory at cost of compute

2. **Micro-batching:**
   - Process smaller batches sequentially
   - Reduces memory per batch
   - Slightly slower but fits in memory

**Common Mistakes:**

1. **Forgetting to cache:**
\\\`\\\`\\\`python
# Wrong
def forward(X):
    z = X @ W + b
    return activation(z)  # Lost z!

# Correct
def forward(X):
    z = X @ W + b
    a = activation(z)
    return a, (X, z)  # Cache for backward
\\\`\\\`\\\`

2. **Modifying cached values:**
\\\`\\\`\\\`python
# Wrong
a = activation(z)
a += 0.1  # Modifies cached value!

# Correct
a = activation(z)
a_modified = a + 0.1  # New array
\\\`\\\`\\\`

3. **Wrong shapes in matrix multiply:**
\\\`\\\`\\\`python
# Wrong
dL_dW = dL_dz @ X  # Wrong order!

# Correct  
dL_dW = X.T @ dL_dz  # Transpose X
\\\`\\\`\\\`

**Conclusion:**

Backpropagation elegantly applies chain rule:
- ∂L/∂z = ∂L/∂a ⊙ σ'(z) - needs cached z or a
- ∂L/∂W = X^T @ (∂L/∂z) / m - needs cached X
- ∂L/∂b = sum(∂L/∂z, axis=0) / m
- ∂L/∂x = (∂L/∂z) @ W^T - gradient propagates backward

Caching forward values is essential for:
- Computing activation derivatives
- Computing weight gradients
- Efficiency (avoid recomputation)
- Numerical stability

Memory-compute trade-off managed through checkpointing for very deep networks.`,
    keyPoints: [
      'Chain rule: ∂L/∂W = ∂L/∂z · ∂z/∂W where ∂L/∂z = ∂L/∂a · ∂a/∂z',
      'Matrix form: ∂L/∂W = (X^T @ ∂L/∂Z) / m where m is batch size',
      "Cached z needed for activation derivatives: σ'(z) depends on forward z or a",
      'Cached X needed for weight gradients: ∂L/∂W requires input X',
      'Gradient propagates backward: ∂L/∂X = (∂L/∂Z) @ W^T',
      'Bias gradient: ∂L/∂b = sum(∂L/∂Z, axis=0) / m',
      'Caching trades memory for speed - essential for efficient training',
      'Gradient checkpointing saves memory by recomputing between checkpoints',
    ],
  },
  {
    id: 'backpropagation-algorithm-dq-2',
    question:
      'Compare the computational complexity of numerical gradient computation versus backpropagation for a network with 1 million parameters. Why is backpropagation O(1) in terms of forward passes while numerical gradients are O(n)?',
    sampleAnswer: `The computational advantage of backpropagation over numerical gradients is one of the most important insights in deep learning. Let's analyze both approaches:

**Numerical Gradients (Finite Differences):**

Definition:
\\\`\\\`\\\`
∂L/∂θ_i ≈ [L(θ + ε·e_i) - L(θ - ε·e_i)] / (2ε)
\\\`\\\`\\\`

Where e_i is the i-th unit vector (1 in position i, 0 elsewhere).

**Complexity Analysis:**

For n parameters:
1. For each parameter θ_i:
   - Compute L(θ + ε·e_i): 1 forward pass
   - Compute L(θ - ε·e_i): 1 forward pass
   - Total: 2 forward passes per parameter

2. Total forward passes: 2n

For 1M parameters:
- Forward passes needed: 2,000,000
- If 1 forward pass = 10ms → Total = 20,000 seconds ≈ 5.5 hours!

**Implementation:**

\\\`\\\`\\\`python
def numerical_gradients(network, X, y, epsilon=1e-5):
    """
    Compute gradients numerically
    Extremely slow but useful for verification
    """
    params = get_all_parameters(network)  # Flatten to vector
    n = len(params)
    gradients = np.zeros(n)
    
    for i in range(n):  # O(n) loop
        # Perturb parameter i
        params[i] += epsilon
        set_parameters(network, params)
        loss_plus = compute_loss(network, X, y)  # Forward pass
        
        params[i] -= 2 * epsilon
        set_parameters(network, params)
        loss_minus = compute_loss(network, X, y)  # Forward pass
        
        # Finite difference
        gradients[i] = (loss_plus - loss_minus) / (2 * epsilon)
        
        # Restore
        params[i] += epsilon
    
    return gradients

# Complexity: O(n) forward passes where n = number of parameters
\\\`\\\`\\\`

**Backpropagation:**

Process:
1. **Forward pass**: Compute predictions and cache activations
2. **Backward pass**: Compute all gradients in one sweep

**Complexity Analysis:**

Let's denote:
- L = number of layers
- n_l = number of neurons in layer l
- Total parameters: n = Σ_l (n_l × n_{l+1})

**Forward Pass:**
- Layer l: Compute z_l = W_l a_{l-1} + b_l
  - Matrix multiply: O(n_{l-1} × n_l) operations
  - Activation: O(n_l) operations
- Total: O(Σ_l n_{l-1} × n_l) = O(n) where n is total connections

**Backward Pass:**
- Layer l: Compute ∂L/∂W_l, ∂L/∂b_l, ∂L/∂a_{l-1}
  - ∂L/∂W_l = a_{l-1}^T @ (∂L/∂z_l): O(n_{l-1} × n_l)
  - ∂L/∂a_{l-1} = (∂L/∂z_l) @ W_l^T: O(n_{l-1} × n_l)
- Total: O(Σ_l n_{l-1} × n_l) = O(n)

**Total Cost:**
- Forward: O(n) operations
- Backward: O(n) operations (approximately 2-3x forward cost)
- Total: O(n) operations ≈ 3 forward passes

For 1M parameters:
- Equivalent forward passes: ~3
- If 1 forward pass = 10ms → Total = 30ms
- **Speedup: 666,666x faster than numerical gradients!**

**Why Backpropagation is O(1) Forward Passes:**

Key insight: **All gradients computed simultaneously in one backward sweep**

1. **Chain Rule Efficiency:**
   - Gradient of parameter in layer l depends on:
     - Gradient from layer l+1 (already computed)
     - Local gradient (cheap to compute)
   - No need to recompute earlier layers

2. **Shared Computation:**
\\\`\\\`\\\`
Numerical gradient for W_l^{ij}:
  - Perturb W_l^{ij}
  - Forward through entire network → loss
  - Repeat for W_l^{ij} - ε
  - Each parameter tested independently

Backpropagation:
  - One forward pass caches all activations
  - One backward pass computes ∂L/∂W_l for ALL i,j simultaneously
  - Matrix operations compute many gradients at once
\\\`\\\`\\\`

3. **Mathematical Elegance:**
\\\`\\\`\\\`
∂L/∂W_l = (∂L/∂z_l) · (∂z_l/∂W_l)
            ^^^^^^^^^   ^^^^^^^^^^^
            from next   a_{l-1}^T
            layer       (cached)

Single matrix multiply computes ALL weight gradients for layer l!
\\\`\\\`\\\`

**Concrete Example:**

Network: 784 → 256 → 128 → 10 (MNIST classifier)
Parameters: (784×256) + 256 + (256×128) + 128 + (128×10) + 10 = 235,146

**Numerical Gradients:**
\\\`\\\`\\\`
For each of 235,146 parameters:
  - 2 forward passes
Total: 470,292 forward passes

At 1ms per forward pass: 470 seconds ≈ 8 minutes
\\\`\\\`\\\`

**Backpropagation:**
\\\`\\\`\\\`
1 forward pass + 1 backward pass ≈ 3 forward passes equivalent

At 1ms per forward pass: 3ms

Speedup: 470,292 / 3 ≈ 156,764x faster!
\\\`\\\`\\\`

**Why O(1) in Terms of Forward Passes:**

The key is that complexity is measured in **forward pass equivalents**:

- Numerical: O(n) forward passes (n = number of parameters)
- Backprop: O(1) forward passes (constant: ~3 regardless of n)

Even if network has 1M or 100M parameters, backprop still takes ~3 forward pass equivalents!

**Memory vs Computation Trade-off:**

Backpropagation achieves O(1) forward passes by:
1. **Storing intermediate activations** (O(batch_size × Σ layer_sizes) memory)
2. **Using stored values in backward pass** (no recomputation)

Alternative (checkpointing):
- Store only some activations
- Recompute others during backward
- Reduces memory, increases compute (but still << numerical gradients)

**Practical Implications:**

**Training Deep Networks:**
\\\`\\\`\\\`
Without backprop: infeasible
- 1M parameters, 1000 training steps
- 2B forward passes
- Days/weeks of training

With backprop: practical
- 3000 forward pass equivalents
- Minutes/hours of training
\\\`\\\`\\\`

**Gradient Checking:**
- Use numerical gradients only for debugging (small networks, few parameters)
- Never for actual training

**Automatic Differentiation:**
- Modern frameworks (PyTorch, TensorFlow) implement backprop automatically
- Users define forward pass, framework computes backward automatically
- Enables rapid experimentation

**Financial ML Context:**

For trading models:
\\\`\\\`\\\`
Typical network: 50 features → 128 → 64 → 32 → 1
Parameters: ~13,000

Numerical gradients:
  - 26,000 forward passes per update
  - 100 updates/epoch, 1000 epochs = 2.6B forward passes
  - Infeasible!

Backpropagation:
  - 3 forward equivalents per update
  - 300,000 forward equivalents total
  - Completes in reasonable time
\\\`\\\`\\\`

**Summary Table:**

| Method | Forward Passes | For 1M params | Time (1ms/pass) |
|--------|---------------|---------------|-----------------|
| Numerical | 2n | 2,000,000 | 33 minutes |
| Backprop | ~3 | 3 | 3 ms |
| Speedup | - | - | 666,666x |

**Conclusion:**

Backpropagation is O(1) forward passes because:
1. One forward pass caches all needed values
2. One backward pass computes ALL gradients simultaneously
3. Matrix operations compute thousands of gradients at once
4. Chain rule shares computation across parameters

This efficiency makes training deep networks practical. Without backpropagation, modern deep learning (and deep trading models) would be impossible.

The O(1) vs O(n) difference is why backpropagation is called "the algorithm that made deep learning possible." It's not just an optimization—it's the fundamental enabler of gradient-based deep learning.`,
    keyPoints: [
      'Numerical gradients: O(n) forward passes for n parameters (2n total)',
      'Backpropagation: O(1) forward passes (~3 equivalent, regardless of n)',
      'For 1M parameters: 2M forward passes vs 3 - speedup of 666,000x',
      'Backprop computes all gradients simultaneously using chain rule',
      'Matrix operations (X^T @ dL_dZ) compute thousands of gradients at once',
      'Cached forward values enable efficient backward computation',
      'Backward pass ≈ 2-3x cost of forward pass (still O(1) forward equivalents)',
      'Without backprop, training deep networks would be computationally infeasible',
    ],
  },
  {
    id: 'backpropagation-algorithm-dq-3',
    question:
      "Gradient checking is used to verify backpropagation implementations. Explain how it works, why it's reliable, when you should use it, and what relative error threshold indicates a correct implementation.",
    sampleAnswer: `Gradient checking is the gold standard for verifying backpropagation implementations. It's reliable because it uses the mathematical definition of derivatives, but it's too expensive for regular use.

**Mathematical Foundation:**

The derivative is defined as:
\\\`\\\`\\\`
f'(x) = lim_{ε→0} [f(x + ε) - f(x - ε)] / (2ε)
\\\`\\\`\\\`

For small ε (e.g., 10^-5), this finite difference approximates the derivative very accurately.

**How Gradient Checking Works:**

**1. Compute Analytical Gradients (Backprop):**
\\\`\\\`\\\`python
def compute_analytical_gradients(network, X, y):
    """Use backpropagation to compute gradients"""
    network.forward(X)
    gradients_analytical = network.backward(y)
    return gradients_analytical
\\\`\\\`\\\`

**2. Compute Numerical Gradients (Finite Differences):**
\\\`\\\`\\\`python
def compute_numerical_gradient(network, X, y, param_name, epsilon=1e-5):
    """
    Compute gradient numerically using finite differences
    
    For parameter θ:
    ∂L/∂θ ≈ [L(θ + ε) - L(θ - ε)] / (2ε)
    """
    param = getattr(network, param_name)
    numerical_grad = np.zeros_like(param)
    
    # Flatten parameter for iteration
    param_flat = param.ravel()
    numerical_grad_flat = numerical_grad.ravel()
    
    for i in range(len(param_flat)):
        # Save original value
        original_value = param_flat[i]
        
        # Evaluate f(θ + ε)
        param_flat[i] = original_value + epsilon
        network.forward(X)
        loss_plus = network.compute_loss(y, network.output)
        
        # Evaluate f(θ - ε)
        param_flat[i] = original_value - epsilon
        network.forward(X)
        loss_minus = network.compute_loss(y, network.output)
        
        # Compute gradient
        numerical_grad_flat[i] = (loss_plus - loss_minus) / (2 * epsilon)
        
        # Restore original value
        param_flat[i] = original_value
    
    return numerical_grad.reshape(param.shape)
\\\`\\\`\\\`

**3. Compare Gradients:**
\\\`\\\`\\\`python
def gradient_check(network, X, y, epsilon=1e-5, threshold=1e-5):
    """
    Verify backprop by comparing with numerical gradients
    
    Returns:
        passed: bool - Whether all gradients match within threshold
        results: dict - Detailed results per parameter
    """
    # Compute analytical gradients
    network.forward(X)
    analytical_grads = network.backward(y)
    
    results = {}
    all_passed = True
    
    print("Gradient Checking:")
    print("=" * 70)
    
    for param_name in analytical_grads:
        # Remove 'd' prefix (dW1 → W1)
        param_name_raw = param_name[1:] if param_name.startswith('d') else param_name
        
        # Get analytical gradient
        grad_analytical = analytical_grads[param_name]
        
        # Compute numerical gradient
        grad_numerical = compute_numerical_gradient(
            network, X, y, param_name_raw, epsilon
        )
        
        # Compute relative difference
        numerator = np.linalg.norm(grad_analytical - grad_numerical)
        denominator = np.linalg.norm(grad_analytical) + np.linalg.norm(grad_numerical)
        relative_diff = numerator / (denominator + 1e-12)
        
        # Check if passes
        passed = relative_diff < threshold
        all_passed = all_passed and passed
        
        # Store results
        results[param_name] = {
            'relative_diff': relative_diff,
            'passed': passed,
            'analytical_norm': np.linalg.norm(grad_analytical),
            'numerical_norm': np.linalg.norm(grad_numerical),
        }
        
        # Print results
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{param_name:12s}: relative_diff = {relative_diff:.2e} {status}")
        
        # If failed, show some values
        if not passed and param_name == 'dW1':  # Show details for first failure
            print(f"  Sample analytical values: {grad_analytical.flat[:5]}")
            print(f"  Sample numerical values:  {grad_numerical.flat[:5]}")
            print(f"  Difference: {(grad_analytical - grad_numerical).flat[:5]}")
    
    print("=" * 70)
    print(f"Overall: {'✓ ALL PASS' if all_passed else '✗ SOME FAILED'}")
    
    return all_passed, results
\\\`\\\`\\\`

**Why Gradient Checking is Reliable:**

**1. Mathematical Guarantee:**
- Finite differences directly implement derivative definition
- As ε → 0, approximation becomes exact
- For ε = 10^-5, error is O(ε²) ≈ 10^-10 (very accurate)

**2. Independent of Implementation:**
- Doesn't rely on chain rule or backprop logic
- Only uses forward pass and loss function
- Catches ALL types of errors

**3. Comprehensive:**
- Checks every single parameter gradient
- Detects subtle bugs (wrong signs, missing terms, shape errors)

**Choosing ε (Epsilon):**

Trade-off between accuracy and numerical stability:

\\\`\\\`\\\`
ε too large (e.g., 0.01):
  - Poor approximation of derivative
  - Relative error > 0.01 even for correct gradients

ε too small (e.g., 10^-10):
  - Subtractive cancellation (f(x+ε) ≈ f(x))
  - Numerical instability
  - Relative error > 0.1 due to floating point errors

ε optimal (10^-5 to 10^-7):
  - Good approximation
  - Numerically stable
  - Relative error < 10^-7 for correct gradients
\\\`\\\`\\\`

Recommended: ε = 10^-5 or 10^-7

**Relative Error Thresholds:**

\\\`\\\`\\\`
Relative difference = ||grad_analytical - grad_numerical|| / 
                     (||grad_analytical|| + ||grad_numerical||)

Interpretation:
  < 10^-7: Excellent - implementation definitely correct
  < 10^-5: Good - likely correct (acceptable for most purposes)
  < 10^-3: Concerning - may have bugs, check carefully
  > 10^-3: Failed - implementation has errors
\\\`\\\`\\\`

**Common Patterns:**

\\\`\\\`\\\`python
# Excellent implementation
relative_diff = 3.5e-9  # Nearly perfect match

# Good implementation (typical with float32)
relative_diff = 8.2e-6  # Small differences due to numerical precision

# Problematic (check for bugs)
relative_diff = 5.3e-4  # Systematic error likely

# Failed (definitely has bugs)
relative_diff = 0.15  # Completely wrong
\\\`\\\`\\\`

**When to Use Gradient Checking:**

**DO use when:**
1. **Implementing new architectures** - verify backprop is correct
2. **Custom layers** - test new operations
3. **Debugging** - track down gradient bugs
4. **Learning** - understand how backprop works
5. **Critical applications** - verify correctness before deployment

**DON'T use when:**
1. **Regular training** - way too slow (O(n) forward passes)
2. **Large networks** - infeasible for millions of parameters
3. **Production** - only needed during development
4. **Every epoch** - once verified, trust your implementation

**Best Practices:**

**1. Use Small Networks for Testing:**
\\\`\\\`\\\`python
# For gradient checking, use tiny network
test_net = Network([5, 3, 2])  # Only ~30 parameters
test_data = X[:1]  # Single sample

# Verify on tiny network first
passed, results = gradient_check(test_net, test_data, y[:1])

# If passes, trust it for larger networks
\\\`\\\`\\\`

**2. Test Each Component Separately:**
\\\`\\\`\\\`python
# Test layers independently
def test_relu_layer():
    layer = ReLULayer(10, 5)
    # ... gradient check just this layer

def test_softmax_layer():
    layer = SoftmaxLayer(10)
    # ... gradient check just this layer
\\\`\\\`\\\`

**3. Use Double Precision:**
\\\`\\\`\\\`python
# float64 for gradient checking (more precise)
network = Network([5, 3, 2], dtype=np.float64)
X = X.astype(np.float64)
y = y.astype(np.float64)

# After verification, use float32 for training (faster)
\\\`\\\`\\\`

**4. Turn Off Regularization:**
\\\`\\\`\\\`python
# Regularization adds to gradient
# Turn off for cleaner gradient checking
network.dropout_rate = 0.0
network.l2_lambda = 0.0
\\\`\\\`\\\`

**5. Use Deterministic Operations:**
\\\`\\\`\\\`python
# Disable dropout, batch norm, etc. during gradient check
network.eval()  # Evaluation mode
\\\`\\\`\\\`

**Typical Bugs Gradient Checking Catches:**

**1. Wrong Sign:**
\\\`\\\`\\\`python
# Bug
dL_dW = -X.T @ dL_dz  # Wrong sign!

# Correct
dL_dW = X.T @ dL_dz
\\\`\\\`\\\`
Relative error: ~2.0 (gradients in opposite direction)

**2. Forgot to Average Over Batch:**
\\\`\\\`\\\`python
# Bug
dL_dW = X.T @ dL_dz  # Missing /m!

# Correct
dL_dW = (X.T @ dL_dz) / m
\\\`\\\`\\\`
Relative error: proportional to batch size

**3. Wrong Activation Derivative:**
\\\`\\\`\\\`python
# Bug
dL_dz = dL_da * a  # Wrong! Should be a * (1 - a)

# Correct
dL_dz = dL_da * a * (1 - a)  # For sigmoid
\\\`\\\`\\\`
Relative error: ~0.5-2.0

**4. Shape Errors:**
\\\`\\\`\\\`python
# Bug
dL_dW = dL_dz @ X  # Wrong order!

# Correct
dL_dW = X.T @ dL_dz  # Transpose X
\\\`\\\`\\\`
Causes shape mismatch or wrong values

**Example Implementation:**

\\\`\\\`\\\`python
# Complete gradient check example
import numpy as np

def full_gradient_check_example():
    # Create small network
    np.random.seed(42)
    network = TwoLayerNetwork(input_size=3, hidden_size=4, output_size=2)
    
    # Single sample (faster)
    X = np.random.randn(1, 3)
    y = np.array([[1, 0]])
    
    # Gradient check
    passed, results = gradient_check(
        network, X, y,
        epsilon=1e-5,
        threshold=1e-5
    )
    
    if passed:
        print("\\n✓ Implementation verified! Safe to train.")
    else:
        print("\\n✗ Implementation has errors. Fix before training.")
        for param, result in results.items():
            if not result['passed',]:
                print(f"  {param}: error = {result['relative_diff',]:.2e}")

full_gradient_check_example()
\\\`\\\`\\\`

**Summary:**

Gradient checking:
- **What**: Compare backprop gradients with numerical finite differences
- **Why**: Verify implementation correctness (catches all bugs)
- **When**: During development/debugging, not during training
- **Threshold**: < 10^-7 excellent, < 10^-5 good, > 10^-3 failed
- **ε**: Use 10^-5 (balance accuracy vs numerical stability)
- **Cost**: O(n) forward passes - expensive but worth it for verification

**Golden Rule**: Always gradient check new implementations before training. Once verified, trust your backprop and never use numerical gradients during actual training.`,
    keyPoints: [
      'Gradient checking compares backprop with finite differences: [L(θ+ε) - L(θ-ε)]/(2ε)',
      'Relative error < 10^-7: excellent, < 10^-5: good, > 10^-3: failed implementation',
      'Use ε = 10^-5 for optimal accuracy vs numerical stability trade-off',
      'Catches ALL types of bugs: wrong signs, missing terms, shape errors, wrong derivatives',
      'Only for debugging/verification - too expensive O(n) for regular training',
      'Test on small networks (few parameters) first before scaling up',
      'Turn off regularization, dropout, and use deterministic mode for checking',
      'Once verified, trust backprop - numerical gradients not needed during training',
    ],
  },
];
