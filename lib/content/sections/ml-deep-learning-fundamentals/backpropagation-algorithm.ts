/**
 * Section: Backpropagation Algorithm
 * Module: Deep Learning Fundamentals
 *
 * Covers the backpropagation algorithm, chain rule application, computing gradients,
 * computational graph differentiation, and implementing backprop from scratch
 */

export const backpropagationAlgorithmSection = {
  id: 'backpropagation-algorithm',
  title: 'Backpropagation Algorithm',
  content: `
# Backpropagation Algorithm

## Introduction

**Backpropagation** (backward propagation of errors) is the algorithm that makes training deep neural networks practical. It efficiently computes gradients of the loss function with respect to all parameters in the network using the chain rule of calculus.

**What You'll Learn:**
- The backpropagation algorithm step-by-step
- Chain rule application in neural networks
- Computing gradients layer by layer
- Implementing backprop from scratch
- Computational efficiency of backprop
- Common implementation pitfalls

**Historical Note:** Backpropagation, popularized by Rumelhart, Hinton, and Williams (1986), was the breakthrough that enabled modern deep learning.

## The Core Problem

### What We Need

To train a neural network, we need:
\`\`\`
∂L/∂W  for all weight matrices
∂L/∂b  for all bias vectors
\`\`\`

Where L is the loss function. These gradients tell us how to adjust parameters to reduce loss.

### Naive Approach (Numerical Gradients)

\`\`\`python
def numerical_gradient (f, x, epsilon=1e-5):
    """
    Compute gradient numerically using finite differences
    SLOW but useful for gradient checking
    """
    grad = np.zeros_like (x)
    
    for i in range (len (x)):
        x_plus = x.copy()
        x_plus[i] += epsilon
        
        x_minus = x.copy()
        x_minus[i] -= epsilon
        
        # Finite difference approximation
        grad[i] = (f (x_plus) - f (x_minus)) / (2 * epsilon)
    
    return grad

# Example: For 1 million parameters
# Need 2 million forward passes!
# Computational cost: O(n) forward passes where n = number of parameters
# Backpropagation cost: O(1) forward pass + O(1) backward pass
\`\`\`

**Problem:** For a network with 1M parameters, numerical gradients require 2M forward passes!

**Solution:** Backpropagation computes all gradients in one forward + one backward pass.

## The Chain Rule: Foundation of Backpropagation

### Univariate Chain Rule

For composition of functions:
\`\`\`
If y = f (u) and u = g (x), then:
dy/dx = dy/du · du/dx
\`\`\`

### Multivariate Chain Rule

For multiple paths:
\`\`\`
If z = f (x, y), x = g (t), y = h (t), then:
dz/dt = (∂z/∂x)(dx/dt) + (∂z/∂y)(dy/dt)
\`\`\`

### Neural Network Application

\`\`\`
Layer structure: x → z = Wx + b → a = σ(z) → L

To find ∂L/∂W, we chain:
∂L/∂W = (∂L/∂a) · (∂a/∂z) · (∂z/∂W)
         ^^^^^^^^   ^^^^^^^^   ^^^^^^^^
         from next  activation  local
         layer      derivative  gradient
\`\`\`

## Backpropagation Algorithm: Step by Step

### Two-Layer Network Example

Network:
\`\`\`
Layer 1: z₁ = W₁x + b₁,  a₁ = σ(z₁)
Layer 2: z₂ = W₂a₁ + b₂, a₂ = σ(z₂)
Loss:    L = MSE(y, a₂)
\`\`\`

### Forward Pass (Compute outputs)

\`\`\`python
import numpy as np

class TwoLayerNetwork:
    """
    Simple 2-layer network for understanding backprop
    """
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights (small random values)
        self.W1 = np.random.randn (input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn (hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid (self, z):
        """Sigmoid activation"""
        return 1 / (1 + np.exp(-np.clip (z, -500, 500)))
    
    def sigmoid_derivative (self, z):
        """Derivative of sigmoid"""
        s = self.sigmoid (z)
        return s * (1 - s)
    
    def forward (self, X):
        """
        Forward propagation
        Returns outputs and cache for backprop
        """
        # Layer 1
        self.z1 = X @ self.W1 + self.b1        # Pre-activation
        self.a1 = self.sigmoid (self.z1)        # Post-activation
        
        # Layer 2
        self.z2 = self.a1 @ self.W2 + self.b2  # Pre-activation
        self.a2 = self.sigmoid (self.z2)        # Post-activation (output)
        
        # Cache for backward pass
        self.cache = {
            'X': X,
            'z1': self.z1, 'a1': self.a1,
            'z2': self.z2, 'a2': self.a2
        }
        
        return self.a2
    
    def compute_loss (self, y_true, y_pred):
        """MSE loss"""
        return np.mean((y_true - y_pred) ** 2)


# Test forward pass
np.random.seed(42)
network = TwoLayerNetwork (input_size=3, hidden_size=4, output_size=2)

X = np.array([[1, 2, 3]])  # Single sample
y = np.array([[0, 1]])      # True label

output = network.forward(X)
loss = network.compute_loss (y, output)

print("Forward Pass:")
print(f"  Input shape: {X.shape}")
print(f"  Hidden layer (a1) shape: {network.a1.shape}")
print(f"  Output (a2): {output}")
print(f"  Loss: {loss:.4f}")
\`\`\`

### Backward Pass (Compute gradients)

\`\`\`python
def backward (self, y_true):
    """
    Backward propagation - compute all gradients
    
    Mathematical flow:
    1. ∂L/∂a₂ = ∂/∂a₂ [MSE] 
    2. ∂L/∂z₂ = ∂L/∂a₂ · ∂a₂/∂z₂ (chain rule)
    3. ∂L/∂W₂ = a₁ᵀ · ∂L/∂z₂ (gradient for W₂)
    4. ∂L/∂a₁ = ∂L/∂z₂ · W₂ᵀ (gradient flowing back)
    5. ∂L/∂z₁ = ∂L/∂a₁ · ∂a₁/∂z₁ (chain rule)
    6. ∂L/∂W₁ = Xᵀ · ∂L/∂z₁ (gradient for W₁)
    """
    m = y_true.shape[0]  # Batch size
    
    # Get cached values
    X = self.cache['X']
    z1, a1 = self.cache['z1'], self.cache['a1']
    z2, a2 = self.cache['z2'], self.cache['a2']
    
    # ====================
    # OUTPUT LAYER (Layer 2)
    # ====================
    
    # Step 1: Gradient of loss w.r.t. output
    # L = (1/2m) Σ(y - a₂)²
    # ∂L/∂a₂ = (1/m)(a₂ - y)
    dL_da2 = (a2 - y_true) / m
    
    # Step 2: Gradient w.r.t. pre-activation z₂
    # Apply chain rule: ∂L/∂z₂ = ∂L/∂a₂ · ∂a₂/∂z₂
    # where ∂a₂/∂z₂ = σ'(z₂) = a₂(1 - a₂)
    dL_dz2 = dL_da2 * self.sigmoid_derivative (z2)
    
    # Step 3: Gradients for W₂ and b₂
    # z₂ = W₂a₁ + b₂
    # ∂z₂/∂W₂ = a₁, so ∂L/∂W₂ = a₁ᵀ · ∂L/∂z₂
    dL_dW2 = a1.T @ dL_dz2
    dL_db2 = np.sum (dL_dz2, axis=0, keepdims=True)
    
    # ====================
    # HIDDEN LAYER (Layer 1)
    # ====================
    
    # Step 4: Gradient flowing back to a₁
    # z₂ = W₂a₁ + b₂
    # ∂z₂/∂a₁ = W₂, so ∂L/∂a₁ = ∂L/∂z₂ · W₂ᵀ
    dL_da1 = dL_dz2 @ self.W2.T
    
    # Step 5: Gradient w.r.t. pre-activation z₁
    # Apply chain rule: ∂L/∂z₁ = ∂L/∂a₁ · ∂a₁/∂z₁
    dL_dz1 = dL_da1 * self.sigmoid_derivative (z1)
    
    # Step 6: Gradients for W₁ and b₁
    # z₁ = W₁X + b₁
    dL_dW1 = X.T @ dL_dz1
    dL_db1 = np.sum (dL_dz1, axis=0, keepdims=True)
    
    # Store gradients
    self.gradients = {
        'dW1': dL_dW1, 'db1': dL_db1,
        'dW2': dL_dW2, 'db2': dL_db2
    }
    
    return self.gradients

# Add to TwoLayerNetwork class
TwoLayerNetwork.backward = backward

# Test backward pass
gradients = network.backward (y)

print("\\nBackward Pass (Gradients):")
print(f"  dW1 shape: {gradients['dW1'].shape}, mean: {np.mean (np.abs (gradients['dW1'])):.6f}")
print(f"  db1 shape: {gradients['db1'].shape}, mean: {np.mean (np.abs (gradients['db1'])):.6f}")
print(f"  dW2 shape: {gradients['dW2'].shape}, mean: {np.mean (np.abs (gradients['dW2'])):.6f}")
print(f"  db2 shape: {gradients['db2'].shape}, mean: {np.mean (np.abs (gradients['db2'])):.6f}")
\`\`\`

### Gradient Descent Update

\`\`\`python
def update_parameters (self, learning_rate=0.01):
    """Update parameters using computed gradients"""
    self.W1 -= learning_rate * self.gradients['dW1']
    self.b1 -= learning_rate * self.gradients['db1']
    self.W2 -= learning_rate * self.gradients['dW2']
    self.b2 -= learning_rate * self.gradients['db2']

# Add to class
TwoLayerNetwork.update_parameters = update_parameters

# Complete training step
print("\\nTraining Step:")
print(f"  Loss before: {loss:.4f}")

network.update_parameters (learning_rate=0.1)

# Forward pass again
output_after = network.forward(X)
loss_after = network.compute_loss (y, output_after)
print(f"  Loss after: {loss_after:.4f}")
print(f"  Improvement: {loss - loss_after:.6f}")
\`\`\`

## Computational Efficiency of Backpropagation

### Complexity Analysis

For a network with L layers and n parameters:

**Numerical Gradients:**
- Complexity: O(n) forward passes
- For 1M parameters: 2M forward passes

**Backpropagation:**
- Complexity: 1 forward + 1 backward pass
- Backward pass ≈ 2x cost of forward pass
- Total: ≈ 3 forward passes equivalent

**Speedup:** O(n) → O(1) in terms of forward passes!

### Memory vs Computation Trade-off

\`\`\`python
# Two strategies for backpropagation:

# Strategy 1: Store all activations (Fast, Memory-heavy)
def forward_store_all (network, X):
    cache = {}
    a = X
    for l in range(L):
        z = a @ W[l] + b[l]
        a = activation (z)
        cache[l] = (z, a)  # Store everything
    return a, cache

# Memory: O(batch_size × Σ layer_sizes)
# Speed: Fast backward pass (no recomputation)

# Strategy 2: Gradient checkpointing (Slow, Memory-light)
def forward_checkpoint (network, X, checkpoint_layers):
    cache = {}
    a = X
    for l in range(L):
        z = a @ W[l] + b[l]
        a = activation (z)
        if l in checkpoint_layers:
            cache[l] = (z, a)  # Only store checkpoints
    return a, cache

# Memory: O(batch_size × checkpoint_layer_sizes)
# Speed: Slower backward (recompute between checkpoints)
# Used for very deep networks (100+ layers)
\`\`\`

## Backpropagation with Different Activations

### ReLU Backprop

\`\`\`python
def relu_forward (z):
    """ReLU forward"""
    return np.maximum(0, z)

def relu_backward (dL_da, z):
    """
    ReLU backward
    
    ∂a/∂z = 1 if z > 0, else 0
    ∂L/∂z = ∂L/∂a · (1 if z > 0 else 0)
    """
    dL_dz = dL_da * (z > 0).astype (float)
    return dL_dz

# Example
z = np.array([[-1, 0.5, 2], [0, -0.5, 1]])
a = relu_forward (z)
dL_da = np.ones_like (a)  # Assume gradient from next layer

dL_dz = relu_backward (dL_da, z)

print("ReLU Backpropagation:")
print(f"  z:\\n{z}")
print(f"  a (ReLU(z)):\\n{a}")
print(f"  dL/dz:\\n{dL_dz}")
print("  → Gradient passes through only where z > 0")
\`\`\`

### Softmax + Cross-Entropy Backprop

\`\`\`python
def softmax_cross_entropy_backward (y_pred, y_true):
    """
    Combined softmax + cross-entropy backward pass
    
    Forward:
      ŷ = softmax (z)
      L = -Σ yᵢ log(ŷᵢ)
    
    Backward (amazing simplification!):
      ∂L/∂z = ŷ - y
    """
    return y_pred - y_true

# Example: 3-class classification
y_pred = np.array([[0.7, 0.2, 0.1],   # Sample 1
                   [0.1, 0.8, 0.1]])  # Sample 2
y_true = np.array([[1, 0, 0],         # Sample 1: class 0
                   [0, 1, 0]])         # Sample 2: class 1

dL_dz = softmax_cross_entropy_backward (y_pred, y_true)

print("\\nSoftmax + Cross-Entropy Backprop:")
print(f"  Predicted probabilities:\\n{y_pred}")
print(f"  True labels (one-hot):\\n{y_true}")
print(f"  Gradient ∂L/∂z:\\n{dL_dz}")
print("  → Gradient = prediction error!")
\`\`\`

## Matrix Calculus in Backpropagation

### Jacobian Matrices

For vector-to-vector functions, gradients are Jacobians:

\`\`\`python
# For z = Wa + b where W is (n × m), a is (m × 1)
# ∂z/∂W is a Jacobian matrix

def compute_jacobian_dz_dW(a, z):
    """
    Compute ∂z/∂W for z = Wa + b
    
    Jacobian shape: (n, n, m) - one (n × m) matrix per output
    But we don't actually compute full Jacobian (too large!)
    
    Instead, use chain rule cleverly:
    ∂L/∂W = ∂L/∂z · ∂z/∂W = (∂L/∂z) ⊗ a
    
    Where ⊗ is outer product
    """
    # If ∂L/∂z is (batch, n) and a is (batch, m)
    # Then ∂L/∂W = aᵀ · (∂L/∂z) is (m, n)
    pass  # Conceptual - actual implementation in backward()

# The key insight: We never explicitly compute full Jacobians!
# We use chain rule to compute matrix products directly.
\`\`\`

### Dimension Tracking

Critical for debugging backprop:

\`\`\`python
def check_gradient_shapes (network, X, y):
    """
    Verify gradient shapes match parameter shapes
    Essential for debugging!
    """
    network.forward(X)
    gradients = network.backward (y)
    
    print("Gradient Shape Check:")
    print("-" * 60)
    
    # Check W1
    assert gradients['dW1'].shape == network.W1.shape, \
        f"dW1 shape mismatch: {gradients['dW1'].shape} vs {network.W1.shape}"
    print(f"✓ dW1: {gradients['dW1'].shape} matches W1: {network.W1.shape}")
    
    # Check b1
    assert gradients['db1'].shape == network.b1.shape, \
        f"db1 shape mismatch: {gradients['db1'].shape} vs {network.b1.shape}"
    print(f"✓ db1: {gradients['db1'].shape} matches b1: {network.b1.shape}")
    
    # Check W2
    assert gradients['dW2'].shape == network.W2.shape, \
        f"dW2 shape mismatch: {gradients['dW2'].shape} vs {network.W2.shape}"
    print(f"✓ dW2: {gradients['dW2'].shape} matches W2: {network.W2.shape}")
    
    # Check b2
    assert gradients['db2'].shape == network.b2.shape, \
        f"db2 shape mismatch: {gradients['db2'].shape} vs {network.b2.shape}"
    print(f"✓ db2: {gradients['db2'].shape} matches b2: {network.b2.shape}")
    
    print("\\nAll gradient shapes correct!")

# Test
check_gradient_shapes (network, X, y)
\`\`\`

## Gradient Checking

### Numerical Gradient Verification

\`\`\`python
def gradient_check (network, X, y, epsilon=1e-5):
    """
    Verify backprop gradients against numerical gradients
    Should only be used for debugging, not training!
    """
    # Compute analytical gradients (backprop)
    network.forward(X)
    network.backward (y)
    analytical_grads = network.gradients
    
    # Compute numerical gradients
    def compute_numerical_gradient (param_name):
        param = getattr (network, param_name)
        numerical_grad = np.zeros_like (param)
        
        # Iterate over each parameter
        it = np.nditer (param, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            old_value = param[idx]
            
            # f (x + epsilon)
            param[idx] = old_value + epsilon
            output_plus = network.forward(X)
            loss_plus = network.compute_loss (y, output_plus)
            
            # f (x - epsilon)
            param[idx] = old_value - epsilon
            output_minus = network.forward(X)
            loss_minus = network.compute_loss (y, output_minus)
            
            # Numerical gradient
            numerical_grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)
            
            # Restore
            param[idx] = old_value
            it.iternext()
        
        return numerical_grad
    
    print("\\nGradient Checking:")
    print("-" * 60)
    
    # Check each parameter
    for param_name in ['W1', 'b1', 'W2', 'b2']:
        analytical = analytical_grads[f'd{param_name}']
        numerical = compute_numerical_gradient (param_name)
        
        # Compute relative difference
        diff = np.linalg.norm (analytical - numerical)
        norm_sum = np.linalg.norm (analytical) + np.linalg.norm (numerical)
        relative_diff = diff / (norm_sum + 1e-8)
        
        status = "✓ PASS" if relative_diff < 1e-5 else "✗ FAIL"
        print(f"  {param_name}: relative diff = {relative_diff:.2e} {status}")
    
    print("\\nGradient check complete!")
    print("Note: This is expensive (O(n) forward passes), only for debugging!")

# Run gradient check (small network for speed)
small_net = TwoLayerNetwork(2, 3, 1)
X_small = np.random.randn(1, 2)
y_small = np.random.randn(1, 1)

gradient_check (small_net, X_small, y_small)
\`\`\`

## Common Backprop Mistakes and Solutions

### Mistake 1: Wrong Gradient Shapes

\`\`\`python
# ❌ WRONG: Gradient shape doesn't match parameter
dW = dL_dz @ a.T  # If dimensions don't match, this fails

# ✓ CORRECT: Ensure shapes align
# For z = Wa + b where W is (n_out, n_in), a is (batch, n_in)
# We need dW to be (n_out, n_in)
# Chain rule: dL/dW = dL/dz · dz/dW = dL/dz · aᵀ
# Shapes: (batch, n_out) → (batch, n_in) → sum over batch
dW = (dL_dz.T @ a) / batch_size  # (n_out, batch) @ (batch, n_in) = (n_out, n_in) ✓
\`\`\`

### Mistake 2: Forgetting Chain Rule

\`\`\`python
# ❌ WRONG: Computing gradient w.r.t. post-activation
dL_dW = dL_da @ a.T  # Skips the activation derivative!

# ✓ CORRECT: Apply chain rule through activation
dL_dz = dL_da * activation_derivative (z)  # Chain through activation
dL_dW = dL_dz.T @ input  # Then compute weight gradient
\`\`\`

### Mistake 3: Not Averaging Over Batch

\`\`\`python
# ❌ WRONG: Sum gradients without averaging
dW = X.T @ dL_dz  # Gradient grows with batch size!

# ✓ CORRECT: Average over batch
dW = (X.T @ dL_dz) / batch_size  # Gradient independent of batch size
\`\`\`

### Mistake 4: Modifying Cached Values

\`\`\`python
# ❌ WRONG: In-place operations corrupt cache
a = relu (z)
a += 0.1  # Modifies cached value!

# ✓ CORRECT: Create new arrays
a = relu (z)
a_modified = a + 0.1  # Doesn't affect cache
\`\`\`

## Visualizing Backpropagation

\`\`\`python
import matplotlib.pyplot as plt

def visualize_gradient_flow (network, X, y):
    """Visualize how gradients flow backward through network"""
    network.forward(X)
    network.backward (y)
    
    # Compute gradient magnitudes
    layers = ['Layer 1', 'Layer 2']
    weight_grads = [
        np.mean (np.abs (network.gradients['dW1'])),
        np.mean (np.abs (network.gradients['dW2']))
    ]
    bias_grads = [
        np.mean (np.abs (network.gradients['db1'])),
        np.mean (np.abs (network.gradients['db2']))
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot weight gradients
    ax1.bar (layers, weight_grads, alpha=0.7, color='blue')
    ax1.set_ylabel('Mean Absolute Gradient')
    ax1.set_title('Weight Gradients by Layer')
    ax1.set_ylim(0, max (weight_grads) * 1.2)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot bias gradients
    ax2.bar (layers, bias_grads, alpha=0.7, color='green')
    ax2.set_ylabel('Mean Absolute Gradient')
    ax2.set_title('Bias Gradients by Layer')
    ax2.set_ylim(0, max (bias_grads) * 1.2)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    print("Gradient Flow Analysis:")
    print(f"  Layer 1 weight gradient: {weight_grads[0]:.6f}")
    print(f"  Layer 2 weight gradient: {weight_grads[1]:.6f}")
    print(f"  Ratio (L1/L2): {weight_grads[0]/weight_grads[1]:.2f}")
    
    if weight_grads[0] / weight_grads[1] < 0.1:
        print("  ⚠ Warning: Potential vanishing gradient in early layers")

visualize_gradient_flow (network, X, y)
\`\`\`

## Key Takeaways

1. **Backpropagation** efficiently computes all gradients using the chain rule
2. **Forward pass** computes outputs and caches values needed for backward
3. **Backward pass** computes gradients layer-by-layer in reverse order
4. **Complexity**: O(1) forward/backward vs O(n) numerical gradients
5. **Chain rule** is applied at every layer: ∂L/∂z = ∂L/∂a · ∂a/∂z
6. **Gradient checking** verifies backprop implementation correctness
7. **Shape checking** prevents common implementation bugs
8. **Clean gradients** (like ŷ - y) emerge from well-designed loss/activation pairs

## What's Next

Backpropagation computes gradients, but how do we use them to update weights optimally? Next:
- **Optimization Algorithms**: SGD, Momentum, Adam, and adaptive methods
- Learning rates and scheduling
- Convergence properties
- Modern optimizers for deep learning
`,
};
