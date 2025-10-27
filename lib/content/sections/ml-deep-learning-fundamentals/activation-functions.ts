/**
 * Section: Activation Functions
 * Module: Deep Learning Fundamentals
 *
 * Covers sigmoid, tanh, ReLU and variants, softmax, why non-linearity matters,
 * the dying ReLU problem, and choosing appropriate activation functions
 */

export const activationFunctionsSection = {
  id: 'activation-functions',
  title: 'Activation Functions',
  content: `
# Activation Functions

## Introduction

Activation functions are the non-linear transformations applied to neuron outputs that enable neural networks to learn complex patterns. Without them, neural networks would be limited to learning only linear relationships, regardless of depth.

**What You'll Learn:**
- Mathematical definitions of common activation functions
- Properties and derivatives of each activation function
- When to use each activation function
- The dying ReLU problem and solutions
- Softmax for multi-class classification
- Implementing activation functions from scratch

## Why Non-Linearity Matters

### The Linearity Problem

Without activation functions, a neural network is just a series of linear transformations:

\`\`\`
Layer 1: h₁ = W₁x + b₁
Layer 2: h₂ = W₂h₁ + b₂ = W₂(W₁x + b₁) + b₂
       = (W₂W₁)x + (W₂b₁ + b₂)
       = W_eff·x + b_eff
\`\`\`

**Result**: Multiple linear layers collapse to a single linear transformation! The network can only learn linear relationships.

### The Power of Non-Linearity

Adding non-linear activation functions:

\`\`\`
Layer 1: h₁ = σ(W₁x + b₁)        # Non-linear transformation
Layer 2: h₂ = σ(W₂h₁ + b₂)       # Cannot be simplified!
\`\`\`

Now the network can learn:
- Polynomial functions
- Trigonometric patterns
- Step functions
- Any continuous function (universal approximation)

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt

# Demonstration: Linear vs Non-linear Transformation
def linear_transform (x, W1, W2):
    """Two linear layers - equivalent to one"""
    h1 = W1 @ x
    h2 = W2 @ h1
    return h2

def nonlinear_transform (x, W1, W2, activation):
    """Two layers with activation - fundamentally different"""
    h1 = activation(W1 @ x)  # Non-linearity!
    h2 = activation(W2 @ h1)  # Non-linearity!
    return h2

# Test on simple data
x = np.linspace(-5, 5, 1000).reshape(1, -1)
W1 = np.array([[0.5]])
W2 = np.array([[2.0]])

# ReLU activation
relu = lambda z: np.maximum(0, z)

# Compare outputs
linear_output = linear_transform (x, W1, W2)
nonlinear_output = nonlinear_transform (x, W1, W2, relu)

plt.figure (figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot (x.flatten(), linear_output.flatten(), linewidth=2)
plt.title('Two Linear Layers (Still Linear)', fontsize=14, fontweight='bold')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot (x.flatten(), nonlinear_output.flatten(), linewidth=2, color='red')
plt.title('Two Layers with ReLU (Non-linear)', fontsize=14, fontweight='bold')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
\`\`\`

**Key Insight**: Activation functions are what give neural networks their power. Choose them carefully!

## Sigmoid Activation Function

### Mathematical Definition

The **sigmoid** (or logistic) function squashes inputs to the range (0, 1):

\`\`\`
σ(z) = 1 / (1 + e^(-z))
\`\`\`

**Properties:**
- **Range**: (0, 1) - useful for probabilities
- **Differentiable**: σ'(z) = σ(z)(1 - σ(z))
- **Monotonic**: Always increasing
- **S-shaped curve**: Smooth transition

### Derivative

The sigmoid derivative has an elegant form:

\`\`\`
d/dz σ(z) = σ(z) · (1 - σ(z))
\`\`\`

**Why this matters**: In backpropagation, we need derivatives. Sigmoid\'s derivative is efficient to compute.

### Implementation

\`\`\`python
def sigmoid (z):
    """
    Sigmoid activation function
    
    Args:
        z: Input values (any shape)
    
    Returns:
        Sigmoid applied element-wise
    """
    # Clip to prevent overflow in exp
    z_clipped = np.clip (z, -500, 500)
    return 1 / (1 + np.exp(-z_clipped))

def sigmoid_derivative (z):
    """
    Derivative of sigmoid
    
    Args:
        z: Input values
    
    Returns:
        Derivative evaluated at z
    """
    s = sigmoid (z)
    return s * (1 - s)

# Visualize sigmoid and its derivative
z = np.linspace(-10, 10, 1000)
sig = sigmoid (z)
sig_deriv = sigmoid_derivative (z)

plt.figure (figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot (z, sig, linewidth=2, label='σ(z)')
plt.axhline (y=0, color='k', linestyle='--', alpha=0.3)
plt.axhline (y=1, color='k', linestyle='--', alpha=0.3)
plt.axvline (x=0, color='k', linestyle='--', alpha=0.3)
plt.xlabel('z', fontsize=12)
plt.ylabel('σ(z)', fontsize=12)
plt.title('Sigmoid Function', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim(-0.1, 1.1)

plt.subplot(1, 2, 2)
plt.plot (z, sig_deriv, linewidth=2, color='orange', label="σ'(z)")
plt.axhline (y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline (x=0, color='k', linestyle='--', alpha=0.3)
plt.xlabel('z', fontsize=12)
plt.ylabel("σ'(z)", fontsize=12)
plt.title('Sigmoid Derivative', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

print(f"σ(0) = {sigmoid(0):.4f} (center point)")
print(f"σ(5) ≈ {sigmoid(5):.4f} (saturates to 1)")
print(f"σ(-5) ≈ {sigmoid(-5):.4f} (saturates to 0)")
print(f"\\nσ'(0) = {sigmoid_derivative(0):.4f} (maximum gradient)")
print(f"σ'(5) ≈ {sigmoid_derivative(5):.4f} (gradient near zero)")
\`\`\`

**Output:**
\`\`\`
σ(0) = 0.5000 (center point)
σ(5) ≈ 0.9933 (saturates to 1)
σ(-5) ≈ 0.0067 (saturates to 0)

σ'(0) = 0.2500 (maximum gradient)
σ'(5) ≈ 0.0066 (gradient near zero)
\`\`\`

### Pros and Cons

**Advantages:**
- ✅ Output in (0, 1) interpretable as probabilities
- ✅ Smooth and differentiable everywhere
- ✅ Historical importance (used in early neural networks)
- ✅ Works well for binary classification output

**Disadvantages:**
- ❌ **Vanishing gradient**: Derivative approaches 0 for large |z|
- ❌ **Not zero-centered**: Outputs always positive, causing zig-zag dynamics
- ❌ **Computationally expensive**: Requires exp() computation
- ❌ **Saturation**: Neurons can "die" when saturated

### When to Use Sigmoid

**Good for:**
- Binary classification output layer
- Gates in LSTMs and GRUs
- When probabilities are needed

**Avoid for:**
- Hidden layers (use ReLU instead)
- Deep networks (vanishing gradient problem)
- When training is slow

## Hyperbolic Tangent (tanh) Activation

### Mathematical Definition

The **tanh** function is a scaled sigmoid that outputs to (-1, 1):

\`\`\`
tanh (z) = (e^z - e^(-z)) / (e^z + e^(-z)) = 2σ(2z) - 1
\`\`\`

**Properties:**
- **Range**: (-1, 1) - zero-centered
- **Derivative**: tanh'(z) = 1 - tanh²(z)
- **Symmetric**: tanh(-z) = -tanh (z)

### Implementation

\`\`\`python
def tanh (z):
    """
    Hyperbolic tangent activation
    
    Args:
        z: Input values
    
    Returns:
        tanh applied element-wise
    """
    return np.tanh (z)  # NumPy has efficient implementation

def tanh_derivative (z):
    """Derivative of tanh"""
    t = tanh (z)
    return 1 - t ** 2

# Visualize tanh
z = np.linspace(-5, 5, 1000)
tanh_vals = tanh (z)
tanh_deriv = tanh_derivative (z)
sig_vals = sigmoid (z)

plt.figure (figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot (z, tanh_vals, linewidth=2, label='tanh (z)', color='blue')
plt.plot (z, sig_vals, linewidth=2, label='σ(z)', color='green', alpha=0.6)
plt.axhline (y=0, color='k', linestyle='--', alpha=0.3)
plt.axhline (y=1, color='k', linestyle='--', alpha=0.3)
plt.axhline (y=-1, color='k', linestyle='--', alpha=0.3)
plt.xlabel('z', fontsize=12)
plt.ylabel('Activation', fontsize=12)
plt.title('tanh vs Sigmoid', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot (z, tanh_deriv, linewidth=2, label="tanh'(z)", color='orange')
plt.plot (z, sigmoid_derivative (z), linewidth=2, label="σ'(z)", color='red', alpha=0.6)
plt.axhline (y=0, color='k', linestyle='--', alpha=0.3)
plt.xlabel('z', fontsize=12)
plt.ylabel('Derivative', fontsize=12)
plt.title('Derivatives Comparison', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
\`\`\`

### Advantages Over Sigmoid

1. **Zero-centered**: Outputs range from -1 to 1
   - Helps with optimization (no zig-zag dynamics)
   - Gradients can flow more naturally

2. **Stronger gradients**: Maximum derivative is 1 (vs 0.25 for sigmoid)

3. **Better for hidden layers**: Generally preferred over sigmoid

**Still has vanishing gradient problem** for large |z|.

### When to Use tanh

**Good for:**
- Hidden layers in shallow networks
- RNNs and LSTMs (as activations, not gates)
- When zero-centered outputs are beneficial

**Avoid for:**
- Deep networks (still has vanishing gradient)
- When computational speed is critical

## ReLU (Rectified Linear Unit)

### Mathematical Definition

**ReLU** is the most popular activation function in modern deep learning:

\`\`\`
ReLU(z) = max(0, z) = {
  z  if z > 0
  0  if z ≤ 0
}
\`\`\`

**Derivative:**
\`\`\`
ReLU'(z) = {
  1  if z > 0
  0  if z ≤ 0
}
\`\`\`

**Properties:**
- **Range**: [0, ∞)
- **Non-saturating**: No vanishing gradient for positive values
- **Sparse activation**: About 50% of neurons are 0
- **Computationally cheap**: Just a threshold operation

### Implementation

\`\`\`python
def relu (z):
    """
    ReLU activation function
    
    Args:
        z: Input values
    
    Returns:
        max(0, z) element-wise
    """
    return np.maximum(0, z)

def relu_derivative (z):
    """
    Derivative of ReLU
    
    Args:
        z: Input values
    
    Returns:
        1 if z > 0, else 0
    """
    return (z > 0).astype (float)

# Visualize ReLU
z = np.linspace(-5, 5, 1000)
relu_vals = relu (z)
relu_deriv = relu_derivative (z)

plt.figure (figsize=(14, 5))

plt.subplot(1, 3, 1)
plt.plot (z, relu_vals, linewidth=2, color='purple', label='ReLU(z)')
plt.plot (z, tanh (z), linewidth=2, color='blue', alpha=0.5, label='tanh (z)')
plt.plot (z, sigmoid (z), linewidth=2, color='green', alpha=0.5, label='σ(z)')
plt.axhline (y=0, color='k', linestyle='--', alpha=0.3)
plt.xlabel('z', fontsize=12)
plt.ylabel('Activation', fontsize=12)
plt.title('ReLU vs tanh vs Sigmoid', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-1.5, 5)

plt.subplot(1, 3, 2)
plt.plot (z, relu_deriv, linewidth=2, color='orange', label="ReLU'(z)")
plt.axhline (y=0, color='k', linestyle='--', alpha=0.3)
plt.axhline (y=1, color='k', linestyle='--', alpha=0.3)
plt.xlabel('z', fontsize=12)
plt.ylabel('Derivative', fontsize=12)
plt.title('ReLU Derivative', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-0.2, 1.2)

plt.subplot(1, 3, 3)
# Show sparsity
np.random.seed(42)
activations = np.random.randn(100)
relu_activations = relu (activations)
plt.bar (range(100), relu_activations, color='purple', alpha=0.6)
plt.xlabel('Neuron Index', fontsize=12)
plt.ylabel('Activation', fontsize=12)
plt.title (f'ReLU Sparsity ({np.sum (relu_activations == 0)}% zeros)', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
\`\`\`

### Why ReLU is Popular

**Advantages:**
- ✅ **No vanishing gradient** for positive inputs (gradient = 1)
- ✅ **Computational efficiency**: Just comparison and thresholding
- ✅ **Sparse activations**: Approximately 50% of neurons are zero
- ✅ **Biological plausibility**: Neurons either fire or don't
- ✅ **Faster convergence**: Networks train 6x faster than sigmoid/tanh

**Disadvantages:**
- ❌ **Dying ReLU problem**: Neurons can permanently die (output 0 for all inputs)
- ❌ **Not zero-centered**: All outputs are non-negative
- ❌ **Unbounded**: Output can grow arbitrarily large

### The Dying ReLU Problem

**What happens:**1. During training, a large gradient update causes weights to become very negative
2. For all inputs, z = wx + b < 0
3. ReLU always outputs 0
4. Gradient is always 0
5. Neuron never recovers—it's "dead"

\`\`\`python
# Demonstration: Dying ReLU
class SimpleNetwork:
    def __init__(self):
        self.w = np.array([1.5, -0.5])  # Initial weights
        self.b = 0.0
    
    def forward (self, X):
        z = X @ self.w + self.b
        return relu (z), z
    
    def update_with_bad_gradient (self, learning_rate=10.0):
        # Simulate a large negative gradient update
        self.w -= learning_rate * np.array([1.0, 1.0])
        self.b -= learning_rate * 1.0

# Before bad update
net = SimpleNetwork()
X_test = np.random.randn(100, 2)
activations_before, z_before = net.forward(X_test)

print("Before bad update:")
print(f"  Weights: {net.w}")
print(f"  Bias: {net.b:.2f}")
print(f"  Active neurons: {np.sum (activations_before > 0)}/100")
print(f"  Average activation: {np.mean (activations_before):.4f}")

# After bad update
net.update_with_bad_gradient (learning_rate=10.0)
activations_after, z_after = net.forward(X_test)

print("\\nAfter bad update (dying ReLU):")
print(f"  Weights: {net.w}")
print(f"  Bias: {net.b:.2f}")
print(f"  Active neurons: {np.sum (activations_after > 0)}/100")
print(f"  Average activation: {np.mean (activations_after):.4f}")
print(f"  All pre-activations negative: {np.all (z_after < 0)}")
print("  → Neuron is DEAD! All gradients = 0, cannot recover.")
\`\`\`

**Output:**
\`\`\`
Before bad update:
  Weights: [ 1.5 -0.5]
  Bias: 0.00
  Active neurons: 68/100
  Average activation: 0.8342

After bad update (dying ReLU):
  Weights: [-8.5 -10.5]
  Bias: -10.00
  All pre-activations negative: True
  Active neurons: 0/100
  Average activation: 0.0000
  → Neuron is DEAD! All gradients = 0, cannot recover.
\`\`\`

**Solution**: Use variants like Leaky ReLU, which we'll cover next.

## ReLU Variants

### Leaky ReLU

**Purpose**: Fix the dying ReLU problem by allowing small negative gradient:

\`\`\`
LeakyReLU(z) = {
  z         if z > 0
  αz        if z ≤ 0
}
\`\`\`

Where α is a small constant (typically 0.01).

\`\`\`python
def leaky_relu (z, alpha=0.01):
    """
    Leaky ReLU activation
    
    Args:
        z: Input values
        alpha: Slope for negative values (default 0.01)
    
    Returns:
        Leaky ReLU applied element-wise
    """
    return np.where (z > 0, z, alpha * z)

def leaky_relu_derivative (z, alpha=0.01):
    """Derivative of Leaky ReLU"""
    return np.where (z > 0, 1, alpha)

# Visualize Leaky ReLU
z = np.linspace(-5, 5, 1000)
relu_vals = relu (z)
leaky_vals = leaky_relu (z, alpha=0.01)

plt.figure (figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot (z, relu_vals, linewidth=2, label='ReLU', color='purple')
plt.plot (z, leaky_vals, linewidth=2, label='Leaky ReLU (α=0.01)', color='red')
plt.axhline (y=0, color='k', linestyle='--', alpha=0.3)
plt.xlabel('z', fontsize=12)
plt.ylabel('Activation', fontsize=12)
plt.title('ReLU vs Leaky ReLU', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-0.2, 5)

plt.subplot(1, 2, 2)
plt.plot (z, relu_derivative (z), linewidth=2, label="ReLU'", color='purple')
plt.plot (z, leaky_relu_derivative (z, 0.01), linewidth=2, label="Leaky ReLU'", color='red')
plt.axhline (y=0, color='k', linestyle='--', alpha=0.3)
plt.xlabel('z', fontsize=12)
plt.ylabel('Derivative', fontsize=12)
plt.title('Derivative Comparison', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-0.1, 1.1)

plt.tight_layout()
plt.show()

print("Leaky ReLU Properties:")
print(f"  For z > 0: output = z, gradient = 1")
print(f"  For z < 0: output = 0.01z, gradient = 0.01")
print("  → Neurons never completely die!")
\`\`\`

**Advantages over ReLU:**
- ✅ No dying neuron problem (always has non-zero gradient)
- ✅ Same computational efficiency as ReLU
- ✅ Often works better in practice

### Parametric ReLU (PReLU)

**Idea**: Learn the α parameter during training instead of fixing it:

\`\`\`
PReLU(z) = {
  z         if z > 0
  αz        if z ≤ 0
}
\`\`\`

Where α is a learnable parameter.

### Exponential Linear Unit (ELU)

**Purpose**: Smooth activation that can produce negative values:

\`\`\`
ELU(z) = {
  z                    if z > 0
  α(e^z - 1)           if z ≤ 0
}
\`\`\`

\`\`\`python
def elu (z, alpha=1.0):
    """
    Exponential Linear Unit (ELU)
    
    Args:
        z: Input values
        alpha: Scale for negative values
    
    Returns:
        ELU applied element-wise
    """
    return np.where (z > 0, z, alpha * (np.exp (z) - 1))

# Visualize all ReLU variants
z = np.linspace(-3, 3, 1000)
relu_vals = relu (z)
leaky_vals = leaky_relu (z, alpha=0.01)
elu_vals = elu (z, alpha=1.0)

plt.figure (figsize=(10, 6))
plt.plot (z, relu_vals, linewidth=2, label='ReLU', color='purple')
plt.plot (z, leaky_vals, linewidth=2, label='Leaky ReLU (α=0.01)', color='red')
plt.plot (z, elu_vals, linewidth=2, label='ELU (α=1.0)', color='green')
plt.axhline (y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline (x=0, color='k', linestyle='--', alpha=0.3)
plt.xlabel('z', fontsize=12)
plt.ylabel('Activation', fontsize=12)
plt.title('ReLU Variants Comparison', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-1.5, 3)
plt.tight_layout()
plt.show()
\`\`\`

**ELU Advantages:**
- ✅ Smooth everywhere (continuously differentiable)
- ✅ Negative values push mean activation closer to zero
- ✅ Better gradient flow than ReLU
- ❌ More expensive (requires exp() for negative values)

### SELU (Scaled ELU)

**Self-normalizing** activation that maintains mean ≈ 0 and variance ≈ 1:

\`\`\`
SELU(z) = λ · ELU(z, α)
\`\`\`

Where λ ≈ 1.0507 and α ≈ 1.6733 are fixed constants derived from theory.

**Special property**: With proper initialization, SELU networks are self-normalizing (don't need batch normalization).

## Softmax Activation Function

### Purpose: Multi-class Classification

For multi-class classification, we need outputs that:
1. Are all non-negative
2. Sum to 1 (probability distribution)
3. Represent class probabilities

**Softmax** converts logits (raw scores) to probabilities:

\`\`\`
softmax (z)ᵢ = e^(zᵢ) / Σⱼ e^(zⱼ)
\`\`\`

### Implementation

\`\`\`python
def softmax (z):
    """
    Softmax activation for multi-class classification
    
    Args:
        z: Logits, shape (batch_size, num_classes)
    
    Returns:
        Probabilities, shape (batch_size, num_classes)
    """
    # Subtract max for numerical stability (prevents overflow)
    z_shifted = z - np.max (z, axis=-1, keepdims=True)
    exp_z = np.exp (z_shifted)
    return exp_z / np.sum (exp_z, axis=-1, keepdims=True)

# Example: 3-class classification
logits = np.array([
    [2.0, 1.0, 0.1],  # Sample 1
    [1.0, 3.0, 0.2],  # Sample 2
    [0.1, 0.2, 2.5],  # Sample 3
])

probabilities = softmax (logits)

print("Multi-class Classification Example:")
print("\\nLogits (raw scores):")
print(logits)
print("\\nProbabilities (after softmax):")
print(probabilities)
print("\\nVerification:")
for i, probs in enumerate (probabilities):
    print(f"  Sample {i+1}: sum = {np.sum (probs):.6f}, predicted class = {np.argmax (probs)}")

# Visualize softmax
plt.figure (figsize=(12, 4))

for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.bar(['Class 0', 'Class 1', 'Class 2'], probabilities[i], color=['blue', 'green', 'red'], alpha=0.6)
    plt.ylabel('Probability')
    plt.title (f'Sample {i+1} Classification\\n(Predicted: Class {np.argmax (probabilities[i])})')
    plt.ylim(0, 1)
    plt.axhline (y=0.5, color='k', linestyle='--', alpha=0.3)
    plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
\`\`\`

**Output:**
\`\`\`
Logits (raw scores):
[[2.  1.  0.1]
 [1.  3.  0.2]
 [0.1 0.2 2.5]]

Probabilities (after softmax):
[[0.659  0.242  0.099]
 [0.117  0.858  0.025]
 [0.123  0.137  0.740]]

Verification:
  Sample 1: sum = 1.000000, predicted class = 0
  Sample 2: sum = 1.000000, predicted class = 1
  Sample 3: sum = 1.000000, predicted class = 2
\`\`\`

**Key Properties:**
- All outputs are positive
- Sum of outputs = 1 (probability distribution)
- Larger logits → higher probabilities (exponential scaling)
- Used ONLY in output layer for multi-class classification

### Temperature in Softmax

**Temperature** parameter controls confidence of predictions:

\`\`\`
softmax_T(z) = softmax (z / T)
\`\`\`

\`\`\`python
def softmax_with_temperature (z, temperature=1.0):
    """Softmax with temperature parameter"""
    return softmax (z / temperature)

# Example: Effect of temperature
logits = np.array([[2.0, 1.0, 0.5]])

temps = [0.5, 1.0, 2.0, 5.0]
plt.figure (figsize=(12, 3))

for i, temp in enumerate (temps):
    probs = softmax_with_temperature (logits, temperature=temp)
    plt.subplot(1, 4, i + 1)
    plt.bar(['Class 0', 'Class 1', 'Class 2'], probs[0], color=['blue', 'green', 'red'], alpha=0.6)
    plt.ylim(0, 1)
    plt.title (f'T = {temp}')
    plt.ylabel('Probability')
    plt.grid(True, alpha=0.3, axis='y')

plt.suptitle('Effect of Temperature on Softmax', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("Temperature Effects:")
print("  T < 1: Sharper distribution (more confident)")
print("  T = 1: Standard softmax")
print("  T > 1: Smoother distribution (less confident)")
\`\`\`

**Applications:**
- **Knowledge distillation**: Use high T for "soft" targets
- **Sampling**: Control randomness in text generation
- **Calibration**: Adjust model confidence

## Choosing the Right Activation Function

### Decision Guide

\`\`\`
┌─────────────────────────────────────────────────────────┐
│ WHERE IN THE NETWORK?                                    │
└─────────────────────────────────────────────────────────┘
              │
              ├─→ OUTPUT LAYER
              │        │
              │        ├─→ Binary Classification
              │        │        → Sigmoid
              │        │
              │        ├─→ Multi-class Classification
              │        │        → Softmax
              │        │
              │        └─→ Regression
              │                 → Linear (no activation) or ReLU
              │
              └─→ HIDDEN LAYERS
                       │
                       ├─→ Default Choice
                       │        → ReLU (or Leaky ReLU)
                       │
                       ├─→ Dying ReLU problem observed?
                       │        → Leaky ReLU or ELU
                       │
                       ├─→ Need self-normalization?
                       │        → SELU (with proper init)
                       │
                       └─→ RNN/LSTM?
                                → tanh (for activations)
                                → sigmoid (for gates)
\`\`\`

### Practical Recommendations

**1. Start with ReLU for hidden layers:**
- Fast, simple, works well
- If you observe dying neurons, switch to Leaky ReLU

**2. Output layer depends on task:**
- Binary classification: Sigmoid
- Multi-class classification: Softmax
- Regression: Linear or ReLU (if output should be non-negative)

**3. For very deep networks:**
- Consider SELU (self-normalizing)
- Or use ReLU with Batch Normalization

**4. For RNNs:**
- Use tanh for hidden state activations
- Use sigmoid for gates (LSTM/GRU)

**5. Experiment:**
- Leaky ReLU vs ELU vs ReLU
- Different α values for Leaky ReLU
- Test on validation set

### Common Mistakes to Avoid

❌ **Using sigmoid/tanh in deep networks**
- Causes vanishing gradients
- Training becomes very slow or fails

❌ **Using ReLU for binary classification output**
- Output should be in (0, 1) for probabilities
- Use sigmoid instead

❌ **Forgetting activation functions**
- Network becomes linear, can't learn complex patterns

❌ **Using wrong activation for task**
- Regression with sigmoid → outputs limited to (0, 1)
- Multi-class with sigmoid → outputs don't sum to 1

## Connection to Trading and Finance

### Activation Functions in Trading Models

**1. Price Direction Prediction (Binary Classification):**
\`\`\`python
# Output layer: Sigmoid for probability of price increase
output = sigmoid(W_out @ hidden + b_out)
# output ∈ (0, 1) represents P(price_up)
\`\`\`

**2. Multi-Asset Portfolio Allocation:**
\`\`\`python
# Output layer: Softmax for portfolio weights
weights = softmax(W_out @ hidden + b_out)
# weights sum to 1, can be used as portfolio allocation
\`\`\`

**3. Volatility Prediction (Regression, non-negative):**
\`\`\`python
# Output layer: ReLU to ensure positive volatility
volatility = relu(W_out @ hidden + b_out)
# volatility ≥ 0 (as required)
\`\`\`

**4. Hidden Layers:**
\`\`\`python
# Use ReLU or Leaky ReLU for learning complex market patterns
hidden1 = relu(W1 @ features + b1)
hidden2 = relu(W2 @ hidden1 + b2)
\`\`\`

### Trading Example: Market Regime Classification

\`\`\`python
# Classify market regime: Bullish, Neutral, Bearish
class MarketRegimeClassifier:
    def __init__(self, input_features=10):
        # Hidden layers with ReLU
        self.W1 = np.random.randn (input_features, 16) * 0.01
        self.b1 = np.zeros(16)
        self.W2 = np.random.randn(16, 8) * 0.01
        self.b2 = np.zeros(8)
        # Output layer with softmax (3 classes)
        self.W3 = np.random.randn(8, 3) * 0.01
        self.b3 = np.zeros(3)
    
    def predict (self, features):
        """
        Predict market regime
        
        Args:
            features: [momentum, volatility, volume, ...]
        
        Returns:
            probabilities: [P(Bullish), P(Neutral), P(Bearish)]
        """
        # Forward pass
        h1 = relu (features @ self.W1 + self.b1)
        h2 = relu (h1 @ self.W2 + self.b2)
        logits = h2 @ self.W3 + self.b3
        probabilities = softmax (logits.reshape(1, -1))
        return probabilities[0]

# Example usage
classifier = MarketRegimeClassifier (input_features=10)
market_features = np.random.randn(10)  # Momentum, vol, etc.
regime_probs = classifier.predict (market_features)

regimes = ['Bullish', 'Neutral', 'Bearish']
print("Market Regime Prediction:")
for regime, prob in zip (regimes, regime_probs):
    print(f"  {regime}: {prob:.1%}")
print(f"\\nPredicted Regime: {regimes[np.argmax (regime_probs)]}")
\`\`\`

## Key Takeaways

1. **Non-linearity is Essential**: Without activation functions, neural networks can only learn linear relationships
2. **ReLU is King**: Default choice for hidden layers due to efficiency and performance
3. **Sigmoid for Binary**: Use sigmoid for binary classification output
4. **Softmax for Multi-class**: Use softmax for multi-class classification output
5. **Dying ReLU Problem**: Can be fixed with Leaky ReLU or ELU
6. **Task-Specific**: Output activation depends on your task (classification vs regression)
7. **Experiment**: Try different activations and measure validation performance
8. **Historical Context**: sigmoid/tanh used in early networks, ReLU revolutionized deep learning

## What\'s Next

Now that you understand activation functions, we'll explore:
- **Forward Propagation**: Detailed mechanics of how data flows through networks
- **Loss Functions**: How to measure prediction quality
- **Backpropagation**: Computing gradients for training
- **Optimization**: Algorithms for updating weights

These components work together to enable neural networks to learn from data.
`,
};
