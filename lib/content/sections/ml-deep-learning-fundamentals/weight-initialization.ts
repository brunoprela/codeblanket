/**
 * Section: Weight Initialization
 * Module: Deep Learning Fundamentals
 *
 * Covers why initialization matters, Xavier/Glorot initialization, He initialization,
 * preventing vanishing/exploding gradients, and layer-specific strategies
 */

export const weightInitializationSection = {
  id: 'weight-initialization',
  title: 'Weight Initialization',
  content: `
# Weight Initialization

## Introduction

**Weight initialization** determines the starting point for training. Poor initialization can cause:
- Vanishing gradients (training stalls)
- Exploding gradients (training diverges)
- Slow convergence
- Poor final performance

Good initialization accelerates training and often improves final accuracy.

**What You'll Learn:**
- Why initialization matters
- Zero, random, and constant initialization
- Xavier/Glorot initialization (for sigmoid/tanh)
- He initialization (for ReLU)
- Layer-specific strategies
- Modern best practices

## Why Initialization Matters

### Problem 1: All Zeros

\`\`\`python
# Bad: Initialize all weights to zero
W = np.zeros((n_in, n_out))
b = np.zeros((1, n_out))
\`\`\`

**Issue**: All neurons compute the same output and receive the same gradient. Network cannot break symmetry—equivalent to single neuron per layer!

### Problem 2: Too Large

\`\`\`python
# Bad: Large random weights
W = np.random.randn (n_in, n_out) * 10  # Too large!
\`\`\`

**Issue**: Activations explode, gradients explode, training diverges.

### Problem 3: Too Small

\`\`\`python
# Bad: Tiny random weights
W = np.random.randn (n_in, n_out) * 0.0001  # Too small!
\`\`\`

**Issue**: Activations vanish, gradients vanish, training stalls.

## Xavier/Glorot Initialization

### For Sigmoid and Tanh Activations

**Goal**: Keep variance of activations and gradients consistent across layers.

**Formula**:
\`\`\`
W ~ Uniform(-√(6/(n_in + n_out)), √(6/(n_in + n_out)))

Or: W ~ Normal(0, √(2/(n_in + n_out)))
\`\`\`

\`\`\`python
def xavier_init (n_in, n_out):
    """Xavier/Glorot initialization"""
    limit = np.sqrt(6 / (n_in + n_out))
    return np.random.uniform(-limit, limit, (n_in, n_out))

# Example
W1 = xavier_init(784, 256)
print(f"Xavier init: std = {np.std(W1):.4f}")
\`\`\`

**Why it works**: Maintains variance ≈ 1 through forward and backward passes for sigmoid/tanh.

## He Initialization  

### For ReLU Activations

ReLU zeros out negative activations, effectively halving variance. Compensate by scaling up:

**Formula**:
\`\`\`
W ~ Normal(0, √(2/n_in))
\`\`\`

\`\`\`python
def he_init (n_in, n_out):
    """He initialization for ReLU"""
    return np.random.randn (n_in, n_out) * np.sqrt(2.0 / n_in)

W_relu = he_init(784, 256)
print(f"He init: std = {np.std(W_relu):.4f}")
\`\`\`

**Rule of thumb**: 
- sigmoid/tanh → Xavier
- ReLU/Leaky ReLU → He

## Comparison of Initialization Methods

\`\`\`python
import matplotlib.pyplot as plt

def test_initialization (init_method, activation, n_layers=10, n_neurons=100):
    """Test activation statistics through deep network"""
    activations = []
    x = np.random.randn(1000, n_neurons)
    
    for layer in range (n_layers):
        if init_method == 'zero':
            W = np.zeros((n_neurons, n_neurons))
        elif init_method == 'too_large':
            W = np.random.randn (n_neurons, n_neurons) * 5
        elif init_method == 'too_small':
            W = np.random.randn (n_neurons, n_neurons) * 0.01
        elif init_method == 'xavier':
            W = xavier_init (n_neurons, n_neurons)
        elif init_method == 'he':
            W = he_init (n_neurons, n_neurons)
        
        z = x @ W
        
        if activation == 'relu':
            x = np.maximum(0, z)
        elif activation == 'tanh':
            x = np.tanh (z)
        
        activations.append (x)
    
    return activations

# Compare methods
methods = ['too_small', 'xavier', 'he']
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, method in enumerate (methods):
    acts = test_initialization (method, 'relu', n_layers=10)
    
    means = [np.mean (a) for a in acts]
    stds = [np.std (a) for a in acts]
    
    axes[idx].plot (means, label='Mean', linewidth=2)
    axes[idx].plot (stds, label='Std', linewidth=2)
    axes[idx].set_xlabel('Layer')
    axes[idx].set_ylabel('Value')
    axes[idx].set_title (f'{method.title()} Init')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("He initialization maintains stable activations through deep networks!")
\`\`\`

## Key Takeaways

1. **Never initialize to zero** - breaks symmetry
2. **ReLU → He initialization** (√(2/n_in))
3. **Sigmoid/tanh → Xavier** (√(2/(n_in+n_out)))
4. **Proper init accelerates training** 10-100x
5. **Modern default: He for ReLU networks**

## What's Next

Initialization starts training well, but **regularization** keeps it on track:
- Dropout, Batch Normalization, L1/L2 
- Preventing overfitting
- Production-ready models
`,
};
