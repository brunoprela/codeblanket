/**
 * Section: Optimization Algorithms
 * Module: Deep Learning Fundamentals
 *
 * Covers SGD, Momentum, RMSprop, Adam, AdamW, learning rate scheduling,
 * and choosing the right optimizer for your problem
 */

export const optimizationAlgorithmsSection = {
  id: 'optimization-algorithms',
  title: 'Optimization Algorithms',
  content: `
# Optimization Algorithms

## Introduction

Backpropagation computes gradients, but **optimization algorithms** determine how we use those gradients to update parameters. The choice of optimizer profoundly affects training speed, convergence, and final model performance.

**What You'll Learn:**
- Gradient Descent and its variants (SGD, Mini-batch)
- Momentum-based methods (Momentum, Nesterov)
- Adaptive learning rate methods (AdaGrad, RMSprop, Adam)
- AdamW and weight decay
- Learning rate scheduling
- Choosing the right optimizer

## Gradient Descent: The Foundation

### Mathematical Formulation

Given loss function L(θ), we want to find θ* that minimizes L:

\`\`\`
θ_{t+1} = θ_t - η ∇_θ L(θ_t)
\`\`\`

Where:
- θ = parameters (weights and biases)
- η = learning rate (step size)
- ∇_θ L = gradient of loss w.r.t. parameters

### Three Variants

**1. Batch Gradient Descent:**
\`\`\`
θ_{t+1} = θ_t - η ∇_θ (1/N) Σᵢ L(θ_t, xᵢ, yᵢ)
\`\`\`
- Uses entire dataset per update
- Accurate gradient but slow
- Deterministic

**2. Stochastic Gradient Descent (SGD):**
\`\`\`
θ_{t+1} = θ_t - η ∇_θ L(θ_t, xᵢ, yᵢ)
\`\`\`
- Uses single sample per update
- Fast but noisy gradient
- Stochastic, can escape local minima

**3. Mini-batch Gradient Descent:**
\`\`\`
θ_{t+1} = θ_t - η ∇_θ (1/B) Σᵢ∈batch L(θ_t, xᵢ, yᵢ)
\`\`\`
- Uses batch of B samples per update
- Balance between accuracy and speed
- Standard in practice (B=32, 64, 128)

### Implementation

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt

class GradientDescent:
    """Vanilla gradient descent optimizer"""
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def update (self, params, gradients):
        """
        Update parameters using gradients
        
        Args:
            params: Dictionary of parameters
            gradients: Dictionary of gradients (same keys as params)
        """
        for key in params:
            params[key] -= self.learning_rate * gradients[key]
        
        return params

# Example: Optimize simple function
def rosenbrock (x, y):
    """Rosenbrock function (hard to optimize)"""
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_gradient (x, y):
    """Gradient of Rosenbrock function"""
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return np.array([dx, dy])

# Optimization
np.random.seed(42)
params = {'xy': np.array([-1.0, -1.0])}  # Starting point
optimizer = GradientDescent (learning_rate=0.001)

history = [params['xy'].copy()]
for i in range(1000):
    x, y = params['xy']
    grad = rosenbrock_gradient (x, y)
    params = optimizer.update (params, {'xy': grad})
    history.append (params['xy'].copy())

history = np.array (history)

print(f"Final position: {params['xy']}")
print(f"Optimal: [1.0, 1.0]")
print(f"Final loss: {rosenbrock(*params['xy']):.6f}")
\`\`\`

## Momentum-Based Methods

### Problem with Vanilla SGD

SGD oscillates in ravines (steep in one direction, shallow in another):
- Slow progress toward minimum
- Oscillations perpendicular to minimum

### Momentum

Adds "velocity" term to accumulate gradients:

\`\`\`
v_t = β v_{t-1} + ∇_θ L(θ_t)
θ_{t+1} = θ_t - η v_t
\`\`\`

Where β ∈ [0, 1] is momentum coefficient (typically 0.9).

**Benefits:**
- Dampens oscillations
- Accelerates in consistent directions
- Faster convergence

\`\`\`python
class MomentumOptimizer:
    """SGD with Momentum"""
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = {}
    
    def update (self, params, gradients):
        # Initialize velocity on first call
        if not self.velocity:
            for key in params:
                self.velocity[key] = np.zeros_like (params[key])
        
        # Update velocity and parameters
        for key in params:
            self.velocity[key] = (self.momentum * self.velocity[key] + 
                                  gradients[key])
            params[key] -= self.learning_rate * self.velocity[key]
        
        return params

# Compare with vanilla GD
momentum_opt = MomentumOptimizer (learning_rate=0.001, momentum=0.9)
params_momentum = {'xy': np.array([-1.0, -1.0])}

history_momentum = [params_momentum['xy'].copy()]
for i in range(500):  # Fewer iterations needed!
    x, y = params_momentum['xy']
    grad = rosenbrock_gradient (x, y)
    params_momentum = momentum_opt.update (params_momentum, {'xy': grad})
    history_momentum.append (params_momentum['xy'].copy())

history_momentum = np.array (history_momentum)

print("\\nMomentum vs Vanilla GD:")
print(f"  Vanilla (1000 iters): loss = {rosenbrock(*history[-1]):.6f}")
print(f"  Momentum (500 iters): loss = {rosenbrock(*history_momentum[-1]):.6f}")
print("  → Momentum converges faster!")
\`\`\`

### Nesterov Accelerated Gradient (NAG)

"Look ahead" variant of momentum:

\`\`\`
θ_lookahead = θ_t - β v_{t-1}
v_t = β v_{t-1} + ∇_θ L(θ_lookahead)
θ_{t+1} = θ_t - η v_t
\`\`\`

Evaluates gradient at the "lookahead" position, often converges faster.

## Adaptive Learning Rate Methods

### Problem with Fixed Learning Rate

- Too large: Overshooting, divergence
- Too small: Slow convergence
- Different parameters need different learning rates

### AdaGrad

Adapts learning rate per parameter based on historical gradients:

\`\`\`
g_t = ∇_θ L(θ_t)
G_t = G_{t-1} + g_t ⊙ g_t  (accumulate squared gradients)
θ_{t+1} = θ_t - (η / √(G_t + ε)) ⊙ g_t
\`\`\`

**Benefit**: Parameters with large gradients get smaller learning rates
**Problem**: Learning rate decays too aggressively (√G_t grows unbounded)

### RMSprop

Fixes AdaGrad by using exponential moving average:

\`\`\`
g_t = ∇_θ L(θ_t)
E[g²]_t = β E[g²]_{t-1} + (1 - β) g_t²
θ_{t+1} = θ_t - (η / √(E[g²]_t + ε)) ⊙ g_t
\`\`\`

**Benefit**: Doesn't accumulate indefinitely, maintains adaptive learning rates

\`\`\`python
class RMSpropOptimizer:
    """RMSprop optimizer"""
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.squared_gradient = {}
    
    def update (self, params, gradients):
        if not self.squared_gradient:
            for key in params:
                self.squared_gradient[key] = np.zeros_like (params[key])
        
        for key in params:
            # Update squared gradient moving average
            self.squared_gradient[key] = (
                self.beta * self.squared_gradient[key] + 
                (1 - self.beta) * gradients[key]**2
            )
            
            # Adaptive learning rate update
            params[key] -= (self.learning_rate * gradients[key] / 
                           (np.sqrt (self.squared_gradient[key]) + self.epsilon))
        
        return params
\`\`\`

## Adam Optimizer

### The Most Popular Optimizer

Adam (Adaptive Moment Estimation) combines momentum and RMSprop:

\`\`\`
g_t = ∇_θ L(θ_t)
m_t = β₁ m_{t-1} + (1 - β₁) g_t          (momentum)
v_t = β₂ v_{t-1} + (1 - β₂) g_t²         (RMSprop)

# Bias correction (important for early iterations)
m̂_t = m_t / (1 - β₁^t)
v̂_t = v_t / (1 - β₂^t)

θ_{t+1} = θ_t - η m̂_t / (√v̂_t + ε)
\`\`\`

**Default hyperparameters:**
- β₁ = 0.9 (momentum)
- β₂ = 0.999 (RMSprop)
- η = 0.001 (learning rate)
- ε = 1e-8 (numerical stability)

\`\`\`python
class AdamOptimizer:
    """Adam optimizer (most commonly used)"""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment (momentum)
        self.v = {}  # Second moment (RMSprop)
        self.t = 0   # Time step
    
    def update (self, params, gradients):
        if not self.m:
            for key in params:
                self.m[key] = np.zeros_like (params[key])
                self.v[key] = np.zeros_like (params[key])
        
        self.t += 1
        
        for key in params:
            # Update biased first moment
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * gradients[key]
            
            # Update biased second moment
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * gradients[key]**2
            
            # Bias correction
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)
            
            # Update parameters
            params[key] -= self.learning_rate * m_hat / (np.sqrt (v_hat) + self.epsilon)
        
        return params

# Test Adam
adam_opt = AdamOptimizer (learning_rate=0.01)
params_adam = {'xy': np.array([-1.0, -1.0])}

history_adam = [params_adam['xy'].copy()]
for i in range(200):  # Even fewer iterations!
    x, y = params_adam['xy']
    grad = rosenbrock_gradient (x, y)
    params_adam = adam_opt.update (params_adam, {'xy': grad})
    history_adam.append (params_adam['xy'].copy())

history_adam = np.array (history_adam)

print("\\nOptimizer Comparison:")
print(f"  Vanilla GD (1000 iters):  {rosenbrock(*history[-1]):.6f}")
print(f"  Momentum (500 iters):     {rosenbrock(*history_momentum[-1]):.6f}")
print(f"  Adam (200 iters):         {rosenbrock(*history_adam[-1]):.6f}")
print("  → Adam converges fastest!")
\`\`\`

### AdamW: Adam with Weight Decay

Proper implementation of L2 regularization with Adam:

\`\`\`
# Standard Adam with L2: INCORRECT
g_t = ∇_θ L(θ_t) + λθ_t  # Add L2 to gradient
# Then apply Adam...

# AdamW: CORRECT
# Apply Adam update as usual
# Then apply weight decay separately
θ_{t+1} = θ_{t+1} - η λ θ_t
\`\`\`

**Why separate?** Adam\'s adaptive learning rates interfere with L2 when added to gradients. AdamW decouples them.

\`\`\`python
class AdamWOptimizer(AdamOptimizer):
    """Adam with decoupled weight decay (AdamW)"""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, 
                 epsilon=1e-8, weight_decay=0.01):
        super().__init__(learning_rate, beta1, beta2, epsilon)
        self.weight_decay = weight_decay
    
    def update (self, params, gradients):
        # Apply Adam update
        params = super().update (params, gradients)
        
        # Apply weight decay separately
        for key in params:
            params[key] -= self.learning_rate * self.weight_decay * params[key]
        
        return params
\`\`\`

## Learning Rate Scheduling

### Why Schedule Learning Rate?

- **Early training**: Large LR for fast progress
- **Late training**: Small LR for fine-tuning
- **Escape plateaus**: Occasional LR increases

### Common Schedules

**1. Step Decay:**
\`\`\`
η_t = η_0 × γ^⌊t/k⌋

E.g., multiply by 0.1 every 30 epochs
\`\`\`

**2. Exponential Decay:**
\`\`\`
η_t = η_0 × e^(-kt)
\`\`\`

**3. Cosine Annealing:**
\`\`\`
η_t = η_min + (η_max - η_min) × (1 + cos(πt/T)) / 2
\`\`\`

**4. Warm Restart (SGDR):**
- Periodically reset to high LR
- Helps escape local minima

\`\`\`python
class LearningRateScheduler:
    """Various LR scheduling strategies"""
    
    @staticmethod
    def step_decay (initial_lr, epoch, step_size=30, gamma=0.1):
        """Multiply LR by gamma every step_size epochs"""
        return initial_lr * (gamma ** (epoch // step_size))
    
    @staticmethod
    def exponential_decay (initial_lr, epoch, decay_rate=0.95):
        """Exponential decay"""
        return initial_lr * (decay_rate ** epoch)
    
    @staticmethod
    def cosine_annealing (initial_lr, epoch, T_max, eta_min=0):
        """Cosine annealing"""
        return eta_min + (initial_lr - eta_min) * (
            1 + np.cos (np.pi * epoch / T_max)
        ) / 2
    
    @staticmethod
    def linear_warmup (initial_lr, epoch, warmup_epochs):
        """Linear warmup (0 to initial_lr)"""
        if epoch < warmup_epochs:
            return initial_lr * (epoch / warmup_epochs)
        return initial_lr

# Visualize schedules
epochs = np.arange(0, 100)
initial_lr = 0.1

plt.figure (figsize=(14, 4))

plt.subplot(1, 3, 1)
lrs_step = [LearningRateScheduler.step_decay (initial_lr, e, step_size=20) for e in epochs]
plt.plot (epochs, lrs_step, linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Step Decay')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
lrs_exp = [LearningRateScheduler.exponential_decay (initial_lr, e, decay_rate=0.95) for e in epochs]
plt.plot (epochs, lrs_exp, linewidth=2, color='orange')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Exponential Decay')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
lrs_cos = [LearningRateScheduler.cosine_annealing (initial_lr, e, T_max=100) for e in epochs]
plt.plot (epochs, lrs_cos, linewidth=2, color='green')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Cosine Annealing')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
\`\`\`

## Choosing the Right Optimizer

### Decision Guide

\`\`\`
START HERE: Adam (most reliable default)
│
├─→ Working well? ✓ Done!
│
├─→ Overfitting?
│   └─→ Try AdamW (better regularization)
│
├─→ Need faster convergence?
│   └─→ Increase learning rate or use LR warmup
│
├─→ Unstable training?
│   └─→ Decrease learning rate or use gradient clipping
│
└─→ For specific cases:
    ├─→ Very deep networks: AdamW + LR warmup
    ├─→ Sparse data (NLP): Adam or AdamW
    ├─→ Small datasets: SGD with momentum
    └─→ Computer vision: SGD with momentum or AdamW
\`\`\`

### Comparison Table

| Optimizer | Speed | Memory | Robustness | Best For |
|-----------|-------|--------|------------|----------|
| SGD | Slow | Low | Medium | Small data, CV |
| SGD+Momentum | Medium | Low | Good | CV, proven |
| RMSprop | Fast | Medium | Good | RNNs |
| Adam | Fast | Medium | Excellent | Default choice |
| AdamW | Fast | Medium | Excellent | Fine-tuning, large models |

### Trading Model Recommendations

**For Daily Trading Models:**
\`\`\`python
# Default: Adam with moderate LR
optimizer = AdamOptimizer(
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999
)

# With L2 regularization: AdamW
optimizer = AdamWOptimizer(
    learning_rate=0.001,
    weight_decay=0.01  # Prevents overfitting
)
\`\`\`

**For Intraday/HFT Models:**
\`\`\`python
# Need fast adaptation: Higher LR + warmup
optimizer = AdamOptimizer (learning_rate=0.01)
# With warmup for stability
\`\`\`

## Key Takeaways

1. **Adam is the default** - works well in most cases
2. **AdamW for regularization** - better than L2 with Adam
3. **Learning rate is critical** - most important hyperparameter
4. **Use scheduling** - start high, end low
5. **Momentum helps** - dampens oscillations
6. **Adaptive methods converge faster** - but may overfit
7. **Monitor training** - adjust based on loss curves
8. **For trading**: AdamW with conservative LR (0.001)

## What's Next

Optimization gets parameters moving in the right direction. But starting from the right place matters too:
- **Weight Initialization**: He, Xavier, and why it matters
- Preventing vanishing/exploding gradients
- Layer-specific initialization strategies
`,
};
