/**
 * Quiz questions for Derivatives Fundamentals section
 */

export const derivativesfundamentalsQuiz = [
  {
    id: 'deriv-disc-1',
    question:
      'Explain why ReLU has become the dominant activation function in deep learning despite not being differentiable at x = 0. How do practitioners handle this non-differentiability?',
    hint: 'Consider computational efficiency, vanishing gradients, and how automatic differentiation libraries handle the point x = 0.',
    sampleAnswer: `ReLU (Rectified Linear Unit) has become the default activation function despite its non-differentiability at x = 0 for several compelling reasons:

**1. Computational Efficiency:**
ReLU is simply max(0, x), requiring only a threshold operation. Compare this to sigmoid or tanh which require expensive exponential computations. In deep networks with millions of activations, this difference is substantial:
- Forward pass: O(1) comparison vs O(1) exp()
- Backward pass: Simple 0/1 assignment vs complex exponential derivatives

**2. Solves Vanishing Gradient Problem:**
Unlike sigmoid (σ'(x) ∈ (0, 0.25]) or tanh (tanh'(x) ∈ (0, 1]), ReLU has:
- Derivative = 1 for x > 0 (no attenuation)
- Derivative = 0 for x < 0 (sparse gradients)

In deep networks where gradients are products of many derivatives, ReLU's gradient of 1 prevents vanishing, enabling training of much deeper networks.

**3. Handling Non-Differentiability at x = 0:**
In practice, the non-differentiability at exactly x = 0 is handled pragmatically:

a) **Probability argument**: The probability that any activation is exactly 0.0 (not ≈0, but exactly 0) is measure zero - effectively impossible with floating-point arithmetic.

b) **Subgradient convention**: Automatic differentiation libraries define:
   ReLU'(0) = 0 or ReLU'(0) = 1 (by convention)
   
   Either choice works because we'll never actually hit exactly 0.

c) **Implementation**: Libraries use:
   \\\`\\\`\\\`python
   def relu_backward(x):
       return (x > 0).astype(float)  # Returns 0 for x ≤ 0, 1 for x > 0
   \\\`\\\`\\\`

**4. Sparse Activations:**
The zero gradient for x < 0 creates sparse representations - only a subset of neurons activate for any input. This:
- Improves computational efficiency
- Provides a form of implicit regularization
- Creates more discriminative features

**5. Trade-offs:**
The "dying ReLU" problem occurs when neurons get stuck with x < 0 for all inputs, permanently outputting zero. Solutions:
- Careful initialization (He initialization)
- Learning rate tuning
- Variants like Leaky ReLU: max(0.01x, x)

**Practical Reality:**
The non-differentiability at a single point doesn't matter in practice. What matters is:
- Fast computation
- Good gradient flow (derivative = 1 for active neurons)
- Simple implementation
- Empirical success across countless architectures

This is a perfect example of where mathematical purity (everywhere differentiable) yields to practical effectiveness (better training dynamics, computational efficiency, and empirical results).`,
    keyPoints: [
      'ReLU is computationally efficient compared to sigmoid/tanh',
      'Derivative of 1 prevents vanishing gradients in deep networks',
      'Non-differentiability at x=0 handled by convention (measure zero event)',
      'Sparse activations provide implicit regularization',
      'Practical effectiveness outweighs mathematical concerns',
    ],
  },
  {
    id: 'deriv-disc-2',
    question:
      'In machine learning, we often compute derivatives numerically during debugging (gradient checking). Explain the trade-offs between accuracy and computational cost, and describe how to implement gradient checking effectively.',
    hint: 'Consider forward vs central differences, choice of h, computational complexity, and when gradient checking is necessary.',
    sampleAnswer: `Gradient checking is a critical debugging tool for verifying backpropagation implementation. Understanding the trade-offs is essential for effective use:

**Methods and Their Trade-offs:**

**1. Forward Difference:**
   Formula: (f(θ + h) - f(θ)) / h
   - Cost: 1 function evaluation per parameter
   - Error: O(h) truncation error
   - Use case: Quick, rough checks

**2. Central Difference:**
   Formula: (f(θ + h) - f(θ - h)) / (2h)
   - Cost: 2 function evaluations per parameter
   - Error: O(h²) truncation error  
   - Use case: Accurate gradient verification

**Implementation Strategy:**

\\\`\\\`\\\`python
def gradient_check(f, x, analytical_grad, epsilon=1e-5):
    """
    Verify analytical gradients against numerical approximation
    """
    numerical_grad = np.zeros_like(x)
    
    # Compute numerical gradient for each parameter
    for i in range(x.size):
        # Store original value
        original = x.flat[i]
        
        # Compute f(x + h)
        x.flat[i] = original + epsilon
        f_plus = f(x)
        
        # Compute f(x - h)
        x.flat[i] = original - epsilon
        f_minus = f(x)
        
        # Central difference
        numerical_grad.flat[i] = (f_plus - f_minus) / (2 * epsilon)
        
        # Restore original
        x.flat[i] = original
    
    # Compute relative error
    numerator = np.linalg.norm(numerical_grad - analytical_grad)
    denominator = np.linalg.norm(numerical_grad) + np.linalg.norm(analytical_grad)
    relative_error = numerator / denominator
    
    return numerical_grad, relative_error
\\\`\\\`\\\`

**Choosing h (epsilon):**

The optimal h balances two error sources:
1. **Truncation error**: Decreases as h → 0 (approximation error)
2. **Round-off error**: Increases as h → 0 (floating-point precision)

**Optimal h ≈ ∛ε for forward difference, √ε for central difference**
Where ε ≈ 2.2×10⁻¹⁶ (machine epsilon for float64)

Practical choice: **h = 1e-5 or 1e-7** for central difference

**Computational Cost Considerations:**

For a model with n parameters:
- Forward pass: O(1) computation
- Backward pass: O(1) computation  
- Gradient checking: O(n) function evaluations

Example: Neural network with 1M parameters
- Backprop: ~2x forward pass cost
- Gradient check: ~2M function evaluations (≈1M× slower)

**Practical Guidelines:**

1. **When to use:**
   - Implementing new layer types
   - Debugging strange training behavior
   - After major architecture changes
   - NOT during regular training (too expensive)

2. **Sampling strategy:**
   \\\`\\\`\\\`python
   # Don't check all parameters - sample instead
   n_samples = min(100, n_parameters)
   indices = np.random.choice(n_parameters, n_samples, replace=False)
   # Check only sampled parameters
   \\\`\\\`\\\`

3. **Tolerance thresholds:**
   - Relative error < 1e-7: Excellent (backprop likely correct)
   - Relative error < 1e-5: Good
   - Relative error < 1e-3: Suspicious (possible bug)
   - Relative error > 1e-3: Likely bug

4. **Special cases requiring care:**
   - Non-differentiable functions (ReLU at 0): Use subgradients
   - Batch normalization: Check in eval mode
   - Dropout: Disable during gradient check
   - RNNs: Check unrolled computational graph

**Advanced: Two-sided error bounds:**

\\\`\\\`\\\`python
# Check both forward and central difference
forward_error = abs(forward_diff - analytical)
central_error = abs(central_diff - analytical)

if forward_error < 1e-7 and central_error < 1e-9:
    print("Gradient implementation verified ✓")
elif central_error < 1e-5:
    print("Gradient likely correct (within numerical precision)")
else:
    print("Gradient BUG detected!")
\\\`\\\`\\\`

**Real-World Application:**

Modern frameworks (PyTorch, TensorFlow) have built-in gradient checking:
\\\`\\\`\\\`python
import torch
from torch.autograd import gradcheck

# PyTorch automatic gradient checking
inputs = torch.randn(20, 20, dtype=torch.double, requires_grad=True)
test = gradcheck(my_function, inputs, eps=1e-6)
print(f"Gradients correct: {test}")
\\\`\\\`\\\`

**Summary:**
Gradient checking is essential for correctness but too expensive for training. Use central difference with h ≈ 1e-5 to 1e-7, sample parameters randomly, and only check during development/debugging phases.`,
    keyPoints: [
      'Central difference O(h²) more accurate than forward difference O(h)',
      'Optimal h balances truncation and round-off errors (≈1e-5 to 1e-7)',
      'Computational cost O(n) makes it unsuitable for training',
      'Sample parameters randomly for large models',
      'Use relative error for tolerance checking',
      'Essential for debugging but disable during training',
    ],
  },
  {
    id: 'deriv-disc-3',
    question:
      "Explain how derivatives enable gradient descent to find optimal parameters. Why is the learning rate critical, and what happens when it's too large or too small?",
    hint: 'Discuss the update rule θ_new = θ_old - α·∇L, convergence conditions, and the relationship between derivatives, step size, and loss landscape.',
    sampleAnswer: `Derivatives are the foundation of gradient descent, the workhorse optimization algorithm in machine learning. Understanding this relationship is crucial for effective model training:

**How Derivatives Enable Optimization:**

**1. The Gradient Descent Update Rule:**
   θ_new = θ_old - α · ∇L(θ_old)
   
   Where:
   - θ: Model parameters (weights, biases)
   - α: Learning rate (step size)
   - ∇L(θ): Gradient of loss with respect to parameters
   - The gradient ∇L points in direction of steepest increase
   - Negative gradient (-∇L) points toward steepest decrease

**2. Intuition from Calculus:**
   
   The derivative tells us the local slope. For a 1D function:
   - If f'(θ) > 0: Function increasing → move left (decrease θ)
   - If f'(θ) < 0: Function decreasing → move right (increase θ)
   - If f'(θ) = 0: At critical point (local min/max/saddle)

   The update θ_new = θ_old - α·f'(θ_old) automatically moves toward a minimum.

**The Learning Rate: Goldilocks Problem**

**Learning Rate Too Small (α → 0):**

\\\`\\\`\\\`python
# Example: α = 0.001 (too small)
# Problem: Slow convergence
θ = 10.0  # Start far from minimum
α = 0.001
iterations = 0

while abs(θ) > 0.01:  # Get close to minimum
    gradient = 2*θ  # Derivative of θ²
    θ = θ - α * gradient
    iterations += 1

print(f"Iterations needed: {iterations}")  # Thousands!
\\\`\\\`\\\`

Consequences:
- Takes many iterations to converge
- Training time becomes prohibitive
- May get stuck in plateaus (very small gradients)
- Wall-clock time: Days instead of hours

**Learning Rate Too Large (α → ∞):**

\\\`\\\`\\\`python
# Example: α = 1.5 (too large)
# Problem: Overshooting
θ = 1.0
α = 1.5
history = [θ]

for i in range(20):
    gradient = 2*θ
    θ = θ - α * gradient
    history.append(θ)
    if abs(θ) > 100:
        print(f"Diverged at iteration {i}!")
        break

# θ oscillates wildly: 1, -2, 4, -8, 16, ...
\\\`\\\`\\\`

Consequences:
- Overshoots the minimum
- Oscillates back and forth
- May diverge to infinity
- Loss increases instead of decreases
- Training becomes unstable

**Mathematical Analysis:**

For convex quadratic loss L(θ) = ½θ²:
- Gradient: ∇L = θ
- Update: θ_{t+1} = θ_t - α·θ_t = (1-α)θ_t

**Convergence condition:** |1 - α| < 1
- This gives: 0 < α < 2
- Optimal α = 1 (reaches minimum in one step)
- If α > 2: Diverges

For general functions with Lipschitz-continuous gradients:
- Maximum stable α ≈ 1/L (L = Lipschitz constant)
- Roughly: α < 2/(largest eigenvalue of Hessian)

**Practical Learning Rate Strategies:**

**1. Learning Rate Schedules:**

\\\`\\\`\\\`python
# Start large, decay over time
def learning_rate_schedule(epoch, initial_lr=0.1):
    # Step decay
    if epoch < 30:
        return initial_lr
    elif epoch < 60:
        return initial_lr * 0.1
    else:
        return initial_lr * 0.01

# Or exponential decay
def exp_decay(epoch, initial_lr=0.1, decay_rate=0.95):
    return initial_lr * (decay_rate ** epoch)

# Or cosine annealing
def cosine_anneal(epoch, initial_lr=0.1, total_epochs=100):
    return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))
\\\`\\\`\\\`

**2. Adaptive Learning Rates (Adam, RMSprop):**

These algorithms automatically adjust α per parameter based on gradient history:

\\\`\\\`\\\`python
# Simplified Adam
m = 0  # First moment (momentum)
v = 0  # Second moment (variance)
β1, β2 = 0.9, 0.999
ε = 1e-8

for t in range(1, num_iterations):
    g = compute_gradient()
    
    m = β1*m + (1-β1)*g        # Exponential moving average
    v = β2*v + (1-β2)*(g**2)   # Exponential moving average of squared gradient
    
    m_hat = m / (1 - β1**t)    # Bias correction
    v_hat = v / (1 - β2**t)
    
    θ = θ - α * m_hat / (√v_hat + ε)  # Adaptive step
\\\`\\\`\\\`

Adam effectively uses:
- Larger steps when gradients are consistent
- Smaller steps when gradients are noisy
- Different rates for different parameters

**3. Learning Rate Warmup:**

\\\`\\\`\\\`python
# Start with small α, increase gradually
def warmup_schedule(epoch, target_lr=0.1, warmup_epochs=5):
    if epoch < warmup_epochs:
        return target_lr * (epoch + 1) / warmup_epochs
    return target_lr
\\\`\\\`\\\`

Useful for:
- Large batch training
- Training from scratch (vs fine-tuning)
- Avoiding early instability

**4. Learning Rate Finder:**

\\\`\\\`\\\`python
# Empirically find good learning rate
def find_learning_rate(model, train_loader, start_lr=1e-7, end_lr=10):
    lrs, losses = [], []
    lr = start_lr
    
    for batch in train_loader:
        loss = train_step(model, batch, lr)
        losses.append(loss)
        lrs.append(lr)
        
        # Exponentially increase lr
        lr *= 1.1
        
        if lr > end_lr or loss > 4 * min(losses):
            break
    
    # Plot and choose lr where loss decreases fastest
    # Typically: 1/10 of lr at minimum loss
    return lrs, losses
\\\`\\\`\\\`

**Visual Understanding:**

Imagine rolling a ball down a hill (loss landscape):
- **Small α**: Baby steps, very slow descent
- **Just right α**: Efficient path to bottom
- **Large α**: Steps so large you jump over the valley repeatedly

**Loss Landscape Visualization:**

\\\`\\\`\\\`python
# Different learning rates on same problem
θ_history_small = gradient_descent(f, initial_θ, α=0.01, iterations=100)
θ_history_good = gradient_descent(f, initial_θ, α=0.1, iterations=100)
θ_history_large = gradient_descent(f, initial_θ, α=0.9, iterations=100)

# Plot trajectories
plt.plot(θ_history_small, label='α=0.01 (too small)')
plt.plot(θ_history_good, label='α=0.1 (good)')  
plt.plot(θ_history_large, label='α=0.9 (too large)')
plt.legend()
# Shows: Small α crawls, good α converges smoothly, large α oscillates
\\\`\\\`\\\`

**Summary:**
Derivatives provide the direction to move (downhill), while the learning rate controls step size. Too small → slow convergence, too large → instability/divergence. Modern practice uses adaptive methods (Adam) with learning rate schedules and warmup for robust training across diverse architectures and datasets.`,
    keyPoints: [
      'Gradient points to steepest increase; negative gradient to steepest decrease',
      'Learning rate controls step size in parameter space',
      'Too small: slow convergence, too large: oscillation/divergence',
      'Convergence requires α < 2/L (L = Lipschitz constant)',
      'Modern methods use adaptive rates (Adam) and schedules',
      'Learning rate warmup prevents early training instability',
      'Derivatives provide direction, α provides magnitude',
    ],
  },
];
