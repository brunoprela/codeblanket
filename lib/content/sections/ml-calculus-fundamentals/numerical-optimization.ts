/**
 * Numerical Optimization Methods Section
 */

export const numericaloptimizationSection = {
  id: 'numerical-optimization',
  title: 'Numerical Optimization Methods',
  content: `
# Numerical Optimization Methods

## Introduction

Practical optimization relies on iterative numerical algorithms. Understanding these methods is crucial for tuning hyperparameters and diagnosing training issues in machine learning.

**Goal:** Minimize f (x) where we can compute f (x) and ∇f (x), but not solve ∇f (x) = 0 analytically.

## Gradient Descent

**Update rule:**
x_{k+1} = x_k - α∇f (x_k)

where α is the learning rate (step size).

**Convergence for convex smooth f:**
- With appropriate α, converges to global minimum
- Rate: O(1/k) for general convex, O(exp(-k)) for strongly convex

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent (f, grad_f, x0, alpha=0.1, max_iter=100, tol=1e-6):
    """
    Gradient descent optimization
    
    Args:
        f: objective function
        grad_f: gradient function
        x0: initial point
        alpha: learning rate
        max_iter: maximum iterations
        tol: convergence tolerance
    
    Returns:
        x: optimum
        history: optimization trajectory
    """
    x = x0.copy()
    history = {'x': [x.copy()], 'f': [f (x)]}
    
    for k in range (max_iter):
        # Compute gradient
        grad = grad_f (x)
        
        # Update
        x = x - alpha * grad
        
        history['x'].append (x.copy())
        history['f'].append (f(x))
        
        # Check convergence
        if np.linalg.norm (grad) < tol:
            print(f"Converged in {k+1} iterations")
            break
    
    return x, history

# Test on quadratic function
def f (x):
    return 0.5 * x[0]**2 + 2 * x[1]**2

def grad_f (x):
    return np.array([x[0], 4*x[1]])

x0 = np.array([4.0, 3.0])
x_opt, history = gradient_descent (f, grad_f, x0, alpha=0.1)

print(f"Optimum: {x_opt}")
print(f"Function value: {f (x_opt):.6f}")

# Visualize
x_history = np.array (history['x'])
plt.figure (figsize=(10, 5))

# Left: Contour plot with trajectory
ax1 = plt.subplot(121)
x1 = np.linspace(-5, 5, 100)
x2 = np.linspace(-5, 5, 100)
X1, X2 = np.meshgrid (x1, x2)
Z = 0.5 * X1**2 + 2 * X2**2

ax1.contour(X1, X2, Z, levels=20)
ax1.plot (x_history[:, 0], x_history[:, 1], 'ro-', linewidth=2, markersize=4)
ax1.plot(0, 0, 'g*', markersize=15, label='Optimum')
ax1.set_xlabel('x₁')
ax1.set_ylabel('x₂')
ax1.set_title('Gradient Descent Trajectory')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: Function value over iterations
ax2 = plt.subplot(122)
ax2.semilogy (history['f'])
ax2.set_xlabel('Iteration')
ax2.set_ylabel('f (x)')
ax2.set_title('Convergence (log scale)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gradient_descent.png', dpi=150, bbox_inches='tight')
print("Saved visualization")
\`\`\`

## Momentum

**Idea:** Accumulate velocity to accelerate in persistent directions.

**Update:**
v_{k+1} = βv_k - α∇f (x_k)
x_{k+1} = x_k + v_{k+1}

where β ∈ [0, 1) is momentum coefficient (typically 0.9).

**Benefits:**
- Faster convergence in ravines
- Dampens oscillations
- Can escape shallow local minima

\`\`\`python
def gradient_descent_momentum (f, grad_f, x0, alpha=0.1, beta=0.9, max_iter=100):
    """Gradient descent with momentum"""
    x = x0.copy()
    v = np.zeros_like (x)
    history = {'x': [x.copy()], 'f': [f (x)]}
    
    for k in range (max_iter):
        grad = grad_f (x)
        
        # Update velocity
        v = beta * v - alpha * grad
        
        # Update position
        x = x + v
        
        history['x'].append (x.copy())
        history['f'].append (f(x))
    
    return x, history

# Compare with plain GD
x_gd, hist_gd = gradient_descent (f, grad_f, x0, alpha=0.1, max_iter=50)
x_mom, hist_mom = gradient_descent_momentum (f, grad_f, x0, alpha=0.1, beta=0.9, max_iter=50)

print("Comparison:")
print(f"GD final loss: {hist_gd['f'][-1]:.6f}")
print(f"Momentum final loss: {hist_mom['f'][-1]:.6f}")
print(f"→ Momentum converges faster!")
\`\`\`

## Adam Optimizer

**Adaptive Moment Estimation** - combines momentum with adaptive learning rates.

**Update:**
m_t = β₁m_{t-1} + (1-β₁)∇f (x_t)  # First moment (mean)
v_t = β₂v_{t-1} + (1-β₂)∇f (x_t)²  # Second moment (variance)
m̂_t = m_t/(1-β₁^t)  # Bias correction
v̂_t = v_t/(1-β₂^t)
x_{t+1} = x_t - α·m̂_t/(√v̂_t + ε)

**Default:** β₁=0.9, β₂=0.999, ε=10⁻⁸

\`\`\`python
def adam (f, grad_f, x0, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8, max_iter=100):
    """Adam optimizer"""
    x = x0.copy()
    m = np.zeros_like (x)
    v = np.zeros_like (x)
    history = {'x': [x.copy()], 'f': [f (x)]}
    
    for t in range(1, max_iter + 1):
        grad = grad_f (x)
        
        # Update biased first moment
        m = beta1 * m + (1 - beta1) * grad
        
        # Update biased second moment
        v = beta2 * v + (1 - beta2) * grad**2
        
        # Bias correction
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        # Update parameters
        x = x - alpha * m_hat / (np.sqrt (v_hat) + eps)
        
        history['x'].append (x.copy())
        history['f'].append (f(x))
    
    return x, history

x_adam, hist_adam = adam (f, grad_f, x0, alpha=0.1, max_iter=50)

print(f"Adam final loss: {hist_adam['f'][-1]:.6f}")
print("→ Adam adapts learning rate per parameter")
\`\`\`

## Line Search

Instead of fixed learning rate, find α_k that minimizes f (x_k - α∇f (x_k)).

**Backtracking line search:**
Start with α, reduce until Armijo condition satisfied:
f (x - α∇f (x)) ≤ f (x) - c·α·||∇f (x)||²

\`\`\`python
def backtracking_line_search (f, grad_f, x, grad, alpha_init=1.0, c=1e-4, rho=0.5):
    """
    Backtracking line search
    
    Args:
        f: objective function
        grad_f: gradient function
        x: current point
        grad: current gradient
        alpha_init: initial step size
        c: Armijo condition parameter
        rho: backtracking factor
    
    Returns:
        alpha: step size satisfying Armijo condition
    """
    alpha = alpha_init
    fx = f (x)
    grad_norm_sq = np.dot (grad, grad)
    
    while f (x - alpha * grad) > fx - c * alpha * grad_norm_sq:
        alpha *= rho
        if alpha < 1e-10:
            break
    
    return alpha

def gradient_descent_line_search (f, grad_f, x0, max_iter=100, tol=1e-6):
    """Gradient descent with line search"""
    x = x0.copy()
    history = {'x': [x.copy()], 'f': [f (x)], 'alpha': []}
    
    for k in range (max_iter):
        grad = grad_f (x)
        
        # Find step size via line search
        alpha = backtracking_line_search (f, grad_f, x, grad)
        history['alpha'].append (alpha)
        
        # Update
        x = x - alpha * grad
        
        history['x'].append (x.copy())
        history['f'].append (f(x))
        
        if np.linalg.norm (grad) < tol:
            print(f"Converged in {k+1} iterations")
            break
    
    return x, history

x_ls, hist_ls = gradient_descent_line_search (f, grad_f, x0, max_iter=50)

print(f"\\nLine search:")
print(f"Final loss: {hist_ls['f'][-1]:.6f}")
print(f"Average step size: {np.mean (hist_ls['alpha']):.4f}")
print("→ Adaptive step size improves convergence")
\`\`\`

## Newton\'s Method

**Update:**
x_{k+1} = x_k - H^{-1}∇f (x_k)

where H = ∇²f (x_k) is the Hessian.

**Pros:**
- Quadratic convergence near optimum
- Automatic step size

**Cons:**
- Expensive (O(n³) per iteration)
- Requires Hessian
- May diverge if H not positive definite

\`\`\`python
def newton_method (f, grad_f, hessian_f, x0, max_iter=20):
    """Newton's method"""
    x = x0.copy()
    history = {'x': [x.copy()], 'f': [f (x)]}
    
    for k in range (max_iter):
        grad = grad_f (x)
        H = hessian_f (x)
        
        # Solve Hx = grad for x
        try:
            direction = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            print("Hessian singular, stopping")
            break
        
        # Newton step
        x = x - direction
        
        history['x'].append (x.copy())
        history['f'].append (f(x))
        
        if np.linalg.norm (grad) < 1e-6:
            print(f"Converged in {k+1} iterations")
            break
    
    return x, history

def hessian_f (x):
    return np.array([[1, 0], [0, 4]])

x_newton, hist_newton = newton_method (f, grad_f, hessian_f, x0, max_iter=10)

# Compare convergence rates
print("\\nConvergence comparison (10 iterations):")
print(f"GD:        {hist_gd['f'][min(10, len (hist_gd['f'])-1)]:.2e}")
print(f"Momentum:  {hist_mom['f'][min(10, len (hist_mom['f'])-1)]:.2e}")
print(f"Adam:      {hist_adam['f'][min(10, len (hist_adam['f'])-1)]:.2e}")
print(f"Newton:    {hist_newton['f'][min(10, len (hist_newton['f'])-1)]:.2e}")
print("→ Newton converges fastest (when Hessian available)")
\`\`\`

## Stochastic Gradient Descent (SGD)

**Motivation:** For large datasets, computing full gradient expensive.

**Idea:** Use gradient estimate from mini-batch:
∇f (x) ≈ (1/B)Σᵢ∈batch ∇fᵢ(x)

**Update:**
x_{k+1} = x_k - α_k · ∇̂f (x_k)

where ∇̂f is noisy gradient estimate.

\`\`\`python
def sgd_example():
    """Demonstrate SGD on simple problem"""
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn (n_samples, 2)
    w_true = np.array([2.0, -1.0])
    y = X @ w_true + np.random.randn (n_samples) * 0.1
    
    def loss_full (w):
        """Full batch loss"""
        return 0.5 * np.mean((X @ w - y)**2)
    
    def grad_full (w):
        """Full batch gradient"""
        return X.T @ (X @ w - y) / n_samples
    
    def grad_mini_batch (w, batch_size=32):
        """Mini-batch gradient"""
        indices = np.random.choice (n_samples, batch_size, replace=False)
        X_batch = X[indices]
        y_batch = y[indices]
        return X_batch.T @ (X_batch @ w - y_batch) / batch_size
    
    # Full batch gradient descent
    w_gd = np.zeros(2)
    losses_gd = []
    for _ in range(100):
        losses_gd.append (loss_full (w_gd))
        w_gd = w_gd - 0.1 * grad_full (w_gd)
    
    # SGD with mini-batches
    w_sgd = np.zeros(2)
    losses_sgd = []
    for _ in range(100):
        losses_sgd.append (loss_full (w_sgd))
        w_sgd = w_sgd - 0.1 * grad_mini_batch (w_sgd, batch_size=32)
    
    print("Final weights:")
    print(f"True: {w_true}")
    print(f"GD:   {w_gd}")
    print(f"SGD:  {w_sgd}")
    
    plt.figure (figsize=(10, 4))
    plt.semilogy (losses_gd, label='Full batch GD')
    plt.semilogy (losses_sgd, label='Mini-batch SGD', alpha=0.7)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Full Batch vs Mini-Batch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('sgd_comparison.png', dpi=150, bbox_inches='tight')
    print("→ SGD noisier but computationally efficient")

sgd_example()
\`\`\`

## Summary

**Optimization Methods:**
| Method | Cost/iter | Convergence | Use case |
|--------|-----------|-------------|----------|
| **GD** | O(n) | Linear | Smooth convex |
| **Momentum** | O(n) | Faster | Ill-conditioned |
| **Adam** | O(n) | Adaptive | General (default) |
| **Newton** | O(n³) | Quadratic | Small n, convex |
| **SGD** | O(batch) | Noisy | Large datasets |

**Practical advice:**
- Start with Adam (α=0.001)
- Use momentum for ill-conditioned problems
- SGD essential for large-scale ML
- Line search for convex optimization
- Newton's method rarely used in deep learning

**Key insight:** Modern deep learning relies almost exclusively on first-order methods (gradient-based) with adaptive learning rates and momentum.
`,
};
