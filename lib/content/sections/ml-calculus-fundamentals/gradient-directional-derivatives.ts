/**
 * Gradient & Directional Derivatives Section
 */

export const gradientdirectionalderivativesSection = {
  id: 'gradient-directional-derivatives',
  title: 'Gradient & Directional Derivatives',
  content: `
# Gradient & Directional Derivatives

## Introduction

The gradient tells us the direction of steepest increase, but what if we want to know the rate of change in a specific direction? Directional derivatives answer this question and are fundamental to understanding optimization landscapes in machine learning.

## Directional Derivative

**Definition**: The directional derivative of f at point **a** in direction **v** is:

D_v f(**a**) = lim_{h→0} [f(**a** + h**v**) - f(**a**)] / h

This measures the rate of change of f when moving from **a** in direction **v**.

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def directional_derivative (f, point, direction, h=1e-7):
    """
    Compute directional derivative of f at point in given direction
    """
    point = np.array (point, dtype=float)
    direction = np.array (direction, dtype=float)
    
    # Normalize direction vector
    direction = direction / np.linalg.norm (direction)
    
    # Compute directional derivative
    f_plus = f (point + h * direction)
    f_point = f (point)
    
    return (f_plus - f_point) / h

# Example function: f (x,y) = x² + 2y²
def f (xy):
    if len (xy.shape) == 1:
        x, y = xy
    else:
        x, y = xy[0], xy[1]
    return x**2 + 2*y**2

# Test point and various directions
point = np.array([1.0, 1.0])

# Directions to test
directions = [
    np.array([1, 0]),      # Along x-axis
    np.array([0, 1]),      # Along y-axis
    np.array([1, 1]),      # Diagonal
    np.array([1, -1]),     # Opposite diagonal
    np.array([-1, 0]),     # Negative x
]

print(f"Directional derivatives at point {point}:")
print("="*60)

for direction in directions:
    dir_deriv = directional_derivative (f, point, direction)
    direction_normalized = direction / np.linalg.norm (direction)
    print(f"Direction {direction_normalized}: {dir_deriv:.4f}")
\`\`\`

## Relationship to Gradient

**Key Theorem**: The directional derivative equals the dot product of the gradient with the direction:

D_v f(**a**) = ∇f(**a**) · **v̂**

where **v̂** is the unit vector in direction **v**.

\`\`\`python
def gradient_2d (f, point, h=1e-7):
    """Compute gradient numerically"""
    x, y = point
    df_dx = (f (np.array([x + h, y])) - f (np.array([x, y]))) / h
    df_dy = (f (np.array([x, y + h])) - f (np.array([x, y]))) / h
    return np.array([df_dx, df_dy])

# Compute gradient
grad = gradient_2d (f, point)
print(f"\\nGradient at {point}: {grad}")

# Verify: directional derivative = gradient · direction
print("\\nVerification (gradient · direction):")
print("="*60)

for direction in directions:
    direction_normalized = direction / np.linalg.norm (direction)
    
    # Method 1: Direct computation
    dir_deriv_direct = directional_derivative (f, point, direction)
    
    # Method 2: Gradient dot product
    dir_deriv_gradient = np.dot (grad, direction_normalized)
    
    error = abs (dir_deriv_direct - dir_deriv_gradient)
    
    print(f"Direction {direction_normalized}:")
    print(f"  Direct: {dir_deriv_direct:.8f}")
    print(f"  Grad·v: {dir_deriv_gradient:.8f}")
    print(f"  Error:  {error:.2e}")
\`\`\`

## Maximum Rate of Change

**Theorem**: The gradient points in the direction of maximum rate of increase, and its magnitude is that maximum rate.

\`\`\`python
def visualize_directional_derivatives (f, point, num_directions=36):
    """
    Visualize directional derivatives in all directions
    """
    # Create directions (unit circle)
    angles = np.linspace(0, 2*np.pi, num_directions, endpoint=False)
    directions = np.column_stack([np.cos (angles), np.sin (angles)])
    
    # Compute directional derivatives
    dir_derivs = np.array([
        directional_derivative (f, point, d) for d in directions
    ])
    
    # Compute gradient
    grad = gradient_2d (f, point)
    grad_magnitude = np.linalg.norm (grad)
    grad_direction = grad / grad_magnitude
    
    # Create polar plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Polar plot of directional derivatives
    ax1 = plt.subplot(121, projection='polar')
    ax1.plot (angles, dir_derivs, 'b-', linewidth=2)
    ax1.fill (angles, dir_derivs, alpha=0.3)
    
    # Mark gradient direction
    grad_angle = np.arctan2(grad_direction[1], grad_direction[0])
    ax1.plot([grad_angle], [grad_magnitude], 'ro', markersize=10, label='Gradient')
    
    ax1.set_title (f'Directional Derivatives\\n (max = {grad_magnitude:.2f})')
    ax1.legend()
    
    # Contour plot with gradient
    x = np.linspace (point[0] - 2, point[0] + 2, 100)
    y = np.linspace (point[1] - 2, point[1] + 2, 100)
    X, Y = np.meshgrid (x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f (np.array([X[i, j], Y[i, j]]))
    
    ax2.contour(X, Y, Z, levels=20, cmap='viridis')
    ax2.quiver (point[0], point[1], grad[0], grad[1], 
               color='r', scale=10, width=0.01, label='Gradient')
    
    # Show several directional derivatives
    for i in range(0, num_directions, 6):
        d = directions[i]
        ax2.quiver (point[0], point[1], d[0], d[1],
                  color='b', alpha=0.3, scale=5, width=0.005)
    
    ax2.scatter(*point, color='red', s=100, zorder=5)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Gradient = Direction of Max Increase')
    ax2.legend()
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\\nMaximum directional derivative: {np.max (dir_derivs):.6f}")
    print(f"Gradient magnitude: {grad_magnitude:.6f}")
    print(f"Direction of max increase: {grad_direction}")

visualize_directional_derivatives (f, point)
\`\`\`

## Gradient Descent Interpretation

Gradient descent moves in the direction of **steepest descent** = -∇f

\`\`\`python
def gradient_descent_with_directions (f, gradient_f, x0, learning_rate=0.1, 
                                    max_iterations=50):
    """
    Gradient descent with directional analysis
    """
    x = np.array (x0, dtype=float)
    history = [x.copy()]
    gradients = []
    
    for i in range (max_iterations):
        grad = gradient_f (x)
        gradients.append (grad.copy())
        
        # Move in negative gradient direction (steepest descent)
        x = x - learning_rate * grad
        history.append (x.copy())
        
        if np.linalg.norm (grad) < 1e-6:
            print(f"Converged in {i+1} iterations")
            break
    
    return np.array (history), np.array (gradients)

# Define function and its gradient
def rosenbrock (xy):
    """Rosenbrock function (banana function)"""
    x, y = xy
    return (1 - x)**2 + 100*(y - x**2)**2

def rosenbrock_gradient (xy):
    """Gradient of Rosenbrock function"""
    x, y = xy
    dx = -2*(1 - x) - 400*x*(y - x**2)
    dy = 200*(y - x**2)
    return np.array([dx, dy])

# Run gradient descent
x0 = np.array([-1.0, 1.0])
history, gradients = gradient_descent_with_directions(
    rosenbrock, rosenbrock_gradient, x0, learning_rate=0.001
)

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Contour plot with path
x = np.linspace(-2, 2, 400)
y = np.linspace(-1, 3, 400)
X, Y = np.meshgrid (x, y)
Z = (1 - X)**2 + 100*(Y - X**2)**2

ax1.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis')
ax1.plot (history[:, 0], history[:, 1], 'r-o', markersize=3, label='GD path')
ax1.scatter([1], [1], color='g', s=200, marker='*', label='Minimum', zorder=5)

# Show gradient vectors
for i in range(0, len (history)-1, 5):
    ax1.quiver (history[i, 0], history[i, 1], 
              -gradients[i, 0], -gradients[i, 1],
              color='red', alpha=0.5, scale=50, width=0.005)

ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Gradient Descent on Rosenbrock Function')
ax1.legend()

# Loss over iterations
losses = [rosenbrock (h) for h in history]
ax2.semilogy (losses, 'b-o', markersize=3)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Loss (log scale)')
ax2.set_title('Convergence')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Starting point: {x0}")
print(f"Final point: {history[-1]}")
print(f"True minimum: [1, 1]")
print(f"Final loss: {rosenbrock (history[-1]):.2e}")
\`\`\`

## Applications in Machine Learning

### 1. Understanding Loss Landscapes

\`\`\`python
def analyze_loss_landscape (model_params, loss_fn, data):
    """
    Analyze loss landscape around current parameters
    """
    grad = compute_gradient (model_params, loss_fn, data)
    
    # Sample random directions
    n_directions = 100
    random_directions = np.random.randn (n_directions, len (model_params))
    random_directions = random_directions / np.linalg.norm (random_directions, axis=1, keepdims=True)
    
    # Compute directional derivatives
    directional_derivs = []
    for direction in random_directions:
        dir_deriv = np.dot (grad, direction)
        directional_derivs.append (dir_deriv)
    
    directional_derivs = np.array (directional_derivs)
    
    print("Loss Landscape Analysis:")
    print(f"  Gradient norm: {np.linalg.norm (grad):.4f}")
    print(f"  Max dir deriv: {np.max (directional_derivs):.4f}")
    print(f"  Min dir deriv: {np.min (directional_derivs):.4f}")
    print(f"  Avg dir deriv: {np.mean (directional_derivs):.4f}")
    
    # The gradient magnitude should equal max directional derivative
    print(f"\\nVerification:")
    print(f"  Gradient magnitude = Max directional derivative?")
    print(f"  {np.linalg.norm (grad):.6f} ≈ {np.max (directional_derivs):.6f}")
    
    return directional_derivs

# Example: Simple quadratic loss
def simple_loss (params, X, y):
    predictions = X @ params
    return np.mean((predictions - y)**2)

def compute_gradient (params, loss_fn, data):
    X, y = data
    predictions = X @ params
    grad = (2/len(X)) * X.T @ (predictions - y)
    return grad

# Generate data
np.random.seed(42)
X = np.random.randn(100, 5)
y = X @ np.array([1, 2, 3, 4, 5]) + 0.1*np.random.randn(100)
params = np.zeros(5)

# Analyze
directional_derivs = analyze_loss_landscape (params, simple_loss, (X, y))
\`\`\`

### 2. Momentum and Gradient Direction

\`\`\`python
class GradientDescentWithMomentum:
    """
    Gradient descent with momentum - uses gradient direction
    but accumulates velocity
    """
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = None
    
    def step (self, params, gradient):
        """
        Update parameters using gradient (directional information)
        and accumulated velocity
        """
        if self.velocity is None:
            self.velocity = np.zeros_like (params)
        
        # Accumulate velocity (exponential moving average of gradients)
        self.velocity = self.momentum * self.velocity - self.lr * gradient
        
        # Update parameters
        params_new = params + self.velocity
        
        return params_new

# Compare gradient descent with and without momentum
def compare_optimizers (f, grad_f, x0, n_iterations=100):
    """Compare GD and GD with momentum"""
    
    # Standard GD
    gd_history = [np.array (x0)]
    x_gd = np.array (x0, dtype=float)
    
    for _ in range (n_iterations):
        grad = grad_f (x_gd)
        x_gd = x_gd - 0.001 * grad
        gd_history.append (x_gd.copy())
    
    # GD with momentum
    momentum_history = [np.array (x0)]
    x_momentum = np.array (x0, dtype=float)
    optimizer = GradientDescentWithMomentum (learning_rate=0.001, momentum=0.9)
    
    for _ in range (n_iterations):
        grad = grad_f (x_momentum)
        x_momentum = optimizer.step (x_momentum, grad)
        momentum_history.append (x_momentum.copy())
    
    return np.array (gd_history), np.array (momentum_history)

# Test on Rosenbrock
gd_history, momentum_history = compare_optimizers(
    rosenbrock, rosenbrock_gradient, [-1.0, 1.0]
)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

x = np.linspace(-2, 2, 400)
y = np.linspace(-1, 3, 400)
X, Y = np.meshgrid (x, y)
Z = (1 - X)**2 + 100*(Y - X**2)**2

for ax, history, title in zip (axes, [gd_history, momentum_history],
                               ['Gradient Descent', 'GD with Momentum']):
    ax.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis', alpha=0.5)
    ax.plot (history[:, 0], history[:, 1], 'r-o', markersize=2, linewidth=1.5)
    ax.scatter([1], [1], color='g', s=200, marker='*', label='Minimum')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title (title)
    ax.legend()

plt.tight_layout()
plt.show()

print("Final positions:")
print(f"  GD:           {gd_history[-1]}")
print(f"  GD+Momentum:  {momentum_history[-1]}")
print(f"  True minimum: [1, 1]")
\`\`\`

### 3. Projected Gradient for Constrained Optimization

\`\`\`python
def projected_gradient_descent (f, grad_f, x0, projection_fn,
                               learning_rate=0.1, max_iter=100):
    """
    Gradient descent with projection onto constraint set
    """
    x = np.array (x0, dtype=float)
    history = [x.copy()]
    
    for i in range (max_iter):
        grad = grad_f (x)
        
        # Take gradient step
        x_new = x - learning_rate * grad
        
        # Project onto constraint set
        x_new = projection_fn (x_new)
        
        history.append (x_new.copy())
        x = x_new
        
        if np.linalg.norm (grad) < 1e-6:
            break
    
    return np.array (history)

# Example: Optimize on unit sphere (||x|| = 1)
def project_to_sphere (x):
    """Project onto unit sphere"""
    return x / np.linalg.norm (x)

# Minimize f (x,y) = x² + 2y² subject to x² + y² = 1
history = projected_gradient_descent(
    lambda xy: xy[0]**2 + 2*xy[1]**2,
    lambda xy: np.array([2*xy[0], 4*xy[1]]),
    [0.6, 0.8],  # Start on sphere
    project_to_sphere,
    learning_rate=0.1
)

# Visualize
theta = np.linspace(0, 2*np.pi, 100)
circle_x = np.cos (theta)
circle_y = np.sin (theta)

plt.figure (figsize=(8, 8))
plt.plot (circle_x, circle_y, 'k--', alpha=0.3, label='Constraint: x²+y²=1')
plt.plot (history[:, 0], history[:, 1], 'r-o', markersize=4, label='Projected GD')
plt.scatter (history[-1, 0], history[-1, 1], color='g', s=200, marker='*', 
           label='Minimum')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Projected Gradient Descent on Unit Sphere')
plt.legend()
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.show()

print(f"Optimal point: {history[-1]}")
print(f"Constraint satisfied: ||x|| = {np.linalg.norm (history[-1]):.6f}")
print(f"Optimal value: {history[-1][0]**2 + 2*history[-1][1]**2:.6f}")
\`\`\`

## Summary

**Key Concepts**:
- Directional derivative D_v f = rate of change in direction v
- D_v f(**a**) = ∇f(**a**) · **v̂** (gradient dot product)
- Gradient points in direction of maximum increase
- ||∇f|| = maximum rate of increase
- Gradient descent uses -∇f (steepest descent)

**ML Applications**:
- Understanding optimization landscapes
- Momentum methods accumulate gradient directions
- Projected gradient for constrained optimization
- Natural gradient (different metric for directions)
`,
};
