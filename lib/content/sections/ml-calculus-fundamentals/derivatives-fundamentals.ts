/**
 * Derivatives Fundamentals Section
 */

export const derivativesfundamentalsSection = {
  id: 'derivatives-fundamentals',
  title: 'Derivatives Fundamentals',
  content: `
# Derivatives Fundamentals

## Introduction

The derivative is the cornerstone of calculus and machine learning. It measures **instantaneous rate of change** - how fast a function is changing at any given point. In machine learning, derivatives tell us how to adjust model parameters to reduce error, forming the foundation of backpropagation and gradient descent.

## Definition of the Derivative

**Formal Definition**: The derivative of f(x) at x = a is:

\\\`\\\`\\\`
f'(a) = lim_{h → 0} [f(a + h) - f(a)] / h
\\\`\\\`\\\`

**Intuition**: The derivative is the slope of the tangent line to f(x) at x = a.

**Alternative notation**:
- f'(x) (Lagrange notation)
- df/dx (Leibniz notation)
- Df(x) (Euler notation)  
- ∂f/∂x (partial derivative, coming later)

\\\`\\\`\\\`python
import numpy as np
import matplotlib.pyplot as plt

def numerical_derivative(f, x, h=1e-5):
    """
    Compute derivative using limit definition
    """
    return (f(x + h) - f(x)) / h

# Example function
def f(x):
    return x**2

# Compute derivative at x = 3
x_point = 3
derivative = numerical_derivative(f, x_point)
print(f"f(x) = x²")
print(f"f'({x_point}) ≈ {derivative}")
print(f"Exact: f'(x) = 2x, so f'(3) = {2 * x_point}")

# Visualize
x = np.linspace(0, 5, 100)
y = f(x)

# Tangent line at x = 3
slope = derivative
y_tangent = f(x_point) + slope * (x - x_point)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', label='f(x) = x²', linewidth=2)
plt.plot(x, y_tangent, 'r--', label=f'Tangent line (slope = {slope:.2f})', linewidth=2)
plt.scatter([x_point], [f(x_point)], color='red', s=100, zorder=5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Derivative as Slope of Tangent Line')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
\\\`\\\`\\\`

## Geometric Interpretation

The derivative at a point is the **slope of the tangent line** at that point.

\\\`\\\`\\\`python
def visualize_derivative_approach(f, x_point, h_values):
    """
    Visualize how secant lines approach tangent line
    """
    x = np.linspace(x_point - 2, x_point + 2, 100)
    y = f(x)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, h in enumerate(h_values):
        ax = axes[idx]
        
        # Secant line
        slope_secant = (f(x_point + h) - f(x_point)) / h
        y_secant = f(x_point) + slope_secant * (x - x_point)
        
        # Plot
        ax.plot(x, y, 'b-', label='f(x) = x²', linewidth=2)
        ax.plot(x, y_secant, 'r--', label=f'Secant (h={h})', linewidth=2)
        ax.scatter([x_point, x_point + h], [f(x_point), f(x_point + h)], 
                   color='red', s=100, zorder=5)
        
        # Draw h and f(x+h) - f(x)
        ax.plot([x_point, x_point + h], [f(x_point), f(x_point)], 'g-', linewidth=2, label='h')
        ax.plot([x_point + h], [f(x_point), f(x_point + h)], 'orange', linewidth=2, label='Δy')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Secant slope ≈ {slope_secant:.3f} (h = {h})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Demonstrate convergence to derivative
h_values = [1.0, 0.5, 0.1, 0.01]
visualize_derivative_approach(lambda x: x**2, 3, h_values)
\\\`\\\`\\\`

## Basic Derivative Rules

### Power Rule

**Rule**: If f(x) = x^n, then f'(x) = n·x^(n-1)

\\\`\\\`\\\`python
def power_rule_derivative(n):
    """Return function that computes derivative of x^n"""
    return lambda x: n * x**(n-1)

# Examples
functions = [
    (lambda x: x**2, power_rule_derivative(2), "x²", "2x"),
    (lambda x: x**3, power_rule_derivative(3), "x³", "3x²"),
    (lambda x: x**0.5, power_rule_derivative(0.5), "√x", "0.5x^(-0.5)"),
]

x_test = 4
print("Power Rule Examples:")
print("="*50)
for f, f_prime, f_str, f_prime_str in functions:
    numerical = numerical_derivative(f, x_test)
    analytical = f_prime(x_test)
    print(f"f(x) = {f_str}")
    print(f"f'(x) = {f_prime_str}")
    print(f"At x = {x_test}:")
    print(f"  Numerical: {numerical:.6f}")
    print(f"  Analytical: {analytical:.6f}")
    print()
\\\`\\\`\\\`

### Constant Rule

**Rule**: If f(x) = c (constant), then f'(x) = 0

\\\`\\\`\\\`python
# Constant function has zero derivative
def constant_function(x):
    return 5

x_range = np.linspace(-5, 5, 100)
derivatives = [numerical_derivative(constant_function, x) for x in x_range]

print(f"Derivative of f(x) = 5: {np.mean(derivatives):.10f} ≈ 0")
\\\`\\\`\\\`

### Constant Multiple Rule

**Rule**: If f(x) = c·g(x), then f'(x) = c·g'(x)

\\\`\\\`\\\`python
# Example: f(x) = 3x²
def f(x):
    return 3 * x**2

# Derivative should be 6x
x_test = 5
derivative = numerical_derivative(f, x_test)
expected = 6 * x_test

print(f"f(x) = 3x²")
print(f"f'({x_test}) ≈ {derivative} (expected: {expected})")
\\\`\\\`\\\`

### Sum and Difference Rules

**Rule**: If f(x) = g(x) ± h(x), then f'(x) = g'(x) ± h'(x)

\\\`\\\`\\\`python
# Example: f(x) = x² + 3x + 5
def f(x):
    return x**2 + 3*x + 5

# Derivative: 2x + 3
x_test = 2
numerical = numerical_derivative(f, x_test)
analytical = 2*x_test + 3

print(f"f(x) = x² + 3x + 5")
print(f"f'(x) = 2x + 3")
print(f"At x = {x_test}:")
print(f"  Numerical: {numerical}")
print(f"  Analytical: {analytical}")
\\\`\\\`\\\`

## Common Derivatives

\\\`\\\`\\\`python
# Common functions and their derivatives
common_derivatives = [
    ("x^n", "n·x^(n-1)", lambda x, n=3: x**n, lambda x, n=3: n*x**(n-1)),
    ("e^x", "e^x", np.exp, np.exp),
    ("ln(x)", "1/x", np.log, lambda x: 1/x),
    ("sin(x)", "cos(x)", np.sin, np.cos),
    ("cos(x)", "-sin(x)", np.cos, lambda x: -np.sin(x)),
]

x_test = 2.0
print("Common Derivatives:")
print("="*60)
for func_str, deriv_str, func, deriv in common_derivatives:
    numerical = numerical_derivative(func, x_test)
    analytical = deriv(x_test)
    error = abs(numerical - analytical)
    print(f"f(x) = {func_str:10s} → f'(x) = {deriv_str:15s}")
    print(f"  At x = {x_test}: {numerical:.6f} (error: {error:.2e})")
\\\`\\\`\\\`

## Derivatives in Machine Learning

### 1. Loss Functions and Gradients

\\\`\\\`\\\`python
def mean_squared_error(y_true, y_pred):
    """MSE loss function"""
    return np.mean((y_true - y_pred)**2)

def mse_derivative(y_true, y_pred):
    """Derivative of MSE with respect to predictions"""
    return 2 * (y_pred - y_true) / len(y_true)

# Example
y_true = np.array([1, 2, 3, 4, 5])
y_pred = np.array([1.1, 2.3, 2.8, 4.2, 4.9])

loss = mean_squared_error(y_true, y_pred)
gradient = mse_derivative(y_true, y_pred)

print(f"MSE Loss: {loss}")
print(f"Gradient (how to adjust predictions): {gradient}")
print(f"Negative gradient points toward improvement")
\\\`\\\`\\\`

### 2. Linear Regression Gradient

\\\`\\\`\\\`python
def linear_model(X, w, b):
    """Linear model: y = wX + b"""
    return X * w + b

def mse_loss(X, y_true, w, b):
    """MSE loss for linear regression"""
    y_pred = linear_model(X, w, b)
    return np.mean((y_true - y_pred)**2)

def compute_gradients(X, y_true, w, b):
    """Compute gradients of MSE with respect to w and b"""
    y_pred = linear_model(X, w, b)
    error = y_pred - y_true
    
    dw = 2 * np.mean(error * X)
    db = 2 * np.mean(error)
    
    return dw, db

# Generate data
np.random.seed(42)
X = np.random.randn(100)
y_true = 2 * X + 1 + 0.1 * np.random.randn(100)

# Initial parameters
w, b = 0.0, 0.0
learning_rate = 0.1
losses = []

# Gradient descent
for epoch in range(100):
    loss = mse_loss(X, y_true, w, b)
    losses.append(loss)
    
    dw, db = compute_gradients(X, y_true, w, b)
    
    # Update parameters using derivatives
    w = w - learning_rate * dw
    b = b - learning_rate * db

print(f"Final parameters: w = {w:.3f}, b = {b:.3f}")
print(f"True parameters: w = 2.0, b = 1.0")

# Plot loss curve
plt.figure(figsize=(10, 5))
plt.plot(losses, 'b-')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Loss Decreases Using Derivatives (Gradient Descent)')
plt.grid(True, alpha=0.3)
plt.show()
\\\`\\\`\\\`

### 3. Activation Function Derivatives

\\\`\\\`\\\`python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of sigmoid: σ'(x) = σ(x)(1 - σ(x))"""
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU"""
    return (x > 0).astype(float)

# Visualize
x = np.linspace(-5, 5, 1000)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Sigmoid
axes[0, 0].plot(x, sigmoid(x), 'b-', label='σ(x)', linewidth=2)
axes[0, 0].set_title('Sigmoid')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(x, sigmoid_derivative(x), 'r-', label="σ'(x)", linewidth=2)
axes[0, 1].set_title('Sigmoid Derivative')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# ReLU
axes[1, 0].plot(x, relu(x), 'b-', label='ReLU(x)', linewidth=2)
axes[1, 0].set_title('ReLU')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(x, relu_derivative(x), 'r-', label="ReLU'(x)", linewidth=2)
axes[1, 1].set_title('ReLU Derivative')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Key insight: Sigmoid derivative vanishes at extremes
print("Sigmoid derivative values:")
for x_val in [-5, -2, 0, 2, 5]:
    deriv = sigmoid_derivative(x_val)
    print(f"  σ'({x_val}) = {deriv:.6f}")

print("\\nNotice: Derivative is largest at x=0, nearly zero at extremes!")
print("This is the vanishing gradient problem.")
\\\`\\\`\\\`

## Numerical vs Analytical Derivatives

\\\`\\\`\\\`python
def compare_derivatives(f, f_prime_analytical, x, h_values):
    """Compare numerical and analytical derivatives"""
    print(f"Comparing derivatives at x = {x}")
    print("="*60)
    print(f"{'h':>12} {'Numerical':>15} {'Analytical':>15} {'Error':>15}")
    print("-"*60)
    
    analytical = f_prime_analytical(x)
    
    for h in h_values:
        numerical = (f(x + h) - f(x)) / h
        error = abs(numerical - analytical)
        print(f"{h:>12.2e} {numerical:>15.8f} {analytical:>15.8f} {error:>15.2e}")

# Test with f(x) = sin(x), f'(x) = cos(x)
h_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
compare_derivatives(np.sin, np.cos, 1.0, h_values)

print("\\nNotice: Error decreases as h gets smaller, but only up to a point!")
print("Too small h causes floating-point precision issues.")
\\\`\\\`\\\`

## Best Practices

### 1. Choosing h for Numerical Derivatives

\\\`\\\`\\\`python
# Optimal h balances truncation error and round-off error
def find_optimal_h(f, x, f_prime_true):
    """Find optimal h for numerical derivative"""
    h_values = 10.0**np.arange(-1, -16, -0.5)
    errors = []
    
    for h in h_values:
        numerical = (f(x + h) - f(x)) / h
        error = abs(numerical - f_prime_true(x))
        errors.append(error)
    
    optimal_idx = np.argmin(errors)
    optimal_h = h_values[optimal_idx]
    
    plt.figure(figsize=(10, 6))
    plt.loglog(h_values, errors, 'bo-')
    plt.scatter([optimal_h], [errors[optimal_idx]], color='red', s=200, zorder=5)
    plt.xlabel('h')
    plt.ylabel('Absolute Error')
    plt.title(f'Optimal h ≈ {optimal_h:.2e}')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return optimal_h

optimal_h = find_optimal_h(np.sin, 1.0, np.cos)
print(f"Optimal h: {optimal_h}")
print(f"Typically around √ε where ε is machine epsilon (~2.2e-16)")
print(f"√ε ≈ {np.sqrt(np.finfo(float).eps)}")
\\\`\\\`\\\`

### 2. Central Difference Method

\\\`\\\`\\\`python
def forward_difference(f, x, h):
    """Forward difference: (f(x+h) - f(x)) / h"""
    return (f(x + h) - f(x)) / h

def central_difference(f, x, h):
    """Central difference: (f(x+h) - f(x-h)) / (2h)"""
    return (f(x + h) - f(x - h)) / (2 * h)

# Compare accuracy
x = 1.0
h = 1e-5
f = np.sin
f_prime = np.cos

forward_error = abs(forward_difference(f, x, h) - f_prime(x))
central_error = abs(central_difference(f, x, h) - f_prime(x))

print(f"Forward difference error: {forward_error:.2e}")
print(f"Central difference error: {central_error:.2e}")
print(f"Central difference is {forward_error/central_error:.0f}x more accurate!")
\\\`\\\`\\\`

## Common Pitfalls

\\\`\\\`\\\`python
# Pitfall 1: Derivative doesn't exist at sharp corners
def absolute_value(x):
    return np.abs(x)

x_point = 0
try:
    derivative = numerical_derivative(absolute_value, x_point)
    print(f"|x| derivative at 0: {derivative}")
    print("This is misleading! Derivative doesn't exist at x=0")
except:
    pass

# Check left and right derivatives
h = 1e-5
left_deriv = (absolute_value(x_point) - absolute_value(x_point - h)) / h
right_deriv = (absolute_value(x_point + h) - absolute_value(x_point)) / h

print(f"Left derivative: {left_deriv:.6f}")
print(f"Right derivative: {right_deriv:.6f}")
print("They differ! Derivative doesn't exist.")

# Pitfall 2: Floating-point issues with very small h
h_too_small = 1e-20
derivative_bad = (np.sin(1 + h_too_small) - np.sin(1)) / h_too_small
print(f"\\nDerivative with h={h_too_small}: {derivative_bad}")
print(f"True value: {np.cos(1)}")
print("Floating-point precision causes large error!")
\\\`\\\`\\\`

## Summary

**Key Takeaways**:
- Derivative measures instantaneous rate of change
- Power rule: d/dx[x^n] = n·x^(n-1)
- Sum/difference/constant multiple rules allow building complex derivatives
- Derivatives are essential for gradient descent and backpropagation
- Numerical derivatives need careful choice of h
- Central difference is more accurate than forward difference
- Be aware of non-differentiable points

**ML Applications**:
- Gradient descent relies on derivatives to update parameters
- Backpropagation computes derivatives through chain rule
- Activation function derivatives affect gradient flow
- Loss function derivatives guide optimization
`,
};
