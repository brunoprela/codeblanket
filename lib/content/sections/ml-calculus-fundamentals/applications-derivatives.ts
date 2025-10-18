/**
 * Applications of Derivatives Section
 */

export const applicationsderivativesSection = {
  id: 'applications-derivatives',
  title: 'Applications of Derivatives',
  content: `
# Applications of Derivatives

## Introduction

Derivatives aren't just abstract math - they solve real problems in optimization, approximation, and analysis. In ML, derivatives power gradient descent, Newton's method, and Taylor approximations.

## Finding Extrema (Maxima and Minima)

**Critical Points**: Where f'(x) = 0 or f'(x) undefined

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# Find minimum of f(x) = x²  - 4x + 5
def f(x):
    return x**2 - 4*x + 5

def f_prime(x):
    return 2*x - 4

# Find critical point
critical_x = -(-4) / (2*1)  # From f'(x) = 0
print(f"Critical point: x = {critical_x}")
print(f"Value: f({critical_x}) = {f(critical_x)}")

# Second derivative test
def f_double_prime(x):
    return 2

second_deriv = f_double_prime(critical_x)
if second_deriv > 0:
    print("Minimum (concave up)")
elif second_deriv < 0:
    print("Maximum (concave down)")
\`\`\`

## Newton's Method for Optimization

**Idea**: Use quadratic approximation to find roots/optima faster

\`\`\`python
def newtons_method(f, f_prime, x0, tol=1e-6, max_iter=100):
    """Find root of f using Newton's method"""
    x = x0
    for i in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            return x, i+1
        x = x - fx / f_prime(x)
    return x, max_iter

# Find root of f(x) = x³ - 2x - 5
def f(x):
    return x**3 - 2*x - 5

def f_prime(x):
    return 3*x**2 - 2

root, iters = newtons_method(f, f_prime, 2.0)
print(f"Root: {root} (found in {iters} iterations)")
print(f"Verification: f({root}) = {f(root)}")
\`\`\`

## Taylor Series Approximation

**Taylor Series**: f(x) ≈ f(a) + f'(a)(x-a) + f''(a)(x-a)²/2! + ...

\`\`\`python
def taylor_approximation(f, derivatives, a, x, order):
    """Compute Taylor approximation of order n"""
    result = 0
    factorial = 1
    power = 1
    
    for n in range(order + 1):
        if n == 0:
            term = f(a)
        else:
            term = derivatives[n-1](a) * power / factorial
            factorial *= (n + 1)
            power *= (x - a)
        result += term
    
    return result

# Approximate e^x around x=0
def exp_deriv(x):
    return np.exp(x)  # All derivatives of e^x are e^x

x_approx = 0.5
orders = [1, 2, 3, 5]

print(f"Approximating e^{x_approx}:")
print(f"True value: {np.exp(x_approx)}")

for order in orders:
    approx = taylor_approximation(
        np.exp,
        [exp_deriv] * order,
        0,
        x_approx,
        order
    )
    error = abs(approx - np.exp(x_approx))
    print(f"Order {order}: {approx:.6f} (error: {error:.2e})")
\`\`\`

## Gradient Descent Applications

\`\`\`python
def gradient_descent_2d(f, grad_f, x0, learning_rate=0.1, max_iter=1000):
    """2D gradient descent"""
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    
    for i in range(max_iter):
        gradient = grad_f(x)
        x = x - learning_rate * gradient
        history.append(x.copy())
        
        if np.linalg.norm(gradient) < 1e-6:
            break
    
    return x, np.array(history)

# Minimize f(x,y) = x² + y²
def f(xy):
    return xy[0]**2 + xy[1]**2

def grad_f(xy):
    return np.array([2*xy[0], 2*xy[1]])

optimal, history = gradient_descent_2d(f, grad_f, [5.0, 5.0])
print(f"Optimal point: {optimal}")
print(f"Minimum value: {f(optimal)}")

# Visualize
plt.figure(figsize=(10, 6))
plt.plot(history[:, 0], history[:, 1], 'b-o', markersize=3)
plt.scatter([0], [0], color='r', s=100, marker='*', label='Minimum')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gradient Descent Path')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
\`\`\`

## Optimization in Machine Learning

\`\`\`python
# Linear regression with gradient descent
class LinearRegressionGD:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
        self.w = None
        self.b = None
    
    def fit(self, X, y, epochs=1000):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        losses = []
        
        for epoch in range(epochs):
            # Forward pass
            y_pred = X @ self.w + self.b
            
            # Compute loss
            loss = np.mean((y_pred - y)**2)
            losses.append(loss)
            
            # Compute gradients (derivatives!)
            dw = (2/n_samples) * X.T @ (y_pred - y)
            db = (2/n_samples) * np.sum(y_pred - y)
            
            # Update using derivatives
            self.w -= self.lr * dw
            self.b -= self.lr * db
        
        return losses
    
    def predict(self, X):
        return X @ self.w + self.b

# Test
np.random.seed(42)
X = np.random.randn(100, 1)
y_true = 3*X.squeeze() + 2 + 0.1*np.random.randn(100)

model = LinearRegressionGD(learning_rate=0.1)
losses = model.fit(X, y_true, epochs=100)

print(f"Learned: w={model.w[0]:.3f}, b={model.b:.3f}")
print(f"True: w=3.0, b=2.0")

plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Loss Decreases (Derivatives in Action!)')
plt.grid(True, alpha=0.3)
plt.show()
\`\`\`

## Summary

Derivatives enable:
- Finding optimal points (gradient = 0)
- Newton's method for fast root-finding
- Taylor approximations
- Gradient descent for ML optimization
`,
};
