/**
 * Partial Derivatives Section
 */

export const partialderivativesSection = {
  id: 'partial-derivatives',
  title: 'Partial Derivatives',
  content: `
# Partial Derivatives

## Introduction

Functions of multiple variables (f(x,y) or f(x₁,...,xₙ)) are everywhere in ML - loss functions depend on many parameters. Partial derivatives measure how f changes with respect to ONE variable while holding others constant.

## Definition

**Partial derivative** of f with respect to x:
∂f/∂x = lim_{h→0} [f(x+h, y) - f(x, y)] / h

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Example: f(x,y) = x² + xy + y²
def f(x, y):
    return x**2 + x*y + y**2

# Partial derivatives
def df_dx(x, y):
    return 2*x + y  # ∂f/∂x

def df_dy(x, y):
    return x + 2*y  # ∂f/∂y

# Test point
x0, y0 = 2, 3
print(f"At ({x0}, {y0}):")
print(f"∂f/∂x = {df_dx(x0, y0)}")
print(f"∂f/∂y = {df_dy(x0, y0)}")

# Visualize
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig = plt.figure(figsize=(14, 5))

# 3D surface
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x,y)')
ax1.set_title('f(x,y) = x² + xy + y²')

# Contour plot with gradient
ax2 = fig.add_subplot(132)
contour = ax2.contour(X, Y, Z, levels=20)
ax2.clabel(contour, inline=True, fontsize=8)
ax2.quiver(x0, y0, df_dx(x0, y0), df_dy(x0, y0), color='r', scale=20)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Contours & Gradient Vector')

plt.tight_layout()
plt.show()
\`\`\`

## Computing Partial Derivatives

\`\`\`python
import sympy as sp

# Symbolic computation
x, y = sp.symbols('x y')
f = x**3 * y**2 + sp.sin(x*y)

print(f"f(x,y) = {f}")
print(f"∂f/∂x = {sp.diff(f, x)}")
print(f"∂f/∂y = {sp.diff(f, y)}")

# Numerical partial derivatives
def numerical_partial_x(f, x, y, h=1e-7):
    return (f(x + h, y) - f(x, y)) / h

def numerical_partial_y(f, x, y, h=1e-7):
    return (f(x, y + h) - f(x, y)) / h

# Test
def test_f(x, y):
    return np.sin(x) * np.cos(y)

x_test, y_test = 1.0, 2.0
print(f"\\nNumerical partials at ({x_test}, {y_test}):")
print(f"∂f/∂x ≈ {numerical_partial_x(test_f, x_test, y_test)}")
print(f"∂f/∂y ≈ {numerical_partial_y(test_f, x_test, y_test)}")

# Analytical
print(f"Analytical:")
print(f"∂f/∂x = {np.cos(x_test) * np.cos(y_test)}")
print(f"∂f/∂y = {-np.sin(x_test) * np.sin(y_test)}")
\`\`\`

## Gradient Vector

The **gradient** ∇f is the vector of all partial derivatives:

∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]

\`\`\`python
def gradient(f, point, h=1e-7):
    """Compute gradient numerically"""
    point = np.array(point, dtype=float)
    grad = np.zeros_like(point)
    
    for i in range(len(point)):
        point_plus = point.copy()
        point_plus[i] += h
        grad[i] = (f(point_plus) - f(point)) / h
    
    return grad

# Example: f(x,y,z) = x²y + yz² + z³
def f_3d(xyz):
    x, y, z = xyz
    return x**2 * y + y * z**2 + z**3

point = np.array([1, 2, 3])
grad = gradient(f_3d, point)
print(f"∇f at {point} = {grad}")
\`\`\`

## Applications in ML

\`\`\`python
# Neural network gradient example
class SimpleMLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros(output_size)
    
    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2
    
    def compute_gradients(self, X, y_true):
        """Compute all partial derivatives"""
        batch_size = X.shape[0]
        y_pred = self.forward(X)
        
        # Gradient of loss wrt final output
        dL_dz2 = (y_pred - y_true) / batch_size
        
        # Partial derivatives wrt W2 and b2
        dL_dW2 = self.a1.T @ dL_dz2
        dL_db2 = np.sum(dL_dz2, axis=0)
        
        # Backprop to hidden layer
        dL_da1 = dL_dz2 @ self.W2.T
        dL_dz1 = dL_da1 * (self.z1 > 0)  # ReLU derivative
        
        # Partial derivatives wrt W1 and b1
        dL_dW1 = X.T @ dL_dz1
        dL_db1 = np.sum(dL_dz1, axis=0)
        
        return {
            'dL_dW1': dL_dW1,
            'dL_db1': dL_db1,
            'dL_dW2': dL_dW2,
            'dL_db2': dL_db2
        }

# Test
model = SimpleMLP(input_size=2, hidden_size=3, output_size=1)
X = np.random.randn(10, 2)
y = np.random.randn(10, 1)

grads = model.compute_gradients(X, y)
print("Gradients (partial derivatives):")
for name, grad in grads.items():
    print(f"{name}: shape {grad.shape}, norm {np.linalg.norm(grad):.4f}")
\`\`\`

## Summary

- Partial derivatives: ∂f/∂xᵢ measures change wrt one variable
- Gradient ∇f is vector of all partials
- Essential for multivariable optimization in ML
`,
};
