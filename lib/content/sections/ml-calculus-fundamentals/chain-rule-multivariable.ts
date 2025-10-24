/**
 * Chain Rule for Multiple Variables Section
 */

export const chainrulemultivariableSection = {
  id: 'chain-rule-multivariable',
  title: 'Chain Rule for Multiple Variables',
  content: `
# Chain Rule for Multiple Variables

## Introduction

The multivariable chain rule is the mathematical foundation of backpropagation in neural networks. Understanding how gradients flow through compositions of vector-valued functions is essential for deep learning.

## Single Variable Chain Rule Review

For y = f(g(x)):
dy/dx = f'(g(x)) · g'(x)

## Multivariable Chain Rule

For z = f(x, y) where x = g(t), y = h(t):

dz/dt = (∂f/∂x)(dx/dt) + (∂f/∂y)(dy/dt)

**Intuition**: Total derivative accounts for all paths from t to z.

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt

# Example: z = x² + y², where x = cos(t), y = sin(t)
# Find dz/dt

def f(x, y):
    """Function z = f(x,y)"""
    return x**2 + y**2

def df_dx(x, y):
    """∂f/∂x"""
    return 2*x

def df_dy(x, y):
    """∂f/∂y"""
    return 2*y

def x(t):
    """x = cos(t)"""
    return np.cos(t)

def y(t):
    """y = sin(t)"""
    return np.sin(t)

def dx_dt(t):
    """dx/dt"""
    return -np.sin(t)

def dy_dt(t):
    """dy/dt"""
    return np.cos(t)

def dz_dt_chain_rule(t):
    """
    Chain rule: dz/dt = (∂f/∂x)(dx/dt) + (∂f/∂y)(dy/dt)
    """
    x_val = x(t)
    y_val = y(t)
    return df_dx(x_val, y_val) * dx_dt(t) + df_dy(x_val, y_val) * dy_dt(t)

# Test at t = π/4
t_test = np.pi / 4
print(f"At t = {t_test:.4f}:")
print(f"x(t) = {x(t_test):.4f}")
print(f"y(t) = {y(t_test):.4f}")
print(f"z(t) = {f(x(t_test), y(t_test)):.4f}")
print(f"dz/dt = {dz_dt_chain_rule(t_test):.4f}")

# Verify numerically
h = 1e-7
numerical_derivative = (f(x(t_test + h), y(t_test + h)) - f(x(t_test), y(t_test))) / h
print(f"Numerical dz/dt = {numerical_derivative:.4f}")
print(f"Error: {abs(dz_dt_chain_rule(t_test) - numerical_derivative):.2e}")

# Note: Since z = x² + y² = cos²(t) + sin²(t) = 1, dz/dt = 0!
print(f"\\nExpected: dz/dt = 0 (constant function)")
\`\`\`

## General Multivariable Chain Rule

For z = f(x₁, ..., xₙ) where each xᵢ = gᵢ(t₁, ..., tₘ):

∂z/∂tⱼ = Σᵢ (∂f/∂xᵢ)(∂xᵢ/∂tⱼ)

**Matrix form** (Jacobian):
∂z/∂t = (∂f/∂x) · (∂x/∂t)

\`\`\`python
def jacobian_chain_rule():
    """
    Demonstrate Jacobian form of chain rule
    """
    # z = f(x, y) = x²y + y³
    # x = u + v, y = uv
    # Find ∂z/∂u and ∂z/∂v
    
    def f(x, y):
        return x**2 * y + y**3
    
    def df_dx(x, y):
        return 2*x*y
    
    def df_dy(x, y):
        return x**2 + 3*y**2
    
    def x_from_uv(u, v):
        return u + v
    
    def y_from_uv(u, v):
        return u * v
    
    # Jacobian of (x, y) with respect to (u, v)
    def dx_du(u, v):
        return 1
    
    def dx_dv(u, v):
        return 1
    
    def dy_du(u, v):
        return v
    
    def dy_dv(u, v):
        return u
    
    # Chain rule: ∂z/∂u = (∂z/∂x)(∂x/∂u) + (∂z/∂y)(∂y/∂u)
    def dz_du(u, v):
        x = x_from_uv(u, v)
        y = y_from_uv(u, v)
        return df_dx(x, y) * dx_du(u, v) + df_dy(x, y) * dy_du(u, v)
    
    def dz_dv(u, v):
        x = x_from_uv(u, v)
        y = y_from_uv(u, v)
        return df_dx(x, y) * dx_dv(u, v) + df_dy(x, y) * dy_dv(u, v)
    
    # Test
    u_test, v_test = 2.0, 3.0
    print(f"At (u, v) = ({u_test}, {v_test}):")
    print(f"∂z/∂u = {dz_du(u_test, v_test):.4f}")
    print(f"∂z/∂v = {dz_dv(u_test, v_test):.4f}")
    
    # Verify numerically
    h = 1e-7
    def z(u, v):
        return f(x_from_uv(u, v), y_from_uv(u, v))
    
    numerical_du = (z(u_test + h, v_test) - z(u_test, v_test)) / h
    numerical_dv = (z(u_test, v_test + h) - z(u_test, v_test)) / h
    
    print(f"\\nNumerical verification:")
    print(f"∂z/∂u (numerical) = {numerical_du:.4f}")
    print(f"∂z/∂v (numerical) = {numerical_dv:.4f}")

jacobian_chain_rule()
\`\`\`

## Backpropagation as Chain Rule

Neural networks are function compositions: y = f_L(...f_2(f_1(x)))

**Forward pass**: Compute outputs layer by layer
**Backward pass**: Apply chain rule to compute gradients

\`\`\`python
class NeuralNetworkChainRule:
    """
    Demonstrate backpropagation as repeated chain rule application
    """
    def __init__(self):
        # Simple 2-layer network
        np.random.seed(42)
        self.W1 = np.random.randn(2, 3) * 0.1
        self.b1 = np.zeros(3)
        self.W2 = np.random.randn(3, 1) * 0.1
        self.b2 = np.zeros(1)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def forward(self, X):
        """Forward pass - save intermediates for backprop"""
        self.X = X
        
        # Layer 1
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Layer 2
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2
    
    def backward(self, y_true):
        """
        Backpropagation: Chain rule application
        
        Network: X → z1 → a1 → z2 → a2 → Loss
        
        Chain rule paths:
        ∂L/∂W2 = ∂L/∂a2 · ∂a2/∂z2 · ∂z2/∂W2
        ∂L/∂W1 = ∂L/∂a2 · ∂a2/∂z2 · ∂z2/∂a1 · ∂a1/∂z1 · ∂z1/∂W1
        """
        batch_size = self.X.shape[0]
        
        # Loss: L = (a2 - y)²
        # ∂L/∂a2 = 2(a2 - y)
        dL_da2 = 2 * (self.a2 - y_true) / batch_size
        
        # Chain rule: ∂L/∂z2 = ∂L/∂a2 · ∂a2/∂z2
        da2_dz2 = self.sigmoid_derivative(self.z2)
        dL_dz2 = dL_da2 * da2_dz2
        
        # Chain rule: ∂L/∂W2 = ∂L/∂z2 · ∂z2/∂W2
        # Since z2 = a1 @ W2 + b2, ∂z2/∂W2 = a1
        dL_dW2 = self.a1.T @ dL_dz2
        dL_db2 = np.sum(dL_dz2, axis=0)
        
        # Chain rule: ∂L/∂a1 = ∂L/∂z2 · ∂z2/∂a1
        # Since z2 = a1 @ W2, ∂z2/∂a1 = W2
        dL_da1 = dL_dz2 @ self.W2.T
        
        # Chain rule: ∂L/∂z1 = ∂L/∂a1 · ∂a1/∂z1
        da1_dz1 = self.sigmoid_derivative(self.z1)
        dL_dz1 = dL_da1 * da1_dz1
        
        # Chain rule: ∂L/∂W1 = ∂L/∂z1 · ∂z1/∂W1
        dL_dW1 = self.X.T @ dL_dz1
        dL_db1 = np.sum(dL_dz1, axis=0)
        
        return {
            'dL_dW1': dL_dW1,
            'dL_db1': dL_db1,
            'dL_dW2': dL_dW2,
            'dL_db2': dL_db2
        }
    
    def numerical_gradient(self, X, y_true, param_name, h=1e-5):
        """Compute gradient numerically for verification"""
        param = getattr(self, param_name)
        grad = np.zeros_like(param)
        
        it = np.nditer(param, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            old_value = param[idx]
            
            # f(x + h)
            param[idx] = old_value + h
            y_pred_plus = self.forward(X)
            loss_plus = np.mean((y_pred_plus - y_true)**2)
            
            # f(x - h)
            param[idx] = old_value - h
            y_pred_minus = self.forward(X)
            loss_minus = np.mean((y_pred_minus - y_true)**2)
            
            # Central difference
            grad[idx] = (loss_plus - loss_minus) / (2 * h)
            
            param[idx] = old_value
            it.iternext()
        
        return grad

# Test backpropagation
nn = NeuralNetworkChainRule()
X = np.random.randn(10, 2)
y_true = np.random.randn(10, 1)

# Forward and backward
y_pred = nn.forward(X)
analytical_grads = nn.backward(y_true)

print("Backpropagation Gradient Check:")
print("="*60)

for param_name in ['W1', 'W2']:
    analytical = analytical_grads[f'dL_d{param_name}']
    numerical = nn.numerical_gradient(X, y_true, param_name)
    
    diff = np.linalg.norm(analytical - numerical)
    norm_sum = np.linalg.norm(analytical) + np.linalg.norm(numerical)
    relative_error = diff / (norm_sum + 1e-8)
    
    print(f"\\n{param_name}:")
    print(f"  Analytical grad norm: {np.linalg.norm(analytical):.6f}")
    print(f"  Numerical grad norm:  {np.linalg.norm(numerical):.6f}")
    print(f"  Relative error: {relative_error:.2e}")
    
    if relative_error < 1e-5:
        print(f"  ✓ Gradient correct!")
    else:
        print(f"  ✗ Gradient may have error")
\`\`\`

## Computational Graph

Modern deep learning frameworks use computational graphs to automatically apply chain rule.

\`\`\`python
class ComputationNode:
    """Node in a computational graph"""
    def __init__(self, value, children=(), op='):
        self.value = value
        self.grad = 0
        self.children = children
        self.op = op
        self._backward = lambda: None
    
    def __repr__(self):
        return f"Node(value={self.value:.4f}, grad={self.grad:.4f})"
    
    def __add__(self, other):
        other = other if isinstance(other, ComputationNode) else ComputationNode(other)
        out = ComputationNode(self.value + other.value, (self, other), '+')
        
        def _backward():
            # Chain rule: ∂L/∂self = ∂L/∂out · ∂out/∂self = ∂L/∂out · 1
            self.grad += out.grad
            other.grad += out.grad
        
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, ComputationNode) else ComputationNode(other)
        out = ComputationNode(self.value * other.value, (self, other), '*')
        
        def _backward():
            # Chain rule: ∂L/∂self = ∂L/∂out · ∂out/∂self = ∂L/∂out · other
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad
        
        out._backward = _backward
        return out
    
    def exp(self):
        out = ComputationNode(np.exp(self.value), (self,), 'exp')
        
        def _backward():
            # Chain rule: ∂L/∂self = ∂L/∂out · ∂out/∂self = ∂L/∂out · exp(self)
            self.grad += np.exp(self.value) * out.grad
        
        out._backward = _backward
        return out
    
    def backward(self):
        """Topological sort and apply chain rule"""
        topo = []
        visited = set()
        
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node.children:
                    build_topo(child)
                topo.append(node)
        
        build_topo(self)
        
        self.grad = 1  # dL/dL = 1
        for node in reversed(topo):
            node._backward()

# Example: f(x, y) = (x + y) * exp(x)
x = ComputationNode(2.0)
y = ComputationNode(3.0)

# Forward pass
z1 = x + y
z2 = x.exp()
f = z1 * z2

print("Computational Graph Example:")
print(f"x = {x.value}, y = {y.value}")
print(f"f(x, y) = (x + y) * exp(x) = {f.value:.4f}")

# Backward pass (automatic differentiation)
f.backward()

print(f"\\nGradients (via automatic differentiation):")
print(f"∂f/∂x = {x.grad:.4f}")
print(f"∂f/∂y = {y.grad:.4f}")

# Verify analytically
# f = (x + y) * exp(x)
# ∂f/∂x = exp(x) + (x + y) * exp(x) = (1 + x + y) * exp(x)
# ∂f/∂y = exp(x)
analytical_dx = (1 + x.value + y.value) * np.exp(x.value)
analytical_dy = np.exp(x.value)

print(f"\\nAnalytical verification:")
print(f"∂f/∂x (analytical) = {analytical_dx:.4f}")
print(f"∂f/∂y (analytical) = {analytical_dy:.4f}")
print(f"Error in ∂f/∂x: {abs(x.grad - analytical_dx):.2e}")
print(f"Error in ∂f/∂y: {abs(y.grad - analytical_dy):.2e}")
\`\`\`

## Vector-Valued Functions

For **F**: ℝⁿ → ℝᵐ, the chain rule involves Jacobian matrices.

If z = g(**F**(x)), then:
dz/dx = (∂g/∂**F**) · J_**F**

where J_**F** is the Jacobian matrix of **F**.

\`\`\`python
def vector_chain_rule_example():
    """
    Example with vector-valued function
    F: ℝ² → ℝ³, g: ℝ³ → ℝ
    """
    # F(x, y) = [x², xy, y²]
    def F(x, y):
        return np.array([x**2, x*y, y**2])
    
    # Jacobian of F
    def jacobian_F(x, y):
        return np.array([
            [2*x, 0],      # ∂F₁/∂x, ∂F₁/∂y
            [y, x],        # ∂F₂/∂x, ∂F₂/∂y
            [0, 2*y]       # ∂F₃/∂x, ∂F₃/∂y
        ])
    
    # g(u, v, w) = u + 2v + 3w
    def g(uvw):
        u, v, w = uvw
        return u + 2*v + 3*w
    
    # Gradient of g
    def grad_g(uvw):
        return np.array([1, 2, 3])
    
    # Chain rule: ∇_xy (g∘F) = J_F^T · ∇_uvw g
    def gradient_composition(x, y):
        uvw = F(x, y)
        J_F = jacobian_F(x, y)
        grad_g_at_F = grad_g(uvw)
        
        # Chain rule (matrix multiplication)
        return J_F.T @ grad_g_at_F
    
    # Test
    x_test, y_test = 2.0, 3.0
    grad = gradient_composition(x_test, y_test)
    
    print("Vector-Valued Chain Rule:")
    print(f"At (x, y) = ({x_test}, {y_test})")
    print(f"F(x, y) = {F(x_test, y_test)}")
    print(f"g(F(x, y)) = {g(F(x_test, y_test)):.4f}")
    print(f"\\n∇(g∘F) = {grad}")
    
    # Verify numerically
    def composed(x, y):
        return g(F(x, y))
    
    h = 1e-7
    numerical_dx = (composed(x_test + h, y_test) - composed(x_test, y_test)) / h
    numerical_dy = (composed(x_test, y_test + h) - composed(x_test, y_test)) / h
    
    print(f"\\nNumerical verification:")
    print(f"∂(g∘F)/∂x = {numerical_dx:.4f} (analytical: {grad[0]:.4f})")
    print(f"∂(g∘F)/∂y = {numerical_dy:.4f} (analytical: {grad[1]:.4f})")

vector_chain_rule_example()
\`\`\`

## Summary

**Key Concepts**:
- Multivariable chain rule: Sum over all paths
- Backpropagation = Repeated chain rule application
- Jacobian matrices for vector-valued functions
- Computational graphs automate chain rule
- Forward pass: Compute values
- Backward pass: Compute gradients via chain rule

**Why This Matters**:
Without the chain rule, we couldn't train deep networks. It's the mathematical foundation of backpropagation, enabling gradient-based optimization of composed functions.
`,
};
