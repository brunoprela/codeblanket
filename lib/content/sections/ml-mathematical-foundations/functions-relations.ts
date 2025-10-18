/**
 * Functions & Relations Section
 */

export const functionsrelationsSection = {
  id: 'functions-relations',
  title: 'Functions & Relations',
  content: `
# Functions & Relations

## Introduction

Functions are the foundation of machine learning. Every ML model is essentially a function that maps inputs to outputs. Understanding function notation, properties, and types is crucial for grasping how neural networks, loss functions, and activation functions work.

## Function Notation and Domain/Range

### Definition

A **function** is a relation that assigns exactly one output to each input.

**Notation**: f(x) = y
- f: function name
- x: input (independent variable)
- y: output (dependent variable)

**Domain**: Set of all possible input values
**Range**: Set of all possible output values

**Example**:
\`\`\`
f(x) = 2x + 1
Domain: all real numbers ℝ
Range: all real numbers ℝ
f(3) = 2(3) + 1 = 7
\`\`\`

### Python Implementation

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt

# Define a function
def f(x):
    """Simple linear function"""
    return 2*x + 1

# Evaluate function at specific points
x_values = np.array([0, 1, 2, 3, 4])
y_values = f(x_values)

print("x:", x_values)
print("f(x):", y_values)

# Visualize
plt.figure(figsize=(8, 5))
plt.plot(x_values, y_values, 'bo-', label='f(x) = 2x + 1')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Linear Function')
plt.legend()
plt.grid(True)
plt.show()
\`\`\`

### ML Context: Hypothesis Function

In machine learning, we call our model a **hypothesis function**:

\`\`\`python
# Linear regression hypothesis
def h_theta(x, theta_0, theta_1):
    """
    Hypothesis function for linear regression
    h_θ(x) = θ₀ + θ₁x
    """
    return theta_0 + theta_1 * x

# Example: predicting house prices
# x = square footage, y = price
theta_0 = 50000  # base price
theta_1 = 100    # price per sq ft

square_footage = np.array([1000, 1500, 2000, 2500])
predicted_prices = h_theta(square_footage, theta_0, theta_1)

print("Square Footage:", square_footage)
print("Predicted Prices:", predicted_prices)

# Vectorized version (more efficient)
def h_theta_vectorized(X, theta):
    """
    X: feature matrix (with intercept column)
    theta: parameter vector
    """
    return X @ theta

# Add intercept column
X = np.c_[np.ones(len(square_footage)), square_footage]
theta = np.array([theta_0, theta_1])
predicted_prices_vec = h_theta_vectorized(X, theta)
print("\\nVectorized predictions:", predicted_prices_vec)
\`\`\`

## Types of Functions

### Linear Functions

**Form**: f(x) = mx + b
- m: slope
- b: y-intercept

**Properties**:
- Constant rate of change
- Graph is a straight line

\`\`\`python
def plot_linear_functions():
    x = np.linspace(-5, 5, 100)
    
    plt.figure(figsize=(10, 6))
    
    # Different slopes
    plt.plot(x, 2*x + 1, label='f(x) = 2x + 1', linewidth=2)
    plt.plot(x, -x + 3, label='f(x) = -x + 3', linewidth=2)
    plt.plot(x, 0.5*x - 2, label='f(x) = 0.5x - 2', linewidth=2)
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Linear Functions')
    plt.legend()
    plt.grid(True)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.show()

plot_linear_functions()
print("Linear functions plotted")
\`\`\`

**ML Application**: Linear regression, linear layers in neural networks

### Quadratic Functions

**Form**: f(x) = ax² + bx + c
- a: determines concavity (a > 0: opens up, a < 0: opens down)
- Vertex at x = -b/(2a)

**Properties**:
- Parabolic shape
- One global extremum (min or max)

\`\`\`python
def plot_quadratic_functions():
    x = np.linspace(-5, 5, 100)
    
    plt.figure(figsize=(10, 6))
    
    # Different quadratics
    plt.plot(x, x**2, label='f(x) = x²', linewidth=2)
    plt.plot(x, -x**2 + 4, label='f(x) = -x² + 4', linewidth=2)
    plt.plot(x, 0.5*x**2 - 2*x + 1, label='f(x) = 0.5x² - 2x + 1', linewidth=2)
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Quadratic Functions')
    plt.legend()
    plt.grid(True)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.ylim(-5, 5)
    plt.show()

plot_quadratic_functions()
print("Quadratic functions plotted")
\`\`\`

**ML Application**: Convex optimization, loss function landscapes

### Polynomial Functions

**Form**: f(x) = aₙxⁿ + aₙ₋₁xⁿ⁻¹ + ... + a₁x + a₀
- n: degree of polynomial
- aᵢ: coefficients

\`\`\`python
from numpy.polynomial import Polynomial

# Create polynomial: 2x³ - 3x² + x - 5
coefficients = [-5, 1, -3, 2]  # constant to highest degree
poly = Polynomial(coefficients)

x = np.linspace(-3, 3, 100)
y = poly(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, linewidth=2)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Polynomial: f(x) = 2x³ - 3x² + x - 5')
plt.grid(True)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.show()

print(f"Polynomial: {poly}")
print(f"Degree: {poly.degree()}")
\`\`\`

**ML Application**: Polynomial regression, feature engineering with polynomial features

\`\`\`python
from sklearn.preprocessing import PolynomialFeatures

# Polynomial feature expansion
X = np.array([[2], [3], [4]])
poly_features = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly_features.fit_transform(X)

print("Original features:\\n", X)
print("\\nPolynomial features (degree 3):\\n", X_poly)
print("\\nFeature names:", poly_features.get_feature_names_out(['x']))
\`\`\`

### Exponential Functions

**Form**: f(x) = a·bˣ or f(x) = a·eˣ
- Base b > 1: exponential growth
- Base 0 < b < 1: exponential decay
- e ≈ 2.71828: natural exponential base

**Properties**:
- Always positive (for real inputs)
- Rapid growth or decay
- Never touches x-axis

\`\`\`python
def plot_exponential_functions():
    x = np.linspace(-2, 3, 100)
    
    plt.figure(figsize=(10, 6))
    
    # Growth and decay
    plt.plot(x, np.exp(x), label='f(x) = eˣ (growth)', linewidth=2)
    plt.plot(x, np.exp(-x), label='f(x) = e⁻ˣ (decay)', linewidth=2)
    plt.plot(x, 2**x, label='f(x) = 2ˣ', linewidth=2)
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Exponential Functions')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 10)
    plt.show()

plot_exponential_functions()
print("Exponential functions plotted")
\`\`\`

**ML Application**: Softmax activation, exponential learning rate decay

\`\`\`python
def softmax(x):
    """Softmax activation function"""
    exp_x = np.exp(x - np.max(x))  # subtract max for numerical stability
    return exp_x / exp_x.sum()

# Example: converting logits to probabilities
logits = np.array([2.0, 1.0, 0.1])
probabilities = softmax(logits)

print("Logits:", logits)
print("Probabilities:", probabilities)
print("Sum:", probabilities.sum())  # Should be 1.0
\`\`\`

### Logarithmic Functions

**Form**: f(x) = logₐ(x) or f(x) = ln(x)
- Inverse of exponential function
- Domain: x > 0
- Range: all real numbers

**Properties**:
- Slow growth
- Undefined for x ≤ 0
- log(ab) = log(a) + log(b)
- log(aⁿ) = n·log(a)

\`\`\`python
def plot_logarithmic_functions():
    x = np.linspace(0.1, 10, 100)
    
    plt.figure(figsize=(10, 6))
    
    # Different bases
    plt.plot(x, np.log(x), label='f(x) = ln(x) (natural log)', linewidth=2)
    plt.plot(x, np.log10(x), label='f(x) = log₁₀(x)', linewidth=2)
    plt.plot(x, np.log2(x), label='f(x) = log₂(x)', linewidth=2)
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Logarithmic Functions')
    plt.legend()
    plt.grid(True)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=1, color='k', linewidth=0.5)
    plt.show()

plot_logarithmic_functions()
print("Logarithmic functions plotted")
\`\`\`

**ML Application**: Log loss (cross-entropy), log-likelihood

\`\`\`python
def binary_cross_entropy(y_true, y_pred, epsilon=1e-10):
    """
    Binary cross-entropy loss
    Uses logarithms to penalize wrong predictions
    """
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Example
y_true = np.array([1, 0, 1, 1, 0])
y_pred = np.array([0.9, 0.1, 0.8, 0.7, 0.2])
loss = binary_cross_entropy(y_true, y_pred)
print(f"Binary Cross-Entropy Loss: {loss:.4f}")
\`\`\`

## Inverse Functions

### Definition

If f(x) = y, then f⁻¹(y) = x

**Properties**:
- f(f⁻¹(x)) = x
- f⁻¹(f(x)) = x
- Graph of f⁻¹ is reflection of f across y = x line

\`\`\`python
# Example: f(x) = 2x + 1
def f(x):
    return 2*x + 1

# Inverse: f⁻¹(x) = (x - 1) / 2
def f_inverse(x):
    return (x - 1) / 2

# Verify
x_test = 5
print(f"f({x_test}) = {f(x_test)}")
print(f"f⁻¹(f({x_test})) = {f_inverse(f(x_test))}")

# Visualize
x = np.linspace(-5, 5, 100)
plt.figure(figsize=(8, 8))
plt.plot(x, f(x), label='f(x) = 2x + 1', linewidth=2)
plt.plot(x, f_inverse(x), label='f⁻¹(x) = (x-1)/2', linewidth=2)
plt.plot(x, x, 'k--', label='y = x', linewidth=1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Function and Its Inverse')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
\`\`\`

**ML Application**: Inverse transformations in autoencoders, invertible neural networks

## Composition of Functions

### Definition

**Composition**: (f ∘ g)(x) = f(g(x))

First apply g, then apply f to the result.

\`\`\`python
# Example: f(x) = x² and g(x) = x + 1
def f(x):
    return x**2

def g(x):
    return x + 1

def compose(f, g):
    """Return the composition f ∘ g"""
    return lambda x: f(g(x))

# f(g(x)) = (x + 1)²
f_compose_g = compose(f, g)

x_test = 3
print(f"f(x) = x²")
print(f"g(x) = x + 1")
print(f"(f ∘ g)({x_test}) = f(g({x_test})) = f({g(x_test)}) = {f_compose_g(x_test)}")

# Note: composition is NOT commutative
g_compose_f = compose(g, f)
print(f"\\n(g ∘ f)({x_test}) = g(f({x_test})) = g({f(x_test)}) = {g_compose_f(x_test)}")
print(f"(f ∘ g) ≠ (g ∘ f): {f_compose_g(x_test)} ≠ {g_compose_f(x_test)}")
\`\`\`

**ML Application**: Neural network layers (function composition!)

\`\`\`python
# Neural network as function composition
def layer1(x, W1, b1):
    """First layer: linear transformation"""
    return x @ W1 + b1

def activation_relu(x):
    """ReLU activation"""
    return np.maximum(0, x)

def layer2(x, W2, b2):
    """Second layer: linear transformation"""
    return x @ W2 + b2

def neural_network(x, W1, b1, W2, b2):
    """
    Two-layer neural network as composition:
    f(x) = layer2(ReLU(layer1(x)))
    """
    h1 = layer1(x, W1, b1)
    h1_activated = activation_relu(h1)
    output = layer2(h1_activated, W2, b2)
    return output

# Example
x = np.array([[1, 2, 3]])  # 1 sample, 3 features
W1 = np.random.randn(3, 4)  # 3 inputs, 4 hidden units
b1 = np.random.randn(4)
W2 = np.random.randn(4, 2)  # 4 hidden, 2 outputs
b2 = np.random.randn(2)

output = neural_network(x, W1, b1, W2, b2)
print(f"Neural network output shape: {output.shape}")
print(f"Output: {output}")
\`\`\`

## Activation Functions in ML

Activation functions are crucial non-linear functions in neural networks:

### Sigmoid Function

**Formula**: σ(x) = 1 / (1 + e⁻ˣ)
- Range: (0, 1)
- Output can be interpreted as probability
- Smooth, differentiable

\`\`\`python
def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))

x = np.linspace(-10, 10, 100)
y = sigmoid(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, linewidth=2)
plt.xlabel('x')
plt.ylabel('σ(x)')
plt.title('Sigmoid Function: σ(x) = 1/(1 + e⁻ˣ)')
plt.grid(True)
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
plt.show()

# Derivative
def sigmoid_derivative(x):
    """Derivative of sigmoid: σ'(x) = σ(x)(1 - σ(x))"""
    s = sigmoid(x)
    return s * (1 - s)

print(f"σ(0) = {sigmoid(0)}")
print(f"σ'(0) = {sigmoid_derivative(0)}")
\`\`\`

### ReLU Function

**Formula**: ReLU(x) = max(0, x)
- Most common in modern deep learning
- Computationally efficient
- Helps with vanishing gradient problem

\`\`\`python
def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)

x = np.linspace(-5, 5, 100)
y = relu(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, linewidth=2)
plt.xlabel('x')
plt.ylabel('ReLU(x)')
plt.title('ReLU Function: max(0, x)')
plt.grid(True)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.show()

# Derivative
def relu_derivative(x):
    """Derivative of ReLU"""
    return (x > 0).astype(float)

print(f"ReLU(2) = {relu(2)}")
print(f"ReLU(-2) = {relu(-2)}")
\`\`\`

### Tanh Function

**Formula**: tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)
- Range: (-1, 1)
- Zero-centered (unlike sigmoid)
- Similar to sigmoid but symmetric

\`\`\`python
def tanh(x):
    """Tanh activation function"""
    return np.tanh(x)

x = np.linspace(-5, 5, 100)
y = tanh(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, linewidth=2)
plt.xlabel('x')
plt.ylabel('tanh(x)')
plt.title('Tanh Function')
plt.grid(True)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.show()

print(f"tanh(0) = {tanh(0)}")
print(f"tanh(2) = {tanh(2):.4f}")
print(f"tanh(-2) = {tanh(-2):.4f}")
\`\`\`

## Piecewise Functions

Functions defined differently on different intervals:

\`\`\`python
def piecewise_function(x):
    """
    f(x) = { x²     if x < 0
           { x      if 0 ≤ x < 2
           { 4      if x ≥ 2
    """
    return np.where(x < 0, x**2,
                    np.where(x < 2, x, 4))

x = np.linspace(-3, 4, 1000)
y = piecewise_function(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, linewidth=2)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Piecewise Function')
plt.grid(True)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
plt.axvline(x=2, color='r', linestyle='--', alpha=0.5)
plt.show()
\`\`\`

**ML Application**: ReLU and its variants are piecewise functions

## Function Transformations

Understanding how functions transform is crucial for feature engineering:

### Vertical Shift: f(x) + c

\`\`\`python
x = np.linspace(-5, 5, 100)
f_x = x**2

plt.figure(figsize=(10, 6))
plt.plot(x, f_x, label='f(x) = x²', linewidth=2)
plt.plot(x, f_x + 2, label='f(x) + 2', linewidth=2)
plt.plot(x, f_x - 3, label='f(x) - 3', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Vertical Shifts')
plt.legend()
plt.grid(True)
plt.show()
\`\`\`

### Horizontal Shift: f(x - c)

\`\`\`python
x = np.linspace(-10, 10, 100)
f_x = x**2

plt.figure(figsize=(10, 6))
plt.plot(x, f_x, label='f(x) = x²', linewidth=2)
plt.plot(x, (x - 2)**2, label='f(x - 2)', linewidth=2)
plt.plot(x, (x + 3)**2, label='f(x + 3)', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Horizontal Shifts')
plt.legend()
plt.grid(True)
plt.show()
\`\`\`

### Vertical Scaling: c·f(x)

\`\`\`python
x = np.linspace(-3, 3, 100)
f_x = x**2

plt.figure(figsize=(10, 6))
plt.plot(x, f_x, label='f(x) = x²', linewidth=2)
plt.plot(x, 2*f_x, label='2·f(x)', linewidth=2)
plt.plot(x, 0.5*f_x, label='0.5·f(x)', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Vertical Scaling')
plt.legend()
plt.grid(True)
plt.show()
\`\`\`

**ML Application**: Feature scaling and normalization

## Even and Odd Functions

### Even Functions: f(-x) = f(x)
- Symmetric about y-axis
- Examples: x², cos(x), |x|

### Odd Functions: f(-x) = -f(x)
- Symmetric about origin
- Examples: x, x³, sin(x)

\`\`\`python
x = np.linspace(-5, 5, 100)

plt.figure(figsize=(12, 5))

# Even function
plt.subplot(1, 2, 1)
plt.plot(x, x**2, linewidth=2)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Even Function: f(x) = x²')
plt.grid(True)
plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)

# Odd function
plt.subplot(1, 2, 2)
plt.plot(x, x**3, linewidth=2)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Odd Function: f(x) = x³')
plt.grid(True)
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# Check properties
x_test = 2
print(f"Even function: f({x_test}) = {x_test**2}, f({-x_test}) = {(-x_test)**2}")
print(f"Odd function: f({x_test}) = {x_test**3}, f({-x_test}) = {(-x_test)**3}")
\`\`\`

## Summary

- **Functions** map inputs to outputs following a rule
- **Domain** and **range** define valid inputs and possible outputs
- **Linear, quadratic, polynomial** functions model various relationships
- **Exponential** and **logarithmic** functions are inverses
- **Composition** chains functions together (crucial for neural networks)
- **Activation functions** (sigmoid, ReLU, tanh) add non-linearity to neural networks
- **Transformations** (shifts, scaling) are used in feature engineering
- Every ML model is essentially a function: y = f(x; θ)

These function concepts are the foundation for understanding:
- How neural networks work (composition of functions)
- Loss functions and their properties
- Activation functions and their role
- Feature transformations
- Model predictions as function evaluations
`,
};
