/**
 * Limits & Continuity Section
 */

export const limitscontinuitySection = {
  id: 'limits-continuity',
  title: 'Limits & Continuity',
  content: `
# Limits & Continuity

## Introduction

Limits and continuity are the foundational concepts of calculus. They allow us to understand the behavior of functions as inputs approach specific values, which is essential for derivatives (rates of change) and integrals (accumulation). In machine learning, limits help us understand convergence of optimization algorithms and the behavior of activation functions at boundaries.

## What is a Limit?

A **limit** describes the value a function f (x) approaches as x gets arbitrarily close to some value a.

**Notation**: \`lim_{x → a} f (x) = L\`

This reads: "The limit of f (x) as x approaches a is L"

**Intuition**: As x gets closer and closer to a, f (x) gets closer and closer to L.

### Example: Limits by Numerical Approximation

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt

# Example: Find lim_{x → 2} (x^2 - 4)/(x - 2)
def f (x):
    return (x**2 - 4) / (x - 2)

# Approach from the left (x < 2)
x_left = [1.9, 1.99, 1.999, 1.9999]
for x in x_left:
    print(f"f({x}) = {f (x)}")

# Approach from the right (x > 2)
x_right = [2.1, 2.01, 2.001, 2.0001]
for x in x_right:
    print(f"\\nf({x}) = {f (x)}")

# Visualization
x = np.linspace(0, 4, 1000)
x = x[x != 2]  # Remove x = 2 (undefined)
y = (x**2 - 4) / (x - 2)

plt.figure (figsize=(10, 6))
plt.plot (x, y, 'b-', label='f (x) = (x² - 4)/(x - 2)')
plt.axhline (y=4, color='r', linestyle='--', label='Limit = 4')
plt.axvline (x=2, color='g', linestyle='--', alpha=0.3, label='x = 2 (undefined)')
plt.scatter([2], [4], color='r', s=100, zorder=5, facecolors='none', edgecolors='r')
plt.xlabel('x')
plt.ylabel('f (x)')
plt.title('Limit as x approaches 2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-2, 8)
plt.show()
\`\`\`

**Output**: Both left and right approaches converge to 4, so \`lim_{x → 2} f (x) = 4\`.

Note: f(2) is undefined (division by zero), but the limit exists!

## Formal Definition of a Limit

**Epsilon-Delta Definition**: \`lim_{x → a} f (x) = L\` means:

For every ε > 0, there exists a δ > 0 such that:
- If 0 < |x - a| < δ, then |f (x) - L| < ε

**Translation**: No matter how small a tolerance (ε) we choose for f (x) being close to L, we can find a corresponding range (δ) around a where all x values (except a itself) produce f (x) within that tolerance.

\`\`\`python
def verify_limit (f, a, L, epsilon=0.1):
    """
    Verify limit using epsilon-delta definition
    """
    # Try to find delta
    delta = epsilon / 10  # Initial guess
    
    # Test points around a
    test_points = np.linspace (a - delta, a + delta, 20)
    test_points = test_points[test_points != a]  # Exclude a itself
    
    for x in test_points:
        fx = f (x)
        if abs (fx - L) >= epsilon:
            return False, f"Failed at x={x}, f (x)={fx}, |f (x) - L| = {abs (fx - L)}"
    
    return True, f"Limit verified for ε={epsilon}, δ={delta}"

# Example
def g (x):
    return 2*x + 1

result, message = verify_limit (g, 3, 7, epsilon=0.01)
print(f"Verification: {result}")
print(message)
\`\`\`

## One-Sided Limits

**Left-hand limit**: \`lim_{x → a⁻} f (x)\` (x approaches a from left)
**Right-hand limit**: \`lim_{x → a⁺} f (x)\` (x approaches a from right)

A limit exists if and only if both one-sided limits exist and are equal.

\`\`\`python
def piecewise_function (x):
    """Example of function with different one-sided limits"""
    if x < 0:
        return x**2
    elif x >= 0:
        return x + 1

# Evaluate limits at x = 0
x_left = np.array([-0.1, -0.01, -0.001])
x_right = np.array([0, 0.001, 0.01, 0.1])

print("Left-hand limit (x → 0⁻):")
for x in x_left:
    print(f"  f({x}) = {piecewise_function (x)}")

print("\\nRight-hand limit (x → 0⁺):")
for x in x_right:
    print(f"  f({x}) = {piecewise_function (x)}")

# Visualization
x_neg = np.linspace(-2, 0, 100)
x_pos = np.linspace(0, 2, 100)

plt.figure (figsize=(10, 6))
plt.plot (x_neg, x_neg**2, 'b-', label='f (x) = x² (x < 0)')
plt.plot (x_pos, x_pos + 1, 'r-', label='f (x) = x + 1 (x ≥ 0)')
plt.scatter([0], [0], color='b', s=100, zorder=5, facecolors='none', edgecolors='b')
plt.scatter([0], [1], color='r', s=100, zorder=5)
plt.xlabel('x')
plt.ylabel('f (x)')
plt.title('Piecewise Function with Jump Discontinuity')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline (y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline (x=0, color='k', linestyle='-', alpha=0.3)
plt.show()
\`\`\`

## Limits at Infinity

**Horizontal asymptotes**: Behavior of f (x) as x → ±∞

\`\`\`python
# Example: lim_{x → ∞} (3x² + 2x + 1)/(x² + 1)
def rational_function (x):
    return (3*x**2 + 2*x + 1) / (x**2 + 1)

x_values = [10, 100, 1000, 10000]
for x in x_values:
    print(f"f({x}) = {rational_function (x)}")

# As x → ∞, the limit is 3 (ratio of leading coefficients)

# Visualization
x = np.linspace(0, 100, 1000)
y = rational_function (x)

plt.figure (figsize=(10, 6))
plt.plot (x, y, 'b-', label='f (x) = (3x² + 2x + 1)/(x² + 1)')
plt.axhline (y=3, color='r', linestyle='--', label='Horizontal Asymptote: y = 3')
plt.xlabel('x')
plt.ylabel('f (x)')
plt.title('Limit at Infinity')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 5)
plt.show()
\`\`\`

**Key technique**: For rational functions, divide numerator and denominator by highest power of x:

\`\`\`
lim_{x → ∞} (3x² + 2x + 1)/(x² + 1) = lim_{x → ∞} (3 + 2/x + 1/x²)/(1 + 1/x²) = 3/1 = 3
\`\`\`

## Continuity

A function f is **continuous** at x = a if:
1. f (a) is defined
2. \`lim_{x → a} f (x)\` exists
3. \`lim_{x → a} f (x) = f (a)\`

**Intuition**: You can draw the function without lifting your pencil.

### Types of Discontinuities

1. **Removable discontinuity**: Limit exists but doesn't equal f (a)
2. **Jump discontinuity**: Left and right limits differ
3. **Infinite discontinuity**: Function approaches ±∞

\`\`\`python
def check_continuity (f, a, epsilon=1e-6):
    """
    Check if function is continuous at point a
    """
    try:
        # Check if f (a) is defined
        fa = f (a)
        print(f"f({a}) = {fa} ✓ (defined)")
        
        # Check left and right limits
        left_limit = f (a - epsilon)
        right_limit = f (a + epsilon)
        
        print(f"Left limit: {left_limit}")
        print(f"Right limit: {right_limit}")
        
        # Check if limits are close
        if abs (left_limit - right_limit) < 0.01:
            avg_limit = (left_limit + right_limit) / 2
            print(f"Limit exists: {avg_limit} ✓")
            
            # Check if limit equals f (a)
            if abs (avg_limit - fa) < 0.01:
                print(f"Continuous at x = {a} ✓")
                return True
            else:
                print(f"Removable discontinuity at x = {a}")
                return False
        else:
            print(f"Jump discontinuity at x = {a}")
            return False
            
    except:
        print(f"f({a}) is undefined")
        return False

# Test examples
def continuous_func (x):
    return x**2 + 1

def removable_disc (x):
    if x != 2:
        return (x**2 - 4) / (x - 2)
    return 0  # Wrong value

print("Testing continuous function:")
check_continuity (continuous_func, 2)

print("\\n" + "="*50)
print("Testing removable discontinuity:")
check_continuity (removable_disc, 2)
\`\`\`

## Intermediate Value Theorem (IVT)

**Theorem**: If f is continuous on [a, b] and k is between f (a) and f (b), then there exists c ∈ [a, b] such that f (c) = k.

**Application**: Root finding, proving existence of solutions

\`\`\`python
def intermediate_value_theorem (f, a, b, target, tolerance=1e-6):
    """
    Use IVT to find x where f (x) ≈ target
    Binary search approach
    """
    fa, fb = f (a), f (b)
    
    # Check if target is between f (a) and f (b)
    if not (min (fa, fb) <= target <= max (fa, fb)):
        return None, "Target not in range [f (a), f (b)]"
    
    # Binary search
    iterations = 0
    while abs (b - a) > tolerance:
        mid = (a + b) / 2
        fmid = f (mid)
        
        if abs (fmid - target) < tolerance:
            return mid, f"Found at x = {mid}, f (x) = {fmid}, iterations: {iterations}"
        
        # Decide which half to search
        if (fmid - target) * (fa - target) < 0:
            b = mid
            fb = fmid
        else:
            a = mid
            fa = fmid
        
        iterations += 1
    
    return (a + b) / 2, f"Approximate solution found in {iterations} iterations"

# Example: Find root of f (x) = x^3 - 2x - 5
def cubic (x):
    return x**3 - 2*x - 5

root, message = intermediate_value_theorem (cubic, 2, 3, 0)
print(f"Root: {root}")
print(message)
print(f"Verification: f({root}) = {cubic (root)}")

# Visualization
x = np.linspace(1, 3, 1000)
y = cubic (x)

plt.figure (figsize=(10, 6))
plt.plot (x, y, 'b-', label='f (x) = x³ - 2x - 5')
plt.axhline (y=0, color='k', linestyle='-', alpha=0.3)
plt.scatter([root], [0], color='r', s=100, zorder=5, label=f'Root ≈ {root:.4f}')
plt.xlabel('x')
plt.ylabel('f (x)')
plt.title('Intermediate Value Theorem - Root Finding')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
\`\`\`

## Connection to Machine Learning

### 1. Activation Functions

Neural network activation functions must be well-behaved. Understanding their continuity and limits is crucial:

\`\`\`python
def sigmoid (x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))

def relu (x):
    """ReLU activation function"""
    return np.maximum(0, x)

def leaky_relu (x, alpha=0.01):
    """Leaky ReLU activation function"""
    return np.where (x > 0, x, alpha * x)

# Analyze limits
x_range = np.linspace(-10, 10, 1000)

plt.figure (figsize=(15, 5))

# Sigmoid
plt.subplot(131)
plt.plot (x_range, sigmoid (x_range), 'b-', label='Sigmoid')
plt.axhline (y=0, color='r', linestyle='--', alpha=0.5, label='lim_{x→-∞} = 0')
plt.axhline (y=1, color='g', linestyle='--', alpha=0.5, label='lim_{x→+∞} = 1')
plt.xlabel('x')
plt.ylabel('σ(x)')
plt.title('Sigmoid: Continuous everywhere')
plt.legend()
plt.grid(True, alpha=0.3)

# ReLU
plt.subplot(132)
plt.plot (x_range, relu (x_range), 'b-', label='ReLU')
plt.scatter([0], [0], color='r', s=100, zorder=5, label='Not differentiable at x=0')
plt.xlabel('x')
plt.ylabel('ReLU(x)')
plt.title('ReLU: Continuous but not differentiable at 0')
plt.legend()
plt.grid(True, alpha=0.3)

# Leaky ReLU
plt.subplot(133)
plt.plot (x_range, leaky_relu (x_range), 'b-', label='Leaky ReLU')
plt.xlabel('x')
plt.ylabel('Leaky ReLU(x)')
plt.title('Leaky ReLU: Continuous everywhere')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Sigmoid limits:")
print(f"  lim_(x→-∞) σ(x) = {sigmoid(-100):.10f} ≈ 0")
print(f"  lim_(x→+∞) σ(x) = {sigmoid(100):.10f} ≈ 1")
\`\`\`

### 2. Convergence of Optimization Algorithms

Gradient descent convergence relies on limits:

\`\`\`python
def gradient_descent_convergence (learning_rate=0.01, max_iterations=100):
    """
    Demonstrate convergence using limits
    Minimize f (x) = x^2
    """
    x = 10.0  # Initial point
    history = [x]
    
    for i in range (max_iterations):
        gradient = 2*x  # derivative of x^2
        x = x - learning_rate * gradient
        history.append (x)
        
        # Check convergence (limit reached)
        if abs (x) < 1e-6:
            print(f"Converged at iteration {i+1}")
            break
    
    return np.array (history)

# Test different learning rates
rates = [0.01, 0.1, 0.5]
plt.figure (figsize=(15, 5))

for idx, lr in enumerate (rates, 1):
    history = gradient_descent_convergence (learning_rate=lr)
    
    plt.subplot(1, 3, idx)
    plt.plot (history, 'b-o', markersize=4)
    plt.axhline (y=0, color='r', linestyle='--', label='Limit = 0')
    plt.xlabel('Iteration')
    plt.ylabel('x value')
    plt.title (f'Learning Rate = {lr}')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
\`\`\`

## Common Limit Techniques

\`\`\`python
import sympy as sp

# Define symbolic variable
x = sp.Symbol('x')

# Example 1: Direct substitution
expr1 = x**2 + 3*x + 2
limit1 = sp.limit (expr1, x, 2)
print(f"lim_(x→2) (x² + 3x + 2) = {limit1}")

# Example 2: Factoring
expr2 = (x**2 - 9) / (x - 3)
limit2 = sp.limit (expr2, x, 3)
print(f"lim_(x→3) (x² - 9)/(x - 3) = {limit2}")

# Example 3: L'Hôpital's rule (preview)
expr3 = sp.sin (x) / x
limit3 = sp.limit (expr3, x, 0)
print(f"lim_(x→0) sin (x)/x = {limit3}")

# Example 4: Conjugate multiplication
expr4 = (sp.sqrt (x + 1) - 1) / x
limit4 = sp.limit (expr4, x, 0)
print(f"lim_(x→0) (√(x+1) - 1)/x = {limit4}")
\`\`\`

## Best Practices & Common Pitfalls

### Best Practices:
1. **Always check if direct substitution works first**2. **Verify limits numerically before symbolic computation**3. **Check both one-sided limits for piecewise functions**4. **Use graphical visualization to understand behavior**5. **Be aware of floating-point precision issues**

### Common Pitfalls:
1. **Assuming f (a) = lim_{x→a} f (x)** (not always true!)
2. **Ignoring undefined points**3. **Confusing limit with function value**4. **Not checking continuity before applying theorems**5. **Numerical errors for limits at infinity**

\`\`\`python
# Pitfall example: Floating-point precision
def problematic_limit (x):
    return (1 + 1/x)**x  # Approaches e as x → ∞

# Bad: Using very large x directly
x_large = 1e16
result_bad = problematic_limit (x_large)
print(f"At x = {x_large}: {result_bad}")  # May be inaccurate

# Better: Use logarithm properties or symbolic computation
import sympy as sp
x_sym = sp.Symbol('x')
expr = (1 + 1/x_sym)**x_sym
limit_correct = sp.limit (expr, x_sym, sp.oo)
print(f"Correct limit: {limit_correct}")  # e
print(f"Numerical value of e: {np.e}")
\`\`\`

## Summary

**Key Takeaways**:
- Limits describe function behavior as inputs approach values
- Continuity requires: f (a) defined, limit exists, limit = f (a)
- One-sided limits can differ (jump discontinuities)
- IVT guarantees intermediate values for continuous functions
- Activation functions in neural networks rely on continuity
- Gradient descent convergence is fundamentally about limits

**ML Applications**:
- Understanding activation function boundaries
- Convergence analysis of optimization algorithms
- Numerical stability in computations
- Behavior of loss functions near optima
`,
};
