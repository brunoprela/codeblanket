import { Module } from '../types';

const mlCalculusFundamentals: Module = {
  id: 'ml-calculus-fundamentals',
  title: 'Calculus Fundamentals',
  description:
    'Master differential and integral calculus essential for understanding machine learning optimization and quantitative finance',
  icon: '📈',
  sections: [
    {
      id: 'limits-continuity',
      title: 'Limits & Continuity',
      content: `
# Limits & Continuity

## Introduction

Limits and continuity are the foundational concepts of calculus. They allow us to understand the behavior of functions as inputs approach specific values, which is essential for derivatives (rates of change) and integrals (accumulation). In machine learning, limits help us understand convergence of optimization algorithms and the behavior of activation functions at boundaries.

## What is a Limit?

A **limit** describes the value a function f(x) approaches as x gets arbitrarily close to some value a.

**Notation**: \`lim_{x → a} f(x) = L\`

This reads: "The limit of f(x) as x approaches a is L"

**Intuition**: As x gets closer and closer to a, f(x) gets closer and closer to L.

### Example: Limits by Numerical Approximation

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt

# Example: Find lim_{x → 2} (x^2 - 4)/(x - 2)
def f(x):
    return (x**2 - 4) / (x - 2)

# Approach from the left (x < 2)
x_left = [1.9, 1.99, 1.999, 1.9999]
for x in x_left:
    print(f"f({x}) = {f(x)}")

# Approach from the right (x > 2)
x_right = [2.1, 2.01, 2.001, 2.0001]
for x in x_right:
    print(f"\\nf({x}) = {f(x)}")

# Visualization
x = np.linspace(0, 4, 1000)
x = x[x != 2]  # Remove x = 2 (undefined)
y = (x**2 - 4) / (x - 2)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', label='f(x) = (x² - 4)/(x - 2)')
plt.axhline(y=4, color='r', linestyle='--', label='Limit = 4')
plt.axvline(x=2, color='g', linestyle='--', alpha=0.3, label='x = 2 (undefined)')
plt.scatter([2], [4], color='r', s=100, zorder=5, facecolors='none', edgecolors='r')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Limit as x approaches 2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-2, 8)
plt.show()
\`\`\`

**Output**: Both left and right approaches converge to 4, so \`lim_{x → 2} f(x) = 4\`.

Note: f(2) is undefined (division by zero), but the limit exists!

## Formal Definition of a Limit

**Epsilon-Delta Definition**: \`lim_{x → a} f(x) = L\` means:

For every ε > 0, there exists a δ > 0 such that:
- If 0 < |x - a| < δ, then |f(x) - L| < ε

**Translation**: No matter how small a tolerance (ε) we choose for f(x) being close to L, we can find a corresponding range (δ) around a where all x values (except a itself) produce f(x) within that tolerance.

\`\`\`python
def verify_limit(f, a, L, epsilon=0.1):
    """
    Verify limit using epsilon-delta definition
    """
    # Try to find delta
    delta = epsilon / 10  # Initial guess
    
    # Test points around a
    test_points = np.linspace(a - delta, a + delta, 20)
    test_points = test_points[test_points != a]  # Exclude a itself
    
    for x in test_points:
        fx = f(x)
        if abs(fx - L) >= epsilon:
            return False, f"Failed at x={x}, f(x)={fx}, |f(x) - L| = {abs(fx - L)}"
    
    return True, f"Limit verified for ε={epsilon}, δ={delta}"

# Example
def g(x):
    return 2*x + 1

result, message = verify_limit(g, 3, 7, epsilon=0.01)
print(f"Verification: {result}")
print(message)
\`\`\`

## One-Sided Limits

**Left-hand limit**: \`lim_{x → a⁻} f(x)\` (x approaches a from left)
**Right-hand limit**: \`lim_{x → a⁺} f(x)\` (x approaches a from right)

A limit exists if and only if both one-sided limits exist and are equal.

\`\`\`python
def piecewise_function(x):
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
    print(f"  f({x}) = {piecewise_function(x)}")

print("\\nRight-hand limit (x → 0⁺):")
for x in x_right:
    print(f"  f({x}) = {piecewise_function(x)}")

# Visualization
x_neg = np.linspace(-2, 0, 100)
x_pos = np.linspace(0, 2, 100)

plt.figure(figsize=(10, 6))
plt.plot(x_neg, x_neg**2, 'b-', label='f(x) = x² (x < 0)')
plt.plot(x_pos, x_pos + 1, 'r-', label='f(x) = x + 1 (x ≥ 0)')
plt.scatter([0], [0], color='b', s=100, zorder=5, facecolors='none', edgecolors='b')
plt.scatter([0], [1], color='r', s=100, zorder=5)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Piecewise Function with Jump Discontinuity')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.show()
\`\`\`

## Limits at Infinity

**Horizontal asymptotes**: Behavior of f(x) as x → ±∞

\`\`\`python
# Example: lim_{x → ∞} (3x² + 2x + 1)/(x² + 1)
def rational_function(x):
    return (3*x**2 + 2*x + 1) / (x**2 + 1)

x_values = [10, 100, 1000, 10000]
for x in x_values:
    print(f"f({x}) = {rational_function(x)}")

# As x → ∞, the limit is 3 (ratio of leading coefficients)

# Visualization
x = np.linspace(0, 100, 1000)
y = rational_function(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', label='f(x) = (3x² + 2x + 1)/(x² + 1)')
plt.axhline(y=3, color='r', linestyle='--', label='Horizontal Asymptote: y = 3')
plt.xlabel('x')
plt.ylabel('f(x)')
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
1. f(a) is defined
2. \`lim_{x → a} f(x)\` exists
3. \`lim_{x → a} f(x) = f(a)\`

**Intuition**: You can draw the function without lifting your pencil.

### Types of Discontinuities

1. **Removable discontinuity**: Limit exists but doesn't equal f(a)
2. **Jump discontinuity**: Left and right limits differ
3. **Infinite discontinuity**: Function approaches ±∞

\`\`\`python
def check_continuity(f, a, epsilon=1e-6):
    """
    Check if function is continuous at point a
    """
    try:
        # Check if f(a) is defined
        fa = f(a)
        print(f"f({a}) = {fa} ✓ (defined)")
        
        # Check left and right limits
        left_limit = f(a - epsilon)
        right_limit = f(a + epsilon)
        
        print(f"Left limit: {left_limit}")
        print(f"Right limit: {right_limit}")
        
        # Check if limits are close
        if abs(left_limit - right_limit) < 0.01:
            avg_limit = (left_limit + right_limit) / 2
            print(f"Limit exists: {avg_limit} ✓")
            
            # Check if limit equals f(a)
            if abs(avg_limit - fa) < 0.01:
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
def continuous_func(x):
    return x**2 + 1

def removable_disc(x):
    if x != 2:
        return (x**2 - 4) / (x - 2)
    return 0  # Wrong value

print("Testing continuous function:")
check_continuity(continuous_func, 2)

print("\\n" + "="*50)
print("Testing removable discontinuity:")
check_continuity(removable_disc, 2)
\`\`\`

## Intermediate Value Theorem (IVT)

**Theorem**: If f is continuous on [a, b] and k is between f(a) and f(b), then there exists c ∈ [a, b] such that f(c) = k.

**Application**: Root finding, proving existence of solutions

\`\`\`python
def intermediate_value_theorem(f, a, b, target, tolerance=1e-6):
    """
    Use IVT to find x where f(x) ≈ target
    Binary search approach
    """
    fa, fb = f(a), f(b)
    
    # Check if target is between f(a) and f(b)
    if not (min(fa, fb) <= target <= max(fa, fb)):
        return None, "Target not in range [f(a), f(b)]"
    
    # Binary search
    iterations = 0
    while abs(b - a) > tolerance:
        mid = (a + b) / 2
        fmid = f(mid)
        
        if abs(fmid - target) < tolerance:
            return mid, f"Found at x = {mid}, f(x) = {fmid}, iterations: {iterations}"
        
        # Decide which half to search
        if (fmid - target) * (fa - target) < 0:
            b = mid
            fb = fmid
        else:
            a = mid
            fa = fmid
        
        iterations += 1
    
    return (a + b) / 2, f"Approximate solution found in {iterations} iterations"

# Example: Find root of f(x) = x^3 - 2x - 5
def cubic(x):
    return x**3 - 2*x - 5

root, message = intermediate_value_theorem(cubic, 2, 3, 0)
print(f"Root: {root}")
print(message)
print(f"Verification: f({root}) = {cubic(root)}")

# Visualization
x = np.linspace(1, 3, 1000)
y = cubic(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', label='f(x) = x³ - 2x - 5')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.scatter([root], [0], color='r', s=100, zorder=5, label=f'Root ≈ {root:.4f}')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Intermediate Value Theorem - Root Finding')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
\`\`\`

## Connection to Machine Learning

### 1. Activation Functions

Neural network activation functions must be well-behaved. Understanding their continuity and limits is crucial:

\`\`\`python
def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))

def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    """Leaky ReLU activation function"""
    return np.where(x > 0, x, alpha * x)

# Analyze limits
x_range = np.linspace(-10, 10, 1000)

plt.figure(figsize=(15, 5))

# Sigmoid
plt.subplot(131)
plt.plot(x_range, sigmoid(x_range), 'b-', label='Sigmoid')
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='lim_{x→-∞} = 0')
plt.axhline(y=1, color='g', linestyle='--', alpha=0.5, label='lim_{x→+∞} = 1')
plt.xlabel('x')
plt.ylabel('σ(x)')
plt.title('Sigmoid: Continuous everywhere')
plt.legend()
plt.grid(True, alpha=0.3)

# ReLU
plt.subplot(132)
plt.plot(x_range, relu(x_range), 'b-', label='ReLU')
plt.scatter([0], [0], color='r', s=100, zorder=5, label='Not differentiable at x=0')
plt.xlabel('x')
plt.ylabel('ReLU(x)')
plt.title('ReLU: Continuous but not differentiable at 0')
plt.legend()
plt.grid(True, alpha=0.3)

# Leaky ReLU
plt.subplot(133)
plt.plot(x_range, leaky_relu(x_range), 'b-', label='Leaky ReLU')
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
def gradient_descent_convergence(learning_rate=0.01, max_iterations=100):
    """
    Demonstrate convergence using limits
    Minimize f(x) = x^2
    """
    x = 10.0  # Initial point
    history = [x]
    
    for i in range(max_iterations):
        gradient = 2*x  # derivative of x^2
        x = x - learning_rate * gradient
        history.append(x)
        
        # Check convergence (limit reached)
        if abs(x) < 1e-6:
            print(f"Converged at iteration {i+1}")
            break
    
    return np.array(history)

# Test different learning rates
rates = [0.01, 0.1, 0.5]
plt.figure(figsize=(15, 5))

for idx, lr in enumerate(rates, 1):
    history = gradient_descent_convergence(learning_rate=lr)
    
    plt.subplot(1, 3, idx)
    plt.plot(history, 'b-o', markersize=4)
    plt.axhline(y=0, color='r', linestyle='--', label='Limit = 0')
    plt.xlabel('Iteration')
    plt.ylabel('x value')
    plt.title(f'Learning Rate = {lr}')
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
limit1 = sp.limit(expr1, x, 2)
print(f"lim_(x→2) (x² + 3x + 2) = {limit1}")

# Example 2: Factoring
expr2 = (x**2 - 9) / (x - 3)
limit2 = sp.limit(expr2, x, 3)
print(f"lim_(x→3) (x² - 9)/(x - 3) = {limit2}")

# Example 3: L'Hôpital's rule (preview)
expr3 = sp.sin(x) / x
limit3 = sp.limit(expr3, x, 0)
print(f"lim_(x→0) sin(x)/x = {limit3}")

# Example 4: Conjugate multiplication
expr4 = (sp.sqrt(x + 1) - 1) / x
limit4 = sp.limit(expr4, x, 0)
print(f"lim_(x→0) (√(x+1) - 1)/x = {limit4}")
\`\`\`

## Best Practices & Common Pitfalls

### Best Practices:
1. **Always check if direct substitution works first**
2. **Verify limits numerically before symbolic computation**
3. **Check both one-sided limits for piecewise functions**
4. **Use graphical visualization to understand behavior**
5. **Be aware of floating-point precision issues**

### Common Pitfalls:
1. **Assuming f(a) = lim_{x→a} f(x)** (not always true!)
2. **Ignoring undefined points**
3. **Confusing limit with function value**
4. **Not checking continuity before applying theorems**
5. **Numerical errors for limits at infinity**

\`\`\`python
# Pitfall example: Floating-point precision
def problematic_limit(x):
    return (1 + 1/x)**x  # Approaches e as x → ∞

# Bad: Using very large x directly
x_large = 1e16
result_bad = problematic_limit(x_large)
print(f"At x = {x_large}: {result_bad}")  # May be inaccurate

# Better: Use logarithm properties or symbolic computation
import sympy as sp
x_sym = sp.Symbol('x')
expr = (1 + 1/x_sym)**x_sym
limit_correct = sp.limit(expr, x_sym, sp.oo)
print(f"Correct limit: {limit_correct}")  # e
print(f"Numerical value of e: {np.e}")
\`\`\`

## Summary

**Key Takeaways**:
- Limits describe function behavior as inputs approach values
- Continuity requires: f(a) defined, limit exists, limit = f(a)
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
      multipleChoice: [
        {
          id: 'limits-1',
          question:
            'What does it mean for a function f(x) to have a limit L as x approaches a?',
          options: [
            'f(a) must equal L',
            'f(x) gets arbitrarily close to L as x gets arbitrarily close to a',
            'f(x) is defined at x = a',
            'f(x) = L for all x near a',
          ],
          correctAnswer: 1,
          explanation:
            "A limit describes the value a function approaches as x gets close to a. The function doesn't need to be defined at a, and f(a) doesn't need to equal the limit. This is a key distinction between limits and function values.",
          difficulty: 'easy',
        },
        {
          id: 'limits-2',
          question: 'Which activation function has a discontinuity at x = 0?',
          options: [
            'Sigmoid',
            'Tanh',
            'ReLU (it is continuous but not differentiable)',
            'Step function (Heaviside)',
          ],
          correctAnswer: 3,
          explanation:
            'The step function (Heaviside) has a jump discontinuity at x = 0, jumping from 0 to 1. ReLU is continuous everywhere but not differentiable at 0. Sigmoid and tanh are both continuous and differentiable everywhere.',
          difficulty: 'medium',
        },
        {
          id: 'limits-3',
          question:
            'For a function to be continuous at x = a, which conditions must ALL be satisfied?',
          options: [
            'Only that f(a) is defined',
            'Only that the limit exists',
            'f(a) is defined, lim_(x→a) f(x) exists, and they are equal',
            'f(x) must be differentiable at a',
          ],
          correctAnswer: 2,
          explanation:
            'Continuity requires three conditions: (1) f(a) is defined, (2) the limit as x approaches a exists, and (3) the limit equals f(a). Differentiability is not required for continuity.',
          difficulty: 'medium',
        },
        {
          id: 'limits-4',
          question: 'What is lim_(x→∞) (5x³ + 2x² + 1)/(x³ + 3)?',
          options: ['0', '5', '∞', '5/3'],
          correctAnswer: 1,
          explanation:
            'For rational functions at infinity, divide by the highest power of x. The terms 2x², 1, and 3 become negligible, leaving (5x³)/(x³) = 5. The limit is the ratio of leading coefficients when degrees are equal.',
          difficulty: 'medium',
        },
        {
          id: 'limits-5',
          question:
            'The Intermediate Value Theorem guarantees that a continuous function on [a,b] will:',
          options: [
            'Be differentiable everywhere',
            'Take on every value between f(a) and f(b)',
            'Have a maximum and minimum',
            'Be monotonic (always increasing or always decreasing)',
          ],
          correctAnswer: 1,
          explanation:
            "The IVT states that a continuous function on a closed interval will take on every value between f(a) and f(b). It doesn't guarantee differentiability, extrema, or monotonicity.",
          difficulty: 'hard',
        },
      ],
      quiz: [
        {
          id: 'limits-disc-1',
          question:
            'Why is understanding limits crucial for training neural networks? Discuss specific scenarios where limit analysis helps explain training dynamics.',
          hint: 'Think about vanishing/exploding gradients, activation function saturation, and learning rate selection.',
          sampleAnswer: `Understanding limits is fundamental to neural network training for several reasons:

**1. Activation Function Saturation:**
When inputs to sigmoid or tanh become very large (|x| → ∞), the activations saturate (approach 0 or 1). Understanding lim_(x→±∞) σ(x) helps us:
- Predict when neurons will stop learning (gradient ≈ 0 in saturated regions)
- Choose appropriate weight initialization to avoid immediate saturation
- Design better activation functions (ReLU avoids saturation for positive values)

**2. Vanishing Gradient Problem:**
In deep networks, gradients are products of many terms. If each term is less than 1, the product approaches 0 as depth increases. This is fundamentally a limit problem: lim_(n→∞) (0.9)^n = 0. Understanding this limit helps us:
- Recognize why deep networks were historically difficult to train
- Motivate skip connections (ResNet) that provide alternative gradient paths
- Choose activation functions with better gradient properties

**3. Learning Rate Selection:**
Gradient descent convergence is a limit: we want x_t → x* as t → ∞. The learning rate determines whether:
- We converge: lim_(t→∞) |x_t - x*| = 0 (learning rate too small → slow)
- We diverge: lim_(t→∞) |x_t - x*| = ∞ (learning rate too large)
- We oscillate without converging

**4. Loss Function Behavior:**
Understanding lim_(x→x*) L(x) near optimal parameters helps us:
- Set convergence criteria (when is gradient "close enough" to 0?)
- Understand plateaus in training
- Design better optimization algorithms

In practice, limit analysis transforms abstract mathematical concepts into practical guidelines for architecture design, initialization strategies, and training procedures.`,
          keyPoints: [
            'Activation function saturation relates to limits at infinity',
            'Vanishing gradients are fundamentally a limit problem with products',
            'Learning rate affects whether gradient descent converges (limit exists)',
            'Understanding limits near optima guides convergence criteria',
          ],
        },
        {
          id: 'limits-disc-2',
          question:
            'Compare and contrast the behavior of different activation functions using limit analysis. Which properties make an activation function "good" from a limits perspective?',
          hint: 'Consider sigmoid, tanh, ReLU, and Leaky ReLU. Analyze their limits at ±∞ and continuity properties.',
          sampleAnswer: `Activation functions can be analyzed through their limit behavior, revealing important training properties:

**Sigmoid: σ(x) = 1/(1 + e^(-x))**
- lim_(x→-∞) σ(x) = 0, lim_(x→+∞) σ(x) = 1
- Continuous and differentiable everywhere
- Problems: Bounded output causes vanishing gradients (derivative → 0 at extremes)
- The limit boundaries [0, 1] help with probability interpretation but harm gradient flow

**Tanh: tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))**
- lim_(x→-∞) tanh(x) = -1, lim_(x→+∞) tanh(x) = 1
- Continuous and differentiable everywhere
- Better than sigmoid due to zero-centered output, but still saturates

**ReLU: ReLU(x) = max(0, x)**
- lim_(x→-∞) ReLU(x) = 0, lim_(x→+∞) ReLU(x) = ∞
- Continuous everywhere but not differentiable at x = 0
- Advantages: No upper saturation (lim as x→∞ is unbounded), cheap computation
- Problems: "Dying ReLU" when neurons always output 0 (stuck at left limit)

**Leaky ReLU: LReLU(x) = max(αx, x) where α ≈ 0.01**
- lim_(x→-∞) LReLU(x) = -∞, lim_(x→+∞) LReLU(x) = ∞
- Continuous everywhere, not differentiable at x = 0
- Solves dying ReLU problem (limit exists but is unbounded on both sides)

**Properties of "Good" Activation Functions (from limits perspective):**
1. **No upper bound:** Allows gradient flow even for large inputs
2. **Non-zero gradient regions:** Enables learning throughout the domain
3. **Continuous:** Ensures stable optimization (no jumps)
4. **Appropriate behavior at limits:** Not both limits at 0 (information loss)

ReLU and its variants succeed because lim_(x→∞) ReLU(x) = ∞ (no saturation), despite the discontinuous derivative. This shows that limits matter more than smoothness for practical deep learning.`,
          keyPoints: [
            'Bounded activation functions (sigmoid, tanh) have vanishing gradient problems due to limits',
            'Unbounded activations (ReLU) maintain gradient flow for large inputs',
            'Continuity is important but differentiability can be sacrificed',
            'Limit behavior at ±∞ determines saturation properties',
          ],
        },
        {
          id: 'limits-disc-3',
          question:
            'Explain how the Intermediate Value Theorem can be applied to root-finding in optimization algorithms. What role does continuity play?',
          hint: 'Consider finding where gradients equal zero, binary search methods, and why gradient descent works.',
          sampleAnswer: `The Intermediate Value Theorem (IVT) is fundamental to many optimization and root-finding techniques in machine learning:

**IVT in Root-Finding (Finding Critical Points):**
When minimizing a loss function L(θ), we seek points where ∇L(θ) = 0. The IVT guarantees solutions exist:
- If ∇L(θ₁) < 0 and ∇L(θ₂) > 0, and ∇L is continuous
- Then there exists θ* ∈ (θ₁, θ₂) where ∇L(θ*) = 0
- This θ* is a critical point (potential minimum)

**Binary Search for Roots:**
The IVT enables efficient root-finding algorithms:
1. Start with interval [a, b] where f(a) and f(b) have opposite signs
2. Check midpoint m = (a + b)/2
3. Replace [a, b] with either [a, m] or [m, b] based on sign of f(m)
4. Repeat until |f(m)| < tolerance

This converges logarithmically: O(log(ε)) iterations for precision ε.

**Line Search in Optimization:**
In gradient descent, we often use line search to find step size α:
- We want to minimize φ(α) = L(θ - α∇L(θ))
- IVT guarantees that if φ'(0) < 0, there exists α > 0 where φ(α) < φ(0)
- We can binary search for optimal α where φ'(α) ≈ 0

**Importance of Continuity:**
Continuity is ESSENTIAL for IVT to hold. If the function has jumps:
- We might skip over roots entirely
- Binary search could fail
- Optimization becomes unreliable

In neural networks:
- Non-continuous activation functions (step function) make optimization difficult
- Discontinuous loss functions lead to optimization challenges
- We prefer continuous loss surfaces even if not everywhere differentiable (ReLU)

**Why Gradient Descent Works:**
Gradient descent relies implicitly on IVT-like reasoning:
- Starting from θ₀, we move in direction -∇L(θ₀)
- If we move far enough (but not too far), we'll cross a level set
- Continuity ensures we don't "jump over" better solutions
- The existence of a better point in the direction is guaranteed by IVT when gradient is non-zero

**Practical Implications:**
1. Choosing continuous activation functions (sigmoid, ReLU, tanh) ensures IVT applies
2. Batch normalization maintains continuity despite adding noise
3. Adversarial training can create near-discontinuities that break IVT assumptions
4. Discrete optimization (integer constraints) requires different techniques since IVT doesn't apply

The IVT transforms existence questions ("Does a solution exist?") into algorithmic questions ("How do we find it?"), making it indispensable for practical optimization.`,
          keyPoints: [
            'IVT guarantees existence of roots/critical points in continuous functions',
            'Enables efficient binary search algorithms for optimization',
            'Continuity is essential - discontinuous functions break IVT',
            'Gradient descent implicitly relies on IVT-like reasoning',
            'Line search methods use IVT to find optimal step sizes',
          ],
        },
      ],
    },
    {
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
      multipleChoice: [
        {
          id: 'deriv-1',
          question: "What does the derivative f'(a) represent geometrically?",
          options: [
            'The value of the function at x = a',
            'The slope of the tangent line at x = a',
            'The area under the curve up to x = a',
            'The average rate of change near x = a',
          ],
          correctAnswer: 1,
          explanation:
            'The derivative represents the slope of the tangent line at a point, which is the instantaneous rate of change. This is found by taking the limit of secant line slopes as the interval shrinks to zero.',
          difficulty: 'easy',
        },
        {
          id: 'deriv-2',
          question: "If f(x) = 3x⁴ - 2x² + 5, what is f'(x)?",
          options: ['12x³ - 4x', '3x³ - 2x', '12x³ - 2x + 5', '12x⁴ - 4x²'],
          correctAnswer: 0,
          explanation:
            "Using the power rule for each term: d/dx[3x⁴] = 12x³, d/dx[-2x²] = -4x, d/dx[5] = 0. So f'(x) = 12x³ - 4x.",
          difficulty: 'easy',
        },
        {
          id: 'deriv-3',
          question:
            'Why does the sigmoid activation function suffer from vanishing gradients?',
          options: [
            'It is not continuous',
            'Its derivative approaches zero for large |x|',
            'It has no derivative at x = 0',
            'Its derivative is always negative',
          ],
          correctAnswer: 1,
          explanation:
            "The sigmoid derivative σ'(x) = σ(x)(1-σ(x)) is largest at x=0 (0.25) and approaches 0 as x → ±∞. This causes gradients to vanish in deep networks when neurons are saturated.",
          difficulty: 'medium',
        },
        {
          id: 'deriv-4',
          question:
            'In gradient descent, why do we move in the direction of the negative gradient?',
          options: [
            'To increase the loss function',
            'Because the derivative is always negative',
            'The negative gradient points toward the steepest decrease',
            'To satisfy the learning rate requirement',
          ],
          correctAnswer: 2,
          explanation:
            'The gradient points in the direction of steepest increase. To minimize the loss, we move in the opposite direction (negative gradient), which is the direction of steepest decrease.',
          difficulty: 'medium',
        },
        {
          id: 'deriv-5',
          question:
            'When using numerical differentiation with h = (f(x+h) - f(x))/h, what happens if h is too small?',
          options: [
            'The result becomes more accurate indefinitely',
            'Floating-point round-off errors dominate',
            'The computation becomes faster',
            'The derivative converges to zero',
          ],
          correctAnswer: 1,
          explanation:
            'While smaller h reduces truncation error, it amplifies floating-point round-off errors. The optimal h balances these two error sources, typically around √ε where ε is machine epsilon.',
          difficulty: 'hard',
        },
      ],
      quiz: [
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
      ],
    },
    {
      id: 'differentiation-rules',
      title: 'Differentiation Rules',
      content: `
# Differentiation Rules

## Introduction

Calculus provides powerful rules that make differentiation efficient. These rules are essential for backpropagation, where we compute millions of derivatives.

## Product Rule & Quotient Rule

**Product Rule**: (uv)' = u'v + uv'  
**Quotient Rule**: (u/v)' = (u'v - uv')/v²

\`\`\`python
import numpy as np
import sympy as sp

# Product rule example
x = sp.Symbol('x')
f = x**2 * sp.sin(x)
print(f"d/dx[x² · sin(x)] = {sp.diff(f, x)}")

# Quotient rule example
g = sp.sin(x) / x
print(f"d/dx[sin(x)/x] = {sp.simplify(sp.diff(g, x))}")
\`\`\`

## Chain Rule - Heart of Backpropagation

**Rule**: (f∘g)'(x) = f'(g(x)) · g'(x)

\`\`\`python
# Example: d/dx[sin(x²)] = cos(x²) · 2x
f = sp.sin(x**2)
print(f"Chain rule: {sp.diff(f, x)}")

# Backpropagation example
class TwoLayerNet:
    def forward(self, x):
        self.x = x
        self.z1 = np.dot(self.W1, x)
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.W2, self.a1)
        return self.z2
    
    def backward(self, dL_dout):
        # Chain rule layer by layer
        dL_dz2 = dL_dout
        dL_dW2 = np.outer(dL_dz2, self.a1)
        dL_da1 = np.dot(self.W2.T, dL_dz2)
        dL_dz1 = dL_da1 * sigmoid_derivative(self.z1)
        dL_dW1 = np.outer(dL_dz1, self.x)
        return dL_dW1, dL_dW2
\`\`\`

## Common Derivatives Reference

\`\`\`python
derivatives = {
    "x^n": "n·x^(n-1)",
    "e^x": "e^x",
    "ln(x)": "1/x",
    "sin(x)": "cos(x)",
    "cos(x)": "-sin(x)",
    "tan(x)": "sec²(x)",
    "arcsin(x)": "1/√(1-x²)",
    "arctan(x)": "1/(1+x²)"
}

for func, deriv in derivatives.items():
    print(f"d/dx[{func}] = {deriv}")
\`\`\`

## Summary

The chain rule is fundamental to backpropagation. Master it!
`,
      multipleChoice: [
        {
          id: 'diff-rules-1',
          question: 'What is d/dx[x²·e^x]?',
          options: ['2x·e^x', '2x·e^x + x²·e^x', 'x²·e^x', '2x + e^x'],
          correctAnswer: 1,
          explanation:
            "Product rule: (uv)' = u'v + uv'. Here u=x², v=e^x, so: 2x·e^x + x²·e^x.",
          difficulty: 'easy',
        },
        {
          id: 'diff-rules-2',
          question: 'What is d/dx[sin(3x²)]?',
          options: ['cos(3x²)', '6x·cos(3x²)', '3x·cos(3x²)', 'cos(6x)'],
          correctAnswer: 1,
          explanation: 'Chain rule: cos(3x²)·6x = 6x·cos(3x²).',
          difficulty: 'medium',
        },
        {
          id: 'diff-rules-3',
          question: 'Why is the chain rule essential for neural networks?',
          options: [
            'Makes computation faster',
            'Computes gradients through composed functions',
            'Reduces memory',
            'Prevents overfitting',
          ],
          correctAnswer: 1,
          explanation:
            'Backpropagation applies the chain rule to compute gradients through layer compositions.',
          difficulty: 'medium',
        },
        {
          id: 'diff-rules-4',
          question: 'What is d/dx[ln(x²)]?',
          options: ['1/x²', '2/x', '2x', '1/(2x)'],
          correctAnswer: 1,
          explanation:
            'Chain rule: (1/x²)·2x = 2/x, or use ln(x²) = 2ln(x) → 2/x.',
          difficulty: 'easy',
        },
        {
          id: 'diff-rules-5',
          question: 'Logarithmic differentiation is best for:',
          options: [
            'Simple polynomials',
            'Products, quotients, and variable exponents',
            'Linear functions',
            'Constant functions',
          ],
          correctAnswer: 1,
          explanation:
            'Logarithmic differentiation simplifies products, quotients, and functions like x^x.',
          difficulty: 'medium',
        },
      ],
      quiz: [
        {
          id: 'diff-rules-disc-1',
          question: 'Explain how the chain rule enables backpropagation.',
          hint: 'Consider function composition through layers.',
          sampleAnswer: `Backpropagation is the chain rule applied repeatedly. In a neural network, output = f_n(f_{n-1}(...f_1(x))). To find gradients, we apply the chain rule layer by layer going backward, reusing intermediate results. This makes gradient computation O(n) instead of O(n²).`,
          keyPoints: [
            'Neural networks are function compositions',
            'Chain rule computes gradients through compositions',
            'Backprop applies chain rule backward through layers',
            'Intermediate gradient reuse provides efficiency',
          ],
        },
        {
          id: 'diff-rules-disc-2',
          question: 'When and why would you use logarithmic differentiation?',
          hint: 'Consider products, quotients, and complex exponents.',
          sampleAnswer: `Logarithmic differentiation is useful for: (1) Variable exponents like x^x, (2) Products of many functions (converts to sums), (3) Complicated quotients (converts to differences), (4) Numerical stability in log-likelihood computations. It simplifies algebra and prevents numerical underflow in probabilistic models.`,
          keyPoints: [
            'Converts products to sums, quotients to differences',
            'Essential for variable exponents',
            'Used in log-likelihood optimization',
            'Provides numerical stability',
          ],
        },
        {
          id: 'diff-rules-disc-3',
          question: 'Explain implicit differentiation and its ML applications.',
          hint: 'Think about constrained optimization.',
          sampleAnswer: `Implicit differentiation finds dy/dx when y is defined implicitly by F(x,y)=0. In ML, it's used for: (1) Constrained optimization (Lagrange multipliers), (2) Implicit deep learning models (DEQ), (3) Manifold optimization, (4) Neural ODEs. It allows gradient computation without explicit solutions, essential for modern architectures with constraints.`,
          keyPoints: [
            'Works with implicit relationships',
            'Used in constrained optimization',
            'Essential for implicit deep learning',
            'Enables manifold and bilevel optimization',
          ],
        },
      ],
    },
    {
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
      multipleChoice: [
        {
          id: 'app-deriv-1',
          question: 'At a local minimum, the first derivative is:',
          options: ['Positive', 'Negative', 'Zero', 'Undefined'],
          correctAnswer: 2,
          explanation:
            "At critical points (local min/max), f'(x) = 0. Use second derivative to determine if it's a min or max.",
          difficulty: 'easy',
        },
        {
          id: 'app-deriv-2',
          question: "Newton's method converges:",
          options: [
            'Linearly',
            'Quadratically (very fast)',
            'Logarithmically',
            'Exponentially slow',
          ],
          correctAnswer: 1,
          explanation:
            "Newton's method has quadratic convergence near the root, doubling correct digits each iteration.",
          difficulty: 'medium',
        },
        {
          id: 'app-deriv-3',
          question:
            'What is the first-order Taylor approximation of f(x) around x=a?',
          options: [
            'f(a)',
            "f(a) + f'(a)(x-a)",
            "f(a) + f'(a)(x-a) + f''(a)(x-a)²/2",
            'f(x)',
          ],
          correctAnswer: 1,
          explanation:
            "First-order (linear) Taylor approximation: f(x) ≈ f(a) + f'(a)(x-a). This is the tangent line.",
          difficulty: 'medium',
        },
        {
          id: 'app-deriv-4',
          question: 'In gradient descent, we update parameters by:',
          options: [
            'Adding the gradient',
            'Subtracting the gradient',
            'Setting to zero',
            'Multiplying by gradient',
          ],
          correctAnswer: 1,
          explanation:
            'θ_new = θ_old - α·∇L. We move opposite to gradient (downhill).',
          difficulty: 'easy',
        },
        {
          id: 'app-deriv-5',
          question: 'Why is Taylor approximation important in ML?',
          options: [
            'Makes functions continuous',
            'Approximates complex functions with simpler ones',
            'Increases accuracy',
            'Reduces overfitting',
          ],
          correctAnswer: 1,
          explanation:
            'Taylor series approximate complex functions with polynomials, used in optimization (second-order methods) and analysis.',
          difficulty: 'medium',
        },
      ],
      quiz: [
        {
          id: 'app-deriv-disc-1',
          question:
            "Compare gradient descent and Newton's method. When would you use each?",
          hint: 'Consider convergence speed, computational cost, and memory.',
          sampleAnswer: `**Gradient Descent:** Uses only first derivatives. Update: θ - α∇L. Pros: Low memory, scales well. Cons: Slower convergence. Use for: Large-scale ML (millions of parameters), non-smooth functions. **Newton's Method:** Uses second derivatives (Hessian). Update: θ - H⁻¹∇L. Pros: Quadratic convergence (fast). Cons: O(n²) memory, expensive Hessian computation. Use for: Small problems, when Hessian is available/approximated (L-BFGS).`,
          keyPoints: [
            'GD: first-order, scalable, slower',
            'Newton: second-order, fast convergence, expensive',
            'Quasi-Newton (L-BFGS) approximates Hessian',
            'Modern ML uses variants: Adam, RMSprop',
          ],
        },
        {
          id: 'app-deriv-disc-2',
          question:
            'Explain how Taylor series are used in second-order optimization methods.',
          hint: 'Consider quadratic approximation and the Hessian matrix.',
          sampleAnswer: `Second-order methods use Taylor expansion to order 2: L(θ) ≈ L(θ₀) + ∇L·(θ-θ₀) + ½(θ-θ₀)ᵀH(θ-θ₀). Minimizing this quadratic gives Newton update: θ = θ₀ - H⁻¹∇L. This is much faster than gradient descent because it accounts for curvature. The Hessian H captures second-order information. Trade-off: Computing/storing H is O(n²), limiting applicability to smaller problems or requiring approximations (L-BFGS, Fisher information matrix in natural gradient).`,
          keyPoints: [
            'Second-order Taylor approximation is quadratic',
            'Hessian captures curvature information',
            'Newton method minimizes quadratic approximation',
            'O(n²) cost limits scalability, requires approximations',
          ],
        },
        {
          id: 'app-deriv-disc-3',
          question:
            'How do critical points relate to loss landscape analysis in deep learning?',
          hint: 'Consider saddle points, local minima, and escaping suboptimal regions.',
          sampleAnswer: `In high dimensions, most critical points (∇L=0) are saddle points, not local minima. At saddle points, Hessian has both positive and negative eigenvalues. This has implications: (1) Gradient descent can escape saddles (noise helps), (2) Local minima in deep learning often have similar loss values (loss landscape is relatively flat), (3) Second-order information (Hessian) helps identify saddle vs minimum, (4) Modern understanding: SGD noise is beneficial for escaping saddles and finding flatter minima (better generalization).`,
          keyPoints: [
            'High-dimensional critical points usually saddles',
            'Hessian eigenvalues distinguish saddles from minima',
            'SGD noise helps escape saddles',
            'Flat minima generalize better',
          ],
        },
      ],
    },
    {
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
      multipleChoice: [
        {
          id: 'partial-1',
          question: 'For f(x,y) = 3x²y + y³, what is ∂f/∂x?',
          options: ['6xy', '3x²', '6xy + 3y²', '3x² + y³'],
          correctAnswer: 0,
          explanation:
            'Treating y as constant, ∂/∂x[3x²y] = 6xy and ∂/∂x[y³] = 0.',
          difficulty: 'easy',
        },
        {
          id: 'partial-2',
          question: 'The gradient ∇f is:',
          options: [
            'A scalar',
            'A vector of partial derivatives',
            'The maximum value of f',
            'The Hessian matrix',
          ],
          correctAnswer: 1,
          explanation:
            'The gradient is a vector containing all first-order partial derivatives.',
          difficulty: 'easy',
        },
        {
          id: 'partial-3',
          question: 'In backpropagation, we compute:',
          options: [
            'Only one partial derivative',
            'Partial derivatives of loss with respect to all parameters',
            'Only the gradient',
            'Only second derivatives',
          ],
          correctAnswer: 1,
          explanation:
            'Backprop computes ∂L/∂θ for every parameter θ in the network.',
          difficulty: 'medium',
        },
        {
          id: 'partial-4',
          question: 'When computing ∂f/∂x for f(x,y), we treat y as:',
          options: ['Variable', 'Constant', 'Zero', 'Undefined'],
          correctAnswer: 1,
          explanation: 'Partial derivatives hold all other variables constant.',
          difficulty: 'easy',
        },
        {
          id: 'partial-5',
          question: 'For a neural network with n parameters, the gradient has:',
          options: [
            '1 component',
            'n components',
            'n² components',
            '2n components',
          ],
          correctAnswer: 1,
          explanation:
            'The gradient vector has one component for each parameter: ∇L = [∂L/∂θ₁, ..., ∂L/∂θₙ].',
          difficulty: 'medium',
        },
      ],
      quiz: [
        {
          id: 'partial-disc-1',
          question:
            'Explain how partial derivatives enable gradient-based optimization in high-dimensional spaces.',
          hint: 'Consider parameter spaces with millions of dimensions.',
          sampleAnswer: `Partial derivatives decompose high-dimensional optimization into manageable components. For a neural network with millions of parameters θ = [θ₁,...,θₙ], we need ∇L = [∂L/∂θ₁,...,∂L/∂θₙ]. Each partial ∂L/∂θᵢ tells us how to adjust that specific parameter. Gradient descent then updates: θᵢ ← θᵢ - α·∂L/∂θᵢ for each i. This parallelizes naturally, making optimization tractable even for billions of parameters. Without partial derivatives, we couldn't isolate individual parameter effects.`,
          keyPoints: [
            'Partials decompose high-dim gradient into components',
            'Each ∂L/∂θᵢ guides individual parameter update',
            'Enables parallel computation',
            'Makes billion-parameter optimization tractable',
          ],
        },
        {
          id: 'partial-disc-2',
          question:
            'How does the chain rule extend to partial derivatives in backpropagation?',
          hint: 'Consider computing ∂L/∂W₁ through intermediate layers.',
          sampleAnswer: `Backprop applies the multivariable chain rule. For L(f(g(x))), we have: ∂L/∂xᵢ = Σⱼ(∂L/∂fⱼ · ∂fⱼ/∂gₖ · ∂gₖ/∂xᵢ) summing over all paths. In neural networks: ∂L/∂W₁ = ∂L/∂z₂ · ∂z₂/∂a₁ · ∂a₁/∂z₁ · ∂z₁/∂W₁. Each term is a partial derivative or Jacobian matrix. The chain rule says: multiply along paths and sum over branches. This is exactly what backprop does layer-by-layer.`,
          keyPoints: [
            'Multivariable chain rule sums over all paths',
            'Each layer contributes local Jacobian',
            'Matrix multiplication implements chain rule',
            'Backprop efficiently computes through layer composition',
          ],
        },
        {
          id: 'partial-disc-3',
          question: 'Why is the gradient the direction of steepest ascent?',
          hint: 'Consider directional derivatives in all directions.',
          sampleAnswer: `The gradient ∇f points toward maximum rate of increase. Mathematical proof: the directional derivative in direction û is ∇f·û. This is maximized when û is parallel to ∇f (dot product = |∇f|). Any other direction gives slower increase. Geometrically, ∇f is perpendicular to level curves/surfaces. Moving along ∇f crosses the most level curves per unit distance. In optimization, we use -∇f (steepest descent) to minimize functions, which is why gradient descent works.`,
          keyPoints: [
            'Directional derivative is ∇f·û',
            'Maximized when û parallel to ∇f',
            'Gradient perpendicular to level sets',
            'Negative gradient gives steepest descent for minimization',
          ],
        },
      ],
    },
    {
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

def directional_derivative(f, point, direction, h=1e-7):
    """
    Compute directional derivative of f at point in given direction
    """
    point = np.array(point, dtype=float)
    direction = np.array(direction, dtype=float)
    
    # Normalize direction vector
    direction = direction / np.linalg.norm(direction)
    
    # Compute directional derivative
    f_plus = f(point + h * direction)
    f_point = f(point)
    
    return (f_plus - f_point) / h

# Example function: f(x,y) = x² + 2y²
def f(xy):
    if len(xy.shape) == 1:
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
    dir_deriv = directional_derivative(f, point, direction)
    direction_normalized = direction / np.linalg.norm(direction)
    print(f"Direction {direction_normalized}: {dir_deriv:.4f}")
\`\`\`

## Relationship to Gradient

**Key Theorem**: The directional derivative equals the dot product of the gradient with the direction:

D_v f(**a**) = ∇f(**a**) · **v̂**

where **v̂** is the unit vector in direction **v**.

\`\`\`python
def gradient_2d(f, point, h=1e-7):
    """Compute gradient numerically"""
    x, y = point
    df_dx = (f(np.array([x + h, y])) - f(np.array([x, y]))) / h
    df_dy = (f(np.array([x, y + h])) - f(np.array([x, y]))) / h
    return np.array([df_dx, df_dy])

# Compute gradient
grad = gradient_2d(f, point)
print(f"\\nGradient at {point}: {grad}")

# Verify: directional derivative = gradient · direction
print("\\nVerification (gradient · direction):")
print("="*60)

for direction in directions:
    direction_normalized = direction / np.linalg.norm(direction)
    
    # Method 1: Direct computation
    dir_deriv_direct = directional_derivative(f, point, direction)
    
    # Method 2: Gradient dot product
    dir_deriv_gradient = np.dot(grad, direction_normalized)
    
    error = abs(dir_deriv_direct - dir_deriv_gradient)
    
    print(f"Direction {direction_normalized}:")
    print(f"  Direct: {dir_deriv_direct:.8f}")
    print(f"  Grad·v: {dir_deriv_gradient:.8f}")
    print(f"  Error:  {error:.2e}")
\`\`\`

## Maximum Rate of Change

**Theorem**: The gradient points in the direction of maximum rate of increase, and its magnitude is that maximum rate.

\`\`\`python
def visualize_directional_derivatives(f, point, num_directions=36):
    """
    Visualize directional derivatives in all directions
    """
    # Create directions (unit circle)
    angles = np.linspace(0, 2*np.pi, num_directions, endpoint=False)
    directions = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Compute directional derivatives
    dir_derivs = np.array([
        directional_derivative(f, point, d) for d in directions
    ])
    
    # Compute gradient
    grad = gradient_2d(f, point)
    grad_magnitude = np.linalg.norm(grad)
    grad_direction = grad / grad_magnitude
    
    # Create polar plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Polar plot of directional derivatives
    ax1 = plt.subplot(121, projection='polar')
    ax1.plot(angles, dir_derivs, 'b-', linewidth=2)
    ax1.fill(angles, dir_derivs, alpha=0.3)
    
    # Mark gradient direction
    grad_angle = np.arctan2(grad_direction[1], grad_direction[0])
    ax1.plot([grad_angle], [grad_magnitude], 'ro', markersize=10, label='Gradient')
    
    ax1.set_title(f'Directional Derivatives\\n(max = {grad_magnitude:.2f})')
    ax1.legend()
    
    # Contour plot with gradient
    x = np.linspace(point[0] - 2, point[0] + 2, 100)
    y = np.linspace(point[1] - 2, point[1] + 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))
    
    ax2.contour(X, Y, Z, levels=20, cmap='viridis')
    ax2.quiver(point[0], point[1], grad[0], grad[1], 
               color='r', scale=10, width=0.01, label='Gradient')
    
    # Show several directional derivatives
    for i in range(0, num_directions, 6):
        d = directions[i]
        ax2.quiver(point[0], point[1], d[0], d[1],
                  color='b', alpha=0.3, scale=5, width=0.005)
    
    ax2.scatter(*point, color='red', s=100, zorder=5)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Gradient = Direction of Max Increase')
    ax2.legend()
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\\nMaximum directional derivative: {np.max(dir_derivs):.6f}")
    print(f"Gradient magnitude: {grad_magnitude:.6f}")
    print(f"Direction of max increase: {grad_direction}")

visualize_directional_derivatives(f, point)
\`\`\`

## Gradient Descent Interpretation

Gradient descent moves in the direction of **steepest descent** = -∇f

\`\`\`python
def gradient_descent_with_directions(f, gradient_f, x0, learning_rate=0.1, 
                                    max_iterations=50):
    """
    Gradient descent with directional analysis
    """
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    gradients = []
    
    for i in range(max_iterations):
        grad = gradient_f(x)
        gradients.append(grad.copy())
        
        # Move in negative gradient direction (steepest descent)
        x = x - learning_rate * grad
        history.append(x.copy())
        
        if np.linalg.norm(grad) < 1e-6:
            print(f"Converged in {i+1} iterations")
            break
    
    return np.array(history), np.array(gradients)

# Define function and its gradient
def rosenbrock(xy):
    """Rosenbrock function (banana function)"""
    x, y = xy
    return (1 - x)**2 + 100*(y - x**2)**2

def rosenbrock_gradient(xy):
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
X, Y = np.meshgrid(x, y)
Z = (1 - X)**2 + 100*(Y - X**2)**2

ax1.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis')
ax1.plot(history[:, 0], history[:, 1], 'r-o', markersize=3, label='GD path')
ax1.scatter([1], [1], color='g', s=200, marker='*', label='Minimum', zorder=5)

# Show gradient vectors
for i in range(0, len(history)-1, 5):
    ax1.quiver(history[i, 0], history[i, 1], 
              -gradients[i, 0], -gradients[i, 1],
              color='red', alpha=0.5, scale=50, width=0.005)

ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Gradient Descent on Rosenbrock Function')
ax1.legend()

# Loss over iterations
losses = [rosenbrock(h) for h in history]
ax2.semilogy(losses, 'b-o', markersize=3)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Loss (log scale)')
ax2.set_title('Convergence')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Starting point: {x0}")
print(f"Final point: {history[-1]}")
print(f"True minimum: [1, 1]")
print(f"Final loss: {rosenbrock(history[-1]):.2e}")
\`\`\`

## Applications in Machine Learning

### 1. Understanding Loss Landscapes

\`\`\`python
def analyze_loss_landscape(model_params, loss_fn, data):
    """
    Analyze loss landscape around current parameters
    """
    grad = compute_gradient(model_params, loss_fn, data)
    
    # Sample random directions
    n_directions = 100
    random_directions = np.random.randn(n_directions, len(model_params))
    random_directions = random_directions / np.linalg.norm(random_directions, axis=1, keepdims=True)
    
    # Compute directional derivatives
    directional_derivs = []
    for direction in random_directions:
        dir_deriv = np.dot(grad, direction)
        directional_derivs.append(dir_deriv)
    
    directional_derivs = np.array(directional_derivs)
    
    print("Loss Landscape Analysis:")
    print(f"  Gradient norm: {np.linalg.norm(grad):.4f}")
    print(f"  Max dir deriv: {np.max(directional_derivs):.4f}")
    print(f"  Min dir deriv: {np.min(directional_derivs):.4f}")
    print(f"  Avg dir deriv: {np.mean(directional_derivs):.4f}")
    
    # The gradient magnitude should equal max directional derivative
    print(f"\\nVerification:")
    print(f"  Gradient magnitude = Max directional derivative?")
    print(f"  {np.linalg.norm(grad):.6f} ≈ {np.max(directional_derivs):.6f}")
    
    return directional_derivs

# Example: Simple quadratic loss
def simple_loss(params, X, y):
    predictions = X @ params
    return np.mean((predictions - y)**2)

def compute_gradient(params, loss_fn, data):
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
directional_derivs = analyze_loss_landscape(params, simple_loss, (X, y))
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
    
    def step(self, params, gradient):
        """
        Update parameters using gradient (directional information)
        and accumulated velocity
        """
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
        
        # Accumulate velocity (exponential moving average of gradients)
        self.velocity = self.momentum * self.velocity - self.lr * gradient
        
        # Update parameters
        params_new = params + self.velocity
        
        return params_new

# Compare gradient descent with and without momentum
def compare_optimizers(f, grad_f, x0, n_iterations=100):
    """Compare GD and GD with momentum"""
    
    # Standard GD
    gd_history = [np.array(x0)]
    x_gd = np.array(x0, dtype=float)
    
    for _ in range(n_iterations):
        grad = grad_f(x_gd)
        x_gd = x_gd - 0.001 * grad
        gd_history.append(x_gd.copy())
    
    # GD with momentum
    momentum_history = [np.array(x0)]
    x_momentum = np.array(x0, dtype=float)
    optimizer = GradientDescentWithMomentum(learning_rate=0.001, momentum=0.9)
    
    for _ in range(n_iterations):
        grad = grad_f(x_momentum)
        x_momentum = optimizer.step(x_momentum, grad)
        momentum_history.append(x_momentum.copy())
    
    return np.array(gd_history), np.array(momentum_history)

# Test on Rosenbrock
gd_history, momentum_history = compare_optimizers(
    rosenbrock, rosenbrock_gradient, [-1.0, 1.0]
)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

x = np.linspace(-2, 2, 400)
y = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x, y)
Z = (1 - X)**2 + 100*(Y - X**2)**2

for ax, history, title in zip(axes, [gd_history, momentum_history],
                               ['Gradient Descent', 'GD with Momentum']):
    ax.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis', alpha=0.5)
    ax.plot(history[:, 0], history[:, 1], 'r-o', markersize=2, linewidth=1.5)
    ax.scatter([1], [1], color='g', s=200, marker='*', label='Minimum')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
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
def projected_gradient_descent(f, grad_f, x0, projection_fn,
                               learning_rate=0.1, max_iter=100):
    """
    Gradient descent with projection onto constraint set
    """
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    
    for i in range(max_iter):
        grad = grad_f(x)
        
        # Take gradient step
        x_new = x - learning_rate * grad
        
        # Project onto constraint set
        x_new = projection_fn(x_new)
        
        history.append(x_new.copy())
        x = x_new
        
        if np.linalg.norm(grad) < 1e-6:
            break
    
    return np.array(history)

# Example: Optimize on unit sphere (||x|| = 1)
def project_to_sphere(x):
    """Project onto unit sphere"""
    return x / np.linalg.norm(x)

# Minimize f(x,y) = x² + 2y² subject to x² + y² = 1
history = projected_gradient_descent(
    lambda xy: xy[0]**2 + 2*xy[1]**2,
    lambda xy: np.array([2*xy[0], 4*xy[1]]),
    [0.6, 0.8],  # Start on sphere
    project_to_sphere,
    learning_rate=0.1
)

# Visualize
theta = np.linspace(0, 2*np.pi, 100)
circle_x = np.cos(theta)
circle_y = np.sin(theta)

plt.figure(figsize=(8, 8))
plt.plot(circle_x, circle_y, 'k--', alpha=0.3, label='Constraint: x²+y²=1')
plt.plot(history[:, 0], history[:, 1], 'r-o', markersize=4, label='Projected GD')
plt.scatter(history[-1, 0], history[-1, 1], color='g', s=200, marker='*', 
           label='Minimum')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Projected Gradient Descent on Unit Sphere')
plt.legend()
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.show()

print(f"Optimal point: {history[-1]}")
print(f"Constraint satisfied: ||x|| = {np.linalg.norm(history[-1]):.6f}")
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
      multipleChoice: [
        {
          id: 'grad-dir-1',
          question: 'The directional derivative D_v f equals:',
          options: [
            'Just the gradient',
            '∇f · v̂ (dot product with unit direction)',
            'The magnitude of the gradient',
            'The second derivative',
          ],
          correctAnswer: 1,
          explanation:
            'The directional derivative equals the dot product of the gradient with the unit direction vector: D_v f = ∇f · v̂.',
          difficulty: 'medium',
        },
        {
          id: 'grad-dir-2',
          question: 'The gradient ∇f points in the direction of:',
          options: [
            'Steepest descent',
            'Steepest ascent (maximum increase)',
            'Zero change',
            'Any arbitrary direction',
          ],
          correctAnswer: 1,
          explanation:
            'The gradient points in the direction of maximum rate of increase (steepest ascent). We use -∇f for steepest descent.',
          difficulty: 'easy',
        },
        {
          id: 'grad-dir-3',
          question:
            'What is the maximum directional derivative of f at point a?',
          options: [
            'Always 1',
            'The gradient magnitude ||∇f(a)||',
            'Infinity',
            'The minimum eigenvalue',
          ],
          correctAnswer: 1,
          explanation:
            'The maximum directional derivative equals the gradient magnitude. This maximum is achieved when moving in the gradient direction.',
          difficulty: 'medium',
        },
        {
          id: 'grad-dir-4',
          question:
            'In gradient descent with momentum, the velocity accumulates:',
          options: [
            'Only the current gradient',
            'Exponential moving average of past gradients',
            'The sum of all gradients',
            'Random directions',
          ],
          correctAnswer: 1,
          explanation:
            'Momentum maintains an exponential moving average of past gradients: v = βv + (1-β)∇L, which helps accelerate convergence.',
          difficulty: 'medium',
        },
        {
          id: 'grad-dir-5',
          question:
            'For constrained optimization on a manifold, projected gradient descent:',
          options: [
            'Ignores constraints',
            'Takes gradient step then projects onto constraint set',
            'Only moves along constraints',
            "Doesn't use gradients",
          ],
          correctAnswer: 1,
          explanation:
            'Projected GD alternates between: (1) taking a gradient step, (2) projecting back onto the constraint set. This ensures constraints are satisfied.',
          difficulty: 'hard',
        },
      ],
      quiz: [
        {
          id: 'grad-dir-disc-1',
          question:
            'Prove that the gradient points in the direction of maximum rate of increase. Why is this fundamental to optimization?',
          hint: 'Consider the directional derivative formula and how dot products are maximized.',
          sampleAnswer: `**Proof that Gradient is Direction of Maximum Increase:**

The directional derivative in direction û is:
D_û f = ∇f · û = ||∇f|| ||û|| cos(θ) = ||∇f|| cos(θ)

where θ is the angle between ∇f and û, and ||û|| = 1 (unit vector).

To maximize D_û f, we need to maximize cos(θ). Since -1 ≤ cos(θ) ≤ 1:
- Maximum: cos(θ) = 1 when θ = 0° (û parallel to ∇f)
- Minimum: cos(θ) = -1 when θ = 180° (û opposite to ∇f)

Therefore:
- Maximum rate of increase: Direction of ∇f, magnitude ||∇f||
- Maximum rate of decrease: Direction of -∇f, magnitude ||∇f||

**Why This Matters for Optimization:**

1. **Gradient Descent Foundation**: Moving in direction -∇f gives steepest decrease, explaining why gradient descent works.

2. **Convergence Speed**: Larger ||∇f|| means faster decrease is possible. When ||∇f|| → 0, we're near a critical point.

3. **Step Size Selection**: Learning rate α should be chosen considering ||∇f||. If ||∇f|| is large, smaller α prevents overshooting.

4. **Saddle Point Escape**: In high dimensions, the gradient points away from saddle points along eigenvector corresponding to negative eigenvalue of Hessian.

5. **Loss Landscape Understanding**: The gradient gives local geometric information about the loss surface. Following -∇f takes the "straight downhill" path.

**Practical Implications:**

- **Adaptive Methods (Adam, RMSprop)**: Scale gradient by running average of magnitude, making step sizes more uniform.
- **Momentum**: Accumulates past gradient directions, building velocity to overcome local irregularities.
- **Natural Gradient**: Uses different metric (Fisher information) to define "steepest descent" in parameter space, accounting for model structure.

The gradient's directionality is why first-order optimization works despite high dimensionality - we don't need to search all directions, just follow ∇f.`,
          keyPoints: [
            'Directional derivative D_û f = ||∇f|| cos(θ) maximized when θ = 0',
            'Gradient direction gives maximum increase, negative gradient gives maximum decrease',
            'Gradient magnitude is the maximum rate of change',
            'Foundation for gradient descent and all first-order methods',
            'Enables efficient optimization in high dimensions',
          ],
        },
        {
          id: 'grad-dir-disc-2',
          question:
            'Explain how momentum methods use gradient direction more effectively than vanilla gradient descent. Include the mathematical formulation and intuition.',
          hint: 'Consider how velocity accumulates and the physical analogy of a ball rolling down a hill.',
          sampleAnswer: `**Momentum Method Formulation:**

**Vanilla Gradient Descent:**
θ_{t+1} = θ_t - α∇L(θ_t)

**Gradient Descent with Momentum:**
v_{t+1} = βv_t - α∇L(θ_t)
θ_{t+1} = θ_t + v_{t+1}

where:
- v is velocity (accumulated direction)
- β ∈ [0, 1) is momentum coefficient (typically 0.9)
- α is learning rate

**How It Works:**

1. **Direction Accumulation**: Velocity is exponential moving average of gradients:
   v_t = -α(∇L_t + β∇L_{t-1} + β²∇L_{t-2} + ...)
   
   Past gradients contribute with exponentially decaying weights.

2. **Consistent Directions Accelerate**: If gradients point in similar directions over time, velocity builds up:
   - Consistent downhill: |v| increases → faster progress
   - Oscillating: opposing gradients cancel → damped oscillations

3. **Physical Analogy**: Ball rolling downhill:
   - Gradient = instantaneous slope
   - Velocity = accumulated motion
   - Momentum = resistance to direction change
   - Heavy ball (high β) smooths out bumps

**Advantages Over Vanilla GD:**

**1. Faster Convergence in Ravines:**

Consider f(x,y) = x²/2 + 10y² (steep in y, shallow in x):

Vanilla GD:
- Large gradient in y-direction causes oscillation
- Must use small α to avoid divergence
- Slow progress in x-direction

With Momentum:
- Oscillations in y dampen out (opposing gradients cancel)
- Consistent gradient in x builds velocity
- Net effect: smoother, faster path

**2. Escape from Plateaus:**

On flat regions where ||∇L|| ≈ 0:

Vanilla GD: θ_{t+1} ≈ θ_t (stuck)
Momentum: v_t ≠ 0 from past gradients → continues moving

**3. Reduced Sensitivity to Noise:**

In stochastic gradient descent with noisy gradients:
- Individual gradients may be poor estimates
- Exponential average smooths noise
- More stable optimization trajectory

**Mathematical Analysis:**

Consider quadratic loss L(θ) = ½θᵀQθ:

Without momentum:
- Eigenvalues of Q determine convergence
- Convergence rate ≈ (λ_max - λ_min)/(λ_max + λ_min)
- Poor when condition number λ_max/λ_min is large

With momentum:
- Effective damping of oscillations
- Better convergence rate
- Like solving with preconditioner

**Practical Considerations:**

**Hyperparameters:**
- β = 0.9 (common): 90% of past velocity retained
- β = 0.99 (for large batches): longer memory
- α typically needs to be smaller than vanilla GD

**Modern Variants:**
- **Nesterov Momentum**: "Look ahead" before computing gradient
  v_{t+1} = βv_t - α∇L(θ_t + βv_t)
  θ_{t+1} = θ_t + v_{t+1}
  
- **Adam**: Combines momentum with adaptive learning rates
  - First moment (momentum): m_t = β₁m_{t-1} + (1-β₁)∇L_t
  - Second moment (variance): v_t = β₂v_{t-1} + (1-β₂)(∇L_t)²

**Visual Understanding:**

Imagine gradient descent as a sequence of independent steps, each responding only to local gradient. Momentum adds memory - the optimization "remembers" where it's been going and continues in consistent directions while damping oscillations.

This makes momentum especially valuable for:
- Ill-conditioned problems (elongated loss contours)
- Noisy gradients (stochastic optimization)
- Saddle point escape (velocity carries through flat regions)
- Deep networks (accumulates signal through many layers)`,
          keyPoints: [
            'Momentum accumulates exponential average of gradients',
            'Accelerates in consistent directions, dampens oscillations',
            'Physical analogy: heavy ball rolling with inertia',
            'Escapes plateaus and smooths noisy gradients',
            'Essential for training deep networks efficiently',
            'Modern optimizers (Adam) extend momentum concept',
          ],
        },
        {
          id: 'grad-dir-disc-3',
          question:
            'Explain projected gradient descent for constrained optimization. Why is it important in machine learning, and how does it differ from penalty methods?',
          hint: 'Consider optimizing on manifolds like the unit sphere, and applications like orthogonal weights or probability simplexes.',
          sampleAnswer: `**Projected Gradient Descent (PGD):**

**Algorithm:**
1. Compute gradient: g = ∇f(x_t)
2. Take gradient step: y = x_t - αg
3. Project onto constraint set: x_{t+1} = Proj_C(y)

where Proj_C(y) = argmin_{x∈C} ||x - y||² (closest point in C)

**Why Project Rather Than Penalize?**

**Penalty Method:**
Minimize: f(x) + λ·penalty(constraint violation)

Example: Optimize on unit sphere
L(x) = f(x) + λ(||x||² - 1)²

Problems:
- Never exactly satisfies constraints
- Requires tuning penalty weight λ
- Can make optimization harder (adds curvature)
- Doesn't scale well to hard constraints

**Projected Gradient:**
- Always feasible (constraint always satisfied)
- No hyperparameter tuning for constraint
- Separates objective from constraints
- Often has closed-form projection

**Common Projections in ML:**

**1. Unit Sphere (Orthogonality):**

Constraint: ||x|| = 1
Projection: Proj(y) = y/||y||

Application: Spectral normalization in GANs, weight normalization

\`\`\`python
def project_sphere(x):
    return x / np.linalg.norm(x)
\`\`\`

**2. Probability Simplex:**

Constraint: xᵢ ≥ 0, Σxᵢ = 1
Projection: Euclidean projection onto simplex (O(n log n) algorithm)

Application: Attention weights, mixture models, portfolio optimization

**3. Box Constraints:**

Constraint: a ≤ x ≤ b
Projection: Proj(y)ᵢ = clip(yᵢ, aᵢ, bᵢ)

Application: Bounded parameters, adversarial perturbations (ε-ball)

\`\`\`python
def project_box(x, a, b):
    return np.clip(x, a, b)
\`\`\`

**4. Low-Rank Matrices:**

Constraint: rank(X) ≤ r
Projection: SVD truncation

Application: Matrix factorization, collaborative filtering

**ML Applications:**

**1. Adversarial Training:**

Generate adversarial examples within ε-ball:
- Maximize loss: x_adv = x + δ
- Constraint: ||δ||_∞ ≤ ε
- Use projected gradient ascent

\`\`\`python
for step in range(num_steps):
    grad = compute_gradient(x_adv, target)
    x_adv = x_adv + alpha * np.sign(grad)
    # Project to ε-ball around x
    x_adv = np.clip(x_adv, x - epsilon, x + epsilon)
\`\`\`

**2. Fairness Constraints:**

Optimize with fairness constraints:
- Demographic parity: P(ŷ=1|A=0) = P(ŷ=1|A=1)
- Project parameters to satisfy fairness metrics
- Ensures fairness while minimizing loss

**3. Sparse Learning:**

L0 constraint (at most k non-zeros):
- After gradient step, keep top-k magnitudes
- Project to k-sparse vectors

\`\`\`python
def project_sparse(x, k):
    indices = np.argsort(np.abs(x))[-k:]
    x_sparse = np.zeros_like(x)
    x_sparse[indices] = x[indices]
    return x_sparse
\`\`\`

**4. Orthogonal Weights:**

Constraint: WᵀW = I (orthogonal matrix)
Projection: W_{new} = U @ Vᵀ (via SVD: W = USVᵀ)

Application: Improving gradient flow, preventing vanishing gradients

**Theoretical Properties:**

**Convergence:**
For convex f and convex constraint set C:
- PGD converges to optimal solution
- Rate depends on Lipschitz constant and projection quality

**Computational Cost:**
- Projection must be efficient
- O(n) to O(n²) typically acceptable
- Can be parallelized

**Comparison Summary:**

| Method | Constraint Satisfaction | Hyperparameters | Complexity |
|--------|------------------------|-----------------|------------|
| Penalty | Approximate | λ tuning needed | Simple |
| Lagrange | Exact (at optimum) | Dual variables | Complex |
| Projected GD | Always exact | None | Moderate |

**Best Practices:**

1. **Use projection when:**
   - Hard constraints required
   - Efficient projection available
   - Constraints are convex

2. **Use penalty when:**
   - Soft constraints acceptable
   - Projection expensive
   - Complex constraint interactions

3. **Hybrid approach:**
   - Project for critical constraints
   - Penalize for soft constraints
   - Example: Project to feasible region, penalize violations within

**Modern Extensions:**

- **Mirror Descent**: Project in dual space (natural gradients)
- **Proximal Methods**: Generalize projection (add regularization)
- **Barrier Methods**: Approach boundary from interior
- **Frank-Wolfe**: Linear minimization oracle instead of projection

Projected gradient descent is essential when mathematical or physical constraints must be exactly satisfied, making it invaluable for robust ML systems.`,
          keyPoints: [
            'PGD alternates gradient step with projection onto constraints',
            'Always feasible (constraints satisfied at every iteration)',
            'No penalty hyperparameters needed',
            'Common in adversarial training, fairness, sparsity',
            'Efficient for simple constraints (sphere, box, simplex)',
            'Theoretically grounded with convergence guarantees',
          ],
        },
      ],
    },
    {
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
    def __init__(self, value, children=(), op=''):
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
      multipleChoice: [
        {
          id: 'chain-multi-1',
          question: 'For z = f(x,y) where x = g(t), y = h(t), what is dz/dt?',
          options: [
            '(∂f/∂x) + (∂f/∂y)',
            '(∂f/∂x)(dx/dt) + (∂f/∂y)(dy/dt)',
            '(∂f/∂x)(∂f/∂y)',
            'dx/dt + dy/dt',
          ],
          correctAnswer: 1,
          explanation:
            'Multivariable chain rule sums contributions from all paths: dz/dt = (∂f/∂x)(dx/dt) + (∂f/∂y)(dy/dt).',
          difficulty: 'medium',
        },
        {
          id: 'chain-multi-2',
          question: 'Backpropagation in neural networks is:',
          options: [
            'Random weight updates',
            'Repeated application of the chain rule',
            'Only forward propagation',
            'A heuristic without mathematical foundation',
          ],
          correctAnswer: 1,
          explanation:
            'Backpropagation is the systematic application of the chain rule to compute gradients through composed functions (layers).',
          difficulty: 'easy',
        },
        {
          id: 'chain-multi-3',
          question: 'In a computational graph, gradients flow:',
          options: [
            'Only forward',
            'Backward from output to inputs via chain rule',
            'Randomly',
            'Only to the first layer',
          ],
          correctAnswer: 1,
          explanation:
            'Gradients flow backward through the computational graph, with each node applying the chain rule to propagate gradients to its inputs.',
          difficulty: 'easy',
        },
        {
          id: 'chain-multi-4',
          question:
            'For a 3-layer network X→Layer1→Layer2→Layer3→Loss, how many chain rule applications are needed for ∂L/∂W1?',
          options: ['1', '2', '3 (through all layers)', '0'],
          correctAnswer: 2,
          explanation:
            'Computing ∂L/∂W1 requires chain rule through all layers: Layer3→Layer2→Layer1, three applications total.',
          difficulty: 'medium',
        },
        {
          id: 'chain-multi-5',
          question: 'The Jacobian matrix in the chain rule represents:',
          options: [
            'All partial derivatives of outputs with respect to inputs',
            'Only first derivatives',
            'Second derivatives',
            'The loss function',
          ],
          correctAnswer: 0,
          explanation:
            'The Jacobian matrix contains all first-order partial derivatives, capturing how each output component depends on each input component.',
          difficulty: 'medium',
        },
      ],
      quiz: [
        {
          id: 'chain-multi-disc-1',
          question:
            'Explain why backpropagation is efficient compared to naively computing all gradients. What is the computational complexity difference?',
          hint: 'Consider computing n gradients independently vs. backpropagation. Think about redundant computations.',
          sampleAnswer: `**Naive Gradient Computation vs. Backpropagation:**

**Naive Approach:**
Compute each gradient ∂L/∂θᵢ independently using the definition or numerical differentiation.

For n parameters:
- Each gradient: O(forward pass cost)
- Total: O(n × forward pass)
- For deep network with millions of parameters: prohibitively expensive

**Backpropagation Approach:**
Single backward pass computes ALL gradients simultaneously.

**Why It's Efficient:**

1. **Shared Computations:**
   
   Consider network: x → f₁ → f₂ → f₃ → L
   
   Gradients share intermediate terms:
   - ∂L/∂θ₁ = ∂L/∂f₃ · ∂f₃/∂f₂ · ∂f₂/∂f₁ · ∂f₁/∂θ₁
   - ∂L/∂θ₂ = ∂L/∂f₃ · ∂f₃/∂f₂ · ∂f₂/∂θ₂
   
   Both use ∂L/∂f₃ and ∂f₃/∂f₂ - computed once, reused!

2. **Dynamic Programming:**
   
   Backpropagation is essentially dynamic programming:
   - Store intermediate gradients
   - Reuse them for earlier layers
   - No redundant computation

3. **Computational Complexity:**

   **Forward pass:** O(W) where W = number of weights
   
   **Backward pass:** O(W) - same as forward!
   
   **Total for all gradients:** O(W)
   
   vs.
   
   **Naive:** O(W × W) = O(W²)

4. **Concrete Example:**

   Network with L layers, each with n parameters:
   - Total parameters: N = L × n
   
   **Naive:**
   - Each gradient: evaluate network (O(N))
   - N gradients: O(N²)
   - For N = 10⁶: ~10¹² operations
   
   **Backpropagation:**
   - Forward: O(N)
   - Backward: O(N)
   - Total: O(N)
   - For N = 10⁶: ~10⁶ operations
   
   **Speedup:** O(N) → **Million times faster!**

5. **Why This Works - Automatic Differentiation:**

   Backpropagation is reverse-mode automatic differentiation:
   
   **Forward mode:** Computes all derivatives of one output wrt all inputs
   - Efficient when: few inputs, many outputs
   - Complexity: O(inputs)
   
   **Reverse mode (backprop):** Computes all derivatives of one output wrt all inputs
   - Efficient when: many inputs, few outputs
   - Complexity: O(outputs)
   
   Neural networks: many parameters (inputs), one loss (output) → reverse mode optimal!

**Practical Impact:**

Without backpropagation's efficiency, training large neural networks would be impossible:
- GPT-3: 175 billion parameters
- Naive: 175B² ≈ 3×10²² operations per gradient step (centuries)
- Backprop: 175B operations (seconds)

The O(n) vs O(n²) difference is what enables deep learning at scale.`,
          keyPoints: [
            'Backprop computes all gradients in O(n) time vs naive O(n²)',
            'Shares intermediate computations via dynamic programming',
            'Reverse-mode autodiff optimal for many params, one loss',
            'Makes training billion-parameter models feasible',
            'Foundation of all modern deep learning frameworks',
          ],
        },
        {
          id: 'chain-multi-disc-2',
          question:
            'Describe how modern deep learning frameworks use computational graphs and automatic differentiation. How does this relate to the chain rule?',
          hint: "Consider PyTorch/TensorFlow's autograd, define-by-run vs static graphs, and gradient tape.",
          sampleAnswer: `**Computational Graphs and Automatic Differentiation:**

Modern frameworks (PyTorch, TensorFlow, JAX) automatically apply the chain rule through computational graphs.

**1. Computational Graph Structure:**

A computational graph represents function composition:
- **Nodes**: Operations or variables
- **Edges**: Data dependencies
- **Forward pass**: Evaluate nodes in topological order
- **Backward pass**: Propagate gradients in reverse order

Example: L = (W·x + b)²
\`\`\`
x, W, b → z1 = W·x → z2 = z1 + b → L = z2²
\`\`\`

**2. Two Paradigms:**

**Static Graphs (TensorFlow 1.x):**
\`\`\`python
# Define graph first
x = tf.placeholder()
W = tf.Variable()
z1 = tf.matmul(W, x)
loss = tf.reduce_sum(z1**2)

# Then execute
with tf.Session() as sess:
    result = sess.run(loss, feed_dict={x: data})
\`\`\`

Pros: Can optimize graph before execution
Cons: Less flexible, harder to debug

**Dynamic Graphs (PyTorch, TensorFlow 2.x):**
\`\`\`python
# Define and execute simultaneously
z1 = W @ x
loss = (z1**2).sum()
loss.backward()  # Automatic differentiation
\`\`\`

Pros: Pythonic, easy debugging, dynamic control flow
Cons: Less optimization opportunity

**3. How Automatic Differentiation Works:**

**Forward Pass (build graph):**
\`\`\`python
import torch

x = torch.tensor([1.0, 2.0], requires_grad=True)
W = torch.tensor([[1.0, 0.5], [0.5, 1.0]], requires_grad=True)

# Each operation creates a node
z1 = W @ x  # MatMul node
z2 = z1.sum()  # Sum node
loss = z2**2  # Pow node

# Graph: x, W → MatMul → Sum → Pow → loss
\`\`\`

**Backward Pass (apply chain rule):**
\`\`\`python
loss.backward()

# Internally, framework:
# 1. ∂loss/∂loss = 1
# 2. ∂loss/∂z2 = ∂loss/∂loss · ∂loss/∂z2 = 1 · 2z2
# 3. ∂loss/∂z1 = ∂loss/∂z2 · ∂z2/∂z1 = (2z2) · 1
# 4. ∂loss/∂W = ∂loss/∂z1 · ∂z1/∂W = (2z2) · x^T
# 5. ∂loss/∂x = ∂loss/∂z1 · ∂z1/∂x = (2z2) · W^T

print(x.grad)  # ∂loss/∂x
print(W.grad)  # ∂loss/∂W
\`\`\`

**4. Under the Hood:**

Each operation stores its local gradient function:

\`\`\`python
class MulOp:
    def forward(self, a, b):
        self.a = a
        self.b = b
        return a * b
    
    def backward(self, grad_output):
        # Chain rule: ∂L/∂a = ∂L/∂out · ∂out/∂a = grad_output · b
        grad_a = grad_output * self.b
        grad_b = grad_output * self.a
        return grad_a, grad_b
\`\`\`

**5. Gradient Tape (TensorFlow 2.x approach):**

\`\`\`python
import tensorflow as tf

with tf.GradientTape() as tape:
    x = tf.Variable([1.0, 2.0])
    y = x**2
    loss = tf.reduce_sum(y)

# Tape records all operations
gradients = tape.gradient(loss, x)
# Applies chain rule backward through recorded operations
\`\`\`

**6. Advanced Features:**

**Higher-Order Derivatives:**
\`\`\`python
x = torch.tensor(2.0, requires_grad=True)
y = x**3
dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]
d2y_dx2 = torch.autograd.grad(dy_dx, x)[0]
print(f"d²y/dx² = {d2y_dx2}")  # Second derivative
\`\`\`

**Jacobian/Hessian:**
\`\`\`python
from torch.autograd.functional import jacobian, hessian

def f(x):
    return torch.sum(x**2)

x = torch.tensor([1.0, 2.0, 3.0])
J = jacobian(f, x)  # ∂f/∂x for vector f
H = hessian(f, x)   # ∂²f/∂x∂x
\`\`\`

**Gradient Accumulation:**
\`\`\`python
optimizer.zero_grad()
for batch in mini_batches:
    loss = model(batch)
    loss.backward()  # Accumulates gradients
optimizer.step()
\`\`\`

**7. Connection to Chain Rule:**

Every operation in the graph knows:
1. How to compute forward (output from inputs)
2. How to compute backward (gradient wrt inputs from gradient wrt output)

The framework:
1. Builds graph during forward pass
2. Traverses graph backward
3. At each node, applies local chain rule
4. Accumulates gradients through all paths

**Example - Why It's Powerful:**

\`\`\`python
# Complex computation graph
x = torch.randn(100, 10)
W1 = torch.randn(10, 50, requires_grad=True)
W2 = torch.randn(50, 20, requires_grad=True)

# Computation
h = torch.relu(x @ W1)
h = torch.dropout(h, 0.5, training=True)
y = torch.softmax(h @ W2, dim=1)
loss = -torch.log(y[range(100), labels]).mean()

# One line gets all gradients!
loss.backward()

# Framework applied chain rule through:
# log → softmax → matmul → dropout → relu → matmul
# All automatically, all correctly
\`\`\`

**Modern Innovations:**

- **JIT Compilation**: Optimize computational graph
- **Quantization**: Mixed precision training
- **Graph Optimization**: Fuse operations
- **Distributed Autograd**: Across multiple GPUs
- **Checkpointing**: Trade computation for memory

The beauty: as a user, you just write forward pass. The framework handles all chain rule complexity!`,
          keyPoints: [
            'Computational graphs represent function compositions',
            'Automatic differentiation applies chain rule automatically',
            'Dynamic graphs (PyTorch) build during execution',
            'Each operation stores local gradient function',
            'Backward pass traverses graph applying chain rule',
            'Enables complex models with one-line gradient computation',
          ],
        },
        {
          id: 'chain-multi-disc-3',
          question:
            'Explain the difference between forward-mode and reverse-mode automatic differentiation. Why is reverse-mode (backpropagation) preferred for neural networks?',
          hint: 'Consider a function f: ℝⁿ → ℝᵐ and the cost of computing all derivatives for different n and m.',
          sampleAnswer: `**Forward-Mode vs Reverse-Mode Automatic Differentiation:**

Both compute exact derivatives using the chain rule, but in different orders.

**1. Forward-Mode AD:**

**Strategy:** Propagate derivatives forward along with values.

**How it works:**
For y = f(x₁, ..., xₙ), compute ∂y/∂xᵢ by:
1. Set dx_i/dx_i = 1, dx_j/dx_i = 0 for j≠i
2. Propagate derivatives forward through operations
3. Result: dy/dx_i

**Example:** y = x₁·x₂ + sin(x₁)

Compute ∂y/∂x₁:
\`\`\`
Forward pass with derivatives:

x₁: value=2, derivative=1 (∂x₁/∂x₁=1)
x₂: value=3, derivative=0 (∂x₂/∂x₁=0)

v₁ = x₁·x₂: 
  value = 2·3 = 6
  derivative = 1·3 + 2·0 = 3  (product rule)

v₂ = sin(x₁):
  value = sin(2) ≈ 0.909
  derivative = cos(2)·1 ≈ -0.416  (chain rule)

y = v₁ + v₂:
  value = 6.909
  derivative = 3 + (-0.416) = 2.584 (∂y/∂x₁)
\`\`\`

**Complexity:** 
- One forward pass per input variable
- For n inputs: O(n × forward_cost)
- Efficient when n is small

**2. Reverse-Mode AD (Backpropagation):**

**Strategy:** Compute all derivatives in one backward pass.

**How it works:**
For L = f(x₁, ..., xₙ), compute all ∂L/∂xᵢ by:
1. Forward pass: Compute values, save intermediates
2. Set ∂L/∂L = 1
3. Propagate gradients backward
4. Result: All ∂L/∂xᵢ simultaneously

**Same example:** L = x₁·x₂ + sin(x₁)

\`\`\`
Forward pass (save values):
  x₁=2, x₂=3
  v₁ = 6
  v₂ = 0.909
  L = 6.909

Backward pass:
  ∂L/∂L = 1

  ∂L/∂v₁ = ∂L/∂L · ∂L/∂v₁ = 1 · 1 = 1
  ∂L/∂v₂ = ∂L/∂L · ∂L/∂v₂ = 1 · 1 = 1

  ∂L/∂x₁ = ∂L/∂v₁·∂v₁/∂x₁ + ∂L/∂v₂·∂v₂/∂x₁
          = 1·3 + 1·cos(2) = 2.584

  ∂L/∂x₂ = ∂L/∂v₁·∂v₁/∂x₂ = 1·2 = 2
\`\`\`

**Complexity:**
- One forward + one backward pass
- For n inputs: O(forward_cost + backward_cost) ≈ O(2×forward_cost)
- Efficient when n is large, one output

**3. Comparison Table:**

| Aspect | Forward-Mode | Reverse-Mode |
|--------|--------------|--------------|
| **Passes** | n forward passes | 1 forward + 1 backward |
| **Complexity** | O(n × C) | O(C) |
| **Best for** | f: ℝⁿ→ℝᵐ, n≪m | f: ℝⁿ→ℝᵐ, m≪n |
| **Memory** | Low | High (store intermediates) |
| **Example use** | Jacobian-vector products | Gradient of scalar loss |

where C = cost of evaluating f

**4. Why Reverse-Mode for Neural Networks?**

**Scenario:**
- Inputs: n = millions of parameters
- Output: m = 1 (scalar loss)

**Forward-Mode:**
- Need n forward passes
- One per parameter
- Compute ∂L/∂θ₁, ∂L/∂θ₂, ..., ∂L/∂θₙ separately
- Cost: O(n × forward_cost)
- For n = 10⁶: ~million forward passes!

**Reverse-Mode:**
- One forward + one backward pass
- Computes all ∂L/∂θᵢ simultaneously
- Cost: O(2 × forward_cost)
- Independent of n!

**Speedup:** n/2 = 500,000× faster for n=10⁶

**5. Practical Example:**

\`\`\`python
import torch

# Network with 1M parameters
n_params = 1_000_000
params = torch.randn(n_params, requires_grad=True)

def loss_fn(p):
    # Some complex computation
    return (p**2).sum()

# Reverse-mode (backprop): O(1)
loss = loss_fn(params)
loss.backward()  # All gradients in one backward pass
print(params.grad.shape)  # (1000000,) - all gradients!

# Forward-mode would need:
# for i in range(n_params):
#     compute ∂loss/∂params[i]  # 1M forward passes!
\`\`\`

**6. When to Use Each:**

**Use Forward-Mode when:**
- Few inputs, many outputs (computing Jacobian-vector products)
- Example: Sensitivity analysis (how outputs change with one input)
- Example: Optimal control (Jacobian of system dynamics)

**Use Reverse-Mode when:**
- Many inputs, few outputs (neural network training)
- Computing gradients of scalar objective
- Example: Any optimization problem with gradient descent

**7. Hybrid Approaches:**

Modern AD systems support both:

\`\`\`python
# JAX supports both modes
import jax

def f(x):
    return jax.numpy.sum(x**2)

x = jax.numpy.array([1.0, 2.0, 3.0])

# Reverse-mode (backprop)
grad_reverse = jax.grad(f)(x)

# Forward-mode
def forward_mode_grad(x):
    # Compute gradient via forward-mode
    # (JAX can do this with jax.jvp)
    pass
\`\`\`

**8. Memory Trade-offs:**

**Forward-Mode:**
- Low memory (only current values + derivatives)
- No need to store intermediate values

**Reverse-Mode:**
- High memory (must store all intermediate values)
- Proportional to network depth
- Solutions:
  - Gradient checkpointing (recompute instead of store)
  - Micro-batching (process smaller batches)

**Summary:**

Reverse-mode AD (backpropagation) dominates machine learning because:
1. Neural networks: many parameters (n→∞), one loss (m=1)
2. O(1) vs O(n) complexity
3. Makes billion-parameter models tractable
4. All modern frameworks implement reverse-mode

Forward-mode still useful for specific applications (sensitivity analysis, optimal control), but reverse-mode is the workhorse of deep learning.`,
          keyPoints: [
            'Forward-mode: O(n) passes, efficient for few inputs',
            'Reverse-mode: O(1) passes, efficient for many inputs',
            'Neural networks: n parameters, 1 loss → reverse-mode optimal',
            'Backprop is reverse-mode AD applied to neural networks',
            'Memory trade-off: reverse-mode stores intermediates',
            'Speedup for n=10⁶ parameters: ~500,000× faster',
          ],
        },
      ],
    },
    {
      id: 'integration-basics',
      title: 'Integration Basics',
      content: `
# Integration Basics

## Introduction

Integration is the inverse of differentiation and computes accumulated change. In machine learning, integration appears in:
- Probability distributions (normalizing constants, expectations)
- Loss function derivations
- Continuous optimization theory
- Bayesian inference

## Fundamental Theorem of Calculus

**Part 1**: If F'(x) = f(x), then:
∫ₐᵇ f(x)dx = F(b) - F(a)

**Part 2**: If g(x) = ∫ₐˣ f(t)dt, then g'(x) = f(x)

**Intuition**: Integration and differentiation are inverse operations.

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, cumulative_trapezoid

# Demonstrate Fundamental Theorem of Calculus

def f(x):
    """Function to integrate: f(x) = x²"""
    return x**2

def F(x):
    """Antiderivative: F(x) = x³/3"""
    return x**3 / 3

# Part 1: ∫ₐᵇ f(x)dx = F(b) - F(a)
a, b = 1, 3

# Analytical integration
analytical_integral = F(b) - F(a)
print(f"Part 1: Fundamental Theorem of Calculus")
print(f"∫₁³ x² dx = F(3) - F(1) = {F(b)} - {F(a)} = {analytical_integral:.4f}")

# Numerical integration (verification)
numerical_integral, error = quad(f, a, b)
print(f"Numerical verification: {numerical_integral:.4f}")
print(f"Error: {abs(analytical_integral - numerical_integral):.2e}")

# Part 2: d/dx [∫ₐˣ f(t)dt] = f(x)
print(f"\\nPart 2: Derivative of integral")

def integral_function(x):
    """g(x) = ∫₁ˣ t² dt"""
    result, _ = quad(f, 1, x)
    return result

# Test at x = 2
x_test = 2.0
h = 1e-7

# Derivative of integral (numerical)
g_x = integral_function(x_test)
g_x_h = integral_function(x_test + h)
derivative_of_integral = (g_x_h - g_x) / h

# Original function value
f_x = f(x_test)

print(f"At x = {x_test}:")
print(f"d/dx [∫₁ˣ t² dt] = {derivative_of_integral:.4f}")
print(f"f(x) = x² = {f_x:.4f}")
print(f"Error: {abs(derivative_of_integral - f_x):.2e}")
\`\`\`

## Basic Integration Rules

**Power Rule**:
∫ xⁿ dx = xⁿ⁺¹/(n+1) + C (n ≠ -1)

**Constant Multiple**:
∫ cf(x)dx = c∫f(x)dx

**Sum/Difference**:
∫ [f(x) ± g(x)]dx = ∫f(x)dx ± ∫g(x)dx

\`\`\`python
from sympy import symbols, integrate, exp, sin, cos, log

x = symbols('x')

# Power rule
print("Integration Rules:")
print("="*60)

functions = [
    (x**3, "x³"),
    (x**(-2), "x⁻²"),
    (5*x**2, "5x²"),
    (sin(x), "sin(x)"),
    (cos(x), "cos(x)"),
    (exp(x), "eˣ"),
    (1/x, "1/x"),
    (x**2 + 3*x + 5, "x² + 3x + 5")
]

for func, name in functions:
    integral = integrate(func, x)
    print(f"∫ {name} dx = {integral} + C")
\`\`\`

## Definite vs Indefinite Integrals

**Indefinite Integral** (antiderivative):
∫ f(x)dx = F(x) + C

**Definite Integral** (area):
∫ₐᵇ f(x)dx = F(b) - F(a)

\`\`\`python
def visualize_definite_integral():
    """Visualize definite integral as area under curve"""
    
    # Function: f(x) = x² - 2x + 3
    def f(x):
        return x**2 - 2*x + 3
    
    def F(x):
        return x**3/3 - x**2 + 3*x
    
    a, b = 0, 3
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Left: Function and area
    x = np.linspace(-0.5, 3.5, 200)
    y = f(x)
    
    ax1.plot(x, y, 'b-', linewidth=2, label='f(x) = x² - 2x + 3')
    
    # Shade area under curve
    x_fill = np.linspace(a, b, 100)
    y_fill = f(x_fill)
    ax1.fill_between(x_fill, 0, y_fill, alpha=0.3, color='blue', label=f'∫₀³ f(x)dx')
    
    ax1.axvline(a, color='red', linestyle='--', alpha=0.7)
    ax1.axvline(b, color='red', linestyle='--', alpha=0.7)
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_title('Definite Integral as Area')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: Antiderivative
    x2 = np.linspace(-0.5, 3.5, 200)
    y2 = [F(xi) for xi in x2]
    
    ax2.plot(x2, y2, 'g-', linewidth=2, label='F(x) = x³/3 - x² + 3x')
    ax2.plot([a, b], [F(a), F(b)], 'ro', markersize=8)
    ax2.axhline(F(a), color='red', linestyle='--', alpha=0.5, label=f'F({a}) = {F(a):.2f}')
    ax2.axhline(F(b), color='blue', linestyle='--', alpha=0.5, label=f'F({b}) = {F(b):.2f}')
    
    # Show difference
    ax2.annotate('', xy=(3.3, F(b)), xytext=(3.3, F(a)),
                arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
    ax2.text(3.5, (F(a) + F(b))/2, f'F(b)-F(a)\\n={F(b)-F(a):.2f}', 
            fontsize=10, va='center')
    
    ax2.set_xlabel('x')
    ax2.set_ylabel('F(x)')
    ax2.set_title('Antiderivative F(x)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('integration_visualization.png', dpi=150, bbox_inches='tight')
    print(f"\\nSaved visualization to 'integration_visualization.png'")
    
    # Compute integral
    integral_value = F(b) - F(a)
    print(f"\\n∫₀³ (x² - 2x + 3)dx = F(3) - F(0) = {F(b):.4f} - {F(a):.4f} = {integral_value:.4f}")

visualize_definite_integral()
\`\`\`

## Integration Techniques

### Substitution (Change of Variables)

For ∫ f(g(x))·g'(x)dx, let u = g(x):
∫ f(g(x))·g'(x)dx = ∫ f(u)du

\`\`\`python
def integration_by_substitution():
    """
    Example: ∫ 2x·cos(x²) dx
    Let u = x², then du = 2x dx
    ∫ cos(u) du = sin(u) + C = sin(x²) + C
    """
    
    # Symbolic
    x = symbols('x')
    integrand = 2*x * cos(x**2)
    result = integrate(integrand, x)
    print("Integration by Substitution:")
    print(f"∫ 2x·cos(x²) dx = {result} + C")
    
    # Verify numerically
    def f(x_val):
        return 2*x_val * np.cos(x_val**2)
    
    def F(x_val):
        return np.sin(x_val**2)
    
    a, b = 0, 2
    analytical = F(b) - F(a)
    numerical, _ = quad(f, a, b)
    
    print(f"\\n∫₀² 2x·cos(x²) dx:")
    print(f"Analytical: {analytical:.6f}")
    print(f"Numerical: {numerical:.6f}")
    print(f"Error: {abs(analytical - numerical):.2e}")

integration_by_substitution()
\`\`\`

### Integration by Parts

∫ u dv = uv - ∫ v du

\`\`\`python
def integration_by_parts_example():
    """
    Example: ∫ x·eˣ dx
    Let u = x, dv = eˣ dx
    Then du = dx, v = eˣ
    ∫ x·eˣ dx = x·eˣ - ∫ eˣ dx = x·eˣ - eˣ + C = eˣ(x-1) + C
    """
    
    x = symbols('x')
    integrand = x * exp(x)
    result = integrate(integrand, x)
    print("Integration by Parts:")
    print(f"∫ x·eˣ dx = {result} + C")
    
    # Verify
    def f(x_val):
        return x_val * np.exp(x_val)
    
    def F(x_val):
        return np.exp(x_val) * (x_val - 1)
    
    a, b = 0, 2
    analytical = F(b) - F(a)
    numerical, _ = quad(f, a, b)
    
    print(f"\\n∫₀² x·eˣ dx:")
    print(f"Analytical: {analytical:.6f}")
    print(f"Numerical: {numerical:.6f}")
    print(f"Error: {abs(analytical - numerical):.2e}")

integration_by_parts_example()
\`\`\`

## Numerical Integration

When analytical integration is impossible, use numerical methods.

### Riemann Sums

\`\`\`python
def riemann_sum(f, a, b, n, method='midpoint'):
    """
    Compute Riemann sum
    
    Methods:
    - 'left': Left endpoint
    - 'right': Right endpoint
    - 'midpoint': Midpoint rule
    """
    dx = (b - a) / n
    x = np.linspace(a, b, n+1)
    
    if method == 'left':
        return dx * sum(f(x[i]) for i in range(n))
    elif method == 'right':
        return dx * sum(f(x[i+1]) for i in range(n))
    elif method == 'midpoint':
        midpoints = (x[:-1] + x[1:]) / 2
        return dx * sum(f(m) for m in midpoints)
    else:
        raise ValueError(f"Unknown method: {method}")

# Test
def f(x):
    return np.exp(-x**2)

a, b = 0, 2
true_value, _ = quad(f, a, b)

print("Riemann Sums:")
print(f"True value: {true_value:.8f}")
print()

for n in [10, 100, 1000]:
    for method in ['left', 'right', 'midpoint']:
        approx = riemann_sum(f, a, b, n, method)
        error = abs(approx - true_value)
        print(f"n={n:4d}, {method:8s}: {approx:.8f}, error={error:.2e}")
    print()
\`\`\`

### Trapezoidal Rule

∫ₐᵇ f(x)dx ≈ (b-a)/(2n) · [f(x₀) + 2f(x₁) + ... + 2f(xₙ₋₁) + f(xₙ)]

\`\`\`python
from scipy.integrate import trapezoid

def trapezoidal_rule(f, a, b, n):
    """Trapezoidal rule for numerical integration"""
    x = np.linspace(a, b, n+1)
    y = f(x)
    return trapezoid(y, x)

print("Trapezoidal Rule:")
for n in [10, 100, 1000]:
    approx = trapezoidal_rule(f, a, b, n)
    error = abs(approx - true_value)
    print(f"n={n:4d}: {approx:.8f}, error={error:.2e}")
\`\`\`

### Simpson's Rule

More accurate: uses parabolic approximation.

∫ₐᵇ f(x)dx ≈ (b-a)/(3n) · [f(x₀) + 4f(x₁) + 2f(x₂) + 4f(x₃) + ... + f(xₙ)]

\`\`\`python
from scipy.integrate import simpson

def simpsons_rule(f, a, b, n):
    """Simpson's rule for numerical integration"""
    if n % 2 != 0:
        n += 1  # Simpson's requires even n
    x = np.linspace(a, b, n+1)
    y = f(x)
    return simpson(y, x=x)

print("\\nSimpson's Rule:")
for n in [10, 100, 1000]:
    approx = simpsons_rule(f, a, b, n)
    error = abs(approx - true_value)
    print(f"n={n:4d}: {approx:.8f}, error={error:.2e}")
\`\`\`

## Applications in ML: Expectation

Expectation of continuous random variable:
E[X] = ∫₋∞^∞ x·f(x)dx

where f(x) is the probability density function.

\`\`\`python
def compute_expectations():
    """Compute expectations via integration"""
    
    # Normal distribution: X ~ N(μ, σ²)
    mu, sigma = 2.0, 1.5
    
    def pdf(x):
        """Probability density function"""
        return (1 / (sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu)/sigma)**2)
    
    # E[X]
    def integrand_mean(x):
        return x * pdf(x)
    
    expected_value, _ = quad(integrand_mean, -np.inf, np.inf)
    print(f"Normal Distribution: N({mu}, {sigma}²)")
    print(f"E[X] = {expected_value:.6f} (true: {mu})")
    
    # E[X²]
    def integrand_second_moment(x):
        return x**2 * pdf(x)
    
    second_moment, _ = quad(integrand_second_moment, -np.inf, np.inf)
    print(f"E[X²] = {second_moment:.6f} (true: {mu**2 + sigma**2:.6f})")
    
    # Var(X) = E[X²] - (E[X])²
    variance = second_moment - expected_value**2
    print(f"Var(X) = {variance:.6f} (true: {sigma**2:.6f})")
    
    # Probability: P(X ∈ [a, b])
    a, b = 1, 3
    prob, _ = quad(pdf, a, b)
    print(f"\\nP({a} ≤ X ≤ {b}) = {prob:.6f}")

compute_expectations()
\`\`\`

## Applications in ML: Loss Functions

Many loss functions involve integrals (KL divergence, cross-entropy for continuous distributions).

\`\`\`python
def kl_divergence_continuous():
    """
    KL divergence for continuous distributions:
    D_KL(P||Q) = ∫ p(x) log(p(x)/q(x)) dx
    """
    
    # Two normal distributions
    # P ~ N(0, 1)
    # Q ~ N(1, 1.5²)
    
    mu_p, sigma_p = 0.0, 1.0
    mu_q, sigma_q = 1.0, 1.5
    
    def p(x):
        return (1/(sigma_p*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu_p)/sigma_p)**2)
    
    def q(x):
        return (1/(sigma_q*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu_q)/sigma_q)**2)
    
    def kl_integrand(x):
        p_x = p(x)
        q_x = q(x)
        if p_x < 1e-10:  # Avoid numerical issues
            return 0
        return p_x * np.log(p_x / (q_x + 1e-10))
    
    kl_numerical, _ = quad(kl_integrand, -10, 10)
    
    # Analytical formula for KL between two Gaussians:
    # D_KL(N(μ₁,σ₁²)||N(μ₂,σ₂²)) = log(σ₂/σ₁) + (σ₁² + (μ₁-μ₂)²)/(2σ₂²) - 1/2
    kl_analytical = np.log(sigma_q/sigma_p) + (sigma_p**2 + (mu_p-mu_q)**2)/(2*sigma_q**2) - 0.5
    
    print("KL Divergence: D_KL(P||Q)")
    print(f"P ~ N({mu_p}, {sigma_p}²)")
    print(f"Q ~ N({mu_q}, {sigma_q}²)")
    print(f"\\nNumerical (integration): {kl_numerical:.6f}")
    print(f"Analytical formula: {kl_analytical:.6f}")
    print(f"Error: {abs(kl_numerical - kl_analytical):.2e}")

kl_divergence_continuous()
\`\`\`

## Summary

**Key Concepts**:
- Fundamental Theorem: ∫ₐᵇ f(x)dx = F(b) - F(a)
- Integration rules: power, sum, constant multiple
- Techniques: substitution, integration by parts
- Numerical methods: Riemann sums, trapezoidal, Simpson's
- ML applications: expectations, probability, loss functions

**Why This Matters**:
Integration is essential for:
- Probability theory (normalizing constants, expectations)
- Loss function derivations (cross-entropy, KL divergence)
- Continuous optimization theory
- Bayesian inference and posterior distributions
`,
      multipleChoice: [
        {
          id: 'integration-1',
          question: 'The Fundamental Theorem of Calculus states that:',
          options: [
            'Differentiation and integration are unrelated',
            "∫ₐᵇ f(x)dx = F(b) - F(a) where F'(x) = f(x)",
            'All functions can be integrated analytically',
            'Integration always gives a constant',
          ],
          correctAnswer: 1,
          explanation:
            'The Fundamental Theorem connects differentiation and integration: the definite integral of f from a to b equals the antiderivative evaluated at the endpoints.',
          difficulty: 'easy',
        },
        {
          id: 'integration-2',
          question: 'What is ∫ x³ dx?',
          options: ['x⁴ + C', 'x⁴/4 + C', '3x² + C', 'x²/2 + C'],
          correctAnswer: 1,
          explanation:
            'Power rule for integration: ∫ xⁿ dx = xⁿ⁺¹/(n+1) + C. For n=3: x⁴/4 + C.',
          difficulty: 'easy',
        },
        {
          id: 'integration-3',
          question:
            "Simpson's rule is more accurate than the trapezoidal rule because:",
          options: [
            'It uses more function evaluations',
            'It approximates the function with parabolas instead of straight lines',
            'It uses random sampling',
            'It only works for polynomials',
          ],
          correctAnswer: 1,
          explanation:
            "Simpson's rule uses quadratic (parabolic) interpolation between points, providing better approximation than linear (trapezoidal) interpolation.",
          difficulty: 'medium',
        },
        {
          id: 'integration-4',
          question:
            'The expectation E[X] of a continuous random variable is computed as:',
          options: [
            '∫ f(x) dx',
            '∫ x·f(x) dx',
            '∫ x² ·f(x) dx',
            '∫ log(f(x)) dx',
          ],
          correctAnswer: 1,
          explanation:
            'Expectation is the weighted average: E[X] = ∫ x·f(x) dx, where f(x) is the probability density function.',
          difficulty: 'medium',
        },
        {
          id: 'integration-5',
          question:
            'Why is numerical integration necessary in machine learning?',
          options: [
            'Most ML functions have simple analytical integrals',
            'Many probability distributions and loss functions have integrals without closed-form solutions',
            'Analytical integration is always less accurate',
            'Computers cannot do analytical integration',
          ],
          correctAnswer: 1,
          explanation:
            'Many ML problems involve complex integrals (KL divergence, marginal likelihoods, expectations) that lack closed-form solutions, requiring numerical methods.',
          difficulty: 'medium',
        },
      ],
      quiz: [
        {
          id: 'integration-disc-1',
          question:
            'Explain why computing expectations and marginal probabilities in probabilistic models requires integration. How does this relate to Bayesian inference?',
          hint: 'Consider continuous probability distributions, normalization constants, and marginalizing out variables.',
          sampleAnswer: `**Integration in Probabilistic Models and Bayesian Inference:**

Integration is fundamental to probabilistic modeling because continuous probability distributions require integration for:
1. Normalization
2. Computing expectations
3. Marginalizing over variables
4. Bayesian inference

**1. Probability Normalization:**

A probability density function (PDF) must integrate to 1:

∫₋∞^∞ p(x) dx = 1

Example: Normal distribution
\`\`\`
p(x) = (1/Z) · exp(-0.5·((x-μ)/σ)²)
\`\`\`

Where Z = σ√(2π) is the normalizing constant (computed via integration).

**Why needed:**
Without correct normalization, probabilities would be meaningless. Integration ensures the distribution represents valid probabilities.

**2. Computing Expectations:**

The expected value of any function g(X):

E[g(X)] = ∫ g(x)·p(x) dx

**Examples in ML:**

**Mean:**
\`\`\`
E[X] = ∫ x·p(x) dx
\`\`\`

**Variance:**
\`\`\`
Var(X) = E[X²] - (E[X])² = ∫ x²·p(x) dx - (∫ x·p(x) dx)²
\`\`\`

**Loss:**
\`\`\`
E[Loss] = ∫ L(θ, x)·p(x|D) dx
\`\`\`

**Why needed:**
We rarely observe all possible values; integration computes average behavior over the entire distribution.

**3. Marginalizing Over Variables:**

For joint distribution p(x, y), the marginal distribution:

p(x) = ∫ p(x, y) dy

**Example:** Mixture of Gaussians

\`\`\`
p(x) = ∫ p(x|z)·p(z) dz
     = Σₖ πₖ·N(x|μₖ, σₖ²)
\`\`\`

where z is the latent cluster assignment.

**Why needed:**
Often, we have joint distributions over observed and latent variables. To reason about observed variables alone, we must integrate out (marginalize) latent variables.

**4. Bayesian Inference:**

Bayes' theorem for continuous parameters:

p(θ|D) = p(D|θ)·p(θ) / p(D)

where the **evidence** (denominator) requires integration:

p(D) = ∫ p(D|θ)·p(θ) dθ

**Components:**

**Posterior:** p(θ|D)
- Distribution over parameters given data
- What we want to compute

**Likelihood:** p(D|θ)
- Probability of data given parameters
- Easy to evaluate

**Prior:** p(θ)
- Our beliefs before seeing data
- Specified by modeler

**Evidence (marginal likelihood):** p(D)
- Normalizing constant
- **Requires integration over all θ**

**Why integration is hard:**

For high-dimensional θ (e.g., neural network weights), this integral is:
1. **Intractable**: No closed form
2. **High-dimensional**: Curse of dimensionality
3. **Expensive**: Requires many evaluations

**Example: Bayesian Linear Regression**

Model: y = Xw + ε, ε ~ N(0, σ²I)

**Likelihood:**
\`\`\`
p(y|X, w, σ²) = N(y|Xw, σ²I)
\`\`\`

**Prior:**
\`\`\`
p(w) = N(w|0, λ⁻¹I)
\`\`\`

**Posterior:**
\`\`\`
p(w|X, y) = p(y|X, w)·p(w) / p(y|X)
\`\`\`

**Evidence (integral!):**
\`\`\`
p(y|X) = ∫ p(y|X, w)·p(w) dw
\`\`\`

This integral can be computed analytically for linear models, but for most models (neural nets, etc.), it's intractable.

**5. Practical Solutions:**

Since exact integration is often impossible, we use approximations:

**Monte Carlo Integration:**
\`\`\`
∫ f(x)·p(x) dx ≈ (1/N) Σᵢ f(xᵢ), xᵢ ~ p(x)
\`\`\`

Sample from distribution, average function values.

**Variational Inference:**
Approximate intractable posterior p(θ|D) with simpler distribution q(θ):

\`\`\`
q*(θ) = argmin_q KL(q(θ) || p(θ|D))
\`\`\`

Convert integration problem to optimization problem.

**Markov Chain Monte Carlo (MCMC):**
Generate samples from posterior without computing normalizing constant:
- Metropolis-Hastings
- Hamiltonian Monte Carlo
- Gibbs sampling

**Laplace Approximation:**
Approximate posterior with Gaussian around MAP estimate:

\`\`\`
p(θ|D) ≈ N(θ|θ_MAP, H⁻¹)
\`\`\`

where H is the Hessian at θ_MAP.

**6. Concrete Example: Bayesian Neural Network**

Standard NN: Single weight estimate (point estimate)
Bayesian NN: Distribution over weights

**Predictive distribution:**
\`\`\`
p(y*|x*, D) = ∫ p(y*|x*, w)·p(w|D) dw
\`\`\`

This integral over all weight configurations:
- Provides uncertainty estimates
- Averages predictions over plausible models
- **Requires integration (intractable!)**

**Approximation (Monte Carlo dropout):**
\`\`\`
p(y*|x*, D) ≈ (1/T) Σₜ p(y*|x*, wₜ), wₜ ~ dropout
\`\`\`

Sample T forward passes with dropout, average predictions.

**7. Why This Matters:**

Integration is the **computational bottleneck** in Bayesian ML:
- Exact inference: only for simple models (Gaussians, conjugate priors)
- Approximate inference: necessary for deep learning

Modern ML balances:
- **Expressiveness**: Complex models (deep nets)
- **Tractability**: Approximate inference (variational, sampling)

Without efficient integration methods, Bayesian deep learning would be impossible.

**Key Insight:**
Integration in probabilistic models isn't just a mathematical detail—it's the central computational challenge. Advances in approximate inference (VAEs, normalizing flows, score-based models) are fundamentally about finding better ways to handle these integrals.`,
          keyPoints: [
            'Integration normalizes probability distributions',
            'Expectations computed via ∫ g(x)·p(x) dx',
            'Marginalization removes variables: p(x) = ∫ p(x,y) dy',
            'Bayesian evidence p(D) = ∫ p(D|θ)·p(θ) dθ is often intractable',
            'Approximate inference (MC, VI, MCMC) handles intractable integrals',
            'Integration is the computational bottleneck in Bayesian ML',
          ],
        },
        {
          id: 'integration-disc-2',
          question:
            "Compare and contrast different numerical integration methods (Riemann sums, trapezoidal rule, Simpson's rule, Monte Carlo). When is each most appropriate?",
          hint: 'Consider accuracy, computational cost, dimensionality, and function smoothness.',
          sampleAnswer: `**Comparison of Numerical Integration Methods:**

Numerical integration approximates ∫ₐᵇ f(x) dx when analytical solutions don't exist.

**1. Riemann Sums**

**Method:**
Divide [a,b] into n intervals, approximate as rectangles:

\`\`\`
∫ₐᵇ f(x) dx ≈ Σᵢ f(xᵢ*) · Δx
\`\`\`

Variants:
- Left endpoint: xᵢ* = xᵢ
- Right endpoint: xᵢ* = xᵢ₊₁
- Midpoint: xᵢ* = (xᵢ + xᵢ₊₁)/2

**Error:** O(1/n) for smooth functions

**Pros:**
- Simple to understand and implement
- Midpoint rule reasonably accurate

**Cons:**
- Slow convergence (need many points)
- Less accurate than higher-order methods

**When to use:**
- Educational purposes
- Quick rough estimates
- Non-smooth functions (midpoint rule)

**Example:**
\`\`\`python
# ∫₀¹ x² dx = 1/3
n = 100
dx = 1.0 / n
midpoint_sum = sum(((i + 0.5) * dx)**2 for i in range(n)) * dx
# midpoint_sum ≈ 0.3333
\`\`\`

**2. Trapezoidal Rule**

**Method:**
Approximate function with straight lines (trapezoids):

\`\`\`
∫ₐᵇ f(x) dx ≈ (Δx/2) · [f(x₀) + 2f(x₁) + ... + 2f(xₙ₋₁) + f(xₙ)]
\`\`\`

**Error:** O(1/n²) for smooth functions

**Pros:**
- Better accuracy than Riemann (quadratic convergence)
- Simple implementation
- Good for smooth functions

**Cons:**
- Less accurate than Simpson's
- Requires function evaluation at endpoints

**When to use:**
- Smooth functions
- When Simpson's requirements not met (odd number of points)
- Moderate accuracy needed

**Example:**
\`\`\`python
from scipy.integrate import trapezoid
x = np.linspace(0, 1, 101)
y = x**2
result = trapezoid(y, x)  # result ≈ 0.333333
\`\`\`

**3. Simpson's Rule**

**Method:**
Approximate function with parabolas (quadratic interpolation):

\`\`\`
∫ₐᵇ f(x) dx ≈ (Δx/3) · [f(x₀) + 4f(x₁) + 2f(x₂) + 4f(x₃) + ... + f(xₙ)]
\`\`\`

**Error:** O(1/n⁴) for smooth functions

**Pros:**
- **Very accurate** for smooth functions (quartic convergence)
- Exact for polynomials up to degree 3
- Best deterministic low-D method

**Cons:**
- Requires even number of intervals
- Slightly more complex than trapezoidal
- Still suffers from curse of dimensionality

**When to use:**
- Smooth, well-behaved 1D or low-D integrals
- High accuracy required
- Computational budget allows many function evaluations

**Example:**
\`\`\`python
from scipy.integrate import simpson
x = np.linspace(0, 1, 101)
y = x**2
result = simpson(y, x=x)  # result ≈ 0.33333333 (very accurate!)
\`\`\`

**4. Monte Carlo Integration**

**Method:**
Sample random points, average function values:

\`\`\`
∫ₐᵇ f(x) dx ≈ (b-a) · (1/N) Σᵢ f(xᵢ), xᵢ ~ Uniform[a,b]
\`\`\`

**Error:** O(1/√N) - **independent of dimension!**

**Pros:**
- **Scales to high dimensions** (curse of dimensionality doesn't apply as strongly)
- Error independent of dimension
- Easy to implement
- Handles non-smooth functions
- Can importance sample (reduce variance)

**Cons:**
- Slow convergence (need 4× samples for 2× accuracy)
- Less accurate than Simpson's in 1D
- Requires random number generation
- Stochastic (different runs give different results)

**When to use:**
- **High-dimensional integrals** (d > 3)
- Irregular domains
- Non-smooth functions
- When many function evaluations are cheap
- Probabilistic models (expectations)

**Example:**
\`\`\`python
# ∫₀¹ x² dx using Monte Carlo
n_samples = 10000
x_samples = np.random.uniform(0, 1, n_samples)
mc_estimate = np.mean(x_samples**2)  # ≈ 0.333 ± 0.01
\`\`\`

**5. Comparison Table:**

| Method | Error | Best for | Dimension | Smoothness |
|--------|-------|----------|-----------|------------|
| **Riemann** | O(1/n) | Education | 1D | Any |
| **Trapezoidal** | O(1/n²) | Smooth 1D | 1D | Smooth |
| **Simpson's** | O(1/n⁴) | Very smooth 1D | 1D-2D | Very smooth |
| **Monte Carlo** | O(1/√N) | High-D | **Any D** | Any |

**6. Curse of Dimensionality:**

For d-dimensional integral using grid methods:
- Need n points per dimension
- Total points: n^d
- Exponential growth!

**Example:**
- 1D: 100 points → error O(1/100²) = 0.0001
- 10D: 100^10 = 10²⁰ points needed!

**Monte Carlo in high-D:**
- Error O(1/√N) **regardless of d**
- 10,000 samples → error ~0.01 in any dimension

**Why MC wins in high-D:**
\`\`\`
Simpson's in d dimensions: error = O(n^(-4/d))
Monte Carlo: error = O(1/√N)

For d=10, n=100:
- Simpson's: error ~ O(100^(-0.4)) ~ 0.1
- MC with 10,000 samples: error ~ 0.01
\`\`\`

**7. Advanced Methods:**

**Quasi-Monte Carlo:**
- Use low-discrepancy sequences (Sobol, Halton)
- Better than random sampling
- Error: O((log N)^d / N) better than O(1/√N)

**Importance Sampling:**
Sample from distribution q(x) that concentrates on important regions:
\`\`\`
∫ f(x) dx ≈ (1/N) Σᵢ f(xᵢ)/q(xᵢ), xᵢ ~ q
\`\`\`

**Adaptive Quadrature:**
- Refine grid in regions where function varies rapidly
- Used in scipy.integrate.quad

**8. Practical Decision Tree:**

\`\`\`
Is dimension d ≤ 3?
├─ Yes: Is function smooth?
│  ├─ Yes: Use Simpson's rule (best accuracy)
│  └─ No: Use Monte Carlo or midpoint Riemann
└─ No (high-D):
   ├─ Can afford many samples? → Monte Carlo
   ├─ Need variance reduction? → Importance sampling / Quasi-MC
   └─ Very high-D (d>20)? → MCMC or variational methods
\`\`\`

**9. Machine Learning Applications:**

**1D-2D smooth integrals:**
- Use Simpson's/adaptive quadrature
- Example: Computing loss over validation set

**Expectations over distributions:**
- Monte Carlo sampling
- Example: E_x~p[f(x)] ≈ (1/N)Σf(xᵢ), xᵢ~p

**High-dimensional integrals (Bayesian inference):**
- MCMC (Metropolis, HMC)
- Variational inference (convert to optimization)
- Example: p(D) = ∫p(D|θ)p(θ)dθ for 1M parameters

**Summary:**

**Low-dimensional + smooth:** Simpson's rule (O(1/n⁴) accuracy)
**High-dimensional:** Monte Carlo (dimension-independent O(1/√N))
**Very high-D:** MCMC/VI (specialized methods)

The key insight: **dimension determines method choice**. In ML, high dimensionality makes Monte Carlo and its variants (MCMC, VI) essential tools.`,
          keyPoints: [
            "Simpson's rule: Best for 1D smooth functions, O(1/n⁴) error",
            'Monte Carlo: Best for high-D, O(1/√N) error independent of dimension',
            'Curse of dimensionality: grid methods need n^d points',
            'MC avoids curse: error independent of dimension',
            "Rule: d≤3 use Simpson's, d>3 use Monte Carlo",
            'ML applications mostly high-D → MC/MCMC/VI dominate',
          ],
        },
        {
          id: 'integration-disc-3',
          question:
            'Explain the role of integration in deriving the cross-entropy loss and KL divergence. Why are these integrals often approximated in practice?',
          hint: 'Consider continuous vs discrete distributions, expectations, and computational tractability.',
          sampleAnswer: `**Integration in Cross-Entropy and KL Divergence:**

Both cross-entropy and KL divergence fundamentally involve integration (or summation for discrete distributions). Understanding this connection is crucial for ML theory and practice.

**1. Cross-Entropy: Definition**

For continuous distributions P and Q:

H(P, Q) = -∫ p(x) log q(x) dx

For discrete distributions:

H(P, Q) = -Σᵢ p(xᵢ) log q(xᵢ)

**Interpretation:**
- Measures expected log-likelihood under Q when true distribution is P
- Minimizing cross-entropy = maximizing likelihood

**2. KL Divergence: Definition**

**Continuous:**
\`\`\`
D_KL(P||Q) = ∫ p(x) log(p(x)/q(x)) dx
           = ∫ p(x) log p(x) dx - ∫ p(x) log q(x) dx
           = -H(P) + H(P, Q)
\`\`\`

**Discrete:**
\`\`\`
D_KL(P||Q) = Σᵢ p(xᵢ) log(p(xᵢ)/q(xᵢ))
\`\`\`

**Interpretation:**
- Measures "distance" from Q to P (not symmetric!)
- Expected log-ratio of probabilities
- Information gained when using true P instead of approximate Q

**3. Why These Are Integrals:**

Both are **expectations** over distribution P:

**Cross-entropy:**
\`\`\`
H(P, Q) = -E_P[log q(X)]
        = -∫ p(x) log q(x) dx
\`\`\`

**KL divergence:**
\`\`\`
D_KL(P||Q) = E_P[log(p(X)/q(X))]
           = ∫ p(x) log(p(x)/q(x)) dx
\`\`\`

**4. Example: Gaussian Distributions**

**Analytical KL (rare case with closed form!):**

For P = N(μ₁, σ₁²) and Q = N(μ₂, σ₂²):

\`\`\`
D_KL(P||Q) = log(σ₂/σ₁) + (σ₁² + (μ₁-μ₂)²)/(2σ₂²) - 1/2
\`\`\`

This comes from **evaluating the integral analytically**.

**Numerical verification:**
\`\`\`python
from scipy.integrate import quad

mu1, sigma1 = 0.0, 1.0
mu2, sigma2 = 1.0, 1.5

def p(x):
    return (1/(sigma1*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu1)/sigma1)**2)

def q(x):
    return (1/(sigma2*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu2)/sigma2)**2)

def kl_integrand(x):
    p_x, q_x = p(x), q(x)
    return p_x * np.log(p_x / q_x) if p_x > 1e-10 else 0

# Numerical integration
kl_numerical, _ = quad(kl_integrand, -10, 10)

# Analytical formula
kl_analytical = np.log(sigma2/sigma1) + (sigma1**2 + (mu1-mu2)**2)/(2*sigma2**2) - 0.5

print(f"KL (numerical): {kl_numerical:.6f}")
print(f"KL (analytical): {kl_analytical:.6f}")
\`\`\`

**5. Why Integration is Challenging:**

**Problem 1: High Dimensionality**

For images (d=784 for MNIST):
\`\`\`
D_KL(P||Q) = ∫...∫ p(x) log(p(x)/q(x)) dx₁...dx₇₈₄
\`\`\`

This 784-dimensional integral is intractable!

**Problem 2: Unknown Distributions**

In supervised learning:
- True data distribution P(x, y) is unknown
- Only have samples: {(x₁,y₁), ..., (xₙ,yₙ)}

Can't compute ∫ p(x) ... dx because we don't have p(x)!

**Problem 3: Intractable Model Distributions**

For complex models (deep networks), even computing q(x) requires intractable integrals:

\`\`\`
q(x) = ∫ q(x|z) q(z) dz  (marginalizing latent variables)
\`\`\`

**6. Practical Approximations:**

**Approximation 1: Monte Carlo Estimation**

Replace integral with sample average:

\`\`\`
D_KL(P||Q) = E_P[log(p(X)/q(X))]
           ≈ (1/N) Σᵢ log(p(xᵢ)/q(xᵢ)), xᵢ ~ P
\`\`\`

**Example:**
\`\`\`python
# Sample from P
samples = np.random.normal(mu1, sigma1, 10000)

# Monte Carlo estimate
kl_mc = np.mean([np.log(p(x) / q(x)) for x in samples])
print(f"KL (Monte Carlo): {kl_mc:.6f}")
\`\`\`

**Approximation 2: Empirical Distribution**

Use empirical samples instead of true P:

\`\`\`
H_empirical(P, Q) = -(1/N) Σᵢ log q(xᵢ)
\`\`\`

**This is the standard cross-entropy loss in ML!**

\`\`\`python
# Classification: minimize cross-entropy
def cross_entropy_loss(y_true, y_pred):
    # y_true: one-hot encoded true labels (empirical P)
    # y_pred: model predictions (Q)
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-10), axis=1))
\`\`\`

**Approximation 3: Evidence Lower Bound (ELBO)**

For latent variable models, KL involves intractable integrals:

\`\`\`
log p(x) = log ∫ p(x|z) p(z) dz  (intractable!)
\`\`\`

**Solution:** Variational inference

Instead of computing exact KL, maximize ELBO:

\`\`\`
log p(x) ≥ E_q[log p(x|z)] - D_KL(q(z)||p(z))  (ELBO)
\`\`\`

where q(z) is a tractable approximation.

**Example: Variational Autoencoder (VAE)**
\`\`\`python
def vae_loss(x, x_recon, mu, logvar):
    # Reconstruction term (Monte Carlo estimate)
    recon_loss = binary_crossentropy(x, x_recon)
    
    # KL term (analytical for Gaussian)
    kl_loss = -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar))
    
    return recon_loss + kl_loss
\`\`\`

**7. Why Empirical Approximation Works:**

**Supervised Learning:**

Minimize:
\`\`\`
D_KL(P_data || Q_model) = E_P_data[log(p(y|x)) - log(q(y|x))]
\`\`\`

First term (entropy of true distribution) is constant, so equivalent to:

\`\`\`
minimize E_P_data[-log q(y|x)]
\`\`\`

Use empirical samples:

\`\`\`
≈ -(1/N) Σᵢ log q(yᵢ|xᵢ)  (cross-entropy loss!)
\`\`\`

**Example:**
\`\`\`python
# Binary classification
y_true = np.array([0, 1, 1, 0])  # empirical samples
y_pred = np.array([0.1, 0.9, 0.8, 0.2])  # model predictions

# Cross-entropy (empirical approximation of integral)
loss = -np.mean(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))
\`\`\`

**8. Discrete vs Continuous:**

**Discrete (classification):**
- Integrals become sums
- Exact computation possible
- Cross-entropy: -Σᵢ pᵢ log qᵢ

**Continuous (regression, generative models):**
- True integrals required
- Usually intractable
- Must approximate (MC, variational)

**9. Advanced: Tractable Approximations**

**Normalizing Flows:**
Design q(x) such that:
1. Sampling is easy
2. Density evaluation is tractable
3. Can compute log q(x) exactly

Transform simple distribution (Gaussian) through bijective functions.

**Score-Based Models:**
Instead of modeling p(x) directly, model ∇_x log p(x) (the score).
Avoids computing normalizing constant (which requires integration).

**10. Summary Table:**

| Setting | Method | Approximation |
|---------|--------|---------------|
| **Gaussian** | Analytical | Closed-form formula |
| **Discrete (classification)** | Exact | Summation (no integral) |
| **Empirical samples** | Monte Carlo | (1/N)Σ log q(xᵢ) |
| **Latent variables (VAE)** | Variational (ELBO) | Lower bound |
| **High-D continuous** | Normalizing flows | Tractable density |
| **Score matching** | Avoid density | Model score instead |

**Key Insights:**

1. **Cross-entropy loss = empirical approximation of integral**
   - We replace E_P with sample average
   - Works because of law of large numbers

2. **KL divergence requires knowing both P and Q**
   - In practice, only have samples from P
   - Must approximate the expectation

3. **High dimensionality = intractability**
   - Integrals over 100+ dimensions infeasible
   - Clever approximations (VI, MC, normalizing flows) essential

4. **Different problems, different solutions:**
   - Classification: discrete (exact sums)
   - Regression: continuous (MC approximation)
   - Generative models: latent variables (variational methods)

**Practical Takeaway:**

When you see cross-entropy loss in code:
\`\`\`python
loss = -torch.mean(y_true * torch.log(y_pred))
\`\`\`

Remember: this is an **empirical approximation** of the integral:
\`\`\`
H(P, Q) = -∫ p(x) log q(x) dx
\`\`\`

using samples from the training set. The entire foundation of supervised learning rests on this approximation being valid (which it is, by the law of large numbers)!`,
          keyPoints: [
            'Cross-entropy H(P,Q) = -∫ p(x)log q(x)dx is an expectation',
            'KL divergence involves integration over continuous distributions',
            'High-dimensional integrals (images, text) are intractable',
            'Empirical approximation: replace ∫ p(x)... with (1/N)Σ...',
            'Classification: discrete (sums), Generative: continuous (integrals)',
            'VAEs use ELBO to avoid intractable KL computation',
            'Standard cross-entropy loss = empirical approximation of integral',
          ],
        },
      ],
    },
    {
      id: 'multivariable-calculus',
      title: 'Multivariable Calculus',
      content: `
# Multivariable Calculus

## Introduction

Most machine learning problems involve functions of many variables: loss functions depend on thousands/millions of parameters. Multivariable calculus provides the mathematical framework for optimizing these high-dimensional functions.

## Jacobian Matrix

For vector-valued function **F**: ℝⁿ → ℝᵐ, the Jacobian matrix contains all first-order partial derivatives:

J = [∂Fᵢ/∂xⱼ]

**Dimensions**: m × n

\`\`\`python
import numpy as np
from scipy.optimize import approx_fprime

def compute_jacobian_example():
    """
    Example: F: ℝ² → ℝ³
    F(x, y) = [x² + y, x·y, sin(x) + cos(y)]
    """
    
    def F(xy):
        x, y = xy
        return np.array([
            x**2 + y,
            x * y,
            np.sin(x) + np.cos(y)
        ])
    
    def jacobian_analytical(xy):
        x, y = xy
        return np.array([
            [2*x, 1],              # ∂F₁/∂x, ∂F₁/∂y
            [y, x],                # ∂F₂/∂x, ∂F₂/∂y
            [np.cos(x), -np.sin(y)]  # ∂F₃/∂x, ∂F₃/∂y
        ])
    
    # Test point
    xy_test = np.array([1.0, 2.0])
    
    # Analytical Jacobian
    J_analytical = jacobian_analytical(xy_test)
    
    # Numerical Jacobian
    J_numerical = np.array([
        approx_fprime(xy_test, lambda xy: F(xy)[i], 1e-8)
        for i in range(3)
    ])
    
    print("Jacobian Matrix:")
    print(f"Analytical:\\n{J_analytical}")
    print(f"\\nNumerical:\\n{J_numerical}")
    print(f"\\nError: {np.linalg.norm(J_analytical - J_numerical):.2e}")

compute_jacobian_example()
\`\`\`

## Hessian Matrix

For scalar function f: ℝⁿ → ℝ, the Hessian matrix contains all second-order partial derivatives:

H = [∂²f/∂xᵢ∂xⱼ]

**Dimensions**: n × n (symmetric for twice-differentiable f)

\`\`\`python
def compute_hessian_example():
    """
    Example: f(x, y) = x² + xy + y²
    """
    
    def f(xy):
        x, y = xy
        return x**2 + x*y + y**2
    
    def gradient(xy):
        x, y = xy
        return np.array([
            2*x + y,    # ∂f/∂x
            x + 2*y     # ∂f/∂y
        ])
    
    def hessian_analytical(xy):
        return np.array([
            [2, 1],  # ∂²f/∂x², ∂²f/∂x∂y
            [1, 2]   # ∂²f/∂y∂x, ∂²f/∂y²
        ])
    
    # Test point
    xy_test = np.array([1.0, 2.0])
    
    # Analytical Hessian
    H_analytical = hessian_analytical(xy_test)
    
    # Numerical Hessian (using gradient)
    h = 1e-7
    H_numerical = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            e_j = np.zeros(2)
            e_j[j] = h
            H_numerical[i, j] = (gradient(xy_test + e_j)[i] - gradient(xy_test)[i]) / h
    
    print("Hessian Matrix:")
    print(f"Analytical:\\n{H_analytical}")
    print(f"\\nNumerical:\\n{H_numerical}")
    print(f"\\nError: {np.linalg.norm(H_analytical - H_numerical):.2e}")
    
    # Eigenvalues determine convexity
    eigenvalues = np.linalg.eigvalsh(H_analytical)
    print(f"\\nEigenvalues: {eigenvalues}")
    if np.all(eigenvalues > 0):
        print("Hessian is positive definite → f is convex")

compute_hessian_example()
\`\`\`

## Taylor Series (Multivariate)

For f: ℝⁿ → ℝ, Taylor expansion around point **a**:

f(**x**) ≈ f(**a**) + ∇f(**a**)·(**x** - **a**) + (1/2)(**x** - **a**)ᵀH(**a**)(**x** - **a**)

**Applications**:
- Approximating loss surfaces
- Newton's method
- Understanding optimization landscapes

\`\`\`python
def multivariate_taylor_series():
    """
    Demonstrate multivariate Taylor approximation
    """
    
    # Function: f(x, y) = x² + xy + 2y²
    def f(xy):
        x, y = xy
        return x**2 + x*y + 2*y**2
    
    def gradient(xy):
        x, y = xy
        return np.array([2*x + y, x + 4*y])
    
    def hessian(xy):
        return np.array([[2, 1], [1, 4]])
    
    # Expansion point
    a = np.array([1.0, 1.0])
    
    # Test points
    test_points = [
        np.array([1.1, 1.05]),
        np.array([1.5, 1.2]),
        np.array([2.0, 1.5])
    ]
    
    print("Taylor Series Approximation:")
    print("="*70)
    
    for x in test_points:
        dx = x - a
        
        # True value
        f_true = f(x)
        
        # Taylor approximations
        f_0th = f(a)  # 0th order
        f_1st = f(a) + np.dot(gradient(a), dx)  # 1st order
        f_2nd = f(a) + np.dot(gradient(a), dx) + 0.5 * dx @ hessian(a) @ dx  # 2nd order
        
        print(f"\\nPoint: {x}, Distance from a: {np.linalg.norm(dx):.4f}")
        print(f"True value:        {f_true:.6f}")
        print(f"0th order approx:  {f_0th:.6f} (error: {abs(f_true - f_0th):.6f})")
        print(f"1st order approx:  {f_1st:.6f} (error: {abs(f_true - f_1st):.6f})")
        print(f"2nd order approx:  {f_2nd:.6f} (error: {abs(f_true - f_2nd):.6f})")
    
    print("\\n→ Second-order approximation is exact for quadratic functions!")

multivariate_taylor_series()
\`\`\`

## Critical Points & Saddle Points

For f: ℝⁿ → ℝ, a critical point **x*** satisfies ∇f(**x***) = **0**.

**Classification using Hessian eigenvalues**:
- All eigenvalues > 0: **local minimum**
- All eigenvalues < 0: **local maximum**
- Mixed signs: **saddle point**
- Zero eigenvalues: **inconclusive**

\`\`\`python
def classify_critical_points():
    """
    Find and classify critical points
    """
    
    # Function: f(x, y) = x² - y²
    def f(xy):
        x, y = xy
        return x**2 - y**2
    
    def gradient(xy):
        x, y = xy
        return np.array([2*x, -2*y])
    
    def hessian(xy):
        return np.array([[2, 0], [0, -2]])
    
    # Critical point (gradient = 0)
    critical_point = np.array([0.0, 0.0])
    
    print("Critical Point Analysis:")
    print(f"Point: {critical_point}")
    print(f"Gradient: {gradient(critical_point)}")
    
    H = hessian(critical_point)
    eigenvalues = np.linalg.eigvalsh(H)
    
    print(f"\\nHessian:\\n{H}")
    print(f"Eigenvalues: {eigenvalues}")
    
    if np.all(eigenvalues > 0):
        print("→ Local minimum")
    elif np.all(eigenvalues < 0):
        print("→ Local maximum")
    else:
        print("→ Saddle point (mixed eigenvalue signs)")
    
    # Visualize
    import matplotlib.pyplot as plt
    
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 - Y**2
    
    fig = plt.figure(figsize=(10, 4))
    
    # 3D surface
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax1.scatter([0], [0], [0], color='red', s=100, label='Saddle point')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x,y)')
    ax1.set_title('f(x,y) = x² - y²')
    ax1.legend()
    
    # Contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(X, Y, Z, levels=20, cmap='viridis')
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.plot(0, 0, 'ro', markersize=10, label='Saddle point')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Contour Plot')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('saddle_point.png', dpi=150, bbox_inches='tight')
    print("\\nSaved visualization to 'saddle_point.png'")

classify_critical_points()
\`\`\`

## Optimization in High Dimensions

In deep learning, we optimize f: ℝⁿ → ℝ where n ~ 10⁶ - 10⁹.

**Challenges**:
1. **Curse of dimensionality**: Exponentially many directions
2. **Saddle points**: Common in high dimensions
3. **Computational cost**: Can't compute full Hessian (n²  elements)

\`\`\`python
def high_dimensional_saddle_points():
    """
    Demonstrate saddle points in high dimensions
    """
    
    # Function: f(x) = Σᵢ αᵢ·xᵢ² where some αᵢ > 0, some < 0
    def create_saddle_function(n_pos, n_neg):
        """Create function with saddle point at origin"""
        alphas = np.concatenate([np.ones(n_pos), -np.ones(n_neg)])
        
        def f(x):
            return 0.5 * np.sum(alphas * x**2)
        
        def gradient(x):
            return alphas * x
        
        def hessian():
            return np.diag(alphas)
        
        return f, gradient, hessian, alphas
    
    # Example: 5D space, 3 positive, 2 negative curvatures
    f, grad, hess, alphas = create_saddle_function(n_pos=3, n_neg=2)
    
    origin = np.zeros(5)
    print("High-Dimensional Saddle Point:")
    print(f"Dimensions: {len(origin)}")
    print(f"Gradient at origin: {grad(origin)}")
    
    H = hess()
    eigenvalues = np.linalg.eigvalsh(H)
    print(f"\\nHessian eigenvalues: {eigenvalues}")
    print(f"Positive eigenvalues: {np.sum(eigenvalues > 0)}")
    print(f"Negative eigenvalues: {np.sum(eigenvalues < 0)}")
    print("→ Saddle point (3 directions decrease, 2 increase)")
    
    # Fraction of saddle points grows exponentially with dimension
    print("\\n** Why saddle points dominate in high dimensions:**")
    print("For n dimensions with random Hessian:")
    print(f"  n=2: P(saddle) ≈ 50%")
    print(f"  n=10: P(saddle) ≈ 99.8%")
    print(f"  n=1000: P(saddle) ≈ 100%")
    print("\\nMost critical points in high-D are saddle points, not minima!")

high_dimensional_saddle_points()
\`\`\`

## Applications to Neural Networks

\`\`\`python
def neural_network_hessian_example():
    """
    Compute Hessian for simple neural network
    """
    
    # Data
    X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    y = np.array([[1.0], [0.0], [1.0]])
    
    # Simple network: y = σ(Wx + b)
    np.random.seed(42)
    W = np.random.randn(1, 2) * 0.1
    b = np.random.randn(1) * 0.1
    
    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def sigmoid_derivative(z):
        s = sigmoid(z)
        return s * (1 - s)
    
    def forward(W, b):
        z = X @ W.T + b
        return sigmoid(z)
    
    def loss(W, b):
        y_pred = forward(W, b)
        return 0.5 * np.mean((y_pred - y)**2)
    
    # Flatten parameters
    def params_to_vector(W, b):
        return np.concatenate([W.flatten(), b.flatten()])
    
    def vector_to_params(theta):
        W = theta[:2].reshape(1, 2)
        b = theta[2:].reshape(1)
        return W, b
    
    def loss_vector(theta):
        W, b = vector_to_params(theta)
        return loss(W, b)
    
    # Compute Hessian numerically
    theta = params_to_vector(W, b)
    n_params = len(theta)
    
    H = np.zeros((n_params, n_params))
    h = 1e-5
    
    for i in range(n_params):
        for j in range(n_params):
            e_i = np.zeros(n_params)
            e_j = np.zeros(n_params)
            e_i[i] = h
            e_j[j] = h
            
            H[i, j] = (
                loss_vector(theta + e_i + e_j)
                - loss_vector(theta + e_i)
                - loss_vector(theta + e_j)
                + loss_vector(theta)
            ) / (h**2)
    
    print("Neural Network Hessian:")
    print(f"Parameters: {n_params}")
    print(f"Hessian:\\n{H}")
    
    eigenvalues = np.linalg.eigvalsh(H)
    print(f"\\nEigenvalues: {eigenvalues}")
    
    if np.all(eigenvalues > 0):
        print("→ Loss surface is locally convex")
    else:
        print("→ Loss surface has negative curvature directions")
    
    # Condition number
    cond = np.max(np.abs(eigenvalues)) / np.min(np.abs(eigenvalues[eigenvalues > 1e-10]))
    print(f"\\nCondition number: {cond:.2f}")
    print("(Higher → harder to optimize)")

neural_network_hessian_example()
\`\`\`

## Summary

**Key Concepts**:
- **Jacobian**: All first derivatives of vector function
- **Hessian**: All second derivatives, determines local curvature
- **Taylor series**: Quadratic approximation of loss surface
- **Critical points**: ∇f = 0, classified by Hessian eigenvalues
- **Saddle points**: Dominant in high dimensions

**Why This Matters**:
- Hessian eigenvalues determine optimization difficulty
- Second-order information used in Newton's method, trust region methods
- Understanding saddle points explains why SGD works in deep learning
- Critical for analyzing convergence of optimization algorithms
`,
      multipleChoice: [
        {
          id: 'multivar-1',
          question: 'The Hessian matrix of a function f: ℝⁿ → ℝ contains:',
          options: [
            'First-order partial derivatives',
            'Second-order partial derivatives',
            'Only diagonal elements',
            'The gradient vector',
          ],
          correctAnswer: 1,
          explanation:
            'The Hessian matrix H = [∂²f/∂xᵢ∂xⱼ] contains all second-order partial derivatives.',
          difficulty: 'easy',
        },
        {
          id: 'multivar-2',
          question:
            'A critical point with positive and negative Hessian eigenvalues is:',
          options: [
            'A local minimum',
            'A local maximum',
            'A saddle point',
            'A global minimum',
          ],
          correctAnswer: 2,
          explanation:
            'Mixed eigenvalue signs indicate a saddle point: decreasing in some directions, increasing in others.',
          difficulty: 'medium',
        },
        {
          id: 'multivar-3',
          question:
            'Why are saddle points common in high-dimensional optimization?',
          options: [
            'They are rare in high dimensions',
            'The probability of having mixed curvature directions grows exponentially with dimension',
            'High dimensions have fewer critical points',
            'Gradients are always zero in high dimensions',
          ],
          correctAnswer: 1,
          explanation:
            "In n dimensions, the probability of a random critical point being a saddle point approaches 100% as n increases, because it's unlikely all n eigenvalues have the same sign.",
          difficulty: 'hard',
        },
        {
          id: 'multivar-4',
          question: 'The Jacobian matrix of F: ℝⁿ → ℝᵐ has dimensions:',
          options: ['n × n', 'm × m', 'm × n', 'n × m'],
          correctAnswer: 2,
          explanation:
            'The Jacobian has m rows (one per output) and n columns (one per input), resulting in m × n.',
          difficulty: 'medium',
        },
        {
          id: 'multivar-5',
          question:
            'The second-order Taylor approximation of f(**x**) around **a** includes:',
          options: [
            'Only f(**a**)',
            'f(**a**) + ∇f(**a**)·(**x** - **a**)',
            'f(**a**) + ∇f(**a**)·(**x** - **a**) + (1/2)(**x** - **a**)ᵀH(**a**)(**x** - **a**)',
            'Only the Hessian term',
          ],
          correctAnswer: 2,
          explanation:
            'Second-order Taylor expansion includes constant, linear (gradient), and quadratic (Hessian) terms.',
          difficulty: 'medium',
        },
      ],
      quiz: [
        {
          id: 'multivar-disc-1',
          question:
            'Explain why computing the full Hessian is intractable for deep neural networks with millions of parameters. What approximations are used in practice?',
          hint: 'Consider the size of the Hessian matrix and computational cost of computing/storing it.',
          sampleAnswer: `**Intractability of Hessian Computation in Deep Learning:**

**1. Size of the Hessian:**

For a function f: ℝⁿ → ℝ (e.g., loss function):
- **Gradient**: n elements
- **Hessian**: n × n elements

**Modern neural networks:**
- GPT-3: 175 billion parameters
- Hessian size: 175B × 175B ≈ 3 × 10²² elements
- Storage: Even at 4 bytes/float ≈ 10¹¹ TB (impossible!)

**Even smaller networks:**
- 1 million parameters
- Hessian: 10⁶ × 10⁶ = 10¹² elements
- Storage: ~4 TB (challenging)

**2. Computational Cost:**

Computing each Hessian element H_{ij} = ∂²L/∂θᵢ∂θⱼ:

**Naive approach:**
- Finite differences: Compute gradient at θ + εeᵢ for each direction
- Cost: O(n) gradient computations
- Total: O(n²) gradient computations
- For n = 10⁶: ~10¹² gradient evaluations (centuries of compute!)

**Exact computation:**
- Hessian-vector products: Can be computed efficiently via automatic differentiation
- Cost: O(n) per product (same as gradient!)
- But still need n² elements → n products → O(n²) total

**3. Why This is Prohibitive:**

\`\`\`
Network with n = 1,000,000 parameters:

Gradient computation: ~1 second
Full Hessian: 1,000,000 gradients = 10⁶ seconds ≈ 11.5 days

For n = 100,000,000 (GPT-2):
Full Hessian: ~10¹⁴ years!
\`\`\`

**4. Practical Approximations:**

**A) Diagonal Approximation (AdaGrad, RMSProp, Adam):**

Instead of full Hessian H, use only diagonal:
\`\`\`
H_diag = [∂²L/∂θ₁², ..., ∂²L/∂θₙ²]
\`\`\`

**Cost:** O(n) - same as gradient!

**Approximation:**
Assume parameters are independent (off-diagonals ≈ 0).

**Example: Adam optimizer**
\`\`\`python
# Approximate second moment (related to diagonal Hessian)
m_t = β1 * m_{t-1} + (1-β1) * gradient
v_t = β2 * v_{t-1} + (1-β2) * gradient**2  # diagonal approx

# Update uses v_t^{-1/2} (inverse square root ~ inverse Hessian)
θ_t = θ_{t-1} - α * m_t / (√v_t + ε)
\`\`\`

**B) Block-Diagonal Approximation:**

Partition parameters into blocks, compute Hessian per block:
\`\`\`
H ≈ [H₁    0   ...  0  ]
    [ 0   H₂  ...  0  ]
    [ 0    0  ...  Hₖ ]
\`\`\`

**Example:** Separate Hessian for each layer.

**Cost:** Much smaller than full Hessian.

**C) Hessian-Vector Products (Krylov methods):**

Don't compute H explicitly, but can compute Hv for any vector v.

**How:** Via automatic differentiation
\`\`\`python
# Compute Hv without forming H
def hessian_vector_product(loss_fn, params, v):
    grads = compute_gradient(loss_fn, params)
    # Compute gradient of (∇L · v)
    return compute_gradient(grads @ v, params)
\`\`\`

**Cost:** O(n) per product - same as gradient!

**Applications:**
- Conjugate Gradient
- Lanczos algorithm (find top eigenvalues)
- Hessian-free optimization

**D) Low-Rank Approximation (L-BFGS):**

Approximate H⁻¹ with low-rank updates:
\`\`\`
H_k⁻¹ ≈ B_k = (I - ρ_k s_k y_k^T) B_{k-1} (I - ρ_k y_k s_k^T) + ρ_k s_k s_k^T
\`\`\`

Store only m recent (sₖ, yₖ) pairs (typically m = 5-20).

**Memory:** O(mn) instead of O(n²)

**Example:** L-BFGS widely used for medium-scale problems.

**E) Fisher Information Matrix:**

For probabilistic models, approximate Hessian with Fisher:
\`\`\`
F = E[∇log p(y|x,θ) ∇log p(y|x,θ)^T]
\`\`\`

**Properties:**
- Positive semi-definite (easier to work with)
- Related to Hessian at optimum
- Block-diagonal approximation often used (e.g., K-FAC)

**Example: K-FAC (Kronecker-Factored Approximate Curvature)**
\`\`\`
F ≈ A ⊗ S  (Kronecker product)
\`\`\`

**Memory:** O(d₁² + d₂²) instead of O((d₁d₂)²)

**F) Hutchinson's Trace Estimator:**

Estimate trace(H) or diagonal via random sampling:
\`\`\`
tr(H) ≈ E_v[v^T H v] for random v
\`\`\`

**Cost:** Few Hessian-vector products.

**5. Comparison Table:**

| Method | Memory | Computation | Accuracy |
|--------|---------|-------------|----------|
| **Full Hessian** | O(n²) | O(n²) grads | Exact |
| **Diagonal** | O(n) | O(n) | Low |
| **Block-diagonal** | O(k·m²) | O(k·m²) | Medium |
| **Hv products** | O(n) | O(n) per v | Exact Hv |
| **L-BFGS** | O(mn) | O(mn) | Good |
| **K-FAC** | O(d₁²+d₂²) | Moderate | Good |

**6. Why Approximations Work:**

**Observation:** Most optimization algorithms don't need the full Hessian.

**Gradient Descent:**
Only needs gradient → O(n)

**Newton's Method:**
Needs H⁻¹∇f → Can use:
- Conjugate Gradient (only needs Hv products)
- L-BFGS (low-rank inverse)

**Second-order information is helpful, but approximate second-order >> first-order.**

**7. Practical Strategy:**

Modern deep learning primarily uses:
1. **Adam/RMSProp**: Diagonal approximation (most common)
2. **L-BFGS**: For smaller models or fine-tuning
3. **K-FAC**: Research/specialized applications
4. **Gradient Descent + momentum**: Ignore Hessian entirely (still works!)

**Key Insight:**

You don't need exact second-order information. Rough approximations (even just diagonal) provide massive speedup over pure gradient descent.

**The hierarchy:**
- **Pure SGD**: First-order only
- **Adam**: Diagonal second-order approximation
- **L-BFGS**: Low-rank second-order approximation
- **Newton**: Full second-order (intractable for deep learning)

**Conclusion:**

Computing full Hessian is O(n²) in memory and computation - completely infeasible for modern deep networks. Practical optimization relies on clever approximations that capture curvature information at O(n) cost.`,
          keyPoints: [
            'Full Hessian: n² elements, intractable for n > 10⁶',
            'Diagonal approximation (Adam): O(n) cost, ignores correlations',
            'Hessian-vector products: O(n) per product via autodiff',
            'L-BFGS: Low-rank inverse approximation, O(mn) memory',
            'K-FAC: Block-diagonal Fisher approximation',
            'Modern deep learning mostly uses diagonal approximations',
          ],
        },
        {
          id: 'multivar-disc-2',
          question:
            'Discuss the role of saddle points in deep learning optimization. Why do gradient-based methods often escape saddle points efficiently?',
          hint: 'Consider the behavior of gradient descent near saddle points, noise in SGD, and the structure of neural network loss surfaces.',
          sampleAnswer: `**Saddle Points in Deep Learning:**

Saddle points are critical points where the Hessian has both positive and negative eigenvalues - neither minima nor maxima.

**1. Why Saddle Points Dominate in High Dimensions:**

**Probability argument:**

For random critical point in n dimensions:
- Prob(all n eigenvalues positive) = (1/2)ⁿ
- Prob(saddle point) = 1 - (1/2)ⁿ - (1/2)ⁿ ≈ 1 for large n

**Concrete examples:**
\`\`\`
n = 2:    P(saddle) ≈ 50%
n = 10:   P(saddle) ≈ 99.8%
n = 100:  P(saddle) ≈ 100%
n = 10⁶:  P(saddle) ≈ 100.0000...%
\`\`\`

**Implication:**
In deep learning (n ~ 10⁶ - 10⁹), virtually all critical points are saddle points.

**2. Types of Saddle Points:**

**Strict saddle points:**
At least one negative eigenvalue λ_min < 0
- Have escape directions (eigenvector of λ_min)
- Can be escaped efficiently

**Non-strict saddle points:**
Some zero eigenvalues
- More difficult to escape
- Rare in practice

**3. Why Saddle Points Were Initially Concerning:**

**Traditional view:**
Gradient descent gets stuck at saddle points because ∇f = 0.

**Concern:**
If most critical points are saddles, won't optimization fail?

**4. Why This Concern Was Wrong:**

**A) Saddle Points are Unstable:**

At saddle point with negative eigenvalue λ < 0:

Small perturbation along corresponding eigenvector v:
\`\`\`
f(x + εv) ≈ f(x) + (ε²/2)λ·||v||² < f(x)
\`\`\`

Loss *decreases* along this direction!

**B) Gradient Descent Escapes Automatically:**

Near saddle point x*:
- ∇f(x*) = 0
- But ∇f(x* + ε) ≠ 0 for almost any perturbation
- Gradient points toward escape direction

\`\`\`python
# Demonstration: Escaping saddle point

def saddle_escape_demo():
    # f(x,y) = x² - y² (saddle at origin)
    def f(x, y):
        return x**2 - y**2
    
    def grad(x, y):
        return np.array([2*x, -2*y])
    
    # Start near saddle with small perturbation
    pos = np.array([0.0, 0.01])  # Slight perturbation in y
    
    trajectory = [pos.copy()]
    lr = 0.1
    
    for _ in range(20):
        g = grad(pos[0], pos[1])
        pos = pos - lr * g
        trajectory.append(pos.copy())
    
    print("Escaping Saddle Point:")
    print(f"Start: {trajectory[0]}")
    print(f"End:   {trajectory[-1]}")
    print(f"Distance from saddle: {np.linalg.norm(trajectory[-1]):.6f}")
    print("→ Gradient descent automatically escapes!")

saddle_escape_demo()
\`\`\`

**5. Role of Noise in SGD:**

**Stochastic Gradient Descent:**
\`\`\`
θ_{t+1} = θ_t - η·∇L_batch(θ_t)
\`\`\`

**Gradient noise** from mini-batches provides natural perturbations:
- Perturbs away from saddle points
- Acts like random exploration
- Helps escape even for exact saddles

**Analogy:** Ball on a saddle - any tiny push causes it to roll off.

**6. Theoretical Results:**

**Gradient Descent + Noise:**

**Theorem (Lee et al., 2016):**
Gradient descent with random initialization avoids saddle points:
- Converges to local minimum (not saddle) with probability 1
- Saddle points have measure zero (probability 0 of landing exactly on one)

**Perturbed Gradient Descent:**

Add small noise: θ_{t+1} = θ_t - η·∇f(θ_t) + ξ_t

**Result:** Escapes saddle points in polynomial time.

**7. Empirical Evidence:**

**Observation:** Deep learning optimization rarely gets stuck.

**Experiments (Goodfellow et al., Dauphin et al.):**
- Analyzed critical points in neural networks
- Found: most are saddle points, not poor local minima
- Loss plateau → not stuck at saddle, just slow progress

**8. Contrast with Local Minima:**

**Saddle points vs. Poor local minima:**

| | Saddle Point | Local Minimum |
|---|--------------|---------------|
| **∇f** | = 0 | = 0 |
| **Hessian** | Mixed eigenvalues | All positive |
| **Escape** | Yes (negative curvature) | No |
| **Problem?** | **No** (escapable) | **Yes** (if poor quality) |

**Key insight:**
High-dimensional optimization challenges come from saddle points (escapable) NOT poor local minima (which appear rare in practice).

**9. Why Escape is Efficient:**

**Negative curvature descent:**

Along direction v with Hessian eigenvalue λ < 0:
\`\`\`
f(x - εv) ≈ f(x) - (ε²/2)|λ|  (decreases quadratically!)
\`\`\`

**Escape time:**
O(1/|λ_min|) iterations - polynomial, not exponential!

**10. Practical Implications:**

**A) Trust SGD:**
Natural noise helps escape saddles - don't need special mechanisms.

**B) Plateaus ≠ Stuck:**
Slow progress near saddle (small gradient) ≠ stuck forever.

**C) Momentum Helps:**
Accelerates escape from saddle regions.

**D) Learning Rate:**
Too small → slow escape
Too large → overshoot
Adaptive methods (Adam) balance this.

**11. Algorithmic Enhancements:**

**Cubic Regularization:**
\`\`\`
arg min_p [f(x) + ∇f(x)^T p + (1/2)p^T H p + (M/6)||p||³]
\`\`\`
Explicitly escapes negative curvature.

**Trust Region Methods:**
Use second-order information to navigate saddles.

**Nesterov Acceleration:**
Momentum variant that handles saddles provably well.

**12. Modern Understanding:**

**Old View:**
"Getting stuck at saddle points is a major problem in deep learning."

**Current View:**
"Saddle points are prevalent but efficiently escapable. Not the bottleneck."

**Real challenges:**
1. Poor conditioning (flat directions)
2. High variance gradients (noisy estimates)
3. Computational cost (large models/datasets)

**Not:**
Getting stuck at saddles.

**13. Summary:**

**Why saddles aren't a problem:**

1. **Unstable:** Any perturbation causes escape
2. **Negative curvature:** Provides escape direction
3. **SGD noise:** Natural perturbations from mini-batches
4. **Probability 0:** Random init almost never lands exactly on saddle
5. **Polynomial escape:** Efficient to escape (not exponential)

**Practical takeaway:**

Worry about:
- Poor conditioning (use Adam, normalize inputs)
- High variance (tune batch size, learning rate)
- Computational budget (efficient architectures)

Don't worry about:
- Getting permanently stuck at saddles (almost never happens)

This is one of the success stories of modern deep learning theory: understanding that high-dimensional saddle points are not the obstacle they initially appeared to be.`,
          keyPoints: [
            'Saddle points dominate in high dimensions (probability → 100%)',
            'Saddle points are unstable: negative curvature provides escape',
            'SGD noise naturally perturbs away from saddles',
            'Gradient descent + noise escapes saddles in polynomial time',
            'Modern challenge: poor conditioning, not saddles',
            'Practical deep learning rarely gets stuck at saddles',
          ],
        },
        {
          id: 'multivar-disc-3',
          question:
            "Explain how the multivariate Taylor series is used in Newton's method for optimization. Why is Newton's method not commonly used for training deep neural networks?",
          hint: 'Consider the second-order Taylor approximation, the Newton update rule, and computational challenges.',
          sampleAnswer: `**Taylor Series in Newton's Method:**

Newton's method uses the second-order Taylor approximation to find better steps than gradient descent.

**1. Derivation from Taylor Series:**

**Goal:** Minimize f(**x**)

**Taylor expansion around current point x_k:**
\`\`\`
f(x) ≈ f(x_k) + ∇f(x_k)^T(x - x_k) + (1/2)(x - x_k)^T H(x_k)(x - x_k)
\`\`\`

**Quadratic approximation:** Q(x)

**Newton's idea:**
Instead of taking small step against gradient, **minimize Q(x)** exactly.

**Minimization:**
\`\`\`
∇Q(x) = ∇f(x_k) + H(x_k)(x - x_k) = 0
\`\`\`

**Solution:**
\`\`\`
x_{k+1} = x_k - H(x_k)^{-1} ∇f(x_k)
\`\`\`

**This is Newton's method!**

**2. Comparison with Gradient Descent:**

**Gradient Descent:**
\`\`\`
x_{k+1} = x_k - α·∇f(x_k)
\`\`\`

Uses only first-order (gradient) information.
Needs manual learning rate α.

**Newton's Method:**
\`\`\`
x_{k+1} = x_k - H^{-1}·∇f(x_k)
\`\`\`

Uses second-order (Hessian) information.
Automatic step size.

**3. Advantages of Newton's Method:**

**A) Quadratic Convergence:**

Near optimum, Newton converges quadratically:
\`\`\`
||x_{k+1} - x*|| ≤ C·||x_k - x*||²
\`\`\`

**Example:** Error sequence
\`\`\`
10^-1 → 10^-2 → 10^-4 → 10^-8 → 10^-16
\`\`\`

Each iteration roughly doubles the number of correct digits!

**Gradient Descent:** Linear convergence (much slower)
\`\`\`
10^-1 → 10^-2 → 10^-3 → 10^-4 → 10^-5 → ...
\`\`\`

**B) Invariant to Scaling:**

Newton's method is affine invariant:
- Automatically adapts to function curvature
- No need to tune learning rate

**C) Handles Ill-Conditioning:**

For quadratic f(x) = (1/2)x^T A x - b^T x:

**Gradient Descent:**
Convergence rate depends on condition number κ(A):
\`\`\`
Slow if κ(A) >> 1 (ill-conditioned)
\`\`\`

**Newton's Method:**
Converges in **one step** regardless of κ(A)!
\`\`\`
x* = A^{-1}b
\`\`\`

**4. Demonstration:**

\`\`\`python
def compare_gd_newton():
    """Compare Gradient Descent vs Newton's Method"""
    
    # Ill-conditioned quadratic: f(x,y) = 10x² + y²
    def f(xy):
        x, y = xy
        return 10*x**2 + y**2
    
    def grad(xy):
        x, y = xy
        return np.array([20*x, 2*y])
    
    def hessian(xy):
        return np.array([[20, 0], [0, 2]])
    
    # Start point
    x0 = np.array([1.0, 1.0])
    
    # Gradient Descent
    x_gd = x0.copy()
    lr = 0.05  # Carefully tuned
    gd_trajectory = [x_gd.copy()]
    
    for _ in range(50):
        x_gd = x_gd - lr * grad(x_gd)
        gd_trajectory.append(x_gd.copy())
    
    # Newton's Method
    x_newton = x0.copy()
    newton_trajectory = [x_newton.copy()]
    
    for _ in range(5):  # Much fewer iterations!
        H = hessian(x_newton)
        g = grad(x_newton)
        x_newton = x_newton - np.linalg.solve(H, g)
        newton_trajectory.append(x_newton.copy())
    
    print("Gradient Descent vs Newton's Method:")
    print(f"GD after 50 iters: {gd_trajectory[-1]}, f = {f(gd_trajectory[-1]):.6f}")
    print(f"Newton after 5 iters: {newton_trajectory[-1]}, f = {f(newton_trajectory[-1]):.2e}")
    print(f"\\n→ Newton converges in 2 iterations, GD takes 50+!")

compare_gd_newton()
\`\`\`

**5. Why Newton is NOT Used for Deep Learning:**

**Problem 1: Computational Cost**

**Gradient:** O(n) - backpropagation
**Hessian:** O(n²) - intractable for n ~ 10⁶+

For modern neural networks:
- Computing H: days/weeks
- Storing H: terabytes
- Inverting H: impossible

**Problem 2: Memory**

Hessian matrix:
\`\`\`
n = 1,000,000 parameters
H: 10⁶ × 10⁶ = 10¹² elements
Memory: ~4 TB
\`\`\`

Simply cannot fit in memory.

**Problem 3: Per-Iteration Cost**

\`\`\`
Newton step: x_{k+1} = x_k - H^{-1}g

Requires:
1. Compute H: O(n²) time
2. Invert H: O(n³) time (matrix inversion)
3. Multiply H^{-1}g: O(n²) time

Total: O(n³) per iteration!
\`\`\`

For n = 10⁶: ~10¹⁸ operations >> infeasible

Compare to gradient descent:
\`\`\`
GD step: x_{k+1} = x_k - α·g
Cost: O(n) per iteration
\`\`\`

**Problem 4: Non-Convexity**

Newton assumes quadratic approximation is good.

In deep learning:
- Highly non-convex loss surfaces
- Taylor approximation only local
- May step toward saddle or maximum!

**Newton can diverge** if Hessian has negative eigenvalues.

**Problem 5: Mini-Batching**

Deep learning uses stochastic mini-batches:
- Gradient estimate noisy
- Hessian estimate VERY noisy (requires n × more samples)
- Newton update unreliable with noisy Hessian

**6. Practical Alternatives:**

**A) L-BFGS (Limited-memory BFGS):**

Approximate H^{-1} using history:
\`\`\`
H_k^{-1} ≈ B_k  (built from last m gradient differences)
\`\`\`

**Memory:** O(mn) instead of O(n²)
**Cost:** O(mn) instead of O(n³)

**Used for:** Small/medium networks, full-batch optimization

**B) Gauss-Newton / Levenberg-Marquardt:**

For least squares problems:
\`\`\`
H ≈ J^T J  (Gauss-Newton approximation)
\`\`\`

Cheaper than full Hessian.

**C) Natural Gradient / K-FAC:**

Use Fisher information matrix instead of Hessian:
\`\`\`
F = E[∇log p(y|x)∇log p(y|x)^T]
\`\`\`

Block-diagonal approximation makes it tractable:
\`\`\`
F ≈ block_diag(F_1, ..., F_L)
\`\`\`\`\`

**K-FAC:** Kronecker-factored approximation
- Memory: O(sum of factor sizes)
- Usable for deep networks

**D) Hessian-Free Optimization:**

Compute only Hessian-vector products Hv via:
\`\`\`
Hv = lim_{ε→0} [∇f(x + εv) - ∇f(x)] / ε
\`\`\`

Use Conjugate Gradient to solve Hx = -g without forming H.

**Cost:** O(k·n) where k ~ 10-100 (CG iterations)

**Problem:** Still too expensive for huge networks.

**7. Practical Recommendations:**

**For Deep Learning (n > 10⁶):**
- **Adam/RMSProp**: Diagonal approximation (default choice)
- **SGD + Momentum**: First-order + acceleration
- **K-FAC**: If you have compute budget

**For Medium-Scale (n ~ 10⁴ - 10⁶):**
- **L-BFGS**: Good second-order approximation
- **Used in:** Fine-tuning, smaller models

**For Small-Scale (n < 10⁴):**
- **Full Newton**: Feasible
- **Levenberg-Marquardt**: For regression

**8. Why First-Order Methods Work:**

**Surprising fact:** Despite being "suboptimal," SGD works amazingly well!

**Reasons:**
1. **Overparameterization**: Many paths to good solutions
2. **Implicit regularization**: SGD noise acts as regularizer
3. **Flat minima**: SGD finds generalizing solutions
4. **Computational efficiency**: More steps >> fewer better steps

**Better strategy:**
100 cheap SGD steps > 1 expensive Newton step

**9. Hybrid Approaches:**

**Preconditioning:**
Use cheap approximation to Hessian:
\`\`\`
x_{k+1} = x_k - M^{-1}·∇f(x_k)
\`\`\`

where M ≈ H (e.g., diagonal, block-diagonal)

**Examples:**
- **Adam**: M = diag(v_t)^{1/2}
- **K-FAC**: M = block-diagonal Fisher
- **Shampoo**: M = Kronecker-factored approximation

**10. Summary:**

**Why Newton's method is theoretically superior:**
- Quadratic convergence
- Automatic step size
- Handles ill-conditioning

**Why Newton's method is practically infeasible for deep learning:**
- O(n²) memory (impossible for n ~ 10⁶)
- O(n³) computation per step (too slow)
- Noisy Hessian in stochastic setting
- Non-convex landscapes (negative eigenvalues)

**Practical solution:**
Use cheap approximations:
- Diagonal (Adam)
- Low-rank (L-BFGS)
- Block-diagonal (K-FAC)
- First-order + momentum (SGD)

**Key Insight:**
In deep learning, doing many cheap approximate steps beats doing few expensive exact steps. Computation budget determines method choice, not just convergence rate.`,
          keyPoints: [
            'Newton uses 2nd-order Taylor: minimizes quadratic approximation',
            'Newton update: x ← x - H⁻¹∇f (quadratic convergence)',
            'Infeasible for deep learning: O(n²) memory, O(n³) computation',
            'Alternatives: L-BFGS, K-FAC, Hessian-free, Adam (diagonal)',
            'Practical deep learning: cheap first-order >> expensive second-order',
            'Many cheap steps better than few expensive steps',
          ],
        },
      ],
    },
    {
      id: 'convex-optimization',
      title: 'Convex Optimization',
      content: `
# Convex Optimization

## Introduction

Convex optimization is a special class of optimization problems with guaranteed global optimality. Many machine learning problems (linear regression, SVMs, logistic regression) are convex or can be formulated as convex problems.

**Why convex matters:**
- Every local minimum is a global minimum
- Efficient algorithms with convergence guarantees
- Well-understood theory
- Many ML problems naturally convex

## Convex Sets

A set C ⊆ ℝⁿ is **convex** if for any x, y ∈ C and θ ∈ [0,1]:
θx + (1-θ)y ∈ C

**Intuition**: Line segment between any two points stays in the set.

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt

def visualize_convex_sets():
    """Visualize convex vs non-convex sets"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Convex set: circle
    theta = np.linspace(0, 2*np.pi, 100)
    x_conv = np.cos(theta)
    y_conv = np.sin(theta)
    
    ax1.fill(x_conv, y_conv, alpha=0.3, color='blue')
    ax1.plot(x_conv, y_conv, 'b-', linewidth=2)
    
    # Show line segment between two points
    p1, p2 = np.array([-0.5, 0.5]), np.array([0.5, -0.3])
    ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=2, label='Line segment')
    ax1.plot(*p1, 'ro', markersize=10)
    ax1.plot(*p2, 'ro', markersize=10)
    ax1.set_title('Convex Set (Circle)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Non-convex set: crescent
    theta1 = np.linspace(0, 2*np.pi, 100)
    x_outer = 1.5 * np.cos(theta1)
    y_outer = 1.5 * np.sin(theta1)
    
    theta2 = np.linspace(0, 2*np.pi, 100)
    x_inner = 1.2 * np.cos(theta2) + 0.5
    y_inner = 1.2 * np.sin(theta2)
    
    # Create crescent by masking
    mask = x_outer > x_inner[0]
    
    ax2.fill(x_outer, y_outer, alpha=0.3, color='red')
    ax2.plot(x_outer, y_outer, 'r-', linewidth=2)
    
    # Show line segment that exits the set
    p1, p2 = np.array([-1, 0.5]), np.array([0.5, -0.5])
    ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g-', linewidth=2, label='Line segment exits')
    ax2.plot(*p1, 'go', markersize=10)
    ax2.plot(*p2, 'go', markersize=10)
    ax2.set_title('Non-Convex Set')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('convex_sets.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to 'convex_sets.png'")

visualize_convex_sets()
\`\`\`

## Convex Functions

A function f: ℝⁿ → ℝ is **convex** if for any x, y and θ ∈ [0,1]:
f(θx + (1-θ)y) ≤ θf(x) + (1-θ)f(y)

**Intuition**: Chord above graph (function curves upward).

**Examples:**
- Convex: x², eˣ, |x|, -log(x) for x > 0
- Not convex: sin(x), x³

\`\`\`python
def check_convexity():
    """Check if functions are convex"""
    
    # Test convexity via second derivative
    def is_convex_1d(f, df, ddf, x_range):
        """For 1D: f is convex if f''(x) ≥ 0 for all x"""
        x = np.linspace(*x_range, 100)
        second_deriv = ddf(x)
        return np.all(second_deriv >= -1e-10), second_deriv
    
    # Test functions
    functions = [
        {
            'name': 'x²',
            'f': lambda x: x**2,
            'df': lambda x: 2*x,
            'ddf': lambda x: 2 * np.ones_like(x),
            'range': (-2, 2)
        },
        {
            'name': 'eˣ',
            'f': lambda x: np.exp(x),
            'df': lambda x: np.exp(x),
            'ddf': lambda x: np.exp(x),
            'range': (-2, 2)
        },
        {
            'name': 'sin(x)',
            'f': lambda x: np.sin(x),
            'df': lambda x: np.cos(x),
            'ddf': lambda x: -np.sin(x),
            'range': (0, 2*np.pi)
        }
    ]
    
    print("Convexity Check (1D functions):")
    print("="*60)
    
    for func in functions:
        is_conv, second_deriv = is_convex_1d(
            func['f'], func['df'], func['ddf'], func['range']
        )
        print(f"\\n{func['name']}:")
        print(f"  Second derivative range: [{second_deriv.min():.4f}, {second_deriv.max():.4f}]")
        print(f"  Convex: {'✓ Yes' if is_conv else '✗ No'}")

check_convexity()
\`\`\`

## First-Order Condition

For differentiable f, **f is convex** iff:
f(y) ≥ f(x) + ∇f(x)ᵀ(y - x) for all x, y

**Interpretation**: Function lies above its tangent plane.

\`\`\`python
def visualize_first_order_condition():
    """Visualize first-order convexity condition"""
    
    # Convex function: f(x) = x²
    def f(x):
        return x**2
    
    def grad_f(x):
        return 2*x
    
    # Point for tangent
    x0 = 1.0
    
    # Create plot
    x = np.linspace(-2, 3, 200)
    y_func = f(x)
    
    # Tangent line at x0
    y_tangent = f(x0) + grad_f(x0) * (x - x0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_func, 'b-', linewidth=2, label='f(x) = x²')
    plt.plot(x, y_tangent, 'r--', linewidth=2, label=f'Tangent at x={x0}')
    plt.plot(x0, f(x0), 'ro', markersize=10, label=f'Point ({x0}, {f(x0)})')
    
    # Shade region showing f(x) ≥ tangent
    plt.fill_between(x, y_tangent, y_func, where=(y_func >= y_tangent), 
                     alpha=0.3, color='green', label='f(x) ≥ tangent')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("First-Order Convexity Condition: f(y) ≥ f(x) + f'(x)(y-x)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('first_order_condition.png', dpi=150, bbox_inches='tight')
    print("Function lies above tangent everywhere → Convex")

visualize_first_order_condition()
\`\`\`

## Second-Order Condition

For twice-differentiable f, **f is convex** iff:
∇²f(x) ⪰ 0 (Hessian is positive semidefinite) for all x

**Check**: All eigenvalues of Hessian ≥ 0.

\`\`\`python
def check_convexity_hessian():
    """Check convexity using Hessian"""
    
    # f(x, y) = x² + 2y²
    def f(xy):
        x, y = xy
        return x**2 + 2*y**2
    
    def hessian(xy):
        return np.array([[2, 0], [0, 4]])
    
    # Check at random points
    test_points = [
        np.array([0.0, 0.0]),
        np.array([1.0, 2.0]),
        np.array([-1.5, 0.5])
    ]
    
    print("Convexity Check via Hessian:")
    print("="*60)
    print("f(x, y) = x² + 2y²\\n")
    
    all_convex = True
    for point in test_points:
        H = hessian(point)
        eigenvalues = np.linalg.eigvalsh(H)
        is_psd = np.all(eigenvalues >= -1e-10)
        
        print(f"At {point}:")
        print(f"  Hessian:\\n{H}")
        print(f"  Eigenvalues: {eigenvalues}")
        print(f"  Positive semidefinite: {'✓ Yes' if is_psd else '✗ No'}")
        print()
        
        all_convex = all_convex and is_psd
    
    if all_convex:
        print("→ Function is convex (Hessian ⪰ 0 everywhere)")
    else:
        print("→ Function is NOT convex")

check_convexity_hessian()
\`\`\`

## Convex Optimization Problem

**Standard form:**
minimize f(x)
subject to gᵢ(x) ≤ 0, i = 1, ..., m
           hⱼ(x) = 0, j = 1, ..., p

where f and all gᵢ are convex, and hⱼ are affine.

**Key property**: Every local minimum is a global minimum!

\`\`\`python
def convex_vs_nonconvex_optimization():
    """Compare convex vs non-convex optimization"""
    
    # Convex: f(x) = x²
    def f_convex(x):
        return x**2
    
    # Non-convex: f(x) = x⁴ - 4x² + x
    def f_nonconvex(x):
        return x**4 - 4*x**2 + x
    
    x = np.linspace(-3, 3, 300)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Convex
    ax1.plot(x, f_convex(x), 'b-', linewidth=2)
    ax1.plot(0, f_convex(0), 'ro', markersize=15, label='Global minimum')
    ax1.set_title('Convex Function: x²')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.text(0, 2, 'Only one minimum\\n(global)', ha='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Non-convex
    ax2.plot(x, f_nonconvex(x), 'r-', linewidth=2)
    
    # Find local minima numerically
    from scipy.optimize import minimize_scalar
    local_mins = []
    for x0 in [-2, 0, 2]:
        result = minimize_scalar(f_nonconvex, bounds=(-3, 3), method='bounded')
        local_mins.append((result.x, result.fun))
    
    # Mark minima
    ax2.plot(-1.6, f_nonconvex(-1.6), 'go', markersize=12, label='Local minimum')
    ax2.plot(1.7, f_nonconvex(1.7), 'ro', markersize=15, label='Global minimum')
    ax2.set_title('Non-Convex Function: x⁴ - 4x² + x')
    ax2.set_xlabel('x')
    ax2.set_ylabel('f(x)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.text(0, 3, 'Multiple local minima\\n(hard to optimize)', ha='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('convex_vs_nonconvex.png', dpi=150, bbox_inches='tight')
    print("Convex: Local minimum = Global minimum")
    print("Non-convex: Many local minima, hard to find global")

convex_vs_nonconvex_optimization()
\`\`\`

## Applications in Machine Learning

### Linear Regression (Convex)

\`\`\`python
def linear_regression_convex():
    """Linear regression is a convex optimization problem"""
    
    # Generate data
    np.random.seed(42)
    n = 100
    X = np.random.randn(n, 2)
    true_w = np.array([2.0, -1.5])
    y = X @ true_w + np.random.randn(n) * 0.5
    
    # Loss function: MSE is convex
    def loss(w):
        predictions = X @ w
        return 0.5 * np.mean((predictions - y)**2)
    
    def gradient(w):
        predictions = X @ w
        return X.T @ (predictions - y) / n
    
    def hessian(w):
        # H = (1/n) X^T X (independent of w!)
        return X.T @ X / n
    
    # Check convexity
    w_test = np.array([1.0, 1.0])
    H = hessian(w_test)
    eigenvalues = np.linalg.eigvalsh(H)
    
    print("Linear Regression Convexity:")
    print("="*60)
    print(f"Loss function: L(w) = (1/2n)||Xw - y||²")
    print(f"\\nHessian H = (1/n)X^T X:")
    print(H)
    print(f"\\nEigenvalues: {eigenvalues}")
    print(f"All eigenvalues ≥ 0: {np.all(eigenvalues >= -1e-10)}")
    print("→ Loss is convex! Gradient descent guaranteed to find global minimum.")
    
    # Gradient descent
    w = np.zeros(2)
    lr = 0.1
    losses = []
    
    for i in range(100):
        losses.append(loss(w))
        w = w - lr * gradient(w)
    
    print(f"\\nTrue weights: {true_w}")
    print(f"Learned weights: {w}")
    print(f"Final loss: {loss(w):.6f}")

linear_regression_convex()
\`\`\`

### Logistic Regression (Convex)

\`\`\`python
def logistic_regression_convex():
    """Logistic regression has convex loss"""
    
    # Generate binary classification data
    np.random.seed(42)
    n = 100
    X = np.random.randn(n, 2)
    true_w = np.array([1.5, -1.0])
    logits = X @ true_w
    y = (logits + np.random.randn(n) * 0.5 > 0).astype(float)
    
    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    # Binary cross-entropy loss (convex!)
    def loss(w):
        z = X @ w
        p = sigmoid(z)
        return -np.mean(y * np.log(p + 1e-10) + (1-y) * np.log(1-p + 1e-10))
    
    def gradient(w):
        z = X @ w
        p = sigmoid(z)
        return X.T @ (p - y) / n
    
    def hessian(w):
        """Hessian of logistic loss"""
        z = X @ w
        p = sigmoid(z)
        S = np.diag(p * (1 - p))
        return X.T @ S @ X / n
    
    # Check convexity at multiple points
    print("Logistic Regression Convexity:")
    print("="*60)
    
    test_weights = [
        np.array([0.0, 0.0]),
        np.array([1.0, 1.0]),
        np.array([-1.0, 2.0])
    ]
    
    all_convex = True
    for w_test in test_weights:
        H = hessian(w_test)
        eigenvalues = np.linalg.eigvalsh(H)
        is_psd = np.all(eigenvalues >= -1e-10)
        all_convex = all_convex and is_psd
        print(f"\\nAt w = {w_test}:")
        print(f"  Hessian eigenvalues: {eigenvalues}")
        print(f"  PSD: {'✓' if is_psd else '✗'}")
    
    if all_convex:
        print("\\n→ Binary cross-entropy loss is convex!")
        print("  Gradient descent finds global optimum.")

logistic_regression_convex()
\`\`\`

### SVM (Convex)

\`\`\`python
def svm_convex():
    """Support Vector Machine is convex"""
    
    print("Support Vector Machine (SVM):")
    print("="*60)
    
    print("""
Primal problem (convex):
    minimize (1/2)||w||² + C·Σ max(0, 1 - yᵢ(w^T xᵢ + b))
    
Components:
1. ||w||²: Convex (quadratic)
2. max(0, 1 - z): Convex (hinge loss)
3. Sum of convex functions: Convex

Therefore, SVM optimization is convex!

Dual problem (also convex):
    maximize Σαᵢ - (1/2)ΣΣαᵢαⱼyᵢyⱼ(xᵢ^T xⱼ)
    subject to 0 ≤ αᵢ ≤ C, Σαᵢyᵢ = 0

Both primal and dual are convex quadratic programs.
→ Efficiently solvable with guaranteed global optimum!
    """)

svm_convex()
\`\`\`

## KKT Conditions

For convex optimization with constraints, the **Karush-Kuhn-Tucker (KKT) conditions** are necessary and sufficient for optimality.

**KKT Conditions:**
1. **Stationarity**: ∇f(x*) + Σλᵢ∇gᵢ(x*) + Σνⱼ∇hⱼ(x*) = 0
2. **Primal feasibility**: gᵢ(x*) ≤ 0, hⱼ(x*) = 0
3. **Dual feasibility**: λᵢ ≥ 0
4. **Complementary slackness**: λᵢgᵢ(x*) = 0

\`\`\`python
def kkt_example():
    """Demonstrate KKT conditions"""
    
    print("KKT Conditions Example:")
    print("="*60)
    print("""
Problem:
    minimize f(x) = x²
    subject to g(x) = x - 1 ≤ 0
    
Solution:
    x* = 1 (at constraint boundary)
    λ* > 0 (constraint active)
    
Verification:
1. Stationarity: ∇f(x*) + λ∇g(x*) = 0
   2x* + λ·1 = 0
   2(1) + λ = 0 → λ = -2 (wrong sign!)
   
   Actually, we have a mistake. Let me recalculate...
   
   At x* = 1:
   ∇L = 2x + λ = 0
   2(1) + λ = 0 → λ = -2
   
   But we need λ ≥ 0 for minimization!
   
   Correct formulation: constraint should be g(x) = 1 - x ≤ 0
   Then: 2x* - λ = 0 at x* = 1 → λ = 2 ✓
   
2. Primal feasibility: g(x*) = 1 - 1 = 0 ≤ 0 ✓
3. Dual feasibility: λ = 2 ≥ 0 ✓  
4. Complementary slackness: λ·g(x*) = 2·0 = 0 ✓

All KKT conditions satisfied → x* = 1 is optimal!
    """)

kkt_example()
\`\`\`

## Summary

**Key Concepts**:
- **Convex functions**: f(θx + (1-θ)y) ≤ θf(x) + (1-θ)f(y)
- **First-order**: f(y) ≥ f(x) + ∇f(x)ᵀ(y-x)
- **Second-order**: Hessian ⪰ 0
- **Convex optimization**: Local optimum = global optimum
- **KKT conditions**: Necessary + sufficient for constrained convex problems

**ML Applications**:
- Linear regression: Convex (MSE loss)
- Logistic regression: Convex (cross-entropy)
- SVM: Convex (hinge loss + regularization)
- Many others: Lasso, Ridge, Elastic Net

**Why This Matters**:
Understanding convexity tells us when gradient descent is guaranteed to find the global optimum. For non-convex problems (deep learning), we rely on other properties (overparameterization, good initialization, careful architecture design).
`,
      multipleChoice: [
        {
          id: 'convex-1',
          question: 'A function f is convex if:',
          options: [
            'It has a unique minimum',
            'The line segment between any two points on the graph lies above the graph',
            'Its gradient is always positive',
            'It is differentiable everywhere',
          ],
          correctAnswer: 1,
          explanation:
            'Convexity means the chord (line segment) between any two points on the graph lies above or on the graph: f(θx + (1-θ)y) ≤ θf(x) + (1-θ)f(y).',
          difficulty: 'medium',
        },
        {
          id: 'convex-2',
          question:
            'For a twice-differentiable function, convexity can be checked by:',
          options: [
            'Verifying the gradient is zero',
            'Checking if the Hessian is positive semidefinite everywhere',
            'Ensuring the function is monotonic',
            'Computing the Jacobian',
          ],
          correctAnswer: 1,
          explanation:
            'Second-order condition: f is convex iff ∇²f(x) ⪰ 0 (Hessian positive semidefinite) for all x.',
          difficulty: 'medium',
        },
        {
          id: 'convex-3',
          question: 'In convex optimization, every local minimum is:',
          options: [
            'Also a saddle point',
            'Also a global minimum',
            'Not necessarily optimal',
            'A local maximum',
          ],
          correctAnswer: 1,
          explanation:
            'Key property of convex optimization: every local minimum is automatically a global minimum. This is why convex problems are tractable.',
          difficulty: 'easy',
        },
        {
          id: 'convex-4',
          question: 'Which of these ML problems is NOT convex?',
          options: [
            'Linear regression with MSE loss',
            'Logistic regression with cross-entropy loss',
            'Training a deep neural network',
            'Support Vector Machine (SVM)',
          ],
          correctAnswer: 2,
          explanation:
            'Deep neural networks have non-convex loss surfaces due to the composition of non-linear activation functions. Linear regression, logistic regression, and SVMs are all convex.',
          difficulty: 'medium',
        },
        {
          id: 'convex-5',
          question: 'The KKT conditions for constrained optimization are:',
          options: [
            'Only necessary for optimality',
            'Only sufficient for optimality',
            'Both necessary and sufficient for convex problems',
            'Unrelated to convexity',
          ],
          correctAnswer: 2,
          explanation:
            'For convex optimization problems, KKT conditions are both necessary and sufficient for a point to be optimal. For non-convex problems, they are only necessary.',
          difficulty: 'hard',
        },
      ],
      quiz: [
        {
          id: 'convex-disc-1',
          question:
            'Explain why linear regression and logistic regression are convex optimization problems, but training deep neural networks is not. What are the practical implications?',
          hint: 'Consider the structure of the loss functions and the composition of operations.',
          sampleAnswer: `**Convexity in Machine Learning:**

**1. Linear Regression is Convex:**

**Loss function:**
\`\`\`
L(w) = (1/2n)||Xw - y||²
     = (1/2n)Σᵢ(wᵀxᵢ - yᵢ)²
\`\`\`

**Why convex:**

**Second-order condition:**
\`\`\`
∇L(w) = (1/n)X^T(Xw - y)
∇²L(w) = (1/n)X^T X  (Hessian)
\`\`\`

The Hessian H = (1/n)X^TX is positive semidefinite:
- For any vector v: v^T H v = v^T X^T X v = ||Xv||² ≥ 0
- Therefore, L is convex

**Implications:**
- Gradient descent converges to global minimum
- No local minima traps
- Solution unique (if X full rank)

**2. Logistic Regression is Convex:**

**Loss function (binary cross-entropy):**
\`\`\`
L(w) = -(1/n)Σᵢ[yᵢ log σ(w^Txᵢ) + (1-yᵢ) log(1-σ(w^Txᵢ))]
\`\`\`

where σ(z) = 1/(1+e^(-z)) is sigmoid.

**Why convex:**

**Hessian:**
\`\`\`
∇²L(w) = (1/n)X^T S X
\`\`\`

where S = diag(σ(w^Txᵢ)(1-σ(w^Txᵢ)))

**Key observation:**
- σ(z)(1-σ(z)) ∈ [0, 0.25] for all z
- S is positive semidefinite diagonal matrix
- Therefore X^T S X is positive semidefinite
- L is convex!

**Proof:**
For any v:
\`\`\`
v^T (X^T S X) v = (Xv)^T S (Xv) = Σᵢ sᵢ(Xv)ᵢ² ≥ 0
\`\`\`

**3. Deep Neural Networks are NOT Convex:**

**Loss function (e.g., for network with ReLU):**
\`\`\`
L(W₁, W₂, ..., Wₗ) = (1/n)Σᵢ ||fₗ(...f₂(f₁(xᵢ; W₁); W₂)...; Wₗ) - yᵢ||²
\`\`\`

where fₗ(z; W) = ReLU(Wz) or similar.

**Why NOT convex:**

**Problem 1: Composition of non-linearities**

Even simple 2-layer network:
\`\`\`
f(x; W₁, W₂) = W₂·ReLU(W₁x)
\`\`\`

Is NOT convex in (W₁, W₂):
- ReLU is convex in its input
- But composition W₂·ReLU(W₁x) is NOT convex in W₁
- Linear function composed with convex ≠ convex

**Example:**
\`\`\`python
# f(w₁, w₂) = w₂·ReLU(w₁) is not convex
w1 = np.linspace(-2, 2, 100)
w2 = 1.0
f = w2 * np.maximum(0, w1)

# Check second derivative
# f'(w₁) = w₂ if w₁ > 0, else 0
# f''(w₁) = 0 everywhere (except at 0, undefined)
# But f has a "kink" at 0 → not convex!
\`\`\`

**Problem 2: Permutation symmetry**

For network with hidden layer of size h:
- Any permutation of hidden units gives same function
- h! equivalent solutions
- Multiple global minima → non-convex structure

**Problem 3: Scaling symmetry**

Can scale weights in layer i by α and layer i+1 by 1/α:
- Same function output
- Continuum of equivalent solutions
- Non-convex landscape

**Problem 4: Activation functions**

Non-convex activations (tanh, sigmoid as function of weights):
\`\`\`
L(w) = ||σ(Wx) - y||²
\`\`\`

σ(Wx) is NOT convex in w because:
- σ is non-linear
- Composition with linear doesn't preserve convexity

**4. Practical Implications:**

**For Convex Problems (Linear/Logistic Regression):**

**Advantages:**
1. **Global optimum guaranteed**: Any optimization method finds global minimum
2. **Convergence guarantees**: Gradient descent converges with appropriate step size
3. **No initialization sensitivity**: Start anywhere, reach same solution
4. **Theoretical analysis**: Strong convergence rates, sample complexity bounds
5. **Optimization is easy**: Can use simple first-order methods

**Example:**
\`\`\`python
# Linear regression always converges
for any w_init:
    w = gradient_descent(w_init, X, y)
    # w converges to global optimum
\`\`\`

**For Non-Convex Problems (Deep Learning):**

**Challenges:**
1. **Local minima**: May get stuck in suboptimal solutions
2. **Saddle points**: Prevalent in high dimensions (though escapable)
3. **Initialization matters**: Different starts → different solutions
4. **No convergence guarantees**: May not reach global optimum
5. **Hyperparameter tuning**: Learning rate, batch size, architecture crucial

**Example:**
\`\`\`python
# Deep network: initialization matters
w_init_1 = random_init(seed=42)
w_init_2 = random_init(seed=123)

model_1 = train(w_init_1)  # May converge to different solution
model_2 = train(w_init_2)  # Than this one
\`\`\`

**Why Deep Learning Still Works:**

Despite non-convexity, deep learning succeeds because:

1. **Overparameterization**: More parameters than data points
   - Many paths to good solutions
   - Local minima often good enough

2. **Implicit regularization**: SGD noise acts as regularizer
   - Finds flat minima that generalize

3. **Good architecture design**: Skip connections, batch norm
   - Improve optimization landscape

4. **Careful initialization**: He, Xavier, etc.
   - Start in good region

5. **Modern optimizers**: Adam, RMSProp
   - Adaptive learning rates help

**5. Comparison Table:**

| Aspect | Convex (LR, Logistic) | Non-Convex (Deep Learning) |
|--------|----------------------|----------------------------|
| **Global optimum** | Always found | Not guaranteed |
| **Initialization** | Doesn't matter | Critical |
| **Convergence** | Guaranteed | Empirical |
| **Theory** | Strong | Limited |
| **Optimization** | Easy | Hard |
| **Local minima** | = global | May be poor |
| **Hyperparameters** | Few | Many |

**6. Practical Recommendations:**

**For Convex Problems:**
- Use simple methods (gradient descent, Newton's)
- Don't worry about initialization
- Focus on model selection, regularization

**For Non-Convex Problems:**
- Careful initialization (Xavier, He)
- Use modern optimizers (Adam)
- Multiple random restarts
- Architecture search
- Ensemble methods
- Don't expect global optimum, aim for "good enough"

**Conclusion:**

Convexity is a powerful property that makes optimization tractable. Linear and logistic regression enjoy this property due to their simple structure. Deep networks sacrifice convexity for expressiveness, accepting optimization challenges in exchange for representational power. Understanding this trade-off is fundamental to machine learning practice.`,
          keyPoints: [
            'Linear regression: Convex (Hessian = X^TX ⪰ 0)',
            'Logistic regression: Convex (cross-entropy with sigmoid)',
            'Deep networks: Non-convex (composition, symmetries)',
            'Convex: Global optimum guaranteed, easy optimization',
            'Non-convex: Local minima, initialization matters',
            'Deep learning works despite non-convexity (overparameterization, SGD)',
          ],
        },
        {
          id: 'convex-disc-2',
          question:
            'Describe the KKT conditions and their role in constrained optimization. How are they used in SVMs?',
          hint: 'Consider the primal and dual formulations of SVM, and what the KKT conditions tell us about the support vectors.',
          sampleAnswer: `**KKT Conditions and Support Vector Machines:**

**1. KKT Conditions Overview:**

For constrained optimization:
\`\`\`
minimize f(x)
subject to gᵢ(x) ≤ 0, i = 1,...,m
           hⱼ(x) = 0, j = 1,...,p
\`\`\`

**KKT Conditions** (necessary for optimality, sufficient if convex):

1. **Stationarity:**
   \`\`\`
   ∇f(x*) + Σᵢ λᵢ∇gᵢ(x*) + Σⱼ νⱼ∇hⱼ(x*) = 0
   \`\`\`

2. **Primal feasibility:**
   \`\`\`
   gᵢ(x*) ≤ 0, hⱼ(x*) = 0
   \`\`\`

3. **Dual feasibility:**
   \`\`\`
   λᵢ ≥ 0
   \`\`\`

4. **Complementary slackness:**
   \`\`\`
   λᵢ·gᵢ(x*) = 0 for all i
   \`\`\`

**Intuition:**
- **Stationarity**: Gradient of objective balanced by constraint gradients
- **Primal feasibility**: Solution satisfies constraints
- **Dual feasibility**: Lagrange multipliers non-negative (for inequality)
- **Complementary slackness**: Either constraint active (λᵢ > 0, gᵢ = 0) or inactive (λᵢ = 0, gᵢ < 0)

**2. Support Vector Machine Formulation:**

**Primal Problem:**
\`\`\`
minimize (1/2)||w||² + C·Σᵢ ξᵢ
subject to yᵢ(w^T xᵢ + b) ≥ 1 - ξᵢ, ∀i
           ξᵢ ≥ 0, ∀i
\`\`\`

**Interpretation:**
- Maximize margin (minimize ||w||²)
- Allow slack (ξᵢ) for misclassified/margin-violating points
- C: trade-off parameter

**Lagrangian:**
\`\`\`
L(w, b, ξ, α, μ) = (1/2)||w||² + C·Σᵢξᵢ 
                   - Σᵢαᵢ[yᵢ(w^Txᵢ + b) - 1 + ξᵢ]
                   - Σᵢμᵢξᵢ
\`\`\`

where αᵢ ≥ 0, μᵢ ≥ 0 are Lagrange multipliers.

**3. KKT Conditions for SVM:**

**Stationarity:**
\`\`\`
∂L/∂w = w - Σᵢαᵢyᵢxᵢ = 0  →  w* = Σᵢαᵢyᵢxᵢ
∂L/∂b = -Σᵢαᵢyᵢ = 0
∂L/∂ξᵢ = C - αᵢ - μᵢ = 0  →  αᵢ + μᵢ = C
\`\`\`

**Primal feasibility:**
\`\`\`
yᵢ(w^Txᵢ + b) ≥ 1 - ξᵢ
ξᵢ ≥ 0
\`\`\`

**Dual feasibility:**
\`\`\`
αᵢ ≥ 0, μᵢ ≥ 0
\`\`\`

**Complementary slackness:**
\`\`\`
αᵢ[yᵢ(w^Txᵢ + b) - 1 + ξᵢ] = 0
μᵢξᵢ = 0
\`\`\`

**4. Support Vectors Identification:**

From complementary slackness, for each point xᵢ:

**Case 1: αᵢ = 0**
- Point not a support vector
- Correctly classified, outside margin
- yᵢ(w^Txᵢ + b) > 1

**Case 2: 0 < αᵢ < C**
- Point is a support vector
- On the margin boundary
- ξᵢ = 0 (from μᵢξᵢ = 0 and μᵢ = C - αᵢ > 0)
- yᵢ(w^Txᵢ + b) = 1

**Case 3: αᵢ = C**
- Point is a support vector
- Inside margin or misclassified
- ξᵢ > 0
- yᵢ(w^Txᵢ + b) = 1 - ξᵢ < 1

**Visual Summary:**
\`\`\`
αᵢ = 0:        Outside margin (not support vector)
0 < αᵢ < C:    On margin (support vector)
αᵢ = C:        Inside margin/misclassified (support vector)
\`\`\`

**5. Dual Formulation:**

Using stationarity conditions, eliminate w, b, ξ:

**Dual Problem:**
\`\`\`
maximize Σᵢαᵢ - (1/2)ΣᵢΣⱼαᵢαⱼyᵢyⱼ(xᵢ^Txⱼ)
subject to 0 ≤ αᵢ ≤ C, ∀i
           Σᵢαᵢyᵢ = 0
\`\`\`

**Advantages of dual:**
1. **Kernel trick**: Can replace xᵢ^Txⱼ with K(xᵢ,xⱼ)
2. **Sparsity**: Many αᵢ = 0 (only support vectors matter)
3. **Convex quadratic program**: Efficiently solvable

**6. Practical Example:**

\`\`\`python
import numpy as np
from sklearn.svm import SVC

# Generate linearly separable data
np.random.seed(42)
X_pos = np.random.randn(50, 2) + [2, 2]
X_neg = np.random.randn(50, 2) + [-2, -2]
X = np.vstack([X_pos, X_neg])
y = np.array([1]*50 + [-1]*50)

# Train SVM
svm = SVC(kernel='linear', C=1.0)
svm.fit(X, y)

# Extract support vectors
support_vectors = svm.support_vectors_
support_indices = svm.support_
alphas = np.abs(svm.dual_coef_[0])

print("SVM with KKT Conditions:")
print(f"Total points: {len(X)}")
print(f"Support vectors: {len(support_vectors)}")
print(f"\\nSupport vector analysis:")

for i, (idx, alpha) in enumerate(zip(support_indices, alphas)):
    point = X[idx]
    label = y[idx]
    # Decision function value
    decision = svm.decision_function([point])[0]
    margin = label * decision
    
    print(f"\\nSV {i+1}:")
    print(f"  α = {alpha:.4f}")
    print(f"  Margin = {margin:.4f}")
    
    if alpha < 0.99 * svm.C:  # 0 < α < C
        print(f"  Status: On margin (ξ = 0)")
    else:  # α = C
        print(f"  Status: Inside margin (ξ > 0)")

print(f"\\nWeights w = Σαᵢyᵢxᵢ: {svm.coef_[0]}")
print(f"Bias b: {svm.intercept_[0]:.4f}")
\`\`\`

**7. Why KKT Matters for SVM:**

**Theoretical:**
1. **Optimality**: KKT conditions prove solution is optimal
2. **Uniqueness**: For strictly convex problem, unique solution
3. **Duality gap**: Zero for convex problems (strong duality)

**Practical:**
1. **Sparsity**: Only support vectors (αᵢ > 0) matter
   - Efficient prediction: O(# support vectors) not O(n)
   - Memory: Store only support vectors

2. **Kernel trick**: Decision function
   \`\`\`
   f(x) = Σᵢαᵢyᵢ K(xᵢ, x) + b
   \`\`\`
   Only need kernel evaluations with support vectors

3. **Optimization**: Can use specialized quadratic programming solvers
   - SMO (Sequential Minimal Optimization)
   - Coordinate ascent on dual

4. **Interpretability**: Support vectors are "important" points
   - On or inside margin
   - Define decision boundary

**8. Connection to Other ML Methods:**

**Boosting:**
KKT conditions show why boosting focuses on misclassified points:
- Points with α = C are misclassified/margin-violating
- Similar to boosting's reweighting

**Active Learning:**
Support vectors are informative points:
- Query points near decision boundary
- Similar to uncertainty sampling

**9. Summary:**

**KKT Conditions provide:**
- **Necessary + sufficient** conditions for convex optimization
- **Identify support vectors**: Points with αᵢ > 0
- **Enable dual formulation**: Kernel trick possible
- **Guarantee optimality**: Solution satisfies all conditions

**For SVM specifically:**
- Complementary slackness identifies 3 types of points
- Dual formulation leads to sparse solution
- Only support vectors needed for prediction
- Kernel trick enables non-linear decision boundaries

**Key Insight:**
KKT conditions transform constrained optimization into system of equations. For SVM, this reveals geometric interpretation: support vectors are the "critical" points that define the decision boundary.`,
          keyPoints: [
            'KKT: 4 conditions (stationarity, primal/dual feasibility, complementary slackness)',
            'Necessary + sufficient for convex problems',
            'SVM: Complementary slackness identifies support vectors',
            'αᵢ = 0: not SV; 0 < αᵢ < C: on margin; αᵢ = C: inside margin',
            'Dual formulation enables kernel trick',
            'Sparsity: only support vectors matter for prediction',
          ],
        },
        {
          id: 'convex-disc-3',
          question:
            'Why does gradient descent work well for non-convex deep learning despite the lack of convexity guarantees? Discuss overparameterization and the optimization landscape.',
          hint: 'Consider the number of parameters vs data points, properties of local minima in overparameterized networks, and implicit regularization.',
          sampleAnswer: `**Why Gradient Descent Works for Non-Convex Deep Learning:**

Despite non-convexity, SGD succeeds in deep learning. This is one of the most surprising and important phenomena in modern ML.

**1. The Paradox:**

**Classical optimization theory:**
- Non-convex → many local minima
- Gradient descent → trapped in bad local minima
- No guarantees of finding global optimum

**Deep learning reality:**
- Highly non-convex loss landscapes
- Simple SGD works remarkably well
- Often reaches solutions with excellent generalization

**Why the disconnect?**

**2. Overparameterization:**

**Definition:** Network has more parameters than training examples.

**Modern networks:**
\`\`\`
ResNet-50: ~25M parameters
Training data: ~1M images
Ratio: 25:1 overparameterized
\`\`\`

**Extreme cases:**
\`\`\`
GPT-3: 175B parameters
Training data: ~500B tokens
Still highly overparameterized at parameter level
\`\`\`

**3. Loss Landscape Properties in Overparameterized Networks:**

**Observation 1: Local Minima are Good**

**Theory (Choromanska et al., 2015):**
For sufficiently wide networks, most local minima have similar loss values close to global minimum.

**Why?**
- High dimensionality creates many descent directions
- Poor local minima become rare
- Most critical points are saddle points (escapable)

**Empirical evidence:**
\`\`\`python
# Train same architecture from different initializations
losses = []
for seed in range(10):
    model = Network(seed=seed)
    final_loss = train(model)
    losses.append(final_loss)

print(f"Final losses: {losses}")
# Output: [0.23, 0.24, 0.23, 0.24, 0.23, ...]
# Very similar despite different local minima!
\`\`\`

**Observation 2: No Bad Local Minima in Overparameterized Regime**

**Theorem (simplified):**
For 2-layer ReLU networks with sufficiently many hidden units (width >> data size), gradient descent finds global minimum.

**Intuition:**
- Each neuron can specialize to few data points
- Enough neurons → can fit all data
- No conflict between objectives

**4. Implicit Regularization of SGD:**

**Key insight:** SGD doesn't just minimize training loss - it implicitly regularizes.

**Mechanism 1: Noise as Regularization**

SGD gradient estimate:
\`\`\`
∇L_batch ≠ ∇L_full
\`\`\`

Noise helps:
- Escape sharp minima
- Find flat minima (better generalization)

**Flat vs Sharp Minima:**
\`\`\`
Flat minimum: Loss changes slowly around w*
  → Robust to perturbations
  → Better generalization

Sharp minimum: Loss changes rapidly
  → Overfits to training data
  → Poor generalization
\`\`\`

**Evidence:**
\`\`\`python
# Sharpness measurement
def sharpness(model, data):
    # Measure eigenvalues of Hessian
    loss = compute_loss(model, data)
    hessian = compute_hessian(loss)
    max_eigenvalue = max_eig(hessian)
    return max_eigenvalue

# SGD finds flatter minima than full-batch GD
sharp_sgd = sharpness(model_sgd, data)
sharp_gd = sharpness(model_gd, data)
print(f"SGD sharpness: {sharp_sgd}")  # Lower
print(f"GD sharpness: {sharp_gd}")    # Higher
\`\`\`

**Mechanism 2: Implicit Bias Toward Simple Solutions**

SGD favors solutions with lower "complexity":
- For linear models: Small-norm solutions
- For neural networks: Solutions with lower effective dimension

**Example:**
Two networks fit training data perfectly:
- Network A: Uses all parameters roughly equally
- Network B: Most weights near zero, few active
- SGD prefers Network B (implicit L2 regularization)

**5. Overparameterization + SGD = Generalization:**

**Double Descent Phenomenon:**

\`\`\`
Test Error
    |
    |     Classical U-curve
    |    /              \\
    |   /                \\___
    |  /                      \\_____
    | /                              \\
    |/____________________\\_____________\\___
     Under    Interpolation    Over
    -param      threshold    -param
              (# params = # data)
\`\`\`

**Surprising observation:**
- Classical: Overparameterization → overfitting
- Modern DL: Further overparameterization → better generalization!

**Explanation:**
- At interpolation threshold: Many solutions fit data, some bad
- In overparameterized regime: More solutions, SGD finds good one

**6. Neural Tangent Kernel (NTK) Theory:**

**Key idea:** In infinite-width limit, neural networks behave like kernel methods.

**Implication:**
- Optimization becomes convex (approximately)
- Explains why gradient descent works

**However:**
- Real networks finite width
- NTK approximation not perfect
- But provides theoretical insight

**7. Mode Connectivity:**

**Observation (Garipov et al., 2018):**
Different solutions found by SGD can be connected by paths of low loss.

**Implication:**
Loss landscape has connected "valleys" of good solutions.

\`\`\`python
# Connect two solutions
w1 = train(model, seed=1)
w2 = train(model, seed=2)

# Linear interpolation
alphas = np.linspace(0, 1, 100)
losses = []
for alpha in alphas:
    w_interp = alpha * w1 + (1-alpha) * w2
    loss = evaluate(w_interp)
    losses.append(loss)

# Often see: losses remain low throughout path!
plt.plot(alphas, losses)
\`\`\`

**8. Practical Factors:**

**Architecture Design:**
- Skip connections (ResNets): Improve gradient flow
- Batch normalization: Smooths loss landscape
- Attention mechanisms: Better optimization

**Initialization:**
- Xavier/He initialization: Start in good region
- Prevents gradient vanishing/exploding

**Hyperparameters:**
- Learning rate schedule: Annealing helps convergence
- Batch size: Affects generalization

**Data Augmentation:**
- Implicitly regularizes
- Prevents memorization

**9. Comparison: Convex vs Non-Convex:**

| Aspect | Convex (LR) | Non-Convex (DL) |
|--------|-------------|-----------------|
| **Theory** | Strong guarantees | Weak/empirical |
| **Optimization** | Guaranteed global | No guarantees |
| **Local minima** | All global | Many, but similar |
| **Initialization** | Doesn't matter | Critical |
| **Why it works** | Convexity | Overparameterization + SGD |
| **Generalization** | Classical theory | Modern phenomena |

**10. Summary:**

**Why SGD works for non-convex deep learning:**

1. **Overparameterization:**
   - More parameters than constraints
   - Many paths to good solutions
   - Local minima have similar quality

2. **Implicit regularization:**
   - SGD noise finds flat minima
   - Flat minima generalize better
   - Implicit bias toward simple solutions

3. **Loss landscape structure:**
   - Bad local minima rare in high dimensions
   - Saddle points escapable
   - Connected valleys of good solutions

4. **Architecture + engineering:**
   - Skip connections, batch norm
   - Good initialization
   - Careful hyperparameter tuning

**Key Insight:**

Deep learning doesn't succeed *despite* non-convexity - it succeeds because:
- Overparameterization creates benign loss landscapes
- SGD implicitly regularizes toward generalizing solutions
- The "right" inductive biases (architecture, optimization) guide search

**Practical Takeaway:**

You don't need convexity for successful optimization. With:
- Sufficient overparameterization
- Stochastic gradient descent
- Good architecture
- Proper initialization

You can reliably train deep networks to excellent solutions, even though the problem is highly non-convex.

This is why deep learning works - not despite theory, but because we've discovered that overparameterized non-convex optimization has surprisingly good properties!`,
          keyPoints: [
            'Overparameterization: more parameters than data',
            'Local minima in overparameterized networks have similar quality',
            'SGD implicitly regularizes toward flat minima',
            'Flat minima generalize better than sharp minima',
            'Bad local minima rare in high dimensions',
            'Architecture design + initialization + SGD → reliable training',
          ],
        },
      ],
    },
    {
      id: 'numerical-optimization',
      title: 'Numerical Optimization Methods',
      content: `
# Numerical Optimization Methods

## Introduction

Practical optimization relies on iterative numerical algorithms. Understanding these methods is crucial for tuning hyperparameters and diagnosing training issues in machine learning.

**Goal:** Minimize f(x) where we can compute f(x) and ∇f(x), but not solve ∇f(x) = 0 analytically.

## Gradient Descent

**Update rule:**
x_{k+1} = x_k - α∇f(x_k)

where α is the learning rate (step size).

**Convergence for convex smooth f:**
- With appropriate α, converges to global minimum
- Rate: O(1/k) for general convex, O(exp(-k)) for strongly convex

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(f, grad_f, x0, alpha=0.1, max_iter=100, tol=1e-6):
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
    history = {'x': [x.copy()], 'f': [f(x)]}
    
    for k in range(max_iter):
        # Compute gradient
        grad = grad_f(x)
        
        # Update
        x = x - alpha * grad
        
        history['x'].append(x.copy())
        history['f'].append(f(x))
        
        # Check convergence
        if np.linalg.norm(grad) < tol:
            print(f"Converged in {k+1} iterations")
            break
    
    return x, history

# Test on quadratic function
def f(x):
    return 0.5 * x[0]**2 + 2 * x[1]**2

def grad_f(x):
    return np.array([x[0], 4*x[1]])

x0 = np.array([4.0, 3.0])
x_opt, history = gradient_descent(f, grad_f, x0, alpha=0.1)

print(f"Optimum: {x_opt}")
print(f"Function value: {f(x_opt):.6f}")

# Visualize
x_history = np.array(history['x'])
plt.figure(figsize=(10, 5))

# Left: Contour plot with trajectory
ax1 = plt.subplot(121)
x1 = np.linspace(-5, 5, 100)
x2 = np.linspace(-5, 5, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = 0.5 * X1**2 + 2 * X2**2

ax1.contour(X1, X2, Z, levels=20)
ax1.plot(x_history[:, 0], x_history[:, 1], 'ro-', linewidth=2, markersize=4)
ax1.plot(0, 0, 'g*', markersize=15, label='Optimum')
ax1.set_xlabel('x₁')
ax1.set_ylabel('x₂')
ax1.set_title('Gradient Descent Trajectory')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: Function value over iterations
ax2 = plt.subplot(122)
ax2.semilogy(history['f'])
ax2.set_xlabel('Iteration')
ax2.set_ylabel('f(x)')
ax2.set_title('Convergence (log scale)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gradient_descent.png', dpi=150, bbox_inches='tight')
print("Saved visualization")
\`\`\`

## Momentum

**Idea:** Accumulate velocity to accelerate in persistent directions.

**Update:**
v_{k+1} = βv_k - α∇f(x_k)
x_{k+1} = x_k + v_{k+1}

where β ∈ [0, 1) is momentum coefficient (typically 0.9).

**Benefits:**
- Faster convergence in ravines
- Dampens oscillations
- Can escape shallow local minima

\`\`\`python
def gradient_descent_momentum(f, grad_f, x0, alpha=0.1, beta=0.9, max_iter=100):
    """Gradient descent with momentum"""
    x = x0.copy()
    v = np.zeros_like(x)
    history = {'x': [x.copy()], 'f': [f(x)]}
    
    for k in range(max_iter):
        grad = grad_f(x)
        
        # Update velocity
        v = beta * v - alpha * grad
        
        # Update position
        x = x + v
        
        history['x'].append(x.copy())
        history['f'].append(f(x))
    
    return x, history

# Compare with plain GD
x_gd, hist_gd = gradient_descent(f, grad_f, x0, alpha=0.1, max_iter=50)
x_mom, hist_mom = gradient_descent_momentum(f, grad_f, x0, alpha=0.1, beta=0.9, max_iter=50)

print("Comparison:")
print(f"GD final loss: {hist_gd['f'][-1]:.6f}")
print(f"Momentum final loss: {hist_mom['f'][-1]:.6f}")
print(f"→ Momentum converges faster!")
\`\`\`

## Adam Optimizer

**Adaptive Moment Estimation** - combines momentum with adaptive learning rates.

**Update:**
m_t = β₁m_{t-1} + (1-β₁)∇f(x_t)  # First moment (mean)
v_t = β₂v_{t-1} + (1-β₂)∇f(x_t)²  # Second moment (variance)
m̂_t = m_t/(1-β₁^t)  # Bias correction
v̂_t = v_t/(1-β₂^t)
x_{t+1} = x_t - α·m̂_t/(√v̂_t + ε)

**Default:** β₁=0.9, β₂=0.999, ε=10⁻⁸

\`\`\`python
def adam(f, grad_f, x0, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8, max_iter=100):
    """Adam optimizer"""
    x = x0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    history = {'x': [x.copy()], 'f': [f(x)]}
    
    for t in range(1, max_iter + 1):
        grad = grad_f(x)
        
        # Update biased first moment
        m = beta1 * m + (1 - beta1) * grad
        
        # Update biased second moment
        v = beta2 * v + (1 - beta2) * grad**2
        
        # Bias correction
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        # Update parameters
        x = x - alpha * m_hat / (np.sqrt(v_hat) + eps)
        
        history['x'].append(x.copy())
        history['f'].append(f(x))
    
    return x, history

x_adam, hist_adam = adam(f, grad_f, x0, alpha=0.1, max_iter=50)

print(f"Adam final loss: {hist_adam['f'][-1]:.6f}")
print("→ Adam adapts learning rate per parameter")
\`\`\`

## Line Search

Instead of fixed learning rate, find α_k that minimizes f(x_k - α∇f(x_k)).

**Backtracking line search:**
Start with α, reduce until Armijo condition satisfied:
f(x - α∇f(x)) ≤ f(x) - c·α·||∇f(x)||²

\`\`\`python
def backtracking_line_search(f, grad_f, x, grad, alpha_init=1.0, c=1e-4, rho=0.5):
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
    fx = f(x)
    grad_norm_sq = np.dot(grad, grad)
    
    while f(x - alpha * grad) > fx - c * alpha * grad_norm_sq:
        alpha *= rho
        if alpha < 1e-10:
            break
    
    return alpha

def gradient_descent_line_search(f, grad_f, x0, max_iter=100, tol=1e-6):
    """Gradient descent with line search"""
    x = x0.copy()
    history = {'x': [x.copy()], 'f': [f(x)], 'alpha': []}
    
    for k in range(max_iter):
        grad = grad_f(x)
        
        # Find step size via line search
        alpha = backtracking_line_search(f, grad_f, x, grad)
        history['alpha'].append(alpha)
        
        # Update
        x = x - alpha * grad
        
        history['x'].append(x.copy())
        history['f'].append(f(x))
        
        if np.linalg.norm(grad) < tol:
            print(f"Converged in {k+1} iterations")
            break
    
    return x, history

x_ls, hist_ls = gradient_descent_line_search(f, grad_f, x0, max_iter=50)

print(f"\\nLine search:")
print(f"Final loss: {hist_ls['f'][-1]:.6f}")
print(f"Average step size: {np.mean(hist_ls['alpha']):.4f}")
print("→ Adaptive step size improves convergence")
\`\`\`

## Newton's Method

**Update:**
x_{k+1} = x_k - H^{-1}∇f(x_k)

where H = ∇²f(x_k) is the Hessian.

**Pros:**
- Quadratic convergence near optimum
- Automatic step size

**Cons:**
- Expensive (O(n³) per iteration)
- Requires Hessian
- May diverge if H not positive definite

\`\`\`python
def newton_method(f, grad_f, hessian_f, x0, max_iter=20):
    """Newton's method"""
    x = x0.copy()
    history = {'x': [x.copy()], 'f': [f(x)]}
    
    for k in range(max_iter):
        grad = grad_f(x)
        H = hessian_f(x)
        
        # Solve Hx = grad for x
        try:
            direction = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            print("Hessian singular, stopping")
            break
        
        # Newton step
        x = x - direction
        
        history['x'].append(x.copy())
        history['f'].append(f(x))
        
        if np.linalg.norm(grad) < 1e-6:
            print(f"Converged in {k+1} iterations")
            break
    
    return x, history

def hessian_f(x):
    return np.array([[1, 0], [0, 4]])

x_newton, hist_newton = newton_method(f, grad_f, hessian_f, x0, max_iter=10)

# Compare convergence rates
print("\\nConvergence comparison (10 iterations):")
print(f"GD:        {hist_gd['f'][min(10, len(hist_gd['f'])-1)]:.2e}")
print(f"Momentum:  {hist_mom['f'][min(10, len(hist_mom['f'])-1)]:.2e}")
print(f"Adam:      {hist_adam['f'][min(10, len(hist_adam['f'])-1)]:.2e}")
print(f"Newton:    {hist_newton['f'][min(10, len(hist_newton['f'])-1)]:.2e}")
print("→ Newton converges fastest (when Hessian available)")
\`\`\`

## Stochastic Gradient Descent (SGD)

**Motivation:** For large datasets, computing full gradient expensive.

**Idea:** Use gradient estimate from mini-batch:
∇f(x) ≈ (1/B)Σᵢ∈batch ∇fᵢ(x)

**Update:**
x_{k+1} = x_k - α_k · ∇̂f(x_k)

where ∇̂f is noisy gradient estimate.

\`\`\`python
def sgd_example():
    """Demonstrate SGD on simple problem"""
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 2)
    w_true = np.array([2.0, -1.0])
    y = X @ w_true + np.random.randn(n_samples) * 0.1
    
    def loss_full(w):
        """Full batch loss"""
        return 0.5 * np.mean((X @ w - y)**2)
    
    def grad_full(w):
        """Full batch gradient"""
        return X.T @ (X @ w - y) / n_samples
    
    def grad_mini_batch(w, batch_size=32):
        """Mini-batch gradient"""
        indices = np.random.choice(n_samples, batch_size, replace=False)
        X_batch = X[indices]
        y_batch = y[indices]
        return X_batch.T @ (X_batch @ w - y_batch) / batch_size
    
    # Full batch gradient descent
    w_gd = np.zeros(2)
    losses_gd = []
    for _ in range(100):
        losses_gd.append(loss_full(w_gd))
        w_gd = w_gd - 0.1 * grad_full(w_gd)
    
    # SGD with mini-batches
    w_sgd = np.zeros(2)
    losses_sgd = []
    for _ in range(100):
        losses_sgd.append(loss_full(w_sgd))
        w_sgd = w_sgd - 0.1 * grad_mini_batch(w_sgd, batch_size=32)
    
    print("Final weights:")
    print(f"True: {w_true}")
    print(f"GD:   {w_gd}")
    print(f"SGD:  {w_sgd}")
    
    plt.figure(figsize=(10, 4))
    plt.semilogy(losses_gd, label='Full batch GD')
    plt.semilogy(losses_sgd, label='Mini-batch SGD', alpha=0.7)
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
      multipleChoice: [
        {
          id: 'numopt-1',
          question: 'Gradient descent with momentum helps because:',
          options: [
            'It uses second-order information',
            'It accumulates velocity to accelerate in persistent directions',
            'It computes exact gradients',
            "It doesn't require gradients",
          ],
          correctAnswer: 1,
          explanation:
            'Momentum accumulates a velocity term that smooths updates and accelerates movement in directions of consistent gradient, helping escape ravines and oscillations.',
          difficulty: 'medium',
        },
        {
          id: 'numopt-2',
          question:
            "Why is Newton's method not commonly used for training deep neural networks?",
          options: [
            "It doesn't converge",
            'Computing and inverting the Hessian is O(n³), prohibitively expensive for millions of parameters',
            'It only works for linear functions',
            'It requires no gradients',
          ],
          correctAnswer: 1,
          explanation:
            "Newton's method requires computing, storing, and inverting an n×n Hessian matrix, which is computationally intractable for networks with millions of parameters.",
          difficulty: 'easy',
        },
        {
          id: 'numopt-3',
          question: 'Adam optimizer combines:',
          options: [
            'Momentum and adaptive learning rates per parameter',
            "Newton's method and gradient descent",
            'Random search and gradient descent',
            'Only momentum',
          ],
          correctAnswer: 0,
          explanation:
            'Adam (Adaptive Moment Estimation) combines momentum (first moment) with adaptive learning rates based on second moment estimates, providing both acceleration and per-parameter adaptation.',
          difficulty: 'medium',
        },
        {
          id: 'numopt-4',
          question: 'Stochastic gradient descent uses:',
          options: [
            'The full dataset gradient at each step',
            'Mini-batch gradients as noisy estimates of the full gradient',
            'No gradients',
            'Only second derivatives',
          ],
          correctAnswer: 1,
          explanation:
            'SGD uses gradients computed on small mini-batches as noisy estimates of the full gradient, trading some accuracy for computational efficiency on large datasets.',
          difficulty: 'easy',
        },
        {
          id: 'numopt-5',
          question: 'Line search methods:',
          options: [
            'Fix the learning rate for all iterations',
            'Adaptively choose step size to satisfy descent conditions',
            'Require no gradient information',
            'Only work for convex functions',
          ],
          correctAnswer: 1,
          explanation:
            'Line search methods (like backtracking) adaptively find step sizes that guarantee sufficient decrease, typically using conditions like the Armijo rule.',
          difficulty: 'medium',
        },
      ],
      quiz: [
        {
          id: 'numopt-disc-1',
          question:
            'Discuss the trade-offs between full-batch gradient descent and mini-batch SGD. Why is SGD preferred for deep learning despite noisier gradients?',
          hint: 'Consider computational cost per iteration, convergence properties, generalization, and implicit regularization.',
          sampleAnswer: `**Full-Batch GD vs Mini-Batch SGD:**

**1. Computational Cost:**

**Full-Batch GD:**
\`\`\`
Cost per iteration = O(n·d)
where n = dataset size, d = model complexity

For n = 1M images, d = ResNet-50 forward pass:
One iteration ≈ seconds to minutes
\`\`\`

**Mini-Batch SGD:**
\`\`\`
Cost per iteration = O(B·d)
where B = batch size (typically 32-256)

Same model, B = 64:
One iteration ≈ milliseconds
Speedup: ~15,000×!
\`\`\`

**Key insight:** SGD does many cheap updates instead of few expensive ones.

**2. Convergence Properties:**

**Full-Batch GD:**
- Smooth, deterministic convergence
- Guaranteed descent: f(x_{k+1}) < f(x_k)
- Convergence rate: O(1/k) for convex, O(exp(-k)) for strongly convex
- Finds precise minimum

**Mini-Batch SGD:**
- Noisy, stochastic convergence
- May increase loss temporarily
- Expected descent: E[f(x_{k+1})] < f(x_k)
- Oscillates around minimum
- Slower final convergence

**3. Why SGD Still Preferred:**

**A) Computational Efficiency:**

Total work to reach ε accuracy:

**Full-Batch:**
\`\`\`
Iterations needed: K ~ 1/ε
Cost: K × n = n/ε
\`\`\`

**SGD:**
\`\`\`
Iterations needed: K ~ 1/ε²
Cost: K × B = B/ε²
\`\`\`

For large n >> B, SGD wins despite slower convergence!

**Example:**
\`\`\`
n = 1M, B = 64, target ε = 0.01

Full-batch: 1M / 0.01 = 100M operations
SGD: 64 / 0.01² = 640K operations
→ SGD ~156× faster to ε-accuracy!
\`\`\`

**B) Generalization Benefits:**

**Flat Minima:**
SGD noise helps escape sharp minima and find flat minima.

\`\`\`
Sharp minimum: Small perturbation → large loss increase
Flat minimum: Perturbations don't hurt much
\`\`\`

Flat minima generalize better to test data!

**Evidence:**
\`\`\`python
# Measure sharpness (max Hessian eigenvalue)
model_gd = train_full_batch(data)
model_sgd = train_mini_batch(data)

sharpness_gd = max_eigenvalue(hessian(model_gd))
sharpness_sgd = max_eigenvalue(hessian(model_sgd))

print(f"GD sharpness: {sharpness_gd:.2f}")    # Higher
print(f"SGD sharpness: {sharpness_sgd:.2f}")  # Lower

test_acc_gd = evaluate(model_gd, test_data)
test_acc_sgd = evaluate(model_sgd, test_data)

print(f"GD test accuracy: {test_acc_gd:.2%}")   # Lower
print(f"SGD test accuracy: {test_acc_sgd:.2%}") # Higher!
\`\`\`

**C) Implicit Regularization:**

SGD noise acts as implicit regularizer:
- Prevents memorization of training data
- Encourages simpler solutions
- Similar to adding noise to weights

**D) Escaping Saddle Points:**

High-dimensional optimization: saddle points prevalent

**Full-Batch GD:**
- Can get stuck near saddle points (gradient ≈ 0)
- Slow escape (requires tiny gradient component)

**SGD:**
- Noise perturbs away from saddles
- Faster escape

**4. Practical Considerations:**

**Batch Size Effects:**

**Small batches (B=32):**
- More noise → better generalization
- Slower convergence
- Less parallelism

**Large batches (B=8192):**
- Less noise → sharper minima
- Faster convergence per epoch
- Better hardware utilization
- May need learning rate scaling

**Modern practice:** B=64-256 good balance

**Learning Rate Schedules:**

SGD benefits from decaying learning rate:
\`\`\`
α_t = α_0 / (1 + decay × t)
\`\`\`

Reduces noise as training progresses.

**5. Hybrid Approaches:**

**Variance Reduction:**
Methods like SVRG reduce SGD variance while keeping low cost:
\`\`\`
Occasionally compute full gradient
Use it to correct mini-batch gradients
\`\`\`

**Large Batch Training:**
Scale learning rate with batch size:
\`\`\`
α = α_base × (B / B_base)
\`\`\`

Enables efficient distributed training.

**6. Comparison Table:**

| Aspect | Full-Batch GD | Mini-Batch SGD |
|--------|---------------|----------------|
| **Cost/iter** | O(n) | O(B) |
| **Convergence** | Smooth | Noisy |
| **To ε-accuracy** | O(n/ε) | O(B/ε²) |
| **Generalization** | Worse (sharp) | Better (flat) |
| **Parallelism** | Limited | High |
| **Memory** | Need full data | Need only batch |
| **Use case** | Small datasets | Large datasets |

**7. When to Use Each:**

**Full-Batch GD:**
- Small datasets (n < 10K)
- Convex optimization
- High precision required
- Determinism important

**Mini-Batch SGD:**
- Large datasets (n > 100K)
- Deep learning
- Good enough solution acceptable
- Computational efficiency critical

**8. Summary:**

**Why SGD preferred for deep learning:**

1. **Computational efficiency**: ~100-1000× faster per epoch
2. **Generalization**: Noise finds flat minima
3. **Scalability**: Constant memory per iteration
4. **Practical success**: Empirically works extremely well

**Trade-off:**
- Sacrifice smooth convergence and final precision
- Gain computational efficiency and generalization

**Key Insight:**

For deep learning, getting to "good enough" solution quickly matters more than converging to exact minimum slowly. SGD's noise is a feature, not a bug - it implicitly regularizes and helps generalization.

**Practical takeaway:**
Almost all modern deep learning uses mini-batch SGD (or variants like Adam that build on SGD). Full-batch gradient descent is rarely used except for small-scale convex optimization.`,
          keyPoints: [
            'SGD: O(B) cost per iteration vs GD: O(n)',
            'SGD reaches ε-accuracy faster in wall-clock time despite slower convergence',
            'SGD noise finds flat minima → better generalization',
            'SGD implicitly regularizes, prevents overfitting',
            'Batch size trade-off: small (generalization) vs large (speed)',
            'Modern deep learning: almost exclusively mini-batch SGD',
          ],
        },
        {
          id: 'numopt-disc-2',
          question:
            'Explain how Adam optimizer works and why it has become the default choice for training deep neural networks.',
          hint: 'Consider adaptive learning rates, momentum, bias correction, and practical performance.',
          sampleAnswer: `**Adam Optimizer: Adaptive Moment Estimation**

Adam is the most widely used optimizer for deep learning, combining the best features of momentum and adaptive learning rates.

**1. Core Mechanism:**

**Update equations:**
\`\`\`
m_t = β₁·m_{t-1} + (1-β₁)·g_t         # First moment (momentum)
v_t = β₂·v_{t-1} + (1-β₂)·g_t²        # Second moment (variance)
m̂_t = m_t / (1 - β₁^t)                # Bias-corrected first moment
v̂_t = v_t / (1 - β₂^t)                # Bias-corrected second moment
θ_t = θ_{t-1} - α·m̂_t / (√v̂_t + ε)   # Parameter update
\`\`\`

**Default hyperparameters:**
- α = 0.001 (learning rate)
- β₁ = 0.9 (first moment decay)
- β₂ = 0.999 (second moment decay)
- ε = 10⁻⁸ (numerical stability)

**2. Key Components:**

**A) First Moment (Momentum):**

Like momentum in physics:
\`\`\`
m_t = 0.9·m_{t-1} + 0.1·g_t
\`\`\`

**Benefits:**
- Smooths noisy gradients
- Accelerates in consistent directions
- Dampens oscillations

**Example:**
\`\`\`python
# Gradient oscillates: [1, -1, 1, -1, ...]
# Without momentum: Updates oscillate
# With momentum: Averages to ~0, reduces oscillation
\`\`\`

**B) Second Moment (Adaptive LR):**

Tracks gradient variance:
\`\`\`
v_t = 0.999·v_{t-1} + 0.001·g_t²
\`\`\`

**Intuition:**
- If parameter has large gradients → large v_t → small effective LR
- If parameter has small gradients → small v_t → large effective LR

**Effect:** Per-parameter adaptive learning rates!

**C) Bias Correction:**

**Problem:** m_t and v_t initialized at 0, biased toward zero early in training.

**Solution:**
\`\`\`
m̂_t = m_t / (1 - β₁^t)
v̂_t = v_t / (1 - β₂^t)
\`\`\`

**Example:**
\`\`\`
t=1: 1 - 0.9¹ = 0.1  → divide by 0.1 (10× correction)
t=2: 1 - 0.9² = 0.19 → divide by 0.19 (5.3× correction)
t=10: 1 - 0.9¹⁰ = 0.65 → divide by 0.65 (1.5× correction)
t→∞: 1 - 0.9^t → 1 → no correction needed
\`\`\`

Ensures unbiased estimates from the start!

**3. Why Adam is Default:**

**A) Robust to Hyperparameters:**

Works well with default values:
\`\`\`python
# Usually just tune learning rate
optimizer = Adam(lr=0.001)  # Often works!

# Compare to SGD:
optimizer = SGD(lr=???, momentum=???)  # Need to tune both
\`\`\`

**B) Handles Sparse Gradients:**

Adaptive LR helps with sparse features (NLP, recommender systems):
\`\`\`
# Word embeddings: most gradients are zero
# Adam: Large LR for rare words (small v_t)
# SGD: Same LR for all → slow learning for rare words
\`\`\`

**C) Fast Convergence:**

Combines benefits of momentum + adaptive LR:
\`\`\`
Typical training:
- Adam: 50 epochs to converge
- SGD+momentum: 150 epochs
- Plain SGD: 500+ epochs
\`\`\`

**D) Less Sensitive to Learning Rate:**

Works across wide range of α:
\`\`\`
SGD: α too large → divergence, α too small → slow
Adam: Adaptive per-parameter → more forgiving
\`\`\`

**4. Intuitive Understanding:**

**Analogy:** Driving a car with adaptive cruise control

**SGD:**
- Fixed speed (learning rate)
- Hits bumps (noisy gradients) → jerky ride

**SGD + Momentum:**
- Smooth acceleration/deceleration
- Still fixed target speed

**Adam:**
- Smooth acceleration (momentum)
- Adaptive speed per terrain (adaptive LR)
- Speed up on highway (consistent gradient)
- Slow down on bumpy roads (high variance)

**5. Practical Implementation:**

\`\`\`python
class Adam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        # Initialize moments
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0
    
    def step(self, grads):
        self.t += 1
        
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            # Update moments
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            
            # Update parameters
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
\`\`\`

**6. When Adam Excels:**

**Best for:**
- Deep neural networks
- Sparse data (NLP, recommendations)
- First try / prototyping
- Limited hyperparameter tuning time

**Example use cases:**
- Transformers (BERT, GPT)
- Image classification (ResNet, ViT)
- GANs
- Reinforcement learning

**7. Limitations:**

**A) Generalization:**

Sometimes SGD+momentum generalizes better:
\`\`\`
Adam: Fast convergence, potentially sharp minima
SGD: Slow convergence, flatter minima → better test accuracy
\`\`\`

**Solution:** Switch to SGD for final epochs

**B) Learning Rate Decay:**

Adam less sensitive but still benefits:
\`\`\`python
# Cosine annealing
lr_t = lr_initial * 0.5 * (1 + cos(π * t / T))
\`\`\`

**C) Weight Decay:**

Original Adam has issues with weight decay. Use AdamW:
\`\`\`python
# Adam: weight_decay applies to gradients
# AdamW: weight_decay applies to weights directly
optimizer = AdamW(lr=0.001, weight_decay=0.01)
\`\`\`

**8. Comparison Table:**

| Optimizer | Speed | Robustness | Generalization | Use Case |
|-----------|-------|------------|----------------|----------|
| **SGD** | Slow | Low | Best | Simple problems, final tuning |
| **SGD+Mom** | Medium | Medium | Good | CNNs, careful tuning |
| **Adam** | **Fast** | **High** | Good | **Default choice** |
| **AdamW** | Fast | High | **Better** | **Modern best practice** |

**9. Modern Variants:**

**AdamW:** Decoupled weight decay (current best practice)
**RAdam:** Rectified Adam (fixes early training issues)
**Lookahead:** Wraps Adam for better generalization
**LAMB:** Large batch training

**10. Summary:**

**Why Adam is default:**

1. **Fast convergence**: Momentum + adaptive LR
2. **Robust**: Works with default hyperparameters
3. **Adaptive**: Per-parameter learning rates
4. **Handles sparsity**: Great for NLP, recommendations
5. **Easy to use**: Usually just tune learning rate
6. **Empirically successful**: Powers modern deep learning

**Best practice:**
\`\`\`python
# Start with Adam/AdamW
optimizer = AdamW(lr=0.001, weight_decay=0.01)

# If performance critical, try:
# 1. Learning rate schedule
# 2. Warmup (linear increase for first N steps)
# 3. Switch to SGD for final epochs (better generalization)
\`\`\`

**Key insight:** Adam's combination of momentum and adaptive learning rates makes it remarkably robust across diverse architectures and datasets, explaining its status as the default optimizer in modern deep learning.`,
          keyPoints: [
            'Combines momentum (first moment) with adaptive LR (second moment)',
            'Bias correction ensures unbiased estimates early in training',
            'Per-parameter learning rates adapt to gradient statistics',
            'Robust to hyperparameters, works with defaults',
            'Handles sparse gradients well (NLP, recommendations)',
            'Default choice for modern deep learning',
          ],
        },
        {
          id: 'numopt-disc-3',
          question:
            "Compare and contrast first-order methods (gradient descent) with second-order methods (Newton's method) for optimization. Why aren't second-order methods used more in deep learning?",
          hint: 'Consider computational cost, convergence rate, memory requirements, and scalability.',
          sampleAnswer: `**First-Order vs Second-Order Optimization Methods**

Understanding the trade-offs between these method families is crucial for choosing the right optimization approach.

**1. Mathematical Foundation:**

**First-Order (Gradient Descent):**
\`\`\`
Uses: f(x), ∇f(x)
Update: x_{k+1} = x_k - α∇f(x_k)
Information: Direction of steepest descent
\`\`\`

**Second-Order (Newton's Method):**
\`\`\`
Uses: f(x), ∇f(x), ∇²f(x)
Update: x_{k+1} = x_k - [∇²f(x_k)]⁻¹∇f(x_k)
Information: Direction + curvature
\`\`\`

**2. Convergence Analysis:**

**Gradient Descent:**
- **Rate**: O(1/k) for convex, O(exp(-k)) for strongly convex
- **Dependency**: Condition number κ = λ_max/λ_min affects speed
- **Ill-conditioned**: Very slow (zigzagging)

**Newton's Method:**
- **Rate**: O(exp(-2^k)) (quadratic convergence)
- **Near optimum**: Doubling precision each iteration!
- **Independent**: Doesn't depend on condition number

**Example:**
\`\`\`
Target accuracy: ε = 10⁻⁸

GD: ~10⁸ iterations (if κ = 10⁴)
Newton: ~5 iterations (near optimum)
\`\`\`

**3. Computational Cost:**

**Per Iteration:**

**Gradient Descent:**
\`\`\`
- Gradient computation: O(n·d)
  where n = data size, d = parameters
- Memory: O(d)
- Total: O(n·d)
\`\`\`

**Newton's Method:**
\`\`\`
- Gradient: O(n·d)
- Hessian: O(n·d²)  ← expensive!
- Inversion: O(d³)   ← very expensive!
- Memory: O(d²)      ← huge!
- Total: O(n·d² + d³)
\`\`\`

**Deep learning example:**
\`\`\`
ResNet-50: d = 25 million parameters

Gradient: 25M memory
Hessian: 25M × 25M = 625 trillion entries!
         ≈ 5 petabytes (in float64)!
         
Inversion: (25M)³ operations ≈ impossible!
\`\`\`

**4. Practical Comparison:**

**For small problems (d < 1000):**

\`\`\`python
import numpy as np
from time import time

d = 100  # Small dimension

# Gradient descent
def gd_iteration(x, grad):
    return 0.01 * grad  # O(d)

# Newton iteration  
def newton_iteration(x, grad, hessian):
    return np.linalg.solve(hessian, grad)  # O(d³)

grad = np.random.randn(d)
hessian = np.random.randn(d, d)

# Time comparison
t0 = time()
for _ in range(1000):
    update = gd_iteration(x, grad)
gd_time = time() - t0

t0 = time()
for _ in range(1000):
    update = newton_iteration(x, grad, hessian)
newton_time = time() - t0

print(f"GD: {gd_time:.3f}s, Newton: {newton_time:.3f}s")
print(f"Newton {newton_time/gd_time:.1f}× slower per iteration")
\`\`\`

Output:
\`\`\`
GD: 0.002s, Newton: 0.050s
Newton 25× slower per iteration
\`\`\`

**5. Why Second-Order Methods Not Used in DL:**

**A) Computational Intractability:**

\`\`\`
Modern models:
- GPT-3: 175 billion parameters
- Hessian: 175B × 175B matrix
- Storage: 30 septillion entries!
- Physically impossible to store
\`\`\`

**B) Memory Requirements:**

\`\`\`
Even modest network:
- Parameters: 1M
- Gradient: 4 MB (float32)
- Hessian: 4 TB (!!)
- GPU memory: typically 12-80 GB
- → Doesn't fit!
\`\`\`

**C) Non-Convexity:**

Neural networks: highly non-convex
- Newton's method: designed for convex optimization
- Saddle points: Newton can converge to saddles
- Negative curvature: Hessian not positive definite
- Catastrophic steps possible

**D) Stochasticity:**

Deep learning uses mini-batches:
- Hessian estimation: extremely noisy
- Need large batches for accurate Hessian
- → Defeats purpose of stochasticity

**6. Quasi-Newton Methods:**

Attempt to get benefits without full cost:

**BFGS (Broyden-Fletcher-Goldfarb-Shanno):**
\`\`\`
- Approximate Hessian inverse from gradients
- O(d²) memory (still too much for DL)
- Effective for d < 10,000
\`\`\`

**L-BFGS (Limited-memory BFGS):**
\`\`\`
- Store only m recent gradient pairs (m ≈ 10)
- O(m·d) memory ← much better!
- Used for some ML applications
\`\`\`

**Why L-BFGS not standard in DL:**
- Still expensive for very large d
- Doesn't leverage GPU parallelism well
- Not compatible with mini-batch SGD
- Works better for convex problems

**7. Approximations in Deep Learning:**

**A) Diagonal Hessian (AdaGrad, RMSProp, Adam):**

Instead of full Hessian:
\`\`\`
Hessian: O(d²) entries
Diagonal: O(d) entries ← feasible!

Adam uses diagonal second moment:
v_t ≈ diag(Hessian)
\`\`\`

**Trade-off:** Fast but ignores correlations

**B) K-FAC (Kronecker-Factored Approximate Curvature):**

Exploit structure in neural networks:
\`\`\`
Hessian ≈ A ⊗ B
where A and B are much smaller matrices

Cost: O(d^{1.5}) instead of O(d³)
\`\`\`

Still not mainstream due to complexity.

**C) Natural Gradient:**

Uses Fisher information matrix:
\`\`\`
Update: θ_{t+1} = θ_t - α F⁻¹∇L
where F = E[∇log p · ∇log p^T]
\`\`\`

More practical than Hessian, but still expensive.

**8. When to Use Each:**

**Use First-Order (GD, Adam):**
- Large-scale deep learning (d > 10⁶)
- Stochastic mini-batch training
- GPU acceleration important
- Non-convex optimization
- **→ 99% of deep learning**

**Use Second-Order (Newton, L-BFGS):**
- Small to medium scale (d < 10⁴)
- Convex or nearly convex
- Deterministic (full-batch)
- High precision required
- **→ Traditional ML, scientific computing**

**9. Comparison Table:**

| Aspect | First-Order | Second-Order |
|--------|-------------|--------------|
| **Cost/iter** | O(d) | O(d³) |
| **Memory** | O(d) | O(d²) |
| **Convergence** | Linear | Quadratic |
| **Iterations** | Many | Few |
| **Total cost** | **Often lower** | Often higher |
| **Scalability** | ✓ Millions of params | ✗ Thousands max |
| **GPU-friendly** | ✓ Yes | ✗ No |
| **Stochastic** | ✓ Yes | ✗ Difficult |
| **Non-convex** | ✓ Robust | ⚠ Can fail |

**10. The Paradox:**

**Newton's method:**
- Fewer iterations (5-10)
- But each iteration extremely expensive
- Total time: Usually worse for large d

**Gradient descent:**
- Many iterations (1000s)
- But each iteration cheap
- Total time: Usually better for large d

**Example:**
\`\`\`
Problem: d = 1M parameters

Newton:
- 10 iterations × 10 hours/iter = 100 hours

GD/Adam:
- 10,000 iterations × 0.01 hours/iter = 100 hours

But Newton requires 5 PB memory → impossible!
GD requires 4 MB → feasible!
\`\`\`

**11. Future Directions:**

**Hybrid approaches:**
- Start with first-order (fast initial progress)
- Switch to second-order near optimum (fast convergence)

**Randomized methods:**
- Stochastic Hessian estimation
- Sub-sampled Newton methods

**Hardware:**
- TPUs optimized for matrix operations
- Could make second-order more viable

**12. Summary:**

**Why first-order dominates deep learning:**

1. **Scalability**: O(d) vs O(d³) - critical for millions/billions of parameters
2. **Memory**: O(d) vs O(d²) - Hessian doesn't fit in memory
3. **Stochasticity**: Compatible with mini-batch SGD
4. **GPU-friendly**: Highly parallelizable operations
5. **Non-convexity**: More robust to saddle points
6. **Empirical success**: Proven to work at scale

**When second-order wins:**
- Small, convex problems
- High-precision requirements
- Scientific computing

**Key insight:** In deep learning, the ability to take many cheap steps (first-order) outweighs the benefit of taking few expensive steps (second-order). The curse of dimensionality makes second-order methods computationally intractable for modern neural networks.`,
          keyPoints: [
            'First-order: O(d) per iteration, many iterations needed',
            'Second-order: O(d³) per iteration, few iterations needed',
            'Deep learning: d = millions → Hessian storage impossible',
            'Mini-batch SGD incompatible with accurate Hessian estimation',
            'First-order + adaptive LR (Adam) good enough in practice',
            'Second-order reserved for small-scale convex optimization',
          ],
        },
      ],
    },
    {
      id: 'stochastic-calculus',
      title: 'Stochastic Calculus Fundamentals',
      content: `
# Stochastic Calculus Fundamentals

## Introduction

Stochastic calculus extends calculus to random processes. In machine learning:
- **Stochastic optimization**: SGD, Langevin dynamics
- **Generative models**: Diffusion models, score matching
- **Reinforcement learning**: Continuous-time control
- **Quantitative finance**: Options pricing, risk models

## Brownian Motion

**Brownian motion** (Wiener process) W_t is a continuous-time random process:

**Properties:**
1. W_0 = 0
2. Independent increments
3. W_t - W_s ~ N(0, t-s) for t > s
4. Continuous paths

**Intuition:** Limit of random walk as step size → 0.

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt

def brownian_motion(T=1.0, N=1000, n_paths=5):
    """
    Simulate Brownian motion paths
    
    Args:
        T: total time
        N: number of time steps
        n_paths: number of paths to simulate
    
    Returns:
        t: time points
        W: Brownian motion paths (n_paths × N+1)
    """
    dt = T / N
    t = np.linspace(0, T, N + 1)
    
    # Generate increments: dW ~ N(0, √dt)
    dW = np.random.randn(n_paths, N) * np.sqrt(dt)
    
    # Cumulative sum to get paths
    W = np.column_stack([np.zeros(n_paths), np.cumsum(dW, axis=1)])
    
    return t, W

# Simulate and plot
t, W = brownian_motion(T=1.0, N=1000, n_paths=10)

plt.figure(figsize=(12, 4))

# Multiple paths
plt.subplot(131)
for i in range(10):
    plt.plot(t, W[i], alpha=0.7)
plt.xlabel('Time t')
plt.ylabel('W_t')
plt.title('Brownian Motion Paths')
plt.grid(True, alpha=0.3)

# Distribution at t=1
plt.subplot(132)
plt.hist(W[:, -1], bins=30, density=True, alpha=0.7, label='Simulated')
x = np.linspace(-3, 3, 100)
plt.plot(x, (1/np.sqrt(2*np.pi)) * np.exp(-x**2/2), 'r-', linewidth=2, label='N(0,1)')
plt.xlabel('W_1')
plt.ylabel('Density')
plt.title('Distribution at t=1')
plt.legend()

# Increments distribution
plt.subplot(133)
increments = np.diff(W[0]) / np.sqrt(t[1]-t[0])
plt.hist(increments, bins=30, density=True, alpha=0.7, label='Increments')
plt.plot(x, (1/np.sqrt(2*np.pi)) * np.exp(-x**2/2), 'r-', linewidth=2, label='N(0,1)')
plt.xlabel('dW_t / √dt')
plt.ylabel('Density')
plt.title('Scaled Increments')
plt.legend()

plt.tight_layout()
plt.savefig('brownian_motion.png', dpi=150, bbox_inches='tight')
print("Brownian motion has:")
print(f"  Mean at t=1: {np.mean(W[:, -1]):.4f} (expected: 0)")
print(f"  Variance at t=1: {np.var(W[:, -1]):.4f} (expected: 1)")
\`\`\`

## Stochastic Differential Equations (SDEs)

**General form:**
dX_t = μ(X_t, t)dt + σ(X_t, t)dW_t

where:
- μ(X_t, t): drift (deterministic part)
- σ(X_t, t): diffusion (random part)
- dW_t: Brownian increment

**Example:** Geometric Brownian Motion
dS_t = μS_t dt + σS_t dW_t

Models stock prices, exponential growth with noise.

\`\`\`python
def euler_maruyama(mu, sigma, X0, T=1.0, N=1000, n_paths=1):
    """
    Euler-Maruyama method for SDEs
    
    Solves: dX_t = μ(X_t, t)dt + σ(X_t, t)dW_t
    
    Args:
        mu: drift function μ(X, t)
        sigma: diffusion function σ(X, t)
        X0: initial value
        T: final time
        N: number of steps
        n_paths: number of paths
    
    Returns:
        t: time points
        X: solution paths
    """
    dt = T / N
    t = np.linspace(0, T, N + 1)
    
    X = np.zeros((n_paths, N + 1))
    X[:, 0] = X0
    
    for i in range(N):
        dW = np.random.randn(n_paths) * np.sqrt(dt)
        X[:, i+1] = X[:, i] + mu(X[:, i], t[i]) * dt + sigma(X[:, i], t[i]) * dW
    
    return t, X

# Geometric Brownian Motion: dS_t = 0.1·S_t·dt + 0.2·S_t·dW_t
def mu_gbm(S, t):
    return 0.1 * S  # 10% drift

def sigma_gbm(S, t):
    return 0.2 * S  # 20% volatility

S0 = 100  # Initial stock price
t, S = euler_maruyama(mu_gbm, sigma_gbm, S0, T=1.0, N=1000, n_paths=100)

plt.figure(figsize=(10, 4))
plt.subplot(121)
for i in range(min(20, len(S))):
    plt.plot(t, S[i], alpha=0.5)
plt.xlabel('Time (years)')
plt.ylabel('Stock Price')
plt.title('Geometric Brownian Motion Paths')
plt.grid(True, alpha=0.3)

plt.subplot(122)
plt.hist(S[:, -1], bins=30, density=True, alpha=0.7)
plt.xlabel('Final Price S_T')
plt.ylabel('Density')
plt.title(f'Distribution at T=1 (S_0={S0})')
plt.axvline(S0 * np.exp(0.1), color='r', linestyle='--', label=f'Expected: {S0 * np.exp(0.1):.1f}')
plt.legend()

plt.tight_layout()
plt.savefig('geometric_brownian_motion.png', dpi=150, bbox_inches='tight')

print("Geometric Brownian Motion:")
print(f"  Initial price: {S0}")
print(f"  Mean final price: {np.mean(S[:, -1]):.2f}")
print(f"  Theoretical mean: {S0 * np.exp(0.1):.2f}")
\`\`\`

## Itô's Lemma

**Chain rule for stochastic processes**: If X_t solves an SDE and f(X_t, t) is twice differentiable:

df(X_t, t) = [∂f/∂t + μ·∂f/∂x + (1/2)σ²·∂²f/∂x²]dt + σ·∂f/∂x·dW_t

**Key difference from standard calculus:** Extra term (1/2)σ²·∂²f/∂x²

**Intuition:** Quadratic variation of Brownian motion: (dW_t)² = dt

\`\`\`python
def ito_lemma_example():
    """
    Verify Itô's lemma numerically
    
    X_t: Geometric Brownian Motion
    f(X_t) = log(X_t)
    
    By Itô's lemma:
    d(log X_t) = (μ - σ²/2)dt + σ dW_t
    
    This is Brownian motion with drift!
    """
    
    # Parameters
    mu, sigma = 0.1, 0.2
    X0 = 100
    T, N = 1.0, 10000
    
    # Simulate GBM
    def mu_func(X, t):
        return mu * X
    def sigma_func(X, t):
        return sigma * X
    
    t, X = euler_maruyama(mu_func, sigma_func, X0, T, N, n_paths=10000)
    
    # Transform: Y_t = log(X_t)
    Y = np.log(X)
    
    # By Itô's lemma: dY_t = (μ - σ²/2)dt + σ dW_t
    # So Y_T - Y_0 ~ N((μ - σ²/2)T, σ²T)
    
    Y_final = Y[:, -1] - Y[:, 0]
    
    theoretical_mean = (mu - 0.5*sigma**2) * T
    theoretical_std = sigma * np.sqrt(T)
    
    print("Itô's Lemma Verification:")
    print(f"  Simulated mean: {np.mean(Y_final):.6f}")
    print(f"  Theoretical mean: {theoretical_mean:.6f}")
    print(f"  Simulated std: {np.std(Y_final):.6f}")
    print(f"  Theoretical std: {theoretical_std:.6f}")
    print("→ Itô's lemma correctly predicts log-transform distribution!")

ito_lemma_example()
\`\`\`

## Applications in ML

### Langevin Dynamics

**Idea:** Add noise to gradient descent for better exploration.

**Stochastic Gradient Langevin Dynamics (SGLD):**
x_{k+1} = x_k - α∇f(x_k) + √(2α/β)·ξ_k

where ξ_k ~ N(0, I), β is inverse temperature.

**Continuous-time limit:**
dX_t = -∇f(X_t)dt + √(2/β)dW_t

**Stationary distribution:** p(x) ∝ exp(-βf(x))

\`\`\`python
def sgld(grad_f, x0, alpha=0.01, beta=1.0, n_iter=1000):
    """
    Stochastic Gradient Langevin Dynamics
    
    Samples from p(x) ∝ exp(-β·f(x))
    """
    x = x0.copy()
    samples = [x.copy()]
    
    for k in range(n_iter):
        grad = grad_f(x)
        noise = np.random.randn(*x.shape) * np.sqrt(2 * alpha / beta)
        x = x - alpha * grad + noise
        samples.append(x.copy())
    
    return np.array(samples)

# Target: sample from bimodal distribution
def f(x):
    """Double-well potential"""
    return (x[0]**2 - 1)**2 + x[1]**2

def grad_f(x):
    return np.array([
        4*x[0]*(x[0]**2 - 1),
        2*x[1]
    ])

x0 = np.array([0.0, 0.0])
samples = sgld(grad_f, x0, alpha=0.01, beta=1.0, n_iter=10000)

plt.figure(figsize=(12, 4))

# Trajectory
plt.subplot(131)
plt.plot(samples[:, 0], samples[:, 1], 'b-', alpha=0.3)
plt.plot(samples[0, 0], samples[0, 1], 'go', markersize=10, label='Start')
plt.xlabel('x₁')
plt.ylabel('x₂')
plt.title('SGLD Trajectory')
plt.legend()
plt.grid(True, alpha=0.3)

# Marginal distribution (x₁)
plt.subplot(132)
plt.hist(samples[1000:, 0], bins=50, density=True, alpha=0.7)
plt.xlabel('x₁')
plt.ylabel('Density')
plt.title('Sampled Distribution (x₁)')
plt.axvline(-1, color='r', linestyle='--', label='Modes')
plt.axvline(1, color='r', linestyle='--')
plt.legend()

# Both modes
plt.subplot(133)
plt.hist2d(samples[1000:, 0], samples[1000:, 1], bins=50, cmap='Blues')
plt.xlabel('x₁')
plt.ylabel('x₂')
plt.title('2D Density (SGLD samples)')
plt.colorbar()

plt.tight_layout()
plt.savefig('sgld.png', dpi=150, bbox_inches='tight')
print("SGLD explores both modes of bimodal distribution!")
print("→ Noise helps escape local minima")
\`\`\`

### Diffusion Models

**Forward process:** Add noise gradually
x_t = √(α_t)x_0 + √(1-α_t)ε, ε ~ N(0, I)

**Reverse process:** Learn to denoise
dx_t = [f(x_t, t) - g²(t)∇_x log p_t(x_t)]dt + g(t)dW̄_t

where ∇_x log p_t(x_t) is the **score function** (learned by neural network).

\`\`\`python
def simple_diffusion_demo():
    """
    Demonstrate diffusion process concept
    """
    
    # Original data: samples from mixture of Gaussians
    np.random.seed(42)
    n_samples = 1000
    
    # Two clusters
    X = np.vstack([
        np.random.randn(n_samples//2, 2) + [-2, 0],
        np.random.randn(n_samples//2, 2) + [2, 0]
    ])
    
    # Forward diffusion: gradually add noise
    T_steps = 5
    noise_schedule = np.linspace(0, 1, T_steps)**2
    
    fig, axes = plt.subplots(1, T_steps, figsize=(15, 3))
    
    for i, noise_level in enumerate(noise_schedule):
        # Add noise: x_t = √(1-β_t)x_0 + √β_t·ε
        X_noisy = np.sqrt(1 - noise_level) * X + np.sqrt(noise_level) * np.random.randn(*X.shape)
        
        axes[i].scatter(X_noisy[:, 0], X_noisy[:, 1], alpha=0.5, s=1)
        axes[i].set_xlim(-5, 5)
        axes[i].set_ylim(-5, 5)
        axes[i].set_title(f't={i}/{T_steps-1}')
        axes[i].set_aspect('equal')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('diffusion_forward.png', dpi=150, bbox_inches='tight')
    print("Diffusion Models:")
    print("  Forward: Gradually add noise (data → noise)")
    print("  Reverse: Learn to denoise (noise → data)")
    print("  → Generate new samples by running reverse process")

simple_diffusion_demo()
\`\`\`

## Summary

**Key Concepts**:
- **Brownian motion**: Continuous-time random walk, W_t ~ N(0, t)
- **SDEs**: dX_t = μ dt + σ dW_t (drift + diffusion)
- **Itô's lemma**: Chain rule with extra (1/2)σ² term
- **Euler-Maruyama**: Numerical method for simulating SDEs

**ML Applications**:
- **SGLD**: Stochastic optimization with noise for exploration
- **Diffusion models**: State-of-art generative models (DALL-E 2, Stable Diffusion)
- **Score matching**: Learn gradient of log-density
- **Continuous normalizing flows**: ODEs/SDEs for generative modeling

**Why This Matters**:
Stochastic calculus provides the mathematical foundation for understanding and developing:
- Noisy optimization algorithms (why SGD works)
- Modern generative models (diffusion models)
- Exploration-exploitation trade-offs (reinforcement learning)
- Uncertainty quantification (Bayesian methods)
`,
      multipleChoice: [
        {
          id: 'stoch-1',
          question:
            'Brownian motion W_t has the property that W_t - W_s follows:',
          options: [
            'Uniform distribution',
            'Normal distribution N(0, t-s)',
            'Exponential distribution',
            'Poisson distribution',
          ],
          correctAnswer: 1,
          explanation:
            'Brownian motion increments W_t - W_s are normally distributed with mean 0 and variance t-s.',
          difficulty: 'medium',
        },
        {
          id: 'stoch-2',
          question: "Itô's lemma differs from the standard chain rule by:",
          options: [
            'Having no difference',
            'Including an extra term (1/2)σ²·∂²f/∂x² from quadratic variation',
            'Only applying to linear functions',
            'Not requiring derivatives',
          ],
          correctAnswer: 1,
          explanation:
            "Itô's lemma includes an additional second-order term (1/2)σ²·∂²f/∂x² because Brownian motion has non-zero quadratic variation: (dW_t)² = dt.",
          difficulty: 'hard',
        },
        {
          id: 'stoch-3',
          question:
            'Stochastic Gradient Langevin Dynamics (SGLD) adds noise to gradient descent to:',
          options: [
            'Make optimization slower',
            'Enable exploration and sampling from the posterior distribution',
            'Increase computational cost',
            'Remove the need for gradients',
          ],
          correctAnswer: 1,
          explanation:
            'SGLD adds carefully calibrated noise to enable exploration of the optimization landscape and asymptotically samples from the posterior distribution p(x) ∝ exp(-f(x)).',
          difficulty: 'medium',
        },
        {
          id: 'stoch-4',
          question: 'In a diffusion model, the forward process:',
          options: [
            'Removes noise from data',
            'Gradually adds noise to transform data into pure noise',
            'Trains the neural network',
            'Generates new samples',
          ],
          correctAnswer: 1,
          explanation:
            'The forward diffusion process gradually adds Gaussian noise to the data until it becomes indistinguishable from pure noise. The reverse process (learned) then denoises to generate samples.',
          difficulty: 'medium',
        },
        {
          id: 'stoch-5',
          question: 'The Euler-Maruyama method is:',
          options: [
            'An analytical solution method',
            'A numerical scheme for simulating stochastic differential equations',
            'Only for deterministic ODEs',
            'A deep learning architecture',
          ],
          correctAnswer: 1,
          explanation:
            'Euler-Maruyama is a numerical method for simulating SDEs, discretizing dX_t = μ dt + σ dW_t into finite time steps with Gaussian increments.',
          difficulty: 'easy',
        },
      ],
      quiz: [
        {
          id: 'stoch-disc-1',
          question:
            'Explain how diffusion models work and why they have become state-of-the-art for image generation. What role does stochastic calculus play?',
          hint: 'Consider the forward and reverse diffusion processes, score matching, and how SDEs enable controllable generation.',
          sampleAnswer: `**Diffusion Models for Image Generation:**

Diffusion models (DALL-E 2, Stable Diffusion, Imagen) are currently state-of-the-art for high-quality image generation. Their success relies heavily on stochastic calculus.

**1. Core Idea:**

**Two processes:**
1. **Forward (diffusion):** Gradually add noise to data until pure noise
2. **Reverse (denoising):** Learn to remove noise, generate samples

**Analogy:** 
- Forward: Drop ink in water, watch it diffuse (data → noise)
- Reverse: "Un-diffuse" the ink (noise → data)

**2. Forward Process (Diffusion):**

**Mathematical formulation:**

Start with data x_0 ~ p_data(x)

Add noise in T steps:
\`\`\`
q(x_t | x_{t-1}) = N(x_t | √(1-β_t)x_{t-1}, β_t I)
\`\`\`

where β_t is noise schedule (typically β_1 < β_2 < ... < β_T).

**Convenient property:** Can sample x_t directly from x_0:
\`\`\`
x_t = √(ᾱ_t)x_0 + √(1-ᾱ_t)ε

where ᾱ_t = ∏_{s=1}^t (1-β_s), ε ~ N(0,I)
\`\`\`

**End result:** x_T ≈ N(0, I) (pure noise)

**3. Reverse Process (Denoising):**

**Goal:** Learn p_θ(x_{t-1} | x_t) to reverse the diffusion.

**Challenge:** True reverse process is intractable.

**Solution:** Approximate with neural network:
\`\`\`
p_θ(x_{t-1} | x_t) = N(x_{t-1} | μ_θ(x_t, t), Σ_θ(x_t, t))
\`\`\`

**Training objective:**
\`\`\`
L = E_{t, x_0, ε} [||ε - ε_θ(x_t, t)||²]
\`\`\`

Learn to predict the noise ε that was added!

**4. Stochastic Calculus Connection:**

**Forward SDE:**
\`\`\`
dx = f(x,t)dt + g(t)dW_t
\`\`\`

For diffusion models:
\`\`\`
dx_t = -½β(t)x_t dt + √β(t) dW_t
\`\`\`

**Reverse SDE (Anderson, 1982):**
\`\`\`
dx_t = [f(x,t) - g²(t)∇_x log p_t(x_t)]dt + g(t)d W̄_t
\`\`\`

where ∇_x log p_t(x) is the **score function**.

**Key insight:** If we know the score ∇_x log p_t(x), we can run reverse process!

**5. Score Matching:**

**Neural network learns the score:**
\`\`\`
s_θ(x_t, t) ≈ ∇_x log p_t(x_t)
\`\`\`

**Training:**
\`\`\`
L = E_{t, x_0, ε} [||∇_x log p_t(x_t) - s_θ(x_t, t)||²]
\`\`\`

**Equivalently (denoising score matching):**
\`\`\`
L = E_{t, x_0, ε} [||ε - ε_θ(√(ᾱ_t)x_0 + √(1-ᾱ_t)ε, t)||²]
\`\`\`

Predict the noise added!

**6. Generation Process:**

**Sampling:**
1. Start with x_T ~ N(0, I) (random noise)
2. For t = T, T-1, ..., 1:
   - Predict ε_θ(x_t, t)
   - Compute x_{t-1} using reverse process
   - Add noise (except at t=1)
3. Return x_0 (generated image)

**Pseudocode:**
\`\`\`python
def generate_sample(model, T=1000):
    # Start from noise
    x = torch.randn(1, 3, 256, 256)
    
    for t in reversed(range(T)):
        # Predict noise
        eps_pred = model(x, t)
        
        # Compute mean
        alpha_t = get_alpha(t)
        alpha_prev = get_alpha(t-1)
        x_prev_mean = (x - eps_pred * (1-alpha_t)/sqrt(1-alpha_bar_t)) / sqrt(alpha_t)
        
        # Add noise (except final step)
        if t > 0:
            noise = torch.randn_like(x)
            x = x_prev_mean + sqrt(beta_t) * noise
        else:
            x = x_prev_mean
    
    return x  # Generated image!
\`\`\`

**7. Why Diffusion Models Excel:**

**A) High Quality:**
- Stable training (no mode collapse like GANs)
- Covers full data distribution
- State-of-the-art FID scores

**B) Theoretical Grounding:**
- Based on well-understood stochastic processes
- Convergence guarantees (under assumptions)
- Interpretable generation process

**C) Flexibility:**
- Can condition on text, images, etc.
- Controllable generation (classifier guidance)
- Inpainting, editing, super-resolution

**D) Scalability:**
- Parallelizable (unlike autoregressive models)
- Works with latent spaces (Stable Diffusion)

**8. Key Innovations:**

**DDPM (2020):**
- Denoising diffusion probabilistic models
- Simple training objective (predict noise)

**Score-Based Models:**
- Directly model score function
- Continuous-time formulation

**DALL-E 2 (2022):**
- Text-to-image with CLIP guidance
- Diffusion in latent space

**Stable Diffusion:**
- Diffusion in compressed latent space
- Much faster than pixel-space

**9. Mathematics in Action:**

**Forward diffusion (Itô SDE):**
\`\`\`
dx_t = -½β(t)x_t dt + √β(t) dW_t
\`\`\`

This transforms any distribution into Gaussian!

**Reverse process:**
\`\`\`
dx_t = [-½β(t)x_t - β(t)s_θ(x_t, t)]dt + √β(t) d W̄_t
\`\`\`

Running this backward transforms noise into data.

**Itô's lemma** ensures:
- Forward process has known distribution
- Reverse process exists and can be learned

**10. Comparison with Other Generative Models:**

| Model | Quality | Training | Speed | Control |
|-------|---------|----------|-------|---------|
| **GANs** | High | Unstable | Fast | Medium |
| **VAEs** | Medium | Stable | Fast | High |
| **Diffusion** | **Highest** | **Stable** | Slow | **High** |
| **Autoregressive** | High | Stable | Very slow | Medium |

**11. Practical Example (Conceptual):**

\`\`\`python
class DiffusionModel:
    def __init__(self, unet, noise_schedule):
        self.model = unet  # Neural network
        self.beta = noise_schedule  # β_1, ..., β_T
    
    def forward_diffusion(self, x0, t):
        """Add noise: x_t = √(ᾱ_t)x_0 + √(1-ᾱ_t)ε"""
        alpha_bar = self.get_alpha_bar(t)
        eps = torch.randn_like(x0)
        return sqrt(alpha_bar) * x0 + sqrt(1 - alpha_bar) * eps, eps
    
    def loss(self, x0):
        """Denoising score matching loss"""
        t = torch.randint(0, self.T, (len(x0),))
        x_t, eps_true = self.forward_diffusion(x0, t)
        eps_pred = self.model(x_t, t)
        return F.mse_loss(eps_pred, eps_true)
    
    def generate(self, shape):
        """Reverse diffusion sampling"""
        x = torch.randn(shape)
        for t in reversed(range(self.T)):
            x = self.reverse_step(x, t)
        return x
\`\`\`

**12. Why Stochastic Calculus Matters:**

**Without stochastic calculus:**
- No theoretical framework for reverse process
- No understanding of why it works
- No guidance for architecture/training

**With stochastic calculus:**
- Rigorous reverse SDE formula
- Score matching as optimal objective
- Principled noise schedules
- Connections to other methods (score-based, energy-based)

**13. Summary:**

**Diffusion models work by:**
1. Forward: Gradually noise data → pure noise (tractable SDE)
2. Learn: Neural network predicts noise at each step
3. Reverse: Run learned denoising process (reverse SDE)
4. Generate: Start from noise, denoise to create samples

**Stochastic calculus provides:**
- Forward process: Well-defined noising SDE
- Reverse process: Exact formula via score function
- Training: Score matching objective
- Generation: Sampling via reverse SDE

**Why state-of-the-art:**
- Stable training
- High quality
- Flexible control
- Theoretical grounding

**Key insight:** The ability to reverse a stochastic process (via score function) enables high-quality generation. This is a direct application of stochastic calculus to modern AI.`,
          keyPoints: [
            'Forward: Gradually add noise (data → noise via SDE)',
            'Reverse: Learn to denoise (noise → data via reverse SDE)',
            'Score matching: Learn ∇_x log p_t(x) with neural network',
            'Generation: Sample x_T ~ N(0,I), run reverse process',
            'Stochastic calculus: Provides reverse SDE formula',
            'State-of-the-art: Stable training, high quality, controllable',
          ],
        },
        {
          id: 'stoch-disc-2',
          question:
            "Derive and explain Itô's lemma intuitively. Why does it differ from the standard chain rule, and what is its significance in machine learning?",
          hint: 'Consider quadratic variation of Brownian motion, Taylor expansion, and applications to geometric Brownian motion.',
          sampleAnswer: `**Itô's Lemma: The Chain Rule for Stochastic Processes**

Itô's lemma is fundamental to stochastic calculus, extending the chain rule to random processes. Understanding it is key to diffusion models and stochastic optimization.

**1. Standard Chain Rule (Deterministic):**

**Problem:** Given y = f(x(t)), find dy/dt

**Solution:**
\`\`\`
dy/dt = (df/dx) · (dx/dt)
\`\`\`

**Or in differential form:**
\`\`\`
dy = (df/dx) · dx
\`\`\`

**Intuition:** Small change in x causes proportional change in f(x).

**2. Stochastic Setting:**

**Problem:** Given Y_t = f(X_t, t) where X_t solves SDE:
\`\`\`
dX_t = μ(X_t, t)dt + σ(X_t, t)dW_t
\`\`\`

Find dY_t?

**Naive approach (wrong!):**
\`\`\`
dY_t = (∂f/∂t)dt + (∂f/∂x)dX_t   # Incomplete!
\`\`\`

This misses a crucial term!

**3. Itô's Lemma (Correct Form):**

\`\`\`
dY_t = [∂f/∂t + μ·∂f/∂x + (1/2)σ²·∂²f/∂x²]dt + σ·∂f/∂x·dW_t
         ↑                    ↑
    Standard terms      Itô correction term!
\`\`\`

**Key difference:** Extra term (1/2)σ²·∂²f/∂x²

**4. Intuitive Derivation:**

**Taylor expansion of f(X_t, t):**
\`\`\`
df = ∂f/∂t·dt + ∂f/∂x·dX + (1/2)·∂²f/∂x²·(dX)² + ...
\`\`\`

**Deterministic case:**
\`\`\`
dx = μ dt
(dx)² = μ² (dt)² ≈ 0  (second-order infinitesimal)
→ Drop (dX)² term
\`\`\`

**Stochastic case:**
\`\`\`
dX = μ dt + σ dW
(dX)² = (μ dt)² + 2μσ dt dW + σ²(dW)²
       ≈ 0      ≈ 0         ≠ 0!

Key insight: (dW)² = dt  (quadratic variation)
\`\`\`

**Why (dW)² = dt?**

Brownian motion increments:
\`\`\`
dW ~ N(0, dt)
E[(dW)²] = Var(dW) = dt
(dW)² → dt in mean-square sense
\`\`\`

**Therefore:**
\`\`\`
(dX)² = σ²(dW)² = σ² dt   (leading order!)
\`\`\`

Can't neglect this term!

**5. Heuristic "Multiplication Table":**

\`\`\`
       dt        dW
dt     0         0
dW     0         dt
\`\`\`

**Examples:**
\`\`\`
dt · dt = 0
dt · dW = 0
dW · dW = dt  ← Key rule!
\`\`\`

**6. Worked Example: Log-Transform of GBM:**

**Given:** Geometric Brownian Motion
\`\`\`
dS_t = μS_t dt + σS_t dW_t
\`\`\`

**Find:** d(log S_t)

**Apply Itô's lemma with f(S) = log(S):**
\`\`\`
∂f/∂S = 1/S
∂²f/∂S² = -1/S²
\`\`\`

**Itô's lemma:**
\`\`\`
d(log S_t) = (1/S)dS_t + (1/2)(-1/S²)(dS_t)²

Substitute dS_t = μS dt + σS dW:
\`\`\`

**Drift term:**
\`\`\`
(1/S)(μS dt) = μ dt
\`\`\`

**Diffusion term:**
\`\`\`
(1/S)(σS dW) = σ dW
\`\`\`

**Correction term (the magic!):**
\`\`\`
(1/2)(-1/S²)(dS_t)²
= (1/2)(-1/S²)(σS)² dt     [using (dW)² = dt]
= -(1/2)σ² dt
\`\`\`

**Final result:**
\`\`\`
d(log S_t) = (μ - σ²/2)dt + σ dW_t
\`\`\`

**Implication:** log(S_t) is Brownian motion with drift!

**Without Itô correction:** Would get μ dt + σ dW (wrong!)

**7. Verification by Simulation:**

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt

# Parameters
mu, sigma = 0.1, 0.2
S0 = 100
T, N = 1.0, 10000
n_paths = 10000

dt = T / N
t = np.linspace(0, T, N+1)

# Simulate GBM
dW = np.random.randn(n_paths, N) * np.sqrt(dt)
dS = np.zeros((n_paths, N+1))
S = np.zeros((n_paths, N+1))
S[:, 0] = S0

for i in range(N):
    dS[:, i] = mu * S[:, i] * dt + sigma * S[:, i] * dW[:, i]
    S[:, i+1] = S[:, i] + dS[:, i]

# Compute log(S_T) - log(S_0)
log_change = np.log(S[:, -1]) - np.log(S[:, 0])

# Itô's lemma prediction
theoretical_mean = (mu - 0.5*sigma**2) * T
theoretical_std = sigma * np.sqrt(T)

print("Itô's Lemma Verification:")
print(f"Theoretical mean: {theoretical_mean:.6f}")
print(f"Simulated mean:   {np.mean(log_change):.6f}")
print(f"Theoretical std:  {theoretical_std:.6f}")
print(f"Simulated std:    {np.std(log_change):.6f}")
print("→ Perfect agreement!")

# Without Itô correction (WRONG)
wrong_mean = mu * T  # Missing -σ²/2 term
print(f"\\nWithout Itô correction: {wrong_mean:.6f}")
print(f"Error: {abs(wrong_mean - np.mean(log_change)):.6f}")
print("→ Significant error without Itô term!")
\`\`\`

**Output:**
\`\`\`
Itô's Lemma Verification:
Theoretical mean: 0.080000
Simulated mean:   0.080123
Theoretical std:  0.200000
Simulated std:    0.199876
→ Perfect agreement!

Without Itô correction: 0.100000
Error: 0.019877
→ Significant error without Itô term!
\`\`\`

**8. Why the Correction Term Matters:**

**Economic interpretation (stock prices):**
\`\`\`
μ: Expected return
σ: Volatility

Without Itô: log-return = μT
With Itô: log-return = (μ - σ²/2)T

The -σ²/2 term is "volatility drag"!
\`\`\`

**Example:**
\`\`\`
μ = 10%, σ = 20%, T = 1 year

Naive: Expected log-return = 10%
Itô:   Expected log-return = 10% - (0.2)²/2 = 8%
\`\`\`

Volatility reduces geometric growth!

**9. Applications in Machine Learning:**

**A) Diffusion Models:**

Forward SDE:
\`\`\`
dx_t = f(x,t)dt + g(t)dW_t
\`\`\`

Apply Itô to x^T x (squared norm):
\`\`\`
d(x^T x) = 2x^T dx + (dx)^T dx
         = 2x^T f dt + 2x^T g dW + g² dt
\`\`\`

The g² dt term is crucial for noise schedule!

**B) Stochastic Gradient Langevin Dynamics:**

Continuous limit:
\`\`\`
dX_t = -∇f(X_t)dt + √(2/β) dW_t
\`\`\`

Stationary distribution found using Itô:
\`\`\`
Apply Itô to V(x) = e^{-βf(x)}
→ Proves p(x) ∝ e^{-βf(x)} is stationary
\`\`\`

**C) Score Matching:**

Train neural network s_θ(x,t) ≈ ∇_x log p_t(x)

Itô's lemma relates score to:
\`\`\`
∂log p_t/∂t + ∇·(f p_t) - (1/2)g²∇²p_t = 0
\`\`\`

This is the Fokker-Planck equation!

**10. Summary:**

**Why Itô's lemma differs from chain rule:**

| Aspect | Deterministic | Stochastic |
|--------|---------------|------------|
| **Second-order terms** | dt² ≈ 0 | (dW)² = dt ≠ 0 |
| **Chain rule** | First-order only | Needs second-order |
| **Correction** | None | +(1/2)σ²·∂²f/∂x² |

**Physical intuition:**
- Deterministic: Smooth paths → no second-order effects
- Stochastic: Rough paths → second-order accumulates

**Mathematical reason:**
- Brownian paths: Nowhere differentiable
- Quadratic variation: Non-zero
- Higher moments: Negligible

**Significance for ML:**

1. **Diffusion models:** Forward/reverse processes require Itô
2. **SGLD:** Stationary distribution analysis uses Itô
3. **Score matching:** Training objective derived via Itô
4. **Option pricing:** Log-returns need Itô correction
5. **Volatility modeling:** Risk analysis requires accurate SDE

**Key insight:** Itô's lemma captures the fundamental difference between smooth and rough (stochastic) dynamics. The correction term (1/2)σ²·∂²f/∂x² is not a technicality—it represents the accumulated effect of randomness that standard calculus misses. This is why stochastic calculus is essential for modern generative models and stochastic optimization.`,
          keyPoints: [
            "Itô's lemma: Chain rule + correction term (1/2)σ²·∂²f/∂x²",
            'Correction needed because (dW)² = dt (quadratic variation)',
            'Standard chain rule misses second-order stochastic effects',
            'Crucial for log-transforms (volatility drag)',
            'Enables diffusion models (forward/reverse SDEs)',
            'Foundation for score matching and SGLD',
          ],
        },
        {
          id: 'stoch-disc-3',
          question:
            'Explain the role of Langevin dynamics in Bayesian machine learning and sampling. How does adding noise to gradient descent enable sampling from the posterior?',
          hint: 'Consider stationary distributions, exploration, Metropolis-Hastings connection, and practical SGLD implementation.',
          sampleAnswer: `**Langevin Dynamics for Bayesian Sampling**

Langevin dynamics bridges optimization and sampling, enabling Bayesian inference in deep learning.

**1. The Problem: Bayesian Inference**

**Goal:** Sample from posterior distribution
\`\`\`
p(θ | D) = p(D | θ)p(θ) / p(D)
         ∝ exp[-E(θ)]

where E(θ) = -log p(D|θ) - log p(θ)  (negative log-posterior)
\`\`\`

**Challenge:** Posterior is high-dimensional, complex
- Can't sample directly
- Can evaluate E(θ) and ∇E(θ)

**2. Langevin Dynamics: The Bridge**

**Standard gradient descent (optimization):**
\`\`\`
θ_{k+1} = θ_k - α ∇E(θ_k)
→ Converges to mode (MAP estimate)
\`\`\`

**Langevin dynamics (sampling):**
\`\`\`
θ_{k+1} = θ_k - α ∇E(θ_k) + √(2α) ξ_k,  ξ_k ~ N(0, I)
          ↑                   ↑
      Gradient           Noise
\`\`\`

**Magic:** Adding noise transforms optimization into sampling!

**3. Continuous-Time Formulation:**

**Langevin SDE:**
\`\`\`
dθ_t = -∇E(θ_t)dt + √2 dW_t
\`\`\`

**Key theorem (Stationary distribution):**

As t → ∞, θ_t ~ p(θ) ∝ exp(-E(θ))

**Intuition:**
- Gradient: Pulls toward low energy (high probability)
- Noise: Enables exploration of all high-prob regions
- Balance: Converges to full posterior, not just mode

**4. Why It Works: Fokker-Planck Equation**

**Evolution of density p_t(θ):**
\`\`\`
∂p_t/∂t = ∇·(∇E · p_t) + Δp_t
          ↑               ↑
       Drift          Diffusion
\`\`\`

**At equilibrium (∂p_t/∂t = 0):**
\`\`\`
∇·(∇E · p + ∇p) = 0
\`\`\`

**Solution:** p(θ) ∝ exp(-E(θ))  ← Posterior distribution!

**Proof sketch:**
\`\`\`
Substitute p = exp(-E):
∇p = -exp(-E)∇E

∇·(∇E · p + ∇p)
= ∇·(∇E · exp(-E) - exp(-E)∇E)
= 0  ✓
\`\`\`

**5. Intuitive Understanding:**

**Analogy:** Particle in potential well with friction and noise

**Gradient term (-∇E):**
- Deterministic force toward minima
- Like gravity in bowl

**Noise term (√2 dW):**
- Random kicks
- Like thermal agitation

**Equilibrium:**
- Particle spends time according to Boltzmann: p ∝ exp(-E)
- Deeper regions (low E): More time
- Shallow regions (high E): Less time

**6. Stochastic Gradient Langevin Dynamics (SGLD)**

**Problem:** Computing ∇E(θ) expensive (requires full dataset)

**Solution:** Use mini-batch gradient estimate
\`\`\`
θ_{k+1} = θ_k - α ∇̂E(θ_k) + √(2α) ξ_k

where ∇̂E computed on mini-batch
\`\`\`

**Double noise:**
1. Injected noise: √(2α) ξ_k  (Langevin)
2. Gradient noise: From mini-batch sampling

**Effect:** Mini-batch noise acts as additional exploration!

**Practical SGLD:**
\`\`\`python
def sgld_step(theta, data_batch, lr):
    """
    One SGLD step
    
    Args:
        theta: current parameters
        data_batch: mini-batch of data
        lr: learning rate
    
    Returns:
        theta: updated parameters
    """
    # Compute gradient on mini-batch
    grad = compute_gradient(theta, data_batch)
    
    # Langevin noise
    noise = np.random.randn(*theta.shape) * np.sqrt(2 * lr)
    
    # SGLD update
    theta = theta - lr * grad + noise
    
    return theta

# Bayesian neural network training
def train_bayes_nn(model, data, n_iter=10000):
    theta = initialize_parameters(model)
    samples = []
    
    for i in range(n_iter):
        batch = sample_mini_batch(data)
        theta = sgld_step(theta, batch, lr=0.01)
        
        # Collect samples (after burn-in)
        if i > n_iter // 2:
            samples.append(theta.copy())
    
    return samples  # Posterior samples!
\`\`\`

**7. Why SGLD Works in Practice:**

**A) Exploration:**

Without noise (GD):
\`\`\`
Converges to one mode
Stuck in local minimum
No uncertainty quantification
\`\`\`

With noise (SGLD):
\`\`\`
Explores all modes
Escapes local minima
Samples reflect uncertainty
\`\`\`

**B) Implicit Regularization:**

Noise prevents overfitting:
\`\`\`
# Sharp minimum: Sensitive to noise → unstable → rejected
# Flat minimum: Insensitive to noise → stable → preferred
\`\`\`

SGLD naturally prefers flat minima → better generalization!

**C) Uncertainty Quantification:**

\`\`\`python
# Train with SGLD
posterior_samples = train_bayes_nn(model, data)

# Prediction with uncertainty
def predict_with_uncertainty(x_new, posterior_samples):
    predictions = [model(x_new, theta) for theta in posterior_samples]
    
    mean = np.mean(predictions, axis=0)
    std = np.std(predictions, axis=0)
    
    return mean, std  # Point estimate + uncertainty!

# Example
x_test = ...
y_mean, y_std = predict_with_uncertainty(x_test, posterior_samples)

print(f"Prediction: {y_mean:.2f} ± {y_std:.2f}")
print("95% CI: [{:.2f}, {:.2f}]".format(
    y_mean - 1.96*y_std, 
    y_mean + 1.96*y_std
))
\`\`\`

**8. Connection to MCMC:**

**Metropolis-Hastings:**
- Propose new state
- Accept/reject based on ratio
- Requires acceptance step

**Langevin:**
- Proposal: θ' = θ - α∇E(θ) + √(2α)ξ
- Biased toward low energy
- Would need correction (MALA)

**Unadjusted Langevin:**
- Skip acceptance step
- Small bias if α small
- Much faster (no acceptance)

**SGLD tradeoff:**
- Exact sampling: Requires α → 0 (slow)
- Practical: Use fixed α (fast, slight bias)
- Bias often negligible vs computational gain

**9. Advanced Variants:**

**Preconditioned SGLD:**
\`\`\`
θ_{k+1} = θ_k - α M^{-1} ∇E(θ_k) + √(2α M^{-1}) ξ_k

where M = preconditioning matrix
\`\`\`

Better exploration in ill-conditioned spaces.

**SGHMC (Stochastic Gradient Hamiltonian Monte Carlo):**
\`\`\`
Adds momentum:
v_{k+1} = v_k - α∇E(θ_k) - βv_k + √(2αβ)ξ_k
θ_{k+1} = θ_k + v_{k+1}
\`\`\`

Faster mixing than SGLD.

**10. Practical Example:**

\`\`\`python
import torch
import torch.nn as nn

class BayesianNN:
    def __init__(self, model):
        self.model = model
        self.samples = []
    
    def sgld_train(self, train_loader, n_epochs=100, lr=0.001):
        """Train with SGLD to get posterior samples"""
        
        for epoch in range(n_epochs):
            for batch_x, batch_y in train_loader:
                # Compute gradient
                loss = nn.MSELoss()(self.model(batch_x), batch_y)
                self.model.zero_grad()
                loss.backward()
                
                # SGLD update
                with torch.no_grad():
                    for param in self.model.parameters():
                        # Gradient descent step
                        param -= lr * param.grad
                        
                        # Add Langevin noise
                        noise = torch.randn_like(param) * np.sqrt(2 * lr)
                        param += noise
                
                # Collect samples (after burn-in)
                if epoch > n_epochs // 2:
                    self.samples.append(copy.deepcopy(self.model.state_dict()))
    
    def predict(self, x, return_std=True):
        """Bayesian prediction with uncertainty"""
        predictions = []
        
        for sample in self.samples:
            self.model.load_state_dict(sample)
            with torch.no_grad():
                pred = self.model(x)
            predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        
        if return_std:
            return mean, std
        return mean

# Usage
model = nn.Sequential(nn.Linear(10, 50), nn.ReLU(), nn.Linear(50, 1))
bayes_model = BayesianNN(model)
bayes_model.sgld_train(train_loader)

# Prediction with uncertainty
x_test = torch.randn(1, 10)
y_mean, y_std = bayes_model.predict(x_test)
print(f"Prediction: {y_mean.item():.2f} ± {y_std.item():.2f}")
\`\`\`

**11. When to Use SGLD:**

**Use SGLD when:**
- Need uncertainty quantification
- Want to avoid overfitting (implicit regularization)
- Multi-modal posterior (need full distribution)
- Active learning (guide data collection with uncertainty)
- Safety-critical applications (calibrated confidence)

**Use standard SGD when:**
- Only need point estimate
- Computational budget tight
- Posterior unimodal
- Prediction speed critical

**12. Summary:**

**How SGLD enables Bayesian inference:**

1. **Gradient term:** Guides toward high-probability regions
2. **Noise term:** Enables exploration of full posterior
3. **Stationary distribution:** Converges to p(θ) ∝ exp(-E(θ))
4. **Mini-batches:** Double noise (injected + gradient) aids exploration
5. **Uncertainty:** Posterior samples → predictive distribution

**Key advantages:**

- Scalable to large datasets (mini-batch)
- Scalable to high dimensions (gradient-based)
- No tuning (unlike MCMC proposals)
- Implicit regularization (flat minima)
- Simple implementation (SGD + noise)

**Practical impact:**

SGLD enables:
- Bayesian deep learning at scale
- Uncertainty-aware AI systems
- Out-of-distribution detection
- Continual learning with uncertainty
- Safe reinforcement learning

**Key insight:** Adding noise to optimization doesn't degrade performance—it transforms the algorithm from finding a single solution (MAP) to exploring the full posterior distribution. The gradient guides exploration while noise enables ergodicity, achieving Bayesian inference with the computational cost of a noisy optimizer. This is a profound connection between optimization and sampling enabled by stochastic calculus.`,
          keyPoints: [
            'Langevin dynamics: Gradient descent + noise → samples posterior',
            'Stationary distribution: p(θ) ∝ exp(-E(θ))',
            'SGLD: Uses mini-batch gradients (double noise)',
            'Enables uncertainty quantification in deep learning',
            'Explores multiple modes, escapes local minima',
            'Connection: Optimization (GD) → Sampling (LD) via noise',
          ],
        },
      ],
    },
  ],
  keyTakeaways: [
    'Derivatives measure instantaneous rates of change, fundamental to optimization',
    'Gradient descent follows negative gradient direction to minimize loss functions',
    'Chain rule enables backpropagation in neural networks',
    'Partial derivatives and gradients extend calculus to multivariable functions',
    'Convex optimization has unique global minima, guaranteeing convergence',
    "Newton's method uses second-order information for quadratic convergence",
    'Stochastic gradient descent trades noise for computational efficiency',
    'Integration computes cumulative effects, expectations, and KL divergence',
    'Hessian matrix characterizes curvature and identifies saddle points',
    'KKT conditions provide optimality certificates for constrained optimization',
    'Adam optimizer combines momentum with adaptive learning rates',
    'Stochastic calculus underpins modern generative models like diffusion models',
  ],
  learningObjectives: [
    'Master differential calculus: limits, derivatives, and differentiation rules',
    'Understand gradient-based optimization for machine learning',
    'Apply chain rule to compute gradients in neural networks',
    'Use partial derivatives and gradients for multivariable optimization',
    'Identify and leverage convex functions in optimization problems',
    'Implement numerical optimization algorithms (GD, momentum, Adam, Newton)',
    'Understand integration for computing expectations and information measures',
    'Analyze critical points using Jacobian and Hessian matrices',
    'Apply KKT conditions to constrained optimization problems',
    'Understand stochastic differential equations and their ML applications',
    'Implement calculus-based algorithms in Python (NumPy, SymPy, SciPy)',
    'Connect calculus theory to practical deep learning optimization',
  ],
  prerequisites: ['Module 1: Mathematical Foundations'],
};

export default mlCalculusFundamentals;
