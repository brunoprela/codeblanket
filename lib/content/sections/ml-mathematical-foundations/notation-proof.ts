/**
 * Mathematical Notation & Proof Section
 */

export const notationproofSection = {
  id: 'notation-proof',
  title: 'Mathematical Notation & Proof',
  content: `
# Mathematical Notation & Proof

## Introduction

Mathematical notation provides precise, unambiguous communication. In machine learning, we use notation to define models, losses, and algorithms. Understanding how to read and write mathematical proofs helps us understand why algorithms work and debug when they don't.

## Common Notation

### Variables and Constants

- **Lowercase letters**: variables (x, y, w, b)
- **Uppercase letters**: matrices, random variables (X, Y, W)
- **Greek letters**: parameters (α learning rate, θ parameters, λ regularization)
- **Subscripts**: indices (x₁, x₂, ..., xₙ or xᵢ)
- **Superscripts**: powers (x²) or sample indices (x⁽¹⁾, x⁽²⁾)

\`\`\`python
import numpy as np

# Example: Linear regression notation
# Model: ŷ = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
# Or in vector form: ŷ = wᵀx + b

# Data
X = np.array([[1, 2], [3, 4], [5, 6]])  # Matrix X: (n_samples × n_features)
y = np.array([5, 11, 17])               # Vector y: (n_samples,)
w = np.array([2, 1])                     # Weights w: (n_features,)
b = 1                                    # Bias b: scalar

# Prediction for sample i: ŷ⁽ⁱ⁾ = wᵀx⁽ⁱ⁾ + b
y_pred = X @ w + b

print("Notation example:")
print(f"X shape: {X.shape} (n_samples × n_features)")
print(f"w shape: {w.shape} (n_features,)")
print(f"y_pred: {y_pred}")
print(f"\\nFor sample i=0:")
print(f"x⁽⁰⁾ = {X[0]}")
print(f"ŷ⁽⁰⁾ = w₁x₁ + w₂x₂ + b = {w[0]}×{X[0,0]} + {w[1]}×{X[0,1]} + {b} = {y_pred[0]}")
\`\`\`

### Summation (Σ)

**Σ(i=1 to n) xᵢ**: Sum x₁ + x₂ + ... + xₙ

\`\`\`python
# Example: Mean Squared Error
# MSE = (1/n) Σ(i=1 to n) (yᵢ - ŷᵢ)²

y_true = np.array([5, 11, 17])
y_pred = np.array([4, 10, 18])
n = len(y_true)

# Using summation notation
mse_sum = sum((y_true[i] - y_pred[i])**2 for i in range(n)) / n

# Using vectorized operations
mse_vec = np.mean((y_true - y_pred)**2)

print(f"MSE (summation): {mse_sum}")
print(f"MSE (vectorized): {mse_vec}")
print(f"\\nSummation: Σ(i=1 to {n}) (yᵢ - ŷᵢ)² / {n}")
print(f"Expanded: ({y_true[0]}-{y_pred[0]})² + ({y_true[1]}-{y_pred[1]})² + ({y_true[2]}-{y_pred[2]})² / {n}")
\`\`\`

### Product (Π)

**Π(i=1 to n) xᵢ**: Product x₁ × x₂ × ... × xₙ

\`\`\`python
# Example: Probability of independent events
# P(A₁ ∩ A₂ ∩ ... ∩ Aₙ) = Π P(Aᵢ)

probabilities = np.array([0.9, 0.8, 0.95, 0.85])

# Using product notation
joint_prob = np.prod(probabilities)

print(f"Individual probabilities: {probabilities}")
print(f"Joint probability: Π P(Aᵢ) = {joint_prob:.4f}")
print(f"Expanded: {probabilities[0]} × {probabilities[1]} × {probabilities[2]} × {probabilities[3]} = {joint_prob:.4f}")
\`\`\`

### Set Notation

- **∈**: element of (x ∈ S means x is in set S)
- **∉**: not element of
- **⊆**: subset
- **∪**: union
- **∩**: intersection
- **∅**: empty set

\`\`\`python
# Example: Training/test split notation
# D = {(xᵢ, yᵢ)}ᵢ₌₁ⁿ (dataset of n samples)
# D_train ∪ D_test = D
# D_train ∩ D_test = ∅

D = set(range(100))  # Dataset indices
D_train = set(range(70))
D_test = set(range(70, 100))

print(f"|D| = {len(D)} (dataset size)")
print(f"|D_train| = {len(D_train)}")
print(f"|D_test| = {len(D_test)}")
print(f"\\nD_train ∪ D_test = D: {D_train | D_test == D}")
print(f"D_train ∩ D_test = ∅: {len(D_train & D_test) == 0}")
\`\`\`

### Functions and Mappings

- **f: X → Y**: function f maps from X to Y
- **f(x)**: function application
- **f ∘ g**: function composition

\`\`\`python
# Example: Neural network as function composition
# y = f₃(f₂(f₁(x))) where fᵢ are layer functions

def layer1(x):
    """f₁: ℝⁿ → ℝᵐ"""
    W1 = np.array([[1, 0], [0, 1], [1, 1]])  # (3, 2)
    return W1 @ x

def layer2(x):
    """f₂: ℝᵐ → ℝᵏ with ReLU"""
    W2 = np.array([[1, 0, 1], [0, 1, 1]])  # (2, 3)
    return np.maximum(0, W2 @ x)  # ReLU activation

def layer3(x):
    """f₃: ℝᵏ → ℝ"""
    W3 = np.array([1, 1])  # (2,)
    return W3 @ x

# Composition: f = f₃ ∘ f₂ ∘ f₁
def neural_network(x):
    """f: ℝ² → ℝ"""
    return layer3(layer2(layer1(x)))

x = np.array([1, 2])
y = neural_network(x)

print(f"Input x: {x} ∈ ℝ²")
print(f"After layer 1: {layer1(x)} ∈ ℝ³")
print(f"After layer 2: {layer2(layer1(x))} ∈ ℝ²")
print(f"Output y: {y} ∈ ℝ")
print(f"\\nNeural network = f₃ ∘ f₂ ∘ f₁")
\`\`\`

## Logical Statements

### Quantifiers

- **∀** (for all): ∀x ∈ S, P(x) means "for all x in S, property P holds"
- **∃** (there exists): ∃x ∈ S, P(x) means "there exists an x in S such that P holds"

\`\`\`python
# Example: ∀x ∈ training set, loss(x) >= 0

def loss(y_true, y_pred):
    """MSE loss"""
    return (y_true - y_pred)**2

y_true_samples = np.array([1, 2, 3, 4, 5])
y_pred_samples = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

losses = [loss(yt, yp) for yt, yp in zip(y_true_samples, y_pred_samples)]

print("Loss values:", losses)
print(f"∀ samples, loss >= 0: {all(L >= 0 for L in losses)}")

# Example: ∃x such that gradient = 0 (local minimum)
def f(x):
    return (x - 2)**2

def gradient(x):
    return 2 * (x - 2)

x_values = np.linspace(0, 4, 100)
gradients = [gradient(x) for x in x_values]

# Find where gradient ≈ 0
zero_grad_indices = [i for i, g in enumerate(gradients) if abs(g) < 0.1]
print(f"\\n∃x where |∇f(x)| < 0.1: {len(zero_grad_indices) > 0}")
if zero_grad_indices:
    print(f"Found at x ≈ {x_values[zero_grad_indices[0]]:.2f}")
\`\`\`

### Implications

- **⇒** (implies): P ⇒ Q means "if P then Q"
- **⇔** (if and only if): P ⇔ Q means "P implies Q and Q implies P"

\`\`\`python
# Example: Convexity
# f is convex ⇔ f''(x) ≥ 0 ∀x

def f_convex(x):
    """Convex function: f(x) = x²"""
    return x**2

def f_second_derivative(x):
    """f''(x) = 2"""
    return 2

x_test = np.linspace(-5, 5, 100)
second_derivs = [f_second_derivative(x) for x in x_test]

is_convex = all(d >= 0 for d in second_derivs)
print(f"f(x) = x²")
print(f"f''(x) = 2 ≥ 0 ∀x: {is_convex}")
print(f"Therefore, f is convex")
\`\`\`

## Proof Techniques

### Direct Proof

Prove P ⇒ Q by assuming P and deriving Q.

**Example**: Prove that gradient descent with appropriate learning rate converges for convex functions.

\`\`\`python
# Simplified demonstration (not rigorous proof)

def gradient_descent_convex(f, grad_f, x0, lr, n_iterations):
    """
    Gradient descent on convex function
    """
    x = x0
    trajectory = [x]
    
    for _ in range(n_iterations):
        x = x - lr * grad_f(x)
        trajectory.append(x)
    
    return np.array(trajectory)

# Convex function: f(x) = x²
def f(x):
    return x**2

def grad_f(x):
    return 2*x

# Test convergence
x0 = 10
lr = 0.1
trajectory = gradient_descent_convex(f, grad_f, x0, lr, 50)

print("Gradient Descent on Convex Function:")
print(f"Initial: x₀ = {x0}")
print(f"Final: x₅₀ = {trajectory[-1]:.10f}")
print(f"Minimum at x* = 0")
print(f"Converged: {abs(trajectory[-1]) < 1e-6}")

# Plot
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
iterations = range(len(trajectory))
plt.plot(iterations, trajectory, 'bo-', markersize=4, linewidth=2)
plt.axhline(y=0, color='r', linestyle='--', label='Optimal x*=0')
plt.xlabel('Iteration')
plt.ylabel('x value')
plt.title('Gradient Descent Convergence (Convex Function)')
plt.legend()
plt.grid(True)
plt.show()
\`\`\`

### Proof by Contradiction

Assume ¬Q and derive a contradiction, thus proving Q.

**Example**: Prove that L2 regularization prevents weights from growing unbounded.

### Proof by Induction

Prove base case, then prove inductive step: if true for n, then true for n+1.

**Example**: Prove properties of recursive algorithms.

\`\`\`python
# Example: Prove Σ(i=1 to n) i = n(n+1)/2 by induction

def sum_first_n(n):
    """Compute 1 + 2 + ... + n"""
    return sum(range(1, n+1))

def formula_first_n(n):
    """Formula: n(n+1)/2"""
    return n * (n + 1) // 2

print("Proof by Induction: Σ(i=1 to n) i = n(n+1)/2")
print("\\nBase case (n=1):")
print(f"  LHS: Σ(i=1 to 1) i = {sum_first_n(1)}")
print(f"  RHS: 1(1+1)/2 = {formula_first_n(1)}")
print(f"  Equal: {sum_first_n(1) == formula_first_n(1)} ✓")

print("\\nInductive step: Assume true for n=k, prove for n=k+1")
print("  Σ(i=1 to k+1) i = [Σ(i=1 to k) i] + (k+1)")
print("                  = k(k+1)/2 + (k+1)    [by inductive hypothesis]")
print("                  = [k(k+1) + 2(k+1)]/2")
print("                  = (k+1)(k+2)/2")
print("                  = (k+1)((k+1)+1)/2   [formula for n=k+1] ✓")

print("\\nVerification for several values:")
for n in [1, 5, 10, 50, 100]:
    computed = sum_first_n(n)
    formula = formula_first_n(n)
    print(f"  n={n:>3}: computed={computed:>5}, formula={formula:>5}, match={computed == formula}")
\`\`\`

## Reading Mathematical Papers

### Common Patterns

**Theorem Statement**:
"Let f: ℝⁿ → ℝ be convex. Then gradient descent with learning rate α ≤ 1/L converges to global minimum."

**Translation**:
- f is a convex function (input: n-dimensional vector, output: scalar)
- If we use gradient descent with small enough learning rate
- Then it will reach the best solution

\`\`\`python
# Practical implementation of theorem

def is_lipschitz_smooth(f, grad_f, L, x_samples):
    """
    Check if gradient is L-Lipschitz smooth:
    ‖∇f(x) - ∇f(y)‖ ≤ L‖x - y‖
    """
    for i in range(len(x_samples)):
        for j in range(i+1, len(x_samples)):
            x, y = x_samples[i], x_samples[j]
            grad_diff = abs(grad_f(x) - grad_f(y))
            x_diff = abs(x - y)
            
            if grad_diff > L * x_diff + 1e-6:  # Small tolerance
                return False
    return True

# Example: f(x) = x², ∇f(x) = 2x, L = 2
def f(x):
    return x**2

def grad_f(x):
    return 2*x

L = 2
x_samples = np.linspace(-10, 10, 50)

is_smooth = is_lipschitz_smooth(f, grad_f, L, x_samples)
print(f"f(x) = x² is {L}-Lipschitz smooth: {is_smooth}")
print(f"Theorem says: use α ≤ 1/L = 1/{L} = {1/L}")
print(f"\\nTesting learning rates:")

for alpha in [0.3, 0.5, 0.7]:
    traj = gradient_descent_convex(f, grad_f, 10, alpha, 50)
    converged = abs(traj[-1]) < 1e-6
    print(f"  α = {alpha}: {'✓ converged' if converged else '✗ diverged'}")
\`\`\`

## Summary

- **Notation**: Precise, unambiguous communication
- **Subscripts/superscripts**: Indices and powers
- **Σ, Π**: Summation and product
- **∀, ∃**: Universal and existential quantifiers
- **⇒, ⇔**: Logical implications
- **Proofs**: Direct, contradiction, induction
- **Reading papers**: Translate math to code/intuition

**Key Skill**: Bidirectional translation between math notation and code

**Practice**: Read papers, implement algorithms, verify theorems numerically
`,
};
