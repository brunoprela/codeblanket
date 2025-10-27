/**
 * Sequences & Series Section
 */

export const sequencesseriesSection = {
  id: 'sequences-series',
  title: 'Sequences & Series',
  content: `
# Sequences & Series

## Introduction

Sequences and series are fundamental concepts that appear throughout machine learning and data science: gradient descent iterations, time series analysis, convergence analysis, loss curves, and compound returns in trading. Understanding their properties, convergence behavior, and summation is crucial for analyzing algorithmic behavior and financial models.

## Sequences

### Definition

A **sequence** is an ordered list of numbers: a₁, a₂, a₃, ..., aₙ, ...

**Notation**: {aₙ} or (aₙ)

**Index**: n (usually starts at 0 or 1)

**Term**: aₙ is the nth term

### Types of Sequences

#### Arithmetic Sequences

**Definition**: Constant difference between consecutive terms

**Formula**: aₙ = a₁ + (n-1)d
- a₁: first term
- d: common difference
- n: term number

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt

def arithmetic_sequence (a1, d, n_terms):
    """Generate arithmetic sequence"""
    n = np.arange(1, n_terms + 1)
    return a1 + (n - 1) * d

# Example: 3, 7, 11, 15, ...
a1, d = 3, 4
n_terms = 10
seq = arithmetic_sequence (a1, d, n_terms)

print("Arithmetic sequence:", seq)
print(f"First term: {seq[0]}")
print(f"Common difference: {d}")
print(f"10th term: {seq[-1]}")

# Verify constant difference
differences = np.diff (seq)
print(f"Differences: {differences}")
print(f"All equal to d? {np.all (differences == d)}")

# Visualize
plt.figure (figsize=(10, 6))
plt.plot (range(1, n_terms + 1), seq, 'bo-', markersize=8, linewidth=2)
plt.xlabel('n (term number)')
plt.ylabel('aₙ')
plt.title (f'Arithmetic Sequence: aₙ = {a1} + {d}(n-1)')
plt.grid(True)
plt.show()
\`\`\`

**ML Application**: Learning rate schedules with linear decay

\`\`\`python
def linear_lr_decay (initial_lr, decay_rate, epoch):
    """Linear learning rate decay (arithmetic sequence)"""
    return initial_lr - decay_rate * epoch

initial_lr = 0.1
decay_rate = 0.001
epochs = np.arange(0, 100)
lrs = linear_lr_decay (initial_lr, decay_rate, epochs)

plt.figure (figsize=(10, 6))
plt.plot (epochs, lrs, linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Linear Learning Rate Decay')
plt.grid(True)
plt.show()

print(f"Initial LR: {lrs[0]:.4f}")
print(f"LR at epoch 50: {lrs[50]:.4f}")
print(f"Final LR: {lrs[-1]:.4f}")
\`\`\`

#### Geometric Sequences

**Definition**: Constant ratio between consecutive terms

**Formula**: aₙ = a₁ · rⁿ⁻¹
- a₁: first term
- r: common ratio
- n: term number

\`\`\`python
def geometric_sequence (a1, r, n_terms):
    """Generate geometric sequence"""
    n = np.arange(1, n_terms + 1)
    return a1 * r**(n - 1)

# Example: 2, 6, 18, 54, ... (r=3)
a1, r = 2, 3
n_terms = 10
seq = geometric_sequence (a1, r, n_terms)

print("Geometric sequence:", seq)
print(f"First term: {seq[0]}")
print(f"Common ratio: {r}")
print(f"10th term: {seq[-1]}")

# Verify constant ratio
ratios = seq[1:] / seq[:-1]
print(f"Ratios: {ratios}")
print(f"All equal to r? {np.allclose (ratios, r)}")

# Visualize (log scale to show exponential growth)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot (range(1, n_terms + 1), seq, 'ro-', markersize=8, linewidth=2)
ax1.set_xlabel('n')
ax1.set_ylabel('aₙ')
ax1.set_title (f'Geometric Sequence (Linear Scale): aₙ = {a1}·{r}^(n-1)')
ax1.grid(True)

ax2.plot (range(1, n_terms + 1), seq, 'ro-', markersize=8, linewidth=2)
ax2.set_xlabel('n')
ax2.set_ylabel('aₙ')
ax2.set_yscale('log')
ax2.set_title (f'Geometric Sequence (Log Scale)')
ax2.grid(True)

plt.tight_layout()
plt.show()
\`\`\`

**ML Application**: Exponential learning rate decay

\`\`\`python
def exponential_lr_decay (initial_lr, decay_rate, epoch):
    """Exponential learning rate decay (geometric sequence)"""
    return initial_lr * decay_rate**epoch

initial_lr = 0.1
decay_rate = 0.96  # 4% decay per epoch
epochs = np.arange(0, 100)
lrs = exponential_lr_decay (initial_lr, decay_rate, epochs)

plt.figure (figsize=(10, 6))
plt.plot (epochs, lrs, linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Exponential Learning Rate Decay')
plt.grid(True)
plt.yscale('log')
plt.show()

print(f"Initial LR: {lrs[0]:.4f}")
print(f"LR at epoch 50: {lrs[50]:.6f}")
print(f"Final LR: {lrs[-1]:.8f}")
\`\`\`

#### Recursive Sequences

**Definition**: Each term defined in terms of previous term (s)

**Example - Fibonacci**: aₙ = aₙ₋₁ + aₙ₋₂, with a₁=1, a₂=1

\`\`\`python
def fibonacci (n):
    """Generate first n Fibonacci numbers"""
    if n <= 0:
        return []
    elif n == 1:
        return [1]
    elif n == 2:
        return [1, 1]

    fib = [1, 1]
    for i in range(2, n):
        fib.append (fib[-1] + fib[-2])
    return fib

# Generate first 20 Fibonacci numbers
fib_seq = fibonacci(20)
print("Fibonacci sequence:", fib_seq)

# Golden ratio approximation
ratios = [fib_seq[i+1] / fib_seq[i] for i in range (len (fib_seq)-1)]
print(f"\\nRatios converge to golden ratio φ ≈ 1.618...")
print(f"Last 5 ratios: {ratios[-5:]}")

golden_ratio = (1 + np.sqrt(5)) / 2
print(f"Golden ratio: {golden_ratio:.10f}")
print(f"Final ratio: {ratios[-1]:.10f}")

# Visualize
plt.figure (figsize=(10, 6))
plt.plot (range(1, len (fib_seq) + 1), fib_seq, 'go-', markersize=8, linewidth=2)
plt.xlabel('n')
plt.ylabel('Fibonacci (n)')
plt.title('Fibonacci Sequence')
plt.grid(True)
plt.yscale('log')
plt.show()
\`\`\`

**ML Application**: Recurrent relationships in RNNs

\`\`\`python
# Simple RNN: hidden state is recursive sequence
def simple_rnn_sequence (input_sequence, W_hh, W_xh, h0):
    """
    Generate hidden state sequence in RNN
    hₜ = tanh(W_hh·hₜ₋₁ + W_xh·xₜ)
    """
    hidden_states = [h0]
    h = h0

    for x in input_sequence:
        h = np.tanh(W_hh @ h + W_xh @ x)
        hidden_states.append (h)

    return np.array (hidden_states)

# Example
input_seq = [np.array([1.0]), np.array([0.5]), np.array([0.8])]
W_hh = np.array([[0.9]])  # Weight for previous hidden state
W_xh = np.array([[0.5]])  # Weight for input
h0 = np.array([0.0])      # Initial hidden state

hidden_seq = simple_rnn_sequence (input_seq, W_hh, W_xh, h0)
print("Hidden state sequence:", hidden_seq.flatten())
\`\`\`

### Convergence of Sequences

**Definition**: A sequence {aₙ} **converges** to L if for any ε > 0, there exists N such that |aₙ - L| < ε for all n > N.

**Notation**: lim (n→∞) aₙ = L

\`\`\`python
def visualize_convergence (sequence, limit, title):
    """Visualize sequence convergence"""
    n = len (sequence)

    plt.figure (figsize=(10, 6))
    plt.plot (range(1, n + 1), sequence, 'bo-', markersize=6, linewidth=2, label='Sequence')
    plt.axhline (y=limit, color='r', linestyle='--', linewidth=2, label=f'Limit = {limit}')

    # Show epsilon bands
    epsilon = 0.1
    plt.axhline (y=limit + epsilon, color='g', linestyle=':', alpha=0.5, label=f'ε = {epsilon}')
    plt.axhline (y=limit - epsilon, color='g', linestyle=':', alpha=0.5)

    plt.xlabel('n')
    plt.ylabel('aₙ')
    plt.title (title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Example 1: aₙ = 1/n → 0
n = np.arange(1, 101)
seq1 = 1 / n
visualize_convergence (seq1, 0, 'Convergence: aₙ = 1/n → 0')

# Example 2: aₙ = (1 + 1/n)^n → e
seq2 = (1 + 1/n)**n
visualize_convergence (seq2, np.e, 'Convergence: aₙ = (1 + 1/n)^n → e')

print(f"lim (n→∞) 1/n = {seq1[-1]:.6f}")
print(f"lim (n→∞) (1 + 1/n)^n = {seq2[-1]:.10f}")
print(f"Actual e = {np.e:.10f}")
\`\`\`

**ML Application**: Convergence of gradient descent

\`\`\`python
def gradient_descent_sequence (f, grad_f, x0, learning_rate, n_iterations):
    """
    Generate sequence of iterates in gradient descent
    xₙ₊₁ = xₙ - α·∇f (xₙ)
    """
    x_sequence = [x0]
    x = x0

    for _ in range (n_iterations):
        x = x - learning_rate * grad_f (x)
        x_sequence.append (x)

    return np.array (x_sequence)

# Example: f (x) = x^2, minimum at x=0
def f (x):
    return x**2

def grad_f (x):
    return 2*x

x0 = 10.0
lr = 0.1
n_iter = 50

x_seq = gradient_descent_sequence (f, grad_f, x0, lr, n_iter)

plt.figure (figsize=(10, 6))
plt.plot (range (len (x_seq)), x_seq, 'bo-', markersize=4, linewidth=2)
plt.axhline (y=0, color='r', linestyle='--', linewidth=2, label='Minimum')
plt.xlabel('Iteration')
plt.ylabel('x value')
plt.title('Gradient Descent Sequence Converging to Minimum')
plt.legend()
plt.grid(True)
plt.show()

print(f"Initial x: {x_seq[0]:.6f}")
print(f"Final x: {x_seq[-1]:.10f}")
print(f"Converged to 0? {np.abs (x_seq[-1]) < 1e-6}")
\`\`\`

## Series

### Definition

A **series** is the sum of terms in a sequence: S = a₁ + a₂ + a₃ + ... + aₙ + ...

**Notation**: Σ(n=1 to ∞) aₙ or Σ aₙ

**Partial sum**: Sₙ = Σ(k=1 to n) aₖ

### Arithmetic Series

**Sum formula**: Sₙ = n/2 · (a₁ + aₙ) = n/2 · (2a₁ + (n-1)d)

\`\`\`python
def arithmetic_series_sum (a1, d, n):
    """Sum of first n terms of arithmetic sequence"""
    # Method 1: Direct formula
    sum_formula = n/2 * (2*a1 + (n-1)*d)

    # Method 2: Explicit computation (verification)
    terms = arithmetic_sequence (a1, d, n)
    sum_explicit = np.sum (terms)

    return sum_formula, sum_explicit

# Example: Sum of 1 + 2 + 3 + ... + 100
a1, d, n = 1, 1, 100
sum_formula, sum_explicit = arithmetic_series_sum (a1, d, n)

print(f"Sum of first {n} natural numbers:")
print(f"Formula: {sum_formula:.0f}")
print(f"Explicit: {sum_explicit:.0f}")
print(f"Gauss formula: n (n+1)/2 = {n*(n+1)/2}")

# Visualize partial sums
n_values = np.arange(1, 101)
partial_sums = [arithmetic_series_sum(1, 1, n)[0] for n in n_values]

plt.figure (figsize=(10, 6))
plt.plot (n_values, partial_sums, linewidth=2)
plt.xlabel('n')
plt.ylabel('Sₙ (sum of first n terms)')
plt.title('Arithmetic Series: Sₙ = 1 + 2 + ... + n')
plt.grid(True)
plt.show()
\`\`\`

**ML Application**: Analyzing training time complexity

\`\`\`python
# Total number of operations in training with variable batch sizes
def total_operations (batch_sizes):
    """
    If batch sizes form arithmetic sequence,
    total ops = arithmetic series sum
    """
    n = len (batch_sizes)
    a1 = batch_sizes[0]
    d = batch_sizes[1] - batch_sizes[0] if n > 1 else 0

    total = n/2 * (2*a1 + (n-1)*d)
    return int (total)

# Example: batch sizes 32, 36, 40, ..., 100
batch_sizes = list (range(32, 101, 4))
total_ops = total_operations (batch_sizes)
print(f"Batch sizes: {batch_sizes[:5]} ... {batch_sizes[-3:]}")
print(f"Total operations: {total_ops:,}")
\`\`\`

### Geometric Series

**Sum formula (finite)**: Sₙ = a₁ · (1 - rⁿ) / (1 - r) for r ≠ 1

**Infinite series**: S = a₁ / (1 - r) if |r| < 1 (converges)

\`\`\`python
def geometric_series_sum (a1, r, n):
    """Sum of first n terms of geometric sequence"""
    if r == 1:
        return a1 * n

    # Finite sum formula
    sum_formula = a1 * (1 - r**n) / (1 - r)

    # Explicit computation (verification)
    terms = geometric_sequence (a1, r, n)
    sum_explicit = np.sum (terms)

    return sum_formula, sum_explicit

# Example: 1 + 1/2 + 1/4 + 1/8 + ...
a1, r = 1, 0.5
n_terms = [5, 10, 20, 50, 100]

print("Geometric series: 1 + 1/2 + 1/4 + ...")
for n in n_terms:
    sum_n, _ = geometric_series_sum (a1, r, n)
    print(f"S_{n:>3} = {sum_n:.10f}")

# Infinite sum (converges for |r| < 1)
infinite_sum = a1 / (1 - r)
print(f"\\nInfinite sum (theoretical): {infinite_sum}")
print(f"S_100 is very close to infinite sum: {np.isclose (sum_n, infinite_sum)}")

# Visualize convergence
n_range = np.arange(1, 101)
partial_sums = [geometric_series_sum (a1, r, n)[0] for n in n_range]

plt.figure (figsize=(10, 6))
plt.plot (n_range, partial_sums, linewidth=2, label='Partial sums')
plt.axhline (y=infinite_sum, color='r', linestyle='--', linewidth=2, label=f'Limit = {infinite_sum}')
plt.xlabel('n')
plt.ylabel('Sₙ')
plt.title('Geometric Series Convergence')
plt.legend()
plt.grid(True)
plt.show()
\`\`\`

**ML Application**: Discount factor in reinforcement learning

\`\`\`python
def discounted_return (rewards, gamma):
    """
    Compute discounted return (geometric series)
    G = r₁ + γr₂ + γ²r₃ + ... = Σ γᵗrₜ
    """
    n = len (rewards)
    discount_factors = gamma ** np.arange (n)
    return np.sum (rewards * discount_factors)

# Example: sequence of rewards in RL
rewards = np.array([1, 2, 3, 4, 5])
gamma = 0.9  # Discount factor

G = discounted_return (rewards, gamma)
print(f"Rewards: {rewards}")
print(f"Discount factor γ: {gamma}")
print(f"Discounted return: {G:.4f}")

# Compare with undiscounted
undiscounted = np.sum (rewards)
print(f"Undiscounted sum: {undiscounted}")
print(f"Discount effect: {(1 - G/undiscounted)*100:.1f}% reduction")
\`\`\`

### Infinite Series and Convergence

#### Tests for Convergence

**1. Divergence Test**: If lim (n→∞) aₙ ≠ 0, then Σaₙ diverges

**2. Ratio Test**: If lim |aₙ₊₁/aₙ| < 1, series converges

**3. Comparison Test**: If 0 ≤ aₙ ≤ bₙ and Σbₙ converges, then Σaₙ converges

\`\`\`python
def ratio_test (sequence, n_terms=100):
    """
    Apply ratio test to check convergence
    Returns limit of |aₙ₊₁/aₙ|
    """
    ratios = np.abs (sequence[1:] / sequence[:-1])

    # Take last several ratios (should stabilize)
    limit = np.mean (ratios[-10:])

    print(f"Ratio test: lim |aₙ₊₁/aₙ| ≈ {limit:.6f}")
    if limit < 1:
        print("Series converges (ratio < 1)")
    elif limit > 1:
        print("Series diverges (ratio > 1)")
    else:
        print("Test inconclusive (ratio = 1)")

    return limit

# Example 1: Σ 1/n² (converges)
n = np.arange(1, 101)
seq1 = 1 / n**2
print("Series: Σ 1/n²")
print(f"Partial sum S_100: {np.sum (seq1):.6f}")
print(f"Known limit: π²/6 = {np.pi**2/6:.6f}")
# Ratio test
ratio1 = ratio_test (seq1)

# Example 2: Σ 1/2^n (converges - geometric with r=1/2)
seq2 = 1 / 2**n
print(f"\\nSeries: Σ 1/2^n")
print(f"Partial sum S_100: {np.sum (seq2):.10f}")
print(f"Known limit: 1/(1-1/2) = 2 (minus first term)")
ratio2 = ratio_test (seq2)

# Example 3: Σ n (diverges)
seq3 = n
print(f"\\nSeries: Σ n")
print(f"Partial sum S_100: {np.sum (seq3):.0f}")
print("This series diverges (terms don't approach 0)")
\`\`\`

### Power Series

**Definition**: Σ(n=0 to ∞) cₙxⁿ

Important power series in ML:

**1. Exponential**: eˣ = Σ xⁿ/n!

**2. Sine**: sin (x) = Σ (-1)ⁿx^(2n+1)/(2n+1)!

**3. Cosine**: cos (x) = Σ (-1)ⁿx^(2n)/(2n)!

**4. Geometric**: 1/(1-x) = Σ xⁿ for |x| < 1

\`\`\`python
from math import factorial

def exp_series (x, n_terms=20):
    """Approximate e^x using power series"""
    return sum (x**n / factorial (n) for n in range (n_terms))

def sin_series (x, n_terms=20):
    """Approximate sin (x) using power series"""
    return sum((-1)**n * x**(2*n+1) / factorial(2*n+1) for n in range (n_terms))

def cos_series (x, n_terms=20):
    """Approximate cos (x) using power series"""
    return sum((-1)**n * x**(2*n) / factorial(2*n) for n in range (n_terms))

# Test approximations
x_test = 1.5

print("Power series approximations vs actual:")
print(f"\\nx = {x_test}")
print(f"e^x: series = {exp_series (x_test):.10f}, actual = {np.exp (x_test):.10f}")
print(f"sin (x): series = {sin_series (x_test):.10f}, actual = {np.sin (x_test):.10f}")
print(f"cos (x): series = {cos_series (x_test):.10f}, actual = {np.cos (x_test):.10f}")

# Visualize convergence of e^x series
x_range = np.linspace(-2, 2, 100)
n_terms_list = [1, 2, 3, 5, 10, 20]

plt.figure (figsize=(12, 6))
for n in n_terms_list:
    approx = [exp_series (x, n) for x in x_range]
    plt.plot (x_range, approx, label=f'{n} terms', linewidth=2)

plt.plot (x_range, np.exp (x_range), 'k--', linewidth=2, label='Actual e^x')
plt.xlabel('x')
plt.ylabel('e^x')
plt.title('Power Series Approximation of e^x')
plt.legend()
plt.grid(True)
plt.ylim(-5, 10)
plt.show()
\`\`\`

**ML Application**: Taylor approximation of activation functions

\`\`\`python
def sigmoid (x):
    """Standard sigmoid"""
    return 1 / (1 + np.exp(-x))

def sigmoid_taylor (x, n_terms=5):
    """
    Taylor series approximation of sigmoid around x=0
    σ(x) ≈ 1/2 + x/4 - x³/48 + ...
    """
    # First few terms of Taylor series
    if n_terms >= 1:
        result = 0.5
    if n_terms >= 2:
        result += x / 4
    if n_terms >= 3:
        result -= x**3 / 48
    if n_terms >= 4:
        result += x**5 / 480
    return result

# Compare
x_range = np.linspace(-2, 2, 100)
sig_actual = sigmoid (x_range)
sig_taylor = [sigmoid_taylor (x, 3) for x in x_range]

plt.figure (figsize=(10, 6))
plt.plot (x_range, sig_actual, 'b-', linewidth=2, label='Actual sigmoid')
plt.plot (x_range, sig_taylor, 'r--', linewidth=2, label='Taylor approx (3 terms)')
plt.xlabel('x')
plt.ylabel('σ(x)')
plt.title('Sigmoid vs Taylor Approximation')
plt.legend()
plt.grid(True)
plt.show()

print("Taylor approximation is good near x=0!")
print("This is used for efficient approximate computations")
\`\`\`

## Summation Notation

### Sigma Notation

**Σ(i=m to n) f (i)**: Sum f (i) for i from m to n

**Properties**:
- Σ(c·aᵢ) = c·Σaᵢ (constant multiple)
- Σ(aᵢ + bᵢ) = Σaᵢ + Σbᵢ (linearity)
- Σ(i=1 to n) c = n·c (constant sum)

\`\`\`python
def evaluate_sum (f, start, end):
    """Evaluate Σ f (i) from start to end"""
    return sum (f(i) for i in range (start, end + 1))

# Common summation formulas
def sum_first_n_natural (n):
    """Σ(i=1 to n) i = n (n+1)/2"""
    return n * (n + 1) // 2

def sum_first_n_squares (n):
    """Σ(i=1 to n) i² = n (n+1)(2n+1)/6"""
    return n * (n + 1) * (2*n + 1) // 6

def sum_first_n_cubes (n):
    """Σ(i=1 to n) i³ = [n (n+1)/2]²"""
    return (n * (n + 1) // 2) ** 2

n = 100
print(f"Summation formulas for n = {n}:")
print(f"Σ i     = {sum_first_n_natural (n):,}")
print(f"Σ i²    = {sum_first_n_squares (n):,}")
print(f"Σ i³    = {sum_first_n_cubes (n):,}")

# Verify with explicit computation
print(f"\\nVerification:")
print(f"Σ i     = {sum (range(1, n+1)):,}")
print(f"Σ i²    = {sum (i**2 for i in range(1, n+1)):,}")
print(f"Σ i³    = {sum (i**3 for i in range(1, n+1)):,}")
\`\`\`

**ML Application**: Loss function over dataset

\`\`\`python
def mean_squared_error (y_true, y_pred):
    """
    MSE = (1/n) Σ(yᵢ - ŷᵢ)²
    Summation over all data points
    """
    n = len (y_true)
    return np.sum((y_true - y_pred)**2) / n

def cross_entropy_loss (y_true, y_pred, epsilon=1e-10):
    """
    CE = -(1/n) Σ[yᵢlog(ŷᵢ) + (1-yᵢ)log(1-ŷᵢ)]
    Summation over all data points
    """
    y_pred = np.clip (y_pred, epsilon, 1 - epsilon)
    n = len (y_true)
    return -np.sum (y_true * np.log (y_pred) + (1 - y_true) * np.log(1 - y_pred)) / n

# Example
y_true = np.array([0, 1, 1, 0, 1])
y_pred = np.array([0.1, 0.9, 0.8, 0.2, 0.7])

mse = mean_squared_error (y_true, y_pred)
ce = cross_entropy_loss (y_true, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Cross-Entropy Loss: {ce:.4f}")
\`\`\`

## Applications in Trading

### Compound Returns (Geometric Series)

\`\`\`python
def compound_return (returns):
    """
    Total return with compounding
    (1+r₁)(1+r₂)...(1+rₙ) - 1
    """
    return np.prod(1 + returns) - 1

def arithmetic_mean_return (returns):
    """Simple average return"""
    return np.mean (returns)

def geometric_mean_return (returns):
    """Geometric mean (compound average)"""
    return np.prod(1 + returns)**(1/len (returns)) - 1

# Monthly returns
monthly_returns = np.array([0.02, -0.01, 0.03, 0.01, -0.02, 0.04])

total = compound_return (monthly_returns)
arith_mean = arithmetic_mean_return (monthly_returns)
geom_mean = geometric_mean_return (monthly_returns)

print("Monthly returns:", monthly_returns)
print(f"\\nTotal compounded return: {total*100:.2f}%")
print(f"Arithmetic mean: {arith_mean*100:.2f}%")
print(f"Geometric mean: {geom_mean*100:.2f}%")
print("\\nGeometric mean ≤ Arithmetic mean (equality only if all returns equal)")
\`\`\`

### Moving Averages (Arithmetic Series)

\`\`\`python
def simple_moving_average (prices, window):
    """
    SMA(t) = (1/n) Σ(i=t-n+1 to t) priceᵢ
    Arithmetic series with equal weights
    """
    sma = []
    for i in range (window - 1, len (prices)):
        window_prices = prices[i - window + 1:i + 1]
        sma.append (np.mean (window_prices))
    return np.array (sma)

def exponential_moving_average (prices, span):
    """
    EMA uses exponentially decaying weights (geometric series)
    """
    alpha = 2 / (span + 1)
    ema = [prices[0]]

    for price in prices[1:]:
        ema.append (alpha * price + (1 - alpha) * ema[-1])

    return np.array (ema)

# Example stock prices
np.random.seed(42)
prices = 100 + np.cumsum (np.random.randn(100) * 2)

sma_20 = simple_moving_average (prices, 20)
ema_20 = exponential_moving_average (prices, 20)

plt.figure (figsize=(12, 6))
plt.plot (prices, label='Price', linewidth=1, alpha=0.7)
plt.plot (range(19, len (prices)), sma_20, label='SMA(20)', linewidth=2)
plt.plot (ema_20, label='EMA(20)', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Moving Averages')
plt.legend()
plt.grid(True)
plt.show()

print(f"SMA uses equal weights: 1/n for each term")
print(f"EMA uses exponentially decaying weights: α(1-α)^i (geometric)")
\`\`\`

## Summary

- **Sequences**: Ordered lists of numbers (arithmetic, geometric, recursive)
- **Convergence**: Sequences approaching a limit
- **Series**: Sums of sequence terms
- **Arithmetic series**: Sum = n/2(a₁ + aₙ)
- **Geometric series**: Sum = a₁(1-rⁿ)/(1-r), infinite sum = a₁/(1-r) if |r|<1
- **Power series**: Represent functions as infinite polynomials
- **Summation notation**: Σ for compact representation

**ML Applications**:
- Gradient descent iterations (convergent sequence)
- Learning rate schedules (arithmetic/geometric sequences)
- Loss over epochs (convergent sequence)
- Discounted returns in RL (geometric series)
- Taylor approximations (power series)
- Dataset summations (loss functions)

**Trading Applications**:
- Compound returns (geometric series)
- Moving averages (arithmetic series, exponential weights)
- Portfolio growth over time
- Time value of money
`,
};
