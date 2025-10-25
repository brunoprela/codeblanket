/**
 * Exponents & Logarithms Section
 */

export const exponentslogarithmsSection = {
  id: 'exponents-logarithms',
  title: 'Exponents & Logarithms',
  content: `
# Exponents & Logarithms

## Introduction

Exponents and logarithms are fundamental operations that appear everywhere in machine learning: learning rate schedules, activation functions (sigmoid, softmax), information theory (entropy, cross-entropy), complexity analysis, and time series forecasting. Understanding their properties and relationship is crucial for both theory and implementation.

## Laws of Exponents

### Basic Rules

**Product Rule**: aᵐ · aⁿ = aᵐ⁺ⁿ
**Quotient Rule**: aᵐ / aⁿ = aᵐ⁻ⁿ
**Power Rule**: (aᵐ)ⁿ = aᵐⁿ
**Power of Product**: (ab)ᵐ = aᵐbᵐ
**Power of Quotient**: (a/b)ᵐ = aᵐ/bᵐ
**Zero Exponent**: a⁰ = 1 (for a ≠ 0)
**Negative Exponent**: a⁻ⁿ = 1/aⁿ
**Fractional Exponent**: a^(m/n) = ⁿ√(aᵐ)

### Python Implementation

\`\`\`python
import numpy as np

# Basic exponent operations
a, m, n = 2, 3, 4

# Product rule
print(f"{a}^{m} · {a}^{n} = {a**m * a**n} = {a**(m+n)}")  # 2³ · 2⁴ = 128 = 2⁷

# Quotient rule
print(f"{a}^{m} / {a}^{n} = {a**m / a**n} = {a**(m-n)}")  # 2³ / 2⁴ = 0.125 = 2⁻¹

# Power rule
print(f"({a}^{m})^{n} = {(a**m)**n} = {a**(m*n)}")  # (2³)⁴ = 4096 = 2¹²

# Zero exponent
print(f"{a}^0 = {a**0}")  # 2⁰ = 1

# Negative exponent
print(f"{a}^-{m} = {a**(-m)} = {1/a**m}")  # 2⁻³ = 0.125

# Fractional exponent
print(f"{a}^(1/{m}) = {a**(1/m)} ≈ {a**(1/m):.4f}")  # 2^(1/3) ≈ 1.2599 (cube root)
\`\`\`

### ML Application: Learning Rate Decay

Exponential decay is common for learning rate schedules:

\`\`\`python
def exponential_decay (initial_lr, epoch, decay_rate):
    """
    Learning rate with exponential decay
    lr = lr₀ · e^(-decay_rate · epoch)
    """
    return initial_lr * np.exp(-decay_rate * epoch)

initial_lr = 0.1
decay_rate = 0.05
epochs = np.arange(0, 100)

# Calculate learning rates
lrs = exponential_decay (initial_lr, epochs, decay_rate)

# Visualize
import matplotlib.pyplot as plt
plt.figure (figsize=(10, 6))
plt.plot (epochs, lrs, linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Exponential Learning Rate Decay')
plt.grid(True)
plt.show()

print(f"Initial LR: {lrs[0]:.4f}")
print(f"LR at epoch 50: {lrs[50]:.6f}")
print(f"LR at epoch 99: {lrs[99]:.6f}")
\`\`\`

## Logarithms

### Definition

**Logarithm**: The inverse of exponentiation

If aˣ = y, then logₐ(y) = x
- a: base
- y: argument
- x: result

**Common bases**:
- **Natural log**: ln (x) = logₑ(x), where e ≈ 2.71828
- **Common log**: log₁₀(x)
- **Binary log**: log₂(x) (used in information theory)

### Python Implementation

\`\`\`python
# Different logarithm bases
x = 8

# Natural logarithm (base e)
ln_x = np.log (x)
print(f"ln({x}) = {ln_x:.4f}")
print(f"Verify: e^{ln_x:.4f} = {np.exp (ln_x):.4f}")

# Common logarithm (base 10)
log10_x = np.log10(x)
print(f"\\nlog₁₀({x}) = {log10_x:.4f}")
print(f"Verify: 10^{log10_x:.4f} = {10**log10_x:.4f}")

# Binary logarithm (base 2)
log2_x = np.log2(x)
print(f"\\nlog₂({x}) = {log2_x:.4f}")
print(f"Verify: 2^{log2_x:.4f} = {2**log2_x:.4f}")

# Change of base formula: logₐ(x) = ln (x) / ln (a)
base = 5
logbase_x = np.log (x) / np.log (base)
print(f"\\nlog₅({x}) = {logbase_x:.4f}")
print(f"Verify: 5^{logbase_x:.4f} = {base**logbase_x:.4f}")
\`\`\`

## Laws of Logarithms

### Basic Rules

**Product Rule**: log (xy) = log (x) + log (y)
**Quotient Rule**: log (x/y) = log (x) - log (y)
**Power Rule**: log (xⁿ) = n·log (x)
**Change of Base**: logₐ(x) = logᵦ(x) / logᵦ(a)
**Identity**: logₐ(a) = 1
**Inverse**: logₐ(1) = 0

### Python Verification

\`\`\`python
x, y, n = 4, 16, 3

# Product rule
print(f"log({x}·{y}) = {np.log (x*y):.4f}")
print(f"log({x}) + log({y}) = {np.log (x) + np.log (y):.4f}")
print(f"Equal: {np.isclose (np.log (x*y), np.log (x) + np.log (y))}")

# Quotient rule
print(f"\\nlog({y}/{x}) = {np.log (y/x):.4f}")
print(f"log({y}) - log({x}) = {np.log (y) - np.log (x):.4f}")
print(f"Equal: {np.isclose (np.log (y/x), np.log (y) - np.log (x))}")

# Power rule
print(f"\\nlog({x}^{n}) = {np.log (x**n):.4f}")
print(f"{n}·log({x}) = {n * np.log (x):.4f}")
print(f"Equal: {np.isclose (np.log (x**n), n * np.log (x))}")
\`\`\`

### ML Application: Log Space Computation

Many ML operations are more stable in log space:

\`\`\`python
# Problem: Computing product of many small probabilities
probs = np.array([0.1, 0.2, 0.15, 0.08, 0.12])

# Direct multiplication (can underflow)
product_direct = np.prod (probs)
print(f"Direct product: {product_direct}")
print(f"Scientific notation: {product_direct:.2e}")

# Log space computation (more stable)
log_probs = np.log (probs)
log_product = np.sum (log_probs)  # log (a·b·c) = log (a) + log (b) + log (c)
product_log_space = np.exp (log_product)
print(f"\\nLog space product: {product_log_space}")
print(f"Scientific notation: {product_log_space:.2e}")

# Even with very small probabilities
tiny_probs = np.full(100, 0.01)  # 100 probabilities of 0.01
print(f"\\n100 probabilities of 0.01:")
print(f"Direct: {np.prod (tiny_probs):.2e}")  # May underflow to 0
print(f"Log space: {np.exp (np.sum (np.log (tiny_probs))):.2e}")
\`\`\`

## Exponential Growth and Compound Interest

### Compound Interest Formula

A = P(1 + r/n)ⁿᵗ
- A: final amount
- P: principal (initial amount)
- r: annual interest rate
- n: number of times compounded per year
- t: time in years

**Continuous compounding**: A = Peʳᵗ (as n → ∞)

\`\`\`python
def compound_interest (principal, rate, times_per_year, years):
    """Calculate compound interest"""
    return principal * (1 + rate/times_per_year)**(times_per_year * years)

def continuous_compound (principal, rate, years):
    """Calculate continuous compound interest"""
    return principal * np.exp (rate * years)

# Example: $1000 at 5% for 10 years
P, r, t = 1000, 0.05, 10

# Different compounding frequencies
print(f"Initial investment: \${P}")
print(f"Annual rate: {r*100}%")
print(f"Time: {t} years\\n")

print(f"Annual compounding: \${compound_interest(P, r, 1, t):.2f}")
print(f"Monthly compounding: \${compound_interest(P, r, 12, t):.2f}")
print(f"Daily compounding: \${compound_interest(P, r, 365, t):.2f}")
print(f"Continuous compounding: \${continuous_compound(P, r, t):.2f}")
\`\`\`

### Trading Application: Portfolio Growth

\`\`\`python
def portfolio_growth (initial_value, monthly_return, months):
    """
    Calculate portfolio growth with compound returns
    Similar to compound interest but with monthly returns
    """
    return initial_value * (1 + monthly_return)**months

# Example: $10,000 portfolio, 2% average monthly return
initial = 10000
monthly_return = 0.02
months = 12

final_value = portfolio_growth (initial, monthly_return, months)
total_return = (final_value - initial) / initial

print(f"Initial portfolio: \${initial:,.2f}")
print(f"Average monthly return: {monthly_return*100}%")
print(f"After {months} months: \${final_value:,.2f}")
print(f"Total return: {total_return*100:.2f}%")
print(f"Simple (non-compounded) would be: {monthly_return*months*100:.2f}%")
\`\`\`

## Logarithmic Scales

Logarithmic scales are useful when data spans many orders of magnitude:

\`\`\`python
# Training loss often decreases exponentially
epochs = np.arange(1, 101)
loss = 10 * np.exp(-0.05 * epochs) + 0.1  # Exponential decay + noise

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Linear scale
ax1.plot (epochs, loss, linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss (Linear Scale)')
ax1.grid(True)

# Logarithmic scale
ax2.plot (epochs, loss, linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_yscale('log')
ax2.set_title('Training Loss (Log Scale)')
ax2.grid(True)

plt.tight_layout()
plt.show()

print("Log scale makes exponential trends linear!")
\`\`\`

## Natural Exponential and e

### The number e

e ≈ 2.71828... is the base of natural logarithms

**Properties**:
- e = lim (n→∞) (1 + 1/n)ⁿ
- e = Σ(1/k!) for k=0 to ∞
- eˣ is the only function equal to its own derivative: d/dx (eˣ) = eˣ

### Why e is "Natural"

\`\`\`python
# Demonstrating e through compound interest
n_values = [1, 10, 100, 1000, 10000, 100000, 1000000]
e_approx = [(1 + 1/n)**n for n in n_values]

print("Approximating e with (1 + 1/n)^n:")
for n, e_val in zip (n_values, e_approx):
    print(f"n = {n:>7}: e ≈ {e_val:.10f}")
print(f"\\nActual e: {np.e:.10f}")

# e through series
def e_series (terms):
    """Approximate e using series: e = 1 + 1/1! + 1/2! + 1/3! + ..."""
    from math import factorial
    return sum(1/factorial (k) for k in range (terms))

print(f"\\ne from series (10 terms): {e_series(10):.10f}")
print(f"e from series (20 terms): {e_series(20):.10f}")
\`\`\`

## Information Theory: Entropy

Logarithms are fundamental in information theory:

### Shannon Entropy

H(X) = -Σ p (x) log₂(p (x))

Measures average information content or uncertainty.

\`\`\`python
def entropy (probabilities):
    """
    Calculate Shannon entropy
    Uses log base 2 (bits of information)
    """
    # Remove zeros to avoid log(0)
    probs = probabilities[probabilities > 0]
    return -np.sum (probs * np.log2(probs))

# Example 1: Fair coin
fair_coin = np.array([0.5, 0.5])
H_fair = entropy (fair_coin)
print(f"Fair coin entropy: {H_fair:.4f} bits")  # 1 bit

# Example 2: Biased coin
biased_coin = np.array([0.9, 0.1])
H_biased = entropy (biased_coin)
print(f"Biased coin entropy: {H_biased:.4f} bits")  # Less than 1

# Example 3: Certain outcome
certain = np.array([1.0, 0.0])
H_certain = entropy (certain)
print(f"Certain outcome entropy: {H_certain:.4f} bits")  # 0

# Example 4: Uniform distribution over 8 outcomes
uniform_8 = np.ones(8) / 8
H_uniform = entropy (uniform_8)
print(f"Uniform over 8 outcomes: {H_uniform:.4f} bits")  # 3 bits (log₂(8))

print("\\nHigher entropy = more uncertainty = more information needed")
\`\`\`

### Cross-Entropy Loss

Cross-entropy between true distribution p and predicted distribution q:

H(p, q) = -Σ p (x) log (q(x))

\`\`\`python
def cross_entropy (y_true, y_pred, epsilon=1e-10):
    """
    Cross-entropy loss (using natural log)
    y_true: true probabilities
    y_pred: predicted probabilities
    """
    y_pred = np.clip (y_pred, epsilon, 1 - epsilon)
    return -np.sum (y_true * np.log (y_pred))

# Binary classification example
y_true = np.array([1, 0, 1, 1, 0])  # True labels

# One-hot encode for binary case
y_true_onehot = np.column_stack([1 - y_true, y_true])

# Good predictions (confident and correct)
y_pred_good = np.array([[0.1, 0.9],   # Predict 1, true is 1 ✓
                        [0.9, 0.1],   # Predict 0, true is 0 ✓
                        [0.2, 0.8],   # Predict 1, true is 1 ✓
                        [0.15, 0.85], # Predict 1, true is 1 ✓
                        [0.85, 0.15]])# Predict 0, true is 0 ✓

# Bad predictions (wrong)
y_pred_bad = np.array([[0.6, 0.4],    # Predict 0, true is 1 ✗
                       [0.4, 0.6],    # Predict 1, true is 0 ✗
                       [0.5, 0.5],    # Uncertain, true is 1
                       [0.7, 0.3],    # Predict 0, true is 1 ✗
                       [0.3, 0.7]])   # Predict 1, true is 0 ✗

loss_good = sum (cross_entropy (y_true_onehot[i], y_pred_good[i]) 
                for i in range (len (y_true))) / len (y_true)
loss_bad = sum (cross_entropy (y_true_onehot[i], y_pred_bad[i]) 
               for i in range (len (y_true))) / len (y_true)

print(f"Cross-entropy (good predictions): {loss_good:.4f}")
print(f"Cross-entropy (bad predictions): {loss_bad:.4f}")
print(f"\\nLower is better - good predictions have lower loss")
\`\`\`

## Logarithmic Complexity

Algorithm complexity often involves logarithms:

\`\`\`python
import time

# Binary search: O(log n)
def binary_search (arr, target):
    """Binary search in sorted array"""
    left, right = 0, len (arr) - 1
    comparisons = 0
    
    while left <= right:
        comparisons += 1
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid, comparisons
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1, comparisons

# Linear search: O(n)
def linear_search (arr, target):
    """Linear search"""
    comparisons = 0
    for i, val in enumerate (arr):
        comparisons += 1
        if val == target:
            return i, comparisons
    return -1, comparisons

# Compare on different sizes
sizes = [100, 1000, 10000, 100000]
print("Comparisons needed to find element at end:\\n")
print(f"{'Size':>10} {'Linear':>12} {'Binary':>12} {'log₂(n)':>12}")
print("-" * 50)

for n in sizes:
    arr = list (range (n))
    target = n - 1  # Last element
    
    _, linear_comps = linear_search (arr, target)
    _, binary_comps = binary_search (arr, target)
    log2_n = np.log2(n)
    
    print(f"{n:>10} {linear_comps:>12} {binary_comps:>12} {log2_n:>12.2f}")

print("\\nBinary search comparisons ≈ log₂(n)")
print("This is why logarithmic algorithms scale so well!")
\`\`\`

## Practical ML Applications

### Softmax with Log-Sum-Exp Trick

Softmax is numerically unstable. The log-sum-exp trick uses logarithms for stability:

\`\`\`python
def softmax_naive (x):
    """Naive softmax (can overflow)"""
    exp_x = np.exp (x)
    return exp_x / np.sum (exp_x)

def softmax_stable (x):
    """Numerically stable softmax using log-sum-exp trick"""
    # Subtract max for stability
    exp_x = np.exp (x - np.max (x))
    return exp_x / np.sum (exp_x)

# Example with large values
logits_large = np.array([1000, 1001, 1002])

try:
    result_naive = softmax_naive (logits_large)
    print(f"Naive softmax: {result_naive}")
except:
    print("Naive softmax: OVERFLOW ERROR")

result_stable = softmax_stable (logits_large)
print(f"Stable softmax: {result_stable}")
print(f"Sum: {np.sum (result_stable):.10f}")  # Should be 1.0

# Why it works:
# softmax (x) = exp (x) / Σexp (x)
# = exp (x - max (x)) / Σexp (x - max (x))
# Subtracting max keeps exponentials from overflowing
\`\`\`

### Log-Likelihood in Training

Many loss functions are negative log-likelihoods:

\`\`\`python
def negative_log_likelihood (y_true, y_pred_prob, epsilon=1e-10):
    """
    Negative log-likelihood loss
    Equivalent to cross-entropy for classification
    """
    y_pred_prob = np.clip (y_pred_prob, epsilon, 1 - epsilon)
    return -np.mean (np.log (y_pred_prob[np.arange (len (y_true)), y_true]))

# Multi-class classification
y_true = np.array([0, 2, 1, 0, 2])  # Class indices
y_pred_prob = np.array([
    [0.7, 0.2, 0.1],  # Predict class 0, true is 0 ✓
    [0.1, 0.2, 0.7],  # Predict class 2, true is 2 ✓
    [0.2, 0.6, 0.2],  # Predict class 1, true is 1 ✓
    [0.8, 0.1, 0.1],  # Predict class 0, true is 0 ✓
    [0.1, 0.3, 0.6],  # Predict class 2, true is 2 ✓
])

nll = negative_log_likelihood (y_true, y_pred_prob)
print(f"Negative Log-Likelihood: {nll:.4f}")
print("\\nLower NLL = better predictions")
print("NLL is minimized when predicted probabilities match true labels")
\`\`\`

## Summary

- **Exponents** model growth (compound interest, neural network depth, learning curves)
- **Logarithms** are inverses of exponents and compress large ranges
- **Laws** of exponents and logarithms simplify complex calculations
- **Log space** provides numerical stability for products of small numbers
- **e** and natural log appear naturally in continuous growth and calculus
- **Entropy** and **cross-entropy** use logarithms to measure information and loss
- **Logarithmic complexity** O(log n) is extremely efficient for large datasets
- **Numerical tricks** (log-sum-exp, log-likelihood) prevent overflow/underflow

**Key ML Applications**:
- Softmax and log-softmax
- Cross-entropy loss
- Learning rate schedules
- Information theory metrics
- Complexity analysis
- Numerical stability
`,
};
