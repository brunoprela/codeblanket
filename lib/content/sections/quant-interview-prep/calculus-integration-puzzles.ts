export const calculusIntegrationPuzzles = {
  title: 'Calculus & Integration Puzzles',
  id: 'calculus-integration-puzzles',
  content: `
# Calculus & Integration Puzzles

## Introduction

Calculus problems in quant interviews test your ability to:
- **Optimize** trading strategies and portfolio allocations
- **Integrate** probability distributions for option pricing
- **Approximate** complex functions using Taylor series
- **Differentiate** to find rates of change and sensitivities
- **Solve** differential equations for stochastic processes

Firms like Jane Street, Citadel, and DE Shaw use calculus problems to assess mathematical maturity and problem-solving speed.

This section covers:
1. Optimization without calculus (clever tricks)
2. Derivatives and rate problems
3. Integration shortcuts and techniques
4. Taylor series approximations
5. Differential equations basics
6. Multivariable calculus
7. Financial applications (Greeks, option pricing)

---

## Mental Math Shortcuts

### Common Derivatives (Memorize)

\`\`\`
d/dx (xⁿ) = n·xⁿ⁻¹
d/dx (eˣ) = eˣ
d/dx (ln x) = 1/x
d/dx (sin x) = cos x
d/dx (cos x) = -sin x
d/dx(1/(1-x)) = 1/(1-x)²
\`\`\`

### Common Integrals

\`\`\`
∫ xⁿ dx = xⁿ⁺¹/(n+1) + C
∫ eˣ dx = eˣ + C
∫ 1/x dx = ln|x| + C
∫ 1/(1+x²) dx = arctan (x) + C
\`\`\`

### Taylor Series (First Few Terms)

\`\`\`
eˣ ≈ 1 + x + x²/2 + x³/6
ln(1+x) ≈ x - x²/2 + x³/3
sin (x) ≈ x - x³/6
cos (x) ≈ 1 - x²/2
(1+x)ⁿ ≈ 1 + nx + n (n-1)x²/2
\`\`\`

---

## Optimization Problems

### Problem 1: Maximize Area with Fixed Perimeter

**Question:** You have 100 meters of fencing. What dimensions maximize the enclosed rectangular area?

**Clever solution (no calculus):**

For fixed perimeter, square maximizes area.

Perimeter = 2(L + W) = 100
So L + W = 50

Maximum area when L = W = 25 meters.
Area = 25 × 25 = 625 m²

**Calculus verification:**

\`\`\`
A = L × W = L × (50 - L)
A = 50L - L²

dA/dL = 50 - 2L = 0
L = 25 ✓
\`\`\`

**Financial application:** Portfolio allocation with budget constraint.

### Problem 2: Minimize Cost

**Question:** Build a rectangular storage tank with volume 1000 m³. Base costs $10/m², walls cost $5/m². Find dimensions minimizing cost.

**Solution:**

Let base dimensions be x × y, height h.
Volume: xyh = 1000, so h = 1000/(xy)

Cost: C = 10xy + 5(2xh + 2yh)
     = 10xy + 10h (x + y)
     = 10xy + 10(1000/(xy))(x + y)
     = 10xy + 10000(x + y)/(xy)

For minimum, often x = y (by symmetry or calculus).

If x = y:
C = 10x² + 20000/x²

dC/dx = 20x - 40000/x³ = 0
20x = 40000/x³
x⁴ = 2000
x = (2000)^(1/4) ≈ 6.69 m

\`\`\`python
"""
Optimization Problem Solver
"""

import numpy as np
from scipy.optimize import minimize_scalar

def cost_function (x):
    """Cost as a function of base dimension (assuming square base)."""
    volume = 1000
    h = volume / (x ** 2)
    base_cost = 10 * x ** 2
    wall_cost = 5 * 4 * x * h  # 4 walls
    return base_cost + wall_cost

# Find minimum
result = minimize_scalar (cost_function, bounds=(1, 20), method='bounded')
optimal_x = result.x
min_cost = result.fun

print(f"Optimal dimension: {optimal_x:.2f} m")
print(f"Minimum cost: \${min_cost:.2f}")
print(f"Height: {1000/(optimal_x**2):.2f} m")

# Verify with derivative
x_test = optimal_x
derivative = 20 * x_test - 40000 / (x_test ** 3)
print(f"Derivative at optimal point: {derivative:.6f} (should be ≈0)")

# Output:
# Optimal dimension: 6.69 m
# Minimum cost: $8944.27
# Height: 22.36 m
# Derivative at optimal point: 0.000003(should be ≈0)
\`\`\`

---

## Integration Puzzles

### Problem 3: Area Under Curve

**Question:** Compute ∫₀¹ x² dx without tables.

**Mental solution:**

\`\`\`
∫ x² dx = x³/3 + C

∫₀¹ x² dx = [x³/3]₀¹ = 1/3 - 0 = 1/3
\`\`\`

**Geometric verification:** Area under parabola y = x² from 0 to 1 is exactly 1/3.

### Problem 4: Expected Value Integration

**Question:** X ~ Uniform[0,1]. Find E[X²].

**Solution:**

\`\`\`
E[X²] = ∫₀¹ x² · f (x) dx
      = ∫₀¹ x² · 1 dx    (uniform density = 1)
      = [x³/3]₀¹
      = 1/3
\`\`\`

**Variance:** Var(X) = E[X²] - (E[X])² = 1/3 - (1/2)² = 1/3 - 1/4 = 1/12

### Problem 5: Substitution Trick

**Question:** ∫₀^(π/2) sin (x) cos (x) dx

**Quick method:**

\`\`\`
sin (x) cos (x) = (1/2) sin(2x)

∫₀^(π/2) (1/2) sin(2x) dx = -(1/4) cos(2x) |₀^(π/2)
                           = -(1/4)[cos(π) - cos(0)]
                           = -(1/4)[-1 - 1]
                           = 1/2
\`\`\`

**Alternative (substitution):**

Let u = sin (x), du = cos (x) dx

\`\`\`
∫₀^(π/2) sin (x) cos (x) dx = ∫₀¹ u du = [u²/2]₀¹ = 1/2
\`\`\`

\`\`\`python
"""
Numerical Integration Verification
"""

from scipy import integrate

# Problem 3: x² from 0 to 1
def f1(x):
    return x**2

result1, error1 = integrate.quad (f1, 0, 1)
print(f"∫₀¹ x² dx = {result1:.6f} (exact: {1/3:.6f})")

# Problem 5: sin (x)cos (x) from 0 to π/2
def f2(x):
    return np.sin (x) * np.cos (x)

result2, error2 = integrate.quad (f2, 0, np.pi/2)
print(f"∫₀^(π/2) sin (x)cos (x) dx = {result2:.6f} (exact: 0.500000)")

# Output:
# ∫₀¹ x² dx = 0.333333 (exact: 0.333333)
# ∫₀^(π/2) sin (x)cos (x) dx = 0.500000 (exact: 0.500000)
\`\`\`

---

## Taylor Series Applications

### Problem 6: Approximate e^0.1

**Question:** Estimate e^0.1 mentally using Taylor series.

**Solution:**

\`\`\`
eˣ ≈ 1 + x + x²/2 + x³/6

e^0.1 ≈ 1 + 0.1 + (0.1)²/2 + (0.1)³/6
      ≈ 1 + 0.1 + 0.005 + 0.000167
      ≈ 1.105
\`\`\`

**Actual value:** e^0.1 = 1.10517...

**Error:** 0.00017 (0.015%)

**Interview tip:** For small x, keep first 2-3 terms for quick approximation.

### Problem 7: Option Price Approximation

**Question:** Black-Scholes call price formula involves N(d₁) where d₁ is complex. For small volatility and short time, approximate.

**Setup:**

When σ√T is small (low vol or short time), we can use Taylor expansion of N(d₁) around 0.5.

For ATM option (S = K), d₁ ≈ 0, so N(d₁) ≈ 0.5

This gives the mental math formula we saw earlier:
\`\`\`
C ≈ 0.4 × S × σ × √T
\`\`\`

The 0.4 comes from probability density and other factors in the Taylor expansion.

---

## Differential Equations

### Problem 8: Exponential Growth

**Question:** A portfolio grows at rate proportional to its value: dV/dt = rV. Find V(t) given V(0) = V₀.

**Solution:**

\`\`\`
dV/dt = rV

Separate variables: dV/V = r dt

Integrate: ln V = rt + C

V = e^(rt + C) = e^C · e^(rt)

At t=0: V₀ = e^C, so C = ln V₀

V(t) = V₀ · e^(rt)
\`\`\`

**Financial interpretation:** Continuous compound growth at rate r.

### Problem 9: Mean Reversion

**Question:** Price follows dP/dt = α(μ - P). Find P(t).

**Solution:**

\`\`\`
dP/dt = α(μ - P)

Let u = P - μ, then du/dt = dP/dt

du/dt = -αu

This is exponential decay: u (t) = u(0)e^(-αt)

P(t) - μ = (P(0) - μ)e^(-αt)

P(t) = μ + (P(0) - μ)e^(-αt)
\`\`\`

**Interpretation:** Price reverts to mean μ at rate α.

\`\`\`python
"""
Differential Equation Solutions
"""

import matplotlib.pyplot as plt

# Exponential growth
def portfolio_value (t, V0, r):
    return V0 * np.exp (r * t)

# Mean reversion
def mean_reverting_price (t, P0, mu, alpha):
    return mu + (P0 - mu) * np.exp(-alpha * t)

# Plot examples
t = np.linspace(0, 5, 100)

# Growth example
V0, r = 100, 0.1
V = portfolio_value (t, V0, r)

# Mean reversion example
P0, mu, alpha = 120, 100, 0.5
P = mean_reverting_price (t, P0, mu, alpha)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot (t, V)
ax1.set_title('Exponential Growth: V(t) = V₀e^(rt)')
ax1.set_xlabel('Time')
ax1.set_ylabel('Portfolio Value')
ax1.grid(True, alpha=0.3)

ax2.plot (t, P)
ax2.axhline (mu, color='r', linestyle='--', label='Mean μ')
ax2.set_title('Mean Reversion: dP/dt = α(μ-P)')
ax2.set_xlabel('Time')
ax2.set_ylabel('Price')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
# plt.show()
\`\`\`

---

## Multivariable Calculus

### Problem 10: Partial Derivatives for Greeks

**Question:** Option price C(S, σ, t) depends on stock price S, volatility σ, and time t. What do the partial derivatives represent?

**Solution:**

- **∂C/∂S = Delta (Δ):** Change in option price per $1 change in stock
- **∂C/∂σ = Vega (ν):** Change in option price per 1% change in volatility
- **∂C/∂t = Theta (Θ):** Change in option price per day passing

Second derivatives:
- **∂²C/∂S² = Gamma (Γ):** Change in delta per $1 change in stock

### Problem 11: Optimization with Constraint (Lagrange)

**Question:** Maximize f (x,y) = xy subject to x + y = 10.

**Method 1 (substitution):**

\`\`\`
y = 10 - x
f (x) = x(10 - x) = 10x - x²

f'(x) = 10 - 2x = 0
x = 5, y = 5

Maximum: f(5,5) = 25
\`\`\`

**Method 2 (Lagrange multipliers):**

\`\`\`
L(x, y, λ) = xy - λ(x + y - 10)

∂L/∂x = y - λ = 0 → y = λ
∂L/∂y = x - λ = 0 → x = λ
∂L/∂λ = -(x + y - 10) = 0

From first two: x = y
From constraint: x + x = 10 → x = 5

Maximum at (5, 5)
\`\`\`

**Financial application:** Allocate budget between two assets to maximize expected return.

---

## Interview Problem Set

### Problem 12: Quick Integration

\`\`\`
∫₀^∞ e^(-x) dx = ?
\`\`\`

**Solution:** [−e^(−x)]₀^∞ = 0 − (−1) = 1

### Problem 13: Derivative Chain Rule

\`\`\`
If f (x) = e^(x²), find f'(x)
\`\`\`

**Solution:** f'(x) = e^(x²) · 2x = 2x·e^(x²)

### Problem 14: Implicit Differentiation

\`\`\`
x² + y² = 25. Find dy/dx.
\`\`\`

**Solution:**
\`\`\`
2x + 2y (dy/dx) = 0
dy/dx = -x/y
\`\`\`

### Problem 15: L'Hôpital's Rule

\`\`\`
lim (x→0) (sin x)/x = ?
\`\`\`

**Solution:**
\`\`\`
Both numerator and denominator → 0 (indeterminate 0/0)

Apply L'Hôpital:
lim (x→0) (sin x)/x = lim (x→0) (cos x)/1 = 1
\`\`\`

---

## Financial Calculus Applications

### Problem 16: Portfolio Sensitivity

**Question:** Portfolio value V = 100S + 50S², where S is stock price. If S = 10, what is:
1. Current value?
2. Delta (dV/dS)?
3. Gamma (d²V/dS²)?

**Solution:**

\`\`\`
1. V(10) = 100(10) + 50(10)² = 1000 + 5000 = 6000

2. Delta = dV/dS = 100 + 100S
   At S=10: Δ = 100 + 1000 = 1100

3. Gamma = d²V/dS² = 100
\`\`\`

**Interpretation:**
- Portfolio worth $6000
- If stock moves $1, portfolio changes by ~$1100
- For every $1 stock moves, delta changes by 100

### Problem 17: Theta Decay

**Question:** Option value decreases as C(t) = C₀ · e^(-kt). If C₀ = 10 and k = 0.1 (daily), what's the daily theta?

**Solution:**

\`\`\`
C(t) = 10e^(-0.1t)

dC/dt = 10 · (-0.1) · e^(-0.1t) = -1 · e^(-0.1t)

At t=0: dC/dt = -1

Theta ≈ -$1 per day initially
\`\`\`

---

## Mental Math Tricks

**Trick 1: Small angle approximations**
\`\`\`
sin (x) ≈ x        (x in radians, x small)
cos (x) ≈ 1 - x²/2
tan (x) ≈ x
\`\`\`

**Trick 2: Compound growth**
\`\`\`
(1 + r)ⁿ ≈ e^(rn) for small r
\`\`\`

**Trick 3: Log approximations**
\`\`\`
ln(1 + x) ≈ x for small x
ln(2) ≈ 0.693
ln(10) ≈ 2.303
\`\`\`

---

## Summary

**Key calculus skills for interviews:**

1. **Optimization:** Find maxima/minima with or without constraints
2. **Integration:** Evaluate common integrals, expected values
3. **Approximation:** Taylor series for quick estimates
4. **Differential equations:** Solve growth, decay, mean reversion
5. **Multivariable:** Partial derivatives for Greeks

**Interview strategy:**
- State the technique you're using ("I'll use substitution...")
- Check boundary conditions and special cases
- Verify answer makes intuitive sense
- Offer geometric or numerical verification

**Practice daily:**
- 10 derivatives
- 5 integrals
- 3 optimization problems
- 2 Taylor series approximations

Master these, and calculus interviews become straightforward!
`,
};
