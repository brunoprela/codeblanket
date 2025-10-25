/**
 * Integration Basics Section
 */

export const integrationbasicsSection = {
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

**Part 1**: If F'(x) = f (x), then:
∫ₐᵇ f (x)dx = F(b) - F(a)

**Part 2**: If g (x) = ∫ₐˣ f (t)dt, then g'(x) = f (x)

**Intuition**: Integration and differentiation are inverse operations.

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, cumulative_trapezoid

# Demonstrate Fundamental Theorem of Calculus

def f (x):
    """Function to integrate: f (x) = x²"""
    return x**2

def F(x):
    """Antiderivative: F(x) = x³/3"""
    return x**3 / 3

# Part 1: ∫ₐᵇ f (x)dx = F(b) - F(a)
a, b = 1, 3

# Analytical integration
analytical_integral = F(b) - F(a)
print(f"Part 1: Fundamental Theorem of Calculus")
print(f"∫₁³ x² dx = F(3) - F(1) = {F(b)} - {F(a)} = {analytical_integral:.4f}")

# Numerical integration (verification)
numerical_integral, error = quad (f, a, b)
print(f"Numerical verification: {numerical_integral:.4f}")
print(f"Error: {abs (analytical_integral - numerical_integral):.2e}")

# Part 2: d/dx [∫ₐˣ f (t)dt] = f (x)
print(f"\\nPart 2: Derivative of integral")

def integral_function (x):
    """g (x) = ∫₁ˣ t² dt"""
    result, _ = quad (f, 1, x)
    return result

# Test at x = 2
x_test = 2.0
h = 1e-7

# Derivative of integral (numerical)
g_x = integral_function (x_test)
g_x_h = integral_function (x_test + h)
derivative_of_integral = (g_x_h - g_x) / h

# Original function value
f_x = f (x_test)

print(f"At x = {x_test}:")
print(f"d/dx [∫₁ˣ t² dt] = {derivative_of_integral:.4f}")
print(f"f (x) = x² = {f_x:.4f}")
print(f"Error: {abs (derivative_of_integral - f_x):.2e}")
\`\`\`

## Basic Integration Rules

**Power Rule**:
∫ xⁿ dx = xⁿ⁺¹/(n+1) + C (n ≠ -1)

**Constant Multiple**:
∫ cf (x)dx = c∫f (x)dx

**Sum/Difference**:
∫ [f (x) ± g (x)]dx = ∫f (x)dx ± ∫g (x)dx

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
    (sin (x), "sin (x)"),
    (cos (x), "cos (x)"),
    (exp (x), "eˣ"),
    (1/x, "1/x"),
    (x**2 + 3*x + 5, "x² + 3x + 5")
]

for func, name in functions:
    integral = integrate (func, x)
    print(f"∫ {name} dx = {integral} + C")
\`\`\`

## Definite vs Indefinite Integrals

**Indefinite Integral** (antiderivative):
∫ f (x)dx = F(x) + C

**Definite Integral** (area):
∫ₐᵇ f (x)dx = F(b) - F(a)

\`\`\`python
def visualize_definite_integral():
    """Visualize definite integral as area under curve"""
    
    # Function: f (x) = x² - 2x + 3
    def f (x):
        return x**2 - 2*x + 3
    
    def F(x):
        return x**3/3 - x**2 + 3*x
    
    a, b = 0, 3
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Left: Function and area
    x = np.linspace(-0.5, 3.5, 200)
    y = f (x)
    
    ax1.plot (x, y, 'b-', linewidth=2, label='f (x) = x² - 2x + 3')
    
    # Shade area under curve
    x_fill = np.linspace (a, b, 100)
    y_fill = f (x_fill)
    ax1.fill_between (x_fill, 0, y_fill, alpha=0.3, color='blue', label=f'∫₀³ f (x)dx')
    
    ax1.axvline (a, color='red', linestyle='--', alpha=0.7)
    ax1.axvline (b, color='red', linestyle='--', alpha=0.7)
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('f (x)')
    ax1.set_title('Definite Integral as Area')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: Antiderivative
    x2 = np.linspace(-0.5, 3.5, 200)
    y2 = [F(xi) for xi in x2]
    
    ax2.plot (x2, y2, 'g-', linewidth=2, label='F(x) = x³/3 - x² + 3x')
    ax2.plot([a, b], [F(a), F(b)], 'ro', markersize=8)
    ax2.axhline(F(a), color='red', linestyle='--', alpha=0.5, label=f'F({a}) = {F(a):.2f}')
    ax2.axhline(F(b), color='blue', linestyle='--', alpha=0.5, label=f'F({b}) = {F(b):.2f}')
    
    # Show difference
    ax2.annotate(', xy=(3.3, F(b)), xytext=(3.3, F(a)),
                arrowprops=dict (arrowstyle='<->', color='purple', lw=2))
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

For ∫ f (g(x))·g'(x)dx, let u = g (x):
∫ f (g(x))·g'(x)dx = ∫ f (u)du

\`\`\`python
def integration_by_substitution():
    """
    Example: ∫ 2x·cos (x²) dx
    Let u = x², then du = 2x dx
    ∫ cos (u) du = sin (u) + C = sin (x²) + C
    """
    
    # Symbolic
    x = symbols('x')
    integrand = 2*x * cos (x**2)
    result = integrate (integrand, x)
    print("Integration by Substitution:")
    print(f"∫ 2x·cos (x²) dx = {result} + C")
    
    # Verify numerically
    def f (x_val):
        return 2*x_val * np.cos (x_val**2)
    
    def F(x_val):
        return np.sin (x_val**2)
    
    a, b = 0, 2
    analytical = F(b) - F(a)
    numerical, _ = quad (f, a, b)
    
    print(f"\\n∫₀² 2x·cos (x²) dx:")
    print(f"Analytical: {analytical:.6f}")
    print(f"Numerical: {numerical:.6f}")
    print(f"Error: {abs (analytical - numerical):.2e}")

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
    integrand = x * exp (x)
    result = integrate (integrand, x)
    print("Integration by Parts:")
    print(f"∫ x·eˣ dx = {result} + C")
    
    # Verify
    def f (x_val):
        return x_val * np.exp (x_val)
    
    def F(x_val):
        return np.exp (x_val) * (x_val - 1)
    
    a, b = 0, 2
    analytical = F(b) - F(a)
    numerical, _ = quad (f, a, b)
    
    print(f"\\n∫₀² x·eˣ dx:")
    print(f"Analytical: {analytical:.6f}")
    print(f"Numerical: {numerical:.6f}")
    print(f"Error: {abs (analytical - numerical):.2e}")

integration_by_parts_example()
\`\`\`

## Numerical Integration

When analytical integration is impossible, use numerical methods.

### Riemann Sums

\`\`\`python
def riemann_sum (f, a, b, n, method='midpoint'):
    """
    Compute Riemann sum
    
    Methods:
    - 'left': Left endpoint
    - 'right': Right endpoint
    - 'midpoint': Midpoint rule
    """
    dx = (b - a) / n
    x = np.linspace (a, b, n+1)
    
    if method == 'left':
        return dx * sum (f(x[i]) for i in range (n))
    elif method == 'right':
        return dx * sum (f(x[i+1]) for i in range (n))
    elif method == 'midpoint':
        midpoints = (x[:-1] + x[1:]) / 2
        return dx * sum (f(m) for m in midpoints)
    else:
        raise ValueError (f"Unknown method: {method}")

# Test
def f (x):
    return np.exp(-x**2)

a, b = 0, 2
true_value, _ = quad (f, a, b)

print("Riemann Sums:")
print(f"True value: {true_value:.8f}")
print()

for n in [10, 100, 1000]:
    for method in ['left', 'right', 'midpoint']:
        approx = riemann_sum (f, a, b, n, method)
        error = abs (approx - true_value)
        print(f"n={n:4d}, {method:8s}: {approx:.8f}, error={error:.2e}")
    print()
\`\`\`

### Trapezoidal Rule

∫ₐᵇ f (x)dx ≈ (b-a)/(2n) · [f (x₀) + 2f (x₁) + ... + 2f (xₙ₋₁) + f (xₙ)]

\`\`\`python
from scipy.integrate import trapezoid

def trapezoidal_rule (f, a, b, n):
    """Trapezoidal rule for numerical integration"""
    x = np.linspace (a, b, n+1)
    y = f (x)
    return trapezoid (y, x)

print("Trapezoidal Rule:")
for n in [10, 100, 1000]:
    approx = trapezoidal_rule (f, a, b, n)
    error = abs (approx - true_value)
    print(f"n={n:4d}: {approx:.8f}, error={error:.2e}")
\`\`\`

### Simpson\'s Rule

More accurate: uses parabolic approximation.

∫ₐᵇ f (x)dx ≈ (b-a)/(3n) · [f (x₀) + 4f (x₁) + 2f (x₂) + 4f (x₃) + ... + f (xₙ)]

\`\`\`python
from scipy.integrate import simpson

def simpsons_rule (f, a, b, n):
    """Simpson's rule for numerical integration"""
    if n % 2 != 0:
        n += 1  # Simpson's requires even n
    x = np.linspace (a, b, n+1)
    y = f (x)
    return simpson (y, x=x)

print("\\nSimpson\'s Rule:")
for n in [10, 100, 1000]:
    approx = simpsons_rule (f, a, b, n)
    error = abs (approx - true_value)
    print(f"n={n:4d}: {approx:.8f}, error={error:.2e}")
\`\`\`

## Applications in ML: Expectation

Expectation of continuous random variable:
E[X] = ∫₋∞^∞ x·f (x)dx

where f (x) is the probability density function.

\`\`\`python
def compute_expectations():
    """Compute expectations via integration"""
    
    # Normal distribution: X ~ N(μ, σ²)
    mu, sigma = 2.0, 1.5
    
    def pdf (x):
        """Probability density function"""
        return (1 / (sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu)/sigma)**2)
    
    # E[X]
    def integrand_mean (x):
        return x * pdf (x)
    
    expected_value, _ = quad (integrand_mean, -np.inf, np.inf)
    print(f"Normal Distribution: N({mu}, {sigma}²)")
    print(f"E[X] = {expected_value:.6f} (true: {mu})")
    
    # E[X²]
    def integrand_second_moment (x):
        return x**2 * pdf (x)
    
    second_moment, _ = quad (integrand_second_moment, -np.inf, np.inf)
    print(f"E[X²] = {second_moment:.6f} (true: {mu**2 + sigma**2:.6f})")
    
    # Var(X) = E[X²] - (E[X])²
    variance = second_moment - expected_value**2
    print(f"Var(X) = {variance:.6f} (true: {sigma**2:.6f})")
    
    # Probability: P(X ∈ [a, b])
    a, b = 1, 3
    prob, _ = quad (pdf, a, b)
    print(f"\\nP({a} ≤ X ≤ {b}) = {prob:.6f}")

compute_expectations()
\`\`\`

## Applications in ML: Loss Functions

Many loss functions involve integrals (KL divergence, cross-entropy for continuous distributions).

\`\`\`python
def kl_divergence_continuous():
    """
    KL divergence for continuous distributions:
    D_KL(P||Q) = ∫ p (x) log (p(x)/q (x)) dx
    """
    
    # Two normal distributions
    # P ~ N(0, 1)
    # Q ~ N(1, 1.5²)
    
    mu_p, sigma_p = 0.0, 1.0
    mu_q, sigma_q = 1.0, 1.5
    
    def p (x):
        return (1/(sigma_p*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu_p)/sigma_p)**2)
    
    def q (x):
        return (1/(sigma_q*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu_q)/sigma_q)**2)
    
    def kl_integrand (x):
        p_x = p (x)
        q_x = q (x)
        if p_x < 1e-10:  # Avoid numerical issues
            return 0
        return p_x * np.log (p_x / (q_x + 1e-10))
    
    kl_numerical, _ = quad (kl_integrand, -10, 10)
    
    # Analytical formula for KL between two Gaussians:
    # D_KL(N(μ₁,σ₁²)||N(μ₂,σ₂²)) = log(σ₂/σ₁) + (σ₁² + (μ₁-μ₂)²)/(2σ₂²) - 1/2
    kl_analytical = np.log (sigma_q/sigma_p) + (sigma_p**2 + (mu_p-mu_q)**2)/(2*sigma_q**2) - 0.5
    
    print("KL Divergence: D_KL(P||Q)")
    print(f"P ~ N({mu_p}, {sigma_p}²)")
    print(f"Q ~ N({mu_q}, {sigma_q}²)")
    print(f"\\nNumerical (integration): {kl_numerical:.6f}")
    print(f"Analytical formula: {kl_analytical:.6f}")
    print(f"Error: {abs (kl_numerical - kl_analytical):.2e}")

kl_divergence_continuous()
\`\`\`

## Summary

**Key Concepts**:
- Fundamental Theorem: ∫ₐᵇ f (x)dx = F(b) - F(a)
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
};
