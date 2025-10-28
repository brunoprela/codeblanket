/**
 * Algebraic Expressions & Equations Section
 */

export const algebraicexpressionsSection = {
  id: 'algebraic-expressions',
  title: 'Algebraic Expressions & Equations',
  content: `
# Algebraic Expressions & Equations

## Introduction

Algebra forms the backbone of machine learning mathematics. From understanding linear regression equations to manipulating loss functions, algebraic thinking is essential. This section covers variables, expressions, equations, and their solutions—skills you'll use daily in ML work.

## Variables, Coefficients, and Constants

### Definitions

**Variable**: A symbol (usually x, y, z, or θ) representing an unknown or changing quantity.

**Coefficient**: A number multiplied by a variable.

**Constant**: A fixed numerical value.

**Example Expression**: \`3x² + 5x - 7\`
- Variables: x
- Coefficients: 3 (for x²), 5 (for x)
- Constant: -7

### Variables in Machine Learning

In ML, we use specific variable conventions:

| Symbol | Common Use | Example |
|--------|------------|---------|
| x | Input features | x₁, x₂, ..., xₙ |
| y | Output/target | y = f (x) |
| θ, w | Model parameters/weights | θ = [θ₀, θ₁, ..., θₙ] |
| α | Learning rate | α = 0.01 |
| λ | Regularization parameter | λ = 0.1 |
| ε | Error term or small constant | ε = 1e-8 |

**Python Example**:
\`\`\`python
import numpy as np

# Variables in linear regression: y = θ₀ + θ₁x₁ + θ₂x₂
# θ₀ is the intercept (constant term)
# θ₁, θ₂ are coefficients (weights)

theta_0 = 2.5  # intercept
theta_1 = 1.3  # coefficient for x₁
theta_2 = -0.7  # coefficient for x₂

# Input features
x1 = 3.0
x2 = 4.0

# Prediction
y_pred = theta_0 + theta_1 * x1 + theta_2 * x2
print(f"Prediction: y = {theta_0} + {theta_1}*{x1} + {theta_2}*{x2} = {y_pred}")

# Vectorized form (more efficient)
theta = np.array([theta_0, theta_1, theta_2])
x = np.array([1, x1, x2])  # Note: 1 for intercept
y_pred_vec = np.dot (theta, x)
print(f"Vectorized prediction: {y_pred_vec}")
\`\`\`

## Simplifying Expressions

### Combining Like Terms

**Like terms**: Terms with the same variables raised to the same powers.

**Example**: \`3x + 5x - 2x = (3 + 5 - 2)x = 6x\`

**Python Implementation**:
\`\`\`python
from sympy import symbols, simplify, expand

x = symbols('x')

# Expression: 3x + 5x - 2x
expr = 3*x + 5*x - 2*x
simplified = simplify (expr)
print(f"3x + 5x - 2x = {simplified}")

# More complex expression
expr2 = 2*x**2 + 3*x + 4*x**2 - x + 5
simplified2 = simplify (expr2)
print(f"2x² + 3x + 4x² - x + 5 = {simplified2}")

# Application: Combining gradient terms
theta = symbols('theta')
gradient_term1 = 2 * theta - 3
gradient_term2 = 4 * theta + 1
gradient_term3 = -theta + 5

total_gradient = simplify (gradient_term1 + gradient_term2 + gradient_term3)
print(f"\\nTotal gradient: {total_gradient}")
\`\`\`

### Distributive Property

**Rule**: \`a (b + c) = ab + ac\`

**Python Example**:
\`\`\`python
from sympy import symbols, expand, factor

x, y = symbols('x y')

# Expanding
expr = 2*x * (3*x + 4)
expanded = expand (expr)
print(f"2x(3x + 4) = {expanded}")

# Factoring (reverse)
expr2 = 6*x**2 + 8*x
factored = factor (expr2)
print(f"6x² + 8x = {factored}")

# ML application: Expanding loss function
# L(θ) = (y - θx)²
theta, x_var, y_var = symbols('theta x y')
loss = (y_var - theta * x_var)**2
expanded_loss = expand (loss)
print(f"\\nExpanded loss: (y - θx)² = {expanded_loss}")
\`\`\`

### Factoring

Common factoring patterns:

1. **Common factor**: \`ax + ay = a (x + y)\`
2. **Difference of squares**: \`a² - b² = (a + b)(a - b)\`
3. **Perfect square**: \`a² + 2ab + b² = (a + b)²\`
4. **Quadratic**: \`ax² + bx + c = a (x - r₁)(x - r₂)\`

\`\`\`python
from sympy import symbols, factor, expand

x = symbols('x')

# Common factor
expr1 = 3*x**2 + 6*x
print(f"Factor {expr1}: {factor (expr1)}")

# Difference of squares
expr2 = x**2 - 9
print(f"Factor {expr2}: {factor (expr2)}")

# Perfect square
expr3 = x**2 + 6*x + 9
print(f"Factor {expr3}: {factor (expr3)}")

# Quadratic
expr4 = x**2 - 5*x + 6
print(f"Factor {expr4}: {factor (expr4)}")

# Verify by expanding back
factored = factor (expr4)
expanded_back = expand (factored)
print(f"Expand {factored}: {expanded_back}")
\`\`\`

## Solving Linear Equations

### Single Variable

**Standard form**: \`ax + b = 0\`
**Solution**: \`x = -b/a\` (provided a ≠ 0)

**Example**: \`3x - 12 = 0\`
\`\`\`
3x = 12
x = 4
\`\`\`

**Python Example**:
\`\`\`python
from sympy import symbols, Eq, solve

x = symbols('x')

# Equation: 3x - 12 = 0
equation = Eq(3*x - 12, 0)
solution = solve (equation, x)
print(f"Solution to 3x - 12 = 0: x = {solution}")

# More complex: 2(x + 3) = 4x - 10
equation2 = Eq(2*(x + 3), 4*x - 10)
solution2 = solve (equation2, x)
print(f"Solution to 2(x + 3) = 4x - 10: x = {solution2}")

# Verification
x_val = solution2[0]
left_side = 2*(x_val + 3)
right_side = 4*x_val - 10
print(f"Verification: {left_side} = {right_side}")
\`\`\`

### Application: Solving for Learning Rate

In gradient descent, we want to find when gradient = 0:

\`\`\`python
# Find learning rate α such that new_loss = target_loss
# new_loss = old_loss - α * gradient

theta, alpha, gradient = symbols('theta alpha gradient')
old_loss = symbols('old_loss')
target_loss = symbols('target_loss')

# Equation: old_loss - α * gradient = target_loss
equation = Eq (old_loss - alpha * gradient, target_loss)
alpha_solution = solve (equation, alpha)
print(f"Optimal α = {alpha_solution[0]}")

# Numerical example
old_loss_val = 10.0
target_loss_val = 8.0
gradient_val = 5.0

alpha_val = (old_loss_val - target_loss_val) / gradient_val
print(f"\\nNumerical: α = {alpha_val}")
\`\`\`

## Quadratic Equations

### Standard Form

**ax² + bx + c = 0** where a ≠ 0

### Quadratic Formula

**x = (-b ± √(b² - 4ac)) / (2a)**

**Discriminant**: Δ = b² - 4ac
- If Δ > 0: Two distinct real solutions
- If Δ = 0: One repeated real solution
- If Δ < 0: Two complex solutions

**Python Implementation**:
\`\`\`python
import numpy as np
from sympy import symbols, solve, Eq

def solve_quadratic (a, b, c):
    """Solve quadratic equation ax² + bx + c = 0"""
    discriminant = b**2 - 4*a*c

    if discriminant > 0:
        x1 = (-b + np.sqrt (discriminant)) / (2*a)
        x2 = (-b - np.sqrt (discriminant)) / (2*a)
        return f"Two solutions: x₁ = {x1:.4f}, x₂ = {x2:.4f}"
    elif discriminant == 0:
        x = -b / (2*a)
        return f"One solution: x = {x:.4f}"
    else:
        real_part = -b / (2*a)
        imag_part = np.sqrt(-discriminant) / (2*a)
        return f"Complex solutions: x = {real_part:.4f} ± {imag_part:.4f}i"

# Examples
print("1. x² - 5x + 6 = 0")
print(solve_quadratic(1, -5, 6))

print("\\n2. x² - 6x + 9 = 0")
print(solve_quadratic(1, -6, 9))

print("\\n3. x² + x + 1 = 0")
print(solve_quadratic(1, 1, 1))

# Using SymPy
x = symbols('x')
equation = x**2 - 5*x + 6
solutions = solve (equation, x)
print(f"\\nSymPy solution: {solutions}")
\`\`\`

### Application: Finding Optimal Parameters

Many optimization problems reduce to quadratic equations:

\`\`\`python
# Example: Minimize f(θ) = θ² - 4θ + 5
# Find θ where f'(θ) = 0

theta = symbols('theta')
f = theta**2 - 4*theta + 5

# Take derivative
f_prime = f.diff (theta)
print(f"f(θ) = {f}")
print(f"f'(θ) = {f_prime}")

# Solve f'(θ) = 0
optimal_theta = solve (f_prime, theta)
print(f"Optimal θ = {optimal_theta}")

# Evaluate minimum value
min_value = f.subs (theta, optimal_theta[0])
print(f"Minimum value: f({optimal_theta[0]}) = {min_value}")

# Verify with NumPy
theta_range = np.linspace(-1, 5, 100)
f_values = theta_range**2 - 4*theta_range + 5
optimal_idx = np.argmin (f_values)
print(f"\\nNumerical verification: θ ≈ {theta_range[optimal_idx]:.4f}")
\`\`\`

## Systems of Equations

### Two Variables

**System**:
\`\`\`
a₁x + b₁y = c₁
a₂x + b₂y = c₂
\`\`\`

**Methods**:
1. Substitution
2. Elimination
3. Matrix methods (covered in Linear Algebra)

**Python Example - Substitution Method**:
\`\`\`python
from sympy import symbols, Eq, solve

x, y = symbols('x y')

# System:
# 2x + 3y = 8
# x - y = 1

eq1 = Eq(2*x + 3*y, 8)
eq2 = Eq (x - y, 1)

# Solve system
solution = solve((eq1, eq2), (x, y))
print(f"Solution: {solution}")
print(f"x = {solution[x]}, y = {solution[y]}")

# Verification
x_val, y_val = solution[x], solution[y]
print(f"\\nVerification:")
print(f"2({x_val}) + 3({y_val}) = {2*x_val + 3*y_val}")
print(f"{x_val} - {y_val} = {x_val - y_val}")
\`\`\`

**Python Example - Matrix Method (preview)**:
\`\`\`python
import numpy as np

# System in matrix form: Ax = b
# [2  3] [x]   [8]
# [1 -1] [y] = [1]

A = np.array([[2, 3],
              [1, -1]])
b = np.array([8, 1])

# Solve using linear algebra
solution = np.linalg.solve(A, b)
print(f"Matrix solution: x = {solution[0]:.4f}, y = {solution[1]:.4f}")

# Verification
result = A @ solution
print(f"Verification: Ax = {result}, b = {b}")
print(f"Match? {np.allclose (result, b)}")
\`\`\`

### Three Variables

**System**:
\`\`\`
a₁x + b₁y + c₁z = d₁
a₂x + b₂y + c₂z = d₂
a₃x + b₃y + c₃z = d₃
\`\`\`

**Python Example**:
\`\`\`python
from sympy import symbols, Eq, solve
import numpy as np

# Using SymPy
x, y, z = symbols('x y z')

# System:
# x + y + z = 6
# 2x - y + z = 3
# x + 2y - z = 2

eq1 = Eq (x + y + z, 6)
eq2 = Eq(2*x - y + z, 3)
eq3 = Eq (x + 2*y - z, 2)

solution = solve((eq1, eq2, eq3), (x, y, z))
print(f"SymPy solution: {solution}")

# Using NumPy (faster for numerical problems)
A = np.array([[1,  1,  1],
              [2, -1,  1],
              [1,  2, -1]])
b = np.array([6, 3, 2])

solution_np = np.linalg.solve(A, b)
print(f"\\nNumPy solution: x={solution_np[0]:.4f}, y={solution_np[1]:.4f}, z={solution_np[2]:.4f}")
\`\`\`

### Application: Multi-variable Linear Regression

In linear regression with 3 features, we solve for θ₀, θ₁, θ₂:

\`\`\`python
# Normal equation: θ = (XᵀX)⁻¹Xᵀy

# Sample data
X = np.array([
    [1, 2, 3],    # Sample 1: [feature1, feature2, feature3]
    [1, 3, 4],    # Sample 2
    [1, 4, 5],    # Sample 3
    [1, 5, 6],    # Sample 4
])
X_with_intercept = np.c_[np.ones(X.shape[0]), X]  # Add column of 1s for intercept

y = np.array([10, 15, 20, 25])

# Solve normal equation
XtX = X_with_intercept.T @ X_with_intercept
Xty = X_with_intercept.T @ y
theta = np.linalg.solve(XtX, Xty)

print(f"Learned parameters: θ₀={theta[0]:.4f}, θ₁={theta[1]:.4f}, θ₂={theta[2]:.4f}, θ₃={theta[3]:.4f}")

# Make prediction
x_new = np.array([1, 3, 4, 5])  # Include 1 for intercept
y_pred = x_new @ theta
print(f"Prediction for {x_new[1:]}: {y_pred:.4f}")
\`\`\`

## Real-World Problem Modeling

### Example 1: Portfolio Optimization

**Problem**: You have $10,000 to invest in stocks (S) and bonds (B). You want:
- At least $3,000 in bonds
- Stock investment should be at most 2× bond investment
- Expected return: 8% for stocks, 5% for bonds
- Maximize total return

**Mathematical Model**:
\`\`\`
Variables: S (stock), B (bonds)
Constraints:
  S + B = 10,000
  B ≥ 3,000
  S ≤ 2B
Objective: Maximize 0.08S + 0.05B
\`\`\`

\`\`\`python
from scipy.optimize import linprog

# This is a preview of optimization (covered in Calculus module)
# For now, we solve the constraint equation

# From S + B = 10,000, we get S = 10,000 - B
# From S ≤ 2B: 10,000 - B ≤ 2B → B ≥ 10,000/3 ≈ 3,333.33

B = 3334  # Round up to meet constraint
S = 10000 - B

print(f"Investment: \${S:,} in stocks, \\$\{B:,} in bonds")
print(f"Expected return: \${0.08*S + 0.05*B:,.2f}")
print(f"Return rate: {(0.08*S + 0.05*B)/10000*100:.2f}%")
\`\`\`

### Example 2: Break-Even Analysis for Trading

**Problem**: A trading strategy has:
- Fixed costs: $1,000/month (software, data feeds)
- Variable cost: $2 per trade
- Revenue: $5 per profitable trade
- How many trades needed to break even?

**Mathematical Model**:
\`\`\`
Let n = number of trades
Cost = 1000 + 2n
Revenue = 5n
Break-even: Revenue = Cost
5n = 1000 + 2n
\`\`\`

\`\`\`python
# Solve: 5n = 1000 + 2n
# 3n = 1000
# n = 1000/3

n = symbols('n')
equation = Eq(5*n, 1000 + 2*n)
breakeven = solve (equation, n)
print(f"Break-even trades: {breakeven[0]:.0f} trades")
print(f"Rounded up: {np.ceil (float (breakeven[0]))} trades")

# Verification
trades = 334
cost = 1000 + 2*trades
revenue = 5*trades
profit = revenue - cost
print(f"\\nAt {trades} trades:")
print(f"Cost: \\$\{cost:,}")
print(f"Revenue: \\$\{revenue:,}")
print(f"Profit: \\$\{profit:,}")
\`\`\`

## Using SymPy for Symbolic Math

SymPy is Python\'s symbolic mathematics library:

\`\`\`python
from sympy import symbols, solve, Eq, simplify, expand, factor, diff

# Define symbols
x, y, theta, alpha = symbols('x y theta alpha')

# 1. Solve equations
eq = x**2 + 3*x - 10
solution = solve (eq, x)
print(f"Solve x² + 3x - 10 = 0: {solution}")

# 2. Simplify expressions
expr = (x + 1)**2 - (x**2 + 2*x + 1)
simplified = simplify (expr)
print(f"\\nSimplify (x+1)² - (x²+2x+1): {simplified}")

# 3. Expand expressions
expr2 = (x + y)**3
expanded = expand (expr2)
print(f"\\nExpand (x+y)³: {expanded}")

# 4. Factor expressions
expr3 = x**3 - y**3
factored = factor (expr3)
print(f"\\nFactor x³ - y³: {factored}")

# 5. Derivatives (preview)
f = x**3 - 2*x**2 + x - 5
f_prime = diff (f, x)
print(f"\\nDerivative of {f}: {f_prime}")

# 6. Substitute values
result = f.subs (x, 2)
print(f"f(2) = {result}")

# 7. Systems of equations
eq1 = Eq(2*x + y, 10)
eq2 = Eq (x - y, 2)
sol = solve((eq1, eq2), (x, y))
print(f"\\nSystem solution: {sol}")
\`\`\`

## Common Mistakes and How to Avoid Them

### 1. Sign Errors

\`\`\`python
# Wrong: -2x + 5 = 13 → x = -4 (forgot to flip sign)
# Right: -2x = 8 → x = -4

x = symbols('x')
eq = Eq(-2*x + 5, 13)
correct = solve (eq, x)
print(f"Correct solution: {correct}")
\`\`\`

### 2. Order of Operations

\`\`\`python
# Wrong: 2 + 3 * 4 = 20 (added first)
# Right: 2 + 3 * 4 = 14 (multiply first)

result_wrong = (2 + 3) * 4  # 20
result_right = 2 + 3 * 4     # 14
print(f"Wrong (left to right): {result_wrong}")
print(f"Right (PEMDAS): {result_right}")
\`\`\`

### 3. Division by Zero

\`\`\`python
def safe_solve (numerator, denominator):
    """Safely solve ax + b = 0"""
    if denominator == 0:
        if numerator == 0:
            return "Infinite solutions"
        else:
            return "No solution"
    return -numerator / denominator

# 0x + 5 = 0 (no solution)
print(f"0x + 5 = 0: {safe_solve(5, 0)}")

# 0x + 0 = 0 (infinite solutions)
print(f"0x + 0 = 0: {safe_solve(0, 0)}")

# 2x + 6 = 0 (x = -3)
print(f"2x + 6 = 0: x = {safe_solve(6, 2)}")
\`\`\`

### 4. Squaring Both Sides (Introduces Extra Solutions)

\`\`\`python
# Solving √x = -2 by squaring both sides
# (√x)² = (-2)² → x = 4
# But √4 = 2, not -2! Need to check solution.

x = symbols('x')
# Original: sqrt (x) = -2 (no real solution)
# After squaring: x = 4 (but this doesn't satisfy original!)

from sympy import sqrt
eq_original = Eq (sqrt (x), -2)
eq_squared = Eq (x, 4)

print(f"Original equation: {eq_original}")
print(f"After squaring: {eq_squared}")
print(f"Check x=4 in original: √4 = 2 ≠ -2")
print(f"Conclusion: No real solution exists")
\`\`\`

## Summary

- **Variables** represent unknowns; **coefficients** multiply variables; **constants** are fixed values
- **Simplifying** combines like terms and applies distributive property
- **Linear equations** (ax + b = 0) have one solution: x = -b/a
- **Quadratic equations** (ax² + bx + c = 0) use quadratic formula
- **Systems of equations** can be solved by substitution, elimination, or matrix methods
- **SymPy** provides powerful symbolic math capabilities in Python
- **Real-world modeling** translates word problems into mathematical equations
- Always **verify solutions** by substituting back into original equation

These algebraic skills are fundamental for:
- Deriving machine learning algorithms
- Solving optimization problems
- Understanding model equations
- Implementing gradient descent
- Debugging mathematical errors in code
`,
};
