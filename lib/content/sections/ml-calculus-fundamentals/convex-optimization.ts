/**
 * Convex Optimization Section
 */

export const convexoptimizationSection = {
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
    x_conv = np.cos (theta)
    y_conv = np.sin (theta)
    
    ax1.fill (x_conv, y_conv, alpha=0.3, color='blue')
    ax1.plot (x_conv, y_conv, 'b-', linewidth=2)
    
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
    x_outer = 1.5 * np.cos (theta1)
    y_outer = 1.5 * np.sin (theta1)
    
    theta2 = np.linspace(0, 2*np.pi, 100)
    x_inner = 1.2 * np.cos (theta2) + 0.5
    y_inner = 1.2 * np.sin (theta2)
    
    # Create crescent by masking
    mask = x_outer > x_inner[0]
    
    ax2.fill (x_outer, y_outer, alpha=0.3, color='red')
    ax2.plot (x_outer, y_outer, 'r-', linewidth=2)
    
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
f(θx + (1-θ)y) ≤ θf (x) + (1-θ)f (y)

**Intuition**: Chord above graph (function curves upward).

**Examples:**
- Convex: x², eˣ, |x|, -log (x) for x > 0
- Not convex: sin (x), x³

\`\`\`python
def check_convexity():
    """Check if functions are convex"""
    
    # Test convexity via second derivative
    def is_convex_1d (f, df, ddf, x_range):
        """For 1D: f is convex if f'(x) ≥ 0 for all x"""
        x = np.linspace(*x_range, 100)
        second_deriv = ddf (x)
        return np.all (second_deriv >= -1e-10), second_deriv
    
    # Test functions
    functions = [
        {
            'name': 'x²',
            'f': lambda x: x**2,
            'df': lambda x: 2*x,
            'ddf': lambda x: 2 * np.ones_like (x),
            'range': (-2, 2)
        },
        {
            'name': 'eˣ',
            'f': lambda x: np.exp (x),
            'df': lambda x: np.exp (x),
            'ddf': lambda x: np.exp (x),
            'range': (-2, 2)
        },
        {
            'name': 'sin (x)',
            'f': lambda x: np.sin (x),
            'df': lambda x: np.cos (x),
            'ddf': lambda x: -np.sin (x),
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
f (y) ≥ f (x) + ∇f (x)ᵀ(y - x) for all x, y

**Interpretation**: Function lies above its tangent plane.

\`\`\`python
def visualize_first_order_condition():
    """Visualize first-order convexity condition"""
    
    # Convex function: f (x) = x²
    def f (x):
        return x**2
    
    def grad_f (x):
        return 2*x
    
    # Point for tangent
    x0 = 1.0
    
    # Create plot
    x = np.linspace(-2, 3, 200)
    y_func = f (x)
    
    # Tangent line at x0
    y_tangent = f (x0) + grad_f (x0) * (x - x0)
    
    plt.figure (figsize=(10, 6))
    plt.plot (x, y_func, 'b-', linewidth=2, label='f (x) = x²')
    plt.plot (x, y_tangent, 'r--', linewidth=2, label=f'Tangent at x={x0}')
    plt.plot (x0, f (x0), 'ro', markersize=10, label=f'Point ({x0}, {f (x0)})')
    
    # Shade region showing f (x) ≥ tangent
    plt.fill_between (x, y_tangent, y_func, where=(y_func >= y_tangent), 
                     alpha=0.3, color='green', label='f (x) ≥ tangent')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("First-Order Convexity Condition: f (y) ≥ f (x) + f'(x)(y-x)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('first_order_condition.png', dpi=150, bbox_inches='tight')
    print("Function lies above tangent everywhere → Convex")

visualize_first_order_condition()
\`\`\`

## Second-Order Condition

For twice-differentiable f, **f is convex** iff:
∇²f (x) ⪰ 0 (Hessian is positive semidefinite) for all x

**Check**: All eigenvalues of Hessian ≥ 0.

\`\`\`python
def check_convexity_hessian():
    """Check convexity using Hessian"""
    
    # f (x, y) = x² + 2y²
    def f (xy):
        x, y = xy
        return x**2 + 2*y**2
    
    def hessian (xy):
        return np.array([[2, 0], [0, 4]])
    
    # Check at random points
    test_points = [
        np.array([0.0, 0.0]),
        np.array([1.0, 2.0]),
        np.array([-1.5, 0.5])
    ]
    
    print("Convexity Check via Hessian:")
    print("="*60)
    print("f (x, y) = x² + 2y²\\n")
    
    all_convex = True
    for point in test_points:
        H = hessian (point)
        eigenvalues = np.linalg.eigvalsh(H)
        is_psd = np.all (eigenvalues >= -1e-10)
        
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
minimize f (x)
subject to gᵢ(x) ≤ 0, i = 1, ..., m
           hⱼ(x) = 0, j = 1, ..., p

where f and all gᵢ are convex, and hⱼ are affine.

**Key property**: Every local minimum is a global minimum!

\`\`\`python
def convex_vs_nonconvex_optimization():
    """Compare convex vs non-convex optimization"""
    
    # Convex: f (x) = x²
    def f_convex (x):
        return x**2
    
    # Non-convex: f (x) = x⁴ - 4x² + x
    def f_nonconvex (x):
        return x**4 - 4*x**2 + x
    
    x = np.linspace(-3, 3, 300)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Convex
    ax1.plot (x, f_convex (x), 'b-', linewidth=2)
    ax1.plot(0, f_convex(0), 'ro', markersize=15, label='Global minimum')
    ax1.set_title('Convex Function: x²')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f (x)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.text(0, 2, 'Only one minimum\\n (global)', ha='center', fontsize=12,
             bbox=dict (boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Non-convex
    ax2.plot (x, f_nonconvex (x), 'r-', linewidth=2)
    
    # Find local minima numerically
    from scipy.optimize import minimize_scalar
    local_mins = []
    for x0 in [-2, 0, 2]:
        result = minimize_scalar (f_nonconvex, bounds=(-3, 3), method='bounded')
        local_mins.append((result.x, result.fun))
    
    # Mark minima
    ax2.plot(-1.6, f_nonconvex(-1.6), 'go', markersize=12, label='Local minimum')
    ax2.plot(1.7, f_nonconvex(1.7), 'ro', markersize=15, label='Global minimum')
    ax2.set_title('Non-Convex Function: x⁴ - 4x² + x')
    ax2.set_xlabel('x')
    ax2.set_ylabel('f (x)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.text(0, 3, 'Multiple local minima\\n (hard to optimize)', ha='center', fontsize=12,
             bbox=dict (boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
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
    X = np.random.randn (n, 2)
    true_w = np.array([2.0, -1.5])
    y = X @ true_w + np.random.randn (n) * 0.5
    
    # Loss function: MSE is convex
    def loss (w):
        predictions = X @ w
        return 0.5 * np.mean((predictions - y)**2)
    
    def gradient (w):
        predictions = X @ w
        return X.T @ (predictions - y) / n
    
    def hessian (w):
        # H = (1/n) X^T X (independent of w!)
        return X.T @ X / n
    
    # Check convexity
    w_test = np.array([1.0, 1.0])
    H = hessian (w_test)
    eigenvalues = np.linalg.eigvalsh(H)
    
    print("Linear Regression Convexity:")
    print("="*60)
    print(f"Loss function: L(w) = (1/2n)||Xw - y||²")
    print(f"\\nHessian H = (1/n)X^T X:")
    print(H)
    print(f"\\nEigenvalues: {eigenvalues}")
    print(f"All eigenvalues ≥ 0: {np.all (eigenvalues >= -1e-10)}")
    print("→ Loss is convex! Gradient descent guaranteed to find global minimum.")
    
    # Gradient descent
    w = np.zeros(2)
    lr = 0.1
    losses = []
    
    for i in range(100):
        losses.append (loss (w))
        w = w - lr * gradient (w)
    
    print(f"\\nTrue weights: {true_w}")
    print(f"Learned weights: {w}")
    print(f"Final loss: {loss (w):.6f}")

linear_regression_convex()
\`\`\`

### Logistic Regression (Convex)

\`\`\`python
def logistic_regression_convex():
    """Logistic regression has convex loss"""
    
    # Generate binary classification data
    np.random.seed(42)
    n = 100
    X = np.random.randn (n, 2)
    true_w = np.array([1.5, -1.0])
    logits = X @ true_w
    y = (logits + np.random.randn (n) * 0.5 > 0).astype (float)
    
    def sigmoid (z):
        return 1 / (1 + np.exp(-np.clip (z, -500, 500)))
    
    # Binary cross-entropy loss (convex!)
    def loss (w):
        z = X @ w
        p = sigmoid (z)
        return -np.mean (y * np.log (p + 1e-10) + (1-y) * np.log(1-p + 1e-10))
    
    def gradient (w):
        z = X @ w
        p = sigmoid (z)
        return X.T @ (p - y) / n
    
    def hessian (w):
        """Hessian of logistic loss"""
        z = X @ w
        p = sigmoid (z)
        S = np.diag (p * (1 - p))
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
        H = hessian (w_test)
        eigenvalues = np.linalg.eigvalsh(H)
        is_psd = np.all (eigenvalues >= -1e-10)
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
1. **Stationarity**: ∇f (x*) + Σλᵢ∇gᵢ(x*) + Σνⱼ∇hⱼ(x*) = 0
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
    minimize f (x) = x²
    subject to g (x) = x - 1 ≤ 0
    
Solution:
    x* = 1 (at constraint boundary)
    λ* > 0 (constraint active)
    
Verification:
1. Stationarity: ∇f (x*) + λ∇g (x*) = 0
   2x* + λ·1 = 0
   2(1) + λ = 0 → λ = -2 (wrong sign!)
   
   Actually, we have a mistake. Let me recalculate...
   
   At x* = 1:
   ∇L = 2x + λ = 0
   2(1) + λ = 0 → λ = -2
   
   But we need λ ≥ 0 for minimization!
   
   Correct formulation: constraint should be g (x) = 1 - x ≤ 0
   Then: 2x* - λ = 0 at x* = 1 → λ = 2 ✓
   
2. Primal feasibility: g (x*) = 1 - 1 = 0 ≤ 0 ✓
3. Dual feasibility: λ = 2 ≥ 0 ✓  
4. Complementary slackness: λ·g (x*) = 2·0 = 0 ✓

All KKT conditions satisfied → x* = 1 is optimal!
    """)

kkt_example()
\`\`\`

## Summary

**Key Concepts**:
- **Convex functions**: f(θx + (1-θ)y) ≤ θf (x) + (1-θ)f (y)
- **First-order**: f (y) ≥ f (x) + ∇f (x)ᵀ(y-x)
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
};
