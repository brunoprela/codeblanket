/**
 * Multivariable Calculus Section
 */

export const multivariablecalculusSection = {
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
};
