/**
 * Quiz questions for Algebraic Expressions & Equations section
 */

export const algebraicexpressionsQuiz = [
  {
    id: 'dq1-normal-equation',
    question:
      'Explain why the normal equation in linear regression θ = (X^T X)^(-1) X^T y involves solving a system of equations. What does each component represent, and when might this approach fail?',
    sampleAnswer: `The normal equation is the closed-form solution to linear regression that minimizes the mean squared error. Let's break down why it involves systems of equations:

**Derivation Context**:
When we have m training examples with n features, we're trying to find parameters θ = [θ₀, θ₁, ..., θₙ] that minimize:
L(θ) = (1/2m) Σ(y⁽ⁱ⁾ - h_θ(x⁽ⁱ⁾))²

Taking the derivative with respect to each θⱼ and setting to zero gives us:
∂L/∂θⱼ = 0 for j = 0, 1, ..., n

This creates n+1 equations with n+1 unknowns—a system of equations!

**Component Breakdown**:
- **X**: m × (n+1) design matrix where each row is one training example
- **X^T X**: (n+1) × (n+1) square matrix of feature correlations
- **(X^T X)^(-1)**: Inverse used to "solve" the system
- **X^T y**: (n+1) × 1 vector representing target correlations
- **θ**: (n+1) × 1 vector of parameters we're solving for

**Why it's a System of Equations**:
X^T X θ = X^T y represents:
[sum of feature cross-products] × [parameters] = [sum of feature-target products]

This is equivalent to n+1 simultaneous equations. For example, with 2 features:
θ₀ Σ1 + θ₁ Σx₁ + θ₂ Σx₂ = Σy
θ₀ Σx₁ + θ₁ Σx₁² + θ₂ Σx₁x₂ = Σx₁y
θ₀ Σx₂ + θ₁ Σx₁x₂ + θ₂ Σx₂² = Σx₂y

**When This Approach Fails**:

1. **Non-invertible X^T X** (singular matrix):
   - Happens when features are linearly dependent
   - Or when m < n (fewer examples than features)
   - Solution: Use regularization (Ridge/Lasso) or pseudo-inverse

2. **Computational Complexity**:
   - Computing (X^T X)^(-1) is O(n³)
   - With millions of features, this is prohibitive
   - Solution: Use iterative methods like gradient descent

3. **Numerical Instability**:
   - If X^T X is ill-conditioned (nearly singular)
   - Small changes in data cause large changes in θ
   - Solution: Feature scaling, regularization

4. **Memory Constraints**:
   - X^T X requires O(n²) memory
   - For very large n, may not fit in memory
   - Solution: Stochastic gradient descent

**Practical Implications**:
- Use normal equation when: n ≤ 10,000 and X^T X is invertible
- Use gradient descent when: n > 10,000 or need online learning
- Modern deep learning almost never uses normal equation due to scale`,
    keyPoints: [
      'Normal equation solves n+1 simultaneous linear equations',
      'X^T X θ = X^T y is the matrix form of the system',
      'Fails when X^T X is not invertible (singular)',
      'Computationally expensive O(n³) for large n',
      'Gradient descent is preferred for high-dimensional problems',
    ],
  },
  {
    id: 'dq2-quadratic-optimization',
    question:
      'Many machine learning optimization problems reduce to quadratic equations. Explain how this happens in the context of finding optimal learning rates or convergence analysis, and why understanding quadratic equations helps debug training issues.',
    sampleAnswer: `Quadratic equations appear frequently in ML optimization due to second-order Taylor approximations and the geometry of loss landscapes:

**1. Learning Rate Optimization**:

Consider gradient descent: θ_{t+1} = θ_t - α∇L(θ_t)

Using Taylor expansion around θ_t:
L(θ_{t+1}) ≈ L(θ_t) + ∇L^T(θ_{t+1} - θ_t) + (1/2)(θ_{t+1} - θ_t)^T H (θ_{t+1} - θ_t)

Where H is the Hessian (second derivatives). Substituting θ_{t+1}:
L(θ_t - α∇L) ≈ L(θ_t) - α||∇L||² + (α²/2)∇L^T H ∇L

This is a **quadratic equation in α**! To find optimal α, set derivative to zero:
dL/dα = -||∇L||² + α ∇L^T H ∇L = 0
α = ||∇L||² / (∇L^T H ∇L)

**2. Convergence Analysis**:

For strongly convex functions with Lipschitz continuous gradients:
L(θ) ≥ L(θ*) + (μ/2)||θ - θ*||²

The convergence rate of gradient descent is:
||θ_t - θ*||² ≤ (1 - αμ)^t ||θ₀ - θ*||²

This is a geometric series (related to sequences), but analyzing when (1 - αμ) < 1 involves:
αμ < 2 (quadratic inequality)

**3. Newton's Method**:

Newton's method uses quadratic approximation explicitly:
θ_{t+1} = θ_t - H^(-1)∇L(θ_t)

This assumes the loss is locally quadratic, which is why it converges faster near optima but can fail in non-convex regions.

**4. Debugging Training Issues**:

**Problem: Loss Diverges (Explodes)**
- Quadratic analysis: If α > 2/λ_max (λ_max = largest eigenvalue of H)
- Solution: Reduce learning rate or use adaptive methods

**Problem: Loss Plateaus**
- Could be at local minimum where ∇L ≈ 0
- Check second derivative (Hessian): 
  - If H > 0: true minimum
  - If H < 0: maximum (shouldn't happen)
  - If H ≈ 0: saddle point (common in deep learning)

**Problem: Oscillating Loss**
- Learning rate too large: overshooting minimum
- Quadratic bowl analogy: α causes "bouncing" back and forth
- Solution: Reduce α or use momentum

**5. Practical Example - Debugging Polynomial Features**:

\`\`\`python
# With polynomial features, loss landscape becomes more complex
# Understanding quadratic terms helps tune regularization

def analyze_loss_curvature(X, y, theta):
    """Compute second derivative (curvature) of loss"""
    n = len(y)
    # For MSE: L = (1/2n)||Xθ - y||²
    # Hessian: H = (1/n)X^T X
    H = (X.T @ X) / n
    eigenvalues = np.linalg.eigvalsh(H)
    
    print(f"Condition number: {eigenvalues.max() / eigenvalues.min():.2e}")
    print(f"Max safe learning rate: {2 / eigenvalues.max():.2e}")
    return eigenvalues
\`\`\`

**6. Portfolio Optimization Context**:

In mean-variance portfolio theory:
min (1/2)θ^T Σ θ - μ^T θ

Where Σ is covariance matrix. This is a **quadratic program**!
- θ: portfolio weights
- Quadratic term: risk (variance)
- Linear term: expected return

Understanding this helps:
- Know when solution is unique (Σ positive definite)
- Understand risk-return tradeoff geometrically
- Debug numerical issues in portfolio solvers

**Key Takeaways**:
Quadratic equations in ML represent local curvature, enable learning rate theory, and help diagnose optimization problems through eigenvalue analysis.`,
    keyPoints: [
      'Taylor expansions create quadratic loss approximations',
      'Optimal learning rate comes from solving quadratic equation',
      'Eigenvalues of Hessian determine convergence properties',
      'Understanding curvature helps debug divergence and oscillation',
      'Quadratic programs appear in portfolio optimization',
    ],
  },
  {
    id: 'dq3-symbolic-numeric',
    question:
      'Compare symbolic computation (SymPy) versus numeric computation (NumPy) for solving algebraic equations. When should you use each in machine learning and trading applications? Provide specific scenarios.',
    sampleAnswer: `The choice between symbolic (SymPy) and numeric (NumPy) computation depends on whether you need exact analytical solutions or fast numerical approximations:

**Symbolic Computation (SymPy)**

**Strengths**:
1. **Exact Solutions**: No floating-point errors
2. **Analytical Insights**: See formulas, not just numbers
3. **Simplification**: Algebraic manipulation and simplification
4. **Derivative Automation**: Exact symbolic derivatives

**Limitations**:
1. **Slow**: Orders of magnitude slower than numeric
2. **Memory Intensive**: Expressions grow large
3. **Limited Scalability**: Struggles with high dimensions
4. **Can't Always Solve**: Some equations have no closed form

**Numeric Computation (NumPy)**

**Strengths**:
1. **Speed**: Highly optimized, uses BLAS/LAPACK
2. **Scalability**: Handles millions of variables
3. **Memory Efficient**: Fixed-size arrays
4. **Robust Algorithms**: Iterative solvers for any problem

**Limitations**:
1. **Numerical Error**: Floating-point precision issues
2. **No Insights**: Just numbers, not formulas
3. **Initial Guess**: Iterative methods need starting point
4. **Convergence**: May fail to converge or find wrong solution

**ML Scenarios - When to Use Each**:

**Use SymPy**:

1. **Deriving Custom Loss Functions**:
\`\`\`python
# Derive gradient symbolically
from sympy import symbols, diff
x, y, theta = symbols('x y theta')
loss = (y - theta*x)**2
gradient = diff(loss, theta)  # Exact: 2*x*(theta*x - y)
\`\`\`

2. **Theoretical Analysis**:
- Proving convergence properties
- Analyzing algorithm behavior
- Computing exact update rules

3. **Small-Scale Prototypes**:
- Research papers: show exact formulas
- Educational purposes
- Verifying numeric implementations

4. **Equation Simplification**:
\`\`\`python
# Simplify complex regularization terms
expr = (theta**2 + 2*theta + 1) / (theta + 1)
simplified = simplify(expr)  # theta + 1
\`\`\`

**Use NumPy**:

1. **Training Neural Networks**:
\`\`\`python
# Millions of parameters - symbolic impossible
weights = np.random.randn(1000, 1000)
gradients = compute_numerical_gradient(weights)  # Fast
\`\`\`

2. **Real-Time Predictions**:
- Trading: Need microsecond latency
- Production systems: Speed critical
- Online learning: Continuous updates

3. **Large-Scale Optimization**:
\`\`\`python
# Solve 10,000 × 10,000 system
A = generate_feature_matrix(10000)
b = target_vector
theta = np.linalg.solve(A, b)  # Fast
\`\`\`

4. **Empirical Analysis**:
- Backtesting strategies
- Monte Carlo simulations
- Performance benchmarking

**Trading Application Scenarios**:

**Scenario 1: Developing New Strategy (Use SymPy)**
\`\`\`python
# Derive exact formula for expected return
from sympy import symbols, integrate, exp
p, mu, sigma = symbols('p mu sigma', real=True, positive=True)

# Option payoff with probability distribution
payoff = max(p - K, 0)  # Call option
expected = integrate(payoff * normal_pdf(p, mu, sigma), (p, -oo, oo))
# Get exact formula for pricing
\`\`\`

**Scenario 2: Backtesting (Use NumPy)**
\`\`\`python
# Process millions of price ticks
prices = load_market_data()  # Shape: (1000000, 100)
returns = np.log(prices[1:] / prices[:-1])
sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
# Fast numerical computation
\`\`\`

**Scenario 3: Risk Analysis (Both)**
\`\`\`python
# Symbolic: Derive VaR formula
from sympy import quantile
var_formula = quantile(returns_distribution, alpha)

# Numeric: Compute actual VaR
returns_array = np.random.normal(mu, sigma, 10000)
var_numeric = np.percentile(returns_array, 5)  # 5% VaR
\`\`\`

**Scenario 4: Order Execution (Use NumPy)**
\`\`\`python
# Real-time: Must execute in microseconds
optimal_order_size = compute_vwap_slice(
    current_volume=volume_array,  # NumPy
    target_quantity=1000,
    time_remaining=60
)
# Symbolic would timeout
\`\`\`

**Scenario 5: Academic Research Paper (Use SymPy)**
\`\`\`python
# Show exact theoretical optimal bid-ask spread
s, lambda_b, lambda_a, sigma = symbols('s lambda_b lambda_a sigma')
expected_profit = derive_market_maker_profit(s, lambda_b, lambda_a, sigma)
optimal_spread = solve(diff(expected_profit, s), s)
# Include in paper as formula
\`\`\`

**Hybrid Approach**:

Best practice: Use both!

1. **Derive with SymPy, Implement with NumPy**:
\`\`\`python
# Step 1: Symbolic derivation
gradient_formula = derive_gradient_symbolically()

# Step 2: Convert to numeric function
gradient_func = lambdify(theta, gradient_formula)

# Step 3: Use in training
for epoch in range(epochs):
    theta -= learning_rate * gradient_func(theta)  # Fast numeric
\`\`\`

2. **Validate Numeric with Symbolic**:
\`\`\`python
# Compute numerically
numeric_result = solve_with_numpy()

# Verify with symbolic (on small subset)
symbolic_result = solve_with_sympy(small_sample)
assert np.isclose(numeric_result, symbolic_result)
\`\`\`

**Summary**:
- **SymPy**: Prototyping, derivation, analysis, small problems
- **NumPy**: Production, training, large-scale, real-time
- **Both**: Derive formulas symbolically, implement numerically
- **Trading**: NumPy for execution, SymPy for strategy development`,
    keyPoints: [
      'SymPy: exact solutions, insights, slow, small-scale',
      'NumPy: fast, scalable, approximate, production-ready',
      'Use SymPy for derivation and theoretical analysis',
      'Use NumPy for training, backtesting, and real-time systems',
      'Hybrid approach: derive symbolically, implement numerically',
    ],
  },
];
