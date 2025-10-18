/**
 * Quiz questions for Multivariable Calculus section
 */

export const multivariablecalculusQuiz = [
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
];
