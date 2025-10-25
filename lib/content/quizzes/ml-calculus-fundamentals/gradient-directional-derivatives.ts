/**
 * Quiz questions for Gradient & Directional Derivatives section
 */

export const gradientdirectionalderivativesQuiz = [
  {
    id: 'grad-dir-disc-1',
    question:
      'Prove that the gradient points in the direction of maximum rate of increase. Why is this fundamental to optimization?',
    hint: 'Consider the directional derivative formula and how dot products are maximized.',
    sampleAnswer: `**Proof that Gradient is Direction of Maximum Increase:**

The directional derivative in direction û is:
D_û f = ∇f · û = ||∇f|| ||û|| cos(θ) = ||∇f|| cos(θ)

where θ is the angle between ∇f and û, and ||û|| = 1 (unit vector).

To maximize D_û f, we need to maximize cos(θ). Since -1 ≤ cos(θ) ≤ 1:
- Maximum: cos(θ) = 1 when θ = 0° (û parallel to ∇f)
- Minimum: cos(θ) = -1 when θ = 180° (û opposite to ∇f)

Therefore:
- Maximum rate of increase: Direction of ∇f, magnitude ||∇f||
- Maximum rate of decrease: Direction of -∇f, magnitude ||∇f||

**Why This Matters for Optimization:**

1. **Gradient Descent Foundation**: Moving in direction -∇f gives steepest decrease, explaining why gradient descent works.

2. **Convergence Speed**: Larger ||∇f|| means faster decrease is possible. When ||∇f|| → 0, we're near a critical point.

3. **Step Size Selection**: Learning rate α should be chosen considering ||∇f||. If ||∇f|| is large, smaller α prevents overshooting.

4. **Saddle Point Escape**: In high dimensions, the gradient points away from saddle points along eigenvector corresponding to negative eigenvalue of Hessian.

5. **Loss Landscape Understanding**: The gradient gives local geometric information about the loss surface. Following -∇f takes the "straight downhill" path.

**Practical Implications:**

- **Adaptive Methods (Adam, RMSprop)**: Scale gradient by running average of magnitude, making step sizes more uniform.
- **Momentum**: Accumulates past gradient directions, building velocity to overcome local irregularities.
- **Natural Gradient**: Uses different metric (Fisher information) to define "steepest descent" in parameter space, accounting for model structure.

The gradient's directionality is why first-order optimization works despite high dimensionality - we don't need to search all directions, just follow ∇f.`,
    keyPoints: [
      'Directional derivative D_û f = ||∇f|| cos(θ) maximized when θ = 0',
      'Gradient direction gives maximum increase, negative gradient gives maximum decrease',
      'Gradient magnitude is the maximum rate of change',
      'Foundation for gradient descent and all first-order methods',
      'Enables efficient optimization in high dimensions',
    ],
  },
  {
    id: 'grad-dir-disc-2',
    question:
      'Explain how momentum methods use gradient direction more effectively than vanilla gradient descent. Include the mathematical formulation and intuition.',
    hint: 'Consider how velocity accumulates and the physical analogy of a ball rolling down a hill.',
    sampleAnswer: `**Momentum Method Formulation:**

**Vanilla Gradient Descent:**
θ_{t+1} = θ_t - α∇L(θ_t)

**Gradient Descent with Momentum:**
v_{t+1} = βv_t - α∇L(θ_t)
θ_{t+1} = θ_t + v_{t+1}

where:
- v is velocity (accumulated direction)
- β ∈ [0, 1) is momentum coefficient (typically 0.9)
- α is learning rate

**How It Works:**

1. **Direction Accumulation**: Velocity is exponential moving average of gradients:
   v_t = -α(∇L_t + β∇L_{t-1} + β²∇L_{t-2} + ...)
   
   Past gradients contribute with exponentially decaying weights.

2. **Consistent Directions Accelerate**: If gradients point in similar directions over time, velocity builds up:
   - Consistent downhill: |v| increases → faster progress
   - Oscillating: opposing gradients cancel → damped oscillations

3. **Physical Analogy**: Ball rolling downhill:
   - Gradient = instantaneous slope
   - Velocity = accumulated motion
   - Momentum = resistance to direction change
   - Heavy ball (high β) smooths out bumps

**Advantages Over Vanilla GD:**

**1. Faster Convergence in Ravines:**

Consider f (x,y) = x²/2 + 10y² (steep in y, shallow in x):

Vanilla GD:
- Large gradient in y-direction causes oscillation
- Must use small α to avoid divergence
- Slow progress in x-direction

With Momentum:
- Oscillations in y dampen out (opposing gradients cancel)
- Consistent gradient in x builds velocity
- Net effect: smoother, faster path

**2. Escape from Plateaus:**

On flat regions where ||∇L|| ≈ 0:

Vanilla GD: θ_{t+1} ≈ θ_t (stuck)
Momentum: v_t ≠ 0 from past gradients → continues moving

**3. Reduced Sensitivity to Noise:**

In stochastic gradient descent with noisy gradients:
- Individual gradients may be poor estimates
- Exponential average smooths noise
- More stable optimization trajectory

**Mathematical Analysis:**

Consider quadratic loss L(θ) = ½θᵀQθ:

Without momentum:
- Eigenvalues of Q determine convergence
- Convergence rate ≈ (λ_max - λ_min)/(λ_max + λ_min)
- Poor when condition number λ_max/λ_min is large

With momentum:
- Effective damping of oscillations
- Better convergence rate
- Like solving with preconditioner

**Practical Considerations:**

**Hyperparameters:**
- β = 0.9 (common): 90% of past velocity retained
- β = 0.99 (for large batches): longer memory
- α typically needs to be smaller than vanilla GD

**Modern Variants:**
- **Nesterov Momentum**: "Look ahead" before computing gradient
  v_{t+1} = βv_t - α∇L(θ_t + βv_t)
  θ_{t+1} = θ_t + v_{t+1}
  
- **Adam**: Combines momentum with adaptive learning rates
  - First moment (momentum): m_t = β₁m_{t-1} + (1-β₁)∇L_t
  - Second moment (variance): v_t = β₂v_{t-1} + (1-β₂)(∇L_t)²

**Visual Understanding:**

Imagine gradient descent as a sequence of independent steps, each responding only to local gradient. Momentum adds memory - the optimization "remembers" where it's been going and continues in consistent directions while damping oscillations.

This makes momentum especially valuable for:
- Ill-conditioned problems (elongated loss contours)
- Noisy gradients (stochastic optimization)
- Saddle point escape (velocity carries through flat regions)
- Deep networks (accumulates signal through many layers)`,
    keyPoints: [
      'Momentum accumulates exponential average of gradients',
      'Accelerates in consistent directions, dampens oscillations',
      'Physical analogy: heavy ball rolling with inertia',
      'Escapes plateaus and smooths noisy gradients',
      'Essential for training deep networks efficiently',
      'Modern optimizers (Adam) extend momentum concept',
    ],
  },
  {
    id: 'grad-dir-disc-3',
    question:
      'Explain projected gradient descent for constrained optimization. Why is it important in machine learning, and how does it differ from penalty methods?',
    hint: 'Consider optimizing on manifolds like the unit sphere, and applications like orthogonal weights or probability simplexes.',
    sampleAnswer: `**Projected Gradient Descent (PGD):**

**Algorithm:**
1. Compute gradient: g = ∇f (x_t)
2. Take gradient step: y = x_t - αg
3. Project onto constraint set: x_{t+1} = Proj_C(y)

where Proj_C(y) = argmin_{x∈C} ||x - y||² (closest point in C)

**Why Project Rather Than Penalize?**

**Penalty Method:**
Minimize: f (x) + λ·penalty (constraint violation)

Example: Optimize on unit sphere
L(x) = f (x) + λ(||x||² - 1)²

Problems:
- Never exactly satisfies constraints
- Requires tuning penalty weight λ
- Can make optimization harder (adds curvature)
- Doesn't scale well to hard constraints

**Projected Gradient:**
- Always feasible (constraint always satisfied)
- No hyperparameter tuning for constraint
- Separates objective from constraints
- Often has closed-form projection

**Common Projections in ML:**

**1. Unit Sphere (Orthogonality):**

Constraint: ||x|| = 1
Projection: Proj (y) = y/||y||

Application: Spectral normalization in GANs, weight normalization

\`\`\`python
def project_sphere (x):
    return x / np.linalg.norm (x)
\`\`\`

**2. Probability Simplex:**

Constraint: xᵢ ≥ 0, Σxᵢ = 1
Projection: Euclidean projection onto simplex (O(n log n) algorithm)

Application: Attention weights, mixture models, portfolio optimization

**3. Box Constraints:**

Constraint: a ≤ x ≤ b
Projection: Proj (y)ᵢ = clip (yᵢ, aᵢ, bᵢ)

Application: Bounded parameters, adversarial perturbations (ε-ball)

\`\`\`python
def project_box (x, a, b):
    return np.clip (x, a, b)
\`\`\`

**4. Low-Rank Matrices:**

Constraint: rank(X) ≤ r
Projection: SVD truncation

Application: Matrix factorization, collaborative filtering

**ML Applications:**

**1. Adversarial Training:**

Generate adversarial examples within ε-ball:
- Maximize loss: x_adv = x + δ
- Constraint: ||δ||_∞ ≤ ε
- Use projected gradient ascent

\`\`\`python
for step in range (num_steps):
    grad = compute_gradient (x_adv, target)
    x_adv = x_adv + alpha * np.sign (grad)
    # Project to ε-ball around x
    x_adv = np.clip (x_adv, x - epsilon, x + epsilon)
\`\`\`

**2. Fairness Constraints:**

Optimize with fairness constraints:
- Demographic parity: P(ŷ=1|A=0) = P(ŷ=1|A=1)
- Project parameters to satisfy fairness metrics
- Ensures fairness while minimizing loss

**3. Sparse Learning:**

L0 constraint (at most k non-zeros):
- After gradient step, keep top-k magnitudes
- Project to k-sparse vectors

\`\`\`python
def project_sparse (x, k):
    indices = np.argsort (np.abs (x))[-k:]
    x_sparse = np.zeros_like (x)
    x_sparse[indices] = x[indices]
    return x_sparse
\`\`\`

**4. Orthogonal Weights:**

Constraint: WᵀW = I (orthogonal matrix)
Projection: W_{new} = U @ Vᵀ (via SVD: W = USVᵀ)

Application: Improving gradient flow, preventing vanishing gradients

**Theoretical Properties:**

**Convergence:**
For convex f and convex constraint set C:
- PGD converges to optimal solution
- Rate depends on Lipschitz constant and projection quality

**Computational Cost:**
- Projection must be efficient
- O(n) to O(n²) typically acceptable
- Can be parallelized

**Comparison Summary:**

| Method | Constraint Satisfaction | Hyperparameters | Complexity |
|--------|------------------------|-----------------|------------|
| Penalty | Approximate | λ tuning needed | Simple |
| Lagrange | Exact (at optimum) | Dual variables | Complex |
| Projected GD | Always exact | None | Moderate |

**Best Practices:**

1. **Use projection when:**
   - Hard constraints required
   - Efficient projection available
   - Constraints are convex

2. **Use penalty when:**
   - Soft constraints acceptable
   - Projection expensive
   - Complex constraint interactions

3. **Hybrid approach:**
   - Project for critical constraints
   - Penalize for soft constraints
   - Example: Project to feasible region, penalize violations within

**Modern Extensions:**

- **Mirror Descent**: Project in dual space (natural gradients)
- **Proximal Methods**: Generalize projection (add regularization)
- **Barrier Methods**: Approach boundary from interior
- **Frank-Wolfe**: Linear minimization oracle instead of projection

Projected gradient descent is essential when mathematical or physical constraints must be exactly satisfied, making it invaluable for robust ML systems.`,
    keyPoints: [
      'PGD alternates gradient step with projection onto constraints',
      'Always feasible (constraints satisfied at every iteration)',
      'No penalty hyperparameters needed',
      'Common in adversarial training, fairness, sparsity',
      'Efficient for simple constraints (sphere, box, simplex)',
      'Theoretically grounded with convergence guarantees',
    ],
  },
];
