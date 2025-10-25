/**
 * Quiz questions for Integration Basics section
 */

export const integrationbasicsQuiz = [
  {
    id: 'integration-disc-1',
    question:
      'Explain why computing expectations and marginal probabilities in probabilistic models requires integration. How does this relate to Bayesian inference?',
    hint: 'Consider continuous probability distributions, normalization constants, and marginalizing out variables.',
    sampleAnswer: `**Integration in Probabilistic Models and Bayesian Inference:**

Integration is fundamental to probabilistic modeling because continuous probability distributions require integration for:
1. Normalization
2. Computing expectations
3. Marginalizing over variables
4. Bayesian inference

**1. Probability Normalization:**

A probability density function (PDF) must integrate to 1:

∫₋∞^∞ p (x) dx = 1

Example: Normal distribution
\`\`\`
p (x) = (1/Z) · exp(-0.5·((x-μ)/σ)²)
\`\`\`

Where Z = σ√(2π) is the normalizing constant (computed via integration).

**Why needed:**
Without correct normalization, probabilities would be meaningless. Integration ensures the distribution represents valid probabilities.

**2. Computing Expectations:**

The expected value of any function g(X):

E[g(X)] = ∫ g (x)·p (x) dx

**Examples in ML:**

**Mean:**
\`\`\`
E[X] = ∫ x·p (x) dx
\`\`\`

**Variance:**
\`\`\`
Var(X) = E[X²] - (E[X])² = ∫ x²·p (x) dx - (∫ x·p (x) dx)²
\`\`\`

**Loss:**
\`\`\`
E[Loss] = ∫ L(θ, x)·p (x|D) dx
\`\`\`

**Why needed:**
We rarely observe all possible values; integration computes average behavior over the entire distribution.

**3. Marginalizing Over Variables:**

For joint distribution p (x, y), the marginal distribution:

p (x) = ∫ p (x, y) dy

**Example:** Mixture of Gaussians

\`\`\`
p (x) = ∫ p (x|z)·p (z) dz
     = Σₖ πₖ·N(x|μₖ, σₖ²)
\`\`\`

where z is the latent cluster assignment.

**Why needed:**
Often, we have joint distributions over observed and latent variables. To reason about observed variables alone, we must integrate out (marginalize) latent variables.

**4. Bayesian Inference:**

Bayes' theorem for continuous parameters:

p(θ|D) = p(D|θ)·p(θ) / p(D)

where the **evidence** (denominator) requires integration:

p(D) = ∫ p(D|θ)·p(θ) dθ

**Components:**

**Posterior:** p(θ|D)
- Distribution over parameters given data
- What we want to compute

**Likelihood:** p(D|θ)
- Probability of data given parameters
- Easy to evaluate

**Prior:** p(θ)
- Our beliefs before seeing data
- Specified by modeler

**Evidence (marginal likelihood):** p(D)
- Normalizing constant
- **Requires integration over all θ**

**Why integration is hard:**

For high-dimensional θ (e.g., neural network weights), this integral is:
1. **Intractable**: No closed form
2. **High-dimensional**: Curse of dimensionality
3. **Expensive**: Requires many evaluations

**Example: Bayesian Linear Regression**

Model: y = Xw + ε, ε ~ N(0, σ²I)

**Likelihood:**
\`\`\`
p (y|X, w, σ²) = N(y|Xw, σ²I)
\`\`\`

**Prior:**
\`\`\`
p (w) = N(w|0, λ⁻¹I)
\`\`\`

**Posterior:**
\`\`\`
p (w|X, y) = p (y|X, w)·p (w) / p (y|X)
\`\`\`

**Evidence (integral!):**
\`\`\`
p (y|X) = ∫ p (y|X, w)·p (w) dw
\`\`\`

This integral can be computed analytically for linear models, but for most models (neural nets, etc.), it's intractable.

**5. Practical Solutions:**

Since exact integration is often impossible, we use approximations:

**Monte Carlo Integration:**
\`\`\`
∫ f (x)·p (x) dx ≈ (1/N) Σᵢ f (xᵢ), xᵢ ~ p (x)
\`\`\`

Sample from distribution, average function values.

**Variational Inference:**
Approximate intractable posterior p(θ|D) with simpler distribution q(θ):

\`\`\`
q*(θ) = argmin_q KL(q(θ) || p(θ|D))
\`\`\`

Convert integration problem to optimization problem.

**Markov Chain Monte Carlo (MCMC):**
Generate samples from posterior without computing normalizing constant:
- Metropolis-Hastings
- Hamiltonian Monte Carlo
- Gibbs sampling

**Laplace Approximation:**
Approximate posterior with Gaussian around MAP estimate:

\`\`\`
p(θ|D) ≈ N(θ|θ_MAP, H⁻¹)
\`\`\`

where H is the Hessian at θ_MAP.

**6. Concrete Example: Bayesian Neural Network**

Standard NN: Single weight estimate (point estimate)
Bayesian NN: Distribution over weights

**Predictive distribution:**
\`\`\`
p (y*|x*, D) = ∫ p (y*|x*, w)·p (w|D) dw
\`\`\`

This integral over all weight configurations:
- Provides uncertainty estimates
- Averages predictions over plausible models
- **Requires integration (intractable!)**

**Approximation (Monte Carlo dropout):**
\`\`\`
p (y*|x*, D) ≈ (1/T) Σₜ p (y*|x*, wₜ), wₜ ~ dropout
\`\`\`

Sample T forward passes with dropout, average predictions.

**7. Why This Matters:**

Integration is the **computational bottleneck** in Bayesian ML:
- Exact inference: only for simple models (Gaussians, conjugate priors)
- Approximate inference: necessary for deep learning

Modern ML balances:
- **Expressiveness**: Complex models (deep nets)
- **Tractability**: Approximate inference (variational, sampling)

Without efficient integration methods, Bayesian deep learning would be impossible.

**Key Insight:**
Integration in probabilistic models isn't just a mathematical detail—it's the central computational challenge. Advances in approximate inference (VAEs, normalizing flows, score-based models) are fundamentally about finding better ways to handle these integrals.`,
    keyPoints: [
      'Integration normalizes probability distributions',
      'Expectations computed via ∫ g (x)·p (x) dx',
      'Marginalization removes variables: p (x) = ∫ p (x,y) dy',
      'Bayesian evidence p(D) = ∫ p(D|θ)·p(θ) dθ is often intractable',
      'Approximate inference (MC, VI, MCMC) handles intractable integrals',
      'Integration is the computational bottleneck in Bayesian ML',
    ],
  },
  {
    id: 'integration-disc-2',
    question:
      "Compare and contrast different numerical integration methods (Riemann sums, trapezoidal rule, Simpson\'s rule, Monte Carlo). When is each most appropriate?",
    hint: 'Consider accuracy, computational cost, dimensionality, and function smoothness.',
    sampleAnswer: `**Comparison of Numerical Integration Methods:**

Numerical integration approximates ∫ₐᵇ f (x) dx when analytical solutions don't exist.

**1. Riemann Sums**

**Method:**
Divide [a,b] into n intervals, approximate as rectangles:

\`\`\`
∫ₐᵇ f (x) dx ≈ Σᵢ f (xᵢ*) · Δx
\`\`\`

Variants:
- Left endpoint: xᵢ* = xᵢ
- Right endpoint: xᵢ* = xᵢ₊₁
- Midpoint: xᵢ* = (xᵢ + xᵢ₊₁)/2

**Error:** O(1/n) for smooth functions

**Pros:**
- Simple to understand and implement
- Midpoint rule reasonably accurate

**Cons:**
- Slow convergence (need many points)
- Less accurate than higher-order methods

**When to use:**
- Educational purposes
- Quick rough estimates
- Non-smooth functions (midpoint rule)

**Example:**
\`\`\`python
# ∫₀¹ x² dx = 1/3
n = 100
dx = 1.0 / n
midpoint_sum = sum(((i + 0.5) * dx)**2 for i in range (n)) * dx
# midpoint_sum ≈ 0.3333
\`\`\`

**2. Trapezoidal Rule**

**Method:**
Approximate function with straight lines (trapezoids):

\`\`\`
∫ₐᵇ f (x) dx ≈ (Δx/2) · [f (x₀) + 2f (x₁) + ... + 2f (xₙ₋₁) + f (xₙ)]
\`\`\`

**Error:** O(1/n²) for smooth functions

**Pros:**
- Better accuracy than Riemann (quadratic convergence)
- Simple implementation
- Good for smooth functions

**Cons:**
- Less accurate than Simpson\'s - Requires function evaluation at endpoints

**When to use:**
- Smooth functions
- When Simpson's requirements not met (odd number of points)
- Moderate accuracy needed

**Example:**
\`\`\`python
from scipy.integrate import trapezoid
x = np.linspace(0, 1, 101)
y = x**2
result = trapezoid (y, x)  # result ≈ 0.333333
\`\`\`

**3. Simpson's Rule**

**Method:**
Approximate function with parabolas (quadratic interpolation):

\`\`\`
∫ₐᵇ f (x) dx ≈ (Δx/3) · [f (x₀) + 4f (x₁) + 2f (x₂) + 4f (x₃) + ... + f (xₙ)]
\`\`\`

**Error:** O(1/n⁴) for smooth functions

**Pros:**
- **Very accurate** for smooth functions (quartic convergence)
- Exact for polynomials up to degree 3
- Best deterministic low-D method

**Cons:**
- Requires even number of intervals
- Slightly more complex than trapezoidal
- Still suffers from curse of dimensionality

**When to use:**
- Smooth, well-behaved 1D or low-D integrals
- High accuracy required
- Computational budget allows many function evaluations

**Example:**
\`\`\`python
from scipy.integrate import simpson
x = np.linspace(0, 1, 101)
y = x**2
result = simpson (y, x=x)  # result ≈ 0.33333333 (very accurate!)
\`\`\`

**4. Monte Carlo Integration**

**Method:**
Sample random points, average function values:

\`\`\`
∫ₐᵇ f (x) dx ≈ (b-a) · (1/N) Σᵢ f (xᵢ), xᵢ ~ Uniform[a,b]
\`\`\`

**Error:** O(1/√N) - **independent of dimension!**

**Pros:**
- **Scales to high dimensions** (curse of dimensionality doesn't apply as strongly)
- Error independent of dimension
- Easy to implement
- Handles non-smooth functions
- Can importance sample (reduce variance)

**Cons:**
- Slow convergence (need 4× samples for 2× accuracy)
- Less accurate than Simpson\'s in 1D
- Requires random number generation
- Stochastic (different runs give different results)

**When to use:**
- **High-dimensional integrals** (d > 3)
- Irregular domains
- Non-smooth functions
- When many function evaluations are cheap
- Probabilistic models (expectations)

**Example:**
\`\`\`python
# ∫₀¹ x² dx using Monte Carlo
n_samples = 10000
x_samples = np.random.uniform(0, 1, n_samples)
mc_estimate = np.mean (x_samples**2)  # ≈ 0.333 ± 0.01
\`\`\`

**5. Comparison Table:**

| Method | Error | Best for | Dimension | Smoothness |
|--------|-------|----------|-----------|------------|
| **Riemann** | O(1/n) | Education | 1D | Any |
| **Trapezoidal** | O(1/n²) | Smooth 1D | 1D | Smooth |
| **Simpson's** | O(1/n⁴) | Very smooth 1D | 1D-2D | Very smooth |
| **Monte Carlo** | O(1/√N) | High-D | **Any D** | Any |

**6. Curse of Dimensionality:**

For d-dimensional integral using grid methods:
- Need n points per dimension
- Total points: n^d
- Exponential growth!

**Example:**
- 1D: 100 points → error O(1/100²) = 0.0001
- 10D: 100^10 = 10²⁰ points needed!

**Monte Carlo in high-D:**
- Error O(1/√N) **regardless of d**
- 10,000 samples → error ~0.01 in any dimension

**Why MC wins in high-D:**
\`\`\`
Simpson's in d dimensions: error = O(n^(-4/d))
Monte Carlo: error = O(1/√N)

For d=10, n=100:
- Simpson's: error ~ O(100^(-0.4)) ~ 0.1
- MC with 10,000 samples: error ~ 0.01
\`\`\`

**7. Advanced Methods:**

**Quasi-Monte Carlo:**
- Use low-discrepancy sequences (Sobol, Halton)
- Better than random sampling
- Error: O((log N)^d / N) better than O(1/√N)

**Importance Sampling:**
Sample from distribution q (x) that concentrates on important regions:
\`\`\`
∫ f (x) dx ≈ (1/N) Σᵢ f (xᵢ)/q (xᵢ), xᵢ ~ q
\`\`\`

**Adaptive Quadrature:**
- Refine grid in regions where function varies rapidly
- Used in scipy.integrate.quad

**8. Practical Decision Tree:**

\`\`\`
Is dimension d ≤ 3?
├─ Yes: Is function smooth?
│  ├─ Yes: Use Simpson\'s rule (best accuracy)
│  └─ No: Use Monte Carlo or midpoint Riemann
└─ No (high-D):
   ├─ Can afford many samples? → Monte Carlo
   ├─ Need variance reduction? → Importance sampling / Quasi-MC
   └─ Very high-D (d>20)? → MCMC or variational methods
\`\`\`

**9. Machine Learning Applications:**

**1D-2D smooth integrals:**
- Use Simpson's/adaptive quadrature
- Example: Computing loss over validation set

**Expectations over distributions:**
- Monte Carlo sampling
- Example: E_x~p[f (x)] ≈ (1/N)Σf (xᵢ), xᵢ~p

**High-dimensional integrals (Bayesian inference):**
- MCMC (Metropolis, HMC)
- Variational inference (convert to optimization)
- Example: p(D) = ∫p(D|θ)p(θ)dθ for 1M parameters

**Summary:**

**Low-dimensional + smooth:** Simpson's rule (O(1/n⁴) accuracy)
**High-dimensional:** Monte Carlo (dimension-independent O(1/√N))
**Very high-D:** MCMC/VI (specialized methods)

The key insight: **dimension determines method choice**. In ML, high dimensionality makes Monte Carlo and its variants (MCMC, VI) essential tools.`,
    keyPoints: [
      "Simpson\'s rule: Best for 1D smooth functions, O(1/n⁴) error",
      'Monte Carlo: Best for high-D, O(1/√N) error independent of dimension',
      'Curse of dimensionality: grid methods need n^d points',
      'MC avoids curse: error independent of dimension',
      "Rule: d≤3 use Simpson's, d>3 use Monte Carlo",
      'ML applications mostly high-D → MC/MCMC/VI dominate',
    ],
  },
  {
    id: 'integration-disc-3',
    question:
      'Explain the role of integration in deriving the cross-entropy loss and KL divergence. Why are these integrals often approximated in practice?',
    hint: 'Consider continuous vs discrete distributions, expectations, and computational tractability.',
    sampleAnswer: `**Integration in Cross-Entropy and KL Divergence:**

Both cross-entropy and KL divergence fundamentally involve integration (or summation for discrete distributions). Understanding this connection is crucial for ML theory and practice.

**1. Cross-Entropy: Definition**

For continuous distributions P and Q:

H(P, Q) = -∫ p (x) log q (x) dx

For discrete distributions:

H(P, Q) = -Σᵢ p (xᵢ) log q (xᵢ)

**Interpretation:**
- Measures expected log-likelihood under Q when true distribution is P
- Minimizing cross-entropy = maximizing likelihood

**2. KL Divergence: Definition**

**Continuous:**
\`\`\`
D_KL(P||Q) = ∫ p (x) log (p(x)/q (x)) dx
           = ∫ p (x) log p (x) dx - ∫ p (x) log q (x) dx
           = -H(P) + H(P, Q)
\`\`\`

**Discrete:**
\`\`\`
D_KL(P||Q) = Σᵢ p (xᵢ) log (p(xᵢ)/q (xᵢ))
\`\`\`

**Interpretation:**
- Measures "distance" from Q to P (not symmetric!)
- Expected log-ratio of probabilities
- Information gained when using true P instead of approximate Q

**3. Why These Are Integrals:**

Both are **expectations** over distribution P:

**Cross-entropy:**
\`\`\`
H(P, Q) = -E_P[log q(X)]
        = -∫ p (x) log q (x) dx
\`\`\`

**KL divergence:**
\`\`\`
D_KL(P||Q) = E_P[log (p(X)/q(X))]
           = ∫ p (x) log (p(x)/q (x)) dx
\`\`\`

**4. Example: Gaussian Distributions**

**Analytical KL (rare case with closed form!):**

For P = N(μ₁, σ₁²) and Q = N(μ₂, σ₂²):

\`\`\`
D_KL(P||Q) = log(σ₂/σ₁) + (σ₁² + (μ₁-μ₂)²)/(2σ₂²) - 1/2
\`\`\`

This comes from **evaluating the integral analytically**.

**Numerical verification:**
\`\`\`python
from scipy.integrate import quad

mu1, sigma1 = 0.0, 1.0
mu2, sigma2 = 1.0, 1.5

def p (x):
    return (1/(sigma1*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu1)/sigma1)**2)

def q (x):
    return (1/(sigma2*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu2)/sigma2)**2)

def kl_integrand (x):
    p_x, q_x = p (x), q (x)
    return p_x * np.log (p_x / q_x) if p_x > 1e-10 else 0

# Numerical integration
kl_numerical, _ = quad (kl_integrand, -10, 10)

# Analytical formula
kl_analytical = np.log (sigma2/sigma1) + (sigma1**2 + (mu1-mu2)**2)/(2*sigma2**2) - 0.5

print(f"KL (numerical): {kl_numerical:.6f}")
print(f"KL (analytical): {kl_analytical:.6f}")
\`\`\`

**5. Why Integration is Challenging:**

**Problem 1: High Dimensionality**

For images (d=784 for MNIST):
\`\`\`
D_KL(P||Q) = ∫...∫ p (x) log (p(x)/q (x)) dx₁...dx₇₈₄
\`\`\`

This 784-dimensional integral is intractable!

**Problem 2: Unknown Distributions**

In supervised learning:
- True data distribution P(x, y) is unknown
- Only have samples: {(x₁,y₁), ..., (xₙ,yₙ)}

Can't compute ∫ p (x) ... dx because we don't have p (x)!

**Problem 3: Intractable Model Distributions**

For complex models (deep networks), even computing q (x) requires intractable integrals:

\`\`\`
q (x) = ∫ q (x|z) q (z) dz  (marginalizing latent variables)
\`\`\`

**6. Practical Approximations:**

**Approximation 1: Monte Carlo Estimation**

Replace integral with sample average:

\`\`\`
D_KL(P||Q) = E_P[log (p(X)/q(X))]
           ≈ (1/N) Σᵢ log (p(xᵢ)/q (xᵢ)), xᵢ ~ P
\`\`\`

**Example:**
\`\`\`python
# Sample from P
samples = np.random.normal (mu1, sigma1, 10000)

# Monte Carlo estimate
kl_mc = np.mean([np.log (p(x) / q (x)) for x in samples])
print(f"KL (Monte Carlo): {kl_mc:.6f}")
\`\`\`

**Approximation 2: Empirical Distribution**

Use empirical samples instead of true P:

\`\`\`
H_empirical(P, Q) = -(1/N) Σᵢ log q (xᵢ)
\`\`\`

**This is the standard cross-entropy loss in ML!**

\`\`\`python
# Classification: minimize cross-entropy
def cross_entropy_loss (y_true, y_pred):
    # y_true: one-hot encoded true labels (empirical P)
    # y_pred: model predictions (Q)
    return -np.mean (np.sum (y_true * np.log (y_pred + 1e-10), axis=1))
\`\`\`

**Approximation 3: Evidence Lower Bound (ELBO)**

For latent variable models, KL involves intractable integrals:

\`\`\`
log p (x) = log ∫ p (x|z) p (z) dz  (intractable!)
\`\`\`

**Solution:** Variational inference

Instead of computing exact KL, maximize ELBO:

\`\`\`
log p (x) ≥ E_q[log p (x|z)] - D_KL(q (z)||p (z))  (ELBO)
\`\`\`

where q (z) is a tractable approximation.

**Example: Variational Autoencoder (VAE)**
\`\`\`python
def vae_loss (x, x_recon, mu, logvar):
    # Reconstruction term (Monte Carlo estimate)
    recon_loss = binary_crossentropy (x, x_recon)
    
    # KL term (analytical for Gaussian)
    kl_loss = -0.5 * np.sum(1 + logvar - mu**2 - np.exp (logvar))
    
    return recon_loss + kl_loss
\`\`\`

**7. Why Empirical Approximation Works:**

**Supervised Learning:**

Minimize:
\`\`\`
D_KL(P_data || Q_model) = E_P_data[log (p(y|x)) - log (q(y|x))]
\`\`\`

First term (entropy of true distribution) is constant, so equivalent to:

\`\`\`
minimize E_P_data[-log q (y|x)]
\`\`\`

Use empirical samples:

\`\`\`
≈ -(1/N) Σᵢ log q (yᵢ|xᵢ)  (cross-entropy loss!)
\`\`\`

**Example:**
\`\`\`python
# Binary classification
y_true = np.array([0, 1, 1, 0])  # empirical samples
y_pred = np.array([0.1, 0.9, 0.8, 0.2])  # model predictions

# Cross-entropy (empirical approximation of integral)
loss = -np.mean (y_true * np.log (y_pred) + (1-y_true) * np.log(1-y_pred))
\`\`\`

**8. Discrete vs Continuous:**

**Discrete (classification):**
- Integrals become sums
- Exact computation possible
- Cross-entropy: -Σᵢ pᵢ log qᵢ

**Continuous (regression, generative models):**
- True integrals required
- Usually intractable
- Must approximate (MC, variational)

**9. Advanced: Tractable Approximations**

**Normalizing Flows:**
Design q (x) such that:
1. Sampling is easy
2. Density evaluation is tractable
3. Can compute log q (x) exactly

Transform simple distribution (Gaussian) through bijective functions.

**Score-Based Models:**
Instead of modeling p (x) directly, model ∇_x log p (x) (the score).
Avoids computing normalizing constant (which requires integration).

**10. Summary Table:**

| Setting | Method | Approximation |
|---------|--------|---------------|
| **Gaussian** | Analytical | Closed-form formula |
| **Discrete (classification)** | Exact | Summation (no integral) |
| **Empirical samples** | Monte Carlo | (1/N)Σ log q (xᵢ) |
| **Latent variables (VAE)** | Variational (ELBO) | Lower bound |
| **High-D continuous** | Normalizing flows | Tractable density |
| **Score matching** | Avoid density | Model score instead |

**Key Insights:**

1. **Cross-entropy loss = empirical approximation of integral**
   - We replace E_P with sample average
   - Works because of law of large numbers

2. **KL divergence requires knowing both P and Q**
   - In practice, only have samples from P
   - Must approximate the expectation

3. **High dimensionality = intractability**
   - Integrals over 100+ dimensions infeasible
   - Clever approximations (VI, MC, normalizing flows) essential

4. **Different problems, different solutions:**
   - Classification: discrete (exact sums)
   - Regression: continuous (MC approximation)
   - Generative models: latent variables (variational methods)

**Practical Takeaway:**

When you see cross-entropy loss in code:
\`\`\`python
loss = -torch.mean (y_true * torch.log (y_pred))
\`\`\`

Remember: this is an **empirical approximation** of the integral:
\`\`\`
H(P, Q) = -∫ p (x) log q (x) dx
\`\`\`

using samples from the training set. The entire foundation of supervised learning rests on this approximation being valid (which it is, by the law of large numbers)!`,
    keyPoints: [
      'Cross-entropy H(P,Q) = -∫ p (x)log q (x)dx is an expectation',
      'KL divergence involves integration over continuous distributions',
      'High-dimensional integrals (images, text) are intractable',
      'Empirical approximation: replace ∫ p (x)... with (1/N)Σ...',
      'Classification: discrete (sums), Generative: continuous (integrals)',
      'VAEs use ELBO to avoid intractable KL computation',
      'Standard cross-entropy loss = empirical approximation of integral',
    ],
  },
];
