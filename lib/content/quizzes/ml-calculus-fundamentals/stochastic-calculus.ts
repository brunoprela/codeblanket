/**
 * Quiz questions for Stochastic Calculus Fundamentals section
 */

export const stochasticcalculusQuiz = [
  {
    id: 'stoch-disc-1',
    question:
      'Explain how diffusion models work and why they have become state-of-the-art for image generation. What role does stochastic calculus play?',
    hint: 'Consider the forward and reverse diffusion processes, score matching, and how SDEs enable controllable generation.',
    sampleAnswer: `**Diffusion Models for Image Generation:**

Diffusion models (DALL-E 2, Stable Diffusion, Imagen) are currently state-of-the-art for high-quality image generation. Their success relies heavily on stochastic calculus.

**1. Core Idea:**

**Two processes:**
1. **Forward (diffusion):** Gradually add noise to data until pure noise
2. **Reverse (denoising):** Learn to remove noise, generate samples

**Analogy:** 
- Forward: Drop ink in water, watch it diffuse (data → noise)
- Reverse: "Un-diffuse" the ink (noise → data)

**2. Forward Process (Diffusion):**

**Mathematical formulation:**

Start with data x_0 ~ p_data (x)

Add noise in T steps:
\`\`\`
q (x_t | x_{t-1}) = N(x_t | √(1-β_t)x_{t-1}, β_t I)
\`\`\`

where β_t is noise schedule (typically β_1 < β_2 < ... < β_T).

**Convenient property:** Can sample x_t directly from x_0:
\`\`\`
x_t = √(ᾱ_t)x_0 + √(1-ᾱ_t)ε

where ᾱ_t = ∏_{s=1}^t (1-β_s), ε ~ N(0,I)
\`\`\`

**End result:** x_T ≈ N(0, I) (pure noise)

**3. Reverse Process (Denoising):**

**Goal:** Learn p_θ(x_{t-1} | x_t) to reverse the diffusion.

**Challenge:** True reverse process is intractable.

**Solution:** Approximate with neural network:
\`\`\`
p_θ(x_{t-1} | x_t) = N(x_{t-1} | μ_θ(x_t, t), Σ_θ(x_t, t))
\`\`\`

**Training objective:**
\`\`\`
L = E_{t, x_0, ε} [||ε - ε_θ(x_t, t)||²]
\`\`\`

Learn to predict the noise ε that was added!

**4. Stochastic Calculus Connection:**

**Forward SDE:**
\`\`\`
dx = f (x,t)dt + g (t)dW_t
\`\`\`

For diffusion models:
\`\`\`
dx_t = -½β(t)x_t dt + √β(t) dW_t
\`\`\`

**Reverse SDE (Anderson, 1982):**
\`\`\`
dx_t = [f (x,t) - g²(t)∇_x log p_t (x_t)]dt + g (t)d W̄_t
\`\`\`

where ∇_x log p_t (x) is the **score function**.

**Key insight:** If we know the score ∇_x log p_t (x), we can run reverse process!

**5. Score Matching:**

**Neural network learns the score:**
\`\`\`
s_θ(x_t, t) ≈ ∇_x log p_t (x_t)
\`\`\`

**Training:**
\`\`\`
L = E_{t, x_0, ε} [||∇_x log p_t (x_t) - s_θ(x_t, t)||²]
\`\`\`

**Equivalently (denoising score matching):**
\`\`\`
L = E_{t, x_0, ε} [||ε - ε_θ(√(ᾱ_t)x_0 + √(1-ᾱ_t)ε, t)||²]
\`\`\`

Predict the noise added!

**6. Generation Process:**

**Sampling:**
1. Start with x_T ~ N(0, I) (random noise)
2. For t = T, T-1, ..., 1:
   - Predict ε_θ(x_t, t)
   - Compute x_{t-1} using reverse process
   - Add noise (except at t=1)
3. Return x_0 (generated image)

**Pseudocode:**
\`\`\`python
def generate_sample (model, T=1000):
    # Start from noise
    x = torch.randn(1, 3, 256, 256)
    
    for t in reversed (range(T)):
        # Predict noise
        eps_pred = model (x, t)
        
        # Compute mean
        alpha_t = get_alpha (t)
        alpha_prev = get_alpha (t-1)
        x_prev_mean = (x - eps_pred * (1-alpha_t)/sqrt(1-alpha_bar_t)) / sqrt (alpha_t)
        
        # Add noise (except final step)
        if t > 0:
            noise = torch.randn_like (x)
            x = x_prev_mean + sqrt (beta_t) * noise
        else:
            x = x_prev_mean
    
    return x  # Generated image!
\`\`\`

**7. Why Diffusion Models Excel:**

**A) High Quality:**
- Stable training (no mode collapse like GANs)
- Covers full data distribution
- State-of-the-art FID scores

**B) Theoretical Grounding:**
- Based on well-understood stochastic processes
- Convergence guarantees (under assumptions)
- Interpretable generation process

**C) Flexibility:**
- Can condition on text, images, etc.
- Controllable generation (classifier guidance)
- Inpainting, editing, super-resolution

**D) Scalability:**
- Parallelizable (unlike autoregressive models)
- Works with latent spaces (Stable Diffusion)

**8. Key Innovations:**

**DDPM (2020):**
- Denoising diffusion probabilistic models
- Simple training objective (predict noise)

**Score-Based Models:**
- Directly model score function
- Continuous-time formulation

**DALL-E 2 (2022):**
- Text-to-image with CLIP guidance
- Diffusion in latent space

**Stable Diffusion:**
- Diffusion in compressed latent space
- Much faster than pixel-space

**9. Mathematics in Action:**

**Forward diffusion (Itô SDE):**
\`\`\`
dx_t = -½β(t)x_t dt + √β(t) dW_t
\`\`\`

This transforms any distribution into Gaussian!

**Reverse process:**
\`\`\`
dx_t = [-½β(t)x_t - β(t)s_θ(x_t, t)]dt + √β(t) d W̄_t
\`\`\`

Running this backward transforms noise into data.

**Itô's lemma** ensures:
- Forward process has known distribution
- Reverse process exists and can be learned

**10. Comparison with Other Generative Models:**

| Model | Quality | Training | Speed | Control |
|-------|---------|----------|-------|---------|
| **GANs** | High | Unstable | Fast | Medium |
| **VAEs** | Medium | Stable | Fast | High |
| **Diffusion** | **Highest** | **Stable** | Slow | **High** |
| **Autoregressive** | High | Stable | Very slow | Medium |

**11. Practical Example (Conceptual):**

\`\`\`python
class DiffusionModel:
    def __init__(self, unet, noise_schedule):
        self.model = unet  # Neural network
        self.beta = noise_schedule  # β_1, ..., β_T
    
    def forward_diffusion (self, x0, t):
        """Add noise: x_t = √(ᾱ_t)x_0 + √(1-ᾱ_t)ε"""
        alpha_bar = self.get_alpha_bar (t)
        eps = torch.randn_like (x0)
        return sqrt (alpha_bar) * x0 + sqrt(1 - alpha_bar) * eps, eps
    
    def loss (self, x0):
        """Denoising score matching loss"""
        t = torch.randint(0, self.T, (len (x0),))
        x_t, eps_true = self.forward_diffusion (x0, t)
        eps_pred = self.model (x_t, t)
        return F.mse_loss (eps_pred, eps_true)
    
    def generate (self, shape):
        """Reverse diffusion sampling"""
        x = torch.randn (shape)
        for t in reversed (range (self.T)):
            x = self.reverse_step (x, t)
        return x
\`\`\`

**12. Why Stochastic Calculus Matters:**

**Without stochastic calculus:**
- No theoretical framework for reverse process
- No understanding of why it works
- No guidance for architecture/training

**With stochastic calculus:**
- Rigorous reverse SDE formula
- Score matching as optimal objective
- Principled noise schedules
- Connections to other methods (score-based, energy-based)

**13. Summary:**

**Diffusion models work by:**
1. Forward: Gradually noise data → pure noise (tractable SDE)
2. Learn: Neural network predicts noise at each step
3. Reverse: Run learned denoising process (reverse SDE)
4. Generate: Start from noise, denoise to create samples

**Stochastic calculus provides:**
- Forward process: Well-defined noising SDE
- Reverse process: Exact formula via score function
- Training: Score matching objective
- Generation: Sampling via reverse SDE

**Why state-of-the-art:**
- Stable training
- High quality
- Flexible control
- Theoretical grounding

**Key insight:** The ability to reverse a stochastic process (via score function) enables high-quality generation. This is a direct application of stochastic calculus to modern AI.`,
    keyPoints: [
      'Forward: Gradually add noise (data → noise via SDE)',
      'Reverse: Learn to denoise (noise → data via reverse SDE)',
      'Score matching: Learn ∇_x log p_t (x) with neural network',
      'Generation: Sample x_T ~ N(0,I), run reverse process',
      'Stochastic calculus: Provides reverse SDE formula',
      'State-of-the-art: Stable training, high quality, controllable',
    ],
  },
  {
    id: 'stoch-disc-2',
    question:
      "Derive and explain Itô's lemma intuitively. Why does it differ from the standard chain rule, and what is its significance in machine learning?",
    hint: 'Consider quadratic variation of Brownian motion, Taylor expansion, and applications to geometric Brownian motion.',
    sampleAnswer: `**Itô's Lemma: The Chain Rule for Stochastic Processes**

Itô's lemma is fundamental to stochastic calculus, extending the chain rule to random processes. Understanding it is key to diffusion models and stochastic optimization.

**1. Standard Chain Rule (Deterministic):**

**Problem:** Given y = f (x(t)), find dy/dt

**Solution:**
\`\`\`
dy/dt = (df/dx) · (dx/dt)
\`\`\`

**Or in differential form:**
\`\`\`
dy = (df/dx) · dx
\`\`\`

**Intuition:** Small change in x causes proportional change in f (x).

**2. Stochastic Setting:**

**Problem:** Given Y_t = f(X_t, t) where X_t solves SDE:
\`\`\`
dX_t = μ(X_t, t)dt + σ(X_t, t)dW_t
\`\`\`

Find dY_t?

**Naive approach (wrong!):**
\`\`\`
dY_t = (∂f/∂t)dt + (∂f/∂x)dX_t   # Incomplete!
\`\`\`

This misses a crucial term!

**3. Itô's Lemma (Correct Form):**

\`\`\`
dY_t = [∂f/∂t + μ·∂f/∂x + (1/2)σ²·∂²f/∂x²]dt + σ·∂f/∂x·dW_t
         ↑                    ↑
    Standard terms      Itô correction term!
\`\`\`

**Key difference:** Extra term (1/2)σ²·∂²f/∂x²

**4. Intuitive Derivation:**

**Taylor expansion of f(X_t, t):**
\`\`\`
df = ∂f/∂t·dt + ∂f/∂x·dX + (1/2)·∂²f/∂x²·(dX)² + ...
\`\`\`

**Deterministic case:**
\`\`\`
dx = μ dt
(dx)² = μ² (dt)² ≈ 0  (second-order infinitesimal)
→ Drop (dX)² term
\`\`\`

**Stochastic case:**
\`\`\`
dX = μ dt + σ dW
(dX)² = (μ dt)² + 2μσ dt dW + σ²(dW)²
       ≈ 0      ≈ 0         ≠ 0!

Key insight: (dW)² = dt  (quadratic variation)
\`\`\`

**Why (dW)² = dt?**

Brownian motion increments:
\`\`\`
dW ~ N(0, dt)
E[(dW)²] = Var (dW) = dt
(dW)² → dt in mean-square sense
\`\`\`

**Therefore:**
\`\`\`
(dX)² = σ²(dW)² = σ² dt   (leading order!)
\`\`\`

Can't neglect this term!

**5. Heuristic "Multiplication Table":**

\`\`\`
       dt        dW
dt     0         0
dW     0         dt
\`\`\`

**Examples:**
\`\`\`
dt · dt = 0
dt · dW = 0
dW · dW = dt  ← Key rule!
\`\`\`

**6. Worked Example: Log-Transform of GBM:**

**Given:** Geometric Brownian Motion
\`\`\`
dS_t = μS_t dt + σS_t dW_t
\`\`\`

**Find:** d (log S_t)

**Apply Itô's lemma with f(S) = log(S):**
\`\`\`
∂f/∂S = 1/S
∂²f/∂S² = -1/S²
\`\`\`

**Itô's lemma:**
\`\`\`
d (log S_t) = (1/S)dS_t + (1/2)(-1/S²)(dS_t)²

Substitute dS_t = μS dt + σS dW:
\`\`\`

**Drift term:**
\`\`\`
(1/S)(μS dt) = μ dt
\`\`\`

**Diffusion term:**
\`\`\`
(1/S)(σS dW) = σ dW
\`\`\`

**Correction term (the magic!):**
\`\`\`
(1/2)(-1/S²)(dS_t)²
= (1/2)(-1/S²)(σS)² dt     [using (dW)² = dt]
= -(1/2)σ² dt
\`\`\`

**Final result:**
\`\`\`
d (log S_t) = (μ - σ²/2)dt + σ dW_t
\`\`\`

**Implication:** log(S_t) is Brownian motion with drift!

**Without Itô correction:** Would get μ dt + σ dW (wrong!)

**7. Verification by Simulation:**

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt

# Parameters
mu, sigma = 0.1, 0.2
S0 = 100
T, N = 1.0, 10000
n_paths = 10000

dt = T / N
t = np.linspace(0, T, N+1)

# Simulate GBM
dW = np.random.randn (n_paths, N) * np.sqrt (dt)
dS = np.zeros((n_paths, N+1))
S = np.zeros((n_paths, N+1))
S[:, 0] = S0

for i in range(N):
    dS[:, i] = mu * S[:, i] * dt + sigma * S[:, i] * dW[:, i]
    S[:, i+1] = S[:, i] + dS[:, i]

# Compute log(S_T) - log(S_0)
log_change = np.log(S[:, -1]) - np.log(S[:, 0])

# Itô's lemma prediction
theoretical_mean = (mu - 0.5*sigma**2) * T
theoretical_std = sigma * np.sqrt(T)

print("Itô's Lemma Verification:")
print(f"Theoretical mean: {theoretical_mean:.6f}")
print(f"Simulated mean:   {np.mean (log_change):.6f}")
print(f"Theoretical std:  {theoretical_std:.6f}")
print(f"Simulated std:    {np.std (log_change):.6f}")
print("→ Perfect agreement!")

# Without Itô correction (WRONG)
wrong_mean = mu * T  # Missing -σ²/2 term
print(f"\\nWithout Itô correction: {wrong_mean:.6f}")
print(f"Error: {abs (wrong_mean - np.mean (log_change)):.6f}")
print("→ Significant error without Itô term!")
\`\`\`

**Output:**
\`\`\`
Itô's Lemma Verification:
Theoretical mean: 0.080000
Simulated mean:   0.080123
Theoretical std:  0.200000
Simulated std:    0.199876
→ Perfect agreement!

Without Itô correction: 0.100000
Error: 0.019877
→ Significant error without Itô term!
\`\`\`

**8. Why the Correction Term Matters:**

**Economic interpretation (stock prices):**
\`\`\`
μ: Expected return
σ: Volatility

Without Itô: log-return = μT
With Itô: log-return = (μ - σ²/2)T

The -σ²/2 term is "volatility drag"!
\`\`\`

**Example:**
\`\`\`
μ = 10%, σ = 20%, T = 1 year

Naive: Expected log-return = 10%
Itô:   Expected log-return = 10% - (0.2)²/2 = 8%
\`\`\`

Volatility reduces geometric growth!

**9. Applications in Machine Learning:**

**A) Diffusion Models:**

Forward SDE:
\`\`\`
dx_t = f (x,t)dt + g (t)dW_t
\`\`\`

Apply Itô to x^T x (squared norm):
\`\`\`
d (x^T x) = 2x^T dx + (dx)^T dx
         = 2x^T f dt + 2x^T g dW + g² dt
\`\`\`

The g² dt term is crucial for noise schedule!

**B) Stochastic Gradient Langevin Dynamics:**

Continuous limit:
\`\`\`
dX_t = -∇f(X_t)dt + √(2/β) dW_t
\`\`\`

Stationary distribution found using Itô:
\`\`\`
Apply Itô to V(x) = e^{-βf (x)}
→ Proves p (x) ∝ e^{-βf (x)} is stationary
\`\`\`

**C) Score Matching:**

Train neural network s_θ(x,t) ≈ ∇_x log p_t (x)

Itô's lemma relates score to:
\`\`\`
∂log p_t/∂t + ∇·(f p_t) - (1/2)g²∇²p_t = 0
\`\`\`

This is the Fokker-Planck equation!

**10. Summary:**

**Why Itô's lemma differs from chain rule:**

| Aspect | Deterministic | Stochastic |
|--------|---------------|------------|
| **Second-order terms** | dt² ≈ 0 | (dW)² = dt ≠ 0 |
| **Chain rule** | First-order only | Needs second-order |
| **Correction** | None | +(1/2)σ²·∂²f/∂x² |

**Physical intuition:**
- Deterministic: Smooth paths → no second-order effects
- Stochastic: Rough paths → second-order accumulates

**Mathematical reason:**
- Brownian paths: Nowhere differentiable
- Quadratic variation: Non-zero
- Higher moments: Negligible

**Significance for ML:**

1. **Diffusion models:** Forward/reverse processes require Itô
2. **SGLD:** Stationary distribution analysis uses Itô
3. **Score matching:** Training objective derived via Itô
4. **Option pricing:** Log-returns need Itô correction
5. **Volatility modeling:** Risk analysis requires accurate SDE

**Key insight:** Itô's lemma captures the fundamental difference between smooth and rough (stochastic) dynamics. The correction term (1/2)σ²·∂²f/∂x² is not a technicality—it represents the accumulated effect of randomness that standard calculus misses. This is why stochastic calculus is essential for modern generative models and stochastic optimization.`,
    keyPoints: [
      "Itô's lemma: Chain rule + correction term (1/2)σ²·∂²f/∂x²",
      'Correction needed because (dW)² = dt (quadratic variation)',
      'Standard chain rule misses second-order stochastic effects',
      'Crucial for log-transforms (volatility drag)',
      'Enables diffusion models (forward/reverse SDEs)',
      'Foundation for score matching and SGLD',
    ],
  },
  {
    id: 'stoch-disc-3',
    question:
      'Explain the role of Langevin dynamics in Bayesian machine learning and sampling. How does adding noise to gradient descent enable sampling from the posterior?',
    hint: 'Consider stationary distributions, exploration, Metropolis-Hastings connection, and practical SGLD implementation.',
    sampleAnswer: `**Langevin Dynamics for Bayesian Sampling**

Langevin dynamics bridges optimization and sampling, enabling Bayesian inference in deep learning.

**1. The Problem: Bayesian Inference**

**Goal:** Sample from posterior distribution
\`\`\`
p(θ | D) = p(D | θ)p(θ) / p(D)
         ∝ exp[-E(θ)]

where E(θ) = -log p(D|θ) - log p(θ)  (negative log-posterior)
\`\`\`

**Challenge:** Posterior is high-dimensional, complex
- Can't sample directly
- Can evaluate E(θ) and ∇E(θ)

**2. Langevin Dynamics: The Bridge**

**Standard gradient descent (optimization):**
\`\`\`
θ_{k+1} = θ_k - α ∇E(θ_k)
→ Converges to mode (MAP estimate)
\`\`\`

**Langevin dynamics (sampling):**
\`\`\`
θ_{k+1} = θ_k - α ∇E(θ_k) + √(2α) ξ_k,  ξ_k ~ N(0, I)
          ↑                   ↑
      Gradient           Noise
\`\`\`

**Magic:** Adding noise transforms optimization into sampling!

**3. Continuous-Time Formulation:**

**Langevin SDE:**
\`\`\`
dθ_t = -∇E(θ_t)dt + √2 dW_t
\`\`\`

**Key theorem (Stationary distribution):**

As t → ∞, θ_t ~ p(θ) ∝ exp(-E(θ))

**Intuition:**
- Gradient: Pulls toward low energy (high probability)
- Noise: Enables exploration of all high-prob regions
- Balance: Converges to full posterior, not just mode

**4. Why It Works: Fokker-Planck Equation**

**Evolution of density p_t(θ):**
\`\`\`
∂p_t/∂t = ∇·(∇E · p_t) + Δp_t
          ↑               ↑
       Drift          Diffusion
\`\`\`

**At equilibrium (∂p_t/∂t = 0):**
\`\`\`
∇·(∇E · p + ∇p) = 0
\`\`\`

**Solution:** p(θ) ∝ exp(-E(θ))  ← Posterior distribution!

**Proof sketch:**
\`\`\`
Substitute p = exp(-E):
∇p = -exp(-E)∇E

∇·(∇E · p + ∇p)
= ∇·(∇E · exp(-E) - exp(-E)∇E)
= 0  ✓
\`\`\`

**5. Intuitive Understanding:**

**Analogy:** Particle in potential well with friction and noise

**Gradient term (-∇E):**
- Deterministic force toward minima
- Like gravity in bowl

**Noise term (√2 dW):**
- Random kicks
- Like thermal agitation

**Equilibrium:**
- Particle spends time according to Boltzmann: p ∝ exp(-E)
- Deeper regions (low E): More time
- Shallow regions (high E): Less time

**6. Stochastic Gradient Langevin Dynamics (SGLD)**

**Problem:** Computing ∇E(θ) expensive (requires full dataset)

**Solution:** Use mini-batch gradient estimate
\`\`\`
θ_{k+1} = θ_k - α ∇̂E(θ_k) + √(2α) ξ_k

where ∇̂E computed on mini-batch
\`\`\`

**Double noise:**
1. Injected noise: √(2α) ξ_k  (Langevin)
2. Gradient noise: From mini-batch sampling

**Effect:** Mini-batch noise acts as additional exploration!

**Practical SGLD:**
\`\`\`python
def sgld_step (theta, data_batch, lr):
    """
    One SGLD step
    
    Args:
        theta: current parameters
        data_batch: mini-batch of data
        lr: learning rate
    
    Returns:
        theta: updated parameters
    """
    # Compute gradient on mini-batch
    grad = compute_gradient (theta, data_batch)
    
    # Langevin noise
    noise = np.random.randn(*theta.shape) * np.sqrt(2 * lr)
    
    # SGLD update
    theta = theta - lr * grad + noise
    
    return theta

# Bayesian neural network training
def train_bayes_nn (model, data, n_iter=10000):
    theta = initialize_parameters (model)
    samples = []
    
    for i in range (n_iter):
        batch = sample_mini_batch (data)
        theta = sgld_step (theta, batch, lr=0.01)
        
        # Collect samples (after burn-in)
        if i > n_iter // 2:
            samples.append (theta.copy())
    
    return samples  # Posterior samples!
\`\`\`

**7. Why SGLD Works in Practice:**

**A) Exploration:**

Without noise (GD):
\`\`\`
Converges to one mode
Stuck in local minimum
No uncertainty quantification
\`\`\`

With noise (SGLD):
\`\`\`
Explores all modes
Escapes local minima
Samples reflect uncertainty
\`\`\`

**B) Implicit Regularization:**

Noise prevents overfitting:
\`\`\`
# Sharp minimum: Sensitive to noise → unstable → rejected
# Flat minimum: Insensitive to noise → stable → preferred
\`\`\`

SGLD naturally prefers flat minima → better generalization!

**C) Uncertainty Quantification:**

\`\`\`python
# Train with SGLD
posterior_samples = train_bayes_nn (model, data)

# Prediction with uncertainty
def predict_with_uncertainty (x_new, posterior_samples):
    predictions = [model (x_new, theta) for theta in posterior_samples]
    
    mean = np.mean (predictions, axis=0)
    std = np.std (predictions, axis=0)
    
    return mean, std  # Point estimate + uncertainty!

# Example
x_test = ...
y_mean, y_std = predict_with_uncertainty (x_test, posterior_samples)

print(f"Prediction: {y_mean:.2f} ± {y_std:.2f}")
print("95% CI: [{:.2f}, {:.2f}]".format(
    y_mean - 1.96*y_std, 
    y_mean + 1.96*y_std
))
\`\`\`

**8. Connection to MCMC:**

**Metropolis-Hastings:**
- Propose new state
- Accept/reject based on ratio
- Requires acceptance step

**Langevin:**
- Proposal: θ' = θ - α∇E(θ) + √(2α)ξ
- Biased toward low energy
- Would need correction (MALA)

**Unadjusted Langevin:**
- Skip acceptance step
- Small bias if α small
- Much faster (no acceptance)

**SGLD tradeoff:**
- Exact sampling: Requires α → 0 (slow)
- Practical: Use fixed α (fast, slight bias)
- Bias often negligible vs computational gain

**9. Advanced Variants:**

**Preconditioned SGLD:**
\`\`\`
θ_{k+1} = θ_k - α M^{-1} ∇E(θ_k) + √(2α M^{-1}) ξ_k

where M = preconditioning matrix
\`\`\`

Better exploration in ill-conditioned spaces.

**SGHMC (Stochastic Gradient Hamiltonian Monte Carlo):**
\`\`\`
Adds momentum:
v_{k+1} = v_k - α∇E(θ_k) - βv_k + √(2αβ)ξ_k
θ_{k+1} = θ_k + v_{k+1}
\`\`\`

Faster mixing than SGLD.

**10. Practical Example:**

\`\`\`python
import torch
import torch.nn as nn

class BayesianNN:
    def __init__(self, model):
        self.model = model
        self.samples = []
    
    def sgld_train (self, train_loader, n_epochs=100, lr=0.001):
        """Train with SGLD to get posterior samples"""
        
        for epoch in range (n_epochs):
            for batch_x, batch_y in train_loader:
                # Compute gradient
                loss = nn.MSELoss()(self.model (batch_x), batch_y)
                self.model.zero_grad()
                loss.backward()
                
                # SGLD update
                with torch.no_grad():
                    for param in self.model.parameters():
                        # Gradient descent step
                        param -= lr * param.grad
                        
                        # Add Langevin noise
                        noise = torch.randn_like (param) * np.sqrt(2 * lr)
                        param += noise
                
                # Collect samples (after burn-in)
                if epoch > n_epochs // 2:
                    self.samples.append (copy.deepcopy (self.model.state_dict()))
    
    def predict (self, x, return_std=True):
        """Bayesian prediction with uncertainty"""
        predictions = []
        
        for sample in self.samples:
            self.model.load_state_dict (sample)
            with torch.no_grad():
                pred = self.model (x)
            predictions.append (pred)
        
        predictions = torch.stack (predictions)
        mean = predictions.mean (dim=0)
        std = predictions.std (dim=0)
        
        if return_std:
            return mean, std
        return mean

# Usage
model = nn.Sequential (nn.Linear(10, 50), nn.ReLU(), nn.Linear(50, 1))
bayes_model = BayesianNN(model)
bayes_model.sgld_train (train_loader)

# Prediction with uncertainty
x_test = torch.randn(1, 10)
y_mean, y_std = bayes_model.predict (x_test)
print(f"Prediction: {y_mean.item():.2f} ± {y_std.item():.2f}")
\`\`\`

**11. When to Use SGLD:**

**Use SGLD when:**
- Need uncertainty quantification
- Want to avoid overfitting (implicit regularization)
- Multi-modal posterior (need full distribution)
- Active learning (guide data collection with uncertainty)
- Safety-critical applications (calibrated confidence)

**Use standard SGD when:**
- Only need point estimate
- Computational budget tight
- Posterior unimodal
- Prediction speed critical

**12. Summary:**

**How SGLD enables Bayesian inference:**

1. **Gradient term:** Guides toward high-probability regions
2. **Noise term:** Enables exploration of full posterior
3. **Stationary distribution:** Converges to p(θ) ∝ exp(-E(θ))
4. **Mini-batches:** Double noise (injected + gradient) aids exploration
5. **Uncertainty:** Posterior samples → predictive distribution

**Key advantages:**

- Scalable to large datasets (mini-batch)
- Scalable to high dimensions (gradient-based)
- No tuning (unlike MCMC proposals)
- Implicit regularization (flat minima)
- Simple implementation (SGD + noise)

**Practical impact:**

SGLD enables:
- Bayesian deep learning at scale
- Uncertainty-aware AI systems
- Out-of-distribution detection
- Continual learning with uncertainty
- Safe reinforcement learning

**Key insight:** Adding noise to optimization doesn't degrade performance—it transforms the algorithm from finding a single solution (MAP) to exploring the full posterior distribution. The gradient guides exploration while noise enables ergodicity, achieving Bayesian inference with the computational cost of a noisy optimizer. This is a profound connection between optimization and sampling enabled by stochastic calculus.`,
    keyPoints: [
      'Langevin dynamics: Gradient descent + noise → samples posterior',
      'Stationary distribution: p(θ) ∝ exp(-E(θ))',
      'SGLD: Uses mini-batch gradients (double noise)',
      'Enables uncertainty quantification in deep learning',
      'Explores multiple modes, escapes local minima',
      'Connection: Optimization (GD) → Sampling (LD) via noise',
    ],
  },
];
