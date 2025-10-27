/**
 * Stochastic Calculus Fundamentals Section
 */

export const stochasticcalculusSection = {
  id: 'stochastic-calculus',
  title: 'Stochastic Calculus Fundamentals',
  content: `
# Stochastic Calculus Fundamentals

## Introduction

Stochastic calculus extends calculus to random processes. In machine learning:
- **Stochastic optimization**: SGD, Langevin dynamics
- **Generative models**: Diffusion models, score matching
- **Reinforcement learning**: Continuous-time control
- **Quantitative finance**: Options pricing, risk models

## Brownian Motion

**Brownian motion** (Wiener process) W_t is a continuous-time random process:

**Properties:**1. W_0 = 0
2. Independent increments
3. W_t - W_s ~ N(0, t-s) for t > s
4. Continuous paths

**Intuition:** Limit of random walk as step size → 0.

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt

def brownian_motion(T=1.0, N=1000, n_paths=5):
    """
    Simulate Brownian motion paths
    
    Args:
        T: total time
        N: number of time steps
        n_paths: number of paths to simulate
    
    Returns:
        t: time points
        W: Brownian motion paths (n_paths × N+1)
    """
    dt = T / N
    t = np.linspace(0, T, N + 1)
    
    # Generate increments: dW ~ N(0, √dt)
    dW = np.random.randn (n_paths, N) * np.sqrt (dt)
    
    # Cumulative sum to get paths
    W = np.column_stack([np.zeros (n_paths), np.cumsum (dW, axis=1)])
    
    return t, W

# Simulate and plot
t, W = brownian_motion(T=1.0, N=1000, n_paths=10)

plt.figure (figsize=(12, 4))

# Multiple paths
plt.subplot(131)
for i in range(10):
    plt.plot (t, W[i], alpha=0.7)
plt.xlabel('Time t')
plt.ylabel('W_t')
plt.title('Brownian Motion Paths')
plt.grid(True, alpha=0.3)

# Distribution at t=1
plt.subplot(132)
plt.hist(W[:, -1], bins=30, density=True, alpha=0.7, label='Simulated')
x = np.linspace(-3, 3, 100)
plt.plot (x, (1/np.sqrt(2*np.pi)) * np.exp(-x**2/2), 'r-', linewidth=2, label='N(0,1)')
plt.xlabel('W_1')
plt.ylabel('Density')
plt.title('Distribution at t=1')
plt.legend()

# Increments distribution
plt.subplot(133)
increments = np.diff(W[0]) / np.sqrt (t[1]-t[0])
plt.hist (increments, bins=30, density=True, alpha=0.7, label='Increments')
plt.plot (x, (1/np.sqrt(2*np.pi)) * np.exp(-x**2/2), 'r-', linewidth=2, label='N(0,1)')
plt.xlabel('dW_t / √dt')
plt.ylabel('Density')
plt.title('Scaled Increments')
plt.legend()

plt.tight_layout()
plt.savefig('brownian_motion.png', dpi=150, bbox_inches='tight')
print("Brownian motion has:")
print(f"  Mean at t=1: {np.mean(W[:, -1]):.4f} (expected: 0)")
print(f"  Variance at t=1: {np.var(W[:, -1]):.4f} (expected: 1)")
\`\`\`

## Stochastic Differential Equations (SDEs)

**General form:**
dX_t = μ(X_t, t)dt + σ(X_t, t)dW_t

where:
- μ(X_t, t): drift (deterministic part)
- σ(X_t, t): diffusion (random part)
- dW_t: Brownian increment

**Example:** Geometric Brownian Motion
dS_t = μS_t dt + σS_t dW_t

Models stock prices, exponential growth with noise.

\`\`\`python
def euler_maruyama (mu, sigma, X0, T=1.0, N=1000, n_paths=1):
    """
    Euler-Maruyama method for SDEs
    
    Solves: dX_t = μ(X_t, t)dt + σ(X_t, t)dW_t
    
    Args:
        mu: drift function μ(X, t)
        sigma: diffusion function σ(X, t)
        X0: initial value
        T: final time
        N: number of steps
        n_paths: number of paths
    
    Returns:
        t: time points
        X: solution paths
    """
    dt = T / N
    t = np.linspace(0, T, N + 1)
    
    X = np.zeros((n_paths, N + 1))
    X[:, 0] = X0
    
    for i in range(N):
        dW = np.random.randn (n_paths) * np.sqrt (dt)
        X[:, i+1] = X[:, i] + mu(X[:, i], t[i]) * dt + sigma(X[:, i], t[i]) * dW
    
    return t, X

# Geometric Brownian Motion: dS_t = 0.1·S_t·dt + 0.2·S_t·dW_t
def mu_gbm(S, t):
    return 0.1 * S  # 10% drift

def sigma_gbm(S, t):
    return 0.2 * S  # 20% volatility

S0 = 100  # Initial stock price
t, S = euler_maruyama (mu_gbm, sigma_gbm, S0, T=1.0, N=1000, n_paths=100)

plt.figure (figsize=(10, 4))
plt.subplot(121)
for i in range (min(20, len(S))):
    plt.plot (t, S[i], alpha=0.5)
plt.xlabel('Time (years)')
plt.ylabel('Stock Price')
plt.title('Geometric Brownian Motion Paths')
plt.grid(True, alpha=0.3)

plt.subplot(122)
plt.hist(S[:, -1], bins=30, density=True, alpha=0.7)
plt.xlabel('Final Price S_T')
plt.ylabel('Density')
plt.title (f'Distribution at T=1 (S_0={S0})')
plt.axvline(S0 * np.exp(0.1), color='r', linestyle='--', label=f'Expected: {S0 * np.exp(0.1):.1f}')
plt.legend()

plt.tight_layout()
plt.savefig('geometric_brownian_motion.png', dpi=150, bbox_inches='tight')

print("Geometric Brownian Motion:")
print(f"  Initial price: {S0}")
print(f"  Mean final price: {np.mean(S[:, -1]):.2f}")
print(f"  Theoretical mean: {S0 * np.exp(0.1):.2f}")
\`\`\`

## Itô's Lemma

**Chain rule for stochastic processes**: If X_t solves an SDE and f(X_t, t) is twice differentiable:

df(X_t, t) = [∂f/∂t + μ·∂f/∂x + (1/2)σ²·∂²f/∂x²]dt + σ·∂f/∂x·dW_t

**Key difference from standard calculus:** Extra term (1/2)σ²·∂²f/∂x²

**Intuition:** Quadratic variation of Brownian motion: (dW_t)² = dt

\`\`\`python
def ito_lemma_example():
    """
    Verify Itô's lemma numerically
    
    X_t: Geometric Brownian Motion
    f(X_t) = log(X_t)
    
    By Itô's lemma:
    d (log X_t) = (μ - σ²/2)dt + σ dW_t
    
    This is Brownian motion with drift!
    """
    
    # Parameters
    mu, sigma = 0.1, 0.2
    X0 = 100
    T, N = 1.0, 10000
    
    # Simulate GBM
    def mu_func(X, t):
        return mu * X
    def sigma_func(X, t):
        return sigma * X
    
    t, X = euler_maruyama (mu_func, sigma_func, X0, T, N, n_paths=10000)
    
    # Transform: Y_t = log(X_t)
    Y = np.log(X)
    
    # By Itô's lemma: dY_t = (μ - σ²/2)dt + σ dW_t
    # So Y_T - Y_0 ~ N((μ - σ²/2)T, σ²T)
    
    Y_final = Y[:, -1] - Y[:, 0]
    
    theoretical_mean = (mu - 0.5*sigma**2) * T
    theoretical_std = sigma * np.sqrt(T)
    
    print("Itô's Lemma Verification:")
    print(f"  Simulated mean: {np.mean(Y_final):.6f}")
    print(f"  Theoretical mean: {theoretical_mean:.6f}")
    print(f"  Simulated std: {np.std(Y_final):.6f}")
    print(f"  Theoretical std: {theoretical_std:.6f}")
    print("→ Itô's lemma correctly predicts log-transform distribution!")

ito_lemma_example()
\`\`\`

## Applications in ML

### Langevin Dynamics

**Idea:** Add noise to gradient descent for better exploration.

**Stochastic Gradient Langevin Dynamics (SGLD):**
x_{k+1} = x_k - α∇f (x_k) + √(2α/β)·ξ_k

where ξ_k ~ N(0, I), β is inverse temperature.

**Continuous-time limit:**
dX_t = -∇f(X_t)dt + √(2/β)dW_t

**Stationary distribution:** p (x) ∝ exp(-βf (x))

\`\`\`python
def sgld (grad_f, x0, alpha=0.01, beta=1.0, n_iter=1000):
    """
    Stochastic Gradient Langevin Dynamics
    
    Samples from p (x) ∝ exp(-β·f (x))
    """
    x = x0.copy()
    samples = [x.copy()]
    
    for k in range (n_iter):
        grad = grad_f (x)
        noise = np.random.randn(*x.shape) * np.sqrt(2 * alpha / beta)
        x = x - alpha * grad + noise
        samples.append (x.copy())
    
    return np.array (samples)

# Target: sample from bimodal distribution
def f (x):
    """Double-well potential"""
    return (x[0]**2 - 1)**2 + x[1]**2

def grad_f (x):
    return np.array([
        4*x[0]*(x[0]**2 - 1),
        2*x[1]
    ])

x0 = np.array([0.0, 0.0])
samples = sgld (grad_f, x0, alpha=0.01, beta=1.0, n_iter=10000)

plt.figure (figsize=(12, 4))

# Trajectory
plt.subplot(131)
plt.plot (samples[:, 0], samples[:, 1], 'b-', alpha=0.3)
plt.plot (samples[0, 0], samples[0, 1], 'go', markersize=10, label='Start')
plt.xlabel('x₁')
plt.ylabel('x₂')
plt.title('SGLD Trajectory')
plt.legend()
plt.grid(True, alpha=0.3)

# Marginal distribution (x₁)
plt.subplot(132)
plt.hist (samples[1000:, 0], bins=50, density=True, alpha=0.7)
plt.xlabel('x₁')
plt.ylabel('Density')
plt.title('Sampled Distribution (x₁)')
plt.axvline(-1, color='r', linestyle='--', label='Modes')
plt.axvline(1, color='r', linestyle='--')
plt.legend()

# Both modes
plt.subplot(133)
plt.hist2d (samples[1000:, 0], samples[1000:, 1], bins=50, cmap='Blues')
plt.xlabel('x₁')
plt.ylabel('x₂')
plt.title('2D Density (SGLD samples)')
plt.colorbar()

plt.tight_layout()
plt.savefig('sgld.png', dpi=150, bbox_inches='tight')
print("SGLD explores both modes of bimodal distribution!")
print("→ Noise helps escape local minima")
\`\`\`

### Diffusion Models

**Forward process:** Add noise gradually
x_t = √(α_t)x_0 + √(1-α_t)ε, ε ~ N(0, I)

**Reverse process:** Learn to denoise
dx_t = [f (x_t, t) - g²(t)∇_x log p_t (x_t)]dt + g (t)dW̄_t

where ∇_x log p_t (x_t) is the **score function** (learned by neural network).

\`\`\`python
def simple_diffusion_demo():
    """
    Demonstrate diffusion process concept
    """
    
    # Original data: samples from mixture of Gaussians
    np.random.seed(42)
    n_samples = 1000
    
    # Two clusters
    X = np.vstack([
        np.random.randn (n_samples//2, 2) + [-2, 0],
        np.random.randn (n_samples//2, 2) + [2, 0]
    ])
    
    # Forward diffusion: gradually add noise
    T_steps = 5
    noise_schedule = np.linspace(0, 1, T_steps)**2
    
    fig, axes = plt.subplots(1, T_steps, figsize=(15, 3))
    
    for i, noise_level in enumerate (noise_schedule):
        # Add noise: x_t = √(1-β_t)x_0 + √β_t·ε
        X_noisy = np.sqrt(1 - noise_level) * X + np.sqrt (noise_level) * np.random.randn(*X.shape)
        
        axes[i].scatter(X_noisy[:, 0], X_noisy[:, 1], alpha=0.5, s=1)
        axes[i].set_xlim(-5, 5)
        axes[i].set_ylim(-5, 5)
        axes[i].set_title (f't={i}/{T_steps-1}')
        axes[i].set_aspect('equal')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('diffusion_forward.png', dpi=150, bbox_inches='tight')
    print("Diffusion Models:")
    print("  Forward: Gradually add noise (data → noise)")
    print("  Reverse: Learn to denoise (noise → data)")
    print("  → Generate new samples by running reverse process")

simple_diffusion_demo()
\`\`\`

## Summary

**Key Concepts**:
- **Brownian motion**: Continuous-time random walk, W_t ~ N(0, t)
- **SDEs**: dX_t = μ dt + σ dW_t (drift + diffusion)
- **Itô's lemma**: Chain rule with extra (1/2)σ² term
- **Euler-Maruyama**: Numerical method for simulating SDEs

**ML Applications**:
- **SGLD**: Stochastic optimization with noise for exploration
- **Diffusion models**: State-of-art generative models (DALL-E 2, Stable Diffusion)
- **Score matching**: Learn gradient of log-density
- **Continuous normalizing flows**: ODEs/SDEs for generative modeling

**Why This Matters**:
Stochastic calculus provides the mathematical foundation for understanding and developing:
- Noisy optimization algorithms (why SGD works)
- Modern generative models (diffusion models)
- Exploration-exploitation trade-offs (reinforcement learning)
- Uncertainty quantification (Bayesian methods)
`,
};
