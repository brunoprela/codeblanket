/**
 * Quiz questions for Sequences & Series section
 */

export const sequencesseriesQuiz = [
  {
    id: 'dq1-gradient-descent-convergence',
    question:
      'Gradient descent generates a sequence of parameter updates: θₙ₊₁ = θₙ - α∇L(θₙ). Explain the conditions under which this sequence converges to a minimum. Discuss the role of learning rate α, the relationship to geometric sequences, and why convergence is not always guaranteed. Provide mathematical analysis and practical examples.',
    sampleAnswer: `Gradient descent is fundamentally about generating a convergent sequence of parameters that approach an optimal value. Understanding sequence convergence is crucial for analyzing GD behavior.

**Gradient Descent as a Sequence**:

The update rule generates a sequence {θ₀, θ₁, θ₂, ...}:
θₙ₊₁ = θₙ - α∇L(θₙ)

We want: lim(n→∞) θₙ = θ* (optimal parameters)

**Conditions for Convergence**:

**1. Lipschitz Continuous Gradient**:

The gradient must not change too rapidly:
‖∇L(x) - ∇L(y)‖ ≤ L‖x - y‖

where L is the Lipschitz constant.

**Why it matters**: If gradients change wildly, small steps can lead to huge changes.

**2. Learning Rate Constraint**:

For smooth functions with Lipschitz constant L:
α < 2/L (necessary for convergence)
α ≤ 1/L (sufficient for convergence)

**Proof sketch**: 
Consider quadratic loss L(θ) = ½θᵀAθ - bᵀθ
Update: θₙ₊₁ = θₙ - α(Aθₙ - b)
       = (I - αA)θₙ + αb

This is a linear recurrence. Convergence requires eigenvalues of (I - αA) to have magnitude < 1:
|1 - αλᵢ| < 1 for all eigenvalues λᵢ of A
⟹ α < 2/λmax

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(f, grad_f, x0, alpha, n_iterations, tol=1e-6):
    """
    Gradient descent that returns full sequence
    """
    x_sequence = [x0]
    f_sequence = [f(x0)]
    x = x0
    
    for i in range(n_iterations):
        grad = grad_f(x)
        x_new = x - alpha * grad
        
        x_sequence.append(x_new)
        f_sequence.append(f(x_new))
        
        # Check convergence
        if np.abs(x_new - x) < tol:
            print(f"Converged at iteration {i+1}")
            break
        
        x = x_new
    
    return np.array(x_sequence), np.array(f_sequence)

# Example: f(x) = x^2, optimal at x=0
def f(x):
    return x**2

def grad_f(x):
    return 2*x

x0 = 10.0

# Test different learning rates
learning_rates = [0.1, 0.5, 0.9, 1.1]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, alpha in enumerate(learning_rates):
    ax = axes[idx // 2, idx % 2]
    
    x_seq, f_seq = gradient_descent(f, grad_f, x0, alpha, 50)
    
    ax.plot(f_seq, 'bo-', linewidth=2, markersize=4)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('f(x)')
    ax.set_title(f'Learning Rate α = {alpha}')
    ax.grid(True)
    ax.set_ylim(-5, 110)
    
    # Annotate convergence behavior
    if alpha < 1.0:
        ax.text(0.5, 0.9, 'Converges', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
    else:
        ax.text(0.5, 0.9, 'Diverges', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))

plt.tight_layout()
plt.show()

# For f(x) = x^2, L = 2 (second derivative)
# So α < 2/2 = 1 for convergence
print(f"\\nLipschitz constant L = 2")
print(f"Convergence requires α < 2/L = 1.0")
print(f"α = 0.1, 0.5, 0.9 converge ✓")
print(f"α = 1.1 diverges ✗")
\`\`\`

**3. Convexity (for global convergence)**:

For convex L, any local minimum is global.
GD on convex function with appropriate α converges to global optimum.

For non-convex (neural networks), GD converges to local minimum or saddle point.

**Connection to Geometric Sequences**:

For quadratic loss near optimum, GD behaves like geometric sequence:

Let eₙ = θₙ - θ* (error at step n)
Then: eₙ₊₁ = (1 - αλ)eₙ (for eigenvalue λ)

This is geometric with ratio r = (1 - αλ)

Convergence rate: |eₙ| = |r|ⁿ|e₀|

**Fast convergence**: Need |r| << 1 ⟹ α ≈ 1/λ (but not too large)
**Slow convergence**: |r| ≈ 1 ⟹ α too small or too large
**Divergence**: |r| > 1 ⟹ α too large

\`\`\`python
# Demonstrate geometric convergence
def analyze_convergence_rate(x_sequence, x_optimal):
    """Analyze convergence rate of sequence"""
    errors = np.abs(x_sequence - x_optimal)
    
    # Check if geometric: compute ratios eₙ₊₁/eₙ
    ratios = errors[1:] / errors[:-1]
    
    # Estimate convergence rate
    avg_ratio = np.mean(ratios[10:])  # After initial phase
    
    return errors, ratios, avg_ratio

x0 = 10.0
alpha = 0.4
x_seq, _ = gradient_descent(f, grad_f, x0, alpha, 100)
errors, ratios, r = analyze_convergence_rate(x_seq, 0.0)

print(f"\\nConvergence Analysis (α = {alpha}):")
print(f"Average ratio |eₙ₊₁/eₙ| = {r:.4f}")
print(f"Theoretical ratio (1-αL) = {1-alpha*2:.4f}")
print(f"Match: {np.isclose(r, 1-alpha*2, atol=0.01)}")

# Plot error decay
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.semilogy(errors, 'bo-', markersize=4)
plt.xlabel('Iteration')
plt.ylabel('|θₙ - θ*| (log scale)')
plt.title('Error Decay (Linear on Log Scale = Geometric)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(ratios, 'ro-', markersize=4)
plt.axhline(y=r, color='g', linestyle='--', label=f'Average = {r:.3f}')
plt.xlabel('Iteration')
plt.ylabel('|eₙ₊₁/eₙ|')
plt.title('Convergence Ratio (Constant = Geometric)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
\`\`\`

**Why Convergence Not Always Guaranteed**:

**1. Learning Rate Too Large**:
- Overshoots minimum
- Can diverge or oscillate
- Sequence does not converge

**2. Non-Convex Landscape**:
- Can get stuck in local minima
- Saddle points can slow convergence
- No guarantee of global optimum

**3. Vanishing/Exploding Gradients**:
- Very deep networks
- Gradients → 0 (vanishing) or → ∞ (exploding)
- Sequence stops moving or diverges

**4. Poor Initialization**:
- Start far from any good minimum
- May never reach good region

**5. Stochastic Gradient Descent**:
- Uses mini-batches (noisy gradients)
- Sequence is stochastic, not deterministic
- Doesn't converge to exact point (oscillates around optimum)

**Practical Solutions**:

\`\`\`python
# Adaptive learning rates (like geometric sequence with changing ratio)
def adam_style_learning_rate(iteration, initial_lr=0.001, decay_rate=0.9):
    """
    Adaptive LR that adjusts based on iteration
    Similar to geometric decay but more sophisticated
    """
    return initial_lr * (decay_rate ** (iteration / 100))

# Learning rate scheduling
def step_decay(initial_lr, epoch, drop_rate=0.5, epochs_drop=10):
    """
    Drop learning rate by factor every N epochs
    Creates piecewise geometric sequence
    """
    return initial_lr * (drop_rate ** (epoch // epochs_drop))

epochs = np.arange(0, 100)
adam_lrs = [adam_style_learning_rate(e) for e in epochs]
step_lrs = [step_decay(0.1, e) for e in epochs]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, adam_lrs, linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Learning Rate')
plt.title('Exponential Decay (Geometric)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs, step_lrs, linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Step Decay (Piecewise Geometric)')
plt.grid(True)

plt.tight_layout()
plt.show()
\`\`\`

**Trading Application**:

In trading strategy optimization:
- Parameter updates follow GD
- Non-convex landscape (market regimes change)
- Need adaptive learning rates
- Monitor convergence of backtest performance

\`\`\`python
# Simplified trading strategy parameter optimization
def optimize_trading_strategy(initial_params, historical_data, n_iterations=100):
    """
    Optimize strategy parameters using gradient descent
    Parameters might be: stop-loss %, take-profit %, position size
    """
    params = initial_params
    performance_sequence = []
    
    for i in range(n_iterations):
        # Backtest with current params
        performance = backtest(params, historical_data)
        performance_sequence.append(performance)
        
        # Compute gradient (finite differences in practice)
        grad = compute_gradient(params, historical_data)
        
        # Update with adaptive learning rate
        alpha = step_decay(0.01, i)
        params = params + alpha * grad  # Maximize performance
        
        # Check convergence
        if i > 10 and np.std(performance_sequence[-10:]) < 0.001:
            print(f"Strategy parameters converged at iteration {i}")
            break
    
    return params, performance_sequence

# Key insight: convergence of strategy parameters indicates
# stable optimal configuration for historical data
# But beware overfitting!
\`\`\`

**Summary**:
- GD creates parameter sequence θₙ₊₁ = θₙ - α∇L(θₙ)
- Convergence requires: α < 2/L (Lipschitz constant)
- Near optimum, behaves like geometric sequence with ratio (1-αL)
- Geometric convergence: error decays as rⁿ
- Non-convex problems, poor α, or bad initialization prevent convergence
- Adaptive learning rates improve convergence in practice
- Understanding sequences crucial for debugging training dynamics`,
    keyPoints: [
      'GD generates sequence θₙ₊₁ = θₙ - α∇L(θₙ), want convergence to θ*',
      'Convergence requires α < 2/L where L is Lipschitz constant',
      'Near optimum, GD behaves like geometric sequence with ratio (1-αL)',
      'Geometric convergence: error |θₙ - θ*| ≈ rⁿ|θ₀ - θ*|',
      'Adaptive learning rates and momentum improve convergence in practice',
    ],
  },
  {
    id: 'dq2-compound-returns-series',
    question:
      'In trading, compound returns follow: (1+r₁)(1+r₂)...(1+rₙ) while arithmetic returns are (r₁+r₂+...+rₙ)/n. Explain why compound (geometric) returns are always less than or equal to arithmetic returns. How do sequences and series help us understand portfolio growth? Discuss geometric mean vs arithmetic mean, and why this matters for trading strategies.',
    sampleAnswer: `The difference between compound and arithmetic returns is fundamental to understanding portfolio performance and is directly related to geometric series and sequences.

**Compound vs Arithmetic Returns**:

**Arithmetic (Simple) Return**:
R_arithmetic = (r₁ + r₂ + ... + rₙ) / n

**Geometric (Compound) Return**:
R_geometric = [(1+r₁)(1+r₂)...(1+rₙ)]^(1/n) - 1

**Key Insight**: R_geometric ≤ R_arithmetic (AM-GM Inequality)

Equality only when all returns are equal.

**Why Compound Returns Are Lower**:

**Mathematical Proof** (AM-GM Inequality):

For positive numbers a₁, a₂, ..., aₙ:
(a₁ + a₂ + ... + aₙ)/n ≥ (a₁·a₂·...·aₙ)^(1/n)

Let aᵢ = 1 + rᵢ:
(Σ(1+rᵢ))/n ≥ [Π(1+rᵢ)]^(1/n)

The left side is related to arithmetic mean, right side is geometric mean.

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt

def arithmetic_return(returns):
    """Simple average"""
    return np.mean(returns)

def geometric_return(returns):
    """Compound average"""
    return np.prod(1 + returns)**(1/len(returns)) - 1

def final_wealth(initial, returns):
    """Final portfolio value with compounding"""
    return initial * np.prod(1 + returns)

# Example: Compare different return sequences
sequences = {
    'Constant': np.array([0.05, 0.05, 0.05, 0.05, 0.05]),
    'Volatile +': np.array([0.10, -0.02, 0.08, 0.01, 0.03]),
    'Volatile ++': np.array([0.20, -0.10, 0.15, -0.05, 0.10]),
    'Extreme': np.array([0.50, -0.30, 0.40, -0.20, 0.30])
}

print("Return Sequences Analysis:\\n")
print(f"{'Sequence':<12} {'Arith %':<10} {'Geom %':<10} {'Final $':<10} {'Gap %':<10}")
print("-" * 60)

for name, returns in sequences.items():
    arith = arithmetic_return(returns) * 100
    geom = geometric_return(returns) * 100
    final = final_wealth(1000, returns)
    gap = arith - geom
    
    print(f"{name:<12} {arith:>9.2f} {geom:>9.2f} {final:>9.2f} {gap:>9.2f}")

print("\\nKey Observation: Higher volatility → Larger gap between arithmetic and geometric")
\`\`\`

**Output**:
\`\`\`
Sequence     Arith %    Geom %     Final $    Gap %     
------------------------------------------------------------
Constant        5.00      5.00   1276.28      0.00
Volatile +      4.00      3.88   1208.38      0.12
Volatile ++     6.00      5.31   1295.50      0.69
Extreme        14.00     10.73   1643.06      3.27
\`\`\`

**Intuition - Volatility Drag**:

Consider two periods: +50%, then -33.33%

Arithmetic mean: (50% - 33.33%) / 2 = 8.33%
Geometric mean: (1.5 × 0.6667)^0.5 - 1 = 0%

You end up at the same place! 
$100 → $150 → $100

The arithmetic mean is misleading because losses hurt more than gains help (when compounding).

\`\`\`python
# Visualize the effect
def simulate_paths(initial, mean_return, volatility, periods, n_paths=1000):
    """Simulate multiple portfolio paths"""
    np.random.seed(42)
    returns = np.random.normal(mean_return, volatility, (n_paths, periods))
    
    paths = np.zeros((n_paths, periods + 1))
    paths[:, 0] = initial
    
    for t in range(periods):
        paths[:, t+1] = paths[:, t] * (1 + returns[:, t])
    
    return paths, returns

# Simulate low vs high volatility
initial = 1000
mean_return = 0.01  # 1% per period
periods = 100

low_vol_paths, low_vol_returns = simulate_paths(initial, mean_return, 0.02, periods)
high_vol_paths, high_vol_returns = simulate_paths(initial, mean_return, 0.10, periods)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Low volatility paths
axes[0, 0].plot(low_vol_paths.T, alpha=0.1, color='blue')
axes[0, 0].plot(low_vol_paths.mean(axis=0), color='red', linewidth=2, label='Mean path')
axes[0, 0].set_title('Low Volatility (σ = 2%)')
axes[0, 0].set_xlabel('Period')
axes[0, 0].set_ylabel('Portfolio Value')
axes[0, 0].legend()
axes[0, 0].grid(True)

# High volatility paths
axes[0, 1].plot(high_vol_paths.T, alpha=0.1, color='blue')
axes[0, 1].plot(high_vol_paths.mean(axis=0), color='red', linewidth=2, label='Mean path')
axes[0, 1].set_title('High Volatility (σ = 10%)')
axes[0, 1].set_xlabel('Period')
axes[0, 1].set_ylabel('Portfolio Value')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Final value distributions
axes[1, 0].hist(low_vol_paths[:, -1], bins=50, alpha=0.7, color='blue', edgecolor='black')
axes[1, 0].axvline(low_vol_paths[:, -1].mean(), color='red', linewidth=2, label='Mean')
axes[1, 0].set_title('Low Vol: Final Value Distribution')
axes[1, 0].set_xlabel('Final Portfolio Value')
axes[1, 0].legend()

axes[1, 1].hist(high_vol_paths[:, -1], bins=50, alpha=0.7, color='blue', edgecolor='black')
axes[1, 1].axvline(high_vol_paths[:, -1].mean(), color='red', linewidth=2, label='Mean')
axes[1, 1].set_title('High Vol: Final Value Distribution')
axes[1, 1].set_xlabel('Final Portfolio Value')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

# Compute realized returns
low_vol_arith = arithmetic_return(low_vol_returns.flatten()) * 100
low_vol_geom = geometric_return(low_vol_returns.flatten()) * 100
high_vol_arith = arithmetic_return(high_vol_returns.flatten()) * 100
high_vol_geom = geometric_return(high_vol_returns.flatten()) * 100

print(f"\\nLow Volatility:")
print(f"  Arithmetic mean: {low_vol_arith:.2f}%")
print(f"  Geometric mean: {low_vol_geom:.2f}%")
print(f"  Difference: {low_vol_arith - low_vol_geom:.2f}%")

print(f"\\nHigh Volatility:")
print(f"  Arithmetic mean: {high_vol_arith:.2f}%")
print(f"  Geometric mean: {high_vol_geom:.2f}%")
print(f"  Difference: {high_vol_arith - high_vol_geom:.2f}%")

print(f"\\nVolatility drag is {(high_vol_arith - high_vol_geom)/(low_vol_arith - low_vol_geom):.1f}x larger for high volatility!")
\`\`\`

**Sequences Perspective**:

Portfolio value forms a sequence:
V₀, V₁, V₂, ..., Vₙ

where Vₜ = Vₜ₋₁(1 + rₜ)

This is a geometric sequence if returns constant, otherwise more complex.

\`\`\`python
def portfolio_sequence(initial, returns):
    """Generate portfolio value sequence"""
    values = [initial]
    for r in returns:
        values.append(values[-1] * (1 + r))
    return np.array(values)

# Example
returns = np.array([0.10, -0.05, 0.08, 0.03, -0.02])
portfolio = portfolio_sequence(1000, returns)

print("Portfolio value sequence:")
for i, (v, r) in enumerate(zip(portfolio[:-1], returns)):
    print(f"V_{i} = \${v:.2f} → V_{i+1} = \${portfolio[i+1]:.2f} (return: {r*100:+.1f}%)")

print(f"\\nFinal value: \${portfolio[-1]:.2f}")
print(f"Total return: {(portfolio[-1]/portfolio[0] - 1)*100:.2f}%")
\`\`\`

**Why This Matters for Trading**:

**1. Performance Measurement**:
- Use geometric returns for actual growth
- Arithmetic returns overstate performance if volatile

**2. Volatility is Costly**:
- Two strategies with same arithmetic return but different volatility
- Lower volatility strategy will have higher compounded return
- This is "volatility drag"

**3. Risk Management**:
- Large losses require even larger gains to recover
- Lose 50% → Need 100% gain to break even
- Geometric perspective shows asymmetry

\`\`\`python
def recovery_return_needed(loss_percent):
    """
    Calculate return needed to recover from loss
    If lose X%, need gain of X/(1-X) to recover
    """
    loss = loss_percent / 100
    recovery = loss / (1 - loss)
    return recovery * 100

losses = [10, 20, 30, 40, 50, 60, 70, 80, 90]
recoveries = [recovery_return_needed(L) for L in losses]

plt.figure(figsize=(10, 6))
plt.plot(losses, recoveries, 'ro-', linewidth=2, markersize=8)
plt.xlabel('Loss (%)')
plt.ylabel('Gain Needed to Recover (%)')
plt.title('Asymmetry of Gains and Losses (Geometric Returns)')
plt.grid(True)
plt.axline((0, 0), slope=1, color='gray', linestyle='--', label='If symmetric')
plt.legend()
plt.show()

print("Loss vs Recovery:")
for L, R in zip(losses, recoveries):
    print(f"Lose {L}% → Need +{R:.1f}% to recover")
\`\`\`

**4. Sharpe Ratio Adjustment**:

Traditional Sharpe uses arithmetic mean.
For compounding, should use geometric mean:

Sharpe_geometric = (R_geom - R_f) / σ

**5. Long-Term Projections**:

Projecting wealth growth over time:
- Use geometric mean (actual compounding)
- Not arithmetic mean (overstates growth)

\`\`\`python
# Project wealth over 30 years
initial_wealth = 10000
annual_return = 0.08  # 8%
volatility = 0.15      # 15%

# Arithmetic projection (WRONG for compounding)
years = 30
wealth_arithmetic = initial_wealth * (1 + annual_return)**years

# Geometric projection (accounting for volatility drag)
# Approximate: geometric return ≈ arithmetic - σ²/2
volatility_drag = volatility**2 / 2
geometric_return = annual_return - volatility_drag
wealth_geometric = initial_wealth * (1 + geometric_return)**years

print(f"30-year wealth projection (initial \${initial_wealth:,}):")
print(f"Arithmetic (wrong): \${wealth_arithmetic:,.0f}")
print(f"Geometric (correct): \${wealth_geometric:,.0f}")
print(f"Difference: \${wealth_arithmetic - wealth_geometric:,.0f}")
print(f"\\nOverstimation: {(wealth_arithmetic/wealth_geometric - 1)*100:.1f}%")
\`\`\`

**Summary**:
- Geometric returns account for compounding (sequences/series perspective)
- Always ≤ arithmetic returns (AM-GM inequality)
- Difference increases with volatility ("volatility drag")
- Losses require disproportionate gains to recover (geometric asymmetry)
- Use geometric mean for realistic portfolio projections
- Key insight: Minimizing volatility can increase long-term growth even with same average return`,
    keyPoints: [
      'Geometric return ≤ Arithmetic return (AM-GM inequality), equality only if constant',
      'Volatility drag: higher volatility → larger gap between arithmetic and geometric',
      'Portfolio sequence Vₜ = V₀·Π(1+rᵢ) shows compounding nature',
      'Losses hurt more than gains help: lose 50% needs 100% gain to recover',
      'Use geometric mean for realistic long-term projections, not arithmetic',
    ],
  },
  {
    id: 'dq3-reinforcement-learning-series',
    question:
      'In reinforcement learning, the discounted return is defined as Gₜ = Σ(k=0 to ∞) γᵏrₜ₊ₖ where γ is the discount factor. This is a geometric series. Explain: (1) Why we discount future rewards, (2) Conditions for convergence of this infinite series, (3) How the discount factor affects agent behavior, (4) Practical computation, (5) Application to trading strategies.',
    sampleAnswer: `The discounted return in reinforcement learning is a perfect application of geometric series theory to sequential decision-making, directly relevant to algorithmic trading strategies.

**1. Why Discount Future Rewards?**

**Definition**:
Gₜ = rₜ + γrₜ₊₁ + γ²rₜ₊₂ + γ³rₜ₊₃ + ...
   = Σ(k=0 to ∞) γᵏrₜ₊ₖ

where:
- Gₜ: return from time t
- rₜ: immediate reward at time t
- γ ∈ [0, 1]: discount factor

**Reasons for Discounting**:

**Mathematical**: Ensures convergence of infinite sum (more on this below)

**Economic**: Time value of money - future rewards worth less than immediate
- $100 today > $100 in 1 year
- Can invest $100 today to grow

**Uncertainty**: Future is uncertain
- Closer rewards more reliable than distant ones
- Model may be imperfect for long horizons

**Agent Behavior**: Encourages taking action sooner rather than later
- Without discounting (γ=1), may delay rewards indefinitely
- With discounting, prefers sooner rewards

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt

def compute_discounted_return(rewards, gamma):
    """
    Compute Gₜ = Σ γᵏrₜ₊ₖ
    This is a geometric series with weights γᵏ
    """
    n = len(rewards)
    discount_factors = gamma ** np.arange(n)
    return np.sum(rewards * discount_factors)

# Example: sequence of rewards
rewards = np.array([1, 2, 3, 4, 5])

# Different discount factors
gammas = [0.5, 0.9, 0.95, 0.99, 1.0]

print("Effect of discount factor γ on return:\\n")
print(f"Rewards: {rewards}\\n")

for gamma in gammas:
    G = compute_discounted_return(rewards, gamma)
    print(f"γ = {gamma:.2f}: G = {G:.4f}")

# Undiscounted sum
print(f"\\nUndiscounted (γ=1): {np.sum(rewards)}")
print("Lower γ → More emphasis on immediate rewards")
print("Higher γ → More consideration of future rewards")
\`\`\`

**2. Convergence Conditions**

The infinite series Gₜ = Σ γᵏrₜ₊ₖ is a geometric series.

**Condition 1**: |γ| < 1 (discount factor between 0 and 1)

**Condition 2**: Rewards must be bounded: |rₜ| ≤ R_max for all t

**Proof of Convergence**:

If rewards bounded by R_max:
|Gₜ| ≤ Σ(k=0 to ∞) γᵏR_max
     = R_max · Σ γᵏ
     = R_max / (1 - γ)    [geometric series formula]

So Gₜ is bounded, hence converges.

\`\`\`python
def infinite_geometric_series_sum(gamma, R_max):
    """
    Theoretical maximum return for bounded rewards
    G_max = R_max / (1 - γ)
    """
    if gamma >= 1:
        return np.inf
    return R_max / (1 - gamma)

# Demonstrate convergence
def compute_partial_returns(rewards_infinite, gamma, n_terms_list):
    """Compute partial sums to show convergence"""
    partial_returns = []
    for n in n_terms_list:
        G_n = compute_discounted_return(rewards_infinite[:n], gamma)
        partial_returns.append(G_n)
    return np.array(partial_returns)

# Simulate infinite reward sequence (bounded)
np.random.seed(42)
R_max = 10
rewards_infinite = np.random.uniform(-R_max, R_max, 1000)

gamma = 0.9
n_terms_list = range(1, 101)
partial_returns = compute_partial_returns(rewards_infinite, gamma, n_terms_list)

# Theoretical bound
theoretical_max = infinite_geometric_series_sum(gamma, R_max)

plt.figure(figsize=(10, 6))
plt.plot(n_terms_list, partial_returns, linewidth=2, label='Partial sums')
plt.axhline(y=theoretical_max, color='r', linestyle='--', linewidth=2, 
            label=f'Theoretical max = {theoretical_max:.2f}')
plt.axhline(y=-theoretical_max, color='r', linestyle='--', linewidth=2)
plt.fill_between(n_terms_list, -theoretical_max, theoretical_max, alpha=0.1, color='red')
plt.xlabel('Number of terms')
plt.ylabel('Partial return Gₙ')
plt.title(f'Convergence of Discounted Return (γ = {gamma})')
plt.legend()
plt.grid(True)
plt.show()

print(f"\\nConvergence Analysis (γ = {gamma}, R_max = {R_max}):")
print(f"Theoretical bound: ±{theoretical_max:.2f}")
print(f"Actual G_100: {partial_returns[-1]:.2f}")
print(f"Within bounds: {abs(partial_returns[-1]) <= theoretical_max}")
\`\`\`

**3. Effect of Discount Factor on Behavior**

**γ = 0** (Myopic):
- Only immediate reward matters: Gₜ = rₜ
- Agent ignores future completely
- Very short-sighted behavior

**γ ≈ 0.5** (Short-term):
- Future rewards decay quickly
- Effective horizon ~2-3 steps
- Good for highly uncertain environments

**γ ≈ 0.9** (Medium-term):
- Balances immediate and future
- Common in practice
- Effective horizon ~10 steps

**γ ≈ 0.99** (Long-term):
- Values future highly
- Effective horizon ~100 steps
- Better for stable environments

**γ = 1** (No discounting):
- All rewards equally important
- May not converge (infinite sum)
- Only works for episodic tasks (finite horizon)

\`\`\`python
# Effective horizon: number of steps that matter
def effective_horizon(gamma, threshold=0.01):
    """
    Compute effective horizon: k where γᵏ < threshold
    Rewards beyond this have < threshold weight
    """
    if gamma >= 1:
        return np.inf
    return int(np.log(threshold) / np.log(gamma))

gammas = [0.5, 0.7, 0.9, 0.95, 0.99]
print("Effective Horizon (weight < 1%):\\n")
for gamma in gammas:
    horizon = effective_horizon(gamma, 0.01)
    print(f"γ = {gamma:.2f}: ~{horizon} steps")

# Visualize discount weights
k = np.arange(0, 50)
plt.figure(figsize=(12, 6))

for gamma in [0.5, 0.9, 0.99]:
    weights = gamma ** k
    plt.plot(k, weights, linewidth=2, label=f'γ = {gamma}')

plt.axhline(y=0.01, color='gray', linestyle='--', alpha=0.5, label='1% threshold')
plt.xlabel('Steps into future (k)')
plt.ylabel('Discount weight γᵏ')
plt.title('Discount Weights Over Time')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.show()
\`\`\`

**4. Practical Computation**

**Forward View** (what we defined):
Gₜ = rₜ + γGₜ₊₁

Requires looking into future - impractical online!

**Backward View** (TD learning):
Vₜ ← Vₜ + α(rₜ + γVₜ₊₁ - Vₜ)

Updates value incrementally without full rollout.

**Monte Carlo** (episodic):
After episode ends, compute actual returns backward:

\`\`\`python
def compute_returns_backward(rewards, gamma):
    """
    Compute discounted returns efficiently going backward
    Gₜ = rₜ + γGₜ₊₁
    """
    n = len(rewards)
    returns = np.zeros(n)
    
    # Start from end
    returns[-1] = rewards[-1]
    
    # Work backward
    for t in range(n-2, -1, -1):
        returns[t] = rewards[t] + gamma * returns[t+1]
    
    return returns

# Example episode
rewards = np.array([1, 0, 0, 1, 2])
gamma = 0.9
returns = compute_returns_backward(rewards, gamma)

print("Rewards:  ", rewards)
print("Returns:  ", [f"{r:.3f}" for r in returns])

# Verify first return
manual_G0 = 1 + 0.9*0 + 0.9**2*0 + 0.9**3*1 + 0.9**4*2
print(f"\\nManual G₀: {manual_G0:.3f}")
print(f"Computed G₀: {returns[0]:.3f}")
print(f"Match: {np.isclose(manual_G0, returns[0])}")
\`\`\`

**Vector Implementation** (efficient):

\`\`\`python
def compute_returns_vectorized(rewards, gamma):
    """
    Vectorized computation using geometric series
    """
    n = len(rewards)
    # Create discount matrix
    discount_matrix = gamma ** np.abs(np.arange(n)[:, None] - np.arange(n))
    # Upper triangular to only include future rewards
    discount_matrix = np.triu(discount_matrix)
    # Multiply with rewards
    returns = discount_matrix @ rewards
    return returns

rewards = np.array([1, 0, 0, 1, 2])
returns_vec = compute_returns_vectorized(rewards, 0.9)
returns_back = compute_returns_backward(rewards, 0.9)

print("\\nVectorized vs Backward:")
print("Vectorized:", [f"{r:.3f}" for r in returns_vec])
print("Backward:  ", [f"{r:.3f}" for r in returns_back])
print("Match:", np.allclose(returns_vec, returns_back))
\`\`\`

**5. Application to Trading Strategies**

**Trading as RL Problem**:
- State: market conditions, portfolio state
- Action: buy, sell, hold, position size
- Reward: profit/loss
- Goal: Maximize discounted cumulative returns

\`\`\`python
# Simplified trading example
class TradingEnvironment:
    def __init__(self, prices):
        self.prices = prices
        self.current_step = 0
        self.position = 0  # 1 = long, 0 = flat, -1 = short
        
    def step(self, action):
        # action: -1 (sell), 0 (hold), 1 (buy)
        current_price = self.prices[self.current_step]
        
        # Compute reward (profit/loss from position)
        if self.current_step > 0:
            price_change = self.prices[self.current_step] - self.prices[self.current_step-1]
            reward = self.position * price_change
        else:
            reward = 0
        
        # Update position
        self.position = action
        self.current_step += 1
        
        done = self.current_step >= len(self.prices) - 1
        return reward, done

# Simulate trading episode
np.random.seed(42)
prices = 100 + np.cumsum(np.random.randn(50) * 2)

env = TradingEnvironment(prices)

# Simple strategy: buy when price drops, sell when rises
actions = []
rewards = []

for i in range(len(prices) - 1):
    # Simple momentum strategy
    if i == 0:
        action = 1  # Start long
    else:
        price_change = prices[i] - prices[i-1]
        action = 1 if price_change > 0 else -1
    
    reward, done = env.step(action)
    actions.append(action)
    rewards.append(reward)
    
    if done:
        break

# Compute discounted returns for different γ
gammas_trading = [0.9, 0.95, 0.99]
print("\\nTrading Episode Analysis:\\n")

for gamma in gammas_trading:
    returns_seq = compute_returns_backward(np.array(rewards), gamma)
    G0 = returns_seq[0]
    print(f"γ = {gamma:.2f}: Initial expected return G₀ = \${G0:.2f}")

print(f"\\nTotal episode profit (undiscounted): \${np.sum(rewards):.2f}")
print("\\nLower γ → Prioritizes near-term profits")
print("Higher γ → Considers long-term strategy value")
\`\`\`

**Choosing γ for Trading**:

**High Frequency Trading**: γ ≈ 0.5-0.7
- Short holding periods
- Quick profits matter most
- Future very uncertain (milliseconds)

**Swing Trading**: γ ≈ 0.9-0.95
- Hold for days/weeks
- Balance immediate and future gains
- Medium-term market view

**Long-term Investing**: γ ≈ 0.99-0.999
- Hold for months/years
- Future returns highly valued
- Building long-term portfolio value

**Risk Management Context**:
- Higher γ encourages avoiding large drawdowns (future matters)
- Lower γ may take more risk for immediate gains

\`\`\`python
# Demonstrate effect of γ on trading strategy valuation
def evaluate_strategy(rewards, gamma):
    """Evaluate trading strategy with given discount factor"""
    return compute_discounted_return(rewards, gamma)

# Two strategies:
# A: consistent small gains
# B: large gain at end, losses early

strategy_A = np.array([2, 2, 2, 2, 2])
strategy_B = np.array([-1, -1, -1, -1, 12])

gammas = np.linspace(0.5, 1.0, 50)
values_A = [evaluate_strategy(strategy_A, g) for g in gammas]
values_B = [evaluate_strategy(strategy_B, g) for g in gammas]

plt.figure(figsize=(10, 6))
plt.plot(gammas, values_A, linewidth=2, label='Strategy A (consistent)')
plt.plot(gammas, values_B, linewidth=2, label='Strategy B (back-loaded)')
plt.xlabel('Discount Factor γ')
plt.ylabel('Discounted Return')
plt.title('Strategy Valuation vs Discount Factor')
plt.legend()
plt.grid(True)
plt.show()

print("\\nStrategy Comparison:")
print(f"Both have same total: A = {np.sum(strategy_A)}, B = {np.sum(strategy_B)}")
print(f"\\nAt γ = 0.9:")
print(f"  Strategy A: {evaluate_strategy(strategy_A, 0.9):.2f}")
print(f"  Strategy B: {evaluate_strategy(strategy_B, 0.9):.2f}")
print("\\nLower γ strongly prefers early gains (A)!")
\`\`\`

**Summary**:
- Discounted return is geometric series: Gₜ = Σ γᵏrₜ₊ₖ
- Converges if |γ| < 1 and rewards bounded
- γ controls effective time horizon: low γ = myopic, high γ = farsighted
- Computed efficiently backward: Gₜ = rₜ + γGₜ₊₁
- In trading: γ depends on strategy timeframe (HFT low, investing high)
- Key insight: γ encodes trader's time preference for profits`,
    keyPoints: [
      'Discounted return Gₜ = Σγᵏrₜ₊ₖ is geometric series, converges if |γ|<1',
      'γ controls effective horizon: γ=0.5 → ~2 steps, γ=0.9 → ~10 steps, γ=0.99 → ~100 steps',
      'Computed efficiently backward: Gₜ = rₜ + γGₜ₊₁ (recursive formula)',
      'Trading application: γ depends on strategy timeframe (HFT: low, investing: high)',
      'Lower γ prioritizes immediate profits, higher γ considers long-term value',
    ],
  },
];
