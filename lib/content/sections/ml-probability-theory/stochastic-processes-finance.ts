/**
 * Stochastic Processes for Finance Section
 */

export const stochasticprocessesfinanceSection = {
  id: 'stochastic-processes-finance',
  title: 'Stochastic Processes for Finance',
  content: `# Stochastic Processes for Finance

## Introduction

**Stochastic processes** model random evolution over time. In finance, asset prices, interest rates, and volatility are all modeled as stochastic processes.

**Key applications**:
- Stock price modeling
- Options pricing
- Risk management
- Algorithmic trading
- Portfolio optimization

## Random Walk

A **random walk** is the simplest stochastic process: each step is random and independent.

\\[ S_t = S_0 + \\sum_{i=1}^t X_i \\]

where \\( X_i \\) are i.i.d. random variables.

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt

def random_walk_demo():
    """Demonstrate random walk"""
    
    np.random.seed(42)
    n_steps = 1000
    n_paths = 5
    
    print("=== Random Walk ===")
    print("S_t = S_0 + Σ X_i where X_i ~ N(0,1)")
    print()
    
    plt.figure (figsize=(12, 6))
    
    for i in range (n_paths):
        steps = np.random.randn (n_steps)
        path = np.cumsum (np.concatenate([[0], steps]))
        plt.plot (path, alpha=0.7, label=f'Path {i+1}')
    
    plt.xlabel('Time Step')
    plt.ylabel('Position')
    plt.title('Random Walk: Multiple Paths')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    
    print("Properties:")
    print("- E[S_t] = S_0 (zero drift)")
    print("- Var(S_t) = t·σ² (variance grows with time)")
    print("- Not realistic for stocks (can go negative!)")

random_walk_demo()
\`\`\`

## Geometric Brownian Motion (GBM)

**GBM** is the standard model for stock prices:

\\[ dS_t = \\mu S_t dt + \\sigma S_t dW_t \\]

where:
- μ: drift (expected return)
- σ: volatility
- dW_t: Brownian motion (Wiener process)

**Solution**:
\\[ S_t = S_0 e^{(\\mu - \\sigma^2/2)t + \\sigma W_t} \\]

**Properties**:
- Always positive (suitable for prices)
- Log-normal distribution
- Used in Black-Scholes model

\`\`\`python
def geometric_brownian_motion():
    """Simulate Geometric Brownian Motion"""
    
    # Parameters
    S0 = 100  # Initial price
    mu = 0.1  # 10% annual drift
    sigma = 0.2  # 20% annual volatility
    T = 1  # 1 year
    n_steps = 252  # Trading days
    n_paths = 10
    
    dt = T / n_steps
    np.random.seed(42)
    
    print("=== Geometric Brownian Motion ===")
    print(f"S_0 = \\$\{S0}")
    print(f"μ (drift) = {mu:.1%} per year")
    print(f"σ (volatility) = {sigma:.1%} per year")
    print()
    
    plt.figure (figsize=(12, 6))
    
    final_prices = []
    
    for i in range (n_paths):
        # Generate random shocks
        dW = np.random.randn (n_steps) * np.sqrt (dt)
        
        # Simulate path
        path = [S0]
        S = S0
        for j in range (n_steps):
            S = S * np.exp((mu - 0.5*sigma**2)*dt + sigma*dW[j])
            path.append(S)
        
        final_prices.append(S)
        plt.plot (path, alpha=0.6)
    
    plt.xlabel('Time Step (Days)')
    plt.ylabel('Stock Price ($)')
    plt.title (f'GBM: Stock Price Paths (μ={mu:.1%}, σ={sigma:.1%})')
    plt.grid(True, alpha=0.3)
    plt.axhline(S0, color='red', linestyle='--', label=f'Initial: \${S0}')
    plt.legend()
    
    print(f"Final prices after {T} year:")
    print(f"  Mean: \\$\{np.mean (final_prices):.2f}")
print(f"  Std: \\$\{np.std (final_prices):.2f}")
print(f"  Min: \\$\{np.min (final_prices):.2f}")
print(f"  Max: \\$\{np.max (final_prices):.2f}")

geometric_brownian_motion()
\`\`\`

## Mean Reversion (Ornstein-Uhlenbeck)

**Mean-reverting process**: Tends to return to long-term mean.

\\[ dX_t = \\theta(\\mu - X_t)dt + \\sigma dW_t \\]

where:
- θ: speed of mean reversion
- μ: long-term mean
- σ: volatility

**Used for**: Interest rates, volatility, pairs trading

\`\`\`python
def mean_reversion_demo():
    """Simulate mean-reverting process"""
    
    # Parameters
    X0 = 100
    mu = 100  # Long-term mean
    theta = 0.5  # Speed of reversion
    sigma = 10  # Volatility
    T = 5
    n_steps = 1000
    dt = T / n_steps
    
    np.random.seed(42)
    
    print("=== Mean Reversion (Ornstein-Uhlenbeck) ===")
    print(f"Long-term mean μ = {mu}")
    print(f"Reversion speed θ = {theta}")
    print()
    
    plt.figure (figsize=(12, 6))
    
    for i in range(5):
        path = [X0]
        X = X0
        
        for _ in range (n_steps):
            dX = theta * (mu - X) * dt + sigma * np.sqrt (dt) * np.random.randn()
            X = X + dX
            path.append(X)
        
        plt.plot (path, alpha=0.6)
    
    plt.axhline (mu, color='red', linestyle='--', linewidth=2, label=f'Mean = {mu}')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Mean-Reverting Process: Tends toward long-term mean')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    print("Applications:")
    print("- Interest rate models")
    print("- Volatility models")
    print("- Pairs trading (spread mean-reverts)")
    print("- Commodity prices")

mean_reversion_demo()
\`\`\`

## Poisson Process

**Poisson process**: Models random events occurring at constant rate λ.

**Properties**:
- Number of events in [0,t]: N_t ~ Poisson(λt)
- Time between events: Exponential(λ)

**Used for**: Jump events, defaults, arrivals

\`\`\`python
def poisson_process_demo():
    """Simulate Poisson process"""
    
    lambda_rate = 5  # 5 events per unit time
    T = 10
    
    np.random.seed(42)
    
    # Generate event times
    event_times = []
    t = 0
    while t < T:
        # Time to next event ~ Exponential(λ)
        dt = np.random.exponential(1/lambda_rate)
        t += dt
        if t < T:
            event_times.append (t)
    
    # Counting process
    times = np.linspace(0, T, 1000)
    counts = [np.sum (np.array (event_times) <= t) for t in times]
    
    print("=== Poisson Process ===")
    print(f"Rate λ = {lambda_rate} events per unit time")
    print(f"Total events in [0, {T}]: {len (event_times)}")
    print(f"Expected: λ·T = {lambda_rate * T}")
    print()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Counting process
    ax1.plot (times, counts, linewidth=2)
    ax1.scatter (event_times, range(1, len (event_times)+1), color='red', s=50, zorder=5)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Number of Events')
    ax1.set_title('Poisson Process: Counting Process N(t)')
    ax1.grid(True, alpha=0.3)
    
    # Inter-arrival times
    if len (event_times) > 1:
        inter_arrival = np.diff([0] + event_times)
        ax2.hist (inter_arrival, bins=30, density=True, alpha=0.7, edgecolor='black')
        x = np.linspace(0, max (inter_arrival), 100)
        from scipy import stats
        ax2.plot (x, stats.expon (scale=1/lambda_rate).pdf (x), 'r-', linewidth=2, label='Exponential(λ)')
        ax2.set_xlabel('Inter-arrival Time')
        ax2.set_ylabel('Density')
        ax2.set_title('Time Between Events (should be Exponential)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    print("Applications:")
    print("- Credit defaults")
    print("- Trade arrivals")
    print("- News events")
    print("- System failures")

poisson_process_demo()
\`\`\`

## ML/Trading Applications

\`\`\`python
def trading_applications():
    """Demonstrate trading applications"""
    
    print("=== Stochastic Processes in Trading ===")
    print()
    print("1. Stock Price Modeling:")
    print("   - GBM for individual stocks")
    print("   - Captures drift and volatility")
    print("   - Used in Monte Carlo simulation")
    print()
    print("2. Options Pricing:")
    print("   - Black-Scholes uses GBM")
    print("   - Simulate thousands of price paths")
    print("   - Estimate option value as expectation")
    print()
    print("3. Pairs Trading:")
    print("   - Model spread as mean-reverting")
    print("   - Buy when below mean, sell when above")
    print("   - Ornstein-Uhlenbeck process")
    print()
    print("4. Risk Management:")
    print("   - Simulate portfolio paths")
    print("   - Estimate Value at Risk (VaR)")
    print("   - Stress testing")
    print()
    print("5. High-Frequency Trading:")
    print("   - Poisson process for order arrivals")
    print("   - Market microstructure models")
    print("   - Queueing theory")

trading_applications()
\`\`\`

## Key Takeaways

1. **Random walk**: Simple model, can go negative
2. **GBM**: Standard for stock prices, always positive, log-normal
3. **Mean reversion**: For spreads, interest rates, volatility
4. **Poisson process**: For discrete events, jumps
5. **Applications**: Pricing, risk, trading strategies
6. **Simulation**: Monte Carlo methods use stochastic processes
7. **ML integration**: Feature engineering from price processes

Stochastic processes provide the mathematical foundation for quantitative finance!
`,
};
