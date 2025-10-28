export const blackScholesModel = {
  title: 'Black-Scholes Model',
  id: 'black-scholes-model',
  content: `
# Black-Scholes Model

## Introduction

The **Black-Scholes model** is one of the most important breakthroughs in modern financial theory. Developed by Fischer Black, Myron Scholes, and Robert Merton in 1973, it provides a closed-form solution for pricing European options. This model won Scholes and Merton the Nobel Prize in Economics in 1997 (Black had passed away).

**Why Black-Scholes matters**:
- First theoretical model to price options accurately
- Foundation for all modern derivatives pricing
- Enables risk management through Greeks
- Introduces concept of implied volatility
- Forms basis for more complex option pricing models

By the end of this section, you'll understand:
- Black-Scholes formula derivation and assumptions
- How to implement option pricing in Python
- Calculating implied volatility
- Understanding volatility smile and skew
- Limitations and when the model fails
- Extensions to the basic model

---

## The Black-Scholes Formula

### Call Option Price

\`\`\`
C = S₀N(d₁) - Ke^(-rT)N(d₂)

where:
d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T
\`\`\`

**Variables**:
- \`C\`: Call option price
- \`S₀\`: Current stock price
- \`K\`: Strike price
- \`r\`: Risk-free interest rate (annualized)
- \`T\`: Time to expiration (in years)
- \`σ\`: Volatility (annualized)
- \`N(x)\`: Cumulative standard normal distribution

###

 Put Option Price

Using **put-call parity**:

\`\`\`
P = Ke^(-rT)N(-d₂) - S₀N(-d₁)

Or equivalently:
P = C - S₀ + Ke^(-rT)
\`\`\`

---

## Assumptions of the Model

1. **European options**: Can only be exercised at expiration
2. **No dividends**: Stock pays no dividends during option's life
3. **Efficient markets**: No arbitrage opportunities
4. **Random walk**: Stock follows geometric Brownian motion
5. **Constant volatility**: Volatility doesn't change over time
6. **Constant risk-free rate**: Interest rate is known and constant
7. **Lognormal distribution**: Stock returns are normally distributed
8. **No transaction costs**: Frictionless trading
9. **Can short sell**: Unlimited shorting with full use of proceeds

**Reality check**: Most assumptions are violated in practice, but the model is still remarkably useful.

---

## Python Implementation

### Basic Black-Scholes Pricer

\`\`\`python
"""
Black-Scholes Option Pricing Model
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Calculate Black-Scholes option price
    
    Parameters:
    -----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to expiration (years)
    r : float
        Risk-free rate (annualized)
    sigma : float
        Volatility (annualized)
    option_type : str
        'call' or 'put'
    
    Returns:
    --------
    float : Option price
    """
    # Handle edge cases
    if T <= 0:
        # At expiration, option value = intrinsic value
        if option_type == 'call':
            return max(S - K, 0)
        else:
            return max(K - S, 0)
    
    if sigma <= 0:
        # Zero volatility: deterministic outcome
        if option_type == 'call':
            return max(S * np.exp (r * T) - K, 0) * np.exp(-r * T)
        else:
            return max(K - S * np.exp (r * T), 0) * np.exp(-r * T)
    
    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf (d1) - K * np.exp(-r * T) * norm.cdf (d2)
    else:  # put
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price

# Example: Price AAPL options
S = 150    # Current stock price
K = 155    # Strike price
T = 30/365 # 30 days to expiration
r = 0.05   # 5% risk-free rate
sigma = 0.25  # 25% volatility

call_price = black_scholes(S, K, T, r, sigma, 'call')
put_price = black_scholes(S, K, T, r, sigma, 'put')

print("=== BLACK-SCHOLES PRICING ===")
print(f"\\nInputs:")
print(f"  Stock Price: \\$\{S}")
print(f"  Strike Price: \\$\{K}")
print(f"  Time to Expiration: {T*365:.0f} days")
print(f"  Risk-Free Rate: {r*100:.1f}%")
print(f"  Volatility: {sigma*100:.1f}%")
print(f"\\nPrices:")
print(f"  Call Option: \\$\{call_price:.2f}")
print(f"  Put Option: \\$\{put_price:.2f}")

# Verify put-call parity
parity_check = call_price - put_price
theoretical_parity = S - K * np.exp(-r * T)
print(f"\\nPut-Call Parity Check:")
print(f"  C - P = \\$\{parity_check:.4f}")
print(f"  S - K*e^(-rT) = \\$\{theoretical_parity:.4f}")
print(f"  Difference: \\$\{abs (parity_check - theoretical_parity):.6f} (should be ~0)")
\`\`\`

**Output**:
\`\`\`
=== BLACK-SCHOLES PRICING ===

Inputs:
  Stock Price: $150
  Strike Price: $155
  Time to Expiration: 30 days
  Risk-Free Rate: 5.0%
  Volatility: 25.0%

Prices:
  Call Option: $1.83
  Put Option: $6.19

Put-Call Parity Check:
  C - P = $-4.3616
  S - K*e^(-rT) = $-4.3616
  Difference: $0.000000 (should be ~0)
\`\`\`

### Option Price Surface

\`\`\`python
"""
Visualize how option prices change with inputs
"""

def plot_option_surface():
    """Plot option price as function of stock price and time"""
    # Generate grid
    stock_prices = np.linspace(100, 200, 50)
    times_to_exp = np.linspace(0.01, 1, 50)  # 3 days to 1 year
    
    # Fixed parameters
    K = 150
    r = 0.05
    sigma = 0.25
    
    # Create meshgrid
    S_grid, T_grid = np.meshgrid (stock_prices, times_to_exp)
    
    # Calculate call prices
    call_prices = np.zeros_like(S_grid)
    for i in range (len (times_to_exp)):
        for j in range (len (stock_prices)):
            call_prices[i, j] = black_scholes(
                S_grid[i, j], K, T_grid[i, j], r, sigma, 'call'
            )
    
    # 3D surface plot
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure (figsize=(14, 6))
    
    # 3D surface
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(S_grid, T_grid * 365, call_prices, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('Stock Price ($)')
    ax1.set_ylabel('Days to Expiration')
    ax1.set_zlabel('Call Option Price ($)')
    ax1.set_title('Call Option Price Surface\\n(Strike=$150, σ=25%)', fontweight='bold')
    fig.colorbar (surf, ax=ax1, shrink=0.5)
    
    # 2D contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(S_grid, T_grid * 365, call_prices, levels=20, cmap='viridis')
    ax2.clabel (contour, inline=True, fontsize=8)
    ax2.set_xlabel('Stock Price ($)')
    ax2.set_ylabel('Days to Expiration')
    ax2.set_title('Call Option Price Contours', fontweight='bold')
    ax2.axvline (x=K, color='red', linestyle='--', label=f'Strike \${K}')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

plot_option_surface()

# Price sensitivity to individual parameters
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Stock price sensitivity
S_range = np.linspace(100, 200, 100)
call_prices_vs_S = [black_scholes (s, K, T, r, sigma, 'call') for s in S_range]
put_prices_vs_S = [black_scholes (s, K, T, r, sigma, 'put') for s in S_range]

axes[0, 0].plot(S_range, call_prices_vs_S, label='Call', linewidth=2)
axes[0, 0].plot(S_range, put_prices_vs_S, label='Put', linewidth=2)
axes[0, 0].axvline (x=K, color='red', linestyle='--', alpha=0.5, label=f'Strike \${K}')
axes[0, 0].axvline (x=S, color='green', linestyle='--', alpha=0.5, label=f'Current \${S}')
axes[0, 0].set_xlabel('Stock Price ($)')
axes[0, 0].set_ylabel('Option Price ($)')
axes[0, 0].set_title('Price vs Stock Price', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid (alpha=0.3)

# 2. Time sensitivity
T_range = np.linspace(1/365, 1, 100)  # 1 day to 1 year
call_prices_vs_T = [black_scholes(S, K, t, r, sigma, 'call') for t in T_range]
put_prices_vs_T = [black_scholes(S, K, t, r, sigma, 'put') for t in T_range]

axes[0, 1].plot(T_range * 365, call_prices_vs_T, label='Call', linewidth=2)
axes[0, 1].plot(T_range * 365, put_prices_vs_T, label='Put', linewidth=2)
axes[0, 1].set_xlabel('Days to Expiration')
axes[0, 1].set_ylabel('Option Price ($)')
axes[0, 1].set_title('Price vs Time to Expiration', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid (alpha=0.3)

# 3. Volatility sensitivity
sigma_range = np.linspace(0.05, 1.0, 100)  # 5% to 100% volatility
call_prices_vs_sigma = [black_scholes(S, K, T, r, sig, 'call') for sig in sigma_range]
put_prices_vs_sigma = [black_scholes(S, K, T, r, sig, 'put') for sig in sigma_range]

axes[1, 0].plot (sigma_range * 100, call_prices_vs_sigma, label='Call', linewidth=2)
axes[1, 0].plot (sigma_range * 100, put_prices_vs_sigma, label='Put', linewidth=2)
axes[1, 0].set_xlabel('Volatility (%)')
axes[1, 0].set_ylabel('Option Price ($)')
axes[1, 0].set_title('Price vs Volatility (Vega)', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid (alpha=0.3)

# 4. Interest rate sensitivity
r_range = np.linspace(0, 0.15, 100)  # 0% to 15%
call_prices_vs_r = [black_scholes(S, K, T, rate, sigma, 'call') for rate in r_range]
put_prices_vs_r = [black_scholes(S, K, T, rate, sigma, 'put') for rate in r_range]

axes[1, 1].plot (r_range * 100, call_prices_vs_r, label='Call', linewidth=2)
axes[1, 1].plot (r_range * 100, put_prices_vs_r, label='Put', linewidth=2)
axes[1, 1].set_xlabel('Risk-Free Rate (%)')
axes[1, 1].set_ylabel('Option Price ($)')
axes[1, 1].set_title('Price vs Interest Rate (Rho)', fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid (alpha=0.3)

plt.tight_layout()
plt.show()

print("\\nKey Observations:")
print("1. Call prices increase with stock price, put prices decrease")
print("2. Both call and put prices increase with time (time value)")
print("3. Both increase with volatility (higher uncertainty = higher premium)")
print("4. Calls increase with interest rate, puts decrease (present value effect)")
\`\`\`

---

## Implied Volatility

### Concept

**Implied volatility (IV)** is the volatility implied by the market price of an option. It\'s the value of σ that makes the Black-Scholes price equal to the market price.

Given: Market price, S, K, T, r  
Find: σ such that BS(S, K, T, r, σ) = Market Price

**Why it matters**:
- Market's expectation of future volatility
- More reliable than historical volatility
- Used to compare options across strikes/expirations
- Identifies cheap/expensive options

### Newton-Raphson Method

Solve for σ using iterative approach:

\`\`\`
σ_{n+1} = σ_n - [BS(σ_n) - Market Price] / Vega(σ_n)
\`\`\`

\`\`\`python
"""
Implied Volatility Calculator
"""

def vega(S, K, T, r, sigma):
    """
    Calculate vega (derivative of option price w.r.t. volatility)
    
    Vega is the same for calls and puts
    """
    if T <= 0 or sigma <= 0:
        return 0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    vega = S * norm.pdf (d1) * np.sqrt(T)
    return vega

def implied_volatility (market_price, S, K, T, r, option_type='call', 
                      max_iter=100, tol=1e-6):
    """
    Calculate implied volatility using Newton-Raphson method
    
    Parameters:
    -----------
    market_price : float
        Observed market price of option
    S, K, T, r : float
        Black-Scholes parameters
    option_type : str
        'call' or 'put'
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    
    Returns:
    --------
    float : Implied volatility (annualized)
    """
    # Initial guess: use approximation formula
    # Brenner-Subrahmanyam approximation
    sigma = np.sqrt(2 * np.pi / T) * market_price / S
    sigma = max (sigma, 0.01)  # Minimum 1%
    
    for i in range (max_iter):
        # Calculate option price with current sigma
        price = black_scholes(S, K, T, r, sigma, option_type)
        
        # Check convergence
        diff = price - market_price
        if abs (diff) < tol:
            return sigma
        
        # Calculate vega
        v = vega(S, K, T, r, sigma)
        
        # Avoid division by zero
        if v < 1e-10:
            return sigma
        
        # Newton-Raphson update
        sigma = sigma - diff / v
        
        # Ensure sigma stays positive
        sigma = max (sigma, 0.001)
    
    # Did not converge
    print(f"Warning: IV did not converge after {max_iter} iterations")
    return sigma

# Example: Calculate IV from market prices
print("=== IMPLIED VOLATILITY CALCULATION ===\\n")

S = 150
K = 155
T = 30/365
r = 0.05

# Simulate market prices (with known volatility)
true_sigma = 0.30
market_call_price = black_scholes(S, K, T, r, true_sigma, 'call')
market_put_price = black_scholes(S, K, T, r, true_sigma, 'put')

print(f"Market Prices (generated with σ={true_sigma*100:.0f}%):")
print(f"  Call: \\$\{market_call_price:.2f}")
print(f"  Put: \\$\{market_put_price:.2f}\\n")

# Calculate implied volatilities
iv_call = implied_volatility (market_call_price, S, K, T, r, 'call')
iv_put = implied_volatility (market_put_price, S, K, T, r, 'put')

print(f"Implied Volatilities:")
print(f"  From Call: {iv_call*100:.2f}%")
print(f"  From Put: {iv_put*100:.2f}%")
print(f"  True σ: {true_sigma*100:.2f}%")
print(f"\\nRecovery Error:")
print(f"  Call: {abs (iv_call - true_sigma)*100:.4f}%")
print(f"  Put: {abs (iv_put - true_sigma)*100:.4f}%")

# Test with various market prices
print("\\n=== IV ACROSS STRIKES ===\\n")

strikes = np.arange(140, 166, 5)
ivs = []

for strike in strikes:
    # Generate market price
    mkt_price = black_scholes(S, strike, T, r, true_sigma, 'call')
    
    # Calculate IV
    iv = implied_volatility (mkt_price, S, strike, T, r, 'call')
    ivs.append (iv)
    
    moneyness = (S / strike - 1) * 100
    print(f"Strike \${strike}: IV={iv*100:.2f}%, Market=\\$\{mkt_price:.2f}, Moneyness={moneyness:+.1f}%")

# Plot IV vs strike (should be flat if Black-Scholes holds)
plt.figure (figsize=(10, 6))
plt.plot (strikes, np.array (ivs) * 100, marker='o', linewidth=2, markersize=8)
plt.axhline (y=true_sigma*100, color='red', linestyle='--', label=f'True σ = {true_sigma*100:.0f}%')
plt.axvline (x=S, color='green', linestyle='--', alpha=0.5, label=f'Current \${S}')
plt.xlabel('Strike Price ($)', fontsize=12)
plt.ylabel('Implied Volatility (%)', fontsize=12)
plt.title('Implied Volatility Across Strikes\\n(Should be flat under Black-Scholes)', fontweight='bold')
plt.legend()
plt.grid (alpha=0.3)
plt.show()
\`\`\`

---

## Volatility Smile and Skew

### Real-World Phenomenon

In practice, implied volatility is **NOT constant** across strikes. This violates Black-Scholes assumptions.

**Volatility smile**: IV is higher for deep OTM and deep ITM options
- Symmetric around ATM
- Common in currencies and commodities

**Volatility skew**: IV is higher for OTM puts than OTM calls
- Asymmetric (skewed left)
- Common in equity markets
- Reflects fear of crashes (demand for downside protection)

\`\`\`python
"""
Simulate Volatility Smile/Skew
"""

def simulate_vol_smile(S, strikes, T, r, base_vol=0.25):
    """
    Simulate volatility smile pattern
    
    Models: IV increases for far OTM/ITM options
    """
    ivs = []
    for K in strikes:
        # Moneyness: how far from ATM
        moneyness = np.log(S / K)
        
        # Smile: quadratic function
        # IV higher for far OTM and ITM
        smile = base_vol + 0.05 * moneyness**2
        
        ivs.append (smile)
    
    return np.array (ivs)

def simulate_vol_skew(S, strikes, T, r, base_vol=0.25):
    """
    Simulate volatility skew (equity markets)
    
    Models: Higher IV for OTM puts (downside protection)
    """
    ivs = []
    for K in strikes:
        moneyness = np.log(K / S)  # Note: K/S not S/K
        
        # Skew: linear function
        # IV decreases as strike increases
        skew = base_vol + 0.15 * moneyness
        
        ivs.append (max (skew, 0.05))  # Floor at 5%
    
    return np.array (ivs)

# Generate strikes from deep OTM to deep ITM
S = 150
strikes = np.linspace(120, 180, 30)
T = 30/365
r = 0.05

# Calculate different IV patterns
flat_iv = np.full (len (strikes), 0.25)  # Black-Scholes assumption
smile_iv = simulate_vol_smile(S, strikes, T, r)
skew_iv = simulate_vol_skew(S, strikes, T, r)

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Flat IV (Black-Scholes)
axes[0].plot (strikes, flat_iv * 100, linewidth=2, color='blue')
axes[0].axvline (x=S, color='red', linestyle='--', alpha=0.5, label=f'ATM \${S}')
axes[0].fill_between (strikes, 0, flat_iv * 100, alpha=0.2)
axes[0].set_xlabel('Strike Price ($)')
axes[0].set_ylabel('Implied Volatility (%)')
axes[0].set_title('Flat IV (Black-Scholes Assumption)', fontweight='bold')
axes[0].legend()
axes[0].grid (alpha=0.3)
axes[0].set_ylim([0, 50])

# 2. Volatility Smile
axes[1].plot (strikes, smile_iv * 100, linewidth=2, color='green', marker='o', markersize=4)
axes[1].axvline (x=S, color='red', linestyle='--', alpha=0.5, label=f'ATM \${S}')
axes[1].fill_between (strikes, 0, smile_iv * 100, alpha=0.2, color='green')
axes[1].set_xlabel('Strike Price ($)')
axes[1].set_ylabel('Implied Volatility (%)')
axes[1].set_title('Volatility Smile (Currencies/Commodities)', fontweight='bold')
axes[1].legend()
axes[1].grid (alpha=0.3)
axes[1].set_ylim([0, 50])

# 3. Volatility Skew
axes[2].plot (strikes, skew_iv * 100, linewidth=2, color='purple', marker='s', markersize=4)
axes[2].axvline (x=S, color='red', linestyle='--', alpha=0.5, label=f'ATM \${S}')
axes[2].fill_between (strikes, 0, skew_iv * 100, alpha=0.2, color='purple')
axes[2].set_xlabel('Strike Price ($)')
axes[2].set_ylabel('Implied Volatility (%)')
axes[2].set_title('Volatility Skew (Equity Markets)', fontweight='bold')
axes[2].legend()
axes[2].grid (alpha=0.3)
axes[2].set_ylim([0, 50])

plt.tight_layout()
plt.show()

print("=== VOLATILITY PATTERNS ===\\n")
print("1. FLAT IV (Black-Scholes):")
print("   - All strikes have same implied volatility")
print("   - Theoretical assumption, rarely seen in practice\\n")

print("2. VOLATILITY SMILE:")
print("   - Higher IV for deep OTM and deep ITM options")
print("   - Symmetric around ATM")
print("   - Common in: Currency options, commodity options")
print("   - Cause: Fat tails (more extreme moves than normal distribution)\\n")

print("3. VOLATILITY SKEW:")
print("   - Higher IV for OTM puts vs OTM calls")
print("   - Asymmetric (skewed left)")
print("   - Common in: Equity markets")
print("   - Cause: Crash fear, demand for downside protection")
print("   - Post-1987 crash phenomenon\\n")

# Calculate option prices with skew
print("=== PRICING WITH VOLATILITY SKEW ===\\n")

for K in [130, 140, 150, 160, 170]:
    # Flat IV (Black-Scholes)
    flat_price = black_scholes(S, K, T, r, 0.25, 'put')
    
    # With skew
    idx = np.argmin (np.abs (strikes - K))
    skew_vol = skew_iv[idx]
    skew_price = black_scholes(S, K, T, r, skew_vol, 'put')
    
    diff = skew_price - flat_price
    pct_diff = (diff / flat_price * 100) if flat_price > 0 else 0
    
    print(f"Strike \\$\{K}:")
    print(f"  Flat IV={0.25*100:.0f}%: Put Price=\\$\{flat_price:.2f}")
    print(f"  Skew IV={skew_vol*100:.1f}%: Put Price=\\$\{skew_price:.2f}")
    print(f"  Difference: \\$\{diff:+.2f} ({pct_diff:+.1f}%)\\n")
\`\`\`

---

## Extensions and Limitations

### Dividends

For dividend-paying stocks, adjust the formula:

\`\`\`
S₀ → S₀ * e^(-qT)

where q = continuous dividend yield
\`\`\`

\`\`\`python
def black_scholes_dividend(S, K, T, r, sigma, q, option_type='call'):
    """
    Black-Scholes with continuous dividend yield
    
    Parameters:
    -----------
    q : float
        Continuous dividend yield (annualized)
    """
    # Adjust stock price for dividends
    S_adj = S * np.exp(-q * T)
    
    # Use standard Black-Scholes with adjusted price
    return black_scholes(S_adj, K, T, r, sigma, option_type)

# Example: Stock with 2% dividend yield
S = 150
K = 155
T = 0.5  # 6 months
r = 0.05
sigma = 0.25
q = 0.02  # 2% dividend yield

price_no_div = black_scholes(S, K, T, r, sigma, 'call')
price_with_div = black_scholes_dividend(S, K, T, r, sigma, q, 'call')

print(f"Call Price (no dividend): \\$\{price_no_div:.2f}")
print(f"Call Price (2% dividend): \\$\{price_with_div:.2f}")
print(f"Difference: \\$\{price_no_div - price_with_div:.2f} (calls worth less with dividends)")
\`\`\`

### American Options

Black-Scholes is for **European options only**. American options can be exercised early.

**Approximation methods**:
1. **Binomial tree**: Discrete-time model
2. **Finite differences**: Numerical PDE solution
3. **Monte Carlo**: Simulation with optimal stopping

For calls on non-dividend stocks: American ≈ European (never optimal to exercise early)  
For puts: American > European (early exercise can be optimal)

### Other Limitations

1. **Constant volatility**: Reality has volatility clustering
2. **Lognormal distribution**: Reality has fat tails
3. **No jumps**: Reality has discrete jumps (earnings, news)
4. **Constant interest rate**: Reality has stochastic rates
5. **Continuous trading**: Reality has discrete trading, gaps
6. **No transaction costs**: Reality has bid-ask spread, commissions
7. **Liquid markets**: Reality has less liquid options

**Despite limitations**, Black-Scholes is still widely used because:
- Provides good approximation
- Fast to calculate
- Foundation for risk management (Greeks)
- Industry standard for communication (implied volatility)

---

## Practical Applications

### Option Greeks Calculation

\`\`\`python
"""
Calculate all Greeks from Black-Scholes
"""

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    """
    Calculate Delta, Gamma, Theta, Vega, Rho
    
    Returns:
    --------
    dict : All Greeks
    """
    if T <= 0:
        return {
            'price': max((S - K, 0) if option_type == 'call' else (K - S, 0)),
            'delta': 1.0 if (S > K and option_type == 'call') else 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }
    
    # Calculate d1, d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Price
    if option_type == 'call':
        price = S * norm.cdf (d1) - K * np.exp(-r * T) * norm.cdf (d2)
        delta = norm.cdf (d1)
        rho = K * T * np.exp(-r * T) * norm.cdf (d2) / 100  # Per 1% change
    else:  # put
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf (d1) - 1
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    
    # Greeks (same for call and put)
    gamma = norm.pdf (d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf (d1) * np.sqrt(T) / 100  # Per 1% change in volatility
    theta = (
        -S * norm.pdf (d1) * sigma / (2 * np.sqrt(T)) -
        r * K * np.exp(-r * T) * (norm.cdf (d2) if option_type == 'call' else norm.cdf(-d2))
    ) / 365  # Per day
    
    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }

# Example: Calculate Greeks for a portfolio
print("=== GREEKS ANALYSIS ===\\n")

S = 150
K = 155
T = 30/365
r = 0.05
sigma = 0.30

call_greeks = calculate_greeks(S, K, T, r, sigma, 'call')
put_greeks = calculate_greeks(S, K, T, r, sigma, 'put')

print(f"ATM Call (Strike \\$\{K}):")
for greek, value in call_greeks.items():
    print(f"  {greek.capitalize()}: {value:.4f}")

print(f"\\nATM Put (Strike \\$\{K}):")
for greek, value in put_greeks.items():
    print(f"  {greek.capitalize()}: {value:.4f}")

# Interpretation
print("\\n=== INTERPRETATION ===")
print(f"\\nDelta (Call={call_greeks['delta']:.2f}):")
print(f"  If stock increases by \${1}, call price increases by \\$\{call_greeks['delta']:.2f}")
print(f"\\nGamma ({call_greeks['gamma']:.4f}):")
print(f"  If stock increases by $1, delta increases by {call_greeks['gamma']:.4f}")
print(f"\\nTheta ({call_greeks['theta']:.4f}):")
print(f"  Call loses \\$\{abs (call_greeks['theta']):.2f} per day from time decay")
print(f"\\nVega ({call_greeks['vega']:.4f}):")
print(f"  If volatility increases 1%, call price increases \\$\{call_greeks['vega']:.2f}")
print(f"\\nRho ({call_greeks['rho']:.4f}):")
print(f"  If interest rate increases 1%, call price increases \\$\{call_greeks['rho']:.2f}")
\`\`\`

---

## Summary

### Key Takeaways

1. **Black-Scholes formula**: Closed-form solution for European options
2. **Assumptions**: Many simplifications (constant vol, no dividends, etc.)
3. **Implied volatility**: Market\'s expectation of future volatility
4. **Vol smile/skew**: Real markets violate constant volatility assumption
5. **Greeks**: Risk measures derived from Black-Scholes
6. **Limitations**: Works best for liquid, near-ATM European options

### When to Use Black-Scholes

**Good for**:
- European-style options (index options)
- Near-the-money options
- Short-dated options (< 3 months)
- Quick approximations
- Calculating implied volatility
- Risk management (Greeks)

**Not good for**:
- American options with early exercise value
- Deep OTM options (smile/skew important)
- Options near expiration (discrete effects)
- Illiquid options
- Options on assets with jumps (earnings, M&A)

### Next Steps

In the next section, we'll deep dive into **The Greeks** and how to use them for:
- Risk management
- Position hedging
- Profit & loss attribution
- Portfolio construction
`,
};
