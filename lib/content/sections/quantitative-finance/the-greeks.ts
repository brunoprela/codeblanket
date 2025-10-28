export const theGreeks = {
  title: 'The Greeks',
  id: 'the-greeks',
  content: `
# The Greeks

## Introduction

**The Greeks** are measures of option price sensitivity to various factors. Named after Greek letters, they quantify how much an option's price changes when underlying parameters change by a small amount. The Greeks are essential for:

- **Risk management**: Understanding position risks
- **Hedging**: Creating delta-neutral or vega-neutral portfolios
- **P&L attribution**: Understanding where profits/losses come from
- **Position sizing**: Determining appropriate exposure
- **Strategy selection**: Choosing options based on Greek profiles

By the end of this section, you'll understand:
- Delta, Gamma, Theta, Vega, and Rho
- How to calculate and interpret each Greek
- Practical applications in trading and risk management
- Portfolio-level Greek management
- Advanced Greek strategies (gamma scalping, vega trading)

---

## Delta (Δ): Price Sensitivity

### Definition

**Delta** measures how much an option's price changes for a $1 change in the underlying stock price.

\`\`\`
Δ = ∂C/∂S (partial derivative of option price w.r.t. stock price)
\`\`\`

**Ranges**:
- **Call delta**: 0 to +1
- **Put delta**: -1 to 0
- **Stock delta**: Exactly 1

**Interpretation**:
- Delta = 0.50: Option price changes $0.50 for every $1 stock move
- Delta also approximates **probability** of finishing in-the-money
- Delta = 0.70 ≈ 70% chance of ITM at expiration

### Calculation

From Black-Scholes:

\`\`\`
Call Delta: Δ_call = N(d1)
Put Delta: Δ_put = N(d1) - 1 = -N(-d1)

where d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
\`\`\`

\`\`\`python
"""
Delta Calculation and Visualization
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def calculate_delta(S, K, T, r, sigma, option_type='call'):
    """
    Calculate option delta
    
    Returns:
    --------
    float : Delta (0 to 1 for calls, -1 to 0 for puts)
    """
    if T <= 0:
        # At expiration, delta is binary
        if option_type == 'call':
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    
    # Calculate d1
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    if option_type == 'call':
        delta = norm.cdf (d1)
    else:  # put
        delta = norm.cdf (d1) - 1
    
    return delta

# Example calculations
S = 150
K = 150
T = 30/365
r = 0.05
sigma = 0.25

print("=== DELTA EXAMPLES ===\\n")

# ATM option
call_delta = calculate_delta(S, K, T, r, sigma, 'call')
put_delta = calculate_delta(S, K, T, r, sigma, 'put')

print(f"ATM (Strike \${K}, Stock \\$\{S}):")
print(f"  Call Delta: {call_delta:.4f} (~50%)")
print(f"  Put Delta: {put_delta:.4f} (~-50%)")
print(f"  Sum: {call_delta + put_delta:.4f} (should be ≈0)\\n")

# ITM call
call_itm_delta = calculate_delta(S, K-10, T, r, sigma, 'call')
print(f"ITM Call (Strike \${K-10}, Stock \\$\{S}):")
print(f"  Delta: {call_itm_delta:.4f} (~80-90%)")
print(f"  Behaves more like stock\\n")

# OTM put
put_otm_delta = calculate_delta(S, K-10, T, r, sigma, 'put')
print(f"OTM Put (Strike \${K-10}, Stock \\$\{S}):")
print(f"  Delta: {put_otm_delta:.4f} (~-10 to -20%)")
print(f"  Small exposure\\n")

# Visualize delta across strikes and time
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Delta vs Stock Price
stock_range = np.linspace(120, 180, 100)
call_deltas = [calculate_delta (s, K, T, r, sigma, 'call') for s in stock_range]
put_deltas = [calculate_delta (s, K, T, r, sigma, 'put') for s in stock_range]

axes[0].plot (stock_range, call_deltas, label='Call Delta', linewidth=2)
axes[0].plot (stock_range, put_deltas, label='Put Delta', linewidth=2)
axes[0].axhline (y=0, color='black', linestyle='-', alpha=0.3)
axes[0].axhline (y=0.5, color='gray', linestyle='--', alpha=0.3, label='Δ=0.50')
axes[0].axhline (y=-0.5, color='gray', linestyle='--', alpha=0.3)
axes[0].axvline (x=K, color='red', linestyle='--', alpha=0.5, label=f'Strike \${K}')
axes[0].set_xlabel('Stock Price ($)')
axes[0].set_ylabel('Delta')
axes[0].set_title('Delta vs Stock Price', fontweight='bold')
axes[0].legend()
axes[0].grid (alpha=0.3)
axes[0].set_ylim([-1.1, 1.1])

# Delta vs Time to Expiration (for ATM)
time_range = np.linspace(1/365, 1, 100)
call_deltas_time = [calculate_delta(S, K, t, r, sigma, 'call') for t in time_range]

axes[1].plot (time_range * 365, call_deltas_time, linewidth=2, color='green')
axes[1].axhline (y=0.5, color='red', linestyle='--', alpha=0.5, label='Δ=0.50')
axes[1].set_xlabel('Days to Expiration')
axes[1].set_ylabel('Call Delta')
axes[1].set_title('ATM Call Delta vs Time', fontweight='bold')
axes[1].legend()
axes[1].grid (alpha=0.3)
axes[1].set_ylim([0.45, 0.55])

plt.tight_layout()
plt.show()

print("\\nKEY OBSERVATIONS:")
print("1. ATM options have delta ≈ 0.50 (±0.50 for puts)")
print("2. Delta increases as option goes ITM (approaches 1.00 or -1.00)")
print("3. Delta decreases as option goes OTM (approaches 0)")
print("4. ATM delta stays near 0.50 regardless of time (slight drift toward binary)")
\`\`\`

### Practical Applications

**1. Position Equivalent**

Delta tells you stock-equivalent exposure:

\`\`\`python
# Portfolio delta exposure
positions = [
    {'type': 'stock', 'quantity': 100, 'delta': 1.0},
    {'type': 'call', 'quantity': 10, 'contracts': 100, 'delta': 0.60},
    {'type': 'put', 'quantity': -5, 'contracts': 100, 'delta': -0.40},
]

total_delta = 0
for pos in positions:
    if pos['type'] == 'stock':
        delta_exposure = pos['quantity'] * pos['delta']
    else:
        delta_exposure = pos['quantity'] * pos['contracts'] * pos['delta']
    
    total_delta += delta_exposure
    print(f"{pos['type'].title()}: {delta_exposure:.0f} delta")

print(f"\\nTotal Portfolio Delta: {total_delta:.0f} shares")
print(f"Interpretation: Portfolio behaves like {total_delta:.0f} shares of stock")

# Output:
# Stock: 100 delta
# Call: 600 delta (10 contracts × 100 shares × 0.60)
# Put: 200 delta (-5 contracts × 100 shares × -0.40)
# Total: 900 shares equivalent
\`\`\`

**2. Delta Hedging**

Create delta-neutral position (zero directional exposure):

\`\`\`python
def delta_hedge (option_position, option_delta, shares_per_contract=100):
    """
    Calculate hedge for delta-neutral position
    
    Parameters:
    -----------
    option_position : int
        Number of option contracts (positive = long, negative = short)
    option_delta : float
        Delta of the option
    
    Returns:
    --------
    int : Number of shares to buy/sell (negative = sell/short)
    """
    option_delta_exposure = option_position * shares_per_contract * option_delta
    hedge_shares = -option_delta_exposure  # Opposite sign
    
    return hedge_shares

# Example: Long 10 call contracts (delta 0.70)
option_pos = 10
opt_delta = 0.70

hedge = delta_hedge (option_pos, opt_delta)

print(f"\\n=== DELTA HEDGING ===")
print(f"Position: Long {option_pos} call contracts (delta {opt_delta})")
print(f"Option delta exposure: {option_pos * 100 * opt_delta:.0f} shares")
print(f"Hedge: {hedge:.0f} shares (short {abs (hedge):.0f} shares)")
print(f"\\nResult: Delta-neutral position")
print(f"  Stock up $1: Calls gain \${option_pos * 100 * opt_delta:.0f}, stock loses \\$\{abs (hedge):.0f}")
print(f"  Net change: $0 (approximately)")
\`\`\`

---

## Gamma (Γ): Delta Sensitivity

### Definition

**Gamma** measures how much delta changes for a $1 change in the underlying stock price.

\`\`\`
Γ = ∂²C/∂S² = ∂Δ/∂S (second derivative or rate of change of delta)
\`\`\`

**Key points**:
- Gamma is **always positive** for long options (calls and puts)
- Gamma is **highest for ATM options**
- Gamma increases as expiration approaches
- Gamma represents **curvature** (convexity) in option payoff

### Calculation

From Black-Scholes:

\`\`\`
Γ = N'(d1) / (S × σ × √T)

where N'(x) = (1/√(2π)) × e^(-x²/2) (standard normal PDF)
\`\`\`

**Same for calls and puts** (gamma is symmetric).

\`\`\`python
"""
Gamma Calculation and Visualization
"""

def calculate_gamma(S, K, T, r, sigma):
    """
    Calculate option gamma (same for calls and puts)
    
    Returns:
    --------
    float : Gamma
    """
    if T <= 0:
        return 0.0  # No gamma at expiration
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    # Standard normal PDF
    n_prime_d1 = np.exp(-d1**2 / 2) / np.sqrt(2 * np.pi)
    
    gamma = n_prime_d1 / (S * sigma * np.sqrt(T))
    
    return gamma

# Example
S = 150
K = 150
T = 30/365
r = 0.05
sigma = 0.25

gamma = calculate_gamma(S, K, T, r, sigma)

print("=== GAMMA EXAMPLE ===\\n")
print(f"Stock: \${S}, Strike: \\$\{K}, T: {T*365:.0f} days")
print(f"Gamma: {gamma:.6f}")
print(f"\\nInterpretation:")
print(f"  If stock moves $1, delta changes by {gamma:.6f}")
print(f"  If stock moves $10, delta changes by ~{gamma*10:.4f}")

# Visualize gamma
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Gamma vs Stock Price (different expirations)
stock_range = np.linspace(120, 180, 100)
times = [5/365, 30/365, 90/365]

for t in times:
    gammas = [calculate_gamma (s, K, t, r, sigma) for s in stock_range]
    axes[0, 0].plot (stock_range, gammas, label=f'{t*365:.0f} days', linewidth=2)

axes[0, 0].axvline (x=K, color='red', linestyle='--', alpha=0.5, label=f'Strike \${K}')
axes[0, 0].set_xlabel('Stock Price ($)')
axes[0, 0].set_ylabel('Gamma')
axes[0, 0].set_title('Gamma vs Stock Price (Different Expirations)', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid (alpha=0.3)

# 2. Gamma vs Time (ATM option)
time_range = np.linspace(1/365, 90/365, 100)
gammas_time = [calculate_gamma(S, K, t, r, sigma) for t in time_range]

axes[0, 1].plot (time_range * 365, gammas_time, linewidth=2, color='purple')
axes[0, 1].set_xlabel('Days to Expiration')
axes[0, 1].set_ylabel('Gamma')
axes[0, 1].set_title('ATM Gamma vs Time (Gamma Risk Increases Near Expiration)', fontweight='bold')
axes[0, 1].grid (alpha=0.3)

# 3. Delta vs Stock Price (showing curvature = gamma)
stock_range_fine = np.linspace(140, 160, 200)
call_deltas_fine = [calculate_delta (s, K, T, r, sigma, 'call') for s in stock_range_fine]

axes[1, 0].plot (stock_range_fine, call_deltas_fine, linewidth=2)
axes[1, 0].axvline (x=K, color='red', linestyle='--', alpha=0.5, label=f'Strike \${K}')
axes[1, 0].set_xlabel('Stock Price ($)')
axes[1, 0].set_ylabel('Delta')
axes[1, 0].set_title('Delta Curve (Gamma = Slope of This Curve)', fontweight='bold')
axes[1, 0].grid (alpha=0.3)
axes[1, 0].legend()

# Add annotation showing gamma as slope
S_point = 150
delta_at_point = calculate_delta(S_point, K, T, r, sigma, 'call')
gamma_at_point = calculate_gamma(S_point, K, T, r, sigma)

# Draw tangent line
x_tangent = np.array([S_point - 5, S_point + 5])
y_tangent = delta_at_point + gamma_at_point * (x_tangent - S_point)
axes[1, 0].plot (x_tangent, y_tangent, 'r--', alpha=0.7, label=f'Tangent (slope={gamma_at_point:.4f})')
axes[1, 0].legend()

# 4. P&L from gamma
stock_moves = np.linspace(-20, 20, 100)
delta_pnl = call_deltas_fine[100] * stock_moves  # Linear (delta only)
gamma_pnl = call_deltas_fine[100] * stock_moves + 0.5 * gamma * stock_moves**2  # With gamma correction

axes[1, 1].plot (stock_moves, delta_pnl, label='Delta P&L Only (Linear)', linestyle='--', linewidth=2)
axes[1, 1].plot (stock_moves, gamma_pnl, label='Delta + Gamma P&L (Actual)', linewidth=2)
axes[1, 1].axhline (y=0, color='black', linestyle='-', alpha=0.3)
axes[1, 1].axvline (x=0, color='black', linestyle='-', alpha=0.3)
axes[1, 1].fill_between (stock_moves, delta_pnl, gamma_pnl, alpha=0.2, label='Gamma Profit')
axes[1, 1].set_xlabel('Stock Price Move ($)')
axes[1, 1].set_ylabel('Option P&L ($)')
axes[1, 1].set_title('Gamma Creates Convexity (Profit From Large Moves)', fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid (alpha=0.3)

plt.tight_layout()
plt.show()

print("\\n=== KEY GAMMA INSIGHTS ===")
print("1. ATM options have highest gamma (most delta sensitivity)")
print("2. Gamma increases dramatically near expiration (gamma risk)")
print("3. Long gamma = profit from volatility (large moves in either direction)")
print("4. Short gamma = profit from stability (stock stays still)")
print("5. Gamma P&L = 0.5 × Γ × (ΔS)²")
\`\`\`

### Gamma Scalping

**Strategy**: Maintain delta-neutral position, profit from gamma as stock moves.

\`\`\`python
"""
Gamma Scalping Simulation
"""

def simulate_gamma_scalping(S_initial, K, T, r, sigma, stock_path, 
                             rehedge_threshold=0.10):
    """
    Simulate gamma scalping strategy
    
    Returns P&L from gamma scalping
    """
    # Initial setup: Long ATM straddle
    from scipy.stats import norm
    
    def bs_price(S, K, T, r, sigma, opt_type):
        if T <= 0:
            return max(S - K, 0) if opt_type == 'call' else max(K - S, 0)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if opt_type == 'call':
            return S * norm.cdf (d1) - K * np.exp(-r * T) * norm.cdf (d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    # Buy straddle
    initial_call = bs_price(S_initial, K, T, r, sigma, 'call')
    initial_put = bs_price(S_initial, K, T, r, sigma, 'put')
    initial_cost = initial_call + initial_put
    
    print(f"Initial Position: Long Straddle at \\$\{S_initial}")
    print(f"  Call: \\$\{initial_call:.2f}")
    print(f"  Put: \\$\{initial_put:.2f}")
    print(f"  Total Cost: \\$\{initial_cost:.2f}\\n")
    
    # Track hedging
    stock_position = 0  # Start with no stock hedge
    total_hedging_pnl = 0
    rehedges = []
    
    for i, S_current in enumerate (stock_path):
        # Current delta
        T_current = T * (1 - i / len (stock_path))
        if T_current <= 0:
            break
        
        delta_call = calculate_delta(S_current, K, T_current, r, sigma, 'call')
        delta_put = calculate_delta(S_current, K, T_current, r, sigma, 'put')
        option_delta = delta_call + delta_put  # Straddle delta
        
        # Check if rehedge needed
        total_delta = option_delta + stock_position / 100
        
        if abs (total_delta) > rehedge_threshold:
            # Rehedge to delta-neutral
            shares_to_trade = -total_delta * 100
            
            # P&L from closing old stock position
            if stock_position != 0:
                stock_pnl = stock_position * (S_current - rehedges[-1]['price']) if rehedges else 0
                total_hedging_pnl += stock_pnl
            else:
                stock_pnl = 0
            
            # Open new position
            stock_position = shares_to_trade
            
            rehedges.append({
                'day': i,
                'price': S_current,
                'shares': shares_to_trade,
                'delta': total_delta,
                'pnl': stock_pnl
            })
    
    # Close out at end
    final_call = bs_price (stock_path[-1], K, 0.01, r, sigma, 'call')
    final_put = bs_price (stock_path[-1], K, 0.01, r, sigma, 'put')
    final_value = final_call + final_put
    
    option_pnl = final_value - initial_cost
    
    # Close stock position
    if stock_position != 0:
        final_stock_pnl = stock_position * (stock_path[-1] - rehedges[-1]['price'])
        total_hedging_pnl += final_stock_pnl
    
    total_pnl = option_pnl + total_hedging_pnl
    
    print(f"\\n=== GAMMA SCALPING RESULTS ===")
    print(f"Number of rehedges: {len (rehedges)}")
    print(f"\\nP&L Breakdown:")
    print(f"  Option P&L: \\$\{option_pnl:.2f}")
    print(f"  Hedging P&L (gamma scalping): \\$\{total_hedging_pnl:.2f}")
    print(f"  Total P&L: \\$\{total_pnl:.2f}\\n")
    
    return total_pnl, rehedges

# Simulate with volatile price path
np.random.seed(42)
days = 30
S_start = 150
volatility = 0.30  # Annualized
daily_vol = volatility / np.sqrt(252)

# Generate stock path with mean reversion
stock_path = [S_start]
for _ in range (days - 1):
    # Random walk with mean reversion
    mean_reversion = 0.02 * (S_start - stock_path[-1])
    change = mean_reversion + daily_vol * S_start * np.random.randn()
    stock_path.append (stock_path[-1] + change)

stock_path = np.array (stock_path)

# Run gamma scalping
total_pnl, rehedges = simulate_gamma_scalping(
    S_initial=150,
    K=150,
    T=30/365,
    r=0.05,
    sigma=0.25,  # Lower than realized!
    stock_path=stock_path,
    rehedge_threshold=0.10
)

# Visualize
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Stock path with rehedges
axes[0].plot (stock_path, linewidth=2, label='Stock Price')
axes[0].axhline (y=150, color='red', linestyle='--', alpha=0.5, label='Strike')
for rehedge in rehedges:
    axes[0].axvline (x=rehedge['day'], color='gray', linestyle=':', alpha=0.3)
    axes[0].plot (rehedge['day'], rehedge['price'], 'ro', markersize=8)
axes[0].set_xlabel('Day')
axes[0].set_ylabel('Stock Price ($)')
axes[0].set_title('Stock Path with Rehedge Points (Red Dots)', fontweight='bold')
axes[0].legend()
axes[0].grid (alpha=0.3)

# Cumulative hedging P&L
cum_pnl = np.cumsum([r['pnl'] for r in rehedges])
axes[1].plot (range (len (cum_pnl)), cum_pnl, linewidth=2, color='green', marker='o')
axes[1].axhline (y=0, color='black', linestyle='-', alpha=0.3)
axes[1].set_xlabel('Rehedge Number')
axes[1].set_ylabel('Cumulative Hedging P&L ($)')
axes[1].set_title('Gamma Scalping P&L (Buy Low, Sell High Automatically)', fontweight='bold')
axes[1].grid (alpha=0.3)

plt.tight_layout()
plt.show()
\`\`\`

---

## Theta (Θ): Time Decay

### Definition

**Theta** measures how much an option's value decreases as one day passes (with all else equal).

\`\`\`
Θ = ∂C/∂T (partial derivative w.r.t. time, but conventionally negative)
\`\`\`

**Key points**:
- Theta is **negative for long options** (time decay hurts)
- Theta is **positive for short options** (time decay helps)
- Theta accelerates as expiration approaches
- ATM options have highest theta (most time value to lose)

### Calculation

From Black-Scholes (per year, divide by 365 for per-day):

\`\`\`
Θ_call = -S×N'(d1)×σ/(2√T) - r×K×e^(-rT)×N(d2)
Θ_put = -S×N'(d1)×σ/(2√T) + r×K×e^(-rT)×N(-d2)
\`\`\`

\`\`\`python
"""
Theta Calculation and Time Decay Analysis
"""

def calculate_theta(S, K, T, r, sigma, option_type='call'):
    """
    Calculate option theta (per day)
    
    Returns:
    --------
    float : Theta (negative for long options)
    """
    if T <= 0:
        return 0.0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Standard normal PDF
    n_prime_d1 = np.exp(-d1**2 / 2) / np.sqrt(2 * np.pi)
    
    # Common term
    common = -S * n_prime_d1 * sigma / (2 * np.sqrt(T))
    
    if option_type == 'call':
        theta = common - r * K * np.exp(-r * T) * norm.cdf (d2)
    else:  # put
        theta = common + r * K * np.exp(-r * T) * norm.cdf(-d2)
    
    # Convert to per-day
    theta = theta / 365
    
    return theta

# Example
S = 150
K = 150
T = 30/365
r = 0.05
sigma = 0.25

call_theta = calculate_theta(S, K, T, r, sigma, 'call')
put_theta = calculate_theta(S, K, T, r, sigma, 'put')

print("=== THETA EXAMPLES ===\\n")
print(f"ATM Call (30 days to expiration):")
print(f"  Theta: \\$\{call_theta:.4f} per day")
print(f"  Interpretation: Loses \\$\{abs (call_theta):.2f} per day from time decay\\n")

print(f"ATM Put (30 days to expiration):")
print(f"  Theta: \\$\{put_theta:.4f} per day\\n")

# Calculate for different expirations
expirations = [5, 15, 30, 60, 90, 180]
thetas = [calculate_theta(S, K, t/365, r, sigma, 'call') for t in expirations]

print("Theta vs Time to Expiration:")
for t, theta in zip (expirations, thetas):
    print(f"  {t:3d} days: \\$\{theta:+.4f}/day")

print("\\n→ Theta accelerates as expiration approaches!")

# Visualize time decay
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Option value decay over time
days_to_exp = np.linspace(90, 0, 91)
call_values = []
put_values = []

for day in days_to_exp:
    T_curr = day / 365
    if T_curr <= 0:
        T_curr = 0.001
    call_val = black_scholes(S, K, T_curr, r, sigma, 'call')
    put_val = black_scholes(S, K, T_curr, r, sigma, 'put')
    call_values.append (call_val)
    put_values.append (put_val)

axes[0, 0].plot(90 - days_to_exp, call_values, label='ATM Call', linewidth=2)
axes[0, 0].plot(90 - days_to_exp, put_values, label='ATM Put', linewidth=2)
axes[0, 0].set_xlabel('Days Elapsed')
axes[0, 0].set_ylabel('Option Value ($)')
axes[0, 0].set_title('Time Decay: Option Value vs Days Passed', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid (alpha=0.3)
axes[0, 0].invert_xaxis()

# 2. Theta vs Time to Expiration
time_range = np.linspace(1/365, 180/365, 100)
call_thetas = [calculate_theta(S, K, t, r, sigma, 'call') for t in time_range]

axes[0, 1].plot (time_range * 365, np.abs (call_thetas), linewidth=2, color='red')
axes[0, 1].set_xlabel('Days to Expiration')
axes[0, 1].set_ylabel('|Theta| ($/day)')
axes[0, 1].set_title('Theta Acceleration Near Expiration', fontweight='bold')
axes[0, 1].grid (alpha=0.3)
axes[0, 1].axvline (x=30, color='orange', linestyle='--', alpha=0.5, label='30 days (theta accelerates)')
axes[0, 1].legend()

# 3. Theta vs Strike (at 30 days)
strike_range = np.linspace(120, 180, 50)
call_thetas_strike = [calculate_theta(S, k, T, r, sigma, 'call') for k in strike_range]

axes[1, 0].plot (strike_range, np.abs (call_thetas_strike), linewidth=2, color='purple')
axes[1, 0].axvline (x=S, color='red', linestyle='--', alpha=0.5, label=f'ATM (Stock \${S})')
axes[1, 0].set_xlabel('Strike Price ($)')
axes[1, 0].set_ylabel('|Theta| ($/day)')
axes[1, 0].set_title('Theta vs Strike (ATM Has Highest Theta)', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid (alpha=0.3)

# 4. Weekend effect: 3-day decay on Friday
# Simulate option value decay Thurs → Mon
days_labels = ['Thursday', 'Friday', 'Monday']
call_value_thurs = black_scholes(S, K, 30/365, r, sigma, 'call')
call_value_fri = black_scholes(S, K, 29/365, r, sigma, 'call')
call_value_mon = black_scholes(S, K, 26/365, r, sigma, 'call')  # 3 days pass

values = [call_value_thurs, call_value_fri, call_value_mon]
decays = [0, call_value_thurs - call_value_fri, call_value_fri - call_value_mon]

axes[1, 1].bar (days_labels, values, color=['blue', 'blue', 'red'], alpha=0.7)
for i, (label, val, decay) in enumerate (zip (days_labels, values, decays)):
    axes[1, 1].text (i, val + 0.1, f'\${val:.2f}', ha='center', fontweight='bold')
    if decay > 0:
        axes[1, 1].text (i, val - 0.3, f'-\${decay:.2f}', ha='center', color='red', fontweight='bold')

axes[1, 1].set_ylabel('Option Value ($)')
axes[1, 1].set_title('Weekend Effect: 3-Day Decay Friday→Monday', fontweight='bold')
axes[1, 1].grid (axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print("\\n=== KEY THETA INSIGHTS ===")
print("1. ATM options have highest theta (most time value)")
print("2. Theta accelerates in final 30 days (avoid holding long options)")
print("3. Weekend: Lose 3 days of value Friday → Monday")
print("4. Short options benefit from theta (collect time decay)")
print("5. Theta is enemy of option buyers, friend of option sellers")
\`\`\`

---

## Vega (ν): Volatility Sensitivity

### Definition

**Vega** (actually Greek letter nu, ν, but called vega) measures how much an option's price changes for a 1% change in implied volatility.

\`\`\`
ν = ∂C/∂σ
\`\`\`

**Key points**:
- Vega is **positive for long options** (higher IV → higher premium)
- Vega is **same for calls and puts** at same strike
- ATM options have highest vega
- Longer-dated options have higher vega

### Calculation

From Black-Scholes (per 1% change in volatility):

\`\`\`
ν = S × N'(d1) × √T / 100
\`\`\`

\`\`\`python
"""
Vega Calculation and Volatility Trading
"""

def calculate_vega(S, K, T, r, sigma):
    """
    Calculate option vega (per 1% volatility change)
    
    Returns:
    --------
    float : Vega
    """
    if T <= 0:
        return 0.0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    # Standard normal PDF
    n_prime_d1 = np.exp(-d1**2 / 2) / np.sqrt(2 * np.pi)
    
    # Vega (per 1% change)
    vega = S * n_prime_d1 * np.sqrt(T) / 100
    
    return vega

# Example
S = 150
K = 150
T = 30/365
r = 0.05
sigma = 0.25

vega = calculate_vega(S, K, T, r, sigma)

print("=== VEGA EXAMPLES ===\\n")
print(f"ATM Option (30 days):")
print(f"  Vega: \\$\{vega:.4f} per 1% volatility change")
print(f"  Current IV: {sigma*100:.0f}%")
print(f"\\n  If IV → 26% (+1%): Option gains \\$\{vega:.2f}")
print(f"  If IV → 30% (+5%): Option gains \\$\{vega*5:.2f}")
print(f"  If IV → 20% (-5%): Option loses \\$\{vega*5:.2f}\\n")

# IV crush example (earnings)
print("=== IV CRUSH EXAMPLE (EARNINGS) ===")
pre_earnings_iv = 0.50  # 50% IV
post_earnings_iv = 0.30  # 30% IV
iv_drop = (post_earnings_iv - pre_earnings_iv) * 100  # -20%

option_price_before = black_scholes(S, K, T, r, pre_earnings_iv, 'call')
option_price_after = black_scholes(S, K, T, r, post_earnings_iv, 'call')
actual_loss = option_price_after - option_price_before

vega_pre = calculate_vega(S, K, T, r, pre_earnings_iv)
vega_loss_estimate = vega_pre * iv_drop

print(f"\\nBefore earnings:")
print(f"  IV: {pre_earnings_iv*100:.0f}%, Call Price: \\$\{option_price_before:.2f}")
print(f"  Vega: \\$\{vega_pre:.2f}")

print(f"\\nAfter earnings (stock unchanged):")
print(f"  IV: {post_earnings_iv*100:.0f}% (dropped {abs (iv_drop):.0f}%)")
print(f"  Call Price: \\$\{option_price_after:.2f}")
print(f"\\nLoss from IV crush: \\$\{actual_loss:.2f}")
print(f"Vega estimate: \\$\{vega_loss_estimate:.2f} (close!)")

# Visualize vega
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Vega vs Stock Price
stock_range = np.linspace(120, 180, 100)
vegas = [calculate_vega (s, K, T, r, sigma) for s in stock_range]

axes[0, 0].plot (stock_range, vegas, linewidth=2, color='orange')
axes[0, 0].axvline (x=K, color='red', linestyle='--', alpha=0.5, label=f'Strike \${K}')
axes[0, 0].set_xlabel('Stock Price ($)')
axes[0, 0].set_ylabel('Vega ($/1% IV)')
axes[0, 0].set_title('Vega vs Stock Price (Max at ATM)', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid (alpha=0.3)

# 2. Vega vs Time
time_range = np.linspace(5/365, 180/365, 100)
vegas_time = [calculate_vega(S, K, t, r, sigma) for t in time_range]

axes[0, 1].plot (time_range * 365, vegas_time, linewidth=2, color='green')
axes[0, 1].set_xlabel('Days to Expiration')
axes[0, 1].set_ylabel('Vega ($/1% IV)')
axes[0, 1].set_title('Vega vs Time (Longer-dated = Higher Vega)', fontweight='bold')
axes[0, 1].grid (alpha=0.3)

# 3. Option price vs IV
iv_range = np.linspace(0.10, 0.60, 100)
call_prices_vs_iv = [black_scholes(S, K, T, r, iv, 'call') for iv in iv_range]

axes[1, 0].plot (iv_range * 100, call_prices_vs_iv, linewidth=2, color='purple')
axes[1, 0].axvline (x=sigma*100, color='red', linestyle='--', alpha=0.5, label=f'Current IV {sigma*100:.0f}%')
axes[1, 0].set_xlabel('Implied Volatility (%)')
axes[1, 0].set_ylabel('Call Option Price ($)')
axes[1, 0].set_title('Option Price vs IV (Vega = Slope)', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid (alpha=0.3)

# Add tangent line at current IV
current_price = black_scholes(S, K, T, r, sigma, 'call')
vega_current = calculate_vega(S, K, T, r, sigma)
iv_tangent = np.array([sigma - 0.05, sigma + 0.05])
price_tangent = current_price + vega_current * (iv_tangent - sigma) * 100
axes[1, 0].plot (iv_tangent * 100, price_tangent, 'r--', alpha=0.7, linewidth=2, label=f'Tangent (vega=\${vega_current:.2f})')
axes[1, 0].legend()

# 4. VIX and option values correlation
# Simulate: VIX changes and option value changes
np.random.seed(42)
days = 60
vix_path = [25]  # Start at 25
option_values = [black_scholes(S, K, 30/365, r, vix_path[0]/100, 'call')]

for _ in range (days - 1):
    # VIX mean-reverting random walk
    mean_reversion = 0.05 * (25 - vix_path[-1])
    vix_change = mean_reversion + 2 * np.random.randn()
    new_vix = max (vix_path[-1] + vix_change, 10)  # Floor at 10
    vix_path.append (new_vix)
    
    opt_val = black_scholes(S, K, max(30 - len (option_values), 1)/365, r, new_vix/100, 'call')
    option_values.append (opt_val)

ax2 = axes[1, 1].twinx()
axes[1, 1].plot (vix_path, color='red', linewidth=2, label='VIX')
ax2.plot (option_values, color='blue', linewidth=2, label='Option Value')
axes[1, 1].set_xlabel('Day')
axes[1, 1].set_ylabel('VIX', color='red')
ax2.set_ylabel('Option Value ($)', color='blue')
axes[1, 1].set_title('VIX and Option Value (Positive Correlation)', fontweight='bold')
axes[1, 1].tick_params (axis='y', labelcolor='red')
ax2.tick_params (axis='y', labelcolor='blue')
axes[1, 1].grid (alpha=0.3)

# Add correlation
correlation = np.corrcoef (vix_path, option_values)[0, 1]
axes[1, 1].text(0.5, 0.95, f'Correlation: {correlation:.3f}', 
                transform=axes[1, 1].transAxes, ha='center', fontsize=12,
                bbox=dict (boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

print("\\n=== KEY VEGA INSIGHTS ===")
print("1. Long options = long vega (benefit from IV increase)")
print("2. Short options = short vega (benefit from IV decrease)")
print("3. ATM options have highest vega (most sensitive to IV)")
print("4. IV crush after earnings hurts long option holders")
print("5. VIX spike = windfall for option holders, disaster for sellers")
\`\`\`

---

## Rho (ρ): Interest Rate Sensitivity

### Definition

**Rho** measures how much an option's price changes for a 1% change in the risk-free interest rate.

\`\`\`
ρ = ∂C/∂r
\`\`\`

**Key points**:
- Least important Greek in practice (rates change slowly)
- Calls have **positive rho** (benefit from higher rates)
- Puts have **negative rho** (hurt by higher rates)
- More important for long-dated options (LEAPS)

### Calculation

\`\`\`
ρ_call = K × T × e^(-rT) × N(d2) / 100
ρ_put = -K × T × e^(-rT) × N(-d2) / 100
\`\`\`

\`\`\`python
"""
Rho Calculation
"""

def calculate_rho(S, K, T, r, sigma, option_type='call'):
    """
    Calculate option rho (per 1% interest rate change)
    
    Returns:
    --------
    float : Rho
    """
    if T <= 0:
        return 0.0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        rho = K * T * np.exp(-r * T) * norm.cdf (d2) / 100
    else:  # put
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    
    return rho

# Example
S = 150
K = 150
T = 30/365
r = 0.05
sigma = 0.25

call_rho = calculate_rho(S, K, T, r, sigma, 'call')
put_rho = calculate_rho(S, K, T, r, sigma, 'put')

print("=== RHO EXAMPLES ===\\n")
print(f"Short-dated options (30 days):")
print(f"  Call Rho: \\$\{call_rho:.4f} per 1% rate change")
print(f"  Put Rho: \\$\{put_rho:.4f} per 1% rate change")
print(f"  → Minimal impact\\n")

# Long-dated (LEAPS)
T_long = 2  # 2 years
call_rho_long = calculate_rho(S, K, T_long, r, sigma, 'call')
put_rho_long = calculate_rho(S, K, T_long, r, sigma, 'put')

print(f"Long-dated options (2 years):")
print(f"  Call Rho: \\$\{call_rho_long:.4f} per 1% rate change")
print(f"  Put Rho: \\$\{put_rho_long:.4f} per 1% rate change")
print(f"  → More significant\\n")

# Rate change scenario
rate_change = 0.02  # Fed raises 2%
call_impact = call_rho_long * 200  # 2% = 200 basis points
put_impact = put_rho_long * 200

print(f"If Fed raises rates by 2% (200 bps):")
print(f"  2-year Call: {'+' if call_impact > 0 else '} \\$\{call_impact:.2f}")
print(f"  2-year Put: \\$\{put_impact:.2f}")

print("\\n=== WHY RHO MATTERS (SOMETIMES) ===")
print("1. LEAPS (long-dated options): Rho can matter for multi-year positions")
print("2. Rising rate environment: Calls benefit, puts suffer")
print("3. Macro hedging: Factor in Fed policy expectations")
print("4. Usually negligible: For <3 month options, ignore rho")
\`\`\`

---

## Portfolio Greeks Management

\`\`\`python
"""
Managing Portfolio Greeks
"""

class OptionPortfolio:
    """
    Portfolio-level Greeks management
    """
    
    def __init__(self):
        self.positions = []
    
    def add_position (self, name, option_type, strike, expiration, 
                     quantity, S, r, sigma):
        """Add option position"""
        # Calculate Greeks
        greeks = calculate_greeks(S, strike, expiration, r, sigma, option_type)
        
        # Scale by quantity (×100 shares per contract)
        position = {
            'name': name,
            'type': option_type,
            'strike': strike,
            'expiration': expiration,
            'quantity': quantity,
            'greeks': {
                'delta': greeks['delta'] * quantity * 100,
                'gamma': greeks['gamma'] * quantity * 100,
                'theta': greeks['theta'] * quantity * 100,
                'vega': greeks['vega'] * quantity * 100,
                'rho': greeks['rho'] * quantity * 100
            }
        }
        
        self.positions.append (position)
    
    def portfolio_greeks (self):
        """Calculate total portfolio Greeks"""
        total = {
            'delta': 0,
            'gamma': 0,
            'theta': 0,
            'vega': 0,
            'rho': 0
        }
        
        for pos in self.positions:
            for greek in total:
                total[greek] += pos['greeks'][greek]
        
        return total
    
    def display (self):
        """Display portfolio summary"""
        print("=== PORTFOLIO GREEKS ===\\n")
        print(f"{'Position':<30} {'Qty':<8} {'Delta':<10} {'Gamma':<10} {'Theta':<10} {'Vega':<10}")
        print("-" * 78)
        
        for pos in self.positions:
            print(f"{pos['name']:<30} {pos['quantity']:<8} "
                  f"{pos['greeks']['delta']:>9.0f} "
                  f"{pos['greeks']['gamma']:>9.4f} "
                  f"{pos['greeks']['theta']:>9.2f} "
                  f"{pos['greeks']['vega']:>9.2f}")
        
        print("-" * 78)
        
        total = self.portfolio_greeks()
        print(f"{'TOTAL':<30} {':8} "
              f"{total['delta']:>9.0f} "
              f"{total['gamma']:>9.4f} "
              f"{total['theta']:>9.2f} "
              f"{total['vega']:>9.2f}")
        
        print("\\n=== INTERPRETATION ===")
        print(f"Portfolio Delta: {total['delta']:.0f} (behaves like {total['delta']:.0f} shares)")
        print(f"Portfolio Gamma: {total['gamma']:.4f} (delta changes by {total['gamma']:.4f} per $1 move)")
        print(f"Portfolio Theta: \${total['theta']:.2f}/day ({'earning' if total['theta'] > 0 else 'losing'} \\$\{abs (total['theta']):.2f}/day)")
        print(f"Portfolio Vega: \\$\{total['vega']:.2f} per 1% IV ({'long' if total['vega'] > 0 else 'short'} vol)")

# Example portfolio
S = 150
r = 0.05
sigma = 0.25

portfolio = OptionPortfolio()

# Long 10 call contracts
portfolio.add_position(
    "Long 10 Call $155 (30d)",
    'call', 155, 30/365, 10, S, r, sigma
)

# Short 10 call contracts (covered call)
portfolio.add_position(
    "Short 10 Call $165 (30d)",
    'call', 165, 30/365, -10, S, r, sigma
)

# Long 5 put contracts (protection)
portfolio.add_position(
    "Long 5 Put $145 (30d)",
    'put', 145, 30/365, 5, S, r, sigma
)

portfolio.display()

# Hedging recommendation
total_greeks = portfolio.portfolio_greeks()

print("\\n=== HEDGING RECOMMENDATIONS ===")

if abs (total_greeks['delta']) > 100:
    hedge_shares = -total_greeks['delta']
    print(f"1. Delta hedge: {'+' if hedge_shares > 0 else '}{hedge_shares:.0f} shares to reach delta-neutral")

if abs (total_greeks['vega']) > 500:
    if total_greeks['vega'] > 0:
        print(f"2. Vega risk: Long \\$\{total_greeks['vega']:.0f} vega → exposed to IV drop")
        print(f"   Consider: Sell some options or buy short-dated options (lower vega)")
    else:
        print(f"2. Vega risk: Short \\$\{abs (total_greeks['vega']):.0f} vega → exposed to IV spike")
        print(f"   Consider: Buy long-dated options to add positive vega")

if total_greeks['theta'] < -50:
    print(f"3. Theta burn: Losing \\$\{abs (total_greeks['theta']):.2f}/day")
    print(f"   Consider: Close or roll positions if expecting low movement")
elif total_greeks['theta'] > 50:
    print(f"3. Theta collection: Earning \\$\{total_greeks['theta']:.2f}/day")
    print(f"   Good for range-bound markets")

if abs (total_greeks['gamma']) < 0.01:
    print(f"4. Low gamma: Portfolio delta will stay stable")
elif total_greeks['gamma'] > 0.5:
    print(f"5. High gamma: Delta will change significantly with stock moves")
    print(f"   → Rehedge frequently (gamma scalping opportunity)")
else:
    print(f"4. Negative gamma: Delta moves against you")
    print(f"   → Large moves hurt (max loss if stock moves far)")
\`\`\`

---

## Summary

### Key Takeaways

1. **Delta**: Directional exposure (stock-equivalent)
2. **Gamma**: Delta sensitivity (curvature, convexity)
3. **Theta**: Time decay (enemy of long options)
4. **Vega**: Volatility sensitivity (IV risk)
5. **Rho**: Interest rate sensitivity (usually negligible)

### Greek Relationships

- **Delta + Gamma**: Together determine P&L from stock moves
- **Gamma vs Theta**: Trade-off (long gamma → negative theta)
- **Vega vs Time**: Longer-dated = higher vega
- **ATM options**: Highest gamma, theta, and vega

### Trading Strategies by Greeks

- **Long gamma + Short theta**: Straddle/strangle (profit from movement)
- **Short gamma + Long theta**: Iron condor, short straddle (profit from stability)
- **Long vega**: Buy options before earnings/events
- **Short vega**: Sell options after IV spike

### Next Steps

In the next sections, we'll apply these Greeks to:
- Portfolio theory and optimization
- Risk management frameworks
- Building complete trading systems
`,
};
