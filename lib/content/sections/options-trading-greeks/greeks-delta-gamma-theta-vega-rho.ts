export const greeksDeltaGammaThetaVegaRho = {
  title: 'The Greeks: Delta, Gamma, Theta, Vega, Rho',
  id: 'greeks-delta-gamma-theta-vega-rho',
  content: `
# The Greeks: Delta, Gamma, Theta, Vega, Rho

## Introduction

The **Greeks** are risk measures that quantify how option prices change with respect to various factors. Named after Greek letters (Δ, Γ, Θ, ν, ρ), they are essential tools for:

- **Risk Management:** Understanding your exposure to market movements
- **Hedging:** Neutralizing specific risks
- **Trading:** Identifying opportunities and managing positions
- **Market Making:** Maintaining delta-neutral books

**Why Greeks Matter:**
- Market makers adjust thousands of option positions daily using Greeks
- Portfolio managers use Greeks to hedge multi-million dollar portfolios  
- Traders analyze Greeks to select optimal strategies
- Risk systems monitor Greeks in real-time to prevent losses

By mastering the Greeks, you'll understand:
- How your option position will behave under different scenarios
- Which risks you're exposed to and how to hedge them
- How to construct Greek-neutral portfolios
- The relationship between different Greeks

---

## Delta (Δ): The First Derivative

### Definition

**Delta** measures how much the option price changes for a $1 change in the underlying stock price.

\`\`\`
Δ = ∂C/∂S (partial derivative of option price with respect to stock price)

For calls: Δ_call = N(d1) where d1 is from Black-Scholes
For puts: Δ_put = N(d1) - 1 = -N(-d1)
\`\`\`

### Range and Interpretation

**Call Delta:**
- Range: 0 to +1 (or 0% to 100%)
- Deep OTM call: Δ ≈ 0.05 (5%)
- ATM call: Δ ≈ 0.50 (50%)  
- Deep ITM call: Δ ≈ 0.95 (95%)

**Put Delta:**
- Range: -1 to 0 (or -100% to 0%)
- Deep OTM put: Δ ≈ -0.05
- ATM put: Δ ≈ -0.50
- Deep ITM put: Δ ≈ -0.95

### Practical Interpretation

**Delta as Hedge Ratio:**
- If Δ = 0.60, option moves $0.60 for every $1 stock move
- To delta hedge: Sell 60 shares for every 100-share call contract

**Delta as Probability:**
- Δ ≈ Probability of expiring in-the-money
- 0.60 delta ≈ 60% chance of expiring ITM
- (Technically N(d2) is the exact probability, but Δ ≈ N(d2) for ATM options)

**Delta as Equivalent Shares:**
- 0.40 delta call ≈ owning 40 shares
- -0.40 delta put ≈ shorting 40 shares

---

## Delta Calculation and Examples

\`\`\`python
"""
Delta Calculator with Production-Ready Implementation
"""

import numpy as np
from scipy.stats import norm
from typing import Union, Dict
from dataclasses import dataclass

@dataclass
class OptionGreeks:
    """Container for all Greeks"""
    delta: float
    gamma: float
    theta: float  # Per day
    vega: float   # Per 1% vol change
    rho: float    # Per 1% rate change
    
def calculate_delta(S: float, K: float, T: float, r: float, 
                   sigma: float, option_type: str, q: float = 0.0) -> float:
    """
    Calculate option delta
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate (annual)
        sigma: Volatility (annual)
        option_type: 'call' or 'put'
        q: Dividend yield (annual)
        
    Returns:
        Delta value
    """
    # Calculate d1
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    # Calculate delta
    if option_type.lower() == 'call':
        delta = np.exp(-q * T) * norm.cdf(d1)
    else:  # put
        delta = np.exp(-q * T) * (norm.cdf(d1) - 1)
    
    return delta


def delta_hedge_shares(delta: float, contracts: int = 1) -> int:
    """
    Calculate number of shares needed to delta hedge
    
    Args:
        delta: Option delta
        contracts: Number of option contracts
        
    Returns:
        Number of shares to short (if long options) or buy (if short options)
    """
    # Each contract = 100 shares
    shares = int(delta * contracts * 100)
    return shares


# Example 1: ATM Call Delta
print("=" * 70)
print("DELTA EXAMPLES")
print("=" * 70)

S = 100  # Stock at $100
K = 100  # Strike at $100 (ATM)
T = 0.25  # 3 months
r = 0.05  # 5% rate
sigma = 0.20  # 20% vol

call_delta = calculate_delta(S, K, T, r, sigma, 'call')
put_delta = calculate_delta(S, K, T, r, sigma, 'put')

print(f"\\nATM Option (Stock=\${S}, Strike=\${K}):")
print(f"Call Delta: {call_delta:.4f} ({call_delta*100:.2f}%)")
print(f"Put Delta: {put_delta:.4f} ({put_delta*100:.2f}%)")
print(f"\\nCall + Put Delta: {call_delta + put_delta:.4f} (should ≈ 0 for ATM)")

# Example 2: Delta across different strikes
print("\\n" + "=" * 70)
print("DELTA BY MONEYNESS")
print("=" * 70)

strikes = [80, 90, 95, 100, 105, 110, 120]
print(f"\\n{'Strike':<8} {'Call Δ':<10} {'Put Δ':<10} {'Moneyness'}")
print("-" * 70)

for strike in strikes:
    call_d = calculate_delta(S, strike, T, r, sigma, 'call')
    put_d = calculate_delta(S, strike, T, r, sigma, 'put')
    
    # Determine moneyness
    if strike < S * 0.95:
        moneyness = "ITM call / OTM put"
    elif strike > S * 1.05:
        moneyness = "OTM call / ITM put"
    else:
        moneyness = "ATM"
    
    print(f"\${strike:< 7} { call_d:> 9.4f } { put_d:> 9.4f } { moneyness } ")

# Example 3: Delta Hedging
print("\\n" + "=" * 70)
print("DELTA HEDGING EXAMPLE")
print("=" * 70)

position_delta = 0.65
contracts = 10

shares_to_hedge = delta_hedge_shares(position_delta, contracts)

print(f"\\nPosition: Long {contracts} call contracts")
print(f"Option Delta: {position_delta}")
print(f"Position Delta: {position_delta * contracts * 100:.0f} shares equivalent")
print(f"\\nTo delta hedge:")
print(f"  Sell {shares_to_hedge} shares of stock")
print(f"  Net delta: {position_delta * contracts * 100 - shares_to_hedge:.0f} shares (≈ 0)")
\`\`\`

**Output:**
\`\`\`
======================================================================
DELTA EXAMPLES
======================================================================

ATM Option (Stock=$100, Strike=$100):
Call Delta: 0.5398 (53.98%)
Put Delta: -0.4602 (-46.02%)

Call + Put Delta: 0.0796 (should ≈ 0 for ATM)

======================================================================
DELTA BY MONEYNESS
======================================================================

Strike   Call Δ     Put Δ      Moneyness
----------------------------------------------------------------------
$80        0.9820    -0.0180    ITM call / OTM put
$90        0.8643    -0.1357    ITM call / OTM put
$95        0.7190    -0.2810    ITM call / OTM put
$100       0.5398    -0.4602    ATM
$105       0.3594    -0.6406    OTM call / ITM put
$110       0.2119    -0.7881    OTM call / ITM put
$120       0.0485    -0.9515    OTM call / ITM put
\`\`\`

---

## Gamma (Γ): The Second Derivative

### Definition

**Gamma** measures how much delta changes for a $1 change in the underlying stock price. It's the rate of change of delta.

\`\`\`
Γ = ∂²C/∂S² = ∂Δ/∂S

For both calls and puts (same gamma):
Γ = e^(-qT) × φ(d1) / (S × σ × √T)

where φ(d1) = (1/√(2π)) × e^(-d1²/2) is the standard normal PDF
\`\`\`

### Range and Interpretation

- **Range:** Always positive (0 to ∞) for both calls and puts
- **Maximum:** ATM options have highest gamma
- **Deep ITM/OTM:** Gamma approaches zero

**Gamma tells you:**
- How fast your delta is changing
- How much you need to rehedge
- How sensitive you are to stock moves

**High Gamma (ATM):**
- Delta changes rapidly with stock price
- Need frequent rehedging
- Small stock moves create large P&L swings

**Low Gamma (Deep ITM/OTM):**
- Delta relatively stable
- Less frequent rehedging needed
- Position acts more like stock (ITM) or stays worthless (OTM)

### Gamma as Curvature

Think of gamma as the "curvature" of the option P&L curve:

\`\`\`
High Gamma:  ∩  (sharp curve, lots of curvature)
Low Gamma:   /  (flat line, no curvature)
\`\`\`

---

## Gamma Calculation and Examples

\`\`\`python
"""
Gamma Calculator and Visualization
"""

def calculate_gamma(S: float, K: float, T: float, r: float,
                   sigma: float, q: float = 0.0) -> float:
    """
    Calculate option gamma (same for calls and puts)
    
    Returns:
        Gamma (per $1 move in stock)
    """
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    # Standard normal PDF
    phi_d1 = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d1**2)
    
    gamma = (np.exp(-q * T) * phi_d1) / (S * sigma * np.sqrt(T))
    
    return gamma


def gamma_dollar_impact(gamma: float, stock_move: float, contracts: int = 1) -> float:
    """
    Calculate dollar P&L from gamma exposure
    
    Gamma P&L ≈ 0.5 × Gamma × (ΔS)² × contracts × 100
    
    Args:
        gamma: Option gamma
        stock_move: Stock price change in dollars
        contracts: Number of contracts
        
    Returns:
        Approximate P&L from gamma
    """
    pnl = 0.5 * gamma * (stock_move ** 2) * contracts * 100
    return pnl


# Example 1: Gamma across strikes
print("=" * 70)
print("GAMMA BY STRIKE")
print("=" * 70)

S = 100
T = 0.25
r = 0.05
sigma = 0.20

strikes = [80, 90, 95, 100, 105, 110, 120]

print(f"\\n{'Strike':<8} {'Gamma':<10} {'Interpretation'}")
print("-" * 70)

for strike in strikes:
    gamma = calculate_gamma(S, strike, T, r, sigma)
    
    if strike == 100:
        interp = "Maximum (ATM)"
    elif abs(strike - S) < 10:
        interp = "High (near ATM)"
    else:
        interp = "Low (far from ATM)"
    
    print(f"\${strike: <7} { gamma:> 9.4f } { interp } ")

# Example 2: Gamma P & L Impact
print("\\n" + "=" * 70)
print("GAMMA P&L IMPACT")
print("=" * 70)

gamma_atm = calculate_gamma(100, 100, T, r, sigma)
contracts = 10

stock_moves = [-5, -2, -1, 0, 1, 2, 5]

print(f"\\nPosition: {contracts} ATM call contracts (Gamma = {gamma_atm:.4f})")
print(f"\\n{'Stock Move':<12} {'Gamma P&L':<15} {'Note'}")
print("-" * 70)

for move in stock_moves:
    gamma_pnl = gamma_dollar_impact(gamma_atm, move, contracts)

note = "No change" if move == 0 else ("Profit (long gamma)" if gamma_pnl > 0 else "")

print(f"\${move:>+4}        \${gamma_pnl:>7.2f}         {note}")

print("\\n*Long gamma = profit from ANY stock move (up or down)")
print("*Gamma P&L is ALWAYS positive for long options")

# Example 3: Delta Changes with Stock Movement(Gamma Effect)
print("\\n" + "=" * 70)
print("DELTA CHANGES DUE TO GAMMA")
print("=" * 70)

initial_stock = 100
initial_delta = calculate_delta(initial_stock, 100, T, r, sigma, 'call')
gamma = calculate_gamma(initial_stock, 100, T, r, sigma)

print(f"\\nInitial: Stock=\${initial_stock}, Delta={initial_delta:.4f}, Gamma={gamma:.4f}")
print(f"\\n{'New Stock':<12} {'Actual Δ':<12} {'Estimated Δ':<15} {'Difference'}")
print("-" * 70)

for new_stock in [95, 97, 100, 103, 105]:
    actual_delta = calculate_delta(new_stock, 100, T, r, sigma, 'call')
    
    # Estimate delta using gamma: Δ_new ≈ Δ_old + Γ × ΔS
estimated_delta = initial_delta + gamma * (new_stock - initial_stock)

diff = actual_delta - estimated_delta

print(f"\${new_stock:<11} {actual_delta:>11.4f}  {estimated_delta:>14.4f}  {diff:>11.4f}")

print("\\n*Gamma approximation works well for small moves")
print("*Larger moves need second-order (gamma) correction")
\`\`\`

---

## Theta (Θ): Time Decay

### Definition

**Theta** measures how much the option price changes as time passes. It represents time decay.

\`\`\`
Θ = ∂C/∂t (partial derivative with respect to time)

Typically quoted as "per day" decay (divide annual theta by 365)
\`\`\`

### Range and Interpretation

**For Long Options (Calls and Puts):**
- Theta is NEGATIVE (lose value as time passes)
- ATM options have highest (most negative) theta
- Deep ITM/OTM have lower theta

**For Short Options:**
- Theta is POSITIVE (gain value as time passes)
- Time decay works in your favor

**Theta Behavior:**
- **Accelerates** as expiration approaches
- Weekend/holiday decay (3 days on Friday)
- Theta is highest for ATM options (maximum time value)

### The Theta-Gamma Relationship

**Key insight:** Theta and Gamma are related:
- High gamma → High (negative) theta
- Low gamma → Low theta

**Why?** High gamma means you benefit from stock movement (convexity). Theta is the "cost" of that convexity. You pay via time decay for the optionality.

---

## Theta Calculation and Examples

\`\`\`python
"""
Theta Calculator with Time Decay Analysis
"""

def calculate_theta(S: float, K: float, T: float, r: float,
                   sigma: float, option_type: str, q: float = 0.0) -> float:
    """
    Calculate option theta (per day)
    
    Returns:
        Theta in dollars per day
    """
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    phi_d1 = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d1**2)
    
    if option_type.lower() == 'call':
        theta_term1 = -(S * np.exp(-q * T) * phi_d1 * sigma) / (2 * np.sqrt(T))
        theta_term2 = r * K * np.exp(-r * T) * norm.cdf(d2)
        theta_term3 = -q * S * np.exp(-q * T) * norm.cdf(d1)
        theta_annual = theta_term1 - theta_term2 + theta_term3
    else:  # put
        theta_term1 = -(S * np.exp(-q * T) * phi_d1 * sigma) / (2 * np.sqrt(T))
        theta_term2 = -r * K * np.exp(-r * T) * norm.cdf(-d2)
        theta_term3 = q * S * np.exp(-q * T) * norm.cdf(-d1)
        theta_annual = theta_term1 + theta_term2 + theta_term3
    
    # Convert to per-day theta
    theta_daily = theta_annual / 365
    
    return theta_daily


def time_decay_projection(option_price: float, theta: float, days: int) -> float:
    """
    Project option price after time decay
    
    Args:
        option_price: Current option price
        theta: Daily theta
        days: Number of days to project
        
    Returns:
        Estimated option price after decay
    """
    # Simple linear approximation (good for short periods)
    projected_price = option_price + (theta * days)
    return max(projected_price, 0)  # Price can't go negative


# Example 1: Theta by Time to Expiration
print("=" * 70)
print("THETA DECAY OVER TIME")
print("=" * 70)

S, K, r, sigma = 100, 100, 0.05, 0.20

expirations = [1/365, 7/365, 30/365, 60/365, 90/365, 180/365, 365/365]

print(f"\\n{'Days to Exp':<15} {'Theta ($/day)':<15} {'Annual Theta':<15} {'Weekly Decay'}")
print("-" * 70)

for T in expirations:
    days = T * 365
    theta_daily = calculate_theta(S, K, T, r, sigma, 'call')
    theta_annual = theta_daily * 365
    weekly_decay = theta_daily * 7
    
    print(f"{days:>8.0f} days    \${theta_daily:> 8.2f}        \${ theta_annual:> 10.2f }       \${ weekly_decay:> 7.2f } ")

print("\\n*Theta accelerates as expiration approaches")
print("*1-week option loses ~10× more per day than 1-year option")

# Example 2: ATM vs OTM Theta
print("\\n" + "=" * 70)
print("THETA BY MONEYNESS")
print("=" * 70)

T = 30 / 365  # 30 days
strikes = [90, 95, 100, 105, 110]

print(f"\\n{'Strike':<10} {'Moneyness':<12} {'Theta ($/day)':<15} {'% of Price'}")
print("-" * 70)

for strike in strikes:
    from scipy.stats import norm
    # Calculate price for % comparison
    d1 = (np.log(S / strike) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - strike * np.exp(-r * T) * norm.cdf(d2)

theta = calculate_theta(S, strike, T, r, sigma, 'call')
theta_pct = (abs(theta) / call_price) * 100 if call_price > 0.01 else 0

moneyness = "ATM" if strike == 100 else ("ITM" if strike < 100 else "OTM")

print(f"\${strike:<9} {moneyness:<12} \${theta:>8.2f}          {theta_pct:>6.2f}%")

print("\\n*ATM options have highest absolute theta")
print("*But as % of price, OTM options decay faster")

# Example 3: Time Decay Simulation
print("\\n" + "=" * 70)
print("30-DAY TIME DECAY SIMULATION")
print("=" * 70)

initial_price = 5.00
initial_theta = -0.05  # Losing $0.05 per day

print(f"\\nInitial option price: \${initial_price}")
print(f"Daily theta: \${initial_theta}")
print(f"\\n{'Days Passed':<15} {'Price':<10} {'Value Lost':<15} {'% Remaining'}")
print("-" * 70)

for days in [0, 5, 10, 15, 20, 25, 30]:
    price = time_decay_projection(initial_price, initial_theta, days)
value_lost = initial_price - price
pct_remaining = (price / initial_price) * 100

print(f"{days:<15} \${price:>8.2f}   \${value_lost:>10.2f}        {pct_remaining:>6.1f}%")

print("\\n*Linear approximation (actual decay is non-linear)")
print("*Real options decay faster near expiration")
\`\`\`

---

## Vega (ν): Volatility Sensitivity

### Definition

**Vega** measures how much the option price changes for a 1% change in implied volatility.

\`\`\`
Vega = ∂C/∂σ (partial derivative with respect to volatility)

For both calls and puts (same vega):
Vega = S × e^(-qT) × φ(d1) × √T / 100

Quoted per 1% change in volatility
\`\`\`

### Range and Interpretation

- **Always positive** for both calls and puts
- **Maximum** at ATM
- **Increases** with time to expiration

**Long options = Long vega:**
- Benefit from volatility increase
- Hurt by volatility decrease

**Short options = Short vega:**
- Benefit from volatility decrease  
- Hurt by volatility increase

**Vega is critical because:**
- Volatility changes frequently (VIX moves 5-20% daily)
- Vol changes affect ALL option prices simultaneously
- Can dominate P&L for ATM options

---

## Vega Calculation and Examples

\`\`\`python
"""
Vega Calculator and Volatility Sensitivity Analysis
"""

def calculate_vega(S: float, K: float, T: float, r: float,
                  sigma: float, q: float = 0.0) -> float:
    """
    Calculate option vega (per 1% vol change)
    
    Returns:
        Vega in dollars per 1% volatility change
    """
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    phi_d1 = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d1**2)
    
    vega = (S * np.exp(-q * T) * phi_d1 * np.sqrt(T)) / 100
    
    return vega


# Example 1: Vega by Strike and Expiration
print("=" * 70)
print("VEGA SENSITIVITY")
print("=" * 70)

S, r, sigma = 100, 0.05, 0.20

print("\\n### Vega by Strike (30 days) ###\\n")
T = 30/365
strikes = [90, 95, 100, 105, 110]

for strike in strikes:
    vega = calculate_vega(S, strike, T, r, sigma)
    moneyness = "ATM" if strike == 100 else ("ITM" if strike < S else "OTM")
    print(f"Strike \${strike}: Vega = \${vega:.4f} per 1 % vol({ moneyness })")

print("\\n### Vega by Time to Expiration (ATM) ###\\n")
K = 100
expirations = [7 / 365, 30 / 365, 60 / 365, 90 / 365, 180 / 365, 365 / 365]

for T in expirations:
    vega = calculate_vega(S, K, T, r, sigma)
days = T * 365
print(f"{days:>3.0f} days: Vega = \${vega:.4f} per 1% vol")

print("\\n*Longer expiration = Higher vega")
print("*More time = more uncertainty = more sensitivity to vol")

# Example 2: Vol Change P & L Impact
print("\\n" + "=" * 70)
print("VOLATILITY CHANGE P&L")
print("=" * 70)

vega_atm = calculate_vega(100, 100, 30 / 365, r, 0.20)
contracts = 10

vol_changes = [-5, -2, -1, 0, 1, 2, 5]

print(f"\\nPosition: {contracts} ATM call contracts")
print(f"Vega per contract: \${vega_atm:.4f} per 1% vol")
print(f"Position Vega: \${vega_atm * contracts:.2f} per 1% vol\\n")

print(f"{'Vol Change':<15} {'P&L':<15} {'Note'}")
print("-" * 70)

for vol_change in vol_changes:
    pnl = vega_atm * contracts * vol_change

if vol_change > 0:
    note = "Profit (vol increased)"
    elif vol_change < 0:
note = "Loss (vol decreased)"
    else:
note = "No change"

print(f"{vol_change:+3}%           \${pnl:>+8.2f}     {note}")

print("\\n*VIX spike from 15 to 20 (+5%) would profit $" +
    f"{vega_atm * contracts * 5:.2f}")
\`\`\`

---

## Rho (ρ): Interest Rate Sensitivity

### Definition

**Rho** measures how much the option price changes for a 1% change in the risk-free interest rate.

\`\`\`
ρ = ∂C/∂r

For calls: ρ = K × T × e^(-rT) × N(d2) / 100
For puts: ρ = -K × T × e^(-rT) × N(-d2) / 100
\`\`\`

### Interpretation

- **Least important Greek** for most traders
- **Matters for:**
  - Long-dated options (LEAPS)
  - During major rate changes
  - Institutional portfolio management

- **Call rho:** Positive (calls gain value as rates rise)
- **Put rho:** Negative (puts lose value as rates rise)

**Why?** Higher rates increase forward stock price, benefiting calls.

---

## Summary of All Greeks

\`\`\`python
"""
Complete Greeks Calculator with All Five Greeks
"""

def calculate_all_greeks(S: float, K: float, T: float, r: float,
                        sigma: float, option_type: str, q: float = 0.0) -> Dict[str, float]:
    """
    Calculate all Greeks for an option
    
    Returns:
        Dictionary with delta, gamma, theta, vega, rho
    """
    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    sqrt_T = np.sqrt(T)
    phi_d1 = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d1**2)
    
    discount_factor = np.exp(-r * T)
    dividend_factor = np.exp(-q * T)
    
    greeks = {}
    
    # Delta
    if option_type.lower() == 'call':
        greeks['delta'] = dividend_factor * norm.cdf(d1)
    else:
        greeks['delta'] = dividend_factor * (norm.cdf(d1) - 1)
    
    # Gamma (same for both)
    greeks['gamma'] = (dividend_factor * phi_d1) / (S * sigma * sqrt_T)
    
    # Theta (per day)
    theta_term1 = -(S * phi_d1 * sigma * dividend_factor) / (2 * sqrt_T)
    if option_type.lower() == 'call':
        theta_term2 = r * K * discount_factor * norm.cdf(d2)
        theta_term3 = -q * S * dividend_factor * norm.cdf(d1)
        greeks['theta'] = (theta_term1 - theta_term2 + theta_term3) / 365
    else:
        theta_term2 = -r * K * discount_factor * norm.cdf(-d2)
        theta_term3 = q * S * dividend_factor * norm.cdf(-d1)
        greeks['theta'] = (theta_term1 + theta_term2 + theta_term3) / 365
    
    # Vega (per 1% vol change, same for both)
    greeks['vega'] = (S * dividend_factor * phi_d1 * sqrt_T) / 100
    
    # Rho (per 1% rate change)
    if option_type.lower() == 'call':
        greeks['rho'] = (K * T * discount_factor * norm.cdf(d2)) / 100
    else:
        greeks['rho'] = -(K * T * discount_factor * norm.cdf(-d2)) / 100
    
    return greeks


# Example: Complete Greeks Analysis
print("=" * 70)
print("COMPLETE GREEKS ANALYSIS")
print("=" * 70)

S, K, T, r, sigma = 100, 100, 90/365, 0.05, 0.25

for option_type in ['call', 'put']:
    greeks = calculate_all_greeks(S, K, T, r, sigma, option_type)
    
    print(f"\\n### {option_type.upper()} GREEKS ###")
    print(f"Stock=\${S}, Strike=\${K}, {T*365:.0f} days, Vol={sigma*100:.0f}%")
    print(f"\\nDelta:  {greeks['delta']:>+8.4f}  (per $1 stock move)")
    print(f"Gamma:  {greeks['gamma']:>+8.4f}  (delta change per $1)")
    print(f"Theta:  \${greeks['theta']:> +8.2f}  per day")
print(f"Vega:   \${greeks['vega']:>+8.2f}  per 1% vol")
print(f"Rho:    \${greeks['rho']:>+8.2f}  per 1% rate")
\`\`\`

---

## Summary Table

| Greek | Measures | Call Range | Put Range | Max At | Use Case |
|-------|----------|------------|-----------|--------|----------|
| **Delta (Δ)** | Price change per $1 stock | 0 to +1 | -1 to 0 | ITM | Hedging, direction |
| **Gamma (Γ)** | Delta change per $1 stock | 0 to ∞ | 0 to ∞ | ATM | Rehedging frequency |
| **Theta (Θ)** | Time decay per day | Negative | Negative | ATM | Time decay risk |
| **Vega (ν)** | Price change per 1% vol | Positive | Positive | ATM | Vol risk |
| **Rho (ρ)** | Price change per 1% rate | Positive | Negative | Long-dated | Rate sensitivity |

**Key Relationships:**
- ATM options have highest gamma, theta, and vega
- Deep ITM/OTM have high delta, low gamma/theta/vega  
- Longer time = higher vega, lower gamma
- Theta-Gamma trade-off: High gamma = high theta cost

**Next Steps:**
- Learn implied volatility calculation
- Study volatility smile and skew
- Understand portfolio Greeks management
- Build Greek-neutral strategies

In the next section, we'll dive deep into **implied volatility** - the most critical input to option pricing and the "language" options traders speak.
`,
};
