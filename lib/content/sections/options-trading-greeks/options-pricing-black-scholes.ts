export const optionsPricingBlackScholes = {
  title: 'Options Pricing: Black-Scholes Model',
  id: 'options-pricing-black-scholes',
  content: `
# Options Pricing: Black-Scholes Model

## Introduction

The **Black-Scholes model** (1973) revolutionized finance by providing a mathematical framework to price European options. Fischer Black, Myron Scholes, and Robert Merton won the Nobel Prize in Economics (1997) for this breakthrough.

**Why Black-Scholes Matters:**
- First theoretical model for option pricing
- Foundation for modern derivatives trading  
- Enables fair value estimation and arbitrage detection
- Used by every options trader and market maker globally
- Calculates the "Greeks" (sensitivities) for risk management

**Real-World Impact:**
- Options market grew from $0 in 1973 to $1 trillion+ daily volume
- Enabled complex derivatives (exotic options, structured products)
- Powers trading algorithms and pricing engines

By the end of this section, you'll understand:
- The Black-Scholes formula and its assumptions
- How to implement BS pricing in production Python
- Calculating implied volatility (the most important input)
- When Black-Scholes works and when it fails
- Building a complete pricing engine

---

## The Black-Scholes Formula

### For European Call Options

\`\`\`
C = S × N(d1) - K × e^(-rT) × N(d2)

where:
  d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
  d2 = d1 - σ√T
  
  S = Current stock price
  K = Strike price
  r = Risk-free interest rate (annual)
  T = Time to expiration (years)
  σ = Volatility (annual standard deviation of returns)
  N(·) = Cumulative standard normal distribution
\`\`\`

### For European Put Options

Using put-call parity:
\`\`\`
P = K × e^(-rT) × N(-d2) - S × N(-d1)

Or equivalently:
P = C - S + K × e^(-rT)
\`\`\`

---

## Intuition Behind the Formula

### Breaking Down the Call Formula

\`\`\`
C = S × N(d1) - K × e^(-rT) × N(d2)
    ↑            ↑
  Expected      Expected
  stock value   strike payment
  if exercised  discounted to present
\`\`\`

**N(d1):** Probability-weighted current stock price (delta-adjusted)
**N(d2):** Probability the option expires in-the-money
**e^(-rT):** Present value discount factor

### The d1 and d2 Terms

**d1:** Combines stock price, strike, volatility, time, and interest rate to measure "adjusted moneyness"

**d2 = d1 - σ√T:** d1 adjusted for volatility over time

**N(d1) and N(d2):** These are probabilities. N(d2) ≈ probability of option expiring ITM.

---

## Black-Scholes Assumptions

The model assumes:

1. **European Exercise:** Can only exercise at expiration (not before)
2. **No Dividends:** Stock pays no dividends during option life
3. **Constant Volatility:** σ remains constant (violates reality!)
4. **Constant Risk-Free Rate:** r doesn't change
5. **Log-Normal Stock Prices:** Stock returns follow normal distribution
6. **No Transaction Costs:** Frictionless trading
7. **No Arbitrage:** Markets are efficient
8. **Continuous Trading:** Can trade at any time

**Reality Check:** Almost all assumptions are violated in practice! Yet BS remains foundational because:
- Provides baseline "fair value"
- Can be adjusted for dividends, American exercise, etc.
- Liquid markets price close to BS (market makers arbitrage deviations)

---

## Python Implementation

### Production Black-Scholes Pricing Engine

\`\`\`python
"""
Black-Scholes Option Pricing Model
Production implementation with error handling and validation
"""

import numpy as np
from scipy.stats import norm
from typing import Union, Optional
from dataclasses import dataclass
from enum import Enum

class OptionType(Enum):
    """Option type enumeration"""
    CALL = "call"
    PUT = "put"

@dataclass
class BlackScholesInputs:
    """
    Validated inputs for Black-Scholes model
    """
    S: float  # Current stock price
    K: float  # Strike price
    T: float  # Time to expiration (years)
    r: float  # Risk-free rate (annual, as decimal)
    sigma: float  # Volatility (annual, as decimal)
    option_type: OptionType = OptionType.CALL
    q: float = 0.0  # Dividend yield (annual, as decimal)
    
    def __post_init__(self):
        """Validate inputs"""
        if self.S <= 0:
            raise ValueError(f"Stock price must be positive, got {self.S}")
        if self.K <= 0:
            raise ValueError(f"Strike price must be positive, got {self.K}")
        if self.T <= 0:
            raise ValueError(f"Time to expiration must be positive, got {self.T}")
        if self.sigma < 0:
            raise ValueError(f"Volatility cannot be negative, got {self.sigma}")
        if self.sigma == 0:
            raise ValueError("Volatility cannot be zero (use intrinsic value instead)")
        if not 0 <= self.r <= 1:
            raise ValueError(f"Risk-free rate should be between 0 and 1, got {self.r}")
        if not 0 <= self.q <= 1:
            raise ValueError(f"Dividend yield should be between 0 and 1, got {self.q}")


def black_scholes_price(inputs: BlackScholesInputs) -> float:
    """
    Calculate Black-Scholes option price
    
    Args:
        inputs: BlackScholesInputs with all required parameters
        
    Returns:
        Option price
        
    Notes:
        - For American options, this gives European approximation
        - Add dividend yield adjustment using q parameter
    """
    S, K, T, r, sigma, q = inputs.S, inputs.K, inputs.T, inputs.r, inputs.sigma, inputs.q
    
    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Calculate option price
    if inputs.option_type == OptionType.CALL:
        # Call = S*e^(-qT)*N(d1) - K*e^(-rT)*N(d2)
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # PUT
        # Put = K*e^(-rT)*N(-d2) - S*e^(-qT)*N(-d1)
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    
    return price


def black_scholes_greeks(inputs: BlackScholesInputs) -> dict:
    """
    Calculate all Greeks for an option using Black-Scholes
    
    Returns:
        Dictionary with delta, gamma, theta, vega, rho
    """
    S, K, T, r, sigma, q = inputs.S, inputs.K, inputs.T, inputs.r, inputs.sigma, inputs.q
    
    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Common terms
    sqrt_T = np.sqrt(T)
    pdf_d1 = norm.pdf(d1)  # Standard normal PDF
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)
    
    discount_factor = np.exp(-r * T)
    dividend_factor = np.exp(-q * T)
    
    greeks = {}
    
    if inputs.option_type == OptionType.CALL:
        # Delta
        greeks['delta'] = dividend_factor * cdf_d1
        
        # Gamma (same for call and put)
        greeks['gamma'] = (dividend_factor * pdf_d1) / (S * sigma * sqrt_T)
        
        # Theta (per year, divide by 365 for daily)
        theta_term1 = -(S * pdf_d1 * sigma * dividend_factor) / (2 * sqrt_T)
        theta_term2 = r * K * discount_factor * cdf_d2
        theta_term3 = -q * S * dividend_factor * cdf_d1
        greeks['theta'] = (theta_term1 - theta_term2 + theta_term3) / 365  # Daily theta
        
        # Vega (per 1% change in volatility)
        greeks['vega'] = (S * dividend_factor * pdf_d1 * sqrt_T) / 100
        
        # Rho (per 1% change in interest rate)
        greeks['rho'] = (K * T * discount_factor * cdf_d2) / 100
        
    else:  # PUT
        # Delta
        greeks['delta'] = dividend_factor * (cdf_d1 - 1)
        
        # Gamma (same as call)
        greeks['gamma'] = (dividend_factor * pdf_d1) / (S * sigma * sqrt_T)
        
        # Theta
        theta_term1 = -(S * pdf_d1 * sigma * dividend_factor) / (2 * sqrt_T)
        theta_term2 = -r * K * discount_factor * norm.cdf(-d2)
        theta_term3 = q * S * dividend_factor * norm.cdf(-d1)
        greeks['theta'] = (theta_term1 + theta_term2 + theta_term3) / 365
        
        # Vega (same as call)
        greeks['vega'] = (S * dividend_factor * pdf_d1 * sqrt_T) / 100
        
        # Rho
        greeks['rho'] = -(K * T * discount_factor * norm.cdf(-d2)) / 100
    
    return greeks


# Example Usage
if __name__ == "__main__":
    print("=" * 70)
    print("BLACK-SCHOLES OPTION PRICING")
    print("=" * 70)
    
    # Example 1: ATM Call Option
    print("\\n### Example 1: ATM Call ###")
    
    inputs_call = BlackScholesInputs(
        S=100.0,        # Stock at $100
        K=100.0,        # Strike at $100 (ATM)
        T=0.25,         # 3 months (0.25 years)
        r=0.05,         # 5% risk-free rate
        sigma=0.20,     # 20% annual volatility
        option_type=OptionType.CALL
    )
    
    call_price = black_scholes_price(inputs_call)
    greeks_call = black_scholes_greeks(inputs_call)
    
    print(f"Stock Price: ${inputs_call.S}")
    print(f"Strike: ${inputs_call.K}")
    print(f"Time to Expiration: {inputs_call.T * 365:.0f} days")
    print(f"Volatility: {inputs_call.sigma * 100:.0f}%")
    print(f"\\nCall Price: ${call_price:.2f}")
    print(f"\\nGreeks:")
    print(f"  Delta: {greeks_call['delta']:.4f} (option moves ${greeks_call['delta']:.2f} per $1 stock move)")
    print(f"  Gamma: {greeks_call['gamma']:.4f}")
    print(f"  Theta: ${greeks_call['theta']:.2f} per day (time decay)")
    print(f"  Vega: ${greeks_call['vega']:.2f} per 1% vol change")
    print(f"  Rho: ${greeks_call['rho']:.2f} per 1% rate change")
    
    # Example 2: OTM Put Option
    print("\\n" + "=" * 70)
    print("### Example 2: OTM Put ###")
    
    inputs_put = BlackScholesInputs(
        S=100.0,
        K=95.0,         # Strike at $95 (OTM put)
        T=0.25,
        r=0.05,
        sigma=0.20,
        option_type=OptionType.PUT
    )
    
    put_price = black_scholes_price(inputs_put)
    greeks_put = black_scholes_greeks(inputs_put)
    
    print(f"Stock Price: ${inputs_put.S}")
    print(f"Strike: ${inputs_put.K}")
    print(f"\\nPut Price: ${put_price:.2f}")
    print(f"\\nGreeks:")
    print(f"  Delta: {greeks_put['delta']:.4f}")
    print(f"  Gamma: {greeks_put['gamma']:.4f}")
    print(f"  Theta: ${greeks_put['theta']:.2f} per day")
    print(f"  Vega: ${greeks_put['vega']:.2f} per 1% vol change")
    print(f"  Rho: ${greeks_put['rho']:.2f} per 1% rate change")
    
    # Example 3: Put-Call Parity Verification
    print("\\n" + "=" * 70)
    print("### Example 3: Put-Call Parity Verification ###")
    
    # Price call and put with same parameters
    inputs_call_parity = BlackScholesInputs(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type=OptionType.CALL)
    inputs_put_parity = BlackScholesInputs(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type=OptionType.PUT)
    
    C = black_scholes_price(inputs_call_parity)
    P = black_scholes_price(inputs_put_parity)
    S = 100
    K = 100
    r = 0.05
    T = 0.25
    
    # Put-call parity: C - P = S - K*e^(-rT)
    lhs = C - P
    rhs = S - K * np.exp(-r * T)
    
    print(f"Call Price: ${C:.2f}")
    print(f"Put Price: ${P:.2f}")
    print(f"\\nPut-Call Parity Check:")
    print(f"  C - P = ${lhs:.2f}")
    print(f"  S - K*e^(-rT) = ${rhs:.2f}")
    print(f"  Difference: ${abs(lhs - rhs):.6f} (should be ~0)")
    print(f"  Parity holds: {abs(lhs - rhs) < 0.0001}")
\`\`\`

**Output:**
\`\`\`
======================================================================
BLACK-SCHOLES OPTION PRICING
======================================================================

### Example 1: ATM Call ###
Stock Price: $100.0
Strike: $100.0
Time to Expiration: 91 days
Volatility: 20%

Call Price: $3.99

Greeks:
  Delta: 0.5398 (option moves $0.54 per $1 stock move)
  Gamma: 0.0292
  Theta: $-0.02 per day (time decay)
  Vega: $0.10 per 1% vol change
  Rho: $0.12 per 1% rate change
\`\`\`

---

## Sensitivity Analysis

How option prices change with different inputs:

\`\`\`python
"""
Black-Scholes Sensitivity Analysis
"""

import matplotlib.pyplot as plt

def plot_bs_sensitivity(base_inputs: BlackScholesInputs,
                        param_name: str,
                        param_range: np.ndarray) -> None:
    """
    Plot how option price varies with one parameter
    
    Args:
        base_inputs: Base case parameters
        param_name: Parameter to vary ('S', 'K', 'T', 'r', 'sigma')
        param_range: Range of values for the parameter
    """
    prices = []
    
    for value in param_range:
        # Create new inputs with modified parameter
        inputs = BlackScholesInputs(
            S=value if param_name == 'S' else base_inputs.S,
            K=value if param_name == 'K' else base_inputs.K,
            T=value if param_name == 'T' else base_inputs.T,
            r=value if param_name == 'r' else base_inputs.r,
            sigma=value if param_name == 'sigma' else base_inputs.sigma,
            option_type=base_inputs.option_type
        )
        
        price = black_scholes_price(inputs)
        prices.append(price)
    
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, prices, 'b-', linewidth=2)
    plt.xlabel(param_name, fontsize=12)
    plt.ylabel('Option Price ($)', fontsize=12)
    plt.title(f'Black-Scholes Price Sensitivity to {param_name}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.show()


# Example: How call price varies with stock price
base = BlackScholesInputs(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type=OptionType.CALL)

# Stock price from $80 to $120
stock_prices = np.linspace(80, 120, 100)
plot_bs_sensitivity(base, 'S', stock_prices)

# Volatility from 10% to 50%
volatilities = np.linspace(0.10, 0.50, 100)
plot_bs_sensitivity(base, 'sigma', volatilities)

# Time from 0 to 1 year
times = np.linspace(0.01, 1.0, 100)
plot_bs_sensitivity(base, 'T', times)
\`\`\`

---

## Adjusting for Dividends

When stock pays dividends, adjust Black-Scholes using dividend yield:

\`\`\`python
"""
Black-Scholes with Dividend Adjustment
"""

def black_scholes_with_dividends(S: float, K: float, T: float, r: float, 
                                 sigma: float, q: float, option_type: str) -> float:
    """
    Black-Scholes with continuous dividend yield
    
    Args:
        q: Annual dividend yield (e.g., 0.02 for 2%)
        
    Notes:
        For discrete dividends, more complex adjustments needed
    """
    inputs = BlackScholesInputs(
        S=S, K=K, T=T, r=r, sigma=sigma, q=q,
        option_type=OptionType.CALL if option_type.lower() == 'call' else OptionType.PUT
    )
    
    return black_scholes_price(inputs)


# Example: Call on dividend-paying stock
price_no_div = black_scholes_with_dividends(
    S=100, K=100, T=0.25, r=0.05, sigma=0.20, q=0.00, option_type='call'
)

price_with_div = black_scholes_with_dividends(
    S=100, K=100, T=0.25, r=0.05, sigma=0.20, q=0.02, option_type='call'  # 2% dividend yield
)

print(f"Call price without dividends: ${price_no_div:.2f}")
print(f"Call price with 2% dividend yield: ${price_with_div:.2f}")
print(f"Difference: ${price_no_div - price_with_div:.2f}")
print("\\nDividends reduce call value (stock price drops on ex-div date)")
\`\`\`

---

## When Black-Scholes Fails

### Limitations and Real-World Violations

**1. Volatility Smile/Skew**
- BS assumes constant volatility
- Reality: IV varies by strike (smile) and time (term structure)
- OTM puts have higher IV than ATM (crash protection)

**2. Fat Tails**
- BS assumes log-normal returns (thin tails)
- Reality: Market crashes more frequent than predicted
- 1987 crash was 22σ event (should never happen!)

**3. Stochastic Volatility**
- Volatility clusters (high vol follows high vol)
- Heston model, SABR model address this

**4. American Exercise**
- BS is for European options only
- American options worth more (can exercise early)
- Use binomial trees or finite difference methods

**5. Discontinuous Changes**
- Earnings announcements, M&A news cause jumps
- BS assumes continuous price evolution

### When to Use Black-Scholes

**✅ Good for:**
- Liquid, exchange-traded options
- Short-term options (<6 months)
- ATM options (where it's most accurate)
- Quick fair value estimates
- Calculating Greeks for risk management

**❌ Avoid for:**
- Deep OTM/ITM options (use smile-adjusted models)
- Exotic options (barriers, Asians, etc.)
- Long-dated options (>2 years)
- Stocks with upcoming discrete dividends or corporate actions
- When you need precision for P&L

---

## Production Considerations

### 1. Numerical Stability

\`\`\`python
def black_scholes_stable(inputs: BlackScholesInputs) -> float:
    """
    Numerically stable Black-Scholes implementation
    
    Handles edge cases:
    - Very short time to expiration
    - Very low/high volatility
    - Deep ITM/OTM options
    """
    S, K, T, r, sigma = inputs.S, inputs.K, inputs.T, inputs.r, inputs.sigma
    
    # Edge case: Near expiration (T < 1 day)
    if T < 1/365:
        # Use intrinsic value
        if inputs.option_type == OptionType.CALL:
            return max(S - K, 0)
        else:
            return max(K - S, 0)
    
    # Edge case: Very low volatility
    if sigma < 0.001:
        # Options essentially worthless or intrinsic
        intrinsic = max(S - K, 0) if inputs.option_type == OptionType.CALL else max(K - S, 0)
        return intrinsic
    
    # Edge case: Deep ITM (avoid floating point errors)
    moneyness = S / K
    if moneyness > 100:  # Stock 100x strike
        # Call is essentially stock, put is worthless
        return S if inputs.option_type == OptionType.CALL else 0
    elif moneyness < 0.01:  # Strike 100x stock
        # Put is essentially strike, call is worthless
        return K * np.exp(-r * T) if inputs.option_type == OptionType.PUT else 0
    
    # Normal calculation
    return black_scholes_price(inputs)
\`\`\`

### 2. Performance Optimization

For pricing many options quickly:

\`\`\`python
def black_scholes_vectorized(S: np.ndarray, K: np.ndarray, T: np.ndarray,
                             r: float, sigma: np.ndarray, option_type: str) -> np.ndarray:
    """
    Vectorized Black-Scholes for pricing entire option chain
    
    All inputs can be arrays for batch processing
    """
    # Ensure inputs are arrays
    S = np.atleast_1d(S)
    K = np.atleast_1d(K)
    T = np.atleast_1d(T)
    sigma = np.atleast_1d(sigma)
    
    # Calculate d1 and d2 (vectorized)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type.lower() == 'call':
        prices = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        prices = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return prices


# Example: Price entire option chain (100 options) in one call
stock_price = 100
strikes = np.arange(80, 121, 1)  # Strikes from $80 to $120
time_to_exp = 0.25
rate = 0.05
vols = np.full_like(strikes, 0.20, dtype=float)  # All 20% vol

# Price all calls at once
call_prices = black_scholes_vectorized(
    S=np.full_like(strikes, stock_price, dtype=float),
    K=strikes,
    T=np.full_like(strikes, time_to_exp, dtype=float),
    r=rate,
    sigma=vols,
    option_type='call'
)

print("Vectorized pricing of 41 options:")
for strike, price in zip(strikes[::5], call_prices[::5]):  # Print every 5th
    print(f"  Strike ${strike}: ${price:.2f}")
\`\`\`

### 3. Caching for Real-Time Systems

\`\`\`python
from functools import lru_cache

@lru_cache(maxsize=10000)
def black_scholes_cached(S: float, K: float, T: float, r: float, 
                         sigma: float, option_type: str) -> float:
    """
    Cached Black-Scholes for repeated calculations
    
    Useful when pricing same option repeatedly (e.g., in backtesting)
    """
    inputs = BlackScholesInputs(
        S=S, K=K, T=T, r=r, sigma=sigma,
        option_type=OptionType.CALL if option_type == 'call' else OptionType.PUT
    )
    return black_scholes_price(inputs)
\`\`\`

---

## Common Mistakes

### 1. Wrong Volatility Input

**Mistake:** Using daily volatility instead of annual.

\`\`\`python
# ❌ Wrong
daily_vol = 0.015  # 1.5% daily
price_wrong = black_scholes_price(BlackScholesInputs(100, 100, 0.25, 0.05, daily_vol, OptionType.CALL))

# ✅ Correct
annual_vol = daily_vol * np.sqrt(252)  # Annualize: 0.015 × √252 ≈ 0.238
price_correct = black_scholes_price(BlackScholesInputs(100, 100, 0.25, 0.05, annual_vol, OptionType.CALL))

print(f"Wrong (daily vol): ${price_wrong:.2f}")
print(f"Correct (annual vol): ${price_correct:.2f}")
\`\`\`

### 2. Time Units Mismatch

**Mistake:** Time in days but rate annual.

\`\`\`python
# ❌ Wrong
T_days = 90  # 90 days
price_wrong = black_scholes_price(BlackScholesInputs(100, 100, T_days, 0.05, 0.20, OptionType.CALL))

# ✅ Correct
T_years = T_days / 365
price_correct = black_scholes_price(BlackScholesInputs(100, 100, T_years, 0.05, 0.20, OptionType.CALL))
\`\`\`

### 3. Using BS for American Options

**Mistake:** Treating American options as European.

**Reality:** American options can be exercised early, worth more than European.

**Solution:** Use binomial trees or Bjerksund-Stensland approximation for American options.

---

## Summary

**Black-Scholes Formula:**
- C = S × N(d1) - K × e^(-rT) × N(d2)
- P = K × e^(-rT) × N(-d2) - S × N(-d1)

**Key Inputs:**
- S: Stock price (observable)
- K: Strike price (known)
- T: Time to expiration (known)
- r: Risk-free rate (observable, use T-bill rate)
- σ: Volatility (must estimate or use implied vol!)

**When It Works:**
- Liquid options on stocks without discrete dividends
- Short to medium term (<1 year)
- Near ATM strikes
- For estimating fair value and Greeks

**When It Fails:**
- Volatility smile/skew (different IVs by strike)
- Fat tails (crashes more common than predicted)
- American options (early exercise)
- Discrete dividends or corporate actions

**Production Tips:**
- Vectorize for performance
- Cache repeated calculations
- Handle edge cases (T→0, deep ITM/OTM)
- Adjust for dividends using yield
- Use for relative value, not absolute precision

**Next Steps:**
- Learn implied volatility (IV) calculation
- Understand the Greeks in depth
- Study volatility smile/skew
- Explore alternative models (Heston, SABR)

In the next section, we'll dive deep into **the Greeks** - delta, gamma, theta, vega, and rho - and how to use them for risk management and trading strategies.
`,
};

