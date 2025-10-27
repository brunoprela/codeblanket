export const impliedVolatility = {
  title: 'Implied Volatility',
  id: 'implied-volatility',
  content: `
# Implied Volatility

## Introduction

**Implied Volatility (IV)** is arguably the most important concept in options trading. While stock prices, strikes, time, and interest rates are observable, volatility must be estimated. Implied volatility is the market's estimate of future volatility "implied" by current option prices.

**Why IV Matters:**
- Options are quoted in IV terms, not dollar prices
- IV represents fear/greed in the market
- The VIX (volatility index) is based on IV
- Volatility smile/skew reveals market expectations
- IV trading is more profitable than direction for many professionals

**Key Concepts:**
- Historical vs implied volatility
- IV calculation using Newton-Raphson
- Volatility smile and skew
- VIX index and volatility products
- IV rank and percentile
- Trading strategies based on IV

---

## Historical vs Implied Volatility

### Historical Volatility (HV)

**Historical volatility** measures past price movements (realized volatility).

\`\`\`python
"""
Historical Volatility Calculator
"""

import numpy as np
import pandas as pd
from typing import Union

def calculate_historical_volatility(prices: Union[np.ndarray, pd.Series],
                                   window: int = 30,
                                   annualize: bool = True) -> float:
    """
    Calculate historical volatility from price series
    
    Args:
        prices: Array or Series of historical prices
        window: Lookback window in days
        annualize: If True, annualize the volatility
        
    Returns:
        Historical volatility (annualized if requested)
    """
    # Calculate log returns
    prices = np.array(prices)
    log_returns = np.log(prices[1:] / prices[:-1])
    
    # Take last 'window' returns
    recent_returns = log_returns[-window:]
    
    # Calculate standard deviation
    volatility = np.std(recent_returns, ddof=1)
    
    # Annualize (sqrt of 252 trading days)
    if annualize:
        volatility = volatility * np.sqrt(252)
    
    return volatility


# Example: Calculate HV for SPY
import yfinance as yf

spy = yf.download('SPY', period='1y', progress=False)
spy_prices = spy['Close'].values

hv_10day = calculate_historical_volatility(spy_prices, window=10)
hv_30day = calculate_historical_volatility(spy_prices, window=30)
hv_90day = calculate_historical_volatility(spy_prices, window=90)

print("SPY Historical Volatility:")
print(f"  10-day: {hv_10day*100:.2f}%")
print(f"  30-day: {hv_30day*100:.2f}%")
print(f"  90-day: {hv_90day*100:.2f}%")
\`\`\`

### Implied Volatility (IV)

**Implied volatility** is forward-looking - the volatility value that makes the Black-Scholes price equal to the market price.

**Key Difference:**
- HV: Backward-looking (what happened)
- IV: Forward-looking (what market expects)

**Relationship:**
- IV > HV: Market expects more volatility than historical
- IV < HV: Market expects less volatility than historical
- IV mean-reverts around HV over long periods

---

## Calculating Implied Volatility

### Newton-Raphson Method

The most common method for calculating IV.

\`\`\`python
"""
Implied Volatility Calculator using Newton-Raphson
"""

from scipy.stats import norm

def black_scholes_price(S: float, K: float, T: float, r: float, 
                       sigma: float, option_type: str) -> float:
    """Calculate Black-Scholes option price"""
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type.lower() == 'call':
        price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    
    return price


def black_scholes_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate vega for Newton-Raphson"""
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    return vega / 100  # Per 1% change


def implied_volatility_newton(market_price: float, S: float, K: float, T: float,
                              r: float, option_type: str,
                              initial_guess: float = 0.25,
                              max_iterations: int = 100,
                              tolerance: float = 1e-6) -> tuple:
    """
    Calculate implied volatility using Newton-Raphson method
    
    Args:
        market_price: Observed option price in market
        S, K, T, r: Black-Scholes parameters
        option_type: 'call' or 'put'
        initial_guess: Starting volatility estimate
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance
        
    Returns:
        (implied_volatility, iterations, success)
    """
    # Validate market price
    intrinsic = max(S - K, 0) if option_type.lower() == 'call' else max(K - S, 0)
    if market_price < intrinsic - 0.01:
        return None, 0, False  # Price below intrinsic
    
    sigma = initial_guess
    
    for i in range(max_iterations):
        # Calculate BS price and vega
        try:
            bs_price = black_scholes_price(S, K, T, r, sigma, option_type)
            vega = black_scholes_vega(S, K, T, r, sigma)
        except:
            return None, i, False
        
        # Check convergence
        diff = bs_price - market_price
        if abs(diff) < tolerance:
            return sigma, i+1, True
        
        # Newton step: sigma_new = sigma_old - f(sigma)/f'(sigma)
        if vega < 1e-10:  # Avoid division by zero
            return None, i, False
        
        sigma = sigma - (diff / vega)
        
        # Keep sigma in reasonable range
        sigma = max(0.01, min(sigma, 5.0))
    
    # Did not converge
    return None, max_iterations, False


# Example usage
print("=" * 70)
print("IMPLIED VOLATILITY CALCULATOR")
print("=" * 70)

# Market data
S = 100  # Stock price
K = 100  # Strike (ATM)
T = 30/365  # 30 days
r = 0.05  # 5% rate
market_price = 3.50  # Observed call price

iv, iterations, success = implied_volatility_newton(market_price, S, K, T, r, 'call')

if success:
    print(f"\\nMarket Price: \${market_price}")
    print(f"Implied Volatility: {iv*100:.2f}%")
    print(f"Converged in {iterations} iterations")
    
    # Verify
    bs_price = black_scholes_price(S, K, T, r, iv, 'call')
    print(f"\\nVerification:")
    print(f"  Market price: \${market_price:.4f}")
    print(f"  BS price at IV: \${bs_price:.4f}")
    print(f"  Difference: \${abs(bs_price - market_price):.6f}")
else:
    print("Failed to calculate IV")
\`\`\`

### Bisection Method (Fallback)

More robust but slower method for edge cases.

\`\`\`python
"""
Bisection Method for IV (robust fallback)
"""

def implied_volatility_bisection(market_price: float, S: float, K: float, T: float,
                                r: float, option_type: str,
                                vol_min: float = 0.01,
                                vol_max: float = 5.0,
                                max_iterations: int = 100,
                                tolerance: float = 1e-6) -> tuple:
    """
    Calculate IV using bisection (slower but more robust)
    """
    # Check bounds
    price_min = black_scholes_price(S, K, T, r, vol_min, option_type)
    price_max = black_scholes_price(S, K, T, r, vol_max, option_type)
    
    if market_price < price_min or market_price > price_max:
        return None, 0, False
    
    for i in range(max_iterations):
        vol_mid = (vol_min + vol_max) / 2
        price_mid = black_scholes_price(S, K, T, r, vol_mid, option_type)
        
        if abs(price_mid - market_price) < tolerance:
            return vol_mid, i+1, True
        
        if price_mid < market_price:
            vol_min = vol_mid
        else:
            vol_max = vol_mid
    
    return (vol_min + vol_max) / 2, max_iterations, False


# Combined robust IV calculator
def calculate_implied_volatility(market_price: float, S: float, K: float, T: float,
                                r: float, option_type: str) -> dict:
    """
    Robust IV calculator: Try Newton first, fallback to bisection
    """
    # Try Newton-Raphson (fast)
    iv, iterations, success = implied_volatility_newton(
        market_price, S, K, T, r, option_type
    )
    
    if success and 0.05 <= iv <= 2.0:
        return {
            'iv': iv,
            'iterations': iterations,
            'method': 'Newton-Raphson',
            'success': True
        }
    
    # Fallback to bisection (robust)
    iv, iterations, success = implied_volatility_bisection(
        market_price, S, K, T, r, option_type
    )
    
    return {
        'iv': iv,
        'iterations': iterations,
        'method': 'Bisection',
        'success': success
    }
\`\`\`

---

## Volatility Smile and Skew

### The Smile Phenomenon

For same expiration, IV varies by strike - creating a "smile" or "skew" shape.

\`\`\`python
"""
Volatility Smile Analysis
"""

def analyze_volatility_smile(option_chain: dict, S: float, T: float, r: float):
    """
    Calculate and visualize volatility smile
    
    Args:
        option_chain: Dict with strikes as keys, market prices as values
        S: Current stock price
        T: Time to expiration
        r: Risk-free rate
    """
    strikes = []
    ivs = []
    moneyness = []
    
    for strike, price in option_chain.items():
        result = calculate_implied_volatility(price, S, strike, T, r, 'call')
        
        if result['success']:
            strikes.append(strike)
            ivs.append(result['iv'] * 100)
            moneyness.append(strike / S)
    
    # Plot smile
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # By strike
    ax1.plot(strikes, ivs, 'b-o', linewidth=2)
    ax1.axvline(S, color='r', linestyle='--', label=f'Stock Price \${S}')
    ax1.set_xlabel('Strike Price')
    ax1.set_ylabel('Implied Volatility (%)')
    ax1.set_title('Volatility Smile by Strike')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # By moneyness
    ax2.plot(moneyness, ivs, 'g-o', linewidth=2)
    ax2.axvline(1.0, color='r', linestyle='--', label='ATM')
    ax2.set_xlabel('Moneyness (Strike / Stock)')
    ax2.set_ylabel('Implied Volatility (%)')
    ax2.set_title('Volatility Smile by Moneyness')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    return strikes, ivs


# Example: SPY volatility smile
spy_option_chain = {
    80: 20.50,   # Deep ITM
    85: 15.30,
    90: 10.50,
    95: 6.80,
    100: 4.20,   # ATM
    105: 2.50,
    110: 1.40,
    115: 0.75,
    120: 0.40    # Deep OTM
}

strikes, ivs = analyze_volatility_smile(spy_option_chain, S=100, T=30/365, r=0.05)

print("\\nVolatility Smile:")
for strike, iv in zip(strikes, ivs):
    print(f"  Strike \${strike}: IV = {iv:.2f}%")
\`\`\`

### Volatility Skew in Equities

**Equity markets** typically show **negative skew** (left skew):
- OTM puts: Higher IV (crash protection)
- ATM options: Medium IV
- OTM calls: Lower IV

**Why?**1. **Leverage effect**: Stock down → leverage up → volatility up
2. **Demand**: Portfolio managers buy OTM puts for insurance
3. **Supply/demand**: More put buyers than sellers

\`\`\`python
"""
Analyzing Equity Skew
"""

def calculate_skew_metric(iv_otm_put: float, iv_atm: float, iv_otm_call: float) -> dict:
    """
    Calculate skew metrics
    
    Returns:
        Dict with skew measures
    """
    # 25-delta risk reversal (common measure)
    risk_reversal = iv_otm_put - iv_otm_call
    
    # Butterfly (convexity of smile)
    butterfly = (iv_otm_put + iv_otm_call) / 2 - iv_atm
    
    # Put skew (relative to ATM)
    put_skew = (iv_otm_put - iv_atm) / iv_atm
    
    return {
        'risk_reversal': risk_reversal,
        'butterfly': butterfly,
        'put_skew': put_skew,
        'interpretation': 'Negative skew (equity-like)' if risk_reversal > 0 else 'Positive skew (unusual)'
    }


# Example
skew = calculate_skew_metric(
    iv_otm_put=22.0,   # 95 strike put
    iv_atm=20.0,        # 100 strike
    iv_otm_call=19.0    # 105 strike call
)

print("\\nSkew Analysis:")
print(f"  Risk Reversal: {skew['risk_reversal']:.2f}%")
print(f"  Butterfly: {skew['butterfly']:.2f}%")
print(f"  Put Skew: {skew['put_skew']*100:.1f}%")
print(f"  {skew['interpretation']}")
\`\`\`

---

## VIX Index (Fear Gauge)

### Understanding VIX

The **VIX** (CBOE Volatility Index) measures 30-day implied volatility of S&P 500 options.

**VIX Interpretation:**
- VIX < 15: Low volatility (complacent market)
- VIX 15-20: Normal volatility
- VIX 20-30: Elevated volatility
- VIX 30-40: High volatility (fear)
- VIX > 40: Extreme fear (crashes, crises)

**VIX Properties:**
- Mean-reverting (tends to return to ~15-20)
- Spikes during market crashes
- Negative correlation with S&P 500
- "VIX up = stocks down" (usually)

\`\`\`python
"""
VIX Analysis and Trading Signals
"""

def analyze_vix_regime(vix_level: float) -> dict:
    """
    Determine volatility regime from VIX level
    """
    if vix_level < 12:
        regime = "Extremely Low"
        signal = "Consider buying volatility (cheap insurance)"
        strategies = ["Long straddles", "Long variance swaps"]
    elif vix_level < 15:
        regime = "Low"
        signal = "Volatility relatively cheap"
        strategies = ["Buy options", "Long gamma positions"]
    elif vix_level < 20:
        regime = "Normal"
        signal = "Fair value"
        strategies = ["Neutral strategies", "Hedged positions"]
    elif vix_level < 30:
        regime = "Elevated"
        signal = "Consider selling volatility premium"
        strategies = ["Short strangles", "Iron condors"]
    elif vix_level < 40:
        regime = "High"
        signal = "High volatility premium, but risky to sell"
        strategies = ["Reduce positions", "Defensive hedges"]
    else:
        regime = "Extreme Fear"
        signal = "Crisis mode - be cautious"
        strategies = ["Cash", "Long puts only", "Wait for calm"]
    
    return {
        'vix': vix_level,
        'regime': regime,
        'signal': signal,
        'strategies': strategies
    }


# Example
vix_analysis = analyze_vix_regime(25.5)

print("\\nVIX Analysis:")
print(f"  VIX Level: {vix_analysis['vix']}")
print(f"  Regime: {vix_analysis['regime']}")
print(f"  Signal: {vix_analysis['signal']}")
print(f"  Strategies:")
for strategy in vix_analysis['strategies']:
    print(f"    - {strategy}")
\`\`\`

---

## IV Rank and Percentile

### IV Rank

**IV Rank** compares current IV to its 52-week range.

\`\`\`
IV Rank = (Current IV - 52-week Low) / (52-week High - 52-week Low) × 100
\`\`\`

- IV Rank 0%: At 52-week low
- IV Rank 50%: Middle of range
- IV Rank 100%: At 52-week high

\`\`\`python
"""
IV Rank and Percentile Calculator
"""

def calculate_iv_rank(current_iv: float, iv_history: np.ndarray) -> float:
    """
    Calculate IV Rank (position in 52-week range)
    """
    iv_min = np.min(iv_history)
    iv_max = np.max(iv_history)
    
    if iv_max == iv_min:
        return 50.0  # No range
    
    iv_rank = (current_iv - iv_min) / (iv_max - iv_min) * 100
    return iv_rank


def calculate_iv_percentile(current_iv: float, iv_history: np.ndarray) -> float:
    """
    Calculate IV Percentile (% of days below current IV)
    """
    below_current = np.sum(iv_history < current_iv)
    percentile = (below_current / len(iv_history)) * 100
    return percentile


# Example
iv_history_52w = np.array([15, 18, 20, 22, 25, 28, 19, 16, 17, 30, 22, 18])  # Simplified
current_iv = 24

iv_rank = calculate_iv_rank(current_iv, iv_history_52w)
iv_percentile = calculate_iv_percentile(current_iv, iv_history_52w)

print(f"\\nCurrent IV: {current_iv}%")
print(f"52-week range: {np.min(iv_history_52w):.1f}% - {np.max(iv_history_52w):.1f}%")
print(f"IV Rank: {iv_rank:.1f}%")
print(f"IV Percentile: {iv_percentile:.1f}%")

if iv_rank > 75:
    print("→ High IV rank, consider selling premium")
elif iv_rank < 25:
    print("→ Low IV rank, consider buying options")
else:
    print("→ Neutral IV rank")
\`\`\`

---

## Trading Strategies Based on IV

### High IV Strategies (Sell Premium)

When IV is elevated (IV Rank > 50%):
- **Sell options** to collect inflated premium
- **Short strangles** (sell OTM put + call)
- **Iron condors** (defined risk)
- **Covered calls** (generate income)

### Low IV Strategies (Buy Options)

When IV is depressed (IV Rank < 25%):
- **Buy options** while cheap
- **Long straddles/strangles** before volatility events
- **Calendar spreads** (buy back month, sell front month)
- **Debit spreads** (defined risk directional plays)

### IV Expansion Plays

Expecting IV to increase (before earnings, events):
- **Long ATM options** (highest vega)
- **Long variance swaps**
- **VIX call options**

---

## Common Pitfalls

### 1. Confusing IV with Stock Direction

**Mistake:** Thinking high IV means stock will move.

**Reality:** High IV means market EXPECTS movement, but direction unknown.

### 2. Ignoring IV Rank

**Mistake:** Selling options because "20% vol seems high."

**Reality:** 20% might be low for this stock. Always check IV Rank.

### 3. Mean Reversion Assumption

**Mistake:** "VIX is at 40, must go down soon!"

**Reality:** VIX can stay elevated (2008: VIX > 40 for months).

---

## Summary

**Key Concepts:**
- **IV = Market's future volatility expectation**
- **HV = Past realized volatility**
- **Volatility Smile:** IV varies by strike (OTM puts highest for equities)
- **VIX:** 30-day S&P 500 IV, mean-reverts around 15-20
- **IV Rank:** Current IV vs 52-week range
- **Trading:** Sell premium when IV high, buy options when IV low

**IV is the "language" of options** - professionals quote and trade in IV terms, not dollar prices.

**Next Steps:**
- Learn specific options strategies
- Understand covered calls and protective puts
- Study spreads and multi-leg strategies
- Build complete trading systems

In the next section, we'll explore **options trading strategies** and how to select the right strategy based on market outlook and IV regime.
`,
};
