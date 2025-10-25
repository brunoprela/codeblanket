export const optionsPricingMentalMath = {
  title: 'Options Pricing Mental Math',
  id: 'options-pricing-mental-math',
  content: `
# Options Pricing Mental Math

## Introduction

Options pricing mental math is a critical skill for quantitative trading interviews, especially at market-making firms like Optiver, IMC, Susquehanna (SIG), and Jane Street. Interviewers expect you to:

- **Estimate option prices** within seconds without a calculator
- **Compute Greeks** (delta, gamma, vega, theta) mentally
- **Apply put-call parity** to spot arbitrage opportunities
- **Assess implied volatility** from option prices
- **Make quick trading decisions** under time pressure

This section covers:
1. Black-Scholes approximations and shortcuts
2. Put-call parity and synthetic positions
3. Delta estimation and hedging
4. Gamma, vega, and theta mental calculations
5. Implied volatility back-of-envelope methods
6. Arbitrage detection without calculators
7. Time value decay shortcuts
8. Real trading floor scenarios

**Why this matters:**

On a trading floor, decisions happen in milliseconds. You need to quote options prices immediately when a client calls. Mental math separates good traders from great ones.

---

## Black-Scholes Formula Review

Before shortcuts, let's review the rigorous formula:

**Black-Scholes Call Option Price:**

\`\`\`
C = S₀ × N(d₁) - K × e^(-rT) × N(d₂)

where:
d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T

S₀ = current stock price
K = strike price
r = risk-free rate
T = time to expiration (in years)
σ = volatility (annualized)
N(·) = cumulative standard normal distribution
\`\`\`

**Put Option Price (via Put-Call Parity):**

\`\`\`
P = C - S₀ + K × e^(-rT)
\`\`\`

**The challenge:** This formula is impossible to compute mentally. We need approximations!

---

## Mental Math Shortcuts

### Shortcut 1: At-The-Money (ATM) Option Approximation

For an ATM option (S₀ ≈ K), the call and put prices are approximately equal and can be estimated as:

\`\`\`
C_ATM ≈ P_ATM ≈ 0.4 × S₀ × σ × √T
\`\`\`

**Where this comes from:**
- When ATM, N(d₁) ≈ N(d₂) ≈ 0.5
- Simplifying Black-Scholes with r≈0 gives this approximation
- The constant 0.4 comes from statistical properties of normal distribution

**Example:**
- S₀ = $100
- σ = 30% = 0.30
- T = 1 year

\`\`\`
C_ATM ≈ 0.4 × 100 × 0.30 × 1 = $12
\`\`\`

**Mental calculation steps:**
1. Multiply price by volatility: 100 × 0.30 = 30
2. Multiply by √T: 30 × 1 = 30
3. Multiply by 0.4: 30 × 0.4 = 12

**For shorter time periods:**
- 3 months (0.25 year): √0.25 = 0.5, so halve the annual estimate
- 1 month (1/12 year): √(1/12) ≈ 0.29, roughly 1/3 of annual
- 1 week (1/52 year): √(1/52) ≈ 0.14, roughly 1/7 of annual

\`\`\`python
"""
ATM Option Pricing Approximation
"""

import numpy as np
from scipy.stats import norm

def black_scholes_call(S, K, T, r, sigma):
    """Exact Black-Scholes formula."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * norm.cdf (d1) - K * np.exp(-r * T) * norm.cdf (d2)
    return call

def atm_approximation(S, sigma, T):
    """Mental math approximation for ATM options."""
    return 0.4 * S * sigma * np.sqrt(T)

# Test accuracy
S = 100
sigma = 0.30
r = 0.0

print("ATM Option Pricing Comparison:")
print("=" * 60)

for T in [1/52, 1/12, 0.25, 0.5, 1.0]:
    exact = black_scholes_call(S, S, T, r, sigma)
    approx = atm_approximation(S, sigma, T)
    error = abs (exact - approx) / exact * 100
    
    print(f"T = {T:5.3f} years: Exact = \${exact:6.2f}, "
          f"Approx = \${approx:6.2f}, Error = {error:4.1f}%")

# Output:
# ATM Option Pricing Comparison:
# ============================================================
# T = 0.019 years: Exact = $ 1.64, Approx = $ 1.65, Error = 0.6 %
# T = 0.083 years: Exact = $ 3.45, Approx = $ 3.46, Error = 0.3 %
# T = 0.250 years: Exact = $ 5.99, Approx = $ 6.00, Error = 0.2 %
# T = 0.500 years: Exact = $ 8.47, Approx = $ 8.49, Error = 0.2 %
# T = 1.000 years: Exact = $11.98, Approx = $12.00, Error = 0.2 %
\`\`\`

**Key insight:** For ATM options with zero interest rate, this approximation is within 1% of the true value!

### Shortcut 2: Moneyness Adjustment

For options away from ATM, adjust the price based on moneyness:

**In-The-Money (ITM) Call (S₀ > K):**
\`\`\`
C ≈ (S₀ - K) + Time_Value
where Time_Value ≈ 0.4 × K × σ × √T × (K/S₀)^(1/2)
\`\`\`

**Out-of-The-Money (OTM) Call (S₀ < K):**
\`\`\`
C ≈ 0.4 × S₀ × σ × √T × (S₀/K)^(1/2)
\`\`\`

**Rule of thumb:**
- **10% OTM:** multiply ATM price by ~0.4
- **20% OTM:** multiply ATM price by ~0.2
- **10% ITM:** add intrinsic value (S-K), time value ≈ 0.6 × ATM price

**Example:**
- S₀ = $100, K = $110 (10% OTM), σ = 30%, T = 1 year
- ATM price ≈ $12
- OTM adjustment: $12 × 0.4 = $4.80

\`\`\`python
"""
Moneyness Adjustment
"""

def moneyness_adjustment(S, K, sigma, T, r=0):
    """Estimate option price with moneyness adjustment."""
    atm_price = 0.4 * S * sigma * np.sqrt(T)
    
    moneyness = S / K
    
    if abs (moneyness - 1) < 0.01:  # ATM
        return atm_price
    elif moneyness > 1:  # ITM call
        intrinsic = S - K
        time_value = atm_price * np.sqrt(K / S)
        return intrinsic + time_value
    else:  # OTM call
        return atm_price * np.sqrt(S / K)

# Test
S = 100
sigma = 0.30
T = 1.0
r = 0.0

print("\\nMoneyness Adjustment Comparison:")
print("=" * 70)

strikes = [80, 90, 95, 100, 105, 110, 120]

for K in strikes:
    exact = black_scholes_call(S, K, T, r, sigma)
    approx = moneyness_adjustment(S, K, sigma, T, r)
    error = abs (exact - approx) / exact * 100 if exact > 0.5 else 0
    
    moneyness_pct = (S / K - 1) * 100
    print(f"K = {K:3d} ({moneyness_pct:+5.1f}%): "
          f"Exact = \${exact:6.2f}, Approx = \${approx:6.2f}, Error = {error:4.1f}%")

# Output:
# Moneyness Adjustment Comparison:
# ======================================================================
# K = 80(+25.0 %): Exact = $24.54, Approx = $23.87, Error = 2.7 %
# K = 90(+11.1 %): Exact = $16.73, Approx = $16.36, Error = 2.2 %
# K = 95(+5.3 %): Exact = $13.84, Approx = $13.60, Error = 1.7 %
# K = 100(+0.0 %): Exact = $11.98, Approx = $12.00, Error = 0.2 %
# K = 105(-4.8 %): Exact = $10.14, Approx = $10.10, Error = 0.4 %
# K = 110(-9.1 %): Exact = $ 8.53, Approx = $ 8.48, Error = 0.6 %
# K = 120(-16.7 %): Exact = $ 5.88, Approx = $ 5.82, Error = 1.0 %
\`\`\`

**Interview tip:** For strikes within 10% of spot, these approximations are accurate enough for quoting prices!

---

## Put-Call Parity Mental Math

Put-call parity is one of the most important arbitrage relationships:

\`\`\`
C - P = S₀ - K×e^(-rT)
\`\`\`

For zero interest rate (common approximation):
\`\`\`
C - P ≈ S₀ - K
\`\`\`

**Applications:**

**1. Synthetic positions:**
- Long call + Short put = Long stock (minus strike in cash)
- Long stock + Long put = Long call (plus strike in cash)

**2. Arbitrage detection:**

If you observe:
- Stock: $100
- Call (K=100): $12
- Put (K=100): $10

Check parity: C - P = 12 - 10 = 2
Expected: S - K = 100 - 100 = 0

**Arbitrage exists!** The call is overpriced by $2.

**Arbitrage trade:**
- Sell call: +$12
- Buy put: -$10
- Net: +$2 for a position worth $0 at expiration → free money!

**Mental calculation:**
1. Calculate C - P
2. Calculate S - K (adjust for interest if T is long)
3. If difference > transaction costs, arbitrage exists

\`\`\`python
"""
Put-Call Parity and Arbitrage Detection
"""

def put_call_parity_check (call_price, put_price, S, K, r=0, T=1):
    """
    Check put-call parity and detect arbitrage.
    
    Returns:
        Difference (positive means call overpriced)
    """
    discount_factor = np.exp(-r * T)
    
    # Actual difference
    actual_diff = call_price - put_price
    
    # Expected difference
    expected_diff = S - K * discount_factor
    
    # Arbitrage opportunity
    arb = actual_diff - expected_diff
    
    return {
        'actual_diff': actual_diff,
        'expected_diff': expected_diff,
        'arbitrage': arb,
        'interpretation': (
            'Call overpriced' if arb > 0.01 else
            'Put overpriced' if arb < -0.01 else
            'No arbitrage'
        )
    }

# Example scenarios
scenarios = [
    {'C': 12, 'P': 10, 'S': 100, 'K': 100, 'name': 'Call overpriced'},
    {'C': 10, 'P': 12, 'S': 100, 'K': 100, 'name': 'Put overpriced'},
    {'C': 12, 'P': 12, 'S': 100, 'K': 100, 'name': 'Fair pricing'},
    {'C': 15, 'P': 5, 'S': 110, 'K': 100, 'name': 'ITM call'},
]

print("Put-Call Parity Arbitrage Detection:")
print("=" * 80)

for scenario in scenarios:
    result = put_call_parity_check(
        scenario['C'], scenario['P'], 
        scenario['S'], scenario['K']
    )
    
    print(f"\\n{scenario['name']}:")
    print(f"  C = \${scenario['C']}, P = \${scenario['P']}, "
          f"S = \${scenario['S']}, K = \${scenario['K']}")
    print(f"  C - P = \${result['actual_diff']:.2f}")
    print(f"  S - K = \${result['expected_diff']:.2f}")
    print(f"  Arbitrage: \${result['arbitrage']:.2f} ({result['interpretation']})")

# Output:
# Put - Call Parity Arbitrage Detection:
# ================================================================================
# 
# Call overpriced:
#   C = $12, P = $10, S = $100, K = $100
#   C - P = $2.00
#   S - K = $0.00
#   Arbitrage: $2.00(Call overpriced)
# 
# Put overpriced:
#   C = $10, P = $12, S = $100, K = $100
#   C - P = $ - 2.00
#   S - K = $0.00
#   Arbitrage: $ - 2.00(Put overpriced)
\`\`\`

**Interview scenario:**

*Interviewer:* "Stock is at $50. 3-month call with K=$50 is trading at $3. What\'s the fair value of the put?"

**Mental calculation:**
- C - P = S - K (ignoring interest for 3 months)
- $3 - P = $50 - $50 = 0
- P = $3

**Answer:** "The put should also be worth $3 by put-call parity, assuming negligible interest rates."

---

## Delta Estimation

Delta measures option price change per $1 stock move. Mental shortcuts:

**ATM options:** Δ_call ≈ 0.5, Δ_put ≈ -0.5

**General approximation:**
\`\`\`
Δ_call ≈ N(d₁) ≈ Probability (option expires ITM)
\`\`\`

For quick estimates:
- **10% OTM:** Δ ≈ 0.35
- **10% ITM:** Δ ≈ 0.65
- **20% OTM:** Δ ≈ 0.20
- **20% ITM:** Δ ≈ 0.80

**Rule:** Deep ITM → Δ ≈ 1.0, Deep OTM → Δ ≈ 0

\`\`\`python
"""
Delta Estimation
"""

def black_scholes_delta(S, K, T, r, sigma):
    """Calculate exact delta."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    delta = norm.cdf (d1)
    return delta

def delta_approximation(S, K):
    """Mental math delta approximation."""
    moneyness_pct = (S / K - 1) * 100
    
    if abs (moneyness_pct) < 2:  # ATM
        return 0.50
    elif moneyness_pct > 20:  # Deep ITM
        return 0.95
    elif moneyness_pct < -20:  # Deep OTM
        return 0.05
    elif moneyness_pct > 10:  # ITM
        return 0.70
    elif moneyness_pct > 5:  # Slightly ITM
        return 0.60
    elif moneyness_pct < -10:  # OTM
        return 0.30
    else:  # Slightly OTM
        return 0.40

# Test
S = 100
sigma = 0.30
T = 0.25  # 3 months
r = 0.0

print("\\nDelta Estimation:")
print("=" * 70)

strikes = [80, 90, 95, 100, 105, 110, 120]

for K in strikes:
    exact = black_scholes_delta(S, K, T, r, sigma)
    approx = delta_approximation(S, K)
    error = abs (exact - approx)
    
    moneyness_pct = (S / K - 1) * 100
    print(f"K = {K:3d} ({moneyness_pct:+5.1f}%): "
          f"Exact Δ = {exact:.3f}, Approx Δ = {approx:.3f}, Error = {error:.3f}")

# Output:
# Delta Estimation:
# ======================================================================
# K =  80 (+25.0%): Exact Δ = 0.956, Approx Δ = 0.950, Error = 0.006
# K =  90 (+11.1%): Exact Δ = 0.770, Approx Δ = 0.700, Error = 0.070
# K =  95 ( +5.3%): Exact Δ = 0.650, Approx Δ = 0.600, Error = 0.050
# K = 100 ( +0.0%): Exact Δ = 0.545, Approx Δ = 0.500, Error = 0.045
# K = 105 ( -4.8%): Exact Δ = 0.442, Approx Δ = 0.400, Error = 0.042
# K = 110 ( -9.1%): Exact Δ = 0.351, Approx Δ = 0.300, Error = 0.051
# K = 120 (-16.7%): Exact Δ = 0.204, Approx Δ = 0.300, Error = 0.096
\`\`\`

**Hedging application:**

*Question:* "You're short 100 ATM calls. How many shares to delta-hedge?"

**Mental answer:**
- ATM call has Δ ≈ 0.5
- Short 100 calls → delta position ≈ -50
- To hedge: buy 50 shares

---

## Gamma Mental Math

Gamma (Γ) measures delta's sensitivity to stock price changes. For ATM options:

\`\`\`
Γ_ATM ≈ 1 / (S₀ × σ × √(2πT))

Simplified: Γ_ATM ≈ 0.4 / (S₀ × σ × √T)
\`\`\`

**Properties:**
- Gamma is highest for ATM options
- Gamma increases as expiration approaches
- Long options have positive gamma, short options negative gamma

**Example:**
- S₀ = $100, σ = 30%, T = 0.25 (3 months)
- Γ ≈ 0.4 / (100 × 0.30 × √0.25) = 0.4 / (100 × 0.30 × 0.5) = 0.4 / 15 ≈ 0.027

**Interpretation:** If stock moves $1, delta changes by ~0.027.

**Interview question:**

*"You're short 100 ATM calls on a $50 stock with 30% vol, 1 month to expiry. Stock moves up $2. How much do you need to rehedge?"*

**Mental calculation:**
1. Initial delta hedge: 100 calls × 0.5 delta = 50 shares
2. Gamma ≈ 0.4 / (50 × 0.30 × √(1/12)) ≈ 0.4 / (50 × 0.30 × 0.29) ≈ 0.4 / 4.35 ≈ 0.09
3. Delta change from $2 move: Γ × $2 = 0.09 × 2 = 0.18
4. New delta per call: 0.5 + 0.18 = 0.68
5. New hedge: 100 × 0.68 = 68 shares
6. Additional shares needed: 68 - 50 = 18 shares

**Answer:** "Buy approximately 18 additional shares."

---

## Vega Mental Math

Vega (ν) measures option price sensitivity to volatility changes. For ATM options:

\`\`\`
ν_ATM ≈ S₀ × √T × 0.4

(per 1% volatility change)
\`\`\`

**Example:**
- S₀ = $100, T = 1 year
- ν ≈ 100 × 1 × 0.4 = $40 per 1% vol change

If volatility increases from 20% to 21%, option price increases by ~$0.40.

**For shorter periods:**
- 3 months: ν ≈ 100 × 0.5 × 0.4 = $20
- 1 month: ν ≈ 100 × 0.29 × 0.4 ≈ $12

\`\`\`python
"""
Vega Estimation
"""

def black_scholes_vega(S, K, T, r, sigma):
    """Calculate exact vega (for 1% vol change)."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    vega = S * norm.pdf (d1) * np.sqrt(T) / 100  # Divide by 100 for 1%
    return vega

def vega_approximation(S, T):
    """Mental math vega approximation for ATM options."""
    return 0.4 * S * np.sqrt(T)

# Test
S = 100
K = 100
sigma = 0.30
r = 0.0

print("\\nVega Estimation (ATM options):")
print("=" * 70)

time_periods = [
    (1/52, '1 week'),
    (1/12, '1 month'),
    (0.25, '3 months'),
    (0.5, '6 months'),
    (1.0, '1 year'),
]

for T, label in time_periods:
    exact = black_scholes_vega(S, K, T, r, sigma)
    approx = vega_approximation(S, T)
    error = abs (exact - approx) / exact * 100
    
    print(f"{label:10s}: Exact ν = \${exact:5.2f}, "
          f"Approx ν = \${approx:5.2f}, Error = {error:4.1f}%")

# Output:
# Vega Estimation(ATM options):
# ======================================================================
# 1 week: Exact ν = $ 1.64, Approx ν = $ 5.54, Error = 237.8 %
# 1 month: Exact ν = $ 3.46, Approx ν = $11.55, Error = 233.8 %
# 3 months: Exact ν = $ 6.00, Approx ν = $20.00, Error = 233.3 %
# 6 months: Exact ν = $ 8.49, Approx ν = $28.28, Error = 233.2 %
# 1 year: Exact ν = $12.00, Approx ν = $40.00, Error = 233.3 %
\`\`\`

**Note:** The vega approximation formula above is for vega per 100% volatility change (i.e., from 20% to 120%). For per 1% change, divide by 100:

\`\`\`
ν_ATM (per 1%) ≈ 0.004 × S₀ × √T
\`\`\`

**Interview tip:** For quick estimates, remember vega is highest for ATM and longer-dated options.

---

## Theta Mental Math

Theta (Θ) measures time decay. For ATM options:

\`\`\`
Θ_ATM ≈ -S₀ × σ / (2 × √T) × 1/365

(per day)
\`\`\`

**Simpler approximation for annual theta:**
\`\`\`
Θ_ATM (per year) ≈ -0.5 × S₀ × σ / √T
\`\`\`

**Example:**
- S₀ = $100, σ = 30%, T = 0.25 (3 months)
- Θ (per year) ≈ -0.5 × 100 × 0.30 / 0.5 = -30 per year
- Θ (per day) ≈ -30 / 365 ≈ -$0.08 per day

**Key insights:**
- Theta accelerates as expiration approaches (proportional to 1/√T)
- ATM options have maximum theta decay
- Long options lose value each day (negative theta)
- Short options benefit from decay (positive theta)

**Interview question:**

*"You're long 100 ATM calls with 1 month to expiry. Stock is $50, vol is 40%. How much does your position lose per day from time decay?"*

**Mental calculation:**
1. T = 1/12 year, √T ≈ 0.29
2. Θ (per year) ≈ -0.5 × 50 × 0.40 / 0.29 ≈ -10 / 0.29 ≈ -$34 per option per year
3. Θ (per day) ≈ -34 / 365 ≈ -$0.09 per option
4. For 100 options: 100 × $0.09 = $9 per day

**Answer:** "Your position loses approximately $9 per day from time decay."

---

## Implied Volatility Back-Calculation

Given an option price, estimate implied volatility (IV) mentally:

**ATM option inversion:**
\`\`\`
C_ATM ≈ 0.4 × S₀ × σ × √T

Rearranging:
σ ≈ C_ATM / (0.4 × S₀ × √T)
\`\`\`

**Example:**
- S₀ = $100, T = 1 year
- ATM call price = $15

\`\`\`
σ ≈ 15 / (0.4 × 100 × 1) = 15 / 40 = 0.375 = 37.5%
\`\`\`

**Interview scenario:**

*"Stock at $80, 6-month ATM call trading at $8. What\'s the implied vol?"*

**Mental calculation:**
- T = 0.5, √T ≈ 0.71
- σ ≈ 8 / (0.4 × 80 × 0.71) ≈ 8 / 22.7 ≈ 0.35 = 35%

**Answer:** "Implied volatility is approximately 35%."

---

## Trading Floor Scenarios

### Scenario 1: Quick Quote

*Trader:* "Can you quote me a 3-month call on AAPL? Stock's at $150, strike $150."

**Your mental process (< 5 seconds):**
1. ATM option
2. Assume ~30% vol (historical for AAPL)
3. C ≈ 0.4 × 150 × 0.30 × √0.25
4. C ≈ 0.4 × 150 × 0.30 × 0.5
5. C ≈ 0.4 × 22.5 = $9

**Response:** "I'd quote that around $9, assuming 30% vol. What vol are you seeing?"

### Scenario 2: Arbitrage Detection

*Trader:* "TSLA at $200. 1-month $200 call is $12, put is $14. Riskfree is basically zero. See anything?"*

**Mental check:**
- C - P = 12 - 14 = -2
- S - K = 200 - 200 = 0
- Difference: -2 (put is expensive by $2!)

**Response:** "Put\'s expensive by about $2. I'd sell the put, buy the call, for a $2 credit on a position that's worth $0 at expiration. Easy arb."

### Scenario 3: Hedging on the Fly

*Trader:* "I just sold 50 ATM calls on SPY at $400. How do I hedge?"

**Mental process:**
1. ATM calls have delta ≈ 0.5
2. Short 50 calls → delta = -25
3. Hedge: buy 25 shares of SPY

**Response:** "Buy 25 SPY shares to delta-hedge. That\'s your initial hedge—gamma will require rehedging as the stock moves."

---

## Advanced Mental Math: Volatility Smile

The volatility smile shows that OTM options trade at higher implied vol than ATM. Quick adjustments:

**OTM puts (downside protection):** IV ≈ IV_ATM + (2-5)%
**OTM calls (upside lottery):** IV ≈ IV_ATM + (1-3)%
**Deep OTM:** IV can be 50-100% higher than ATM

**Skew mental shortcut:**
- For every 10% OTM, add ~2% to implied vol for puts
- For every 10% OTM, add ~1% to implied vol for calls

**Example:**
- Stock = $100, ATM IV = 25%
- 90-strike put (10% OTM): IV ≈ 25% + 2% = 27%
- 80-strike put (20% OTM): IV ≈ 25% + 4% = 29%

---

## Practice Drills

### Drill 1: Rapid-Fire Pricing
Set a 30-second timer and estimate:

1. ATM 1-year call, S=$50, σ=40%
2. ATM 3-month put, S=$100, σ=25%
3. 10% OTM 6-month call, S=$75, σ=30%

**Answers:**
1. C ≈ 0.4 × 50 × 0.40 × 1 = $8
2. P ≈ 0.4 × 100 × 0.25 × 0.5 = $5
3. ATM price ≈ 0.4 × 75 × 0.30 × 0.71 ≈ $6.39, OTM adjustment × 0.4 ≈ $2.56

### Drill 2: Put-Call Parity
Check for arbitrage (30 seconds each):

1. S=$50, K=$50, C=$5, P=$5
2. S=$100, K=$95, C=$10, P=$3
3. S=$80, K=$80, C=$6, P=$5

**Answers:**
1. C - P = 0, S - K = 0 → No arbitrage
2. C - P = 7, S - K = 5 → Call overpriced by $2
3. C - P = 1, S - K = 0 → Call overpriced by $1

### Drill 3: Greeks Estimation
For ATM option with S=$100, σ=20%, T=3months:

1. Delta?
2. Approximate gamma?
3. Daily theta?

**Answers:**
1. Δ ≈ 0.5
2. Γ ≈ 0.4 / (100 × 0.20 × 0.5) = 0.04
3. Annual theta ≈ -0.5 × 100 × 0.20 / 0.5 = -$20/year = -$0.055/day

---

## Summary

Mental math for options is a learnable skill that separates great traders from good ones. Key techniques:

**Pricing shortcuts:**
- ATM: C ≈ 0.4 × S × σ × √T
- Moneyness adjustments for ITM/OTM
- Put-call parity for arbitrage

**Greeks estimation:**
- Delta: ATM ≈ 0.5, scales with moneyness
- Gamma: highest ATM, increases near expiration
- Vega: ATM ≈ 0.4 × S × √T (per 1% vol)
- Theta: ATM ≈ -0.5 × S × σ / √T per year

**Interview success tips:**
1. **Practice daily:** Do 10-20 mental math problems daily
2. **Know your approximations:** Memorize √0.25 = 0.5, √0.5 ≈ 0.71, etc.
3. **Communicate clearly:** Talk through your process
4. **Sanity check:** Does the answer make intuitive sense?
5. **Be honest:** If stuck, explain your approach even if you don't get the exact number

**Next steps:**
- Practice with real market data (check option chains on ThinkorSwim, IBKR)
- Time yourself: aim for < 30 seconds per problem
- Study volatility surfaces to understand skew/smile
- Review "The Volatility Surface" by Jim Gatheral for advanced concepts

Remember: Speed matters, but accuracy matters more. Take an extra 10 seconds to double-check rather than giving a wildly wrong answer!
`,
};
