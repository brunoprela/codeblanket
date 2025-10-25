export const derivativesPricing = {
  id: 'derivatives-pricing',
  title: 'Derivatives Pricing',
  content: `
# Derivatives Pricing

## Introduction

Derivatives are financial contracts whose value derives from an underlying asset (stocks, bonds, commodities, currencies, interest rates). Understanding derivatives pricing is fundamental to quantitative finance—used for hedging risk, speculation, and arbitrage.

**Major derivatives:**
- **Forwards & Futures**: Agreements to buy/sell at future date
- **Swaps**: Exchange cash flows (fixed vs floating interest rates)
- **Options**: Already covered in Options Fundamentals and Black-Scholes
- **Exotic options**: Path-dependent, barrier, Asian, lookback options

**Pricing principles:**
1. **No-arbitrage**: Prevents risk-free profits (law of one price)
2. **Replication**: Derivatives can be replicated with underlying assets
3. **Risk-neutral valuation**: Price as expected payoff under risk-neutral measure

This section covers forward/futures pricing, swap valuation, exotic options, and credit derivatives—essential knowledge for structuring desks, risk management, and quantitative trading.

---

## Forwards and Futures

### Forward Contracts

**Definition**: Agreement to buy/sell asset at specified price (forward price \(F\)) on future date \(T\).

**Key characteristics:**
- **OTC (over-the-counter)**: Customized, bilateral, counterparty risk
- **No upfront payment**: Value = 0 at inception
- **Settlement**: Physical delivery or cash settlement at maturity

**Forward price formula** (no-arbitrage):
\[
F_0 = S_0 e^{(r-q)T}
\]

Where:
- \(S_0\): Spot price
- \(r\): Risk-free rate
- \(q\): Dividend yield (or convenience yield for commodities)
- \(T\): Time to maturity

**Derivation (no-arbitrage argument):**

**Strategy 1**: Buy asset today at \(S_0\), hold until \(T\).
- Cost: \(S_0\) (borrow at rate \(r\))
- Receive dividends: \(S_0 q T\) (continuous div yield)
- Net cost at time \(T\): \(S_0 e^{rT} - S_0 q e^{rT} = S_0 e^{(r-q)T}\)

**Strategy 2**: Enter long forward at forward price \(F_0\).
- Cost at time \(T\): \(F_0\)

**No-arbitrage**: Both strategies deliver same asset at \(T\) → equal cost:
\[
F_0 = S_0 e^{(r-q)T}
\]

**Example:** Stock \(S_0 = \\$100\), \(r = 5\\%\), \(q = 2\\%\), \(T = 1\) year.
\[
F_0 = 100 e^{(0.05-0.02) \\times 1} = 100 e^{0.03} = \\$103.05
\]

If forward trades at \(F = \\$105\) (overpriced):
- **Arbitrage**: Short forward, buy stock, borrow \$100
- At maturity: Deliver stock for \$105, repay loan \$103.05
- **Profit**: \$105 - \$103.05 = \$1.95 risk-free

### Futures Contracts

**Differences from forwards:**
- **Exchange-traded**: Standardized, liquid, marked-to-market daily
- **Margin requirements**: Initial margin + variation margin (daily settlement)
- **No counterparty risk**: Clearinghouse guarantees

**Futures price ≈ Forward price** (under deterministic interest rates).

If rates stochastic and correlated with asset:
\[
F_{futures} \\neq F_{forward}
\]

**Contango vs Backwardation:**

**Contango**: \(F > S\) (futures above spot).
- Normal for non-perishable commodities (storage costs)
- Carry costs: \(F = S e^{(r+c-q)T}\) where \(c\) = storage cost

**Backwardation**: \(F < S\) (futures below spot).
- Occurs when convenience yield high (oil producers need immediate inventory)
- Strong demand for physical commodity

**Example:** Crude oil futures, Dec 2024.
- Spot: \$80/barrel
- 1-month future: \$82 (contango, storage + interest costs)
- During supply shock: 1-month future \$78 (backwardation, convenience yield dominates)

### Currency Forwards

**Interest rate parity**:
\[
F_0 = S_0 \\frac{e^{r_d T}}{e^{r_f T}} = S_0 e^{(r_d - r_f)T}
\]

Where:
- \(r_d\): Domestic interest rate
- \(r_f\): Foreign interest rate

**Example:** USD/EUR spot = 1.10, \(r_{USD} = 5\\%\), \(r_{EUR} = 3\\%\), \(T = 1\) year.
\[
F_0 = 1.10 e^{(0.05-0.03) \\times 1} = 1.10 e^{0.02} = 1.122
\]

**Interpretation**: USD has higher interest rate → USD forwards trade at premium (EUR appreciates in forward market).

---

## Interest Rate Swaps

### Plain Vanilla Swap

**Definition**: Exchange fixed-rate payments for floating-rate payments (typically LIBOR or SOFR).

**Structure:**
- **Fixed leg**: Pay fixed rate \(R\) on notional \(N\)
- **Floating leg**: Receive floating rate \(L_t\) on notional \(N\)
- **Net payment** each period: \(N \\times (L_t - R)\)

**Example:** 5-year swap, \$10M notional, fixed rate 4%, floating SOFR.
- Year 1: SOFR = 3.5% → Receive net \$10M × (3.5% - 4%) = -\$50k (pay \$50k)
- Year 2: SOFR = 4.5% → Receive net \$10M × (4.5% - 4%) = +\$50k (receive \$50k)

**Use cases:**
- **Hedging**: Company with floating-rate debt enters pay-fixed swap → converts to fixed rate
- **Speculation**: Bet on interest rate direction
- **Arbitrage**: Exploit mispricing between fixed and floating markets

### Swap Valuation

**No-arbitrage approach**: Swap = Long floating-rate bond + Short fixed-rate bond.

**At inception** (fair swap rate):
\[
V_{swap} = 0
\]

The fixed rate \(R\) is chosen so present value of fixed leg = present value of floating leg:
\[
R = \\frac{1 - P(T_n)}{\\sum_{i=1}^{n} P(T_i)}
\]

Where:
- \(P(T_i)\): Zero-coupon bond price maturing at \(T_i\)
- Denominator: Annuity factor

**Example:** 3-year swap, annual payments, discount factors: \(P(1) = 0.95\), \(P(2) = 0.90\), \(P(3) = 0.85\).
\[
R = \\frac{1 - 0.85}{0.95 + 0.90 + 0.85} = \\frac{0.15}{2.70} = 5.56\\%
\]

**After inception**: Swap value changes with interest rates.
\[
V_{swap} = V_{float} - V_{fixed}
\]

If rates rise:
- Fixed leg value falls (discounted at higher rates)
- Pay-fixed swap gains value (locked in low rate)

### Cross-Currency Swaps

**Exchange principal and interest in different currencies.**

**Example:** \$10M USD ↔ €9M EUR swap, 5 years.
- At inception: Exchange \$10M for €9M
- Each period: Exchange USD interest for EUR interest
- At maturity: Re-exchange principals

**Use case:** Multinational with USD revenue, EUR debt → swap to match cash flows.

---

## Exotic Options

### Asian Options

**Payoff depends on average price** over option life (not just terminal price).

**Average price call**:
\[
\\text{Payoff} = \\max(A - K, 0)
\]
Where \(A\) = arithmetic or geometric average of underlying prices.

**Advantages:**
- **Cheaper** than standard options (averaging reduces volatility)
- **Harder to manipulate**: Can't manipulate average (vs single price at expiry)

**Use case:** Hedging commodity exposure (oil producer locks in average selling price).

**Valuation:** No closed-form for arithmetic average. Use:
- Monte Carlo simulation
- Geometric average has closed-form (modify Black-Scholes)

### Barrier Options

**Activated or deactivated if underlying hits barrier \(H\).**

**Types:**
- **Knock-in**: Becomes active if \(S\) hits \(H\)
- **Knock-out**: Becomes worthless if \(S\) hits \(H\)
- **Up-and-in**: \(H > S_0\) (barrier above spot)
- **Down-and-out**: \(H < S_0\) (barrier below spot)

**Example:** Down-and-out call, \(S_0 = \\$100\), \(K = \\$100\), \(H = \\$90\).
- If \(S\) never falls to \$90: Regular call payoff \(\\max(S_T - 100, 0)\)
- If \(S\) hits \$90 anytime: Option expires worthless

**Pricing:** Closed-form under Black-Scholes (using reflection principle).
\[
C_{down-and-out} = C_{vanilla} - C_{reflected}
\]

**Cheaper than vanilla** (probability of knockout reduces value).

### Lookback Options

**Payoff depends on maximum or minimum price** during option life.

**Floating strike lookback call**:
\[
\\text{Payoff} = S_T - S_{min}
\]
Where \(S_{min}\) = minimum price during option life.

**Example:** Stock ranges \$90-\$110, ends at \$105.
- Payoff = \$105 - \$90 = \$15 (buy at lowest, sell at terminal)

**Expensive** (captures best possible price).

### Digital (Binary) Options

**Fixed payoff** if condition met.

**Cash-or-nothing call**:
\[
\\text{Payoff} = \\begin{cases} Q & S_T > K \\\\ 0 & S_T \\leq K \\end{cases}
\]

**Example:** \$100 payout if stock > \$100 at expiry.

**Valuation** (risk-neutral):
\[
V = Q e^{-rT} N(d_2)
\]
Where \(d_2\) from Black-Scholes.

---

## Credit Derivatives

### Credit Default Swaps (CDS)

**Insurance against default**: Buyer pays premium, receives payoff if reference entity defaults.

**Structure:**
- **Protection buyer**: Pays \(s\) (CDS spread) per period
- **Protection seller**: Pays \((1 - R) \\times N\) if default occurs

Where:
- \(s\): CDS spread (bps per year)
- \(R\): Recovery rate (typically 40% for corporates)
- \(N\): Notional

**Example:** 5-year CDS on Company X, \$10M notional, spread 200 bps, recovery 40%.
- **Annual premium**: \$10M × 2% = \$200k
- **If default**: Seller pays \$10M × (1 - 0.40) = \$6M

**CDS pricing**: CDS spread compensates for default risk.
\[
s \\approx (1 - R) \\times \\lambda
\]
Where \(\\lambda\) = default intensity (annual probability).

**Example:** Default probability 5%, recovery 40%.
\[
s \\approx (1 - 0.40) \\times 0.05 = 0.03 = 300 \\text{ bps}
\]

**Use cases:**
- **Hedging**: Bondholder buys protection against issuer default
- **Speculation**: Short CDS (sell protection) if bullish on credit
- **Arbitrage**: CDS-bond basis trading

---

## Python Implementation

### Forward Pricing

\`\`\`python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

def forward_price (spot, rate, dividend_yield, time_to_maturity):
    """
    Calculate forward price for asset with dividends.
    
    Parameters:
    - spot: Current spot price
    - rate: Risk-free rate (continuous)
    - dividend_yield: Dividend yield (continuous)
    - time_to_maturity: Time to maturity (years)
    
    Returns:
    - forward_price: Fair forward price
    """
    return spot * np.exp((rate - dividend_yield) * time_to_maturity)

# Example: Stock forward
spot = 100
rate = 0.05
div_yield = 0.02
maturity = 1.0

fwd_price = forward_price (spot, rate, div_yield, maturity)
print(f"Forward Price: \${fwd_price:.2f}")

# Arbitrage detection
market_forward = 105
if market_forward > fwd_price:
    profit = market_forward - fwd_price
    print(f"\\nArbitrage Opportunity: Short forward at \${market_forward:.2f}")
    print(f"Fair value: \${fwd_price:.2f}")
    print(f"Risk-free profit: \${profit:.2f} per share")
\`\`\`

### Swap Valuation

\`\`\`python
def swap_rate (discount_factors):
    """
    Calculate fair swap rate given discount factors.
    
    Parameters:
    - discount_factors: List of zero-coupon bond prices [P(T1), P(T2), ..., P(Tn)]
    
    Returns:
    - swap_rate: Fair fixed rate for swap
    """
    annuity_factor = sum (discount_factors)
    swap_rate = (1 - discount_factors[-1]) / annuity_factor
    return swap_rate

# Example: 3-year swap
discount_factors = [0.95, 0.90, 0.85]
fair_rate = swap_rate (discount_factors)
print(f"\\nFair Swap Rate: {fair_rate*100:.2f}%")

# Swap value after inception
def swap_value (fixed_rate, floating_rates, discount_factors, notional=1):
    """Calculate swap value (receive floating, pay fixed)."""
    # Fixed leg value
    fixed_pv = sum (fixed_rate * df for df in discount_factors)
    
    # Floating leg value (at reset, floating leg worth par)
    floating_pv = 1 - discount_factors[-1]
    
    # Swap value
    value = notional * (floating_pv - fixed_pv)
    return value

notional = 10_000_000
locked_rate = 0.04
floating_rates = [0.03, 0.045, 0.05]

swap_val = swap_value (locked_rate, floating_rates, discount_factors, notional)
print(f"\\nSwap Value: \${swap_val:,.0f}")
\`\`\`

### Asian Option Pricing (Monte Carlo)

\`\`\`python
def asian_option_mc (spot, strike, rate, sigma, maturity, num_paths=10000, num_steps=252):
    """
    Price Asian call option using Monte Carlo simulation.
    
    Parameters:
    - spot: Initial stock price
    - strike: Strike price
    - rate: Risk-free rate
    - sigma: Volatility
    - maturity: Time to maturity (years)
    - num_paths: Number of Monte Carlo paths
    - num_steps: Number of time steps
    
    Returns:
    - option_price: Asian call option price
    """
    dt = maturity / num_steps
    discount_factor = np.exp(-rate * maturity)
    
    payoffs = []
    for _ in range (num_paths):
        # Simulate price path
        prices = [spot]
        for _ in range (num_steps):
            dW = np.random.normal(0, np.sqrt (dt))
            S_new = prices[-1] * np.exp((rate - 0.5*sigma**2)*dt + sigma*dW)
            prices.append(S_new)
        
        # Calculate average price
        avg_price = np.mean (prices)
        
        # Payoff
        payoff = max (avg_price - strike, 0)
        payoffs.append (payoff)
    
    option_price = discount_factor * np.mean (payoffs)
    return option_price

# Example: Asian call
print("\\n" + "="*60)
print("ASIAN OPTION PRICING (Monte Carlo)")
print("="*60)

spot = 100
strike = 100
rate = 0.05
sigma = 0.20
maturity = 1.0

asian_price = asian_option_mc (spot, strike, rate, sigma, maturity)
print(f"Asian Call Price: \${asian_price:.2f}")

# Compare to vanilla European call (Black-Scholes)
def black_scholes_call(S, K, r, sigma, T):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf (d1) - K*np.exp(-r*T)*norm.cdf (d2)

vanilla_price = black_scholes_call (spot, strike, rate, sigma, maturity)
print(f"Vanilla European Call: \${vanilla_price:.2f}")
print(f"\\nAsian option is cheaper: \${vanilla_price - asian_price:.2f} ({(vanilla_price-asian_price)/vanilla_price*100:.1f}% discount)")
print("Reason: Averaging reduces volatility → lower option value")
\`\`\`

---

## Real-World Applications

### 1. **Currency Hedging (Forwards)**

**Problem:** U.S. company expects €10M revenue in 6 months, concerned about EUR/USD decline.

**Solution:** Enter EUR/USD forward (sell EUR, buy USD).
- Current spot: 1.10 USD/EUR
- 6-month forward: 1.12 USD/EUR
- Lock in: €10M × 1.12 = \$11.2M

**Outcome:**
- If EUR falls to 1.05: Saved \$700k (would've received only \$10.5M)
- If EUR rises to 1.15: Opportunity cost \$300k (could've received \$11.5M)

### 2. **Interest Rate Risk Management (Swaps)**

**Problem:** Corporation has \$100M floating-rate debt (SOFR + 1%), expects rates to rise.

**Solution:** Enter pay-fixed receive-floating swap at 4%.
- Effective rate: Pay 4% fixed + 1% spread = 5% all-in
- Hedged against rate increases (floating payments offset by swap)

**Example:** SOFR rises to 6%.
- Debt cost: 6% + 1% = 7%
- Swap receives: 6% floating
- Net cost: 7% - 6% + 4% = 5% (protected!)

### 3. **Commodity Price Insurance (Asian Options)**

**Problem:** Airline needs 1M barrels of oil over next year, wants to cap costs.

**Solution:** Buy 1-year Asian call option on oil, strike \$80/barrel.
- Average oil price during year: \$85/barrel
- Payoff: (\$85 - \$80) × 1M barrels = \$5M
- **Effective purchase price**: \$85 - \$5 + premium paid = capped at \$80 + premium

**Advantage**: Asian option cheaper than standard options, still provides protection.

---

## Key Takeaways

1. **Forward price = Spot × e^{(r-q)T}**—driven by cost-of-carry (interest, dividends, storage)
2. **Contango (F>S) vs Backwardation (F<S)**—reflects storage costs vs convenience yield
3. **Swap rate** equalizes PV of fixed and floating legs—determined by zero-coupon curve
4. **Interest rate swaps** convert floating rate to fixed (or vice versa)—\$500T+ notional outstanding
5. **Asian options** cheaper than vanilla (averaging reduces volatility)—popular for commodity hedging
6. **Barrier options** cheaper than vanilla (knockout probability reduces value)—used for cost reduction
7. **CDS spread** compensates for default risk—spread ≈ (1-R) × default probability
8. **No-arbitrage pricing** foundation—replication and risk-neutral valuation ensure consistent prices

Derivatives pricing combines no-arbitrage arguments, stochastic calculus, and numerical methods (Monte Carlo, finite difference) to value complex instruments and manage financial risk.
`,
};
