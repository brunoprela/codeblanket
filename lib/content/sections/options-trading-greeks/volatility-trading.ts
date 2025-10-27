export const volatilityTrading = {
  title: 'Volatility Trading',
  id: 'volatility-trading',
  content: `
# Volatility Trading

## Introduction

**Volatility trading** is the practice of trading options based on volatility expectations rather than directional views. Professional traders often focus exclusively on volatility, treating options as "vol instruments" rather than directional bets.

**Core Philosophy:**
- Trade **relative value** in volatility
- **Implied volatility (IV)** vs **realized volatility (RV)**
- **IV is mean-reverting** - exploit deviations
- **Volatility smile/skew arbitrage**

**Why Volatility Trading:**
- Less crowded than directional trading
- More predictable (vol mean-reverts)
- Multiple strategies for any market condition
- Professional-level edge opportunity

---

## Volatility Premium

### The Concept

**Volatility Risk Premium (VRP):** On average, implied volatility > realized volatility.

**Why?**
- **Insurance premium:** Buyers pay for protection
- **Fear > Greed:** Investors overestimate downside risk
- **Supply/demand:** More put buyers than sellers

\`\`\`python
"""
Volatility Premium Analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_volatility_premium(iv_series, rv_series):
    """
    Calculate volatility risk premium
    
    VRP = IV - RV (averaged over time)
    """
    # Calculate premium
    premium = iv_series - rv_series
    
    # Statistics
    avg_premium = premium.mean()
    std_premium = premium.std()
    
    # Positive premium days
    positive_days = (premium > 0).sum() / len(premium) * 100
    
    return {
        'average_premium': avg_premium,
        'std_premium': std_premium,
        'positive_pct': positive_days,
        'premium_series': premium
    }


# Simulate historical data (SPY example)
np.random.seed(42)
dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')

# IV tends to be higher than RV
iv_data = np.random.normal(20, 5, len(dates))  # Average IV ~20%
rv_data = iv_data - np.random.normal(2, 1, len(dates))  # RV typically 2% lower

# Ensure positive values
iv_data = np.maximum(iv_data, 10)
rv_data = np.maximum(rv_data, 8)

# Create DataFrame
df = pd.DataFrame({
    'Date': dates,
    'IV': iv_data,
    'RV': rv_data
})

# Calculate premium
results = calculate_volatility_premium(df['IV'], df['RV'])

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# IV vs RV
ax1.plot(df['Date'], df['IV'], label='Implied Volatility', linewidth=1.5, alpha=0.8)
ax1.plot(df['Date'], df['RV'], label='Realized Volatility', linewidth=1.5, alpha=0.8)
ax1.fill_between(df['Date'], df['IV'], df['RV'], alpha=0.3, color='green', 
                  label='Volatility Premium')
ax1.set_ylabel('Volatility (%)')
ax1.set_title('Implied vs Realized Volatility (SPY Example)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Premium distribution
premium = results['premium_series']
ax2.hist(premium, bins=50, edgecolor='black', alpha=0.7)
ax2.axvline(premium.mean(), color='red', linestyle='--', linewidth=2, 
            label=f'Mean Premium: {premium.mean():.2f}%')
ax2.set_xlabel('Volatility Premium (IV - RV)')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribution of Volatility Risk Premium')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("=" * 70)
print("VOLATILITY RISK PREMIUM ANALYSIS")
print("=" * 70)
print(f"\\nAverage Premium: {results['average_premium']:.2f}%")
print(f"Standard Deviation: {results['std_premium']:.2f}%")
print(f"Positive Premium: {results['positive_pct']:.1f}% of days")
print(f"\\nInterpretation:")
print(f"  - IV is typically {results['average_premium']:.2f}% higher than RV")
print(f"  - Selling volatility (options) has positive expected value")
print(f"  - Premium is positive {results['positive_pct']:.0f}% of the time")
\`\`\`

### Trading the Premium

**Strategies to capture VRP:**1. **Sell options** systematically (collect premium)
2. **Short strangles/iron condors** (high IV)
3. **Ratio spreads** (sell more than buy)
4. **Calendar spreads** (sell near-term vol)

**Risk:** Tail events (crashes) can wipe out months of premium collection.

---

## VIX Futures and Options

### VIX Futures Structure

**VIX futures** trade at premium/discount to VIX spot.

**Term Structure:**
- **Contango (normal):** Front month < back month
- **Backwardation (stress):** Front month > back month

\`\`\`python
"""
VIX Futures Term Structure Analysis
"""

def analyze_vix_term_structure(vix_spot, futures_prices):
    """
    Analyze VIX term structure
    
    Args:
        vix_spot: Current VIX index level
        futures_prices: Dict of {expiration_month: futures_price}
    """
    print("=" * 70)
    print("VIX FUTURES TERM STRUCTURE")
    print("=" * 70)
    print(f"\\nVIX Spot: {vix_spot:.2f}")
    print(f"\\nFutures Prices:")
    
    expirations = sorted(futures_prices.keys())
    prices = [futures_prices[exp] for exp in expirations]
    
    # Calculate roll yield
    roll_yields = []
    
    for i, (exp, price) in enumerate(zip(expirations, prices)):
        premium = price - vix_spot
        pct_premium = (price / vix_spot - 1) * 100
        
        if i > 0:
            prev_price = prices[i-1]
            roll_yield = (prev_price - price) / prev_price * 100
            roll_yields.append(roll_yield)
            print(f"  {exp}: {price:.2f} (+{premium:.2f}, +{pct_premium:.1f}%) "
                  f"[Roll Yield: {roll_yield:.2f}%]")
        else:
            print(f"  {exp}: {price:.2f} (+{premium:.2f}, +{pct_premium:.1f}%)")
    
    # Determine market state
    if all(p > vix_spot for p in prices):
        state = "CONTANGO"
        implication = "Negative carry for long positions"
        strategy = "Consider short VIX ETPs (SVXY) or VIX call spreads"
    elif all(p < vix_spot for p in prices):
        state = "BACKWARDATION"
        implication = "Positive carry for long positions"
        strategy = "Consider long VIX ETPs (VXX) or VIX put spreads"
    else:
        state = "MIXED"
        implication = "Transition period"
        strategy = "Wait for clear structure or use calendar spreads"
    
    avg_roll_yield = np.mean(roll_yields) if roll_yields else 0
    
    print(f"\\n{'─' * 70}")
    print(f"Market State: {state}")
    print(f"Implication: {implication}")
    print(f"Average Roll Yield: {avg_roll_yield:.2f}%/month")
    print(f"\\nRecommended Strategy: {strategy}")
    
    # Visualize
    plt.figure(figsize=(12, 6))
    months = list(range(1, len(expirations) + 1))
    plt.plot(months, prices, 'b-o', linewidth=2, markersize=8, label='Futures Prices')
    plt.axhline(vix_spot, color='r', linestyle='--', linewidth=2, label=f'VIX Spot ({vix_spot:.2f})')
    plt.fill_between(months, prices, vix_spot, alpha=0.3, 
                     color='green' if state == "CONTANGO" else 'red')
    plt.xlabel('Month')
    plt.ylabel('Price')
    plt.title(f'VIX Futures Term Structure: {state}')
    plt.xticks(months, expirations)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# Example 1: Contango (normal market)
print("\\nEXAMPLE 1: CONTANGO (CALM MARKET)")
analyze_vix_term_structure(
    vix_spot=15.0,
    futures_prices={
        'M1': 16.0,
        'M2': 17.5,
        'M3': 18.5,
        'M4': 19.0,
        'M5': 19.5,
        'M6': 20.0
    }
)

# Example 2: Backwardation (crisis)
print("\\n\\nEXAMPLE 2: BACKWARDATION (CRISIS)")
analyze_vix_term_structure(
    vix_spot=45.0,
    futures_prices={
        'M1': 43.0,
        'M2': 38.0,
        'M3': 33.0,
        'M4': 28.0,
        'M5': 25.0,
        'M6': 22.0
    }
)
\`\`\`

### VIX ETPs: VXX, UVXY, SVXY

**Long Volatility ETPs:**
- **VXX:** 1× short-term VIX futures
- **UVXY:**1.5× leveraged short-term VIX futures

**Short Volatility ETPs:**
- **SVXY:** -0.5× inverse VIX futures

\`\`\`python
"""
VXX Decay Analysis (Contango Effect)
"""

def simulate_vxx_decay(initial_price=100, months=12, monthly_roll_cost=0.05):
    """
    Simulate VXX price decay due to contango
    
    In contango, VXX must constantly roll from cheaper front month
    to more expensive back month, creating structural decay.
    """
    prices = [initial_price]
    
    for month in range(1, months + 1):
        # Monthly decay from roll cost
        new_price = prices[-1] * (1 - monthly_roll_cost)
        prices.append(new_price)
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(range(months + 1), prices, 'r-', linewidth=2, marker='o')
    plt.axhline(initial_price, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('Month')
    plt.ylabel('VXX Price')
    plt.title(f'VXX Decay in Contango ({monthly_roll_cost*100:.0f}% monthly roll cost)')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("=" * 70)
    print("VXX CONTANGO DECAY SIMULATION")
    print("=" * 70)
    print(f"\\nStarting Price: \${initial_price:.2f})"
print(f"Monthly Roll Cost: {monthly_roll_cost*100:.0f}%")
print(f"\\nPrice After:")
for i in [1, 3, 6, 12]:
    if i < len(prices):
        pct_change = (prices[i] / initial_price - 1) * 100
print(f"  {i:2d} month{'s' if i > 1 else ' '}: \${prices[i]:6.2f} ({pct_change:+.1f}%)")

annual_decay = (prices[-1] / initial_price - 1) * 100
print(f"\\n{'─' * 70}")
print(f"Annual Decay: {annual_decay:.1f}%")
print(f"\\nWARNING: VXX is NOT a buy-and-hold investment!")
print(f"         Use only for short-term tactical trades")

return prices

# Simulate
prices = simulate_vxx_decay(initial_price = 100, months = 24, monthly_roll_cost = 0.05)

print(f"\\nREAL-WORLD EXAMPLE:")
print(f"  VXX launched in 2009 at ~$400,000 (split-adjusted)")
print(f"  Current price (2024): ~$5")
print(f"  Total decay: 99.99%")
print(f"  This is due to persistent contango over 15 years")
\`\`\`

---

## Variance Swaps

### Concept

**Variance swap:** Pure volatility instrument, no delta exposure.

**Payoff:**
\`\`\`
Payoff = Notional × (Realized Variance - Strike Variance)
\`\`\`

**Example:**
- Strike: 20% variance (400 variance points)
- Notional: $1,000 per variance point
- Realized vol: 25% (625 variance points)
- Payoff: $1,000 × (625 - 400) = $225,000

\`\`\`python
"""
Variance Swap Replication
"""

def replicate_variance_swap_with_options(spot, strikes, option_prices, strike_variance):
    """
    Replicate variance swap using portfolio of options
    
    Theory: Variance swap = weighted portfolio of OTM options
    Weight = 1 / K²
    """
    print("=" * 70)
    print("VARIANCE SWAP REPLICATION")
    print("=" * 70)
    
    total_weight = 0
    portfolio_cost = 0
    
    print(f"\\nSpot: \${spot:.2f}")
print(f"Strike Variance: {strike_variance:.0f} (vol = {np.sqrt(strike_variance):.1f}%)")
print(f"\\nOption Portfolio:")
print(f"  {'Strike':>8} {'Price':>8} {'Weight':>10} {'Position':>10}")
print("  " + "─" * 45)

for strike, price in zip(strikes, option_prices):
        # Weight inversely proportional to strike squared
weight = 1 / (strike ** 2)
position_value = weight * price * 100

total_weight += weight
portfolio_cost += position_value

print(f"  \${strike:7.2f} \${price:7.2f} {weight:10.6f} \${position_value:9.2f}")

print("  " + "─" * 45)
print(f"  {'Total':>8} \${portfolio_cost:7.2f}")
    
    # Calculate implied variance
implied_variance = portfolio_cost / total_weight
implied_vol = np.sqrt(implied_variance)

print(f"\\n{'─' * 70}")
print(f"Implied Variance: {implied_variance:.0f} variance points")
print(f"Implied Volatility: {implied_vol:.1f}%")
print(f"\\nThis portfolio replicates a variance swap struck at {strike_variance:.0f}")

return implied_variance

# Example
spot = 100
strikes = [80, 85, 90, 95, 100, 105, 110, 115, 120]
option_prices = [0.50, 1.00, 2.00, 3.50, 5.00, 3.50, 2.00, 1.00, 0.50]

replicate_variance_swap_with_options(spot, strikes, option_prices, strike_variance = 400)
\`\`\`

---

## Dispersion Trading

### Concept

**Trade correlation** between index and components.

**Strategy:**
- **Sell index volatility** (SPX options)
- **Buy component volatility** (individual stock options)
- Profit when correlation drops (dispersion increases)

\`\`\`python
"""
Dispersion Trade Analysis
"""

def analyze_dispersion_trade(index_iv, component_ivs, weights, correlation):
    """
    Analyze dispersion trade opportunity
    
    Expected relationship:
    Index Variance = Σ (weight² × component_variance) + 2 × Σ Σ (weight_i × weight_j × cov)
    
    When correlation drops, index vol drops more than weighted average of components.
    """
    print("=" * 70)
    print("DISPERSION TRADE ANALYSIS")
    print("=" * 70)
    
    # Calculate weighted average of component IVs
    weighted_avg_iv = np.average(component_ivs, weights=weights)
    
    # Calculate expected index IV based on correlation
    # Simplified: IV_index² ≈ avg(IV_components²) × avg_correlation
    avg_component_variance = np.average(np.array(component_ivs) ** 2, weights=weights)
    expected_index_variance = avg_component_variance * correlation
    expected_index_iv = np.sqrt(expected_index_variance)
    
    # Dispersion
    dispersion = weighted_avg_iv - index_iv
    expected_dispersion = weighted_avg_iv - expected_index_iv
    
    print(f"\\nIndex IV: {index_iv:.1f}%")
    print(f"Weighted Avg Component IV: {weighted_avg_iv:.1f}%")
    print(f"Expected Index IV (given correlation): {expected_index_iv:.1f}%")
    print(f"\\nActual Dispersion: {dispersion:.1f}%")
    print(f"Expected Dispersion: {expected_dispersion:.1f}%")
    print(f"Correlation: {correlation:.2f}")
    
    # Trading opportunity
    if dispersion > expected_dispersion + 2:
        signal = "BUY DISPERSION"
        explanation = "Components rich vs index, or correlation too high"
        trade = "Sell component vol, buy index vol"
    elif dispersion < expected_dispersion - 2:
        signal = "SELL DISPERSION"
        explanation = "Index rich vs components, or correlation too low"
        trade = "Buy component vol, sell index vol"
    else:
        signal = "FAIR VALUE"
        explanation = "No significant mispricing"
        trade = "No trade"
    
    print(f"\\n{'─' * 70}")
    print(f"Signal: {signal}")
    print(f"Explanation: {explanation}")
    print(f"Trade: {trade}")
    
    return dispersion

# Example
index_iv = 18.0
component_ivs = [25, 30, 20, 22, 28, 24, 26, 21, 23, 27]
weights = [0.15, 0.12, 0.10, 0.10, 0.10, 0.08, 0.08, 0.08, 0.10, 0.09]
correlation = 0.60

analyze_dispersion_trade(index_iv, component_ivs, weights, correlation)
\`\`\`

---

## Volatility Arbitrage

### Concept

**Exploit mispricing** in the volatility surface.

**Strategies:**1. **Calendar spread arbitrage:** Front month vs back month
2. **Strike arbitrage:** OTM vs ATM relative value
3. **Cross-asset arbitrage:** SPX vs SPY vs futures

\`\`\`python
"""
Volatility Arbitrage Scanner
"""

def scan_vol_arbitrage(option_chain):
    """
    Scan for volatility arbitrage opportunities
    
    Look for:
    1. IV inversions (front > back month)
    2. Smile arbitrage (OTM too cheap/expensive)
    3. Put-call parity violations
    """
    opportunities = []
    
    print("=" * 70)
    print("VOLATILITY ARBITRAGE SCANNER")
    print("=" * 70)
    
    # Check calendar spreads
    print("\\n1. CALENDAR SPREAD ANALYSIS:")
    for strike in option_chain['strikes']:
        iv_30day = option_chain['iv_30day'][strike]
        iv_60day = option_chain['iv_60day'][strike]
        
        # Front month should be < back month (usually)
        if iv_30day > iv_60day + 3:  # Inversion
            opportunities.append({
                'type': 'Calendar Inversion',
                'strike': strike,
                'front_iv': iv_30day,
                'back_iv': iv_60day,
                'trade': f'Buy {strike} 60-day, sell {strike} 30-day',
                'edge': iv_30day - iv_60day
            })
            print(f"  OPPORTUNITY: Strike \${strike}")
            print(f"    30-day IV: {iv_30day:.1f}% > 60-day IV: {iv_60day:.1f}%")
            print(f"    Inversion: {iv_30day - iv_60day:.1f}%")
    
    # Check smile arbitrage
    print("\\n2. VOLATILITY SMILE ANALYSIS:")
    atm_strike = option_chain['atm_strike']
    atm_iv = option_chain['iv_30day'][atm_strike]
    
    for strike in option_chain['strikes']:
        iv = option_chain['iv_30day'][strike]
        moneyness = strike / option_chain['spot']
        
        # Check for extreme deviations
        deviation = abs(iv - atm_iv)
        if deviation > 10:  # > 10% deviation
            opportunities.append({
                'type': 'Smile Arbitrage',
                'strike': strike,
                'iv': iv,
                'atm_iv': atm_iv,
                'deviation': deviation,
                'moneyness': moneyness
            })
            print(f"  OPPORTUNITY: Strike \${strike} ({moneyness:.2f} moneyness)")
            print(f"    IV: {iv:.1f}% vs ATM: {atm_iv:.1f}%")
            print(f"    Deviation: {deviation:.1f}%")
    
    # Summary
    print(f"\\n{'─' * 70}")
    print(f"Total Opportunities Found: {len(opportunities)}")
    
    return opportunities

# Example scan
option_chain = {
    'spot': 100,
    'atm_strike': 100,
    'strikes': [90, 95, 100, 105, 110],
    'iv_30day': {
        90: 28.0,  # OTM put
        95: 24.0,
        100: 20.0,  # ATM
        105: 18.0,
        110: 16.0   # OTM call
    },
    'iv_60day': {
        90: 26.0,
        95: 23.0,
        100: 21.0,
        105: 19.5,
        110: 18.0
    }
}

opportunities = scan_vol_arbitrage(option_chain)
\`\`\`

---

## Summary

**Key Volatility Trading Concepts:**
- **VRP:** IV typically > RV (sell premium for edge)
- **VIX futures:** Trade term structure (contango/backwardation)
- **VIX ETPs:** Structural decay in contango (VXX, UVXY)
- **Variance swaps:** Pure volatility exposure
- **Dispersion:** Trade index vs component correlation
- **Arbitrage:** Exploit IV mispricings

**Professional Edge:**
- Volatility more predictable than direction
- Mean reversion in IV
- Structural edges (VRP, contango)
- Sophisticated but learnable

In the next section, we'll learn how to manage portfolio-level Greeks and risk across multiple positions.
`,
};
