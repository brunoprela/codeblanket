export const spreadsStrategies = {
  title: 'Spreads: Bull, Bear, Butterfly, Condor',
  id: 'spreads-strategies',
  content: `
# Options Spreads

## Introduction

**Spreads** are multi-leg options strategies that combine multiple calls or puts to create specific risk/reward profiles. They offer:
- **Defined risk and reward** (know max profit/loss upfront)
- **Lower capital requirements** than naked options
- **Flexible positioning** for any market view

**Major Spread Categories:**1. **Vertical Spreads** (same expiration, different strikes)
2. **Butterfly Spreads** (3 strikes, symmetric)
3. **Condor Spreads** (4 strikes, wider range)
4. **Calendar/Diagonal Spreads** (different expirations)

---

## Vertical Spreads

### Bull Call Spread (Covered Earlier)

Bullish, defined risk/reward using calls.

### Bull Put Spread

**Bullish strategy using puts** (alternative to bull call spread).

**Setup:**
- Sell higher strike put
- Buy lower strike put (protection)
- Net: **Credit received**

\`\`\`python
"""
Bull Put Spread
"""

def bull_put_spread_payoff(stock_prices, short_strike, long_strike, net_credit):
    """
    Bullish: Sell higher strike put, buy lower strike put
    """
    short_put = -np.maximum(short_strike - stock_prices, 0)
    long_put = np.maximum(long_strike - stock_prices, 0)
    payoff = short_put + long_put + net_credit
    return payoff

# Example: 100/95 bull put spread
short_strike = 100
long_strike = 95
short_premium = 3.00
long_premium = 1.00
net_credit = short_premium - long_premium  # $2.00

stock_prices = np.linspace(85, 115, 200)
payoff = bull_put_spread_payoff(stock_prices, short_strike, long_strike, net_credit)

plt.figure(figsize=(10, 6))
plt.plot(stock_prices, payoff, 'g-', linewidth=2)
plt.axhline(0, color='black', linestyle='--', alpha=0.3)
plt.axhline(net_credit, color='green', linestyle=':', label=f'Max Profit \${net_credit}')
plt.axhline(-(long_strike - short_strike) + net_credit, color='red', linestyle=':', label='Max Loss')
plt.xlabel('Stock Price at Expiration')
plt.ylabel('Profit / Loss')
plt.title('Bull Put Spread')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

max_profit = net_credit
max_loss = (short_strike - long_strike) - net_credit
breakeven = short_strike - net_credit

print(f"Bull Put Spread: {short_strike}/{long_strike}")
print(f"  Net Credit: \${net_credit}")
print(f"  Max Profit: \${max_profit} (at \${short_strike}+)")
print(f"  Max Loss: \${max_loss} (at \${long_strike} or below)")
print(f"  Breakeven: \${breakeven}")
print(f"  Return on Risk: {(max_profit / max_loss * 100):.1f}%")
\`\`\`

**Bull Call vs Bull Put Spread:**

| Aspect | Bull Call Spread | Bull Put Spread |
|--------|------------------|-----------------|
| **Structure** | Buy call, sell call | Sell put, buy put |
| **Cash Flow** | Debit (pay) | Credit (receive) |
| **Capital** | Lower | Higher (margin) |
| **Max Profit** | Spread - Debit | Credit |
| **Best IV** | Low IV | High IV |

---

### Bear Put Spread (Covered Earlier)

Bearish, defined risk using puts.

### Bear Call Spread

**Bearish strategy using calls** (alternative to bear put spread).

**Setup:**
- Sell lower strike call
- Buy higher strike call (protection)
- Net: **Credit received**

\`\`\`python
"""
Bear Call Spread
"""

def bear_call_spread_payoff(stock_prices, short_strike, long_strike, net_credit):
    """
    Bearish: Sell lower strike call, buy higher strike call
    """
    short_call = -np.maximum(stock_prices - short_strike, 0)
    long_call = np.maximum(stock_prices - long_strike, 0)
    payoff = short_call + long_call + net_credit
    return payoff

# Example: 100/105 bear call spread
short_strike = 100
long_strike = 105
net_credit = 2.00

stock_prices = np.linspace(85, 115, 200)
payoff = bear_call_spread_payoff(stock_prices, short_strike, long_strike, net_credit)

max_profit = net_credit
max_loss = (long_strike - short_strike) - net_credit
breakeven = short_strike + net_credit

print(f"Bear Call Spread: {short_strike}/{long_strike}")
print(f"  Max Profit: \${max_profit} (at \${short_strike} or below)")
print(f"  Max Loss: \${max_loss} (at \${long_strike}+)")
print(f"  Breakeven: \${breakeven}")
\`\`\`

---

## Butterfly Spreads

### Long Call Butterfly

**Neutral strategy:** Profit if stock stays near middle strike.

**Setup:**
- Buy 1 lower strike call
- **Sell 2 middle strike calls**
- Buy 1 higher strike call
- All same expiration

\`\`\`python
"""
Long Call Butterfly
"""

def long_call_butterfly(stock_prices, lower_strike, middle_strike, upper_strike, net_debit):
    """
    Neutral: Profit at middle strike
    
    Structure:
    +1 Call @ lower
    -2 Calls @ middle
    +1 Call @ upper
    """
    long_lower = np.maximum(stock_prices - lower_strike, 0)
    short_middle = -2 * np.maximum(stock_prices - middle_strike, 0)
    long_upper = np.maximum(stock_prices - upper_strike, 0)
    
    payoff = long_lower + short_middle + long_upper - net_debit
    return payoff

# Example: 95/100/105 butterfly
lower = 95
middle = 100
upper = 105

# Typical pricing: ATM more expensive
lower_call_price = 7.00
middle_call_price = 5.00
upper_call_price = 3.00

net_debit = lower_call_price - 2*middle_call_price + upper_call_price
# = 7 - 10 + 3 = 0 (simplified for symmetry)

stock_prices = np.linspace(85, 115, 200)
payoff = long_call_butterfly(stock_prices, lower, middle, upper, net_debit=1.0)

plt.figure(figsize=(12, 6))
plt.plot(stock_prices, payoff, 'purple', linewidth=2)
plt.axhline(0, color='black', linestyle='--', alpha=0.3)
plt.axvline(middle, color='green', linestyle=':', linewidth=2, label=f'Target \${middle}')
plt.xlabel('Stock Price at Expiration')
plt.ylabel('Profit / Loss')
plt.title('Long Call Butterfly: Peak Profit at Middle Strike')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Calculate key metrics
max_profit = (middle - lower) - net_debit
max_loss = net_debit
breakevens = [lower + net_debit, upper - net_debit]

print(f"\\nLong Call Butterfly: {lower}/{middle}/{upper}")
print(f"  Net Debit: \${net_debit:.2f}")
print(f"  Max Profit: \${max_profit:.2f} (at \${middle})")
print(f"  Max Loss: \${net_debit:.2f} (at \${lower} or below, \${upper} or above)")
print(f"  Breakevens: \${breakevens[0]:.2f} and \${breakevens[1]:.2f}")
print(f"  Profit Range: \${breakevens[0]:.2f} - \${breakevens[1]:.2f}")
\`\`\`

### Long Put Butterfly

Same structure using puts:
- Buy 1 higher strike put
- Sell 2 middle strike puts
- Buy 1 lower strike put

**When to Use Butterflies:**
- **Neutral outlook** (expect minimal movement)
- **Low IV** (cheap to establish)
- **Event-driven:** Expect calm after event
- **High probability,** low profit per trade

---

## Iron Butterfly

Combination of short straddle + protective wings.

**Setup:**
- Sell ATM call
- Sell ATM put
- Buy OTM call (protection)
- Buy OTM put (protection)

\`\`\`python
"""
Iron Butterfly
"""

def iron_butterfly(stock_prices, atm_strike, wing_width, net_credit):
    """
    Short straddle + protective wings
    
    Max profit at ATM strike
    """
    # Sell ATM straddle
    short_call = -np.maximum(stock_prices - atm_strike, 0)
    short_put = -np.maximum(atm_strike - stock_prices, 0)
    
    # Buy wings
    long_call = np.maximum(stock_prices - (atm_strike + wing_width), 0)
    long_put = np.maximum((atm_strike - wing_width) - stock_prices, 0)
    
    payoff = short_call + short_put + long_call + long_put + net_credit
    return payoff

# Example
atm_strike = 100
wing_width = 10
net_credit = 5.00  # Selling straddle collects more than wing cost

stock_prices = np.linspace(80, 120, 200)
payoff = iron_butterfly(stock_prices, atm_strike, wing_width, net_credit)

plt.figure(figsize=(10, 6))
plt.plot(stock_prices, payoff, 'b-', linewidth=2)
plt.axhline(0, color='black', linestyle='--', alpha=0.3)
plt.axvline(atm_strike, color='green', linestyle=':', label='ATM Strike')
plt.xlabel('Stock Price')
plt.ylabel('P&L')
plt.title('Iron Butterfly')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

max_profit = net_credit
max_loss = wing_width - net_credit

print(f"Iron Butterfly: ATM \${atm_strike}, Wings ±\${wing_width}")
print(f"  Max Profit: \${max_profit} (at \${atm_strike})")
print(f"  Max Loss: \${max_loss} (at wings or beyond)")
\`\`\`

---

## Condor Spreads

### Iron Condor (Covered Earlier)

Wide range, 4 different strikes.

### Call Condor

**Setup:**
- Buy 1 call (lowest strike)
- Sell 1 call (lower-middle)
- Sell 1 call (upper-middle)
- Buy 1 call (highest strike)

\`\`\`python
"""
Long Call Condor
"""

def long_call_condor(stock_prices, strikes, net_debit):
    """
    4 strikes: lowest, lower-mid, upper-mid, highest
    """
    long_lowest = np.maximum(stock_prices - strikes[0], 0)
    short_lower_mid = -np.maximum(stock_prices - strikes[1], 0)
    short_upper_mid = -np.maximum(stock_prices - strikes[2], 0)
    long_highest = np.maximum(stock_prices - strikes[3], 0)
    
    payoff = long_lowest + short_lower_mid + short_upper_mid + long_highest - net_debit
    return payoff

# Example: 90/95/105/110 condor
strikes = [90, 95, 105, 110]
net_debit = 2.00

stock_prices = np.linspace(80, 120, 200)
payoff = long_call_condor(stock_prices, strikes, net_debit)

plt.figure(figsize=(10, 6))
plt.plot(stock_prices, payoff, 'orange', linewidth=2)
plt.axhline(0, color='black', linestyle='--', alpha=0.3)
for strike in strikes:
    plt.axvline(strike, color='gray', linestyle=':', alpha=0.5)
plt.xlabel('Stock Price')
plt.ylabel('P&L')
plt.title('Long Call Condor: Wider Profit Zone than Butterfly')
plt.grid(True, alpha=0.3)
plt.show()

print(f"Condor strikes: {strikes}")
print(f"Profit zone: \${strikes[1]}-\${strikes[2]}")
\`\`\`

---

## Calendar/Diagonal Spreads

### Calendar Spread (Time Spread)

**Setup:**
- Sell near-term option (30 days)
- Buy longer-term option (60-90 days)
- **Same strike**

**Profit Driver:** Front-month decays faster than back-month.

\`\`\`python
"""
Calendar Spread Simulation
"""

def simulate_calendar_spread(S, K, T_short, T_long, sigma, days_to_track=30):
    """
    Track P&L of calendar spread over time
    """
    results = []
    
    for day in range(days_to_track + 1):
        # Time remaining
        t_short = max(T_short - day/365, 0.001)
        t_long = T_long - day/365
        
        # Option values
        if day < T_short * 365:
            short_value = black_scholes_price(S, K, t_short, 0.05, sigma, 'call')
        else:
            short_value = 0  # Expired
        
        long_value = black_scholes_price(S, K, t_long, 0.05, sigma, 'call')
        
        # Spread value (long - short, since we're short front)
        spread_value = long_value - short_value
        
        results.append({
            'day': day,
            'short_value': short_value,
            'long_value': long_value,
            'spread_value': spread_value
        })
    
    return pd.DataFrame(results)

# Simulate
S = 100
K = 100  # ATM
T_short = 30/365
T_long = 90/365
sigma = 0.25

df = simulate_calendar_spread(S, K, T_short, T_long, sigma)

# Plot
plt.figure(figsize=(12, 5))
plt.plot(df['day'], df['spread_value'], linewidth=2)
plt.axvline(30, color='red', linestyle='--', label='Front Month Expiration')
plt.xlabel('Days from Entry')
plt.ylabel('Spread Value')
plt.title('Calendar Spread Value Over Time (Stock Stays at $100)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

initial_value = df['spread_value'].iloc[0]
final_value = df['spread_value'].iloc[-1]
profit = final_value - initial_value

print(f"\\nCalendar Spread Simulation:")
print(f"  Initial Cost: \${initial_value:.2f}")
print(f"  Value at Day 30: \${df['spread_value'].iloc[30]:.2f}")
print(f"  Profit: \${profit:.2f}")
\`\`\`

### Diagonal Spread

Like calendar but **different strikes** (e.g., sell ATM near-term, buy OTM long-term).

---

## Spread Comparison

\`\`\`python
"""
Compare Spread Strategies
"""

def compare_all_spreads():
    stock_prices = np.linspace(80, 120, 200)
    
    spreads = {
        'Bull Call Spread': bull_call_spread(stock_prices, 100, 105, 3),
        'Bull Put Spread': bull_put_spread_payoff(stock_prices, 100, 95, 2),
        'Bear Put Spread': bear_put_spread(stock_prices, 100, 95, 2.5),
        'Bear Call Spread': bear_call_spread_payoff(stock_prices, 100, 105, 2),
        'Iron Condor': iron_condor(stock_prices, 95, 90, 105, 110, 2),
        'Iron Butterfly': iron_butterfly(stock_prices, 100, 10, 5),
        'Call Butterfly': long_call_butterfly(stock_prices, 95, 100, 105, 1),
    }
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes = axes.flatten()
    
    for idx, (name, payoff) in enumerate(spreads.items()):
        ax = axes[idx]
        ax.plot(stock_prices, payoff, linewidth=2)
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax.axvline(100, color='gray', linestyle=':', alpha=0.5)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_xlabel('Stock Price')
        ax.set_ylabel('P&L')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

compare_all_spreads()
\`\`\`

---

## Summary

**Vertical Spreads:**
- Directional (bull/bear)
- Defined risk/reward
- Lower cost than naked options

**Butterfly/Condor:**
- Neutral strategies
- Profit in narrow range
- Low risk, limited profit

**Calendar:**
- Theta play (time decay)
- Best with stable stock
- Requires adjustments

**Key Principle:** Match spread to market view + IV regime.
`,
};
