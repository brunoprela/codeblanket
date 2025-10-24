export const optionsFundamentals = {
  title: 'Options Fundamentals',
  id: 'options-fundamentals',
  content: `
# Options Fundamentals

## Introduction

Options are derivative contracts that give the holder the **right, but not the obligation**, to buy or sell an underlying asset at a specified price (strike price) on or before a specified date (expiration). Options are fundamental to quantitative finance for:

- **Risk management**: Hedging portfolio positions
- **Income generation**: Selling covered calls, cash-secured puts
- **Speculation**: Leveraged directional bets
- **Volatility trading**: Profiting from volatility changes
- **Complex strategies**: Spreads, straddles, butterflies

By the end of this section, you'll understand:
- Call and put options mechanics
- Option payoff diagrams and profit/loss
- Intrinsic value vs time value
- European vs American options
- Common option strategies
- How to implement option pricing in Python

### Why Options Matter for Quants

Options provide asymmetric payoffs—limited downside with unlimited (or large) upside. This non-linearity creates opportunities:

1. **Hedging**: A portfolio manager holding stocks can buy put options as insurance
2. **Income**: Writing covered calls generates premium income
3. **Leverage**: Control $10,000 of stock with $1,000 in options
4. **Volatility**: Trade volatility directly (volatility is an asset class)
5. **Tail risk**: Protect against black swan events

---

## Call Options

### Definition

A **call option** gives the buyer the right to **buy** the underlying asset at the strike price.

**Contract specifications**:
- **Underlying**: Stock, index, commodity, currency
- **Strike price (K)**: Price at which you can buy
- **Expiration date (T)**: Last day to exercise
- **Premium**: Price paid for the option
- **Contract size**: Typically 100 shares per contract

### Payoff Diagram

At expiration, call option payoff:

\`\`\`
Long Call Payoff = max(S_T - K, 0)
where S_T = stock price at expiration
\`\`\`

- If S_T > K: Exercise the option, profit = S_T - K - premium
- If S_T ≤ K: Let it expire worthless, loss = premium

\`\`\`python
"""
Call Option Payoff Visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def call_payoff(spot_prices, strike, premium, position='long'):
    """
    Calculate call option payoff
    
    Parameters:
    -----------
    spot_prices : array
        Range of underlying prices at expiration
    strike : float
        Strike price
    premium : float
        Option premium paid/received
    position : str
        'long' or 'short'
    
    Returns:
    --------
    payoff : array
        Payoff at each price point
    """
    intrinsic = np.maximum(spot_prices - strike, 0)
    
    if position == 'long':
        # Long call: pay premium, receive intrinsic value
        payoff = intrinsic - premium
    else:
        # Short call: receive premium, pay intrinsic value
        payoff = premium - intrinsic
    
    return payoff

# Example: AAPL trading at $150
current_price = 150
strike = 155  # Call option with strike $155
premium = 5   # Premium paid: $5 per share

# Range of prices at expiration
spot_range = np.linspace(130, 180, 100)

# Calculate payoffs
long_call = call_payoff(spot_range, strike, premium, 'long')
short_call = call_payoff(spot_range, strike, premium, 'short')

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Long Call
axes[0].plot(spot_range, long_call, label='Long Call P/L', color='green', linewidth=2)
axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[0].axvline(x=strike, color='red', linestyle='--', alpha=0.5, label=f'Strike \${strike}')
axes[0].axvline(x=current_price, color='blue', linestyle='--', alpha=0.5, label=f'Current \${current_price}')
axes[0].fill_between(spot_range, 0, long_call, where=(long_call > 0), alpha=0.3, color='green', label='Profit zone')
axes[0].fill_between(spot_range, 0, long_call, where=(long_call < 0), alpha=0.3, color='red', label='Loss zone')
axes[0].set_xlabel('Stock Price at Expiration ($)', fontsize=12)
axes[0].set_ylabel('Profit/Loss ($)', fontsize=12)
axes[0].set_title(f'Long Call: Strike \${strike}, Premium \${premium}', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Calculate breakeven
breakeven = strike + premium
axes[0].axvline(x=breakeven, color='orange', linestyle=':', linewidth=2, label=f'Breakeven \${breakeven}')
axes[0].text(breakeven + 1, long_call.max()/2, f'BE: \${breakeven}', fontsize=10, fontweight='bold')

# Short Call
axes[1].plot(spot_range, short_call, label='Short Call P/L', color='red', linewidth=2)
axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[1].axvline(x=strike, color='red', linestyle='--', alpha=0.5, label=f'Strike \${strike}')
axes[1].axvline(x=current_price, color='blue', linestyle='--', alpha=0.5, label=f'Current \${current_price}')
axes[1].fill_between(spot_range, 0, short_call, where=(short_call > 0), alpha=0.3, color='green', label='Profit zone')
axes[1].fill_between(spot_range, 0, short_call, where=(short_call < 0), alpha=0.3, color='red', label='Loss zone')
axes[1].set_xlabel('Stock Price at Expiration ($)', fontsize=12)
axes[1].set_ylabel('Profit/Loss ($)', fontsize=12)
axes[1].set_title(f'Short Call: Strike \${strike}, Premium \${premium}', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Calculate key metrics
print("=== CALL OPTION ANALYSIS ===")
print(f"\\nCurrent Stock Price: \${current_price}")
print(f"Strike Price: \${strike}")
print(f"Premium: \${premium}")
print(f"\\nLong Call:")
print(f"  Max Loss: \${premium} (at any price ≤ \${strike})")
print(f"  Max Gain: Unlimited (as stock rises)")
print(f"  Breakeven: \${strike + premium}")
print(f"\\nShort Call:")
print(f"  Max Gain: \${premium} (at any price ≤ \${strike})")
print(f"  Max Loss: Unlimited (as stock rises)")
print(f"  Breakeven: \${strike + premium}")
\`\`\`

**Key insights**:
- **Long call**: Limited risk (premium), unlimited profit potential
- **Short call**: Limited profit (premium), unlimited risk
- **Breakeven**: Strike + premium
- **In-the-money (ITM)**: Current price > strike (call has intrinsic value)
- **At-the-money (ATM)**: Current price ≈ strike
- **Out-of-the-money (OTM)**: Current price < strike (call has no intrinsic value)

---

## Put Options

### Definition

A **put option** gives the buyer the right to **sell** the underlying asset at the strike price.

### Payoff Diagram

At expiration, put option payoff:

\`\`\`
Long Put Payoff = max(K - S_T, 0)
\`\`\`

- If S_T < K: Exercise the option, profit = K - S_T - premium
- If S_T ≥ K: Let it expire worthless, loss = premium

\`\`\`python
"""
Put Option Payoff Visualization
"""

def put_payoff(spot_prices, strike, premium, position='long'):
    """Calculate put option payoff"""
    intrinsic = np.maximum(strike - spot_prices, 0)
    
    if position == 'long':
        payoff = intrinsic - premium
    else:
        payoff = premium - intrinsic
    
    return payoff

# Example: AAPL trading at $150
strike = 145  # Put option with strike $145
premium = 4   # Premium paid: $4 per share

# Calculate payoffs
long_put = put_payoff(spot_range, strike, premium, 'long')
short_put = put_payoff(spot_range, strike, premium, 'short')

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Long Put
axes[0].plot(spot_range, long_put, label='Long Put P/L', color='blue', linewidth=2)
axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[0].axvline(x=strike, color='red', linestyle='--', alpha=0.5, label=f'Strike \${strike}')
axes[0].axvline(x=current_price, color='green', linestyle='--', alpha=0.5, label=f'Current \${current_price}')
axes[0].fill_between(spot_range, 0, long_put, where=(long_put > 0), alpha=0.3, color='green')
axes[0].fill_between(spot_range, 0, long_put, where=(long_put < 0), alpha=0.3, color='red')
axes[0].set_xlabel('Stock Price at Expiration ($)', fontsize=12)
axes[0].set_ylabel('Profit/Loss ($)', fontsize=12)
axes[0].set_title(f'Long Put: Strike \${strike}, Premium \${premium}', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Breakeven for put
breakeven_put = strike - premium
axes[0].axvline(x=breakeven_put, color='orange', linestyle=':', linewidth=2)
axes[0].text(breakeven_put - 8, long_put.max()/2, f'BE: \${breakeven_put}', fontsize=10, fontweight='bold')

# Short Put
axes[1].plot(spot_range, short_put, label='Short Put P/L', color='purple', linewidth=2)
axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[1].axvline(x=strike, color='red', linestyle='--', alpha=0.5, label=f'Strike \${strike}')
axes[1].axvline(x=current_price, color='green', linestyle='--', alpha=0.5, label=f'Current \${current_price}')
axes[1].fill_between(spot_range, 0, short_put, where=(short_put > 0), alpha=0.3, color='green')
axes[1].fill_between(spot_range, 0, short_put, where=(short_put < 0), alpha=0.3, color='red')
axes[1].set_xlabel('Stock Price at Expiration ($)', fontsize=12)
axes[1].set_ylabel('Profit/Loss ($)', fontsize=12)
axes[1].set_title(f'Short Put: Strike \${strike}, Premium \${premium}', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\\n=== PUT OPTION ANALYSIS ===")
print(f"\\nCurrent Stock Price: \${current_price}")
print(f"Strike Price: \${strike}")
print(f"Premium: \${premium}")
print(f"\\nLong Put:")
print(f"  Max Loss: \${premium} (at any price ≥ \${strike})")
print(f"  Max Gain: \${strike - premium} (if stock goes to $0)")
print(f"  Breakeven: \${strike - premium}")
print(f"\\nShort Put:")
print(f"  Max Gain: \${premium} (at any price ≥ \${strike})")
print(f"  Max Loss: \${strike - premium} (if stock goes to $0)")
print(f"  Breakeven: \${strike - premium}")
\`\`\`

**Key insights**:
- **Long put**: Limited risk (premium), large profit potential (stock can drop to $0)
- **Short put**: Limited profit (premium), large risk (if stock crashes)
- **Breakeven**: Strike - premium
- **In-the-money (ITM)**: Current price < strike (put has intrinsic value)
- **Out-of-the-money (OTM)**: Current price > strike (put has no intrinsic value)

---

## Intrinsic Value vs Time Value

### Option Value Components

Option premium = **Intrinsic Value** + **Time Value**

**Intrinsic Value** = Value if exercised immediately
- Call: max(S - K, 0)
- Put: max(K - S, 0)

**Time Value** = Premium - Intrinsic Value
- Represents probability of finishing in-the-money
- Decays to zero at expiration (theta decay)
- Highest for at-the-money options

\`\`\`python
"""
Intrinsic vs Time Value Analysis
"""

class Option:
    """Simple option pricing (intrinsic value only)"""
    
    def __init__(self, spot, strike, option_type='call'):
        self.spot = spot
        self.strike = strike
        self.option_type = option_type
    
    def intrinsic_value(self):
        """Calculate intrinsic value"""
        if self.option_type == 'call':
            return max(self.spot - self.strike, 0)
        else:  # put
            return max(self.strike - self.spot, 0)
    
    def analyze_value(self, market_premium):
        """Analyze option value components"""
        intrinsic = self.intrinsic_value()
        time_value = market_premium - intrinsic
        
        return {
            'spot': self.spot,
            'strike': self.strike,
            'type': self.option_type,
            'market_premium': market_premium,
            'intrinsic_value': intrinsic,
            'time_value': time_value,
            'pct_time_value': (time_value / market_premium * 100) if market_premium > 0 else 0,
            'moneyness': self.get_moneyness()
        }
    
    def get_moneyness(self):
        """Determine if ITM, ATM, or OTM"""
        if self.option_type == 'call':
            if self.spot > self.strike + 0.50:
                return 'ITM'
            elif abs(self.spot - self.strike) <= 0.50:
                return 'ATM'
            else:
                return 'OTM'
        else:  # put
            if self.spot < self.strike - 0.50:
                return 'ITM'
            elif abs(self.spot - self.strike) <= 0.50:
                return 'ATM'
            else:
                return 'OTM'

# Example: Analyze AAPL options
spot_price = 150

# Call options at different strikes
call_chain = [
    {'strike': 140, 'premium': 12.50},  # ITM
    {'strike': 150, 'premium': 6.00},   # ATM
    {'strike': 160, 'premium': 2.00},   # OTM
]

print("=== CALL OPTIONS ANALYSIS ===")
print(f"Spot Price: \${spot_price}\\n")

results = []
for opt in call_chain:
    option = Option(spot_price, opt['strike'], 'call')
    analysis = option.analyze_value(opt['premium'])
    results.append(analysis)
    
    print(f"Strike \${analysis['strike']}: {analysis['moneyness']}")
    print(f"  Market Premium: \${analysis['market_premium']:.2f}")
    print(f"  Intrinsic Value: \${analysis['intrinsic_value']:.2f}")
    print(f"  Time Value: \${analysis['time_value']:.2f} ({analysis['pct_time_value']:.1f}%)")
    print()

# Visualize value components
df = pd.DataFrame(results)

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(df))
width = 0.35

bars1 = ax.bar(x - width/2, df['intrinsic_value'], width, label='Intrinsic Value', color='skyblue')
bars2 = ax.bar(x + width/2, df['time_value'], width, label='Time Value', color='lightcoral')

ax.set_xlabel('Option Strike', fontsize=12)
ax.set_ylabel('Value ($)', fontsize=12)
ax.set_title('Call Option Value Components', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f"\${s['strike']}\\n{Option(spot_price, s['strike'], 'call').get_moneyness()}" 
                     for s in call_chain])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                   f'\${height:.2f}', ha='center', va='center', fontweight='bold')

plt.tight_layout()
plt.show()
\`\`\`

**Key observations**:
1. **ITM options**: Mostly intrinsic value, less time value
2. **ATM options**: Maximum time value (highest uncertainty)
3. **OTM options**: All time value, no intrinsic value
4. **Time decay**: Time value decreases as expiration approaches
5. **Volatility**: Higher volatility → higher time value

---

## European vs American Options

### European Options

- Can only be exercised **at expiration**
- Most index options (SPX, NDX)
- Easier to price (closed-form solutions)
- Generally less valuable than American

### American Options

- Can be exercised **any time before expiration**
- Most stock options
- More valuable (early exercise optionality)
- No closed-form solution (use binomial trees or numerical methods)

### Early Exercise Considerations

**When to exercise American options early?**

**Calls on non-dividend stocks**: NEVER exercise early
- Reason: Time value is always positive
- Better to sell the option than exercise

**Calls on dividend-paying stocks**: Consider exercise just before ex-dividend
- If dividend > time value remaining, exercise to capture dividend

**Puts**: May exercise early if deep in-the-money
- Capture intrinsic value now vs waiting
- Avoid risk of stock price recovering

\`\`\`python
"""
Early Exercise Analysis
"""

def should_exercise_early(option_type, spot, strike, time_value, dividend=0):
    """
    Determine if early exercise is optimal
    
    Parameters:
    -----------
    option_type : str
        'call' or 'put'
    spot : float
        Current stock price
    strike : float
        Strike price
    time_value : float
        Current time value of option
    dividend : float
        Dividend amount (for calls)
    
    Returns:
    --------
    dict : Analysis results
    """
    intrinsic = max(spot - strike, 0) if option_type == 'call' else max(strike - spot, 0)
    
    if option_type == 'call':
        # For calls: exercise if dividend > time value
        exercise_early = dividend > time_value
        reason = f"Dividend (\${dividend}) > Time Value (\${time_value})" if exercise_early else \
                 f"Dividend (\${dividend}) ≤ Time Value (\${time_value})"
    else:  # put
        # For puts: consider if deep ITM and time value is low
        if spot > 0:
            moneyness_ratio = strike / spot
        else:
            moneyness_ratio = float('inf')
        
        # Rule of thumb: exercise if put is >20% ITM and time value < 5% of intrinsic
        deep_itm = moneyness_ratio > 1.20
        low_time_value = (time_value / intrinsic < 0.05) if intrinsic > 0 else False
        
        exercise_early = deep_itm and low_time_value
        reason = f"Deep ITM ({moneyness_ratio:.2f}x) and low time value" if exercise_early else \
                 f"Not deep enough ITM or significant time value remains"
    
    return {
        'option_type': option_type,
        'intrinsic_value': intrinsic,
        'time_value': time_value,
        'total_value': intrinsic + time_value,
        'exercise_early': exercise_early,
        'reason': reason,
        'recommendation': 'EXERCISE' if exercise_early else 'HOLD/SELL'
    }

# Example 1: Call option on dividend-paying stock
print("=== EARLY EXERCISE ANALYSIS ===\\n")
print("Example 1: Call option on dividend-paying stock")
print("  Stock: $100, Strike: $90, Time Value: $2, Dividend: $3\\n")

analysis1 = should_exercise_early('call', spot=100, strike=90, time_value=2, dividend=3)
print(f"  Intrinsic Value: \${analysis1['intrinsic_value']:.2f}")
print(f"  Time Value: \${analysis1['time_value']:.2f}")
print(f"  Reason: {analysis1['reason']}")
print(f"  Recommendation: {analysis1['recommendation']}\\n")

# Example 2: Deep ITM put option
print("Example 2: Deep ITM put option")
print("  Stock: $50, Strike: $70, Time Value: $1\\n")

analysis2 = should_exercise_early('put', spot=50, strike=70, time_value=1)
print(f"  Intrinsic Value: \${analysis2['intrinsic_value']:.2f}")
print(f"  Time Value: \${analysis2['time_value']:.2f}")
print(f"  Reason: {analysis2['reason']}")
print(f"  Recommendation: {analysis2['recommendation']}")
\`\`\`

---

## Common Option Strategies

### 1. Covered Call

**Setup**: Own 100 shares + Sell 1 call option
**Goal**: Generate income from premium
**Risk**: Limited upside (capped at strike)
**Ideal**: Neutral to slightly bullish, expect sideways movement

\`\`\`python
"""
Covered Call Strategy
"""

def covered_call(spot_prices, stock_purchase, strike, premium):
    """Calculate covered call payoff"""
    # Long stock payoff
    stock_pl = spot_prices - stock_purchase
    
    # Short call payoff
    call_pl = premium - np.maximum(spot_prices - strike, 0)
    
    # Combined
    total_pl = stock_pl + call_pl
    
    return stock_pl, call_pl, total_pl

# Example: Own AAPL at $150, sell $160 call for $3
stock_cost = 150
call_strike = 160
call_premium = 3

spot_range = np.linspace(130, 180, 100)
stock_pl, call_pl, total_pl = covered_call(spot_range, stock_cost, call_strike, call_premium)

plt.figure(figsize=(12, 6))
plt.plot(spot_range, stock_pl, label='Long Stock', linestyle='--', alpha=0.7)
plt.plot(spot_range, call_pl, label='Short Call', linestyle='--', alpha=0.7)
plt.plot(spot_range, total_pl, label='Covered Call (Combined)', linewidth=2, color='green')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.axvline(x=stock_cost, color='blue', linestyle=':', label=f'Stock Cost \${stock_cost}')
plt.axvline(x=call_strike, color='red', linestyle=':', label=f'Call Strike \${call_strike}')
plt.fill_between(spot_range, 0, total_pl, where=(total_pl > 0), alpha=0.2, color='green')
plt.fill_between(spot_range, 0, total_pl, where=(total_pl < 0), alpha=0.2, color='red')
plt.xlabel('Stock Price at Expiration ($)')
plt.ylabel('Profit/Loss ($)')
plt.title('Covered Call Strategy', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print("=== COVERED CALL ANALYSIS ===")
print(f"Stock Purchase: \${stock_cost}")
print(f"Call Strike: \${call_strike}")
print(f"Premium Received: \${call_premium}")
print(f"\\nMax Profit: \${(call_strike - stock_cost) + call_premium} (at \${call_strike} or higher)")
print(f"Max Loss: \${stock_cost - call_premium} (if stock goes to $0)")
print(f"Breakeven: \${stock_cost - call_premium}")
\`\`\`

### 2. Protective Put (Married Put)

**Setup**: Own 100 shares + Buy 1 put option
**Goal**: Protect against downside risk (insurance)
**Risk**: Premium paid (cost of insurance)
**Ideal**: Bullish but want protection

\`\`\`python
"""
Protective Put Strategy
"""

def protective_put(spot_prices, stock_purchase, strike, premium):
    """Calculate protective put payoff"""
    # Long stock
    stock_pl = spot_prices - stock_purchase
    
    # Long put
    put_pl = np.maximum(strike - spot_prices, 0) - premium
    
    # Combined
    total_pl = stock_pl + put_pl
    
    return stock_pl, put_pl, total_pl

# Example: Own AAPL at $150, buy $145 put for $4
put_strike = 145
put_premium = 4

stock_pl, put_pl, total_pl = protective_put(spot_range, stock_cost, put_strike, put_premium)

plt.figure(figsize=(12, 6))
plt.plot(spot_range, stock_pl, label='Long Stock', linestyle='--', alpha=0.7)
plt.plot(spot_range, put_pl, label='Long Put', linestyle='--', alpha=0.7)
plt.plot(spot_range, total_pl, label='Protective Put (Combined)', linewidth=2, color='blue')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.axvline(x=stock_cost, color='blue', linestyle=':', label=f'Stock Cost \${stock_cost}')
plt.axvline(x=put_strike, color='red', linestyle=':', label=f'Put Strike \${put_strike}')
plt.fill_between(spot_range, 0, total_pl, where=(total_pl > 0), alpha=0.2, color='green')
plt.fill_between(spot_range, 0, total_pl, where=(total_pl < 0), alpha=0.2, color='red')
plt.xlabel('Stock Price at Expiration ($)')
plt.ylabel('Profit/Loss ($)')
plt.title('Protective Put Strategy', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print("\\n=== PROTECTIVE PUT ANALYSIS ===")
print(f"Stock Purchase: \${stock_cost}")
print(f"Put Strike: \${put_strike}")
print(f"Premium Paid: \${put_premium}")
print(f"\\nMax Loss: \${(stock_cost - put_strike) + put_premium} (protected below \${put_strike})")
print(f"Max Gain: Unlimited (minus premium)")
print(f"Breakeven: \${stock_cost + put_premium}")
\`\`\`

### 3. Bull Call Spread

**Setup**: Buy call at strike K1 + Sell call at strike K2 (K2 > K1)
**Goal**: Reduce cost of long call by capping upside
**Risk**: Limited to net premium paid
**Ideal**: Moderately bullish

\`\`\`python
"""
Bull Call Spread
"""

def bull_call_spread(spot_prices, long_strike, short_strike, long_premium, short_premium):
    """Calculate bull call spread payoff"""
    # Long call
    long_call_pl = np.maximum(spot_prices - long_strike, 0) - long_premium
    
    # Short call
    short_call_pl = short_premium - np.maximum(spot_prices - short_strike, 0)
    
    # Combined
    total_pl = long_call_pl + short_call_pl
    
    return long_call_pl, short_call_pl, total_pl

# Example: Buy $150 call for $6, sell $160 call for $2
long_strike = 150
short_strike = 160
long_premium = 6
short_premium = 2

long_call_pl, short_call_pl, total_pl = bull_call_spread(
    spot_range, long_strike, short_strike, long_premium, short_premium
)

plt.figure(figsize=(12, 6))
plt.plot(spot_range, long_call_pl, label=f'Long \${long_strike} Call', linestyle='--', alpha=0.7)
plt.plot(spot_range, short_call_pl, label=f'Short \${short_strike} Call', linestyle='--', alpha=0.7)
plt.plot(spot_range, total_pl, label='Bull Call Spread', linewidth=2, color='green')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.axvline(x=long_strike, color='blue', linestyle=':', alpha=0.5)
plt.axvline(x=short_strike, color='red', linestyle=':', alpha=0.5)
plt.fill_between(spot_range, 0, total_pl, where=(total_pl > 0), alpha=0.2, color='green')
plt.fill_between(spot_range, 0, total_pl, where=(total_pl < 0), alpha=0.2, color='red')
plt.xlabel('Stock Price at Expiration ($)')
plt.ylabel('Profit/Loss ($)')
plt.title('Bull Call Spread', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

net_debit = long_premium - short_premium
max_profit = (short_strike - long_strike) - net_debit
max_loss = net_debit
breakeven = long_strike + net_debit

print("\\n=== BULL CALL SPREAD ANALYSIS ===")
print(f"Long Call: \${long_strike} strike, \${long_premium} premium")
print(f"Short Call: \${short_strike} strike, \${short_premium} premium")
print(f"\\nNet Debit: \${net_debit}")
print(f"Max Profit: \${max_profit} (at \${short_strike} or higher)")
print(f"Max Loss: \${max_loss} (at \${long_strike} or lower)")
print(f"Breakeven: \${breakeven}")
print(f"Risk/Reward Ratio: {max_loss/max_profit:.2f}:1")
\`\`\`

### 4. Iron Condor

**Setup**: Sell OTM put + Buy further OTM put + Sell OTM call + Buy further OTM call
**Goal**: Profit from low volatility (range-bound)
**Risk**: Limited to difference in strikes minus net credit
**Ideal**: Neutral, expect stock to stay in range

\`\`\`python
"""
Iron Condor Strategy
"""

def iron_condor(spot_prices, put_strikes, call_strikes, put_premiums, call_premiums):
    """
    Calculate iron condor payoff
    
    Parameters:
    -----------
    put_strikes : tuple (long, short)
    call_strikes : tuple (short, long)
    put_premiums : tuple (long, short)
    call_premiums : tuple (short, long)
    """
    # Put spread (bull put spread)
    long_put_pl = np.maximum(put_strikes[0] - spot_prices, 0) - put_premiums[0]
    short_put_pl = put_premiums[1] - np.maximum(put_strikes[1] - spot_prices, 0)
    put_spread_pl = long_put_pl + short_put_pl
    
    # Call spread (bear call spread)
    short_call_pl = call_premiums[0] - np.maximum(spot_prices - call_strikes[0], 0)
    long_call_pl = np.maximum(spot_prices - call_strikes[1], 0) - call_premiums[1]
    call_spread_pl = short_call_pl + long_call_pl
    
    # Combined
    total_pl = put_spread_pl + call_spread_pl
    
    return put_spread_pl, call_spread_pl, total_pl

# Example Iron Condor on AAPL at $150
# Buy $135 put, sell $140 put, sell $160 call, buy $165 call
put_strikes = (135, 140)   # (long, short)
call_strikes = (160, 165)  # (short, long)
put_premiums = (1, 2.5)    # (long, short)
call_premiums = (2.5, 1)   # (short, long)

put_spread_pl, call_spread_pl, total_pl = iron_condor(
    spot_range, put_strikes, call_strikes, put_premiums, call_premiums
)

plt.figure(figsize=(14, 7))
plt.plot(spot_range, put_spread_pl, label='Bull Put Spread', linestyle='--', alpha=0.7)
plt.plot(spot_range, call_spread_pl, label='Bear Call Spread', linestyle='--', alpha=0.7)
plt.plot(spot_range, total_pl, label='Iron Condor', linewidth=2.5, color='darkblue')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

# Mark strikes
for strike in [put_strikes[0], put_strikes[1], call_strikes[0], call_strikes[1]]:
    plt.axvline(x=strike, color='gray', linestyle=':', alpha=0.4)

plt.axvspan(put_strikes[1], call_strikes[0], alpha=0.1, color='green', label='Profit Zone')
plt.fill_between(spot_range, 0, total_pl, where=(total_pl > 0), alpha=0.2, color='green')
plt.fill_between(spot_range, 0, total_pl, where=(total_pl < 0), alpha=0.2, color='red')

plt.xlabel('Stock Price at Expiration ($)')
plt.ylabel('Profit/Loss ($)')
plt.title('Iron Condor Strategy', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

net_credit = (put_premiums[1] - put_premiums[0]) + (call_premiums[0] - call_premiums[1])
max_profit = net_credit
max_loss = (put_strikes[1] - put_strikes[0]) - net_credit

print("\\n=== IRON CONDOR ANALYSIS ===")
print(f"Put Spread: Buy \${put_strikes[0]} / Sell \${put_strikes[1]}")
print(f"Call Spread: Sell \${call_strikes[0]} / Buy \${call_strikes[1]}")
print(f"\\nNet Credit Received: \${net_credit}")
print(f"Max Profit: \${max_profit} (stock between \${put_strikes[1]} and \${call_strikes[0]})")
print(f"Max Loss: \${max_loss} (stock outside wings)")
print(f"\\nProfit Range: \${put_strikes[1]} to \${call_strikes[0]}")
print(f"Breakevens: \${put_strikes[1] - net_credit:.2f} and \${call_strikes[0] + net_credit:.2f}")
print(f"Probability of Profit: ~{(call_strikes[0] - put_strikes[1]) / (2 * current_price) * 100:.1f}% (rough estimate)")
\`\`\`

---

## Option Trading in Python

### Real-World Option Data

\`\`\`python
"""
Fetching Real Option Data
"""

import yfinance as yf
from datetime import datetime, timedelta

def get_option_chain(ticker, expiration_date=None):
    """
    Fetch option chain for a ticker
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    expiration_date : str or None
        Expiration date (YYYY-MM-DD) or None for nearest expiration
    
    Returns:
    --------
    dict : Contains calls and puts dataframes
    """
    stock = yf.Ticker(ticker)
    
    # Get available expiration dates
    expirations = stock.options
    
    if expiration_date is None:
        # Use nearest expiration
        expiration_date = expirations[0]
    
    # Get option chain
    opt_chain = stock.option_chain(expiration_date)
    
    return {
        'ticker': ticker,
        'expiration': expiration_date,
        'calls': opt_chain.calls,
        'puts': opt_chain.puts,
        'current_price': stock.history(period='1d')['Close'].iloc[-1]
    }

# Example: AAPL options
try:
    aapl_options = get_option_chain('AAPL')
    
    print(f"=== {aapl_options['ticker']} OPTIONS ===")
    print(f"Current Price: \${aapl_options['current_price']:.2f}")
    print(f"Expiration: {aapl_options['expiration']}\\n")
    
    # Display ATM calls
    spot = aapl_options['current_price']
    calls = aapl_options['calls']
    
    # Find ATM options (closest to current price)
    calls['distance'] = abs(calls['strike'] - spot)
    atm_calls = calls.nsmallest(5, 'distance')[['strike', 'lastPrice', 'bid', 'ask', 
                                                   'volume', 'openInterest', 'impliedVolatility']]
    
    print("ATM Call Options:")
    print(atm_calls.to_string(index=False))
    
    # Display ATM puts
    puts = aapl_options['puts']
    puts['distance'] = abs(puts['strike'] - spot)
    atm_puts = puts.nsmallest(5, 'distance')[['strike', 'lastPrice', 'bid', 'ask', 
                                                 'volume', 'openInterest', 'impliedVolatility']]
    
    print("\\nATM Put Options:")
    print(atm_puts.to_string(index=False))
    
except Exception as e:
    print(f"Error fetching options: {e}")
    print("Note: Requires internet connection and valid ticker")
\`\`\`

### Strategy Simulator

\`\`\`python
"""
Option Strategy Simulator
"""

class OptionPosition:
    """Represents a single option position"""
    
    def __init__(self, option_type, strike, premium, position='long', quantity=1):
        self.option_type = option_type  # 'call' or 'put'
        self.strike = strike
        self.premium = premium
        self.position = position  # 'long' or 'short'
        self.quantity = quantity
    
    def payoff(self, spot_price):
        """Calculate payoff at given spot price"""
        if self.option_type == 'call':
            intrinsic = max(spot_price - self.strike, 0)
        else:  # put
            intrinsic = max(self.strike - spot_price, 0)
        
        if self.position == 'long':
            return (intrinsic - self.premium) * self.quantity
        else:  # short
            return (self.premium - intrinsic) * self.quantity
    
    def __repr__(self):
        return f"{self.position.title()} {self.quantity} {self.option_type.title()} \${self.strike} @ \${self.premium}"

class OptionStrategy:
    """Combines multiple option positions into a strategy"""
    
    def __init__(self, name):
        self.name = name
        self.positions = []
        self.stock_position = None
    
    def add_option(self, option_type, strike, premium, position='long', quantity=1):
        """Add an option leg"""
        opt = OptionPosition(option_type, strike, premium, position, quantity)
        self.positions.append(opt)
        return self
    
    def add_stock(self, quantity, cost):
        """Add stock position"""
        self.stock_position = {'quantity': quantity, 'cost': cost}
        return self
    
    def calculate_payoff(self, spot_prices):
        """Calculate total strategy payoff"""
        if not isinstance(spot_prices, np.ndarray):
            spot_prices = np.array([spot_prices])
        
        total_payoff = np.zeros_like(spot_prices, dtype=float)
        
        # Add option payoffs
        for position in self.positions:
            for i, spot in enumerate(spot_prices):
                total_payoff[i] += position.payoff(spot)
        
        # Add stock P/L if present
        if self.stock_position:
            stock_pl = (spot_prices - self.stock_position['cost']) * self.stock_position['quantity']
            total_payoff += stock_pl
        
        return total_payoff
    
    def analyze(self, current_price, price_range=None):
        """Analyze strategy"""
        if price_range is None:
            price_range = np.linspace(current_price * 0.7, current_price * 1.3, 100)
        
        payoffs = self.calculate_payoff(price_range)
        
        # Calculate key metrics
        max_profit = payoffs.max()
        max_loss = payoffs.min()
        
        # Find breakevens (where payoff crosses zero)
        breakevens = []
        for i in range(len(payoffs) - 1):
            if (payoffs[i] <= 0 and payoffs[i+1] > 0) or (payoffs[i] >= 0 and payoffs[i+1] < 0):
                # Linear interpolation
                be = price_range[i] + (price_range[i+1] - price_range[i]) * (-payoffs[i]) / (payoffs[i+1] - payoffs[i])
                breakevens.append(be)
        
        return {
            'strategy': self.name,
            'positions': self.positions,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'breakevens': breakevens,
            'price_range': price_range,
            'payoffs': payoffs
        }
    
    def plot(self, current_price, price_range=None):
        """Plot strategy payoff"""
        analysis = self.analyze(current_price, price_range)
        
        plt.figure(figsize=(12, 7))
        plt.plot(analysis['price_range'], analysis['payoffs'], linewidth=2.5, color='darkblue')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.axvline(x=current_price, color='green', linestyle='--', 
                   label=f'Current Price \${current_price:.2f}', alpha=0.7)
        
        # Mark breakevens
        for be in analysis['breakevens']:
            plt.axvline(x=be, color='orange', linestyle=':', alpha=0.7)
            plt.text(be, 0, f'  BE: \${be:.2f}', rotation=90, va='bottom', fontsize=9)
        
        plt.fill_between(analysis['price_range'], 0, analysis['payoffs'], 
                        where=(analysis['payoffs'] > 0), alpha=0.2, color='green', label='Profit')
        plt.fill_between(analysis['price_range'], 0, analysis['payoffs'], 
                        where=(analysis['payoffs'] < 0), alpha=0.2, color='red', label='Loss')
        
        plt.xlabel('Stock Price at Expiration ($)', fontsize=12)
        plt.ylabel('Profit/Loss ($)', fontsize=12)
        plt.title(f'{self.name} Payoff Diagram', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Print analysis
        print(f"\\n=== {self.name.upper()} ANALYSIS ===")
        print(f"\\nPositions:")
        for i, pos in enumerate(self.positions, 1):
            print(f"  {i}. {pos}")
        
        if self.stock_position:
            print(f"  Stock: {self.stock_position['quantity']} shares @ \${self.stock_position['cost']}")
        
        print(f"\\nMax Profit: \${analysis['max_profit']:.2f}")
        print(f"Max Loss: \${analysis['max_loss']:.2f}")
        print(f"Breakevens: {', '.join([f'\${be:.2f}' for be in analysis['breakevens']])}")

# Example: Build and analyze strategies
current = 150

# 1. Covered Call
covered_call = OptionStrategy("Covered Call")
covered_call.add_stock(quantity=100, cost=150)
covered_call.add_option('call', strike=160, premium=3, position='short', quantity=1)
covered_call.plot(current)

# 2. Iron Condor
iron_condor = OptionStrategy("Iron Condor")
iron_condor.add_option('put', strike=135, premium=1, position='long')
iron_condor.add_option('put', strike=140, premium=2.5, position='short')
iron_condor.add_option('call', strike=160, premium=2.5, position='short')
iron_condor.add_option('call', strike=165, premium=1, position='long')
iron_condor.plot(current)

# 3. Long Straddle (volatility play)
straddle = OptionStrategy("Long Straddle")
straddle.add_option('call', strike=150, premium=6, position='long')
straddle.add_option('put', strike=150, premium=5, position='long')
straddle.plot(current)
\`\`\`

---

## Summary

### Key Takeaways

1. **Options provide leverage and flexibility**: Control large positions with small capital
2. **Asymmetric risk/reward**: Calls and puts have different risk profiles
3. **Time decay**: Options lose value as expiration approaches (theta)
4. **Intrinsic vs time value**: Understand what you're paying for
5. **American vs European**: Early exercise considerations
6. **Strategies for every market view**: Bullish, bearish, neutral, volatile
7. **Risk management**: Use options to hedge and protect portfolios

### Trading Considerations

- **Liquidity**: Trade liquid options (high volume, tight bid-ask spread)
- **Volatility**: Implied volatility affects option prices significantly
- **Time decay**: Theta accelerates in final 30 days
- **Transaction costs**: Options have wider spreads than stocks
- **Assignment risk**: Short options can be assigned early
- **Position sizing**: Never risk more than you can afford to lose

### Next Steps

In the next sections, we'll cover:
- **Black-Scholes Model**: How to price options mathematically
- **The Greeks**: Measure and manage option risk (delta, gamma, theta, vega)
- **Advanced strategies**: Volatility trading, complex spreads
- **Risk management**: Position sizing and portfolio hedging with options

Options are powerful tools in quantitative finance. Master the fundamentals here before moving to advanced pricing and trading strategies.
`,
};
