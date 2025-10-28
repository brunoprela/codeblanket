export const optionsTradingStrategies = {
  title: 'Options Trading Strategies',
  id: 'options-trading-strategies',
  content: `
# Options Trading Strategies

## Introduction

Options strategies combine calls and puts in various ways to create specific risk/reward profiles. The key is matching the strategy to your:
- **Market outlook** (bullish, bearish, neutral)
- **Volatility view** (expecting IV expansion or contraction)
- **Risk tolerance** (defined vs undefined risk)
- **Time horizon** (short-term vs long-term)

**Core Strategy Categories:**1. **Directional:** Bullish or bearish plays
2. **Volatility:** Profit from volatility changes (up or down)
3. **Income:** Generate premium from time decay
4. **Hedging:** Protect existing positions

---

## Strategy Selection Framework

### Market Outlook Matrix

| Outlook | IV Regime | Best Strategies |
|---------|-----------|-----------------|
| **Bullish** | Low IV | Long calls, call debit spreads |
| **Bullish** | High IV | Short puts, put credit spreads |
| **Bearish** | Low IV | Long puts, put debit spreads |
| **Bearish** | High IV | Short calls (with stock), call credit spreads |
| **Neutral** | Low IV | Long iron condors, calendar spreads |
| **Neutral** | High IV | Short strangles, short iron condors |
| **Volatile** | Low IV | Long straddles/strangles |
| **Stable** | High IV | Short straddles/strangles (with hedges) |

---

## Bullish Strategies

### 1. Long Call (Basic)

**Setup:**
- Buy 1 call option
- Strike: ATM or slightly OTM
- Expiration: 30-60 days

**Payoff:**
- Max profit: Unlimited
- Max loss: Premium paid
- Breakeven: Strike + premium

\`\`\`python
"""
Long Call Analysis
"""

def long_call_payoff(stock_prices, strike, premium):
    """Calculate long call P&L"""
    payoff = np.maximum(stock_prices - strike, 0) - premium
    return payoff

# Example
strike = 100
premium = 5
stock_prices = np.linspace(80, 120, 100)

payoff = long_call_payoff(stock_prices, strike, premium)

plt.figure(figsize=(10, 6))
plt.plot(stock_prices, payoff, 'g-', linewidth=2)
plt.axhline(0, color='black', linestyle='--', alpha=0.3)
plt.axvline(strike, color='red', linestyle='--', alpha=0.3, label=f'Strike \${strike}')
plt.xlabel('Stock Price at Expiration')
plt.ylabel('Profit / Loss')
plt.title('Long Call Payoff Diagram')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

print(f"Max Loss: \\$\{premium}")
print(f"Breakeven: \\$\{strike + premium}")
print(f"Max Profit: Unlimited")
\`\`\`

**When to Use:**
- Bullish on stock
- Low to medium IV
- Want leveraged upside
- Can afford to lose premium

---

### 2. Bull Call Spread

**Setup:**
- Buy 1 call (lower strike)
- Sell 1 call (higher strike)
- Same expiration

**Advantage:** Lower cost than naked call, defined risk

\`\`\`python
"""
Bull Call Spread
"""

def bull_call_spread(stock_prices, long_strike, short_strike, net_debit):
    """Calculate bull call spread P&L"""
    long_call = np.maximum(stock_prices - long_strike, 0)
    short_call = -np.maximum(stock_prices - short_strike, 0)
    payoff = long_call + short_call - net_debit
    return payoff

# Example: 100/105 bull call spread
long_strike = 100
short_strike = 105
long_premium = 5
short_premium = 2
net_debit = long_premium - short_premium  # $3

stock_prices = np.linspace(90, 115, 100)
payoff = bull_call_spread(stock_prices, long_strike, short_strike, net_debit)

plt.figure(figsize=(10, 6))
plt.plot(stock_prices, payoff, 'g-', linewidth=2)
plt.axhline(0, color='black', linestyle='--', alpha=0.3)
plt.fill_between(stock_prices, payoff, 0, where=(payoff > 0), alpha=0.3, color='green')
plt.fill_between(stock_prices, payoff, 0, where=(payoff < 0), alpha=0.3, color='red')
plt.xlabel('Stock Price at Expiration')
plt.ylabel('Profit / Loss')
plt.title('Bull Call Spread Payoff')
plt.grid(True, alpha=0.3)
plt.show()

max_profit = short_strike - long_strike - net_debit
max_loss = net_debit

print(f"Net Debit: \\$\{net_debit}")
print(f"Max Profit: \${max_profit} (at \\$\{short_strike}+)")
print(f"Max Loss: \\$\{net_debit}")
print(f"Breakeven: \\$\{long_strike + net_debit}")
print(f"Return on Risk: {(max_profit / net_debit * 100):.1f}%")
\`\`\`

---

## Bearish Strategies

### 1. Long Put

**Setup:**
- Buy 1 put option
- Strike: ATM or slightly OTM
- Expiration: 30-60 days

**Payoff:**
- Max profit: Strike - premium (if stock → 0)
- Max loss: Premium paid
- Breakeven: Strike - premium

**When to Use:**
- Bearish on stock
- Low to medium IV
- Want leveraged downside
- As portfolio hedge

---

### 2. Bear Put Spread

**Setup:**
- Buy 1 put (higher strike)
- Sell 1 put (lower strike)
- Same expiration

**Advantage:** Lower cost, defined risk/reward

\`\`\`python
"""
Bear Put Spread
"""

def bear_put_spread(stock_prices, long_strike, short_strike, net_debit):
    """Calculate bear put spread P&L"""
    long_put = np.maximum(long_strike - stock_prices, 0)
    short_put = -np.maximum(short_strike - stock_prices, 0)
    payoff = long_put + short_put - net_debit
    return payoff

# Example: 100/95 bear put spread
long_strike = 100
short_strike = 95
net_debit = 2.50

stock_prices = np.linspace(85, 110, 100)
payoff = bear_put_spread(stock_prices, long_strike, short_strike, net_debit)

max_profit = long_strike - short_strike - net_debit
max_loss = net_debit

print(f"Max Profit: \${max_profit} (at \\$\{short_strike} or below)")
print(f"Max Loss: \\$\{net_debit}")
print(f"Breakeven: \\$\{long_strike - net_debit}")
\`\`\`

---

## Neutral Strategies

### 1. Iron Condor

**Setup:**
- Sell 1 OTM put (lower)
- Buy 1 OTM put (even lower) - protection
- Sell 1 OTM call (higher)
- Buy 1 OTM call (even higher) - protection

**Profile:** Profit if stock stays in range

\`\`\`python
"""
Iron Condor Strategy
"""

def iron_condor(stock_prices, put_short_strike, put_long_strike,
                call_short_strike, call_long_strike, net_credit):
    """
    Calculate iron condor P&L
    
    Example: Stock at $100
    - Sell 95 put for $1.50
    - Buy 90 put for $0.50 (protection)
    - Sell 105 call for $1.50
    - Buy 110 call for $0.50 (protection)
    Net credit = $2.00
    """
    # Put spread
    short_put = -np.maximum(put_short_strike - stock_prices, 0)
    long_put = np.maximum(put_long_strike - stock_prices, 0)
    
    # Call spread
    short_call = -np.maximum(stock_prices - call_short_strike, 0)
    long_call = np.maximum(stock_prices - call_long_strike, 0)
    
    # Total P&L
    payoff = short_put + long_put + short_call + long_call + net_credit
    return payoff


# Example
stock_prices = np.linspace(80, 120, 200)

payoff = iron_condor(
    stock_prices,
    put_short_strike=95,
    put_long_strike=90,
    call_short_strike=105,
    call_long_strike=110,
    net_credit=2.0
)

plt.figure(figsize=(12, 6))
plt.plot(stock_prices, payoff, 'b-', linewidth=2)
plt.axhline(0, color='black', linestyle='--', alpha=0.3)
plt.axvline(100, color='gray', linestyle=':', label='Stock Price')
plt.fill_between(stock_prices, payoff, 0, where=(payoff > 0), alpha=0.3, color='green', label='Profit Zone')
plt.fill_between(stock_prices, payoff, 0, where=(payoff < 0), alpha=0.3, color='red', label='Loss Zone')
plt.xlabel('Stock Price at Expiration')
plt.ylabel('Profit / Loss')
plt.title('Iron Condor Payoff')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

print(f"Max Profit: \\$\{net_credit:.2f} (stock between $95-$105)")
print(f"Max Loss: \${5 - net_credit:.2f} (stock below $90 or above $110)")
print(f"Breakevens: \${95 - net_credit:.2f} and \${105 + net_credit:.2f}")
print(f"Probability of Profit: ~70% (if selling 1 SD)")
\`\`\`

**When to Use:**
- Neutral outlook (expect low movement)
- High IV (selling premium)
- Want defined risk
- Prefer high probability of profit

---

### 2. Calendar Spread (Time Spread)

**Setup:**
- Sell near-term option (30 days)
- Buy longer-term option (60-90 days)
- Same strike (usually ATM)

**Profit Driver:** Theta decay of near-term option

\`\`\`python
"""
Calendar Spread Analysis
"""

def calendar_spread_value(S, K, T_short, T_long, sigma, r=0.05):
    """
    Estimate calendar spread value
    Short front month, long back month
    """
    from scipy.stats import norm
    
    # Price short option (expires first)
    short_price = black_scholes_price(S, K, T_short, r, sigma, 'call')
    
    # Price long option (expires later)
    long_price = black_scholes_price(S, K, T_long, r, sigma, 'call')
    
    # Spread value = long - short (paid to enter)
    spread_value = long_price - short_price
    
    return spread_value, short_price, long_price


# Example
S = 100
K = 100
T_short = 30/365
T_long = 60/365
sigma = 0.25

spread_value, short, long_price = calendar_spread_value(S, K, T_short, T_long, sigma)

print(f"\\nCalendar Spread Analysis:")
print(f"  Front month (30d): \\$\{short:.2f}")
print(f"  Back month (60d): \\$\{long_price:.2f}")
print(f"  Net debit: \\$\{spread_value:.2f}")
print(f"\\nProfit potential:")
print(f"  If stock stays at \\$\{K} at expiration:")
print(f"    Front month expires worthless: +\\$\{short:.2f}")
print(f"    Back month retains value: still worth ~\\$\{long_price - 1:.2f}")
print(f"    Estimated profit: $1-2 per share")
\`\`\`

---

## Strategy Comparison

\`\`\`python
"""
Compare Multiple Strategies
"""

def compare_strategies(stock_prices):
    """Generate payoff diagrams for multiple strategies"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    strategies = [
        {
            'name': 'Long Call',
            'payoff': long_call_payoff(stock_prices, 100, 5),
            'color': 'green'
        },
        {
            'name': 'Bull Call Spread',
            'payoff': bull_call_spread(stock_prices, 100, 105, 3),
            'color': 'lightgreen'
        },
        {
            'name': 'Long Put',
            'payoff': long_put_payoff(stock_prices, 100, 5),
            'color': 'red'
        },
        {
            'name': 'Bear Put Spread',
            'payoff': bear_put_spread(stock_prices, 100, 95, 2.5),
            'color': 'salmon'
        },
        {
            'name': 'Iron Condor',
            'payoff': iron_condor(stock_prices, 95, 90, 105, 110, 2),
            'color': 'blue'
        },
        {
            'name': 'Long Straddle',
            'payoff': long_straddle(stock_prices, 100, 10),
            'color': 'purple'
        }
    ]
    
    for idx, strategy in enumerate(strategies):
        ax = axes[idx]
        ax.plot(stock_prices, strategy['payoff'], 
                color=strategy['color'], linewidth=2)
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax.axvline(100, color='gray', linestyle=':', alpha=0.5)
        ax.set_title(strategy['name'], fontsize=12, fontweight='bold')
        ax.set_xlabel('Stock Price')
        ax.set_ylabel('P&L')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

stock_prices = np.linspace(80, 120, 200)
compare_strategies(stock_prices)
\`\`\`

---

## Key Takeaways

**Strategy Selection:**
- **Directional view** → Calls/Puts or spreads
- **Volatility view** → Straddles/Strangles
- **Income generation** → Covered calls, iron condors
- **Hedge** → Protective puts, collars

**Risk Management:**
- Always know max loss before entering
- Use spreads to define risk
- Size positions appropriately
- Don't trade strategies you don't understand

**Matching IV Regime:**
- **Low IV:** Buy options (cheap)
- **High IV:** Sell options (rich premium)
- **Critical:** Check IV Rank, not absolute IV

In the next sections, we'll dive deep into specific strategies: covered calls, protective puts, spreads, and volatility plays.
`,
};
