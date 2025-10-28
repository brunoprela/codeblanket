export const coveredCallsProtectivePuts = {
  title: 'Covered Calls and Protective Puts',
  id: 'covered-calls-protective-puts',
  content: `
# Covered Calls and Protective Puts

## Introduction

These are two of the most practical and widely-used options strategies:
- **Covered Call:** Own stock + sell call (income generation strategy)
- **Protective Put:** Own stock + buy put (insurance/hedge strategy)

Both are beginner-friendly, have defined risk profiles, and serve different purposes in portfolio management.

---

## Covered Call Strategy

### Mechanics

**Setup:**1. Own 100 shares of stock (or multiples of 100)
2. Sell 1 call option per 100 shares
3. Typically sell **30-45 days** to expiration
4. Strike: **OTM** (above current price) or ATM

**Example:**
- Own 100 shares of AAPL at $150
- Sell 155-strike call for $3.00 (30 days)
- Collect $300 premium

### Profit/Loss Analysis

\`\`\`python
"""
Covered Call Analysis
"""

import numpy as np
import matplotlib.pyplot as plt

def covered_call_payoff(stock_prices, purchase_price, strike, premium):
    """
    Calculate covered call P&L
    
    Args:
        stock_prices: Array of possible stock prices at expiration
        purchase_price: Original stock purchase price
        strike: Sold call strike
        premium: Premium received from selling call
        
    Returns:
        Total P&L array
    """
    # Stock P&L
    stock_pl = stock_prices - purchase_price
    
    # Short call P&L
    short_call_pl = np.minimum(strike - stock_prices, 0) + premium
    
    # Total P&L
    total_pl = stock_pl + short_call_pl
    
    return total_pl, stock_pl, short_call_pl


# Example
purchase_price = 150
strike = 155
premium = 3

stock_prices = np.linspace(130, 170, 200)
total_pl, stock_pl, call_pl = covered_call_payoff(stock_prices, purchase_price, strike, premium)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# P&L diagram
ax1.plot(stock_prices, total_pl, 'b-', linewidth=2, label='Covered Call')
ax1.plot(stock_prices, stock_pl, 'g--', linewidth=1, label='Stock Only', alpha=0.5)
ax1.axhline(0, color='black', linestyle='-', alpha=0.3)
ax1.axvline(purchase_price, color='gray', linestyle=':', alpha=0.5, label=f'Purchase \${purchase_price}')
ax1.axvline(strike, color='red', linestyle=':', alpha=0.5, label=f'Strike \${strike}')
ax1.set_xlabel('Stock Price at Expiration')
ax1.set_ylabel('Profit / Loss per Share')
ax1.set_title('Covered Call Payoff Diagram')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Returns comparison
scenarios = {
    'Stock at $140': 140,
    'Stock at $150': 150,
    'Stock at $155': 155,
    'Stock at $160': 160,
    'Stock at $170': 170
}

stock_only_returns = []
covered_call_returns = []
labels = []

for scenario, price in scenarios.items():
    stock_return = (price - purchase_price) / purchase_price * 100
    cc_return = ((price if price < strike else strike) - purchase_price + premium) / purchase_price * 100
    
    stock_only_returns.append(stock_return)
    covered_call_returns.append(cc_return)
    labels.append(scenario)

x = np.arange(len(labels))
width = 0.35

ax2.bar(x - width/2, stock_only_returns, width, label='Stock Only', color='green', alpha=0.7)
ax2.bar(x + width/2, covered_call_returns, width, label='Covered Call', color='blue', alpha=0.7)
ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
ax2.set_ylabel('Return (%)')
ax2.set_title('Returns: Stock Only vs Covered Call')
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=45, ha='right')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Print analysis
print("=" * 70)
print("COVERED CALL ANALYSIS")
print("=" * 70)
print(f"\\nPosition: Long 100 shares at \${purchase_price}, Short \${strike} call for \\$\{premium}")
print(f"\\nScenario Analysis:")
for scenario, price in scenarios.items():
    stock_pl = price - purchase_price
    if price <= strike:
        call_pl = premium
        total_pl = stock_pl + premium
        status = "Call expires worthless"
    else:
        call_pl = premium - (price - strike)
        total_pl = strike - purchase_price + premium
        status = f"Stock called away at \${strike}"
    
    print(f"\\n{scenario}:")
    print(f"  Stock P&L: \\$\{stock_pl:.2f}")
    print(f"  Call P&L: \\$\{call_pl:.2f}")
    print(f"  Total P&L: \\$\{total_pl:.2f}")
    print(f"  Return: {(total_pl / purchase_price * 100):.2f}%")
    print(f"  {status}")

# Calculate metrics
print(f"\\n{'─' * 70}")
print("Key Metrics:")
print(f"  Max Profit: \${strike - purchase_price + premium:.2f} (at \\$\{strike}+)")
print(f"  Max Profit %: {((strike - purchase_price + premium) / purchase_price * 100):.2f}%")
print(f"  Downside Protection: \${premium:.2f} (breakeven at \\$\{purchase_price - premium:.2f})")
print(f"  Upside Capped At: \\$\{strike}")
\`\`\`

### When to Use Covered Calls

**Ideal Conditions:**1. **Neutral to slightly bullish** outlook
2. **High IV** (sell rich premium)
3. **Stock you're willing to sell** at strike
4. **Income generation** goal

**Not Ideal:**
- Very bullish (upside capped)
- Very bearish (minimal protection)
- Low IV (small premium not worth it)

---

## Covered Call Variations

### 1. ATM Covered Call
- **Strike = Current stock price**
- **Higher premium** (more income)
- **Higher probability** of assignment
- **Best for neutral outlook**

### 2. OTM Covered Call  
- **Strike > Current price** (e.g., 5-10% OTM)
- **Lower premium** but participation in upside
- **Lower probability** of assignment
- **Best for slightly bullish outlook**

### 3. Monthly Income Strategy

\`\`\`python
"""
Monthly Covered Call Income Strategy
"""

def monthly_covered_call_income(stock_price, annual_dividend_yield, 
                                monthly_premium_pct, months=12):
    """
    Project covered call income over time
    
    Args:
        stock_price: Current stock price
        annual_dividend_yield: Annual dividend yield (e.g., 0.02 for 2%)
        monthly_premium_pct: Monthly premium as % of stock price (e.g., 0.02 for 2%)
        months: Number of months to project
        
    Returns:
        DataFrame with income projections
    """
    import pandas as pd
    
    results = []
    cumulative_income = 0
    
    for month in range(1, months + 1):
        # Monthly covered call premium
        monthly_premium = stock_price * monthly_premium_pct
        
        # Quarterly dividend (assuming quarterly payments)
        monthly_dividend = (stock_price * annual_dividend_yield / 4) if month % 3 == 0 else 0
        
        # Total income
        monthly_total = monthly_premium + monthly_dividend
        cumulative_income += monthly_total
        
        results.append({
            'Month': month,
            'Premium': monthly_premium,
            'Dividend': monthly_dividend,
            'Monthly Total': monthly_total,
            'Cumulative': cumulative_income
        })
    
    df = pd.DataFrame(results)
    return df


# Example: AAPL strategy
stock_price = 150
annual_div_yield = 0.005  # 0.5%
monthly_premium_pct = 0.015  # 1.5% per month (selling 30-day calls)

income_df = monthly_covered_call_income(stock_price, annual_div_yield, monthly_premium_pct)

print("\\nMONTHLY COVERED CALL INCOME PROJECTION")
print("=" * 70)
print(f"Stock: \\$\{stock_price}")
print(f"Strategy: Sell 1.5% OTM calls monthly")
print("\\n", income_df.to_string(index=False))

annual_return = (income_df['Cumulative'].iloc[-1] / stock_price) * 100
print(f"\\nAnnual Return from Premiums + Dividends: {annual_return:.2f}%")
print(f"This is BEFORE any stock appreciation (capped at strike)")
\`\`\`

---

## Protective Put Strategy

### Mechanics

**Setup:**1. Own 100 shares of stock
2. Buy 1 put option per 100 shares
3. Strike: Typically 5-10% OTM (below current price)
4. Expiration: Depends on hedge horizon (30-90 days)

**Purpose:** **Insurance** against downside

**Example:**
- Own 100 shares of TSLA at $200
- Buy 180-strike put for $5 (90 days)
- Cost: $500 for 3 months of protection
- Max loss: $20 stock decline + $5 premium = $25/share

### Payoff Analysis

\`\`\`python
"""
Protective Put Analysis
"""

def protective_put_payoff(stock_prices, purchase_price, put_strike, put_premium):
    """
    Calculate protective put P&L
    
    Components:
    - Long stock
    - Long put (insurance)
    """
    # Stock P&L
    stock_pl = stock_prices - purchase_price
    
    # Long put P&L
    put_pl = np.maximum(put_strike - stock_prices, 0) - put_premium
    
    # Total P&L
    total_pl = stock_pl + put_pl
    
    return total_pl, stock_pl, put_pl


# Example
purchase_price = 200
put_strike = 180
put_premium = 5

stock_prices = np.linspace(150, 250, 200)
total_pl, stock_pl, put_pl = protective_put_payoff(stock_prices, purchase_price, 
                                                    put_strike, put_premium)

plt.figure(figsize=(12, 6))
plt.plot(stock_prices, stock_pl, 'r--', linewidth=2, label='Stock Only', alpha=0.7)
plt.plot(stock_prices, total_pl, 'g-', linewidth=2, label='Stock + Protective Put')
plt.axhline(0, color='black', linestyle='-', alpha=0.3)
plt.axvline(purchase_price, color='gray', linestyle=':', label=f'Purchase \${purchase_price}')
plt.axvline(put_strike, color='orange', linestyle=':', label=f'Put Strike \${put_strike}')

# Highlight floor
plt.axhline(put_strike - purchase_price - put_premium, color='green', 
            linestyle='--', linewidth=1.5, label=f'Max Loss Floor')

plt.xlabel('Stock Price at Expiration')
plt.ylabel('Profit / Loss per Share')
plt.title('Protective Put: Downside Protection')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Analysis
print("=" * 70)
print("PROTECTIVE PUT ANALYSIS")
print("=" * 70)
print(f"\\nPosition: Long 100 shares at \\$\{purchase_price}")
print(f"Protection: Long \${put_strike} put for \\$\{put_premium}")
print(f"\\nScenarios:")

scenarios = {
    'Crash ($150)': 150,
    'Down 10% ($180)': 180,
    'Flat ($200)': 200,
    'Up 10% ($220)': 220,
    'Rally ($250)': 250
}

for scenario, price in scenarios.items():
    stock_pl = price - purchase_price
    put_pl = max(put_strike - price, 0) - put_premium
    total_pl = stock_pl + put_pl
    
    print(f"\\n{scenario}:")
    print(f"  Stock P&L: \\$\{stock_pl:.2f}")
    print(f"  Put P&L: \\$\{put_pl:.2f}")
    print(f"  Total P&L: \\$\{total_pl:.2f}")
    print(f"  Return: {(total_pl / purchase_price * 100):.2f}%")

max_loss = put_strike - purchase_price - put_premium
print(f"\\n{'─' * 70}")
print(f"Max Loss: \\$\{max_loss:.2f} per share (no matter how low stock goes!)")
print(f"Cost of Protection: \\$\{put_premium:.2f}/share for 90 days = {(put_premium/purchase_price*100):.2f}%")
print(f"Annualized Insurance Cost: {(put_premium/purchase_price*4*100):.2f}%")
\`\`\`

### When to Use Protective Puts

**Ideal Conditions:**1. **Hold concentrated position** (can't diversify)
2. **Earnings/event risk** ahead
3. **Market uncertainty** (hedge portfolio)
4. **Lock in gains** (have profits, protect them)

**Trade-offs:**
- **Cost:** Premium reduces returns
- **Decay:** Loses value if stock flat/up
- **Peace of mind:** Worth it for large positions

---

## Covered Call vs Protective Put

| Aspect | Covered Call | Protective Put |
|--------|--------------|----------------|
| **Purpose** | Income generation | Downside protection |
| **Direction** | Neutral to slightly bullish | Bullish but hedged |
| **Cash Flow** | Receive premium (credit) | Pay premium (debit) |
| **Max Profit** | Capped at strike | Unlimited |
| **Max Loss** | Stock to zero minus premium | Limited (strike - stock - premium) |
| **IV Preference** | High IV (sell rich premium) | Low IV (buy cheap insurance) |
| **Adjustments** | Roll up/out if stock rises | Roll down/out if more protection needed |

---

## Combined Strategy: Collar

**Setup:**
- Own stock
- Sell OTM call (e.g., 105)
- Buy OTM put (e.g., 95)
- **Net:** Zero-cost or small credit/debit

**Effect:** Caps upside AND downside

\`\`\`python
"""
Collar Strategy (Covered Call + Protective Put)
"""

def collar_payoff(stock_prices, purchase_price, call_strike, call_premium, 
                  put_strike, put_premium):
    """
    Collar = Long Stock + Short Call + Long Put
    """
    stock_pl = stock_prices - purchase_price
    short_call = -np.maximum(stock_prices - call_strike, 0) + call_premium
    long_put = np.maximum(put_strike - stock_prices, 0) - put_premium
    
    total = stock_pl + short_call + long_put
    return total

# Zero-cost collar example
purchase_price = 100
call_strike = 105
call_premium = 2.50
put_strike = 95
put_premium = 2.50  # Exactly offsets call premium

stock_prices = np.linspace(85, 115, 200)
collar_pl = collar_payoff(stock_prices, purchase_price, call_strike, call_premium,
                          put_strike, put_premium)

plt.figure(figsize=(10, 6))
plt.plot(stock_prices, collar_pl, 'purple', linewidth=2, label='Collar')
plt.axhline(0, color='black', linestyle='-', alpha=0.3)
plt.axhline(5, color='green', linestyle='--', alpha=0.5, label='Max Profit')
plt.axhline(-5, color='red', linestyle='--', alpha=0.5, label='Max Loss')
plt.xlabel('Stock Price')
plt.ylabel('P&L')
plt.title('Zero-Cost Collar: Defined Risk and Reward')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

print(f"Collar Analysis:")
print(f"  Max Profit: \${call_strike - purchase_price} (at \\$\{call_strike}+)")
print(f"  Max Loss: \${purchase_price - put_strike} (at \\$\{put_strike} or below)")
print(f"  Net Cost: \\$\{put_premium - call_premium}")
\`\`\`

---

## Summary

**Covered Calls:**
- Generate income from stock holdings
- Best in high IV environments
- Trade-off: Cap upside potential

**Protective Puts:**
- Insurance against downside
- Best in low IV (cheap insurance)
- Cost: Premium paid

**Collars:**
- Combine both for defined risk/reward
- Often zero-cost or small cost
- Suitable for concentrated positions

These strategies form the foundation of sophisticated portfolio management and are used by institutions and retail traders alike.
`,
};
