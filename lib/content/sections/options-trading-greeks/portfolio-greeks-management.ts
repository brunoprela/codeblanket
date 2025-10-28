export const portfolioGreeksManagement = {
  title: 'Options Greeks Portfolio Management',
  id: 'portfolio-greeks-management',
  content: `
# Options Greeks Portfolio Management

## Introduction

Managing a **portfolio of options** requires aggregating and managing Greeks across all positions. Professional traders don't look at individual trades - they manage **portfolio-level risk**.

**Key Concepts:**
- **Aggregate Greeks** across all positions
- **Risk limits** for delta, gamma, theta, vega
- **Stress testing** and scenario analysis
- **Hedging** portfolio Greeks
- **P&L attribution** to each Greek

This is how institutions manage options books worth millions or billions.

---

## Portfolio Delta Management

### Concept

**Portfolio delta** = Sum of all position deltas.

**Target:** Usually close to zero (market-neutral) or small directional bias.

\`\`\`python
"""
Portfolio Delta Management System
"""

import pandas as pd
import numpy as np

class OptionsPortfolio:
    def __init__(self):
        self.positions = []
    
    def add_position(self, position):
        """Add option position to portfolio"""
        self.positions.append(position)
    
    def calculate_portfolio_greeks(self):
        """Calculate aggregate Greeks"""
        total_delta = 0
        total_gamma = 0
        total_theta = 0
        total_vega = 0
        total_rho = 0
        
        for pos in self.positions:
            # Multiply by quantity (contracts × 100)
            multiplier = pos['quantity'] * 100
            
            total_delta += pos['delta'] * multiplier
            total_gamma += pos['gamma'] * multiplier
            total_theta += pos['theta'] * multiplier
            total_vega += pos['vega'] * multiplier
            total_rho += pos['rho'] * multiplier
        
        return {
            'delta': total_delta,
            'gamma': total_gamma,
            'theta': total_theta,
            'vega': total_vega,
            'rho': total_rho
        }
    
    def get_delta_exposure(self, stock_price):
        """Calculate dollar delta exposure"""
        greeks = self.calculate_portfolio_greeks()
        dollar_delta = greeks['delta'] * stock_price
        return dollar_delta
    
    def get_delta_hedge_required(self, target_delta=0):
        """Calculate shares needed to hedge to target delta"""
        greeks = self.calculate_portfolio_greeks()
        current_delta = greeks['delta']
        hedge_shares = target_delta - current_delta
        return hedge_shares
    
    def display_portfolio_summary(self):
        """Display portfolio risk metrics"""
        greeks = self.calculate_portfolio_greeks()
        
        print("=" * 70)
        print("PORTFOLIO GREEKS SUMMARY")
        print("=" * 70)
        print(f"\\nNet Delta: {greeks['delta']:,.0f} shares")
        print(f"  (Equivalent to {abs(greeks['delta']):,.0f} shares {'long' if greeks['delta'] > 0 else 'short'})")
        print(f"\\nNet Gamma: {greeks['gamma']:,.2f}")
        print(f"  (Portfolio delta changes by {greeks['gamma']:,.2f} for $1 stock move)")
        print(f"\\nNet Theta: \\$\{greeks['theta']:,.2f}
} per day")
print(f"  (Portfolio {'gains' if greeks['theta'] > 0 else 'loses'} \\$\{abs(greeks['theta']):,.2f} daily from time decay)")
print(f"\\nNet Vega: \\$\{greeks['vega']:,.2f} per 1% IV")
print(f"  (Portfolio {'gains' if greeks['vega'] > 0 else 'loses'} \\$\{abs(greeks['vega']):,.2f} per 1% IV move)")
print(f"\\nNet Rho: \\$\{greeks['rho']:,.2f} per 1% rate change")
        
        # Risk assessment
print(f"\\n{'─' * 70}")
print("RISK ASSESSMENT:")

if abs(greeks['delta']) > 10000:
    print(f"  ⚠️  HIGH DELTA EXPOSURE: {greeks['delta']:,.0f} shares")
print(f"     Consider hedging to reduce directional risk")
        else:
print(f"  ✓  Delta exposure acceptable: {greeks['delta']:,.0f} shares")

if abs(greeks['gamma']) > 500:
    print(f"  ⚠️  HIGH GAMMA: {greeks['gamma']:,.0f}")
print(f"     Delta will change rapidly with stock moves")
        else:
print(f"  ✓  Gamma manageable: {greeks['gamma']:,.0f}")

if abs(greeks['theta']) > 5000:
    print(f"  ⚠️  HIGH THETA: \\$\{greeks['theta']:,.0f}/day")
print(f"     Significant daily P&L from time decay")
        else:
print(f"  ✓  Theta acceptable: \\$\{greeks['theta']:,.0f}/day")


# Example Portfolio
portfolio = OptionsPortfolio()

# Position 1: Long 10 SPY 450 calls
portfolio.add_position({
    'symbol': 'SPY',
    'type': 'call',
    'strike': 450,
    'quantity': 10,  # contracts
    'delta': 0.60,
    'gamma': 0.015,
    'theta': -15,
    'vega': 25,
    'rho': 18
})

# Position 2: Short 5 SPY 470 calls(covered call)
portfolio.add_position({
    'symbol': 'SPY',
    'type': 'call',
    'strike': 470,
    'quantity': -5,
    'delta': 0.35,
    'gamma': 0.012,
    'theta': -10,
    'vega': 18,
    'rho': 12
})

# Position 3: Long 15 SPY 430 puts(protection)
portfolio.add_position({
    'symbol': 'SPY',
    'type': 'put',
    'strike': 430,
    'quantity': 15,
    'delta': -0.25,
    'gamma': 0.013,
    'theta': -12,
    'vega': 20,
    'rho': -15
})

# Display summary
portfolio.display_portfolio_summary()

# Calculate hedge
spy_price = 450
hedge_shares = portfolio.get_delta_hedge_required(target_delta = 0)

print(f"\\n{'─' * 70}")
print(f"DELTA HEDGING:")
print(f"  To achieve delta-neutral, need to {'buy' if hedge_shares > 0 else 'sell'} {abs(hedge_shares):,.0f} SPY shares")
print(f"  At $450/share, requires \\$\{abs(hedge_shares * spy_price):,.0f} in capital")
\`\`\`

---

## Portfolio Gamma Management

### High Gamma Scenarios

**Gamma** indicates how quickly delta changes.

**High gamma:**
- **Benefit:** Profit from rehedging (gamma scalping)
- **Risk:** Frequent rehedging required (transaction costs)

\`\`\`python
"""
Gamma Scalping P&L Simulation
"""

def simulate_gamma_scalping_pnl(initial_stock, days, gamma, daily_vol):
    """
    Simulate P&L from gamma scalping
    
    Args:
        initial_stock: Starting stock price
        days: Number of days
        gamma: Portfolio gamma
        daily_vol: Daily volatility (e.g., 0.01 for 1%)
    """
    stock = initial_stock
    cumulative_pnl = 0
    daily_pnls = []
    
    for day in range(days):
        # Random stock move
        move = np.random.normal(0, daily_vol) * stock
        new_stock = stock + move
        
        # Gamma P&L = 0.5 × Gamma × (Stock Move)²
        gamma_pnl = 0.5 * gamma * (move ** 2)
        
        cumulative_pnl += gamma_pnl
        daily_pnls.append({
            'day': day,
            'stock': new_stock,
            'move': move,
            'gamma_pnl': gamma_pnl,
            'cumulative_pnl': cumulative_pnl
        })
        
        stock = new_stock
    
    df = pd.DataFrame(daily_pnls)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Stock price
    ax1.plot(df['day'], df['stock'], linewidth=2)
    ax1.set_ylabel('Stock Price')
    ax1.set_title('Stock Price Path')
    ax1.grid(True, alpha=0.3)
    
    # Cumulative P&L
    ax2.plot(df['day'], df['cumulative_pnl'], 'g-', linewidth=2)
    ax2.set_xlabel('Day')
    ax2.set_ylabel('Cumulative Gamma P&L ($)')
    ax2.set_title('Gamma Scalping Profit')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=" * 70)
    print("GAMMA SCALPING SIMULATION")
    print("=" * 70)
    print(f"\\nGamma: {gamma}")
    print(f"Days: {days}")
    print(f"Daily Volatility: {daily_vol*100:.1f}%")
    print(f"\\nResults:")
    print(f"  Starting Stock: \\$\{initial_stock:.2f}")
print(f"  Ending Stock: \\$\{stock:.2f}")
print(f"  Total Gamma P&L: \\$\{cumulative_pnl:,.2f}")
print(f"  Average Daily P&L: \\$\{cumulative_pnl/days:.2f}")

return df

# Simulate
np.random.seed(42)
df = simulate_gamma_scalping_pnl(
    initial_stock = 450,
    days = 30,
    gamma = 500,  # High gamma portfolio
    daily_vol = 0.015  # 1.5 % daily vol
)
\`\`\`

---

## Portfolio Theta Management

### Daily P&L from Theta

**Theta** represents daily time decay profit/loss.

\`\`\`python
"""
Theta Decay Tracking
"""

def project_theta_pnl(portfolio_theta, days):
    """
    Project P&L from theta decay
    
    Note: Theta is not linear - accelerates near expiration
    """
    daily_pnls = []
    cumulative = 0
    
    for day in range(1, days + 1):
        # Theta accelerates as expiration approaches
        # Approximate with sqrt decay
        time_factor = np.sqrt((days - day + 1) / days)
        daily_theta = portfolio_theta * time_factor
        
        cumulative += daily_theta
        daily_pnls.append({
            'day': day,
            'daily_theta': daily_theta,
            'cumulative': cumulative
        })
    
    df = pd.DataFrame(daily_pnls)
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df['day'], df['cumulative'], 'r-', linewidth=2)
    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.xlabel('Days')
    plt.ylabel('Cumulative Theta P&L ($)')
    plt.title('Theta Decay Over Time')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("=" * 70)
    print("THETA DECAY PROJECTION")
    print("=" * 70)
    print(f"\\nPortfolio Theta: \\$\{portfolio_theta:,.2f}/day")
print(f"Days Projected: {days}")
print(f"\\nProjected P&L:")
for milestone in [7, 14, 21, 30]:
    if milestone <= days:
        pnl = df[df['day'] == milestone]['cumulative'].values[0]
print(f"  {milestone} days: \\$\{pnl:,.2f}")

total_pnl = df['cumulative'].iloc[-1]
print(f"\\nTotal Theta P&L: \\$\{total_pnl:,.2f}")

if portfolio_theta > 0:
    print(f"\\n✓ Positive theta: Portfolio gains \\$\{abs(total_pnl):,.0f} from time decay")
else:
print(f"\\n⚠️  Negative theta: Portfolio loses \\$\{abs(total_pnl):,.0f} to time decay")

return df

# Example: Short premium portfolio
df_theta = project_theta_pnl(portfolio_theta = 500, days = 30)
\`\`\`

---

## Portfolio Vega Management

### IV Scenario Analysis

**Vega** measures sensitivity to implied volatility changes.

\`\`\`python
"""
Vega Exposure and IV Scenarios
"""

def analyze_vega_scenarios(portfolio_vega, current_pnl=0):
    """
    Analyze P&L under different IV scenarios
    """
    scenarios = {
        'IV Crash (-10%)': -10,
        'IV Drop (-5%)': -5,
        'Flat IV': 0,
        'IV Rise (+5%)': 5,
        'IV Spike (+10%)': 10,
        'Extreme Spike (+20%)': 20
    }
    
    results = []
    
    print("=" * 70)
    print("VEGA SCENARIO ANALYSIS")
    print("=" * 70)
    print(f"\\nPortfolio Vega: \\$\{portfolio_vega:,.2f} per 1 % IV")
print(f"Current P&L: \\$\{current_pnl:,.2f}")
print(f"\\nIV Scenarios:")
print(f"  {'Scenario':<25} {'IV Change':>12} {'P&L Impact':>15} {'New P&L':>15}")
print("  " + "─" * 70)

for scenario, iv_change in scenarios.items():
    pnl_impact = portfolio_vega * iv_change
new_pnl = current_pnl + pnl_impact

results.append({
    'scenario': scenario,
    'iv_change': iv_change,
    'pnl_impact': pnl_impact,
    'new_pnl': new_pnl
})

color = '🟢' if pnl_impact > 0 else '🔴' if pnl_impact < 0 else '⚪'
print(f"  {scenario:<25} {iv_change:>+11}% {color} \${pnl_impact:>13,.0f} \\$\{new_pnl:>14,.0f}")

df = pd.DataFrame(results)
    
    # Plot
plt.figure(figsize = (12, 6))
colors = ['red' if x < 0 else 'green' if x > 0 else 'gray' for x in df['pnl_impact']]
plt.bar(df['scenario'], df['pnl_impact'], color = colors, alpha = 0.7)
plt.axhline(0, color = 'black', linestyle = '-', linewidth = 0.5)
plt.xlabel('Scenario')
plt.ylabel('P&L Impact ($)')
plt.title('Portfolio P&L Under Different IV Scenarios')
plt.xticks(rotation = 45, ha = 'right')
plt.grid(True, alpha = 0.3, axis = 'y')
plt.tight_layout()
plt.show()
    
    # Risk assessment
print(f"\\n{'─' * 70}")
if portfolio_vega > 0:
    print(f"✓ LONG VEGA: Portfolio benefits from IV increases")
print(f"  Best case (+20% IV): \\$\{df['pnl_impact'].max():,.0f}")
print(f"  Worst case (-10% IV): \\$\{df['pnl_impact'].min():,.0f}")
    else:
print(f"⚠️  SHORT VEGA: Portfolio hurt by IV increases")
print(f"  Best case (-10% IV): \\$\{df['pnl_impact'].min():,.0f}")
print(f"  Worst case (+20% IV): \\$\{df['pnl_impact'].max():,.0f}")

return df

# Example: Long vega portfolio
df_vega = analyze_vega_scenarios(portfolio_vega = 15000, current_pnl = 5000)
\`\`\`

---

## Stress Testing

### Multi-Factor Stress Tests

Test portfolio under extreme scenarios.

\`\`\`python
"""
Comprehensive Stress Testing
"""

def stress_test_portfolio(portfolio_greeks, current_price, current_pnl=0):
    """
    Stress test portfolio under various extreme scenarios
    """
    scenarios = [
        {'name': 'Normal Day', 'stock_pct': 0, 'iv_change': 0},
        {'name': 'Small Up', 'stock_pct': 2, 'iv_change': -1},
        {'name': 'Small Down', 'stock_pct': -2, 'iv_change': 1},
        {'name': 'Large Up', 'stock_pct': 5, 'iv_change': -3},
        {'name': 'Large Down', 'stock_pct': -5, 'iv_change': 3},
        {'name': 'Rally', 'stock_pct': 10, 'iv_change': -5},
        {'name': 'Crash', 'stock_pct': -10, 'iv_change': 10},
        {'name': 'Flash Crash', 'stock_pct': -20, 'iv_change': 20},
        {'name': 'Black Swan', 'stock_pct': -30, 'iv_change': 30},
    ]
    
    results = []
    
    print("=" * 80)
    print("PORTFOLIO STRESS TEST")
    print("=" * 80)
    print(f"\\nCurrent Portfolio Greeks:")
    print(f"  Delta: {portfolio_greeks['delta']:,.0f}")
    print(f"  Gamma: {portfolio_greeks['gamma']:,.2f}")
    print(f"  Theta: \\$\{portfolio_greeks['theta']:,.2f}")
print(f"  Vega: \\$\{portfolio_greeks['vega']:,.2f}")
print(f"\\nStress Scenarios:")

for scenario in scenarios:
    stock_move_pct = scenario['stock_pct']
iv_change = scenario['iv_change']
        
        # Calculate P & L components
stock_move = current_price * (stock_move_pct / 100)
        
        # Delta P & L
delta_pnl = portfolio_greeks['delta'] * stock_move
        
        # Gamma P & L(0.5 × gamma × move²)
gamma_pnl = 0.5 * portfolio_greeks['gamma'] * (stock_move ** 2)
        
        # Vega P & L
vega_pnl = portfolio_greeks['vega'] * iv_change
        
        # Theta P & L(assuming 1 day)
theta_pnl = portfolio_greeks['theta']
        
        # Total P & L
total_pnl = delta_pnl + gamma_pnl + vega_pnl + theta_pnl
new_total_pnl = current_pnl + total_pnl

results.append({
    'scenario': scenario['name'],
    'stock_pct': stock_move_pct,
    'iv_change': iv_change,
    'delta_pnl': delta_pnl,
    'gamma_pnl': gamma_pnl,
    'vega_pnl': vega_pnl,
    'theta_pnl': theta_pnl,
    'total_pnl': total_pnl,
    'new_total': new_total_pnl
})

df = pd.DataFrame(results)
    
    # Display
print(f"\\n{'Scenario':<15} {'Stock':>8} {'IV':>6} {'Delta':>10} {'Gamma':>10} {'Vega':>10} {'Total':>12}")
print("─" * 80)

for _, row in df.iterrows():
    print(f"{row['scenario']:<15} {row['stock_pct']:>+7}% {row['iv_change']:>+5} "
              f"\${row['delta_pnl']:>9,.0f} \${row['gamma_pnl']:>9,.0f} "
              f"\${row['vega_pnl']:>9,.0f} \${row['total_pnl']:>11,.0f}")
    
    # Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 6))
    
    # P & L by scenario
colors = ['green' if x > 0 else 'red' for x in df['total_pnl']]
ax1.barh(df['scenario'], df['total_pnl'], color = colors, alpha = 0.7)
ax1.axvline(0, color = 'black', linestyle = '-', linewidth = 1)
ax1.set_xlabel('P&L ($)')
ax1.set_title('Portfolio P&L by Stress Scenario')
ax1.grid(True, alpha = 0.3, axis = 'x')
    
    # P & L attribution
worst_case = df.loc[df['total_pnl'].idxmin()]
attribution = pd.DataFrame({
    'Greek': ['Delta', 'Gamma', 'Vega', 'Theta'],
    'P&L': [worst_case['delta_pnl'], worst_case['gamma_pnl'],
    worst_case['vega_pnl'], worst_case['theta_pnl']]
})
colors_attr = ['red' if x < 0 else 'green' for x in attribution['P&L']]
ax2.bar(attribution['Greek'], attribution['P&L'], color = colors_attr, alpha = 0.7)
ax2.axhline(0, color = 'black', linestyle = '-', linewidth = 1)
ax2.set_ylabel('P&L ($)')
ax2.set_title(f'P&L Attribution: Worst Case ({worst_case["scenario"]})')
ax2.grid(True, alpha = 0.3, axis = 'y')

plt.tight_layout()
plt.show()
    
    # Risk summary
max_loss = df['total_pnl'].min()
max_gain = df['total_pnl'].max()

print(f"\\n{'─' * 80}")
print("RISK SUMMARY:")
print(f"  Best Case: \\$\{max_gain:,.0f} ({df.loc[df['total_pnl'].idxmax(), 'scenario']})")
print(f"  Worst Case: \\$\{max_loss:,.0f} ({df.loc[df['total_pnl'].idxmin(), 'scenario']})")
print(f"  Range: \\$\{max_gain - max_loss:,.0f}")

if max_loss < -50000:
    print(f"\\n⚠️  CRITICAL: Worst case loss > $50K")
print(f"     Consider reducing position sizes or adding hedges")

return df

# Example
portfolio_greeks = {
    'delta': 5000,
    'gamma': 300,
    'theta': 500,
    'vega': 8000
}

df_stress = stress_test_portfolio(portfolio_greeks, current_price = 450, current_pnl = 10000)
\`\`\`

---

## Summary

**Portfolio Greeks Management:**
- **Aggregate** all positions to portfolio level
- **Monitor** delta, gamma, theta, vega continuously
- **Set limits** for each Greek
- **Hedge** when limits breached
- **Stress test** regularly

**Professional Risk Management:**
- Daily Greeks reporting
- Real-time position monitoring
- Automated alerts for limit breaches
- Regular stress tests and scenario analysis
- P&L attribution by Greek

This is the foundation of institutional-level options trading.
`,
};
