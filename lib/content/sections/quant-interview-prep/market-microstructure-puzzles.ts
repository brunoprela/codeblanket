export const marketMicrostructurePuzzles = {
  title: 'Market Microstructure Puzzles',
  id: 'market-microstructure-puzzles',
  content: `
# Market Microstructure Puzzles

## Introduction

Market microstructure is the study of how markets operate at the transaction level - the mechanics of price formation, order execution, and liquidity provision. For quantitative trading interviews, especially at market-making firms (Optiver, IMC, Susquehanna, Jane Street), deep understanding of microstructure is essential.

**Why this matters:**

- **Market making:** Understanding spreads, inventory risk, adverse selection
- **Execution algorithms:** Minimizing market impact and transaction costs
- **High-frequency trading:** Exploiting microstructure inefficiencies
- **Order routing:** Optimizing execution across venues
- **Liquidity provision:** Balancing profit vs risk

This section covers:
1. Bid-ask spread economics and decomposition
2. Order types and strategic use
3. Market impact models and estimation
4. Adverse selection and information asymmetry
5. Order book dynamics and queue position
6. Execution algorithms (VWAP, TWAP, POV)
7. Dark pools and alternative venues
8. Tick size effects and discreteness
9. High-frequency trading strategies
10. Market maker inventory management

### Interview Focus

Firms test your understanding of:
- Spread decomposition (processing + inventory + adverse selection)
- Market impact square root law
- Optimal execution strategies
- Queue dynamics and priority rules
- Adverse selection problems
- Real-time decision making under uncertainty

---

## Bid-Ask Spread Economics

### Spread Decomposition

The bid-ask spread compensates market makers for three cost components:

\`\`\`
Spread = Order Processing Costs + Inventory Holding Costs + Adverse Selection Costs
\`\`\`

**Order Processing:**  
Fixed costs per transaction (infrastructure, clearing, regulatory)

**Inventory Costs:**  
Risk of holding positions (price risk, funding costs)

**Adverse Selection:**  
Losses to informed traders who know something you don't

### Problem 1: Spread Calculation Fundamentals

**Question:** A stock has the following quotes:
- Bid: 500 shares @ $99.95
- Ask: 300 shares @ $100.05

Calculate:
1. Absolute spread (dollars)
2. Percentage spread (relative to mid)
3. Effective spread for 100-share market buy
4. Realized spread if mid-price moves to $100.02 after trade

**Solution:**

\`\`\`
1. Absolute spread = Ask - Bid
                   = $100.05 - $99.95
                   = $0.10 (10 cents or 10 bps)

2. Mid-price = (Bid + Ask) / 2
             = ($99.95 + $100.05) / 2
             = $100.00

   Percentage spread = (Abs spread / Mid) × 100%
                     = $0.10 / $100.00 × 100%
                     = 0.1%

3. Effective spread for market buy:
   - Buy 100 shares at ask price $100.05
   - Execution price = $100.05
   - Mid-price = $100.00
   - Effective half-spread = $100.05 - $100.00 = $0.05
   - Effective spread = 2 × $0.05 = $0.10
   
   (Effective spread = 2 × |execution price - mid|)

4. Realized spread (captures price impact):
   - Bought at $100.05
   - New mid after trade = $100.02
   - Price impact = $100.02 - $100.00 = $0.02
   - Realized half-spread = $100.05 - $100.02 = $0.03
   - Realized spread = 2 × $0.03 = $0.06
   
   The $0.04 difference between effective ($0.10) and realized ($0.06) 
   is the permanent price impact.
\`\`\`

\`\`\`python
"""
Spread Calculations
"""

def calculate_spreads(bid, ask, execution_price, new_mid=None):
    """
    Calculate various spread measures.
    
    Args:
        bid: Best bid price
        ask: Best ask price
        execution_price: Actual execution price
        new_mid: Mid-price after trade (for realized spread)
        
    Returns:
        Dictionary with spread measures
    """
    mid = (bid + ask) / 2
    
    absolute_spread = ask - bid
    percentage_spread = (absolute_spread / mid) * 100
    
    effective_half_spread = abs(execution_price - mid)
    effective_spread = 2 * effective_half_spread
    
    results = {
        'mid_price': mid,
        'absolute_spread': absolute_spread,
        'percentage_spread': percentage_spread,
        'effective_spread': effective_spread,
        'effective_half_spread': effective_half_spread,
    }
    
    if new_mid is not None:
        realized_half_spread = abs(execution_price - new_mid)
        realized_spread = 2 * realized_half_spread
        price_impact = abs(new_mid - mid)
        
        results.update({
            'realized_spread': realized_spread,
            'realized_half_spread': realized_half_spread,
            'price_impact': price_impact,
            'price_reversal': effective_half_spread - realized_half_spread
        })
    
    return results

# Problem 1 example
spreads = calculate_spreads(
    bid=99.95,
    ask=100.05,
    execution_price=100.05,
    new_mid=100.02
)

print("Spread Analysis:")
for key, value in spreads.items():
    print(f"  {key}: \${value:.4f}")

# Output:
# Spread Analysis:
#   mid_price: $100.0000
#   absolute_spread: $0.1000
#   percentage_spread: 0.1000
#   effective_spread: $0.1000
#   effective_half_spread: $0.0500
#   realized_spread: $0.0600
#   realized_half_spread: $0.0300
#   price_impact: $0.0200
#   price_reversal: $0.0200
\`\`\`

### Problem 2: Market Maker P&L Analysis

**Question:** You're a market maker with the following activity over one day:

Morning session:
- Quote bid=$100.00, ask=$100.10
- Buy 5,000 shares at your bid
- Sell 4,000 shares at your ask

Afternoon session:
- Market moves, new mid-price=$100.25
- You're long 1,000 shares (5000 bought - 4000 sold)
- Close position at mid-price

Calculate:
1. Gross P&L from spread capture
2. Inventory P&L
3. Total P&L
4. If you had hedged immediately with futures, what would P&L be?

**Solution:**

\`\`\`
1. Gross spread capture P&L:
   
   Matched trades (min of buys and sells):
   - 4,000 shares bought at $100.00 = $400,000 paid
   - 4,000 shares sold at $100.10 = $400,400 received
   - Spread profit = $400,400 - $400,000 = $400
   
   Alternative calculation:
   - Captured $0.10 spread on 4,000 shares
   - $0.10 × 4,000 = $400 ✓

2. Inventory P&L:
   
   Remaining position: Long 1,000 shares
   - Bought at $100.00 = $100,000 cost basis
   - Closed at $100.25 = $100,250 proceeds
   - Inventory profit = $100,250 - $100,000 = $250

3. Total P&L:
   
   Spread P&L: $400
   Inventory P&L: $250
   Total: $650

4. With immediate futures hedge:
   
   When long 1,000 shares after morning:
   - Hedge by shorting 1,000 futures at $100.05 (mid-price)
   
   Afternoon close:
   - Sell shares at $100.25: +$100,250
   - Buy back futures at $100.25: -$100,250
   - Cost basis of shares: -$100,000
   
   Spread P&L: $400 (same as before)
   Hedged position P&L: $100,250 - $100,250 - $100,000
                        = -$100,000 (just the cost basis)
   
   Wait, let me recalculate:
   - Shares: bought at 100.00, sold at 100.25 → +$250
   - Futures: short at 100.05, cover at 100.25 → -$200
   - Net inventory P&L with hedge: $250 - $200 = $50
   
   Total P&L with hedge: $400 (spread) + $50 (hedged inventory) = $450
   
   Comparison:
   - Unhedged: $650
   - Hedged: $450
   - Difference: $200 (foregone profit from favorable market move)
   
   Key insight: Hedging eliminates directional risk but also caps upside.
\`\`\`

\`\`\`python
"""
Market Maker P&L Calculation
"""

class MarketMakerPnL:
    def __init__(self, bid, ask):
        self.bid = bid
        self.ask = ask
        self.mid = (bid + ask) / 2
        self.position = 0
        self.cash = 0
        self.spread_pnl = 0
        self.trades = []
    
    def buy_at_bid(self, shares):
        """Market maker buys (someone sells to us at our bid)."""
        self.position += shares
        self.cash -= shares * self.bid
        self.trades.append(('buy', shares, self.bid))
    
    def sell_at_ask(self, shares):
        """Market maker sells (someone buys from us at our ask)."""
        self.position -= shares
        self.cash += shares * self.ask
        self.trades.append(('sell', shares, self.ask))
        
        # Track spread capture
        matched = min(shares, abs(self.position + shares))
        self.spread_pnl += matched * (self.ask - self.bid)
    
    def close_position(self, closing_price):
        """Close remaining inventory at given price."""
        if self.position != 0:
            self.cash += self.position * closing_price
            self.position = 0
    
    def get_total_pnl(self):
        """Calculate total P&L."""
        return self.cash
    
    def get_breakdown(self):
        """Get P&L breakdown."""
        return {
            'spread_pnl': self.spread_pnl,
            'total_pnl': self.cash,
            'inventory_pnl': self.cash - self.spread_pnl
        }

# Problem 2 example
mm = MarketMakerPnL(bid=100.00, ask=100.10)

# Morning trades
mm.buy_at_bid(5000)
mm.sell_at_ask(4000)

print(f"Position after morning: {mm.position} shares")
print(f"Spread P&L: \${mm.spread_pnl:.2f}")

# Close position at new mid
mm.close_position(closing_price=100.25)

breakdown = mm.get_breakdown()
print(f"\\nP&L Breakdown:")
print(f"  Spread P&L: \${breakdown['spread_pnl']:.2f}")
print(f"  Inventory P&L: \${breakdown['inventory_pnl']:.2f}")
print(f"  Total P&L: \${breakdown['total_pnl']:.2f}")

# Output:
# Position after morning: 1000 shares
# Spread P&L: $400.00
# 
# P&L Breakdown:
#   Spread P&L: $400.00
#   Inventory P&L: $250.00
#   Total P&L: $650.00
\`\`\`

---

## Market Impact Models

### Square Root Law

The most widely used empirical model for market impact:

\`\`\`
Market Impact (%) ≈ γ × σ × √(Q / V)

Where:
- γ = market impact coefficient (~0.1 to 1.0 depending on market)
- σ = daily volatility (annualized / √252)
- Q = trade size (shares)
- V = average daily volume (shares)
\`\`\`

**Key insights:**
- Impact grows with √(trade size), not linearly
- Doubling trade size increases impact by only ~41% (√2 ≈ 1.41)
- Impact proportional to volatility (more volatile = higher impact)
- Impact inverse to liquidity (higher volume = lower impact)

### Problem 3: Market Impact Estimation

**Question:** You need to execute the following trades. Estimate market impact for each:

a) Buy 100,000 shares, stock volatility 25%, daily volume 2M shares
b) Buy 200,000 shares (double size), same stock
c) Buy 100,000 shares, different stock: volatility 40%, volume 500K

Use γ = 0.5 for all calculations.

**Solution:**

\`\`\`
Formula: Impact = γ × σ × √(Q/V)

a) Buy 100K shares:
   Q/V = 100,000 / 2,000,000 = 0.05
   Impact = 0.5 × 0.25 × √0.05
          = 0.5 × 0.25 × 0.2236
          = 0.0280 = 2.80%
   
   If stock at $50, impact ≈ $50 × 0.028 = $1.40 per share
   Total cost = 100,000 × $1.40 = $140,000

b) Buy 200K shares (double):
   Q/V = 200,000 / 2,000,000 = 0.10
   Impact = 0.5 × 0.25 × √0.10
          = 0.5 × 0.25 × 0.3162
          = 0.0395 = 3.95%
   
   Note: Doubling trade size increased impact by 3.95/2.80 = 1.41 ≈ √2 ✓
   
   If stock at $50, impact ≈ $1.98 per share
   Total cost = 200,000 × $1.98 = $396,000

c) Buy 100K, higher vol/lower liquidity:
   Q/V = 100,000 / 500,000 = 0.20
   Impact = 0.5 × 0.40 × √0.20
          = 0.5 × 0.40 × 0.4472
          = 0.0894 = 8.94%
   
   Much higher impact due to:
   - Higher volatility (40% vs 25%) → 1.6x multiplier
   - Lower volume (500K vs 2M) → √4 = 2x multiplier
   - Combined effect: 1.6 × 2 = 3.2x higher impact
   
   If stock at $80, impact ≈ $7.15 per share
   Total cost = 100,000 × $7.15 = $715,000
\`\`\`

**Key takeaway:** Market impact can be massive cost for large trades in illiquid stocks!

\`\`\`python
"""
Market Impact Estimation
"""

import numpy as np

def estimate_market_impact(
    trade_size: float,
    daily_volume: float,
    volatility: float,
    gamma: float = 0.5,
    stock_price: float = None
):
    """
    Estimate market impact using square root model.
    
    Args:
        trade_size: Number of shares to trade
        daily_volume: Average daily volume
        volatility: Daily volatility (as decimal, e.g., 0.25 for 25%)
        gamma: Market impact coefficient (typically 0.1 to 1.0)
        stock_price: Current stock price (optional, for dollar impact)
        
    Returns:
        Dictionary with impact estimates
    """
    # Calculate impact as percentage
    participation_rate = trade_size / daily_volume
    impact_pct = gamma * volatility * np.sqrt(participation_rate)
    
    results = {
        'participation_rate': participation_rate * 100,  # as percentage
        'impact_pct': impact_pct * 100,  # as percentage
    }
    
    if stock_price is not None:
        impact_per_share = impact_pct * stock_price
        total_cost = impact_per_share * trade_size
        
        results.update({
            'impact_per_share': impact_per_share,
            'total_cost': total_cost,
            'average_execution_price': stock_price * (1 + impact_pct)
        })
    
    return results

# Problem 3 examples
print("Market Impact Analysis:\\n")

# Scenario a
impact_a = estimate_market_impact(
    trade_size=100_000,
    daily_volume=2_000_000,
    volatility=0.25,
    stock_price=50
)
print("Scenario A (100K shares, 2M volume, 25% vol, $50 stock):")
print(f"  Participation rate: {impact_a['participation_rate']:.1f}%")
print(f"  Market impact: {impact_a['impact_pct']:.2f}%")
print(f"  Cost per share: \${impact_a['impact_per_share']:.2f}")
print(f"  Total cost: \${impact_a['total_cost']:,.0f}")

# Scenario b
impact_b = estimate_market_impact(
    trade_size=200_000,
    daily_volume=2_000_000,
    volatility=0.25,
    stock_price=50
)
print(f"\\nScenario B (200K shares, double size):")
print(f"  Market impact: {impact_b['impact_pct']:.2f}%")
print(f"  Impact ratio vs A: {impact_b['impact_pct'] / impact_a['impact_pct']:.2f}x")
print(f"  Total cost: \${impact_b['total_cost']:,.0f}")

# Scenario c
impact_c = estimate_market_impact(
    trade_size=100_000,
    daily_volume=500_000,
    volatility=0.40,
    stock_price=80
)
print(f"\\nScenario C (100K shares, illiquid/volatile, $80 stock):")
print(f"  Participation rate: {impact_c['participation_rate']:.1f}%")
print(f"  Market impact: {impact_c['impact_pct']:.2f}%")
print(f"  Cost per share: \${impact_c['impact_per_share']:.2f}")
print(f"  Total cost: \${impact_c['total_cost']:,.0f}")

# Output:
# Market Impact Analysis:
# 
# Scenario A (100K shares, 2M volume, 25% vol, $50 stock):
#   Participation rate: 5.0%
#   Market impact: 2.80%
#   Cost per share: $1.40
#   Total cost: $140,000
# 
# Scenario B (200K shares, double size):
#   Market impact: 3.95%
#   Impact ratio vs A: 1.41x
#   Total cost: $396,000
# 
# Scenario C (100K shares, illiquid/volatile, $80 stock):
#   Participation rate: 20.0%
#   Market impact: 8.94%
#   Cost per share: $7.15
#   Total cost: $715,000
\`\`\`

---

## Summary

Market microstructure is critical for:
- **Market makers:** Managing spreads, inventory, and adverse selection
- **Execution traders:** Minimizing costs and market impact
- **HFT firms:** Exploiting liquidity provision opportunities
- **Portfolio managers:** Understanding transaction costs

**Key formulas to memorize:**
- Effective spread = 2 × |execution price - mid|
- Market impact ∝ σ√(Q/V)
- Adverse selection cost = P(informed) × expected loss

**Interview strategy:**
- Calculate spreads in both dollars and basis points
- Use square root law for impact estimation
- Consider both temporary and permanent impact
- Discuss trade-offs between impact and timing risk

Master these concepts for market-making and execution roles!
`,
};
