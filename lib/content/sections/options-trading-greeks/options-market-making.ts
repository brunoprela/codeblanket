export const optionsMarketMaking = {
  title: 'Options Market Making',
  id: 'options-market-making',
  content: `
# Options Market Making

## Introduction

**Market makers** provide liquidity by continuously quoting bid/ask prices for options. They profit from:
- **Bid-ask spread** (buy low, sell high)
- **Inventory management** (hedging positions)
- **Volatility arbitrage** (edge on IV)

This is **advanced** territory - institutional-level strategy requiring sophisticated systems and risk management.

---

## Market Maker Role

### Core Function

**Provide liquidity:**
- Quote bid (willing to buy) and ask (willing to sell)
- Fill orders from retail/institutional traders
- Profit from spread (not directional moves)

**Example:**
- Stock at $100
- MM quotes: Bid $4.90 / Ask $5.10 for 100-strike call
- Spread = $0.20 per share = $20 per contract

---

## Delta Hedging

### Concept

Market makers are **delta-neutral** - hedge stock risk immediately.

\`\`\`python
"""
Delta Hedging Simulation
"""

class MarketMaker:
    def __init__(self, capital=1000000):
        self.capital = capital
        self.option_position = 0  # +Long, -Short
        self.stock_position = 0
        self.pnl = 0
    
    def sell_call(self, contracts, delta, premium):
        """Sell call options and delta hedge"""
        # Sell calls (short position)
        self.option_position -= contracts
        
        # Receive premium
        self.capital += premium * contracts * 100
        
        # Delta hedge: Buy stock to offset
        shares_to_buy = contracts * 100 * delta
        self.stock_position += shares_to_buy
        
        print(f"Sold {contracts} calls (delta {delta:.2f})")
        print(f"  Received: \\$\{premium * contracts * 100:,.0f})"
print(f"  Hedged: Buy {shares_to_buy:.0f} shares")
    
    def rehedge(self, new_delta, stock_price):
"""Adjust hedge as delta changes"""
        # Target hedge
target_shares = self.option_position * 100 * new_delta
        
        # Current hedge
current_shares = self.stock_position
        
        # Adjustment needed
shares_to_trade = target_shares - current_shares

if shares_to_trade > 0:
    print(f"Buy {shares_to_trade:.0f} shares at \\$\{stock_price}")
else:
print(f"Sell {-shares_to_trade:.0f} shares at \\$\{stock_price}")

self.stock_position += shares_to_trade
self.capital -= shares_to_trade * stock_price
    
    def calculate_pnl(self, stock_price, option_price):
"""Calculate current P&L"""
        # Stock P & L
stock_pnl = self.stock_position * stock_price
        
        # Option P & L(negative because we're short)
        option_pnl = self.option_position * option_price * 100
        
        total = stock_pnl + option_pnl + self.capital - 1000000
        return total


# Simulation
mm = MarketMaker()

# Day 1: Sell 10 calls at $100 strike, stock at $100
mm.sell_call(contracts = 10, delta = 0.50, premium = 5.00)

# Day 2: Stock rises to $105, delta increases to 0.70
print("\\nDay 2: Stock at $105, delta = 0.70")
mm.rehedge(new_delta = 0.70, stock_price = 105)

# Day 3: Stock at $110, delta = 0.90
print("\\nDay 3: Stock at $110, delta = 0.90")
mm.rehedge(new_delta = 0.90, stock_price = 110)

# Calculate final P & L
option_value_now = 11.00  # Call now worth $11(stock $110, strike $100)
pnl = mm.calculate_pnl(stock_price = 110, option_price = option_value_now)

print(f"\\nFinal P&L: \\$\{pnl:,.0f}")
print("Market maker stays delta-neutral, profits from spread + gamma scalping")
\`\`\`

---

## Bid-Ask Spread Pricing

### Formula

\`\`\`
Mid Price = Theoretical Value (Black-Scholes)
Bid = Mid - Spread/2
Ask = Mid + Spread/2
\`\`\`

**Spread determined by:**
- **Liquidity** (higher volume = tighter spread)
- **Volatility** (higher vol = wider spread)
- **Inventory** (need to unload inventory = tighter spread on that side)

\`\`\`python
"""
Market Maker Pricing Engine
"""

def calculate_spread(base_spread, liquidity_factor, inventory_position):
    """
    Determine bid-ask spread
    
    Args:
        base_spread: Base spread in dollars (e.g., 0.10)
        liquidity_factor: 1.0 = normal, 2.0 = illiquid (wider)
        inventory_position: >0 = long (want to sell), <0 = short (want to buy)
    """
    # Adjust for liquidity
    spread = base_spread * liquidity_factor
    
    # Skew based on inventory
    if inventory_position > 0:
        # Long inventory - tighten bid, widen ask (incentivize selling)
        bid_adjustment = 0.05
        ask_adjustment = -0.05
    elif inventory_position < 0:
        # Short inventory - widen bid, tighten ask (incentivize buying)
        bid_adjustment = -0.05
        ask_adjustment = 0.05
    else:
        bid_adjustment = 0
        ask_adjustment = 0
    
    return spread, bid_adjustment, ask_adjustment


def quote_option(theoretical_value, base_spread=0.10, liquidity=1.0, inventory=0):
    """Generate bid/ask quotes"""
    spread, bid_adj, ask_adj = calculate_spread(base_spread, liquidity, inventory)
    
    bid = theoretical_value - spread/2 + bid_adj
    ask = theoretical_value + spread/2 + ask_adj
    
    return bid, ask, theoretical_value


# Examples
print("Market Making Quotes:")
print("=" * 50)

# Normal market
bid, ask, mid = quote_option(theoretical_value=5.00, inventory=0)
print(f"\\nNormal Market:")
print(f"  Bid: \${bid:.2f} / Ask: \${ask:.2f} (Mid \\$\{mid:.2f})")

# Long inventory(want to sell)
bid, ask, mid = quote_option(theoretical_value = 5.00, inventory = 100)
print(f"\\nLong Inventory (want to sell):")
print(f"  Bid: \${bid:.2f} / Ask: \\$\{ask:.2f}")

# Short inventory(want to buy)
bid, ask, mid = quote_option(theoretical_value = 5.00, inventory = -100)
print(f"\\nShort Inventory (want to buy):")
print(f"  Bid: \${bid:.2f} / Ask: \\$\{ask:.2f}")

# Illiquid market
bid, ask, mid = quote_option(theoretical_value = 5.00, liquidity = 2.0)
print(f"\\nIlliquid Market (2x spread):")
print(f"  Bid: \${bid:.2f} / Ask: \\$\{ask:.2f}")
\`\`\`

---

## Gamma Scalping

### Concept

**Gamma scalping:** Profit from rehedging a delta-neutral position as stock moves.

**Mechanics:**1. Long gamma (long options) + delta hedge
2. Stock moves → delta changes
3. Rehedge: Buy low, sell high
4. Repeat

\`\`\`python
"""
Gamma Scalping Simulation
"""

def simulate_gamma_scalping(initial_stock, strike, days=30, volatility=0.25):
    """
    Simulate gamma scalping strategy
    """
    import random
    
    # Initial setup
    stock = initial_stock
    position_pnl = []
    hedges = []
    
    # Long 1 ATM straddle
    straddle_cost = 10  # Simplified
    gamma = 0.03  # Gamma value
    
    for day in range(days):
        # Stock moves (random walk)
        daily_return = random.gauss(0, volatility/np.sqrt(252))
        stock = stock * (1 + daily_return)
        
        # Calculate new delta
        delta = calculate_option_delta(stock, strike)
        
        # Rehedge (gamma scalping)
        # When stock up → delta increases → buy more stock (buy high)
        # When stock down → delta decreases → sell stock (sell low)
        # WAIT, that's backwards!
        
        # Actually:
        # Long gamma means: Stock up → sell stock (sell high), Stock down → buy stock (buy low)
        
        # Track
        position_pnl.append({
            'day': day,
            'stock': stock,
            'delta': delta
        })
    
    return position_pnl

# Note: Gamma scalping is complex, requires continuous rehedging
# Market makers profit from realized vol > implied vol
\`\`\`

---

## Pin Risk

**Pin risk:** Stock closes exactly at strike at expiration.

**Problem:**
- Unclear if options will be exercised
- Market maker has large hedge position
- Need to unwind at open Monday (risky)

**Example:**
- MM short 1000 calls at $100 strike
- Hedged with long 50,000 shares
- Friday close: Stock exactly $100
- Uncertain: Will calls be exercised?
- If yes: MM sells 100,000 shares (called away) → now short 50K shares
- If no: MM long 50,000 shares → need to sell Monday

---

## High-Frequency Trading in Options

**HFT strategies:**
- **Latency arbitrage:** Faster quotes, better prices
- **Statistical arbitrage:** Small edges, high volume
- **Spread capture:** Compete to capture bid-ask spread

**Technology:**
- Co-located servers (microseconds matter)
- FPGA chips for ultra-low latency
- Direct market data feeds

---

## Risk Management

**Key Risks:**1. **Delta risk:** Not perfectly hedged
2. **Gamma risk:** Large moves → large rehedging costs
3. **Vega risk:** IV changes (inventory exposed)
4. **Pin risk:** Uncertain exercise at expiration

**Position Limits:**
- Max net delta exposure: e.g., 1000 shares
- Max gamma: e.g., 100 gamma
- Max vega: e.g., $10,000 per 1% IV

---

## Advanced Greeks for Market Makers

### Second-Order Greeks

Market makers track **advanced Greeks** beyond delta/gamma/theta/vega.

\`\`\`python
"""
Advanced Greeks: Volga, Vanna, Charm
"""

def calculate_advanced_greeks(S, K, T, r, sigma):
    """
    Calculate second-order Greeks
    
    - Volga (Vomma): d(Vega)/d(sigma) - convexity of vega
    - Vanna: d(Delta)/d(sigma) or d(Vega)/d(S) - cross-sensitivity
    - Charm: d(Delta)/d(T) - delta decay over time
    """
    from scipy.stats import norm
    
    # Standard d1, d2
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    # Basic Greeks
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    delta = norm.cdf(d1)
    
    # Volga (Vomma)
    # How vega changes with volatility
    volga = vega * d1 * d2 / sigma
    
    # Vanna
    # How delta changes with volatility (or vega with stock)
    vanna = -vega / S * d2 / (sigma * np.sqrt(T))
    
    # Charm (Delta Decay)
    # How delta changes over time
    charm = -norm.pdf(d1) * (2*r*T - d2*sigma*np.sqrt(T)) / (2*T*sigma*np.sqrt(T))
    
    return {
        'volga': volga,
        'vanna': vanna,
        'charm': charm
    }


# Example
advanced = calculate_advanced_greeks(S=100, K=100, T=30/365, r=0.05, sigma=0.25)

print("=" * 70)
print("ADVANCED GREEKS FOR MARKET MAKERS")
print("=" * 70)
print(f"\\nATM Call (S=$100, K=$100, T=30d, σ=25%):")
print(f"\\nVolga: {advanced['volga']:.4f}")
print(f"  Interpretation: If IV increases 1%, vega changes by {advanced['volga']:.4f}")
print(f"  Use: Hedge vega convexity risk")

print(f"\\nVanna: {advanced['vanna']:.4f}")
print(f"  Interpretation: If IV increases 1%, delta changes by {advanced['vanna']:.4f}")
print(f"  Or: If stock moves $1, vega changes by {advanced['vanna']:.4f}")
print(f"  Use: Manage interaction between delta and vega hedges")

print(f"\\nCharm: {advanced['charm']:.4f}")
print(f"  Interpretation: Delta decreases by {advanced['charm']:.4f} per day")
print(f"  Use: Anticipate rehedging needs as expiration approaches")

# Practical example
print(f"\\n{'─' * 70}")
print("PRACTICAL APPLICATION:")
print(f"\\nScenario: Market maker short 1000 ATM calls")
print(f"  Initial delta hedge: Short 50,000 shares (delta 0.50 each)")
print(f"\\n1 Day Later:")
print(f"  Charm effect: Delta decreased by {advanced['charm']*1000:.0f}")
print(f"  New delta: {(0.50 + advanced['charm'])*1000:.0f}")
print(f"  Rehedge needed: Buy back {abs(advanced['charm']*1000):.0f} shares")
print(f"\\nWithout tracking charm, miss rehedging opportunity!")
\`\`\`

### Why Advanced Greeks Matter

**For large positions:**
- **Volga:** Vega changes as IV moves (vega convexity)
- **Vanna:** Delta and vega interact (cross-gamma)
- **Charm:** Delta decays approaching expiration

**Risk:**
- Not tracking → Unexpected P&L swings
- Large inventory → Second-order effects dominate

---

## Inventory Management Strategies

### Optimal Inventory Levels

Market makers manage **inventory risk** - don't want to accumulate large long/short positions.

\`\`\`python
"""
Inventory Management System
"""

class InventoryManager:
    """
    Manage market maker inventory and skew quotes
    """
    
    def __init__(self, max_inventory=100):
        self.inventory = 0  # Net option position (contracts)
        self.max_inventory = max_inventory
        self.target_inventory = 0  # Ideally flat
    
    def calculate_inventory_skew(self):
        """
        Calculate how much to skew bid/ask based on inventory
        
        Returns:
            (bid_adjustment, ask_adjustment) in cents
        """
        # Inventory as % of max
        inventory_pct = self.inventory / self.max_inventory
        
        # Skew quotes to incentivize reducing inventory
        if inventory_pct > 0.5:  # Long inventory
            # Tighten ask (easier to sell), widen bid
            bid_adjustment = -0.05
            ask_adjustment = +0.05
        elif inventory_pct < -0.5:  # Short inventory
            # Tighten bid (easier to buy), widen ask
            bid_adjustment = +0.05
            ask_adjustment = -0.05
        else:
            # Neutral - no skew
            bid_adjustment = 0
            ask_adjustment = 0
        
        # Scale by severity
        severity = abs(inventory_pct)
        bid_adjustment *= severity
        ask_adjustment *= severity
        
        return bid_adjustment, ask_adjustment
    
    def update_quotes(self, theoretical_mid, base_spread=0.10):
        """
        Generate bid/ask quotes with inventory skew
        """
        bid_adj, ask_adj = self.calculate_inventory_skew()
        
        bid = theoretical_mid - base_spread/2 + bid_adj
        ask = theoretical_mid + base_spread/2 + ask_adj
        
        return bid, ask
    
    def trade_occurs(self, direction, quantity):
        """
        Update inventory after trade
        
        Args:
            direction: 'buy' (we buy = long) or 'sell' (we sell = short)
            quantity: Number of contracts
        """
        if direction == 'buy':
            self.inventory += quantity
        else:
            self.inventory -= quantity
        
        # Check if need urgent unwind
        if abs(self.inventory) > self.max_inventory:
            return f"⚠️  CRITICAL: Inventory {self.inventory} exceeds limit {self.max_inventory}"
        
        return None


# Simulation
manager = InventoryManager(max_inventory=100)

print("=" * 70)
print("INVENTORY MANAGEMENT SIMULATION")
print("=" * 70)

# Simulate trades over day
trades = [
    ('buy', 10),
    ('buy', 15),
    ('sell', 5),
    ('buy', 20),  # Now getting long
    ('buy', 30),  # Very long!
    ('sell', 40),  # Unload inventory
]

theoretical_mid = 5.00

for i, (direction, quantity) in enumerate(trades):
    # Update inventory
    alert = manager.trade_occurs(direction, quantity)
    
    # Generate new quotes
    bid, ask = manager.update_quotes(theoretical_mid)
    
    print(f"\\nTrade {i+1}: {direction.upper()} {quantity} contracts")
    print(f"  Inventory: {manager.inventory:+d} contracts")
    print(f"  Quotes: \${bid:.2f} / \\$\{ask:.2f}")

if alert:
    print(f"  {alert}")
    
    # Show skew
if manager.inventory > 50:
    print(f"  📉 Long inventory → Ask tightened to attract sellers")
    elif manager.inventory < -50:
print(f"  📈 Short inventory → Bid tightened to attract buyers")

print(f"\\n{'─' * 70}")
print(f"Final Inventory: {manager.inventory:+d} contracts")
print(f"Status: {'✓ Acceptable' if abs(manager.inventory) < 50 else '⚠️ Need to unwind'}")
\`\`\`

---

## High-Frequency Options Trading

### Technology Infrastructure

**Modern market making requires:**
- **Co-location:** Servers in exchange data center (microseconds matter)
- **FPGA chips:** Hardware-level processing (nanoseconds)
- **Direct feeds:** Fastest market data (no retail delays)
- **Low-latency networks:** Fiber optic, microwave links

\`\`\`python
"""
Latency Arbitrage Simulation
"""

class LatencyArbitrage:
    """
    Simulate HFT market making advantage
    """
    
    def __init__(self, our_latency_ms, competitor_latency_ms):
        self.our_latency = our_latency_ms / 1000  # Convert to seconds
        self.competitor_latency = competitor_latency_ms / 1000
    
    def simulate_quote_race(self, market_move_size=0.05):
        """
        Simulate race to update quotes after market move
        
        Our firm: 0.1ms latency (co-located)
        Competitor: 10ms latency (retail)
        
        When stock moves, who updates quotes first?
        """
        print("=" * 70)
        print("LATENCY ARBITRAGE SIMULATION")
        print("=" * 70)
        
        print(f"\\nStock moves $0.05 in SPY")
        print(f"  Our latency: {self.our_latency*1000:.1f} ms")
        print(f"  Competitor latency: {self.competitor_latency*1000:.1f} ms")
        
        # We update first
        time_advantage = self.competitor_latency - self.our_latency
        
        print(f"\\nTime advantage: {time_advantage*1000:.1f} ms")
        print(f"\\nScenario:")
        print(f"  1. Stock moves at T=0")
        print(f"  2. We update quotes at T={self.our_latency*1000:.1f}ms")
        print(f"  3. Competitor updates at T={self.competitor_latency*1000:.1f}ms")
        print(f"\\nResult:")
        print(f"  → We cancel old stale quotes")
        print(f"  → Competitor trades into our new quotes (we profit)")
        print(f"  → Or: We trade into competitor's stale quotes (we profit)")
        
        # Calculate value
        trades_per_day = 1000  # High volume
        avg_profit_per_trade = 0.01  # 1 cent edge
        daily_profit = trades_per_day * avg_profit_per_trade
        annual_profit = daily_profit * 252
        
        print(f"\\n{'─' * 70}")
        print(f"VALUE OF SPEED:")
        print(f"  Trades per day: {trades_per_day:,}")
        print(f"  Avg edge per trade: \\$\{avg_profit_per_trade:.2f}")
print(f"  Daily profit: \\$\{daily_profit:,.0f}")
print(f"  Annual profit: \\$\{annual_profit:,.0f}")
print(f"\\n  This is why firms pay millions for co-location!")


# Example
latency_arb = LatencyArbitrage(
    our_latency_ms = 0.1,  # 0.1 milliseconds(co - located, FPGA)
    competitor_latency_ms = 10  # 10 milliseconds(retail connection)
)

latency_arb.simulate_quote_race()
\`\`\`

### Statistical Arbitrage

**HFT strategies:**
- **Mean reversion:** Option prices revert to fair value
- **Volatility surface arbitrage:** Exploit mispricings
- **Order flow prediction:** Predict price moves from order flow

---

## Payment for Order Flow (PFOF)

### How It Works

**Retail brokers** (Robinhood, etc.) sell order flow to market makers.

**Business model:**1. Retail investor wants to buy call at market
2. Broker sends order to market maker (Citadel, Virtu)
3. Market maker fills order (gives price improvement)
4. Market maker pays broker $0.50 per contract
5. Market maker profits from spread + information

\`\`\`python
"""
PFOF Economics
"""

def analyze_pfof_economics():
    """
    Analyze profitability of Payment for Order Flow
    """
    print("=" * 70)
    print("PAYMENT FOR ORDER FLOW (PFOF) ECONOMICS")
    print("=" * 70)
    
    # Per contract
    pfof_paid = 0.50  # Pay broker $0.50 per contract
    spread_captured = 0.10  # Capture $0.10 of spread
    information_value = 0.15  # Order flow information worth $0.15
    risk = 0.05  # Risk of adverse selection $0.05
    
    profit_per_contract = spread_captured + information_value - pfof_paid - risk
    
    print(f"\\nPer Contract Economics:")
    print(f"  Revenue:")
    print(f"    Spread captured: \\$\{spread_captured:.2f}")
print(f"    Information value: \\$\{information_value:.2f}")
print(f"    Total revenue: \\$\{spread_captured + information_value:.2f}")

print(f"\\n  Costs:")
print(f"    PFOF paid to broker: \\$\{pfof_paid:.2f}")
print(f"    Adverse selection risk: \\$\{risk:.2f}")
print(f"    Total costs: \\$\{pfof_paid + risk:.2f}")

print(f"\\n  Net profit: \\$\{profit_per_contract:.2f} per contract")
    
    # Scale
daily_volume = 100000  # 100K contracts per day
daily_profit = profit_per_contract * daily_volume
annual_profit = daily_profit * 252

print(f"\\n{'─' * 70}")
print(f"AT SCALE:")
print(f"  Daily volume: {daily_volume:,} contracts")
print(f"  Daily profit: \\$\{daily_profit:,.0f}")
print(f"  Annual profit: \\$\{annual_profit:,.0f}")

print(f"\\n{'─' * 70}")
print(f"WHY PFOF IS VALUABLE:")
print(f"  1. Retail order flow is 'uninformed' (no information edge)")
print(f"  2. Market makers can trade against retail profitably")
print(f"  3. Retail gets price improvement (win-win)")
print(f"  4. Billions in profit for firms like Citadel, Virtu")

analyze_pfof_economics()
\`\`\`

---

## Market Making P&L Attribution

### Daily P&L Breakdown

Market makers track **exactly** where profits come from.

\`\`\`python
"""
Market Maker P&L Attribution
"""

class MarketMakerPnL:
    """
    Track and attribute P&L sources
    """
    
    def __init__(self):
        self.pnl_sources = {
            'spread_capture': 0,
            'gamma_scalping': 0,
            'theta_decay': 0,
            'vega_pnl': 0,
            'inventory_risk': 0,
            'adverse_selection': 0
        }
    
    def attribute_daily_pnl(self, trades_data, portfolio_greeks, market_changes):
        """
        Attribute P&L to sources
        
        Args:
            trades_data: List of trades executed
            portfolio_greeks: Greeks at start of day
            market_changes: Dict of price/IV changes
        """
        # 1. Spread capture
        for trade in trades_data:
            self.pnl_sources['spread_capture'] += trade['spread_captured']
        
        # 2. Gamma scalping
        stock_move = market_changes['price_change']
        gamma_pnl = 0.5 * portfolio_greeks['gamma'] * (stock_move ** 2)
        self.pnl_sources['gamma_scalping'] += gamma_pnl
        
        # 3. Theta decay
        theta_pnl = portfolio_greeks['theta']
        self.pnl_sources['theta_decay'] += theta_pnl
        
        # 4. Vega P&L
        iv_change = market_changes['iv_change']
        vega_pnl = portfolio_greeks['vega'] * iv_change
        self.pnl_sources['vega_pnl'] += vega_pnl
        
        # 5. Inventory risk
        # P&L from holding net long/short position
        inventory_pnl = portfolio_greeks['delta'] * stock_move
        self.pnl_sources['inventory_risk'] += inventory_pnl
        
        # 6. Adverse selection
        # Losses from trading against informed flow
        for trade in trades_data:
            if trade.get('informed', False):
                self.pnl_sources['adverse_selection'] -= trade['adverse_cost']
        
        return self.pnl_sources
    
    def generate_report(self):
        """Generate daily P&L attribution report"""
        total_pnl = sum(self.pnl_sources.values())
        
        print("=" * 70)
        print("DAILY P&L ATTRIBUTION REPORT")
        print("=" * 70)
        
        print(f"\\n{'Source':<25} {'P&L':>12} {'% of Total':>12}")
        print("─" * 70)
        
        for source, pnl in sorted(self.pnl_sources.items(), 
                                  key=lambda x: x[1], reverse=True):
            pct = (pnl / total_pnl * 100) if total_pnl != 0 else 0
            status = '🟢' if pnl > 0 else '🔴' if pnl < 0 else '⚪'
            print(f"{source.replace('_', ' ').title():<25} \\$\{pnl:> 11, .0f} { status } { pct:> 10.1f }% ")

print("─" * 70)
print(f"{'TOTAL':<25} \\$\{total_pnl:>11,.0f}")

print(f"\\n{'─' * 70}")
print("INSIGHTS:")
        
        # Identify main profit driver
max_source = max(self.pnl_sources.items(), key = lambda x: x[1])
print(f"  Largest contributor: {max_source[0].replace('_', ' ').title()} (\\$\{max_source[1]:,.0f})")
        
        # Check if losing to adverse selection
if self.pnl_sources['adverse_selection'] < -1000:
    print(f"  ⚠️  High adverse selection losses - review order flow sources")
        
        # Check gamma scalping effectiveness
if self.pnl_sources['gamma_scalping'] < 0:
    print(f"  ⚠️  Negative gamma scalping - may be overhedging")


# Example
pnl_tracker = MarketMakerPnL()

# Simulate day's activity
trades = [
    { 'spread_captured': 50, 'informed': False },
    { 'spread_captured': 40, 'informed': False },
    { 'spread_captured': 60, 'informed': False },
    { 'spread_captured': -100, 'informed': True, 'adverse_cost': 100 },  # Bad trade
] * 100  # 400 trades total

portfolio_greeks = {
    'delta': 1000,
    'gamma': 500,
    'theta': 2000,
    'vega': 30000
}

market_changes = {
    'price_change': 2.5,  # Stock up $2.50
    'iv_change': - 0.02  # IV down 2 %
}

pnl_tracker.attribute_daily_pnl(trades, portfolio_greeks, market_changes)
pnl_tracker.generate_report()
\`\`\`

---

## Real-World Market Maker Firms

### Major Players

**Equities Options:**
- **Citadel Securities:** Largest ($7B+ annual revenue)
- **Virtu Financial:** High-frequency specialist
- **Susquehanna (SIG):** Options-focused prop shop
- **Jane Street:** Algorithmic market maker
- **IMC Trading:** European leader

**Business Model:**
- Volume × Spread × Efficiency
- 1 cent profit × 100M contracts × 252 days = $252M annually
- Requires: Speed, capital, technology, risk management

---

## Summary

**Market making key concepts:**
- **Provide liquidity,** earn bid-ask spread
- **Delta-neutral** through continuous hedging
- **Gamma scalping** profit from volatility
- **Advanced Greeks** (volga, vanna, charm) for large positions
- **Inventory management** critical to avoid risk accumulation
- **HFT technology** (co-location, FPGA) for competitive edge
- **PFOF** major revenue source (pay for retail order flow)
- **P&L attribution** tracks exact profit sources

**Professional infrastructure:**
- Microsecond latency technology
- Real-time risk monitoring
- Automated hedging systems
- Sophisticated pricing models
- 24/7 operations

**Not for retail traders** - requires institutional capital, technology, and expertise. But understanding market making helps traders:
- Get better fills (know the spread)
- Understand liquidity dynamics
- Appreciate option pricing in real markets

This is the foundation of modern options markets!
`,
};
