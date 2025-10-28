export const marketMakingLiquidityProvision = {
  title: 'Market Making and Liquidity Provision',
  id: 'market-making-liquidity-provision',
  content: `
# Market Making and Liquidity Provision

## Introduction

**Market making** is the practice of simultaneously providing buy and sell quotes for a financial instrument, profiting from the bid-ask spread while managing inventory risk. Market makers are critical to modern financial markets, providing **liquidity**—the ability for other participants to quickly buy or sell without significantly moving prices.

Market makers face a fundamental trade-off:
- **Profit opportunity:** Capture the bid-ask spread on each round-trip trade.
- **Inventory risk:** Accumulating positions exposes them to adverse price movements.
- **Adverse selection risk:** Trading with informed parties who know the price is about to move.

Understanding market making is essential for:
- **Quantitative traders:** Implementing profitable market making strategies.
- **Risk managers:** Managing inventory and adverse selection exposure.
- **Execution traders:** Understanding how market makers respond to order flow.
- **Regulators:** Evaluating market quality and liquidity provision.

This section explores the economics, mathematics, and technology of professional market making.

---

## Deep Technical Explanation: Market Making Theory

### 1. Market Maker Economics

**Revenue Sources:**1. **Bid-Ask Spread:**
   - **Capture:** Buy at bid ($100.00), sell at ask ($100.05) → profit $0.05 per share.
   - **Volume:** Execute thousands of round trips daily.
   - **Example:** 1 million shares daily × $0.03 average spread = $30,000 daily revenue.

2. **Exchange Rebates:**
   - **Maker-Taker Model:** Exchanges pay rebates to liquidity providers (~$0.002 per share).
   - **Impact:** Can represent 30-50% of total revenue.
   - **Example:** Post 1 million shares → receive $2,000 in rebates.

3. **Payment for Order Flow (PFOF):**
   - **Retail Flow:** Brokers (Robinhood, E*TRADE) sell order flow to market makers.
   - **Cost:** Market maker pays broker ~$0.001-0.003 per share.
   - **Benefit:** Retail flow is typically uninformed (lower adverse selection risk).
   - **Profit:** Market maker provides price improvement (better than NBBO), still captures most of spread.

**Costs:**1. **Adverse Selection:**
   - **Losses to Informed Traders:** Trading with parties who have superior information.
   - **Magnitude:** 30-60% of gross spread captured.
   - **Example:** Capture $0.03 spread, but lose $0.015 on average to informed traders → net $0.015.

2. **Inventory Risk:**
   - **Price Risk:** Holding long or short positions exposes to market movements.
   - **Example:** Long 10,000 shares, price drops $0.50 → lose $5,000.

3. **Technology and Infrastructure:**
   - **Co-location:** $10K-50K per month per exchange.
   - **Development:** Millions in software and hardware development.
   - **Personnel:** Highly paid quants, engineers, traders.

4. **Regulatory Costs:**
   - **Compliance:** Legal, reporting, surveillance systems.
   - **Capital Requirements:** Broker-dealers must maintain minimum net capital.

### 2. Inventory Management

**Challenge:** Market makers must balance capturing spreads with avoiding dangerous inventory accumulation.

**Strategies:**

**A. Quote Skewing:**
- **Principle:** Adjust quotes to discourage inventory accumulation and encourage reduction.
- **Implementation:**
  - **Long inventory:** Widen bid (discourage more buys), tighten/lower ask (encourage sells).
  - **Short inventory:** Tighten bid (encourage buys), widen ask (discourage sells).

**Formula:**
\`\`\`
Bid_skew = -α × inventory
Ask_skew = -α × inventory
\`\`\`
Where α is the skew coefficient (e.g., 0.001 = 10 bps per 1000 shares).

**Example:**
- **Neutral:** Bid $100.00, Ask $100.05
- **Long 5,000 shares:** Bid $99.95 (-5 bps), Ask $100.00 (-5 bps)
- **Short 5,000 shares:** Bid $100.05 (+5 bps), Ask $100.10 (+5 bps)

**B. Size Adjustment:**
- **High inventory:** Reduce quote size on the side that increases inventory.
- **Example:** Long 80% of limit → quote only 100 shares on bid (vs. normal 1,000), but 2,000 on ask.

**C. Hedging:**
- **Correlated Instruments:** Offset inventory with futures, ETFs, or baskets.
- **Example:** Long 10,000 shares of AAPL → sell 100 contracts of QQQ (Nasdaq-100 ETF) to hedge market exposure.
- **Dynamic:** Continuously adjust hedge ratio based on correlations.

**D. Temporary Exit:**
- **Extreme Imbalance:** If inventory approaches limits and market is moving against you, exit completely (flatten at market).
- **Loss Acceptance:** Better to take a small certain loss than risk a large uncertain loss.

### 3. Avellaneda-Stoikov Model

The **Avellaneda-Stoikov model** (2008) provides a rigorous framework for optimal market making with inventory risk.

**Problem Setup:**
- **Goal:** Maximize expected utility of wealth at end of trading period.
- **Controls:** Bid and ask quotes (prices and sizes).
- **State:** Current inventory, time remaining.
- **Constraints:** Inventory limits, spread must be non-negative.

**Model Assumptions:**
- **Price dynamics:** Midpoint follows Brownian motion (drift μ, volatility σ).
- **Order arrival:** Poisson process with intensity depending on distance from midpoint.
- **Utility:** Exponential utility with risk aversion parameter γ.

**Key Results:**

**Optimal Spread (Around Midpoint):**
\`\`\`
δ_bid = (1/γ) × ln(1 + γ/k) + q × (γσ²/2k) × (T - t)
δ_ask = (1/γ) × ln(1 + γ/k) - q × (γσ²/2k) × (T - t)
\`\`\`

Where:
- **δ_bid, δ_ask:** Distance of bid/ask from midpoint.
- **γ:** Risk aversion (higher γ → wider spreads, more conservative).
- **k:** Order arrival rate intensity parameter.
- **q:** Current inventory (signed: positive = long, negative = short).
- **σ:** Volatility.
- **T - t:** Time remaining in trading period.

**Intuition:**
- **Base spread:** (1/γ) × ln(1 + γ/k) is the spread when inventory is zero.
- **Inventory adjustment:** q × (γσ²/2k) × (T - t) skews quotes away from accumulating more inventory.
  - **Long (q > 0):** Widen bid (δ_bid increases), tighten ask (δ_ask decreases) → encourage sells.
  - **Short (q < 0):** Tighten bid, widen ask → encourage buys.
- **Time decay:** As T - t → 0, inventory adjustment intensifies (need to flatten before end).

**Order Arrival Intensity:**
The probability of a fill at price level p depends on distance from midpoint:
\`\`\`
λ_bid(δ) = Λ × exp(-k × δ_bid)
λ_ask(δ) = Λ × exp(-k × δ_ask)
\`\`\`
Where Λ is base arrival rate, k controls sensitivity to price.

**Trade-off:**
- **Wider spread:** Lower fill rate, but higher profit per fill and lower inventory risk.
- **Tighter spread:** Higher fill rate, but lower profit per fill and higher inventory risk.

### 4. Adverse Selection Management

**Detection:**1. **Order Flow Analysis:**
   - **Persistent One-Sided Pressure:** Multiple buys (or sells) in succession → likely informed.
   - **Large Orders:** Aggressive market orders of significant size.
   - **Timing:** Orders just before news releases, earnings, macro events.

2. **VPIN (Volume-Synchronized Probability of Informed Trading):**
   - **Calculation:** Measures buy-sell imbalance in fixed volume buckets.
   - **High VPIN:** Indicates high probability of informed trading.
   - **Response:** Widen spreads, reduce size, or stop quoting.

3. **Price Impact Analysis:**
   - **Realized Spread:** If trades consistently move price against you, adverse selection is high.
   - **Formula:** Realized Spread = 2 × (Fill Price - Midpoint 5 minutes later)
   - **Negative Realized Spread:** Indicates adverse selection (you lost money).

**Mitigation:**1. **Spread Widening:**
   - **Before Earnings:** Widen spread by 50-200% to compensate for higher adverse selection risk.
   - **During News:** Stop quoting momentarily, wait for information to be digested.

2. **Quote Size Reduction:**
   - **High VPIN:** Quote smaller sizes (100 shares instead of 1,000).
   - **Limit Exposure:** Cap loss from any single informed trade.

3. **Speed:**
   - **Fast Updates:** Update quotes in microseconds to reduce time window for adverse selection.
   - **Cancel on News:** Immediately pull quotes when news hits (NLP + automated cancellation).

4. **Selective Quoting:**
   - **Avoid Toxic Symbols:** Don't make markets in securities with high informed trading (e.g., biotech before FDA decisions).
   - **Prefer Retail Flow:** Seek PFOF deals (retail flow is less informed).

---

## Code Implementation: Avellaneda-Stoikov Market Maker

### Avellaneda-Stoikov Model Implementation

\`\`\`python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple

@dataclass
class MarketMakerParams:
    """Parameters for Avellaneda-Stoikov market maker."""
    risk_aversion: float = 0.1  # γ (gamma)
    volatility: float = 0.02  # σ (sigma) - per time unit
    time_horizon: float = 1.0  # T (total trading period)
    order_intensity: float = 1.5  # Λ (base arrival rate)
    sensitivity: float = 1.5  # k (price sensitivity)
    tick_size: float = 0.01  # Minimum price increment
    max_inventory: int = 10000  # Position limit

class AvellanedaStoikovMarketMaker:
    """
    Implements Avellaneda-Stoikov optimal market making strategy.
    
    The model determines optimal bid and ask quotes based on:
    - Current inventory
    - Volatility
    - Risk aversion
    - Time remaining
    """
    def __init__(self, params: MarketMakerParams):
        self.params = params
        self.inventory = 0
        self.cash = 0.0
        self.time_elapsed = 0.0
        self.trades = []
        self.pnl_history = []
    
    def calculate_reservation_price(self, midpoint: float, time_remaining: float) -> float:
        """
        Calculate reservation price (indifference price).
        
        This is the price at which the market maker is indifferent to inventory.
        
        Formula: r = s - q × γ × σ² × (T - t)
        """
        inventory_adjustment = (
            self.inventory * 
            self.params.risk_aversion * 
            self.params.volatility ** 2 * 
            time_remaining
        )
        reservation_price = midpoint - inventory_adjustment
        return reservation_price
    
    def calculate_optimal_spread(self, time_remaining: float) -> float:
        """
        Calculate optimal half-spread.
        
        Formula: δ = (1/γ) × ln(1 + γ/k)
        
        This is the base spread when inventory is zero.
        """
        gamma = self.params.risk_aversion
        k = self.params.sensitivity
        
        half_spread = (1 / gamma) * np.log(1 + gamma / k)
        return half_spread
    
    def calculate_quotes(self, midpoint: float, time_remaining: float) -> Tuple[float, float]:
        """
        Calculate optimal bid and ask quotes.
        
        Returns:
        - (bid_price, ask_price)
        """
        # Reservation price (adjusted for inventory)
        reservation_price = self.calculate_reservation_price(midpoint, time_remaining)
        
        # Base half-spread
        half_spread = self.calculate_optimal_spread(time_remaining)
        
        # Inventory adjustment to spread
        inventory_adjustment = (
            self.inventory * 
            self.params.risk_aversion * 
            self.params.volatility ** 2 * 
            time_remaining / 
            (2 * self.params.sensitivity)
        )
        
        # Calculate quotes
        bid_price = reservation_price - half_spread - inventory_adjustment
        ask_price = reservation_price + half_spread - inventory_adjustment
        
        # Round to tick size
        bid_price = round(bid_price / self.params.tick_size) * self.params.tick_size
        ask_price = round(ask_price / self.params.tick_size) * self.params.tick_size
        
        # Ensure positive spread
        if ask_price <= bid_price:
            ask_price = bid_price + self.params.tick_size
        
        return bid_price, ask_price
    
    def calculate_fill_probability(self, quote_distance: float) -> float:
        """
        Calculate probability of order being filled.
        
        Formula: λ = Λ × exp(-k × δ)
        
        Returns: Fill probability (0 to 1)
        """
        intensity = self.params.order_intensity * np.exp(-self.params.sensitivity * abs(quote_distance))
        # Convert intensity to probability for small time step (assuming dt = 1 second)
        probability = 1 - np.exp(-intensity)
        return probability
    
    def execute_trade(self, side: str, price: float, size: int, timestamp: float):
        """Execute a trade and update state."""
        if side == 'buy':
            self.inventory += size
            self.cash -= price * size
        else:  # sell
            self.inventory -= size
            self.cash += price * size
        
        self.trades.append({
            'timestamp': timestamp,
            'side': side,
            'price': price,
            'size': size,
            'inventory': self.inventory,
            'cash': self.cash
        })
    
    def get_pnl(self, current_midpoint: float) -> float:
        """Calculate current mark-to-market P&L."""
        return self.cash + self.inventory * current_midpoint
    
    def step(self, midpoint: float, dt: float) -> dict:
        """
        Execute one time step of market making.
        
        Parameters:
        - midpoint: Current market midpoint
        - dt: Time step size
        
        Returns:
        - Dictionary with bid, ask, fill information
        """
        self.time_elapsed += dt
        time_remaining = max(0, self.params.time_horizon - self.time_elapsed)
        
        # Calculate optimal quotes
        bid_price, ask_price = self.calculate_quotes(midpoint, time_remaining)
        
        # Calculate fill probabilities
        bid_distance = midpoint - bid_price
        ask_distance = ask_price - midpoint
        
        bid_fill_prob = self.calculate_fill_probability(bid_distance)
        ask_fill_prob = self.calculate_fill_probability(ask_distance)
        
        # Simulate fills (Poisson arrivals)
        bid_filled = np.random.random() < bid_fill_prob * dt
        ask_filled = np.random.random() < ask_fill_prob * dt
        
        # Execute trades
        trade_size = 100  # Fixed size for simplicity
        
        if bid_filled and abs(self.inventory + trade_size) <= self.params.max_inventory:
            self.execute_trade('buy', bid_price, trade_size, self.time_elapsed)
        
        if ask_filled and abs(self.inventory - trade_size) <= self.params.max_inventory:
            self.execute_trade('sell', ask_price, trade_size, self.time_elapsed)
        
        # Record P&L
        pnl = self.get_pnl(midpoint)
        self.pnl_history.append({
            'timestamp': self.time_elapsed,
            'midpoint': midpoint,
            'bid': bid_price,
            'ask': ask_price,
            'inventory': self.inventory,
            'pnl': pnl,
            'spread': ask_price - bid_price
        })
        
        return {
            'bid': bid_price,
            'ask': ask_price,
            'bid_filled': bid_filled,
            'ask_filled': ask_filled,
            'inventory': self.inventory,
            'pnl': pnl
        }

# Simulation
def simulate_market_making(num_steps: int = 1000, seed: int = 42):
    """
    Simulate Avellaneda-Stoikov market making over time.
    """
    np.random.seed(seed)
    
    params = MarketMakerParams(
        risk_aversion=0.1,
        volatility=0.02,
        time_horizon=100.0,
        order_intensity=2.0,
        sensitivity=1.5,
        tick_size=0.01,
        max_inventory=5000
    )
    
    mm = AvellanedaStoikovMarketMaker(params)
    
    # Simulate midpoint as geometric Brownian motion
    midpoint = 100.0
    dt = params.time_horizon / num_steps
    
    for step in range(num_steps):
        # Update midpoint (random walk with drift)
        midpoint *= np.exp((0.0 - 0.5 * params.volatility**2) * dt + 
                          params.volatility * np.sqrt(dt) * np.random.randn())
        
        # Market maker quotes and potentially trades
        mm.step(midpoint, dt)
    
    return mm, pd.DataFrame(mm.pnl_history)

# Run simulation
mm, pnl_df = simulate_market_making(num_steps=10000)

print("Avellaneda-Stoikov Market Making Simulation")
print("=" * 70)
print(f"Total trades executed: {len(mm.trades)}")
print(f"Final inventory: {mm.inventory} shares")
print(f"Final P&L: \\$\{mm.get_pnl(pnl_df.iloc[-1]['midpoint']):.2f}")
print(f"Average spread: \\$\{pnl_df['spread'].mean():.4f}")
print(f"Max inventory: {pnl_df['inventory'].abs().max()} shares")
print()

# Visualizations
fig, axes = plt.subplots(3, 1, figsize = (14, 10))

# P & L over time
axes[0].plot(pnl_df['timestamp'], pnl_df['pnl'], linewidth = 1)
axes[0].set_xlabel('Time')
axes[0].set_ylabel('P&L ($)')
axes[0].set_title('Mark-to-Market P&L Over Time')
axes[0].grid(True, alpha = 0.3)

# Inventory over time
axes[1].plot(pnl_df['timestamp'], pnl_df['inventory'], linewidth = 1, color = 'orange')
axes[1].axhline(y = 0, color = 'black', linestyle = '--', alpha = 0.5)
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Inventory (shares)')
axes[1].set_title('Inventory Over Time')
axes[1].grid(True, alpha = 0.3)

# Spread over time
axes[2].plot(pnl_df['timestamp'], pnl_df['spread'], linewidth = 1, color = 'green')
axes[2].set_xlabel('Time')
axes[2].set_ylabel('Spread ($)')
axes[2].set_title('Bid-Ask Spread Over Time (Adjusts with Inventory)')
axes[2].grid(True, alpha = 0.3)

plt.tight_layout()
plt.show()

# Analyze spread vs inventory relationship
fig, ax = plt.subplots(figsize = (10, 6))
scatter = ax.scatter(pnl_df['inventory'], pnl_df['spread'],
    c = pnl_df['timestamp'], cmap = 'viridis', alpha = 0.5, s = 10)
ax.set_xlabel('Inventory (shares)')
ax.set_ylabel('Spread ($)')
ax.set_title('Spread vs Inventory (Color = Time)')
plt.colorbar(scatter, label = 'Time')
ax.grid(True, alpha = 0.3)
plt.tight_layout()
plt.show()
\`\`\`

### Payment for Order Flow (PFOF) Analysis

\`\`\`python
class PFOFAnalyzer:
    """
    Analyzes profitability of Payment for Order Flow market making.
    """
    def __init__(self, pfof_cost: float = 0.002):
        """
        Initialize analyzer.
        
        Parameters:
        - pfof_cost: Cost per share paid to broker for order flow
        """
        self.pfof_cost = pfof_cost
    
    def analyze_retail_flow(self, orders: list, execution_quality: dict) -> dict:
        """
        Analyze profitability of retail order flow.
        
        Parameters:
        - orders: List of retail orders
        - execution_quality: {'spread_captured': float, 'adverse_selection_cost': float}
        
        Returns:
        - Profit analysis
        """
        total_shares = sum(o['size'] for o in orders)
        
        # Revenue
        spread_revenue = execution_quality['spread_captured'] * total_shares
        
        # Costs
        pfof_cost_total = self.pfof_cost * total_shares
        adverse_selection_cost = execution_quality['adverse_selection_cost'] * total_shares
        
        # Net profit
        net_profit = spread_revenue - pfof_cost_total - adverse_selection_cost
        net_profit_per_share = net_profit / total_shares if total_shares > 0 else 0
        
        return {
            'total_shares': total_shares,
            'spread_revenue': spread_revenue,
            'pfof_cost': pfof_cost_total,
            'adverse_selection_cost': adverse_selection_cost,
            'net_profit': net_profit,
            'net_profit_per_share': net_profit_per_share,
            'profit_margin': (net_profit / spread_revenue * 100) if spread_revenue > 0 else 0
        }

# Example analysis
analyzer = PFOFAnalyzer(pfof_cost=0.002)

retail_orders = [{'size': 100} for _ in range(10000)]  # 1M shares
execution_quality = {
    'spread_captured': 0.03,  # 3 cents per share
    'adverse_selection_cost': 0.005  # 0.5 cents (retail is less informed)
}

analysis = analyzer.analyze_retail_flow(retail_orders, execution_quality)
print("\\nPFOF Profitability Analysis")
print("=" * 70)
for key, value in analysis.items():
    if isinstance(value, float):
        if 'per_share' in key:
            print(f"{key:>30}: \\$\{value:.4f}
}")
        elif 'margin' in key:
print(f"{key:>30}: {value:.2f}%")
        else:
print(f"{key:>30}: \\$\{value:,.2f}")
    else:
print(f"{key:>30}: {value:,}")
\`\`\`

---

## Real-World Example: Jane Street

**Jane Street** is one of the world's leading quantitative trading firms, with a major focus on market making across equities, ETFs, options, bonds, and cryptocurrencies.

**Market Making Operations:**
- **ETFs:** Major liquidity provider for over 500 ETFs globally (especially Vanguard, iShares).
- **Options:** One of the largest options market makers (equity options, index options).
- **Fixed Income:** Market making in corporate bonds and treasuries.

**Technology:**
- **OCaml:** Uses functional programming language OCaml for trading systems (unusual choice, prioritizes correctness).
- **Low Latency:** Co-located at all major exchanges, sub-millisecond execution.
- **Risk Management:** Sophisticated real-time systems to monitor inventory, Greeks, and P&L across thousands of instruments.

**Strategy:**
- **Statistical Arbitrage:** Combines market making with stat arb (e.g., ETF vs. constituents arbitrage).
- **Inventory Management:** Aggressively hedges with futures and correlated instruments.
- **Selective Quoting:** Pulls quotes during high volatility or news events.

**Scale:**
- **Volume:** Trades billions of dollars daily.
- **Instruments:** Makes markets in 10,000+ securities globally.
- **Employees:** ~2,000+ employees (many PhDs in math, physics, CS).

**Revenue:**
- **2022:** Estimated $2-3 billion in revenue (private firm, not disclosed).
- **Sources:** Spread capture (50%), rebates (20%), arbitrage profits (30%).

**Regulatory Compliance:**
- **Market Maker Obligations:** Registered as DMM (Designated Market Maker) on NYSE for certain securities.
- **Best Execution:** Must provide competitive quotes and fill rates.
- **Surveillance:** Subject to FINRA and exchange surveillance for manipulation.

---

## Hands-on Exercise: Build a Simple Market Maker

**Task:** Implement a market maker with basic inventory management and adaptive spread.

**Requirements:**1. Start with neutral inventory (0 shares).
2. Post bid and ask quotes around current midpoint.
3. Adjust spread based on inventory (widen when inventory is large).
4. Implement position limits (±5,000 shares).
5. Track P&L over 1,000 simulated time steps.

**Hints:**
- Use exponential utility for inventory aversion: spread_adj = k × inventory²
- Simulate random fills (Poisson arrivals).
- Compare performance with and without inventory management.

\`\`\`python
# Your implementation here
class SimpleMarketMaker:
    def __init__(self, base_spread, inventory_aversion):
        self.base_spread = base_spread
        self.inventory_aversion = inventory_aversion
        # ... complete the implementation
\`\`\`

---

## Common Pitfalls

1. **Ignoring Inventory Risk:** Focusing only on spread capture without managing inventory accumulation. Can lead to catastrophic losses during price moves.

2. **Static Spreads:** Not adjusting spreads based on volatility, inventory, or adverse selection signals. Markets are dynamic; spreads must be too.

3. **Over-Quoting During News:** Leaving quotes active during earnings releases or major news events. High adverse selection risk.

4. **Insufficient Speed:** In competitive markets, being even 100 microseconds slower can mean getting picked off by informed traders.

5. **Poor Risk Controls:** Lacking hard position limits or loss limits. A single bug or market event can wipe out months of profits.

---

## Production Checklist

1. **Real-Time Inventory Tracking:** Sub-millisecond updates to position across all venues and instruments.

2. **Dynamic Spread Adjustment:** Algorithms that adjust quotes based on volatility (VIX), order flow (VPIN), inventory, and time of day.

3. **Multi-Venue Quoting:** Presence on all relevant venues (exchanges, dark pools) with smart order routing.

4. **Hedging Infrastructure:** Automated hedging with futures, ETFs, or baskets when inventory exceeds thresholds.

5. **Adverse Selection Monitoring:** Real-time VPIN calculation, news feed monitoring (NLP), immediate quote cancellation on detection.

6. **Position and Loss Limits:** Hard limits enforced at the system level (not just alerts). Automatic flatting if breached.

7. **Performance Attribution:** Daily breakdown of P&L by source (spread, rebates, adverse selection, hedging costs).

---

## Regulatory Considerations

1. **Market Maker Registration:** If quoting as a designated market maker, must meet uptime and quote quality obligations (e.g., 90% of trading day).

2. **Best Execution:** Quotes must be competitive. Cannot systematically worsen prices to maximize rebates (violates Reg NMS).

3. **Quote Stuffing:** Placing and rapidly canceling large numbers of quotes can be deemed manipulative. Must have legitimate quoting intent.

4. **PFOF Disclosure:** If receiving PFOF, must disclose to clients and demonstrate execution quality (Rule 606 reports).

5. **Spoofing:** Placing quotes with intent to cancel before execution to manipulate prices is illegal (Dodd-Frank Act).
`,
};
