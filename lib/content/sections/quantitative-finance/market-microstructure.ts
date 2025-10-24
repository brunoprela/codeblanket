export const marketMicrostructure = {
  id: 'market-microstructure',
  title: 'Market Microstructure Theory',
  content: `
# Market Microstructure Theory

## Introduction

Market microstructure examines how trading mechanisms, information flows, and market participants interact to determine asset prices and liquidity. Unlike traditional finance (which assumes frictionless markets), microstructure recognizes transaction costs, information asymmetry, and strategic behavior as fundamental drivers of price formation.

**Why microstructure matters:**
- **Execution costs**: Bid-ask spreads and market impact can consume 0.5-2% of portfolio value annually
- **Price discovery**: How information is incorporated into prices (efficient markets hypothesis test)
- **High-frequency trading**: Microsecond advantages worth billions in profits
- **Market design**: Exchange competition, dark pools, payment for order flow

This section covers order types, bid-ask spread components, order book dynamics, market impact models, and optimal execution algorithms-essential knowledge for quantitative traders, execution desk professionals, and market makers.

---

## Order Types and Market Structure

### Basic Order Types

**1. Market Orders**

Execute immediately at best available price (top of book).

**Advantages:**
- Guaranteed execution (fills immediately)
- Simple to implement

**Disadvantages:**
- Price uncertainty (slippage)
- Pays bid-ask spread (takes liquidity)
- Market impact for large orders

**Example:** Buy 1000 AAPL at market.
- Best ask: $180.05 for 500 shares
- Next ask: $180.06 for 300 shares  
- Next ask: $180.07 for 200 shares
- **Fill**: 500 @ $180.05, 300 @ $180.06, 200 @ $180.07
- **Average price**: $180.057 (walked up order book)

**2. Limit Orders**

Execute only at specified price or better.

**Advantages:**
- Price certainty (won't pay more than limit)
- Provides liquidity (earns rebate on some exchanges)
- No adverse selection risk (controls execution price)

**Disadvantages:**
- Execution uncertainty (may not fill)
- Opportunity cost (miss trade if price moves away)
- Queue position matters (FIFO priority)

**Example:** Buy 1000 AAPL limit $180.00.
- Current ask: $180.05
- Order sits in queue at $180.00 bid
- If price falls to $180.00, fills at $180.00 or better
- If price rises to $180.10, no fill (missed trade)

**3. Stop Orders**

Converts to market order when trigger price reached.

**Stop-loss**: Sell stop below current price (limit losses).
**Stop-buy**: Buy stop above current price (momentum entry).

**Example:** Own AAPL at $180, set stop-loss at $175.
- If price hits $175, converts to market sell order
- May fill below $175 if price gapping down (no price guarantee!)

**4. Stop-Limit Orders**

Converts to **limit** order (not market) when stop triggered.

**Advantage**: Price protection (won't sell below limit).
**Risk**: May not execute if price gaps through limit.

**5. Iceberg (Hidden) Orders**

Large order displayed incrementally (e.g., show 100 shares of 10,000 total).

**Purpose**: Hide order size to prevent information leakage.
**Risk**: Gives away size over time (smart algos detect icebergs).

### Order Routing and Venues

**Exchanges:**
- **NYSE, NASDAQ**: Traditional lit exchanges (public order book)
- **BATS, IEX**: Alternative exchanges (IEX has 350μs speed bump to discourage HFT)

**Dark Pools:**
- Private venues where orders hidden until executed
- Institutional-focused (large block trades)
- ~40% of U.S. equity volume trades dark

**Payment for Order Flow (PFOF):**
- Retail brokers (Robinhood, Schwab) sell order flow to market makers (Citadel, Virtu)
- Market makers profit from spread, pay broker $0.001-0.003 per share
- Controversy: Conflict of interest (brokers incentivized to route to highest bidder, not best execution)

---

## Bid-Ask Spread

### Spread Components

The bid-ask spread \(S = P_{ask} - P_{bid}\) compensates market makers for three costs:

**1. Order Processing Costs**

Fixed costs per trade: clearing, settlement, exchange fees, technology.

**Typical**: $0.01-0.02 per share for liquid stocks.

**2. Inventory Risk**

Market makers hold risky inventory between buy and sell.

**Example:**
- Market maker buys 1000 AAPL at $180 (accumulates inventory)
- Before selling, AAPL drops to $179
- Loss: $1,000 (inventory risk)

**Compensation**: Spread must compensate for volatility risk.

**Formula** (Stoll 1978):
\[
S = 2 \\sigma \\sqrt{\\frac{T}{\\pi}}
\]
Where \(\\sigma\) = volatility, \(T\) = holding period.

**3. Adverse Selection**

Informed traders know more than market maker → trade against market maker → market maker loses.

**Example:**
- Market maker quotes AAPL $180.00 bid / $180.10 ask
- Informed trader knows positive earnings surprise → buys at $180.10 (market maker sells at $180.10)
- Earnings released, AAPL jumps to $185 → market maker lost $4.90 (would've been worth $185, sold for $180.10)

**Kyle (1985) model**: Adverse selection cost proportional to \(\\lambda \\times Q\), where \(\\lambda\) = information asymmetry, \(Q\) = order size.

**Total spread:**
\[
S = \\text{Processing} + \\text{Inventory} + \\text{Adverse Selection}
\]
\[
S \\approx c + \\alpha \\sigma + \\lambda Q
\]

**Typical breakdown:**
- Processing: 20% (liquid stocks), 40% (illiquid stocks)
- Inventory: 30%
- Adverse selection: 50%

### Spread Determinants

**Wider spreads for:**
- High volatility stocks (\(\\uparrow \\sigma\) → more inventory risk)
- Low volume stocks (less competition among market makers)
- Small-cap stocks (more information asymmetry)
- During market stress (liquidity dries up)

**Tighter spreads for:**
- Large-cap stocks (AAPL, MSFT: 1-2 cent spreads)
- High-frequency competition (HFT narrows spreads)
- Exchange rebates (maker-taker pricing incentivizes liquidity provision)

---

## Order Book Dynamics

### Order Book Structure

**Limit order book** displays supply (asks) and demand (bids) at each price level.

**Example AAPL order book:**

| Bid Size | Bid Price | Ask Price | Ask Size |
|----------|-----------|-----------|----------|
| 200 | $179.98 | $180.00 | 500 |
| 500 | $179.97 | $180.01 | 300 |
| 300 | $179.96 | $180.02 | 400 |
| 1000 | $179.95 | $180.05 | 600 |

**Best bid**: $179.98 (highest buy price)  
**Best ask**: $180.00 (lowest sell price)  
**Spread**: $0.02 (1 cent on each side)

**Depth**: Total quantity at each level (200 + 500 + 300 + 1000 = 2000 shares within $0.03 of mid).

### Order Book Imbalance

**Order flow imbalance** predicts short-term price movements.

**Formula:**
\[
\\text{Imbalance} = \\frac{\\text{Bid Volume} - \\text{Ask Volume}}{\\text{Bid Volume} + \\text{Ask Volume}}
\]

**Example:**
- Bid volume (within 5 ticks): 2,000 shares
- Ask volume (within 5 ticks): 1,000 shares
- Imbalance: (2000 - 1000) / (2000 + 1000) = +0.33 (33% bid-heavy)

**Interpretation:** Positive imbalance → buying pressure → price likely to rise (next trade more likely at ask).

**Empirical evidence:**
- Imbalance predicts next trade direction with 60-70% accuracy
- Predictive power decays within seconds (information short-lived)
- HFT traders exploit imbalance signals

### Microstructure Noise

**Microstructure noise**: Short-term price deviations from fundamental value due to trading frictions.

**Sources:**
- Bid-ask bounce (price bounces between bid and ask as trades alternate)
- Inventory effects (market makers adjust quotes based on inventory)
- Discreteness (minimum tick size constrains prices)

**Example of bid-ask bounce:**
- True value: $100.00
- Bid: $99.90, Ask: $100.10
- Trade sequence: Buy @ $100.10, Sell @ $99.90, Buy @ $100.10, Sell @ $99.90
- Observed volatility: $0.20 (but true value unchanged!)
- **Noise**: 67% of observed variance is microstructure (Hansen-Lunde, 2006)

**Filtering noise:**
- Use mid-quote instead of last price: \(\\text{Mid} = (\\text{Bid} + \\text{Ask}) / 2\)
- Time-weighted average (TWAP)
- Kalman filter (separate signal from noise)

---

## Market Impact

### Temporary vs Permanent Impact

**Temporary impact** (liquidity effect):
- Price moves adversely during execution
- Reverts after order completion
- Caused by: Inventory effects, urgency premium

**Permanent impact** (information effect):
- Price change persists after execution
- Market learns from order flow
- Caused by: Informed trading, fundamental information

**Example:** Buy 100,000 AAPL (1% of daily volume).
- Pre-trade mid: $180.00
- During execution: Avg fill $180.50 (temporary impact: $0.50)
- 30 minutes later: $180.20 (permanent impact: $0.20)
- Temporary component: $0.30 (reverted)

### Square-Root Law of Market Impact

**Almgren-Chriss (2000) model**: Market impact scales with \(\\sqrt{\\text{order size}}\).

\[
\\text{Impact} = \\eta \\sigma \\sqrt{\\frac{Q}{V}}
\]

**Variables:**
- \(\\eta\): Market impact coefficient (0.1-1.0, higher for illiquid stocks)
- \(\\sigma\): Daily volatility
- \(Q\): Order size (shares)
- \(V\): Average daily volume (shares)

**Example:** Buy 50,000 AAPL.
- \(\\sigma = 2\\%\) daily, \(V = 50M\) shares/day, \(\\eta = 0.5\)
- Impact = \(0.5 \\times 0.02 \\times \\sqrt{50,000 / 50,000,000}\)
- Impact = \(0.01 \\times \\sqrt{0.001} = 0.01 \\times 0.0316 = 0.0316\\%\)
- On $180 stock: $180 × 0.0316% = $0.057 per share
- **Total cost**: 50,000 × $0.057 = $2,850

**Why square-root?**
- Linear impact would imply splitting order into N parts reduces cost to zero (arbitrage)
- Square-root captures diminishing returns to splitting (information still leaks)

### Execution Strategies

**1. VWAP (Volume-Weighted Average Price)**

Trade proportionally to market volume distribution throughout day.

**Goal**: Match average market price (no information leakage).

**Example:** Execute 100,000 shares over full day.
- Hour 1 (10% of daily volume): Buy 10,000 shares
- Hour 2 (15% of daily volume): Buy 15,000 shares
- ... (proportional to volume profile)

**Advantage**: Benchmarkable (easy to compare vs VWAP).  
**Disadvantage**: Doesn't optimize for market impact (mechanical execution).

**2. TWAP (Time-Weighted Average Price)**

Trade evenly over time window.

**Example:** 100,000 shares over 6 hours = 16,667 shares/hour (constant rate).

**Advantage**: Simple, predictable.  
**Disadvantage**: Ignores volume (trades same amount in low-volume periods → high impact).

**3. Implementation Shortfall (Almgren-Chriss)**

Minimize expected cost + risk penalty.

**Objective:**
[
\\min \\mathbb{E}[\\text{Cost}] + \\lambda \\times \\text{Var}[\\text{Cost}]
]

Where:
- Cost = Execution price - decision price
- (\\lambda): Risk aversion (higher → trade slower to reduce variance)

**Solution**: Trade faster when risk aversion low, slower when high.

**4. Arrival Price**

Measure execution vs price at order arrival (decision point).

**Benchmark**: Did we execute better or worse than "doing nothing"?

**Components:**
- Market impact cost (unavoidable)
- Timing cost (price moved before we executed)
- Opportunity cost (didn't execute, price moved away)

---

## High-Frequency Trading (HFT)

### HFT Strategies

**1. Market Making**

Continuously quote bid and ask, profit from spread.

**Example:** Quote AAPL $180.00 / $180.02.
- Buy at $180.00, sell at $180.02 → profit $0.02 per round-trip
- Do this 10,000 times/day → $20,000 profit

**Key**: Manage inventory risk (don't accumulate large position).

**Inventory management:**
- If long 5,000 shares: Skew quotes ($179.99 bid / $180.01 ask) to encourage selling
- If short 5,000 shares: Skew quotes ($180.01 bid / $180.03 ask) to encourage buying

**2. Latency Arbitrage**

Exploit stale quotes on slow exchanges.

**Example:**
- Exchange A (fast): AAPL ask updated to $180.10 (bad news arrived)
- Exchange B (slow): AAPL still shows $180.00 ask (stale quote)
- HFT: Buy at $180.00 on B, sell at $180.09 on A → profit $0.09

**Arms race**: Co-location, microwave towers, laser beams (light speed advantage).

**3. Statistical Arbitrage**

Pairs trading, mean reversion on millisecond timescales.

**Example:** AAPL and SPY usually move together (correlation 0.9).
- AAPL jumps +0.1% but SPY unchanged (temporary deviation)
- HFT: Buy SPY, short AAPL (bet on reversion)
- 100ms later: SPY catches up +0.1% → close positions, profit

**4. Order Anticipation**

Detect large institutional orders, trade ahead.

**Example:**
- Detect 10 iceberg orders on AAPL (100 shares displayed, 10,000 total each)
- Infer: Large institutional buyer (total 100,000+ shares)
- HFT: Buy AAPL ahead of institution, sell to institution at higher price

**Controversy**: Is this "front-running" (illegal) or "legal prediction"? (Flash Boys debate)

### HFT Impact on Markets

**Positive:**
- Tighter spreads (competition drives spreads from $0.10 to $0.01)
- Deeper liquidity (HFT provides 50%+ of displayed liquidity)
- Faster price discovery (information incorporated in milliseconds)

**Negative:**
- Flash crashes (2010: Dow dropped 1000 points in minutes)
- Fragility (HFT pulls liquidity during stress → "ghost liquidity")
- Arms race (billions spent on speed, zero-sum game)
- Retail disadvantage (HFT earns 0.1 cent/share from payment for order flow)

---

## Python Implementation

### Order Book Analysis

\`\`\`python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

class OrderBook:
    """
    Simplified limit order book simulator.
    """
    
    def __init__(self):
        self.bids = {}  # {price: quantity}
        self.asks = {}  # {price: quantity}
        self.trade_history = []
    
    def add_order(self, side, price, quantity):
        """Add limit order to book."""
        if side == 'bid':
            self.bids[price] = self.bids.get(price, 0) + quantity
        else:  # ask
            self.asks[price] = self.asks.get(price, 0) + quantity
    
    def market_order(self, side, quantity):
        """Execute market order (walk the book)."""
        fills = []
        remaining = quantity
        
        if side == 'buy':
            # Buy at asks (ascending price)
            prices = sorted(self.asks.keys())
            for price in prices:
                available = self.asks[price]
                fill_qty = min(remaining, available)
                fills.append((price, fill_qty))
                remaining -= fill_qty
                self.asks[price] -= fill_qty
                if self.asks[price] == 0:
                    del self.asks[price]
                if remaining == 0:
                    break
        else:  # sell
            # Sell at bids (descending price)
            prices = sorted(self.bids.keys(), reverse=True)
            for price in prices:
                available = self.bids[price]
                fill_qty = min(remaining, available)
                fills.append((price, fill_qty))
                remaining -= fill_qty
                self.bids[price] -= fill_qty
                if self.bids[price] == 0:
                    del self.bids[price]
                if remaining == 0:
                    break
        
        avg_price = sum(p * q for p, q in fills) / sum(q for p, q in fills) if fills else None
        self.trade_history.append({
            'side': side,
            'quantity': quantity,
            'fills': fills,
            'avg_price': avg_price,
            'slippage': remaining
        })
        
        return fills, avg_price, remaining
    
    def get_spread(self):
        """Calculate bid-ask spread."""
        if not self.bids or not self.asks:
            return None
        best_bid = max(self.bids.keys())
        best_ask = min(self.asks.keys())
        return best_ask - best_bid
    
    def get_mid_price(self):
        """Calculate mid-price."""
        if not self.bids or not self.asks:
            return None
        return (max(self.bids.keys()) + min(self.asks.keys())) / 2
    
    def get_depth(self, levels=5):
        """Get order book depth."""
        bid_prices = sorted(self.bids.keys(), reverse=True)[:levels]
        ask_prices = sorted(self.asks.keys())[:levels]
        
        return {
            'bids': [(p, self.bids[p]) for p in bid_prices],
            'asks': [(p, self.asks[p]) for p in ask_prices]
        }
    
    def display(self):
        """Display order book."""
        depth = self.get_depth(5)
        
        print("\\n" + "="*50)
        print("ORDER BOOK")
        print("="*50)
        print(f"Spread: \${self.get_spread(): .2f
}")
print(f"Mid Price: \${self.get_mid_price():.2f}")
print("\\nAsks (Sell Orders):")
for price, qty in reversed(depth['asks']):
    print(f"  \${price:.2f}  |  {qty} shares")
print("-" * 50)
print("Bids (Buy Orders):")
for price, qty in depth['bids']:
    print(f"  \${price:.2f}  |  {qty} shares")

# Example: Build order book
book = OrderBook()

# Add limit orders
book.add_order('bid', 179.98, 200)
book.add_order('bid', 179.97, 500)
book.add_order('bid', 179.96, 300)
book.add_order('bid', 179.95, 1000)

book.add_order('ask', 180.00, 500)
book.add_order('ask', 180.01, 300)
book.add_order('ask', 180.02, 400)
book.add_order('ask', 180.05, 600)

book.display()

# Execute market buy order
print("\\n" + "=" * 50)
print("EXECUTING MARKET BUY: 1000 shares")
print("=" * 50)
fills, avg_price, slippage = book.market_order('buy', 1000)
print(f"\\nFills:")
for price, qty in fills:
    print(f"  {qty} shares @ \${price:.2f}")
print(f"\\nAverage Price: \${avg_price:.4f}")
print(f"Slippage: {slippage} shares (unfilled)")

book.display()
\`\`\`

### Market Impact Estimation

\`\`\`python
def market_impact_almgren_chriss(order_size, daily_volume, volatility, eta=0.5):
    """
    Estimate market impact using Almgren-Chriss square-root model.
    
    Parameters:
    - order_size: Number of shares to trade
    - daily_volume: Average daily volume (shares)
    - volatility: Daily volatility (decimal, e.g., 0.02 for 2%)
    - eta: Market impact coefficient (0.1-1.0)
    
    Returns:
    - impact_pct: Impact as percentage of price
    - impact_bps: Impact in basis points
    """
    participation_rate = order_size / daily_volume
    impact_pct = eta * volatility * np.sqrt(participation_rate)
    impact_bps = impact_pct * 10000
    
    return impact_pct, impact_bps

# Example: Calculate impact for various order sizes
print("\\n" + "="*50)
print("MARKET IMPACT ANALYSIS")
print("="*50)

stock_price = 180
daily_volume = 50_000_000  # 50M shares
volatility = 0.02  # 2% daily
eta = 0.5

order_sizes = [10_000, 50_000, 100_000, 500_000, 1_000_000]

results = []
for order_size in order_sizes:
    impact_pct, impact_bps = market_impact_almgren_chriss(
        order_size, daily_volume, volatility, eta
    )
    impact_dollars = stock_price * impact_pct
    total_cost = order_size * impact_dollars
    
    results.append({
        'Order Size': f'{order_size:,}',
        '% of ADV': f'{order_size/daily_volume*100:.2f}%',
        'Impact (bps)': f'{impact_bps:.1f}',
        'Impact ($)': f'\${impact_dollars: .4f}',
'Total Cost': f'\${total_cost:,.0f}'
    })

df = pd.DataFrame(results)
print(df.to_string(index = False))

print("\\nKEY INSIGHT: Impact scales with √(order size)")
print("Doubling order size increases impact by √2 = 1.41× (not 2×)")
\`\`\`

---

## Real-World Applications

### 1. **Optimal Execution for Institutional Orders**

**Problem:** Pension fund needs to buy $500M AAPL over 5 days without moving market.

**Solution:** Almgren-Chriss implementation shortfall algorithm.
- Optimize tradeoff: Urgency (finish quickly) vs market impact (trade slowly)
- Dynamic adjustment: If price moves favorably, accelerate; if unfavorable, slow down
- VWAP slicing: Trade proportionally to intraday volume curve

**Result:** Save 5-15 bps vs naive execution (on $500M, that's $250k-750k).

### 2. **HFT Market Making**

**Strategy:** Quote tight spreads, profit from bid-ask, manage inventory.

**Example:**
- Quote 10,000 stocks simultaneously
- Spread: 1 cent per stock
- Volume: 100 round-trips/second across all stocks = 6,000/minute
- Profit: 6,000 × $0.01 = $60/minute = $3,600/hour = $28,800/day (before costs)
- **Annual**: $7.2M (realistic for small HFT shop)

**Risk management:**
- Inventory limits: Max 10,000 shares per stock (avoid large overnight positions)
- Skew quotes when inventory imbalanced
- Cancel orders if volatility spikes (VIX > 30)

### 3. **Transaction Cost Analysis (TCA)**

**Measure execution quality:**
- Implementation shortfall: Execution price vs decision price
- VWAP: Did we beat VWAP?
- Arrival price: Cost vs doing nothing

**Example TCA:**
- Order: Buy 100,000 AAPL, decision price $180.00
- Execution: Avg fill $180.15
- VWAP (full day): $180.10
- **Shortfall**: $180.15 - $180.00 = $0.15 (15 bps)
- **vs VWAP**: $180.15 - $180.10 = $0.05 (5 bps underperformance)

**Action**: Switch execution algo or broker if consistently underperforming.

---

## Key Takeaways

1. **Bid-ask spread** has three components: processing costs (20%), inventory risk (30%), adverse selection (50%)
2. **Market impact** scales as √(order size)-large orders need careful execution to minimize cost
3. **Order book imbalance** predicts short-term price moves (60-70% accuracy)-HFT traders exploit this
4. **Microstructure noise** accounts for 50-70% of observed high-frequency volatility-use mid-quotes to filter
5. **HFT provides liquidity** (tighter spreads, deeper books) but can withdraw during stress (flash crashes)
6. **Optimal execution** algorithms (Almgren-Chriss, VWAP, TWAP) save 5-15 bps on institutional trades
7. **Transaction cost analysis (TCA)** is essential-measure shortfall, compare to benchmarks, improve execution
8. **Dark pools** and payment for order flow raise conflict-of-interest concerns but dominate retail order flow

Understanding microstructure enables quantitative traders to minimize execution costs, market makers to optimize quote placement, and regulators to design fair market structures.
`,
};
