export const marketMicrostructureFundamentals = {
  title: 'Market Microstructure Fundamentals',
  id: 'market-microstructure-fundamentals',
  content: `
# Market Microstructure Fundamentals

## Introduction

**Market microstructure** is the study of how markets operate at the microscopic level - how orders turn into trades, how prices form, and how different participants interact within the trading infrastructure.

**Why This Matters:**
- **For Traders**: Understanding market mechanics improves execution quality
- **For Engineers**: Building trading systems requires deep knowledge of market structure
- **For Quants**: Market microstructure effects can dominate signal in HFT strategies
- **For Everyone**: The "plumbing" of markets affects all participants

**This Module**: We dive into the nitty-gritty details of how exchanges work, order flow dynamics, and the infrastructure that makes modern markets possible.

---

## What is Market Microstructure?

### Definition

Market microstructure examines:
- **Price formation**: How orders translate to prices
- **Trading mechanisms**: How exchanges match buyers and sellers
- **Market participants**: Who trades and why
- **Information flow**: How news becomes prices
- **Transaction costs**: Bid-ask spreads, slippage, market impact

### Historical Context

**Pre-Electronic Era** (before 1990s):
- Floor traders shouting in pits
- Human specialists maintaining order books
- Phone orders to brokers
- Slow: Minutes to execute

**Electronic Revolution** (1990s-2000s):
- NASDAQ goes fully electronic (1971)
- NYSE introduces DOT system (1976)
- Decimalization (2001)
- Speed: Seconds to execute

**Modern Era** (2010s-present):
- High-frequency trading dominant
- Co-location at exchanges
- Microsecond latencies
- Fragmented liquidity across venues
- Speed: Microseconds to execute

---

## Market Participants

### 1. Retail Investors

**Who**: Individual investors (you and me)

**Characteristics**:
- Small order sizes (10-1000 shares)
- Infrequent trading
- Use brokers (Robinhood, E*TRADE, Interactive Brokers)
- Generally uninformed (no edge)

**Market Impact**:
- Provide liquidity to professional traders
- Orders routed via payment for order flow (PFOF)
- GameStop 2021: Retail can move markets collectively

### 2. Institutional Investors

**Who**: Mutual funds, pension funds, asset managers

**Characteristics**:
- Large order sizes (100K+ shares)
- Execute via algorithms (VWAP, TWAP, POV)
- Price sensitive (need good execution)
- Long-term holders (less informed on short-term moves)

**Challenges**:
- Market impact: Large orders move prices
- Information leakage: Others detect large orders
- Execution horizon: Hours to days

**Example**: BlackRock buying $50M of AAPL
- Can't buy all at once (would spike price)
- Slices into smaller orders over hours/days
- Uses algo to minimize impact

### 3. High-Frequency Traders (HFT)

**Who**: Citadel Securities, Virtu, Tower Research, Jump Trading

**Characteristics**:
- Ultra-fast: Microsecond reaction times
- High volume: 50%+ of US equity volume
- Co-located: Servers next to exchange
- Multiple strategies: Market making, arbitrage, momentum

**Strategies**:
- **Market making**: Provide liquidity, earn spread
- **Arbitrage**: Exploit price differences across venues
- **Latency arbitrage**: Trade on stale quotes before updates
- **Momentum ignition**: Trigger cascading orders

**Impact**:
- Tighter spreads (competition)
- More liquidity (always quoting)
- Increased volatility (rapid trading)
- Controversial: "Predatory" practices vs liquidity provision

### 4. Market Makers

**Who**: Designated market makers (DMMs), specialists

**Characteristics**:
- Obligated to quote bid and ask
- Provide continuous liquidity
- Manage inventory risk
- Earn spread as compensation

**NYSE Designated Market Makers (DMMs)**:
- Assigned stocks (e.g., GTS handles AAPL)
- Must maintain fair and orderly market
- Open and close auctions
- Step in during volatility

**NASDAQ Market Makers**:
- Multiple makers per stock (competitive)
- Not obligated but incentivized
- Tighter spreads due to competition

### 5. Proprietary Trading Firms

**Who**: Jane Street, DRW, IMC, Optiver

**Characteristics**:
- Trade own capital (not client money)
- Quantitative strategies
- Options market making (Jane Street dominates)
- Cross-asset arbitrage

**Focus**:
- Relative value: Pairs, spreads, baskets
- Statistical arbitrage
- Derivatives market making

---

## Order Types

### Basic Order Types

#### Market Order

**Definition**: Buy/sell immediately at best available price

\`\`\`python
"""
Market Order: Execute ASAP at any price
"""

class MarketOrder:
    def __init__(self, symbol: str, side: str, quantity: int):
        self.symbol = symbol
        self.side = side  # 'BUY' or 'SELL'
        self.quantity = quantity
        self.order_type = 'MARKET'
    
    def __repr__(self):
        return f"Market {self.side} {self.quantity} {self.symbol}"

# Example
order = MarketOrder('AAPL', 'BUY', 100)
# Execution: Immediately at $150.25 (best ask)
# Certainty: 100% filled
# Price: Uncertain (could be worse than expected)
\`\`\`

**Pros**:
- Guaranteed execution (if liquidity exists)
- Immediate

**Cons**:
- Price uncertainty (slippage)
- Can walk the book (multiple price levels)
- Expensive for large orders

**When to Use**:
- Small orders in liquid stocks
- Need immediate execution
- Not price sensitive

#### Limit Order

**Definition**: Buy/sell at specified price or better

\`\`\`python
"""
Limit Order: Only execute at limit price or better
"""

class LimitOrder:
    def __init__(self, symbol: str, side: str, quantity: int, limit_price: float):
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.limit_price = limit_price
        self.order_type = 'LIMIT'
    
    def __repr__(self):
        return f"Limit {self.side} {self.quantity} {self.symbol} @ \${self.limit_price}"

# Example
order = LimitOrder('AAPL', 'BUY', 100, 150.00)
# Execution: Only if someone sells at $150.00 or lower
# Certainty: May not fill
# Price: Guaranteed $150.00 or better
\`\`\`

**Pros**:
- Price certainty (won't pay more than limit)
- Control over execution price

**Cons**:
- Execution uncertainty (may not fill)
- Opportunity cost (miss the trade if price moves)

**When to Use**:
- Price sensitive
- Willing to wait
- Passive liquidity provision (earn spread)

#### Stop Order (Stop-Loss)

**Definition**: Becomes market order when stop price hit

\`\`\`python
"""
Stop Order: Trigger at stop price, execute as market order
"""

class StopOrder:
    def __init__(self, symbol: str, side: str, quantity: int, stop_price: float):
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.stop_price = stop_price
        self.order_type = 'STOP'
        self.triggered = False
    
    def check_trigger(self, current_price: float) -> bool:
        if self.side == 'SELL' and current_price <= self.stop_price:
            self.triggered = True
        elif self.side == 'BUY' and current_price >= self.stop_price:
            self.triggered = True
        return self.triggered
    
    def __repr__(self):
        status = "TRIGGERED" if self.triggered else "WAITING"
        return f"Stop {self.side} {self.quantity} {self.symbol} @ \${self.stop_price} [{status}]"

# Example: Stop-loss on long position
order = StopOrder('AAPL', 'SELL', 100, 145.00)
# Current price: $150
# Price drops to $145 → triggers → sells as market order
# Actual fill: $144.90 (slippage possible)
\`\`\`

**Use Case**: Risk management (exit losing positions)

**Risk**: Slippage if fast move (gap down past stop)

### Advanced Order Types

#### Immediate-or-Cancel (IOC)

**Definition**: Execute immediately, cancel unfilled portion

\`\`\`python
"""
IOC: Fill what you can right now, cancel the rest
"""

class IOCOrder:
    def __init__(self, symbol: str, side: str, quantity: int, limit_price: float):
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.limit_price = limit_price
        self.order_type = 'IOC'
        self.time_in_force = 'IOC'
    
    def __repr__(self):
        return f"IOC {self.side} {self.quantity} {self.symbol} @ \${self.limit_price}"

# Example
order = IOCOrder('AAPL', 'BUY', 1000, 150.00)
# Bid: 500 shares at $150.00
# Fill: 500 shares immediately
# Cancel: Remaining 500 shares
# Result: Partial fill (500/1000)
\`\`\`

**Use Case**:
- Testing liquidity without leaving footprint
- HFT strategies (no queue sitting)
- Avoid adverse selection (don't sit in book)

#### Fill-or-Kill (FOK)

**Definition**: Execute entire order immediately or cancel

\`\`\`python
"""
FOK: All or nothing, right now
"""

class FOKOrder:
    def __init__(self, symbol: str, side: str, quantity: int, limit_price: float):
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.limit_price = limit_price
        self.order_type = 'FOK'
        self.time_in_force = 'FOK'
    
    def __repr__(self):
        return f"FOK {self.side} {self.quantity} {self.symbol} @ \${self.limit_price}"

# Example
order = FOKOrder('AAPL', 'BUY', 1000, 150.00)
# Bid: 500 shares at $150.00
# Result: Cancel (not enough liquidity)
# Either fills 1000 shares or nothing
\`\`\`

**Use Case**:
- Arbitrage (need exact quantity for spread)
- Large orders in thin markets (all-or-nothing)

#### Pegged Orders

**Definition**: Price automatically adjusts relative to market

\`\`\`python
"""
Pegged Order: Dynamic pricing relative to BBO
"""

class PeggedOrder:
    def __init__(self, symbol: str, side: str, quantity: int, 
                 peg_type: str, offset: float = 0.0):
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.peg_type = peg_type  # 'MIDPOINT', 'PRIMARY', 'MARKET'
        self.offset = offset  # Offset from peg (+ or -)
        self.order_type = 'PEGGED'
    
    def calculate_price(self, bid: float, ask: float) -> float:
        if self.peg_type == 'MIDPOINT':
            base = (bid + ask) / 2
        elif self.peg_type == 'PRIMARY':
            base = bid if self.side == 'BUY' else ask
        elif self.peg_type == 'MARKET':
            base = ask if self.side == 'BUY' else bid
        return base + self.offset
    
    def __repr__(self):
        return f"Pegged {self.side} {self.quantity} {self.symbol} @ {self.peg_type}+{self.offset}"

# Example: Midpoint peg (dark pool style)
order = PeggedOrder('AAPL', 'BUY', 100, 'MIDPOINT', 0.0)
# Bid: $150.00, Ask: $150.10
# Order price: $150.05 (midpoint)
# Bid/ask changes → order price adjusts automatically
\`\`\`

**Use Cases**:
- Midpoint pegs: Price improvement in dark pools
- Primary pegs: Stay at top of book
- Market pegs: Ensure execution while controlling aggression

---

## Order Matching Algorithms

### Price-Time Priority (FIFO)

**Most Common**: Used by NYSE, NASDAQ, most exchanges

**Rules**:
1. **Price priority**: Best price goes first
2. **Time priority**: Within same price, first-in-first-out

\`\`\`python
"""
Price-Time Priority Matching Engine
"""

from collections import defaultdict, deque
from typing import Dict, Deque, Optional
import time

class PriceLevel:
    """Order queue at a specific price"""
    def __init__(self, price: float):
        self.price = price
        self.orders: Deque[LimitOrder] = deque()
        self.total_quantity = 0
    
    def add_order(self, order: LimitOrder):
        self.orders.append(order)
        self.total_quantity += order.quantity
    
    def remove_order(self, order: LimitOrder):
        self.orders.remove(order)
        self.total_quantity -= order.quantity

class OrderBook:
    """Simple price-time priority order book"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        # Price levels: {price: PriceLevel}
        self.bids: Dict[float, PriceLevel] = {}
        self.asks: Dict[float, PriceLevel] = {}
        self.trades = []
    
    def add_limit_order(self, order: LimitOrder) -> list:
        """Add order to book, return list of trades"""
        trades = []
        
        if order.side == 'BUY':
            # Match against asks (sellers)
            trades = self._match_order(order, self.asks, 'ASK')
            if order.quantity > 0:
                # Remaining quantity: Add to bid side
                self._add_to_book(order, self.bids)
        else:  # SELL
            # Match against bids (buyers)
            trades = self._match_order(order, self.bids, 'BID')
            if order.quantity > 0:
                # Remaining quantity: Add to ask side
                self._add_to_book(order, self.asks)
        
        return trades
    
    def _match_order(self, order: LimitOrder, book: dict, book_side: str) -> list:
        """Match incoming order against book"""
        trades = []
        
        # Get matchable prices (sorted)
        if order.side == 'BUY':
            # Buy order: Match against asks <= limit price
            prices = sorted([p for p in book.keys() if p <= order.limit_price])
        else:
            # Sell order: Match against bids >= limit price
            prices = sorted([p for p in book.keys() if p >= order.limit_price], reverse=True)
        
        for price in prices:
            if order.quantity == 0:
                break
            
            level = book[price]
            
            while level.orders and order.quantity > 0:
                resting_order = level.orders[0]
                
                # Execute trade
                trade_qty = min(order.quantity, resting_order.quantity)
                trade = {
                    'symbol': self.symbol,
                    'price': price,  # Resting order's price (price priority)
                    'quantity': trade_qty,
                    'buyer': order if order.side == 'BUY' else resting_order,
                    'seller': order if order.side == 'SELL' else resting_order,
                    'timestamp': time.time()
                }
                trades.append(trade)
                
                # Update quantities
                order.quantity -= trade_qty
                resting_order.quantity -= trade_qty
                
                # Remove filled order
                if resting_order.quantity == 0:
                    level.orders.popleft()
                    level.total_quantity -= trade_qty
            
            # Remove empty price level
            if len(level.orders) == 0:
                del book[price]
        
        return trades
    
    def _add_to_book(self, order: LimitOrder, book: dict):
        """Add order to book (didn't fully match)"""
        if order.limit_price not in book:
            book[order.limit_price] = PriceLevel(order.limit_price)
        book[order.limit_price].add_order(order)
    
    def get_best_bid(self) -> Optional[float]:
        """Highest bid price"""
        return max(self.bids.keys()) if self.bids else None
    
    def get_best_ask(self) -> Optional[float]:
        """Lowest ask price"""
        return min(self.asks.keys()) if self.asks else None
    
    def get_spread(self) -> Optional[float]:
        """Bid-ask spread"""
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        return (ask - bid) if (bid and ask) else None
    
    def display_book(self, depth: int = 5):
        """Display order book"""
        print(f"\\n=== Order Book: {self.symbol} ===")
        print(f"{'BIDS':<30} | {'ASKS':<30}")
        print("-" * 62)
        
        # Get sorted prices
        bid_prices = sorted(self.bids.keys(), reverse=True)[:depth]
        ask_prices = sorted(self.asks.keys())[:depth]
        
        max_rows = max(len(bid_prices), len(ask_prices))
        
        for i in range(max_rows):
            # Bid side
            if i < len(bid_prices):
                price = bid_prices[i]
                qty = self.bids[price].total_quantity
                bid_str = f"\${price:.2f} x {qty}"
            else:
                bid_str = ""
            
            # Ask side
if i < len(ask_prices):
    price = ask_prices[i]
qty = self.asks[price].total_quantity
ask_str = f"\${price:.2f}
} x { qty }"
            else:
ask_str = ""

print(f"{bid_str:<30} | {ask_str:<30}")

spread = self.get_spread()
if spread:
    print(f"\\nSpread: \\$\{spread:.2f}")

# Example usage
book = OrderBook('AAPL')

# Add sell orders(asks)
book.add_limit_order(LimitOrder('AAPL', 'SELL', 100, 150.10))
book.add_limit_order(LimitOrder('AAPL', 'SELL', 200, 150.11))
book.add_limit_order(LimitOrder('AAPL', 'SELL', 150, 150.10))  # Same price, behind first

# Add buy orders(bids)
book.add_limit_order(LimitOrder('AAPL', 'BUY', 100, 150.00))
book.add_limit_order(LimitOrder('AAPL', 'BUY', 200, 149.99))
book.add_limit_order(LimitOrder('AAPL', 'BUY', 50, 150.00))  # Same price, behind first

book.display_book()

# Output:
# === Order Book: AAPL ===
# BIDS | ASKS
# --------------------------------------------------------------
# $150.00 x 150 | $150.10 x 250
# $149.99 x 200 | $150.11 x 200
#
# Spread: $0.10

# Aggressive buy order(crosses spread)
trades = book.add_limit_order(LimitOrder('AAPL', 'BUY', 300, 150.11))

print(f"\\nExecuted {len(trades)} trades:")
for trade in trades:
    print(f"  {trade['quantity']} shares @ \\$\{trade['price']:.2f}")

# Trades:
# 100 shares @$150.10(first order at $150.10, time priority)
# 150 shares @$150.10(second order at $150.10)
# 50 shares @$150.11(moved to next price level)
\`\`\`

**Advantages**:
- Fair: First come, first served
- Simple: Easy to understand and implement
- Predictable: Transparent queue position

**Disadvantages**:
- Queue gaming: HFT race to front of queue
- Latency sensitive: Speed wins

### Pro-Rata Matching

**Used By**: Some futures exchanges (CME certain products), options exchanges

**Rules**:
1. **Price priority**: Best price still goes first
2. **Pro-rata allocation**: Within price level, distribute proportionally to size

\`\`\`python
"""
Pro-Rata Matching Algorithm
"""

def match_pro_rata(incoming_qty: int, resting_orders: list) -> list:
    """
    Distribute fills proportionally to order sizes
    """
    total_resting = sum(o.quantity for o in resting_orders)
    fills = []
    
    remaining = incoming_qty
    
    # Calculate pro-rata fills (rounded down)
    for order in resting_orders:
        pro_rata_qty = int((order.quantity / total_resting) * incoming_qty)
        fill_qty = min(pro_rata_qty, order.quantity, remaining)
        
        if fill_qty > 0:
            fills.append((order, fill_qty))
            remaining -= fill_qty
    
    # Distribute remainder (if any) using top-up or FIFO
    if remaining > 0:
        for order, filled in fills:
            if remaining == 0:
                break
            can_fill = order.quantity - filled
            top_up = min(can_fill, remaining)
            if top_up > 0:
                # Update fill
                fills = [(o, q + top_up if o == order else q) for o, q in fills]
                remaining -= top_up
    
    return fills

# Example
resting_orders = [
    LimitOrder('ES', 'SELL', 100, 4500.0),  # 100 / 400 = 25%
    LimitOrder('ES', 'SELL', 200, 4500.0),  # 200 / 400 = 50%
    LimitOrder('ES', 'SELL', 100, 4500.0),  # 100 / 400 = 25%
]

incoming = LimitOrder('ES', 'BUY', 150, 4500.0)

fills = match_pro_rata(150, resting_orders)

# Results:
# Order 1: 37 fills (25% of 150 = 37.5, rounded down)
# Order 2: 75 fills (50% of 150 = 75)
# Order 3: 38 fills (25% of 150 = 37.5, rounded down + remainder)
# Total: 150
\`\`\`

**Advantages**:
- Large orders get more fills (fair to size)
- Less queue gaming (can't jump ahead with small order)
- Less speed sensitive

**Disadvantages**:
- Complex: Harder to implement
- Unpredictable: Don't know exact fill percentage
- Encourages quote stuffing (place large orders to get pro-rata fill)

---

## Tick Size and Its Impact

**Tick size**: Minimum price increment

**Examples**:
- Stocks >$1: $0.01 tick
- Stocks <$1: $0.0001 tick
- Futures (ES): 0.25 points = $12.50 per contract
- Bitcoin: $0.01

### Impact on Market Quality

**Smaller Tick Size**:
- **Pros**: Tighter spreads, better price discovery
- **Cons**: More price levels, less depth per level, queue gaming

**Larger Tick Size**:
- **Pros**: More liquidity at each level, less gaming
- **Cons**: Wider spreads, worse execution prices

**Tick Size Pilot (SEC 2016-2018)**:
- Increased tick size for small-cap stocks ($0.05 vs $0.01)
- Goal: Encourage more liquidity
- Result: Wider spreads, worse execution, pilot ended

---

## Real-World: Exchange Matching Engines

### NASDAQ TotalView

**Architecture**:
- In-memory order book (< 1ms latency)
- ITCH protocol for market data (binary, fast)
- OUCH protocol for orders
- Co-located servers for HFT

**Capacity**:
- 100,000+ messages per second per symbol
- Millions of orders per second across all symbols

### NYSE Pillar

**Architecture**:
- Hybrid: Electronic + floor brokers
- DMM involvement in opening/closing auctions
- Price-time priority
- Millisecond latencies

**Market Model**:
- Continuous trading (9:30-16:00 ET)
- Opening auction (9:30)
- Closing auction (16:00)
- After-hours trading (extended hours)

---

## Production Patterns: Event-Driven Architecture

\`\`\`python
"""
Event-driven order processing system
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Callable
import queue
import threading
import time

class EventType(Enum):
    ORDER_NEW = "ORDER_NEW"
    ORDER_CANCEL = "ORDER_CANCEL"
    ORDER_MODIFY = "ORDER_MODIFY"
    TRADE = "TRADE"
    MARKET_DATA = "MARKET_DATA"

@dataclass
class Event:
    event_type: EventType
    timestamp: float
    data: dict

class EventQueue:
    """Thread-safe event queue"""
    def __init__(self):
        self.queue = queue.Queue()
    
    def put(self, event: Event):
        self.queue.put(event)
    
    def get(self, timeout: Optional[float] = None) -> Optional[Event]:
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

class EventProcessor:
    """Process events from queue"""
    
    def __init__(self, event_queue: EventQueue):
        self.event_queue = event_queue
        self.handlers = {}
        self.running = False
    
    def register_handler(self, event_type: EventType, handler: Callable):
        """Register event handler"""
        self.handlers[event_type] = handler
    
    def start(self):
        """Start processing events"""
        self.running = True
        self.thread = threading.Thread(target=self._process_loop)
        self.thread.start()
    
    def stop(self):
        """Stop processing"""
        self.running = False
        self.thread.join()
    
    def _process_loop(self):
        """Main event processing loop"""
        while self.running:
            event = self.event_queue.get(timeout=0.1)
            
            if event:
                handler = self.handlers.get(event.event_type)
                if handler:
                    try:
                        handler(event)
                    except Exception as e:
                        print(f"Error processing event: {e}")

# Example: Simple trading system
event_queue = EventQueue()
processor = EventProcessor(event_queue)

def handle_new_order(event: Event):
    order = event.data['order']
    print(f"[{event.timestamp}] New order: {order}")
    # Process order...

def handle_trade(event: Event):
    trade = event.data
    print(f"[{event.timestamp}] Trade: {trade['quantity']} @ \\$\{trade['price']}")
    # Update positions, PnL...

processor.register_handler(EventType.ORDER_NEW, handle_new_order)
processor.register_handler(EventType.TRADE, handle_trade)

processor.start()

# Simulate events
event_queue.put(Event(
    EventType.ORDER_NEW,
    time.time(),
    {'order': LimitOrder('AAPL', 'BUY', 100, 150.00)}
))

event_queue.put(Event(
    EventType.TRADE,
    time.time(),
    {'symbol': 'AAPL', 'price': 150.00, 'quantity': 100}
))

time.sleep(1)
processor.stop()
\`\`\`

---

## Key Takeaways

1. **Market microstructure** is the infrastructure and mechanics of how markets operate
2. **Multiple participants** with different motivations: retail, institutional, HFT, market makers
3. **Order types** provide different trade-offs between price and execution certainty
4. **Matching algorithms** (price-time priority, pro-rata) determine who gets filled
5. **Event-driven architecture** is essential for building production trading systems
6. **Understanding market mechanics** improves execution quality and system design

**Next Section**: We dive into order book dynamics - how the book evolves, imbalance as a signal, and reconstructing order books from tick data.
`,
};
