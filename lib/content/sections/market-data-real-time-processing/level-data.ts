export const levelData = {
  title: 'Level 1, Level 2, Level 3 Data',
  id: 'level-data',
  content: `
# Level 1, Level 2, Level 3 Data

## Introduction

Market data comes in three levels of depth, each revealing different aspects of market microstructure. Understanding these levels is crucial for algorithmic trading, market making, and building sophisticated trading systems.

**The Three Levels:**
- **Level 1 (L1)**: Best bid/offer (BBO) - top of book
- **Level 2 (L2)**: Full order book depth - multiple price levels
- **Level 3 (L3)**: Individual orders - complete transparency

**Why Levels Matter:**
- **Retail traders**: Use L1 for basic trading (99% of retail)
- **Professional traders**: Need L2 for liquidity analysis
- **Market makers**: Require L3 for order flow internalization
- **HFT firms**: Monitor L2/L3 for alpha signals

**Real-World Usage:**
- **NASDAQ TotalView**: L2 data for NASDAQ stocks ($15/month)
- **NYSE OpenBook**: L2 depth for NYSE stocks
- **CME MDP 3.0**: L2 futures data (free for exchange members)
- **Institutional**: L3 data requires exchange membership + expensive infrastructure

By the end of this section, you'll understand:
- L1/L2/L3 data structures and differences
- How to parse and process order books
- Building order book from L2 updates
- Calculating metrics from depth data
- Production-grade order book implementation

---

## Level 1 Data (Best Bid/Offer)

### L1 Data Structure

\`\`\`python
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

@dataclass
class Level1Quote:
    """Level 1: Best Bid and Offer (top of book)"""
    symbol: str
    exchange_timestamp: datetime
    receive_timestamp: datetime
    
    # Best bid (highest buy order)
    bid_price: Decimal
    bid_size: int
    bid_exchange: str = "NASDAQ"
    
    # Best ask/offer (lowest sell order)
    ask_price: Decimal
    ask_size: int
    ask_exchange: str = "NASDAQ"
    
    # Last trade
    last_price: Decimal
    last_size: int
    
    # Session data
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal
    volume: int
    
    @property
    def spread(self) -> Decimal:
        """Bid-ask spread in dollars"""
        return self.ask_price - self.bid_price
    
    @property
    def spread_bps(self) -> Decimal:
        """Spread in basis points"""
        mid = (self.bid_price + self.ask_price) / 2
        return (self.spread / mid) * Decimal('10000')
    
    @property
    def mid_price(self) -> Decimal:
        """Mid-point between bid and ask"""
        return (self.bid_price + self.ask_price) / 2
    
    @property
    def is_locked(self) -> bool:
        """Locked market: bid >= ask (abnormal)"""
        return self.bid_price >= self.ask_price
    
    @property
    def is_crossed(self) -> bool:
        """Crossed market: bid > ask (error condition)"""
        return self.bid_price > self.ask_price

# Example L1 quote
quote = Level1Quote(
    symbol="AAPL",
    exchange_timestamp=datetime(2024, 1, 15, 9, 30, 0, 123456),
    receive_timestamp=datetime(2024, 1, 15, 9, 30, 0, 125789),
    bid_price=Decimal("150.24"),
    bid_size=500,
    bid_exchange="NASDAQ",
    ask_price=Decimal("150.26"),
    ask_size=300,
    ask_exchange="ARCA",
    last_price=Decimal("150.25"),
    last_size=100,
    open_price=Decimal("150.00"),
    high_price=Decimal("151.00"),
    low_price=Decimal("149.50"),
    close_price=Decimal("150.25"),
    volume=1_250_000
)

print(f"Spread: ${quote.spread} ({quote.spread_bps} bps)")
print(f"Mid: ${quote.mid_price}")
print(f"Latency: {(quote.receive_timestamp - quote.exchange_timestamp).total_seconds() * 1000:.2f}ms")
\`\`\`

### L1 Use Cases

\`\`\`python
class Level1Strategy:
    """Simple trading strategies using L1 data only"""
    
    def __init__(self):
        self.positions = {}
        self.last_quote = {}
    
    def on_quote(self, quote: Level1Quote):
        """Process L1 quote update"""
        symbol = quote.symbol
        
        # 1. Spread Analysis
        if quote.spread_bps > 50:
            print(f"{symbol}: Wide spread {quote.spread_bps:.1f} bps - avoid trading")
        
        # 2. Locked/Crossed Market Detection
        if quote.is_crossed:
            print(f"{symbol}: CROSSED MARKET - bid={quote.bid_price} > ask={quote.ask_price}")
            # Arbitrage opportunity or data error
        
        # 3. Price Move Detection
        if symbol in self.last_quote:
            last = self.last_quote[symbol]
            price_change = quote.mid_price - last.mid_price
            price_change_pct = float(price_change / last.mid_price) * 100
            
            if abs(price_change_pct) > 0.5:  # 0.5% move
                print(f"{symbol}: Large move {price_change_pct:.2f}% "
                      f"(${last.mid_price} → ${quote.mid_price})")
        
        # 4. Imbalance Detection (basic)
        bid_value = float(quote.bid_price * quote.bid_size)
        ask_value = float(quote.ask_price * quote.ask_size)
        imbalance = (bid_value - ask_value) / (bid_value + ask_value)
        
        if imbalance > 0.6:  # Strong buy pressure
            print(f"{symbol}: Buy pressure (imbalance={imbalance:.2f})")
        elif imbalance < -0.6:  # Strong sell pressure
            print(f"{symbol}: Sell pressure (imbalance={imbalance:.2f})")
        
        self.last_quote[symbol] = quote

strategy = Level1Strategy()
strategy.on_quote(quote)
\`\`\`

---

## Level 2 Data (Order Book Depth)

### L2 Data Structure

\`\`\`python
from typing import List, Dict
from collections import defaultdict

@dataclass
class PriceLevel:
    """Single price level in order book"""
    price: Decimal
    size: int
    order_count: int = 1  # Number of orders at this level

@dataclass
class Level2Snapshot:
    """Full order book snapshot"""
    symbol: str
    timestamp: datetime
    
    # Bids: sorted descending by price (best bid first)
    bids: List[PriceLevel]
    
    # Asks: sorted ascending by price (best ask first)
    asks: List[PriceLevel]
    
    @property
    def best_bid(self) -> PriceLevel:
        return self.bids[0] if self.bids else None
    
    @property
    def best_ask(self) -> PriceLevel:
        return self.asks[0] if self.asks else None
    
    @property
    def spread(self) -> Decimal:
        if not self.best_bid or not self.best_ask:
            return Decimal('0')
        return self.best_ask.price - self.best_bid.price
    
    @property
    def mid_price(self) -> Decimal:
        if not self.best_bid or not self.best_ask:
            return Decimal('0')
        return (self.best_bid.price + self.best_ask.price) / 2
    
    def total_bid_volume(self, levels: int = None) -> int:
        """Total bid volume for top N levels"""
        levels = levels or len(self.bids)
        return sum(level.size for level in self.bids[:levels])
    
    def total_ask_volume(self, levels: int = None) -> int:
        """Total ask volume for top N levels"""
        levels = levels or len(self.asks)
        return sum(level.size for level in self.asks[:levels])
    
    def volume_imbalance(self, levels: int = 5) -> float:
        """Order imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol)"""
        bid_vol = self.total_bid_volume(levels)
        ask_vol = self.total_ask_volume(levels)
        total_vol = bid_vol + ask_vol
        
        if total_vol == 0:
            return 0.0
        
        return float(bid_vol - ask_vol) / float(total_vol)
    
    def depth_at_price(self, price: Decimal, side: str) -> int:
        """Volume available at specific price level"""
        levels = self.bids if side == 'bid' else self.asks
        for level in levels:
            if level.price == price:
                return level.size
        return 0

# Example L2 snapshot
snapshot = Level2Snapshot(
    symbol="AAPL",
    timestamp=datetime.now(),
    bids=[
        PriceLevel(Decimal("150.24"), 500, 3),
        PriceLevel(Decimal("150.23"), 800, 5),
        PriceLevel(Decimal("150.22"), 1200, 7),
        PriceLevel(Decimal("150.21"), 600, 4),
        PriceLevel(Decimal("150.20"), 1500, 10),
    ],
    asks=[
        PriceLevel(Decimal("150.26"), 300, 2),
        PriceLevel(Decimal("150.27"), 700, 4),
        PriceLevel(Decimal("150.28"), 1000, 6),
        PriceLevel(Decimal("150.29"), 500, 3),
        PriceLevel(Decimal("150.30"), 1200, 8),
    ]
)

print(f"Best Bid: {snapshot.best_bid.price} × {snapshot.best_bid.size}")
print(f"Best Ask: {snapshot.best_ask.price} × {snapshot.best_ask.size}")
print(f"Spread: ${snapshot.spread}")
print(f"Top 5 bid volume: {snapshot.total_bid_volume(5)}")
print(f"Top 5 ask volume: {snapshot.total_ask_volume(5)}")
print(f"Imbalance: {snapshot.volume_imbalance(5):.2f}")
\`\`\`

### Order Book Builder (Incremental Updates)

\`\`\`python
from enum import Enum

class UpdateType(Enum):
    ADD = "add"
    MODIFY = "modify"
    DELETE = "delete"

@dataclass
class OrderBookUpdate:
    """Incremental order book update"""
    symbol: str
    timestamp: datetime
    side: str  # 'bid' or 'ask'
    update_type: UpdateType
    price: Decimal
    size: int  # New size (0 for DELETE)

class OrderBookBuilder:
    """Build and maintain order book from incremental updates"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        
        # Order book: price -> size
        self.bids: Dict[Decimal, int] = {}  # price -> size
        self.asks: Dict[Decimal, int] = {}
        
        # Metadata
        self.last_update = None
        self.sequence_number = 0
        self.update_count = 0
    
    def process_snapshot(self, snapshot: Level2Snapshot):
        """Initialize from full snapshot"""
        self.bids = {level.price: level.size for level in snapshot.bids}
        self.asks = {level.price: level.size for level in snapshot.asks}
        self.last_update = snapshot.timestamp
        print(f"Snapshot loaded: {len(self.bids)} bids, {len(self.asks)} asks")
    
    def process_update(self, update: OrderBookUpdate):
        """Apply incremental update"""
        book = self.bids if update.side == 'bid' else self.asks
        
        if update.update_type == UpdateType.ADD:
            book[update.price] = update.size
        elif update.update_type == UpdateType.MODIFY:
            if update.price in book:
                book[update.price] = update.size
            else:
                book[update.price] = update.size  # Treat as ADD
        elif update.update_type == UpdateType.DELETE:
            book.pop(update.price, None)
        
        self.last_update = update.timestamp
        self.update_count += 1
    
    def get_snapshot(self, depth: int = 10) -> Level2Snapshot:
        """Generate current order book snapshot"""
        # Sort and convert to PriceLevel objects
        bids_sorted = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)
        asks_sorted = sorted(self.asks.items(), key=lambda x: x[0])
        
        bids = [PriceLevel(price, size) for price, size in bids_sorted[:depth]]
        asks = [PriceLevel(price, size) for price, size in asks_sorted[:depth]]
        
        return Level2Snapshot(
            symbol=self.symbol,
            timestamp=self.last_update,
            bids=bids,
            asks=asks
        )
    
    def best_bid_ask(self) -> tuple:
        """Get current BBO (L1 from L2 data)"""
        best_bid = max(self.bids.keys()) if self.bids else None
        best_ask = min(self.asks.keys()) if self.asks else None
        return (best_bid, best_ask)
    
    def validate(self) -> List[str]:
        """Validate order book integrity"""
        errors = []
        
        # Check for crossed book
        best_bid, best_ask = self.best_bid_ask()
        if best_bid and best_ask and best_bid >= best_ask:
            errors.append(f"Crossed book: bid={best_bid} >= ask={best_ask}")
        
        # Check for zero/negative sizes
        for price, size in list(self.bids.items()) + list(self.asks.items()):
            if size <= 0:
                errors.append(f"Invalid size at {price}: {size}")
        
        # Check for duplicate prices (shouldn't happen with dict)
        # Already handled by using dict
        
        return errors

# Usage
builder = OrderBookBuilder("AAPL")

# Initialize from snapshot
initial_snapshot = Level2Snapshot(
    symbol="AAPL",
    timestamp=datetime.now(),
    bids=[PriceLevel(Decimal("150.24"), 500)],
    asks=[PriceLevel(Decimal("150.26"), 300)]
)
builder.process_snapshot(initial_snapshot)

# Process updates
update1 = OrderBookUpdate(
    symbol="AAPL",
    timestamp=datetime.now(),
    side='bid',
    update_type=UpdateType.ADD,
    price=Decimal("150.23"),
    size=800
)
builder.process_update(update1)

# Get current snapshot
current = builder.get_snapshot()
print(f"Current BBO: {builder.best_bid_ask()}")
print(f"Updates processed: {builder.update_count}")
\`\`\`

---

## Level 3 Data (Individual Orders)

### L3 Data Structure

\`\`\`python
@dataclass
class Order:
    """Individual order (L3 data)"""
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    price: Decimal
    size: int
    timestamp: datetime
    trader_id: str = ""  # Sometimes available
    order_type: str = "limit"  # limit, market, stop, etc.

class Level3OrderBook:
    """Order-by-order book (L3)"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        
        # Order ID -> Order
        self.orders: Dict[str, Order] = {}
        
        # Price level -> List of order IDs (for aggregation to L2)
        self.bid_levels: Dict[Decimal, List[str]] = defaultdict(list)
        self.ask_levels: Dict[Decimal, List[str]] = defaultdict(list)
    
    def add_order(self, order: Order):
        """Add new order"""
        self.orders[order.order_id] = order
        
        # Add to price level
        if order.side == 'buy':
            self.bid_levels[order.price].append(order.order_id)
        else:
            self.ask_levels[order.price].append(order.order_id)
    
    def cancel_order(self, order_id: str):
        """Cancel order"""
        if order_id not in self.orders:
            return
        
        order = self.orders[order_id]
        
        # Remove from price level
        levels = self.bid_levels if order.side == 'buy' else self.ask_levels
        if order.price in levels:
            levels[order.price].remove(order_id)
            if not levels[order.price]:
                del levels[order.price]
        
        # Remove order
        del self.orders[order_id]
    
    def execute_order(self, order_id: str, executed_size: int):
        """Partially execute order"""
        if order_id not in self.orders:
            return
        
        order = self.orders[order_id]
        order.size -= executed_size
        
        if order.size <= 0:
            self.cancel_order(order_id)
    
    def get_level2_snapshot(self) -> Level2Snapshot:
        """Aggregate L3 to L2"""
        bids = []
        for price in sorted(self.bid_levels.keys(), reverse=True):
            order_ids = self.bid_levels[price]
            total_size = sum(self.orders[oid].size for oid in order_ids)
            bids.append(PriceLevel(price, total_size, len(order_ids)))
        
        asks = []
        for price in sorted(self.ask_levels.keys()):
            order_ids = self.ask_levels[price]
            total_size = sum(self.orders[oid].size for oid in order_ids)
            asks.append(PriceLevel(price, total_size, len(order_ids)))
        
        return Level2Snapshot(
            symbol=self.symbol,
            timestamp=datetime.now(),
            bids=bids,
            asks=asks
        )
    
    def order_flow_analysis(self) -> dict:
        """Analyze order flow from L3 data"""
        # Count orders by size category
        small_orders = sum(1 for o in self.orders.values() if o.size < 100)
        medium_orders = sum(1 for o in self.orders.values() if 100 <= o.size < 1000)
        large_orders = sum(1 for o in self.orders.values() if o.size >= 1000)
        
        return {
            'total_orders': len(self.orders),
            'small_orders': small_orders,
            'medium_orders': medium_orders,
            'large_orders': large_orders,
            'avg_order_size': sum(o.size for o in self.orders.values()) / len(self.orders) if self.orders else 0
        }

# Example L3 usage
l3_book = Level3OrderBook("AAPL")

# Add individual orders
l3_book.add_order(Order("ORD001", "AAPL", "buy", Decimal("150.24"), 100, datetime.now()))
l3_book.add_order(Order("ORD002", "AAPL", "buy", Decimal("150.24"), 200, datetime.now()))
l3_book.add_order(Order("ORD003", "AAPL", "buy", Decimal("150.23"), 500, datetime.now()))

# Aggregate to L2
l2_from_l3 = l3_book.get_level2_snapshot()
print(f"L2 from L3: {len(l2_from_l3.bids)} price levels")
print(f"Best bid: {l2_from_l3.best_bid.price} × {l2_from_l3.best_bid.size} ({l2_from_l3.best_bid.order_count} orders)")

# Analyze order flow
flow = l3_book.order_flow_analysis()
print(f"Order flow: {flow}")
\`\`\`

---

## Best Practices

1. **L1 for most strategies** - 90%+ of strategies only need BBO
2. **L2 for liquidity analysis** - Essential for market making and large orders
3. **L3 rarely needed** - Only for specialized microstructure research
4. **Validate order books** - Check for crossed markets, negative sizes
5. **Handle snapshots + updates** - Periodic snapshots, incremental updates
6. **Monitor latency** - exchange_timestamp vs receive_timestamp
7. **Use depth metrics** - Imbalance, total volume at top levels

---

## Data Level Comparison

| Feature | L1 | L2 | L3 |
|---------|----|----|----| 
| **Data** | Best bid/ask | Full depth | Individual orders |
| **Cost** | Free | $15-100/mo | Exchange membership |
| **Latency** | 50-100ms | 100-200ms | 200-500ms |
| **Bandwidth** | 1 KB/s | 10-50 KB/s | 100-500 KB/s |
| **Use Case** | Retail trading | Professional | Market making |

Now you can build order books from L1/L2/L3 data for any trading strategy!
`,
};
