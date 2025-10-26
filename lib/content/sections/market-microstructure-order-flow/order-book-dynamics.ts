export const orderBookDynamics = {
    title: 'Order Book Dynamics',
    id: 'order-book-dynamics',
    content: `
# Order Book Dynamics

## Introduction

The **order book** is the central data structure of modern markets - a real-time ledger showing all buy and sell orders at different price levels. Understanding order book dynamics is crucial for:

- **Traders**: Reading order flow and market sentiment
- **Quants**: Building predictive models from order book data
- **Engineers**: Designing systems that process millions of order book updates per second
- **Market Makers**: Managing inventory based on book shape

**This Section**: We dive deep into order book structure, Level 1/2/3 data, imbalance metrics, and how to reconstruct order books from raw tick data.

---

## Order Book Structure

### Basic Components

**Price Levels**:
- Discrete prices where orders rest
- Bid side: Buy orders (want to buy at specific prices)
- Ask side: Sell orders (want to sell at specific prices)

**Depth**:
- Quantity at each price level
- Total: Sum across all levels
- Cumulative: Running sum from best price outward

\`\`\`python
"""
Order Book Representation
"""

from dataclasses import dataclass, field
from typing import Dict, List
from decimal import Decimal
import heapq

@dataclass
class PriceLevel:
    """Single price level in order book"""
    price: Decimal
    quantity: int
    num_orders: int = 1
    
    def __repr__(self):
        return f"${float(self.price): .2f
} x { self.quantity } ({ self.num_orders })"

class OrderBook:
"""
    Complete order book with bid and ask sides
"""
    
    def __init__(self, symbol: str):
self.symbol = symbol
        # Bids: Max heap(highest price first)
self.bids: Dict[Decimal, PriceLevel] = {}
        # Asks: Min heap(lowest price first)
self.asks: Dict[Decimal, PriceLevel] = {}
self.last_update_time = 0
    
    def add_bid(self, price: Decimal, quantity: int):
"""Add or update bid level"""
if price in self.bids:
    self.bids[price].quantity += quantity
self.bids[price].num_orders += 1
        else:
self.bids[price] = PriceLevel(price, quantity)
    
    def add_ask(self, price: Decimal, quantity: int):
"""Add or update ask level"""
if price in self.asks:
    self.asks[price].quantity += quantity
self.asks[price].num_orders += 1
        else:
self.asks[price] = PriceLevel(price, quantity)
    
    def remove_bid(self, price: Decimal, quantity: int):
"""Remove quantity from bid level"""
if price in self.bids:
    self.bids[price].quantity -= quantity
if self.bids[price].quantity <= 0:
                del self.bids[price]
    
    def remove_ask(self, price: Decimal, quantity: int):
"""Remove quantity from ask level"""
if price in self.asks:
    self.asks[price].quantity -= quantity
if self.asks[price].quantity <= 0:
                del self.asks[price]
    
    def get_best_bid(self) -> Decimal | None:
"""Highest bid price"""
return max(self.bids.keys()) if self.bids else None
    
    def get_best_ask(self) -> Decimal | None:
"""Lowest ask price"""
return min(self.asks.keys()) if self.asks else None
    
    def get_mid_price(self) -> Decimal | None:
"""Mid-point between best bid and ask"""
bid = self.get_best_bid()
ask = self.get_best_ask()
if bid and ask:
return (bid + ask) / 2
return None
    
    def get_spread(self) -> Decimal | None:
"""Bid-ask spread"""
bid = self.get_best_bid()
ask = self.get_best_ask()
if bid and ask:
return ask - bid
return None
    
    def get_spread_bps(self) -> float | None:
"""Spread in basis points"""
spread = self.get_spread()
mid = self.get_mid_price()
if spread and mid and mid > 0:
return float(spread / mid * 10000)
return None
    
    def get_depth(self, side: str, levels: int = 10) -> List[PriceLevel]:
"""
        Get order book depth

Args:
side: 'bid' or 'ask'
levels: Number of price levels to return
"""
if side == 'bid':
            # Sort descending(best bid first)
prices = sorted(self.bids.keys(), reverse = True)[:levels]
return [self.bids[p] for p in prices]
        else:  # ask
            # Sort ascending(best ask first)
prices = sorted(self.asks.keys())[:levels]
return [self.asks[p] for p in prices]
    
    def get_total_volume(self, side: str, levels: int = 10) -> int:
"""Total volume in top N levels"""
depth = self.get_depth(side, levels)
return sum(level.quantity for level in depth)
    
    def get_weighted_mid(self, depth: int = 5) -> Decimal | None:
"""
Volume - weighted mid price
        Uses top N levels on each side
"""
bid_depth = self.get_depth('bid', depth)
ask_depth = self.get_depth('ask', depth)

if not bid_depth or not ask_depth:
return None
        
        # Volume - weighted average bid
bid_value = sum(float(level.price) * level.quantity for level in bid_depth)
    bid_volume = sum(level.quantity for level in bid_depth)
    vwab = Decimal(bid_value / bid_volume) if bid_volume > 0 else Decimal(0)
        
        # Volume - weighted average ask
ask_value = sum(float(level.price) * level.quantity for level in ask_depth)
    ask_volume = sum(level.quantity for level in ask_depth)
    vwaa = Decimal(ask_value / ask_volume) if ask_volume > 0 else Decimal(0)
        
        # Weighted mid
return (vwab + vwaa) / 2
    
    def display(self, levels: int = 10):
"""Pretty print order book"""
print(f"\\n{'='*60}")
print(f"Order Book: {self.symbol}")
print(f"{'='*60}")
print(f"{'BIDS':<30} | {'ASKS':<30}")
print(f"{'-'*30} | {'-'*30}")

bid_depth = self.get_depth('bid', levels)
ask_depth = self.get_depth('ask', levels)

max_rows = max(len(bid_depth), len(ask_depth))

for i in range(max_rows):
    bid_str = f"{bid_depth[i]}" if i < len(bid_depth) else ""
ask_str = f"{ask_depth[i]}" if i < len(ask_depth) else ""
print(f"{bid_str:<30} | {ask_str:<30}")
        
        # Summary stats
spread = self.get_spread()
spread_bps = self.get_spread_bps()
mid = self.get_mid_price()

print(f"\\n{'-'*60}")
if mid:
    print(f"Mid: ${float(mid):.4f}")
if spread:
    print(f"Spread: ${float(spread):.4f} ({spread_bps:.2f} bps)")
print(f"{'='*60}\\n")

# Example usage
book = OrderBook('AAPL')

# Add bid orders
book.add_bid(Decimal('150.00'), 500)
book.add_bid(Decimal('149.99'), 300)
book.add_bid(Decimal('149.98'), 1000)
book.add_bid(Decimal('149.97'), 200)
book.add_bid(Decimal('149.96'), 800)

# Add ask orders
book.add_ask(Decimal('150.01'), 400)
book.add_ask(Decimal('150.02'), 600)
book.add_ask(Decimal('150.03'), 250)
book.add_ask(Decimal('150.04'), 900)
book.add_ask(Decimal('150.05'), 150)

book.display(levels = 5)
\`\`\`

---

## Market Data Levels

### Level 1: Best Bid and Offer (BBO)

**Contains**:
- Best bid price and size
- Best ask price and size
- Last trade price and size

\`\`\`python
"""
Level 1 Market Data
"""

@dataclass
class Level1Data:
    """BBO - Best Bid and Offer"""
    symbol: str
    timestamp: float
    
    # Best bid
    bid_price: Decimal | None
    bid_size: int | None
    
    # Best ask
    ask_price: Decimal | None
    ask_size: int | None
    
    # Last trade
    last_price: Decimal | None
    last_size: int | None
    
    @property
    def mid_price(self) -> Decimal | None:
        if self.bid_price and self.ask_price:
            return (self.bid_price + self.ask_price) / 2
        return None
    
    @property
    def spread(self) -> Decimal | None:
        if self.bid_price and self.ask_price:
            return self.ask_price - self.bid_price
        return None
    
    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'bid': float(self.bid_price) if self.bid_price else None,
            'bid_size': self.bid_size,
            'ask': float(self.ask_price) if self.ask_price else None,
            'ask_size': self.ask_size,
            'last': float(self.last_price) if self.last_price else None,
            'last_size': self.last_size,
            'mid': float(self.mid_price) if self.mid_price else None,
            'spread': float(self.spread) if self.spread else None,
        }

def extract_level1(book: OrderBook, timestamp: float) -> Level1Data:
    """Extract Level 1 data from order book"""
    best_bid = book.get_best_bid()
    best_ask = book.get_best_ask()
    
    bid_size = book.bids[best_bid].quantity if best_bid else None
    ask_size = book.asks[best_ask].quantity if best_ask else None
    
    return Level1Data(
        symbol=book.symbol,
        timestamp=timestamp,
        bid_price=best_bid,
        bid_size=bid_size,
        ask_price=best_ask,
        ask_size=ask_size,
        last_price=None,  # Would come from trade feed
        last_size=None
    )
\`\`\`

**Use Cases**:
- Real-time price displays
- Basic trading decisions
- Retail trading apps (Robinhood)
- Low-bandwidth scenarios

**Limitations**:
- No depth information
- Can't see hidden liquidity
- No queue position
- Misses large orders slightly away from BBO

### Level 2: Full Order Book Depth

**Contains**:
- All price levels (bid and ask)
- Total quantity at each level
- Number of orders at each level (sometimes)

\`\`\`python
"""
Level 2 Market Data
"""

@dataclass
class Level2Data:
    """Full order book depth"""
    symbol: str
    timestamp: float
    bids: List[tuple[Decimal, int, int]]  # (price, quantity, num_orders)
    asks: List[tuple[Decimal, int, int]]
    
    def get_cumulative_volume(self, side: str) -> List[tuple[Decimal, int]]:
        """Cumulative volume from best price outward"""
        levels = self.bids if side == 'bid' else self.asks
        
        cumulative = []
        total = 0
        for price, qty, _ in levels:
            total += qty
            cumulative.append((price, total))
        
        return cumulative
    
    def get_volume_at_price(self, side: str, target_price: Decimal) -> int:
        """Get volume at specific price"""
        levels = self.bids if side == 'bid' else self.asks
        
        for price, qty, _ in levels:
            if price == target_price:
                return qty
        return 0
    
    def get_vwap(self, side: str, depth: int = 10) -> Decimal:
        """Volume-weighted average price for top N levels"""
        levels = (self.bids if side == 'bid' else self.asks)[:depth]
        
        total_value = sum(float(price) * qty for price, qty, _ in levels)
        total_volume = sum(qty for _, qty, _ in levels)
        
        return Decimal(total_value / total_volume) if total_volume > 0 else Decimal(0)

def extract_level2(book: OrderBook, timestamp: float, depth: int = 20) -> Level2Data:
    """Extract Level 2 data from order book"""
    bid_depth = book.get_depth('bid', depth)
    ask_depth = book.get_depth('ask', depth)
    
    bids = [(level.price, level.quantity, level.num_orders) for level in bid_depth]
    asks = [(level.price, level.quantity, level.num_orders) for level in ask_depth]
    
    return Level2Data(
        symbol=book.symbol,
        timestamp=timestamp,
        bids=bids,
        asks=asks
    )
\`\`\`

**Use Cases**:
- Algorithmic trading (depth-aware execution)
- Market impact estimation
- Liquidity analysis
- Professional trading platforms (Bloomberg, thinkorswim)

**Visualization**:
- Depth chart (cumulative volume by price)
- Heatmap (volume intensity at each level)
- Ladder display (traditional DOM - Depth of Market)

### Level 3: Individual Orders

**Contains**:
- Every individual order
- Order ID, price, quantity, timestamp
- Can track specific orders over time

\`\`\`python
"""
Level 3 Market Data
"""

@dataclass
class Order:
    """Individual order"""
    order_id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    price: Decimal
    quantity: int
    timestamp: float
    
    def __hash__(self):
        return hash(self.order_id)

@dataclass
class Level3Data:
    """Individual orders"""
    symbol: str
    timestamp: float
    bid_orders: List[Order]
    ask_orders: List[Order]
    
    def get_orders_at_price(self, side: str, price: Decimal) -> List[Order]:
        """Get all orders at specific price"""
        orders = self.bid_orders if side == 'BUY' else self.ask_orders
        return [o for o in orders if o.price == price]
    
    def get_queue_position(self, order_id: str) -> tuple[int, int] | None:
        """
        Get queue position for order
        Returns (position, total_orders_at_level)
        """
        # Find order
        all_orders = self.bid_orders + self.ask_orders
        target_order = next((o for o in all_orders if o.order_id == order_id), None)
        
        if not target_order:
            return None
        
        # Get orders at same price
        same_price = self.get_orders_at_price(target_order.side, target_order.price)
        
        # Sort by timestamp (FIFO)
        same_price.sort(key=lambda x: x.timestamp)
        
        # Find position
        position = next(i for i, o in enumerate(same_price) if o.order_id == order_id)
        
        return (position + 1, len(same_price))  # 1-indexed
    
    def aggregate_to_level2(self) -> Level2Data:
        """Aggregate Level 3 to Level 2"""
        # Group bid orders by price
        bid_levels = {}
        for order in self.bid_orders:
            if order.price not in bid_levels:
                bid_levels[order.price] = []
            bid_levels[order.price].append(order)
        
        # Group ask orders by price
        ask_levels = {}
        for order in self.ask_orders:
            if order.price not in ask_levels:
                ask_levels[order.price] = []
            ask_levels[order.price].append(order)
        
        # Create Level 2 tuples
        bids = [
            (price, sum(o.quantity for o in orders), len(orders))
            for price, orders in sorted(bid_levels.items(), reverse=True)
        ]
        
        asks = [
            (price, sum(o.quantity for o in orders), len(orders))
            for price, orders in sorted(ask_levels.items())
        ]
        
        return Level2Data(
            symbol=self.symbol,
            timestamp=self.timestamp,
            bids=bids,
            asks=asks
        )
\`\`\`

**Use Cases**:
- Queue position tracking
- Identifying large "icebergs" (hidden orders)
- Detecting order cancellations (spoofing detection)
- Research and forensics

**Availability**:
- Limited: Not all exchanges provide Level 3
- NASDAQ: ITCH feed (full Level 3)
- CME: MDP 3.0 (derivatives Level 3)
- Restricted: Some exchanges charge premium

---

## Order Book Imbalance

### Definition

**Imbalance** measures pressure between buy and sell sides:

\`\`\`
Imbalance = (Bid Volume - Ask Volume) / (Bid Volume + Ask Volume)
\`\`\`

**Range**: -1 to +1
- **+1**: Only bids (extreme buying pressure)
- **0**: Balanced
- **-1**: Only asks (extreme selling pressure)

### Calculation

\`\`\`python
"""
Order Book Imbalance Metrics
"""

def calculate_imbalance(book: OrderBook, depth: int = 5) -> float:
    """
    Calculate order book imbalance
    
    Positive: Bid pressure (likely price increase)
    Negative: Ask pressure (likely price decrease)
    """
    bid_volume = book.get_total_volume('bid', depth)
    ask_volume = book.get_total_volume('ask', depth)
    
    total = bid_volume + ask_volume
    if total == 0:
        return 0.0
    
    return (bid_volume - ask_volume) / total

def calculate_weighted_imbalance(book: OrderBook, depth: int = 5) -> float:
    """
    Price-weighted imbalance
    Gives more weight to levels closer to mid
    """
    mid = book.get_mid_price()
    if not mid:
        return 0.0
    
    bid_depth = book.get_depth('bid', depth)
    ask_depth = book.get_depth('ask', depth)
    
    # Weight by inverse distance from mid
    bid_weighted = sum(
        level.quantity / (abs(float(mid - level.price)) + 0.01)
        for level in bid_depth
    )
    
    ask_weighted = sum(
        level.quantity / (abs(float(mid - level.price)) + 0.01)
        for level in ask_depth
    )
    
    total = bid_weighted + ask_weighted
    if total == 0:
        return 0.0
    
    return (bid_weighted - ask_weighted) / total

def calculate_microprice(book: OrderBook) -> Decimal | None:
    """
    Microprice: Volume-weighted mid price
    
    Incorporates order book imbalance into price
    More accurate than simple mid for "true" price
    """
    best_bid = book.get_best_bid()
    best_ask = book.get_best_ask()
    
    if not best_bid or not best_ask:
        return None
    
    bid_size = book.bids[best_bid].quantity
    ask_size = book.asks[best_ask].quantity
    
    total_size = bid_size + ask_size
    if total_size == 0:
        return (best_bid + best_ask) / 2
    
    # Weight by opposite side volume (more ask volume → price closer to bid)
    microprice = (best_bid * ask_size + best_ask * bid_size) / total_size
    
    return microprice

# Example
book = OrderBook('AAPL')
book.add_bid(Decimal('150.00'), 1000)  # Large bid
book.add_bid(Decimal('149.99'), 500)
book.add_ask(Decimal('150.01'), 200)   # Small ask
book.add_ask(Decimal('150.02'), 300)

print(f"Simple mid: ${float(book.get_mid_price()): .4f}")
# Output: $150.005

microprice = calculate_microprice(book)
print(f"Microprice: ${float(microprice):.4f}")
# Output: $150.0067(closer to ask due to larger bid volume)

imbalance = calculate_imbalance(book, depth = 2)
print(f"Imbalance: {imbalance:.4f}")
# Output: +0.6(60 % bid pressure)
\`\`\`

### Predictive Power

**Research Findings**:
- Imbalance predicts short-term price moves (next 100ms - 1s)
- Stronger signal in liquid stocks
- Decays quickly (information lifespan ~1 second)

**Strategy**:
\`\`\`python
"""
Simple imbalance-based signal
"""

def generate_signal(book: OrderBook, threshold: float = 0.3) -> str:
    """
    Generate trading signal from imbalance
    
    Returns: 'BUY', 'SELL', or 'NEUTRAL'
    """
    imbalance = calculate_imbalance(book, depth=5)
    
    if imbalance > threshold:
        return 'BUY'  # Strong bid pressure
    elif imbalance < -threshold:
        return 'SELL'  # Strong ask pressure
    else:
        return 'NEUTRAL'

# Usage
signal = generate_signal(book, threshold=0.5)
print(f"Signal: {signal}")
# If imbalance > 0.5: "BUY" (heavy buying pressure)
\`\`\`

---

## Order Book Reconstruction from Tick Data

### NASDAQ ITCH Protocol

**ITCH**: NASDAQ's market data feed (binary format)

**Message Types**:
- **Add Order**: New limit order
- **Execute Order**: Order filled
- **Cancel Order**: Order cancelled
- **Replace Order**: Modify price/quantity
- **Delete Order**: Remove order

\`\`\`python
"""
Order Book Reconstruction from ITCH Messages
"""

from enum import Enum
from typing import Dict

class ITCHMessageType(Enum):
    ADD_ORDER = 'A'
    EXECUTE_ORDER = 'E'
    CANCEL_ORDER = 'X'
    DELETE_ORDER = 'D'
    REPLACE_ORDER = 'U'

@dataclass
class ITCHMessage:
    """Simplified ITCH message"""
    msg_type: ITCHMessageType
    timestamp: int  # Nanoseconds
    order_id: int
    side: str  # 'B' or 'S'
    price: int  # Price in 1/10000 of dollar (e.g., 1500000 = $150.00)
    quantity: int
    
class OrderBookRebuilder:
    """Rebuild order book from ITCH messages"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.book = OrderBook(symbol)
        self.orders: Dict[int, Order] = {}  # Track individual orders
    
    def process_message(self, msg: ITCHMessage):
        """Process single ITCH message"""
        
        if msg.msg_type == ITCHMessageType.ADD_ORDER:
            self._handle_add(msg)
        
        elif msg.msg_type == ITCHMessageType.EXECUTE_ORDER:
            self._handle_execute(msg)
        
        elif msg.msg_type == ITCHMessageType.CANCEL_ORDER:
            self._handle_cancel(msg)
        
        elif msg.msg_type == ITCHMessageType.DELETE_ORDER:
            self._handle_delete(msg)
        
        elif msg.msg_type == ITCHMessageType.REPLACE_ORDER:
            self._handle_replace(msg)
    
    def _handle_add(self, msg: ITCHMessage):
        """Add new order to book"""
        price = Decimal(msg.price) / 10000
        
        # Store order
        order = Order(
            order_id=str(msg.order_id),
            symbol=self.symbol,
            side='BUY' if msg.side == 'B' else 'SELL',
            price=price,
            quantity=msg.quantity,
            timestamp=msg.timestamp / 1e9
        )
        self.orders[msg.order_id] = order
        
        # Add to book
        if msg.side == 'B':
            self.book.add_bid(price, msg.quantity)
        else:
            self.book.add_ask(price, msg.quantity)
    
    def _handle_execute(self, msg: ITCHMessage):
        """Execute (fill) order"""
        if msg.order_id not in self.orders:
            return
        
        order = self.orders[msg.order_id]
        executed_qty = msg.quantity
        
        # Remove from book
        if order.side == 'BUY':
            self.book.remove_bid(order.price, executed_qty)
        else:
            self.book.remove_ask(order.price, executed_qty)
        
        # Update order quantity
        order.quantity -= executed_qty
        
        if order.quantity <= 0:
            del self.orders[msg.order_id]
    
    def _handle_cancel(self, msg: ITCHMessage):
        """Cancel portion of order"""
        if msg.order_id not in self.orders:
            return
        
        order = self.orders[msg.order_id]
        cancelled_qty = msg.quantity
        
        # Remove from book
        if order.side == 'BUY':
            self.book.remove_bid(order.price, cancelled_qty)
        else:
            self.book.remove_ask(order.price, cancelled_qty)
        
        # Update order
        order.quantity -= cancelled_qty
        
        if order.quantity <= 0:
            del self.orders[msg.order_id]
    
    def _handle_delete(self, msg: ITCHMessage):
        """Delete entire order"""
        if msg.order_id not in self.orders:
            return
        
        order = self.orders[msg.order_id]
        
        # Remove from book
        if order.side == 'BUY':
            self.book.remove_bid(order.price, order.quantity)
        else:
            self.book.remove_ask(order.price, order.quantity)
        
        # Delete order
        del self.orders[msg.order_id]
    
    def _handle_replace(self, msg: ITCHMessage):
        """Replace order (modify price or quantity)"""
        # Delete old order
        self._handle_delete(msg)
        # Add new order
        self._handle_add(msg)
    
    def get_snapshot(self) -> OrderBook:
        """Get current order book snapshot"""
        return self.book

# Example usage
rebuilder = OrderBookRebuilder('AAPL')

# Process messages
messages = [
    ITCHMessage(ITCHMessageType.ADD_ORDER, 1000000, 1, 'B', 1500000, 100),
    ITCHMessage(ITCHMessageType.ADD_ORDER, 1001000, 2, 'S', 1500100, 50),
    ITCHMessage(ITCHMessageType.EXECUTE_ORDER, 1002000, 1, 'B', 1500000, 30),
    ITCHMessage(ITCHMessageType.CANCEL_ORDER, 1003000, 2, 'S', 1500100, 20),
]

for msg in messages:
    rebuilder.process_message(msg)

book = rebuilder.get_snapshot()
book.display(levels=5)
\`\`\`

---

## Real-World: Exchange Order Book Analysis

### NASDAQ Order Book

**Characteristics**:
- **High frequency**: 1M+ messages per second (peak)
- **Deep**: 20-50 price levels typical
- **Fragmented**: Multiple venues (NASDAQ, BATS, IEX)

### CME Futures Order Book

**Characteristics**:
- **Thick**: Large volume at each level (institutional)
- **Spoofing**: Fake orders placed and cancelled (illegal)
- **Pro-rata**: Some products use pro-rata matching

### Crypto Order Books

**Example: Binance BTC/USDT**:
- **24/7**: Continuous trading
- **Thinner**: Less depth than traditional markets
- **Volatile**: Rapid changes in book shape

\`\`\`python
"""
Crypto order book via Binance WebSocket
"""

import asyncio
import websockets
import json

async def subscribe_orderbook(symbol: str = 'btcusdt'):
    """
    Subscribe to Binance order book updates
    """
    url = f"wss://stream.binance.com:9443/ws/{symbol}@depth"
    
    async with websockets.connect(url) as ws:
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            
            # Parse order book update
            bids = [(Decimal(price), float(qty)) for price, qty in data['bids']]
            asks = [(Decimal(price), float(qty)) for price, qty in data['asks']]
            
            print(f"Bids: {len(bids)}, Asks: {len(asks)}")
            
            # Top level
            if bids and asks:
                best_bid, bid_qty = bids[0]
                best_ask, ask_qty = asks[0]
                spread = best_ask - best_bid
                
                print(f"BBO: ${best_bid} x {bid_qty} / ${best_ask} x {ask_qty}")
                print(f"Spread: ${spread}")

# Run
# asyncio.run(subscribe_orderbook('btcusdt'))
\`\`\`

---

## Visualization

### Depth Chart

\`\`\`python
"""
Order book depth visualization
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_depth_chart(book: OrderBook, depth: int = 20):
    """
    Plot cumulative depth chart
    """
    bid_depth = book.get_depth('bid', depth)
    ask_depth = book.get_depth('ask', depth)
    
    # Cumulative bid volume
    bid_prices = [float(level.price) for level in bid_depth]
    bid_volumes = np.cumsum([level.quantity for level in bid_depth])
    
    # Cumulative ask volume
    ask_prices = [float(level.price) for level in ask_depth]
    ask_volumes = np.cumsum([level.quantity for level in ask_depth])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot bids (green, descending prices)
    ax.fill_between(bid_prices, 0, bid_volumes, alpha=0.3, color='green', step='post')
    ax.plot(bid_prices, bid_volumes, color='green', linewidth=2, label='Bids')
    
    # Plot asks (red, ascending prices)
    ax.fill_between(ask_prices, 0, ask_volumes, alpha=0.3, color='red', step='post')
    ax.plot(ask_prices, ask_volumes, color='red', linewidth=2, label='Asks')
    
    # Mid price line
    mid = book.get_mid_price()
    if mid:
        ax.axvline(float(mid), color='blue', linestyle='--', linewidth=1, label=f'Mid: ${float(mid): .2f}')

ax.set_xlabel('Price ($)')
ax.set_ylabel('Cumulative Volume')
ax.set_title(f'{book.symbol} Order Book Depth')
ax.legend()
ax.grid(True, alpha = 0.3)

plt.tight_layout()
return fig

# Example
book = OrderBook('AAPL')
# ... populate book ...
fig = plot_depth_chart(book)
# plt.show()
\`\`\`

---

## Key Takeaways

1. **Order book** is central data structure - all orders at all price levels
2. **Level 1** (BBO) for basic prices, **Level 2** for depth, **Level 3** for individual orders
3. **Imbalance** predicts short-term price moves - bid/ask volume ratio
4. **Microprice** more accurate than mid - volume-weighted price
5. **Reconstruction** from tick data requires careful message processing
6. **Visualization** (depth charts) reveals liquidity and market sentiment

**Next Section**: Price discovery process - how order flow translates to price movements and information incorporation into markets.
`
};

