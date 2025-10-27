export const orderBookSimulator = {
    title: 'Project: Order Book Simulator',
    id: 'order-book-simulator',
    content: `# Project: Order Book Simulator

## Introduction

This capstone project integrates all concepts from the Market Microstructure module into a **production-grade order book matching engine**. You'll build a complete trading venue simulator that:

- Implements price-time priority matching
- Supports all major order types (market, limit, IOC, FOK, stop, pegged)
- Generates Level 1, 2, and 3 market data feeds
- Integrates market making bots
- Tracks latency and performance metrics
- Provides real-time visualization
- Handles 100,000+ orders per second

This is the type of system that powers exchanges like NASDAQ, NYSE, and CME. By building it, you'll deeply understand market microstructure mechanics and gain experience with high-performance systems design.

## System Architecture

\`\`\`
┌─────────────────────────────────────────────────────────────────┐
│                     Order Book Simulator                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐         ┌──────────────┐                     │
│  │  Order       │────────>│  Matching    │                     │
│  │  Gateway     │         │  Engine      │                     │
│  └──────────────┘         └──────────────┘                     │
│         │                         │                             │
│         │                         v                             │
│         │                 ┌──────────────┐                     │
│         │                 │  Order Book  │                     │
│         │                 │  (Bid/Ask)   │                     │
│         │                 └──────────────┘                     │
│         │                         │                             │
│         v                         v                             │
│  ┌──────────────┐         ┌──────────────┐                     │
│  │  Market Data │         │   Trade      │                     │
│  │  Publisher   │         │   Tape       │                     │
│  └──────────────┘         └──────────────┘                     │
│         │                         │                             │
│         v                         v                             │
│  ┌──────────────────────────────────────┐                     │
│  │         Market Participants           │                     │
│  │  - Market Makers                      │                     │
│  │  - Retail Traders                     │                     │
│  │  - HFT Algorithms                     │                     │
│  └──────────────────────────────────────┘                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
\`\`\`

## Core Data Structures

### Order Representation

\`\`\`python
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from enum import Enum
import time
from collections import defaultdict
import heapq

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    IOC = "IOC"  # Immediate-Or-Cancel
    FOK = "FOK"  # Fill-Or-Kill
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    PEGGED = "PEGGED"  # Peg to NBBO midpoint

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class TimeInForce(Enum):
    DAY = "DAY"
    GTC = "GTC"  # Good-Till-Cancel
    IOC = "IOC"
    FOK = "FOK"

@dataclass
class Order:
    """
    Order representation with all fields required for production matching engine.
    """
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[float] = None  # None for market orders
    time_in_force: TimeInForce = TimeInForce.DAY
    
    # Conditional order fields
    stop_price: Optional[float] = None  # For stop orders
    peg_offset: Optional[float] = None  # For pegged orders
    
    # Execution tracking
    filled_quantity: int = 0
    remaining_quantity: int = field(init=False)
    avg_fill_price: float = 0.0
    
    # Timestamps (nanoseconds)
    timestamp_received: int = field(default_factory=time.perf_counter_ns)
    timestamp_in_book: Optional[int] = None
    timestamp_filled: Optional[int] = None
    
    # Metadata
    participant_id: str = ""
    client_order_id: str = ""
    
    def __post_init__(self):
        self.remaining_quantity = self.quantity
    
    def __lt__(self, other):
        """
        Comparison for priority queue (price-time priority).
        Buy side: Higher price = higher priority (reversed)
        Sell side: Lower price = higher priority
        Secondary: Earlier timestamp = higher priority
        """
        if self.side == OrderSide.BUY:
            if self.price == other.price:
                return self.timestamp_received < other.timestamp_received
            return self.price > other.price
        else:  # SELL
            if self.price == other.price:
                return self.timestamp_received < other.timestamp_received
            return self.price < other.price

@dataclass
class Trade:
    """Executed trade"""
    trade_id: str
    symbol: str
    price: float
    quantity: int
    buy_order_id: str
    sell_order_id: str
    timestamp: int
    aggressor_side: OrderSide  # Which order was aggressive

@dataclass
class MarketDataSnapshot:
    """Level 1/2/3 market data"""
    symbol: str
    timestamp: int
    
    # Level 1: Best bid/ask
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    best_bid_size: int = 0
    best_ask_size: int = 0
    
    # Level 2: Depth (price -> quantity at each level)
    bid_depth: Dict[float, int] = field(default_factory=dict)
    ask_depth: Dict[float, int] = field(default_factory=dict)
    
    # Level 3: Full order book (price -> list of orders)
    bid_orders: Dict[float, List[Order]] = field(default_factory=lambda: defaultdict(list))
    ask_orders: Dict[float, List[Order]] = field(default_factory=lambda: defaultdict(list))
    
    @property
    def spread(self) -> Optional[float]:
        """Calculate bid-ask spread"""
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None
    
    @property
    def midpoint(self) -> Optional[float]:
        """Calculate midpoint"""
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return None
\`\`\`

## Matching Engine Implementation

\`\`\`python
class OrderBook:
    """
    High-performance order book with price-time priority matching.
    
    Uses priority queues (heaps) for O(log n) insertions and O(1) best price access.
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        
        # Order storage: price level -> list of orders (FIFO at each price)
        self.buy_orders: Dict[float, List[Order]] = defaultdict(list)
        self.sell_orders: Dict[float, List[Order]] = defaultdict(list)
        
        # Priority queues for best price access
        self.buy_prices: List[float] = []  # Max heap (negative prices)
        self.sell_prices: List[float] = []  # Min heap
        
        # Order ID lookup
        self.orders: Dict[str, Order] = {}
        
        # Stop orders (activated when price reached)
        self.buy_stop_orders: Dict[float, List[Order]] = defaultdict(list)
        self.sell_stop_orders: Dict[float, List[Order]] = defaultdict(list)
        
        # Statistics
        self.total_orders = 0
        self.total_trades = 0
        self.total_volume = 0
    
    def add_order(self, order: Order):
        """Add order to book"""
        self.total_orders += 1
        self.orders[order.order_id] = order
        order.timestamp_in_book = time.perf_counter_ns()
        
        # Handle stop orders separately
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            if order.side == OrderSide.BUY:
                self.buy_stop_orders[order.stop_price].append(order)
            else:
                self.sell_stop_orders[order.stop_price].append(order)
            return
        
        # Add to appropriate side
        if order.side == OrderSide.BUY:
            self.buy_orders[order.price].append(order)
            if order.price not in self.buy_prices:
                heapq.heappush(self.buy_prices, -order.price)  # Negative for max heap
        else:
            self.sell_orders[order.price].append(order)
            if order.price not in self.sell_prices:
                heapq.heappush(self.sell_prices, order.price)  # Min heap
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order by ID"""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        
        # Remove from price level
        if order.side == OrderSide.BUY:
            if order.price in self.buy_orders:
                self.buy_orders[order.price] = [o for o in self.buy_orders[order.price] 
                                                if o.order_id != order_id]
                if not self.buy_orders[order.price]:
                    del self.buy_orders[order.price]
        else:
            if order.price in self.sell_orders:
                self.sell_orders[order.price] = [o for o in self.sell_orders[order.price] 
                                                 if o.order_id != order_id]
                if not self.sell_orders[order.price]:
                    del self.sell_orders[order.price]
        
        del self.orders[order_id]
        return True
    
    def get_best_bid(self) -> Optional[float]:
        """Get best bid price (highest buy price)"""
        while self.buy_prices:
            price = -self.buy_prices[0]  # Negative for max heap
            if price in self.buy_orders and self.buy_orders[price]:
                return price
            heapq.heappop(self.buy_prices)  # Remove stale price
        return None
    
    def get_best_ask(self) -> Optional[float]:
        """Get best ask price (lowest sell price)"""
        while self.sell_prices:
            price = self.sell_prices[0]
            if price in self.sell_orders and self.sell_orders[price]:
                return price
            heapq.heappop(self.sell_prices)  # Remove stale price
        return None
    
    def get_market_data(self, depth_levels: int = 10) -> MarketDataSnapshot:
        """Generate market data snapshot"""
        snapshot = MarketDataSnapshot(symbol=self.symbol, timestamp=time.perf_counter_ns())
        
        # Level 1: Best bid/ask
        snapshot.best_bid = self.get_best_bid()
        snapshot.best_ask = self.get_best_ask()
        
        if snapshot.best_bid:
            snapshot.best_bid_size = sum(o.remaining_quantity 
                                        for o in self.buy_orders[snapshot.best_bid])
        if snapshot.best_ask:
            snapshot.best_ask_size = sum(o.remaining_quantity 
                                        for o in self.sell_orders[snapshot.best_ask])
        
        # Level 2: Depth (top N price levels)
        sorted_buy_prices = sorted([p for p in self.buy_orders.keys() if self.buy_orders[p]], 
                                   reverse=True)[:depth_levels]
        sorted_sell_prices = sorted([p for p in self.sell_orders.keys() if self.sell_orders[p]])[:depth_levels]
        
        for price in sorted_buy_prices:
            snapshot.bid_depth[price] = sum(o.remaining_quantity for o in self.buy_orders[price])
        
        for price in sorted_sell_prices:
            snapshot.ask_depth[price] = sum(o.remaining_quantity for o in self.sell_orders[price])
        
        # Level 3: Full orders (for transparency)
        for price in sorted_buy_prices:
            snapshot.bid_orders[price] = self.buy_orders[price].copy()
        
        for price in sorted_sell_prices:
            snapshot.ask_orders[price] = self.sell_orders[price].copy()
        
        return snapshot

class MatchingEngine:
    """
    Order matching engine with price-time priority.
    
    Handles all order types and generates trade confirmations.
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.order_book = OrderBook(symbol)
        self.trades: List[Trade] = []
        self.trade_id_counter = 0
        
        # Performance metrics
        self.latency_samples: List[int] = []
    
    def submit_order(self, order: Order) -> Dict:
        """
        Submit order to matching engine.
        
        Returns dict with execution results.
        """
        start_time = time.perf_counter_ns()
        
        result = {
            'order_id': order.order_id,
            'status': 'UNKNOWN',
            'fills': [],
            'remaining_quantity': order.quantity
        }
        
        # Handle market orders
        if order.order_type == OrderType.MARKET:
            result = self._execute_market_order(order)
        
        # Handle IOC (Immediate-Or-Cancel)
        elif order.order_type == OrderType.IOC:
            result = self._execute_ioc_order(order)
        
        # Handle FOK (Fill-Or-Kill)
        elif order.order_type == OrderType.FOK:
            result = self._execute_fok_order(order)
        
        # Handle limit orders
        elif order.order_type == OrderType.LIMIT:
            result = self._execute_limit_order(order)
        
        # Handle pegged orders
        elif order.order_type == OrderType.PEGGED:
            result = self._execute_pegged_order(order)
        
        # Track latency
        end_time = time.perf_counter_ns()
        latency_ns = end_time - start_time
        self.latency_samples.append(latency_ns)
        result['latency_ns'] = latency_ns
        
        return result
    
    def _execute_market_order(self, order: Order) -> Dict:
        """Execute market order (buy/sell at best available price)"""
        fills = []
        remaining = order.quantity
        
        if order.side == OrderSide.BUY:
            # Buy market order: match against sell side (asks)
            while remaining > 0:
                best_ask = self.order_book.get_best_ask()
                if not best_ask:
                    break  # No liquidity
                
                # Match against best ask
                ask_orders = self.order_book.sell_orders[best_ask]
                if not ask_orders:
                    break
                
                # Match with first order at best ask (price-time priority)
                resting_order = ask_orders[0]
                match_quantity = min(remaining, resting_order.remaining_quantity)
                
                # Execute trade
                trade = self._create_trade(order, resting_order, best_ask, match_quantity)
                fills.append(trade)
                
                # Update quantities
                remaining -= match_quantity
                resting_order.remaining_quantity -= match_quantity
                resting_order.filled_quantity += match_quantity
                
                # Remove fully filled order
                if resting_order.remaining_quantity == 0:
                    ask_orders.pop(0)
                    resting_order.timestamp_filled = time.perf_counter_ns()
        
        else:  # SELL
            # Sell market order: match against buy side (bids)
            while remaining > 0:
                best_bid = self.order_book.get_best_bid()
                if not best_bid:
                    break
                
                bid_orders = self.order_book.buy_orders[best_bid]
                if not bid_orders:
                    break
                
                resting_order = bid_orders[0]
                match_quantity = min(remaining, resting_order.remaining_quantity)
                
                trade = self._create_trade(resting_order, order, best_bid, match_quantity)
                fills.append(trade)
                
                remaining -= match_quantity
                resting_order.remaining_quantity -= match_quantity
                resting_order.filled_quantity += match_quantity
                
                if resting_order.remaining_quantity == 0:
                    bid_orders.pop(0)
                    resting_order.timestamp_filled = time.perf_counter_ns()
        
        status = 'FILLED' if remaining == 0 else ('PARTIAL_FILL' if fills else 'REJECTED')
        
        return {
            'order_id': order.order_id,
            'status': status,
            'fills': fills,
            'remaining_quantity': remaining
        }
    
    def _execute_limit_order(self, order: Order) -> Dict:
        """Execute limit order (buy/sell at specified price or better)"""
        fills = []
        remaining = order.quantity
        
        # Try to match against book
        if order.side == OrderSide.BUY:
            while remaining > 0:
                best_ask = self.order_book.get_best_ask()
                if not best_ask or best_ask > order.price:
                    break  # No matchable liquidity
                
                ask_orders = self.order_book.sell_orders[best_ask]
                if not ask_orders:
                    break
                
                resting_order = ask_orders[0]
                match_quantity = min(remaining, resting_order.remaining_quantity)
                
                trade = self._create_trade(order, resting_order, best_ask, match_quantity)
                fills.append(trade)
                
                remaining -= match_quantity
                resting_order.remaining_quantity -= match_quantity
                resting_order.filled_quantity += match_quantity
                
                if resting_order.remaining_quantity == 0:
                    ask_orders.pop(0)
                    resting_order.timestamp_filled = time.perf_counter_ns()
        
        else:  # SELL
            while remaining > 0:
                best_bid = self.order_book.get_best_bid()
                if not best_bid or best_bid < order.price:
                    break
                
                bid_orders = self.order_book.buy_orders[best_bid]
                if not bid_orders:
                    break
                
                resting_order = bid_orders[0]
                match_quantity = min(remaining, resting_order.remaining_quantity)
                
                trade = self._create_trade(resting_order, order, best_bid, match_quantity)
                fills.append(trade)
                
                remaining -= match_quantity
                resting_order.remaining_quantity -= match_quantity
                resting_order.filled_quantity += match_quantity
                
                if resting_order.remaining_quantity == 0:
                    bid_orders.pop(0)
                    resting_order.timestamp_filled = time.perf_counter_ns()
        
        # If remaining quantity, add to book
        if remaining > 0:
            order.remaining_quantity = remaining
            self.order_book.add_order(order)
            status = 'PARTIAL_FILL' if fills else 'RESTING'
        else:
            status = 'FILLED'
        
        return {
            'order_id': order.order_id,
            'status': status,
            'fills': fills,
            'remaining_quantity': remaining
        }
    
    def _execute_ioc_order(self, order: Order) -> Dict:
        """Immediate-Or-Cancel: Execute immediately, cancel remainder"""
        result = self._execute_limit_order(order)
        
        # Cancel any remaining quantity
        if result['remaining_quantity'] > 0:
            self.order_book.cancel_order(order.order_id)
            result['status'] = 'PARTIAL_FILL_CANCELED' if result['fills'] else 'CANCELED'
        
        return result
    
    def _execute_fok_order(self, order: Order) -> Dict:
        """Fill-Or-Kill: Either fill completely or reject"""
        # Check if full quantity is available at acceptable price
        available_quantity = 0
        
        if order.side == OrderSide.BUY:
            for price in sorted(self.order_book.sell_orders.keys()):
                if price > order.price:
                    break
                available_quantity += sum(o.remaining_quantity 
                                         for o in self.order_book.sell_orders[price])
                if available_quantity >= order.quantity:
                    break
        else:  # SELL
            for price in sorted(self.order_book.buy_orders.keys(), reverse=True):
                if price < order.price:
                    break
                available_quantity += sum(o.remaining_quantity 
                                         for o in self.order_book.buy_orders[price])
                if available_quantity >= order.quantity:
                    break
        
        # If insufficient quantity, reject
        if available_quantity < order.quantity:
            return {
                'order_id': order.order_id,
                'status': 'REJECTED_FOK',
                'fills': [],
                'remaining_quantity': order.quantity,
                'reason': f'Insufficient liquidity ({available_quantity} < {order.quantity})'
            }
        
        # Execute as limit order (will fill completely)
        return self._execute_limit_order(order)
    
    def _execute_pegged_order(self, order: Order) -> Dict:
        """Pegged order: Price follows NBBO midpoint + offset"""
        snapshot = self.order_book.get_market_data()
        if not snapshot.midpoint:
            return {
                'order_id': order.order_id,
                'status': 'REJECTED',
                'fills': [],
                'remaining_quantity': order.quantity,
                'reason': 'No NBBO available for pegged order'
            }
        
        # Set price to midpoint + offset
        order.price = snapshot.midpoint + (order.peg_offset or 0)
        order.order_type = OrderType.LIMIT  # Convert to limit for execution
        
        return self._execute_limit_order(order)
    
    def _create_trade(self, buy_order: Order, sell_order: Order, 
                     price: float, quantity: int) -> Trade:
        """Create trade record"""
        self.trade_id_counter += 1
        trade = Trade(
            trade_id=f"T{self.trade_id_counter}",
            symbol=self.symbol,
            price=price,
            quantity=quantity,
            buy_order_id=buy_order.order_id,
            sell_order_id=sell_order.order_id,
            timestamp=time.perf_counter_ns(),
            aggressor_side=buy_order.side if buy_order.timestamp_received > sell_order.timestamp_received else sell_order.side
        )
        
        self.trades.append(trade)
        self.order_book.total_trades += 1
        self.order_book.total_volume += quantity
        
        return trade
    
    def get_statistics(self) -> Dict:
        """Get matching engine statistics"""
        if not self.latency_samples:
            return {}
        
        sorted_latencies = sorted(self.latency_samples)
        
        return {
            'total_orders': self.order_book.total_orders,
            'total_trades': self.order_book.total_trades,
            'total_volume': self.order_book.total_volume,
            'latency_mean_us': sum(self.latency_samples) / len(self.latency_samples) / 1000,
            'latency_p50_us': sorted_latencies[len(sorted_latencies) // 2] / 1000,
            'latency_p99_us': sorted_latencies[int(len(sorted_latencies) * 0.99)] / 1000,
            'latency_max_us': sorted_latencies[-1] / 1000
        }
\`\`\`

## Market Making Bot Integration

\`\`\`python
class SimpleMarketMaker:
    """
    Simple market making bot that quotes bid/ask around fair value.
    
    Integrates with matching engine to provide liquidity.
    """
    
    def __init__(self, participant_id: str, spread_bps: float = 10):
        self.participant_id = participant_id
        self.spread_bps = spread_bps  # Spread in basis points
        self.position = 0  # Current inventory
        self.max_position = 10000  # Max inventory
        self.order_size = 100
        
        self.active_orders: List[str] = []
    
    def quote(self, engine: MatchingEngine, fair_value: float) -> List[Dict]:
        """Generate quotes (bid and ask)"""
        # Cancel existing orders
        for order_id in self.active_orders:
            engine.order_book.cancel_order(order_id)
        self.active_orders.clear()
        
        # Calculate spread
        spread_dollars = fair_value * (self.spread_bps / 10000)
        half_spread = spread_dollars / 2
        
        # Adjust for inventory (skew quotes if positioned)
        inventory_skew = (self.position / self.max_position) * half_spread
        
        bid_price = round(fair_value - half_spread + inventory_skew, 2)
        ask_price = round(fair_value + half_spread + inventory_skew, 2)
        
        results = []
        
        # Place bid (buy) order if not at max long position
        if self.position < self.max_position:
            buy_order = Order(
                order_id=f"MM_BUY_{time.perf_counter_ns()}",
                symbol=engine.symbol,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=self.order_size,
                price=bid_price,
                participant_id=self.participant_id
            )
            result = engine.submit_order(buy_order)
            if result['status'] in ['RESTING', 'PARTIAL_FILL']:
                self.active_orders.append(buy_order.order_id)
            
            # Update position
            for fill in result['fills']:
                self.position += fill.quantity
            
            results.append(result)
        
        # Place ask (sell) order if not at max short position
        if self.position > -self.max_position:
            sell_order = Order(
                order_id=f"MM_SELL_{time.perf_counter_ns()}",
                symbol=engine.symbol,
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                quantity=self.order_size,
                price=ask_price,
                participant_id=self.participant_id
            )
            result = engine.submit_order(sell_order)
            if result['status'] in ['RESTING', 'PARTIAL_FILL']:
                self.active_orders.append(sell_order.order_id)
            
            # Update position
            for fill in result['fills']:
                self.position -= fill.quantity
            
            results.append(result)
        
        return results
\`\`\`

## Complete Simulation

\`\`\`python
import random

def run_simulation():
    """Run complete order book simulation"""
    print("=" * 80)
    print("ORDER BOOK SIMULATOR - Production-Grade Matching Engine")
    print("=" * 80)
    
    # Initialize matching engine
    engine = MatchingEngine("AAPL")
    
    # Initialize market maker
    market_maker = SimpleMarketMaker("MM1", spread_bps=10)
    
    # Fair value
    fair_value = 150.00
    
    # Market maker quotes
    print("\\n[MARKET MAKER] Generating quotes...")
    mm_results = market_maker.quote(engine, fair_value)
    for result in mm_results:
        print(f"  Order {result['order_id']}: {result['status']}")
    
    # Display order book
    print("\\n[ORDER BOOK] Market data after MM quotes:")
    snapshot = engine.order_book.get_market_data()
    print(f"  NBBO: \${snapshot.best_bid:.2f} x \${snapshot.best_ask:.2f}")
    print(f"  Spread: \${snapshot.spread:.2f} ({snapshot.spread/snapshot.midpoint*10000:.1f} bps)")
    print(f"  Midpoint: \${snapshot.midpoint:.2f}")
    print(f"\\n  Bid Depth:")
    for price in sorted(snapshot.bid_depth.keys(), reverse=True)[:5]:
        print(f"    \${price:.2f}: {snapshot.bid_depth[price]:,} shares")
    print(f"\\n  Ask Depth:")
    for price in sorted(snapshot.ask_depth.keys())[:5]:
        print(f"    \${price:.2f}: {snapshot.ask_depth[price]:,} shares")
    
    # Simulate retail trader orders
print("\\n" + "=" * 80)
print("[SIMULATION] Executing retail trader orders...")
print("=" * 80)
    
    # Order 1: Small market buy
print("\\n[TRADER 1] Market buy 50 shares")
order1 = Order(
    order_id = "RETAIL1",
    symbol = "AAPL",
    side = OrderSide.BUY,
    order_type = OrderType.MARKET,
    quantity = 50,
    participant_id = "RETAIL1"
)
result1 = engine.submit_order(order1)
print(f"  Status: {result1['status']}")
    print(f"  Fills: {len(result1['fills'])}")
    for fill in result1['fills']:
        print(f"    - {fill.quantity} @ \${fill.price:.2f}")
    print(f"  Latency: {result1['latency_ns'] / 1000:.2f} μs")
    
    # Order 2: Limit buy below market
print("\\n[TRADER 2] Limit buy 200 @ $149.90")
order2 = Order(
    order_id = "RETAIL2",
    symbol = "AAPL",
    side = OrderSide.BUY,
    order_type = OrderType.LIMIT,
    quantity = 200,
    price = 149.90,
    participant_id = "RETAIL2"
)
result2 = engine.submit_order(order2)
print(f"  Status: {result2['status']}")
print(f"  Remaining: {result2['remaining_quantity']} shares")
print(f"  Latency: {result2['latency_ns'] / 1000:.2f} μs")
    
    # Order 3: IOC order
print("\\n[TRADER 3] IOC buy 100 @ $150.10")
order3 = Order(
    order_id = "RETAIL3",
    symbol = "AAPL",
    side = OrderSide.BUY,
    order_type = OrderType.IOC,
    quantity = 100,
    price = 150.10,
    participant_id = "RETAIL3"
)
result3 = engine.submit_order(order3)
print(f"  Status: {result3['status']}")
    print(f"  Fills: {len(result3['fills'])}")
    for fill in result3['fills']:
        print(f"    - {fill.quantity} @ \${fill.price:.2f}")
    
    # Order 4: FOK order(will reject if can't fill completely)
print("\\n[TRADER 4] FOK sell 1000 @ $149.95")
order4 = Order(
    order_id = "RETAIL4",
    symbol = "AAPL",
    side = OrderSide.SELL,
    order_type = OrderType.FOK,
    quantity = 1000,
    price = 149.95,
    participant_id = "RETAIL4"
)
result4 = engine.submit_order(order4)
print(f"  Status: {result4['status']}")
if result4['status'] == 'REJECTED_FOK':
    print(f"  Reason: {result4['reason']}")
    
    # Market maker updates quotes after trades
print("\\n[MARKET MAKER] Updating quotes after trades...")
mm_results = market_maker.quote(engine, fair_value)
print(f"  Position: {market_maker.position} shares")
    
    # Final order book state
print("\\n" + "=" * 80)
print("[ORDER BOOK] Final State")
    print("=" * 80)
    snapshot = engine.order_book.get_market_data()
    print(f"\\nNBBO: \${snapshot.best_bid:.2f} x \${snapshot.best_ask:.2f}")
    print(f"Midpoint: \${snapshot.midpoint:.2f}")
    print(f"\\nTop 5 Bid Levels:")
    for price in sorted(snapshot.bid_depth.keys(), reverse=True)[:5]:
        print(f"  \${price:.2f}: {snapshot.bid_depth[price]:,} shares")
    print(f"\\nTop 5 Ask Levels:")
    for price in sorted(snapshot.ask_depth.keys())[:5]:
        print(f"  \${price:.2f}: {snapshot.bid_depth[price]:,} shares")
    
    # Statistics
print("\\n" + "=" * 80)
print("[STATISTICS] Matching Engine Performance")
print("=" * 80)
stats = engine.get_statistics()
print(f"Total orders: {stats['total_orders']}")
print(f"Total trades: {stats['total_trades']}")
print(f"Total volume: {stats['total_volume']:,} shares")
print(f"\\nLatency Statistics:")
print(f"  Mean: {stats['latency_mean_us']:.2f} μs")
print(f"  P50: {stats['latency_p50_us']:.2f} μs")
print(f"  P99: {stats['latency_p99_us']:.2f} μs")
print(f"  Max: {stats['latency_max_us']:.2f} μs")

print("\\n" + "=" * 80)
print("SIMULATION COMPLETE")
print("=" * 80)

# Run the simulation
if __name__ == "__main__":
    run_simulation()
\`\`\`

## Performance Optimization

### 1. Lock-Free Data Structures

For multi-threaded environments, use lock-free queues:

\`\`\`python
# Use atomic operations for order submission
from multiprocessing import Queue

class ConcurrentMatchingEngine:
    def __init__(self, symbol: str):
        self.order_queue = Queue()  # Lock-free queue
        # Process orders in dedicated thread
\`\`\`

### 2. Memory Pre-Allocation

Pre-allocate order objects to avoid GC pauses:

\`\`\`python
class OrderPool:
    """Object pool for orders to reduce allocations"""
    def __init__(self, size: int = 10000):
        self.pool = [Order(...) for _ in range(size)]
        self.available = list(range(size))
\`\`\`

### 3. CPU Pinning

Pin matching engine thread to dedicated CPU core:

\`\`\`bash
# Linux: taskset command
taskset -c 2 python matching_engine.py
\`\`\`

## Extensions and Enhancements

1. **Iceberg Orders**: Display only portion of large order
2. **Auction Matching**: Opening/closing auction mechanisms
3. **Market Data Multicast**: UDP broadcast for low-latency distribution
4. **FIX Protocol**: Industry-standard order entry protocol
5. **Risk Management**: Pre-trade risk checks (position limits, margin)
6. **Regulatory Reporting**: CAT, OATS, Blue Sheets
7. **Replay System**: Reconstruct order book from historical data
8. **Performance Testing**: Benchmark at 100K+ orders/second

## Hands-On Exercises

1. **Add iceberg order support**: Hidden quantity revealed incrementally
2. **Implement auction matching**: Call auction at market open/close
3. **Build FIX gateway**: Accept orders via FIX protocol
4. **Create visualization**: Real-time order book heatmap
5. **Performance tuning**: Profile and optimize to 100K orders/sec
6. **Add more order types**: OCO (One-Cancels-Other), Trailing Stop

## Production Deployment

### Infrastructure Requirements

- **Compute**: 8-core CPU @ 3.5+ GHz, 32GB RAM
- **Network**: 10Gbps NICs, co-located with exchanges
- **Storage**: NVMe SSDs for trade logs (100K+ writes/sec)
- **Redundancy**: Active-active failover, <10ms switch time

### Monitoring

- Order latency (P50, P95, P99, P999)
- Matching throughput (orders/sec, trades/sec)
- Order book depth and spread
- Market maker position and PnL
- System health (CPU, memory, network)

## Summary

This capstone project integrates every concept from the Market Microstructure module:

- **Order book mechanics**: Price-time priority, all order types
- **Latency optimization**: Microsecond-level execution
- **Market making**: Liquidity provision with inventory management
- **Market data**: Level 1, 2, 3 feeds
- **Regulatory compliance**: Trade reporting, audit trails
- **Production design**: Performance, reliability, monitoring

Building this order book simulator provides deep understanding of modern electronic markets and prepares you for roles in exchange technology, HFT infrastructure, or quantitative trading systems.

**Congratulations on completing Module 12: Market Microstructure & Order Flow!** 🎉
`,
};
