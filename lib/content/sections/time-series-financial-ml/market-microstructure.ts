export const marketMicrostructure = {
  title: 'Market Microstructure',
  id: 'market-microstructure',
  content: `
# Market Microstructure

## Introduction

Market microstructure is the study of how trades are executed, how prices form, and how information flows through financial markets. Understanding microstructure is crucial for:

- **Algorithmic Trading**: Minimize market impact
- **High-Frequency Trading**: Exploit millisecond-level patterns
- **Execution Quality**: Reduce slippage and transaction costs
- **Price Discovery**: Understand how news becomes prices
- **Liquidity Provision**: Market making strategies

**Core Topics**:
1. Order types and execution mechanisms
2. Bid-ask spread and liquidity
3. Order book dynamics
4. Market impact and slippage
5. Order flow imbalance
6. High-frequency trading patterns

---

## Order Types & Execution Mechanics

### Basic Order Types

\`\`\`python
"""
Complete Order Type Implementation
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

class OrderType(Enum):
    """All standard order types"""
    MARKET = "market"           # Execute immediately at best price
    LIMIT = "limit"             # Execute only at limit price or better
    STOP = "stop"               # Market order triggered at stop price
    STOP_LIMIT = "stop_limit"   # Limit order triggered at stop price
    TRAILING_STOP = "trailing_stop"  # Stop that moves with price
    ICEBERG = "iceberg"         # Hidden quantity
    FILL_OR_KILL = "fok"        # Execute fully or cancel
    IMMEDIATE_OR_CANCEL = "ioc"  # Execute partially, cancel rest

class OrderSide(Enum):
    """Buy or sell"""
    BUY = "buy"
    SELL = "sell"

class TimeInForce(Enum):
    """How long order stays active"""
    GTC = "gtc"  # Good-til-canceled
    DAY = "day"  # Cancel at market close
    IOC = "ioc"  # Immediate or cancel
    FOK = "fok"  # Fill or kill
    GTD = "gtd"  # Good-til-date

class OrderStatus(Enum):
    """Order lifecycle states"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"

@dataclass
class Order:
    """
    Complete order representation
    """
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    
    # Optional price parameters
    price: Optional[float] = None          # Limit price
    stop_price: Optional[float] = None      # Stop trigger price
    trailing_amount: Optional[float] = None  # For trailing stops
    
    # Order properties
    time_in_force: TimeInForce = TimeInForce.GTC
    display_quantity: Optional[int] = None  # For iceberg orders
    
    # Tracking
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        self.updated_at = self.created_at
    
    def remaining_quantity (self) -> int:
        """Quantity not yet filled"""
        return self.quantity - self.filled_quantity
    
    def is_complete (self) -> bool:
        """Is order fully filled?"""
        return self.filled_quantity >= self.quantity
    
    def update_fill (self, fill_quantity: int, fill_price: float):
        """Update order with partial/full fill"""
        self.filled_quantity += fill_quantity
        
        # Update average fill price
        total_filled_value = self.avg_fill_price * (self.filled_quantity - fill_quantity)
        total_filled_value += fill_quantity * fill_price
        self.avg_fill_price = total_filled_value / self.filled_quantity
        
        self.updated_at = datetime.now()
        
        if self.is_complete():
            self.status = OrderStatus.FILLED
        elif self.filled_quantity > 0:
            self.status = OrderStatus.PARTIAL


# ============================================================================
# ORDER EXAMPLES
# ============================================================================

# 1. Market Order - Execute immediately
market_order = Order(
    symbol="AAPL",
    side=OrderSide.BUY,
    order_type=OrderType.MARKET,
    quantity=100
)
print(f"Market Order: Buy 100 AAPL at market price")

# 2. Limit Order - Only at $150 or better
limit_order = Order(
    symbol="AAPL",
    side=OrderSide.BUY,
    order_type=OrderType.LIMIT,
    quantity=100,
    price=150.00
)
print(f"Limit Order: Buy 100 AAPL at $150 or better")

# 3. Stop Loss - Sell if price falls to $145
stop_loss = Order(
    symbol="AAPL",
    side=OrderSide.SELL,
    order_type=OrderType.STOP,
    quantity=100,
    stop_price=145.00
)
print(f"Stop Loss: Sell 100 AAPL if price hits $145")

# 4. Stop-Limit - Limit order triggered at stop
stop_limit = Order(
    symbol="AAPL",
    side=OrderSide.SELL,
    order_type=OrderType.STOP_LIMIT,
    quantity=100,
    stop_price=145.00,
    price=144.50  # Sell at $144.50 if stop triggered
)
print(f"Stop-Limit: If price hits $145, sell at $144.50 or better")

# 5. Trailing Stop - Stop that moves with price
trailing_stop = Order(
    symbol="AAPL",
    side=OrderSide.SELL,
    order_type=OrderType.TRAILING_STOP,
    quantity=100,
    trailing_amount=5.00  # Trail $5 below high
)
print(f"Trailing Stop: Sell if price drops $5 from highest point")

# 6. Iceberg Order - Hide full size
iceberg = Order(
    symbol="AAPL",
    side=OrderSide.BUY,
    order_type=OrderType.LIMIT,
    quantity=10000,
    price=150.00,
    display_quantity=100  # Only show 100 shares
)
print(f"Iceberg: Buy 10,000 AAPL but only display 100 at a time")
\`\`\`

---

## Order Book Dynamics

### Complete Order Book Implementation

\`\`\`python
"""
Full Order Book with Level 2 Data
"""

import heapq
from collections import defaultdict
import pandas as pd
import numpy as np

class OrderBook:
    """
    Complete order book implementation
    Tracks bids, asks, and order book metrics
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        
        # Order book levels: price -> total size
        self.bids = {}  # Buy orders
        self.asks = {}  # Sell orders
        
        # Maintain sorted price levels for fast access
        self._bid_prices = []  # Max heap
        self._ask_prices = []  # Min heap
        
        # Track all orders by ID
        self.orders = {}
        
        # Metrics
        self.last_trade_price = None
        self.last_trade_size = None
    
    def add_order (self, order: Order):
        """Add limit order to book"""
        if order.order_type != OrderType.LIMIT:
            raise ValueError("Only limit orders can be added to book")
        
        self.orders[order.order_id] = order
        
        if order.side == OrderSide.BUY:
            # Add to bids
            if order.price not in self.bids:
                self.bids[order.price] = 0
                heapq.heappush (self._bid_prices, -order.price)  # Negative for max heap
            self.bids[order.price] += order.quantity
        
        else:  # SELL
            # Add to asks
            if order.price not in self.asks:
                self.asks[order.price] = 0
                heapq.heappush (self._ask_prices, order.price)  # Min heap
            self.asks[order.price] += order.quantity
    
    def remove_order (self, order_id: str):
        """Cancel order"""
        if order_id not in self.orders:
            return
        
        order = self.orders[order_id]
        
        if order.side == OrderSide.BUY:
            self.bids[order.price] -= order.remaining_quantity()
            if self.bids[order.price] <= 0:
                del self.bids[order.price]
        else:
            self.asks[order.price] -= order.remaining_quantity()
            if self.asks[order.price] <= 0:
                del self.asks[order.price]
        
        del self.orders[order_id]
    
    def best_bid (self) -> Optional[float]:
        """Highest buy price"""
        while self._bid_prices and -self._bid_prices[0] not in self.bids:
            heapq.heappop (self._bid_prices)
        
        return -self._bid_prices[0] if self._bid_prices else None
    
    def best_ask (self) -> Optional[float]:
        """Lowest sell price"""
        while self._ask_prices and self._ask_prices[0] not in self.asks:
            heapq.heappop (self._ask_prices)
        
        return self._ask_prices[0] if self._ask_prices else None
    
    def spread (self) -> Optional[float]:
        """Bid-ask spread"""
        bid = self.best_bid()
        ask = self.best_ask()
        
        if bid is not None and ask is not None:
            return ask - bid
        return None
    
    def spread_bps (self) -> Optional[float]:
        """Spread in basis points"""
        spread = self.spread()
        mid = self.mid_price()
        
        if spread and mid:
            return (spread / mid) * 10000
        return None
    
    def mid_price (self) -> Optional[float]:
        """Midpoint of bid-ask"""
        bid = self.best_bid()
        ask = self.best_ask()
        
        if bid is not None and ask is not None:
            return (bid + ask) / 2
        return None
    
    def microprice (self) -> Optional[float]:
        """
        Volume-weighted mid price
        More accurate than simple mid
        """
        bid = self.best_bid()
        ask = self.best_ask()
        
        if bid is None or ask is None:
            return None
        
        bid_size = self.bids.get (bid, 0)
        ask_size = self.asks.get (ask, 0)
        
        if bid_size + ask_size == 0:
            return self.mid_price()
        
        return (bid * ask_size + ask * bid_size) / (bid_size + ask_size)
    
    def depth (self, levels: int = 5) -> dict:
        """
        Order book depth at top N levels
        
        Returns bid/ask prices and sizes
        """
        bid_levels = sorted (self.bids.keys(), reverse=True)[:levels]
        ask_levels = sorted (self.asks.keys())[:levels]
        
        return {
            'bids': [(price, self.bids[price]) for price in bid_levels],
            'asks': [(price, self.asks[price]) for price in ask_levels]
        }
    
    def liquidity (self, levels: int = 5) -> dict:
        """Total available size at top levels"""
        bid_levels = sorted (self.bids.keys(), reverse=True)[:levels]
        ask_levels = sorted (self.asks.keys())[:levels]
        
        bid_liquidity = sum (self.bids[price] for price in bid_levels)
        ask_liquidity = sum (self.asks[price] for price in ask_levels)
        
        return {
            'bid_liquidity': bid_liquidity,
            'ask_liquidity': ask_liquidity,
            'total_liquidity': bid_liquidity + ask_liquidity
        }
    
    def imbalance (self, levels: int = 5) -> float:
        """
        Order book imbalance
        
        Imbalance = (Bid Size - Ask Size) / (Bid Size + Ask Size)
        > 0: More buy pressure
        < 0: More sell pressure
        """
        liq = self.liquidity (levels)
        bid_liq = liq['bid_liquidity']
        ask_liq = liq['ask_liquidity']
        
        if bid_liq + ask_liq == 0:
            return 0
        
        return (bid_liq - ask_liq) / (bid_liq + ask_liq)
    
    def execute_market_order (self, side: OrderSide, quantity: int) -> dict:
        """
        Simulate market order execution
        
        Returns execution report with fills and average price
        """
        fills = []
        remaining = quantity
        
        if side == OrderSide.BUY:
            # Walk up the ask ladder
            ask_levels = sorted (self.asks.keys())
            
            for price in ask_levels:
                if remaining <= 0:
                    break
                
                available = self.asks[price]
                fill_size = min (remaining, available)
                
                fills.append({'price': price, 'size': fill_size})
                self.asks[price] -= fill_size
                
                if self.asks[price] <= 0:
                    del self.asks[price]
                
                remaining -= fill_size
        
        else:  # SELL
            # Walk down the bid ladder
            bid_levels = sorted (self.bids.keys(), reverse=True)
            
            for price in bid_levels:
                if remaining <= 0:
                    break
                
                available = self.bids[price]
                fill_size = min (remaining, available)
                
                fills.append({'price': price, 'size': fill_size})
                self.bids[price] -= fill_size
                
                if self.bids[price] <= 0:
                    del self.bids[price]
                
                remaining -= fill_size
        
        # Calculate average fill price
        if fills:
            total_value = sum (f['price'] * f['size'] for f in fills)
            total_size = sum (f['size'] for f in fills)
            avg_price = total_value / total_size
        else:
            avg_price = None
        
        return {
            'fills': fills,
            'filled_quantity': quantity - remaining,
            'remaining_quantity': remaining,
            'avg_fill_price': avg_price,
            'slippage': self.calculate_slippage (fills, side)
        }
    
    def calculate_slippage (self, fills: list, side: OrderSide) -> Optional[float]:
        """Calculate slippage from mid price"""
        if not fills:
            return None
        
        mid = self.mid_price()
        if mid is None:
            return None
        
        total_value = sum (f['price'] * f['size'] for f in fills)
        total_size = sum (f['size'] for f in fills)
        avg_price = total_value / total_size
        
        if side == OrderSide.BUY:
            slippage = avg_price - mid  # Paid more than mid
        else:
            slippage = mid - avg_price  # Received less than mid
        
        return slippage
    
    def print_book (self, levels: int = 5):
        """Pretty print order book"""
        print(f"\\n{'='*60}")
        print(f"ORDER BOOK: {self.symbol}")
        print(f"{'='*60}")
        
        depth = self.depth (levels)
        
        print(f"\\n{'Price':>10}  {'Size':>10}  {'Side':>6}")
        print("-" * 32)
        
        # Asks (reverse order for display)
        for price, size in reversed (depth['asks']):
            print(f"{price:>10.2f}  {size:>10}  {'ASK':>6}")
        
        print("-" * 32)
        print(f"  SPREAD: {self.spread():.2f} ({self.spread_bps():.1f} bps)")
        print(f"  MID: {self.mid_price():.2f}")
        print("-" * 32)
        
        # Bids
        for price, size in depth['bids']:
            print(f"{price:>10.2f}  {size:>10}  {'BID':>6}")
        
        print()
        
        # Metrics
        liq = self.liquidity (levels)
        imb = self.imbalance (levels)
        
        print(f"Bid Liquidity: {liq['bid_liquidity']:,}")
        print(f"Ask Liquidity: {liq['ask_liquidity']:,}")
        print(f"Imbalance: {imb:+.3f}")
        print(f"{'='*60}\\n")


# ============================================================================
# EXAMPLE: BUILD AND ANALYZE ORDER BOOK
# ============================================================================

# Create order book
book = OrderBook("AAPL")

# Add bid orders
book.add_order(Order (symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.LIMIT,
                    quantity=500, price=150.00, order_id="bid1"))
book.add_order(Order (symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.LIMIT,
                    quantity=300, price=149.99, order_id="bid2"))
book.add_order(Order (symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.LIMIT,
                    quantity=200, price=149.98, order_id="bid3"))
book.add_order(Order (symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.LIMIT,
                    quantity=150, price=149.97, order_id="bid4"))
book.add_order(Order (symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.LIMIT,
                    quantity=100, price=149.96, order_id="bid5"))

# Add ask orders
book.add_order(Order (symbol="AAPL", side=OrderSide.SELL, order_type=OrderType.LIMIT,
                    quantity=400, price=150.01, order_id="ask1"))
book.add_order(Order (symbol="AAPL", side=OrderSide.SELL, order_type=OrderType.LIMIT,
                    quantity=600, price=150.02, order_id="ask2"))
book.add_order(Order (symbol="AAPL", side=OrderSide.SELL, order_type=OrderType.LIMIT,
                    quantity=250, price=150.03, order_id="ask3"))
book.add_order(Order (symbol="AAPL", side=OrderSide.SELL, order_type=OrderType.LIMIT,
                    quantity=180, price=150.04, order_id="ask4"))
book.add_order(Order (symbol="AAPL", side=OrderSide.SELL, order_type=OrderType.LIMIT,
                    quantity=120, price=150.05, order_id="ask5"))

# Display book
book.print_book (levels=5)

# Execute market order
print("\\nExecuting BUY market order for 800 shares...")
execution = book.execute_market_order(OrderSide.BUY, 800)

print(f"\\nExecution Report:")
print(f"  Filled: {execution['filled_quantity']} shares")
print(f"  Avg Price: \${execution['avg_fill_price']:.2f}")
print(f"  Slippage: \${execution['slippage']:.4f}")
print(f"  Fills: {len (execution['fills'])} levels")

# Show updated book
book.print_book (levels=5)
\`\`\`

---

## Market Impact & Slippage

\`\`\`python
"""
Estimate market impact for large orders
"""

import numpy as np

class MarketImpactModel:
    """
    Estimate price impact of trades
    """
    
    @staticmethod
    def kyle_lambda (daily_volume: float, volatility: float) -> float:
        """
        Kyle\'s Lambda: permanent price impact coefficient
        
        λ ≈ σ / √V
        where σ = volatility, V = daily volume
        """
        return volatility / np.sqrt (daily_volume)
    
    @staticmethod
    def permanent_impact (order_size: float, daily_volume: float,
                        volatility: float) -> float:
        """
        Permanent price impact
        
        Impact = λ * (Order Size / Daily Volume)
        """
        lambda_coef = MarketImpactModel.kyle_lambda (daily_volume, volatility)
        participation_rate = order_size / daily_volume
        return lambda_coef * participation_rate
    
    @staticmethod
    def temporary_impact (order_size: float, spread: float) -> float:
        """
        Temporary impact (half spread + more for large orders)
        """
        base_impact = spread / 2
        size_impact = 0.1 * np.log(1 + order_size / 100)  # Logarithmic
        return base_impact + size_impact
    
    @staticmethod
    def total_execution_cost (order_size: float, price: float,
                            daily_volume: float, volatility: float,
                            spread: float, is_buy: bool = True) -> dict:
        """
        Complete execution cost breakdown
        """
        # Permanent impact
        perm_impact = MarketImpactModel.permanent_impact(
            order_size, daily_volume, volatility
        )
        
        # Temporary impact
        temp_impact = MarketImpactModel.temporary_impact (order_size, spread)
        
        # Total impact in dollars
        sign = 1 if is_buy else -1
        total_impact = sign * (perm_impact + temp_impact)
        
        # Costs
        execution_price = price + total_impact
        cost_per_share = total_impact
        total_cost = cost_per_share * order_size
        cost_bps = (cost_per_share / price) * 10000
        
        return {
            'permanent_impact': perm_impact,
            'temporary_impact': temp_impact,
            'total_impact': total_impact,
            'execution_price': execution_price,
            'cost_per_share': cost_per_share,
            'total_cost': total_cost,
            'cost_bps': cost_bps
        }


# Example: Large order impact
price = 150.00
order_size = 10000  # 10K shares
daily_volume = 50_000_000  # 50M shares/day
volatility = 2.00  # $2 daily volatility
spread = 0.02  # $0.02 spread

impact = MarketImpactModel.total_execution_cost(
    order_size, price, daily_volume, volatility, spread, is_buy=True
)

print("\\n" + "="*60)
print("MARKET IMPACT ANALYSIS")
print("="*60)
print(f"Order Size: {order_size:,} shares")
print(f"Current Price: \${price:.2f}")
print(f"Daily Volume: {daily_volume:,} shares")
print(f"\\nImpact Breakdown:")
print(f"  Permanent Impact: \${impact['permanent_impact']:.4f}")
print(f"  Temporary Impact: \${impact['temporary_impact']:.4f}")
print(f"  Total Impact: \${impact['total_impact']:.4f}")
print(f"\\nExecution Costs:")
print(f"  Execution Price: \${impact['execution_price']:.2f}")
print(f"  Cost per Share: \${impact['cost_per_share']:.4f}")
print(f"  Total Cost: \${impact['total_cost']:.2f}")
print(f"  Cost (bps): {impact['cost_bps']:.2f} bps")
print("="*60 + "\\n")
\`\`\`

---

## Order Flow Imbalance

\`\`\`python
"""
Analyze order flow for short-term predictions
"""

class OrderFlowAnalyzer:
    """
    Order flow analysis for alpha
    """
    
    @staticmethod
    def calculate_ofi (book_updates: pd.DataFrame) -> pd.Series:
        """
        Order Flow Imbalance (OFI)
        
        Measures net buying/selling pressure
        Predictive of short-term price moves
        """
        # Change in bid size at best bid
        delta_bid = book_updates['bid_size'].diff()
        delta_bid_price = book_updates['best_bid'].diff()
        
        # Change in ask size at best ask
        delta_ask = book_updates['ask_size'].diff()
        delta_ask_price = book_updates['best_ask'].diff()
        
        # OFI formula
        ofi = delta_bid * (delta_bid_price >= 0) - delta_ask * (delta_ask_price <= 0)
        
        return ofi
    
    @staticmethod
    def voi (trades: pd.DataFrame) -> pd.Series:
        """
        Volume Order Imbalance
        
        Buy volume - Sell volume
        """
        buy_volume = trades[trades['side'] == 'buy']['size']
        sell_volume = trades[trades['side'] == 'sell']['size']
        
        voi = buy_volume.sum() - sell_volume.sum()
        return voi
    
    @staticmethod
    def trade_intensity (trades: pd.DataFrame, window: str = '1min') -> pd.Series:
        """Number of trades per time window"""
        trades['timestamp'] = pd.to_datetime (trades['timestamp'])
        trades = trades.set_index('timestamp')
        return trades.resample (window).size()
    
    @staticmethod
    def effective_spread (trades: pd.DataFrame, mid_prices: pd.Series) -> pd.Series:
        """
        Effective Spread = 2 * |Trade Price - Mid Price|
        
        Measures actual execution cost
        """
        trades['mid'] = mid_prices.reindex (trades.index, method='ffill')
        effective_spread = 2 * abs (trades['price'] - trades['mid'])
        return effective_spread


# Simulate order flow data
np.random.seed(42)
n_updates = 1000

book_updates = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01 09:30', periods=n_updates, freq='100ms'),
    'best_bid': 100 + np.cumsum (np.random.randn (n_updates) * 0.01),
    'best_ask': 100 + np.cumsum (np.random.randn (n_updates) * 0.01) + 0.02,
    'bid_size': np.random.randint(100, 1000, n_updates),
    'ask_size': np.random.randint(100, 1000, n_updates)
})

analyzer = OrderFlowAnalyzer()
ofi = analyzer.calculate_ofi (book_updates)

print("\\nOrder Flow Imbalance Statistics:")
print(f"  Mean OFI: {ofi.mean():.2f}")
print(f"  Std OFI: {ofi.std():.2f}")
print(f"  Current OFI: {ofi.iloc[-1]:.2f}")

# OFI predicts short-term returns
returns = book_updates['best_bid'].pct_change()
correlation = ofi.corr (returns.shift(-1))  # Next period return
print(f"\\nOFI vs Next-Period Return Correlation: {correlation:.3f}")
\`\`\`

---

## High-Frequency Trading Patterns

\`\`\`python
"""
Detect HFT patterns in order book
"""

class HFTDetector:
    """
    Identify high-frequency trading patterns
    """
    
    @staticmethod
    def detect_quote_stuffing (book_updates: pd.DataFrame,
                             threshold: int = 100) -> pd.Series:
        """
        Quote Stuffing: Rapid order submissions/cancellations
        
        > threshold updates per second = stuffing
        """
        updates_per_sec = book_updates.resample('1S', on='timestamp').size()
        stuffing = updates_per_sec > threshold
        return stuffing
    
    @staticmethod
    def detect_layering (book_updates: pd.DataFrame) -> pd.Series:
        """
        Layering: Large orders on one side, trade on other
        
        Detect imbalance followed by opposite-side trades
        """
        imbalance = (book_updates['bid_size'] - book_updates['ask_size']) / \
                    (book_updates['bid_size'] + book_updates['ask_size'])
        
        # Large imbalance
        large_imbalance = abs (imbalance) > 0.5
        
        return large_imbalance
    
    @staticmethod
    def detect_spoofing (orders: pd.DataFrame) -> pd.DataFrame:
        """
        Spoofing: Place orders with intent to cancel
        
        Large orders that get canceled before execution
        """
        orders['lifetime'] = (orders['canceled_at'] - orders['placed_at']).dt.total_seconds()
        
        # Large orders canceled quickly
        spoofing = (orders['quantity'] > orders['quantity'].quantile(0.9)) & \
                   (orders['lifetime'] < 1.0)
        
        return orders[spoofing]


# Microstructure patterns are key to understanding modern markets
print("\\nHFT Detection helps identify:")
print("  - Quote stuffing (manipulation)")
print("  - Layering (spoofing)")
print("  - Front-running patterns")
print("  - Liquidity withdrawal")
\`\`\`

---

## Key Takeaways

**Order Types Strategy**:
- **Market**: Fast execution, guaranteed fill, but slippage
- **Limit**: Price control, but may not fill
- **Stop**: Risk management, but can gap through stop
- **Iceberg**: Hide large orders, reduce market impact

**Order Book Insights**:
- **Spread**: Cost of immediacy
- **Depth**: Available liquidity
- **Imbalance**: Predictive signal (buy/sell pressure)
- **Microprice**: Better than mid for predictions

**Market Impact**:
- **Permanent**: Price change that persists
- **Temporary**: Recovers after trade
- **Cost**: Increases with size, volatility, and illiquidity
- **Mitigation**: Split orders, use algos, trade during high liquidity

**Order Flow Alpha**:
- **OFI**: Predicts short-term (seconds) price moves
- **VOI**: Buy/sell volume imbalance
- **Effective Spread**: Actual transaction cost
- **Trade Intensity**: Higher = more information flow

**HFT Awareness**:
- Markets are **fast** (microseconds matter)
- **Adverse selection**: Smart money trades against you
- **Latency arbitrage**: First to act wins
- **Co-location**: Physical proximity to exchange

**Remember**: Understanding microstructure = better execution = higher returns after costs.
`,
};
