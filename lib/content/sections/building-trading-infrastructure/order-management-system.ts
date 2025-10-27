export const orderManagementSystem = {
  title: 'Order Management System (OMS)',
  id: 'order-management-system',
  content: `
# Order Management System (OMS)

## Introduction

The **Order Management System (OMS)** is the heart of any trading system. It manages the **entire lifecycle** of orders from creation to completion.

**Core Responsibilities:**
- Order validation and enrichment
- Order state management (NEW → FILLED → etc.)
- Order routing to execution venues
- Fill tracking and aggregation
- Order amendments and cancellations
- Regulatory reporting and audit trail

**Real-World Scale:**
- **Interactive Brokers**: 2.77M accounts, billions in daily order volume
- **Robinhood**: 23M accounts, handles millions of orders per day
- **Citadel Securities**: Routes 26% of US equity volume

This section builds a **production-grade OMS** with all features required by professional trading firms.

---

## Order Lifecycle

### Order States

\`\`\`
NEW → PENDING_RISK → PENDING_EXEC → PARTIALLY_FILLED → FILLED
 ↓         ↓              ↓                               ↑
 ↓    REJECTED      CANCELLED ──────────────────────────┘
 ↓
EXPIRED
\`\`\`

\`\`\`python
"""
Order Management System - Core Implementation
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime, timedelta
from decimal import Decimal
import uuid

class OrderSide(Enum):
    """Buy or Sell"""
    BUY = "BUY"
    SELL = "SELL"

class OrderType(Enum):
    """Order types"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"

class TimeInForce(Enum):
    """Order duration"""
    DAY = "DAY"  # Good for trading day
    GTC = "GTC"  # Good til cancelled
    IOC = "IOC"  # Immediate or cancel
    FOK = "FOK"  # Fill or kill
    GTD = "GTD"  # Good til date

class OrderState(Enum):
    """Order lifecycle states"""
    NEW = "NEW"
    PENDING_RISK = "PENDING_RISK"
    RISK_REJECTED = "RISK_REJECTED"
    PENDING_EXEC = "PENDING_EXEC"
    ACCEPTED = "ACCEPTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

@dataclass
class Order:
    """
    Order object with all fields
    
    Follows FIX protocol field naming conventions
    """
    # Identification
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    client_order_id: str = ""
    parent_order_id: Optional[str] = None
    
    # Security
    symbol: str = ""
    security_id: str = ""
    exchange: str = ""
    
    # Order details
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: Decimal = Decimal('0')
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    
    # Timing
    time_in_force: TimeInForce = TimeInForce.DAY
    expire_time: Optional[datetime] = None
    
    # State
    state: OrderState = OrderState.NEW
    filled_quantity: Decimal = Decimal('0')
    remaining_quantity: Decimal = Decimal('0')
    avg_fill_price: Decimal = Decimal('0')
    
    # Metadata
    account: str = ""
    strategy: str = ""
    trader: str = ""
    
    # Timestamps
    created_time: datetime = field(default_factory=datetime.utcnow)
    submitted_time: Optional[datetime] = None
    accepted_time: Optional[datetime] = None
    completed_time: Optional[datetime] = None
    
    # Routing
    destination: str = ""
    route: List[str] = field(default_factory=list)
    
    # Tracking
    fills: List[Dict] = field(default_factory=list)
    events: List[Dict] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize computed fields"""
        if not self.client_order_id:
            self.client_order_id = f"CLO-{self.order_id[:8]}"
        self.remaining_quantity = self.quantity
        
        # Log creation event
        self.add_event("ORDER_CREATED", {"state": self.state.value})
    
    def add_event(self, event_type: str, data: Dict):
        """Add event to order history"""
        event = {
            'timestamp': datetime.utcnow(),
            'event_type': event_type,
            'data': data
        }
        self.events.append(event)
    
    def is_terminal(self) -> bool:
        """Check if order is in terminal state"""
        terminal_states = {
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.REJECTED,
            OrderState.EXPIRED
        }
        return self.state in terminal_states
    
    def is_fillable(self) -> bool:
        """Check if order can receive fills"""
        fillable_states = {
            OrderState.ACCEPTED,
            OrderState.PARTIALLY_FILLED
        }
        return self.state in fillable_states
    
    def get_fill_percentage(self) -> float:
        """Get percentage filled"""
        if self.quantity == 0:
            return 0.0
        return float(self.filled_quantity / self.quantity * 100)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'order_id': self.order_id,
            'client_order_id': self.client_order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': float(self.quantity),
            'price': float(self.price) if self.price else None,
            'state': self.state.value,
            'filled_quantity': float(self.filled_quantity),
            'remaining_quantity': float(self.remaining_quantity),
            'avg_fill_price': float(self.avg_fill_price),
            'created_time': self.created_time.isoformat(),
        }


@dataclass
class Fill:
    """
    Order fill (execution report)
    """
    fill_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    order_id: str = ""
    execution_id: str = ""
    
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    
    quantity: Decimal = Decimal('0')
    price: Decimal = Decimal('0')
    
    timestamp: datetime = field(default_factory=datetime.utcnow)
    exchange: str = ""
    
    commission: Decimal = Decimal('0')
    commission_currency: str = "USD"
    
    liquidity_flag: str = ""  # "MAKER" or "TAKER"
    
    def notional_value(self) -> Decimal:
        """Calculate notional value"""
        return self.quantity * self.price
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'fill_id': self.fill_id,
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': float(self.quantity),
            'price': float(self.price),
            'timestamp': self.timestamp.isoformat(),
            'exchange': self.exchange,
            'commission': float(self.commission),
        }
\`\`\`

---

## OMS Core Implementation

\`\`\`python
"""
Order Management System
"""

import asyncio
from typing import Dict, List, Optional, Callable
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class OrderManagementSystem:
    """
    Production-grade Order Management System
    
    Features:
    - Order lifecycle management
    - State transitions with validation
    - Fill aggregation
    - Order amendments
    - Regulatory audit trail
    """
    
    def __init__(self):
        # Order storage
        self.orders: Dict[str, Order] = {}
        self.orders_by_client_id: Dict[str, Order] = {}
        self.orders_by_symbol: Dict[str, List[Order]] = defaultdict(list)
        
        # Fill storage
        self.fills: Dict[str, Fill] = {}
        
        # Callbacks
        self.on_order_update: Optional[Callable] = None
        self.on_fill: Optional[Callable] = None
        
        # Statistics
        self.stats = {
            'orders_created': 0,
            'orders_filled': 0,
            'orders_cancelled': 0,
            'orders_rejected': 0,
            'total_volume': Decimal('0'),
        }
        
        logger.info("OMS initialized")
    
    async def create_order(self, order: Order) -> Order:
        """
        Create new order
        
        Validates order and transitions to PENDING_RISK
        """
        # Validate order
        validation_errors = self._validate_order(order)
        if validation_errors:
            logger.error(f"Order validation failed: {validation_errors}")
            order.state = OrderState.REJECTED
            order.add_event("ORDER_REJECTED", {
                'reason': 'validation_failed',
                'errors': validation_errors
            })
            raise ValueError(f"Invalid order: {validation_errors}")
        
        # Store order
        self.orders[order.order_id] = order
        self.orders_by_client_id[order.client_order_id] = order
        self.orders_by_symbol[order.symbol].append(order)
        
        # Update state
        order.state = OrderState.PENDING_RISK
        order.submitted_time = datetime.utcnow()
        order.add_event("ORDER_SUBMITTED", {'state': order.state.value})
        
        self.stats['orders_created'] += 1
        
        logger.info(f"Order created: {order.order_id} {order.side.value} {order.quantity} {order.symbol}")
        
        # Notify callbacks
        if self.on_order_update:
            await self.on_order_update(order)
        
        return order
    
    def _validate_order(self, order: Order) -> List[str]:
        """Validate order fields"""
        errors = []
        
        if not order.symbol:
            errors.append("Symbol is required")
        
        if order.quantity <= 0:
            errors.append("Quantity must be positive")
        
        if order.order_type == OrderType.LIMIT and not order.price:
            errors.append("Limit order requires price")
        
        if order.order_type == OrderType.STOP and not order.stop_price:
            errors.append("Stop order requires stop price")
        
        if order.time_in_force == TimeInForce.GTD and not order.expire_time:
            errors.append("GTD order requires expire time")
        
        return errors
    
    async def accept_order(self, order_id: str) -> Order:
        """
        Accept order (passed risk checks)
        
        Transitions from PENDING_RISK → ACCEPTED
        """
        order = self.orders.get(order_id)
        if not order:
            raise ValueError(f"Order not found: {order_id}")
        
        if order.state != OrderState.PENDING_RISK:
            raise ValueError(f"Cannot accept order in state {order.state}")
        
        order.state = OrderState.ACCEPTED
        order.accepted_time = datetime.utcnow()
        order.add_event("ORDER_ACCEPTED", {
            'state': order.state.value,
            'accepted_time': order.accepted_time.isoformat()
        })
        
        logger.info(f"Order accepted: {order_id}")
        
        if self.on_order_update:
            await self.on_order_update(order)
        
        return order
    
    async def reject_order(self, order_id: str, reason: str) -> Order:
        """
        Reject order
        
        Transitions to REJECTED (terminal state)
        """
        order = self.orders.get(order_id)
        if not order:
            raise ValueError(f"Order not found: {order_id}")
        
        order.state = OrderState.REJECTED
        order.completed_time = datetime.utcnow()
        order.add_event("ORDER_REJECTED", {
            'reason': reason,
            'rejected_time': order.completed_time.isoformat()
        })
        
        self.stats['orders_rejected'] += 1
        
        logger.warning(f"Order rejected: {order_id} - {reason}")
        
        if self.on_order_update:
            await self.on_order_update(order)
        
        return order
    
    async def fill_order(self, fill: Fill) -> Order:
        """
        Process order fill
        
        Updates order state and calculates average fill price
        """
        order = self.orders.get(fill.order_id)
        if not order:
            raise ValueError(f"Order not found: {fill.order_id}")
        
        if not order.is_fillable():
            raise ValueError(f"Order not fillable: {order.state}")
        
        # Store fill
        self.fills[fill.fill_id] = fill
        order.fills.append(fill.to_dict())
        
        # Update order quantities
        old_filled_qty = order.filled_quantity
        order.filled_quantity += fill.quantity
        order.remaining_quantity = order.quantity - order.filled_quantity
        
        # Update average fill price
        if order.filled_quantity > 0:
            total_cost = (old_filled_qty * order.avg_fill_price) + (fill.quantity * fill.price)
            order.avg_fill_price = total_cost / order.filled_quantity
        
        # Update state
        if order.filled_quantity >= order.quantity:
            order.state = OrderState.FILLED
            order.completed_time = datetime.utcnow()
            self.stats['orders_filled'] += 1
            logger.info(f"Order filled: {order.order_id}")
        else:
            order.state = OrderState.PARTIALLY_FILLED
            logger.info(f"Order partially filled: {order.order_id} ({order.get_fill_percentage():.1f}%)")
        
        # Add event
        order.add_event("ORDER_FILL", {
            'fill_id': fill.fill_id,
            'quantity': float(fill.quantity),
            'price': float(fill.price),
            'filled_quantity': float(order.filled_quantity),
            'remaining_quantity': float(order.remaining_quantity),
            'state': order.state.value
        })
        
        # Update stats
        self.stats['total_volume'] += fill.notional_value()
        
        # Notify callbacks
        if self.on_fill:
            await self.on_fill(fill)
        
        if self.on_order_update:
            await self.on_order_update(order)
        
        return order
    
    async def cancel_order(self, order_id: str, reason: str = "user_request") -> Order:
        """
        Cancel order
        
        Transitions to CANCELLED (terminal state)
        """
        order = self.orders.get(order_id)
        if not order:
            raise ValueError(f"Order not found: {order_id}")
        
        if order.is_terminal():
            raise ValueError(f"Cannot cancel order in terminal state: {order.state}")
        
        order.state = OrderState.CANCELLED
        order.completed_time = datetime.utcnow()
        order.add_event("ORDER_CANCELLED", {
            'reason': reason,
            'filled_quantity': float(order.filled_quantity),
            'cancelled_quantity': float(order.remaining_quantity),
            'cancelled_time': order.completed_time.isoformat()
        })
        
        self.stats['orders_cancelled'] += 1
        
        logger.info(f"Order cancelled: {order_id} - {reason}")
        
        if self.on_order_update:
            await self.on_order_update(order)
        
        return order
    
    async def modify_order(
        self,
        order_id: str,
        new_quantity: Optional[Decimal] = None,
        new_price: Optional[Decimal] = None
    ) -> Order:
        """
        Modify order (replace)
        
        In practice, this creates new order and cancels old one
        """
        order = self.orders.get(order_id)
        if not order:
            raise ValueError(f"Order not found: {order_id}")
        
        if order.is_terminal():
            raise ValueError(f"Cannot modify order in terminal state: {order.state}")
        
        # Cancel original order
        await self.cancel_order(order_id, reason="replaced")
        
        # Create new order with modifications
        new_order = Order(
            parent_order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            quantity=new_quantity or order.quantity,
            price=new_price or order.price,
            time_in_force=order.time_in_force,
            account=order.account,
            strategy=order.strategy,
            trader=order.trader,
        )
        
        await self.create_order(new_order)
        
        logger.info(f"Order modified: {order_id} → {new_order.order_id}")
        
        return new_order
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self.orders.get(order_id)
    
    def get_order_by_client_id(self, client_order_id: str) -> Optional[Order]:
        """Get order by client order ID"""
        return self.orders_by_client_id.get(client_order_id)
    
    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """Get all orders for symbol"""
        return self.orders_by_symbol.get(symbol, [])
    
    def get_open_orders(self) -> List[Order]:
        """Get all open orders"""
        return [
            order for order in self.orders.values()
            if not order.is_terminal()
        ]
    
    def get_statistics(self) -> Dict:
        """Get OMS statistics"""
        open_orders = len(self.get_open_orders())
        return {
            'total_orders': len(self.orders),
            'open_orders': open_orders,
            'filled_orders': self.stats['orders_filled'],
            'cancelled_orders': self.stats['orders_cancelled'],
            'rejected_orders': self.stats['orders_rejected'],
            'total_volume': float(self.stats['total_volume']),
            'total_fills': len(self.fills),
        }
    
    def print_statistics(self):
        """Print OMS statistics"""
        stats = self.get_statistics()
        print("\\n" + "=" * 70)
        print("OMS STATISTICS")
        print("=" * 70)
        for key, value in stats.items():
            print(f"  {key}: {value}")
\`\`\`

---

## Order Validation and Risk Integration

\`\`\`python
"""
Order Validation and Risk Checks
"""

class OrderValidator:
    """
    Pre-submission order validation
    """
    
    @staticmethod
    def validate_market_hours(symbol: str, timestamp: datetime) -> bool:
        """Check if market is open"""
        # Simplified - real implementation checks exchange calendars
        hour = timestamp.hour
        if symbol.endswith("USD"):  # Crypto
            return True  # 24/7
        else:  # Equities
            return 9 <= hour < 16  # 9:30 AM - 4:00 PM ET
    
    @staticmethod
    def validate_lot_size(symbol: str, quantity: Decimal) -> bool:
        """Check minimum lot size"""
        # Simplified - real implementation checks exchange rules
        min_lots = {
            'BTC': Decimal('0.001'),
            'ETH': Decimal('0.01'),
            'AAPL': Decimal('1'),  # Full shares
        }
        min_qty = min_lots.get(symbol, Decimal('1'))
        return quantity >= min_qty
    
    @staticmethod
    def validate_price_increment(symbol: str, price: Decimal) -> bool:
        """Check price follows tick size"""
        # Simplified - real implementation checks exchange rules
        tick_sizes = {
            'BTC': Decimal('0.01'),
            'ETH': Decimal('0.01'),
            'AAPL': Decimal('0.01'),
        }
        tick = tick_sizes.get(symbol, Decimal('0.01'))
        return price % tick == 0


class PreTradeRiskCheck:
    """
    Pre-trade risk management
    """
    
    def __init__(self):
        # Risk limits
        self.max_order_value = Decimal('1000000')  # $1M per order
        self.max_position = Decimal('100000')  # Max shares per symbol
        self.max_daily_loss = Decimal('50000')  # $50K daily loss limit
        
        # Current state
        self.positions: Dict[str, Decimal] = {}
        self.daily_pnl = Decimal('0')
    
    async def check_order(self, order: Order, current_price: Decimal) -> tuple[bool, str]:
        """
        Perform pre-trade risk check
        
        Returns: (passed, reason_if_failed)
        """
        # Check order value
        notional = order.quantity * current_price
        if notional > self.max_order_value:
            return False, f"Order value \${notional} exceeds limit \${self.max_order_value}"
        
        # Check position limit
        current_pos = self.positions.get(order.symbol, Decimal('0'))
        position_delta = order.quantity if order.side == OrderSide.BUY else -order.quantity
        new_position = current_pos + position_delta
        
        if abs(new_position) > self.max_position:
            return False, f"Position {new_position} exceeds limit {self.max_position}"
        
        # Check daily loss limit
        if self.daily_pnl < -self.max_daily_loss:
            return False, f"Daily loss \${abs(self.daily_pnl)} exceeds limit \${self.max_daily_loss}"
        
        return True, "PASS"
\`\`\`

---

## Complete OMS Example

\`\`\`python
"""
Complete OMS Usage Example
"""

async def oms_example():
    """Demonstrate complete OMS functionality"""
    
    # Initialize OMS
    oms = OrderManagementSystem()
    
    # Set up callbacks
    async def on_order_update(order: Order):
        print(f"[ORDER UPDATE] {order.order_id}: {order.state.value}")
    
    async def on_fill(fill: Fill):
        print(f"[FILL] {fill.order_id}: {fill.quantity} @ {fill.price}")
    
    oms.on_order_update = on_order_update
    oms.on_fill = on_fill
    
    print("=" * 70)
    print("OMS DEMO")
    print("=" * 70)
    
    # Create market buy order
    order1 = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal('100'),
        account="ACC-001",
        strategy="momentum",
        trader="algo_trader"
    )
    
    print(f"\\n1. Creating order...")
    await oms.create_order(order1)
    
    # Accept order (passed risk checks)
    print(f"\\n2. Accepting order...")
    await oms.accept_order(order1.order_id)
    
    # Simulate partial fill
    print(f"\\n3. Simulating partial fill...")
    fill1 = Fill(
        order_id=order1.order_id,
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=Decimal('50'),
        price=Decimal('150.25'),
        exchange="NASDAQ",
        commission=Decimal('1.00')
    )
    await oms.fill_order(fill1)
    
    # Simulate second fill (complete)
    print(f"\\n4. Simulating final fill...")
    fill2 = Fill(
        order_id=order1.order_id,
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=Decimal('50'),
        price=Decimal('150.30'),
        exchange="NASDAQ",
        commission=Decimal('1.00')
    )
    await oms.fill_order(fill2)
    
    # Create limit order
    print(f"\\n5. Creating limit order...")
    order2 = Order(
        symbol="AAPL",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=Decimal('100'),
        price=Decimal('155.00'),
        time_in_force=TimeInForce.GTC,
        account="ACC-001",
    )
    await oms.create_order(order2)
    await oms.accept_order(order2.order_id)
    
    # Cancel order
    print(f"\\n6. Cancelling order...")
    await oms.cancel_order(order2.order_id, reason="strategy_exit")
    
    # Print statistics
    oms.print_statistics()
    
    # Print order details
    print(f"\\n" + "=" * 70)
    print("ORDER DETAILS")
    print("=" * 70)
    order = oms.get_order(order1.order_id)
    print(f"\\nOrder: {order.order_id}")
    print(f"  Symbol: {order.symbol}")
    print(f"  Side: {order.side.value}")
    print(f"  Quantity: {order.quantity}")
    print(f"  Filled: {order.filled_quantity} ({order.get_fill_percentage():.1f}%)")
    print(f"  Avg Price: \${order.avg_fill_price:.2f})"
print(f"  State: {order.state.value}")
print(f"\\n  Events:")
for event in order.events:
    print(f"    {event['timestamp'].strftime('%H:%M:%S')}: {event['event_type']}")

# asyncio.run(oms_example())
\`\`\`

---

## Summary

**OMS Core Functions:**1. **Order Creation**: Validate and store orders
2. **State Management**: Track lifecycle (NEW → FILLED)
3. **Fill Processing**: Aggregate fills, calculate avg price
4. **Order Amendments**: Cancel/modify orders
5. **Audit Trail**: Log every event for compliance

**Real-World OMS Features (Not Implemented Here):**
- Multi-account support (prime broker, sub-accounts)
- Order routing rules (venue selection)
- Algo orders (VWAP, TWAP, iceberg)
- Order staging (parent/child orders)
- Basket orders (multi-symbol)
- Cross-asset support (equities, options, futures)
- FIX protocol integration
- Database persistence (PostgreSQL)
- Real-time P&L tracking
- Regulatory reporting (CAT, OATS, MiFID II)

**Next Section**: Module 14.3 - Execution Management System (EMS) will cover order routing, execution algorithms, and broker integration.
`,
};
