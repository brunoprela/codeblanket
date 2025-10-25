export const designingOrderManagementSystems = {
  title: 'Designing Order Management Systems (OMS)',
  id: 'designing-order-management-systems',
  content: `
# Designing Order Management Systems (OMS)

## Introduction

An **Order Management System (OMS)** is the heart of any trading operation. It handles the complete lifecycle of orders from creation to execution, maintains state, enforces risk checks, and provides audit trails for regulatory compliance. For high-frequency trading, latency requirements can be as low as **100 microseconds**, while institutional systems prioritize correctness and auditability.

### What Makes a Great OMS?

A production-grade OMS must balance competing requirements:

- **Low Latency**: Sub-millisecond order routing for competitive advantage
- **High Throughput**: Handle 100K+ orders per second during peak trading
- **Correctness**: No duplicate orders, accurate state management
- **Risk Control**: Pre-trade checks prevent catastrophic losses
- **Auditability**: Complete order history for regulatory compliance
- **Resilience**: Failover, recovery, no data loss

By the end of this section, you'll understand:
- OMS architecture patterns (monolithic vs microservices)
- Order lifecycle and state management
- Latency optimization techniques
- Risk check implementation
- FIX protocol integration
- Production deployment strategies

---

## Order Lifecycle

### States and Transitions

An order goes through multiple states from creation to completion:

\`\`\`
NEW → PENDING_VALIDATION → VALIDATED → SUBMITTED → 
ACKNOWLEDGED → PARTIALLY_FILLED → FILLED → CLOSED

Alternative paths:
VALIDATED → REJECTED (failed risk check)
SUBMITTED → REJECTED (broker rejection)
ACKNOWLEDGED → CANCELED (user cancellation)
\`\`\`

Each transition must be:
- **Atomic**: State changes complete or rollback
- **Logged**: Full audit trail for compliance
- **Fast**: Minimal latency between states

### Order Types

Modern OMS must support diverse order types:

**Market Orders**: Execute immediately at best available price
**Limit Orders**: Execute at specified price or better
**Stop Orders**: Trigger at stop price, become market order
**Stop-Limit**: Trigger at stop, execute as limit order
**Iceberg**: Large order split into smaller visible portions
**TWAP/VWAP**: Algorithmic orders spread over time
**Pegged Orders**: Price pegged to market (bid/ask/mid)

---

## Architecture Patterns

### 1. Monolithic OMS

Traditional architecture with all components in a single process:

\`\`\`python
"""
Monolithic OMS Design
Pros: Lowest latency, simplest deployment
Cons: Scaling challenges, single point of failure
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Callable
from datetime import datetime
import uuid
import threading
from queue import Queue, PriorityQueue

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

class OrderStatus(Enum):
    NEW = "NEW"
    PENDING_VALIDATION = "PENDING_VALIDATION"
    VALIDATED = "VALIDATED"
    SUBMITTED = "SUBMITTED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"

@dataclass
class Order:
    """Core Order object"""
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    
    # System fields
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    account_id: str = "DEFAULT"
    status: OrderStatus = OrderStatus.NEW
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    
    # Timestamps (microsecond precision for HFT)
    created_at: int = field(default_factory=lambda: int(datetime.now().timestamp() * 1_000_000))
    updated_at: int = field(default_factory=lambda: int(datetime.now().timestamp() * 1_000_000))
    
    # Audit trail
    state_history: List[tuple] = field(default_factory=list)
    
    def update_status(self, new_status: OrderStatus, reason: str = ""):
        """Update order status with audit trail"""
        timestamp = int(datetime.now().timestamp() * 1_000_000)
        self.state_history.append((timestamp, self.status, new_status, reason))
        self.status = new_status
        self.updated_at = timestamp

@dataclass
class Fill:
    """Execution fill"""
    order_id: str
    fill_id: str
    quantity: float
    price: float
    timestamp: int
    exchange: str
    commission: float = 0.0

class RiskManager:
    """Pre-trade risk checks"""
    
    def __init__(self, max_position_size: float, max_order_value: float):
        self.max_position_size = max_position_size
        self.max_order_value = max_order_value
        self.positions: Dict[str, float] = {}  # symbol -> quantity
        self.lock = threading.Lock()
    
    def check_order(self, order: Order, current_price: float) -> tuple[bool, str]:
        """
        Pre-trade risk checks
        Returns: (approved, rejection_reason)
        """
        with self.lock:
            # Check 1: Order value
            order_value = order.quantity * current_price
            if order_value > self.max_order_value:
                return False, f"Order value {order_value} exceeds limit {self.max_order_value}"
            
            # Check 2: Position limits
            current_position = self.positions.get(order.symbol, 0.0)
            projected_position = current_position + (
                order.quantity if order.side == OrderSide.BUY else -order.quantity
            )
            
            if abs(projected_position) > self.max_position_size:
                return False, f"Position would be {projected_position}, exceeds {self.max_position_size}"
            
            # Check 3: Price reasonableness (prevent fat finger)
            if order.order_type == OrderType.LIMIT and order.limit_price:
                if order.side == OrderSide.BUY and order.limit_price > current_price * 1.05:
                    return False, "Buy limit price 5%+ above market"
                if order.side == OrderSide.SELL and order.limit_price < current_price * 0.95:
                    return False, "Sell limit price 5%+ below market"
            
            return True, ""
    
    def update_position(self, symbol: str, fill: Fill, side: OrderSide):
        """Update position after fill"""
        with self.lock:
            current = self.positions.get(symbol, 0.0)
            change = fill.quantity if side == OrderSide.BUY else -fill.quantity
            self.positions[symbol] = current + change

class OrderBook:
    """Internal order book for tracking"""
    
    def __init__(self):
        self.orders: Dict[str, Order] = {}
        self.orders_by_symbol: Dict[str, List[str]] = {}
        self.lock = threading.Lock()
    
    def add_order(self, order: Order):
        """Add order to book"""
        with self.lock:
            self.orders[order.order_id] = order
            if order.symbol not in self.orders_by_symbol:
                self.orders_by_symbol[order.symbol] = []
            self.orders_by_symbol[order.symbol].append(order.order_id)
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Retrieve order by ID"""
        with self.lock:
            return self.orders.get(order_id)
    
    def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all active orders"""
        with self.lock:
            active_statuses = {
                OrderStatus.SUBMITTED,
                OrderStatus.ACKNOWLEDGED,
                OrderStatus.PARTIALLY_FILLED
            }
            
            if symbol:
                order_ids = self.orders_by_symbol.get(symbol, [])
                return [
                    self.orders[oid] 
                    for oid in order_ids 
                    if self.orders[oid].status in active_statuses
                ]
            else:
                return [
                    order 
                    for order in self.orders.values() 
                    if order.status in active_statuses
                ]

class MonolithicOMS:
    """
    Simple monolithic OMS
    All components in one process for lowest latency
    """
    
    def __init__(self, risk_manager: RiskManager):
        self.risk_manager = risk_manager
        self.order_book = OrderBook()
        self.order_queue = Queue()
        self.fill_queue = Queue()
        
        # Callbacks for external systems
        self.on_order_update: Optional[Callable] = None
        self.on_fill: Optional[Callable] = None
        
        # Start worker threads
        self.running = False
        self.order_thread = None
        self.fill_thread = None
    
    def start(self):
        """Start OMS threads"""
        self.running = True
        self.order_thread = threading.Thread(target=self._process_orders, daemon=True)
        self.fill_thread = threading.Thread(target=self._process_fills, daemon=True)
        self.order_thread.start()
        self.fill_thread.start()
    
    def stop(self):
        """Stop OMS"""
        self.running = False
        if self.order_thread:
            self.order_thread.join(timeout=5)
        if self.fill_thread:
            self.fill_thread.join(timeout=5)
    
    def submit_order(self, order: Order, current_price: float) -> tuple[bool, str]:
        """
        Submit new order
        Returns: (success, message)
        """
        # Update status
        order.update_status(OrderStatus.PENDING_VALIDATION)
        
        # Risk check
        approved, reason = self.risk_manager.check_order(order, current_price)
        if not approved:
            order.update_status(OrderStatus.REJECTED, reason)
            self.order_book.add_order(order)
            return False, reason
        
        # Validated
        order.update_status(OrderStatus.VALIDATED)
        self.order_book.add_order(order)
        
        # Queue for routing
        self.order_queue.put(order)
        
        return True, f"Order {order.order_id} accepted"
    
    def cancel_order(self, order_id: str) -> tuple[bool, str]:
        """Cancel an order"""
        order = self.order_book.get_order(order_id)
        if not order:
            return False, "Order not found"
        
        if order.status not in {OrderStatus.SUBMITTED, OrderStatus.ACKNOWLEDGED, OrderStatus.PARTIALLY_FILLED}:
            return False, f"Cannot cancel order in status {order.status}"
        
        # In production, send cancel to exchange
        # For now, mark as canceled
        order.update_status(OrderStatus.CANCELED, "User requested")
        return True, f"Order {order_id} canceled"
    
    def report_fill(self, fill: Fill):
        """Report a fill from exchange"""
        self.fill_queue.put(fill)
    
    def _process_orders(self):
        """Worker thread to route orders"""
        while self.running:
            try:
                order = self.order_queue.get(timeout=0.1)
                
                # Mark as submitted
                order.update_status(OrderStatus.SUBMITTED)
                
                # In production: Route to exchange via FIX protocol
                # Simulate: Immediately acknowledge
                order.update_status(OrderStatus.ACKNOWLEDGED)
                
                # Callback
                if self.on_order_update:
                    self.on_order_update(order)
                
            except:
                continue
    
    def _process_fills(self):
        """Worker thread to process fills"""
        while self.running:
            try:
                fill = self.fill_queue.get(timeout=0.1)
                
                # Find order
                order = self.order_book.get_order(fill.order_id)
                if not order:
                    continue
                
                # Update fill quantity
                order.filled_quantity += fill.quantity
                order.average_fill_price = (
                    (order.average_fill_price * (order.filled_quantity - fill.quantity) +
                     fill.price * fill.quantity) / order.filled_quantity
                )
                
                # Update status
                if order.filled_quantity >= order.quantity:
                    order.update_status(OrderStatus.FILLED)
                else:
                    order.update_status(OrderStatus.PARTIALLY_FILLED)
                
                # Update position
                self.risk_manager.update_position(order.symbol, fill, order.side)
                
                # Callback
                if self.on_fill:
                    self.on_fill(order, fill)
                
            except:
                continue

# Example usage
if __name__ == "__main__":
    # Create OMS with risk limits
    risk_mgr = RiskManager(
        max_position_size=1000,
        max_order_value=100_000
    )
    
    oms = MonolithicOMS(risk_mgr)
    
    # Callbacks
    def on_order_update(order: Order):
        print(f"Order {order.order_id}: {order.status}")
    
    def on_fill(order: Order, fill: Fill):
        print(f"Fill: {fill.quantity} @ {fill.price}")
    
    oms.on_order_update = on_order_update
    oms.on_fill = on_fill
    
    # Start OMS
    oms.start()
    
    # Submit order
    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.LIMIT,
        limit_price=150.0
    )
    
    success, msg = oms.submit_order(order, current_price=150.5)
    print(msg)
    
    # Simulate fill
    fill = Fill(
        order_id=order.order_id,
        fill_id=str(uuid.uuid4()),
        quantity=100,
        price=150.0,
        timestamp=int(datetime.now().timestamp() * 1_000_000),
        exchange="NASDAQ"
    )
    oms.report_fill(fill)
    
    import time
    time.sleep(1)
    
    oms.stop()
\`\`\`

---

## Microservices OMS Architecture

For larger firms, microservices provide:
- **Scalability**: Scale components independently
- **Resilience**: Failure isolation
- **Flexibility**: Technology diversity

**Trade-off**: Higher latency due to network calls

\`\`\`python
"""
Microservices OMS Architecture

Components:
1. Order Gateway: Entry point, authentication
2. Risk Service: Pre-trade risk checks
3. Order Router: Route to exchanges
4. Execution Service: Manage fills
5. Position Service: Track positions
6. Audit Service: Logging and compliance
"""

from typing import Protocol
import json
from abc import ABC, abstractmethod

class MessageBus(ABC):
    """Abstract message bus for inter-service communication"""
    
    @abstractmethod
    def publish(self, topic: str, message: dict):
        """Publish message to topic"""
        pass
    
    @abstractmethod
    def subscribe(self, topic: str, callback: Callable):
        """Subscribe to topic"""
        pass

class RiskService:
    """
    Microservice for risk checks
    Deployed separately, scales independently
    """
    
    def __init__(self, message_bus: MessageBus):
        self.bus = message_bus
        self.risk_manager = RiskManager(
            max_position_size=1000,
            max_order_value=100_000
        )
        
        # Subscribe to risk check requests
        self.bus.subscribe("risk.check", self.handle_risk_check)
    
    def handle_risk_check(self, message: dict):
        """Handle risk check request"""
        order_data = message['order']
        current_price = message['price']
        
        # Reconstruct order
        order = Order(**order_data)
        
        # Check risk
        approved, reason = self.risk_manager.check_order(order, current_price)
        
        # Publish result
        result = {
            'order_id': order.order_id,
            'approved': approved,
            'reason': reason
        }
        self.bus.publish("risk.result", result)

class OrderGateway:
    """
    API Gateway for orders
    Handles authentication, rate limiting
    """
    
    def __init__(self, message_bus: MessageBus):
        self.bus = message_bus
        self.pending_orders = {}
        
        # Subscribe to risk results
        self.bus.subscribe("risk.result", self.handle_risk_result)
    
    def submit_order(self, order: Order, current_price: float, api_key: str) -> dict:
        """
        Submit order via API
        Returns: {"success": bool, "order_id": str, "message": str}
        """
        # Authenticate
        if not self.authenticate(api_key):
            return {"success": False, "message": "Invalid API key"}
        
        # Rate limiting
        if not self.check_rate_limit(api_key):
            return {"success": False, "message": "Rate limit exceeded"}
        
        # Store order
        self.pending_orders[order.order_id] = order
        
        # Request risk check
        message = {
            'order': {
                'symbol': order.symbol,
                'side': order.side.value,
                'quantity': order.quantity,
                'order_type': order.order_type.value,
                'limit_price': order.limit_price,
                'order_id': order.order_id
            },
            'price': current_price
        }
        self.bus.publish("risk.check", message)
        
        return {
            "success": True,
            "order_id": order.order_id,
            "message": "Order submitted for validation"
        }
    
    def handle_risk_result(self, message: dict):
        """Handle risk check result"""
        order_id = message['order_id']
        approved = message['approved']
        
        order = self.pending_orders.get(order_id)
        if not order:
            return
        
        if approved:
            # Route order
            self.bus.publish("order.route", {'order_id': order_id})
        else:
            # Reject
            order.update_status(OrderStatus.REJECTED, message['reason'])
    
    def authenticate(self, api_key: str) -> bool:
        """Authenticate API key"""
        # In production: Check against database
        return True
    
    def check_rate_limit(self, api_key: str) -> bool:
        """Check rate limit"""
        # In production: Use Redis for rate limiting
        return True

# In production: Use RabbitMQ, Kafka, or Redis Pub/Sub
# This is a simplified in-memory version
class InMemoryMessageBus(MessageBus):
    """Simple in-memory message bus for demo"""
    
    def __init__(self):
        self.subscribers = {}
    
    def publish(self, topic: str, message: dict):
        if topic in self.subscribers:
            for callback in self.subscribers[topic]:
                callback(message)
    
    def subscribe(self, topic: str, callback: Callable):
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)
\`\`\`

---

## Latency Optimization

### Critical Path Analysis

For HFT, every microsecond matters. Measure latency at each stage:

\`\`\`
Order Submit → Risk Check → Validation → Routing → Exchange
   5μs           10μs          2μs        15μs       30μs
\`\`\`

**Target**: < 100μs total

### Optimization Techniques

\`\`\`python
"""
Latency Optimization Techniques
"""

import time
import mmap
import struct
from typing import NamedTuple

# 1. Pre-allocated memory pools
class OrderPool:
    """
    Pre-allocate order objects to avoid memory allocation
    Critical for ultra-low latency
    """
    
    def __init__(self, pool_size: int = 10000):
        self.pool = [Order(
            symbol="",
            side=OrderSide.BUY,
            quantity=0,
            order_type=OrderType.MARKET
        ) for _ in range(pool_size)]
        self.available = list(range(pool_size))
    
    def acquire(self) -> Optional[Order]:
        """Get order from pool"""
        if not self.available:
            return None
        idx = self.available.pop()
        return self.pool[idx]
    
    def release(self, order: Order):
        """Return order to pool"""
        # Reset order
        order.status = OrderStatus.NEW
        order.filled_quantity = 0.0
        # Find index and return to available
        idx = self.pool.index(order)
        self.available.append(idx)

# 2. Lock-free data structures
from queue import Queue
import threading

class LockFreeCounter:
    """
    Lock-free atomic counter using compare-and-swap
    Faster than locks for high-contention scenarios
    """
    
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()  # Python doesn't have true CAS, using lock
    
    def increment(self) -> int:
        """Atomically increment and return new value"""
        with self._lock:
            self._value += 1
            return self._value
    
    @property
    def value(self) -> int:
        return self._value

# 3. Shared memory for IPC
class SharedMemoryOrderBook:
    """
    Use shared memory for zero-copy IPC between processes
    Critical for multi-process architectures
    """
    
    ORDER_SIZE = 128  # bytes per order
    
    def __init__(self, max_orders: int = 10000):
        self.max_orders = max_orders
        self.size = max_orders * self.ORDER_SIZE
        
        # Create shared memory
        self.shm = mmap.mmap(-1, self.size)
    
    def write_order(self, index: int, order: Order):
        """Write order to shared memory"""
        offset = index * self.ORDER_SIZE
        
        # Pack order data (simplified)
        data = struct.pack(
            'Q Q d d',  # order_id_hash, timestamp, quantity, price
            hash(order.order_id) % (2**64),
            order.created_at,
            order.quantity,
            order.limit_price or 0.0
        )
        
        self.shm.seek(offset)
        self.shm.write(data)
    
    def read_order(self, index: int) -> tuple:
        """Read order from shared memory"""
        offset = index * self.ORDER_SIZE
        self.shm.seek(offset)
        data = self.shm.read(32)  # 4 * 8 bytes
        return struct.unpack('Q Q d d', data)

# 4. Batch processing
class BatchOrderProcessor:
    """
    Batch process orders for higher throughput
    Trade latency for throughput when appropriate
    """
    
    def __init__(self, batch_size: int = 100, timeout_ms: int = 10):
        self.batch_size = batch_size
        self.timeout_ms = timeout_ms
        self.batch = []
        self.last_flush = time.time()
    
    def add_order(self, order: Order) -> Optional[List[Order]]:
        """
        Add order to batch
        Returns: Batch if ready to process, None otherwise
        """
        self.batch.append(order)
        
        # Check if should flush
        if len(self.batch) >= self.batch_size:
            return self.flush()
        
        if (time.time() - self.last_flush) * 1000 > self.timeout_ms:
            return self.flush()
        
        return None
    
    def flush(self) -> List[Order]:
        """Flush current batch"""
        batch = self.batch
        self.batch = []
        self.last_flush = time.time()
        return batch

# 5. Avoid Python GIL for critical paths
# Use C extensions or Cython for hot paths
# Example: Cython code (save as risk_check.pyx)
\`\`\`

**Cython Example** (for 10-20x speedup):

\`\`\`cython
# risk_check.pyx
cdef class FastRiskChecker:
    cdef double max_position
    cdef double max_order_value
    cdef dict positions
    
    def __init__(self, double max_position, double max_order_value):
        self.max_position = max_position
        self.max_order_value = max_order_value
        self.positions = {}
    
    cpdef (bint, str) check_order(self, str symbol, double quantity, double price):
        cdef double order_value = quantity * price
        cdef double current_position
        cdef double projected_position
        
        if order_value > self.max_order_value:
            return False, "Order value exceeds limit"
        
        current_position = self.positions.get(symbol, 0.0)
        projected_position = current_position + quantity
        
        if abs(projected_position) > self.max_position:
            return False, "Position limit exceeded"
        
        return True, ""
\`\`\`

---

## FIX Protocol Integration

Financial Information eXchange (FIX) is the standard protocol for order routing.

\`\`\`python
"""
FIX Protocol Integration
Using QuickFIX library
"""

# Install: pip install quickfix
import quickfix as fix
import quickfix44 as fix44

class FIXApplication(fix.Application):
    """
    FIX Application for OMS
    Handles FIX messages to/from broker
    """
    
    def __init__(self, oms: MonolithicOMS):
        super().__init__()
        self.oms = oms
        self.session_id = None
    
    def onCreate(self, sessionID):
        """Called when session created"""
        self.session_id = sessionID
        print(f"FIX session created: {sessionID}")
    
    def onLogon(self, sessionID):
        """Called when logged on"""
        print(f"FIX session logged on: {sessionID}")
    
    def onLogout(self, sessionID):
        """Called when logged out"""
        print(f"FIX session logged out: {sessionID}")
    
    def toAdmin(self, message, sessionID):
        """Outgoing admin message"""
        pass
    
    def fromAdmin(self, message, sessionID):
        """Incoming admin message"""
        pass
    
    def toApp(self, message, sessionID):
        """Outgoing application message"""
        print(f"Sending: {message}")
    
    def fromApp(self, message, sessionID):
        """Incoming application message"""
        # Parse message type
        msg_type = fix.MsgType()
        message.getHeader().getField(msg_type)
        
        if msg_type.getValue() == fix.MsgType_ExecutionReport:
            self.on_execution_report(message)
    
    def on_execution_report(self, message: fix.Message):
        """Handle execution report (fill)"""
        # Extract fields
        order_id = fix.ClOrdID()
        exec_type = fix.ExecType()
        order_qty = fix.OrderQty()
        last_qty = fix.LastQty()
        last_px = fix.LastPx()
        
        message.getField(order_id)
        message.getField(exec_type)
        message.getField(last_qty)
        message.getField(last_px)
        
        # Report fill to OMS
        if exec_type.getValue() == fix.ExecType_FILL or exec_type.getValue() == fix.ExecType_PARTIAL_FILL:
            fill = Fill(
                order_id=order_id.getValue(),
                fill_id=str(uuid.uuid4()),
                quantity=last_qty.getValue(),
                price=last_px.getValue(),
                timestamp=int(datetime.now().timestamp() * 1_000_000),
                exchange="BROKER"
            )
            self.oms.report_fill(fill)
    
    def send_new_order(self, order: Order):
        """Send new order via FIX"""
        # Create FIX message
        msg = fix44.NewOrderSingle()
        
        # Header
        msg.getHeader().setField(fix.MsgType(fix.MsgType_NewOrderSingle))
        
        # Required fields
        msg.setField(fix.ClOrdID(order.order_id))
        msg.setField(fix.Symbol(order.symbol))
        msg.setField(fix.Side(
            fix.Side_BUY if order.side == OrderSide.BUY else fix.Side_SELL
        ))
        msg.setField(fix.OrderQty(order.quantity))
        msg.setField(fix.OrdType(
            fix.OrdType_MARKET if order.order_type == OrderType.MARKET else fix.OrdType_LIMIT
        ))
        
        if order.order_type == OrderType.LIMIT:
            msg.setField(fix.Price(order.limit_price))
        
        # Send
        fix.Session.sendToTarget(msg, self.session_id)

# FIX Configuration
config = """
[DEFAULT]
ConnectionType=initiator
HeartBtInt=30

[SESSION]
BeginString=FIX.4.4
SenderCompID=OMS
TargetCompID=BROKER
SocketConnectHost=localhost
SocketConnectPort=5001
"""

# Save config and run
# settings = fix.SessionSettings("fix.cfg")
# application = FIXApplication(oms)
# store_factory = fix.FileStoreFactory(settings)
# log_factory = fix.FileLogFactory(settings)
# initiator = fix.SocketInitiator(application, store_factory, settings, log_factory)
# initiator.start()
\`\`\`

---

## Production Deployment

### High Availability Architecture

\`\`\`
                    Load Balancer
                         |
        +----------------+----------------+
        |                                 |
    OMS Primary                      OMS Secondary
    (Active)                         (Hot Standby)
        |                                 |
        +---------> Shared State <--------+
                  (PostgreSQL/Redis)
\`\`\`

**Failover Requirements**:
- Detect primary failure: < 100ms
- Promote secondary: < 500ms
- Resume operations: < 1s
- Zero data loss

\`\`\`python
"""
High Availability OMS with Failover
"""

import redis
from typing import Optional
import pickle

class HAOrderManagementSystem:
    """
    HA OMS using Redis for shared state
    Supports primary-secondary failover
    """
    
    def __init__(self, node_id: str, is_primary: bool, redis_host: str = 'localhost'):
        self.node_id = node_id
        self.is_primary = is_primary
        self.redis = redis.Redis(host=redis_host, decode_responses=False)
        
        # Heartbeat
        self.heartbeat_key = "oms:primary:heartbeat"
        self.heartbeat_interval = 1  # seconds
        
        # Start monitoring
        if not is_primary:
            threading.Thread(target=self._monitor_primary, daemon=True).start()
    
    def _send_heartbeat(self):
        """Primary sends heartbeat"""
        while self.is_primary:
            self.redis.setex(
                self.heartbeat_key,
                self.heartbeat_interval * 3,  # TTL
                self.node_id
            )
            time.sleep(self.heartbeat_interval)
    
    def _monitor_primary(self):
        """Secondary monitors primary heartbeat"""
        while not self.is_primary:
            heartbeat = self.redis.get(self.heartbeat_key)
            
            if heartbeat is None:
                print("Primary failed! Promoting to primary...")
                self._promote_to_primary()
                break
            
            time.sleep(self.heartbeat_interval)
    
    def _promote_to_primary(self):
        """Promote secondary to primary"""
        # Try to acquire lock
        lock = self.redis.set(
            "oms:primary:lock",
            self.node_id,
            nx=True,  # Set if not exists
            ex=30  # Expire in 30s
        )
        
        if lock:
            self.is_primary = True
            print(f"Node {self.node_id} is now primary")
            
            # Recover state from Redis
            self._recover_state()
            
            # Start heartbeat
            threading.Thread(target=self._send_heartbeat, daemon=True).start()
    
    def _recover_state(self):
        """Recover order state from Redis"""
        # Get all orders
        keys = self.redis.keys("order:*")
        for key in keys:
            order_data = self.redis.get(key)
            order = pickle.loads(order_data)
            print(f"Recovered order: {order.order_id}")
    
    def persist_order(self, order: Order):
        """Persist order to Redis for HA"""
        key = f"order:{order.order_id}"
        self.redis.set(key, pickle.dumps(order))
\`\`\`

### Monitoring and Alerting

\`\`\`python
"""
OMS Monitoring and Metrics
"""

from dataclasses import dataclass
from typing import Dict
import time

@dataclass
class OMSMetrics:
    """OMS performance metrics"""
    orders_submitted: int = 0
    orders_filled: int = 0
    orders_rejected: int = 0
    orders_canceled: int = 0
    
    total_fill_value: float = 0.0
    
    # Latency (microseconds)
    avg_order_latency: float = 0.0
    p95_order_latency: float = 0.0
    p99_order_latency: float = 0.0
    
    # Throughput
    orders_per_second: float = 0.0
    
    # Errors
    risk_check_failures: int = 0
    routing_errors: int = 0

class OMSMonitor:
    """Monitor OMS performance"""
    
    def __init__(self):
        self.metrics = OMSMetrics()
        self.latencies = []
        self.start_time = time.time()
    
    def record_order_submitted(self):
        self.metrics.orders_submitted += 1
    
    def record_order_filled(self, fill_value: float):
        self.metrics.orders_filled += 1
        self.metrics.total_fill_value += fill_value
    
    def record_latency(self, latency_us: float):
        """Record order latency in microseconds"""
        self.latencies.append(latency_us)
        
        # Keep only recent latencies (last 10k)
        if len(self.latencies) > 10000:
            self.latencies = self.latencies[-10000:]
        
        # Update metrics
        self.metrics.avg_order_latency = sum(self.latencies) / len(self.latencies)
        sorted_latencies = sorted(self.latencies)
        self.metrics.p95_order_latency = sorted_latencies[int(len(sorted_latencies) * 0.95)]
        self.metrics.p99_order_latency = sorted_latencies[int(len(sorted_latencies) * 0.99)]
    
    def get_metrics(self) -> OMSMetrics:
        """Get current metrics"""
        # Calculate throughput
        elapsed = time.time() - self.start_time
        self.metrics.orders_per_second = self.metrics.orders_submitted / elapsed
        
        return self.metrics
    
    def check_alerts(self) -> List[str]:
        """Check for alert conditions"""
        alerts = []
        
        # High latency
        if self.metrics.p99_order_latency > 1000:  # > 1ms
            alerts.append(f"High latency: P99 = {self.metrics.p99_order_latency:.0f}μs")
        
        # High reject rate
        total_orders = self.metrics.orders_submitted
        if total_orders > 100:
            reject_rate = self.metrics.orders_rejected / total_orders
            if reject_rate > 0.05:  # > 5%
                alerts.append(f"High reject rate: {reject_rate:.1%}")
        
        return alerts

# Example: Export metrics to Prometheus
\`\`\`

---

## Best Practices

### 1. State Machine Validation

Always validate state transitions:

\`\`\`python
VALID_TRANSITIONS = {
    OrderStatus.NEW: [OrderStatus.PENDING_VALIDATION],
    OrderStatus.PENDING_VALIDATION: [OrderStatus.VALIDATED, OrderStatus.REJECTED],
    OrderStatus.VALIDATED: [OrderStatus.SUBMITTED, OrderStatus.REJECTED],
    OrderStatus.SUBMITTED: [OrderStatus.ACKNOWLEDGED, OrderStatus.REJECTED],
    OrderStatus.ACKNOWLEDGED: [OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED, OrderStatus.CANCELED],
    OrderStatus.PARTIALLY_FILLED: [OrderStatus.FILLED, OrderStatus.CANCELED],
}

def is_valid_transition(current: OrderStatus, new: OrderStatus) -> bool:
    return new in VALID_TRANSITIONS.get(current, [])
\`\`\`

### 2. Idempotency

Handle duplicate messages gracefully:

\`\`\`python
def process_fill(self, fill: Fill):
    """Process fill idempotently"""
    # Check if already processed
    if fill.fill_id in self.processed_fills:
        return  # Already handled
    
    # Process
    self._update_order(fill)
    
    # Mark as processed
    self.processed_fills.add(fill.fill_id)
\`\`\`

### 3. Circuit Breakers

Prevent cascade failures:

\`\`\`python
class CircuitBreaker:
    """Prevent sending orders during system issues"""
    
    def __init__(self, failure_threshold: int = 10, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func):
        """Execute function with circuit breaker"""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker OPEN")
        
        try:
            result = func()
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failures = 0
            return result
        except Exception as e:
            self.failures += 1
            self.last_failure_time = time.time()
            
            if self.failures >= self.failure_threshold:
                self.state = "OPEN"
            
            raise
\`\`\`

---

## Summary

Building a production OMS requires careful attention to:

1. **Correctness**: Accurate state management, audit trails
2. **Performance**: Low latency (<100μs for HFT), high throughput (100K+ orders/sec)
3. **Risk Management**: Pre-trade checks, position limits
4. **Resilience**: HA setup, failover, circuit breakers
5. **Compliance**: Complete audit trails, regulatory reporting
6. **Monitoring**: Real-time metrics, alerting

The choice between monolithic and microservices depends on your requirements:
- **Monolithic**: Lowest latency, simplest deployment (good for HFT)
- **Microservices**: Better scalability, resilience (good for large institutions)

In the next section, we'll design market data systems capable of ingesting 1M+ messages per second.
`,
};
