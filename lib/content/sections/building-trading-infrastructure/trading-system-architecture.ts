export const tradingSystemArchitecture = {
    title: 'Trading System Architecture',
    id: 'trading-system-architecture',
    content: `
# Trading System Architecture

## Introduction

Building a **production trading system** is fundamentally different from building typical web applications. Trading systems must handle:

- **Sub-millisecond latency** requirements
- **100% uptime** during market hours (no downtime tolerance)
- **Regulatory compliance** (audit trails, best execution)
- **Financial accuracy** (zero tolerance for P&L errors)
- **High throughput** (millions of messages per second)

**Real-World Context:**
- **Citadel Securities**: Processes 26% of all US equity volume
- **Jane Street**: $17 trillion annual trading volume
- **Interactive Brokers**: 2.77 million client accounts, billions in daily volume

This section covers the **architectural patterns** that make these systems possible.

---

## Core Architecture Principles

### 1. Event-Driven Architecture

**Why Event-Driven?**

Traditional request-response architectures don't work for trading:
- Market data arrives asynchronously (streaming)
- Orders execute asynchronously (callbacks)
- Risk checks must happen in real-time
- Multiple systems need to react to same events

\`\`\`python
"""
Event-Driven Trading System Foundation
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Callable, Dict, List
from datetime import datetime
import asyncio
from decimal import Decimal

class EventType(Enum):
    """All event types in trading system"""
    # Market Data Events
    MARKET_DATA = "MARKET_DATA"
    QUOTE = "QUOTE"
    TRADE = "TRADE"
    
    # Order Events
    NEW_ORDER = "NEW_ORDER"
    ORDER_ACCEPTED = "ORDER_ACCEPTED"
    ORDER_REJECTED = "ORDER_REJECTED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_PARTIALLY_FILLED = "ORDER_PARTIALLY_FILLED"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    
    # Signal Events
    SIGNAL = "SIGNAL"
    
    # Risk Events
    RISK_CHECK = "RISK_CHECK"
    RISK_BREACH = "RISK_BREACH"
    
    # Position Events
    POSITION_UPDATE = "POSITION_UPDATE"
    
    # System Events
    SYSTEM_START = "SYSTEM_START"
    SYSTEM_STOP = "SYSTEM_STOP"
    HEARTBEAT = "HEARTBEAT"


@dataclass
class Event:
    """Base event class"""
    type: EventType
    timestamp: datetime
    data: Dict
    source: str
    
    def __post_init__(self):
        """Ensure timestamp is set"""
        if not self.timestamp:
            self.timestamp = datetime.utcnow()


class EventBus:
    """
    Central event bus for trading system
    
    Implements publish-subscribe pattern for loose coupling
    """
    
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        
    def subscribe(self, event_type: EventType, handler: Callable):
        """Subscribe handler to event type"""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
        print(f"[EventBus] Subscribed {handler.__name__} to {event_type.value}")
    
    async def publish(self, event: Event):
        """Publish event to all subscribers"""
        await self._event_queue.put(event)
    
    async def _process_events(self):
        """Process events from queue"""
        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(), 
                    timeout=1.0
                )
                
                # Get subscribers for this event type
                handlers = self._subscribers.get(event.type, [])
                
                # Call all handlers asynchronously
                tasks = []
                for handler in handlers:
                    task = asyncio.create_task(handler(event))
                    tasks.append(task)
                
                # Wait for all handlers to complete
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"[EventBus] Error processing event: {e}")
    
    async def start(self):
        """Start event bus"""
        self._running = True
        asyncio.create_task(self._process_events())
        print("[EventBus] Started")
    
    async def stop(self):
        """Stop event bus"""
        self._running = False
        print("[EventBus] Stopped")


# Example Usage
async def example_event_bus():
    """Demonstrate event bus pattern"""
    
    bus = EventBus()
    
    # Define handlers
    async def on_market_data(event: Event):
        print(f"[MarketDataHandler] Received: {event.data}")
    
    async def on_new_order(event: Event):
        print(f"[OrderHandler] New order: {event.data}")
    
    async def on_risk_check(event: Event):
        print(f"[RiskHandler] Risk check: {event.data}")
    
    # Subscribe handlers
    bus.subscribe(EventType.MARKET_DATA, on_market_data)
    bus.subscribe(EventType.NEW_ORDER, on_new_order)
    bus.subscribe(EventType.NEW_ORDER, on_risk_check)  # Multiple handlers!
    
    # Start bus
    await bus.start()
    
    # Publish events
    await bus.publish(Event(
        type=EventType.MARKET_DATA,
        timestamp=datetime.utcnow(),
        data={'symbol': 'AAPL', 'price': 150.00},
        source='market_data_feed'
    ))
    
    await bus.publish(Event(
        type=EventType.NEW_ORDER,
        timestamp=datetime.utcnow(),
        data={'symbol': 'AAPL', 'side': 'BUY', 'quantity': 100},
        source='strategy'
    ))
    
    await asyncio.sleep(1)
    await bus.stop()

# asyncio.run(example_event_bus())
\`\`\`

---

## Component Architecture

### High-Level System Components

\`\`\`
┌─────────────────────────────────────────────────────────────────┐
│                     TRADING SYSTEM ARCHITECTURE                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Market Data │───▶│   Strategy   │───▶│     Risk     │      │
│  │     Feed     │    │    Engine    │    │   Manager    │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                    │                    │              │
│         │                    │                    │              │
│         ▼                    ▼                    ▼              │
│  ┌─────────────────────────────────────────────────────┐        │
│  │                    Event Bus                         │        │
│  └─────────────────────────────────────────────────────┘        │
│         │                    │                    │              │
│         │                    │                    │              │
│         ▼                    ▼                    ▼              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Portfolio   │    │     OMS      │    │     EMS      │      │
│  │   Manager    │    │  (Orders)    │    │ (Execution)  │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                    │                    │              │
│         │                    │                    │              │
│         ▼                    ▼                    ▼              │
│  ┌─────────────────────────────────────────────────────┐        │
│  │              Database Layer (PostgreSQL)             │        │
│  └─────────────────────────────────────────────────────┘        │
│         │                    │                    │              │
│         ▼                    ▼                    ▼              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Broker     │    │  Monitoring  │    │    Audit     │      │
│  │  Gateway     │    │  & Alerts    │    │     Log      │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
\`\`\`

### Component Responsibilities

\`\`\`python
"""
Component Definitions
"""

class TradingSystemComponent:
    """Base class for all system components"""
    
    def __init__(self, name: str, event_bus: EventBus):
        self.name = name
        self.event_bus = event_bus
        self._running = False
    
    async def start(self):
        """Start component"""
        self._running = True
        print(f"[{self.name}] Started")
    
    async def stop(self):
        """Stop component"""
        self._running = False
        print(f"[{self.name}] Stopped")


class MarketDataFeed(TradingSystemComponent):
    """
    Receives and normalizes market data from exchanges
    
    Responsibilities:
    - Connect to market data feeds (websockets, FIX)
    - Normalize data across venues
    - Publish market data events
    - Handle reconnections and data gaps
    """
    
    def __init__(self, event_bus: EventBus):
        super().__init__("MarketDataFeed", event_bus)
        self._subscriptions: List[str] = []
    
    async def subscribe(self, symbol: str):
        """Subscribe to symbol"""
        self._subscriptions.append(symbol)
        print(f"[{self.name}] Subscribed to {symbol}")
    
    async def _simulate_market_data(self):
        """Simulate market data stream"""
        import random
        
        while self._running:
            for symbol in self._subscriptions:
                # Simulate quote
                await self.event_bus.publish(Event(
                    type=EventType.QUOTE,
                    timestamp=datetime.utcnow(),
                    data={
                        'symbol': symbol,
                        'bid': 150.00 + random.uniform(-1, 1),
                        'ask': 150.05 + random.uniform(-1, 1),
                        'bid_size': random.randint(100, 1000),
                        'ask_size': random.randint(100, 1000),
                    },
                    source=self.name
                ))
            
            await asyncio.sleep(0.1)  # 10Hz updates
    
    async def start(self):
        await super().start()
        asyncio.create_task(self._simulate_market_data())


class StrategyEngine(TradingSystemComponent):
    """
    Generates trading signals based on market data
    
    Responsibilities:
    - Consume market data
    - Run trading algorithms
    - Generate signals (buy/sell)
    - Manage strategy state
    """
    
    def __init__(self, event_bus: EventBus):
        super().__init__("StrategyEngine", event_bus)
        self.positions: Dict[str, int] = {}
        
        # Subscribe to market data
        event_bus.subscribe(EventType.QUOTE, self._on_market_data)
    
    async def _on_market_data(self, event: Event):
        """React to market data"""
        symbol = event.data['symbol']
        mid_price = (event.data['bid'] + event.data['ask']) / 2
        
        # Simple mean reversion strategy
        if mid_price < 149.00 and self.positions.get(symbol, 0) <= 0:
            # Generate buy signal
            await self.event_bus.publish(Event(
                type=EventType.SIGNAL,
                timestamp=datetime.utcnow(),
                data={
                    'symbol': symbol,
                    'side': 'BUY',
                    'quantity': 100,
                    'reason': 'mean_reversion_buy'
                },
                source=self.name
            ))


class RiskManager(TradingSystemComponent):
    """
    Pre-trade and post-trade risk management
    
    Responsibilities:
    - Pre-trade risk checks (position limits, capital limits)
    - Real-time risk monitoring
    - Position limits enforcement
    - Risk breaches notification
    """
    
    def __init__(self, event_bus: EventBus):
        super().__init__("RiskManager", event_bus)
        
        # Risk limits
        self.max_position_size = 10000
        self.max_notional = 1000000
        self.current_positions: Dict[str, int] = {}
        
        # Subscribe to signals for pre-trade checks
        event_bus.subscribe(EventType.SIGNAL, self._pre_trade_check)
    
    async def _pre_trade_check(self, event: Event):
        """Perform pre-trade risk check"""
        symbol = event.data['symbol']
        side = event.data['side']
        quantity = event.data['quantity']
        
        # Check position limits
        current_pos = self.current_positions.get(symbol, 0)
        new_pos = current_pos + (quantity if side == 'BUY' else -quantity)
        
        if abs(new_pos) > self.max_position_size:
            print(f"[{self.name}] RISK BREACH: Position limit exceeded for {symbol}")
            await self.event_bus.publish(Event(
                type=EventType.RISK_BREACH,
                timestamp=datetime.utcnow(),
                data={
                    'symbol': symbol,
                    'limit': self.max_position_size,
                    'attempted': new_pos,
                    'action': 'REJECT_ORDER'
                },
                source=self.name
            ))
            return
        
        # Risk check passed - convert signal to order
        await self.event_bus.publish(Event(
            type=EventType.NEW_ORDER,
            timestamp=datetime.utcnow(),
            data=event.data,
            source=self.name
        ))


class OrderManagementSystem(TradingSystemComponent):
    """
    Manages order lifecycle
    
    Responsibilities:
    - Order creation and validation
    - Order state management
    - Order routing to execution
    - Order updates and fills
    """
    
    def __init__(self, event_bus: EventBus):
        super().__init__("OMS", event_bus)
        self.orders: Dict[str, Dict] = {}
        self._order_counter = 0
        
        # Subscribe to new orders
        event_bus.subscribe(EventType.NEW_ORDER, self._handle_new_order)
    
    async def _handle_new_order(self, event: Event):
        """Handle new order request"""
        self._order_counter += 1
        order_id = f"ORD-{self._order_counter:08d}"
        
        order = {
            'order_id': order_id,
            'symbol': event.data['symbol'],
            'side': event.data['side'],
            'quantity': event.data['quantity'],
            'status': 'NEW',
            'filled_quantity': 0,
            'timestamp': datetime.utcnow()
        }
        
        self.orders[order_id] = order
        
        print(f"[{self.name}] Created order {order_id}: {order['side']} {order['quantity']} {order['symbol']}")
        
        # Publish order accepted
        await self.event_bus.publish(Event(
            type=EventType.ORDER_ACCEPTED,
            timestamp=datetime.utcnow(),
            data=order,
            source=self.name
        ))


class ExecutionManagementSystem(TradingSystemComponent):
    """
    Handles order execution
    
    Responsibilities:
    - Route orders to brokers/exchanges
    - Execution algorithms (VWAP, TWAP, etc.)
    - Fill reporting
    - Slippage tracking
    """
    
    def __init__(self, event_bus: EventBus):
        super().__init__("EMS", event_bus)
        
        # Subscribe to accepted orders
        event_bus.subscribe(EventType.ORDER_ACCEPTED, self._execute_order)
    
    async def _execute_order(self, event: Event):
        """Execute order"""
        order = event.data
        
        # Simulate execution (in reality, would route to broker)
        await asyncio.sleep(0.5)  # Simulate execution delay
        
        # Publish fill
        await self.event_bus.publish(Event(
            type=EventType.ORDER_FILLED,
            timestamp=datetime.utcnow(),
            data={
                'order_id': order['order_id'],
                'symbol': order['symbol'],
                'side': order['side'],
                'quantity': order['quantity'],
                'price': 150.00,  # Simulated fill price
                'commission': 1.00
            },
            source=self.name
        ))


class PortfolioManager(TradingSystemComponent):
    """
    Tracks positions and P&L
    
    Responsibilities:
    - Position tracking (real-time)
    - P&L calculation
    - Portfolio analytics
    - Position reconciliation
    """
    
    def __init__(self, event_bus: EventBus):
        super().__init__("PortfolioManager", event_bus)
        self.positions: Dict[str, int] = {}
        self.avg_prices: Dict[str, float] = {}
        
        # Subscribe to fills
        event_bus.subscribe(EventType.ORDER_FILLED, self._update_position)
    
    async def _update_position(self, event: Event):
        """Update position on fill"""
        symbol = event.data['symbol']
        side = event.data['side']
        quantity = event.data['quantity']
        price = event.data['price']
        
        # Update position
        current_pos = self.positions.get(symbol, 0)
        qty_delta = quantity if side == 'BUY' else -quantity
        new_pos = current_pos + qty_delta
        
        self.positions[symbol] = new_pos
        
        # Update average price (simplified)
        self.avg_prices[symbol] = price
        
        print(f"[{self.name}] Position update: {symbol} = {new_pos} @ {price}")
        
        # Publish position update
        await self.event_bus.publish(Event(
            type=EventType.POSITION_UPDATE,
            timestamp=datetime.utcnow(),
            data={
                'symbol': symbol,
                'position': new_pos,
                'avg_price': price
            },
            source=self.name
        ))
\`\`\`

---

## Complete System Integration

\`\`\`python
"""
Complete Trading System
"""

class TradingSystem:
    """
    Main trading system orchestrator
    """
    
    def __init__(self):
        # Create event bus
        self.event_bus = EventBus()
        
        # Create components
        self.market_data = MarketDataFeed(self.event_bus)
        self.strategy = StrategyEngine(self.event_bus)
        self.risk = RiskManager(self.event_bus)
        self.oms = OrderManagementSystem(self.event_bus)
        self.ems = ExecutionManagementSystem(self.event_bus)
        self.portfolio = PortfolioManager(self.event_bus)
        
        self.components = [
            self.market_data,
            self.strategy,
            self.risk,
            self.oms,
            self.ems,
            self.portfolio
        ]
    
    async def start(self):
        """Start trading system"""
        print("=" * 70)
        print("STARTING TRADING SYSTEM")
        print("=" * 70)
        
        # Start event bus
        await self.event_bus.start()
        
        # Start all components
        for component in self.components:
            await component.start()
        
        # Subscribe to symbols
        await self.market_data.subscribe('AAPL')
        
        print("\\nTRADING SYSTEM READY\\n")
    
    async def stop(self):
        """Stop trading system"""
        print("\\n" + "=" * 70)
        print("STOPPING TRADING SYSTEM")
        print("=" * 70)
        
        # Stop all components
        for component in reversed(self.components):
            await component.stop()
        
        # Stop event bus
        await self.event_bus.stop()
        
        print("TRADING SYSTEM STOPPED")
    
    async def run(self, duration: int = 10):
        """Run trading system for specified duration"""
        await self.start()
        await asyncio.sleep(duration)
        await self.stop()


# Example: Run trading system
async def main():
    system = TradingSystem()
    await system.run(duration=5)

# asyncio.run(main())
\`\`\`

---

## Microservices Architecture

For **production systems**, components are separated into microservices:

\`\`\`
┌─────────────────────────────────────────────────────────────────┐
│                  MICROSERVICES ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────┐  │
│  │   Market   │  │  Strategy  │  │    Risk    │  │   OMS    │  │
│  │    Data    │  │  Service   │  │  Service   │  │ Service  │  │
│  │  Service   │  └────────────┘  └────────────┘  └──────────┘  │
│  └────────────┘         │                │              │        │
│        │                │                │              │        │
│        └────────────────┴────────────────┴──────────────┘        │
│                              │                                    │
│                    ┌─────────▼─────────┐                         │
│                    │   Message Queue   │                         │
│                    │  (Kafka/RabbitMQ) │                         │
│                    └─────────┬─────────┘                         │
│                              │                                    │
│        ┌─────────────────────┴──────────────────────┐            │
│        │                     │              │        │            │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────┐  │
│  │    EMS     │  │ Portfolio  │  │   Audit    │  │  Alert   │  │
│  │  Service   │  │  Service   │  │  Service   │  │ Service  │  │
│  └────────────┘  └────────────┘  └────────────┘  └──────────┘  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
\`\`\`

**Benefits:**
- **Independent scaling:** Scale market data service separately from strategy
- **Independent deployment:** Deploy risk service without touching others
- **Fault isolation:** Strategy crash doesn't kill market data
- **Technology flexibility:** Use C++ for market data, Python for strategy

---

## Data Flow Patterns

### 1. Market Data Flow

\`\`\`
Exchange → Market Data Service → Normalization → Event Bus → Subscribers
\`\`\`

### 2. Order Flow

\`\`\`
Strategy → Signal → Risk Check → OMS → EMS → Broker → Exchange
                        ↓           ↓      ↓
                     Reject?    Audit   Fill
\`\`\`

### 3. Position Update Flow

\`\`\`
Fill Event → Portfolio Manager → Position Update → Risk Monitor → Alerts
\`\`\`

---

## State Management

\`\`\`python
"""
State Management for Trading Systems
"""

from enum import Enum
from typing import Optional
import json
import redis

class OrderState(Enum):
    """Order lifecycle states"""
    PENDING = "PENDING"
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class StateManager:
    """
    Centralized state management
    
    Uses Redis for shared state across services
    """
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True
        )
    
    def set_order_state(self, order_id: str, state: OrderState, data: Dict):
        """Set order state"""
        key = f"order:{order_id}"
        value = {
            'state': state.value,
            'data': data,
            'timestamp': datetime.utcnow().isoformat()
        }
        self.redis_client.set(key, json.dumps(value))
    
    def get_order_state(self, order_id: str) -> Optional[Dict]:
        """Get order state"""
        key = f"order:{order_id}"
        value = self.redis_client.get(key)
        if value:
            return json.loads(value)
        return None
    
    def set_position(self, symbol: str, quantity: int, avg_price: float):
        """Set current position"""
        key = f"position:{symbol}"
        value = {
            'quantity': quantity,
            'avg_price': avg_price,
            'timestamp': datetime.utcnow().isoformat()
        }
        self.redis_client.set(key, json.dumps(value))
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get current position"""
        key = f"position:{symbol}"
        value = self.redis_client.get(key)
        if value:
            return json.loads(value)
        return None
    
    def get_all_positions(self) -> Dict[str, Dict]:
        """Get all positions"""
        positions = {}
        for key in self.redis_client.scan_iter("position:*"):
            symbol = key.split(':')[1]
            positions[symbol] = self.get_position(symbol)
        return positions
\`\`\`

---

## Error Handling and Recovery

\`\`\`python
"""
Error Handling Strategies
"""

class CircuitBreaker:
    """
    Circuit breaker for external services
    
    Prevents cascading failures when broker/exchange is down
    """
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func, *args, **kwargs):
        """Execute function through circuit breaker"""
        
        if self.state == 'OPEN':
            # Check if timeout has passed
            if datetime.utcnow().timestamp() - self.last_failure_time > self.timeout:
                self.state = 'HALF_OPEN'
                print("[CircuitBreaker] Attempting recovery (HALF_OPEN)")
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            
            if self.state == 'HALF_OPEN':
                self.state = 'CLOSED'
                self.failure_count = 0
                print("[CircuitBreaker] Recovery successful (CLOSED)")
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.utcnow().timestamp()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'OPEN'
                print(f"[CircuitBreaker] OPEN after {self.failure_count} failures")
            
            raise


class RetryStrategy:
    """
    Exponential backoff retry strategy
    """
    
    @staticmethod
    async def retry_with_backoff(
        func,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0
    ):
        """Retry function with exponential backoff"""
        
        for attempt in range(max_retries):
            try:
                return await func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                
                delay = min(base_delay * (2 ** attempt), max_delay)
                print(f"[Retry] Attempt {attempt + 1} failed, retrying in {delay}s")
                await asyncio.sleep(delay)
\`\`\`

---

## Performance Monitoring

\`\`\`python
"""
Performance Monitoring
"""

import time
from functools import wraps

class PerformanceMonitor:
    """
    Monitor system performance metrics
    """
    
    def __init__(self):
        self.metrics = {}
    
    def record_latency(self, component: str, operation: str, latency_ms: float):
        """Record operation latency"""
        key = f"{component}.{operation}"
        if key not in self.metrics:
            self.metrics[key] = {
                'count': 0,
                'total_ms': 0,
                'min_ms': float('inf'),
                'max_ms': 0
            }
        
        m = self.metrics[key]
        m['count'] += 1
        m['total_ms'] += latency_ms
        m['min_ms'] = min(m['min_ms'], latency_ms)
        m['max_ms'] = max(m['max_ms'], latency_ms)
    
    def get_statistics(self) -> Dict:
        """Get performance statistics"""
        stats = {}
        for key, m in self.metrics.items():
            stats[key] = {
                'count': m['count'],
                'avg_ms': m['total_ms'] / m['count'],
                'min_ms': m['min_ms'],
                'max_ms': m['max_ms']
            }
        return stats
    
    def print_statistics(self):
        """Print performance statistics"""
        print("\\n" + "=" * 70)
        print("PERFORMANCE STATISTICS")
        print("=" * 70)
        
        stats = self.get_statistics()
        for key, s in stats.items():
            print(f"\\n{key}:")
            print(f"  Count: {s['count']}")
            print(f"  Avg: {s['avg_ms']:.2f}ms")
            print(f"  Min: {s['min_ms']:.2f}ms")
            print(f"  Max: {s['max_ms']:.2f}ms")


def measure_latency(monitor: PerformanceMonitor, component: str, operation: str):
    """Decorator to measure function latency"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.time()
            result = await func(*args, **kwargs)
            end = time.time()
            latency_ms = (end - start) * 1000
            monitor.record_latency(component, operation, latency_ms)
            return result
        return wrapper
    return decorator
\`\`\`

---

## Summary

**Key Architectural Principles:**

1. **Event-Driven:** Loose coupling, asynchronous processing
2. **Microservices:** Independent scaling, deployment, fault isolation
3. **State Management:** Centralized state in Redis for consistency
4. **Error Handling:** Circuit breakers, retries, graceful degradation
5. **Performance Monitoring:** Track latency at every component
6. **Regulatory Compliance:** Audit logs for every action

**Real-World Architecture Comparison:**

| Component | Interactive Brokers | Citadel Securities | Your System |
|-----------|-------------------|-------------------|-------------|
| Market Data | C++ (low latency) | FPGA (ultra-low) | Python/Asyncio |
| Strategy | Multiple languages | C++/Python | Python |
| OMS | Proprietary | Proprietary | Python + PostgreSQL |
| Message Bus | Custom IPC | Custom + Kafka | Kafka/Redis |
| Execution | FIX to 135+ venues | Direct exchange | Broker API + FIX |

**Next Steps:**
- **Module 14.2:** Deep dive into OMS (Order Management System)
- **Module 14.3:** EMS and execution algorithms
- **Module 14.4:** FIX protocol implementation
- **Module 14.9:** Low-latency programming (C++ optimization)

This architecture provides the **foundation** for a production trading system. Each component will be explored in detail in subsequent sections.
`,
};

