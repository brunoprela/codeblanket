export const completeTradingSystem = {
  title: 'Project: Complete Trading System',
  id: 'complete-trading-system',
  content: `
# Project: Complete Trading System

## Project Overview

Build an **end-to-end production-grade trading system** integrating all concepts from Module 14. This capstone project demonstrates mastery of trading infrastructure.

**System Components:**
1. Order Management System (OMS)
2. Execution Management System (EMS)
3. FIX Protocol connectivity
4. Smart Order Routing (SOR)
5. Real-time position tracking
6. P&L calculation (realized + unrealized)
7. Trade reconciliation
8. Risk management
9. System monitoring and alerting
10. Disaster recovery and failover
11. Production deployment pipeline

**Project Goals:**
- Execute 1,000+ orders/day
- Order latency <100ms (p99)
- Order success rate >99%
- Position tracking <1s latency
- Real-time P&L updates
- Trade reconciliation rate >99%
- System uptime >99.99%

**Real-World Comparison:**

| Metric | Your System (Target) | Retail Broker | Professional Firm |
|--------|----------------------|---------------|-------------------|
| Order Latency | <100ms | <1s | <10ms |
| Throughput | 100 orders/sec | 1K orders/sec | 100K+ orders/sec |
| Uptime | 99.99% | 99.9% | 99.999% |
| Position Updates | <1s | <5s | <100ms |
| Trade Recon Rate | >99% | >98% | >99.9% |

---

## System Architecture

\`\`\`
┌───────────────────────────────────────────────────────────────────────┐
│                        COMPLETE TRADING SYSTEM                          │
├───────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────┐       ┌──────────────┐       ┌──────────────┐         │
│  │   Trader    │──────▶│   REST API   │──────▶│   Strategy   │         │
│  │  (Web UI)   │       │  (FastAPI)   │       │    Engine    │         │
│  └─────────────┘       └──────┬───────┘       └──────┬───────┘         │
│                                │                      │                  │
│                                ▼                      ▼                  │
│  ┌────────────────────────────────────────────────────────────┐        │
│  │                     Event Bus (Kafka)                       │        │
│  └────┬───────────┬───────────┬───────────┬──────────┬────────┘        │
│       │           │           │           │          │                  │
│       ▼           ▼           ▼           ▼          ▼                  │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐ ┌──────────┐         │
│  │  Risk  │  │  OMS   │  │  EMS   │  │Position│ │   P&L    │         │
│  │Manager │  │(Orders)│  │(Exec)  │  │Tracker │ │Calculator│         │
│  └───┬────┘  └───┬────┘  └───┬────┘  └───┬────┘ └─────┬────┘         │
│      │           │           │           │            │                 │
│      └───────────┴───────────┴───────────┴────────────┘                │
│                              │                                           │
│                              ▼                                           │
│          ┌──────────────────────────────────────┐                      │
│          │      PostgreSQL (Orders, Positions)   │                      │
│          │      Redis (Real-time Cache)          │                      │
│          │      TimescaleDB (Market Data)        │                      │
│          └──────────────────┬───────────────────┘                      │
│                              │                                           │
│                              ▼                                           │
│  ┌───────────┐       ┌──────────────┐       ┌──────────────┐          │
│  │  Broker   │◀──────│  FIX Engine  │──────▶│  Market Data │          │
│  │ (Alpaca)  │       │ (QuickFIX)   │       │     Feed     │          │
│  └───────────┘       └──────────────┘       └──────────────┘          │
│                                                                           │
│  ┌───────────────────────────────────────────────────────────────┐    │
│  │            Monitoring & Alerting (Prometheus + Grafana)        │    │
│  └───────────────────────────────────────────────────────────────┘    │
│                                                                           │
└───────────────────────────────────────────────────────────────────────┘
\`\`\`

---

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1)

\`\`\`bash
# Setup project structure
trading-system/
├── backend/
│   ├── app/
│   │   ├── api/          # REST API endpoints
│   │   ├── core/         # Core business logic
│   │   ├── models/       # Database models
│   │   ├── services/     # Business services
│   │   └── utils/        # Utilities
│   ├── tests/
│   ├── requirements.txt
│   └── docker-compose.yml
├── frontend/
│   ├── src/
│   │   ├── components/   # React components
│   │   ├── pages/        # Pages
│   │   └── api/          # API client
│   └── package.json
├── infrastructure/
│   ├── terraform/        # Infrastructure as code
│   ├── kubernetes/       # K8s manifests
│   └── monitoring/       # Prometheus/Grafana configs
└── docs/
    ├── architecture.md
    ├── api.md
    └── deployment.md

# Install dependencies
cd backend
pip install -r requirements.txt

# Setup databases
docker-compose up -d postgres redis timescaledb

# Run migrations
alembic upgrade head

# Start services
python -m app.main
\`\`\`

### Phase 2: OMS Implementation (Week 2)

\`\`\`python
"""
Complete Order Management System
backend/app/services/oms.py
"""

from typing import Optional, List
from decimal import Decimal
from datetime import datetime
from sqlalchemy.orm import Session
from app.models.order import Order, OrderState, OrderSide, OrderType
from app.core.events import EventBus, OrderEvent
from app.services.risk import RiskManager
import logging

logger = logging.getLogger(__name__)

class OrderManagementSystem:
    """
    Production OMS implementation
    
    Features:
    - Order validation
    - State management
    - Event publishing
    - Risk integration
    - Audit logging
    """
    
    def __init__(
        self,
        db: Session,
        event_bus: EventBus,
        risk_manager: RiskManager
    ):
        self.db = db
        self.event_bus = event_bus
        self.risk_manager = risk_manager
        
        # Statistics
        self.orders_created = 0
        self.orders_rejected = 0
    
    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        price: Optional[Decimal] = None,
        account: str = "default",
        strategy: Optional[str] = None
    ) -> Order:
        """
        Create new order
        
        Workflow:
        1. Validate order parameters
        2. Pre-trade risk check
        3. Persist to database
        4. Publish order created event
        5. Return order
        """
        # Step 1: Create order object
        order = Order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            account=account,
            strategy=strategy,
            state=OrderState.PENDING_RISK,
            remaining_quantity=quantity
        )
        
        # Step 2: Validate
        validation_errors = self._validate_order(order)
        if validation_errors:
            order.state = OrderState.REJECTED
            order.rejection_reason = "; ".join(validation_errors)
            self.orders_rejected += 1
            
            logger.warning(f"Order validation failed: {validation_errors}")
            raise ValueError(f"Order validation failed: {validation_errors}")
        
        # Step 3: Risk check
        risk_check = await self.risk_manager.pre_trade_check(order)
        if not risk_check.passed:
            order.state = OrderState.RISK_REJECTED
            order.rejection_reason = risk_check.reason
            self.orders_rejected += 1
            
            logger.warning(f"Risk check failed: {risk_check.reason}")
            raise ValueError(f"Risk check failed: {risk_check.reason}")
        
        # Step 4: Persist
        order.state = OrderState.NEW
        order.submitted_at = datetime.utcnow()
        
        self.db.add(order)
        self.db.commit()
        self.db.refresh(order)
        
        self.orders_created += 1
        
        logger.info(f"Order created: {order.order_id} {order.symbol} {order.side} {order.quantity}")
        
        # Step 5: Publish event
        await self.event_bus.publish(OrderEvent(
            event_type='order_created',
            order_id=order.order_id,
            data=order.to_dict()
        ))
        
        return order
    
    def _validate_order(self, order: Order) -> List[str]:
        """Validate order parameters"""
        errors = []
        
        if not order.symbol:
            errors.append("Symbol is required")
        
        if order.quantity <= 0:
            errors.append("Quantity must be positive")
        
        if order.order_type == OrderType.LIMIT and not order.price:
            errors.append("Limit order requires price")
        
        if order.order_type == OrderType.LIMIT and order.price <= 0:
            errors.append("Price must be positive")
        
        return errors
    
    async def cancel_order(self, order_id: str) -> Order:
        """Cancel order"""
        order = self.db.query(Order).filter(Order.order_id == order_id).first()
        
        if not order:
            raise ValueError(f"Order not found: {order_id}")
        
        if order.state in [OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED]:
            raise ValueError(f"Cannot cancel order in state {order.state}")
        
        order.state = OrderState.CANCELLED
        order.completed_at = datetime.utcnow()
        
        self.db.commit()
        
        logger.info(f"Order cancelled: {order_id}")
        
        await self.event_bus.publish(OrderEvent(
            event_type='order_cancelled',
            order_id=order_id,
            data=order.to_dict()
        ))
        
        return order
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self.db.query(Order).filter(Order.order_id == order_id).first()
    
    def get_open_orders(self, account: Optional[str] = None) -> List[Order]:
        """Get all open orders"""
        query = self.db.query(Order).filter(
            Order.state.in_([OrderState.NEW, OrderState.PARTIALLY_FILLED])
        )
        
        if account:
            query = query.filter(Order.account == account)
        
        return query.all()
    
    def get_statistics(self) -> dict:
        """Get OMS statistics"""
        total_orders = self.db.query(Order).count()
        filled_orders = self.db.query(Order).filter(Order.state == OrderState.FILLED).count()
        
        return {
            'orders_created': self.orders_created,
            'orders_rejected': self.orders_rejected,
            'total_orders': total_orders,
            'filled_orders': filled_orders,
            'fill_rate': filled_orders / total_orders if total_orders > 0 else 0
        }
\`\`\`

### Phase 3: Position Tracking & P&L (Week 3)

\`\`\`python
"""
Real-time Position Tracker with P&L
backend/app/services/position_tracker.py
"""

from typing import Dict, List, Optional
from decimal import Decimal
from datetime import datetime
from app.models.position import Position
from app.models.fill import Fill
import redis
import json

class PositionTracker:
    """
    Real-time position tracking
    
    Features:
    - Update positions on fills
    - Calculate P&L (realized + unrealized)
    - Redis cache for fast lookups
    - PostgreSQL for persistence
    """
    
    def __init__(self, db, redis_client: redis.Redis):
        self.db = db
        self.redis = redis_client
    
    async def process_fill(self, fill: Fill):
        """
        Process fill and update positions
        
        Steps:
        1. Get current position
        2. Update quantity and avg cost
        3. Calculate realized P&L
        4. Update database
        5. Update Redis cache
        """
        position = self._get_or_create_position(
            symbol=fill.symbol,
            account=fill.account,
            strategy=fill.strategy
        )
        
        # Calculate position update
        if fill.side == 'BUY':
            # Buying adds to position
            new_quantity = position.quantity + fill.quantity
            
            # Update average cost
            if new_quantity > 0:
                total_cost = (position.quantity * position.avg_cost) + (fill.quantity * fill.price)
                position.avg_cost = total_cost / new_quantity
            
            position.quantity = new_quantity
            realized_pnl = Decimal('0')
        
        else:  # SELL
            # Selling reduces position
            new_quantity = position.quantity - fill.quantity
            
            # Calculate realized P&L
            realized_pnl = (fill.price - position.avg_cost) * fill.quantity
            position.realized_pnl += realized_pnl
            
            position.quantity = new_quantity
            
            # If position is closed, reset avg cost
            if new_quantity == 0:
                position.avg_cost = Decimal('0')
                position.closed_at = datetime.utcnow()
        
        position.updated_at = datetime.utcnow()
        
        # Save to database
        self.db.commit()
        
        # Update Redis cache
        await self._update_redis_cache(position)
        
        return {
            'position': position,
            'realized_pnl': float(realized_pnl)
        }
    
    def _get_or_create_position(
        self,
        symbol: str,
        account: str,
        strategy: Optional[str] = None
    ) -> Position:
        """Get existing position or create new one"""
        position = self.db.query(Position).filter(
            Position.symbol == symbol,
            Position.account == account,
            Position.strategy == strategy,
            Position.closed_at.is_(None)
        ).first()
        
        if not position:
            position = Position(
                symbol=symbol,
                account=account,
                strategy=strategy,
                quantity=Decimal('0'),
                avg_cost=Decimal('0'),
                realized_pnl=Decimal('0'),
                unrealized_pnl=Decimal('0')
            )
            self.db.add(position)
        
        return position
    
    async def update_unrealized_pnl(
        self,
        symbol: str,
        current_price: Decimal
    ):
        """Update unrealized P&L for all positions in symbol"""
        positions = self.db.query(Position).filter(
            Position.symbol == symbol,
            Position.closed_at.is_(None)
        ).all()
        
        for position in positions:
            if position.quantity != 0:
                position.unrealized_pnl = position.quantity * (current_price - position.avg_cost)
                position.current_price = current_price
                position.price_updated_at = datetime.utcnow()
                
                # Update Redis
                await self._update_redis_cache(position)
        
        self.db.commit()
    
    async def _update_redis_cache(self, position: Position):
        """Update position in Redis for fast lookups"""
        key = f"pos:{position.account}:{position.symbol}"
        
        data = {
            'symbol': position.symbol,
            'account': position.account,
            'quantity': str(position.quantity),
            'avg_cost': str(position.avg_cost),
            'realized_pnl': str(position.realized_pnl),
            'unrealized_pnl': str(position.unrealized_pnl),
            'updated_at': position.updated_at.isoformat()
        }
        
        self.redis.set(key, json.dumps(data), ex=3600)  # Expire in 1 hour
    
    async def get_position_from_cache(
        self,
        symbol: str,
        account: str
    ) -> Optional[Dict]:
        """Get position from Redis (fast)"""
        key = f"pos:{account}:{symbol}"
        data = self.redis.get(key)
        
        if data:
            return json.loads(data)
        
        return None
    
    def get_portfolio_summary(self, account: str) -> Dict:
        """Get portfolio summary for account"""
        positions = self.db.query(Position).filter(
            Position.account == account,
            Position.closed_at.is_(None)
        ).all()
        
        total_realized_pnl = sum(p.realized_pnl for p in positions)
        total_unrealized_pnl = sum(p.unrealized_pnl for p in positions)
        total_pnl = total_realized_pnl + total_unrealized_pnl
        
        return {
            'account': account,
            'num_positions': len(positions),
            'realized_pnl': float(total_realized_pnl),
            'unrealized_pnl': float(total_unrealized_pnl),
            'total_pnl': float(total_pnl),
            'positions': [p.to_dict() for p in positions]
        }
\`\`\`

---

## Complete System Integration

\`\`\`python
"""
Main Trading System Orchestrator
backend/app/main.py
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio

from app.services.oms import OrderManagementSystem
from app.services.ems import ExecutionManagementSystem
from app.services.position_tracker import PositionTracker
from app.services.risk import RiskManager
from app.core.events import EventBus
from app.core.database import SessionLocal, engine, redis_client
from app.api import orders, positions, portfolio
from app.monitoring.metrics import setup_metrics

# Lifespan for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting Trading System...")
    
    # Initialize core services
    app.state.event_bus = EventBus()
    app.state.risk_manager = RiskManager()
    app.state.oms = OrderManagementSystem(
        db=SessionLocal(),
        event_bus=app.state.event_bus,
        risk_manager=app.state.risk_manager
    )
    app.state.ems = ExecutionManagementSystem(
        event_bus=app.state.event_bus
    )
    app.state.position_tracker = PositionTracker(
        db=SessionLocal(),
        redis_client=redis_client
    )
    
    # Start event bus
    await app.state.event_bus.start()
    
    # Setup monitoring
    setup_metrics(app)
    
    print("✓ Trading System Ready")
    
    yield
    
    # Shutdown
    print("Shutting down Trading System...")
    await app.state.event_bus.stop()
    print("✓ Shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Trading System API",
    description="Production trading infrastructure",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(orders.router, prefix="/api/orders", tags=["orders"])
app.include_router(positions.router, prefix="/api/positions", tags=["positions"])
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["portfolio"])

# Health check
@app.get("/health")
async def health_check():
    """System health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "oms": "up",
            "ems": "up",
            "positions": "up",
            "database": "up",
            "redis": "up"
        }
    }

# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
\`\`\`

---

## Project Deliverables

### 1. Working System
- ✅ REST API (FastAPI)
- ✅ Database (PostgreSQL + Redis + TimescaleDB)
- ✅ Order Management (create, cancel, modify)
- ✅ Execution (broker integration via Alpaca API)
- ✅ Position tracking (real-time)
- ✅ P&L calculation (real-time)
- ✅ Risk management (pre-trade checks)
- ✅ Trade reconciliation (EOD)
- ✅ Monitoring (Prometheus + Grafana)

### 2. Documentation
- Architecture diagram
- API documentation (OpenAPI/Swagger)
- Database schema
- Deployment guide
- Runbook (incident response)

### 3. Tests
- Unit tests (>80% coverage)
- Integration tests (order flow end-to-end)
- Load tests (100 orders/sec sustained)
- Chaos tests (failure scenarios)

### 4. Deployment
- Docker containers
- Kubernetes manifests
- Blue-green deployment pipeline
- Monitoring dashboards

### 5. Demo
- Place orders via API
- Show real-time positions
- Display P&L updates
- Demonstrate monitoring
- Show automated reconciliation

---

## Success Criteria

✅ **Functional Requirements:**
- [ ] Execute 1,000+ orders per day
- [ ] Order latency p99 <100ms
- [ ] Order success rate >99%
- [ ] Position updates <1s latency
- [ ] Real-time P&L calculation
- [ ] Trade reconciliation rate >99%
- [ ] Zero data loss (ACID compliance)

✅ **Non-Functional Requirements:**
- [ ] System uptime >99.99%
- [ ] Graceful degradation (no crashes)
- [ ] Instant rollback capability
- [ ] Comprehensive monitoring
- [ ] Audit trail for all actions
- [ ] Documentation complete

✅ **Code Quality:**
- [ ] Type hints throughout (mypy passing)
- [ ] Unit test coverage >80%
- [ ] Integration tests passing
- [ ] Linting clean (black, ruff)
- [ ] Security scan passing

---

## Deployment Instructions

\`\`\`bash
# 1. Clone repository
git clone https://github.com/your-username/trading-system
cd trading-system

# 2. Setup environment
cp .env.example .env
# Edit .env with your configuration

# 3. Start infrastructure
docker-compose up -d postgres redis timescaledb prometheus grafana

# 4. Run database migrations
cd backend
alembic upgrade head

# 5. Start backend
python -m app.main

# 6. Start frontend (in another terminal)
cd frontend
npm install
npm run dev

# 7. Access system
# API: http://localhost:8000
# API Docs: http://localhost:8000/docs
# Frontend: http://localhost:3000
# Grafana: http://localhost:3001

# 8. Run tests
pytest tests/ -v --cov=app

# 9. Load test
locust -f tests/load_test.py --host=http://localhost:8000
\`\`\`

---

## Next Steps

**Congratulations!** 🎉 You've completed Module 14: Building Trading Infrastructure.

You now have the knowledge and practical experience to build production-grade trading systems used by:
- Hedge funds (Citadel, Two Sigma, Jane Street)
- Trading firms (Virtu, Flow Traders, IMC)
- Brokers (Interactive Brokers, TD Ameritrade)
- Banks (Goldman Sachs, Morgan Stanley, JP Morgan)

**Skills Acquired:**
✅ Trading system architecture (microservices, event-driven)
✅ Order Management System (OMS) implementation
✅ Execution Management System (EMS) with smart routing
✅ FIX protocol integration for broker connectivity
✅ Real-time position tracking and P&L calculation
✅ Trade reconciliation and settlement
✅ Low-latency optimization techniques
✅ Production deployment strategies
✅ Disaster recovery and failover
✅ System monitoring and alerting

**Continue Your Journey:**
- **Module 15**: Algorithmic Trading Strategies
- **Module 16**: High-Frequency Trading (HFT)
- **Module 17**: Options Trading Systems
- **Module 18**: Risk Management
- **Module 19**: Portfolio Management
- **Module 20**: Market Making Strategies

**Career Opportunities:**
- Trading Systems Engineer: $150K-$300K+
- Quantitative Developer: $200K-$500K+
- Trading Infrastructure Architect: $250K-$600K+
- Platform Engineer (Trading): $180K-$400K+

**Keep Learning:**
- Contribute to open-source trading projects (QuantConnect, Lean, Zipline)
- Build your own trading strategies
- Participate in algorithmic trading competitions
- Network with trading professionals on LinkedIn/Twitter

**Good luck building the future of trading! 🚀**
`,
};
