import { Content } from '@/lib/types';

const productionBacktestingEngineProject: Content = {
  title: 'Project: Production Backtesting Engine',
  description:
    'Capstone project building a complete production-grade backtesting system with multiple strategy support, performance reporting, and deployment infrastructure',
  sections: [
    {
      title: 'Building a Complete Production System',
      content: `
# Project: Production Backtesting Engine

This capstone project integrates everything learned in this module into a complete, production-grade backtesting system.

## System Requirements

### Functional Requirements
- Support multiple strategies simultaneously
- Handle multi-asset portfolios
- Real-time performance monitoring
- Automated report generation
- Parameter optimization integration
- Paper trading mode
- Production deployment ready

### Non-Functional Requirements
- Process 10+ years of daily data in <5 minutes
- Support 100+ concurrent backtests
- 99.9% uptime for live trading
- Sub-second latency for order execution
- Complete audit trail
- Disaster recovery capability

## Architecture Overview

\`\`\`
┌─────────────────────────────────────────────────────────┐
│                   API Gateway                            │
│              (FastAPI / REST)                            │
└──────────────────┬──────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
┌───────▼────────┐   ┌────────▼────────┐
│  Backtest      │   │   Live Trading   │
│  Engine        │   │   Engine         │
└───────┬────────┘   └────────┬────────┘
        │                     │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │   Strategy Manager   │
        └──────────┬──────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
┌───────▼────────┐   ┌────────▼────────┐
│  Data Handler  │   │  Execution      │
│                │   │  Handler        │
└───────┬────────┘   └────────┬────────┘
        │                     │
┌───────▼────────┐   ┌────────▼────────┐
│  PostgreSQL    │   │  Redis Cache    │
│  (Timescale)   │   │                 │
└────────────────┘   └─────────────────┘
\`\`\`

## Complete Implementation

\`\`\`python
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import redis
import psycopg2
from sqlalchemy import create_engine
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradingMode(Enum):
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"

@dataclass
class Order:
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    timestamp: datetime
    limit_price: Optional[float] = None
    order_id: str = field(default_factory=lambda: f"ORD_{int(datetime.now().timestamp())}")

@dataclass
class Fill:
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    fill_price: float
    timestamp: datetime
    commission: float = 0.0

@dataclass
class Position:
    symbol: str
    quantity: int
    avg_price: float
    market_value: float = 0.0
    unrealized_pnl: float = 0.0

class Strategy(ABC):
    """Abstract strategy interface"""
    
    def __init__(self, strategy_id: str, params: Dict):
        self.strategy_id = strategy_id
        self.params = params
        self.positions: Dict[str, Position] = {}
    
    @abstractmethod
    async def on_data(self, data: Dict) -> List[Order]:
        """Process market data and generate orders"""
        pass
    
    @abstractmethod
    async def on_fill(self, fill: Fill):
        """Process fill notification"""
        pass

class DataManager:
    """
    Centralized data management
    """
    
    def __init__(self, db_url: str, redis_url: str):
        self.engine = create_engine(db_url)
        self.redis = redis.from_url(redis_url)
        logger.info("DataManager initialized")
    
    async def get_historical_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch historical data"""
        
        # Check Redis cache first
        cache_key = f"data:{','.join(symbols)}:{start_date}:{end_date}"
        cached = self.redis.get(cache_key)
        
        if cached:
            logger.info(f"Cache hit for {cache_key}")
            return pd.read_json(cached)
        
        # Fetch from database
        query = f"""
            SELECT * FROM market_data
            WHERE symbol IN ({','.join([f"'{s}'" for s in symbols])})
            AND timestamp BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY timestamp
        """
        
        df = pd.read_sql(query, self.engine)
        
        # Cache for 1 hour
        self.redis.setex(cache_key, 3600, df.to_json())
        
        logger.info(f"Loaded {len(df)} rows from database")
        return df
    
    async def store_backtest_results(
        self,
        backtest_id: str,
        results: Dict
    ):
        """Store backtest results"""
        
        query = """
            INSERT INTO backtest_results 
            (backtest_id, strategy_id, metrics, timestamp)
            VALUES (%s, %s, %s, %s)
        """
        
        with self.engine.connect() as conn:
            conn.execute(
                query,
                (
                    backtest_id,
                    results['strategy_id'],
                    json.dumps(results['metrics']),
                    datetime.now()
                )
            )
        
        logger.info(f"Stored results for backtest {backtest_id}")

class ExecutionEngine:
    """
    Handles order execution across all modes
    """
    
    def __init__(self, mode: TradingMode):
        self.mode = mode
        self.pending_orders: Dict[str, Order] = {}
        logger.info(f"ExecutionEngine initialized in {mode.value} mode")
    
    async def submit_order(self, order: Order) -> Fill:
        """Submit order for execution"""
        
        if self.mode == TradingMode.BACKTEST:
            return await self._simulate_fill(order)
        elif self.mode == TradingMode.PAPER:
            return await self._paper_fill(order)
        else:  # LIVE
            return await self._live_fill(order)
    
    async def _simulate_fill(self, order: Order) -> Fill:
        """Simulate fill in backtest"""
        
        # Simple simulation: immediate fill with slippage
        slippage = 0.0005  # 5 bps
        
        if order.side == OrderSide.BUY:
            fill_price = order.limit_price * (1 + slippage) if order.limit_price else 100.0
        else:
            fill_price = order.limit_price * (1 - slippage) if order.limit_price else 100.0
        
        commission = max(1.0, order.quantity * 0.005)  # $1 min or $0.005/share
        
        fill = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            fill_price=fill_price,
            timestamp=order.timestamp,
            commission=commission
        )
        
        logger.debug(f"Simulated fill: {order.symbol} {order.quantity}@{fill_price:.2f}")
        return fill
    
    async def _paper_fill(self, order: Order) -> Fill:
        """Execute in paper trading"""
        # Similar to simulate but with real-time data
        return await self._simulate_fill(order)
    
    async def _live_fill(self, order: Order) -> Fill:
        """Execute in live trading"""
        # Connect to real broker API
        logger.critical(f"LIVE ORDER: {order.symbol} {order.quantity}")
        raise NotImplementedError("Live trading requires broker integration")

class BacktestEngine:
    """
    Core backtesting engine
    """
    
    def __init__(
        self,
        data_manager: DataManager,
        execution_engine: ExecutionEngine
    ):
        self.data_manager = data_manager
        self.execution = execution_engine
        self.strategies: Dict[str, Strategy] = {}
        logger.info("BacktestEngine initialized")
    
    def register_strategy(self, strategy: Strategy):
        """Register a strategy"""
        self.strategies[strategy.strategy_id] = strategy
        logger.info(f"Registered strategy: {strategy.strategy_id}")
    
    async def run_backtest(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """
        Run backtest for all registered strategies
        """
        logger.info(f"Starting backtest: {start_date} to {end_date}")
        
        # Load data
        data = await self.data_manager.get_historical_data(
            symbols, start_date, end_date
        )
        
        # Initialize portfolio
        portfolio = Portfolio(initial_capital=100000)
        equity_history = []
        
        # Event loop
        for idx, row in data.iterrows():
            timestamp = row['timestamp']
            
            # Create market data event
            market_data = {
                'timestamp': timestamp,
                'symbol': row['symbol'],
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume']
            }
            
            # Process each strategy
            for strategy in self.strategies.values():
                try:
                    # Generate orders
                    orders = await strategy.on_data(market_data)
                    
                    # Execute orders
                    for order in orders:
                        fill = await self.execution.submit_order(order)
                        portfolio.update_fill(fill)
                        await strategy.on_fill(fill)
                        
                except Exception as e:
                    logger.error(f"Strategy {strategy.strategy_id} error: {e}")
            
            # Update portfolio value
            current_prices = {row['symbol']: row['close']}
            equity = portfolio.get_total_equity(current_prices)
            equity_history.append((timestamp, equity))
        
        # Calculate metrics
        results = self._calculate_results(portfolio, equity_history)
        
        logger.info(f"Backtest complete: Sharpe={results['sharpe']:.2f}")
        return results
    
    def _calculate_results(
        self,
        portfolio: 'Portfolio',
        equity_history: List[Tuple]
    ) -> Dict:
        """Calculate backtest metrics"""
        
        df = pd.DataFrame(equity_history, columns=['timestamp', 'equity'])
        df.set_index('timestamp', inplace=True)
        
        returns = df['equity'].pct_change().dropna()
        
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        total_return = (df['equity'].iloc[-1] / portfolio.initial_capital) - 1
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'sharpe': sharpe,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'num_trades': len(portfolio.trade_history),
            'final_equity': df['equity'].iloc[-1]
        }

class Portfolio:
    """Portfolio management"""
    
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Fill] = []
    
    def update_fill(self, fill: Fill):
        """Process fill"""
        if fill.side == OrderSide.BUY:
            self.cash -= (fill.quantity * fill.fill_price + fill.commission)
        else:
            self.cash += (fill.quantity * fill.fill_price - fill.commission)
        
        self.trade_history.append(fill)
    
    def get_total_equity(self, current_prices: Dict[str, float]) -> float:
        """Calculate total equity"""
        positions_value = sum(
            pos.quantity * current_prices.get(pos.symbol, 0)
            for pos in self.positions.values()
        )
        return self.cash + positions_value

# FastAPI Application
app = FastAPI(title="Production Backtesting Engine")

# Global instances
data_manager = DataManager(
    db_url="postgresql://user:pass@localhost/quant_db",
    redis_url="redis://localhost:6379"
)

backtest_engine = BacktestEngine(
    data_manager=data_manager,
    execution_engine=ExecutionEngine(TradingMode.BACKTEST)
)

class BacktestRequest(BaseModel):
    strategy_id: str
    symbols: List[str]
    start_date: str
    end_date: str
    parameters: Dict

@app.post("/backtest")
async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """Submit backtest job"""
    
    backtest_id = f"bt_{int(datetime.now().timestamp())}"
    
    logger.info(f"Received backtest request: {backtest_id}")
    
    # Run in background
    background_tasks.add_task(
        execute_backtest,
        backtest_id,
        request
    )
    
    return {
        "backtest_id": backtest_id,
        "status": "submitted"
    }

async def execute_backtest(backtest_id: str, request: BacktestRequest):
    """Execute backtest asynchronously"""
    try:
        results = await backtest_engine.run_backtest(
            symbols=request.symbols,
            start_date=datetime.fromisoformat(request.start_date),
            end_date=datetime.fromisoformat(request.end_date)
        )
        
        # Store results
        await data_manager.store_backtest_results(backtest_id, results)
        
        logger.info(f"Backtest {backtest_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Backtest {backtest_id} failed: {e}")

@app.get("/backtest/{backtest_id}")
async def get_backtest_results(backtest_id: str):
    """Retrieve backtest results"""
    
    # Query from database
    query = f"SELECT * FROM backtest_results WHERE backtest_id = '{backtest_id}'"
    
    with data_manager.engine.connect() as conn:
        result = conn.execute(query).fetchone()
    
    if result:
        return {
            "backtest_id": result[0],
            "status": "completed",
            "results": json.loads(result[2])
        }
    else:
        return {"backtest_id": backtest_id, "status": "not_found"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "mode": backtest_engine.execution.mode.value
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
\`\`\`

## Deployment

### Docker Deployment

\`\`\`dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
\`\`\`

### Kubernetes Deployment

\`\`\`yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backtest-engine
spec:
  replicas: 3
  selector:
    matchLabels:
      app: backtest-engine
  template:
    metadata:
      labels:
        app: backtest-engine
    spec:
      containers:
      - name: backtest
        image: backtest-engine:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: url
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
\`\`\`

## Production Checklist

- [ ] All components implemented and tested
- [ ] Database schema created
- [ ] Redis configured
- [ ] API documented
- [ ] Logging and monitoring enabled
- [ ] Error handling comprehensive
- [ ] Unit tests (>80% coverage)
- [ ] Integration tests
- [ ] Load testing completed
- [ ] Security review passed
- [ ] Disaster recovery plan documented
- [ ] Deployment pipeline configured
- [ ] Documentation complete

**Congratulations!** You've built a production-grade backtesting engine. This system can handle real trading operations at scale.
`,
    },
  ],
};

export default productionBacktestingEngineProject;
