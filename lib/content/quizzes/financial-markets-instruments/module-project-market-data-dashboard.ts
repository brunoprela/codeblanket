export const moduleProjectMarketDataDashboardQuiz = [
    {
        id: 'fm-1-14-q-1',
        question:
            'Build a complete end-to-end market data dashboard that: (1) Ingests real-time websocket feeds from multiple exchanges, (2) Stores tick data in TimescaleDB, (3) Calculates real-time analytics (VWAP, order book imbalance, spread), (4) Visualizes with React/Next.js, (5) Alerts on unusual patterns. Provide full architecture and code.',
        sampleAnswer: `**Complete Market Data Dashboard Architecture:**

**Tech Stack:**
- **Data Ingestion:** Python with websockets (Polygon.io, Alpaca, IEX)
- **Storage:** TimescaleDB (PostgreSQL extension for time-series)
- **Processing:** Python (pandas, numpy) for analytics
- **API:** FastAPI for real-time data serving
- **Frontend:** Next.js + React + Recharts for visualization
- **Alerts:** Redis pub/sub + Twilio for SMS alerts

**Backend Implementation:**

\`\`\`python
import asyncio
import websockets
import json
from datetime import datetime
import psycopg2
from typing import Dict, List

class MarketDataIngestion:
    """
    Real-time websocket data ingestion
    """
    def __init__(self, db_config):
        self.conn = psycopg2.connect(**db_config)
        self.cursor = self.conn.cursor()
        
    async def connect_to_feed(self, ws_url: str, symbols: List[str]):
        """Connect to websocket feed"""
        async with websockets.connect(ws_url) as ws:
            # Subscribe to symbols
            subscribe_msg = {
                "action": "subscribe",
                "symbols": symbols
            }
            await ws.send(json.dumps(subscribe_msg))
            
            # Process incoming ticks
            async for message in ws:
                tick = json.loads(message)
                self.process_tick(tick)
    
    def process_tick(self, tick: Dict):
        """Store tick and calculate analytics"""
        # Insert tick data
        self.cursor.execute("""
            INSERT INTO ticks (timestamp, symbol, price, size, bid, ask)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            tick['timestamp'],
            tick['symbol'],
            tick['price'],
            tick['size'],
            tick['bid'],
            tick['ask']
        ))
        self.conn.commit()
        
        # Calculate real-time analytics
        analytics = self.calculate_analytics(tick['symbol'])
        
        # Check for alerts
        self.check_alerts(tick['symbol'], analytics)
    
    def calculate_analytics(self, symbol: str):
        """Calculate VWAP, spread, imbalance"""
        # VWAP (last 5 minutes)
        self.cursor.execute("""
            SELECT 
                SUM(price * size) / SUM(size) as vwap,
                AVG(ask - bid) as avg_spread,
                (AVG(bid_size) - AVG(ask_size)) / (AVG(bid_size) + AVG(ask_size)) as imbalance
            FROM ticks
            WHERE symbol = %s 
                AND timestamp > NOW() - INTERVAL '5 minutes'
        """, (symbol,))
        
        result = self.cursor.fetchone()
        
        return {
            'vwap': result[0],
            'avg_spread': result[1],
            'imbalance': result[2]
        }
    
    def check_alerts(self, symbol: str, analytics: Dict):
        """Alert on unusual patterns"""
        # Alert 1: Spread widening >2x normal
        if analytics['avg_spread'] > self.get_normal_spread(symbol) * 2:
            self.send_alert(f"{symbol}: Spread widened to {analytics['avg_spread']:.4f}")
        
        # Alert 2: Extreme imbalance
        if abs(analytics['imbalance']) > 0.5:
            direction = "BUY" if analytics['imbalance'] > 0 else "SELL"
            self.send_alert(f"{symbol}: {direction} pressure (imbalance={analytics['imbalance']:.2f})")

class RealTimeAnalytics:
    """
    Calculate rolling analytics
    """
    def calculate_vwap(self, symbol: str, window_minutes: int = 5):
        """Volume-Weighted Average Price"""
        # Query from TimescaleDB
        query = f"""
            SELECT time_bucket('1 minute', timestamp) as bucket,
                   SUM(price * size) / SUM(size) as vwap
            FROM ticks
            WHERE symbol = '{symbol}'
                AND timestamp > NOW() - INTERVAL '{window_minutes} minutes'
            GROUP BY bucket
            ORDER BY bucket
        """
        # Execute and return
        return self.execute_query(query)
    
    def calculate_order_book_imbalance(self, symbol: str):
        """
        Order book imbalance = (Bid Size - Ask Size) / (Bid Size + Ask Size)
        >0 = buying pressure, <0 = selling pressure
        """
        query = f"""
            SELECT timestamp,
                   (bid_size - ask_size)::float / (bid_size + ask_size) as imbalance
            FROM ticks
            WHERE symbol = '{symbol}'
                AND timestamp > NOW() - INTERVAL '1 hour'
            ORDER BY timestamp
        """
        return self.execute_query(query)
    
    def detect_large_trades(self, symbol: str, threshold_pct: float = 0.05):
        """
        Detect trades >5% of minute volume
        """
        query = f"""
            WITH minute_volumes AS (
                SELECT time_bucket('1 minute', timestamp) as bucket,
                       SUM(size) as minute_volume
                FROM ticks
                WHERE symbol = '{symbol}'
                GROUP BY bucket
            )
            SELECT t.timestamp, t.size, t.price, mv.minute_volume
            FROM ticks t
            JOIN minute_volumes mv ON time_bucket('1 minute', t.timestamp) = mv.bucket
            WHERE t.symbol = '{symbol}'
                AND t.size > mv.minute_volume * {threshold_pct}
            ORDER BY t.timestamp DESC
            LIMIT 100
        """
        return self.execute_query(query)

# FastAPI for serving data
from fastapi import FastAPI, WebSocket
app = FastAPI()

@app.websocket("/ws/ticks/{symbol}")
async def websocket_ticks(websocket: WebSocket, symbol: str):
    """Stream real-time ticks to frontend"""
    await websocket.accept()
    
    # Stream from database
    while True:
        tick = get_latest_tick(symbol)
        await websocket.send_json(tick)
        await asyncio.sleep(0.1)  # 10 ticks/second

@app.get("/api/analytics/{symbol}")
def get_analytics(symbol: str):
    """Get current analytics"""
    analytics = RealTimeAnalytics()
    return {
        'vwap': analytics.calculate_vwap(symbol),
        'imbalance': analytics.calculate_order_book_imbalance(symbol),
        'large_trades': analytics.detect_large_trades(symbol)
    }
\`\`\`

**Frontend (Next.js + React):**

\`\`\`typescript
// components/MarketDataDashboard.tsx
import { useEffect, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip } from 'recharts';

export default function MarketDataDashboard({ symbol }: { symbol: string }) {
  const [ticks, setTicks] = useState([]);
  const [analytics, setAnalytics] = useState(null);
  
  useEffect(() => {
    // WebSocket connection for real-time ticks
    const ws = new WebSocket(\`ws://localhost:8000/ws/ticks/\${symbol}\`);
    
    ws.onmessage = (event) => {
      const tick = JSON.parse(event.data);
      setTicks(prev => [...prev.slice(-100), tick]);  // Keep last 100
    };
    
    // Poll analytics
    const interval = setInterval(async () => {
      const response = await fetch(\`/api/analytics/\${symbol}\`);
      const data = await response.json();
      setAnalytics(data);
    }, 1000);
    
    return () => {
      ws.close();
      clearInterval(interval);
    };
  }, [symbol]);
  
  return (
    <div className="dashboard">
      <h1>{symbol} Market Data</h1>
      
      {/* Real-time price chart */}
      <LineChart width={800} height={400} data={ticks}>
        <XAxis dataKey="timestamp" />
        <YAxis domain={['auto', 'auto']} />
        <Line type="monotone" dataKey="price" stroke="#8884d8" dot={false} />
        <Tooltip />
      </LineChart>
      
      {/* Analytics cards */}
      <div className="analytics-grid">
        <div className="card">
          <h3>VWAP (5min)</h3>
          <p className="value">${analytics?.vwap?.toFixed(2)}</p>
        </div>
        
        <div className="card">
          <h3>Spread</h3>
          <p className="value">{analytics?.avg_spread?.toFixed(4)}</p>
        </div>
        
        <div className="card">
          <h3>Order Book Imbalance</h3>
          <p className={analytics?.imbalance > 0 ? 'positive' : 'negative'}>
            {analytics?.imbalance?.toFixed(2)}
          </p>
          <p className="label">{analytics?.imbalance > 0 ? 'BUY pressure' : 'SELL pressure'}</p>
        </div>
      </div>
      
      {/* Large trades table */}
      <div className="large-trades">
        <h3>Large Trades (>5% minute volume)</h3>
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>Price</th>
              <th>Size</th>
              <th>% of Volume</th>
            </tr>
          </thead>
          <tbody>
            {analytics?.large_trades?.map((trade, i) => (
              <tr key={i}>
                <td>{new Date(trade.timestamp).toLocaleTimeString()}</td>
                <td>${trade.price}</td>
                <td>{trade.size.toLocaleString()}</td>
                <td>{((trade.size / trade.minute_volume) * 100).toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
\`\`\`

**Database Schema (TimescaleDB):**

\`\`\`sql
CREATE TABLE ticks (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    price NUMERIC(10, 2),
    size INTEGER,
    bid NUMERIC(10, 2),
    ask NUMERIC(10, 2),
    bid_size INTEGER,
    ask_size INTEGER
);

-- Create hypertable for time-series optimization
SELECT create_hypertable('ticks', 'timestamp');

-- Indexes for fast queries
CREATE INDEX idx_symbol_time ON ticks (symbol, timestamp DESC);
CREATE INDEX idx_large_trades ON ticks (symbol, size DESC);

-- Continuous aggregates for pre-computed analytics
CREATE MATERIALIZED VIEW vwap_1min
WITH (timescaledb.continuous) AS
SELECT time_bucket('1 minute', timestamp) AS bucket,
       symbol,
       SUM(price * size) / SUM(size) as vwap,
       SUM(size) as volume
FROM ticks
GROUP BY bucket, symbol;
\`\`\`

**Key Features:**
1. **Real-time ingestion** from multiple websocket sources
2. **TimescaleDB** for efficient time-series storage (100K+ ticks/second)
3. **Continuous aggregates** for pre-computed VWAP, spreads
4. **FastAPI** websocket streaming to frontend
5. **React dashboard** with real-time charts
6. **Alert system** for unusual patterns

**Production Enhancements:**
- Add Redis for caching hot data
- Implement circuit breakers for failed feeds
- Add authentication/authorization
- Monitoring with Prometheus/Grafana
- Load balancing for multiple concurrent users`,
        keyPoints: [
            'Architecture: Websockets (ingest) → TimescaleDB (store) → FastAPI (serve) → React (visualize)',
            'Analytics: Real-time VWAP, order book imbalance, spread monitoring, large trade detection',
            'Alerts: Spread widening >2× normal, extreme imbalance >0.5, large trades >5% volume',
            'Performance: TimescaleDB handles 100K+ ticks/second, continuous aggregates for speed',
            'Production: Add Redis cache, monitoring, auth, circuit breakers, load balancing',
        ],
    },
    {
        id: 'fm-1-14-q-2',
        question:
            'Extend the dashboard to include: (1) Multi-symbol watchlist with real-time P&L tracking, (2) Backtesting engine for simple strategies, (3) Paper trading simulation, (4) Risk management with position limits and stop-losses. Show how to integrate all Module 1 concepts.',
        sampleAnswer: `[Comprehensive extension showing portfolio tracking, strategy backtesting framework, simulated order execution, risk management system, and integration of all module concepts]`,
        keyPoints: [
            'Watchlist: Real-time P&L for 10-20 symbols, aggregate portfolio metrics',
            'Backtesting: Historical data replay, strategy signal generation, execution simulation',
            'Paper trading: Simulate orders against real-time market, track fills, calculate slippage',
            'Risk management: Position limits (max 10% per symbol), portfolio VaR, stop-losses',
            'Integration: Uses order types, market data, liquidity analysis, execution algos from module',
        ],
    },
    {
        id: 'fm-1-14-q-3',
        question:
            'Document best practices for production deployment: database optimization, API rate limiting, websocket scaling, error handling, monitoring, and disaster recovery. How would you handle exchange downtime or data feed failures?',
        sampleAnswer: `[Production deployment guide covering database indexing, connection pooling, rate limit middleware, websocket heartbeats, comprehensive error handling, Prometheus monitoring, backup strategies, and failover procedures]`,
        keyPoints: [
            'Database: TimescaleDB hypertables, compression, retention policies, continuous aggregates',
            'Scaling: Redis for caching, multiple websocket instances, load balancer',
            'Monitoring: Prometheus metrics, Grafana dashboards, PagerDuty alerts',
            'Error handling: Circuit breakers for feeds, retry logic with exponential backoff',
            'Disaster recovery: Hot standby database, automated failover, data replication',
        ],
    },
];
