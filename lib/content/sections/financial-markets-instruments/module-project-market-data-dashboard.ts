export const moduleProjectMarketDataDashboard = {
  title: 'Module Project: Market Data Dashboard',
  slug: 'module-project-market-data-dashboard',
  description: 'Build a production-ready real-time market data dashboard',
  content: `
# Module Project: Market Data Dashboard

## Project Overview

Build a comprehensive market data dashboard that brings together everything you've learned:
- 📊 **Real-time quotes** from multiple exchanges
- 📈 **Order book visualization** (Level 2 data)
- 💹 **Trade feed** with directional analysis
- 🎯 **Market impact calculator**
- 📉 **Technical indicators** and charts
- ⚡ **Performance**: <100ms latency end-to-end

**Skills you'll practice:**
- Market data APIs (Polygon, Alpha Vantage, IEX)
- WebSocket real-time streaming
- Order book management
- Data normalization
- React/Next.js UI
- Python backend (Fast API)
- PostgreSQL/TimescaleDB for historical data

**What you'll build:**1. Backend: FastAPI market data server
2. Frontend: React dashboard with real-time updates
3. Database: TimescaleDB for time-series storage
4. Analytics: Market impact, liquidity metrics
5. Visualization: Charts, order book depth
6. Testing: Unit tests, integration tests

---

## Part 1: Backend - Market Data Server

\`\`\`python
# backend/market_data_server.py

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import aiohttp
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
from pydantic import BaseModel
import logging

# Configure logging
logging.basicConfig (level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Market Data Dashboard API")

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class Quote(BaseModel):
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    last: float
    last_size: int
    
    def spread_bps (self) -> float:
        mid = (self.bid + self.ask) / 2
        return ((self.ask - self.bid) / mid) * 10000

class OrderBookLevel(BaseModel):
    price: float
    size: int
    num_orders: int

class OrderBook(BaseModel):
    symbol: str
    timestamp: datetime
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    
    def imbalance (self, levels: int = 5) -> float:
        """Calculate order book imbalance"""
        bid_vol = sum (level.size for level in self.bids[:levels])
        ask_vol = sum (level.size for level in self.asks[:levels])
        total = bid_vol + ask_vol
        return (bid_vol - ask_vol) / total if total > 0 else 0

class Trade(BaseModel):
    symbol: str
    timestamp: datetime
    price: float
    size: int
    side: str  # 'BUY' or 'SELL'

# Market Data Provider
class MarketDataProvider:
    """
    Connect to market data APIs
    
    Supports: Polygon, Alpha Vantage, IEX Cloud
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()
    
    async def get_quote (self, symbol: str) -> Quote:
        """
        Fetch real-time quote
        
        Free tier: 5 API calls per minute (Polygon)
        Paid tier: Real-time WebSocket
        """
        # Example: Polygon API
        url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}"
        params = {"apiKey": self.api_key}
        
        async with self.session.get (url, params=params) as response:
            data = await response.json()
            
            if response.status != 200:
                raise Exception (f"API error: {data}")
            
            ticker = data['ticker']
            
            return Quote(
                symbol=symbol,
                timestamp=datetime.now(),
                bid=ticker['day']['c'],  # Simplified: use close as bid
                ask=ticker['day']['c'] + 0.01,  # Mock ask
                bid_size=100,
                ask_size=100,
                last=ticker['day']['c'],
                last_size=ticker['day']['v']
            )
    
    async def get_order_book (self, symbol: str) -> OrderBook:
        """
        Fetch Level 2 order book
        
        Note: Full order book requires premium API access
        This simulates order book for demonstration
        """
        quote = await self.get_quote (symbol)
        
        # Simulate order book levels
        bids = [
            OrderBookLevel (price=quote.bid - i * 0.01, size=np.random.randint(100, 1000), num_orders=np.random.randint(1, 10))
            for i in range(10)
        ]
        
        asks = [
            OrderBookLevel (price=quote.ask + i * 0.01, size=np.random.randint(100, 1000), num_orders=np.random.randint(1, 10))
            for i in range(10)
        ]
        
        return OrderBook(
            symbol=symbol,
            timestamp=datetime.now(),
            bids=bids,
            asks=asks
        )
    
    async def stream_trades (self, symbol: str, callback):
        """
        Stream real-time trades
        
        Production: WebSocket to exchange
        Demo: Poll API
        """
        while True:
            try:
                quote = await self.get_quote (symbol)
                
                # Simulate trade
                trade = Trade(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    price=quote.last,
                    size=quote.last_size,
                    side='BUY' if np.random.random() > 0.5 else 'SELL'
                )
                
                await callback (trade)
                
                # Wait before next update
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error (f"Error streaming trades: {e}")
                await asyncio.sleep(5)

# Market Impact Calculator
class MarketImpactCalculator:
    """Calculate estimated market impact"""
    
    @staticmethod
    def estimate_impact(
        order_size: int,
        adv: int,
        volatility: float
    ) -> Dict:
        """
        Square-root law implementation
        """
        participation = order_size / adv
        gamma = 0.8  # Market impact coefficient
        
        # Temporary impact
        temp_impact_bps = gamma * volatility * np.sqrt (participation) * 10000
        
        # Permanent impact (40% of temporary)
        perm_impact_bps = 0.4 * temp_impact_bps
        
        total_impact_bps = temp_impact_bps + perm_impact_bps
        
        # Estimate total cost
        avg_price = 100  # Assume $100 stock
        impact_per_share = (total_impact_bps / 10000) * avg_price
        total_cost = impact_per_share * order_size
        
        return {
            'order_size': order_size,
            'adv': adv,
            'participation_rate_pct': participation * 100,
            'temp_impact_bps': temp_impact_bps,
            'perm_impact_bps': perm_impact_bps,
            'total_impact_bps': total_impact_bps,
            'total_cost_usd': total_cost
        }

# WebSocket Manager
class ConnectionManager:
    """Manage WebSocket connections"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect (self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append (websocket)
        logger.info (f"Client connected. Total: {len (self.active_connections)}")
    
    def disconnect (self, websocket: WebSocket):
        self.active_connections.remove (websocket)
        logger.info (f"Client disconnected. Total: {len (self.active_connections)}")
    
    async def broadcast (self, message: dict):
        """Send to all connected clients"""
        for connection in self.active_connections:
            try:
                await connection.send_json (message)
            except Exception as e:
                logger.error (f"Error broadcasting: {e}")

manager = ConnectionManager()

# API Endpoints

@app.get("/")
async def root():
    return {"message": "Market Data Dashboard API", "version": "1.0"}

@app.get("/api/quote/{symbol}")
async def get_quote (symbol: str):
    """Get current quote for symbol"""
    try:
        async with MarketDataProvider (api_key="YOUR_API_KEY") as provider:
            quote = await provider.get_quote (symbol.upper())
            return quote.dict()
    except Exception as e:
        return {"error": str (e)}, 500

@app.get("/api/orderbook/{symbol}")
async def get_order_book (symbol: str):
    """Get Level 2 order book"""
    try:
        async with MarketDataProvider (api_key="YOUR_API_KEY") as provider:
            book = await provider.get_order_book (symbol.upper())
            return book.dict()
    except Exception as e:
        return {"error": str (e)}, 500

@app.post("/api/impact")
async def calculate_impact(
    order_size: int,
    adv: int,
    volatility: float
):
    """Calculate market impact for order"""
    calculator = MarketImpactCalculator()
    impact = calculator.estimate_impact (order_size, adv, volatility)
    return impact

@app.websocket("/ws/quotes/{symbol}")
async def websocket_quotes (websocket: WebSocket, symbol: str):
    """
    WebSocket endpoint for real-time quotes
    
    Streams quotes every second
    """
    await manager.connect (websocket)
    
    try:
        async with MarketDataProvider (api_key="YOUR_API_KEY") as provider:
            while True:
                # Fetch latest quote
                quote = await provider.get_quote (symbol.upper())
                
                # Send to client
                await websocket.send_json({
                    "type": "quote",
                    "data": quote.dict()
                })
                
                # Wait before next update
                await asyncio.sleep(1)
                
    except WebSocketDisconnect:
        manager.disconnect (websocket)
    except Exception as e:
        logger.error (f"WebSocket error: {e}")
        manager.disconnect (websocket)

@app.websocket("/ws/trades/{symbol}")
async def websocket_trades (websocket: WebSocket, symbol: str):
    """Stream real-time trades"""
    await manager.connect (websocket)
    
    try:
        async with MarketDataProvider (api_key="YOUR_API_KEY") as provider:
            
            async def send_trade (trade: Trade):
                await websocket.send_json({
                    "type": "trade",
                    "data": trade.dict()
                })
            
            # Start streaming
            await provider.stream_trades (symbol.upper(), send_trade)
            
    except WebSocketDisconnect:
        manager.disconnect (websocket)
    except Exception as e:
        logger.error (f"WebSocket error: {e}")
        manager.disconnect (websocket)

# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run (app, host="0.0.0.0", port=8000)
\`\`\`

---

## Part 2: Frontend - React Dashboard

\`\`\`typescript
// frontend/components/MarketDataDashboard.tsx

import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface Quote {
  symbol: string;
  timestamp: string;
  bid: number;
  ask: number;
  bid_size: number;
  ask_size: number;
  last: number;
  last_size: number;
}

interface OrderBookLevel {
  price: number;
  size: number;
  num_orders: number;
}

interface OrderBook {
  symbol: string;
  timestamp: string;
  bids: OrderBookLevel[];
  asks: OrderBookLevel[];
}

interface Trade {
  symbol: string;
  timestamp: string;
  price: number;
  size: number;
  side: 'BUY' | 'SELL';
}

export const MarketDataDashboard: React.FC<{ symbol: string }> = ({ symbol }) => {
  const [quote, setQuote] = useState<Quote | null>(null);
  const [orderBook, setOrderBook] = useState<OrderBook | null>(null);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [priceHistory, setPriceHistory] = useState<{ time: string; price: number }[]>([]);
  
  // WebSocket connection for real-time quotes
  useEffect(() => {
    const ws = new WebSocket(\`ws://localhost:8000/ws/quotes/\${symbol}\`);
    
    ws.onmessage = (event) => {
      const message = JSON.parse (event.data);
      
      if (message.type === 'quote') {
        setQuote (message.data);
        
        // Update price history
        setPriceHistory (prev => {
          const newPoint = {
            time: new Date (message.data.timestamp).toLocaleTimeString(),
            price: message.data.last
          };
          return [...prev.slice(-50), newPoint];  // Keep last 50 points
        });
      }
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
    
    return () => {
      ws.close();
    };
  }, [symbol]);
  
  // WebSocket connection for trades
  useEffect(() => {
    const ws = new WebSocket(\`ws://localhost:8000/ws/trades/\${symbol}\`);
    
    ws.onmessage = (event) => {
      const message = JSON.parse (event.data);
      
      if (message.type === 'trade') {
        setTrades (prev => [message.data, ...prev.slice(0, 19)]);  // Keep last 20
      }
    };
    
    return () => {
      ws.close();
    };
  }, [symbol]);
  
  // Fetch order book
  const fetchOrderBook = useCallback (async () => {
    try {
      const response = await fetch(\`http://localhost:8000/api/orderbook/\${symbol}\`);
      const data = await response.json();
      setOrderBook (data);
    } catch (error) {
      console.error('Error fetching order book:', error);
    }
  }, [symbol]);
  
  useEffect(() => {
    fetchOrderBook();
    const interval = setInterval (fetchOrderBook, 1000);  // Update every second
    return () => clearInterval (interval);
  }, [fetchOrderBook]);
  
  return (
    <div className="p-4 space-y-4">
      <h1 className="text-3xl font-bold">{symbol} - Market Data Dashboard</h1>
      
      {/* Quote Display */}
      <Card>
        <CardHeader>
          <CardTitle>Real-Time Quote</CardTitle>
        </CardHeader>
        <CardContent>
          {quote ? (
            <div className="grid grid-cols-4 gap-4">
              <div>
                <div className="text-sm text-gray-500">Bid</div>
                <div className="text-2xl font-bold">\${quote.bid.toFixed(2)}</div>
                <div className="text-sm text-gray-500">{quote.bid_size} shares</div>
              </div>
              <div>
                <div className="text-sm text-gray-500">Ask</div>
                <div className="text-2xl font-bold">\${quote.ask.toFixed(2)}</div>
                <div className="text-sm text-gray-500">{quote.ask_size} shares</div>
              </div>
              <div>
                <div className="text-sm text-gray-500">Last</div>
                <div className="text-2xl font-bold">\${quote.last.toFixed(2)}</div>
                <div className="text-sm text-gray-500">{quote.last_size} shares</div>
              </div>
              <div>
                <div className="text-sm text-gray-500">Spread</div>
                <div className="text-2xl font-bold">
                  {((quote.ask - quote.bid) / ((quote.ask + quote.bid) / 2) * 10000).toFixed(1)} bps
                </div>
              </div>
            </div>
          ) : (
            <div>Loading...</div>
          )}
        </CardContent>
      </Card>
      
      {/* Price Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Price History</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={priceHistory}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis domain={['auto', 'auto']} />
              <Tooltip />
              <Line type="monotone" dataKey="price" stroke="#8884d8" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
      
      {/* Order Book */}
      <Card>
        <CardHeader>
          <CardTitle>Order Book (Level 2)</CardTitle>
        </CardHeader>
        <CardContent>
          {orderBook ? (
            <div className="grid grid-cols-2 gap-4">
              {/* Bids */}
              <div>
                <h3 className="font-bold mb-2 text-green-600">Bids</h3>
                <table className="w-full text-sm">
                  <thead>
                    <tr>
                      <th className="text-left">Price</th>
                      <th className="text-right">Size</th>
                      <th className="text-right">Orders</th>
                    </tr>
                  </thead>
                  <tbody>
                    {orderBook.bids.slice(0, 10).map((level, i) => (
                      <tr key={i} className="border-t">
                        <td>\${level.price.toFixed(2)}</td>
                        <td className="text-right">{level.size}</td>
                        <td className="text-right">{level.num_orders}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              
              {/* Asks */}
              <div>
                <h3 className="font-bold mb-2 text-red-600">Asks</h3>
                <table className="w-full text-sm">
                  <thead>
                    <tr>
                      <th className="text-left">Price</th>
                      <th className="text-right">Size</th>
                      <th className="text-right">Orders</th>
                    </tr>
                  </thead>
                  <tbody>
                    {orderBook.asks.slice(0, 10).map((level, i) => (
                      <tr key={i} className="border-t">
                        <td>\${level.price.toFixed(2)}</td>
                        <td className="text-right">{level.size}</td>
                        <td className="text-right">{level.num_orders}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <div>Loading...</div>
          )}
        </CardContent>
      </Card>
      
      {/* Trade Feed */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Trades</CardTitle>
        </CardHeader>
        <CardContent>
          <table className="w-full text-sm">
            <thead>
              <tr>
                <th className="text-left">Time</th>
                <th className="text-left">Side</th>
                <th className="text-right">Price</th>
                <th className="text-right">Size</th>
              </tr>
            </thead>
            <tbody>
              {trades.map((trade, i) => (
                <tr key={i} className="border-t">
                  <td>{new Date (trade.timestamp).toLocaleTimeString()}</td>
                  <td>
                    <span className={trade.side === 'BUY' ? 'text-green-600' : 'text-red-600'}>
                      {trade.side}
                    </span>
                  </td>
                  <td className="text-right">\${trade.price.toFixed(2)}</td>
                  <td className="text-right">{trade.size}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </CardContent>
      </Card>
    </div>
  );
};
\`\`\`

---

## Part 3: Testing & Deployment

\`\`\`python
# tests/test_market_data_server.py

import pytest
from fastapi.testclient import TestClient
from backend.market_data_server import app, MarketImpactCalculator

client = TestClient (app)

def test_root():
    """Test API root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_market_impact_calculator():
    """Test market impact calculation"""
    calculator = MarketImpactCalculator()
    
    impact = calculator.estimate_impact(
        order_size=10_000,
        adv=1_000_000,
        volatility=0.30
    )
    
    assert impact['order_size'] == 10_000
    assert impact['participation_rate_pct'] == 1.0
    assert impact['total_impact_bps'] > 0
    assert impact['total_cost_usd'] > 0

def test_impact_api_endpoint():
    """Test market impact API"""
    response = client.post(
        "/api/impact",
        params={
            "order_size": 10000,
            "adv": 1000000,
            "volatility": 0.30
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert 'total_impact_bps' in data
    assert data['participation_rate_pct'] == 1.0

@pytest.mark.asyncio
async def test_websocket_connection():
    """Test WebSocket connection"""
    with client.websocket_connect("/ws/quotes/AAPL") as websocket:
        data = websocket.receive_json()
        assert data['type'] == 'quote'
        assert 'data' in data
\`\`\`

---

## Summary & Next Steps

**What You've Built:**
- ✅ FastAPI backend with WebSocket support
- ✅ React frontend with real-time updates
- ✅ Order book visualization
- ✅ Market impact calculator
- ✅ Trade feed monitoring
- ✅ Test coverage

**Production Enhancements:**1. **Add authentication** (JWT tokens)
2. **Rate limiting** (Redis)
3. **Historical data** (TimescaleDB)
4. **Advanced charts** (TradingView library)
5. **Alert system** (price alerts, unusual volume)
6. **Portfolio tracking** (P&L, positions)

**Deployment:**
\`\`\`bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn market_data_server:app --reload

# Frontend
cd frontend
npm install
npm run dev

# Access dashboard
http://localhost:3000
\`\`\`

**Congratulations!** You've completed Module 1 and built a production-ready market data dashboard!

**Next Module:** Module 2 - Professional Tools (Bloomberg, Python libraries, APIs)
`,
  exercises: [
    {
      prompt:
        'Enhance the dashboard with technical indicators (Moving Average, RSI, MACD, Bollinger Bands) calculated in real-time from the price history. Add indicator charts and buy/sell signals when indicators cross thresholds.',
      solution:
        '// Implementation: 1) Calculate MA: rolling average of last 20/50 prices, 2) RSI: (100 - 100/(1 + RS)), RS = avg_gain/avg_loss over 14 periods, 3) MACD: EMA(12) - EMA(26), signal = EMA(9) of MACD, 4) Bollinger: MA ± 2×std_dev, 5) Display on separate chart panels below price, 6) Signals: RSI<30 = oversold (buy), RSI>70 = overbought (sell), MACD cross = trend change, 7) Store calculations in React state, update on each new price',
    },
    {
      prompt:
        'Add a position tracker that allows entering trades (buy/sell), calculates unrealized P&L based on current market price, shows aggregate position, and tracks realized P&L from closed trades. Include position sizing based on market impact estimates.',
      solution:
        '// Implementation: 1) Position model: {symbol, quantity, avg_cost, current_price, unrealized_pnl}, 2) Trade entry form: symbol, side, quantity, price, 3) On trade: update position (qty += buy qty, -= sell qty), recalc avg_cost, 4) Unrealized P&L = (current_price - avg_cost) × quantity, 5) Realized P&L on close: (sell_price - cost_basis) × qty, 6) Market impact warning: if order_size > 5% ADV, show estimated slippage, 7) Display: current positions table, P&L chart, trade history, 8) Persist to backend API + database',
    },
  ],
};
