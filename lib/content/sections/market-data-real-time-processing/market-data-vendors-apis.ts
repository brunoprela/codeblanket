export const marketDataVendorsApis = {
  title: 'Market Data Vendors and APIs',
  id: 'market-data-vendors-apis',
  content: `
# Market Data Vendors and APIs

## Introduction

Market data vendors are the bridge between exchanges and traders. Understanding vendor landscape, pricing models, API capabilities, and integration patterns is essential for building production trading systems.

**Major Market Data Vendors:**
- **Bloomberg Terminal**: Industry standard, $24K/year/user
- **Refinitiv (Thomson Reuters)**: Eikon platform, enterprise data
- **IEX Cloud**: Modern API, affordable ($0-9K/year)
- **Polygon.io**: Real-time + historical, $200-1K/month
- **Alpha Vantage**: Free tier, API-first
- **Interactive Brokers**: Free with brokerage account
- **Nasdaq Data Link (Quandl)**: Alternative data focus

**Why Vendors Matter:**
- **Exchanges don't sell directly to retail** - need intermediaries
- **Data consolidation** - aggregate all US exchanges (NASDAQ, NYSE, etc.)
- **Historical data** - vendors maintain tick archives going back years
- **Normalization** - vendors standardize data formats across exchanges
- **Support & SLAs** - production systems need reliability guarantees

This section covers vendor selection, API integration, cost optimization, and best practices.

---

## Vendor Comparison

### Tier 1: Professional/Enterprise

**Bloomberg Terminal** ($24K/year/user)
- **Pros**: Industry standard, every dataset imaginable, excellent support, real-time news
- **Cons**: Extremely expensive, closed ecosystem, limited API access
- **Use Case**: Buy-side firms, research analysts, portfolio managers
- **API**: Bloomberg API (C++/Python), B-PIPE for real-time data

**Refinitiv Eikon** ($15-20K/year/user)
- **Pros**: Comprehensive data, strong in fixed income/FX, DataStream historical
- **Cons**: Expensive, clunky interface vs Bloomberg
- **Use Case**: Banks, institutional asset managers
- **API**: Eikon Data API (Python), Refinitiv Real-Time (formerly Reuters)

### Tier 2: Professional/Developer-Friendly

**IEX Cloud** ($0-9K/year, usage-based)
- **Pros**: Modern REST API, affordable, generous free tier, excellent docs
- **Cons**: US equities only, limited historical depth (5 years)
- **Use Case**: Algorithmic traders, fintech startups, retail platforms
- **API**: REST + WebSocket, Python/Node.js SDKs

**Polygon.io** ($200-1K/month)
- **Pros**: Real-time + historical, stocks/options/forex/crypto, unlimited API calls
- **Cons**: Some data delays on lower tiers, less institutional support
- **Use Case**: Quant trading, backtesting, application development
- **API**: REST + WebSocket, comprehensive Python SDK

### Tier 3: Free/Retail

**Alpha Vantage** (Free + Premium $50-500/month)
- **Pros**: Free tier available, good for learning/prototyping
- **Cons**: Rate limits (5 calls/minute free), 15-20 min delayed on free tier
- **Use Case**: Students, hobbyists, small projects
- **API**: Simple REST API, JSON responses

**Yahoo Finance** (Free, unofficial)
- **Pros**: Completely free, no API key needed
- **Cons**: No official API (use yfinance library), can break anytime, delayed data
- **Use Case**: Personal projects, education (not production)
- **API**: Unofficial via yfinance Python library

---

## API Integration Patterns

### REST API Pattern

\`\`\`python
import requests
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict
import time

class MarketDataAPI:
    """Generic market data API client"""
    
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
        
        # Rate limiting
        self.requests_per_minute = 60
        self.request_times = []
    
    def _rate_limit(self):
        """Enforce rate limits"""
        now = time.time()
        
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times 
                             if now - t < 60]
        
        # Check if we're at limit
        if len(self.request_times) >= self.requests_per_minute:
            sleep_time = 60 - (now - self.request_times[0])
            if sleep_time > 0:
                print(f"Rate limit reached, sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
        
        self.request_times.append(now)
    
    def get_quote(self, symbol: str) -> Dict:
        """Get real-time quote"""
        self._rate_limit()
        
        url = f"{self.base_url}/quote/{symbol}"
        response = self.session.get(url)
        response.raise_for_status()
        
        data = response.json()
        
        return {
            'symbol': data['symbol'],
            'bid': Decimal(str(data['bidPrice'])),
            'ask': Decimal(str(data['askPrice'])),
            'last': Decimal(str(data['lastPrice'])),
            'volume': data['volume'],
            'timestamp': datetime.fromtimestamp(data['timestamp'] / 1000)
        }
    
    def get_historical_bars(self, symbol: str, start_date: datetime, 
                           end_date: datetime, interval: str = '1min') -> List[Dict]:
        """Get historical OHLCV bars"""
        self._rate_limit()
        
        url = f"{self.base_url}/bars/{symbol}"
        params = {
            'start': start_date.isoformat(),
            'end': end_date.isoformat(),
            'interval': interval,
            'limit': 10000  # Max per request
        }
        
        response = self.session.get(url, params=params)
        response.raise_for_status()
        
        bars = []
        for bar_data in response.json()['bars']:
            bars.append({
                'timestamp': datetime.fromtimestamp(bar_data['t'] / 1000),
                'open': Decimal(str(bar_data['o'])),
                'high': Decimal(str(bar_data['h'])),
                'low': Decimal(str(bar_data['l'])),
                'close': Decimal(str(bar_data['c'])),
                'volume': bar_data['v']
            })
        
        return bars
    
    def get_bulk_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get quotes for multiple symbols efficiently"""
        self._rate_limit()
        
        # Batch request (if API supports)
        url = f"{self.base_url}/quotes"
        params = {'symbols': ','.join(symbols)}
        
        response = self.session.get(url, params=params)
        response.raise_for_status()
        
        quotes = {}
        for symbol, data in response.json().items():
            quotes[symbol] = {
                'bid': Decimal(str(data['bidPrice'])),
                'ask': Decimal(str(data['askPrice'])),
                'last': Decimal(str(data['lastPrice'])),
                'timestamp': datetime.fromtimestamp(data['timestamp'] / 1000)
            }
        
        return quotes

# Usage examples

# IEX Cloud
iex_api = MarketDataAPI(
    api_key="pk_xxx...",
    base_url="https://cloud.iexapis.com/stable"
)

# Get real-time quote
quote = iex_api.get_quote("AAPL")
print(f"AAPL: Bid \${quote['bid']}, Ask \\$\{quote['ask']}")

# Get historical data
start = datetime(2024, 1, 1)
end = datetime(2024, 1, 31)
bars = iex_api.get_historical_bars("AAPL", start, end, interval='1hour')
print(f"Retrieved {len(bars)} hourly bars")

# Bulk quotes for portfolio
symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
quotes = iex_api.get_bulk_quotes(symbols)
for symbol, quote in quotes.items():
    print(f"{symbol}: \\\$\{quote['last']}")
\`\`\`

### WebSocket Real-Time Pattern

\`\`\`python
import asyncio
import websockets
import json
from typing import Callable, List
from datetime import datetime

class RealtimeDataFeed:
    """WebSocket-based real-time market data"""
    
    def __init__(self, api_key: str, websocket_url: str):
        self.api_key = api_key
        self.websocket_url = websocket_url
        self.ws = None
        self.subscriptions = set()
        self.callbacks = []
        
        # Connection state
        self.connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
    
    def register_callback(self, callback: Callable):
        """Register callback for market data updates"""
        self.callbacks.append(callback)
    
    async def connect(self):
        """Establish WebSocket connection"""
        try:
            self.ws = await websockets.connect(
                self.websocket_url,
                extra_headers={'Authorization': f'Bearer {self.api_key}'}
            )
            self.connected = True
            self.reconnect_attempts = 0
            print("WebSocket connected")
            
            # Authenticate (if required)
            auth_msg = {
                'action': 'auth',
                'key': self.api_key
            }
            await self.ws.send(json.dumps(auth_msg))
            
            # Resubscribe to previous subscriptions
            if self.subscriptions:
                await self._resubscribe()
            
        except Exception as e:
            print(f"Connection error: {e}")
            await self._handle_reconnect()
    
    async def subscribe(self, symbols: List[str], channels: List[str] = None):
        """Subscribe to symbols"""
        channels = channels or ['quotes', 'trades']
        
        for symbol in symbols:
            for channel in channels:
                key = f"{channel}:{symbol}"
                self.subscriptions.add(key)
        
        # Send subscription message
        subscribe_msg = {
            'action': 'subscribe',
            'symbols': symbols,
            'channels': channels
        }
        
        if self.ws and self.connected:
            await self.ws.send(json.dumps(subscribe_msg))
            print(f"Subscribed to {len(symbols)} symbols on {channels}")
    
    async def _resubscribe(self):
        """Resubscribe after reconnection"""
        symbols_by_channel = {}
        for sub in self.subscriptions:
            channel, symbol = sub.split(':')
            if channel not in symbols_by_channel:
                symbols_by_channel[channel] = []
            symbols_by_channel[channel].append(symbol)
        
        for channel, symbols in symbols_by_channel.items():
            await self.subscribe(symbols, [channel])
    
    async def listen(self):
        """Listen for incoming messages"""
        try:
            async for message in self.ws:
                data = json.loads(message)
                
                # Handle different message types
                if data.get('type') == 'quote':
                    await self._handle_quote(data)
                elif data.get('type') == 'trade':
                    await self._handle_trade(data)
                elif data.get('type') == 'error':
                    print(f"Error: {data.get('message')}")
                
        except websockets.ConnectionClosed:
            print("WebSocket connection closed")
            self.connected = False
            await self._handle_reconnect()
        except Exception as e:
            print(f"Listen error: {e}")
            await self._handle_reconnect()
    
    async def _handle_quote(self, data: dict):
        """Process quote update"""
        quote = {
            'type': 'quote',
            'symbol': data['symbol'],
            'bid_price': Decimal(str(data['bidPrice'])),
            'bid_size': data['bidSize'],
            'ask_price': Decimal(str(data['askPrice'])),
            'ask_size': data['askSize'],
            'timestamp': datetime.fromtimestamp(data['timestamp'] / 1000)
        }
        
        # Execute callbacks
        for callback in self.callbacks:
            await callback(quote)
    
    async def _handle_trade(self, data: dict):
        """Process trade update"""
        trade = {
            'type': 'trade',
            'symbol': data['symbol'],
            'price': Decimal(str(data['price'])),
            'size': data['size'],
            'timestamp': datetime.fromtimestamp(data['timestamp'] / 1000)
        }
        
        for callback in self.callbacks:
            await callback(trade)
    
    async def _handle_reconnect(self):
        """Handle reconnection with exponential backoff"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            print(f"Max reconnect attempts reached ({self.max_reconnect_attempts})")
            return
        
        self.reconnect_attempts += 1
        wait_time = min(2 ** self.reconnect_attempts, 60)  # Cap at 60s
        print(f"Reconnecting in {wait_time}s (attempt {self.reconnect_attempts})...")
        
        await asyncio.sleep(wait_time)
        await self.connect()
    
    async def disconnect(self):
        """Close WebSocket connection"""
        if self.ws:
            await self.ws.close()
            self.connected = False
            print("WebSocket disconnected")

# Usage
async def market_data_handler(data: dict):
    """Handle incoming market data"""
    if data['type'] == 'quote':
        print(f"{data['symbol']}: Bid \\\$\{data['bid_price']} × {data['bid_size']}, "
              f"Ask \${data['ask_price']} × {data['ask_size']}")
    elif data['type'] == 'trade':
        print(f"{data['symbol']}: Trade \\\$\{data['price']} × {data['size']}")

async def main():
    # Initialize feed
    feed = RealtimeDataFeed(
        api_key="YOUR_API_KEY",
        websocket_url="wss://api.example.com/v1/quotes"
    )
    
    # Register callback
    feed.register_callback(market_data_handler)
    
    # Connect
    await feed.connect()
    
    # Subscribe to symbols
    await feed.subscribe(['AAPL', 'GOOGL', 'MSFT'], channels=['quotes', 'trades'])
    
    # Listen for updates
    await feed.listen()

# Run
# asyncio.run(main())
\`\`\`

---

## Vendor-Specific Implementations

### Polygon.io

\`\`\`python
from polygon import RESTClient, WebSocketClient
from polygon.websocket.models import WebSocketMessage, EquityTrade, EquityQuote

class PolygonDataFeed:
    """Polygon.io integration"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.rest_client = RESTClient(api_key)
    
    def get_daily_bars(self, symbol: str, from_date: str, to_date: str):
        """Get daily OHLCV bars"""
        aggs = self.rest_client.get_aggs(
            ticker=symbol,
            multiplier=1,
            timespan="day",
            from_=from_date,
            to=to_date
        )
        
        bars = []
        for agg in aggs:
            bars.append({
                'timestamp': datetime.fromtimestamp(agg.timestamp / 1000),
                'open': Decimal(str(agg.open)),
                'high': Decimal(str(agg.high)),
                'low': Decimal(str(agg.low)),
                'close': Decimal(str(agg.close)),
                'volume': agg.volume
            })
        
        return bars
    
    def stream_realtime(self, symbols: List[str]):
        """Stream real-time data via WebSocket"""
        ws = WebSocketClient(
            api_key=self.api_key,
            feed="delayed.polygon.io",  # or "wss.polygon.io" for real-time
            market="stocks",
            on_message=self._handle_message,
            on_error=self._handle_error
        )
        
        # Subscribe
        ws.subscribe("Q.*", "T.*")  # Quotes and Trades for all symbols
        ws.run()
    
    def _handle_message(self, msgs: List[WebSocketMessage]):
        """Handle WebSocket messages"""
        for msg in msgs:
            if isinstance(msg, EquityQuote):
                print(f"Quote: {msg.symbol} \${msg.bid_price} x \\$\{msg.ask_price}")
            elif isinstance(msg, EquityTrade):
                print(f"Trade: {msg.symbol} \\\$\{msg.price} x {msg.size}")
    
    def _handle_error(self, error):
        print(f"WebSocket error: {error}")

# Usage
polygon = PolygonDataFeed("YOUR_API_KEY")
bars = polygon.get_daily_bars("AAPL", "2024-01-01", "2024-01-31")
\`\`\`

---

## Cost Optimization Strategies

1. **Cache aggressively**: Store historical bars locally (don't re-download)
2. **Batch requests**: Use bulk endpoints instead of individual calls
3. **Tiered access**: Use free APIs for development, paid for production
4. **Rate limit management**: Implement client-side rate limiting
5. **Data compression**: Enable gzip encoding for REST APIs
6. **Selective subscriptions**: Only subscribe to symbols you're trading
7. **Off-peak downloads**: Fetch historical data outside market hours

---

## Vendor Selection Criteria

| Criterion | Weight | Questions |
|-----------|--------|-----------|
| **Data Coverage** | 30% | US only? Global? Options/Futures? |
| **Latency** | 25% | Real-time or 15-min delayed? |
| **Cost** | 20% | Per user? Per API call? Flat fee? |
| **API Quality** | 15% | REST? WebSocket? Python SDK? |
| **Historical Depth** | 10% | How many years of tick data? |

---

## Best Practices

1. **Multi-vendor strategy** - Don't rely on single vendor (redundancy)
2. **Local caching** - Reduce API costs and improve latency
3. **Error handling** - Implement retries, exponential backoff
4. **Rate limiting** - Respect vendor limits to avoid bans
5. **Monitoring** - Track API latency, error rates, costs
6. **Data validation** - Verify data quality (missing ticks, bad prices)

Now you can integrate any market data vendor into your trading system!
`,
};
