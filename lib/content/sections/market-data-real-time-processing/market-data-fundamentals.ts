export const marketDataFundamentals = {
  title: 'Market Data Fundamentals',
  id: 'market-data-fundamentals',
  content: `
# Market Data Fundamentals

## Introduction

Market data is the lifeblood of modern financial markets. Every trading decision, risk calculation, and price discovery mechanism depends on the accurate, timely flow of market information. For engineers building financial systems, understanding market data is fundamental - it's not just about displaying prices on a screen, but about processing millions of events per second, maintaining microsecond-level latency, and ensuring data integrity across distributed systems.

**Why Market Data Matters for Engineers:**
- **Scale**: Major exchanges process 10+ million messages per second during peak trading
- **Latency**: Microseconds matter - HFT firms co-locate servers to save 4-5 microseconds
- **Accuracy**: A single bad tick can trigger millions in erroneous trades
- **Cost**: Premium market data feeds cost $10K-100K+ per month
- **Complexity**: 60+ exchanges, multiple asset classes, different protocols

**Market Data in Production:**
- **Trading Firms**: Two Sigma processes 50+ terabytes of market data daily
- **Exchanges**: NASDAQ ITCH feed delivers 400+ million messages per day
- **Brokers**: Interactive Brokers serves real-time data to 2+ million users
- **Fintech**: Robinhood handles billions of quote updates for retail traders

By the end of this section, you'll understand:
- Types and hierarchy of market data
- Real-time vs delayed vs historical data
- Data feed characteristics and requirements
- Market data consumption patterns
- How to build production market data consumers

---

## Types of Market Data

### Quote Data (Level 1 - BBO)

**Best Bid and Offer (BBO)** - The most basic and widely consumed market data.

**Structure:**
\`\`\`python
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

@dataclass
class Quote:
    symbol: str
    timestamp: datetime
    bid_price: Decimal
    bid_size: int
    ask_price: Decimal
    ask_size: int
    exchange: str
    sequence_number: int
    
# Example Quote
quote = Quote(
    symbol="AAPL",
    timestamp=datetime(2024, 1, 15, 9, 30, 0, 123456),
    bid_price=Decimal("150.25"),
    bid_size=500,
    ask_price=Decimal("150.26"),
    ask_size=300,
    exchange="NASDAQ",
    sequence_number=12345678
)
\`\`\`

**Key Fields:**
- **Bid Price**: Highest price buyers are willing to pay
- **Bid Size**: Number of shares available at bid
- **Ask Price**: Lowest price sellers are willing to accept
- **Ask Size**: Number of shares available at ask
- **Spread**: ask_price - bid_price (here: $0.01)

**Update Frequency:**
- Liquid stocks: 100-1000+ updates per second
- Less liquid stocks: 1-10 updates per second
- Options: Can be slower (10-60 second delays)

### Trade Data (Time & Sales)

Records of actual executed transactions.

**Structure:**
\`\`\`python
@dataclass
class Trade:
    symbol: str
    timestamp: datetime
    price: Decimal
    size: int
    exchange: str
    trade_id: str
    conditions: list[str]  # Sale conditions
    sequence_number: int
    
# Example Trade
trade = Trade(
    symbol="AAPL",
    timestamp=datetime(2024, 1, 15, 9, 30, 0, 234567),
    price=Decimal("150.25"),
    size=100,
    exchange="NASDAQ",
    trade_id="T123456789",
    conditions=["@", "T"],  # @ = Regular sale, T = Extended hours
    sequence_number=12345679
)
\`\`\`

**Trade Conditions:**
- **Regular Sale**: Normal market trade
- **Block Trade**: Large institutional trade
- **Odd Lot**: < 100 shares
- **Extended Hours**: Pre-market or after-hours
- **Correction**: Trade was corrected
- **Cancel**: Trade was cancelled

**Volume:**
- **AAPL**: 50-100 million shares per day = ~5,000 trades per minute
- **SPY**: 80-100 million shares per day = ~8,000 trades per minute
- **Total US Equities**: 10-15 billion shares per day

### Market Depth (Level 2)

Order book showing multiple price levels beyond BBO.

**Structure:**
\`\`\`python
@dataclass
class OrderBookLevel:
    price: Decimal
    size: int
    num_orders: int  # Number of orders at this price
    
@dataclass
class OrderBook:
    symbol: str
    timestamp: datetime
    bids: list[OrderBookLevel]  # Sorted high to low
    asks: list[OrderBookLevel]  # Sorted low to high
    sequence_number: int
    
# Example Order Book (5 levels)
order_book = OrderBook(
    symbol="AAPL",
    timestamp=datetime.now(),
    bids=[
        OrderBookLevel(Decimal("150.25"), 500, 3),
        OrderBookLevel(Decimal("150.24"), 800, 5),
        OrderBookLevel(Decimal("150.23"), 1200, 7),
        OrderBookLevel(Decimal("150.22"), 600, 4),
        OrderBookLevel(Decimal("150.21"), 1000, 6),
    ],
    asks=[
        OrderBookLevel(Decimal("150.26"), 300, 2),
        OrderBookLevel(Decimal("150.27"), 700, 4),
        OrderBookLevel(Decimal("150.28"), 900, 6),
        OrderBookLevel(Decimal("150.29"), 400, 3),
        OrderBookLevel(Decimal("150.30"), 1500, 8),
    ],
    sequence_number=12345680
)
\`\`\`

**Depth Levels:**
- **Level 2 (5-10 levels)**: $50-200/month per symbol
- **Full Depth (all levels)**: $500-5000/month per symbol
- **NASDAQ TotalView**: Shows every order (most expensive)

**Update Rate:**
- High-volume stocks: 1000+ updates per second
- Data size: 10-50× larger than Level 1

### OHLCV Bars (Aggregated Data)

Time-aggregated summaries of trades.

**Structure:**
\`\`\`python
@dataclass
class Bar:
    symbol: str
    timestamp: datetime  # Bar start time
    interval: str  # "1min", "5min", "1hour", etc.
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    trade_count: int  # Optional
    vwap: Decimal  # Volume-Weighted Average Price
    
# Example 1-minute bar
bar = Bar(
    symbol="AAPL",
    timestamp=datetime(2024, 1, 15, 9, 30, 0),
    interval="1min",
    open=Decimal("150.25"),
    high=Decimal("150.35"),
    low=Decimal("150.20"),
    close=Decimal("150.30"),
    volume=12500,
    trade_count=234,
    vwap=Decimal("150.275")
)
\`\`\`

**Common Intervals:**
- **1 second**: HFT analysis
- **1 minute**: Day trading
- **5 minutes**: Swing trading
- **1 hour**: Position trading
- **1 day**: Long-term investing

**Storage Requirements:**
- 1-minute bars: ~400 bars per symbol per day
- 1-second bars: 23,400 bars per symbol per day
- SPY (5 years, 1-min): ~500K bars = ~40 MB

---

## Real-Time vs Delayed vs Historical Data

### Real-Time Data

**Definition**: Data delivered with minimal delay from exchange execution.

**Latency:**
- **Direct Exchange Feed**: 1-10 microseconds (co-located)
- **Vendor Feed (Bloomberg)**: 10-100 milliseconds
- **Retail API (Polygon, IEX)**: 100-1000 milliseconds
- **WebSocket Consumer**: Additional 10-50 milliseconds

**Cost:**
- **Exchange Direct**: $1,000-10,000/month per feed
- **Bloomberg Terminal**: $2,000/month (includes real-time)
- **Refinitiv**: $500-5,000/month depending on package
- **IEX Cloud**: $0.05 per 1000 messages (affordable)
- **Polygon.io**: $200-2,000/month (retail-friendly)

**Use Cases:**
- Algorithmic trading (requires real-time)
- Market making (requires ultra-low latency)
- Day trading platforms
- Risk management systems

**Licensing:**
- Must sign exchange agreements
- Per-user fees for professional traders
- Display-only vs trading use (different pricing)

### Delayed Data (15-20 Minutes)

**Definition**: Free, exchange-delayed quotes available to all.

**Delay:**
- **US Exchanges**: 15 minutes required by regulation
- **Futures**: 10 minutes for CME
- **International**: Varies by exchange (some 15-60 minutes)

**Cost:**
- **Free**: Yahoo Finance, Google Finance
- **No licensing**: Can display publicly

**Use Cases:**
- Retail information displays
- Educational purposes
- Historical analysis
- Personal portfolio tracking

**Limitations:**
- Useless for active trading
- May miss important moves
- Old news by the time you see it

### Historical Data

**Definition**: Past market data for backtesting and analysis.

**Granularity:**
- **Tick Data**: Every quote and trade
- **1-second bars**: Popular for HFT research
- **1-minute bars**: Standard for day trading backtests
- **Daily bars**: Long-term analysis

**Storage:**
- **SPY tick data**: ~20 GB per year (uncompressed)
- **SPY 1-minute bars**: ~100 MB per year
- **SPY daily bars**: 10,000 rows = 1 MB for 40 years

**Sources:**
- **Free**: Yahoo Finance (daily), Alpha Vantage (limited)
- **Affordable**: Polygon.io ($200/month), IEX Cloud ($100/month)
- **Professional**: TickData ($500-10K), Nanex ($1K-5K)
- **Premium**: Bloomberg, Refinitiv ($10K-100K)

**Quality Issues:**
- Survivorship bias (delisted stocks missing)
- Split/dividend adjustments
- Bad ticks not cleaned
- Missing data gaps

---

## Data Feed Characteristics

### Message Rate

**Typical Rates by Asset Class:**

**US Equities:**
- Peak: 10-20 million messages/second (market-wide)
- SPY: 100-500 messages/second
- AAPL: 50-200 messages/second
- Low-volume stock: 0.1-10 messages/second

**Options:**
- Peak: 50+ million messages/second (market-wide)
- Single option: 0.01-1 message/second
- SPY options chain (1000 strikes): 100-1000 msg/sec

**Futures:**
- ES (S&P 500 E-mini): 500-2000 messages/second
- NQ (NASDAQ E-mini): 200-1000 messages/second

**Forex:**
- Major pairs: 100-1000 ticks/second (all venues)
- Single venue: 10-50 ticks/second

**Crypto:**
- BTC/USD: 50-500 updates/second (aggregated)
- Low-volume altcoin: 0.1-5 updates/second

### Latency Requirements

**By Use Case:**

| Use Case | Max Latency | Notes |
|----------|-------------|-------|
| High-Frequency Trading | < 100 μs | Co-location required |
| Market Making | < 1 ms | Direct exchange feed |
| Algorithmic Trading | 1-10 ms | Vendor feed acceptable |
| Day Trading Platform | 10-100 ms | Websocket to users |
| Portfolio Management | 100-1000 ms | Less critical |
| Long-term Investing | > 1 second | Can use delayed |

**Latency Components:**
\`\`\`
Exchange → Feed Handler → Normalization → Your App → User Display
  (1μs)      (10μs)           (100μs)        (1ms)      (10ms)
\`\`\`

### Bandwidth Requirements

**Example Calculation for SPY:**
- Message rate: 200 messages/second
- Average message size: 100 bytes
- Bandwidth: 200 × 100 = 20 KB/sec = 160 Kb/sec

**Market-Wide:**
- All US equities: 20 million msg/sec × 100 bytes = 2 GB/sec = 16 Gbps
- Compressed: ~500 Mbps with efficient protocol

**Your Capacity Planning:**
- 100 symbols @ 100 msg/sec each = 10K msg/sec = 1 MB/sec
- Network: 10 Mbps minimum (allow 10× headroom = 100 Mbps)
- CPU: Single core can handle 100K msg/sec (with efficient code)

---

## Market Data Consumption Patterns

### REST API Polling

**Use Case**: Low-frequency updates, simple implementation.

**Python Example:**
\`\`\`python
import requests
import time
from decimal import Decimal

class RESTMarketDataConsumer:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})
    
    def get_quote(self, symbol: str) -> dict:
        """Fetch latest quote for symbol"""
        response = self.session.get(
            f"{self.base_url}/quotes/{symbol}"
        )
        response.raise_for_status()
        return response.json()
    
    def poll_quotes(self, symbols: list[str], interval_seconds: int = 1):
        """Poll quotes at regular intervals"""
        while True:
            for symbol in symbols:
                try:
                    quote = self.get_quote(symbol)
                    print(f"{symbol}: Bid {quote['bid']} Ask {quote['ask']}")
                except Exception as e:
                    print(f"Error fetching {symbol}: {e}")
            time.sleep(interval_seconds)

# Usage
consumer = RESTMarketDataConsumer(
    api_key="your_api_key",
    base_url="https://api.example.com/v1"
)

# Poll every 1 second
consumer.poll_quotes(["AAPL", "MSFT", "GOOGL"], interval_seconds=1)
\`\`\`

**Limitations:**
- ❌ High latency (100-1000 ms)
- ❌ Rate limits (often 5-100 requests/second)
- ❌ Inefficient (separate request per symbol)
- ❌ Can't scale to many symbols
- ✅ Simple to implement
- ✅ Good for low-frequency updates

### WebSocket Streaming

**Use Case**: Real-time updates, efficient for multiple symbols.

**Python Example:**
\`\`\`python
import asyncio
import websockets
import json
from datetime import datetime

class WebSocketMarketDataConsumer:
    def __init__(self, ws_url: str, api_key: str):
        self.ws_url = ws_url
        self.api_key = api_key
        self.ws = None
        self.subscriptions = set()
    
    async def connect(self):
        """Establish WebSocket connection"""
        self.ws = await websockets.connect(
            self.ws_url,
            extra_headers={"Authorization": f"Bearer {self.api_key}"}
        )
        print("WebSocket connected")
        
        # Authenticate
        await self.ws.send(json.dumps({
            "action": "authenticate",
            "key": self.api_key
        }))
        response = await self.ws.recv()
        print(f"Auth response: {response}")
    
    async def subscribe(self, symbols: list[str], data_type: str = "quotes"):
        """Subscribe to symbols"""
        for symbol in symbols:
            await self.ws.send(json.dumps({
                "action": "subscribe",
                "symbols": [symbol],
                "type": data_type
            }))
            self.subscriptions.add((symbol, data_type))
            print(f"Subscribed to {symbol} {data_type}")
    
    async def listen(self, callback):
        """Listen for market data updates"""
        try:
            async for message in self.ws:
                data = json.loads(message)
                await callback(data)
        except websockets.ConnectionClosed:
            print("WebSocket connection closed")
            # Reconnect logic here
    
    async def run(self, symbols: list[str]):
        """Main run loop"""
        await self.connect()
        await self.subscribe(symbols)
        await self.listen(self.handle_message)
    
    async def handle_message(self, data: dict):
        """Handle incoming market data"""
        if data.get("type") == "quote":
            symbol = data["symbol"]
            bid = data["bid"]
            ask = data["ask"]
            timestamp = datetime.now()
            print(f"[{timestamp}] {symbol}: {bid} x {ask}")

# Usage
async def main():
    consumer = WebSocketMarketDataConsumer(
        ws_url="wss://api.example.com/v1/stream",
        api_key="your_api_key"
    )
    await consumer.run(["AAPL", "MSFT", "GOOGL", "TSLA"])

asyncio.run(main())
\`\`\`

**Advantages:**
- ✅ Low latency (10-100 ms)
- ✅ Efficient (single connection, multiple symbols)
- ✅ Push-based (no polling needed)
- ✅ Scales to 100s-1000s of symbols

**Challenges:**
- 🔧 Reconnection logic needed
- 🔧 Heartbeat/keepalive required
- 🔧 Message ordering not guaranteed
- 🔧 Backpressure handling needed

### Direct Exchange Feed

**Use Case**: Ultra-low latency for professional trading.

**Characteristics:**
- Binary protocols (FIX, ITCH, FAST)
- Multicast UDP (no TCP overhead)
- Co-located servers (same data center as exchange)
- Latency: 1-100 microseconds

**Example Protocol: NASDAQ ITCH**
\`\`\`python
import struct
from enum import Enum

class ITCHMessageType(Enum):
    ADD_ORDER = b'A'
    TRADE = b'P'
    ORDER_CANCEL = b'X'
    # ... more message types

def parse_itch_add_order(data: bytes):
    """Parse NASDAQ ITCH Add Order message"""
    # Format: https://www.nasdaqtrader.com/content/technicalsupport/specifications/dataproducts/NQTVITCHSpecification.pdf
    
    message_type = chr(data[0])
    stock_locate = struct.unpack('>H', data[1:3])[0]
    tracking_number = struct.unpack('>H', data[3:5])[0]
    timestamp = struct.unpack('>Q', data[5:11])[0]  # Nanoseconds since midnight
    order_ref = struct.unpack('>Q', data[11:19])[0]
    buy_sell = chr(data[19])
    shares = struct.unpack('>I', data[20:24])[0]
    stock = data[24:32].decode('ascii').strip()
    price = struct.unpack('>I', data[32:36])[0] / 10000  # Price in 1/10000
    
    return {
        'type': 'add_order',
        'stock': stock,
        'side': 'buy' if buy_sell == 'B' else 'sell',
        'price': price,
        'shares': shares,
        'timestamp': timestamp,
        'order_ref': order_ref
    }
\`\`\`

**Professional Setup:**
- Direct exchange connectivity: $5K-50K setup
- Co-location fees: $1K-10K/month
- Market data fees: $1K-20K/month
- Required for HFT and market making

---

## Production Consumer Example

**Complete Production-Grade WebSocket Consumer:**

\`\`\`python
import asyncio
import websockets
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Callable, Dict, List
import logging
from decimal import Decimal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Quote:
    symbol: str
    timestamp: datetime
    bid_price: Decimal
    bid_size: int
    ask_price: Decimal
    ask_size: int
    sequence: int

class ProductionMarketDataConsumer:
    """Production-grade WebSocket market data consumer"""
    
    def __init__(
        self,
        ws_url: str,
        api_key: str,
        reconnect_delay: int = 5,
        heartbeat_interval: int = 30
    ):
        self.ws_url = ws_url
        self.api_key = api_key
        self.reconnect_delay = reconnect_delay
        self.heartbeat_interval = heartbeat_interval
        
        self.ws = None
        self.subscriptions: Dict[str, List[str]] = {}  # type -> symbols
        self.callbacks: Dict[str, Callable] = {}
        self.running = False
        self.message_count = 0
        self.last_heartbeat = None
    
    async def connect(self):
        """Establish WebSocket connection with retry logic"""
        while True:
            try:
                logger.info(f"Connecting to {self.ws_url}")
                self.ws = await websockets.connect(
                    self.ws_url,
                    extra_headers={"Authorization": f"Bearer {self.api_key}"},
                    ping_interval=self.heartbeat_interval,
                    ping_timeout=self.heartbeat_interval * 2
                )
                
                # Authenticate
                await self.ws.send(json.dumps({
                    "action": "authenticate",
                    "key": self.api_key
                }))
                
                auth_response = await asyncio.wait_for(
                    self.ws.recv(),
                    timeout=10.0
                )
                logger.info(f"Authentication: {auth_response}")
                
                # Resubscribe after reconnection
                await self.resubscribe()
                
                logger.info("Successfully connected and subscribed")
                return
                
            except Exception as e:
                logger.error(f"Connection failed: {e}. Retrying in {self.reconnect_delay}s")
                await asyncio.sleep(self.reconnect_delay)
    
    async def resubscribe(self):
        """Resubscribe to all previous subscriptions"""
        for data_type, symbols in self.subscriptions.items():
            if symbols:
                await self.ws.send(json.dumps({
                    "action": "subscribe",
                    "type": data_type,
                    "symbols": symbols
                }))
                logger.info(f"Resubscribed to {len(symbols)} {data_type}")
    
    async def subscribe(self, symbols: List[str], data_type: str = "quotes"):
        """Subscribe to market data"""
        if data_type not in self.subscriptions:
            self.subscriptions[data_type] = []
        
        self.subscriptions[data_type].extend(symbols)
        
        if self.ws:
            await self.ws.send(json.dumps({
                "action": "subscribe",
                "type": data_type,
                "symbols": symbols
            }))
            logger.info(f"Subscribed to {len(symbols)} symbols for {data_type}")
    
    def register_callback(self, data_type: str, callback: Callable):
        """Register callback for data type"""
        self.callbacks[data_type] = callback
    
    async def listen(self):
        """Main message listening loop"""
        try:
            async for message in self.ws:
                self.message_count += 1
                
                try:
                    data = json.loads(message)
                    data_type = data.get("type")
                    
                    if data_type in self.callbacks:
                        await self.callbacks[data_type](data)
                    
                    # Log throughput every 10K messages
                    if self.message_count % 10000 == 0:
                        logger.info(f"Processed {self.message_count} messages")
                        
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    
        except websockets.ConnectionClosed as e:
            logger.warning(f"WebSocket closed: {e}. Reconnecting...")
            await self.connect()
            await self.listen()  # Resume listening
    
    async def run(self):
        """Main run loop"""
        self.running = True
        await self.connect()
        await self.listen()
    
    async def stop(self):
        """Graceful shutdown"""
        self.running = False
        if self.ws:
            await self.ws.close()
        logger.info("Consumer stopped")

# Usage Example
async def quote_handler(data: dict):
    """Handle quote updates"""
    quote = Quote(
        symbol=data["symbol"],
        timestamp=datetime.fromisoformat(data["timestamp"]),
        bid_price=Decimal(str(data["bid"])),
        bid_size=data["bidSize"],
        ask_price=Decimal(str(data["ask"])),
        ask_size=data["askSize"],
        sequence=data.get("sequence", 0)
    )
    
    # Process quote (e.g., update database, trigger strategies)
    logger.info(f"{quote.symbol}: {quote.bid_price} x {quote.ask_price}")

async def main():
    consumer = ProductionMarketDataConsumer(
        ws_url="wss://api.polygon.io/stocks",
        api_key="your_api_key_here",
        reconnect_delay=5,
        heartbeat_interval=30
    )
    
    # Register handlers
    consumer.register_callback("quote", quote_handler)
    
    # Subscribe to symbols
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    await consumer.subscribe(symbols, "quotes")
    
    # Run consumer
    try:
        await consumer.run()
    except KeyboardInterrupt:
        await consumer.stop()

if __name__ == "__main__":
    asyncio.run(main())
\`\`\`

---

## Common Pitfalls

### 1. Not Handling Reconnections

**Problem**: WebSocket disconnects, application stops receiving data.

**Solution**: Implement automatic reconnection with exponential backoff:
\`\`\`python
async def connect_with_backoff(self, max_retries=10):
    for attempt in range(max_retries):
        try:
            await self.connect()
            return
        except Exception as e:
            delay = min(300, 2 ** attempt)  # Cap at 5 minutes
            logger.error(f"Connection failed (attempt {attempt+1}): {e}")
            await asyncio.sleep(delay)
    raise Exception("Failed to connect after max retries")
\`\`\`

### 2. Ignoring Sequence Numbers

**Problem**: Missing or duplicate messages go undetected.

**Solution**: Track sequence numbers and detect gaps:
\`\`\`python
class SequenceTracker:
    def __init__(self):
        self.last_seq = {}
    
    def check(self, symbol: str, seq: int) -> str:
        if symbol not in self.last_seq:
            self.last_seq[symbol] = seq
            return "OK"
        
        expected = self.last_seq[symbol] + 1
        if seq == expected:
            self.last_seq[symbol] = seq
            return "OK"
        elif seq < expected:
            return f"DUPLICATE: got {seq}, expected {expected}"
        else:
            gap = seq - expected
            self.last_seq[symbol] = seq
            return f"GAP: missing {gap} messages"
\`\`\`

### 3. Not Converting Timestamps

**Problem**: Using string timestamps, losing precision, timezone issues.

**Solution**: Parse to datetime immediately, use UTC:
\`\`\`python
from datetime import datetime, timezone

def parse_exchange_timestamp(ts_string: str) -> datetime:
    """Parse exchange timestamp to UTC datetime"""
    # Many exchanges use RFC3339
    dt = datetime.fromisoformat(ts_string.replace('Z', '+00:00'))
    return dt.astimezone(timezone.utc)
\`\`\`

### 4. Blocking the Event Loop

**Problem**: Slow callback blocks receiving new messages.

**Solution**: Process messages asynchronously:
\`\`\`python
async def fast_handler(data: dict):
    # Quick validation
    if not data.get("symbol"):
        return
    
    # Offload heavy work to thread pool or queue
    await asyncio.create_task(process_heavy_work(data))

async def process_heavy_work(data: dict):
    # Database writes, calculations, etc.
    pass
\`\`\`

---

## Best Practices Summary

1. **Always handle reconnections** - Markets don't wait for your app to restart
2. **Track sequence numbers** - Detect gaps and duplicates
3. **Use UTC timestamps** - Avoid timezone headaches
4. **Monitor latency** - Track receive_time - exchange_time
5. **Buffer messages** - Use queues to handle bursts
6. **Rate limit subscriptions** - Don't overwhelm the feed
7. **Log everything** - You'll need it for debugging
8. **Test failover** - Simulate disconnections in staging
9. **Validate data** - Check for negative prices, zero sizes
10. **Document your feed** - Future you will thank you

**Performance Targets:**
- WebSocket latency: < 50ms p99
- Message processing: < 1ms per message
- Reconnection time: < 5 seconds
- Memory usage: < 100 MB for 1000 symbols
- CPU usage: < 25% of one core

---

## Next Steps

Now that you understand market data fundamentals, you're ready to:
1. **Implement a production consumer** using the patterns above
2. **Learn data feed protocols** (FIX, ITCH) in the next section
3. **Build tick data processors** for high-frequency data
4. **Design storage systems** for billions of ticks
5. **Optimize for latency** with advanced techniques

Market data is the foundation - everything else (trading, risk, analytics) builds on top of reliable, fast, accurate data feeds.
`,
};
