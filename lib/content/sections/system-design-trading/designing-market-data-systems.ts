export const designingMarketDataSystems = {
  title: 'Designing Market Data Systems',
  id: 'designing-market-data-systems',
  content: `
# Designing Market Data Systems

## Introduction

A **Market Data System** is the nervous system of any trading operation. It ingests, normalizes, stores, and serves market data at massive scale—often **1 million+ messages per second** during peak trading. The system must provide:

- **Real-time feeds**: Tick-by-tick data with <1ms latency
- **Historical data**: Years of OHLCV bars for backtesting
- **Data quality**: Handle exchange outages, missing ticks, bad prices
- **Multi-source**: NYSE, NASDAQ, Binance, plus alternative data
- **Query performance**: Retrieve 1 year of data in <100ms

Poor market data infrastructure leads to:
- Missed trading opportunities (stale data)
- Bad decisions (incorrect prices)
- Failed backtests (survivorship bias)
- Regulatory violations (incomplete audit trail)

By the end of this section, you'll understand:
- Tick data ingestion at 1M+ msgs/sec
- Data normalization across exchanges
- Storage strategies (hot/warm/cold)
- Query optimization for backtesting
- Real-time aggregation (OHLCV bars)
- Historical replay architecture

---

## Data Types and Scale

### Market Data Hierarchy

\`\`\`
Level 0: Trade ticks (price, quantity, timestamp)
         Volume: 100-500K msgs/sec (liquid stocks)
         
Level 1: Top of book (best bid/ask)
         Volume: 50-100K msgs/sec
         
Level 2: Order book depth (5-10 levels)
         Volume: 500K-1M msgs/sec
         
Level 3: Full order book (all levels)
         Volume: 2-5M msgs/sec
\`\`\`

### Storage Requirements

For a single symbol (AAPL):
- **Tick data**: 500K msgs/day × 252 days/year × 100 bytes = **12.6 GB/year**
- **1-minute bars**: 390 bars/day × 252 days × 100 bytes = **10 MB/year**
- **Daily bars**: 252 bars/year × 100 bytes = **25 KB/year**

For 3000 symbols: **37.8 TB/year** (tick data only)

---

## Architecture Overview

\`\`\`
Exchange Feeds (FIX, WebSocket)
         ↓
  Feed Handlers (normalize)
         ↓
   Message Queue (Kafka)
         ↓
    ┌────────┬────────┬────────┐
    ↓        ↓        ↓        ↓
  Storage  Aggr   Analytics  OMS
  (Time    (Bars) (Signals)  (Exec)
   Series)
\`\`\`

---

## Feed Handler Design

### Multi-Protocol Ingestion

\`\`\`python
"""
Feed Handler for Multiple Exchanges
Normalizes different protocols to common format
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable, Dict
from datetime import datetime
import asyncio
import websockets
import json

class MessageType(Enum):
    TRADE = "TRADE"
    QUOTE = "QUOTE"  # Best bid/ask
    BOOK = "BOOK"    # Order book update
    
@dataclass
class Tick:
    """Normalized market data tick"""
    symbol: str
    timestamp: int  # Microseconds since epoch
    message_type: MessageType
    
    # Trade
    trade_price: Optional[float] = None
    trade_size: Optional[float] = None
    trade_side: Optional[str] = None  # BUY/SELL
    
    # Quote (Level 1)
    bid_price: Optional[float] = None
    bid_size: Optional[float] = None
    ask_price: Optional[float] = None
    ask_size: Optional[float] = None
    
    # Book (Level 2)
    bids: Optional[list] = None  # [(price, size), ...]
    asks: Optional[list] = None
    
    # Exchange metadata
    exchange: str = ""
    sequence_number: Optional[int] = None

class ExchangeFeedHandler:
    """Base class for exchange-specific handlers"""
    
    def __init__(self, exchange_name: str):
        self.exchange_name = exchange_name
        self.callbacks = []
    
    def subscribe (self, callback: Callable[[Tick], None]):
        """Subscribe to tick updates"""
        self.callbacks.append (callback)
    
    def publish (self, tick: Tick):
        """Publish tick to subscribers"""
        for callback in self.callbacks:
            callback (tick)
    
    async def connect (self):
        """Connect to exchange feed"""
        raise NotImplementedError
    
    def parse_message (self, raw_message: dict) -> Optional[Tick]:
        """Parse exchange-specific message to Tick"""
        raise NotImplementedError

class BinanceFeedHandler(ExchangeFeedHandler):
    """
    Binance WebSocket feed handler
    Public API: wss://stream.binance.com:9443
    """
    
    def __init__(self, symbols: list[str]):
        super().__init__("BINANCE")
        self.symbols = [s.lower() for s in symbols]  # Binance uses lowercase
        self.ws = None
    
    async def connect (self):
        """Connect to Binance WebSocket"""
        # Create stream path
        streams = [f"{symbol}@trade" for symbol in self.symbols]
        streams += [f"{symbol}@bookTicker" for symbol in self.symbols]
        stream_path = "/".join (streams)
        
        url = f"wss://stream.binance.com:9443/stream?streams={stream_path}"
        
        print(f"Connecting to Binance: {url}")
        async with websockets.connect (url) as ws:
            self.ws = ws
            async for message in ws:
                try:
                    data = json.loads (message)
                    tick = self.parse_message (data)
                    if tick:
                        self.publish (tick)
                except Exception as e:
                    print(f"Error parsing Binance message: {e}")
    
    def parse_message (self, data: dict) -> Optional[Tick]:
        """Parse Binance message"""
        if 'data' not in data:
            return None
        
        msg = data['data']
        stream = data.get('stream', '')
        
        # Trade message
        if '@trade' in stream:
            return Tick(
                symbol=msg['s'],  # BTCUSDT
                timestamp=int (msg['T'] * 1000),  # Convert ms to μs
                message_type=MessageType.TRADE,
                trade_price=float (msg['p']),
                trade_size=float (msg['q']),
                trade_side="BUY" if msg['m'] else "SELL",  # m=true means buyer is maker
                exchange=self.exchange_name,
                sequence_number=msg['t']
            )
        
        # Quote message (bookTicker)
        elif '@bookTicker' in stream:
            return Tick(
                symbol=msg['s'],
                timestamp=int (msg['E'] * 1000),  # Event time in μs
                message_type=MessageType.QUOTE,
                bid_price=float (msg['b']),
                bid_size=float (msg['B']),
                ask_price=float (msg['a']),
                ask_size=float (msg['A']),
                exchange=self.exchange_name
            )
        
        return None

class AlpacaFeedHandler(ExchangeFeedHandler):
    """
    Alpaca WebSocket feed handler
    For US stocks
    """
    
    def __init__(self, symbols: list[str], api_key: str, api_secret: str):
        super().__init__("ALPACA")
        self.symbols = symbols
        self.api_key = api_key
        self.api_secret = api_secret
        self.ws = None
    
    async def connect (self):
        """Connect to Alpaca WebSocket"""
        url = "wss://stream.data.alpaca.markets/v2/iex"
        
        async with websockets.connect (url) as ws:
            self.ws = ws
            
            # Authenticate
            auth_msg = {
                "action": "auth",
                "key": self.api_key,
                "secret": self.api_secret
            }
            await ws.send (json.dumps (auth_msg))
            auth_response = await ws.recv()
            print(f"Alpaca auth: {auth_response}")
            
            # Subscribe to trades and quotes
            subscribe_msg = {
                "action": "subscribe",
                "trades": self.symbols,
                "quotes": self.symbols
            }
            await ws.send (json.dumps (subscribe_msg))
            
            # Process messages
            async for message in ws:
                try:
                    data = json.loads (message)
                    for item in data:
                        tick = self.parse_message (item)
                        if tick:
                            self.publish (tick)
                except Exception as e:
                    print(f"Error parsing Alpaca message: {e}")
    
    def parse_message (self, msg: dict) -> Optional[Tick]:
        """Parse Alpaca message"""
        msg_type = msg.get('T')
        
        if msg_type == 't':  # Trade
            return Tick(
                symbol=msg['S'],
                timestamp=int (datetime.fromisoformat (msg['t'].replace('Z', '+00:00')).timestamp() * 1_000_000),
                message_type=MessageType.TRADE,
                trade_price=msg['p'],
                trade_size=msg['s'],
                exchange=self.exchange_name
            )
        
        elif msg_type == 'q':  # Quote
            return Tick(
                symbol=msg['S'],
                timestamp=int (datetime.fromisoformat (msg['t'].replace('Z', '+00:00')).timestamp() * 1_000_000),
                message_type=MessageType.QUOTE,
                bid_price=msg['bp'],
                bid_size=msg['bs'],
                ask_price=msg['ap'],
                ask_size=msg['as'],
                exchange=self.exchange_name
            )
        
        return None

# Example: Unified feed aggregator
class MarketDataAggregator:
    """
    Aggregates data from multiple exchanges
    Publishes to unified stream
    """
    
    def __init__(self):
        self.handlers = []
        self.tick_count = 0
        self.last_report_time = datetime.now()
    
    def add_handler (self, handler: ExchangeFeedHandler):
        """Add exchange handler"""
        handler.subscribe (self.on_tick)
        self.handlers.append (handler)
    
    def on_tick (self, tick: Tick):
        """Handle incoming tick"""
        self.tick_count += 1
        
        # Process tick (store, publish, etc.)
        # In production: Send to Kafka
        
        # Report throughput
        now = datetime.now()
        if (now - self.last_report_time).total_seconds() >= 10:
            rate = self.tick_count / 10
            print(f"Throughput: {rate:.0f} msgs/sec")
            self.tick_count = 0
            self.last_report_time = now
    
    async def start (self):
        """Start all handlers"""
        tasks = [handler.connect() for handler in self.handlers]
        await asyncio.gather(*tasks)

# Usage
async def main():
    aggregator = MarketDataAggregator()
    
    # Add Binance (crypto)
    binance = BinanceFeedHandler(['BTCUSDT', 'ETHUSDT'])
    aggregator.add_handler (binance)
    
    # Add Alpaca (stocks) - requires credentials
    # alpaca = AlpacaFeedHandler(['AAPL', 'TSLA'], 'key', 'secret')
    # aggregator.add_handler (alpaca)
    
    await aggregator.start()

# Run
# asyncio.run (main())
\`\`\`

---

## Data Storage Strategy

### Tiered Storage Architecture

\`\`\`
Hot Storage (0-7 days)
  ↓ TimescaleDB/ClickHouse (in-memory)
  ↓ Query latency: <10ms
  ↓ Cost: $$$

Warm Storage (7-90 days)
  ↓ ClickHouse/Parquet on SSD
  ↓ Query latency: 100ms
  ↓ Cost: $$

Cold Storage (90+ days)
  ↓ S3/Parquet compressed
  ↓ Query latency: 1-5s
  ↓ Cost: $
\`\`\`

### TimescaleDB Implementation

\`\`\`python
"""
TimescaleDB for tick data storage
PostgreSQL extension optimized for time series
"""

import psycopg2
from psycopg2.extras import execute_batch

class TickDatabase:
    """
    Store ticks in TimescaleDB
    Partitioned by time for efficient queries
    """
    
    def __init__(self, conn_string: str):
        self.conn = psycopg2.connect (conn_string)
        self.cursor = self.conn.cursor()
        self._create_schema()
    
    def _create_schema (self):
        """Create tables and hypertables"""
        # Create trades table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                timestamp TIMESTAMPTZ NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                exchange VARCHAR(20),
                price DOUBLE PRECISION,
                size DOUBLE PRECISION,
                side VARCHAR(10),
                sequence_number BIGINT
            );
        """)
        
        # Convert to hypertable (TimescaleDB feature)
        try:
            self.cursor.execute("""
                SELECT create_hypertable('trades', 'timestamp',
                    chunk_time_interval => INTERVAL '1 day',
                    if_not_exists => TRUE);
            """)
        except:
            pass  # Already a hypertable
        
        # Create indexes
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS trades_symbol_time_idx 
            ON trades (symbol, timestamp DESC);
        """)
        
        # Create quotes table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS quotes (
                timestamp TIMESTAMPTZ NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                exchange VARCHAR(20),
                bid_price DOUBLE PRECISION,
                bid_size DOUBLE PRECISION,
                ask_price DOUBLE PRECISION,
                ask_size DOUBLE PRECISION
            );
        """)
        
        try:
            self.cursor.execute("""
                SELECT create_hypertable('quotes', 'timestamp',
                    chunk_time_interval => INTERVAL '1 day',
                    if_not_exists => TRUE);
            """)
        except:
            pass
        
        self.conn.commit()
    
    def insert_trades (self, ticks: list[Tick]):
        """Batch insert trades"""
        data = [
            (
                datetime.fromtimestamp (tick.timestamp / 1_000_000),
                tick.symbol,
                tick.exchange,
                tick.trade_price,
                tick.trade_size,
                tick.trade_side,
                tick.sequence_number
            )
            for tick in ticks
            if tick.message_type == MessageType.TRADE
        ]
        
        if data:
            execute_batch(
                self.cursor,
                """
                INSERT INTO trades (timestamp, symbol, exchange, price, size, side, sequence_number)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                data,
                page_size=1000
            )
            self.conn.commit()
    
    def query_trades (self, symbol: str, start_time: datetime, end_time: datetime):
        """Query trades for symbol in time range"""
        self.cursor.execute("""
            SELECT timestamp, price, size, side
            FROM trades
            WHERE symbol = %s
            AND timestamp >= %s
            AND timestamp < %s
            ORDER BY timestamp
        """, (symbol, start_time, end_time))
        
        return self.cursor.fetchall()
    
    def create_bars (self, symbol: str, interval: str = '1 minute'):
        """
        Create OHLCV bars using time_bucket
        TimescaleDB feature for fast aggregation
        """
        self.cursor.execute (f"""
            SELECT
                time_bucket('{interval}', timestamp) AS bar_time,
                symbol,
                FIRST(price, timestamp) AS open,
                MAX(price) AS high,
                MIN(price) AS low,
                LAST(price, timestamp) AS close,
                SUM(size) AS volume
            FROM trades
            WHERE symbol = %s
            GROUP BY bar_time, symbol
            ORDER BY bar_time
        """, (symbol,))
        
        return self.cursor.fetchall()
\`\`\`

### ClickHouse for Analytics

\`\`\`python
"""
ClickHouse for fast analytical queries
Columnar storage, excellent compression
"""

from clickhouse_driver import Client

class ClickHouseMarketData:
    """
    ClickHouse for market data analytics
    10-100x faster than PostgreSQL for aggregations
    """
    
    def __init__(self, host: str = 'localhost'):
        self.client = Client (host)
        self._create_tables()
    
    def _create_tables (self):
        """Create tables with optimized schema"""
        # Trades table with compression
        self.client.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                timestamp DateTime64(6),
                symbol LowCardinality(String),
                exchange LowCardinality(String),
                price Float64,
                size Float64,
                side Enum8('BUY' = 1, 'SELL' = -1)
            )
            ENGINE = MergeTree()
            PARTITION BY toYYYYMM(timestamp)
            ORDER BY (symbol, timestamp)
            SETTINGS index_granularity = 8192
        """)
        
        # Materialized view for 1-minute bars
        self.client.execute("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS bars_1m
            ENGINE = SummingMergeTree()
            PARTITION BY toYYYYMM(bar_time)
            ORDER BY (symbol, bar_time)
            AS SELECT
                toStartOfMinute (timestamp) AS bar_time,
                symbol,
                argMin (price, timestamp) AS open,
                max (price) AS high,
                min (price) AS low,
                argMax (price, timestamp) AS close,
                sum (size) AS volume,
                count() AS trades
            FROM trades
            GROUP BY symbol, bar_time
        """)
    
    def insert_trades (self, ticks: list[Tick]):
        """Bulk insert trades"""
        data = [
            (
                tick.timestamp / 1_000_000,
                tick.symbol,
                tick.exchange,
                tick.trade_price,
                tick.trade_size,
                tick.trade_side
            )
            for tick in ticks
            if tick.message_type == MessageType.TRADE
        ]
        
        if data:
            self.client.execute(
                'INSERT INTO trades (timestamp, symbol, exchange, price, size, side) VALUES',
                data
            )
    
    def get_bars (self, symbol: str, start: str, end: str, interval: str = '1m'):
        """Get OHLCV bars"""
        query = """
            SELECT
                bar_time,
                open,
                high,
                low,
                close,
                volume,
                trades
            FROM bars_1m
            WHERE symbol = %(symbol)s
            AND bar_time >= %(start)s
            AND bar_time < %(end)s
            ORDER BY bar_time
        """
        
        return self.client.execute (query, {
            'symbol': symbol,
            'start': start,
            'end': end
        })
\`\`\`

---

## Real-Time Aggregation

### Streaming Bar Generation

\`\`\`python
"""
Real-time OHLCV bar generation from tick stream
"""

from collections import defaultdict
from typing import Dict

@dataclass
class Bar:
    """OHLCV bar"""
    timestamp: int
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    trades: int

class BarAggregator:
    """
    Generate real-time bars from tick stream
    Supports multiple timeframes simultaneously
    """
    
    def __init__(self, intervals_seconds: list[int] = [60, 300, 3600]):
        self.intervals = intervals_seconds
        self.current_bars: Dict[tuple, Bar] = {}  # (symbol, interval) -> Bar
        self.callbacks = defaultdict (list)  # interval -> [callbacks]
    
    def subscribe (self, interval_seconds: int, callback: Callable[[Bar], None]):
        """Subscribe to bar updates"""
        self.callbacks[interval_seconds].append (callback)
    
    def on_tick (self, tick: Tick):
        """Process tick and update bars"""
        if tick.message_type != MessageType.TRADE:
            return
        
        for interval in self.intervals:
            bar_timestamp = (tick.timestamp // (interval * 1_000_000)) * (interval * 1_000_000)
            key = (tick.symbol, interval)
            
            # Get or create bar
            if key not in self.current_bars:
                self.current_bars[key] = Bar(
                    timestamp=bar_timestamp,
                    symbol=tick.symbol,
                    open=tick.trade_price,
                    high=tick.trade_price,
                    low=tick.trade_price,
                    close=tick.trade_price,
                    volume=tick.trade_size,
                    trades=1
                )
            else:
                bar = self.current_bars[key]
                
                # Check if new bar started
                if tick.timestamp >= bar.timestamp + (interval * 1_000_000):
                    # Publish completed bar
                    for callback in self.callbacks[interval]:
                        callback (bar)
                    
                    # Start new bar
                    self.current_bars[key] = Bar(
                        timestamp=bar_timestamp,
                        symbol=tick.symbol,
                        open=tick.trade_price,
                        high=tick.trade_price,
                        low=tick.trade_price,
                        close=tick.trade_price,
                        volume=tick.trade_size,
                        trades=1
                    )
                else:
                    # Update current bar
                    bar.high = max (bar.high, tick.trade_price)
                    bar.low = min (bar.low, tick.trade_price)
                    bar.close = tick.trade_price
                    bar.volume += tick.trade_size
                    bar.trades += 1

# Usage
aggregator = BarAggregator (intervals_seconds=[60, 300])  # 1m, 5m

def on_1m_bar (bar: Bar):
    print(f"{bar.symbol} 1m: O={bar.open} H={bar.high} L={bar.low} C={bar.close} V={bar.volume}")

aggregator.subscribe(60, on_1m_bar)
\`\`\`

---

## Data Quality and Gap Handling

\`\`\`python
"""
Data quality checks and gap detection
"""

from datetime import timedelta

class DataQualityMonitor:
    """
    Monitor data quality
    Detect gaps, outliers, exchange outages
    """
    
    def __init__(self):
        self.last_tick_time = {}  # symbol -> timestamp
        self.tick_counts = defaultdict (int)  # symbol -> count
        self.price_history = defaultdict (list)  # symbol -> recent prices
    
    def check_tick (self, tick: Tick) -> list[str]:
        """
        Check tick quality
        Returns: list of warnings
        """
        warnings = []
        symbol = tick.symbol
        
        # Check 1: Gap detection
        if symbol in self.last_tick_time:
            gap = tick.timestamp - self.last_tick_time[symbol]
            if gap > 5_000_000:  # > 5 seconds
                warnings.append (f"Gap detected: {gap / 1_000_000:.1f}s since last tick")
        
        self.last_tick_time[symbol] = tick.timestamp
        
        # Check 2: Price spike detection
        if tick.message_type == MessageType.TRADE and tick.trade_price:
            history = self.price_history[symbol]
            
            if len (history) >= 10:
                avg_price = sum (history[-10:]) / 10
                std_price = (sum((p - avg_price)**2 for p in history[-10:]) / 10) ** 0.5
                
                if abs (tick.trade_price - avg_price) > 5 * std_price:
                    warnings.append (f"Price spike: {tick.trade_price} vs avg {avg_price:.2f}")
            
            history.append (tick.trade_price)
            if len (history) > 100:
                history.pop(0)
        
        # Check 3: Tick rate monitoring
        self.tick_counts[symbol] += 1
        
        return warnings
    
    def get_stats (self) -> dict:
        """Get monitoring statistics"""
        return {
            'symbols': len (self.last_tick_time),
            'total_ticks': sum (self.tick_counts.values()),
            'ticks_per_symbol': dict (self.tick_counts)
        }
\`\`\`

---

## Historical Replay

\`\`\`python
"""
Replay historical data for backtesting
Simulate real-time feed from historical data
"""

import pandas as pd
from typing import Iterator

class HistoricalReplay:
    """
    Replay historical tick data
    Simulates real-time feed for backtesting
    """
    
    def __init__(self, db: TickDatabase):
        self.db = db
    
    def replay(
        self, 
        symbols: list[str],
        start_time: datetime,
        end_time: datetime,
        speed: float = 1.0  # 1.0 = real-time, 10.0 = 10x faster
    ) -> Iterator[Tick]:
        """
        Replay ticks in chronological order
        Yields ticks with appropriate timing
        """
        # Fetch all ticks
        all_ticks = []
        for symbol in symbols:
            ticks = self.db.query_trades (symbol, start_time, end_time)
            for timestamp, price, size, side in ticks:
                all_ticks.append(Tick(
                    symbol=symbol,
                    timestamp=int (timestamp.timestamp() * 1_000_000),
                    message_type=MessageType.TRADE,
                    trade_price=price,
                    trade_size=size,
                    trade_side=side,
                    exchange="HISTORICAL"
                ))
        
        # Sort by timestamp
        all_ticks.sort (key=lambda t: t.timestamp)
        
        # Replay with timing
        start_replay_time = datetime.now().timestamp() * 1_000_000
        start_data_time = all_ticks[0].timestamp if all_ticks else 0
        
        for tick in all_ticks:
            # Calculate sleep time
            data_elapsed = tick.timestamp - start_data_time
            replay_elapsed = datetime.now().timestamp() * 1_000_000 - start_replay_time
            
            sleep_time = (data_elapsed - replay_elapsed) / speed
            if sleep_time > 0:
                import time
                time.sleep (sleep_time / 1_000_000)  # Convert to seconds
            
            yield tick

# Usage in backtest
def backtest_with_replay():
    db = TickDatabase("postgresql://localhost/marketdata")
    replay = HistoricalReplay (db)
    
    strategy = MyTradingStrategy()
    
    for tick in replay.replay(
        symbols=['AAPL', 'TSLA'],
        start_time=datetime(2024, 1, 1),
        end_time=datetime(2024, 1, 31),
        speed=1000.0  # 1000x faster
    ):
        strategy.on_tick (tick)
\`\`\`

---

## Production Best Practices

### 1. Data Redundancy

**Multiple data sources**: Never rely on single exchange feed
**Conflict resolution**: Take median price when sources disagree
**Failover**: Automatic switchover to backup feed

### 2. Compression

**Tick data**: Use columnar format (Parquet) with compression
**Typical compression**: 10:1 for tick data, 50:1 for bars
**S3 storage**: $0.023/GB/month vs $100/GB/month for RAM

### 3. Data Validation

\`\`\`python
# Validate all ticks
assert 0 < tick.trade_price < 1_000_000  # Reasonable price
assert tick.trade_size > 0  # Positive size
assert tick.timestamp > 0  # Valid timestamp
assert tick.symbol.isalnum()  # Valid symbol
\`\`\`

### 4. Monitoring

\`\`\`
Metrics to track:
- Tick ingestion rate (msgs/sec)
- Tick-to-database latency
- Data gaps (per symbol)
- Storage growth rate (GB/day)
- Query latency (p50, p95, p99)
\`\`\`

---

## Summary

A production market data system requires:

1. **High throughput**: 1M+ msgs/sec ingestion
2. **Low latency**: <1ms tick-to-strategy
3. **Tiered storage**: Hot (TimescaleDB), Warm (ClickHouse), Cold (S3/Parquet)
4. **Data quality**: Gap detection, spike detection, validation
5. **Multi-source**: Aggregate multiple exchanges
6. **Efficient queries**: ClickHouse for analytics, indexes for point queries
7. **Historical replay**: Simulate real-time for backtesting

Cost optimization:
- **Hot data** (7 days): $1000/month (in-memory)
- **Warm data** (90 days): $500/month (SSD)
- **Cold data** (3 years): $50/month (S3)

In the next section, we'll design backtesting engines that can replay this historical data accurately.
`,
};
