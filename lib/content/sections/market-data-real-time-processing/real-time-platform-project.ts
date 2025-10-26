export const realTimePlatformProject = {
  title: 'Project: Real-Time Market Data Platform',
  id: 'real-time-platform-project',
  content: `
# Project: Real-Time Market Data Platform

## Introduction

This capstone project brings together all concepts from Module 13 into a production-grade real-time market data platform. You'll build a complete system that ingests data from multiple vendors, normalizes it, distributes via Kafka, stores in TimescaleDB, validates quality, and monitors performance - exactly like systems used by professional trading firms.

**Project Scope:**
- **Data Ingestion**: WebSocket feeds from IEX Cloud and Polygon.io
- **Normalization**: Unified quote format across exchanges
- **Distribution**: Kafka pipeline (5K messages/second)
- **Storage**: TimescaleDB with compression (1 year retention)
- **Validation**: Real-time quality checks and anomaly detection
- **Monitoring**: Prometheus metrics + Grafana dashboards

**Technical Requirements:**
- Support 500 symbols (S&P 500 constituents)
- Process 50K ticks/second aggregate
- End-to-end latency < 10ms (ingestion → Kafka → consumer)
- 99.9% uptime (8.76 hours downtime/year max)
- 1-year historical storage (~50 billion ticks)

**Real-World Comparable Systems:**
- **Bloomberg Terminal**: Market data for 300+ exchanges
- **Refinitiv**: 8M+ instruments, sub-second updates
- **Trading Firms**: Internal platforms processing 1M+ msgs/sec

By completing this project, you'll demonstrate mastery of production market data systems.

---

## Architecture Overview

### System Components

\`\`\`
┌─────────────────────────────────────────────────────────────┐
│                   DATA INGESTION LAYER                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │ IEX Cloud│  │Polygon.io│  │ Future   │                  │
│  │WebSocket │  │WebSocket │  │ Vendors  │                  │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘                  │
│        │             │             │                         │
│        └─────────────┼─────────────┘                         │
│                      │                                       │
│                      ▼                                       │
│             ┌────────────────┐                               │
│             │  Normalizer    │  (Unified Quote Format)      │
│             └────────┬───────┘                               │
│                      │                                       │
│                      ▼                                       │
│             ┌────────────────┐                               │
│             │   Validator    │  (Quality Checks)            │
│             └────────┬───────┘                               │
└──────────────────────┼───────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                 DISTRIBUTION LAYER (KAFKA)                   │
│  Topic: market-data-quotes                                  │
│  Partitions: 50 (by symbol hash)                            │
│  Replication: 3 (high availability)                          │
└─────────┬───────────────────────────────────────────┬───────┘
          │                                           │
          ▼                                           ▼
┌──────────────────┐                        ┌─────────────────┐
│  STORAGE LAYER   │                        │ STRATEGY LAYER  │
│  (TimescaleDB)   │                        │  (Consumers)    │
│                  │                        │                 │
│  - Tick Storage  │                        │ - Strategy 1    │
│  - Compression   │                        │ - Strategy 2    │
│  - 1 Year Data   │                        │ - Strategy N    │
└──────────────────┘                        └─────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│              MONITORING & ALERTING                           │
│  Prometheus (metrics) + Grafana (dashboards)                │
│  - Latency tracking                                          │
│  - Throughput monitoring                                     │
│  - Error rate alerts                                         │
│  - Data quality scores                                       │
└─────────────────────────────────────────────────────────────┘
\`\`\`

---

## Implementation Guide

### Phase 1: Infrastructure Setup (Week 1)

**1.1 Kafka Setup**

\`\`\`bash
# Docker Compose for Kafka cluster
version: '3'
services:
  zookeeper:
    image: confluentinc/cp-zookeeper:7.5.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    ports:
      - "2181:2181"
  
  kafka:
    image: confluentinc/cp-kafka:7.5.0
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1

# Start: docker-compose up -d
\`\`\`

**1.2 TimescaleDB Setup**

\`\`\`bash
# Docker for TimescaleDB
docker run -d \\
  --name timescaledb \\
  -p 5432:5432 \\
  -e POSTGRES_PASSWORD=password \\
  timescale/timescaledb:latest-pg15

# Create database and schema
psql -h localhost -U postgres -c "CREATE DATABASE market_data;"
psql -h localhost -U postgres -d market_data < schema.sql
\`\`\`

\`\`\`sql
-- schema.sql
CREATE TABLE ticks (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    exchange TEXT,
    bid_price DECIMAL(12, 4),
    ask_price DECIMAL(12, 4),
    bid_size INTEGER,
    ask_size INTEGER,
    conditions TEXT
);

-- Convert to hypertable (time-series optimized)
SELECT create_hypertable('ticks', 'time', chunk_time_interval => INTERVAL '1 day');

-- Create indexes
CREATE INDEX idx_symbol_time ON ticks (symbol, time DESC);

-- Enable compression
ALTER TABLE ticks SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);

-- Auto-compress data older than 7 days
SELECT add_compression_policy('ticks', INTERVAL '7 days');
\`\`\`

**1.3 Redis Setup (optional, for caching)**

\`\`\`bash
docker run -d --name redis -p 6379:6379 redis:7-alpine
\`\`\`

---

### Phase 2: Data Ingestion (Week 2)

**2.1 WebSocket Connectors**

\`\`\`python
# ingestion/iex_connector.py
import asyncio
import websockets
import json
from typing import Callable

class IEXConnector:
    """IEX Cloud WebSocket connector"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = f"wss://ws.iexcloud.io/stable?token={api_key}"
        self.callbacks = []
    
    def register_callback(self, callback: Callable):
        self.callbacks.append(callback)
    
    async def connect(self, symbols: list):
        async with websockets.connect(self.url) as ws:
            # Subscribe to symbols
            subscribe_msg = json.dumps({
                "action": "subscribe",
                "symbols": symbols,
                "channels": ["quotes"]
            })
            await ws.send(subscribe_msg)
            
            # Listen for updates
            async for message in ws:
                data = json.loads(message)
                
                for callback in self.callbacks:
                    await callback(data)

# ingestion/polygon_connector.py
from polygon import WebSocketClient

class PolygonConnector:
    """Polygon.io WebSocket connector"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = WebSocketClient(api_key, "stocks")
    
    async def connect(self, symbols: list):
        # Subscribe and handle messages
        self.client.run(self._handle_message)
    
    def _handle_message(self, msgs):
        for msg in msgs:
            # Process message
            pass
\`\`\`

**2.2 Data Normalizer**

\`\`\`python
# normalization/normalizer.py
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

@dataclass
class NormalizedQuote:
    symbol: str
    timestamp: datetime
    exchange: str
    bid_price: Decimal
    ask_price: Decimal
    bid_size: int
    ask_size: int

class DataNormalizer:
    """Normalize quotes from different vendors"""
    
    def normalize_iex(self, raw: dict) -> NormalizedQuote:
        return NormalizedQuote(
            symbol=raw['symbol'],
            timestamp=datetime.fromtimestamp(raw['timestamp'] / 1000),
            exchange='IEX',
            bid_price=Decimal(str(raw['bidPrice'])),
            ask_price=Decimal(str(raw['askPrice'])),
            bid_size=raw['bidSize'],
            ask_size=raw['askSize']
        )
    
    def normalize_polygon(self, raw: dict) -> NormalizedQuote:
        return NormalizedQuote(
            symbol=raw['sym'],
            timestamp=datetime.fromtimestamp(raw['t'] / 1000000000),
            exchange='POLYGON',
            bid_price=Decimal(str(raw['bp'])),
            ask_price=Decimal(str(raw['ap'])),
            bid_size=raw['bs'],
            ask_size=raw['as']
        )
\`\`\`

---

### Phase 3: Kafka Pipeline (Week 3)

**3.1 Producer**

\`\`\`python
# pipeline/kafka_producer.py
from aiokafka import AIOKafkaProducer
import json
import asyncio

class MarketDataProducer:
    """Kafka producer for market data"""
    
    def __init__(self, bootstrap_servers: str = "localhost:9092"):
        self.producer = AIOKafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            compression_type='lz4',
            linger_ms=1,  # Low latency
            acks=1  # Wait for leader only
        )
    
    async def start(self):
        await self.producer.start()
    
    async def publish_quote(self, quote: NormalizedQuote):
        """Publish quote to Kafka"""
        message = {
            'symbol': quote.symbol,
            'timestamp': quote.timestamp.isoformat(),
            'exchange': quote.exchange,
            'bid_price': str(quote.bid_price),
            'ask_price': str(quote.ask_price),
            'bid_size': quote.bid_size,
            'ask_size': quote.ask_size
        }
        
        await self.producer.send(
            'market-data-quotes',
            value=message,
            key=quote.symbol.encode('utf-8')  # Partition by symbol
        )
    
    async def stop(self):
        await self.producer.stop()
\`\`\`

**3.2 Consumer**

\`\`\`python
# pipeline/kafka_consumer.py
from aiokafka import AIOKafkaConsumer
import json

class MarketDataConsumer:
    """Kafka consumer for strategies"""
    
    def __init__(self, group_id: str, bootstrap_servers: str = "localhost:9092"):
        self.consumer = AIOKafkaConsumer(
            'market-data-quotes',
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest'
        )
    
    async def start(self):
        await self.consumer.start()
    
    async def consume(self):
        """Consume messages"""
        async for message in self.consumer:
            quote = message.value
            yield quote
    
    async def stop(self):
        await self.consumer.stop()
\`\`\`

---

### Phase 4: Storage Layer (Week 4)

\`\`\`python
# storage/timescale_writer.py
import psycopg2
from psycopg2.extras import execute_batch

class TimescaleWriter:
    """Write ticks to TimescaleDB"""
    
    def __init__(self, conn_string: str):
        self.conn = psycopg2.connect(conn_string)
        self.batch = []
        self.batch_size = 1000
    
    def write_tick(self, quote: NormalizedQuote):
        """Add tick to batch"""
        self.batch.append((
            quote.timestamp,
            quote.symbol,
            quote.exchange,
            quote.bid_price,
            quote.ask_price,
            quote.bid_size,
            quote.ask_size
        ))
        
        if len(self.batch) >= self.batch_size:
            self.flush()
    
    def flush(self):
        """Write batch to database"""
        if not self.batch:
            return
        
        with self.conn.cursor() as cur:
            execute_batch(cur, """
                INSERT INTO ticks (time, symbol, exchange, 
                                  bid_price, ask_price, bid_size, ask_size)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, self.batch)
        
        self.conn.commit()
        self.batch = []
\`\`\`

---

### Phase 5: Monitoring (Week 5)

**5.1 Prometheus Metrics**

\`\`\`python
# monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Metrics
quotes_received = Counter('quotes_received_total', 'Total quotes received', ['symbol'])
quotes_validated = Counter('quotes_validated_total', 'Validated quotes', ['valid'])
processing_latency = Histogram('processing_latency_seconds', 'Processing latency')
kafka_lag = Gauge('kafka_consumer_lag', 'Consumer lag', ['group'])

# Start metrics server
start_http_server(8000)
\`\`\`

**5.2 Grafana Dashboard**

\`\`\`json
{
  "dashboard": {
    "title": "Market Data Platform",
    "panels": [
      {
        "title": "Throughput",
        "targets": [
          {
            "expr": "rate(quotes_received_total[1m])"
          }
        ]
      },
      {
        "title": "Latency P99",
        "targets": [
          {
            "expr": "histogram_quantile(0.99, processing_latency_seconds)"
          }
        ]
      }
    ]
  }
}
\`\`\`

---

## Testing & Deployment

### Load Testing

\`\`\`python
# tests/load_test.py
import asyncio
import time

async def load_test():
    """Simulate 50K ticks/second"""
    producer = MarketDataProducer()
    await producer.start()
    
    start = time.time()
    count = 0
    
    for _ in range(50000):
        await producer.publish_quote(generate_random_quote())
        count += 1
    
    elapsed = time.time() - start
    print(f"Published {count} quotes in {elapsed:.2f}s")
    print(f"Throughput: {count/elapsed:.0f} quotes/sec")

# Run: python -m tests.load_test
\`\`\`

### Docker Deployment

\`\`\`dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "main.py"]
\`\`\`

---

## Success Criteria

✅ **Functional Requirements:**
- [ ] Ingest data from 2+ vendors
- [ ] Support 500 symbols
- [ ] Process 50K ticks/second
- [ ] Store 1 year of data

✅ **Performance Requirements:**
- [ ] Latency < 10ms (P99)
- [ ] Throughput > 50K ticks/sec
- [ ] Data quality score > 95%

✅ **Operational Requirements:**
- [ ] 99.9% uptime
- [ ] Automated monitoring
- [ ] Alerting on errors

**Deliverables:**
1. Complete source code (GitHub repo)
2. Docker Compose deployment
3. Grafana dashboards
4. Load test results
5. Architecture documentation

Congratulations! You've built a production-grade market data platform! 🎉
`,
};
