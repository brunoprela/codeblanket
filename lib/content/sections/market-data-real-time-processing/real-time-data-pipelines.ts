export const realTimeDataPipelines = {
  title: 'Real-Time Data Pipelines',
  id: 'real-time-data-pipelines',
  content: `
# Real-Time Data Pipelines

## Introduction

Real-time data pipelines are the backbone of algorithmic trading systems. They ingest, process, and distribute market data at microsecond latencies to thousands of strategies simultaneously.

**Pipeline Components:**
- **Ingestion**: Receive data from exchanges/vendors (WebSocket, FIX, UDP multicast)
- **Processing**: Parse, validate, normalize, enrich
- **Distribution**: Publish to strategies (pub/sub, message queues)
- **Storage**: Persist for compliance and backtesting

**Production Systems:**
- **Trading Firms**: Process 100K-1M msgs/sec per symbol
- **Exchanges**: Handle 10M+ orders/sec (NASDAQ peak)
- **Data Vendors**: Distribute to 1000s of clients simultaneously
- **Latency Targets**: Exchange→Strategy in < 1ms (HFT), < 100ms (retail)

This section covers architecture patterns, message queues (Kafka, Redis), processing frameworks, and building production pipelines.

---

## Pipeline Architecture Patterns

### Pattern 1: Simple Pipeline (WebSocket → Strategy)

\`\`\`python
import asyncio
from typing import Callable

class SimplePipeline:
    """Direct WebSocket to strategy - lowest latency"""
    
    def __init__(self, strategy: Callable):
        self.strategy = strategy
    
    async def run(self):
        async with websocket_connect() as ws:
            async for message in ws:
                # Parse
                data = parse(message)
                
                # Execute strategy directly
                await self.strategy(data)

# Pros: Lowest latency (< 1ms), simple
# Cons: No persistence, no replay, single strategy only
\`\`\`

### Pattern 2: Pub/Sub Pipeline (Redis)

\`\`\`python
import redis.asyncio as redis
import asyncio
import json
from datetime import datetime

class RedisPubSubPipeline:
    """Redis pub/sub for multi-strategy distribution"""
    
    def __init__(self, redis_url: str = "redis://localhost"):
        self.redis = redis.from_url(redis_url)
        self.pubsub = self.redis.pubsub()
    
    async def publish_quote(self, symbol: str, quote: dict):
        """Publish quote to Redis channel"""
        channel = f"quotes:{symbol}"
        message = json.dumps({
            'symbol': symbol,
            'bid': str(quote['bid']),
            'ask': str(quote['ask']),
            'timestamp': quote['timestamp'].isoformat()
        })
        
        await self.redis.publish(channel, message)
    
    async def subscribe_quotes(self, symbols: list, callback: Callable):
        """Subscribe to quotes"""
        channels = [f"quotes:{s}" for s in symbols]
        await self.pubsub.subscribe(*channels)
        
        async for message in self.pubsub.listen():
            if message['type'] == 'message':
                data = json.loads(message['data'])
                await callback(data)

# Usage
pipeline = RedisPubSubPipeline()

# Publisher (data ingestion)
await pipeline.publish_quote("AAPL", {
    'bid': Decimal('150.00'),
    'ask': Decimal('150.02'),
    'timestamp': datetime.now()
})

# Subscriber (strategy)
async def strategy_handler(quote):
    print(f"Strategy received: {quote}")

await pipeline.subscribe_quotes(['AAPL', 'GOOGL'], strategy_handler)

# Pros: Multi-strategy, low latency (< 5ms), simple
# Cons: No persistence (messages lost if no subscriber), no replay
\`\`\`

### Pattern 3: Kafka Pipeline (Production)

\`\`\`python
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
import asyncio
import json
from datetime import datetime
from decimal import Decimal

class KafkaPipeline:
    """Production Kafka pipeline with persistence"""
    
    def __init__(self, bootstrap_servers: str = "localhost:9092"):
        self.bootstrap_servers = bootstrap_servers
        self.producer = None
        self.consumers = {}
    
    async def start_producer(self):
        """Initialize Kafka producer"""
        self.producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            compression_type='lz4',  # Compress messages
            linger_ms=0,  # Send immediately (low latency)
            acks=1  # Wait for leader acknowledgment
        )
        await self.producer.start()
    
    async def publish_quote(self, symbol: str, quote: dict):
        """Publish quote to Kafka topic"""
        topic = f"market-data-quotes"
        
        message = {
            'symbol': symbol,
            'bid': str(quote['bid']),
            'ask': str(quote['ask']),
            'timestamp': quote['timestamp'].isoformat(),
            'source': 'iex-cloud'
        }
        
        # Use symbol as partition key (all AAPL quotes go to same partition)
        await self.producer.send_and_wait(
            topic,
            value=message,
            key=symbol.encode('utf-8')
        )
    
    async def consume_quotes(self, group_id: str, callback: Callable):
        """Consume quotes from Kafka"""
        consumer = AIOKafkaConsumer(
            'market-data-quotes',
            bootstrap_servers=self.bootstrap_servers,
            group_id=group_id,  # Consumer group for load balancing
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest',  # Start from latest (or 'earliest' for replay)
            enable_auto_commit=True
        )
        
        await consumer.start()
        
        try:
            async for message in consumer:
                quote = message.value
                await callback(quote)
        finally:
            await consumer.stop()
    
    async def replay_historical(self, start_time: datetime, end_time: datetime):
        """Replay historical data from Kafka"""
        # Kafka retains messages (default 7 days, configurable to months/years)
        consumer = AIOKafkaConsumer(
            'market-data-quotes',
            bootstrap_servers=self.bootstrap_servers,
            auto_offset_reset='earliest',  # Start from beginning
            enable_auto_commit=False
        )
        
        await consumer.start()
        
        messages = []
        async for message in consumer:
            quote = json.loads(message.value.decode('utf-8'))
            timestamp = datetime.fromisoformat(quote['timestamp'])
            
            if start_time <= timestamp <= end_time:
                messages.append(quote)
            elif timestamp > end_time:
                break
        
        await consumer.stop()
        return messages

# Usage
pipeline = KafkaPipeline()
await pipeline.start_producer()

# Publish 1000 quotes/sec
for i in range(1000):
    await pipeline.publish_quote("AAPL", {
        'bid': Decimal('150.00'),
        'ask': Decimal('150.02'),
        'timestamp': datetime.now()
    })

# Consume with strategy
async def strategy_handler(quote):
    print(f"Strategy: {quote['symbol']} @ {quote['bid']}")

await pipeline.consume_quotes("strategy-group-1", strategy_handler)

# Replay backtest
historical = await pipeline.replay_historical(
    datetime(2024, 1, 1),
    datetime(2024, 1, 31)
)
print(f"Replayed {len(historical)} historical quotes")
\`\`\`

---

## Performance Benchmarks

| Pipeline | Latency | Throughput | Persistence | Cost |
|----------|---------|------------|-------------|------|
| Direct WS | < 1ms | 100K msg/s | None | $0 |
| Redis Pub/Sub | 1-5ms | 1M msg/s | None | $50/mo |
| Kafka | 5-20ms | 10M msg/s | Days/Years | $200/mo |
| AWS Kinesis | 50-200ms | 1M msg/s | 24 hours | $500/mo |

**Recommendation**: Use Kafka for production (persistence + replay + scale).

---

## Best Practices

1. **Partition by symbol** - Ensures ordering per symbol
2. **Monitor lag** - Consumer lag indicates bottlenecks
3. **Compress messages** - LZ4 reduces bandwidth 50-70%
4. **Set retention** - Kafka: 7-30 days (compliance)
5. **Use consumer groups** - Scale consumers horizontally
6. **Implement backpressure** - Don't overwhelm slow consumers

Now you can build production-grade real-time data pipelines!
\`,
};
