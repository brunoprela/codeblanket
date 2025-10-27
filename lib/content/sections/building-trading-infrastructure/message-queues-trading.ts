export const messageQueuesTrading = {
  title: 'Message Queues for Trading',
  id: 'message-queues-trading',
  content: `
# Message Queues for Trading

## Introduction

**Message queues** enable async communication between trading system components. They're the nervous system of modern trading infrastructure.

**Why Message Queues Matter:**
- **Decoupling**: OMS ↔ EMS ↔ Risk System communicate independently
- **Reliability**: Queue persists messages if consumer is down
- **Scalability**: Add consumers to handle load spikes
- **Audit trail**: Every message logged for compliance
- **Load leveling**: Handle bursts of market data without losing messages

**Real-World Usage:**
- **Citadel Securities**: Custom message bus + Kafka for market data (~26% of US equity volume)
- **Jane Street**: ZeroMQ for ultra-low latency IPC between strategies
- **Interactive Brokers**: RabbitMQ for order routing across 135 venues
- **Bloomberg**: Custom pub/sub for enterprise-wide market data distribution

**Message Queue Selection by Use Case:**

| Use Case | Queue | Latency | Throughput | Persistence |
|----------|-------|---------|------------|-------------|
| Market Data | Kafka | ~5ms | Millions/sec | Yes |
| Order Routing | RabbitMQ | ~1ms | 100K/sec | Yes |
| IPC (Strategy ↔ Execution) | ZeroMQ | <100μs | Millions/sec | No |
| Real-time Positions | Redis Streams | <1ms | 1M/sec | Optional |

This section covers production message queue patterns for trading systems.

---

## Kafka for Market Data Distribution

\`\`\`python
"""
Kafka for High-Throughput Market Data

Benefits:
- Handles millions of ticks per second
- Persistent log (replay data for backtesting)
- Partitioning for parallelism
- Consumer groups for scaling

Drawbacks:
- Higher latency (~5ms) vs ZeroMQ
- More complex setup
"""

from kafka import KafkaProducer, KafkaConsumer, TopicPartition
from kafka.admin import KafkaAdminClient, NewTopic
import json
from datetime import datetime
from typing import Dict, List
from decimal import Decimal
import asyncio

class MarketDataKafkaProducer:
    """
    Publish market data to Kafka
    
    Design:
    - Topic per asset class (equities, options, futures)
    - Partitioned by symbol for parallelism
    - Key by symbol for ordering guarantees
    """
    
    def __init__(self, bootstrap_servers: List[str] = ['localhost:9092']):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            compression_type='snappy',  # Fast compression
            linger_ms=10,  # Batch for 10ms for throughput
            acks='all',  # Wait for all replicas (reliability)
            retries=3
        )
        
        print("[Kafka Producer] Connected to Kafka")
    
    def publish_quote(
        self,
        symbol: str,
        bid_price: Decimal,
        bid_size: int,
        ask_price: Decimal,
        ask_size: int,
        exchange: str
    ):
        """Publish quote update"""
        
        message = {
            'type': 'QUOTE',
            'symbol': symbol,
            'bid_price': float(bid_price),
            'bid_size': bid_size,
            'ask_price': float(ask_price),
            'ask_size': ask_size,
            'exchange': exchange,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Send to topic, keyed by symbol (ensures order)
        future = self.producer.send(
            'market-data-quotes',
            key=symbol,
            value=message
        )
        
        # Non-blocking: returns immediately
        return future
    
    def publish_trade(
        self,
        symbol: str,
        price: Decimal,
        quantity: int,
        side: str,
        exchange: str
    ):
        """Publish trade execution"""
        
        message = {
            'type': 'TRADE',
            'symbol': symbol,
            'price': float(price),
            'quantity': quantity,
            'side': side,
            'exchange': exchange,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        future = self.producer.send(
            'market-data-trades',
            key=symbol,
            value=message
        )
        
        return future
    
    def flush(self):
        """Flush pending messages"""
        self.producer.flush()
    
    def close(self):
        """Close producer"""
        self.producer.close()


class MarketDataKafkaConsumer:
    """
    Consume market data from Kafka
    
    Features:
    - Consumer groups for scaling
    - Offset management for replay
    - Batch processing for efficiency
    """
    
    def __init__(
        self,
        bootstrap_servers: List[str] = ['localhost:9092'],
        group_id: str = 'strategy-group',
        topics: List[str] = None
    ):
        if topics is None:
            topics = ['market-data-quotes', 'market-data-trades']
        
        self.consumer = KafkaConsumer(
            *topics,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest',  # Start from latest message
            enable_auto_commit=True,
            max_poll_records=1000  # Batch size
        )
        
        print(f"[Kafka Consumer] Subscribed to {topics}")
    
    def consume_batch(self, timeout_ms: int = 1000) -> List[Dict]:
        """
        Consume batch of messages
        
        Returns: List of market data messages
        """
        messages = []
        
        records = self.consumer.poll(timeout_ms=timeout_ms)
        
        for topic_partition, partition_records in records.items():
            for record in partition_records:
                messages.append({
                    'topic': record.topic,
                    'partition': record.partition,
                    'offset': record.offset,
                    'key': record.key.decode('utf-8') if record.key else None,
                    'value': record.value,
                    'timestamp': record.timestamp
                })
        
        return messages
    
    def reset_to_timestamp(self, timestamp_ms: int):
        """
        Reset consumer to specific timestamp (for replay)
        
        Useful for backtesting or debugging
        """
        # Get assigned partitions
        partitions = self.consumer.assignment()
        
        # Get offsets for timestamp
        offsets = self.consumer.offsets_for_times({
            p: timestamp_ms for p in partitions
        })
        
        # Seek to offsets
        for partition, offset_and_timestamp in offsets.items():
            if offset_and_timestamp:
                self.consumer.seek(partition, offset_and_timestamp.offset)
        
        print(f"[Kafka Consumer] Reset to timestamp {timestamp_ms}")
    
    def close(self):
        """Close consumer"""
        self.consumer.close()


# Example: Real-time market data pipeline
async def kafka_market_data_example():
    """Demonstrate Kafka market data pipeline"""
    
    producer = MarketDataKafkaProducer()
    consumer = MarketDataKafkaConsumer(group_id='example-strategy')
    
    print("=" * 80)
    print("KAFKA MARKET DATA PIPELINE")
    print("=" * 80)
    
    # Simulate publishing market data
    print("\\nPublishing market data...")
    for i in range(10):
        producer.publish_quote(
            symbol='AAPL',
            bid_price=Decimal('150.00') + Decimal(i * 0.01),
            bid_size=1000,
            ask_price=Decimal('150.01') + Decimal(i * 0.01),
            ask_size=900,
            exchange='NASDAQ'
        )
    
    producer.flush()
    print("Published 10 quotes")
    
    # Consume messages
    print("\\nConsuming market data...")
    await asyncio.sleep(1)  # Wait for messages to propagate
    
    messages = consumer.consume_batch(timeout_ms=2000)
    print(f"\\nReceived {len(messages)} messages:")
    
    for msg in messages[:5]:  # Print first 5
        value = msg['value']
        print(f"  {value['symbol']}: Bid ${value['bid_price']}, Ask ${value['ask_price']}")
    
    consumer.close()
    producer.close()

# asyncio.run(kafka_market_data_example())
\`\`\`

---

## RabbitMQ for Order Routing

\`\`\`python
"""
RabbitMQ for Reliable Order Flow

Benefits:
- Message acknowledgments (at-least-once delivery)
- Flexible routing (exchanges, queues, bindings)
- Dead-letter queues for failed orders
- Priority queues for urgent orders

Drawbacks:
- Higher latency than ZeroMQ (~1ms)
- Single broker (no horizontal scaling like Kafka)
"""

import pika
from pika import BasicProperties
import json
from typing import Callable, Dict
from datetime import datetime
from enum import Enum

class OrderPriority(Enum):
    """Order priority levels"""
    LOW = 1
    NORMAL = 5
    HIGH = 9
    URGENT = 10

class RabbitMQOrderRouter:
    """
    Order routing via RabbitMQ
    
    Architecture:
    - OMS publishes orders to 'orders' exchange
    - EMS consumes from 'orders.pending' queue
    - Risk system consumes from 'orders.risk-check' queue
    - Dead-letter queue for rejected orders
    """
    
    def __init__(self, host: str = 'localhost', port: int = 5672):
        # Connection
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=host,
                port=port,
                heartbeat=600,
                blocked_connection_timeout=300
            )
        )
        
        self.channel = self.connection.channel()
        
        # Declare exchange
        self.channel.exchange_declare(
            exchange='orders',
            exchange_type='topic',
            durable=True
        )
        
        # Declare queues
        self._setup_queues()
        
        print("[RabbitMQ] Connected and queues configured")
    
    def _setup_queues(self):
        """Setup queues and bindings"""
        
        # Main order queue (with priority)
        self.channel.queue_declare(
            queue='orders.pending',
            durable=True,
            arguments={
                'x-max-priority': 10,  # Enable priority
                'x-message-ttl': 60000,  # 60 second TTL
                'x-dead-letter-exchange': 'orders.dead-letter'
            }
        )
        
        # Risk check queue
        self.channel.queue_declare(
            queue='orders.risk-check',
            durable=True
        )
        
        # Dead-letter queue
        self.channel.exchange_declare(
            exchange='orders.dead-letter',
            exchange_type='fanout',
            durable=True
        )
        
        self.channel.queue_declare(
            queue='orders.failed',
            durable=True
        )
        
        self.channel.queue_bind(
            queue='orders.failed',
            exchange='orders.dead-letter'
        )
        
        # Bind queues to exchange
        self.channel.queue_bind(
            queue='orders.pending',
            exchange='orders',
            routing_key='order.new.*'
        )
        
        self.channel.queue_bind(
            queue='orders.risk-check',
            exchange='orders',
            routing_key='order.*.risk-check'
        )
    
    def publish_order(
        self,
        order: Dict,
        priority: OrderPriority = OrderPriority.NORMAL,
        routing_key: str = 'order.new.equity'
    ):
        """
        Publish order to RabbitMQ
        
        Args:
            order: Order dict
            priority: Order priority (affects queue position)
            routing_key: Routing key for exchange
        """
        
        # Add metadata
        order['published_at'] = datetime.utcnow().isoformat()
        
        # Message properties
        properties = BasicProperties(
            delivery_mode=2,  # Persistent
            priority=priority.value,
            content_type='application/json',
            correlation_id=order.get('order_id'),
            timestamp=int(datetime.utcnow().timestamp())
        )
        
        # Publish
        self.channel.basic_publish(
            exchange='orders',
            routing_key=routing_key,
            body=json.dumps(order),
            properties=properties
        )
        
        print(f"[RabbitMQ] Published order: {order.get('order_id')} (priority: {priority.value})")
    
    def consume_orders(
        self,
        queue: str,
        callback: Callable,
        prefetch_count: int = 10
    ):
        """
        Consume orders from queue
        
        Args:
            queue: Queue name
            callback: Callback function (ch, method, properties, body)
            prefetch_count: Max unacknowledged messages
        """
        
        # Set QoS (prefetch)
        self.channel.basic_qos(prefetch_count=prefetch_count)
        
        # Start consuming
        self.channel.basic_consume(
            queue=queue,
            on_message_callback=callback,
            auto_ack=False  # Manual acknowledgment
        )
        
        print(f"[RabbitMQ] Consuming from queue: {queue}")
        self.channel.start_consuming()
    
    def close(self):
        """Close connection"""
        self.connection.close()


class OrderProcessor:
    """Process orders from RabbitMQ"""
    
    def __init__(self, router: RabbitMQOrderRouter):
        self.router = router
        self.processed_count = 0
    
    def process_order(self, ch, method, properties, body):
        """
        Process order callback
        
        Must acknowledge or reject message
        """
        try:
            order = json.loads(body)
            
            print(f"[Processor] Processing order: {order.get('order_id')}")
            
            # Simulate order processing
            # In production: validate, risk check, route to exchange
            
            # Acknowledge message
            ch.basic_ack(delivery_tag=method.delivery_tag)
            
            self.processed_count += 1
            
        except Exception as e:
            print(f"[Processor] Error processing order: {e}")
            
            # Reject message (send to dead-letter queue)
            ch.basic_nack(
                delivery_tag=method.delivery_tag,
                requeue=False  # Don't requeue, send to DLQ
            )


# Example: Order routing pipeline
def rabbitmq_order_example():
    """Demonstrate RabbitMQ order routing"""
    
    router = RabbitMQOrderRouter()
    
    print("=" * 80)
    print("RABBITMQ ORDER ROUTING")
    print("=" * 80)
    
    # Publish orders with different priorities
    print("\\nPublishing orders...")
    
    # Normal priority order
    router.publish_order(
        order={
            'order_id': 'ORD-001',
            'symbol': 'AAPL',
            'side': 'BUY',
            'quantity': 100,
            'price': 150.00
        },
        priority=OrderPriority.NORMAL
    )
    
    # High priority order (market maker)
    router.publish_order(
        order={
            'order_id': 'ORD-002',
            'symbol': 'TSLA',
            'side': 'SELL',
            'quantity': 50,
            'price': 250.00
        },
        priority=OrderPriority.HIGH
    )
    
    # Urgent priority order (risk close-out)
    router.publish_order(
        order={
            'order_id': 'ORD-003',
            'symbol': 'GOOGL',
            'side': 'SELL',
            'quantity': 200,
            'price': 140.00
        },
        priority=OrderPriority.URGENT,
        routing_key='order.new.urgent'
    )
    
    print("\\nOrders published (check with: rabbitmqctl list_queues)")
    
    router.close()

# rabbitmq_order_example()
\`\`\`

---

## ZeroMQ for Ultra-Low Latency IPC

\`\`\`python
"""
ZeroMQ for Ultra-Low Latency Inter-Process Communication

Benefits:
- Ultra-low latency (<100μs)
- No broker (direct process-to-process)
- Multiple patterns (pub-sub, req-rep, push-pull)
- Very simple API

Drawbacks:
- No persistence (messages lost if consumer down)
- No delivery guarantees
- No broker features (routing, queues, etc.)

Use case: Communication between strategy and execution on same machine
"""

import zmq
from typing import Dict
import json
from datetime import datetime
import time

class ZeroMQMarketDataPublisher:
    """
    Publish market data via ZeroMQ PUB-SUB
    
    Latency: <10μs for IPC
    """
    
    def __init__(self, port: int = 5555):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        
        # Bind to port
        self.socket.bind(f"tcp://*:{port}")
        
        # High water mark (max queued messages)
        self.socket.setsockopt(zmq.SNDHWM, 10000)
        
        print(f"[ZeroMQ Publisher] Listening on port {port}")
        
        # Small delay to allow subscribers to connect
        time.sleep(0.1)
    
    def publish_tick(self, symbol: str, price: float, quantity: int):
        """
        Publish tick (ultra-fast)
        
        Message format: "SYMBOL|PRICE|QUANTITY|TIMESTAMP"
        Binary format for speed
        """
        
        timestamp_ns = time.perf_counter_ns()
        
        # Use msgpack or protobuf for production (faster than JSON)
        message = f"{symbol}|{price}|{quantity}|{timestamp_ns}"
        
        # Send with topic prefix (symbol)
        self.socket.send_string(message)
    
    def close(self):
        """Close socket"""
        self.socket.close()
        self.context.term()


class ZeroMQMarketDataSubscriber:
    """
    Subscribe to market data via ZeroMQ
    """
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 5555,
        symbols: list[str] = None
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        
        # Connect to publisher
        self.socket.connect(f"tcp://{host}:{port}")
        
        # High water mark
        self.socket.setsockopt(zmq.RCVHWM, 10000)
        
        # Subscribe to symbols
        if symbols:
            for symbol in symbols:
                self.socket.setsockopt_string(zmq.SUBSCRIBE, symbol)
        else:
            # Subscribe to all
            self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        
        print(f"[ZeroMQ Subscriber] Connected to {host}:{port}")
    
    def receive_tick(self) -> Dict:
        """
        Receive tick (blocking)
        
        Returns: Tick dict
        """
        message = self.socket.recv_string()
        
        # Parse message
        parts = message.split('|')
        
        return {
            'symbol': parts[0],
            'price': float(parts[1]),
            'quantity': int(parts[2]),
            'timestamp_ns': int(parts[3])
        }
    
    def close(self):
        """Close socket"""
        self.socket.close()
        self.context.term()


# Example: Low-latency market data pipeline
def zeromq_example():
    """Demonstrate ZeroMQ ultra-low latency"""
    
    import threading
    
    # Publisher thread
    def publisher_thread():
        pub = ZeroMQMarketDataPublisher(port=5555)
        
        for i in range(100):
            pub.publish_tick('AAPL', 150.00 + i * 0.01, 1000)
            time.sleep(0.001)  # 1ms interval
        
        pub.close()
    
    # Subscriber thread
    def subscriber_thread():
        sub = ZeroMQMarketDataSubscriber(
            host='localhost',
            port=5555,
            symbols=['AAPL']
        )
        
        latencies = []
        
        for _ in range(10):
            tick = sub.receive_tick()
            
            # Calculate latency
            now_ns = time.perf_counter_ns()
            latency_ns = now_ns - tick['timestamp_ns']
            latencies.append(latency_ns)
            
            print(f"  {tick['symbol']}: ${tick['price']}, latency: {latency_ns/1000:.2f} μs")
        
        print(f"\\nAverage latency: {sum(latencies)/len(latencies)/1000:.2f} μs")
        
        sub.close()
    
    print("=" * 80)
    print("ZEROMQ ULTRA-LOW LATENCY IPC")
    print("=" * 80)
    
    # Start threads
    pub_thread = threading.Thread(target=publisher_thread)
    sub_thread = threading.Thread(target=subscriber_thread)
    
    pub_thread.start()
    time.sleep(0.1)  # Let publisher start
    sub_thread.start()
    
    pub_thread.join()
    sub_thread.join()

# zeromq_example()
\`\`\`

---

## Summary

**Message Queue Selection:**

| Criteria | Kafka | RabbitMQ | ZeroMQ |
|----------|-------|----------|--------|
| **Latency** | ~5ms | ~1ms | <100μs |
| **Throughput** | Millions/sec | 100K/sec | Millions/sec |
| **Persistence** | Yes | Yes | No |
| **Delivery Guarantees** | At-least-once | At-least-once | None |
| **Scalability** | Excellent | Good | N/A (no broker) |
| **Use Case** | Market data archive | Order routing | Strategy ↔ Execution IPC |

**Production Recommendations:**
1. **Market Data Distribution**: Kafka (persistence for replay, high throughput)
2. **Order Routing**: RabbitMQ (reliability, acknowledgments, dead-letter queues)
3. **Strategy ↔ Execution IPC**: ZeroMQ (ultra-low latency)
4. **Real-time Positions**: Redis Streams (fast, optional persistence)

**Message Queue Architecture Pattern:**
\`\`\`
Market Data Feed → Kafka → [Strategy 1, Strategy 2, ..., Strategy N]
                             ↓
                    ZeroMQ (IPC)
                             ↓
                    Execution Engine → RabbitMQ → [EMS, Risk, Audit]
                                                        ↓
                                                    Broker/Exchange
\`\`\`

**Next Section**: Module 14.11 - Database Design for Trading Systems
`,
};
