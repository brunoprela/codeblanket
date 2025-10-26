export const messageQueuesTradingMC = [
    {
        id: 'message-queues-trading-mc-1',
        question:
            'Which message queue is BEST for distributing 10 million market data ticks per second?',
        options: [
            'RabbitMQ',
            'Kafka',
            'Redis Pub/Sub',
            'PostgreSQL LISTEN/NOTIFY',
        ],
        correctAnswer: 1,
        explanation:
            'Answer: Kafka.\n\n' +
            'Throughput comparison:\n' +
            '- **Kafka**: 10M+ msgs/sec (designed for high throughput)\n' +
            '- **RabbitMQ**: 100K-1M msgs/sec (moderate throughput)\n' +
            '- **Redis Pub/Sub**: 1M msgs/sec (good, but no persistence)\n' +
            '- **PostgreSQL**: 10K msgs/sec (not designed for this)\n\n' +
            'Kafka advantages for market data:\n' +
            '1. Partitioning: Distribute load across brokers\n' +
            '2. Persistence: Replay data for backtesting\n' +
            '3. Consumer groups: Multiple strategies consume independently\n\n' +
            'Real-world: Bloomberg, Reuters, and exchanges use Kafka-like systems for market data.',
    },
    {
        id: 'message-queues-trading-mc-2',
        question:
            'What does "message acknowledgment" (ACK) mean in RabbitMQ?',
        options: [
            'The producer confirms the message was sent',
            'The consumer confirms the message was successfully processed',
            'The broker confirms the message was stored',
            'The network confirms the message was delivered',
        ],
        correctAnswer: 1,
        explanation:
            'Answer: The consumer confirms the message was successfully processed.\n\n' +
            'RabbitMQ ACK flow:\n' +
            '1. Producer sends message to queue\n' +
            '2. Broker stores message\n' +
            '3. Consumer receives message\n' +
            '4. Consumer processes message\n' +
            '5. **Consumer sends ACK** to broker\n' +
            '6. Broker deletes message (it was processed)\n\n' +
            'If consumer crashes before ACK:\n' +
            '- Broker redelivers message to another consumer\n' +
            '- Ensures no orders are lost\n\n' +
            'Why important for trading:\n' +
            '- Order processing must be reliable\n' +
            '- Cannot lose orders due to consumer crashes\n' +
            '- ACK guarantees at-least-once processing',
    },
    {
        id: 'message-queues-trading-mc-3',
        question:
            'Your order queue latency is 50ms. What is the MOST likely cause?',
        options: [
            'Network congestion',
            'Message batching (waiting to fill a batch before sending)',
            'Slow consumer processing',
            'Broker disk writes',
        ],
        correctAnswer: 1,
        explanation:
            'Answer: Message batching.\n\n' +
            'Kafka batching example:\n' +
            '- Producer batches messages every 10-100ms for throughput\n' +
            '- This adds 10-100ms latency\n' +
            '- Trade-off: Higher throughput vs higher latency\n\n' +
            'Solution for low latency:\n' +
            '```python\n' +
            'producer = KafkaProducer(\n' +
            '    linger_ms=0,  # Send immediately (no batching)\n' +
            '    batch_size=1  # One message per batch\n' +
            ')\n' +
            '```\n' +
            'Result: Latency drops to 1-5ms.\n\n' +
            'Trade-off: Lower throughput (10K msgs/sec vs 1M msgs/sec with batching).',
    },
    {
        id: 'message-queues-trading-mc-4',
        question:
            'What is the advantage of ZeroMQ over Kafka for trading systems?',
        options: [
            'Higher throughput',
            'Better persistence',
            'Lower latency (<10μs)',
            'Easier to use',
        ],
        correctAnswer: 2,
        explanation:
            'Answer: Lower latency (<10μs).\n\n' +
            'Latency comparison:\n' +
            '- **ZeroMQ**: <10μs (in-memory, no broker)\n' +
            '- **Kafka**: 5-10ms (broker, disk writes, batching)\n\n' +
            'Why ZeroMQ is faster:\n' +
            '1. No broker (direct process-to-process)\n' +
            '2. No disk writes (pure in-memory)\n' +
            '3. No serialization overhead (send raw bytes)\n\n' +
            'Trade-offs:\n' +
            '- No persistence (messages lost if consumer down)\n' +
            '- No consumer groups (must implement manually)\n' +
            '- No scaling (limited to single machine)\n\n' +
            'Use case: HFT systems where 10μs matters more than reliability.',
    },
    {
        id: 'message-queues-trading-mc-5',
        question:
            'Your trading system crashed and restarted. Which message queue allows you to replay the last hour of market data?',
        options: [
            'RabbitMQ',
            'Kafka',
            'ZeroMQ',
            'Redis Pub/Sub',
        ],
        correctAnswer: 1,
        explanation:
            'Answer: Kafka.\n\n' +
            'Kafka persistent log:\n' +
            '- All messages stored on disk (retained for days/weeks)\n' +
            '- Each consumer tracks offset (position in log)\n' +
            '- Can rewind offset to replay messages\n\n' +
            'Replay example:\n' +
            '```python\n' +
            '# Rewind to 1 hour ago\n' +
            'consumer.seek_to_time(\n' +
            '    timestamp=datetime.now() - timedelta(hours=1)\n' +
            ')\n' +
            '# Replay all messages from 1 hour ago to now\n' +
            'for message in consumer:\n' +
            '    process(message)\n' +
            '```\n\n' +
            'Why others don\'t support replay:\n' +
            '- **RabbitMQ**: Deletes messages after ACK\n' +
            '- **ZeroMQ**: No persistence (in-memory only)\n' +
            '- **Redis Pub/Sub**: Fire-and-forget (no history)\n\n' +
            'Real-world use case: Restart strategy after crash, replay market data to restore state.',
    },
];

