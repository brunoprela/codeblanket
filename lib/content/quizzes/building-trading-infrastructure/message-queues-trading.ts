export const messageQueuesTradingQuiz = [
    {
        id: 'message-queues-trading-q-1',
        question:
            'Your trading system needs to distribute 10 million market data ticks per second to 50 strategy processes. Should you use Kafka, RabbitMQ, or ZeroMQ? Explain your choice.',
        sampleAnswer:
            'Use Kafka for High-Throughput Market Data:\n\n' +
            'Kafka is best for 10M ticks/sec because:\n' +
            '1. **High throughput**: Handles millions of messages/sec\n' +
            '2. **Persistent log**: Can replay market data for backtesting\n' +
            '3. **Partitioning**: Distribute load across multiple brokers\n' +
            '4. **Consumer groups**: 50 strategies can consume independently\n\n' +
            'Architecture:\n' +
            '- Topic: market-data (partitioned by symbol)\n' +
            '- Producers: Market data feeds (1-10 producers)\n' +
            '- Consumers: 50 strategy processes (consumer group)\n' +
            '- Throughput: 10M msgs/sec (100K msgs/sec per partition × 100 partitions)\n\n' +
            'Why not RabbitMQ: Lower throughput (~100K msgs/sec), not designed for high volume.\n' +
            'Why not ZeroMQ: No persistence, no consumer groups (would need custom logic).',
        keyPoints: [
            'Kafka best for 10M ticks/sec: High throughput, persistent log for replay, partitioning for scale',
            'Architecture: market-data topic partitioned by symbol, 100 partitions × 100K msgs/sec = 10M msgs/sec',
            'Consumer groups: 50 strategy processes consume independently without coordination',
            'RabbitMQ: Not suitable (lower throughput ~100K msgs/sec, designed for order routing not market data)',
            'ZeroMQ: Not suitable (no persistence, no consumer groups, would require custom coordination)',
        ],
    },
    {
        id: 'message-queues-trading-q-2',
        question:
            'Compare Kafka vs RabbitMQ for order routing in a trading system. What are the trade-offs?',
        sampleAnswer:
            'Kafka vs RabbitMQ for Order Routing:\n\n' +
            '**RabbitMQ** (better for orders):\n' +
            '- **ACKs**: Explicit acknowledgment ensures order processed\n' +
            '- **Routing**: Flexible exchange types (direct, topic, fanout)\n' +
            '- **Priority**: Priority queues for urgent orders\n' +
            '- **Latency**: Lower latency (~1ms) for small volumes\n\n' +
            '**Kafka** (better for logs/analytics):\n' +
            '- **Throughput**: Higher throughput (millions of orders/sec)\n' +
            '- **Persistence**: Durable log for audit/replay\n' +
            '- **Scalability**: Easier to scale horizontally\n' +
            '- **Latency**: Higher latency (~5-10ms) due to batching\n\n' +
            'Recommendation: RabbitMQ for critical order flow (reliability), Kafka for order logs/analytics',
        keyPoints: [
            'RabbitMQ better for orders: Explicit ACKs ensure processing, flexible routing, priority queues, lower latency (~1ms)',
            'Kafka better for logs: Higher throughput (millions/sec), persistent log for audit, easier horizontal scaling',
            'RabbitMQ trade-off: Lower throughput (~100K orders/sec), more complex to scale',
            'Kafka trade-off: Higher latency (~5-10ms) due to batching, no priority queues',
            'Production pattern: RabbitMQ for live orders, Kafka for order logs/analytics/backtesting',
        ],
    },
    {
        id: 'message-queues-trading-q-3',
        question:
            'Your order queue has a backlog of 10,000 orders during a market spike. How would you handle this without dropping orders or overwhelming the system?',
        sampleAnswer:
            'Order Queue Backlog Handling:\n\n' +
            '**Immediate Actions:**\n' +
            '1. **Scale consumers**: Spin up 10 additional EMS consumers (auto-scaling)\n' +
            '2. **Throttle producers**: Rate-limit OMS to prevent further backlog growth\n' +
            '3. **Alert traders**: Notify that orders may be delayed\n\n' +
            '**Processing Strategy:**\n' +
            '- Priority queue: Process urgent orders first (market orders > limit orders)\n' +
            '- Batch processing: Process 100 orders at once (better throughput)\n' +
            '- Parallel processing: 10 consumers × 100 orders/sec = 1,000 orders/sec\n\n' +
            '**Timeline:**\n' +
            '- 10,000 orders ÷ 1,000 orders/sec = 10 seconds to clear backlog\n\n' +
            'Production pattern: Pre-scale consumers during known volatility (earnings, FOMC)',
        keyPoints: [
            'Immediate actions: Scale consumers (10 additional EMS), throttle producers (rate-limit OMS), alert traders of delays',
            'Priority queue: Process urgent orders first (market orders before limit orders)',
            'Batch processing: Process 100 orders at once for higher throughput',
            'Parallel processing: 10 consumers × 100 orders/sec = 1,000 orders/sec, clears 10K backlog in 10 seconds',
            'Prevention: Pre-scale consumers during known volatility (earnings, FOMC announcements, market open/close)',
        ],
    },
];

