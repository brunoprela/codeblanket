export const tradingSystemArchitectureMC = [
    {
        id: 'trading-system-architecture-mc-1',
        question:
            'What is the PRIMARY advantage of using an event-driven architecture for trading systems over traditional request-response architecture?',
        options: [
            'Event-driven is easier to implement and debug',
            'Event-driven handles asynchronous market data and order execution naturally with loose coupling between components',
            'Event-driven requires less infrastructure and lower costs',
            'Event-driven guarantees faster execution for all operations',
        ],
        correctAnswer: 1,
        explanation:
            'Event-driven architecture is IDEAL for trading because: (1) Market data arrives asynchronously (streaming quotes), (2) Order execution is asynchronous (fill callbacks), (3) Multiple systems need to react to same events (strategy + risk + audit all listen to fills), (4) Loose coupling enables independent component development/scaling. Example: Market data event published once, consumed by 10 strategies simultaneously without coupling. Request-response would require strategy to poll for data (inefficient). Trade-off: Event-driven is more complex to implement (event bus, message queues) but essential for trading systems that must handle millions of asynchronous events per second.',
    },
    {
        id: 'trading-system-architecture-mc-2',
        question:
            'In a microservices architecture for trading, what is the MAIN trade-off between monolithic and microservices design?',
        options: [
            'Microservices always have lower latency than monolithic',
            'Monolithic is always cheaper to operate than microservices',
            'Microservices add network latency (1-2ms per hop) but enable independent scaling, fault isolation, and technology flexibility',
            'Monolithic can handle higher throughput than microservices',
        ],
        correctAnswer: 2,
        explanation:
            'Microservices Trade-off: ADVANTAGES: (1) Independent scaling (scale market data 10×, strategy 2×), (2) Fault isolation (strategy crash doesn\'t kill data feed), (3) Technology flexibility (C++ for market data, Python for strategy), (4) Independent deployment (deploy risk service without touching others). DISADVANTAGES: (1) Network latency (+1-2ms per service hop), (2) Operational complexity (service discovery, monitoring, distributed tracing), (3) Harder debugging (spans multiple services). Example: Citadel uses microservices for flexibility at scale (millions of messages/sec) despite 1-2ms latency penalty. Monolithic better for small systems (<100K events/sec) where simplicity matters more than scale. Hybrid approach: Hot path monolithic (market data → signal < 1ms), cold path microservices (reporting).',
    },
    {
        id: 'trading-system-architecture-mc-3',
        question:
            'When designing state management for a distributed trading system, why is Redis commonly chosen over PostgreSQL for real-time trading state (positions, open orders)?',
        options: [
            'Redis is cheaper than PostgreSQL',
            'Redis provides sub-millisecond read/write latency (<1ms) compared to PostgreSQL (5-10ms), critical for real-time trading decisions',
            'Redis has better query capabilities than PostgreSQL',
            'Redis provides stronger consistency guarantees than PostgreSQL',
        ],
        correctAnswer: 1,
        explanation:
            'Redis for Real-Time State: LATENCY: Redis: <1ms read/write (in-memory), PostgreSQL: 5-10ms (disk-based). Trading decision at 1000 trades/sec needs <1ms state access. THROUGHPUT: Redis: 100K+ ops/sec per instance, PostgreSQL: 10K+ ops/sec. DATA STRUCTURE: Redis: Native support for hash maps, sorted sets (perfect for positions, order books), PostgreSQL: Relational (overkill for key-value). USE CASES: Redis: Current positions, open orders, last prices (hot data, need speed), PostgreSQL: Order history, trade log, audit trail (cold data, need persistence). PATTERN: Write to Redis (immediate), async write to PostgreSQL (persistence). Example: Position update must reflect in <1ms for risk check, but historical position log can wait 100ms. Hybrid approach: Redis + PostgreSQL is industry standard (Jane Street, Citadel).',
    },
    {
        id: 'trading-system-architecture-mc-4',
        question:
            'What is the purpose of a "circuit breaker" pattern in trading system architecture?',
        options: [
            'To prevent electrical overloads in the data center',
            'To stop trading when market volatility is too high',
            'To prevent cascading failures by temporarily stopping calls to a failing external service (broker/exchange) until it recovers',
            'To limit the number of orders that can be sent per second',
        ],
        correctAnswer: 2,
        explanation:
            'Circuit Breaker Pattern: PURPOSE: Prevent cascading failures when external service (broker, exchange, data feed) is down or slow. Without circuit breaker: 1000s of requests pile up waiting for timeout (30s each) → system freezes. With circuit breaker: After 5 failures → OPEN (stop sending requests immediately, fail fast) → Wait 60s → HALF_OPEN (try 1 request to test) → If success: CLOSED (resume normal operation) → If failure: OPEN again. STATES: CLOSED: Normal operation, all requests pass through. OPEN: Service is down, fail immediately (don\'t wait for timeout), save resources. HALF_OPEN: Testing recovery, allow limited requests. EXAMPLE: Broker API goes down at 9:30am market open (overloaded). Circuit breaker opens after 5 failures (5 seconds). System rejects new orders gracefully instead of hanging. Broker recovers at 9:35am, circuit breaker detects in half-open test, closes, resumes trading. Used by: Netflix (Hystrix), AWS (resilience), all production trading systems.',
    },
    {
        id: 'trading-system-architecture-mc-5',
        question:
            'In an event-driven trading system, what is the PRIMARY reason for using a message queue (Kafka, RabbitMQ) between components instead of direct API calls?',
        options: [
            'Message queues are faster than API calls',
            'Message queues are easier to implement than APIs',
            'Message queues provide decoupling, buffering, persistence, and allow multiple consumers to process same event asynchronously',
            'Message queues use less memory than direct API calls',
        ],
        correctAnswer: 2,
        explanation:
            'Message Queue Benefits: DECOUPLING: Services don\'t know about each other (loose coupling). Market data publishes to queue, doesn\'t know who consumes. Add new strategy without modifying data feed. BUFFERING: Queue absorbs spikes (market opens: 100K orders/sec → queue buffers → OMS processes at 10K/sec). Prevents system overload. PERSISTENCE: Messages persist on disk (Kafka retention: 7 days). If consumer crashes, messages aren\'t lost, replay from last offset. MULTIPLE CONSUMERS: Single market data message consumed by 10 strategies (fan-out). With API calls, would need 10 separate calls (inefficient). ASYNCHRONOUS: Producer doesn\'t wait for consumer (non-blocking). Market data publishes and continues (doesn\'t wait for strategy to process). ORDERING: Kafka partitions guarantee order within partition (critical for trade sequence). TRADE-OFF: Added latency (+1-2ms) vs reliability and scalability. Example: Interactive Brokers uses message queues to handle 1M+ orders/day, decouple order entry from execution. Without queue: Direct API call from strategy to OMS (tight coupling, no buffering, lose messages on crash).',
    },
];

