/**
 * Discussion Questions for Kafka Streams
 */

import { QuizQuestion } from '../../../types';

export const kafkastreamsQuiz: QuizQuestion[] = [
  {
    id: 'kafka-streams-dq-1',
    question:
      'Explain the difference between KStream and KTable in Kafka Streams. Design a real-time e-commerce analytics system that tracks product purchases and maintains running totals. Should you use KStream, KTable, or both? Include windowing strategy and state management.',
    hint: 'Consider the nature of data (events vs state), join operations, and queryable state for dashboards.',
    sampleAnswer: `KStream and KTable represent fundamentally different data abstractions in Kafka Streams, each suited for different use cases.

**KStream vs KTable:**

**KStream (Event Stream):**
- Represents unbounded stream of facts/events
- Each record is independent event
- Inserts only (append-only)
- Same key can appear multiple times

**KTable (Changelog Stream):**
- Represents current state of table
- Each record is update/upsert
- Latest value per key
- null value = deletion

**E-Commerce Analytics System Design:**

**Requirements:**
- Track product purchases in real-time
- Maintain running totals per product
- Display on dashboard (queryable)
- Calculate metrics: total sales, purchase count, average price

**Architecture:**

\`\`\`java
StreamsBuilder builder = new StreamsBuilder();

// 1. KStream: Purchase events (raw events)
KStream<String, Purchase> purchases = builder.stream("purchases");
// Event: {productId: "prod_123", userId: "user_456", amount: 99.99, timestamp: ...}

// 2. Group by product and aggregate
KTable<String, ProductMetrics> productMetrics = purchases
    .groupBy((key, purchase) -> purchase.getProductId())
    .aggregate(
        () -> new ProductMetrics(),  // Initializer
        (productId, purchase, metrics) -> {
            metrics.incrementCount();
            metrics.addAmount(purchase.getAmount());
            metrics.updateAverage();
            return metrics;
        },
        Materialized.<String, ProductMetrics, KeyValueStore<Bytes, byte[]>>as("product-metrics-store")
            .withKeySerde(Serdes.String())
            .withValueSerde(productMetricsSerde)
    );

// 3. Queryable state for dashboard
ReadOnlyKeyValueStore<String, ProductMetrics> store = 
    streams.store("product-metrics-store", QueryableStoreTypes.keyValueStore());

// REST API endpoint:
@GetMapping("/products/{productId}/metrics")
public ProductMetrics getMetrics(@PathVariable String productId) {
    return store.get(productId);
}
\`\`\`

**Use both KStream and KTable:**
- **KStream**: Purchase events (raw data)
- **KTable**: Aggregated metrics (derived state)

**Why this design:**
✅ KStream preserves all purchase events
✅ KTable maintains current metrics
✅ Queryable state for dashboards
✅ Automatically updates on new purchases`,
    keyPoints: [
      'KStream: Event stream (inserts), KTable: Changelog (upserts)',
      'Use KStream for raw events, KTable for aggregated state',
      'Aggregate KStream to KTable for running totals',
      'Materialized state enables queryable stores for dashboards',
      'Windowing for time-based analytics (tumbling, hopping, session)',
      'State management automatic with changelog topics for fault tolerance',
    ],
  },
  {
    id: 'kafka-streams-dq-2',
    question:
      'Design a Kafka Streams application for fraud detection that processes credit card transactions in real-time. The system should detect suspicious patterns (>5 transactions in 5 minutes, transactions from multiple countries in 1 hour). Explain windowing, stateful processing, and exactly-once semantics.',
    hint: 'Consider tumbling/session windows, state stores for tracking patterns, and how to ensure no duplicate alerts.',
    sampleAnswer: `Real-time fraud detection requires stateful stream processing with windowing to detect patterns over time.

**Fraud Detection System:**

**Requirements:**
- Detect >5 transactions in 5-minute window
- Detect transactions from multiple countries in 1-hour window
- Send alerts without duplicates (exactly-once)
- Low latency (<1 second)

**Implementation:**

\`\`\`java
StreamsBuilder builder = new StreamsBuilder();

// Input stream: credit card transactions
KStream<String, Transaction> transactions = builder.stream("transactions");

// Rule 1: >5 transactions in 5 minutes
KTable<Windowed<String>, Long> transactionCounts = transactions
    .groupBy((key, tx) -> tx.getCardId())
    .windowedBy(TimeWindows.of(Duration.ofMinutes(5)))
    .count(Materialized.as("transaction-count-store"));

// Detect high frequency
KStream<String, FraudAlert> highFrequencyAlerts = transactionCounts
    .toStream()
    .filter((windowedKey, count) -> count > 5)
    .map((windowedKey, count) -> {
        String cardId = windowedKey.key();
        return KeyValue.pair(cardId, new FraudAlert(
            cardId, 
            "HIGH_FREQUENCY", 
            "5+ transactions in 5 minutes", 
            count
        ));
    });

// Rule 2: Multiple countries in 1 hour
KTable<Windowed<String>, Set<String>> countryTracker = transactions
    .groupBy((key, tx) -> tx.getCardId())
    .windowedBy(TimeWindows.of(Duration.ofHours(1)))
    .aggregate(
        () -> new HashSet<String>(),
        (cardId, tx, countries) -> {
            countries.add(tx.getCountry());
            return countries;
        },
        Materialized.with(Serdes.String(), countrySetSerde)
    );

KStream<String, FraudAlert> multiCountryAlerts = countryTracker
    .toStream()
    .filter((windowedKey, countries) -> countries.size() > 2)
    .map((windowedKey, countries) -> {
        String cardId = windowedKey.key();
        return KeyValue.pair(cardId, new FraudAlert(
            cardId,
            "MULTI_COUNTRY",
            "Transactions in " + countries.size() + " countries",
            countries
        ));
    });

// Merge alerts
KStream<String, FraudAlert> allAlerts = highFrequencyAlerts.merge(multiCountryAlerts);

// Send to alerts topic (exactly-once)
allAlerts.to("fraud-alerts");

// Configure exactly-once
Properties props = new Properties();
props.put(StreamsConfig.PROCESSING_GUARANTEE_CONFIG, StreamsConfig.EXACTLY_ONCE_V2);
\`\`\`

**State Management:**
- Transaction counts and country sets stored in RocksDB
- Changelog topics backup state for fault tolerance
- On failure, restore from changelog

**Exactly-Once Semantics:**
- Read transaction → Process → Write alert atomically
- No duplicate alerts even if processing fails and retries
- Transactional guarantees across input/output topics`,
    keyPoints: [
      'Windowing (tumbling 5-min, 1-hour) for time-based pattern detection',
      'Stateful aggregation tracks counts and countries per card',
      'RocksDB state stores with changelog topics for fault tolerance',
      'Exactly-once processing prevents duplicate fraud alerts',
      'Low latency (<1s) achieved through stream processing',
      'Merge multiple detection rules into unified alert stream',
    ],
  },
  {
    id: 'kafka-streams-dq-3',
    question:
      'Compare Kafka Streams to Apache Flink for stream processing. For which scenarios would you choose each? Design a stream processing application that needs to process 100K events/sec with stateful operations. Which framework would you choose and why?',
    hint: 'Consider deployment complexity, exactly-once semantics, scalability, and operational overhead.',
    sampleAnswer: `Kafka Streams and Apache Flink are both powerful stream processing frameworks with different trade-offs.

**Kafka Streams vs Apache Flink:**

**Kafka Streams:**
- **Deployment**: Library (embedded in application)
- **Cluster**: None (just Kafka brokers)
- **Exactly-Once**: Yes (built-in)
- **Complexity**: Low (add dependency, run JAR)
- **Scalability**: Horizontal (add instances)
- **State**: Local RocksDB + Kafka changelog

**Apache Flink:**
- **Deployment**: Framework (requires cluster)
- **Cluster**: JobManager + TaskManagers
- **Exactly-Once**: Yes (checkpointing)
- **Complexity**: Higher (cluster management)
- **Scalability**: Horizontal (add TaskManagers)
- **State**: RocksDB + Checkpoints

**When to Choose Each:**

**Choose Kafka Streams:**
✅ Already using Kafka
✅ Want simple deployment (no separate cluster)
✅ Moderate complexity (aggregations, joins, windowing)
✅ Kafka-centric architecture
✅ Small team, low ops overhead

**Choose Flink:**
✅ Complex stream processing (CEP, iterative algorithms)
✅ Need advanced windowing (custom triggers)
✅ Multiple data sources (Kafka, Kinesis, files, databases)
✅ Large-scale ML inference
✅ Have ops team for cluster management

**100K Events/Sec Design:**

**Scenario**: E-commerce click stream analytics
- Volume: 100K clicks/sec
- Operations: Aggregate clicks per product, user sessions, recommendations
- State: User profiles, product metadata
- Latency: <100ms

**Choice: Kafka Streams ✅**

**Why:**
- Kafka already in architecture (click events in Kafka)
- Moderate complexity (aggregations, joins, windowing)
- Simple deployment (Kubernetes pods, no separate cluster)
- Team familiar with Kafka ecosystem
- 100K events/sec well within Kafka Streams capacity

**Architecture:**

\`\`\`
Kafka Topic: clicks (100K events/sec)
  ↓
Kafka Streams App (10 instances)
  - Each processes 10K events/sec
  - Local state (RocksDB)
  - Changelog to Kafka
  ↓
Output Topics: 
  - product-analytics
  - user-sessions
  - recommendation-scores

Deployment:
- Kubernetes StatefulSet
- 10 pods (1:1 with Kafka partitions)
- Auto-scaling based on lag
- Graceful rolling updates

Capacity:
- 10K events/sec per instance
- Processing time: 1ms per event
- Throughput: 10K events/sec × 10 instances = 100K ✅
\`\`\`

**If Choosing Flink:**

Would choose if:
- Need complex event processing (patterns across multiple events)
- Multiple sources (Kafka + Database CDC + Files)
- Advanced ML model inference (TensorFlow models)
- Iterative algorithms (graph processing)

**Trade-offs:**

| Aspect | Kafka Streams | Flink |
|--------|---------------|-------|
| **Ops Complexity** | Low | High |
| **Deployment** | Simple (JAR) | Complex (Cluster) |
| **Exactly-Once** | Yes (Kafka) | Yes (Checkpoints) |
| **State Size** | Moderate (GB) | Large (TB) |
| **Throughput** | High (1M/sec) | Very High (10M/sec) |
| **Latency** | Low (ms) | Very Low (ms) |
| **Use Case** | Kafka-centric | Complex processing |

**Conclusion**: For 100K events/sec with Kafka, Kafka Streams is the right choice due to simplicity, lower ops overhead, and sufficient capacity.`,
    keyPoints: [
      'Kafka Streams: Library (no cluster), Flink: Framework (requires cluster)',
      'Choose Kafka Streams for Kafka-centric, moderate complexity (100K events/sec)',
      'Choose Flink for complex CEP, multiple sources, very high scale (10M+ events/sec)',
      'Kafka Streams simpler deployment (JAR vs cluster management)',
      'Both support exactly-once, stateful processing, horizontal scaling',
      'Trade-off: Simplicity (Kafka Streams) vs Advanced features (Flink)',
    ],
  },
];
