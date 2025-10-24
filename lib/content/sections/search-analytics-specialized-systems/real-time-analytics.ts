import { Section } from '@/lib/types';

const realTimeAnalyticsSection: Section = {
  id: 'real-time-analytics',
  title: 'Real-Time Analytics',
  content: `
# Real-Time Analytics

## Introduction

**Real-time analytics** is the ability to process, analyze, and act on data as it's generated—within milliseconds to seconds. Unlike batch processing that analyzes historical data hours or days after collection, real-time systems provide immediate insights enabling instant decision-making.

In today's fast-paced digital world, the ability to respond immediately to events is critical:
- **Fraud detection**: Block fraudulent transactions before they complete
- **Personalization**: Recommend products while user is browsing
- **Monitoring**: Alert on system issues before users notice
- **Trading**: Execute trades based on market conditions in milliseconds
- **Real-time dashboards**: Business metrics updated every second

This section covers the architecture, technologies, and trade-offs involved in building real-time analytics systems at scale.

## Real-Time vs Near-Real-Time vs Batch

Understanding the spectrum of data processing latencies is essential for choosing the right architecture.

### Latency Comparison

| Processing Type | Latency | Update Frequency | Use Cases | Technologies |
|----------------|---------|------------------|-----------|--------------|
| **Real-Time** | <1 second | Continuous (event-driven) | Fraud detection, trading, real-time bidding | Apache Flink, Druid |
| **Near-Real-Time** | 1-60 seconds | Every few seconds | Live dashboards, monitoring alerts | Kafka + Spark Streaming |
| **Micro-Batch** | 1-15 minutes | Periodic batches | Metrics aggregation, reporting | Spark Structured Streaming |
| **Batch** | Hours to days | Scheduled (nightly, weekly) | Reports, ML training, data warehouse ETL | Spark, Hadoop MapReduce |

### When Real-Time is Worth the Complexity

Real-time systems are significantly more complex than batch systems. Only use real-time when:

**Good Reasons:**
- **Time-sensitive decisions**: Fraud must be blocked NOW, not tomorrow
- **User experience**: Real-time personalization improves conversion
- **Operational necessity**: Trading systems can't wait for batch processing
- **Competitive advantage**: First to act on market signals wins

**Bad Reasons:**
- "Because it sounds cool"
- User doesn't actually need sub-second updates
- Batch would work fine but "real-time is the future"
- Not considering 10x cost and complexity

**Reality Check**: Most "real-time" requirements are actually fine with near-real-time (10-60 second delay).

## Stream Processing Architecture

Real-time analytics requires a fundamentally different architecture than batch processing.

### Core Components

\`\`\`
Data Sources → Message Queue → Stream Processor → Data Store → Dashboard/Alerts
   (events)      (buffer)         (transform)       (query)      (visualize)
\`\`\`

**1. Data Sources**: Applications, IoT sensors, clickstreams, logs
**2. Message Queue**: Kafka, Kinesis (decouples producers from consumers)
**3. Stream Processor**: Flink, Spark Streaming (transform, aggregate, join)
**4. Data Store**: Druid, ClickHouse, TimescaleDB (optimized for analytics)
**5. Dashboard**: Grafana, custom dashboards (visualize results)

### Example: E-Commerce Real-Time Analytics

\`\`\`
Web/Mobile Apps (millions of users)
    ↓
  Events: pageview, add_to_cart, purchase, search
    ↓
  Kafka (buffer 1M events/sec)
    ↓
  Apache Flink (aggregate, enrich, detect patterns)
    ↓
  Apache Druid (real-time OLAP)
    ↓
  Dashboard: Live sales, conversion rates, popular products
\`\`\`

**Requirements:**
- Ingest 1M events/second
- Update dashboards every second
- Query latency <100ms
- 7-day retention in hot storage
- Handle traffic spikes (Black Friday 10x)

## Apache Kafka: The Foundation

Kafka is the de facto standard message queue for real-time systems.

### Why Kafka?

**Traditional message queues** (RabbitMQ, ActiveMQ):
- Low throughput (~10k messages/sec)
- Messages deleted after consumption
- Not designed for replay

**Kafka advantages:**
- **High throughput**: Millions of messages/second
- **Durable**: Messages persisted to disk
- **Replayable**: Consumers can re-read from any offset
- **Scalable**: Horizontal scaling via partitions
- **Fault-tolerant**: Replication ensures no data loss

### Kafka Architecture

\`\`\`
Producer → Topic (partitioned) → Consumer Group
           ├─ Partition 0 (leader: broker 1, replicas: broker 2,3)
           ├─ Partition 1 (leader: broker 2, replicas: broker 1,3)
           └─ Partition 2 (leader: broker 3, replicas: broker 1,2)
\`\`\`

**Key Concepts:**

**Topic**: Logical channel for messages (e.g., "pageviews", "purchases")
**Partition**: Physical subdivision for parallelism
**Producer**: Writes messages to topics
**Consumer Group**: Multiple consumers processing in parallel
**Offset**: Position in partition (enables replay)

### Producing Events

\`\`\`python
from kafka import KafkaProducer
import json
from datetime import datetime

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Send pageview event
event = {
    'event_type': 'pageview',
    'user_id': 'user_12345',
    'page': '/products/laptop',
    'timestamp': datetime.utcnow().isoformat(),
    'session_id': 'session_abc',
    'referrer': 'google',
    'device': 'mobile'
}

producer.send('pageviews', value=event, key=event['user_id'].encode())
producer.flush()
\`\`\`

**Key parameter: \`key\`**
- Messages with same key go to same partition
- Ensures ordering for related events (e.g., same user_id)
- Enables stateful processing (all events for a user processed by same worker)

### Consuming Events

\`\`\`python
from kafka import KafkaConsumer

consumer = KafkaConsumer(
    'pageviews',
    bootstrap_servers=['localhost:9092'],
    group_id='analytics-pipeline',
    auto_offset_reset='earliest',  # Start from beginning
    enable_auto_commit=True,
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

for message in consumer:
    event = message.value
    print(f"User {event['user_id']} viewed {event['page']}")
    # Process event...
\`\`\`

**Consumer Groups**: Multiple consumers in a group share partitions
- 3 partitions, 3 consumers → each gets 1 partition
- 3 partitions, 6 consumers → 3 consumers idle (can't exceed partitions)
- 3 partitions, 1 consumer → processes all 3 sequentially

### Retention and Replay

Unlike traditional queues, Kafka retains messages after consumption.

\`\`\`bash
# Configure 7-day retention
retention.ms=604800000  # 7 days in milliseconds
\`\`\`

**Use cases for replay:**
- **Bug fix**: Reprocess with corrected logic
- **New feature**: Backfill data for new analytics
- **Disaster recovery**: Rebuild downstream state from Kafka
- **Testing**: Replay production traffic in staging

**Example: Reprocessing**
\`\`\`python
# Bug found in aggregation logic deployed 2024-01-10
# Fix deployed 2024-01-15
# Reprocess from 2024-01-10

consumer = KafkaConsumer(
    'pageviews',
    group_id='analytics-pipeline-reprocess',  # New group!
    auto_offset_reset='earliest'
)

# Seek to timestamp 2024-01-10 00:00:00
timestamp_ms = 1704844800000
consumer.seek_to_timestamp(timestamp_ms)

# Reprocess from that point
for message in consumer:
    # Process with fixed logic
    ...
\`\`\`

## Apache Flink: True Stream Processing

Flink is a stream processing framework designed for low-latency, high-throughput, stateful computations.

### Flink vs Spark Streaming

| Feature | Apache Flink | Spark Streaming |
|---------|-------------|-----------------|
| **Model** | True streaming (event-by-event) | Micro-batches (small batches) |
| **Latency** | Sub-millisecond to milliseconds | 200ms to seconds |
| **State Management** | Built-in, sophisticated | Limited, requires external store |
| **Event Time** | Native support | Added later, less mature |
| **Exactly-Once** | Native | Supported |
| **Best For** | <100ms latency, complex state | >1 second latency, Spark ecosystem |

**When to use Flink:**
- Latency requirements <100ms
- Complex stateful operations (session windows, joins)
- Event time processing critical
- Financial systems, fraud detection, real-time recommendations

**When to use Spark:**
- Latency requirements >1 second acceptable
- Already using Spark for batch (unified codebase)
- Simpler operations (transformations, basic aggregations)

### Flink Programming Model

\`\`\`java
StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

// Source: Read from Kafka
DataStream<Event> events = env
    .addSource(new FlinkKafkaConsumer<>("events", new EventSchema(), properties))
    .assignTimestampsAndWatermarks(new EventTimeExtractor());

// Transformation: Filter, map, aggregate
DataStream<Metric> metrics = events
    .filter(event -> event.getType().equals("pageview"))
    .keyBy(Event::getUserId)
    .window(TumblingEventTimeWindows.of(Time.minutes(1)))
    .aggregate(new PageViewCounter());

// Sink: Write results
metrics.addSink(new FlinkKafkaProducer<>("metrics", new MetricSchema(), properties));

env.execute("Real-Time Analytics Job");
\`\`\`

### Stateful Stream Processing

Flink's killer feature is **managed state**—the ability to store data per key across events.

**Example: Fraud Detection**

\`\`\`java
public class FraudDetector extends KeyedProcessFunction<String, Transaction, Alert> {
    
    // Managed state (persisted, fault-tolerant)
    private ValueState<Double> totalSpentLast24h;
    private ValueState<Integer> uniqueMerchantsLast24h;
    private ListState<Transaction> recentTransactions;
    
    @Override
    public void open(Configuration parameters) {
        // Initialize state
        totalSpentLast24h = getRuntimeContext().getState(
            new ValueStateDescriptor<>("totalSpent", Double.class)
        );
        uniqueMerchantsLast24h = getRuntimeContext().getState(
            new ValueStateDescriptor<>("uniqueMerchants", Integer.class)
        );
        recentTransactions = getRuntimeContext().getListState(
            new ListStateDescriptor<>("recentTxns", Transaction.class)
        );
    }
    
    @Override
    public void processElement(Transaction txn, Context ctx, Collector<Alert> out) 
            throws Exception {
        
        // Get current state
        Double currentTotal = totalSpentLast24h.value();
        if (currentTotal == null) currentTotal = 0.0;
        
        // Update state
        currentTotal += txn.getAmount();
        totalSpentLast24h.update(currentTotal);
        
        // Check fraud rules
        if (currentTotal > 10000.0) {
            out.collect(new Alert(
                txn.getUserId(),
                "High spending detected",
                currentTotal,
                "CRITICAL"
            ));
        }
        
        // Check velocity (transactions per hour)
        List<Transaction> recent = new ArrayList<>();
        for (Transaction t : recentTransactions.get()) {
            if (t.getTimestamp() > ctx.timestamp() - 3600000) {  // Last hour
                recent.add(t);
            }
        }
        
        if (recent.size() > 20) {
            out.collect(new Alert(
                txn.getUserId(),
                "High transaction velocity",
                recent.size(),
                "WARNING"
            ));
        }
        
        // Add current transaction to recent
        recentTransactions.add(txn);
        
        // Set timer to clean up old transactions
        ctx.timerService().registerEventTimeTimer(
            ctx.timestamp() + 86400000  // 24 hours
        );
    }
    
    @Override
    public void onTimer(long timestamp, OnTimerContext ctx, Collector<Alert> out) 
            throws Exception {
        // Clean up state older than 24 hours
        List<Transaction> toKeep = new ArrayList<>();
        for (Transaction t : recentTransactions.get()) {
            if (t.getTimestamp() > timestamp - 86400000) {
                toKeep.add(t);
            }
        }
        recentTransactions.update(toKeep);
    }
}

// Usage
transactions
    .keyBy(Transaction::getUserId)
    .process(new FraudDetector())
    .addSink(new AlertSink());
\`\`\`

**Why state matters:**
- Track per-user/per-key aggregations (spending, counts)
- Maintain windows of recent events
- Implement complex business logic (fraud rules)
- All state is fault-tolerant (checkpointed to persistent storage)

### Checkpointing and Fault Tolerance

Flink provides **exactly-once processing guarantees** through checkpointing.

**How it works:**
1. Every N seconds (e.g., 60s), Flink takes a **checkpoint**
2. Checkpoint captures:
   - Current offset in Kafka
   - All operator state (totalSpent, recentTransactions, etc.)
   - In-flight records
3. Checkpoint written to persistent storage (HDFS, S3)
4. If node fails:
   - Flink restarts from last checkpoint
   - Reprocesses from last checkpointed Kafka offset
   - State restored exactly as it was

**Configuration:**
\`\`\`java
env.enableCheckpointing(60000);  // Checkpoint every 60 seconds
env.getCheckpointConfig().setCheckpointingMode(CheckpointingMode.EXACTLY_ONCE);
env.getCheckpointConfig().setMinPauseBetweenCheckpoints(30000);
env.getCheckpointConfig().setCheckpointTimeout(600000);
\`\`\`

**Result**: Node failure doesn't lose data or duplicate processing.

## Windowing

Stream processing often requires **aggregating events over time windows**.

### Window Types

#### 1. Tumbling Windows
Fixed, non-overlapping time periods.

\`\`\`
Time:     0    5   10   15   20   25   30
Events:   ├────┼────┼────┼────┼────┼────┤
Window 1: [─────]
Window 2:      [─────]
Window 3:           [─────]
\`\`\`

**Example: Count events every 5 minutes**
\`\`\`java
events
    .keyBy(Event::getUserId)
    .window(TumblingEventTimeWindows.of(Time.minutes(5)))
    .aggregate(new CountAggregator())
\`\`\`

**Use cases**: Metrics every X minutes, hourly reports

#### 2. Sliding Windows
Fixed size, overlapping windows.

\`\`\`
Time:     0    5   10   15   20   25   30
Window 1: [──────────]
Window 2:      [──────────]
Window 3:           [──────────]
Size: 10 minutes, Slide: 5 minutes
\`\`\`

**Example: Continuous 10-minute average, updated every minute**
\`\`\`java
events
    .keyBy(Event::getUserId)
    .window(SlidingEventTimeWindows.of(Time.minutes(10), Time.minutes(1)))
    .aggregate(new AverageAggregator())
\`\`\`

**Use cases**: Moving averages, "last N minutes" continuously updated

#### 3. Session Windows
Dynamic windows based on inactivity gaps.

\`\`\`
User Events:  E1 E2    E3       E4 E5 E6
Time:         0  2     15       30 31 33
Gap: 10s
              [─────]           [────────]
              Session 1         Session 2
\`\`\`

**Example: User sessions with 30-minute inactivity timeout**
\`\`\`java
events
    .keyBy(Event::getUserId)
    .window(EventTimeSessionWindows.withGap(Time.minutes(30)))
    .aggregate(new SessionAggregator())
\`\`\`

**Use cases**: User sessions, activity bursts, fraud detection

### Event Time vs Processing Time

**Processing Time**: When event is processed by the system
**Event Time**: When event actually occurred (timestamp in event)

\`\`\`
Event occurs: 14:00:00 (event time)
Network delay...
Arrives at Flink: 14:00:15 (processing time)
\`\`\`

**Why event time matters:**
- Out-of-order events common (network delays, mobile offline)
- Reprocessing should produce same results
- Business logic based on when events happened, not when processed

**Example: Sales by hour**
- Processing time: Sales grouped by when they arrived (wrong if delayed)
- Event time: Sales grouped by purchase time (correct)

### Watermarks and Late Data

**Watermarks** tell Flink "all events with timestamp < T have arrived."

\`\`\`
Watermark Strategy:
watermark_time = max_event_time_seen - allowed_lateness

Example with 30-second lateness:
Event times:     10:00:00, 10:00:10, 10:00:20, 10:00:15 (late), 10:00:30
Watermarks:      09:59:30, 09:59:40, 09:59:50, (no change), 10:00:00
                                                            ^
                                                   Window [09:55-10:00] closes
\`\`\`

**Configuration:**
\`\`\`java
WatermarkStrategy<Event> watermarkStrategy = WatermarkStrategy
    .<Event>forBoundedOutOfOrderness(Duration.ofSeconds(30))
    .withTimestampAssigner((event, timestamp) -> event.getTimestamp());

DataStream<Event> withWatermarks = events
    .assignTimestampsAndWatermarks(watermarkStrategy);
\`\`\`

**Handling very late data:**
\`\`\`java
OutputTag<Event> lateDataTag = new OutputTag<Event>("late-data"){};

SingleOutputStreamOperator<Result> result = events
    .keyBy(Event::getKey)
    .window(TumblingEventTimeWindows.of(Time.minutes(5)))
    .allowedLateness(Time.minutes(2))  // Accept up to 2min late
    .sideOutputLateData(lateDataTag)   // Send very late data to side output
    .aggregate(new Aggregator());

// Process late data separately (e.g., write to special table)
DataStream<Event> lateData = result.getSideOutput(lateDataTag);
lateData.addSink(new LateDataSink());
\`\`\`

## Apache Druid: Real-Time OLAP

Druid is a real-time analytics database optimized for time-series data with sub-second query latency.

### Architecture

\`\`\`
Real-Time Ingestion → Real-Time Nodes (recent data)
                            ↓
Batch Ingestion → Historical Nodes (older data)
                            ↓
                      Query Brokers
                            ↓
                        Dashboards
\`\`\`

**Components:**

**Real-Time Nodes**: Ingest and query recent data (last few hours)
**Historical Nodes**: Store and query historical data
**Brokers**: Route queries, merge results
**Coordinators**: Manage data distribution
**Deep Storage**: S3/HDFS for durability

### Ingestion from Kafka

\`\`\`json
{
  "type": "kafka",
  "spec": {
    "dataSchema": {
      "dataSource": "pageviews",
      "timestampSpec": {
        "column": "timestamp",
        "format": "iso"
      },
      "dimensionsSpec": {
        "dimensions": [
          "user_id",
          "page",
          "referrer",
          "device"
        ]
      },
      "metricsSpec": [
        {
          "type": "count",
          "name": "views"
        },
        {
          "type": "longSum",
          "name": "total_time_spent",
          "fieldName": "time_spent_ms"
        },
        {
          "type": "hyperUnique",
          "name": "unique_users",
          "fieldName": "user_id"
        }
      ],
      "granularitySpec": {
        "type": "uniform",
        "segmentGranularity": "HOUR",
        "queryGranularity": "MINUTE",
        "rollup": true
      }
    },
    "ioConfig": {
      "topic": "pageviews",
      "consumerProperties": {
        "bootstrap.servers": "localhost:9092"
      },
      "taskDuration": "PT1H",
      "useEarliestOffset": false
    },
    "tuningConfig": {
      "type": "kafka",
      "maxRowsInMemory": 100000
    }
  }
}
\`\`\`

**Key features:**

**Rollup**: Pre-aggregation at ingestion
- Raw: 1M events → Rolled up: 10k rows (100x compression)
- Groups by dimensions, aggregates metrics

**HyperLogLog**: Approximate unique counts
- Exact unique count: Store all user_ids (GBs)
- HyperLogLog: 12KB with 2% error

### Querying Druid

\`\`\`json
{
  "queryType": "timeseries",
  "dataSource": "pageviews",
  "intervals": ["2024-01-15T00:00/2024-01-15T23:59"],
  "granularity": "hour",
  "filter": {
    "type": "selector",
    "dimension": "device",
    "value": "mobile"
  },
  "aggregations": [
    {
      "type": "longSum",
      "name": "total_views",
      "fieldName": "views"
    },
    {
      "type": "hyperUnique",
      "name": "unique_users",
      "fieldName": "unique_users"
    }
  ],
  "postAggregations": [
    {
      "type": "arithmetic",
      "name": "views_per_user",
      "fn": "/",
      "fields": [
        {"type": "fieldAccess", "fieldName": "total_views"},
        {"type": "hyperUniqueCardinality", "fieldName": "unique_users"}
      ]
    }
  ]
}
\`\`\`

**Result** (in <100ms for billions of events):
\`\`\`json
[
  {
    "timestamp": "2024-01-15T14:00:00.000Z",
    "result": {
      "total_views": 1523849,
      "unique_users": 342567,
      "views_per_user": 4.45
    }
  },
  ...
]
\`\`\`

## Approximation Algorithms

Real-time analytics at scale often requires trading perfect accuracy for performance.

### HyperLogLog: Unique Counting

**Problem**: Count unique visitors across billions of events

**Exact solution**: Store every user_id in a set
- 1B unique users × 16 bytes = 16GB RAM
- Not feasible for real-time

**HyperLogLog solution**:
- Fixed memory: 12KB (regardless of cardinality!)
- Error rate: ~2%
- Result: Count 1B unique users with 12KB

**How it works** (simplified):
1. Hash user_id → binary number
2. Count leading zeros (e.g., 00001010... → 4 leading zeros)
3. Max leading zeros estimates cardinality: ~2^max_zeros
4. Use multiple buckets and harmonic mean for accuracy

**Implementation:**
\`\`\`python
from hyperloglog import HyperLogLog

hll = HyperLogLog(0.01)  # 1% error rate

# Add millions of user_ids
for event in event_stream:
    hll.add(event.user_id)

unique_count = len(hll)  # O(1), 12KB memory
\`\`\`

**Real-world usage**:
- Redis: PFADD, PFCOUNT commands
- Druid: hyperUnique aggregation
- BigQuery: APPROX_COUNT_DISTINCT

### Count-Min Sketch: Frequency Estimation

**Problem**: Find top 10 most viewed pages from billions of pageviews

**Exact solution**: Hash map of all pages → counts
- 10M unique pages × 16 bytes = 160MB
- Still manageable, but doesn't scale to 100M pages

**Count-Min Sketch**:
- Fixed memory: 1MB (regardless of unique pages)
- Error: Overestimates by ε, never underestimates
- Result: Find top-K with bounded error

**How it works**:
\`\`\`
Count-Min Sketch: 2D array (width × depth)
width = 10,000, depth = 5

To increment count for "page_A":
1. Hash "page_A" with 5 different hash functions
2. Increment count at 5 positions:
   row[0][hash1(page_A) % width]++
   row[1][hash2(page_A) % width]++
   ...
   row[4][hash5(page_A) % width]++

To query count for "page_A":
   return MIN(row[0][pos1], row[1][pos2], ..., row[4][pos5])
\`\`\`

**Why it works**: Collisions only increase counts (overestimate), taking MIN across multiple hashes reduces error.

**Implementation**:
\`\`\`python
from count_min_sketch import CountMinSketch

cms = CountMinSketch(width=10000, depth=5)

# Count pageviews
for event in event_stream:
    cms.add(event.page_url)

# Top 10 pages
top_pages = cms.top_k(10)
\`\`\`

### Bloom Filter: Set Membership

**Problem**: Check if user has already seen a recommendation (deduplication)

**Exact solution**: Store all seen user_ids in set
- 100M users × 16 bytes = 1.6GB

**Bloom Filter**:
- Fixed memory: 120MB (for 1% false positive rate)
- No false negatives ("not in set" is always correct)
- Possible false positives ("in set" might be wrong)

**Use case**: "Don't show ads user already clicked"
- False positive: User sees ad they clicked (minor issue)
- False negative: Never happens (user never sees already-clicked ad twice)

## Real-World Example: Real-Time Fraud Detection

Let's design a complete real-time fraud detection system.

### Requirements

- **Throughput**: 50,000 transactions/second
- **Latency**: <100ms (block fraud before transaction completes)
- **Accuracy**: 99%+ (false positives cost customer trust)
- **Rules**:
  1. User spends >$10k in 24 hours
  2. User makes purchases in >3 countries in 24 hours
  3. User shops at >5 new merchants in 1 hour
  4. Transaction amount >3× user's average

### Architecture

\`\`\`
Credit Card Transactions (50k/sec)
    ↓
  Kafka (transactions topic, 100 partitions)
    ↓
  Flink (fraud detection, 50 workers)
    ↓
  Redis (low-latency state: user profiles)
    ↓
  Alerts → Block transaction / Manual review
\`\`\`

### Flink Implementation

\`\`\`java
public class FraudDetectionJob {
    
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = 
            StreamExecutionEnvironment.getExecutionEnvironment();
        
        // Configure for low latency
        env.setBufferTimeout(0);  // No buffering
        env.enableCheckpointing(60000);  // Checkpoint every minute
        
        // Source: Read transactions from Kafka
        DataStream<Transaction> transactions = env
            .addSource(new FlinkKafkaConsumer<>(
                "transactions",
                new TransactionDeserializer(),
                kafkaProps
            ))
            .assignTimestampsAndWatermarks(
                WatermarkStrategy
                    .<Transaction>forBoundedOutOfOrderness(Duration.ofSeconds(5))
                    .withTimestampAssigner((txn, ts) -> txn.getTimestamp())
            );
        
        // Fraud detection
        DataStream<FraudAlert> alerts = transactions
            .keyBy(Transaction::getUserId)
            .process(new FraudDetector());
        
        // Sink: Write alerts
        alerts.addSink(new RedisSink());
        alerts.addSink(new KafkaAlertSink());
        
        env.execute("Real-Time Fraud Detection");
    }
}

public class FraudDetector extends KeyedProcessFunction<String, Transaction, FraudAlert> {
    
    // State: Recent transactions
    private ListState<Transaction> recentTransactions;
    
    // State: Spending by country
    private MapState<String, Double> spendingByCountry;
    
    // State: User profile (average transaction amount)
    private ValueState<UserProfile> userProfile;
    
    @Override
    public void open(Configuration config) {
        recentTransactions = getRuntimeContext().getListState(
            new ListStateDescriptor<>("recentTxns", Transaction.class)
        );
        spendingByCountry = getRuntimeContext().getMapState(
            new MapStateDescriptor<>("countrySpending", String.class, Double.class)
        );
        userProfile = getRuntimeContext().getState(
            new ValueStateDescriptor<>("profile", UserProfile.class)
        );
    }
    
    @Override
    public void processElement(
            Transaction txn, 
            Context ctx, 
            Collector<FraudAlert> out) throws Exception {
        
        long now = ctx.timestamp();
        long last24h = now - 86400000;
        long lastHour = now - 3600000;
        
        // Load user profile
        UserProfile profile = userProfile.value();
        if (profile == null) {
            profile = new UserProfile(txn.getUserId());
        }
        
        // Get recent transactions (last 24h)
        List<Transaction> recent24h = new ArrayList<>();
        for (Transaction t : recentTransactions.get()) {
            if (t.getTimestamp() >= last24h) {
                recent24h.add(t);
            }
        }
        
        // === RULE 1: High spending (>$10k in 24h) ===
        double total24h = recent24h.stream()
            .mapToDouble(Transaction::getAmount)
            .sum() + txn.getAmount();
        
        if (total24h > 10000.0) {
            out.collect(new FraudAlert(
                txn,
                "HIGH_SPENDING",
                String.format("$%.2f spent in 24h", total24h),
                RiskLevel.CRITICAL
            ));
        }
        
        // === RULE 2: Geographic anomaly (>3 countries in 24h) ===
        Set<String> countries = recent24h.stream()
            .map(Transaction::getCountry)
            .collect(Collectors.toSet());
        countries.add(txn.getCountry());
        
        if (countries.size() > 3) {
            out.collect(new FraudAlert(
                txn,
                "GEOGRAPHIC_ANOMALY",
                String.format("%d countries: %s", countries.size(), countries),
                RiskLevel.HIGH
            ));
        }
        
        // === RULE 3: New merchant velocity (>5 new merchants in 1h) ===
        Set<String> knownMerchants = profile.getKnownMerchants();
        long newMerchantsLastHour = recent24h.stream()
            .filter(t -> t.getTimestamp() >= lastHour)
            .map(Transaction::getMerchantId)
            .filter(m -> !knownMerchants.contains(m))
            .distinct()
            .count();
        
        if (!knownMerchants.contains(txn.getMerchantId())) {
            newMerchantsLastHour++;
        }
        
        if (newMerchantsLastHour > 5) {
            out.collect(new FraudAlert(
                txn,
                "NEW_MERCHANT_VELOCITY",
                String.format("%d new merchants in 1h", newMerchantsLastHour),
                RiskLevel.MEDIUM
            ));
        }
        
        // === RULE 4: Amount anomaly (>3× average) ===
        double avgAmount = profile.getAverageAmount();
        if (avgAmount > 0 && txn.getAmount() > avgAmount * 3) {
            out.collect(new FraudAlert(
                txn,
                "AMOUNT_ANOMALY",
                String.format("$%.2f vs avg $%.2f", txn.getAmount(), avgAmount),
                RiskLevel.MEDIUM
            ));
        }
        
        // === Update state ===
        recent24h.add(txn);
        recentTransactions.update(recent24h);
        
        profile.addTransaction(txn);
        userProfile.update(profile);
        
        // Clean up old transactions after 24h
        ctx.timerService().registerEventTimeTimer(now + 86400000);
    }
    
    @Override
    public void onTimer(long timestamp, OnTimerContext ctx, Collector<FraudAlert> out) 
            throws Exception {
        // Remove transactions older than 24h
        List<Transaction> toKeep = new ArrayList<>();
        for (Transaction t : recentTransactions.get()) {
            if (t.getTimestamp() >= timestamp - 86400000) {
                toKeep.add(t);
            }
        }
        recentTransactions.update(toKeep);
    }
}
\`\`\`

### Performance Characteristics

**Latency breakdown** (target: <100ms):
- Kafka ingestion: 5ms
- Flink processing: 20ms
  - State lookup: 5ms
  - Rule evaluation: 10ms
  - State update: 5ms
- Alert write to Redis: 10ms
- **Total: 35ms** ✅ Well under 100ms

**Throughput**:
- 50,000 txns/sec ÷ 100 Kafka partitions = 500 txns/sec/partition
- 50 Flink workers × 2 partitions each = 1,000 txns/sec/worker
- Each worker easily handles 1,000 txns/sec (simple rules, local state)

**State size per user**:
- Recent transactions (24h avg): 50 txns × 200 bytes = 10KB
- User profile: 500 bytes
- Total: ~11KB per active user
- 1M active users: 11GB total state (fits in memory!)

**Fault tolerance**:
- Checkpoint every 60 seconds
- Failure recovery time: <2 minutes (restore from S3 + reprocess from Kafka)
- Zero data loss (exactly-once guarantees)

## Best Practices

### 1. Choose the Right Latency Target

Don't over-engineer:
- **Sub-second**: Fraud detection, trading → Use Flink
- **1-10 seconds**: Live dashboards → Spark Streaming fine
- **Minutes**: Metrics aggregation → Batch every minute

### 2. Use Approximation Algorithms

For unique counts, top-K, set membership:
- HyperLogLog for unique counts (2% error, 12KB vs GBs)
- Count-Min Sketch for top-K (bounded error)
- Bloom Filter for membership (false positives OK)

### 3. Design for Out-of-Order Data

- Use event time, not processing time
- Set appropriate watermarks (balance latency vs completeness)
- Handle late data with side outputs

### 4. Optimize State Size

State is the bottleneck:
- Store only what you need (not full objects)
- Use TTL to expire old state
- Consider external state (Redis) for large datasets

### 5. Monitor Lag

Key metrics:
- **Kafka consumer lag**: How far behind real-time?
- **Processing latency**: Event time vs processing time
- **Checkpoint duration**: Should be <1 minute

### 6. Test with Production Traffic

Replay production events from Kafka:
- Test new logic before deployment
- Verify performance under load
- Catch edge cases

## Trade-Offs

### Real-Time vs Batch

| Aspect | Real-Time | Batch |
|--------|-----------|-------|
| **Latency** | Milliseconds | Hours |
| **Complexity** | High (streaming, state, watermarks) | Low (read, process, write) |
| **Cost** | 2-5× (always running) | 1× (periodic) |
| **Accuracy** | Approximation often needed | Exact |
| **Debugging** | Hard (state, ordering) | Easy (rerun with logs) |
| **Use case** | Act NOW | Understand trends |

### Micro-Batch vs True Streaming

**Spark Structured Streaming (micro-batch)**:
- Pros: Simpler model, Spark ecosystem
- Cons: Minimum 200ms latency, limited state

**Apache Flink (true streaming)**:
- Pros: <10ms latency, sophisticated state
- Cons: Steeper learning curve, smaller ecosystem

## Summary

Real-time analytics enables immediate insights and actions through:

**Core Technologies:**
- **Kafka**: Durable message queue (replay, high throughput)
- **Flink**: True stream processing (<100ms latency, stateful)
- **Druid**: Real-time OLAP (sub-second queries on billions of events)

**Key Concepts:**
- **Windowing**: Tumbling, sliding, session windows
- **Event time**: Process by when events occurred, not when received
- **Watermarks**: Handle out-of-order data
- **State**: Maintain per-key state (fraud rules, user profiles)
- **Approximation**: HyperLogLog, Count-Min Sketch (trade accuracy for speed)

**When to use**:
- Time-sensitive decisions (fraud, trading)
- Operational necessity (can't wait for batch)
- User experience (real-time personalization)

**When NOT to use**:
- Batch is sufficient (overnight reports)
- Complexity not justified
- Perfect accuracy required

Real-time analytics is powerful but complex. Use it when the business value justifies the engineering cost.
`,
  mcqQuizId: 'real-time-analytics-mcq',
  discussionQuizId: 'real-time-analytics-discussion',
};

export default realTimeAnalyticsSection;
