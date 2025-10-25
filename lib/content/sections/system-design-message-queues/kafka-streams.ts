/**
 * Kafka Streams Section
 */

export const kafkastreamsSection = {
  id: 'kafka-streams',
  title: 'Kafka Streams',
  content: `Kafka Streams is a client library for building real-time stream processing applications on top of Kafka. It provides a high-level DSL for transforming, aggregating, and joining data streams without the need for a separate processing cluster like Spark or Flink.

## What is Kafka Streams?

**Kafka Streams** = Lightweight stream processing library + Built on Kafka consumer/producer + No separate cluster required

### **Key Characteristics:**

1. **Library, not framework**: Just add dependency to your application
2. **No external dependencies**: Only Kafka needed (no YARN, Mesos, Kubernetes)
3. **Exactly-once semantics**: Transactional processing
4. **Stateful processing**: Local state stores with fault tolerance
5. **Elastic scaling**: Add/remove instances dynamically
6. **Time-based operations**: Windowing, joins, aggregations

### **Kafka Streams vs Other Processing:**

\`\`\`
Apache Spark Streaming:
- Separate cluster (YARN, Mesos, K8s)
- Micro-batching (latency: seconds)
- Rich ecosystem (ML, SQL)
- Higher operational complexity

Apache Flink:
- Separate cluster
- True streaming (latency: milliseconds)
- Advanced windowing
- Complex deployment

Kafka Streams:
- Embedded library (no separate cluster)
- True streaming (latency: milliseconds)
- Simple deployment (just a JAR)
- Tight Kafka integration
- Lower operational overhead

✅ Use Kafka Streams when:
- Already using Kafka
- Want simple deployment
- Need exactly-once processing
- Moderate complexity (no complex ML)
\`\`\`

---

## Core Abstractions

### **1. KStream (Event Stream)**

A **KStream** is an unbounded stream of records, where each record is an independent event.

\`\`\`
KStream: Continuous events (inserts only)

Time:  t1      t2      t3      t4      t5
Key:   alice   bob     alice   alice   bob
Value: login   click   purchase logout view

Characteristics:
- Each record is independent event
- Same key can appear multiple times
- Represents facts: "User alice logged in at t1"
- Append-only

Example: Click stream, sensor readings, transactions
\`\`\`

**KStream Operations:**

\`\`\`java
// Create KStream from topic
KStream<String, String> clickStream = builder.stream("user-clicks");

// Filter: Keep only premium users
KStream<String, String> premiumClicks = clickStream
    .filter((key, value) -> value.contains("premium"));

// Map: Transform values
KStream<String, Purchase> purchases = clickStream
    .mapValues (value -> parsePurchase (value));

// FlatMap: One record → Multiple records
KStream<String, String> words = textStream
    .flatMapValues (sentence -> Arrays.asList (sentence.split(" ")));

// Peek: Side effect (logging)
clickStream
    .peek((key, value) -> logger.info("Processing: " + value))
    .to("processed-clicks");

// Branch: Split stream based on predicates
KStream<String, String>[] branches = clickStream.branch(
    (key, value) -> value.contains("mobile"),     // Branch 0: mobile
    (key, value) -> value.contains("desktop"),    // Branch 1: desktop
    (key, value) -> true                          // Branch 2: other
);
\`\`\`

### **2. KTable (Changelog Stream)**

A **KTable** is a changelog stream, representing the current state of a table where each record is an update.

\`\`\`
KTable: Current state (upserts and deletes)

Time:  t1           t2          t3           t4          t5
Key:   alice        bob         alice        alice       bob
Value: {city: NY}   {city: SF}  {city: LA}   null        {city: DC}

Current State at t5:
alice: null     (deleted at t4)
bob: {city: DC} (latest update)

Characteristics:
- Each key has latest value
- Updates override previous values
- null value = deletion
- Represents current state: "User alice's current city is LA"
- Like database table with primary key

Example: User profiles, inventory, account balances
\`\`\`

**KTable Operations:**

\`\`\`java
// Create KTable from topic (compacted)
KTable<String, String> userProfiles = builder.table("user-profiles");

// Filter: Keep active users only
KTable<String, String> activeUsers = userProfiles
    .filter((key, value) -> value != null && value.contains("active"));

// MapValues: Transform values
KTable<String, Integer> userAges = userProfiles
    .mapValues (profile -> extractAge (profile));

// Convert to KStream (changelog)
KStream<String, String> profileUpdates = userProfiles.toStream();

// Aggregate (from KStream to KTable)
KTable<String, Long> userCounts = clickStream
    .groupByKey()
    .count();  // Count events per key
\`\`\`

### **3. GlobalKTable**

A **GlobalKTable** is a fully replicated KTable available on all application instances.

\`\`\`
KTable (Partitioned):
Instance 1: Partitions 0, 1 → Subset of data
Instance 2: Partitions 2, 3 → Different subset

GlobalKTable (Replicated):
Instance 1: ALL data from ALL partitions
Instance 2: ALL data from ALL partitions

Use case: Reference data (product catalog, countries, config)
\`\`\`

**Example:**

\`\`\`java
// GlobalKTable: Product catalog (small dataset)
GlobalKTable<String, Product> products = builder.globalTable("products");

// Join KStream with GlobalKTable
KStream<String, Order> orders = builder.stream("orders");

KStream<String, EnrichedOrder> enriched = orders
    .join(
        products,
        (orderId, order) -> order.getProductId(),  // Key extractor
        (order, product) -> new EnrichedOrder (order, product)
    );

✅ Join works even if order and product have different keys
✅ No co-partitioning required
❌ Use only for small datasets (fully replicated)
\`\`\`

---

## Stateful Processing

### **State Stores**

Kafka Streams provides **local state stores** for stateful operations:

\`\`\`
State Store Types:

1. KeyValue Store (RocksDB)
   - Persistent key-value storage
   - Range queries supported
   - Use: Aggregations, joins, caching

2. Window Store (RocksDB)
   - Stores values per time window
   - Use: Windowed aggregations

3. Session Store (RocksDB)
   - Dynamic windows based on activity
   - Use: User sessions, click streams

4. In-Memory Store
   - Fast but not persistent
   - Use: Small datasets, non-critical state

Backend: RocksDB (embedded database)
\`\`\`

### **State Store Architecture:**

\`\`\`
Kafka Streams Application Instance:

┌────────────────────────────────┐
│  Processing Topology           │
│  (Transforms, Aggregations)    │
│            ↓                   │
│  ┌──────────────────────────┐ │
│  │  State Store (RocksDB)   │ │
│  │  {key1: value1,          │ │
│  │   key2: value2}          │ │
│  └──────────────────────────┘ │
│            ↓                   │
│  Changelog Topic (Kafka)       │
│  [Change1, Change2, ...]       │
└────────────────────────────────┘

Fault Tolerance:
1. All state changes written to changelog topic
2. If instance crashes, new instance restores from changelog
3. No data loss

✅ Local state = Fast reads/writes
✅ Changelog = Durability
\`\`\`

### **Aggregations with State:**

\`\`\`java
// Word count example (stateful aggregation)
KStream<String, String> textLines = builder.stream("text-input");

KTable<String, Long> wordCounts = textLines
    // Split into words
    .flatMapValues (line -> Arrays.asList (line.toLowerCase().split("\\s+")))
    
    // Group by word
    .groupBy((key, word) -> word)
    
    // Count occurrences (stateful!)
    .count(Materialized.<String, Long, KeyValueStore<Bytes, byte[]>>as("word-count-store")
        .withKeySerde(Serdes.String())
        .withValueSerde(Serdes.Long()));

// State store tracks count for each word:
// {"hello": 5, "world": 3, "kafka": 7, ...}

// Query state store:
ReadOnlyKeyValueStore<String, Long> store = 
    streams.store("word-count-store", QueryableStoreTypes.keyValueStore());
Long count = store.get("kafka");  // Returns 7
\`\`\`

---

## Windowing Operations

### **Why Windowing?**

In infinite streams, we need bounded time windows for aggregations:

\`\`\`
Problem: "Count user clicks"
Without window: Count from beginning of time (unbounded, keeps growing)
With window: Count per 5-minute window (bounded, manageable)
\`\`\`

### **1. Tumbling Windows (Fixed-Size, Non-Overlapping)**

\`\`\`
Window Size: 5 minutes

Time: 0:00    0:05    0:10    0:15    0:20
      |-------|-------|-------|-------|
      Window1 Window2 Window3 Window4

Event at 0:03 → Window 1
Event at 0:07 → Window 2
Event at 0:12 → Window 3

Each event belongs to exactly one window
\`\`\`

**Example:**

\`\`\`java
KStream<String, Purchase> purchases = builder.stream("purchases");

KTable<Windowed<String>, Double> salesPerUser = purchases
    .groupByKey()
    .windowedBy(TimeWindows.of(Duration.ofMinutes(5)))
    .aggregate(
        () -> 0.0,  // Initial value
        (key, purchase, total) -> total + purchase.getAmount(),
        Materialized.with(Serdes.String(), Serdes.Double())
    );

// Result: Sales per user per 5-minute window
// {(alice, [0:00-0:05]): 150.0, (bob, [0:00-0:05]): 200.0, ...}
\`\`\`

### **2. Hopping Windows (Fixed-Size, Overlapping)**

\`\`\`
Window Size: 10 minutes, Hop: 5 minutes

Time: 0:00    0:05    0:10    0:15    0:20
      |-----------|
            |-----------|
                  |-----------|
                        |-----------|

Event at 0:03 → Window [0:00-0:10] AND [0:00-0:10]
Event at 0:07 → Window [0:00-0:10] AND [0:05-0:15]

Each event can belong to multiple windows
\`\`\`

**Example:**

\`\`\`java
// Moving average: 10-minute window, update every 5 minutes
KTable<Windowed<String>, Double> movingAverage = clickStream
    .groupByKey()
    .windowedBy(TimeWindows.of(Duration.ofMinutes(10))
                           .advanceBy(Duration.ofMinutes(5)))
    .aggregate(
        () -> new AverageTracker(),
        (key, click, tracker) -> tracker.add (click),
        Materialized.with(Serdes.String(), averageTrackerSerde)
    );

// Use case: Smoothed metrics, rolling averages
\`\`\`

### **3. Sliding Windows (Event-Driven)**

\`\`\`
Window Size: 10 minutes

Event at 0:03 → Window [0:03 - 0:13]
Event at 0:07 → Window [0:07 - 0:17]
Event at 0:12 → Window [0:12 - 0:22]

Window starts at each event timestamp
Used for joins (correlate events within time range)
\`\`\`

### **4. Session Windows (Dynamic, Gap-Based)**

\`\`\`
Inactivity Gap: 5 minutes

Events:
0:00 - Click      ┐
0:02 - Click      ├ Session 1 [0:00 - 0:07]
0:05 - Click      ┘
       (5 min gap)
0:12 - Click      ┐ Session 2 [0:12 - 0:15]
0:15 - Click      ┘

Sessions defined by inactivity gaps
Windows merge if events within gap
\`\`\`

**Example:**

\`\`\`java
// User sessions: Group activity with <5min gaps
KTable<Windowed<String>, Long> sessions = clickStream
    .groupByKey()
    .windowedBy(SessionWindows.with(Duration.ofMinutes(5)))
    .count();

// Result: Number of clicks per user session
// {(alice, [0:00-0:07]): 3, (alice, [0:12-0:15]): 2, ...}

// Use case: User sessions, behavior analysis
\`\`\`

---

## Stream Joins

### **1. KStream-KStream Join (Event-to-Event)**

Join two event streams within a time window:

\`\`\`java
// Join clicks with purchases within 10-minute window
KStream<String, Click> clicks = builder.stream("clicks");
KStream<String, Purchase> purchases = builder.stream("purchases");

KStream<String, ClickPurchase> joined = clicks
    .join(
        purchases,
        (click, purchase) -> new ClickPurchase (click, purchase),
        JoinWindows.of(Duration.ofMinutes(10)),
        StreamJoined.with(Serdes.String(), clickSerde, purchaseSerde)
    );

// Example:
// Click at 10:00 joins with Purchase at 10:05 ✅ (within window)
// Click at 10:00 joins with Purchase at 10:15 ❌ (outside window)

// Use case: Attribution (ad click → purchase)
\`\`\`

**Join Types:**

\`\`\`
Inner Join: Only matching records
Left Join: All left records, matched right records (or null)
Outer Join: All records from both sides
\`\`\`

### **2. KStream-KTable Join (Event-to-State)**

Join stream with latest table state:

\`\`\`java
// Enrich orders with latest user profile
KStream<String, Order> orders = builder.stream("orders");
KTable<String, UserProfile> users = builder.table("user-profiles");

KStream<String, EnrichedOrder> enriched = orders
    .join(
        users,
        (order, profile) -> new EnrichedOrder (order, profile)
    );

// Example:
// Order key: "alice" at 10:00
// Joins with latest UserProfile for "alice"
// If profile updated at 9:00, uses that version
// If profile updated at 11:00, previous order still uses 9:00 version

// Use case: Enrichment with reference data
\`\`\`

### **3. KTable-KTable Join (State-to-State)**

Join two tables (triggers on updates):

\`\`\`java
// Join user profiles with account settings
KTable<String, UserProfile> profiles = builder.table("user-profiles");
KTable<String, AccountSettings> settings = builder.table("account-settings");

KTable<String, UserAccount> combined = profiles
    .join(
        settings,
        (profile, setting) -> new UserAccount (profile, setting)
    );

// Triggers:
// - Profile updated → Re-join with current settings
// - Settings updated → Re-join with current profile
// - Result updates in combined table

// Use case: Materialized views, derived tables
\`\`\`

---

## Exactly-Once Semantics

Kafka Streams provides **exactly-once processing** guarantees:

\`\`\`java
Properties props = new Properties();
props.put(StreamsConfig.PROCESSING_GUARANTEE_CONFIG, 
          StreamsConfig.EXACTLY_ONCE_V2);  // Exactly-once

// How it works:
1. Read from input topic (offset: 100)
2. Process record
3. Write to output topic AND commit offset atomically (transaction)
4. If crash before commit, restart from offset 100
5. Process again, but output already written → Idempotent

✅ No duplicates in output
✅ No lost records
✅ No partial state updates

Requirements:
- Kafka 2.5+ (exactly_once_v2)
- Transactional producer
- Read-committed isolation
\`\`\`

---

## Kafka Streams Topology

### **Building a Topology:**

\`\`\`java
StreamsBuilder builder = new StreamsBuilder();

// Source: Read from input topic
KStream<String, String> source = builder.stream("input-topic");

// Processor: Transform
KStream<String, String> filtered = source
    .filter((key, value) -> value != null)
    .mapValues (value -> value.toUpperCase());

// Sink: Write to output topic
filtered.to("output-topic");

// Build topology
Topology topology = builder.build();

// Describe topology (for debugging)
System.out.println (topology.describe());

// Output:
// Topologies:
//   Sub-topology: 0
//     Source: KSTREAM-SOURCE-0000000000 (topics: [input-topic])
//       --> KSTREAM-FILTER-0000000001
//     Processor: KSTREAM-FILTER-0000000001 (stores: [])
//       --> KSTREAM-MAPVALUES-0000000002
//       <-- KSTREAM-SOURCE-0000000000
//     Processor: KSTREAM-MAPVALUES-0000000002 (stores: [])
//       --> KSTREAM-SINK-0000000003
//       <-- KSTREAM-FILTER-0000000001
//     Sink: KSTREAM-SINK-0000000003 (topic: output-topic)
//       <-- KSTREAM-MAPVALUES-0000000002
\`\`\`

---

## Practical Example: Real-Time Analytics

\`\`\`java
// Real-world example: Real-time user activity analytics

StreamsBuilder builder = new StreamsBuilder();

// Input: User activity events
KStream<String, String> activities = builder.stream("user-activities");

// Parse JSON
KStream<String, Activity> parsed = activities
    .mapValues (json -> parseActivity (json));

// Filter: Only purchases
KStream<String, Activity> purchases = parsed
    .filter((key, activity) -> activity.getType().equals("purchase"));

// Aggregate: Total sales per user per 1-hour window
KTable<Windowed<String>, Double> hourlySales = purchases
    .groupBy((key, activity) -> activity.getUserId())
    .windowedBy(TimeWindows.of(Duration.ofHours(1)))
    .aggregate(
        () -> 0.0,
        (userId, activity, total) -> total + activity.getAmount(),
        Materialized.with(Serdes.String(), Serdes.Double())
    );

// Alert: Users with >$1000 purchases in 1 hour
hourlySales
    .toStream()
    .filter((windowedKey, total) -> total > 1000.0)
    .mapValues((windowedKey, total) -> 
        new Alert (windowedKey.key(), total, "High spender"))
    .to("alerts");

// Dashboard: Top spenders (queryable state)
ReadOnlyWindowStore<String, Double> store = 
    streams.store("hourly-sales-store", QueryableStoreTypes.windowStore());

// REST API endpoint to query:
// GET /api/user/{userId}/hourly-sales
@GetMapping("/api/user/{userId}/hourly-sales")
public Map<String, Double> getHourlySales(@PathVariable String userId) {
    Instant now = Instant.now();
    Instant hourAgo = now.minus(Duration.ofHours(1));
    
    WindowStoreIterator<Double> iterator = 
        store.fetch (userId, hourAgo, now);
    
    Map<String, Double> result = new HashMap<>();
    while (iterator.hasNext()) {
        KeyValue<Long, Double> next = iterator.next();
        result.put (new Date (next.key).toString(), next.value);
    }
    return result;
}

✅ Real-time processing (millisecond latency)
✅ Stateful aggregations (hourly totals)
✅ Windowing (tumbling 1-hour windows)
✅ Queryable state (REST API)
✅ Exactly-once semantics
\`\`\`

---

## Kafka Streams Deployment

### **Scaling:**

\`\`\`
Application Instances = Processing Parallelism

Topic: 8 partitions

1 Instance:  Handles all 8 partitions (slow)
2 Instances: Each handles 4 partitions (2× faster)
4 Instances: Each handles 2 partitions (4× faster)
8 Instances: Each handles 1 partition (8× faster)

Max Parallelism = Number of partitions

Adding 9th instance: Idle (no work)

✅ Elastic scaling: Just start/stop instances
✅ No coordination needed (Kafka handles)
\`\`\`

### **Deployment Patterns:**

\`\`\`
1. Standalone JAR:
   java -jar my-streams-app.jar
   ✅ Simple, no dependencies
   ❌ Manual scaling

2. Docker Container:
   docker run my-streams-app
   ✅ Portable
   ✅ Easy scaling (docker-compose scale)

3. Kubernetes:
   kubectl scale deployment my-streams-app --replicas=10
   ✅ Auto-scaling
   ✅ Self-healing
   ✅ Production-grade

4. Serverless (AWS Lambda):
   ❌ Not ideal (Kafka Streams expects long-running process)
   ✅ Use Kafka Connector or Flink instead
\`\`\`

---

## Kafka Streams Best Practices

### **1. Design for Reprocessing:**

\`\`\`java
// Always design idempotent processing
// Kafka Streams may reprocess on rebalance

// Bad: Increment counter
counter++;  // Reprocessing doubles count ❌

// Good: Stateless or idempotent
max = Math.max (max, value);  // Idempotent ✅
\`\`\`

### **2. Choose Appropriate Serdes:**

\`\`\`java
// Use efficient serialization
props.put(StreamsConfig.DEFAULT_KEY_SERDE_CLASS_CONFIG, Serdes.String().getClass());
props.put(StreamsConfig.DEFAULT_VALUE_SERDE_CLASS_CONFIG, Serdes.ByteArray().getClass());

// Custom serdes for complex objects
Serde<MyObject> mySerde = Serdes.serdeFrom (new MySerializer(), new MyDeserializer());
\`\`\`

### **3. Monitor State Store Size:**

\`\`\`
State stores grow unbounded if not managed

Solutions:
- Windowing: Automatically prunes old windows
- Compaction: Keep only latest per key
- TTL: Delete old records (custom logic)

Monitor: Disk usage, restore time
\`\`\`

### **4. Handle Topology Changes Carefully:**

\`\`\`
Changing topology requires:
1. New application.id (fresh start)
   OR
2. Reset tool: kafka-streams-application-reset

// Incompatible changes:
- Changing state store names
- Changing window sizes
- Changing keys

// Compatible changes:
- Adding new branches
- Adding new output topics
\`\`\`

---

## Kafka Streams in System Design Interviews

### **When to Propose Kafka Streams:**

✅ **Real-time stream processing** (low latency)
✅ **Already using Kafka** (tight integration)
✅ **Simple deployment** (no separate cluster)
✅ **Exactly-once required** (financial data)
✅ **Stateful processing** (aggregations, joins, windowing)

### **Trade-offs:**

**Pros:**
- Simple deployment (just a library)
- Exactly-once semantics
- Tight Kafka integration
- Elastic scaling
- Queryable state (REST API)

**Cons:**
- Tied to Kafka (not for non-Kafka sources)
- Less mature than Flink/Spark
- Limited ML capabilities
- State restore can be slow (GB of state)

### **Example Interview Discussion:**

\`\`\`
Interviewer: "Design real-time fraud detection for credit card transactions"

You:
"I'll use Kafka Streams for real-time fraud detection:

Architecture:
Transactions → Kafka Topic → Kafka Streams App → Alerts/Blocked

Kafka Streams Topology:
1. KStream<String, Transaction> from "transactions" topic
2. Group by card_id
3. Window: Tumbling 5-minute windows
4. Aggregate: Count transactions per window
5. Filter: > 10 transactions in 5 min (fraud threshold)
6. Alert: Send to "fraud-alerts" topic

Fraud Rules (multiple sub-topologies):
- Rule 1: >10 transactions in 5 min
- Rule 2: Multiple countries in 1 hour (session window + location tracking)
- Rule 3: Amount >$5000 (high-value threshold)

Enrich with KTable join:
- Join transaction with user_profile (spending history)
- Join with merchant_data (risk score)

Exactly-Once:
- processing.guarantee = exactly_once_v2
- Critical for financial data

Scaling:
- Topic: 20 partitions
- Deploy 20 instances (1 partition each)
- Capacity: 100K transactions/sec

State Management:
- Local RocksDB stores (user history, aggregates)
- Changelog topics (fault tolerance)
- State size: ~10GB per instance (manageable)

Monitoring:
- Stream lag (input vs processing rate)
- State store size
- Alert latency (p99 < 100ms)

Why Kafka Streams over Flink:
- Simpler deployment (no Flink cluster)
- Exactly-once built-in
- Already using Kafka
- Scale sufficient (100K tx/sec)
"
\`\`\`

---

## Key Takeaways

1. **Kafka Streams = Library for stream processing** → No separate cluster
2. **KStream vs KTable** → Events vs State
3. **Stateful processing with local stores** → RocksDB + Changelog topics
4. **Windowing for bounded aggregations** → Tumbling, hopping, sliding, session
5. **Stream joins correlate data** → KStream-KStream, KStream-KTable, KTable-KTable
6. **Exactly-once semantics** → Transactional processing
7. **Elastic scaling** → Add instances = more parallelism
8. **Queryable state** → REST API for real-time queries
9. **Use for real-time processing on Kafka** → Not for batch or non-Kafka sources
10. **In interviews: Discuss topology, windowing, scaling** → Show streaming knowledge

---

**Next:** We'll explore **RabbitMQ**—exchanges, queues, routing, and when to choose RabbitMQ over Kafka.`,
};
