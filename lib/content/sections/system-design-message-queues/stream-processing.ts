/**
 * Stream Processing Section
 */

export const streamprocessingSection = {
  id: 'stream-processing',
  title: 'Stream Processing',
  content: `Stream processing is the practice of processing data in motion—analyzing, transforming, and deriving insights from continuous streams of events in real-time. It\'s fundamental to modern data platforms powering applications from fraud detection to real-time recommendations.

## Stream vs Batch Processing

### **Batch Processing:**

\`\`\`
Collect → Store → Process → Results

Example: Daily sales report
- Collect: Orders throughout the day
- Store: Write to database
- Process: At midnight, run aggregation
- Results: Sales report generated once

Characteristics:
- Process bounded datasets
- High latency (hours, days)
- High throughput
- Simpler programming model
- Good for: Historical analysis, ETL

Technologies: Apache Spark, Hadoop MapReduce
\`\`\`

### **Stream Processing:**

\`\`\`
Continuous → Process → Results (real-time)

Example: Real-time fraud detection
- Continuous: Transaction events streaming
- Process: Analyze each transaction immediately
- Results: Alert within milliseconds if fraud detected

Characteristics:
- Process unbounded datasets (infinite)
- Low latency (milliseconds to seconds)
- Continuous results
- Complex programming model (windowing, state)
- Good for: Real-time analytics, monitoring, alerts

Technologies: Kafka Streams, Apache Flink, Apache Storm
\`\`\`

### **Comparison:**

| Aspect | Batch | Stream |
|--------|-------|--------|
| **Data** | Bounded (finite) | Unbounded (infinite) |
| **Latency** | Hours/days | Milliseconds/seconds |
| **Use Case** | Historical analysis | Real-time insights |
| **Complexity** | Simple | Complex (state, windowing) |
| **Example** | Monthly report | Live dashboard |

---

## Stream Processing Concepts

### **1. Unbounded Data**

Stream data has no defined end:

\`\`\`
Batch Data:
Records: [1, 2, 3, 4, 5]
         ^ Start      ^ End
Process all at once

Stream Data:
Time:    t1   t2   t3   t4   t5   t6   ...   ∞
Records: [1] [2] [3] [4] [5] [6] [...] [...]
         ^ Start                              ^ No end

Process continuously

Challenge: How to aggregate infinite data?
Solution: Windowing
\`\`\`

### **2. Event Time vs Processing Time**

**Event Time**: When event actually occurred
**Processing Time**: When event processed by system

\`\`\`
Example: Mobile app analytics

User clicks button at 10:00 AM (event time)
Phone offline, event buffered
Phone comes online at 2:00 PM
Event processed at 2:01 PM (processing time)

Event time: 10:00 AM
Processing time: 2:01 PM
Lag: 4 hours 1 minute

Why it matters:
- Windowing based on event time → Accurate results
- Windowing based on processing time → Skewed results

Best practice: Use event time for correctness
\`\`\`

### **3. Watermarks**

**Watermark** = Notion of progress in event time (how far we've processed)

\`\`\`
Watermark: "All events with timestamp < T have been processed"

Example:
Current watermark: 10:05
Meaning: All events before 10:05 have arrived (probably)

Late event: Event with timestamp 10:03 arrives at 10:07
→ Event is "late" (past watermark)
→ How to handle? Configure allowed lateness

Perfect watermark: Wait forever (all events arrive eventually)
Heuristic watermark: Wait N seconds (assume events within N seconds)

Tradeoff:
- Wait longer: More complete results, higher latency
- Wait shorter: Faster results, may miss late events
\`\`\`

---

## Windowing

### **Why Windowing?**

\`\`\`
Problem: Aggregate infinite stream

"Count user clicks" → Count from beginning of time? ∞!

Solution: Divide stream into finite windows

"Count user clicks per 5-minute window" → Manageable!
\`\`\`

### **1. Fixed Windows (Tumbling)**

Non-overlapping, fixed-size windows:

\`\`\`
Window size: 5 minutes

Time:  00:00          00:05          00:10          00:15
       |-------------|-------------|-------------|
        Window 1      Window 2      Window 3

Events:
00:01 → Window 1
00:04 → Window 1
00:06 → Window 2
00:09 → Window 2
00:11 → Window 3

Use case: Periodic aggregations, regular intervals
Example: Transactions per minute, error count per hour
\`\`\`

### **2. Sliding Windows**

Overlapping, fixed-size windows:

\`\`\`
Window size: 10 minutes
Slide: 5 minutes

Time:  00:00          00:05          00:10          00:15
       |-------------|
                     |-------------|
                                   |-------------|

Windows:
[00:00 - 00:10]
[00:05 - 00:15]
[00:10 - 00:20]

Event at 00:06:
→ In window [00:00 - 00:10] ✅
→ In window [00:05 - 00:15] ✅

Use case: Moving averages, smoothed metrics
Example: Average response time over last 10 minutes (updated every 5)
\`\`\`

### **3. Session Windows**

Dynamically sized windows based on inactivity gaps:

\`\`\`
Gap: 5 minutes of inactivity

Events:
00:01 - Click      ┐
00:02 - View       ├ Session 1
00:04 - Purchase   ┘
       (5 min gap)
00:11 - Click      ┐
00:13 - View       ├ Session 2
00:14 - Logout     ┘

Session 1: [00:01 - 00:09] (ends 5 min after last event)
Session 2: [00:11 - 00:19]

Use case: User sessions, activity bursts
Example: Web browsing sessions, gaming sessions
\`\`\`

### **4. Global Windows**

Single window for all events (unbounded):

\`\`\`
Used for:
- Non-windowed aggregations
- Explicit triggers (process every N events or N seconds)

Example: Running total, cumulative sum
\`\`\`

---

## Late Data Handling

### **Problem:**

\`\`\`
Window: [00:00 - 00:05]
Watermark: 00:05 (window closed, results emitted)

Late event: timestamp 00:03 arrives at 00:07
→ Window already closed!
→ Results already sent to downstream!

What to do?
\`\`\`

### **Strategies:**

**1. Drop Late Data:**
\`\`\`
Simply ignore late events

Pros: Simple, predictable
Cons: Data loss, inaccurate results

Use case: Acceptable data loss (approximate metrics)
\`\`\`

**2. Allow Lateness:**
\`\`\`
Keep window open for N seconds past watermark

Window: [00:00 - 00:05]
Allowed lateness: 2 minutes
Actually closes: 00:07

Event at 00:03 arrives at 00:06 → Included! ✅
Event at 00:03 arrives at 00:08 → Dropped (too late)

Pros: More complete results
Cons: Delayed output, more state (keep windows open)

Configuration (Flink):
stream
  .windowAll(TumblingEventTimeWindows.of(Time.minutes(5)))
  .allowedLateness(Time.minutes(2))
\`\`\`

**3. Retractions/Updates:**
\`\`\`
Emit initial result, then update if late data arrives

00:05 - Emit result: Count = 100
00:07 - Late event arrives
00:07 - Emit update: Count = 101 (retraction: 100, addition: 101)

Downstream must handle updates

Pros: Timely results + accuracy
Cons: Downstream complexity (handle retractions)
\`\`\`

---

## Stateful Stream Processing

### **Why State?**

\`\`\`
Stateless: Each event processed independently
Example: Filter events (no memory needed)

Stateful: Remember information across events
Example: Count events (must remember count)
\`\`\`

### **State Types:**

**1. Keyed State:**

\`\`\`
State per key (like user_id, device_id)

Example: Count clicks per user
State: {
  "user_123": 5,
  "user_456": 3,
  "user_789": 10
}

Event: user_123 clicks
→ Retrieve state for user_123: 5
→ Increment: 6
→ Store: user_123 → 6
\`\`\`

**2. Operator State:**

\`\`\`
State for entire operator (shared across all keys)

Example: Kafka consumer offsets
State: {
  partition_0: offset 100,
  partition_1: offset 200
}

Used for: Buffering, connectors, sources/sinks
\`\`\`

**3. Broadcast State:**

\`\`\`
State replicated to all parallel instances

Example: Configuration rules
State: {
  fraud_threshold: 1000,
  countries_blacklist: ["XX", "YY"]
}

All operators have copy for evaluation
\`\`\`

### **State Backends:**

**Flink State Backends:**

\`\`\`
1. MemoryStateBackend:
   - In-memory (JVM heap)
   - Fast, limited size
   - Use: Testing, small state

2. FsStateBackend:
   - In-memory + checkpoints to filesystem
   - Moderate size
   - Use: Production (GB of state)

3. RocksDBStateBackend:
   - RocksDB (disk-based)
   - Large state (TB)
   - Use: Very large state
\`\`\`

**Kafka Streams State:**

\`\`\`
RocksDB (local disk) + Changelog topic (Kafka)

State changes written to changelog topic
On restart, restore from changelog
Fault-tolerant, scalable
\`\`\`

---

## Stream Joins

### **1. Stream-Stream Join:**

Join two event streams within a time window:

\`\`\`
Use case: Join ad clicks with ad impressions

Impressions stream:
00:01 - impression_1 (user_123, ad_A)
00:03 - impression_2 (user_456, ad_B)

Clicks stream:
00:02 - click_1 (user_123, ad_A)
00:05 - click_2 (user_456, ad_B)

Join window: 2 minutes

Results:
impression_1 + click_1 → Matched (within 2 min) ✅
impression_2 + click_2 → Matched (within 2 min) ✅

Implementation (Flink):
impressions
  .join (clicks)
  .where (imp -> imp.adId)
  .equalTo (click -> click.adId)
  .window(TumblingEventTimeWindows.of(Time.minutes(2)))
  .apply((imp, click) -> new ImpressionClick (imp, click))
\`\`\`

### **2. Stream-Table Join:**

Enrich stream with reference data:

\`\`\`
Use case: Enrich transactions with user profile

Transaction stream:
{transaction_id, user_id, amount}

User table (slowly changing):
{user_id, name, tier}

Join:
transaction → Lookup user_id in table → Enriched transaction
{transaction_id, user_id, amount, name, tier}

Implementation (Kafka Streams):
KStream<String, Transaction> transactions = ...;
KTable<String, UserProfile> users = ...;

transactions
  .join (users, (transaction, user) -> 
    new EnrichedTransaction (transaction, user))
\`\`\`

### **3. Temporal Join:**

Join with table at specific point in time:

\`\`\`
Use case: Currency conversion at transaction time

Transaction at 10:00: 100 EUR → USD?
→ Lookup EUR/USD rate at 10:00: 1.10
→ Result: 110 USD

Transaction at 14:00: 100 EUR → USD?
→ Lookup EUR/USD rate at 14:00: 1.12
→ Result: 112 USD

Uses historical rate, not latest
\`\`\`

---

## Exactly-Once Processing

### **Challenge:**

\`\`\`
Stream processing failures:

Read message → Process → Write result → Crash before ACK

On restart:
- Read message again (duplicate read)
- Process again (duplicate processing)
- Write result again (duplicate write)

Potential duplicate results ❌
\`\`\`

### **Solution: Exactly-Once Semantics:**

**Kafka Streams:**

\`\`\`
Transactional processing:
1. Read from input topic (offset: 100)
2. Process
3. Write to output topic AND commit offset atomically (transaction)
4. If crash, restart from offset 100
5. Output already written → Deduplicated

Result: Exactly-once ✅

Configuration:
props.put(StreamsConfig.PROCESSING_GUARANTEE_CONFIG, 
          StreamsConfig.EXACTLY_ONCE_V2);
\`\`\`

**Apache Flink:**

\`\`\`
Checkpointing + Two-Phase Commit:
1. Process records
2. Create checkpoint (snapshot of state + positions)
3. Commit checkpoint (two-phase commit with sinks)
4. On failure, restore from last checkpoint

Barriers coordinate checkpoints across parallel streams

Result: Exactly-once ✅

Configuration:
env.enableCheckpointing(60000); // Checkpoint every 60 sec
\`\`\`

---

## Stream Processing Frameworks

### **1. Apache Kafka Streams**

\`\`\`
Type: Library (embedded in application)
Language: Java, Scala
Deployment: Standalone JARs, no cluster

Pros:
✅ Simple deployment (just a library)
✅ Elastic scaling (add instances)
✅ Exactly-once semantics
✅ Tight Kafka integration

Cons:
❌ Kafka-only (no other sources)
❌ Less mature than Flink
❌ Limited ML capabilities

Use case: Kafka-based, simple-moderate complexity
\`\`\`

### **2. Apache Flink**

\`\`\`
Type: Framework (requires cluster)
Language: Java, Scala, Python
Deployment: YARN, Kubernetes, Standalone

Pros:
✅ True streaming (event-by-event)
✅ Advanced windowing
✅ Exactly-once semantics
✅ Rich connectors (Kafka, Kinesis, files, databases)
✅ Powerful state management

Cons:
❌ Complex deployment (cluster needed)
❌ Steeper learning curve
❌ Operational overhead

Use case: Complex stream processing, high scale, low latency
\`\`\`

### **3. Apache Storm**

\`\`\`
Type: Framework (requires cluster)
Language: Java, Python, others
Deployment: YARN, Mesos, Standalone

Pros:
✅ Mature, battle-tested
✅ Low latency (microseconds)
✅ At-least-once / at-most-once

Cons:
❌ No exactly-once (without Trident)
❌ Less active development
❌ Older programming model

Use case: Legacy systems, very low latency
\`\`\`

### **4. Apache Spark Streaming**

\`\`\`
Type: Framework (requires cluster)
Language: Java, Scala, Python
Deployment: YARN, Kubernetes, Standalone

Model: Micro-batching (not true streaming)

Pros:
✅ Unified batch and stream (same API)
✅ Rich ecosystem (ML, SQL, GraphX)
✅ Exactly-once semantics

Cons:
❌ Higher latency (seconds, not milliseconds)
❌ Micro-batching overhead
❌ Complex deployment

Use case: Unified analytics, integration with Spark ecosystem
\`\`\`

---

## Stream Processing in System Design Interviews

### **When to Propose:**

✅ **Real-time analytics** (dashboards, metrics)
✅ **Fraud detection** (immediate alerts)
✅ **Recommendations** (personalized, real-time)
✅ **Monitoring** (anomaly detection)
✅ **IoT** (sensor data processing)

### **Example Discussion:**

\`\`\`
Interviewer: "Design real-time trending topics for Twitter"

You:
"I'll use stream processing for real-time trend detection:

Architecture:
Tweets → Kafka → Flink Stream Processing → Trending API

Stream Processing Pipeline:
1. Source: Kafka topic "tweets"
   - Partition by region (US, EU, APAC)
   - 100K tweets/sec

2. Extract hashtags:
   - FlatMap: Extract #hashtags from tweet text
   - Filter: Ignore common words, spam

3. Window: Tumbling 5-minute windows
   - Count hashtags per window
   - TopN: Keep top 100 hashtags

4. Sink: Write to Redis (trending hashtags)
   - Key: "trending:US:2023-06-15T10:05"
   - Value: [{hashtag: "WorldCup", count: 1500}, ...]

Handling Scale:
- Partition by region (parallel processing)
- Keyed state per hashtag
- RocksDB state backend (millions of hashtags)

Handling Late Data:
- Event time processing (tweet timestamp)
- Watermark: 30 seconds allowed lateness
- Late tweets still counted (acceptable)

Exactly-Once:
- Flink checkpointing (every 60 sec)
- Transactional writes to Redis
- No double-counting

Capacity:
- Input: 100K tweets/sec
- After hashtag extraction: ~200K hashtags/sec (2 per tweet)
- Flink cluster: 20 TaskManagers × 4 cores = 80 parallelism
- 200K / 80 = 2.5K hashtags/core/sec (easily handled)

API Response:
GET /trending?region=US
→ Redis lookup: O(1)
→ Return top 100 hashtags
→ Latency: <10ms

Why Stream Processing:
✅ Real-time (5-minute delay acceptable)
✅ High throughput (100K tweets/sec)
✅ Windowed aggregation (5-min windows)
✅ Stateful (count per hashtag)

vs Batch Processing:
❌ Hourly trends: Too slow (users want real-time)
❌ Spark batch: 10-minute delay minimum
✅ Flink streaming: 5-minute windows, results every 5 min
"
\`\`\`

---

## Key Takeaways

1. **Stream processing = Real-time data processing** → Continuous, unbounded
2. **Event time vs processing time** → Use event time for correctness
3. **Windowing divides infinite streams** → Tumbling, sliding, session
4. **Watermarks track progress** → Allow late data handling
5. **Stateful processing requires state management** → Keyed state, checkpointing
6. **Stream joins correlate events** → Within time windows
7. **Exactly-once semantics** → Kafka Streams, Flink (transactions + checkpoints)
8. **Choose framework based on needs** → Flink (complex), Kafka Streams (simple)
9. **Use for real-time analytics, fraud, monitoring** → Not for batch ETL
10. **In interviews: Discuss windowing, scale, exactly-once** → Show streaming expertise

---

**Next:** We'll explore **Message Schema Evolution**—versioning, compatibility, schema registry, and handling breaking changes in distributed systems.`,
};
