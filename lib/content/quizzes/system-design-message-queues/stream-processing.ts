/**
 * Discussion Questions for Stream Processing
 */

import { QuizQuestion } from '../../../types';

export const streamprocessingQuiz: QuizQuestion[] = [
  {
    id: 'stream-processing-dq-1',
    question:
      'Explain windowing strategies (tumbling, hopping, sliding, session) in stream processing. Design a real-time analytics system for a music streaming service that tracks "songs played in the last hour" and "most played songs in 5-minute windows". Which windowing strategies would you use and why?',
    hint: 'Consider window overlap, state size, and update frequency for each metric.',
    sampleAnswer: `Windowing divides infinite streams into finite chunks for aggregation.

**Windowing Strategies:**

**1. Tumbling Window:**
- Fixed size, non-overlapping
- Example: [0-5min], [5-10min], [10-15min]
- Use: Periodic aggregations

**2. Hopping Window:**
- Fixed size, overlapping (hop < window)
- Example: [0-5min], [2-7min], [4-9min]
- Use: Sliding aggregations

**3. Sliding Window:**
- Continuous, updates on every event
- Example: Last 60 minutes (recalculated per event)
- Use: Real-time continuous metrics

**4. Session Window:**
- Dynamic size based on inactivity gap
- Example: User session (ends after 30min inactivity)
- Use: User behavior analysis

**Music Streaming Analytics:**

\`\`\`java
// Metric 1: Songs played in last hour (Sliding Window)

StreamsBuilder builder = new StreamsBuilder();

KStream<String, SongPlay> plays = builder.stream("song-plays");

// Sliding window: Last 60 minutes
KTable<Windowed<String>, Long> recentPlays = plays
    .groupByKey()
    .windowedBy(TimeWindows.of(Duration.ofMinutes(60)).advanceBy(Duration.ofSeconds(1)))
    .count();

// Problem: Updates every second (high overhead)
// Better: Hopping window (close approximation)

KTable<Windowed<String>, Long> recentPlaysHopping = plays
    .groupByKey()
    .windowedBy(TimeWindows.of(Duration.ofMinutes(60)).advanceBy(Duration.ofMinutes(1)))
    .count();

// Windows: [0-60min], [1-61min], [2-62min], ...
// Updates every 1 minute (60× less overhead) ✅

// Query:
ReadOnlyWindowStore<String, Long> store = streams.store(
    StoreQueryParameters.fromNameAndType("recent-plays-store", QueryableStoreTypes.windowStore())
);

// Get count for last 60 minutes
long count = store.fetch(
    songId,
    Instant.now().minus(60, ChronoUnit.MINUTES),
    Instant.now()
).stream().mapToLong(KeyValue::value).sum();

// Metric 2: Most played songs in 5-minute windows (Tumbling Window)

KTable<Windowed<String>, Long> topSongs = plays
    .groupBy((key, play) -> play.getSongId())
    .windowedBy(TimeWindows.of(Duration.ofMinutes(5)))  // Non-overlapping
    .count();

// Windows: [0-5min], [5-10min], [10-15min]
// Each window independent ✅

// Top 10 songs per window
KStream<Windowed<String>, Long> topSongsStream = topSongs.toStream();

KTable<Windowed<String>, List<Song>> top10 = topSongsStream
    .groupBy(
        (windowedKey, count) -> windowedKey.window(),  // Group by window
        Grouped.with(windowSerde, Serdes.Long())
    )
    .aggregate(
        () -> new ArrayList<>(),
        (window, songCount, topList) -> {
            topList.add(new Song(songCount.getKey(), songCount.getValue()));
            topList.sort((a, b) -> Long.compare(b.getCount(), a.getCount()));
            if (topList.size() > 10) {
                topList.remove(10);  // Keep top 10
            }
            return topList;
        }
    );
\`\`\`

**Why Each Windowing Strategy:**

\`\`\`
Songs played in last hour (Hopping Window):
- Need: Continuous metric (always last 60 minutes)
- Sliding would be perfect but expensive (update per event)
- Hopping approximates sliding with less overhead
- Window: 60 min, Hop: 1 min
- Trade-off: Slight delay (up to 1 min old) but 60× fewer updates

Most played songs in 5-minute windows (Tumbling Window):
- Need: Discrete time periods
- Each 5-min window independent (no overlap)
- Clear reporting (10:00-10:05, 10:05-10:10)
- Tumbling perfect for this ✅

Alternative with Session Windows:
User listening session (Session Window):
- Track songs played per session
- Session = continuous listening
- End session after 30 min inactivity
- Dynamic window size (session can be 5 min or 3 hours)

KTable<Windowed<String>, Long> userSessions = plays
    .groupByKey()  // Group by user_id
    .windowedBy(SessionWindows.with(Duration.ofMinutes(30)))  // 30 min gap
    .count();

// User plays 3 songs:
// 10:00 - Song A
// 10:05 - Song B
// 10:10 - Song C
// 10:50 - Song D (40 min gap → new session)

// Sessions:
// Session 1: [10:00-10:10] (3 songs)
// Session 2: [10:50-10:50] (1 song)

✅ Dynamic windows based on user behavior
\`\`\`

**State Management:**

\`\`\`
Tumbling Window State:
- Store counts for current window only
- Windows: [0-5min], [5-10min]
- State size: 1 window × num_songs
- Memory: O(num_songs)

Hopping Window State:
- Store counts for overlapping windows
- Windows: [0-60min], [1-61min], ..., [59-119min]
- State size: 60 windows × num_songs
- Memory: O(60 × num_songs) = 60× larger ❌

Optimization: Store raw events, aggregate on query
- Store: Events in last 60 minutes
- Query time: Aggregate events
- Memory: O(num_events_per_hour)
- Trade-off: Higher query latency, lower memory

Sliding Window State:
- Update on every event (continuous)
- State size: Events in window
- Memory: O(num_events_in_window)
- Overhead: Recalculate per event ❌

Session Window State:
- Store per-session counts
- Variable window sizes
- Memory: O(num_active_sessions)
- Cleanup: After session gap timeout
\`\`\`

**Late Data Handling:**

\`\`\`
Problem: Event arrives late (out-of-order)

Timeline:
10:00:00 - Event A (timestamp 10:00:00)
10:00:05 - Event B (timestamp 10:00:05)
10:00:10 - Event C (timestamp 10:00:10)
10:00:15 - Event D (timestamp 10:00:02) ← Late! (13 sec delay)

Window: [10:00:00-10:00:10)
Already closed and emitted ❌

Solutions:

1. Allowed Lateness (Flink/Kafka Streams)
builder.stream("song-plays")
    .groupByKey()
    .windowedBy(TimeWindows.of(Duration.ofMinutes(5))
        .grace(Duration.ofSeconds(30)))  // Accept late data up to 30 sec
    .count();

// Window [10:00-10:05) closes at 10:05:30 (with grace period)
// Late events before 10:05:30 included ✅
// Late events after 10:05:30 dropped ❌

2. Watermarks (event time progress indicator)
// Watermark = max(event_time) - allowed_lateness
// Example: Max event time 10:05:00, lateness 30s
// Watermark: 10:04:30

// Window [10:00-10:05) closes when watermark >= 10:05:00
// i.e., when max event time >= 10:05:30

// Late event at 10:00:02 arrives at 10:05:20:
// Watermark: 10:04:50 (10:05:20 - 30s)
// Window still open (10:04:50 < 10:05:00) ✅

// Late event arrives at 10:06:00:
// Watermark: 10:05:30
// Window closed (10:05:30 >= 10:05:00) ❌

3. Retractions (update previous results)
// Window [10:00-10:05) emitted: count = 100
// Late event arrives (in window)
// Emit retraction: count = 101 (updated)

// Downstream consumers handle retractions:
topSongs.toStream().foreach((windowedKey, count) -> {
    if (isRetraction(count)) {
        updateDashboard(windowedKey, count);
    }
});
\`\`\`

**Performance Comparison:**

\`\`\`
Metric: Songs in last hour

Option 1: Sliding Window (update per event)
- Events: 1M plays/hour = 277 events/sec
- Updates: 277 aggregations/sec
- Overhead: High ❌

Option 2: Hopping Window (1-min hop)
- Windows: 60 overlapping windows
- Updates: 60 aggregations/hour = 1/min
- Overhead: Low ✅
- Approximation: Up to 1-min old

Option 3: Tumbling Window (1-min windows)
- Windows: 60 non-overlapping windows
- Query: Sum last 60 windows on read
- Updates: 1 aggregation/min
- Read overhead: Sum 60 values (fast)
- Overhead: Lowest ✅

Recommended: Option 3 (Tumbling 1-min + query-time aggregation)
\`\`\`

**Key Takeaways:**
✅ Tumbling: Non-overlapping, periodic (top songs per 5 min)
✅ Hopping: Overlapping, approximates sliding (songs in last hour)
✅ Sliding: Continuous, expensive (real-time per event)
✅ Session: Dynamic based on gaps (user sessions)
✅ Handle late data with grace period and watermarks`,
    keyPoints: [
      'Tumbling: Fixed non-overlapping windows (periodic aggregations)',
      'Hopping: Overlapping windows (approximates sliding with less overhead)',
      'Sliding: Continuous updates per event (expensive, real-time)',
      'Session: Dynamic windows based on inactivity gap (user behavior)',
      'Use hopping for "last N minutes" metrics (cheaper than sliding)',
      'Handle late data with grace period and watermarks',
    ],
  },
  {
    id: 'stream-processing-dq-2',
    question:
      'Design a stream processing pipeline for fraud detection that processes 100K transactions/sec with stateful pattern matching (detect >5 transactions from same card in different countries within 10 minutes). Include state management, checkpointing, and exactly-once processing guarantees.',
    hint: 'Consider state stores, checkpointing for fault tolerance, and exactly-once semantics to prevent duplicate alerts.',
    sampleAnswer: `Stateful stream processing with exactly-once semantics for fraud detection.

**Architecture:**

\`\`\`
Kafka Topic: transactions (100K/sec)
  ↓
Flink Job: Fraud Detection (stateful)
  - State: Transactions per card (last 10 minutes)
  - Pattern: >5 transactions from different countries
  - Output: Fraud alerts
  ↓
Kafka Topic: fraud-alerts

State Backend: RocksDB (local disk + S3 checkpoints)
Checkpointing: Every 1 minute
Exactly-Once: Kafka transactions
\`\`\`

**Flink Implementation:**

\`\`\`java
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.api.common.state.*;

// Setup
StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

// Enable checkpointing (fault tolerance)
env.enableCheckpointing(60000);  // 1 minute
env.getCheckpointConfig().setCheckpointingMode(CheckpointingMode.EXACTLY_ONCE);
env.getCheckpointConfig().setMinPauseBetweenCheckpoints(30000);  // 30 sec
env.getCheckpointConfig().setCheckpointTimeout(180000);  // 3 min timeout
env.getCheckpointConfig().setMaxConcurrentCheckpoints(1);

// State backend (RocksDB for large state)
env.setStateBackend(new RocksDBStateBackend("s3://checkpoints/fraud-detection"));

// Kafka source (exactly-once)
FlinkKafkaConsumer<Transaction> source = new FlinkKafkaConsumer<>(
    "transactions",
    new TransactionDeserializationSchema(),
    properties
);
source.setStartFromLatest();

DataStream<Transaction> transactions = env.addSource(source);

// Fraud detection (stateful function)
DataStream<FraudAlert> alerts = transactions
    .keyBy(Transaction::getCardId)  // Partition by card_id
    .process(new FraudDetectionFunction());

// Kafka sink (exactly-once)
FlinkKafkaProducer<FraudAlert> sink = new FlinkKafkaProducer<>(
    "fraud-alerts",
    new FraudAlertSerializationSchema(),
    properties,
    FlinkKafkaProducer.Semantic.EXACTLY_ONCE
);

alerts.addSink(sink);

env.execute("Fraud Detection");

// Stateful fraud detection
public class FraudDetectionFunction extends KeyedProcessFunction<String, Transaction, FraudAlert> {
    
    // State: Recent transactions for this card
    private transient ListState<Transaction> recentTransactions;
    
    // State: Countries seen
    private transient MapState<String, Integer> countryCount;  // country → count
    
    // State: Alert deduplication (prevent duplicate alerts)
    private transient ValueState<Long> lastAlertTime;
    
    @Override
    public void open(Configuration parameters) {
        // Initialize state
        recentTransactions = getRuntimeContext().getListState(
            new ListStateDescriptor<>("recentTransactions", Transaction.class)
        );
        
        countryCount = getRuntimeContext().getMapState(
            new MapStateDescriptor<>("countryCount", String.class, Integer.class)
        );
        
        lastAlertTime = getRuntimeContext().getState(
            new ValueStateDescriptor<>("lastAlertTime", Long.class)
        );
    }
    
    @Override
    public void processElement(Transaction tx, Context ctx, Collector<FraudAlert> out) throws Exception {
        String cardId = tx.getCardId();
        long now = tx.getTimestamp();
        
        // Add current transaction to state
        recentTransactions.add(tx);
        
        // Update country count
        String country = tx.getCountry();
        Integer count = countryCount.get(country);
        countryCount.put(country, (count == null ? 0 : count) + 1);
        
        // Clean up old transactions (>10 minutes)
        List<Transaction> validTransactions = new ArrayList<>();
        int totalCount = 0;
        
        for (Transaction t : recentTransactions.get()) {
            if (now - t.getTimestamp() <= 600_000) {  // 10 minutes
                validTransactions.add(t);
                totalCount++;
            } else {
                // Remove from country count
                String oldCountry = t.getCountry();
                Integer oldCount = countryCount.get(oldCountry);
                if (oldCount != null) {
                    if (oldCount == 1) {
                        countryCount.remove(oldCountry);
                    } else {
                        countryCount.put(oldCountry, oldCount - 1);
                    }
                }
            }
        }
        
        // Update state with valid transactions
        recentTransactions.update(validTransactions);
        
        // Fraud detection: >5 transactions from >2 countries
        int uniqueCountries = 0;
        for (String c : countryCount.keys()) {
            uniqueCountries++;
        }
        
        if (totalCount >= 5 && uniqueCountries >= 3) {
            // Fraud detected!
            
            // Deduplication: Only alert once per 10 minutes
            Long lastAlert = lastAlertTime.value();
            if (lastAlert == null || now - lastAlert > 600_000) {
                // Send alert
                FraudAlert alert = new FraudAlert(
                    cardId,
                    "MULTI_COUNTRY_FRAUD",
                    totalCount + " transactions in " + uniqueCountries + " countries",
                    validTransactions,
                    now
                );
                
                out.collect(alert);
                
                // Update last alert time
                lastAlertTime.update(now);
            }
        }
        
        // Register timer to clean up state (10 minutes)
        ctx.timerService().registerEventTimeTimer(now + 600_000);
    }
    
    @Override
    public void onTimer(long timestamp, OnTimerContext ctx, Collector<FraudAlert> out) throws Exception {
        // Timer fired: Clean up expired transactions
        long now = timestamp;
        
        List<Transaction> validTransactions = new ArrayList<>();
        for (Transaction t : recentTransactions.get()) {
            if (now - t.getTimestamp() <= 600_000) {
                validTransactions.add(t);
            }
        }
        
        recentTransactions.update(validTransactions);
        
        // Rebuild country count
        countryCount.clear();
        for (Transaction t : validTransactions) {
            String country = t.getCountry();
            Integer count = countryCount.get(country);
            countryCount.put(country, (count == null ? 0 : count) + 1);
        }
    }
}
\`\`\`

**State Management:**

\`\`\`
State per card:
- Recent transactions (last 10 minutes): List<Transaction>
- Country counts: Map<String, Integer>
- Last alert time: Long

State size estimation:
- 100K tx/sec × 600 sec = 60M transactions in window
- Average: 10 transactions per card
- Cards: 60M / 10 = 6M cards
- State per card: ~1 KB (10 transactions × 100 bytes)
- Total state: 6M × 1 KB = 6 GB

RocksDB stores state on local disk (SSD)
Periodic snapshots to S3 (checkpoints)
\`\`\`

**Checkpointing (Fault Tolerance):**

\`\`\`
Timeline:
T0: Processing transactions (state accumulating)
T60: Checkpoint triggered
  - Pause processing
  - Snapshot state to S3 (all cards' state)
  - Record Kafka offsets
  - Resume processing
T120: Checkpoint triggered again

State snapshots:
s3://checkpoints/fraud-detection/
  chk-1/  (T60 snapshot)
    state/
      card-state-shard-0
      card-state-shard-1
      ...
    offsets
      partition-0-offset-1234567
      partition-1-offset-2345678
      ...
  chk-2/  (T120 snapshot)
    ...

Failure recovery:
T90: Flink task crashes
T91: Flink restarts task
  - Load state from last checkpoint (chk-1 at T60)
  - Restore Kafka offsets (partition-0-offset-1234567)
  - Resume processing from T60
  - Reprocess T60-T90 transactions

Exactly-Once guarantee:
- No transactions lost (restart from checkpoint offset)
- No duplicate alerts (idempotent state updates)
- State consistent with input ✅

Checkpoint overhead:
- State size: 6 GB
- Snapshot time: ~30 seconds (6 GB / 200 MB/s)
- Frequency: 1 minute (30s / 60s = 50% overhead) ❌

Optimization:
- Incremental checkpoints (only changes)
- First checkpoint: 6 GB
- Subsequent: ~500 MB (changes only)
- Snapshot time: ~2.5 seconds ✅
\`\`\`

**Exactly-Once Processing:**

\`\`\`
How exactly-once works:

1. Atomic read-process-write:
   - Read transaction from Kafka (offset 1234)
   - Process (update state, detect fraud)
   - Write alert to Kafka (offset 5678)
   - Commit Kafka offsets atomically

2. Kafka transactions:
producer.initTransactions();
producer.beginTransaction();

// Write alert
producer.send(new ProducerRecord<>("fraud-alerts", alert));

// Commit input offset
producer.sendOffsetsToTransaction(offsets, consumerGroupId);

producer.commitTransaction();

// All or nothing: Alert written + offset committed, or neither ✅

3. Failure scenarios:

Scenario A: Crash before commit
- Transaction not committed
- Alert not visible to consumers
- Kafka offset not advanced
- Restart processes same transaction again
- Idempotent state update (same result)
- Alert sent again, transaction committed ✅

Scenario B: Crash after commit
- Transaction committed
- Alert visible
- Kafka offset advanced
- Restart skips this transaction ✅

Result: Exactly-once (no duplicates, no loss)

4. State consistency:
- State checkpointed with Kafka offsets
- On recovery, state and offsets restored together
- State matches input data ✅
\`\`\`

**Scalability:**

\`\`\`
Throughput: 100K transactions/sec

Single Flink task:
- Max: ~10K tx/sec (RocksDB state operations)
- Need: 100K / 10K = 10 tasks (parallelism)

Kafka partitions: 10 (match Flink parallelism)

Flink cluster:
- TaskManagers: 5 (2 tasks each)
- Slots per TM: 2
- Total slots: 10 ✅

Per-task state:
- Total state: 6 GB
- Per task: 6 GB / 10 = 600 MB
- RocksDB local disk: 1 TB SSD per TM
- Memory: 4 GB per task (RocksDB cache)

Checkpointing:
- Incremental checkpoints: 50 MB per task
- Total: 50 MB × 10 = 500 MB
- Time: 500 MB / 200 MB/s = 2.5 seconds
- Frequency: 1 minute (2.5s / 60s = 4% overhead) ✅

Capacity:
- 10 tasks × 10K tx/sec = 100K tx/sec ✅
- Headroom: 20% (120K tx/sec capacity)
\`\`\`

**Monitoring:**

\`\`\`
Key metrics:
1. Checkpoint duration (alert if >30 sec)
2. State size per task (alert if >1 GB)
3. Processing latency (alert if >1 sec)
4. Backpressure (alert if >10%)
5. Kafka lag (alert if >100K messages)

Dashboard:
- Throughput: 100K tx/sec
- Latency: P99 500ms
- Checkpoint time: 2.5 sec
- State size: 6 GB (600 MB per task)
- Fraud alerts: 50/sec
\`\`\`

**Key Takeaways:**
✅ Keyed state partitioned by card_id (parallel processing)
✅ RocksDB state backend (large state, disk-based)
✅ Checkpointing every 1 minute (fault tolerance)
✅ Incremental checkpoints (low overhead)
✅ Exactly-once with Kafka transactions
✅ Scalability: 10 tasks handle 100K tx/sec`,
    keyPoints: [
      'Keyed state partitioned by card_id enables parallel processing',
      'RocksDB state backend for large state (6 GB, disk-based)',
      'Checkpointing (1-min interval) snapshots state to S3 for fault tolerance',
      'Incremental checkpoints reduce overhead (only changes, not full state)',
      'Exactly-once semantics via Kafka transactions (atomic read-process-write)',
      'Scale horizontally: 10 Flink tasks × 10K tx/sec = 100K tx/sec capacity',
    ],
  },
  {
    id: 'stream-processing-dq-3',
    question:
      'Compare stream processing frameworks (Kafka Streams, Apache Flink, Apache Spark Streaming). For each use case, which would you choose: (1) Simple aggregations within Kafka, (2) Complex CEP with ML inference, (3) Micro-batch processing for cost, (4) Event-time processing with late data. Explain trade-offs.',
    hint: 'Consider deployment complexity, processing model, exactly-once, state management, and latency requirements.',
    sampleAnswer: `Stream processing frameworks offer different trade-offs.

**Framework Comparison:**

| Feature | Kafka Streams | Flink | Spark Streaming |
|---------|---------------|-------|------------------|
| **Deployment** | Library (JAR) | Cluster | Cluster |
| **Model** | Event-by-event | Event-by-event | Micro-batch |
| **Latency** | Low (ms) | Very low (ms) | Higher (sec) |
| **Throughput** | High (100K/s) | Very high (1M+/s) | Very high (1M+/s) |
| **State** | RocksDB + Kafka | RocksDB + Checkpoints | Memory + Checkpoints |
| **Exactly-Once** | Yes (Kafka) | Yes (Checkpoints) | Yes (Checkpoints) |
| **Complexity** | Low | Medium | Medium |

**Use Case 1: Simple aggregations within Kafka**

**Choice: Kafka Streams ✅**

**Why:**
- Already using Kafka (no new cluster)
- Simple deployment (add library)
- Event-by-event processing (low latency)
- Native Kafka integration
- Low ops overhead

**Example:** Count clicks per product

\`\`\`java
StreamsBuilder builder = new StreamsBuilder();

KStream<String, Click> clicks = builder.stream("clicks");

KTable<String, Long> clickCounts = clicks
    .groupBy((key, click) -> click.getProductId())
    .count();

clickCounts.toStream().to("click-counts");

// Deploy:
java -jar click-aggregator.jar

// That's it! No cluster management ✅
\`\`\`

**Use Case 2: Complex CEP with ML inference**

**Choice: Apache Flink ✅**

**Why:**
- Advanced CEP library (pattern matching)
- Low latency (event-by-event)
- Large state support (RocksDB)
- ML inference integration
- Powerful windowing

**Example:** Detect fraudulent patterns + ML scoring

\`\`\`java
// CEP pattern
Pattern<Transaction, ?> pattern = Pattern.<Transaction>begin("start")
    .where(new SimpleCondition<Transaction>() {
        @Override
        public boolean filter(Transaction tx) {
            return tx.getAmount() > 1000;  // Large transaction
        }
    })
    .next("next")
    .where(new SimpleCondition<Transaction>() {
        @Override
        public boolean filter(Transaction tx) {
            return tx.getCountry() != getPreviousCountry();  // Different country
        }
    })
    .within(Time.minutes(10));

PatternStream<Transaction> patternStream = CEP.pattern(transactions, pattern);

DataStream<FraudAlert> alerts = patternStream.select(
    new PatternSelectFunction<Transaction, FraudAlert>() {
        @Override
        public FraudAlert select(Map<String, List<Transaction>> pattern) {
            Transaction t1 = pattern.get("start").get(0);
            Transaction t2 = pattern.get("next").get(0);
            
            // ML inference
            double fraudScore = mlModel.predict(t1, t2);
            
            if (fraudScore > 0.8) {
                return new FraudAlert(t1, t2, fraudScore);
            }
            return null;
        }
    }
);
\`\`\`

**Why not Kafka Streams?**
- No built-in CEP library
- Manual pattern matching (complex)

**Why not Spark?**
- Micro-batch model (higher latency)
- CEP harder to implement

**Use Case 3: Micro-batch processing for cost**

**Choice: Apache Spark Streaming ✅**

**Why:**
- Micro-batches reduce costs (batch operations cheaper)
- Share cluster with batch jobs (cost efficiency)
- Familiar Spark API
- Good for less latency-sensitive workloads

**Example:** Aggregate logs every 30 seconds

\`\`\`python
from pyspark.sql import SparkSession
from pyspark.sql.functions import window, count

spark = SparkSession.builder.appName("LogAggregation").getOrCreate()

logs = spark.readStream.format("kafka") \\
    .option("kafka.bootstrap.servers", "localhost:9092") \\
    .option("subscribe", "logs") \\
    .load()

# Micro-batch: 30 seconds
aggregated = logs.groupBy(
    window(logs.timestamp, "30 seconds"),
    logs.level
).agg(count("*").alias("count"))

query = aggregated.writeStream \\
    .outputMode("complete") \\
    .format("console") \\
    .trigger(processingTime="30 seconds") \\  # Micro-batch interval
    .start()

# Cost savings:
# Flink: Continuous processing (always running)
# Spark: Micro-batches (can use spot instances)
# 30% cost reduction ✅
\`\`\`

**Why not Flink?**
- Continuous processing (no batching benefits)
- Higher cost (always running)

**Why not Kafka Streams?**
- Event-by-event (no batching)

**Use Case 4: Event-time processing with late data**

**Choice: Apache Flink ✅**

**Why:**
- Best event-time support (watermarks)
- Flexible late data handling
- Allowed lateness configuration
- Retractions (update previous results)

**Example:** Aggregate events with late arrivals

\`\`\`java
DataStream<Event> events = ...; // Kafka source

events
    .assignTimestampsAndWatermarks(
        WatermarkStrategy
            .<Event>forBoundedOutOfOrderness(Duration.ofSeconds(30))  // 30 sec lateness
            .withTimestampAssigner((event, timestamp) -> event.getTimestamp())
    )
    .keyBy(Event::getUserId)
    .window(TumblingEventTimeWindows.of(Time.minutes(5)))
    .allowedLateness(Time.minutes(1))  // Accept late data up to 1 min after window closes
    .sum("value")
    .addSink(...);

// Behavior:
// Window [10:00-10:05) processes events with timestamps in that range
// Watermark: max(event_time) - 30 sec
// Window closes when watermark >= 10:05:00 (i.e., event time >= 10:05:30)
// Late data accepted until 10:06:00 (allowed lateness)
// Late data after 10:06:00 dropped ❌

// Flink handles this elegantly ✅
\`\`\`

**Why not Spark?**
- Less flexible watermarking
- No allowed lateness (late data dropped immediately)

**Why not Kafka Streams?**
- Watermarking less powerful
- Late data handling less flexible

**Trade-offs:**

\`\`\`
Kafka Streams:
Pros:
✅ Simple deployment (library, no cluster)
✅ Low ops overhead
✅ Native Kafka integration
✅ Good for simple to moderate complexity

Cons:
❌ Limited to Kafka ecosystem
❌ No advanced CEP
❌ Less scalable than Flink/Spark

Flink:
Pros:
✅ Very low latency (event-by-event)
✅ Advanced features (CEP, late data)
✅ Large state support (TB+)
✅ Exactly-once with any source

Cons:
❌ Cluster management overhead
❌ Steeper learning curve
❌ Higher operational complexity

Spark Streaming:
Pros:
✅ Unified batch + streaming
✅ Cost-effective (micro-batches)
✅ Mature ecosystem
✅ Good for less latency-sensitive

Cons:
❌ Higher latency (seconds)
❌ Micro-batch model limitations
❌ Less flexible event-time handling

**Decision Matrix:**

Kafka-centric + Simple → Kafka Streams
Complex CEP + Low latency → Flink
Cost optimization + Batch jobs → Spark
Event-time + Late data → Flink
\`\`\`

**Key Takeaways:**
✅ Kafka Streams: Simple, library-based, Kafka-centric
✅ Flink: Advanced features, low latency, complex CEP
✅ Spark: Micro-batch, cost-effective, unified batch/streaming
✅ Choose based on: Complexity, latency, cost, ecosystem`,
    keyPoints: [
      'Kafka Streams: Simple deployment (library), Kafka-centric, low ops',
      'Flink: Advanced CEP, very low latency, best event-time handling',
      'Spark Streaming: Micro-batch (cost-effective), unified batch/streaming',
      'Choose Kafka Streams for simple aggregations within Kafka',
      'Choose Flink for complex CEP, ML inference, event-time with late data',
      'Choose Spark for cost optimization and less latency-sensitive workloads',
    ],
  },
];
