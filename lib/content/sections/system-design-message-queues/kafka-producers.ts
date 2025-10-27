/**
 * Kafka Producers Section
 */

export const kafkaproducersSection = {
  id: 'kafka-producers',
  title: 'Kafka Producers',
  content: `Kafka producers are responsible for publishing records to Kafka topics. Understanding producer configuration, partitioning strategies, idempotence, and performance tuning is crucial for building reliable, high-throughput systems.

## Producer Architecture

### **Basic Producer Flow:**

\`\`\`
Application Code
      ↓
  Producer API
      ↓
Serializer (Key & Value)
      ↓
Partitioner (Which partition?)
      ↓
Record Accumulator (Batching)
      ↓
Sender Thread (Network I/O)
      ↓
Kafka Broker (Leader)
      ↓
Replication to Followers
      ↓
Acknowledgment (acks)
\`\`\`

### **Producer Components:**1. **Serializer**: Converts objects to bytes
2. **Partitioner**: Determines target partition
3. **Record Accumulator**: Batches records
4. **Sender**: Asynchronous network I/O
5. **Metadata**: Caches cluster topology

---

## Sending Messages

### **1. Fire-and-Forget (Async)**

Send and don't wait for response:

\`\`\`java
ProducerRecord<String, String> record = 
    new ProducerRecord<>("topic", "key", "value");

producer.send (record);  // Fire and forget

// ❌ No error handling
// ✅ Highest throughput
// ✅ Lowest latency
\`\`\`

**Use Case:**
- Metrics, logs (loss acceptable)
- Non-critical events

### **2. Synchronous Send**

Wait for acknowledgment:

\`\`\`java
ProducerRecord<String, String> record = 
    new ProducerRecord<>("orders", "order_123", orderData);

try {
    RecordMetadata metadata = producer.send (record).get();
    System.out.println("Sent to partition: " + metadata.partition() + 
                       " offset: " + metadata.offset());
} catch (Exception e) {
    System.err.println("Failed to send: " + e);
}

// ✅ Guaranteed delivery before proceeding
// ❌ Low throughput (blocks on each send)
// ✅ Error handling
\`\`\`

**Use Case:**
- Critical messages (payments)
- When you need immediate confirmation
- Testing, debugging

### **3. Asynchronous Send with Callback**

Best of both worlds:

\`\`\`java
ProducerRecord<String, String> record = 
    new ProducerRecord<>("user-activity", userId, activityData);

producer.send (record, new Callback() {
    @Override
    public void onCompletion(RecordMetadata metadata, Exception e) {
        if (e != null) {
            System.err.println("Error: " + e);
            // Handle error (retry, log, alert)
        } else {
            System.out.println("Sent to partition: " + metadata.partition());
        }
    }
});

// ✅ High throughput (non-blocking)
// ✅ Error handling via callback
// ✅ Production-ready pattern
\`\`\`

**Use Case:**
- Most production scenarios
- High throughput with error handling

---

## Partitioning Strategies

### **Why Partition Keys Matter:**

\`\`\`
Topic: "user-activity" (3 partitions)

Without Key (Round-Robin):
Message 1 (user_123) → Partition 0
Message 2 (user_123) → Partition 1  ← Different partition!
Message 3 (user_123) → Partition 2  ← Different partition!

❌ No ordering guarantee for user_123

With Key (Hash Partitioning):
Message 1 (key: user_123) → Partition 1
Message 2 (key: user_123) → Partition 1  ← Same partition!
Message 3 (key: user_123) → Partition 1  ← Same partition!

✅ Ordering guaranteed for user_123
\`\`\`

### **1. Default Partitioner (Hash-Based)**

\`\`\`java
// Partitioning Logic:
partition = hash (key) % numPartitions

Example:
Topic: "orders", 4 partitions

Order 1 (key: "customer_A") → hash("customer_A") % 4 = 2 → Partition 2
Order 2 (key: "customer_B") → hash("customer_B") % 4 = 1 → Partition 1
Order 3 (key: "customer_A") → hash("customer_A") % 4 = 2 → Partition 2

Same key → Same partition → Ordered
\`\`\`

**Characteristics:**
- Uniform distribution (if keys uniformly distributed)
- Ordering within key
- Load balancing

**When to Use:**
- Need ordering per key (user, order, session)
- Keys evenly distributed

### **2. Round-Robin Partitioner (No Key)**

\`\`\`java
ProducerRecord<String, String> record = 
    new ProducerRecord<>("topic", null, "value");  // No key

// Partitioning:
Msg 1 → Partition 0
Msg 2 → Partition 1
Msg 3 → Partition 2
Msg 4 → Partition 0  (wraps around)

✅ Uniform load distribution
❌ No ordering guarantee
\`\`\`

**When to Use:**
- Ordering not required
- Maximum throughput
- Independent events (metrics)

### **3. Custom Partitioner**

Implement custom logic:

\`\`\`java
public class CustomPartitioner implements Partitioner {
    @Override
    public int partition(String topic, Object key, byte[] keyBytes,
                         Object value, byte[] valueBytes, Cluster cluster) {
        int numPartitions = cluster.partitionCountForTopic (topic);
        
        // Custom logic: VIP customers to partition 0
        if (key.toString().startsWith("VIP_")) {
            return 0;  // Dedicated partition for VIPs
        }
        
        // Others: Hash-based
        return Math.abs(Utils.murmur2(keyBytes)) % numPartitions;
    }
}

// Configuration:
props.put("partitioner.class", "com.example.CustomPartitioner");
\`\`\`

**Use Cases:**
- VIP customers (dedicated resources)
- Geographic partitioning (US, EU, APAC)
- Priority-based routing
- Hot key mitigation

### **4. Explicit Partition**

Specify partition directly:

\`\`\`java
// Send to specific partition
ProducerRecord<String, String> record = 
    new ProducerRecord<>("topic", 2, "key", "value");
                              ↑
                          Partition 2

// Use case: Manual partition assignment
\`\`\`

---

## Producer Configuration

### **Essential Settings:**

\`\`\`java
Properties props = new Properties();

// 1. Bootstrap Servers (Required)
props.put("bootstrap.servers", "broker1:9092,broker2:9092,broker3:9092");
// Initial connection to cluster
// Discovers full cluster topology

// 2. Serializers (Required)
props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");
// Convert objects to bytes

// 3. Acknowledgments (acks)
props.put("acks", "all");  // Wait for all ISR
// Options: 0 (none), 1 (leader), all/-1 (all ISR)
// Trade-off: Durability vs Latency

// 4. Retries
props.put("retries", 3);  // Retry on transient errors
props.put("retry.backoff.ms", 100);  // Wait between retries

// 5. Idempotence (Prevent duplicates)
props.put("enable.idempotence", "true");
// Exactly-once semantics (within partition)

// 6. Compression
props.put("compression.type", "snappy");
// Options: none, gzip, snappy, lz4, zstd
// Trade-off: CPU vs Network/Storage

// 7. Batching (Performance)
props.put("batch.size", 16384);  // 16KB batch
props.put("linger.ms", 10);      // Wait 10ms to fill batch
// Trade-off: Latency vs Throughput

// 8. Buffer Memory
props.put("buffer.memory", 33554432);  // 32MB buffer
// Memory for unsent messages

// 9. Request Timeout
props.put("request.timeout.ms", 30000);  // 30 seconds
// Max wait for broker response
\`\`\`

---

## Idempotent Producer

### **Problem: Duplicates on Retry**

\`\`\`
Scenario (Without Idempotence):
Producer sends message → Leader receives → Leader crashes before ACK
                                         ↓
Producer retries → New leader receives (DUPLICATE!)

Result: Message written twice ❌
\`\`\`

### **Solution: Idempotence**

\`\`\`java
props.put("enable.idempotence", "true");

// How it works:
Producer assigns:
- Producer ID (PID)
- Sequence number per partition

Message: { PID: 12345, Seq: 0, data: "..." }
Next:    { PID: 12345, Seq: 1, data: "..." }

Broker:
- Tracks (PID, Seq) per partition
- If duplicate seq, rejects (already written)
- Returns success to producer

Result: Exactly-once per partition ✅
\`\`\`

**Automatic Settings with Idempotence:**

\`\`\`java
When enable.idempotence = true:

// These are automatically set:
acks = all             // Wait for all ISR
retries = MAX_INT      // Retry until success
max.in.flight.requests.per.connection = 5  // Pipelining

✅ Exactly-once within partition
✅ Ordering preserved
✅ No application changes needed
\`\`\`

**When to Enable:**
- ✅ All production systems (low overhead)
- ✅ Critical data (payments, orders)
- ✅ Want exactly-once per partition
- ❌ Only if older Kafka (< 0.11) - upgrade instead!

---

## Transactions

For **exactly-once across multiple partitions/topics**:

\`\`\`java
Properties props = new Properties();
props.put("enable.idempotence", "true");
props.put("transactional.id", "order-processor-1");  // Unique ID

KafkaProducer<String, String> producer = new KafkaProducer<>(props);
producer.initTransactions();

try {
    producer.beginTransaction();
    
    // Send to multiple topics/partitions atomically
    producer.send (new ProducerRecord<>("orders", "order_123", orderData));
    producer.send (new ProducerRecord<>("inventory", "item_456", inventoryUpdate));
    producer.send (new ProducerRecord<>("analytics", "event_789", analyticsEvent));
    
    producer.commitTransaction();  // All succeed or none
    
} catch (Exception e) {
    producer.abortTransaction();  // Rollback all
}

// ✅ All messages written atomically
// ✅ Exactly-once across topics
// ❌ Higher latency (two-phase commit)
\`\`\`

**Use Cases:**
- Exactly-once stream processing (Kafka Streams)
- Atomic multi-topic writes
- Read-process-write patterns

---

## Batching and Compression

### **Batching:**

\`\`\`java
// Without Batching:
Message 1 → Network call (100μs)
Message 2 → Network call (100μs)
Message 3 → Network call (100μs)
Total: 300μs, 3 network calls

// With Batching:
Message 1 ┐
Message 2 ├→ Network call (120μs)
Message 3 ┘
Total: 120μs, 1 network call

Throughput: 3× higher!
\`\`\`

**Batching Configuration:**

\`\`\`java
// Batch by size:
props.put("batch.size", 16384);  // 16KB
// Send when batch reaches 16KB

// Batch by time:
props.put("linger.ms", 10);  // 10ms
// Send after 10ms even if batch not full

// Trade-off:
batch.size = 1, linger.ms = 0   → Low latency, low throughput
batch.size = 64KB, linger.ms = 100 → High latency, high throughput

Production typical:
batch.size = 16KB-64KB
linger.ms = 10-50ms
\`\`\`

### **Compression:**

\`\`\`java
props.put("compression.type", "snappy");

// Compression Comparison:

| Type   | Ratio | CPU   | Speed | Use Case |
|--------|-------|-------|-------|----------|
| none   | 1×    | None  | N/A   | Low CPU, high bandwidth |
| gzip   | 4-6×  | High  | Slow  | Bandwidth-constrained |
| snappy | 2-3×  | Low   | Fast  | Balanced (default choice) |
| lz4    | 2-3×  | Low   | Fast  | Similar to snappy |
| zstd   | 3-5×  | Med   | Fast  | Best overall (Kafka 2.1+) |

Bandwidth Savings:
No compression: 100MB/s
Snappy (3×):    33MB/s  ✅ 67% reduction
\`\`\`

**Best Practice:**
\`\`\`
For most workloads:
- compression.type = snappy or zstd
- Compress at producer (not broker)
- Entire batch compressed (better ratio)
\`\`\`

---

## Producer Performance Tuning

### **1. Increase Throughput:**

\`\`\`java
// Maximize throughput:
props.put("batch.size", 65536);  // 64KB (larger batches)
props.put("linger.ms", 50);      // Wait longer
props.put("compression.type", "lz4");  // Fast compression
props.put("acks", "1");          // Leader-only (if acceptable)
props.put("buffer.memory", 67108864);  // 64MB buffer

// Result: Higher throughput, higher latency
\`\`\`

### **2. Minimize Latency:**

\`\`\`java
// Minimize latency:
props.put("batch.size", 0);      // No batching
props.put("linger.ms", 0);       // Send immediately
props.put("compression.type", "none");  // No compression
props.put("acks", "1");          // Leader-only

// Result: Low latency, low throughput
\`\`\`

### **3. Balanced (Production):**

\`\`\`java
// Production-ready:
props.put("acks", "all");        // Durability
props.put("enable.idempotence", "true");  // Exactly-once
props.put("compression.type", "snappy");  // Balanced compression
props.put("batch.size", 16384);  // 16KB
props.put("linger.ms", 10);      // 10ms wait
props.put("retries", 3);
props.put("max.in.flight.requests.per.connection", 5);

// Result: Good throughput, acceptable latency, reliable
\`\`\`

---

## Error Handling

### **1. Retriable Errors:**

\`\`\`
LEADER_NOT_AVAILABLE       → Retry (leader election in progress)
NOT_ENOUGH_REPLICAS        → Retry (replicas catching up)
NETWORK_EXCEPTION          → Retry (transient network issue)

Producer automatically retries (if retries > 0)
\`\`\`

### **2. Non-Retriable Errors:**

\`\`\`
INVALID_TOPIC_EXCEPTION    → Don't retry (topic doesn't exist)
RECORD_TOO_LARGE           → Don't retry (message exceeds max.message.bytes)
UNSUPPORTED_VERSION        → Don't retry (incompatible versions)

Producer throws exception → Application must handle
\`\`\`

### **Error Handling Pattern:**

\`\`\`java
producer.send (record, (metadata, exception) -> {
    if (exception != null) {
        if (exception instanceof RetriableException) {
            // Already retried, still failed
            logger.error("Failed after retries: " + exception);
            // Send to DLQ (Dead Letter Queue)
            sendToDLQ(record);
        } else {
            // Non-retriable error
            logger.error("Non-retriable error: " + exception);
            // Alert, investigate
        }
    } else {
        // Success
        logger.info("Sent: partition=" + metadata.partition() + 
                    " offset=" + metadata.offset());
    }
});
\`\`\`

---

## Producer Monitoring

### **Key Metrics:**

\`\`\`
1. record-send-rate
   - Messages sent per second
   - Track throughput

2. record-error-rate
   - Failed sends per second
   - Alert if > threshold

3. request-latency-avg
   - Average request latency
   - Detect slowdowns

4. batch-size-avg
   - Average batch size
   - Optimize batching

5. buffer-available-bytes
   - Free buffer memory
   - Alert if low (backpressure)

6. record-retry-rate
   - Retries per second
   - High → Cluster issues

7. compression-rate-avg
   - Compression ratio
   - Validate compression working
\`\`\`

---

## Best Practices

### **1. Always Enable Idempotence:**
\`\`\`java
props.put("enable.idempotence", "true");
// No downside, prevents duplicates
\`\`\`

### **2. Choose Appropriate acks:**
\`\`\`java
// Critical data (payments):
props.put("acks", "all");

// Logs/metrics (loss acceptable):
props.put("acks", "1");
\`\`\`

### **3. Use Compression:**
\`\`\`java
props.put("compression.type", "snappy");
// Reduces bandwidth, storage
\`\`\`

### **4. Tune Batching:**
\`\`\`java
// Balance latency and throughput:
props.put("batch.size", 16384);  // 16KB
props.put("linger.ms", 10);      // 10ms
\`\`\`

### **5. Handle Errors:**
\`\`\`java
// Use callback for error handling:
producer.send (record, callback);
\`\`\`

### **6. Close Producer Gracefully:**
\`\`\`java
// Flush pending messages before shutdown:
producer.flush();
producer.close();
\`\`\`

### **7. Monitor Metrics:**
\`\`\`java
// Track send rate, error rate, latency
// Alert on anomalies
\`\`\`

---

## Producer in System Design Interviews

### **Discussion Points:**1. **Partitioning Strategy:**
   - "Use user_id as partition key for ordering per user"
   - "12 partitions for parallelism"

2. **Reliability:**
   - "Enable idempotence for exactly-once"
   - "acks=all for critical data (payments)"

3. **Performance:**
   - "Batch size 16KB, linger 10ms for throughput"
   - "Snappy compression (3× reduction)"

4. **Error Handling:**
   - "Async send with callback"
   - "Dead letter queue for failed messages"

5. **Monitoring:**
   - "Track send rate, error rate, latency"
   - "Alert on high retry rate"

### **Example:**

\`\`\`
Interviewer: "How would you handle high-volume order processing?"

You:
"Kafka producer with these settings:

1. Idempotence enabled (exactly-once per order)
2. acks=all (order durability critical)
3. Partition by customer_id (order sequence per customer)
4. Batch size 32KB, linger 20ms (handle 100K orders/sec)
5. Snappy compression (reduce network usage)
6. Async send with callback (error handling)
7. Dead letter queue (handle malformed orders)
8. Monitor: send rate, error rate, p99 latency

Scale: 100K orders/sec, 12 partitions = 8.3K/partition
Kafka broker handles 100K+ msg/sec, well within capacity."
\`\`\`

---

## Key Takeaways

1. **Three send patterns: Fire-forget, sync, async+callback** → Async+callback for production
2. **Partition keys ensure ordering** → Same key → Same partition
3. **Idempotence prevents duplicates** → Enable always, low overhead
4. **acks=all for durability** → Wait for all ISR, no data loss
5. **Batching improves throughput** → batch.size + linger.ms tuning
6. **Compression reduces bandwidth** → Snappy/zstd best choice
7. **Transactions for exactly-once across topics** → Higher latency
8. **Handle retriable vs non-retriable errors** → Retry logic + DLQ
9. **Monitor send rate, error rate, latency** → Detect issues early
10. **In interviews: Discuss partitioning, reliability, performance** → Show production thinking

---

**Next:** We'll explore **Kafka Consumers**—consumer groups, offset management, rebalancing, and scaling patterns.`,
};
