/**
 * Discussion Questions for Kafka Producers
 */

import { QuizQuestion } from '../../../types';

export const kafkaproducersQuiz: QuizQuestion[] = [
  {
    id: 'kafka-producers-dq-1',
    question:
      'Explain Kafka producer idempotence and how it prevents duplicate messages. Walk through a failure scenario where a producer crashes after sending a message but before receiving acknowledgment. How does idempotence handle this, and what are the performance implications?',
    hint: 'Consider producer IDs, sequence numbers, broker-side deduplication, and the trade-offs between reliability and throughput.',
    sampleAnswer: `Kafka producer idempotence ensures exactly-once delivery semantics per partition by preventing duplicate messages even when producers retry on failure. Here\'s how it works:

**The Duplicate Problem (Without Idempotence):**

\`\`\`
Scenario:
1. Producer sends Message A
2. Broker receives Message A
3. Broker writes Message A to log (offset 100)
4. Broker sends ACK
5. Network timeout - Producer doesn't receive ACK
6. Producer retries (thinks message failed)
7. Broker receives Message A again
8. Broker writes Message A to log (offset 101) - DUPLICATE! ❌

Result: Message A appears twice in partition
\`\`\`

**Idempotent Producer Solution:**

**Configuration:**
\`\`\`java
Properties props = new Properties();
props.put("enable.idempotence", "true");

// Idempotence automatically sets:
// acks = all (wait for all ISR)
// retries = Integer.MAX_VALUE (retry until success)
// max.in.flight.requests.per.connection = 5 (pipelining)
\`\`\`

**How It Works:**

**1. Producer ID (PID) Assignment:**
\`\`\`
First message sent:
Producer → InitProducerIdRequest → Broker
Broker → Assigns unique PID (e.g., PID=12345) → Producer

PID persists for producer lifetime
Reused across all messages from this producer
\`\`\`

**2. Sequence Numbers:**
\`\`\`
Producer assigns sequence number per partition:

Partition 0:
Message 1: {PID: 12345, Seq: 0, Data: "A"}
Message 2: {PID: 12345, Seq: 1, Data: "B"}
Message 3: {PID: 12345, Seq: 2, Data: "C"}

Partition 1:
Message 1: {PID: 12345, Seq: 0, Data: "X"}  // Separate sequence per partition
Message 2: {PID: 12345, Seq: 1, Data: "Y"}

Sequence numbers start at 0 per partition
Increment by 1 for each message
\`\`\`

**3. Broker-Side Deduplication:**
\`\`\`
Broker tracks last sequence number per (PID, Partition):

State: {(PID=12345, Partition=0): LastSeq=1}

Message arrives: {PID: 12345, Seq: 2, Data: "C"}
Broker checks: Seq=2 == LastSeq+1? Yes (2 == 1+1) ✅
Broker writes message, updates: LastSeq=2

Duplicate arrives: {PID: 12345, Seq: 1, Data: "B"}  // Retry
Broker checks: Seq=1 == LastSeq+1? No (1 < 2) ❌
Broker: Duplicate detected, ignore write
Broker: Still sends ACK (idempotent, already written)

Result: No duplicate in log ✅
\`\`\`

**Failure Scenario Walkthrough:**

**Initial State:**
\`\`\`
Producer: PID=12345, Partition 0, NextSeq=5
Broker: LastSeq for (PID=12345, Partition 0) = 4
\`\`\`

**Step-by-Step:**

\`\`\`
T1: Producer sends Message 5
    {PID: 12345, Seq: 5, Data: "Order 789"}

T2: Message arrives at broker
    Broker checks: Seq=5 == LastSeq+1? (5 == 4+1) ✅
    Broker writes to log at offset 1000
    Broker updates: LastSeq=5

T3: Broker sends ACK
    Response: {Offset: 1000, Success}

T4: Network timeout - Producer doesn't receive ACK ⚠️
    Producer thinks: Message failed (no ACK received)

T5: Producer retries (automatic with idempotence)
    Producer resends same message: {PID: 12345, Seq: 5, Data: "Order 789"}

T6: Retry arrives at broker
    Broker checks: Seq=5 == LastSeq+1? (5 == 5+1? No, 5 == 5) ❌
    Broker: This is a duplicate (Seq <= LastSeq)
    Broker: DON'T write to log (already at offset 1000)
    Broker: DO send ACK (acknowledge receipt)

T7: Producer receives ACK
    Producer: Success! Message delivered
    Producer: Increment NextSeq=6

Result:
- Message appears once in log (offset 1000) ✅
- Producer eventually gets ACK ✅
- Exactly-once delivery achieved ✅
\`\`\`

**Edge Cases:**

**Case 1: Out-of-Order Retry**
\`\`\`
Broker LastSeq=10

Message arrives: Seq=12
Expected: Seq=11 (gap detected)
Action: Broker returns OutOfOrderSequenceException
Producer: Must resend from Seq=11 onwards

Why: Ensures no missing messages
Maintains strict ordering per partition
\`\`\`

**Case 2: Producer Restart**
\`\`\`
Producer crashes, restarts
New PID assigned (PID changes on restart)
Sequence numbers reset to 0

Implication:
- Can't deduplicate across producer restarts
- Need transactional.id for cross-restart idempotence (transactions)

With transactional.id:
props.put("transactional.id", "order-processor-1");
- PID deterministically derived from transactional.id
- Survives producer restarts
- Enables exactly-once across restarts
\`\`\`

**Performance Implications:**

**1. Latency Impact:**
\`\`\`
Non-idempotent (acks=1):
Send → Leader writes → ACK
Latency: ~5ms (single broker)

Idempotent (acks=all, automatic):
Send → Leader writes → Followers replicate → All ISR ACK → Producer ACK
Latency: ~10-15ms (replication overhead)

Increase: 2-3× latency
Acceptable for reliability ✅
\`\`\`

**2. Throughput Impact:**
\`\`\`
Pipelining helps offset latency:
max.in.flight.requests.per.connection = 5

Producer can have 5 unacknowledged batches in flight
Maintains high throughput despite higher per-message latency

Throughput: ~90% of non-idempotent (minimal impact)
\`\`\`

**3. Broker-Side Overhead:**
\`\`\`
Broker tracks: (PID, Partition) → LastSeq

Memory per producer per partition: ~24 bytes
1000 producers × 100 partitions = 100K entries × 24 bytes = ~2.4 MB
Negligible memory overhead ✅

CPU overhead: Sequence number check per message
Cost: ~1-2 microseconds (negligible)
\`\`\`

**4. Producer-Side Overhead:**
\`\`\`
Assign sequence numbers: Negligible (counter increment)
Network: Additional metadata in request (8 bytes per message)
Minimal overhead ✅
\`\`\`

**Production Recommendation:**

\`\`\`java
// ALWAYS enable idempotence (Kafka 3.0+ default)
props.put("enable.idempotence", "true");

Benefits:
✅ No duplicates (exactly-once per partition)
✅ Minimal performance impact (2-3× latency, 90% throughput)
✅ No application-level deduplication needed
✅ Simpler code (producer handles retries)

Cost:
⚠️ Slightly higher latency (acceptable for reliability)
⚠️ Requires acks=all (durability improved)

When NOT to use:
❌ Never! Always use idempotence (default in recent Kafka)
Exception: Legacy Kafka (<0.11) doesn't support it
\`\`\`

**Idempotence vs Transactions:**

\`\`\`
Idempotence:
- Exactly-once per partition
- Automatic (just enable flag)
- Low overhead
- Use for: All producers

Transactions:
- Exactly-once across partitions
- Exactly-once across producer restarts
- Read-process-write atomicity
- Higher overhead
- Use for: Critical cross-partition writes, stream processing

Example: Payment processing
- Single partition writes: Idempotence sufficient ✅
- Multi-partition writes (payment + inventory): Transactions needed ✅
\`\`\`

**Key Takeaways:**

1. **Idempotence prevents duplicates through PID + Sequence numbers**
2. **Broker deduplicates based on sequence number tracking**
3. **Handles retries transparently (no application logic needed)**
4. **Performance impact minimal: 2-3× latency, 90% throughput**
5. **Always enable idempotence (default in modern Kafka)**
6. **Use transactions for cross-partition exactly-once**`,
    keyPoints: [
      'Idempotence uses Producer ID (PID) + Sequence numbers to prevent duplicates',
      'Broker tracks last sequence number per (PID, Partition) for deduplication',
      'Automatic retry on failure without duplicates (exactly-once per partition)',
      'Performance impact: 2-3× latency, minimal throughput decrease (~90%)',
      'Automatically enables acks=all and unlimited retries',
      'Always enable idempotence (default in Kafka 3.0+, no downside)',
    ],
  },
  {
    id: 'kafka-producers-dq-2',
    question:
      'Design a Kafka producer configuration for a payment processing system that must guarantee no duplicate charges and no lost payments. Explain your choices for acks, retries, idempotence, compression, batching, and timeout settings. What monitoring metrics would you track?',
    hint: 'Consider reliability vs performance trade-offs, failure scenarios, and production best practices for critical financial data.',
    sampleAnswer: `Payment processing requires the highest reliability guarantees. Here\'s a comprehensive producer configuration with justifications:

**Requirements:**
- No duplicate charges (exactly-once processing)
- No lost payments (100% reliability)
- Reasonable performance (100ms p99 latency acceptable)
- Audit trail (track all payment attempts)
- Graceful error handling (retry transient, fail permanent errors)

**Producer Configuration:**

\`\`\`java
Properties props = new Properties();

// 1. BOOTSTRAP SERVERS
props.put("bootstrap.servers", "kafka1:9092,kafka2:9092,kafka3:9092");
// Rationale: Multiple brokers for discovery (if one down)
// All 3 brokers listed (client discovers full cluster)

// 2. SERIALIZATION
props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
props.put("value.serializer", "org.apache.kafka.common.serialization.ByteArraySerializer");
// Rationale: 
//   - Key: String (payment_id for partitioning)
//   - Value: ByteArray (Avro-serialized payment data)

// 3. IDEMPOTENCE (CRITICAL)
props.put("enable.idempotence", "true");
// Rationale: Prevents duplicate charges on retry
// Guarantees: Exactly-once delivery per partition
// Automatic settings: acks=all, retries=MAX, max.in.flight=5

// 4. ACKNOWLEDGMENTS (CRITICAL)
// (Automatically set by idempotence, but explicit for clarity)
props.put("acks", "all");  // Wait for all ISR replicas
// Rationale: No data loss (replicated to all in-sync replicas)
// Trade-off: Higher latency (10-15ms) acceptable for payments

// 5. RETRIES (CRITICAL)
// (Automatically set by idempotence)
props.put("retries", Integer.MAX_VALUE);  // Retry until success
// Rationale: Never give up on transient errors
// Producer retries: Network timeouts, leader elections
// Application handles permanent errors (card declined)

// 6. RETRY BACKOFF
props.put("retry.backoff.ms", 100);  // 100ms between retries
// Rationale: Avoid overwhelming broker during failures
// Exponential backoff not built-in (consider custom logic if needed)

// 7. REQUEST TIMEOUT
props.put("request.timeout.ms", 30000);  // 30 seconds
// Rationale: Payment gateway calls take 1-5 seconds
// 30 sec allows for: Processing (5s) + Network (2s) + Buffer (23s)

// 8. DELIVERY TIMEOUT
props.put("delivery.timeout.ms", 120000);  // 2 minutes total
// Rationale: Maximum time for delivery including retries
// 2 minutes allows multiple retry attempts
// After 2 min, producer throws exception (application handles)

// 9. MAX IN-FLIGHT REQUESTS
// (Automatically set by idempotence)
props.put("max.in.flight.requests.per.connection", 5);
// Rationale: Balance throughput and ordering
// 5 = good pipelining while maintaining order (idempotence ensures correctness)

// 10. COMPRESSION
props.put("compression.type", "snappy");
// Rationale: 
//   - Reduces network/storage by ~3× (payment JSON compresses well)
//   - Snappy: Fast compression, low CPU overhead
//   - Alternative: lz4 (similar), zstd (better compression, slightly slower)

// 11. BATCHING (Performance optimization)
props.put("batch.size", 16384);  // 16 KB batch size
// Rationale: Batch multiple payments together
// ~8 payments/batch (2KB each) = 8× fewer network calls

props.put("linger.ms", 10);  // Wait 10ms to fill batch
// Rationale: Small delay acceptable (10ms)
// Better batching = higher throughput
// User sees response in ~100ms total (including processing)

// 12. BUFFER MEMORY
props.put("buffer.memory", 33554432);  // 32 MB
// Rationale: Buffer for unsent messages (traffic spikes)
// 32 MB = ~16K messages buffered (32MB / 2KB per payment)
// Producer blocks if buffer full (backpressure)

// 13. MAX BLOCK TIME
props.put("max.block.ms", 60000);  // 60 seconds
// Rationale: How long producer waits if buffer full
// 60 sec allows burst traffic to subside
// After 60 sec, throws exception (application handles gracefully)

// 14. CLIENT ID
props.put("client.id", "payment-producer-" + InetAddress.getLocalHost().getHostName());
// Rationale: Identify producer in broker logs/metrics
// Include hostname for debugging (which instance sent message)

// 15. PARTITIONER
props.put("partitioner.class", "com.example.PaymentPartitioner");
// Rationale: Custom partitioner for payment_id-based partitioning
// Ensures ordering per payment (critical for idempotency)

KafkaProducer<String, byte[]> producer = new KafkaProducer<>(props);
\`\`\`

**Custom Partitioner:**

\`\`\`java
public class PaymentPartitioner implements Partitioner {
    @Override
    public int partition(String topic, Object key, byte[] keyBytes,
                        Object value, byte[] valueBytes, Cluster cluster) {
        // Partition by payment_id (ensures all messages for same payment to same partition)
        String paymentId = (String) key;
        int numPartitions = cluster.partitionCountForTopic (topic);
        
        // Hash-based partitioning
        return Math.abs (paymentId.hashCode()) % numPartitions;
    }
}
\`\`\`

**Sending Payments:**

\`\`\`java
public class PaymentProducer {
    private final KafkaProducer<String, byte[]> producer;
    private final Counter paymentsAttempted;
    private final Counter paymentsSucceeded;
    private final Counter paymentsFailed;
    private final Histogram paymentLatency;
    
    public void sendPayment(Payment payment) {
        paymentsAttempted.inc();
        long startTime = System.currentTimeMillis();
        
        String key = payment.getPaymentId();  // Partition key
        byte[] value = serializePayment (payment);  // Avro serialization
        
        ProducerRecord<String, byte[]> record = new ProducerRecord<>(
            "payments",  // Topic
            key,         // Key (payment_id)
            value        // Value (serialized payment)
        );
        
        // Add headers (metadata for tracing, debugging)
        record.headers()
            .add("correlation_id", payment.getCorrelationId().getBytes())
            .add("timestamp", String.valueOf(System.currentTimeMillis()).getBytes())
            .add("version", "v1".getBytes());
        
        // Async send with callback (production best practice)
        producer.send (record, (metadata, exception) -> {
            long latency = System.currentTimeMillis() - startTime;
            paymentLatency.observe (latency);
            
            if (exception == null) {
                // Success
                paymentsSucceeded.inc();
                logger.info("Payment sent successfully: payment_id={}, partition={}, offset={}, latency={}ms",
                    key, metadata.partition(), metadata.offset(), latency);
                
                // Update application state (mark as sent)
                updatePaymentStatus (payment.getPaymentId(), "SENT");
                
            } else {
                // Failure (after all retries exhausted)
                paymentsFailed.inc();
                logger.error("Payment failed: payment_id={}, error={}", key, exception.getMessage(), exception);
                
                // Distinguish error types
                if (exception instanceof RetriableException) {
                    // Should not happen (producer retries indefinitely)
                    // Log critical alert
                    alerting.sendAlert("Payment producer exhausted retries", exception);
                } else {
                    // Non-retriable (e.g., message too large, serialization error)
                    logger.error("Non-retriable error: {}", exception.getMessage());
                }
                
                // Update application state (mark as failed)
                updatePaymentStatus (payment.getPaymentId(), "FAILED");
                
                // Send to dead letter queue or retry table
                sendToDeadLetterQueue (payment, exception);
            }
        });
    }
    
    // Graceful shutdown (flush pending messages)
    public void close() {
        logger.info("Closing producer, flushing pending messages...");
        producer.flush();  // Wait for all pending messages
        producer.close(Duration.ofSeconds(30));  // Close with timeout
        logger.info("Producer closed successfully");
    }
}
\`\`\`

**Error Handling Strategy:**

\`\`\`java
// Application-level error handling (before sending to Kafka)
public void processPayment(PaymentRequest request) {
    try {
        // 1. Validate payment request
        validatePayment (request);
        
        // 2. Check idempotency (application-level)
        if (paymentAlreadyProcessed (request.getPaymentId())) {
            logger.info("Payment already processed: {}", request.getPaymentId());
            return;  // Skip duplicate
        }
        
        // 3. Create payment object
        Payment payment = Payment.builder()
            .paymentId (request.getPaymentId())
            .customerId (request.getCustomerId())
            .amount (request.getAmount())
            .timestamp(Instant.now())
            .build();
        
        // 4. Send to Kafka (async)
        paymentProducer.sendPayment (payment);
        
        // 5. Return immediately (don't wait for Kafka ACK)
        return PaymentResponse.accepted (request.getPaymentId());
        
    } catch (ValidationException e) {
        // Permanent error (don't send to Kafka)
        logger.warn("Payment validation failed: {}", e.getMessage());
        return PaymentResponse.rejected (e.getMessage());
    }
}
\`\`\`

**Monitoring Metrics:**

\`\`\`
Producer Metrics (JMX + Prometheus):

1. record-send-rate
   - Messages sent per second
   - Track throughput
   - Alert if drops significantly (< expected rate)

2. record-error-rate
   - Failed sends per second
   - Alert if > 0 (should be 0 with infinite retries)
   - Investigate immediately

3. request-latency-avg / request-latency-p99
   - Average and p99 request latency
   - Track performance degradation
   - Alert if p99 > 100ms (investigate slow brokers)

4. batch-size-avg
   - Average batch size
   - Optimize batching configuration
   - Higher = more efficient

5. compression-rate-avg
   - Compression ratio achieved
   - Validate compression working (~3× expected)

6. buffer-available-bytes
   - Free buffer memory
   - Alert if < 10% (producer under pressure)
   - Scale or optimize

7. record-retry-rate
   - Retries per second
   - Alert if high (> 5% of send rate)
   - Indicates broker or network issues

8. produce-throttle-time-avg
   - Time producer throttled by broker (quota exceeded)
   - Alert if > 0 (hitting quota)

Custom Application Metrics:

9. payments_attempted_total (Counter)
   - Total payment attempts
   - Track overall volume

10. payments_succeeded_total (Counter)
    - Successfully sent to Kafka
    - Success rate: succeeded / attempted

11. payments_failed_total (Counter)
    - Failed sends (after retries)
    - Alert if > 0 (critical)

12. payment_send_latency_seconds (Histogram)
    - End-to-end latency (send + ACK)
    - Track p50, p95, p99
    - Alert if p99 > 100ms

13. payments_in_dead_letter_queue (Gauge)
    - Payments failed permanently
    - Alert if > 0 (manual intervention needed)

Kafka Cluster Metrics:

14. kafka_broker_count
    - Number of live brokers
    - Alert if < 3 (losing redundancy)

15. kafka_topic_partition_count
    - Partitions available
    - Monitor for scaling

16. kafka_topic_replication_factor
    - Verify replication = 3
    - Alert if degraded

17. kafka_under_replicated_partitions
    - Partitions not fully replicated
    - Alert if > 0 (risk of data loss)
\`\`\`

**Alerting Rules:**

\`\`\`yaml
alerts:
  - name: PaymentProducerFailure
    expr: rate (payments_failed_total[5m]) > 0
    severity: critical
    message: "Payments failing after retries"
    
  - name: PaymentHighLatency
    expr: histogram_quantile(0.99, payment_send_latency_seconds) > 0.100
    severity: warning
    message: "Payment send latency p99 > 100ms"
    
  - name: ProducerHighRetryRate
    expr: rate (record_retry_rate[5m]) > 5
    severity: warning
    message: "High producer retry rate (broker issues?)"
    
  - name: ProducerBufferExhausted
    expr: buffer_available_bytes < 3355443  # 10% of 32MB
    severity: warning
    message: "Producer buffer low (backpressure)"
    
  - name: DeadLetterQueueNonEmpty
    expr: payments_in_dead_letter_queue > 0
    severity: critical
    message: "Payments in DLQ require investigation"
\`\`\`

**Dashboard (Grafana):**

\`\`\`
Panel 1: Payment Volume
- Graph: payments_attempted_total rate
- Shows: Payments/second over time

Panel 2: Success Rate
- Gauge: (payments_succeeded / payments_attempted) × 100
- Target: 100%

Panel 3: Latency Distribution
- Heatmap: payment_send_latency_seconds
- Shows: p50, p95, p99 latency

Panel 4: Error Rate
- Graph: payments_failed_total rate
- Alert threshold: 0

Panel 5: Producer Metrics
- Batch size average
- Compression rate
- Buffer available

Panel 6: Kafka Cluster Health
- Broker count
- Under-replicated partitions
- ISR shrink/expand rate
\`\`\`

**Summary:**

This configuration guarantees:
✅ No duplicate charges (idempotence + PID + sequence numbers)
✅ No lost payments (acks=all + infinite retries + replication factor 3)
✅ Graceful error handling (retries + DLQ for permanent failures)
✅ Good performance (batching + compression + pipelining)
✅ Comprehensive monitoring (metrics + alerts + dashboards)
✅ Production-ready reliability for financial transactions`,
    keyPoints: [
      'Enable idempotence, acks=all, infinite retries for exactly-once + no data loss',
      'Use compression (snappy) and batching (16KB, 10ms linger) for performance',
      'Custom partitioner ensures ordering per payment_id (critical for idempotency)',
      'Async send with callback for error handling and monitoring',
      'Monitor: send rate, error rate, latency (p99), retry rate, buffer available',
      'Alert on: Any failures, high latency (>100ms p99), high retry rate, low buffer',
    ],
  },
  {
    id: 'kafka-producers-dq-3',
    question:
      'Explain the trade-offs between batching (batch.size, linger.ms) and latency in Kafka producers. For different scenarios (real-time analytics, batch ETL, user-facing API), what batch configuration would you choose and why? How do compression and batching interact?',
    hint: 'Consider throughput, latency, network efficiency, and the relationship between batch size, linger time, and compression ratio.',
    sampleAnswer: `Batching is critical for Kafka performance, but comes with latency trade-offs. Understanding how to configure batching for different use cases is essential.

**Batching Fundamentals:**

**batch.size**: Maximum bytes per batch (default: 16384 = 16 KB)
**linger.ms**: Maximum time to wait for batch to fill (default: 0)

**How Batching Works:**

\`\`\`
Producer accumulates messages in memory batches:

Without batching (batch.size=1, linger.ms=0):
Message 1 → Network call
Message 2 → Network call
Message 3 → Network call
Total: 3 network calls, 3ms latency each = 9ms

With batching (batch.size=16KB, linger.ms=10):
Message 1 → Accumulate
Message 2 → Accumulate  } Wait up to 10ms
Message 3 → Accumulate  } or until 16KB
Batch ready → Network call (all 3 messages)
Total: 1 network call, 3ms latency + 10ms wait = 13ms

Trade-off:
✅ 3× fewer network calls (better throughput)
❌ 1.4× higher latency (13ms vs 9ms)
\`\`\`

**Batch Triggers:**

Batch sent when EITHER condition met:
1. **batch.size reached**: Batch full (16KB of messages)
2. **linger.ms expired**: Time limit reached (10ms wait)
3. **Flush called**: Application explicitly flushes

**Example:**
\`\`\`
batch.size=16KB, linger.ms=10ms

Scenario 1: High traffic (many messages)
- Messages accumulate quickly
- 16KB batch filled in 2ms
- Sent after 2ms (batch.size trigger)
- linger.ms not reached

Scenario 2: Low traffic (few messages)
- Messages accumulate slowly
- Only 2KB accumulated after 10ms
- Sent after 10ms (linger.ms trigger)
- batch.size not reached (partial batch)

Result: Adapts to traffic patterns ✅
\`\`\`

**Scenario 1: Real-Time Analytics (Click Stream)**

**Requirements:**
- Latency: <100ms acceptable (near real-time)
- Throughput: 100K clicks/second
- Message size: ~500 bytes per click

**Configuration:**
\`\`\`java
props.put("batch.size", 65536);  // 64 KB (larger batch)
props.put("linger.ms", 50);      // Wait 50ms to fill batch
props.put("compression.type", "lz4");  // Fast compression

Rationale:

batch.size=64KB:
- 64KB / 500 bytes = ~128 clicks per batch
- Higher batch size = better throughput (fewer network calls)
- Latency impact acceptable (user doesn't see clicks immediately)

linger.ms=50ms:
- Wait up to 50ms to accumulate messages
- Still within 100ms budget (50ms + network)
- Better batching during traffic spikes

Throughput:
- 100K clicks/sec, 128 per batch = 781 batches/sec
- vs No batching: 100K network calls/sec (128× more!)

Latency:
- Average: ~50ms (linger time) + 10ms (network) = 60ms ✅
- Acceptable for analytics (not user-facing)
\`\`\`

**Scenario 2: Batch ETL (Data Warehouse Loading)**

**Requirements:**
- Latency: Don't care (batch processing)
- Throughput: Maximize (100M records/day)
- Message size: ~2KB per record

**Configuration:**
\`\`\`java
props.put("batch.size", 1048576);  // 1 MB (maximum batch)
props.put("linger.ms", 1000);      // Wait 1 second
props.put("compression.type", "zstd");  // Best compression

Rationale:

batch.size=1MB:
- 1MB / 2KB = ~500 records per batch
- Maximize throughput (fewest network calls)
- Latency irrelevant (batch job)

linger.ms=1000ms:
- Wait up to 1 second to fill batch
- Accumulate as many messages as possible
- ETL runs overnight (latency doesn't matter)

compression.type=zstd:
- Best compression ratio (~5×)
- More CPU but worth it (network/storage savings)
- ETL has spare CPU cycles

Throughput:
- 100M records/day = 1,157 records/sec
- 1,157 / 500 per batch = 2.3 batches/sec
- Minimal network overhead ✅

Network savings:
- No batching: 1,157 network calls/sec
- With batching: 2.3 network calls/sec (502× reduction!)
\`\`\`

**Scenario 3: User-Facing API (Order Processing)**

**Requirements:**
- Latency: <50ms (user waiting for response)
- Throughput: 1,000 orders/second
- Message size: ~5KB per order

**Configuration:**
\`\`\`java
props.put("batch.size", 16384);  // 16 KB (small batch)
props.put("linger.ms", 5);       // Wait only 5ms
props.put("compression.type", "snappy");  // Fast compression

Rationale:

batch.size=16KB:
- 16KB / 5KB = ~3 orders per batch
- Small batch size (prioritize latency over throughput)
- Still benefits from batching at high traffic

linger.ms=5ms:
- Minimal wait time (only 5ms)
- Low latency critical (user sees response quickly)
- Still allows some batching

compression.type=snappy:
- Fast (low CPU overhead)
- Decent compression (~3×)
- Balanced choice

Latency:
- Low traffic: 5ms (linger) + 10ms (network) = 15ms ✅
- High traffic: 2ms (batch fills) + 10ms (network) = 12ms ✅
- Both well under 50ms budget

Throughput:
- 1,000 orders/sec, ~3 per batch = 333 batches/sec
- vs No batching: 1,000 network calls/sec (3× reduction)
\`\`\`

**Compression and Batching Interaction:**

**Key Insight:** Larger batches → Better compression ratio

**Why:**
\`\`\`
Compression algorithms (e.g., zstd, snappy) work better on larger data:

Small batch (1 message, 1KB):
Before: 1KB
After compression: 700 bytes
Compression ratio: 1.4×

Large batch (100 messages, 100KB):
Before: 100KB
After compression: 25KB
Compression ratio: 4×

Why? Compression finds patterns:
- More data = more repetition found
- Dictionary compression more effective
- Overhead amortized across more data
\`\`\`

**Example:**

\`\`\`
Message: {"order_id":"12345","customer":"user_A","amount":99.99}

Without batching:
- Compress 1 message: 58 bytes → 52 bytes (1.1× compression)
- "order_id", "customer", "amount" strings not compressed much

With batching (100 messages):
- Compress 100 messages: 5,800 bytes → 1,450 bytes (4× compression)
- "order_id", "customer", "amount" repeated 100 times → Compressed once in dictionary
- Patterns like "user_A" repeated → Compressed

Result: Batch compression 4× better than individual message compression
\`\`\`

**Compression Types:**

\`\`\`
| Type   | Ratio | Speed | CPU   | Use Case |
|--------|-------|-------|-------|----------|
| none   | 1×    | N/A   | None  | Already compressed data (images) |
| snappy | 2-3×  | Fast  | Low   | User-facing (low latency needed) |
| lz4    | 2-3×  | Fast  | Low   | Similar to snappy, slightly faster |
| gzip   | 4-6×  | Slow  | High  | Bandwidth-constrained (slow network) |
| zstd   | 3-5×  | Medium| Medium| Best overall (modern choice) |

Recommendation:
- User-facing: snappy or lz4 (low latency)
- Batch ETL: zstd or gzip (best compression)
- Real-time: lz4 (balanced)
\`\`\`

**Decision Matrix:**

\`\`\`
| Use Case | batch.size | linger.ms | compression | Rationale |
|----------|-----------|-----------|-------------|-----------|
| User-facing API | 16 KB | 5 ms | snappy | Low latency critical |
| Real-time analytics | 64 KB | 50 ms | lz4 | Balance latency/throughput |
| Batch ETL | 1 MB | 1000 ms | zstd | Maximize throughput |
| Log shipping | 256 KB | 100 ms | snappy | High volume, moderate latency |
| IoT telemetry | 128 KB | 200 ms | lz4 | High throughput, latency OK |
| Financial transactions | 16 KB | 10 ms | snappy | Low latency, reliable |
\`\`\`

**Testing Your Configuration:**

\`\`\`java
// Test different configurations
public void benchmarkBatching() {
    // Configuration 1: No batching
    Props config1 = new Props();
    config1.put("batch.size", 1);
    config1.put("linger.ms", 0);
    
    // Configuration 2: Moderate batching
    Props config2 = new Props();
    config2.put("batch.size", 16384);
    config2.put("linger.ms", 10);
    
    // Configuration 3: Aggressive batching
    Props config3 = new Props();
    config3.put("batch.size", 131072);  // 128 KB
    config3.put("linger.ms", 100);
    
    // Send 10,000 messages with each configuration
    for (Props config : Arrays.asList (config1, config2, config3)) {
        KafkaProducer producer = new KafkaProducer (config);
        
        long start = System.currentTimeMillis();
        for (int i = 0; i < 10000; i++) {
            producer.send (new ProducerRecord("test", "message" + i));
        }
        producer.flush();
        long duration = System.currentTimeMillis() - start;
        
        System.out.println("Config: " + config);
        System.out.println("Time: " + duration + "ms");
        System.out.println("Throughput: " + (10000.0 / duration * 1000) + " msg/sec");
    }
}

// Typical results:
// Config 1 (no batching): 30,000ms, 333 msg/sec
// Config 2 (moderate): 5,000ms, 2,000 msg/sec (6× faster)
// Config 3 (aggressive): 2,000ms, 5,000 msg/sec (15× faster)
\`\`\`

**Key Takeaways:**

1. **Batching dramatically improves throughput** (10-100× fewer network calls)
2. **linger.ms adds latency** (trade-off for throughput)
3. **Larger batches → Better compression** (4× vs 1.4×)
4. **User-facing: Small batches, short linger** (latency critical)
5. **Batch jobs: Large batches, long linger** (latency irrelevant)
6. **Compression choice depends on CPU budget** (snappy for speed, zstd for ratio)
7. **Test configurations for your workload** (optimal settings vary)`,
    keyPoints: [
      'Batching reduces network calls (10-100×) but adds latency (linger.ms)',
      'batch.size and linger.ms: Send when either condition met',
      'User-facing: Small batch (16KB), short linger (5ms) for low latency',
      'Batch ETL: Large batch (1MB), long linger (1s) for max throughput',
      'Larger batches → Better compression (4× vs 1.4× ratio)',
      'Compression choice: snappy (fast), zstd (best ratio), lz4 (balanced)',
    ],
  },
];
