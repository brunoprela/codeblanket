/**
 * Discussion Questions for Kafka Consumers
 */

import { QuizQuestion } from '../../../types';

export const kafkaconsumersQuiz: QuizQuestion[] = [
  {
    id: 'kafka-consumers-dq-1',
    question:
      'Explain Kafka consumer groups and how partition assignment works. Design a consumer group for processing 1 million order events per day from a topic with 24 partitions. How many consumers would you deploy, how would you handle rebalancing, and what happens when a consumer crashes?',
    hint: 'Consider parallelism limits, rebalancing strategies, and failure recovery with offset management.',
    sampleAnswer: `Kafka consumer groups enable parallel processing while maintaining scalability and fault tolerance. Here's a comprehensive analysis:

**Consumer Group Fundamentals:**

A consumer group is a set of consumers that cooperate to consume a topic, with each partition consumed by exactly one consumer in the group at any time.

**Key Rules:**
1. Each partition assigned to exactly ONE consumer in a group
2. Each consumer can handle MULTIPLE partitions
3. Max parallelism = Number of partitions
4. Multiple consumer groups can read the same topic independently

**Architecture:**
\`\`\`
Topic: orders (24 partitions)

Consumer Group: "order-processors"
┌─────────────┬─────────────┬─────────────┐
│ Consumer 1  │ Consumer 2  │ Consumer 3  │
│ Part 0,1,2  │ Part 3,4,5  │ Part 6,7,8  │
│ Part 9,10   │ Part 11,12  │ Part 13,14  │
│ Part 15,16  │ Part 17,18  │ Part 19,20  │
│             │             │ Part 21,22  │
│             │             │ Part 23     │
└─────────────┴─────────────┴─────────────┘
\`\`\`

**Order Processing System Design:**

**Requirements:**
- Volume: 1M orders/day = 11.57 orders/sec average
- Peak: 3× during flash sales = 34.7 orders/sec
- Topic: orders (24 partitions)
- Processing time: ~2 seconds per order (DB write + business logic)

**Consumer Count Calculation:**

\`\`\`
Throughput needed: 35 orders/sec (peak)
Processing time: 2 sec/order
Consumer capacity: 1 order / 2 sec = 0.5 orders/sec per consumer

Minimum consumers: 35 / 0.5 = 70 consumers ❌ (Too many!)

Wait - we have only 24 partitions!
Max useful consumers = 24 (one per partition)

Per-consumer throughput needed:
35 orders/sec / 24 partitions = 1.46 orders/sec per consumer
Time per order: 2 seconds
Capacity per consumer: 0.5 orders/sec

We need 1.46 / 0.5 = 2.92 consumers per partition

Since we can't assign multiple consumers to one partition:
❌ Can't scale beyond 24 consumers with current partition count

Options:
1. Increase partitions (requires new topic, data migration)
2. Optimize processing time (2 sec → 0.5 sec = 4× improvement)
3. Scale vertically (bigger instances, more CPU)

Recommended: Option 2 (optimize processing)
- Batch database writes (10 orders per transaction)
- Async non-critical operations (email, analytics)
- Cache frequent lookups (product data, customer data)

With optimization (0.5 sec/order):
Capacity per consumer: 2 orders/sec
Required: 35 / 2 = 17.5 → 18 consumers

Deployment: 18-20 consumers (buffer for spikes)
- Each handles 1-2 partitions (24 partitions / 20 consumers)
- Capacity: 20 × 2 = 40 orders/sec (15% headroom)
\`\`\`

**Partition Assignment Strategies:**

**1. RangeAssignor (Default):**
\`\`\`
Assigns contiguous partition ranges to consumers

20 consumers, 24 partitions:
Consumer 1: Partitions 0, 1 (2 partitions)
Consumer 2: Partitions 2, 3 (2 partitions)
...
Consumer 12: Partitions 22, 23 (2 partitions)
Consumer 13-20: No partitions (idle)

Problem: Uneven distribution (4 idle consumers)

Better with 24 consumers (1:1):
Consumer 1: Partition 0
Consumer 2: Partition 1
...
Consumer 24: Partition 23

Perfect distribution ✅
\`\`\`

**2. RoundRobinAssignor:**
\`\`\`
Distributes partitions evenly in round-robin

20 consumers, 24 partitions:
Consumer 1: Partitions 0, 20
Consumer 2: Partitions 1, 21
Consumer 3: Partitions 2, 22
Consumer 4: Partitions 3, 23
Consumer 5: Partition 4
Consumer 6: Partition 5
...
Consumer 20: Partition 19

Distribution: 4 consumers with 2 partitions, 16 with 1 partition
More balanced than Range ✅
\`\`\`

**3. StickyAssignor (Recommended):**
\`\`\`
Minimizes partition movement during rebalancing

Initial assignment (20 consumers, 24 partitions):
Consumer 1: Partitions 0, 20
Consumer 2: Partitions 1, 21
... (round-robin distribution)

Consumer 5 crashes:
- RoundRobinAssignor: Reassigns ALL partitions (disruptive!)
- StickyAssignor: Only reassigns partition 4 (minimal disruption)

Consumer 21 added:
- RoundRobinAssignor: Reassigns many partitions
- StickyAssignor: Moves only 1 partition to new consumer

Benefits:
✅ Minimal partition movement on rebalancing
✅ Preserves consumer state (cache, connections)
✅ Lower rebalancing latency

Use: CooperativeStickyAssignor (Kafka 2.4+)
- Incremental rebalancing (no stop-the-world)
- Even better than StickyAssignor
\`\`\`

**Rebalancing Process:**

**When Rebalancing Triggers:**
1. Consumer joins group (scale up)
2. Consumer leaves group (shutdown/crash)
3. Consumer session timeout (heartbeat failure)
4. Partitions added to topic
5. Consumer unsubscribes

**Rebalancing Steps:**

\`\`\`
Eager Rebalancing (Default):

Initial State:
Consumer 1: Partitions 0-7
Consumer 2: Partitions 8-15
Consumer 3: Partitions 16-23

Consumer 4 joins:

Step 1: All consumers revoke ALL partitions
  - Consumer 1, 2, 3: Stop processing
  - Commit offsets
  - Revoke partitions

Step 2: Group coordinator reassigns partitions
  - Consumer 1: Partitions 0-5 (6 partitions)
  - Consumer 2: Partitions 6-11 (6 partitions)
  - Consumer 3: Partitions 12-17 (6 partitions)
  - Consumer 4: Partitions 18-23 (6 partitions)

Step 3: Consumers resume
  - All start processing from committed offsets

Downtime: 5-10 seconds (all consumers stopped)
\`\`\`

\`\`\`
Cooperative Rebalancing (CooperativeStickyAssignor):

Initial State:
Consumer 1: Partitions 0-7
Consumer 2: Partitions 8-15
Consumer 3: Partitions 16-23

Consumer 4 joins:

Round 1: Identify partitions to move
  - Consumer 1: Keep 0-5, revoke 6-7
  - Consumer 2: Keep 8-13, revoke 14-15
  - Consumer 3: Keep 16-21, revoke 22-23
  - Consumer 4: Receive 6-7, 14-15, 22-23 (6 partitions)

Consumers 1, 2, 3 continue processing partitions they keep
Only partitions 6-7, 14-15, 22-23 paused briefly

Round 2: Complete reassignment
  - Consumer 4 starts consuming 6-7, 14-15, 22-23

Downtime: Minimal (only affected partitions paused)
✅ Much better for production
\`\`\`

**Handling Consumer Crashes:**

**Scenario: Consumer 2 crashes**

\`\`\`
Initial State:
Consumer 1: Partitions 0-7
Consumer 2: Partitions 8-15 (CRASHES)
Consumer 3: Partitions 16-23

Timeline:

T0: Consumer 2 crashes (stops sending heartbeats)

T1-T10: Group coordinator waits (session.timeout.ms = 10 seconds)
  - Partitions 8-15 not being consumed ⏸️
  - Orders accumulating in these partitions
  - No rebalancing yet (within timeout)

T10: Session timeout expires
  - Coordinator detects Consumer 2 failure
  - Initiates rebalancing

T11-T15: Rebalancing (5 seconds)
  - Consumer 1 and 3 revoke partitions (if eager)
  - Reassignment:
    * Consumer 1: Partitions 0-11 (12 partitions)
    * Consumer 3: Partitions 12-23 (12 partitions)
  - Consumers resume from last committed offsets

T15: Processing resumes
  - Consumer 1 now handles partitions 8-11 (previously Consumer 2's)
  - Consumer 3 handles partitions 12-15 (previously Consumer 2's)
  - Backlog processed

Total downtime for partitions 8-15: ~15 seconds
\`\`\`

**Offset Management During Crash:**

\`\`\`
Consumer 2 state before crash:
Partition 8:
  - Last committed offset: 1000
  - Currently processing: 1005
  - Consumer crashes at offset 1005

After rebalancing:
Consumer 1 takes over Partition 8:
  - Reads from last committed offset: 1000
  - Reprocesses 1000-1004 (duplicates!)
  - Then processes 1005+ (new messages)

Result: At-least-once delivery (duplicates possible)

Handling Duplicates (Idempotency):
def process_order(order):
    order_id = order['order_id']
    
    # Check if already processed
    if redis.exists(f"processed:{order_id}"):
        return  # Skip duplicate
    
    # Process order
    save_order_to_db(order)
    
    # Mark as processed
    redis.setex(f"processed:{order_id}", 86400, "true")

✅ Idempotent processing handles duplicates safely
\`\`\`

**Minimizing Rebalancing Impact:**

\`\`\`java
Properties props = new Properties();

// 1. Increase session timeout (tolerate longer heartbeat gaps)
props.put("session.timeout.ms", "30000");  // 30 seconds
// Prevents rebalancing due to temporary GC pauses or network hiccups

// 2. Decrease heartbeat interval (detect failures faster)
props.put("heartbeat.interval.ms", "3000");  // 3 seconds
// More frequent heartbeats = faster failure detection (within session timeout)

// 3. Increase max poll interval (allow longer processing time)
props.put("max.poll.interval.ms", "300000");  // 5 minutes
// Consumer can process for 5 minutes before coordinator marks it dead
// Important if processing individual messages takes long

// 4. Use cooperative rebalancing (minimize downtime)
props.put("partition.assignment.strategy", 
          "org.apache.kafka.clients.consumer.CooperativeStickyAssignor");
// Incremental rebalancing, only affected partitions paused

// 5. Configure max poll records (control batch size)
props.put("max.poll.records", "100");  // Fetch 100 records per poll
// Smaller batches = faster processing = more frequent heartbeats
\`\`\`

**Production Deployment:**

\`\`\`
Deployment Strategy:

1. Initial Deployment (20 consumers):
   - Deploy consumers gradually (5 at a time)
   - Allow rebalancing to complete between batches
   - Monitor consumer lag after each rebalancing

2. Scaling Up (add 4 consumers → 24 total):
   - Rolling deployment: Add 1 consumer, wait 30 sec, repeat
   - Minimizes rebalancing frequency
   - Cooperative rebalancing helps

3. Scaling Down (remove 4 consumers → 20 total):
   - Graceful shutdown: Send SIGTERM (not SIGKILL)
   - Consumer commits offsets, leaves group cleanly
   - Coordinator reassigns partitions immediately (no timeout wait)

4. Rolling Restarts (update consumer version):
   - Stop 1 consumer at a time
   - Wait for rebalancing (30 sec)
   - Start new version
   - Repeat for next consumer
   - Total time: 20 consumers × 30 sec = 10 minutes

Automation (Kubernetes):
- HorizontalPodAutoscaler based on consumer lag
- PodDisruptionBudget (max 2 pods down simultaneously)
- Graceful termination (terminationGracePeriodSeconds: 60)
\`\`\`

**Monitoring:**

\`\`\`
Key Metrics:

1. Consumer Lag (Critical):
   - Current offset - Committed offset per partition
   - Alert if lag > 10,000 messages (consumers slow)
   - Metric: kafka_consumer_lag

2. Rebalancing Rate:
   - Rebalances per hour
   - Alert if > 10/hour (instability)
   - Metric: consumer_group_rebalance_count

3. Consumer Heartbeat:
   - Last heartbeat timestamp
   - Alert if > session.timeout.ms (consumer unhealthy)

4. Partition Assignment:
   - Partitions per consumer (should be balanced)
   - Alert if uneven (> 20% deviation)

5. Processing Rate:
   - Messages processed per second per consumer
   - Compare to target throughput

Dashboard:
- Consumer lag heatmap (per partition)
- Rebalancing timeline
- Consumer count over time
- Processing rate per consumer
\`\`\`

**Key Takeaways:**

1. **Max parallelism = Number of partitions** (24 partitions = max 24 useful consumers)
2. **CooperativeStickyAssignor minimizes rebalancing disruption** (incremental, not stop-the-world)
3. **Consumer crash causes rebalancing** (session timeout + reassignment = 10-15 sec downtime)
4. **Idempotent processing handles duplicate messages** (reprocessing after crash)
5. **Configure timeouts carefully** (session timeout, heartbeat interval, max poll interval)
6. **Monitor consumer lag religiously** (indicates consumer health and capacity)
7. **Deploy consumers gradually** (minimize rebalancing frequency)

This design handles 1M orders/day with fault tolerance, automatic failover, and minimal downtime during failures.`,
    keyPoints: [
      'Consumer groups enable parallelism: max parallelism = number of partitions',
      'CooperativeStickyAssignor minimizes rebalancing disruption (incremental reassignment)',
      'Consumer crash triggers rebalancing after session timeout (10-15 sec)',
      'Idempotent processing handles duplicate messages from rebalancing',
      'Configure timeouts: session timeout (30s), heartbeat interval (3s), max poll interval (5min)',
      'Monitor consumer lag per partition (alert if > 10K messages)',
    ],
  },
  {
    id: 'kafka-consumers-dq-2',
    question:
      'Compare manual offset management strategies (commitSync vs commitAsync vs per-record commits) for Kafka consumers. For a payment processing system that cannot tolerate duplicate charges, which strategy would you use and why? Walk through failure scenarios and recovery.',
    hint: 'Consider exactly-once processing, performance trade-offs, and failure recovery with different commit strategies.',
    sampleAnswer: `Offset management is critical for reliable message processing. The commit strategy determines message processing guarantees and system performance. Here's a comprehensive analysis:

**Offset Management Fundamentals:**

**Offset**: Position in partition (0, 1, 2, ...)
**Committed Offset**: Last offset saved to Kafka (__consumer_offsets topic)
**Current Position**: Offset consumer is currently reading

**Why Offsets Matter:**
\`\`\`
Consumer processes messages 100-105:

Without committing:
  - Consumer crashes
  - Restarts from last committed offset (e.g., 95)
  - Reprocesses 95-105 (duplicates!) ❌

With committing after processing:
  - Consumer processes 100-105
  - Commits offset 106
  - Consumer crashes
  - Restarts from 106
  - No duplicates ✅
\`\`\`

**Commit Strategies:**

**1. Auto Commit (Default - Not Recommended):**

\`\`\`java
props.put("enable.auto.commit", "true");
props.put("auto.commit.interval.ms", "5000");  // Every 5 seconds

while (true) {
    ConsumerRecords<> records = consumer.poll(Duration.ofMillis(100));
    
    for (ConsumerRecord<> record : records) {
        process(record);  // Process message
    }
    // Offset auto-committed every 5 seconds
}

Timeline:
T0: Poll messages 100-199
T1: Processing messages...
T2: Still processing...
T5: Auto-commit fires → Commits offset 200
T6: Crash during processing message 150

Result:
- Offset 200 committed (up to 199 processed)
- But only processed up to 150 ❌
- Messages 151-199 lost!

Problem: Commits before processing completes
Risk: Message loss
\`\`\`

**2. Manual Commit - Synchronous (commitSync):**

\`\`\`java
props.put("enable.auto.commit", "false");

while (true) {
    ConsumerRecords<> records = consumer.poll(Duration.ofMillis(100));
    
    for (ConsumerRecord<> record : records) {
        process(record);
    }
    
    consumer.commitSync();  // Blocks until commit succeeds
}

Characteristics:
✅ Commit after processing (no loss)
✅ Synchronous (guaranteed commit before continuing)
✅ Simple error handling
❌ Blocks on commit (lower throughput)
❌ Latency spike on commit failure

Timeline:
T0: Poll messages 100-109
T1: Process all messages (100-109)
T2: commitSync() blocks → Wait for Kafka response
T3: Commit succeeds, offset 110 saved
T4: Continue to next poll

If crash at T2 (before commit):
- Restart from last committed offset (100)
- Reprocess 100-109 (duplicates)
- But no loss ✅

Performance:
- Commit every batch: ~10ms overhead per batch
- For 1000 msg/sec, 100 msg/batch: 10 commits/sec = 100ms overhead
- Acceptable for most use cases
\`\`\`

**3. Manual Commit - Asynchronous (commitAsync):**

\`\`\`java
props.put("enable.auto.commit", "false");

while (true) {
    ConsumerRecords<> records = consumer.poll(Duration.ofMillis(100));
    
    for (ConsumerRecord<> record : records) {
        process(record);
    }
    
    consumer.commitAsync((offsets, exception) -> {
        if (exception != null) {
            logger.error("Commit failed: " + exception);
        }
    });
    // Returns immediately, commit happens asynchronously
}

Characteristics:
✅ Non-blocking (higher throughput)
✅ Callback for error handling
⚠️ Commits may arrive out of order
❌ No guarantee commit succeeds before crash

Timeline:
T0: Poll messages 100-109
T1: Process all messages
T2: commitAsync() returns immediately → Continue
T3: Poll messages 110-119
T4: Commit for 100-109 completes (asynchronously)
T5: Process 110-119
T6: commitAsync() for 110-119

If crash at T3 (after commitAsync but before commit):
- commitAsync() in flight, may not complete
- Restart from last committed offset (100)
- Reprocess 100-109 (duplicates)

Performance:
- No blocking overhead
- Higher throughput (5-10% improvement over sync)
- Good for high-volume, non-critical data
\`\`\`

**4. Per-Record Commit (Maximum Safety):**

\`\`\`java
props.put("enable.auto.commit", "false");

while (true) {
    ConsumerRecords<> records = consumer.poll(Duration.ofMillis(100));
    
    for (ConsumerRecord<> record : records) {
        process(record);  // Process single message
        
        // Commit this specific record
        Map<TopicPartition, OffsetAndMetadata> offsets = Collections.singletonMap(
            new TopicPartition(record.topic(), record.partition()),
            new OffsetAndMetadata(record.offset() + 1)
        );
        consumer.commitSync(offsets);  // Commit after each message
    }
}

Characteristics:
✅ Maximum safety (minimal reprocessing)
✅ Precise control
❌ Very slow (commit per message)
❌ 100× overhead vs batch commit

Timeline:
Process message 100 → commitSync() → 10ms
Process message 101 → commitSync() → 10ms
Process message 102 → commitSync() → 10ms

Throughput:
- 10ms per message (processing + commit)
- Max: 100 messages/sec per consumer
- vs Batch commit: 1000+ messages/sec

Use case: Only when absolutely necessary (very rare)
\`\`\`

**Payment Processing System Design:**

**Requirements:**
- No duplicate charges (exactly-once processing)
- No lost payments (100% reliability)
- High throughput (1000 payments/sec)
- Audit trail (track all attempts)

**Chosen Strategy: Manual commitSync + Idempotency**

**Why Not Exactly-Once?**

Kafka exactly-once works only within Kafka ecosystem:
\`\`\`
Read from Kafka → Process → Write to Kafka (exactly-once ✅)

Read from Kafka → Process → External system (payment gateway) → Write to Kafka (exactly-once ❌)

Problem: Payment gateway outside Kafka transaction
Can't roll back charge if commit fails
\`\`\`

**Solution: At-Least-Once + Idempotency**

**Architecture:**

\`\`\`java
public class PaymentConsumer {
    private final KafkaConsumer<String, Payment> consumer;
    private final PaymentGateway paymentGateway;
    private final Database db;
    private final RedisClient redis;
    
    public void processPayments() {
        while (true) {
            ConsumerRecords<String, Payment> records = 
                consumer.poll(Duration.ofMillis(100));
            
            for (ConsumerRecord<String, Payment> record : records) {
                Payment payment = record.value();
                
                try {
                    processPaymentIdempotent(payment);
                    
                } catch (Exception e) {
                    logger.error("Payment processing failed: " + payment.getId(), e);
                    // Don't commit offset → Message will be redelivered
                    throw e;  // Exit loop, restart consumer
                }
            }
            
            // Commit batch after all processed successfully
            try {
                consumer.commitSync();
                logger.info("Committed {} records", records.count());
                
            } catch (CommitFailedException e) {
                logger.error("Commit failed", e);
                // Coordinator moved, will reprocess batch
                throw e;
            }
        }
    }
    
    private void processPaymentIdempotent(Payment payment) {
        String paymentId = payment.getId();
        
        // Step 1: Check if already processed (Redis cache)
        if (redis.exists("processed:" + paymentId)) {
            logger.info("Payment {} already processed, skipping", paymentId);
            return;  // Skip duplicate ✅
        }
        
        // Step 2: Check database (Redis could be cleared)
        if (db.paymentExists(paymentId)) {
            logger.info("Payment {} exists in DB, skipping", paymentId);
            redis.setex("processed:" + paymentId, 86400, "true");  // Cache for 24h
            return;  // Skip duplicate ✅
        }
        
        // Step 3: Process payment with payment gateway
        String idempotencyKey = "pay_" + paymentId;
        
        try {
            Charge charge = paymentGateway.createCharge(
                payment.getAmount(),
                payment.getCurrency(),
                payment.getCustomerId(),
                idempotencyKey  // Payment gateway deduplicates
            );
            
            // Step 4: Save to database + Redis atomically
            db.transaction(() -> {
                db.savePayment(paymentId, charge.getId(), payment.getAmount());
                redis.setex("processed:" + paymentId, 86400, "true");
            });
            
            logger.info("Payment {} processed successfully", paymentId);
            
        } catch (CardDeclinedException e) {
            // Non-retriable error
            logger.warn("Card declined for payment {}", paymentId);
            db.saveFailedPayment(paymentId, "CARD_DECLINED", e.getMessage());
            // Don't throw - mark as processed (don't retry card declines)
            
        } catch (PaymentGatewayException e) {
            // Retriable error (network, timeout)
            logger.error("Payment gateway error for {}", paymentId, e);
            throw e;  // Will cause consumer to restart, reprocess
        }
    }
}
\`\`\`

**Failure Scenarios:**

**Scenario 1: Crash After Processing, Before Commit**

\`\`\`
Timeline:
T1: Consumer polls payments 100-109
T2: Process payment 100 → Charge card $50 ✅
T3: Process payment 101 → Charge card $30 ✅
T4: Process payment 102 → Charge card $20 ✅
T5: Consumer crashes before commitSync() ❌

Recovery:
T6: Consumer restarts
T7: Reads from last committed offset (100)
T8: Reprocess payment 100:
    - Check Redis: exists("processed:pay_100")? YES
    - Skip processing ✅
T9: Reprocess payment 101:
    - Check Redis: exists("processed:pay_101")? YES
    - Skip processing ✅
T10: Reprocess payment 102:
    - Check Redis: exists("processed:pay_102")? YES
    - Skip processing ✅
T11: Process payment 103 (new) → Charge card ✅
T12: commitSync() succeeds
T13: Continue normally

Result:
✅ No duplicate charges (Redis idempotency)
✅ All payments processed
✅ Recovery automatic
\`\`\`

**Scenario 2: Database Write Fails After Charging**

\`\`\`
Timeline:
T1: Process payment 100
T2: Charge card $50 via payment gateway ✅
T3: Save to database → Database connection error ❌
T4: Exception thrown, no commitSync()
T5: Consumer restarts

Recovery:
T6: Reprocess payment 100:
    - Check Redis: exists("processed:pay_100")? NO (wasn't cached)
    - Check database: paymentExists(pay_100)? NO (write failed)
    - Call payment gateway with idempotency key "pay_100"
    - Gateway: "Already charged this key!" → Returns original charge ✅
    - Save to database ✅
    - Cache in Redis ✅
T7: commitSync() succeeds

Result:
✅ No duplicate charge (payment gateway idempotency)
✅ Database eventually consistent
✅ Automatic recovery
\`\`\`

**Scenario 3: Redis Cleared (Cache Miss)**

\`\`\`
Timeline:
T1: Payments 100-200 processed successfully
T2: Redis cleared (deployment, failover, etc.)
T3: Consumer crashes, restarts from offset 150
T4: Reprocess payment 150:
    - Check Redis: exists("processed:pay_150")? NO (cache cleared)
    - Check database: paymentExists(pay_150)? YES ✅
    - Skip processing ✅
T5: Continue normally

Result:
✅ Database as source of truth
✅ Redis cache rebuilt
✅ No duplicate charges
\`\`\`

**Performance Comparison:**

\`\`\`
Scenario: 1000 payments/sec

Strategy 1: Auto-commit (every 5 sec)
Throughput: 1000 msg/sec ✅
Latency: 1ms per message
Risk: Message loss ❌

Strategy 2: commitSync per batch (100 messages)
Throughput: 990 msg/sec (10 commits/sec × 10ms = 100ms overhead)
Latency: 1ms + 1ms per message (amortized commit)
Risk: None (with idempotency) ✅

Strategy 3: commitAsync per batch
Throughput: 1000 msg/sec ✅
Latency: 1ms per message
Risk: Commit may fail silently ⚠️

Strategy 4: commitSync per message
Throughput: 100 msg/sec ❌ (10× slower)
Latency: 11ms per message (1ms + 10ms commit)
Risk: None ✅

Chosen: Strategy 2 (commitSync per batch)
- Good throughput (990 msg/sec)
- No message loss
- Simple error handling
- Production-proven
\`\`\`

**Best Practices:**

\`\`\`java
// 1. Commit after batch, not per message
for (record : records) {
    process(record);
}
consumer.commitSync();  // Batch commit ✅

// 2. Always disable auto-commit for critical data
props.put("enable.auto.commit", "false");

// 3. Use commitAsync for throughput, commitSync on shutdown
while (running) {
    records = consumer.poll(...);
    process(records);
    consumer.commitAsync();  // Non-blocking
}
consumer.commitSync();  // Final sync commit on shutdown

// 4. Commit specific offsets (advanced)
Map<TopicPartition, OffsetAndMetadata> offsets = ...;
consumer.commitSync(offsets);

// 5. Handle commit failures
try {
    consumer.commitSync();
} catch (CommitFailedException e) {
    // Rebalancing occurred, consumer no longer owns partition
    logger.warn("Commit failed: {}", e.getMessage());
    // Don't retry - will reprocess after rebalancing
}
\`\`\`

**Key Takeaways:**

1. **commitSync guarantees commit before continuing** (safest, slight overhead)
2. **commitAsync improves throughput** (non-blocking, but no guarantee)
3. **For payments: commitSync + idempotency** (no duplicates, reliable)
4. **Idempotency has three layers** (Redis cache, database, payment gateway)
5. **Batch commits balance safety and performance** (commit per batch, not per message)
6. **Always disable auto-commit for critical data** (manual control required)
7. **Handle commit failures gracefully** (rebalancing may invalidate commit)

This strategy ensures exactly-once payment processing through idempotency while maintaining high throughput with batch commits.`,
    keyPoints: [
      'commitSync: Blocks until commit succeeds (safest, slight overhead)',
      'commitAsync: Non-blocking, higher throughput (no guarantee before crash)',
      'Payment processing: Use commitSync + idempotency (no duplicate charges)',
      'Idempotency layers: Redis cache, database check, payment gateway deduplication',
      'Batch commits (not per-message) balance safety and performance',
      'Always disable auto-commit for critical data (manual control required)',
    ],
  },
  {
    id: 'kafka-consumers-dq-3',
    question:
      'Design a dead letter queue (DLQ) strategy for Kafka consumers processing order events. What types of errors warrant DLQ, what metadata should you include, how would you monitor DLQ depth, and how would you reprocess messages from the DLQ? Include retry logic and alerting.',
    hint: 'Consider retriable vs non-retriable errors, poison messages, monitoring strategies, and manual intervention workflows.',
    sampleAnswer: `Dead letter queues (DLQ) are essential for handling messages that repeatedly fail processing. A well-designed DLQ strategy prevents poison messages from blocking the main queue while preserving messages for investigation and recovery.

**DLQ Fundamentals:**

**Purpose:**
- Isolate problematic messages
- Prevent blocking healthy message processing
- Enable manual investigation and recovery
- Maintain audit trail of failures

**Architecture:**

\`\`\`
Main Topic: orders
  ↓ Consumer processes
  ↓ Success → Continue
  ↓ Failure → Retry
  ↓ Max retries exceeded → DLQ

DLQ Topic: orders-dlq
  ↓ Monitoring
  ↓ Alerting
  ↓ Manual investigation
  ↓ Optional: Reprocessing
\`\`\`

**Error Classification:**

**1. Retriable Errors (Don't send to DLQ immediately):**

\`\`\`
Transient failures that may succeed on retry:

Network Errors:
- Connection timeout
- Socket timeout
- Network unreachable

Database Errors:
- Connection pool exhausted
- Deadlock detected
- Temporary unavailability

External Service Errors:
- HTTP 500 (Internal Server Error)
- HTTP 503 (Service Unavailable)
- Timeout

Resource Errors:
- Out of memory (transient)
- CPU throttling

Strategy: Retry with exponential backoff
Max retries: 3-5 attempts
After max retries: Send to DLQ
\`\`\`

**2. Non-Retriable Errors (Send to DLQ immediately):**

\`\`\`
Permanent failures that won't succeed on retry:

Data Validation Errors:
- Invalid format (malformed JSON)
- Missing required fields
- Type mismatch (expected number, got string)

Business Logic Errors:
- Order amount negative
- Customer ID doesn't exist
- Product out of stock (business rule)

Configuration Errors:
- Invalid API endpoint
- Missing environment variable
- Invalid credentials

Schema Errors:
- Incompatible message version
- Unknown event type

Strategy: Send to DLQ immediately (no retry)
Log detailed error information
Alert on-call team
\`\`\`

**DLQ Implementation:**

\`\`\`java
public class OrderConsumer {
    private final KafkaConsumer<String, Order> consumer;
    private final KafkaProducer<String, DLQMessage> dlqProducer;
    private final RedisClient redis;  // Track retry counts
    
    public void processOrders() {
        while (true) {
            ConsumerRecords<String, Order> records = consumer.poll(Duration.ofMillis(100));
            
            for (ConsumerRecord<String, Order> record : records) {
                try {
                    processOrder(record.value());
                    
                    // Success: Clear retry count
                    redis.del("retry:" + record.key());
                    
                } catch (NonRetriableException e) {
                    // Immediate DLQ (validation, schema errors)
                    logger.error("Non-retriable error, sending to DLQ: {}", record.key(), e);
                    sendToDLQ(record, e, 0, "NON_RETRIABLE");
                    
                } catch (RetriableException e) {
                    // Check retry count
                    int retryCount = getRetryCount(record.key());
                    
                    if (retryCount >= MAX_RETRIES) {
                        // Max retries exceeded: Send to DLQ
                        logger.error("Max retries exceeded, sending to DLQ: {}", record.key(), e);
                        sendToDLQ(record, e, retryCount, "MAX_RETRIES_EXCEEDED");
                    } else {
                        // Retry with exponential backoff
                        retryCount++;
                        redis.setex("retry:" + record.key(), 3600, String.valueOf(retryCount));
                        
                        long backoffMs = calculateBackoff(retryCount);
                        logger.warn("Retriable error (retry {}/{}), backoff {}ms: {}", 
                                   retryCount, MAX_RETRIES, backoffMs, record.key(), e);
                        
                        Thread.sleep(backoffMs);
                        
                        // Retry same message
                        processOrder(record.value());
                        
                        // If we get here, retry succeeded
                        redis.del("retry:" + record.key());
                    }
                }
            }
            
            consumer.commitSync();
        }
    }
    
    private long calculateBackoff(int retryCount) {
        // Exponential backoff: 1s, 2s, 4s, 8s, 16s
        return (long) Math.pow(2, retryCount - 1) * 1000;
    }
    
    private int getRetryCount(String key) {
        String count = redis.get("retry:" + key);
        return count != null ? Integer.parseInt(count) : 0;
    }
}
\`\`\`

**DLQ Message Structure:**

\`\`\`java
public class DLQMessage {
    // Original message
    private String originalTopic;
    private int originalPartition;
    private long originalOffset;
    private String key;
    private byte[] value;  // Original message payload
    
    // Error information
    private String errorType;  // NON_RETRIABLE, MAX_RETRIES_EXCEEDED, POISON_MESSAGE
    private String errorMessage;
    private String errorStackTrace;
    private int retryCount;
    
    // Metadata
    private String consumerGroupId;
    private String consumerId;
    private Instant firstAttemptTimestamp;
    private Instant lastAttemptTimestamp;
    private Instant dlqTimestamp;
    
    // Context
    private String environment;  // prod, staging
    private String applicationVersion;
    private Map<String, String> additionalContext;
    
    // Reprocessing
    private boolean reprocessed;
    private String reprocessedBy;
    private Instant reprocessedAt;
}

// Send to DLQ
private void sendToDLQ(ConsumerRecord<String, Order> record, Exception e, 
                       int retryCount, String errorType) {
    DLQMessage dlqMessage = DLQMessage.builder()
        .originalTopic(record.topic())
        .originalPartition(record.partition())
        .originalOffset(record.offset())
        .key(record.key())
        .value(serialize(record.value()))
        .errorType(errorType)
        .errorMessage(e.getMessage())
        .errorStackTrace(getStackTrace(e))
        .retryCount(retryCount)
        .consumerGroupId(consumerGroupId)
        .consumerId(consumerId)
        .firstAttemptTimestamp(getFirstAttemptTimestamp(record.key()))
        .lastAttemptTimestamp(Instant.now())
        .dlqTimestamp(Instant.now())
        .environment(System.getenv("ENVIRONMENT"))
        .applicationVersion(getAppVersion())
        .additionalContext(buildContext(record))
        .build();
    
    ProducerRecord<String, DLQMessage> dlqRecord = new ProducerRecord<>(
        "orders-dlq",  // DLQ topic
        record.key(),
        dlqMessage
    );
    
    dlqProducer.send(dlqRecord, (metadata, exception) -> {
        if (exception != null) {
            logger.error("Failed to send to DLQ: {}", record.key(), exception);
            // Fallback: Write to database, file, or secondary DLQ
            writeToDatabaseDLQ(dlqMessage);
        } else {
            logger.info("Sent to DLQ: {} (partition={}, offset={})", 
                       record.key(), metadata.partition(), metadata.offset());
            
            // Increment DLQ metric
            dlqCounter.inc();
        }
    });
}
\`\`\`

**Monitoring Strategy:**

\`\`\`
1. DLQ Depth (Critical Metric):
   - Messages in DLQ topic
   - Alert if > 0 (immediate attention)
   - Dashboard: Real-time DLQ depth

2. DLQ Arrival Rate:
   - Messages/second added to DLQ
   - Alert if > 5/min (something broke)
   - Dashboard: DLQ arrival rate over time

3. Error Type Distribution:
   - Count by error type (NON_RETRIABLE, MAX_RETRIES, etc.)
   - Identify patterns (e.g., all validation errors)
   - Dashboard: Pie chart of error types

4. Affected Orders:
   - Which order IDs in DLQ
   - Customer impact (VIP customers in DLQ?)
   - Dashboard: List of DLQ messages with details

5. Time in DLQ:
   - How long messages sit in DLQ
   - Alert if > 1 hour (needs investigation)
   - Dashboard: Age of oldest DLQ message

Prometheus Metrics:
// DLQ depth
dlq_depth{topic="orders-dlq"} 15

// DLQ arrival rate
dlq_messages_total{error_type="MAX_RETRIES_EXCEEDED"} 10
dlq_messages_total{error_type="NON_RETRIABLE"} 5

// Oldest message age
dlq_oldest_message_age_seconds 3600  // 1 hour old
\`\`\`

**Alerting Rules:**

\`\`\`yaml
# Alert immediately on any DLQ message
- alert: DLQNonEmpty
  expr: dlq_depth > 0
  for: 1m
  severity: critical
  annotations:
    summary: "Messages in DLQ for {{ $labels.topic }}"
    description: "{{ $value }} messages in DLQ, immediate investigation required"
    runbook: "https://wiki.company.com/runbooks/kafka-dlq"

# Alert on high DLQ arrival rate
- alert: DLQHighArrivalRate
  expr: rate(dlq_messages_total[5m]) > 1
  for: 5m
  severity: warning
  annotations:
    summary: "High DLQ arrival rate"
    description: "{{ $value }} messages/sec arriving in DLQ"

# Alert on old messages in DLQ
- alert: DLQStaleMessages
  expr: dlq_oldest_message_age_seconds > 3600
  severity: warning
  annotations:
    summary: "Stale messages in DLQ"
    description: "Oldest message {{ $value }}s old, needs reprocessing"

# Alert on specific error types
- alert: DLQValidationErrors
  expr: rate(dlq_messages_total{error_type="NON_RETRIABLE"}[5m]) > 0.5
  for: 10m
  severity: critical
  annotations:
    summary: "Validation errors increasing"
    description: "Check for schema changes or bad data"
\`\`\`

**DLQ Processing Workflow:**

\`\`\`
Step 1: Detection
- Alert fires (DLQ depth > 0)
- On-call engineer notified (PagerDuty)

Step 2: Investigation
- Query DLQ topic for messages
- Review error type, stack trace, context
- Identify root cause:
  * Code bug?
  * Data issue?
  * External service down?

Step 3: Resolution
Based on root cause:

Case A: Code Bug
1. Fix bug in code
2. Deploy fix
3. Reprocess DLQ messages (see below)

Case B: Bad Data
1. Identify data source
2. Fix upstream data pipeline
3. Decide: Reprocess or discard?

Case C: External Service Down
1. Wait for service recovery
2. Reprocess automatically (retry succeeds)

Step 4: Reprocessing
- Controlled reprocessing from DLQ
- Monitor for failures
- Update reprocessing metadata
\`\`\`

**Reprocessing from DLQ:**

\`\`\`java
public class DLQReprocessor {
    private final KafkaConsumer<String, DLQMessage> dlqConsumer;
    private final KafkaProducer<String, Order> mainProducer;
    
    public void reprocessDLQ(String filter) {
        logger.info("Starting DLQ reprocessing with filter: {}", filter);
        
        dlqConsumer.subscribe(Collections.singletonList("orders-dlq"));
        
        int reprocessed = 0;
        int skipped = 0;
        int failed = 0;
        
        while (true) {
            ConsumerRecords<String, DLQMessage> records = 
                dlqConsumer.poll(Duration.ofMillis(1000));
            
            if (records.isEmpty()) {
                break;  // No more messages
            }
            
            for (ConsumerRecord<String, DLQMessage> record : records) {
                DLQMessage dlqMessage = record.value();
                
                // Apply filter (e.g., only error_type=X, only after date Y)
                if (!matchesFilter(dlqMessage, filter)) {
                    skipped++;
                    continue;
                }
                
                try {
                    // Deserialize original message
                    Order order = deserialize(dlqMessage.getValue());
                    
                    // Send back to main topic
                    ProducerRecord<String, Order> mainRecord = new ProducerRecord<>(
                        dlqMessage.getOriginalTopic(),
                        dlqMessage.getKey(),
                        order
                    );
                    
                    mainProducer.send(mainRecord).get();  // Synchronous
                    
                    // Mark as reprocessed
                    dlqMessage.setReprocessed(true);
                    dlqMessage.setReprocessedBy(System.getenv("USER"));
                    dlqMessage.setReprocessedAt(Instant.now());
                    
                    // Optional: Save reprocessing audit trail
                    saveReprocessingAudit(dlqMessage);
                    
                    reprocessed++;
                    logger.info("Reprocessed: {}", dlqMessage.getKey());
                    
                } catch (Exception e) {
                    failed++;
                    logger.error("Reprocessing failed: {}", dlqMessage.getKey(), e);
                }
            }
            
            dlqConsumer.commitSync();
        }
        
        logger.info("DLQ reprocessing complete: reprocessed={}, skipped={}, failed={}", 
                   reprocessed, skipped, failed);
    }
}

// Usage:
// Reprocess all messages
reprocessor.reprocessDLQ("all");

// Reprocess only MAX_RETRIES_EXCEEDED (transient errors)
reprocessor.reprocessDLQ("error_type=MAX_RETRIES_EXCEEDED");

// Reprocess messages from specific date
reprocessor.reprocessDLQ("date>=2023-06-01");
\`\`\`

**DLQ Best Practices:**

\`\`\`
1. Separate DLQ per main topic
   orders → orders-dlq
   payments → payments-dlq
   
2. Include rich metadata
   - Error details, stack trace
   - Retry history
   - Environment context
   - Timestamps
   
3. Alert immediately on DLQ depth > 0
   - PagerDuty notification
   - Runbook link
   - Dashboard link
   
4. Distinguish error types
   - Retriable vs non-retriable
   - Different handling strategies
   
5. Exponential backoff for retries
   - 1s, 2s, 4s, 8s, 16s
   - Prevent overwhelming failing service
   
6. Monitor DLQ trends
   - Error type distribution
   - Arrival rate
   - Time to resolution
   
7. Automate reprocessing when possible
   - Transient errors (service recovered)
   - Fixed code bugs (deploy + reprocess)
   
8. Preserve audit trail
   - Track reprocessing attempts
   - Who reprocessed, when, why
   
9. Consider secondary DLQ
   - If sending to DLQ fails
   - Fallback to database or file
   
10. Regular DLQ reviews
    - Weekly review of DLQ patterns
    - Identify systemic issues
    - Improve error handling
\`\`\`

**Key Takeaways:**

1. **DLQ prevents poison messages from blocking queue** (isolate failures)
2. **Distinguish retriable vs non-retriable errors** (retry transient, DLQ permanent)
3. **Exponential backoff for retries** (1s, 2s, 4s, 8s, max 5 retries)
4. **Rich metadata in DLQ messages** (error details, context, timestamps)
5. **Alert immediately on DLQ depth > 0** (critical issue needing investigation)
6. **Reprocessing workflow** (investigate → fix → reprocess with filters)
7. **Monitor DLQ trends** (error types, arrival rate, time to resolution)

This DLQ strategy ensures problematic messages don't block healthy processing while providing tools for investigation and recovery.`,
    keyPoints: [
      'DLQ isolates failed messages to prevent blocking main queue',
      'Distinguish errors: Retriable (retry with backoff) vs Non-retriable (immediate DLQ)',
      'Exponential backoff for retries: 1s, 2s, 4s, 8s (max 5 attempts)',
      'Rich DLQ metadata: error details, stack trace, context, retry history',
      'Alert immediately when DLQ depth > 0 (critical issue)',
      'Reprocessing workflow: Investigate root cause → Fix → Reprocess with filters',
    ],
  },
];
