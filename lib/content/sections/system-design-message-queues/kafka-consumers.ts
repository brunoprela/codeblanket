/**
 * Kafka Consumers Section
 */

export const kafkaconsumersSection = {
  id: 'kafka-consumers',
  title: 'Kafka Consumers',
  content: `Kafka consumers read records from Kafka topics. Understanding consumer groups, offset management, rebalancing, and error handling is essential for building scalable,

 fault-tolerant data pipelines.

## Consumer Architecture

### **Basic Consumer Flow:**

\`\`\`
Application Code
      ↓
  Consumer API
      ↓
Fetch Request (poll)
      ↓
Coordinator (Partition Assignment)
      ↓
Kafka Broker (Read from Leader)
      ↓
Deserializer (Bytes → Objects)
      ↓
Application Processing
      ↓
Offset Commit (Mark as Processed)
\`\`\`

### **Consumer Components:**

1. **Group Coordinator**: Manages consumer group membership
2. **Partition Assignment**: Distributes partitions across consumers
3. **Offset Manager**: Tracks processed offsets
4. **Deserializer**: Converts bytes to objects
5. **Rebalance Protocol**: Handles consumer joins/leaves

---

## Consumer Groups

### **What is a Consumer Group?**

A **consumer group** is a set of consumers that cooperate to consume a topic, with each partition consumed by exactly one consumer in the group.

\`\`\`
Topic: "orders" (4 partitions)

Consumer Group: "order-processors"
┌──────────┬──────────┬──────────┬──────────┐
│  Part 0  │  Part 1  │  Part 2  │  Part 3  │
└────┬─────┴────┬─────┴────┬─────┴────┬─────┘
     │          │          │          │
  Consumer 1  Consumer 2  Consumer 3  Consumer 4

Each partition → Exactly one consumer in group
Each consumer → One or more partitions
\`\`\`

### **Key Benefits:**

1. **Load Balancing**: Work distributed across consumers
2. **Scalability**: Add consumers to increase throughput
3. **Fault Tolerance**: If consumer fails, partitions reassigned
4. **Independent Processing**: Different groups process independently

---

## Consumer Group Scaling

### **Scenario 1: Consumers = Partitions (Optimal)**

\`\`\`
Topic: 4 partitions
Group: 4 consumers

Consumer 1: Part 0  ✅ Each consumer has 1 partition
Consumer 2: Part 1  ✅ Full parallelism
Consumer 3: Part 2  ✅ Maximum throughput
Consumer 4: Part 3
\`\`\`

### **Scenario 2: Consumers < Partitions**

\`\`\`
Topic: 6 partitions
Group: 3 consumers

Consumer 1: Part 0, Part 3  ✅ Still works
Consumer 2: Part 1, Part 4  ✅ Each consumer handles 2
Consumer 3: Part 2, Part 5  ⚠️ Higher load per consumer
\`\`\`

### **Scenario 3: Consumers > Partitions**

\`\`\`
Topic: 3 partitions
Group: 5 consumers

Consumer 1: Part 0  ✅ Active
Consumer 2: Part 1  ✅ Active
Consumer 3: Part 2  ✅ Active
Consumer 4: (idle)  ❌ No work
Consumer 5: (idle)  ❌ No work

Max parallelism = Number of partitions
Extra consumers waste resources
\`\`\`

### **Key Insight:**

**Partitions limit parallelism!**

\`\`\`
Want to scale to 10 consumers?
→ Need at least 10 partitions

Scaling strategy:
1. Estimate max consumers needed
2. Create topic with that many partitions
3. Can't increase partitions later without data migration
   (Well, you can add partitions, but existing keys may remap)
\`\`\`

---

## Offset Management

### **What is an Offset?**

An **offset** is a unique sequential integer assigned to each record in a partition.

\`\`\`
Partition 0:
Offset:  0    1    2    3    4    5    6    7
Record: [A]  [B]  [C]  [D]  [E]  [F]  [G]  [H]
                               ↑
                    Current Position: Offset 4
                    Last Committed: Offset 3

Consumer has read up to offset 4
Consumer has committed offset 3 (processed successfully)
\`\`\`

### **Committed Offset:**

The **committed offset** is the last offset that has been saved to Kafka, indicating "I've successfully processed up to here."

\`\`\`
Scenario:
1. Consumer reads messages at offsets 0, 1, 2
2. Processes them successfully
3. Commits offset 3 (next to read)
4. Consumer crashes
5. New consumer starts from offset 3
6. No messages lost or reprocessed

✅ At-least-once delivery with idempotency
\`\`\`

### **Offset Storage:**

\`\`\`
Kafka stores committed offsets in internal topic:
__consumer_offsets

Entry:
{
  group: "order-processors",
  topic: "orders",
  partition: 0,
  offset: 1234,
  metadata: "...",
  timestamp: 1609459200000
}

✅ Highly available (replicated)
✅ Survives consumer restarts
✅ Shared across all consumers in group
\`\`\`

---

## Offset Commit Strategies

### **1. Auto Commit (Easiest)**

\`\`\`java
Properties props = new Properties();
props.put("enable.auto.commit", "true");
props.put("auto.commit.interval.ms", "5000");  // Every 5 seconds

while (true) {
    ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
    for (ConsumerRecord<String, String> record : records) {
        process (record);  // Process message
    }
    // Offset auto-committed every 5 seconds
}

✅ Simple, no code needed
❌ Risk of message loss or duplication
\`\`\`

**Failure Scenario:**
\`\`\`
1. Consumer polls offsets 0-99
2. Processes offsets 0-49
3. Crashes (before 5-second commit)
4. Restart from last commit (offset 0)
5. Reprocesses 0-49 (duplicates!)

❌ At-least-once, not exactly-once
\`\`\`

### **2. Manual Commit - Synchronous**

\`\`\`java
props.put("enable.auto.commit", "false");

while (true) {
    ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
    for (ConsumerRecord<String, String> record : records) {
        process (record);
    }
    consumer.commitSync();  // Block until committed
}

✅ Guaranteed commit after processing
✅ No message loss
❌ Blocks on commit (lower throughput)
\`\`\`

### **3. Manual Commit - Asynchronous**

\`\`\`java
props.put("enable.auto.commit", "false");

while (true) {
    ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
    for (ConsumerRecord<String, String> record : records) {
        process (record);
    }
    consumer.commitAsync((offsets, exception) -> {
        if (exception != null) {
            logger.error("Commit failed: " + exception);
        }
    });
}

✅ Non-blocking (higher throughput)
✅ Callback for error handling
⚠️ Commits may arrive out of order
\`\`\`

### **4. Manual Commit Per Record**

\`\`\`java
while (true) {
    ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
    for (ConsumerRecord<String, String> record : records) {
        process (record);
        
        // Commit after each record
        Map<TopicPartition, OffsetAndMetadata> offsets = Collections.singletonMap(
            new TopicPartition (record.topic(), record.partition()),
            new OffsetAndMetadata (record.offset() + 1)
        );
        consumer.commitSync (offsets);
    }
}

✅ Precise control
✅ Minimal reprocessing on failure
❌ Very slow (commit per message)
\`\`\`

### **5. Manual Commit Per Batch (Recommended)**

\`\`\`java
props.put("enable.auto.commit", "false");

while (true) {
    ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
    
    for (ConsumerRecord<String, String> record : records) {
        try {
            process (record);  // Idempotent processing
        } catch (Exception e) {
            logger.error("Failed to process: " + record, e);
            // Handle error: skip, DLQ, retry
        }
    }
    
    // Commit entire batch
    try {
        consumer.commitSync();
    } catch (CommitFailedException e) {
        logger.error("Commit failed: " + e);
        // Handle: retry, alert
    }
}

✅ Good throughput (batch commits)
✅ Controlled semantics
✅ Error handling per record
🎯 Production best practice
\`\`\`

---

## Rebalancing

### **What is Rebalancing?**

**Rebalancing** is the process of redistributing partitions among consumers when the group membership changes.

### **Rebalance Triggers:**

1. **Consumer joins group** (scale up)
2. **Consumer leaves/crashes** (scale down, failure)
3. **Partitions added to topic**
4. **Consumer exceeds session timeout** (heartbeat fails)

### **Rebalance Process:**

\`\`\`
Initial State:
Consumer 1: Part 0, Part 1
Consumer 2: Part 2, Part 3

Consumer 3 joins:
1. Group coordinator detects new member
2. Rebalance initiated
3. All consumers stop processing
4. Coordinator assigns partitions:
   - Consumer 1: Part 0
   - Consumer 2: Part 1
   - Consumer 3: Part 2, Part 3
5. Consumers resume from committed offsets

⚠️ Stop-the-world event (no processing during rebalance)
\`\`\`

### **Rebalance Types:**

**1. Eager Rebalancing (Default)**
\`\`\`
1. All consumers revoke ALL partitions
2. Coordinator reassigns partitions
3. Consumers resume

⏱️ Downtime: 1-5 seconds (all consumers stopped)
\`\`\`

**2. Cooperative (Incremental) Rebalancing**
\`\`\`
1. Only affected partitions revoked
2. Reassignment in multiple rounds
3. Most consumers keep processing

⏱️ Minimal downtime (only reassigned partitions stopped)
✅ Kafka 2.4+ (set partition.assignment.strategy)
\`\`\`

### **Minimizing Rebalance Impact:**

\`\`\`java
// 1. Increase session timeout
props.put("session.timeout.ms", "30000");  // 30 seconds
// Consumer can be unresponsive for 30s before rebalance

// 2. Decrease heartbeat interval
props.put("heartbeat.interval.ms", "3000");  // 3 seconds
// More frequent heartbeats = faster failure detection

// 3. Increase max poll interval
props.put("max.poll.interval.ms", "300000");  // 5 minutes
// Consumer can process for 5 minutes before rebalance

// 4. Use cooperative rebalancing
props.put("partition.assignment.strategy", 
          "org.apache.kafka.clients.consumer.CooperativeStickyAssignor");
\`\`\`

---

## Consumer Configuration

### **Essential Settings:**

\`\`\`java
Properties props = new Properties();

// 1. Bootstrap Servers (Required)
props.put("bootstrap.servers", "broker1:9092,broker2:9092,broker3:9092");

// 2. Group ID (Required)
props.put("group.id", "order-processors");
// Identifies consumer group

// 3. Deserializers (Required)
props.put("key.deserializer", "org.apache.kafka.common.serialization.StringDeserializer");
props.put("value.deserializer", "org.apache.kafka.common.serialization.StringDeserializer");

// 4. Auto Offset Reset
props.put("auto.offset.reset", "earliest");
// Options: earliest (from beginning), latest (new messages), none (error)

// 5. Commit Strategy
props.put("enable.auto.commit", "false");  // Manual control

// 6. Session Timeout
props.put("session.timeout.ms", "10000");  // 10 seconds

// 7. Max Poll Records
props.put("max.poll.records", "500");  // Fetch up to 500 records per poll

// 8. Max Poll Interval
props.put("max.poll.interval.ms", "300000");  // 5 minutes

// 9. Fetch Min/Max
props.put("fetch.min.bytes", "1");     // Min bytes to fetch
props.put("fetch.max.wait.ms", "500");  // Max wait time
\`\`\`

---

## Consumer Patterns

### **1. Exactly-Once Processing**

\`\`\`java
// Idempotent processing + manual offset management

props.put("isolation.level", "read_committed");  // Only read committed transactions

while (true) {
    ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
    
    for (ConsumerRecord<String, String> record : records) {
        // Process idempotently (check if already processed)
        if (!isProcessed (record.key())) {
            processMessage (record);
            markAsProcessed (record.key());  // Store in database
        }
    }
    
    // Commit offset in same transaction as processing
    commitOffsetInTransaction();
}

✅ Exactly-once semantics
✅ No duplicates, no losses
❌ Requires transactional storage
\`\`\`

### **2. At-Least-Once Processing**

\`\`\`java
// Most common pattern: Process + Commit

while (true) {
    ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
    
    for (ConsumerRecord<String, String> record : records) {
        processMessage (record);  // Idempotent processing
    }
    
    consumer.commitSync();  // Commit after batch
}

✅ Simpler than exactly-once
✅ No messages lost
⚠️ Possible duplicates on failure (handle with idempotency)
\`\`\`

### **3. Dead Letter Queue (DLQ)**

\`\`\`java
// Handle poison messages that repeatedly fail

while (true) {
    ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
    
    for (ConsumerRecord<String, String> record : records) {
        try {
            processMessage (record);
        } catch (Exception e) {
            int retries = getRetryCount (record);
            if (retries >= MAX_RETRIES) {
                // Send to DLQ after max retries
                sendToDLQ(record, e);
                logger.error("Sent to DLQ: " + record);
            } else {
                // Increment retry count and reprocess later
                incrementRetryCount (record);
                throw e;  // Trigger reprocessing
            }
        }
    }
    
    consumer.commitSync();
}

✅ Prevents poison messages from blocking queue
✅ Allows investigation of failed messages
\`\`\`

---

## Consumer Best Practices

### **1. Idempotent Processing:**

\`\`\`java
// Design consumers to handle duplicate messages

void processOrder(String orderId, OrderData data) {
    if (orderAlreadyProcessed (orderId)) {
        logger.info("Order already processed: " + orderId);
        return;  // Skip duplicate
    }
    
    // Process order
    saveOrder (orderId, data);
    markAsProcessed (orderId);
}

✅ Safe to reprocess messages
✅ Handles at-least-once delivery
\`\`\`

### **2. Batch Processing:**

\`\`\`java
// Process multiple records efficiently

while (true) {
    ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
    
    // Batch database writes
    List<Order> orders = records.stream()
        .map (record -> parseOrder (record.value()))
        .collect(Collectors.toList());
    
    batchInsert (orders);  // Single database call
    consumer.commitSync();
}

✅ Higher throughput (fewer DB calls)
✅ Better resource utilization
\`\`\`

### **3. Graceful Shutdown:**

\`\`\`java
Runtime.getRuntime().addShutdownHook (new Thread(() -> {
    logger.info("Shutting down consumer...");
    consumer.wakeup();  // Interrupt poll()
}));

try {
    while (true) {
        ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
        // Process records
        consumer.commitSync();
    }
} catch (WakeupException e) {
    // Expected during shutdown
} finally {
    consumer.commitSync();  // Final commit
    consumer.close();  // Graceful close (triggers rebalance)
}

✅ Commit in-flight work
✅ Trigger rebalance for other consumers
\`\`\`

### **4. Monitor Consumer Lag:**

\`\`\`
Consumer Lag = Latest Offset - Committed Offset

Example:
Partition 0:
  Latest offset: 10000
  Committed offset: 9500
  Lag: 500 messages

High lag indicates:
- Consumer too slow
- Traffic burst
- Need to scale (add consumers)

Monitor with: JMX metrics, Kafka Manager, Confluent Control Center
\`\`\`

---

## Consumer in System Design Interviews

### **Discussion Points:**

1. **Consumer Group Design:**
   - "12 partitions, 12 consumers for parallelism"
   - "Consumer group per use case (analytics, email, warehouse)"

2. **Offset Management:**
   - "Manual commit after batch for control"
   - "Store offsets in database for exactly-once"

3. **Failure Handling:**
   - "Idempotent processing for duplicates"
   - "Dead letter queue for poison messages"

4. **Scaling:**
   - "Monitor lag, scale consumers as needed"
   - "Max consumers = Number of partitions"

5. **Performance:**
   - "Batch database writes for throughput"
   - "Tune fetch size and poll interval"

### **Example:**

\`\`\`
Interviewer: "How would you consume order events at high scale?"

You:
"Kafka consumer group with these characteristics:

1. Consumer Group per Purpose:
   - Group 1: Payment processing
   - Group 2: Email notifications
   - Group 3: Analytics pipeline

2. Scaling:
   - Topic: 12 partitions
   - Payment processors: 12 consumers (1:1 ratio)
   - Email service: 6 consumers (2 partitions each)
   - Analytics: 3 consumers (4 partitions each)

3. Offset Management:
   - Manual commit after batch
   - Idempotent processing (check order_id)
   - Store processed IDs in Redis (dedup)

4. Error Handling:
   - Try-catch per message
   - Max 3 retries with exponential backoff
   - Dead letter queue for failed orders
   - Alert on DLQ depth > 100

5. Monitoring:
   - Consumer lag per partition (alert if > 1000)
   - Processing rate (messages/sec)
   - Error rate (failed messages/total)

Capacity:
100K orders/sec, 12 partitions = 8.3K/partition
Each consumer processes 8.3K/sec, well within capacity
Monitor lag, scale to 24 consumers if lag increases"
\`\`\`

---

## Key Takeaways

1. **Consumer groups enable parallel processing** → Each partition consumed by one consumer
2. **Max parallelism = Number of partitions** → Plan partition count carefully
3. **Manual offset commit recommended** → Better control than auto-commit
4. **Rebalancing stops processing** → Minimize with cooperative rebalancing
5. **Idempotent processing handles duplicates** → At-least-once delivery pattern
6. **Monitor consumer lag** → Scale when lag increases
7. **Dead letter queue for poison messages** → Prevent blocking
8. **Batch processing improves throughput** → Fewer I/O operations
9. **Graceful shutdown commits offsets** → No message loss
10. **In interviews: Discuss scaling, offset management, error handling** → Show production knowledge

---

**Next:** We'll explore **Kafka Streams**—stream processing, KStream vs KTable, windowing, and stateful processing.`,
};
