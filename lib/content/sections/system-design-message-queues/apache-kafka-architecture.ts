/**
 * Apache Kafka Architecture Section
 */

export const apachekafkaarchitectureSection = {
  id: 'apache-kafka-architecture',
  title: 'Apache Kafka Architecture',
  content: `Apache Kafka is a distributed streaming platform designed for high-throughput, fault-tolerant, and scalable event streaming. It\'s widely used at companies like LinkedIn, Netflix, Uber, and Airbnb for real-time data pipelines and stream processing.

## What is Apache Kafka?

**Kafka** = Distributed Commit Log + Pub/Sub Messaging + Stream Processing

### **Key Capabilities:**1. **Publish/Subscribe** to streams of records
2. **Store** streams durably and reliably
3. **Process** streams in real-time

### **Not Just a Message Queue:**

\`\`\`
Traditional Message Queue:
Producer → Queue → Consumer (message deleted after consumption)

Kafka:
Producer → Kafka → Consumer 1 (reads from offset 0)
                 → Consumer 2 (reads from offset 100)
                 → Consumer 3 (reads from offset 50)

Messages retained (configurable: days/weeks)
Multiple consumers can read independently
Can replay historical data
\`\`\`

---

## Core Concepts

### **1. Topic**

A **topic** is a category or feed name to which records are published.

\`\`\`
Topic: "user-activity"
  Messages: 
    - { user_id: "123", action: "login", timestamp: ... }
    - { user_id: "456", action: "purchase", timestamp: ... }
    - { user_id: "789", action: "logout", timestamp: ... }

Think of it as: A database table, but append-only
\`\`\`

**Characteristics:**
- Multi-subscriber (many consumers can read)
- Append-only log (messages never modified)
- Retention policy (7 days default, configurable)
- Partitioned for parallelism

### **2. Partition**

A **partition** is an ordered, immutable sequence of records.

\`\`\`
Topic: "orders" (3 partitions)

Partition 0: [Msg0, Msg3, Msg6, Msg9, ...]
Partition 1: [Msg1, Msg4, Msg7, Msg10, ...]
Partition 2: [Msg2, Msg5, Msg8, Msg11, ...]

Each partition:
- Ordered sequence (within partition)
- Independently consumed
- Stored on different brokers (distributed)
\`\`\`

**Why Partitions?**1. **Scalability**: Topic can exceed single server capacity
2. **Parallelism**: Multiple consumers process simultaneously
3. **Throughput**: Write to multiple partitions in parallel

**Partition Assignment:**
\`\`\`
message = {
  key: "user_123",  // Partition key
  value: { ... }
}

Partition = hash (key) % num_partitions

Same key → Same partition → Ordering guaranteed per key
No key → Round-robin across partitions
\`\`\`

### **3. Offset**

An **offset** is a unique sequential ID for each message within a partition.

\`\`\`
Partition 0:
Offset: 0    1    2    3    4    5
Msg:   [A] [B] [C] [D] [E] [F]
                     ↑
                Consumer's current position (offset 3)

Consumer tracks: "I've processed up to offset 3"
Next read: Offset 4 (message E)
\`\`\`

**Offset Tracking:**
- Each consumer maintains its offset per partition
- Stored in special Kafka topic: \`__consumer_offsets\`
- Enables replay: Reset offset to reprocess messages
- Commit offset: Acknowledge successful processing

### **4. Broker**

A **broker** is a Kafka server that stores data and serves clients.

\`\`\`
Kafka Cluster (3 brokers)

Broker 1:                Broker 2:                Broker 3:
- Topic A, Partition 0   - Topic A, Partition 1   - Topic A, Partition 2
- Topic B, Partition 1   - Topic B, Partition 0   - Topic C, Partition 0
- Topic C, Partition 1   - Topic C, Partition 2   - Topic B, Partition 2

Each broker:
- Handles read/write requests
- Stores partitions
- Replicates data
- Communicates with other brokers
\`\`\`

**Broker Responsibilities:**1. Receive messages from producers
2. Assign offsets
3. Store messages to disk (commit log)
4. Serve messages to consumers
5. Replicate partitions for fault tolerance
6. Participate in leader election

### **5. Replication**

Each partition has **replicas** across multiple brokers for fault tolerance.

\`\`\`
Topic: "orders", Partition 0, Replication Factor: 3

Broker 1 (Leader):    [Offset 0][Offset 1][Offset 2][Offset 3]
                       ↓ Replicate
Broker 2 (Follower):  [Offset 0][Offset 1][Offset 2][Offset 3]
                       ↓ Replicate
Broker 3 (Follower):  [Offset 0][Offset 1][Offset 2][Offset 3]

Leader: Handles all reads and writes
Followers: Replicate from leader, standby for failover
\`\`\`

**Leader and Followers:**

- **Leader**: One replica handles all client requests
- **Follower**: Other replicas sync from leader
- **In-Sync Replica (ISR)**: Followers caught up with leader
- **Failover**: If leader fails, follower promoted to leader

**Example:**
\`\`\`
Normal Operation:
Producer → Leader (Broker 1) → Followers sync

Broker 1 Failure:
1. Broker 1 down (leader lost)
2. Zookeeper detects failure
3. Broker 2 promoted to leader
4. Broker 2 now handles requests
5. When Broker 1 recovers, becomes follower

✅ Zero downtime
✅ No data loss (replicated)
\`\`\`

---

## Kafka Architecture Components

### **Cluster Architecture**

\`\`\`
                      ┌─────────────────────┐
                      │  ZooKeeper Ensemble │
                      │  (Coordination)     │
                      └──────────┬──────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
   ┌────▼────┐              ┌────▼────┐              ┌────▼────┐
   │ Broker 1│              │ Broker 2│              │ Broker 3│
   │         │              │         │              │         │
   │ Part A-0│◄──replicate──│ Part A-0│◄──replicate──│ Part A-0│
   │ Part B-1│              │ Part B-0│              │ Part B-2│
   └────▲────┘              └────▲────┘              └────▲────┘
        │                        │                        │
        │                        │                        │
   ┌────┴────┐              ┌────┴────┐              ┌────┴────┐
   │Producer │              │Producer │              │Consumer │
   │         │              │         │              │ Group   │
   └─────────┘              └─────────┘              └─────────┘
\`\`\`

### **1. ZooKeeper (Coordination Service)**

**Responsibilities:**
- **Broker coordination**: Track alive brokers
- **Leader election**: Elect partition leaders
- **Configuration**: Store topic metadata
- **Consumer coordination**: (legacy, now moved to Kafka)

**Example:**
\`\`\`
ZooKeeper stores:
/brokers/ids/1 → { "host": "broker1.com", "port": 9092, ... }
/brokers/ids/2 → { "host": "broker2.com", "port": 9092, ... }
/brokers/ids/3 → { "host": "broker3.com", "port": 9092, ... }

/brokers/topics/orders → { "partitions": { "0": [1,2,3], ... } }
                                              ↑
                                          Leader: 1, Replicas: [1,2,3]
\`\`\`

**Note:** Kafka 2.8+ supports **KRaft** (Kafka Raft), removing ZooKeeper dependency. KRaft is now production-ready (Kafka 3.3+).

### **2. Producers**

Producers publish records to Kafka topics.

**Producer Flow:**
\`\`\`
1. Producer creates message
2. Serialization (convert to bytes)
3. Partitioner determines partition
4. Send to leader broker
5. Leader writes to log
6. Leader replicates to followers
7. Acknowledgment to producer
\`\`\`

**Producer Configuration:**
\`\`\`javascript
{
  "bootstrap.servers": "broker1:9092,broker2:9092,broker3:9092",
  "key.serializer": "StringSerializer",
  "value.serializer": "JsonSerializer",
  "acks": "all",  // Wait for all replicas
  "retries": 3,
  "compression.type": "snappy"
}
\`\`\`

### **3. Consumers**

Consumers subscribe to topics and process records.

**Consumer Flow:**
\`\`\`
1. Consumer joins consumer group
2. Partition assignment (group coordinator)
3. Fetch records from assigned partitions
4. Process records
5. Commit offset (mark as processed)
\`\`\`

**Consumer Configuration:**
\`\`\`javascript
{
  "bootstrap.servers": "broker1:9092,broker2:9092",
  "group.id": "order-processing-group",
  "key.deserializer": "StringDeserializer",
  "value.deserializer": "JsonDeserializer",
  "auto.offset.reset": "earliest",
  "enable.auto.commit": false  // Manual commit
}
\`\`\`

### **4. Consumer Groups**

A **consumer group** is a group of consumers that cooperate to consume a topic.

\`\`\`
Topic: "orders" (4 partitions)

Consumer Group: "order-processors"
- Consumer 1: Partition 0
- Consumer 2: Partition 1
- Consumer 3: Partition 2
- Consumer 4: Partition 3

Each partition consumed by exactly ONE consumer in the group
Multiple consumer groups can read the same topic independently
\`\`\`

**Scaling:**
\`\`\`
Scenario 1: More consumers than partitions
Topic: 3 partitions
Consumer Group: 5 consumers

Assignment:
- Consumer 1: Partition 0
- Consumer 2: Partition 1
- Consumer 3: Partition 2
- Consumer 4: Idle (no partition)
- Consumer 5: Idle (no partition)

Max parallelism = Number of partitions
\`\`\`

\`\`\`
Scenario 2: Add more partitions
Topic: 6 partitions (scaled up)
Consumer Group: 5 consumers

Assignment:
- Consumer 1: Partition 0, Partition 5
- Consumer 2: Partition 1
- Consumer 3: Partition 2
- Consumer 4: Partition 3
- Consumer 5: Partition 4

Better load distribution
\`\`\`

---

## Kafka\'s Write Path

### **Step-by-Step Write Flow:**

\`\`\`
1. Producer sends message
   ↓
2. Partitioner determines partition (based on key)
   ↓
3. Message sent to partition leader
   ↓
4. Leader appends to local commit log
   ↓
5. Leader replicates to followers (ISR)
   ↓
6. Followers acknowledge replication
   ↓
7. Leader sends acknowledgment to producer
   ↓
8. Producer receives confirmation
\`\`\`

### **Write Guarantees (acks):**

**acks=0 (Fire and forget):**
\`\`\`
Producer → Leader → [Don't wait for acknowledgment]

Latency: Lowest
Throughput: Highest
Durability: None (message might be lost)

Use case: Metrics, logs (loss acceptable)
\`\`\`

**acks=1 (Leader acknowledgment):**
\`\`\`
Producer → Leader (writes to log) → [ACK]
           Followers (sync asynchronously)

Latency: Medium
Throughput: High
Durability: Medium (lost if leader fails before replication)

Use case: Most applications (good balance)
\`\`\`

**acks=all (All ISR acknowledgment):**
\`\`\`
Producer → Leader (writes to log)
           ↓ Replicate to all ISR
           Followers (all acknowledge)
           ↓
           [ACK to producer]

Latency: Highest
Throughput: Lower
Durability: Highest (no data loss)

Use case: Critical data (payments, orders)
\`\`\`

---

## Kafka's Read Path

### **Step-by-Step Read Flow:**

\`\`\`
1. Consumer requests messages from partition leader
   ↓
2. Leader reads from disk (or page cache)
   ↓
3. Leader sends batch of messages to consumer
   ↓
4. Consumer processes messages
   ↓
5. Consumer commits offset (acknowledges)
   ↓
6. Offset stored in __consumer_offsets topic
\`\`\`

### **Zero-Copy Optimization:**

Kafka uses **sendfile()** system call for efficient data transfer:

\`\`\`
Traditional (4 copies):
Disk → OS Buffer → Kafka App Buffer → Socket Buffer → NIC

Zero-Copy (2 copies):
Disk → OS Buffer → NIC (via DMA)
                   ↑
              No application buffer

Result: 2-3× faster, lower CPU usage
\`\`\`

---

## Kafka Storage Model

### **Commit Log Structure:**

\`\`\`
Partition 0 Directory:
  00000000000000000000.log   (segment 0)
  00000000000000005000.log   (segment 5000)
  00000000000000010000.log   (segment 10000)
  00000000000000010000.index (index for segment 10000)
  00000000000000010000.timeindex

Each segment:
- Max 1GB (default)
- Immutable (append-only)
- Indexed for fast lookup
\`\`\`

### **Segment Compaction:**

Kafka retains data in two ways:

**1. Time-Based Retention:**
\`\`\`
retention.ms = 7 days (default)

Day 0: [Msg1, Msg2, Msg3]
Day 7: [Msg1, Msg2, Msg3] (still available)
Day 8: [Msg1, Msg2, Msg3] deleted

Use case: Event logs, metrics
\`\`\`

**2. Log Compaction:**
\`\`\`
Keep only latest value for each key

Before compaction:
Key A → Value 1 (offset 0)
Key B → Value 1 (offset 1)
Key A → Value 2 (offset 2)
Key B → Value 2 (offset 3)
Key A → Value 3 (offset 4)

After compaction:
Key A → Value 3 (offset 4)  ← Latest value for A
Key B → Value 2 (offset 3)  ← Latest value for B

Use case: Database changelog, state snapshots
\`\`\`

---

## Kafka Performance Characteristics

### **Why Kafka is Fast:**1. **Sequential Disk I/O**
   - Append-only log (sequential writes)
   - Sequential writes faster than random (even on HDD)
   - Modern OS page cache optimization

2. **Zero-Copy Transfer**
   - Kernel-level data transfer (no application buffer)
   - Reduced CPU usage
   - Higher throughput

3. **Batch Processing**
   - Producer batches messages
   - Consumer fetches batches
   - Amortizes network overhead

4. **Compression**
   - Supports Snappy, LZ4, GZIP, Zstd
   - Compress entire batch (better ratio)
   - Decompressed by consumers

5. **Partition Parallelism**
   - Parallel writes across partitions
   - Parallel reads by multiple consumers
   - Linear scalability

### **Throughput Numbers:**

\`\`\`
Single Kafka Broker (typical):
- Writes: 100K - 500K messages/sec
- Reads: 500K - 1M messages/sec
- Throughput: 100MB/s - 500MB/s

Kafka Cluster (3 brokers, replication factor 3):
- Writes: 1M+ messages/sec
- Reads: 5M+ messages/sec
- Throughput: 1GB/s+

Real-World (LinkedIn):
- 7 trillion messages/day
- Peak: 13 million messages/sec
- 7 petabytes/day
\`\`\`

---

## Kafka Use Cases

### **1. Event Streaming**
\`\`\`
User Activity Tracking:
User actions → Kafka → Analytics, Recommendations, Personalization

Example: Netflix (tracking viewing patterns)
\`\`\`

### **2. Log Aggregation**
\`\`\`
Application Logs:
App1, App2, App3 → Kafka → Elasticsearch, S3, Splunk

Example: Uber (centralizing logs from 1000+ microservices)
\`\`\`

### **3. Stream Processing**
\`\`\`
Real-Time Analytics:
Click Stream → Kafka → Kafka Streams → Dashboards

Example: LinkedIn (real-time member metrics)
\`\`\`

### **4. Data Integration (CDC)**
\`\`\`
Change Data Capture:
Database → Kafka → Data Warehouse, Cache, Search

Example: Shopify (syncing MySQL changes to Elasticsearch)
\`\`\`

### **5. Commit Log / Event Sourcing**
\`\`\`
Event Store:
Commands → Kafka (immutable log) → Read Models

Example: Banking (transaction audit log)
\`\`\`

---

## Kafka vs Traditional Message Queues

| Feature | Kafka | RabbitMQ |
|---------|-------|----------|
| **Throughput** | Very High (millions/sec) | Moderate (thousands/sec) |
| **Latency** | Low (ms) | Very Low (μs) |
| **Persistence** | Always (commit log) | Optional |
| **Message Retention** | Configurable (days/weeks) | Until consumed |
| **Replay** | Yes (seek to offset) | No (message deleted) |
| **Ordering** | Per-partition | Per-queue |
| **Scalability** | Horizontal (add brokers) | Limited |
| **Use Case** | Event streaming, high throughput | Task queues, routing |

---

## Kafka in System Design Interviews

### **When to Propose Kafka:**

✅ **High throughput** (millions of events/sec)
✅ **Event streaming** (user activity, logs, metrics)
✅ **Multiple consumers** (analytics, ML, storage)
✅ **Replay capability** (reprocess historical data)
✅ **Durability** (persist data for days/weeks)

### **Trade-offs to Mention:**

**Pros:**
- Extremely high throughput
- Durable, replicated storage
- Horizontal scalability
- Supports multiple consumer groups
- Message replay

**Cons:**
- More complex (ZooKeeper/KRaft, partitions, offsets)
- Higher operational overhead
- Not ideal for low-latency messaging (μs)
- Overkill for simple task queues
- Learning curve for developers

### **Example Interview Discussion:**

\`\`\`
Interviewer: "Design a system to track user activity on a website"

You:
"I'll use Kafka for event streaming. Here\'s why:

Architecture:
User Actions → API Servers → Kafka Topic: 'user-activity'
                                ↓ Consumer Group 1: Real-time analytics
                                ↓ Consumer Group 2: ML recommendations
                                ↓ Consumer Group 3: Data warehouse (batch)

Topic Design:
- Topic: 'user-activity', 12 partitions
- Partition key: user_id (ordering per user)
- Retention: 7 days (replay for ML retraining)
- Replication: 3× (fault tolerance)

Scale Estimation:
- 10M DAU, 100 actions/user/day = 1B events/day
- 1B / 86,400 sec = ~12K events/sec (average)
- Peak (3×): 36K events/sec
- Kafka handles easily (brokers support 100K+ msg/sec)

Why Kafka over SQS/RabbitMQ:
- Multiple consumers (analytics, ML, warehouse)
- High throughput (12K sustained, 36K peak)
- Replay capability (ML wants historical data)
- Kafka Streams for real-time processing

Trade-offs:
- More complex than SQS (manage cluster)
- Could use managed Kafka (AWS MSK, Confluent Cloud)
- Consider cost (vs serverless SQS)
"
\`\`\`

---

## Key Takeaways

1. **Kafka = Distributed commit log + Pub/sub + Stream processing**2. **Topics partitioned for scalability** → Parallel reads/writes
3. **Offsets enable replay** → Reprocess historical data
4. **Replication ensures durability** → Leader/follower model
5. **Consumer groups enable scaling** → Add consumers for parallelism
6. **ZooKeeper/KRaft for coordination** → Leader election, metadata
7. **Sequential I/O + Zero-copy = High throughput** → Millions of messages/sec
8. **Retention configurable** → Time-based or log compaction
9. **Use for high-throughput event streaming** → Not for low-latency task queues
10. **Understand partitions, offsets, consumer groups** → Core to Kafka interviews

---

**Next:** We'll explore **Kafka Producers** in depth—partitioning strategies, idempotence, transactions, and performance tuning.`,
};
