/**
 * Discussion Questions for Apache Kafka Architecture
 */

import { QuizQuestion } from '../../../types';

export const apachekafkaarchitectureQuiz: QuizQuestion[] = [
  {
    id: 'kafka-arch-dq-1',
    question:
      "Explain Kafka's partitioning strategy and how it enables horizontal scalability. Design a Kafka topic for a high-traffic e-commerce platform processing 1 million orders per day. How many partitions would you create, how would you choose partition keys, and how would this design handle hot keys (e.g., celebrity orders)?",
    hint: 'Consider ordering guarantees, parallelism, rebalancing, and strategies for handling skewed data distribution.',
    sampleAnswer: `Kafka's partitioning is fundamental to its scalability, enabling parallel processing while maintaining ordering guarantees within partitions. Here's a comprehensive analysis for an e-commerce platform:

**Kafka Partitioning Fundamentals:**

Partitions divide a topic into multiple ordered logs distributed across brokers. Each partition:
- Is an ordered, immutable sequence of records
- Has a leader (handles reads/writes) and followers (replicas)
- Can be consumed independently
- Maintains ordering within the partition only

**Scalability Through Partitioning:**

1. **Write Scalability:** Multiple partitions = parallel writes across brokers
2. **Read Scalability:** Multiple consumers = parallel reads (one consumer per partition in a group)
3. **Storage Scalability:** Partitions distributed across brokers (topic larger than single server)

**E-Commerce Topic Design: "orders"**

**Requirements Analysis:**
- Volume: 1 million orders/day = 11.57 orders/sec average
- Peak: 3× during flash sales = 34.7 orders/sec
- Ordering requirement: Orders per customer must be ordered
- Data: ~2KB per order message (JSON with items)
- Retention: 7 days (168 hours)

**Partition Count Decision:**

**Calculation:**
\`\`\`
Throughput per partition (rule of thumb):
- Write: ~10 MB/sec per partition
- Read: ~30 MB/sec per partition

Our throughput:
- Peak: 35 orders/sec × 2KB = 70 KB/sec (well below limits)

Consumer parallelism:
- Want ability to scale to 20 consumers (future growth)
- Max parallelism = Number of partitions
- Therefore: Need at least 20 partitions

Recommendation: 24 partitions

Reasoning:
1. Allows 24 parallel consumers (current need: 5-10)
2. Room for growth (2× current peak)
3. Divisible by many numbers (2, 3, 4, 6, 8, 12) - flexible consumer groups
4. Not too many (rebalancing overhead)
5. Each partition: 35/24 = 1.45 orders/sec (manageable)
\`\`\`

**Partition Key Strategy:**

**Option 1: Customer ID (Recommended)**
\`\`\`java
ProducerRecord<String, Order> record = new ProducerRecord<>(
    "orders",
    order.getCustomerId(),  // Partition key
    order
);

Partition = hash(customerId) % 24

Benefits:
✅ Orders per customer guaranteed ordered
✅ Good distribution (many customers)
✅ Customer service queries efficient (all orders in one partition)

Drawbacks:
⚠️ Hot customers (celebrities) cause hot partitions
⚠️ Single customer limited to one partition's throughput
\`\`\`

**Option 2: Order ID**
\`\`\`java
record = new ProducerRecord<>(
    "orders",
    order.getOrderId(),  // Partition key
    order
);

Benefits:
✅ Perfect distribution (each order independent)
✅ No hot keys

Drawbacks:
❌ No ordering guarantees per customer
❌ Customer queries require scanning all partitions
\`\`\`

**Option 3: Composite Key (Customer + Timestamp)**
\`\`\`java
String key = order.getCustomerId() + "_" + (System.currentTimeMillis() / 60000);
// Group customer orders by minute

record = new ProducerRecord<>("orders", key, order);

Benefits:
✅ Orders within same minute ordered per customer
✅ Better distribution than pure customer ID
✅ Time-based partitioning

Drawbacks:
⚠️ Orders across minute boundaries not ordered
⚠️ More complex key management
\`\`\`

**Chosen Strategy: Customer ID with Hot Key Mitigation**

**Handling Hot Keys (Celebrity Orders):**

**Problem:**
\`\`\`
Normal customer: 1-2 orders/day
Celebrity (e.g., Kylie Jenner posts about product): 10,000 orders/hour

If Kylie's customerId = "customer_123":
hash("customer_123") % 24 = Partition 5

Partition 5: 10,000 orders/hour (overloaded!)
Other partitions: ~40 orders/hour (underutilized)

Result: Partition 5 becomes bottleneck ❌
\`\`\`

**Solution 1: Custom Partitioner with VIP Detection**
\`\`\`java
public class VipAwarePartitioner implements Partitioner {
    private static final Set<String> VIP_CUSTOMERS = loadVipCustomers();
    
    @Override
    public int partition(String topic, Object key, byte[] keyBytes, 
                        Object value, byte[] valueBytes, Cluster cluster) {
        int numPartitions = cluster.partitionCountForTopic(topic);
        String customerId = (String) key;
        
        // VIP customers: Use round-robin across dedicated partitions
        if (VIP_CUSTOMERS.contains(customerId)) {
            // Dedicated partitions for VIPs: last 4 partitions (20-23)
            int vipPartition = 20 + (ThreadLocalRandom.current().nextInt(4));
            return vipPartition;
        }
        
        // Regular customers: Hash to partitions 0-19
        return Math.abs(Utils.murmur2(keyBytes)) % 20;
    }
}

Configuration:
- Partitions 0-19: Regular customers (hash-based, ordered per customer)
- Partitions 20-23: VIP customers (round-robin, ordering relaxed)

Benefits:
✅ Hot keys distributed across multiple partitions
✅ Regular customers unaffected
✅ VIP throughput 4× higher (4 partitions vs 1)

Tradeoffs:
⚠️ VIP ordering not guaranteed (acceptable for high volume)
⚠️ Requires VIP detection logic
\`\`\`

**Solution 2: Sub-key Salting**
\`\`\`java
// Add random salt to hot keys
String partitionKey;
if (isHotCustomer(order.getCustomerId())) {
    // Salt hot keys: distribute across 4 partitions
    int salt = ThreadLocalRandom.current().nextInt(4);
    partitionKey = order.getCustomerId() + "_" + salt;
} else {
    partitionKey = order.getCustomerId();
}

record = new ProducerRecord<>("orders", partitionKey, order);

Benefits:
✅ Hot customer orders distributed across multiple partitions
✅ Dynamic (no hardcoded VIP list)

Tradeoffs:
⚠️ Ordering lost for hot customers
⚠️ Consumer must aggregate across sub-keys if needed
\`\`\`

**Solution 3: Separate Topic for High-Volume Customers**
\`\`\`
Topics:
- orders-regular (24 partitions, customer_id key)
- orders-vip (48 partitions, customer_id + salt key)

Producer logic:
if (order.getCustomer().isVip()) {
    producer.send(new ProducerRecord<>("orders-vip", saltedKey, order));
} else {
    producer.send(new ProducerRecord<>("orders-regular", customerId, order));
}

Benefits:
✅ Complete isolation (VIPs don't affect regular customers)
✅ Different configuration (VIP: higher throughput, lower retention)
✅ Clear separation

Tradeoffs:
❌ Topic proliferation
❌ Consumers must subscribe to both topics
❌ Operational overhead (2 topics to manage)
\`\`\`

**Recommended: Solution 1 (Custom Partitioner)**

**Complete Architecture:**

\`\`\`
Kafka Cluster: 6 brokers

Topic: orders
Partitions: 24 (0-23)
Replication Factor: 3 (data on 3 brokers)
Retention: 7 days

Partition Assignment:
Broker 1: Partitions 0, 6, 12, 18 (leader)
Broker 2: Partitions 1, 7, 13, 19 (leader)
Broker 3: Partitions 2, 8, 14, 20 (leader)
Broker 4: Partitions 3, 9, 15, 21 (leader)
Broker 5: Partitions 4, 10, 16, 22 (leader)
Broker 6: Partitions 5, 11, 17, 23 (leader)

Each partition also has 2 follower replicas on other brokers

Consumer Group: order-processors
Consumers: 12 instances (at peak)
Assignment: Each consumer handles 2 partitions

Storage per partition:
- Orders/day per partition: 1M / 24 = 41,666
- Size per partition: 41,666 × 2KB = ~83 MB/day
- 7-day retention: 83 MB × 7 = ~581 MB per partition
- Total topic size: 581 MB × 24 = ~14 GB (easily manageable)
\`\`\`

**Handling Growth:**

**Scaling to 10M orders/day:**
\`\`\`
Current: 24 partitions, 1M orders/day
Future: 10M orders/day

Option 1: Keep 24 partitions
- 10M / 24 = 416K orders/partition/day
- 416K × 2KB = 832 MB/day per partition (acceptable)
- Peak: 115 orders/sec per partition (manageable)
✅ No changes needed!

Option 2: Increase to 48 partitions (if needed)
- Create new topic: orders-v2 (48 partitions)
- Dual-write to both topics (migration period)
- Migrate consumers to orders-v2
- Deprecate orders-v1

⚠️ Note: Can't increase partitions in-place without data migration
         (existing keys would remap to different partitions)
\`\`\`

**Monitoring:**

\`\`\`
Metrics to track:
1. Records per second per partition (detect hot partitions)
2. Bytes per second per partition (detect data skew)
3. Consumer lag per partition (detect slow consumers)
4. Leader distribution (ensure balanced across brokers)

Alerts:
- Partition throughput > 80% of capacity
- Consumer lag > 10,000 messages
- Partition size > 1 GB/day (unexpected growth)

Dashboard:
- Heatmap: Throughput per partition (identify hot partitions)
- Time series: Consumer lag per partition
- Bar chart: Partition size distribution
\`\`\`

**Key Takeaways:**

1. **Partition count balances parallelism and overhead** (24 partitions = sweet spot)
2. **Customer ID as partition key ensures ordering** per customer
3. **Hot keys handled with custom partitioner** (VIP partitions or salting)
4. **Over-partition for future growth** (easier than repartitioning)
5. **Monitor partition distribution** (detect and fix hot partitions)
6. **Replication factor 3 provides fault tolerance** (lose 2 brokers, still available)

This design handles 1M orders/day with room to scale 10×, maintains ordering per customer, and mitigates hot key problems through intelligent partitioning strategies.`,
    keyPoints: [
      'Partitioning enables scalability: parallel writes, reads, and storage distribution',
      '24 partitions recommended for 1M orders/day (allows 24 parallel consumers)',
      'Customer ID as partition key maintains ordering per customer',
      'Hot keys (celebrity orders) handled with custom partitioner or salting',
      'Custom partitioner dedicates partitions to VIP customers (prevents bottlenecks)',
      'Monitor partition throughput and consumer lag to detect hot partitions',
    ],
  },
  {
    id: 'kafka-arch-dq-2',
    question:
      "Explain Kafka's replication mechanism, ISR (In-Sync Replicas), and how it provides fault tolerance. Walk through what happens when the leader for a partition fails, how a new leader is elected, and what guarantees are maintained. What are the trade-offs between acks=0, acks=1, and acks=all?",
    hint: 'Consider data durability, throughput, latency, and failure scenarios. Think about leader election, replication lag, and the role of ZooKeeper/KRaft.',
    sampleAnswer: `Kafka's replication mechanism is critical for fault tolerance and data durability. Understanding ISR, leader election, and acks configuration is essential for designing reliable systems.

**Replication Architecture:**

Each partition has multiple replicas across different brokers:
\`\`\`
Topic: orders, Partition 0, Replication Factor: 3

Broker 1 (Leader):    [Msg0, Msg1, Msg2, Msg3, Msg4]
Broker 2 (Follower):  [Msg0, Msg1, Msg2, Msg3, Msg4]
Broker 3 (Follower):  [Msg0, Msg1, Msg2, Msg3, Msg4]

Leader: Handles all reads and writes
Followers: Continuously replicate from leader
\`\`\`

**In-Sync Replicas (ISR):**

ISR = Set of replicas that are "caught up" with leader

**Criteria for being in ISR:**
1. Replica is alive (sending heartbeats to ZooKeeper)
2. Replica has fetched messages within last 10 seconds (default: replica.lag.time.max.ms)
3. Replica has fetched all messages up to high watermark

**Example:**
\`\`\`
Leader (Broker 1):    [Msg0, Msg1, Msg2, Msg3, Msg4, Msg5] (offset 5)
Follower 1 (Broker 2): [Msg0, Msg1, Msg2, Msg3, Msg4, Msg5] (offset 5) ✅ In ISR
Follower 2 (Broker 3): [Msg0, Msg1, Msg2, Msg3]             (offset 3) ❌ Out of ISR (lagging)

ISR = {Broker 1, Broker 2}
\`\`\`

**Why ISR Matters:**
- Committed messages = Replicated to all ISR members
- Only ISR members eligible for leader election
- acks=all waits for all ISR (not all replicas)

**Leader Failure and Election:**

**Scenario: Leader fails**

\`\`\`
Initial State:
Leader: Broker 1, ISR: {Broker 1, Broker 2, Broker 3}

Time 0: Broker 1 crashes (leader fails)

Time 1: ZooKeeper detects leader failure (heartbeat timeout)
  - Broker 1 session expired
  - Controller broker notified

Time 2: Controller initiates leader election
  - Candidates: ISR members (Broker 2, Broker 3)
  - Selection: First replica in ISR list (Broker 2)

Time 3: Broker 2 promoted to leader
  - Update metadata: Leader = Broker 2
  - Notify all brokers of new leader

Time 4: Producers/consumers redirect to Broker 2
  - Metadata refresh (automatic)
  - New leader serves requests

Time 5: Broker 3 (follower) starts replicating from Broker 2

Failover time: Typically 5-10 seconds
  - Heartbeat timeout: 6 seconds (default)
  - Leader election: 1-2 seconds
  - Metadata propagation: 1-2 seconds

Total downtime: ~10 seconds (unclean.leader.election.enable=false)
\`\`\`

**Data Guarantees During Failover:**

**With acks=all (Recommended):**
\`\`\`
Scenario:
1. Producer sends Msg5 (acks=all)
2. Leader (Broker 1) receives Msg5
3. Leader waits for ISR replication
4. Follower (Broker 2) replicates Msg5 ✅
5. Follower (Broker 3) replicates Msg5 ✅
6. Leader crashes before sending ACK to producer

Result after failover:
- Msg5 replicated to Broker 2 and 3
- Broker 2 becomes leader (has Msg5)
- Producer retries (no ACK received)
- Broker 2 detects duplicate (idempotent producer)
- Broker 2 returns success

✅ No data loss
✅ No duplicates (with idempotence)
\`\`\`

**Without acks=all (acks=1):**
\`\`\`
Scenario:
1. Producer sends Msg5 (acks=1)
2. Leader (Broker 1) receives Msg5
3. Leader ACKs immediately (doesn't wait for followers)
4. Producer receives ACK
5. Leader crashes before followers replicate

Result after failover:
- Msg5 not replicated to Broker 2
- Broker 2 becomes leader (doesn't have Msg5)
- Msg5 lost ❌

❌ Data loss possible
\`\`\`

**Acknowledgment Levels (acks):**

**acks=0 (Fire and Forget):**

\`\`\`
Producer: Sends message, doesn't wait for acknowledgment

Flow:
1. Producer sends Msg5
2. Producer continues immediately (doesn't wait)
3. Network error? Producer doesn't know ❌
4. Broker crashes? Producer doesn't know ❌

Guarantees: None
Durability: Lowest (message may be lost)
Throughput: Highest (no waiting)
Latency: Lowest (no network round-trip)

Use case: Metrics, logs where loss acceptable
Example: Temperature sensor readings (1000/sec, losing 5 not critical)

Configuration:
props.put("acks", "0");
\`\`\`

**acks=1 (Leader Acknowledgment):**

\`\`\`
Producer: Waits for leader to write to log

Flow:
1. Producer sends Msg5
2. Leader writes to local log
3. Leader sends ACK to producer
4. Producer receives ACK
5. Followers replicate asynchronously (after ACK)

Guarantees: Message written to leader
Durability: Medium (lost if leader fails before replication)
Throughput: High (doesn't wait for followers)
Latency: Medium (one network round-trip)

Use case: Most applications (good balance)

Failure scenario:
- Leader ACKs
- Leader crashes before followers replicate
- New leader doesn't have message
- Message lost ❌

Configuration:
props.put("acks", "1");
\`\`\`

**acks=all (or acks=-1) (All ISR Acknowledgment):**

\`\`\`
Producer: Waits for all in-sync replicas to acknowledge

Flow:
1. Producer sends Msg5
2. Leader writes to local log
3. Leader waits for followers in ISR to replicate
4. Follower 1 replicates, sends ACK to leader
5. Follower 2 replicates, sends ACK to leader
6. Leader sends ACK to producer
7. Producer receives ACK

Guarantees: Message replicated to all ISR
Durability: Highest (survives min.insync.replicas-1 failures)
Throughput: Lower (waits for all ISR)
Latency: Highest (multiple network round-trips)

Use case: Critical data (payments, orders, financial transactions)

Configuration:
props.put("acks", "all");
props.put("min.insync.replicas", "2");  // Require at least 2 replicas in ISR

With min.insync.replicas=2:
- Replication factor 3: Can lose 1 broker (2 in ISR ✅)
- Replication factor 3, min=2: If only 1 replica in ISR, writes blocked ❌

Tradeoff: Availability vs Durability
\`\`\`

**Trade-offs Summary:**

| acks | Durability | Throughput | Latency | Data Loss | Use Case |
|------|-----------|-----------|---------|-----------|----------|
| 0 | None | Highest | Lowest | Possible | Metrics, logs |
| 1 | Medium | High | Medium | If leader fails | Most apps |
| all | Highest | Lower | Highest | No (with min.insync.replicas) | Critical data |

**Unclean Leader Election:**

\`\`\`
unclean.leader.election.enable=true (not recommended)

Scenario:
- Leader crashes
- All ISR replicas down (e.g., power outage in datacenter)
- Non-ISR replica available (e.g., lagging replica in different datacenter)

With unclean=false (default):
- Partition unavailable until ISR replica recovers
- Data integrity preserved ✅
- Availability sacrificed ⏸️

With unclean=true:
- Non-ISR replica elected leader
- Data loss (messages not replicated to this replica) ❌
- Availability maintained ✅

Trade-off: Availability vs Consistency
\`\`\`

**Recommended: unclean=false for critical data**

**Real-World Example: Payment System**

\`\`\`
Requirements:
- No lost payments
- No duplicate charges
- High availability

Configuration:
- Replication factor: 3 (data on 3 brokers)
- min.insync.replicas: 2 (require 2 replicas for write)
- acks: all (wait for both ISR replicas)
- unclean.leader.election.enable: false (never lose data)
- Producer idempotence: true (no duplicates)

Result:
- Can lose 1 broker (2 replicas remain ✅)
- Can't lose 2 brokers (partition unavailable until recovery)
- No data loss
- No duplicates
- 99.99% availability (rare to lose 2 brokers simultaneously)

Failure Handling:
1 broker down:
  - Partition still available (2 in ISR) ✅
  - Continue processing payments ✅

2 brokers down:
  - Partition unavailable ❌
  - Writes blocked (min.insync.replicas=2 not met)
  - Reads from remaining replica still possible
  - Wait for broker recovery

Recovery:
  - Broker restarts
  - Catches up from remaining replicas
  - Joins ISR
  - Partition fully available again
\`\`\`

**Monitoring:**

\`\`\`
Key metrics:
1. UnderReplicatedPartitions
   - Partitions with replicas not in ISR
   - Alert if > 0 (replication lagging)

2. OfflinePartitionsCount
   - Partitions with no leader
   - Alert immediately (data unavailable)

3. ISR shrink/expand rate
   - Frequent ISR changes indicate instability
   - Investigate slow replicas

4. Leader election rate
   - Frequent leader elections indicate broker instability
   - Check broker health

Alerts:
- UnderReplicatedPartitions > 0 for 5 minutes → Warning
- OfflinePartitionsCount > 0 → Critical
- Leader election rate > 10/hour → Warning
\`\`\`

**Key Takeaways:**

1. **ISR guarantees data durability** (committed = replicated to all ISR)
2. **acks=all + min.insync.replicas=2 prevents data loss** (recommended for critical data)
3. **Leader election takes 5-10 seconds** (automatic, no data loss with acks=all)
4. **Replication factor 3, min.insync.replicas=2** (balance: tolerate 1 failure, no data loss)
5. **Trade-off: Availability vs Consistency** (unclean leader election)
6. **Monitor ISR status** (detect replication issues early)

This replication mechanism makes Kafka highly fault-tolerant, surviving broker failures without data loss when properly configured.`,
    keyPoints: [
      'ISR (In-Sync Replicas) = replicas caught up with leader, eligible for leader election',
      'acks=all + min.insync.replicas=2 prevents data loss (requires replication to 2 replicas)',
      'Leader election takes 5-10 seconds on failure (automatic, no data loss)',
      'acks=0: No guarantees, highest throughput; acks=1: Medium durability; acks=all: No data loss',
      'Replication factor 3, min.insync.replicas=2 tolerates 1 broker failure',
      'unclean.leader.election=false preserves data integrity (never elect non-ISR leader)',
    ],
  },
  {
    id: 'kafka-arch-dq-3',
    question:
      'Compare Kafka to other messaging systems (RabbitMQ, AWS SQS, Redis Pub/Sub). For each scenario below, which technology would you choose and why: (1) Real-time click stream analytics with replay capability, (2) Traditional task queue for background jobs, (3) Low-latency notifications to connected clients, (4) Event sourcing for audit-critical financial system.',
    hint: 'Consider throughput, latency, persistence, replay capability, operational complexity, and cost.',
    sampleAnswer: `Choosing the right messaging system depends on specific requirements. Let's analyze each technology and match them to the scenarios.

**Technology Comparison:**

**1. Apache Kafka:**
- **Model:** Distributed commit log
- **Throughput:** Very high (millions/sec)
- **Latency:** Low (milliseconds)
- **Persistence:** Yes (days/weeks, configurable)
- **Replay:** Yes (seek to any offset)
- **Ordering:** Per-partition
- **Delivery:** At-least-once, exactly-once
- **Scalability:** Horizontal (add brokers)
- **Ops Complexity:** High (cluster management)
- **Cost:** Medium (self-hosted) to High (managed)

**2. RabbitMQ:**
- **Model:** Message broker
- **Throughput:** Medium (tens of thousands/sec)
- **Latency:** Very low (microseconds to milliseconds)
- **Persistence:** Optional (durable queues)
- **Replay:** No (messages deleted after consumption)
- **Ordering:** Per-queue
- **Delivery:** At-least-once, at-most-once
- **Scalability:** Limited (vertical + clustering)
- **Ops Complexity:** Medium
- **Cost:** Low (self-hosted)

**3. AWS SQS:**
- **Model:** Managed message queue
- **Throughput:** Unlimited (Standard), 3K/sec (FIFO)
- **Latency:** Low (tens of milliseconds)
- **Persistence:** Yes (up to 14 days)
- **Replay:** No
- **Ordering:** Best-effort (Standard), FIFO (FIFO queue)
- **Delivery:** At-least-once
- **Scalability:** Automatic
- **Ops Complexity:** Zero (fully managed)
- **Cost:** Very low (pay-per-request)

**4. Redis Pub/Sub:**
- **Model:** In-memory pub/sub
- **Throughput:** Very high (hundreds of thousands/sec)
- **Latency:** Very low (sub-millisecond)
- **Persistence:** No (fire-and-forget)
- **Replay:** No
- **Ordering:** No guarantees
- **Delivery:** At-most-once (subscribers must be connected)
- **Scalability:** Limited (single instance)
- **Ops Complexity:** Low
- **Cost:** Low

**Scenario Analysis:**

**Scenario 1: Real-Time Click Stream Analytics with Replay**

**Requirements:**
- High volume (millions of clicks/day)
- Real-time processing (seconds latency)
- Replay capability (reprocess historical data)
- Multiple consumers (analytics, ML, data warehouse)
- Data retention (30 days)

**Choice: Apache Kafka ✅**

**Why Kafka:**

1. **High Throughput:**
\`\`\`
Click volume: 10M clicks/day = 115 clicks/sec average
Peak: 3× = 345 clicks/sec
Kafka throughput: 1M+ messages/sec ✅
\`\`\`

2. **Replay Capability:**
\`\`\`
Use case: New ML model needs training on historical data

With Kafka:
- Set consumer offset to 30 days ago
- Replay all clicks from last 30 days
- Train model on historical data
- No special ETL needed ✅

With RabbitMQ/SQS:
- Messages deleted after consumption ❌
- Need separate data lake for historical data
- Complex ETL pipeline
\`\`\`

3. **Multiple Consumers:**
\`\`\`
Consumer Group 1: Real-time analytics (dashboards)
Consumer Group 2: ML recommendations (model training)
Consumer Group 3: Data warehouse (batch loading)
Consumer Group 4: Fraud detection (real-time alerts)

Each group independent:
- Read at their own pace
- Different offsets
- Don't affect each other ✅
\`\`\`

4. **Data Retention:**
\`\`\`
retention.ms = 30 days (2,592,000,000 ms)

Click data retained for 30 days
Any consumer can read any message within 30 days
Perfect for analytics use cases ✅
\`\`\`

**Architecture:**
\`\`\`
Website → Load Balancer → API Servers → Kafka Topic: "clicks"
                                              ↓ Consumer Group 1
                                           Real-time analytics
                                              ↓ Consumer Group 2
                                           ML model training (replay from day 0)
                                              ↓ Consumer Group 3
                                           Data warehouse (batch, once/day)
                                              ↓ Consumer Group 4
                                           Fraud detection (real-time)
\`\`\`

**Why Not Others:**
- RabbitMQ: ❌ No replay, messages deleted after consumption
- SQS: ❌ No replay (max 14 days retention, no offset seeking)
- Redis Pub/Sub: ❌ No persistence, no replay, at-most-once

**Scenario 2: Traditional Task Queue for Background Jobs**

**Requirements:**
- Task distribution (image processing, email sending)
- Load balancing across workers
- Reliable delivery (no lost jobs)
- Low ops overhead
- Moderate throughput (1,000 jobs/hour)

**Choice: AWS SQS (if on AWS) or RabbitMQ (self-hosted) ✅**

**Why SQS (AWS environment):**

1. **Zero Ops:**
\`\`\`
- No servers to manage
- Auto-scaling (no capacity planning)
- High availability (built-in)
- Pay-per-use (no idle costs) ✅
\`\`\`

2. **Perfect for Task Queues:**
\`\`\`
Producer: API server (enqueues background jobs)
Queue: SQS Standard queue
Consumers: Worker instances (process jobs, auto-scaled)

Pattern:
1. User uploads image
2. API server enqueues job {"image_id": "123", "task": "resize"}
3. Worker polls queue, receives job
4. Worker processes image
5. Worker deletes message (ACK)

Load balancing automatic (workers compete for messages) ✅
\`\`\`

3. **Reliability:**
\`\`\`
Visibility timeout: 5 minutes
- Worker receives message → Invisible to others for 5 min
- Worker crashes → Message visible again after 5 min
- Another worker picks it up ✅

Dead letter queue:
- After 3 failed attempts → Moved to DLQ
- Manual investigation ✅
\`\`\`

4. **Cost:**
\`\`\`
Volume: 1,000 jobs/hour = 24,000 jobs/day
SQS cost: 24,000 / 1M × $0.40 = $0.0096/day = $0.29/month

Essentially free! ✅

vs Kafka cluster: $500+/month (3 brokers)
vs RabbitMQ instance: $50+/month (t3.medium)
\`\`\`

**Why RabbitMQ (self-hosted):**

If not on AWS or need lower latency:
- Simple task queue pattern (perfect fit)
- Easy to set up and operate
- Lower latency than SQS (<10ms vs ~50ms)
- Work queues well-documented, battle-tested

**Why Not Others:**
- Kafka: ❌ Overkill (complex, expensive for simple task queue)
- Redis Pub/Sub: ❌ No persistence (lost if subscriber disconnected)

**Scenario 3: Low-Latency Notifications to Connected Clients**

**Requirements:**
- Real-time (sub-second latency)
- Push notifications to web/mobile clients
- High concurrency (100K concurrent connections)
- Fire-and-forget (no persistence needed)

**Choice: Redis Pub/Sub ✅**

**Why Redis Pub/Sub:**

1. **Lowest Latency:**
\`\`\`
Redis latency: <1ms (in-memory)
vs Kafka: 5-10ms
vs RabbitMQ: 1-5ms
vs SQS: 50-100ms

For notifications (user comments, likes, real-time updates):
- <1ms latency critical ✅
- Users expect instant feedback
\`\`\`

2. **Simple Pub/Sub Model:**
\`\`\`
Backend: Publishes notification to channel "user:123:notifications"
Frontend: Subscribes to channel "user:123:notifications"
Redis: Instantly delivers to all subscribers

Example:
1. User A comments on User B's post
2. Backend: PUBLISH user:B:notifications {"type": "comment", "from": "A"}
3. User B's connected clients receive instantly ✅
4. No persistence needed (notification shown once)
\`\`\`

3. **High Concurrency:**
\`\`\`
Redis handles 100K+ concurrent subscribers easily
Each WebSocket connection = 1 Redis subscription
Memory footprint low (no message persistence)
\`\`\`

**Architecture:**
\`\`\`
API Server → Redis Pub/Sub (channel per user) → WebSocket Server → Clients

Example:
1. User A likes User B's post
2. API server: PUBLISH user:B:notifications '{"type": "like", "from": "A"}'
3. WebSocket server (subscribed to user:B:*): Receives message
4. WebSocket server pushes to User B's connected browsers/apps
5. User B sees notification instantly (<100ms total) ✅
\`\`\`

**Why Not Others:**
- Kafka: ❌ Overkill, higher latency (not sub-millisecond)
- RabbitMQ: ⚠️ Could work, but more complex than Redis
- SQS: ❌ Pull-based (not push), higher latency

**Alternative: Redis Streams**

If need persistence + replay:
\`\`\`
XADD user:123:notifications * type like from userA
XREAD BLOCK 0 STREAMS user:123:notifications 0

Benefits:
✅ Persistent (retained for X time)
✅ Replay capability
✅ Consumer groups (multiple subscribers)

Still low latency (~1ms) but more features than Pub/Sub
\`\`\`

**Scenario 4: Event Sourcing for Audit-Critical Financial System**

**Requirements:**
- Immutable audit log (all events forever)
- Replay capability (reconstruct state at any point)
- Exactly-once processing (no duplicates)
- Strict ordering (per account)
- High durability (no data loss)

**Choice: Apache Kafka ✅**

**Why Kafka:**

1. **Immutable Log:**
\`\`\`
Kafka = append-only commit log (perfect for event sourcing)

Events (immutable):
Event 1: AccountCreated {account_id: "A1", balance: 0}
Event 2: MoneyDeposited {account_id: "A1", amount: 100}
Event 3: MoneyWithdrawn {account_id: "A1", amount: 30}

Current state derived by replaying events:
replay(Event 1) → balance: 0
replay(Event 2) → balance: 100
replay(Event 3) → balance: 70

✅ Complete audit trail (who, when, what)
✅ Can reconstruct state at any point in time
✅ Never delete events (infinite retention)
\`\`\`

2. **Infinite Retention:**
\`\`\`
retention.ms = -1 (infinite)

or

log.cleanup.policy = compact (keep latest per key)

For audit/compliance:
- Retain all events forever
- Multi-tiered storage (hot: SSD, cold: S3) ✅
\`\`\`

3. **Exactly-Once Processing:**
\`\`\`
Critical for financial transactions:
- No duplicate charges
- No lost transactions

Kafka exactly-once:
props.put("enable.idempotence", "true");
props.put("transactional.id", "account-processor-1");

producer.beginTransaction();
  producer.send(event1);
  producer.send(event2);
  producer.sendOffsetsToTransaction(offsets, groupId);
producer.commitTransaction();

✅ All events written atomically
✅ No duplicates, no losses
\`\`\`

4. **Ordering:**
\`\`\`
Partition by account_id:
- All events for account A1 → Partition 5
- Processed in order ✅

Example:
Partition 5:
  Event 1: Deposit $100 (t1)
  Event 2: Withdraw $30 (t2)
  Event 3: Deposit $50 (t3)

Processed sequentially:
  0 → 100 → 70 → 120 ✅

If out of order:
  0 → 30 (withdraw before deposit) ❌ (prevented by partitioning)
\`\`\`

5. **Replay and Time Travel:**
\`\`\`
Use case: Tax audit for year 2022

replay(account_id, from: "2022-01-01", to: "2022-12-31"):
- Seek to offset at 2022-01-01
- Replay all events for that year
- Reconstruct account state at any point
- Generate transaction history ✅

Use case: Bug in transaction processing

- Discover bug affected accounts from June 1-15
- Replay events for affected accounts
- Reprocess with bug fix
- Generate corrected state ✅
\`\`\`

**Architecture:**
\`\`\`
Banking API → Kafka Topic: "account-events" (event sourcing)
                    ↓ Partition by account_id
              Consumer: State Projector
                    ↓ Materializes current state
              Database: account_balances (current state)
                    ↓ Query API
              Read Model: Get current balance

Write path (commands):
1. User withdraws $50
2. API validates (check balance in read model)
3. API publishes event: MoneyWithdrawn {account: "A1", amount: 50}
4. Event persisted to Kafka (immutable) ✅

Read path (queries):
1. User checks balance
2. API queries read model (materialized state)
3. Returns current balance ✅

Audit path:
1. Auditor requests transaction history
2. Replay events from Kafka (source of truth)
3. Generate complete audit trail ✅
\`\`\`

**Configuration:**
\`\`\`
replication.factor = 3
min.insync.replicas = 2
acks = all
enable.idempotence = true
retention.ms = -1  // Infinite
log.cleanup.policy = none  // Never delete

Result:
✅ No data loss (replication + acks=all)
✅ No duplicates (idempotence)
✅ Complete audit trail (infinite retention)
✅ Reconstruct state at any point (replay)
\`\`\`

**Why Not Others:**
- RabbitMQ: ❌ Messages deleted after consumption (no audit trail)
- SQS: ❌ Max 14 days retention (insufficient for compliance)
- Redis: ❌ Not durable (in-memory), no long-term retention

**Summary Table:**

| Scenario | Choice | Key Reason |
|----------|--------|------------|
| Click stream analytics | Kafka | High throughput, replay, multiple consumers |
| Background task queue | SQS/RabbitMQ | Simple pattern, low cost, zero ops (SQS) |
| Real-time notifications | Redis Pub/Sub | Lowest latency (<1ms), simple pub/sub |
| Event sourcing (financial) | Kafka | Immutable log, exactly-once, infinite retention |

**Key Takeaway:** Choose technology based on specific requirements, not popularity. Simple use cases don't need complex solutions. Critical use cases need battle-tested durability guarantees.`,
    keyPoints: [
      'Kafka: High throughput, replay, event sourcing (click streams, audit logs)',
      'RabbitMQ: Task queues, flexible routing, lower latency than Kafka',
      'AWS SQS: Zero ops, pay-per-use, perfect for AWS task queues',
      'Redis Pub/Sub: Lowest latency (<1ms), real-time notifications, no persistence',
      'Event sourcing requires immutable log with infinite retention (Kafka)',
      'Choose based on requirements, not popularity (simple use cases = simple solutions)',
    ],
  },
];
