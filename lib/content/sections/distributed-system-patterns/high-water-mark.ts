/**
 * High-Water Mark Section
 */

export const highwatermarkSection = {
  id: 'high-water-mark',
  title: 'High-Water Mark',
  content: `High-Water Mark is a pattern used in distributed systems to track which data has been successfully replicated and is safe to read. It\'s crucial for maintaining consistency in leader-follower architectures and is prominently used in systems like Apache Kafka.

## What is High-Water Mark?

The **High-Water Mark (HWM)** is an **offset** or **position** in a log that indicates the **last committed entry** that has been replicated to a sufficient number of followers.

**Key Concept**: Data before the high-water mark is **safe** to read (durable, replicated). Data after it is **uncommitted** and may be lost if the leader fails.

\`\`\`
Leader's Log:
  Offset 0:  Entry A  ✅ Replicated to all followers
  Offset 1:  Entry B  ✅ Replicated to all followers
  Offset 2:  Entry C  ✅ Replicated to majority
  Offset 3:  Entry D  ⏳ Replicated to some followers
  Offset 4:  Entry E  ⏳ Only on leader
  
High-Water Mark: Offset 2

Consumers can read up to offset 2.
Offsets 3-4 are not yet committed.
\`\`\`

---

## Why High-Water Mark Matters

### **1. Consistency Guarantee**

Without HWM:
\`\`\`
1. Leader writes entry at offset 5
2. Immediately returns to producer
3. Consumer reads offset 5
4. Leader crashes before replicating to followers
5. New leader elected (doesn't have offset 5)
6. Consumer saw data that is now lost ❌
\`\`\`

With HWM:
\`\`\`
1. Leader writes entry at offset 5
2. Waits for replication to majority
3. Advances HWM to 5
4. Returns to producer
5. Consumer can now read offset 5
6. Even if leader crashes, data is safe ✅
\`\`\`

**Result**: Consumers never see data that could be lost.

### **2. Preventing Phantom Reads**

**Phantom Read**: Reading data that later disappears.

\`\`\`
Scenario without HWM:
  T0: Consumer reads offset 10 from Leader A
  T1: Leader A crashes (offset 10 not replicated)
  T2: Leader B elected (only has offsets 0-9)
  T3: Consumer tries to read offset 10 → Not found
  
Consumer saw data that no longer exists!
\`\`\`

HWM ensures consumers only read data that won't disappear.

### **3. Coordinating Replication**

HWM provides a **common reference point** for all nodes:
- Followers know how much data is committed
- Consumers know how much data is safe to read
- Producers know when writes are durable

---

## How High-Water Mark Works

### **Basic Flow**

**Writing**:
\`\`\`
1. Producer sends message to leader
2. Leader appends to local log (offset N)
3. Leader replicates to followers
4. Followers acknowledge replication
5. When majority acknowledges:
   - Leader advances HWM to N
   - Returns success to producer
\`\`\`

**Reading**:
\`\`\`
1. Consumer requests data from offset N
2. If N <= HWM:
   → Return data (committed, safe)
3. If N > HWM:
   → Wait or return empty (uncommitted)
\`\`\`

### **Replication Protocol**

**Follower Actions**:
\`\`\`
1. Fetch data from leader
2. Append to local log
3. Update local replica offset
4. Send acknowledgment to leader
5. Leader sends updated HWM
6. Follower updates its view of HWM
\`\`\`

**Leader Actions**:
\`\`\`
1. Write to local log (LEO = Log End Offset increases)
2. Wait for follower acknowledgments
3. When majority acknowledges offset N:
   - HWM = min(LEO of all in-sync replicas)
4. Send HWM to followers in next replication request
\`\`\`

### **Log End Offset (LEO) vs High-Water Mark**

**Log End Offset (LEO)**: Last offset in the log (highest offset + 1)
**High-Water Mark (HWM)**: Last committed offset

\`\`\`
Leader:
  Log: [0, 1, 2, 3, 4, 5]
  LEO: 6 (next offset to write)
  HWM: 3 (last committed)

Follower 1:
  Log: [0, 1, 2, 3, 4]
  LEO: 5

Follower 2:
  Log: [0, 1, 2, 3]
  LEO: 4

HWM Calculation:
  HWM = min(LEO of in-sync replicas)
  HWM = min(6, 5, 4) - 1 = 3
  
(Leader can advance HWM to 4 once Follower 2 catches up)
\`\`\`

---

## High-Water Mark in Apache Kafka

Kafka extensively uses HWM for managing replication and consumer visibility.

### **Kafka Architecture**

\`\`\`
Topic: orders, Partition: 0

Leader (Broker 1):
  Log: [msg0, msg1, msg2, msg3, msg4]
  LEO: 5
  HWM: 3

Follower (Broker 2):
  Log: [msg0, msg1, msg2, msg3]
  LEO: 4
  HWM: 3 (from leader)

Follower (Broker 3):
  Log: [msg0, msg1, msg2, msg3]
  LEO: 4
  HWM: 3 (from leader)

Consumer:
  Can read offsets 0-3 (up to HWM)
  Cannot read offsets 4+ (not committed yet)
\`\`\`

### **Kafka Replication Protocol**

**Producer Write**:
\`\`\`
1. Producer sends message to leader
2. Leader appends to log (offset 5, LEO = 6)
3. Leader replicates to in-sync replicas (ISR)
4. Followers fetch data:
   - Follower sends fetch request with its LEO
   - Leader sends data + current HWM
5. Followers append to log and acknowledge
6. Leader updates HWM when all ISR acknowledge
7. Leader returns to producer (acks=all)
\`\`\`

**Consumer Read**:
\`\`\`
1. Consumer sends fetch request with offset
2. Leader checks: offset <= HWM?
   - Yes: Return data
   - No: Return empty or wait
3. Consumer processes data
4. Consumer commits offset
\`\`\`

### **In-Sync Replicas (ISR)**

**ISR**: Set of replicas that are caught up with the leader.

**Criteria for ISR**:
- Replica is alive (sending heartbeats)
- Replica has fetched data recently (within \`replica.lag.time.max.ms\`, default 10s)
- Replica\'s LEO is close to leader's LEO

**HWM Calculation**:
\`\`\`
HWM = min(LEO of all replicas in ISR)
\`\`\`

**Why ISR Matters**:
- Slow replicas don't block HWM advancement
- Only replicas in ISR participate in HWM calculation
- If leader fails, only ISR members can become new leader

**Example**:
\`\`\`
Replication Factor: 3 (Broker 1, 2, 3)
ISR: {Broker 1, Broker 2} (Broker 3 is lagging)

Leader (Broker 1): LEO = 100
Follower (Broker 2): LEO = 99
Follower (Broker 3): LEO = 80 (not in ISR)

HWM = min(100, 99) - 1 = 98
(Broker 3 doesn't affect HWM)
\`\`\`

---

## Leader Election and High-Water Mark

When a leader fails, the HWM ensures consistency during failover.

### **Scenario: Leader Failure**

**Before Failure**:
\`\`\`
Leader (Broker 1):
  Log: [0, 1, 2, 3, 4, 5]
  LEO: 6, HWM: 4

Follower (Broker 2, in ISR):
  Log: [0, 1, 2, 3, 4]
  LEO: 5, HWM: 4

Follower (Broker 3, in ISR):
  Log: [0, 1, 2, 3, 4]
  LEO: 5, HWM: 4
\`\`\`

**Leader Crashes**:
\`\`\`
1. Broker 1 fails
2. Broker 2 elected as new leader
3. Broker 2 truncates log to HWM (offset 4)
   - Discards offset 5 (not committed)
4. Broker 2 becomes leader with LEO = 5
5. When Broker 1 recovers:
   - Broker 1 also truncates to HWM (offset 4)
   - Discards offsets 5-6 (not committed)
   - Becomes follower
\`\`\`

**Why Truncate?**
- Offset 5 on old leader was not committed (HWM = 4)
- New leader doesn't have offset 5
- Discarding offset 5 maintains consistency
- Consumers never saw offset 5 (it was beyond HWM)

**Result**: All nodes agree on committed data (offsets 0-4).

---

## Trade-offs

### **Consistency vs Latency**

**Strict HWM** (wait for all replicas):
\`\`\`
Pros:
  + Strong consistency
  + Higher durability
Cons:
  - Higher latency (wait for slowest replica)
  - Reduced availability (slow replica blocks writes)
\`\`\`

**Relaxed HWM** (wait for majority):
\`\`\`
Pros:
  + Lower latency
  + Better availability
Cons:
  - Slightly weaker durability (minority can lose data)
\`\`\`

### **Kafka Producer Acknowledgment Modes**

**acks=0** (no wait):
\`\`\`
Producer doesn't wait for any acknowledgment.
Fastest, but no guarantees.
\`\`\`

**acks=1** (leader only):
\`\`\`
Producer waits for leader to write.
Data visible once on leader (not replicated).
Fast, but risk of loss if leader fails.
\`\`\`

**acks=all** (all in-sync replicas):
\`\`\`
Producer waits for all ISR to acknowledge.
Data visible once HWM advances.
Slower, but durable.
\`\`\`

**Trade-off**: Latency vs durability
- Financial transactions: acks=all
- Log aggregation: acks=1 or acks=0

---

## High-Water Mark in Other Systems

### **Raft Consensus**

Raft uses a concept similar to HWM called **commit index**.

\`\`\`
Leader:
  Log: [1, 2, 3, 4, 5]
  Commit Index: 3

Follower 1:
  Log: [1, 2, 3, 4]
  Commit Index: 3 (from leader)

Follower 2:
  Log: [1, 2, 3]
  Commit Index: 3
\`\`\`

**Commit Index**: Highest log entry known to be replicated to majority.

**Once majority acknowledges entry N**:
- Leader advances commit index to N
- Entry N is durable and can be applied to state machine
- Leader sends commit index to followers

### **PostgreSQL Replication**

PostgreSQL uses **WAL (Write-Ahead Log)** with replication.

\`\`\`
Primary:
  WAL Position: 0/300000
  Flush Position: 0/300000
  
Standby:
  WAL Position: 0/2F0000 (lagging)
  Replay Position: 0/2E0000
\`\`\`

**Synchronous Replication**:
- Primary waits for standby to acknowledge WAL write
- Similar to HWM: Data visible once replicated

**Asynchronous Replication**:
- Primary doesn't wait
- Standby may lag (risk of data loss on primary failure)

### **MySQL Group Replication**

MySQL Group Replication uses **GTID (Global Transaction ID)** with a commit protocol.

\`\`\`
Certified Transactions: GTIDs that have been approved by majority
Applied Transactions: GTIDs that have been executed
\`\`\`

**Similar to HWM**: Only certified transactions are visible to reads.

---

## Monitoring High-Water Mark

### **Key Metrics**

**1. HWM Lag**:
\`\`\`
HWM Lag = LEO - HWM
\`\`\`

**Interpretation**:
- Low lag (< 10): Healthy replication
- High lag (> 1000): Replication issues, slow followers
- Growing lag: Followers can't keep up

**2. Replica Lag**:
\`\`\`
Replica Lag = Leader LEO - Follower LEO
\`\`\`

**Interpretation**:
- 0: Fully caught up
- < 100: Acceptable lag
- > 1000: Replica falling behind

**3. ISR Shrink/Expand Events**:

**ISR Shrink**: Replica removed from ISR (lagging or failed)
- Indicates replica issues
- May reduce redundancy

**ISR Expand**: Replica added back to ISR (caught up)
- Indicates recovery

**Frequent shrink/expand**: Network or broker issues

**4. Under-Replicated Partitions**:
\`\`\`
Under-Replicated = Replication Factor - ISR Size
\`\`\`

**Example**:
\`\`\`
Replication Factor: 3
ISR: {Broker 1, Broker 2} (size 2)
Under-Replicated: 3 - 2 = 1
\`\`\`

**Alert**: Under-replicated partitions > 0 (reduced durability)

---

## Common Issues

### **1. Slow Followers**

**Problem**: Followers can't keep up with leader write rate.

**Causes**:
- Network latency
- Disk I/O bottleneck
- CPU saturation
- GC pauses

**Effects**:
- Followers removed from ISR
- HWM advancement slows
- Higher producer latency (waiting for fewer replicas)
- Reduced durability

**Solutions**:
- Add more followers
- Upgrade hardware (faster disks, more CPU)
- Reduce write load
- Optimize application (smaller messages, batching)

### **2. Stuck High-Water Mark**

**Problem**: HWM not advancing despite writes.

**Causes**:
- All followers out of ISR
- Leader not sending HWM updates
- Bug in replication protocol

**Effects**:
- Consumers can't read new data
- Data piling up as uncommitted

**Diagnosis**:
\`\`\`
Check:
  - ISR size (should be >= 1)
  - Follower fetch requests (should be happening)
  - Leader logs (errors or warnings)
\`\`\`

**Solutions**:
- Restart lagging followers
- Manually adjust ISR (emergency only)
- Fix underlying issues (network, disk)

### **3. HWM Rollback After Leader Failure**

**Problem**: New leader's HWM is lower than old leader's LEO.

**Effects**:
- Some data discarded (offsets beyond HWM)
- Producers may see errors (duplicate detection)

**Expected Behavior**: This is correct (uncommitted data is lost).

**Mitigation**:
- Use \`acks=all\` (ensures data replicated before acknowledging)
- Enable idempotent producer (handles duplicates)
- Monitor under-replicated partitions

---

## Implementation Considerations

### **Atomic HWM Update**

**Challenge**: HWM update must be atomic with replication acknowledgment.

\`\`\`
Bad:
  1. Update HWM in memory
  2. [Crash here]
  3. Send HWM to followers
  
Result: Leader and followers have inconsistent HWM

Good:
  1. Persist HWM update
  2. Send HWM to followers
  3. [Crash safe]
\`\`\`

### **HWM Propagation**

**Challenge**: Followers need to know current HWM.

**Solutions**:
- Leader includes HWM in replication responses
- Followers fetch HWM periodically
- Leader broadcasts HWM updates

**Kafka**: HWM sent with every fetch response from leader.

### **Handling Network Partitions**

**Scenario**: Network partition splits cluster.

\`\`\`
Partition 1: Leader + Follower 1 (majority)
Partition 2: Follower 2 (minority)
\`\`\`

**Behavior**:
- Partition 1: HWM advances (has majority)
- Partition 2: Follower can't be elected leader (not in ISR)

**After Partition Heals**:
- Follower 2 catches up
- Rejoins ISR

**Safety**: Minority partition can't elect leader, preventing split-brain.

---

## Interview Tips

### **Key Concepts to Explain**1. **What is HWM**: Last committed offset that is safe to read
2. **Why it matters**: Consistency, prevents phantom reads, coordinates replication
3. **How it works**: Calculated as min(LEO) of in-sync replicas
4. **Kafka example**: Explain producer acks, consumer reads, ISR
5. **Failover**: Truncate to HWM on leader election

### **Common Interview Questions**

**Q: What happens if a consumer reads beyond the high-water mark?**
A: "Consumers are blocked from reading beyond HWM. The broker will either return empty data or make the consumer wait. This ensures consumers only see committed data that won't disappear if the leader fails. Once HWM advances (data is replicated), the consumer can read the data."

**Q: Why does Kafka truncate the log to the HWM after a leader election?**
A: "When a leader fails, uncommitted data (beyond HWM) may exist only on that leader. The new leader doesn't have this data. To maintain consistency, when the old leader recovers, it truncates its log to the HWM—the last offset that was replicated to majority. This ensures all replicas agree on the committed data. Consumers never saw the truncated data anyway since it was beyond HWM."

**Q: What\'s the trade-off between acks=1 and acks=all in Kafka?**
A: "acks=1: Producer waits only for leader write. Fast (low latency) but data may be lost if leader fails before replication. acks=all: Producer waits for all in-sync replicas. Slower (higher latency) but durable—data is safe even if leader fails. Choice depends on requirements: performance-critical logs use acks=1, financial transactions use acks=all."

**Q: How does HWM prevent split-brain in Kafka?**
A: "HWM is calculated based on in-sync replicas (ISR). Only replicas in ISR can become leader. If network partitions the cluster, only the partition with majority can advance HWM and elect a new leader. The minority partition can't make progress. This prevents two leaders from accepting writes and diverging."

---

## Summary

High-Water Mark is essential for maintaining consistency in replicated logs:

1. **Core Idea**: Track last committed offset safe to read
2. **Guarantees**: Consumers only see durable, replicated data
3. **Calculation**: min(LEO) of in-sync replicas
4. **Failover**: Truncate to HWM ensures consistency
5. **Real-World**: Kafka (partitions), Raft (commit index), PostgreSQL (replication)
6. **Trade-offs**: Consistency vs latency (acks=all vs acks=1)

**Interview Focus**: Understand what HWM prevents (phantom reads, inconsistency), how it's calculated (min LEO of ISR), and what happens during failover (truncate to HWM).
`,
};
