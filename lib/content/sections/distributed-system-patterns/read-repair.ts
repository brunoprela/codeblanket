/**
 * Read Repair Section
 */

export const readrepairSection = {
  id: 'read-repair',
  title: 'Read Repair',
  content: `Read Repair is a technique used in distributed databases to detect and fix inconsistencies during read operations. Instead of relying solely on background processes to maintain consistency, read repair opportunistically fixes stale data whenever it's encountered during reads. This ensures that inconsistencies are resolved quickly and that clients receive consistent data.

## What is Read Repair?

**Read Repair** is a mechanism where, during a read operation, the coordinator:
1. Reads from multiple replicas
2. Compares their responses
3. Detects inconsistencies (different versions)
4. Updates stale replicas with the latest version

**Purpose**: Fix inconsistencies on-the-fly during reads, rather than waiting for background repairs.

\`\`\`
Example:
  Read key "user:123" with read quorum R=2

  Coordinator reads from replicas A, B, C:
    Replica A: value="Alice", timestamp=100
    Replica B: value="Alice", timestamp=100
    Replica C: value="Bob", timestamp=50  (stale!)

  Actions:
    1. Return "Alice" to client (latest version, timestamp=100)
    2. Detect C is stale
    3. Send update to C: "user:123" = "Alice", timestamp=100
    4. C updates its copy
\`\`\`

---

## Why Read Repair?

### **1. Fast Inconsistency Resolution**

Without read repair:
\`\`\`
Inconsistency exists.
Must wait for background anti-entropy (hours/days).
Many clients may read stale data. ❌
\`\`\`

With read repair:
\`\`\`
Inconsistency exists.
First read detects and fixes it immediately.
Subsequent reads get consistent data. ✅
\`\`\`

**Benefit**: Inconsistencies resolved quickly, proportional to read traffic.

### **2. Reduced Background Repair Load**

Active read repair fixes many inconsistencies:
\`\`\`
Without read repair:
  - Anti-entropy must scan entire dataset
  - High network and CPU overhead
  
With read repair:
  - Frequently accessed data auto-repaired
  - Anti-entropy only scans infrequently accessed data
  - Lower overall overhead
\`\`\`

### **3. Automatic Healing**

Hot data (frequently read) is continuously repaired:
\`\`\`
Popular key read 1000 times/day.
Any inconsistency fixed on first read.
Stays consistent due to frequent reads.
\`\`\`

**Benefit**: Self-healing system for popular data.

### **4. Client Gets Latest Data**

Even with eventual consistency:
\`\`\`
Client reads key with quorum R=2.
Coordinator checks multiple replicas.
Returns latest version to client. ✅
\`\`\`

**Benefit**: Stronger consistency guarantees than simple "read from any replica".

---

## How Read Repair Works

### **Synchronous Read Repair**

**Read happens in two phases**:

**Phase 1: Read for Quorum**
\`\`\`
1. Coordinator sends read to R replicas (quorum)
2. Replicas respond with data
3. Coordinator returns response to client immediately
\`\`\`

**Phase 2: Background Digest Check**
\`\`\`
4. Coordinator sends read to remaining replicas
5. Replicas respond with digests (hashes)
6. Coordinator compares all responses
7. If inconsistency detected:
   - Determine latest version (highest timestamp)
   - Send update to stale replicas
\`\`\`

**Example (RF=3, R=2)**:
\`\`\`
T=0ms:  Read from A and B (quorum)
T=10ms: A returns value="Alice", ts=100
T=12ms: B returns value="Alice", ts=100
T=13ms: Return "Alice" to client ✓

T=15ms: Read digest from C (background)
T=20ms: C returns digest=hash("Bob"), ts=50
T=21ms: Detect mismatch (C has stale data)
T=22ms: Send repair: Update C with "Alice", ts=100
\`\`\`

### **Blocking Read Repair**

Wait for all replicas before responding:

\`\`\`
1. Coordinator sends read to ALL replicas
2. Wait for ALL responses
3. Compare all responses
4. Determine latest version
5. If inconsistency:
   - Update stale replicas
   - Wait for updates to complete
6. Return result to client
\`\`\`

**Pros**: Guaranteed consistent response
**Cons**: Higher latency (wait for all replicas + repair)

**Use when**: Strong consistency required.

### **Probabilistic Read Repair**

Read from more replicas than quorum, probabilistically:

\`\`\`
read_repair_chance = 0.1  // 10% of reads

On each read:
  if random() < read_repair_chance:
      read_from_all_replicas()  // Trigger digest check
  else:
      read_from_quorum_only()   // Fast path
\`\`\`

**Benefit**: Balance between consistency and performance.

**Cassandra Configuration**:
\`\`\`
dclocal_read_repair_chance = 0.1   // 10% within DC
read_repair_chance = 0.0           // 0% across DCs
\`\`\`

---

## Read Repair Algorithms

### **Timestamp-Based (Last-Write-Wins)**

Use timestamps to determine latest version:

\`\`\`
Replica A: value="Alice", timestamp=100
Replica B: value="Alice", timestamp=100
Replica C: value="Bob", timestamp=50

Latest: timestamp=100 (A and B)
Stale: C (timestamp=50)

Action: Update C to "Alice", timestamp=100
\`\`\`

**Simple but requires synchronized clocks.**

### **Version Vector-Based**

Use version vectors for causality:

\`\`\`
Replica A: value="Alice", vector={A:5, B:3, C:2}
Replica B: value="Alice", vector={A:5, B:3, C:2}
Replica C: value="Bob", vector={A:3, B:2, C:2}

Compare vectors:
  A's vector dominates C's vector (A:5>3, B:3>2, C:2=2)
  A has latest version

Action: Update C with A's data and vector
\`\`\`

**Handles causality without synchronized clocks.**

### **Quorum Union**

When no single version dominates (concurrent writes):

\`\`\`
Replica A: value="Alice", vector={A:5, B:2, C:1}
Replica B: value="Bob", vector={A:3, B:4, C:1}
Replica C: value="Carol", vector={A:3, B:2, C:3}

No single version dominates (concurrent writes).

Actions:
  1. Return all versions to client: ["Alice", "Bob", "Carol"]
  2. Let client resolve conflict
  3. Write resolved version back
  4. Repair all replicas with resolved version
\`\`\`

**Amazon Dynamo approach**: Expose conflicts to application.

---

## Read Repair in Real-World Systems

### **Apache Cassandra**

Cassandra has two types of read repair:

**1. Blocking Read Repair** (always on):
\`\`\`
Coordinator reads from quorum.
Compares responses.
If mismatch, repairs stale replica.
Waits for repair before returning to client.
\`\`\`

**Benefit**: Guarantees client gets consistent data.

**2. Background Read Repair** (probabilistic):
\`\`\`yaml
# cassandra.yaml
dclocal_read_repair_chance: 0.1   # 10% within DC
read_repair_chance: 0.0           # Cross-DC (disabled by default)
\`\`\`

10% of reads check ALL replicas for inconsistencies.

**Monitoring**:
\`\`\`
nodetool netstats
  Read Repair Statistics: Shows repairs performed
\`\`\`

### **Amazon DynamoDB**

**Consistent Reads** use read repair:

\`\`\`
GetItem with ConsistentRead=true:
  - Read from all replicas in quorum
  - Compare responses
  - Return latest
  - Repair stale replicas
\`\`\`

**Eventually Consistent Reads**: May return stale data (faster).

### **Apache Riak**

**Read Repair Strategy**:
\`\`\`
Read from R replicas (quorum).
If inconsistency detected:
  - Use vector clocks to determine causality
  - If concurrent: Return siblings to client
  - Client resolves and writes back
\`\`\`

**Active Anti-Entropy**: Additional background process for missed inconsistencies.

### **Scylla**

Similar to Cassandra, but optimized:

\`\`\`
Read from quorum (fast path).
Asynchronously check additional replicas.
Repair in background without blocking.
\`\`\`

**Performance**: Lower latency than blocking repair.

---

## Trade-offs

### **Synchronous vs Asynchronous**

**Synchronous**:
\`\`\`
Wait for repair before returning to client.
Pros: Guaranteed consistent read
Cons: Higher latency
\`\`\`

**Asynchronous**:
\`\`\`
Return to client immediately, repair in background.
Pros: Lower latency
Cons: Client may still see stale data on THIS read
\`\`\`

**Typical**: Synchronous for critical data, asynchronous for hot paths.

### **Read Amplification**

Reading from all replicas increases load:

\`\`\`
Without read repair:
  Read from R=2 replicas
  2 read requests

With read repair:
  Read from all RF=3 replicas
  3 read requests (50% more!)
\`\`\`

**Trade-off**: Consistency vs read load.

**Solution**: Probabilistic repair (10% of reads), not every read.

### **Latency Impact**

Blocking read repair increases latency:

\`\`\`
Normal read: 10ms (read from 2 replicas)
Read with repair: 50ms (read from 3, detect mismatch, repair, wait)
\`\`\`

**Trade-off**: Latency vs consistency.

**Solution**: Async repair for latency-sensitive reads.

---

## Challenges and Solutions

### **1. False Positives (Clock Skew)**

**Problem**: Different timestamps due to clock skew, not actual inconsistency.

\`\`\`
Replica A: timestamp=100 (clock accurate)
Replica B: timestamp=102 (clock 2s ahead)

Coordinator thinks B is newer, repairs A unnecessarily. ❌
\`\`\`

**Solutions**:
- Use logical clocks (version vectors) instead of timestamps
- Synchronize clocks with NTP (reduce skew)
- Use write coordinator timestamp (single source)

### **2. Repair Storms**

**Problem**: Many reads trigger many repairs, overwhelming replicas.

\`\`\`
100,000 reads/second.
10% trigger full replica check.
10,000 digest reads + potential repairs.
Replicas overwhelmed. ❌
\`\`\`

**Solutions**:
- Lower read repair probability (1% instead of 10%)
- Rate limit repairs per replica
- Async repairs (don't block reads)
- Monitor replica load, disable repair if overloaded

### **3. Infinite Repair Loop**

**Problem**: Repair write triggers another read repair, which triggers repair, ...

\`\`\`
Read detects stale data on C.
Repair writes to C.
Repair write triggers read on C (with read repair).
Detects own write as stale (?!).
Infinite loop. ❌
\`\`\`

**Solutions**:
- Mark repair writes (don't trigger read repair on repair writes)
- Use timestamps to detect repair is newer
- Idempotent repairs (same repair multiple times is ok)

### **4. Concurrent Writes During Repair**

**Problem**: Repair races with new write.

\`\`\`
T=0:  Read detects C has stale value="Alice"
T=1:  New write: value="Bob" to all replicas
T=2:  Repair sends value="Alice" to C
T=3:  C now has stale "Alice" (should be "Bob") ❌
\`\`\`

**Solutions**:
- Use timestamps/versions: C rejects repair if local data is newer
- Last-write-wins: Highest timestamp wins
- Version vectors: Detect causality

### **5. High Memory Usage**

**Problem**: Coordinator must store all responses to compare.

\`\`\`
Read 1MB value from 3 replicas.
Coordinator holds 3MB in memory. ❌
\`\`\`

**Solutions**:
- Use digests (hashes) for comparison: 32 bytes vs 1MB
- Only fetch full value from stale replica if digest mismatches
- Stream large values instead of buffering

---

## Implementation Considerations

### **Digest Calculation**

**Fast digest**:
\`\`\`
digest = hash(value + timestamp)

Example:
  value = "Alice"
  timestamp = 100
  digest = hash("Alice100") = 0xABC123...
\`\`\`

**Fast comparison**, but requires re-hashing on coordinator.

**Include metadata**:
\`\`\`
digest = hash(value + timestamp + tombstone + ttl)
\`\`\`

Ensures all aspects are checked.

### **Partial Repair**

For large values, repair only changed portions:

\`\`\`
Replica A: {field1: "x", field2: "y", field3: "z"}
Replica B: {field1: "x", field2: "y", field3: "old"}

Repair only field3 (delta), not entire record.
\`\`\`

**Benefit**: Reduce network transfer.

**Complexity**: Requires column-level versioning.

### **Batching Repairs**

Group multiple repairs into single request:

\`\`\`
Read 10 keys, 5 need repair on Replica C.

Instead of 5 separate repair requests:
  Batch repair: Send all 5 updates in one request.
\`\`\`

**Benefit**: Reduce network overhead.

### **Monitoring**

**Key Metrics**:

**1. Read Repair Count**: Number of repairs performed
- High: Many inconsistencies (investigate cause)

**2. Repair Latency**: Time to perform repair
- High: Network issues or slow replicas

**3. Repair Success Rate**: Repairs completed successfully
- Low: Replicas failing or unreachable

**4. Stale Replica Rate**: % of reads that find stale data
- High: Writes not propagating correctly

**Alerts**:
- Read repair count > 1000/s (too many inconsistencies)
- Repair success rate < 95%
- Stale replica rate > 10%

---

## Read Repair vs Anti-Entropy

### **Read Repair**

**Triggered by**: Read operations
**Scope**: Only keys that are read
**Speed**: Immediate for read keys
**Overhead**: Proportional to read traffic

**Best for**: Frequently accessed data

### **Anti-Entropy**

**Triggered by**: Background process (scheduled)
**Scope**: Entire dataset
**Speed**: Slow (hours/days for full scan)
**Overhead**: Constant, independent of reads

**Best for**: Infrequently accessed data, full consistency guarantee

### **Complementary**

Use both:
\`\`\`
- Read repair: Fix hot data automatically
- Anti-entropy: Catch cold data, ensure full consistency
\`\`\`

---

## Interview Tips

### **Key Concepts to Explain**

1. **What is read repair**: Fix inconsistencies during reads by comparing replicas
2. **Why needed**: Fast inconsistency resolution, self-healing for hot data
3. **How it works**: Read from multiple replicas, compare, update stale ones
4. **Synchronous vs async**: Trade-off latency vs consistency
5. **Probabilistic**: Not every read triggers repair (e.g., 10%)
6. **Real-world**: Cassandra (blocking + probabilistic), DynamoDB (consistent reads)

### **Common Interview Questions**

**Q: How does read repair improve consistency?**
A: "Read repair opportunistically fixes inconsistencies during reads. Coordinator reads from multiple replicas (often more than quorum), compares responses, and detects stale data. It then updates stale replicas with the latest version. This means frequently accessed data is continuously repaired, improving consistency over time. Example: Hot key read 1000x/day—any inconsistency fixed on first read, stays consistent due to frequent reads."

**Q: What's the difference between read repair and anti-entropy?**
A: "Read repair is triggered by reads, only fixes keys that are accessed, immediate for those keys, overhead proportional to read traffic. Anti-entropy is a background process that scans the entire dataset, slow but comprehensive. They're complementary: read repair fixes hot data automatically, anti-entropy catches cold data and ensures full consistency. Production systems use both—read repair for fast fixes, anti-entropy for completeness."

**Q: How do you prevent read repair from impacting read latency?**
A: "Use asynchronous read repair: (1) Read from quorum (e.g., 2 replicas), return to client immediately. (2) In background, read digests from remaining replicas, detect inconsistencies, repair. Client doesn't wait for repair. Also use probabilistic repair: only 10% of reads check all replicas (configurable). For remaining 90%, just read from quorum (fast path). This balances consistency (most inconsistencies caught) with latency (most reads are fast)."

**Q: What happens if two replicas have conflicting concurrent writes?**
A: "Depends on conflict resolution strategy: (1) Last-write-wins: Use timestamps, keep write with highest timestamp (requires clock sync). (2) Version vectors: Detect concurrent writes (neither causally dominates). Return all versions to client, let application resolve (e.g., merge shopping carts). (3) Application-defined: Custom merge logic (CRDTs automatically merge). After resolution, repair all replicas with resolved version."

---

## Summary

Read Repair is an essential technique for maintaining consistency in distributed databases:

1. **Core Idea**: Detect and fix inconsistencies during read operations
2. **Benefits**: Fast resolution for hot data, self-healing, reduced anti-entropy load
3. **Mechanism**: Read from multiple replicas, compare, update stale ones
4. **Types**: Synchronous (blocking, slower, guaranteed consistent), asynchronous (fast, eventual consistency)
5. **Probabilistic**: Not every read triggers repair (e.g., 10%), balances cost and benefit
6. **Real-World**: Cassandra (blocking + probabilistic), DynamoDB (consistent reads), Riak (vector clocks)
7. **Trade-offs**: Latency impact, read amplification, repair overhead
8. **Complementary**: Use with anti-entropy (read repair for hot data, anti-entropy for cold data)

**Interview Focus**: Understand how read repair works (compare replicas during reads), why it's useful (fast fixes for hot data), types (sync vs async), and trade-offs (latency, read amplification).
`,
};
