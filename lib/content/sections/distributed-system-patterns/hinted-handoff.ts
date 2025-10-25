/**
 * Hinted Handoff Section
 */

export const hintedhandoffSection = {
  id: 'hinted-handoff',
  title: 'Hinted Handoff',
  content: `Hinted Handoff is a technique used in distributed databases to improve availability during temporary node failures. Instead of failing writes when a replica is down, the system stores "hints" about the writes and replays them when the node recovers. This ensures data eventually reaches all replicas without sacrificing write availability.

## What is Hinted Handoff?

**Hinted Handoff** is a mechanism where, when a replica node is temporarily unavailable, another node temporarily stores the write operations (as "hints") and later forwards them to the unavailable node once it recovers.

**Analogy**: Like leaving a message with a neighbor when someone is not home. The neighbor holds the message and delivers it when the person returns.

\`\`\`
Normal Write (all replicas up):
  Client → Coordinator → Replicas {A, B, C}
  All replicas receive write immediately ✓

Write with Node C Down:
  Client → Coordinator → Replicas {A, B, C (down)}
  A and B receive write
  Coordinator stores "hint" for C
  Later: C comes back, coordinator sends hint to C ✓
\`\`\`

---

## Why Hinted Handoff?

### **1. Improved Write Availability**

Without hinted handoff:
\`\`\`
Replication factor: 3
Required writes (W): 3
Node C is down

Write request arrives:
  - Can only write to A and B (2 < 3)
  - Write fails ❌
\`\`\`

With hinted handoff:
\`\`\`
Replication factor: 3
Required writes (W): 3
Node C is down

Write request arrives:
  - Write to A and B
  - Store hint for C on node D
  - Write succeeds (3 writes: A, B, hint) ✓
  - Later: C recovers, D sends hint to C
\`\`\`

**Benefit**: Writes succeed even during failures.

### **2. Faster Recovery**

Without hinted handoff:
\`\`\`
Node C down for 1 hour.
During that hour: 10,000 writes.

C comes back:
  - Must read entire dataset to find missing data
  - Or: Must run anti-entropy repair (slow)
\`\`\`

With hinted handoff:
\`\`\`
Node C down for 1 hour.
During that hour: 10,000 writes stored as hints.

C comes back:
  - Receive 10,000 hints (exactly what's missing)
  - Apply hints
  - Fully caught up in seconds ✓
\`\`\`

**Benefit**: Faster recovery, precise catch-up.

### **3. Reduced Load on Cluster**

Without hinted handoff:
\`\`\`
Node C recovers.
Must run full anti-entropy:
  - Read all data from other replicas
  - Compare with local data
  - Transfer missing/updated data
  - Network and CPU intensive ❌
\`\`\`

With hinted handoff:
\`\`\`
Node C recovers.
Just replay hints:
  - Sequential writes of missing operations
  - Minimal network and CPU
  - Other nodes unaffected ✓
\`\`\`

---

## How Hinted Handoff Works

### **Normal Write Path**

\`\`\`
Client writes key "user:123" with replication factor 3:

1. Hash key to determine replicas: {Node A, Node B, Node C}
2. Coordinator sends write to A, B, C
3. Wait for W=2 acknowledgments
4. Return success to client
\`\`\`

### **Write Path with Failed Node**

\`\`\`
Client writes key "user:123" with replication factor 3:
Node C is down.

1. Hash key to determine replicas: {A, B, C (down)}
2. Coordinator attempts to write to A, B, C
3. A and B succeed
4. C times out or unreachable
5. Coordinator selects another node (D) to store hint
6. D stores hint: "Write 'user:123' to Node C"
7. Wait for acknowledgments (A, B, hint on D)
8. Return success to client

Later:
9. Node C comes back online
10. Node D detects C is up
11. D sends hint to C: "Write 'user:123'"
12. C applies write
13. D deletes hint
\`\`\`

### **Hint Structure**

\`\`\`json
{
  "hint_id": "uuid-1234",
  "target_node": "Node C",
  "key": "user:123",
  "value": "Alice",
  "timestamp": 1704067200000,
  "ttl": 86400000,  // 24 hours
  "created_at": 1704067200000
}
\`\`\`

**Fields**:
- **target_node**: Which node should receive this hint
- **key/value**: The actual data to write
- **timestamp**: When the write occurred
- **ttl**: How long to keep the hint before discarding
- **created_at**: When the hint was created

---

## Hint Storage and Management

### **Where to Store Hints?**

**Option 1: In-Memory**
\`\`\`
hints_map = {
  "Node C": [hint1, hint2, hint3, ...],
  "Node D": [hint4, hint5, ...]
}
\`\`\`

**Pros**: Fast access
**Cons**: Lost on crash, limited by memory

**Option 2: Dedicated Hints Directory**
\`\`\`
/var/db/hints/
  node-C-hints/
    hint-0001.data
    hint-0002.data
    ...
  node-D-hints/
    hint-0003.data
\`\`\`

**Pros**: Durable, survives crashes
**Cons**: Disk I/O overhead

**Best Practice**: In-memory with periodic flush to disk (WAL-like).

### **Hint Size Limits**

Problem: If node is down for days, hints accumulate.

**Solutions**:

**1. Time-to-Live (TTL)**
\`\`\`
hints older than 3 hours: discard
\`\`\`

**Reason**: After TTL, rely on anti-entropy instead.

**Cassandra Default**: 3 hours

**2. Size Limit per Node**
\`\`\`
if hints_for_node_C > 10 GB:
    stop storing new hints
    rely on anti-entropy
\`\`\`

**3. Total Hints Limit**
\`\`\`
if total_hints > 100 GB:
    delete oldest hints
    or stop accepting new hints
\`\`\`

### **Hint Replay**

When target node recovers:

\`\`\`
1. Detect node is back (heartbeat, gossip)
2. For each hint for that node:
   a. Send hint to node
   b. Wait for acknowledgment
   c. Delete hint
3. Continue until all hints replayed
\`\`\`

**Replay Strategy**:

**Sequential**:
\`\`\`
for hint in hints_for_node_C:
    send_hint_to_C(hint)
    wait_for_ack()
    delete_hint (hint)
\`\`\`

**Pros**: Simple, maintains order
**Cons**: Slow

**Batched**:
\`\`\`
batch = hints_for_node_C[:100]
send_batch_to_C(batch)
wait_for_ack()
delete_hints (batch)
\`\`\`

**Pros**: Faster, less overhead
**Cons**: More complex error handling

**Parallel** (multiple senders):
\`\`\`
Multiple nodes with hints for C send simultaneously.
\`\`\`

**Pros**: Very fast
**Cons**: May overwhelm recovering node

**Best**: Start slow (sequential), ramp up (batched) as node catches up.

---

## Hinted Handoff in Real-World Systems

### **Apache Cassandra**

Cassandra extensively uses hinted handoff.

**Configuration**:
\`\`\`yaml
# cassandra.yaml
hinted_handoff_enabled: true
max_hint_window_in_ms: 10800000  # 3 hours
hinted_handoff_throttle_in_kb: 1024  # 1 MB/s per node
\`\`\`

**Behavior**:
- Write fails to replica → Store hint
- Hints stored in dedicated directory
- Background thread monitors hints
- When node recovers, replay hints
- Hints older than 3 hours discarded

**Monitoring**:
\`\`\`
nodetool statushandoff
\`\`\`

Shows pending hints per node.

**Manual Control**:
\`\`\`
nodetool disablehandoff
nodetool enablehandoff
\`\`\`

### **Amazon Dynamo**

Original Dynamo paper describes hinted handoff as key technique.

**Sloppy Quorum**:
\`\`\`
Normal quorum: W=2 out of {A, B, C}
Node C down: W=2 out of {A, B} + hint on D

Sloppy quorum: Accept writes as long as W nodes (including hints) ack.
\`\`\`

**Benefit**: Availability during failures.

### **Riak**

Similar to Cassandra, uses hinted handoff for availability.

**Configuration**:
\`\`\`
handoff_concurrency = 2  # Concurrent hint replays
\`\`\`

**Fallback Replicas**: Nodes that temporarily store hints.

---

## Consistency Considerations

### **Hinted Handoff ≠ True Replica**

Important: **Hint is not a replica**, it's a temporary stand-in.

\`\`\`
Replication factor: 3
W (write quorum): 2
R (read quorum): 2

Write with C down:
  - Write to A, B
  - Hint on D
  - W=2 satisfied (A, B) ✓

Read:
  - Read from A, B, D
  - D doesn't have actual data (only hint) ❌
  - R=2 satisfied only if A and B both respond
\`\`\`

**Key Point**: Hints don't count toward read quorum. They're for write availability, not read consistency.

### **Read Repair Still Needed**

Even with hinted handoff, reads may return stale data:

\`\`\`
T=0:  Write to A, B, hint on D (C down)
T=1:  C comes back
T=2:  D starts replaying hints (slow, many hints)
T=3:  Read from B, C
      - C doesn't have latest data yet ❌
\`\`\`

**Solution**: Read repair detects and fixes inconsistencies.

### **Eventual Consistency**

Hinted handoff provides **eventual consistency**:
- Writes succeed immediately (availability)
- Data eventually reaches all replicas (consistency)
- Temporary inconsistency is acceptable

**Not suitable for**: Strong consistency requirements (financial transactions).

---

## Challenges and Solutions

### **1. Hint Accumulation**

**Problem**: Node down for days, millions of hints.

\`\`\`
Node C down for 3 days.
100,000 writes/hour.
Total hints: 7.2 million ❌
\`\`\`

**Solutions**:
- **TTL**: Discard hints after 3 hours (Cassandra default)
- **Size limit**: Cap hints per node
- **Throttling**: Limit hint storage rate
- **Fallback to anti-entropy**: After threshold, rely on full repair

### **2. Hint Replay Overload**

**Problem**: Node recovers, receives millions of hints, overwhelmed.

\`\`\`
Node C comes back.
D starts replaying 7 million hints.
C can't keep up, falls behind again. ❌
\`\`\`

**Solutions**:
- **Throttling**: Limit hint replay rate (1 MB/s per node)
- **Prioritization**: Replay recent hints first
- **Backpressure**: C tells D to slow down
- **Gradual ramp-up**: Start slow, increase rate if C keeps up

### **3. Stale Hints**

**Problem**: Hint contains old data, newer write already on node.

\`\`\`
T=0:  Write X=1 to A, B, hint on D (C down)
T=1:  C comes back
T=2:  Write X=2 directly to A, B, C
T=3:  D replays hint X=1 to C
      - C now has stale data X=1 (should be X=2) ❌
\`\`\`

**Solutions**:
- **Timestamps**: Include timestamp in hint, C rejects if older than current
- **Versioning**: Use version vectors to detect conflicts
- **TTL**: Short TTL reduces window for staleness

### **4. Hint Node Failure**

**Problem**: Node storing hints crashes, hints lost.

\`\`\`
Node D storing hints for C crashes.
Hints on D lost.
C never receives those writes. ❌
\`\`\`

**Solutions**:
- **Replicate hints**: Store hints on multiple nodes
- **Persist hints**: Write hints to disk (WAL)
- **Anti-entropy**: Eventually catches missing data
- **Accept risk**: Hints are best-effort, not guaranteed

### **5. Cascading Failures**

**Problem**: Multiple nodes down, hints accumulate everywhere.

\`\`\`
Nodes C, D, E down.
All writes create hints.
Hint storage nodes overwhelmed. ❌
\`\`\`

**Solutions**:
- **Circuit breaker**: Stop storing hints if too many nodes down
- **Prioritize critical data**: Only hint for high-priority writes
- **Aggressive TTL**: Reduce TTL during cluster instability

---

## Implementation Considerations

### **Hint Ordering**

Should hints be replayed in order?

**Unordered**:
\`\`\`
Replay hints in any order (faster).
Rely on timestamps/versions for conflict resolution.
\`\`\`

**Ordered**:
\`\`\`
Replay hints in chronological order (slower).
Ensures causality preserved.
\`\`\`

**Trade-off**: Speed vs causality guarantees.

### **Hint Compression**

Large hints (e.g., binary blobs) can be compressed:

\`\`\`
hint = {
  key: "image:123",
  value: compress (image_data),  // Compressed
  compressed: true
}
\`\`\`

**Benefit**: Reduce disk usage and network transfer.

### **Hint Expiration Check**

Before replaying:

\`\`\`
if current_time - hint.created_at > TTL:
    discard_hint()
    continue
else:
    replay_hint()
\`\`\`

**Avoid**: Replaying very old hints (likely stale).

### **Monitoring**

**Key Metrics**:

**1. Hints Pending**: Number of hints waiting to be replayed
- High: Node down for long time or replay slow

**2. Hint Replay Rate**: Hints/second being replayed
- Low: Replay is slow, tune throttling

**3. Hint Age**: Oldest hint in queue
- Old hints: May exceed TTL soon

**4. Hints Discarded**: Hints discarded due to TTL
- High: TTL too short or node down too long

**Alerts**:
- Hints pending > 1 million
- Oldest hint > 2 hours (approaching 3-hour TTL)
- Hints discarded rate > 1000/s

---

## Alternatives to Hinted Handoff

### **1. Fail the Write**

\`\`\`
Replica down → Write fails → Client retries
\`\`\`

**Pros**: Simple, no hint management
**Cons**: Reduced availability

### **2. Immediate Anti-Entropy**

\`\`\`
Replica down → Write succeeds to available replicas
Later: Run anti-entropy to sync missing replica
\`\`\`

**Pros**: No hint storage
**Cons**: Slower recovery, more load on cluster

### **3. Write-Ahead Log Streaming**

\`\`\`
Write to WAL on available nodes.
Stream WAL to recovering node.
\`\`\`

**Pros**: Efficient, precise
**Cons**: More complex, requires WAL infrastructure

**Used by**: PostgreSQL replication

---

## Interview Tips

### **Key Concepts to Explain**

1. **What is hinted handoff**: Temporary storage of writes for unavailable replica
2. **Why needed**: Improve write availability during failures, faster recovery
3. **How it works**: Store hints on other nodes, replay when target recovers
4. **Not a replica**: Hints don't count toward quorum reads
5. **TTL**: Hints expire to prevent indefinite accumulation
6. **Real-world**: Cassandra (3-hour TTL), Dynamo (sloppy quorum)

### **Common Interview Questions**

**Q: How does hinted handoff improve availability?**
A: "Without hinted handoff, writes fail if required replicas are down. With hinted handoff: If replica C is down, coordinator stores the write as a 'hint' on another node (D). The write succeeds (availability maintained). When C recovers, D replays the hint to C. This allows writes to succeed during temporary failures while ensuring data eventually reaches all replicas (eventual consistency)."

**Q: What\'s the difference between a hint and a replica?**
A: "A hint is a temporary stand-in, not a true replica. It doesn't count toward read quorum—if you read from the hint node, it doesn't have the actual data to return. Hints are only for catching up the down node when it recovers. Example: RF=3, W=2, node C down. Write to A, B, hint on D. Write succeeds (2 acks). But read quorum R=2 must come from actual replicas (A, B), not D."

**Q: What happens if hints accumulate for a node that's down for a long time?**
A: "Hints are capped by TTL (time-to-live, typically 3 hours in Cassandra). After TTL, hints are discarded. This prevents infinite accumulation. If node is down longer than TTL, it must use anti-entropy (read repair, full sync) to catch up. Also enforce size limits: if hints exceed threshold (e.g., 10GB), stop storing new hints and rely on anti-entropy. Trade-off: Faster recovery (hints) vs resource limits (TTL/size caps)."

**Q: How do you handle stale hints?**
A: "Hints include timestamps. When replaying hint to recovered node, node checks timestamp against its current data version. If hint timestamp is older than local data, reject hint (already have newer version). Also, short TTL reduces window for staleness—3-hour hints are less likely stale than 3-day hints. In systems with version vectors, use causal ordering to detect and resolve conflicts."

---

## Summary

Hinted Handoff is a technique for improving availability during temporary failures:

1. **Core Idea**: Store writes for unavailable replica as "hints" on other nodes, replay later
2. **Benefits**: Improved write availability, faster recovery, reduced anti-entropy load
3. **Mechanism**: Coordinator detects down replica, stores hint on fallback node, replays when target recovers
4. **Not a Replica**: Hints don't count toward read quorum, only for write availability
5. **TTL**: Hints expire (typically 3 hours) to prevent accumulation
6. **Real-World**: Cassandra (3-hour TTL), Dynamo (sloppy quorum), Riak
7. **Trade-offs**: Eventual consistency, hint management overhead, not suitable for strong consistency

**Interview Focus**: Understand why hints improve availability (write succeeds despite down replica), how they differ from replicas (don't count toward reads), TTL management (prevent accumulation), and trade-offs (eventual consistency).
`,
};
