/**
 * Vector Clocks & Version Vectors Section
 */

export const vectorclocksSection = {
  id: 'vector-clocks-version-vectors',
  title: 'Vector Clocks & Version Vectors',
  content: `Vector clocks are a fundamental mechanism for tracking causality and detecting conflicts in distributed systems. They answer a critical question: **"In a system with multiple replicas accepting writes, how do we know which version of data is newer when timestamps are unreliable?"**

## The Distributed Timestamp Problem

In distributed systems, we cannot rely on wall-clock timestamps due to:
- **Clock skew**: Different servers have slightly different times
- **Clock drift**: Clocks advance at different rates  
- **NTP limitations**: Network Time Protocol has ~100ms accuracy
- **Malicious actors**: Clocks can be manually adjusted

### Problem Example

\`\`\`
Server A (clock: 10:00:00): Write user:123 = "Alice"
Server B (clock: 09:59:50): Write user:123 = "Alicia"  

Which write is newer?
\`\`\`

If we use wall-clock timestamps:
- Server A timestamp: 10:00:00 (appears newer)
- Server B timestamp: 09:59:50 (appears older)

But what if Server B's write happened **after** A's write, and Server B just has a slow clock?

**We cannot trust wall clocks for ordering distributed events.**

---

## Vector Clocks: The Solution

**Vector clocks** track causality using logical timestamps that are **independent of physical time**.

**Key insight**: Each server maintains a counter of events it has seen. A vector clock is an array of these counters, one per server.

### Structure

For N servers, a vector clock is an array of N integers:

\`\`\`
Vector Clock = [Server1_counter, Server2_counter, ..., ServerN_counter]

Example with 3 servers:
[0, 0, 0]  // Initial state
[1, 0, 0]  // Server 1 performed 1 event
[2, 1, 0]  // Server 1: 2 events, Server 2: 1 event
\`\`\`

**Each server tracks**:
1. **Its own counter**: Incremented on every local event
2. **Other servers' counters**: Updated based on messages received

---

## How Vector Clocks Work

### Rules

**Rule 1: Internal Events**
When server i performs a local write:
- Increment its own counter: \`VC[i]++\`

**Rule 2: Sending Messages**
When server i sends a message:
- Increment its own counter: \`VC[i]++\`
- Attach current vector clock to message

**Rule 3: Receiving Messages**
When server i receives message with VC_msg:
- Increment its own counter: \`VC[i]++\`
- Update all counters to maximum: \`VC[j] = max(VC[j], VC_msg[j])\` for all j

### Concrete Example

\`\`\`
System: 3 servers (A, B, C)

Event 1: A writes "Alice"
  A: [1, 0, 0]  (A increments its counter)

Event 2: B writes "Bob"  
  B: [0, 1, 0]  (B increments its counter)

Event 3: A writes "Alice v2", sends to B
  A: [2, 0, 0]  (A increments before sending)
  Message: {value: "Alice v2", VC: [2, 0, 0]}

Event 4: B receives message from A
  B: [2, 2, 0]  (takes max([0,1,0], [2,0,0]) + increments own)
  B now knows: A has done 2 events, B has done 2 events

Event 5: C writes "Carol"
  C: [0, 0, 1]  (independent, no knowledge of A/B)

Event 6: C sends to B
  C: [0, 0, 2]
  Message: {value: "Carol", VC: [0, 0, 2]}

Event 7: B receives from C
  B: [2, 3, 2]  (takes max([2,2,0], [0,0,2]) + increments own)
  B now knows: A did 2 events, B did 3 events, C did 2 events
\`\`\`

---

## Comparing Vector Clocks

Given two vector clocks \`VC1\` and \`VC2\`, exactly **one** of three relationships holds:

### 1. VC1 Happens Before VC2 (VC1 < VC2)

\`\`\`
VC1[i] ≤ VC2[i] for all i, AND VC1 ≠ VC2
\`\`\`

**Meaning**: All events in VC1 causally precede VC2.

\`\`\`
VC1 = [1, 2, 0]
VC2 = [2, 3, 1]

VC1 < VC2 because:
  1 ≤ 2  ✓
  2 ≤ 3  ✓  
  0 ≤ 1  ✓
  
→ VC1 happened before VC2
\`\`\`

### 2. VC2 Happens Before VC1 (VC2 < VC1)

\`\`\`
VC2[i] ≤ VC1[i] for all i, AND VC1 ≠ VC2
\`\`\`

\`\`\`
VC1 = [3, 5, 2]
VC2 = [2, 4, 1]

VC2 < VC1 because:
  2 ≤ 3  ✓
  4 ≤ 5  ✓
  1 ≤ 2  ✓
  
→ VC2 happened before VC1
\`\`\`

### 3. Concurrent (VC1 || VC2)

\`\`\`
Neither VC1 < VC2 nor VC2 < VC1
\`\`\`

**Meaning**: Events happened concurrently (no causal relationship).

\`\`\`
VC1 = [2, 1, 0]
VC2 = [1, 2, 0]

Not VC1 < VC2 because 2 > 1 (first element)
Not VC2 < VC1 because 2 > 1 (second element)

→ VC1 and VC2 are CONCURRENT (conflict!)
\`\`\`

---

## Conflict Detection

Vector clocks excel at detecting **concurrent writes** (conflicts):

\`\`\`
Server A writes: user:123 = "Alice", VC: [2, 0, 0]
Server B writes: user:123 = "Alicia", VC: [0, 2, 0]

Compare:
  [2, 0, 0] vs [0, 2, 0]
  
  2 > 0 (A ahead in first position)
  0 < 2 (B ahead in second position)
  
→ CONCURRENT writes! Both versions need to be preserved.
\`\`\`

---

## Implementation

\`\`\`python
from typing import Dict, List, Optional

class VectorClock:
    def __init__(self, server_id: str, num_servers: int):
        self.server_id = server_id
        self.server_index = int(server_id.split('_')[1])  # e.g., "server_0" → 0
        self.clock = [0] * num_servers
    
    def increment(self):
        """Increment this server's counter"""
        self.clock[self.server_index] += 1
    
    def update(self, other_clock: List[int]):
        """Update clock based on received message"""
        # Take maximum of each element
        for i in range(len(self.clock)):
            self.clock[i] = max(self.clock[i], other_clock[i])
        
        # Increment own counter
        self.increment()
    
    def happens_before(self, other_clock: List[int]) -> bool:
        """Check if this clock happens before other clock"""
        # All elements ≤ AND at least one element <
        all_leq = all(self.clock[i] <= other_clock[i] for i in range(len(self.clock)))
        at_least_one_less = any(self.clock[i] < other_clock[i] for i in range(len(self.clock)))
        
        return all_leq and at_least_one_less
    
    def concurrent_with(self, other_clock: List[int]) -> bool:
        """Check if this clock is concurrent with other clock"""
        return (not self.happens_before(other_clock) and 
                not VectorClock.happens_before_static(other_clock, self.clock))
    
    @staticmethod
    def happens_before_static(clock1: List[int], clock2: List[int]) -> bool:
        all_leq = all(clock1[i] <= clock2[i] for i in range(len(clock1)))
        at_least_one_less = any(clock1[i] < clock2[i] for i in range(len(clock1)))
        return all_leq and at_least_one_less

# Usage
class DistributedStore:
    def __init__(self, server_id: str, num_servers: int):
        self.server_id = server_id
        self.vc = VectorClock(server_id, num_servers)
        self.data = {}
    
    def write(self, key: str, value: any):
        """Local write"""
        self.vc.increment()
        self.data[key] = {
            'value': value,
            'vector_clock': self.vc.clock.copy()
        }
        return self.data[key]
    
    def receive_write(self, key: str, value: any, remote_vc: List[int]):
        """Receive write from another server"""
        self.vc.update(remote_vc)
        
        if key not in self.data:
            # New key, just store
            self.data[key] = {'value': value, 'vector_clock': remote_vc}
        else:
            local_vc = self.data[key]['vector_clock']
            
            if self.vc.happens_before_static(local_vc, remote_vc):
                # Remote write is newer, replace
                self.data[key] = {'value': value, 'vector_clock': remote_vc}
            elif self.vc.happens_before_static(remote_vc, local_vc):
                # Local write is newer, keep local
                pass
            else:
                # CONCURRENT! Store both versions
                if 'conflicts' not in self.data[key]:
                    self.data[key]['conflicts'] = []
                self.data[key]['conflicts'].append({
                    'value': value,
                    'vector_clock': remote_vc
                })

# Example
store_a = DistributedStore("server_0", num_servers=3)
store_b = DistributedStore("server_1", num_servers=3)

# Server A writes
write_a = store_a.write("user:123", "Alice")
print(f"A writes: {write_a}")  # VC: [1, 0, 0]

# Server B writes (concurrent with A)
write_b = store_b.write("user:123", "Alicia")
print(f"B writes: {write_b}")  # VC: [0, 1, 0]

# B receives A's write
store_b.receive_write("user:123", "Alice", write_a['vector_clock'])
# Detects conflict! Both versions stored
\`\`\`

---

## Real-World Use Cases

### 1. Amazon Dynamo / Riak

**Problem**: Multi-master replication with concurrent writes

**Solution**: Vector clocks to detect conflicts
\`\`\`
Shopping cart example:
User adds item via Server A: {item1}, VC:[1,0,0]
Simultaneously adds item via Server B: {item2}, VC:[0,1,0]

Conflict detected: VC:[1,0,0] || VC:[0,1,0]

Resolution: Merge carts → {item1, item2}
\`\`\`

**Riak**:
\`\`\`bash
# Riak returns vector clock with every read
GET /buckets/shopping_cart/keys/user_123
X-Riak-Vclock: a85hYGBgzGDKBVIcR4M2cgczH7HPYEpkymNlSJcs6XNkYA==

# Client includes vector clock in write
PUT /buckets/shopping_cart/keys/user_123
X-Riak-Vclock: a85hYGBgzGDKBVIcR4M2cgczH7HPYEpkymNlSJcs6XNkYA==
\`\`\`

### 2. Cassandra

**Conflict resolution**:
- Last-Write-Wins (timestamp-based, simpler)
- Vector clocks available but not default

### 3. CouchDB

**Multi-version concurrency control**:
\`\`\`json
{
  "_id": "user:123",
  "_rev": "3-a1b2c3d4",  // Revision vector
  "name": "Alice"
}
\`\`\`

### 4. Distributed Version Control (Git)

**Git commits use causality**:
\`\`\`
Commit A: [1, 0]  (on branch main)
Commit B: [0, 1]  (on branch feature)

Merge conflict: A || B (concurrent commits)
→ Manual resolution required
\`\`\`

---

## Version Vectors

**Version vectors** are a simplified variant of vector clocks used in production systems.

**Difference**:
- **Vector clocks**: Track every event (writes, messages)
- **Version vectors**: Track only data versions (writes only)

**Advantage**: Smaller size, simpler implementation

\`\`\`python
# Vector clock (full causality)
{
  "value": "Alice",
  "vector_clock": [5, 12, 8]  # All events on all servers
}

# Version vector (only writes)
{
  "value": "Alice",  
  "version_vector": [2, 3, 1]  # Only writes from each server
}
\`\`\`

**Trade-off**: Version vectors may report more false conflicts (conservative), but are more practical.

**Used by**: Riak (calls them vector clocks but actually version vectors), Voldemort

---

## Handling Conflicts

When vector clocks detect concurrent writes, application must resolve:

### 1. Last-Write-Wins (LWW)

\`\`\`python
def resolve_conflict(versions):
    # Pick version with highest timestamp
    return max(versions, key=lambda v: v['timestamp'])
\`\`\`

**Pros**: Simple, deterministic
**Cons**: Loses data, arbitrary ordering

### 2. Merge Conflicting Versions

\`\`\`python
def merge_shopping_carts(cart1, cart2):
    # Union of items
    return cart1['items'] + cart2['items']
\`\`\`

**Pros**: No data loss
**Cons**: Requires domain-specific logic

### 3. CRDTs (Conflict-free Replicated Data Types)

**Automatic conflict resolution** through commutative operations:
\`\`\`
G-Counter (Grow-only counter):
  Server A: increment → [1, 0, 0]
  Server B: increment → [0, 1, 0]
  Merge: [1, 1, 0] → total = 2
  
  Order doesn't matter! Commutative.
\`\`\`

### 4. Client-Side Resolution

\`\`\`javascript
// Client sees conflict
GET /shopping_cart/user_123
Response: {
  "conflicts": [
    {"items": ["item1"], "vc": [1,0,0]},
    {"items": ["item2"], "vc": [0,1,0]}
  ]
}

// Client merges and writes back
PUT /shopping_cart/user_123
{
  "items": ["item1", "item2"],
  "resolved_vcs": [[1,0,0], [0,1,0]]
}
\`\`\`

---

## Limitations and Alternatives

### Size Growth Problem

Vector clocks grow with number of servers:
\`\`\`
10 servers → 10 integers (40 bytes)
1000 servers → 1000 integers (4 KB)
10,000 servers → 10,000 integers (40 KB per object!)
\`\`\`

**Solution**: Dotted version vectors (prune old entries)

### Comparison: Vector Clocks vs Physical Timestamps

| Approach | Pros | Cons |
|----------|------|------|
| **Physical timestamps** | Simple, small (8 bytes) | Clock skew, not causally accurate |
| **Vector clocks** | Precise causality | Size grows with servers |
| **Hybrid clocks** | Best of both | More complex |

### Hybrid Logical Clocks (HLC)

**Combines physical time + logical counter**:
\`\`\`
HLC = (physical_time, logical_counter)

Benefits:
- Small size (constant)
- Causality tracking
- Approximate physical time

Used by: CockroachDB, YugabyteDB
\`\`\`

---

## Interview Tips

### Key Talking Points

1. **Problem**: Cannot trust wall clocks in distributed systems
2. **Vector clocks**: Track causality with logical counters (one per server)
3. **Comparison**: Can determine if events are ordered or concurrent
4. **Conflict detection**: Concurrent writes detected → need resolution
5. **Real-world**: DynamoDB, Riak, CouchDB, Git

### Common Interview Questions

**"Why can't we use timestamps for ordering?"**
- Clock skew and drift between servers
- Network delays ≠ causality
- Example: Event B caused by A, but timestamp(B) < timestamp(A) due to slow clock

**"How do vector clocks work?"**
- Each server maintains counter array
- Increment on local events
- Merge on message receive (take max + increment own)
- Can determine: happens-before or concurrent

**"What happens when vector clocks detect a conflict?"**
- Application must resolve (not automatic)
- Options: LWW, merge, CRDT, client-side resolution
- Design choice depends on domain

**"What are limitations?"**
- Size: O(N) per object where N = number of servers
- Doesn't work well for 1000+ servers
- Alternatives: Dotted version vectors, hybrid logical clocks

### Design Exercise

Design a distributed shopping cart with vector clocks:

\`\`\`
1. Each cart operation includes vector clock
2. On write: increment server's counter, attach VC
3. On read: Return cart + VC
4. Client includes VC in next write (causality preserved)
5. On conflict: Merge carts (union of items)

Example:
User adds item1 via Server A: VC:[1,0,0], cart:[item1]
User adds item2 via Server B: VC:[0,1,0], cart:[item2]

Conflict detected: [1,0,0] || [0,1,0]
Resolution: Merge → cart:[item1, item2], VC:[1,1,0]
\`\`\`

---

## Summary

**Vector clocks** are a fundamental mechanism for tracking causality and detecting conflicts in distributed systems.

**Key principles**:
- ✅ Logical timestamps independent of physical clocks
- ✅ Each server maintains counter array (one per server)
- ✅ Can determine: happens-before or concurrent
- ✅ Detects conflicts that require resolution
- ✅ Foundation for multi-master replication

**Limitations**:
- ❌ Size grows with number of servers
- ❌ Requires application-level conflict resolution
- ❌ More complex than timestamps

**Industry adoption**: DynamoDB, Riak, Voldemort, CouchDB, Git—every multi-master system needs causality tracking.

Understanding vector clocks is **essential** for designing systems with concurrent writes and conflict resolution.`,
};
