/**
 * Quorum Consensus Section
 */

export const quorumconsensusSection = {
  id: 'quorum-consensus',
  title: 'Quorum Consensus',
  content: `Quorum consensus is a fundamental technique in distributed systems that answers a critical question: **"How do we ensure data consistency across replicated servers without requiring all servers to agree?"**

## The Replication Consistency Problem

When you replicate data across multiple servers for fault tolerance, you face a dilemma:

**Scenario**: You store user profile data on 5 servers (replicas) for high availability.

\`\`\`
Replicas: [Server A, Server B, Server C, Server D, Server E]
Data: { "user:123": { "name": "Alice", "email": "alice@example.com" } }
\`\`\`

**Question**: When a client writes new data, how many servers must acknowledge before you return success?

### Option 1: Wait for ALL 5 servers (Strong Consistency)

\`\`\`python
def write(key, value):
    success_count = 0
    for server in replicas:
        server.write(key, value)
        success_count += 1
    
    if success_count == 5:
        return "Success"
    else:
        return "Failed"
\`\`\`

**Problems**:
- ❌ If ANY server is down/slow → Write fails
- ❌ System availability = product of individual availabilities
- ❌ If 99% uptime per server → 99%⁵ = 95% system uptime
- ❌ Very slow (limited by slowest server)

### Option 2: Wait for just 1 server (High Availability)

\`\`\`python
def write(key, value):
    replicas[0].write(key, value)
    return "Success"  # Don't wait for others
\`\`\`

**Problems**:
- ❌ Other replicas have stale data
- ❌ Read might return old value
- ❌ Consistency violations

### Option 3: Quorum (The Sweet Spot)

**Key insight**: Don't need ALL servers to agree, just a **majority**.

---

## Quorum Consensus Basics

**Quorum**: A subset of replicas that is large enough to ensure consistency.

### The Three Numbers: N, W, R

- **N**: Total number of replicas (e.g., 5)
- **W**: Write quorum (servers that must acknowledge write)
- **R**: Read quorum (servers to query for reads)

**The Golden Rule**:
\`\`\`
W + R > N
\`\`\`

**Why this works**: Read and write quorums MUST overlap, ensuring reads see latest writes.

---

## How Quorum Works

### Example Setup

\`\`\`
N = 5 replicas (Servers A, B, C, D, E)
W = 3 (write to at least 3 servers)
R = 3 (read from at least 3 servers)

W + R = 6 > N = 5 ✓
\`\`\`

### Write Operation

Client writes \`user:123 = "Alice v2"\`:

\`\`\`
1. Send write to all 5 servers
2. Wait for 3 servers to acknowledge (W=3)
3. Return success to client

Result:
Server A: "Alice v2" ✓
Server B: "Alice v2" ✓  
Server C: "Alice v2" ✓
Server D: "Alice v1" (slow, not yet updated)
Server E: "Alice v1" (slow, not yet updated)
\`\`\`

**Write succeeded** because 3 servers (≥W) acknowledged.

### Read Operation

Client reads \`user:123\`:

\`\`\`
1. Send read to all 5 servers
2. Wait for 3 responses (R=3)
3. Return the latest value (highest timestamp/version)

Responses:
Server A: "Alice v2" (timestamp: 1000)
Server B: "Alice v2" (timestamp: 1000)
Server D: "Alice v1" (timestamp: 500)

Latest: "Alice v2" ✓
\`\`\`

**Why it works**: Since W=3 servers have v2, and we read from R=3 servers, we're **guaranteed** to read from at least one server with v2 (because W + R > N).

### The Math: Why W + R > N Ensures Consistency

\`\`\`
Write touches W servers
Read touches R servers
Total replicas: N servers

If W + R > N:
  → There must be AT LEAST ONE server in both write quorum and read quorum
  → Read is guaranteed to see latest write

Example: N=5, W=3, R=3
Write touches {A, B, C}
Read touches {C, D, E}
Overlap: Server C (has latest data)
\`\`\`

---

## Quorum Configurations

### 1. Strong Consistency: W + R > N

\`\`\`
N=5, W=3, R=3 → 3+3 = 6 > 5 ✓

Guarantees: Read always sees latest write
Use case: Banking, inventory systems
Trade-off: Need quorum for all operations (slower)
\`\`\`

### 2. Eventual Consistency: W + R ≤ N

\`\`\`
N=5, W=1, R=1 → 1+1 = 2 ≤ 5

Guarantees: None, data eventually consistent
Use case: Social media feeds, analytics
Trade-off: Fastest, but may read stale data
\`\`\`

### 3. Write-Heavy Optimization: Low W

\`\`\`
N=5, W=2, R=4 → 2+4 = 6 > 5 ✓

Guarantees: Strong consistency
Optimized for: Write-heavy workloads (faster writes)
Trade-off: Slower reads
Use case: Logging systems
\`\`\`

### 4. Read-Heavy Optimization: Low R

\`\`\`
N=5, W=4, R=2 → 4+2 = 6 > 5 ✓

Guarantees: Strong consistency  
Optimized for: Read-heavy workloads (faster reads)
Trade-off: Slower writes
Use case: Caching, content delivery
\`\`\`

### 5. Highest Consistency: W + R = 2N

\`\`\`
N=5, W=5, R=5 → 5+5 = 10 >> 5

Guarantees: Strongest possible consistency
Trade-off: Requires ALL replicas (low availability)
Use case: Rarely used, too strict
\`\`\`

---

## Quorum Across Consistency Models

### Strict Quorum: W + R > N

**Example**: N=3, W=2, R=2

\`\`\`
Write to 2 of 3 servers → Success
Read from 2 of 3 servers → Get latest

Guaranteed overlap ensures strong consistency
\`\`\`

### Sloppy Quorum (Amazon Dynamo)

**Problem**: Strict quorum fails if too many replicas are down.

**Solution**: Accept writes from ANY N servers (not just designated replicas).

\`\`\`
N=3 designated replicas: {A, B, C}
A and B are down

Strict quorum: Write fails (can't reach W=2)
Sloppy quorum: Write to C and D (temporary replica)
  → Later, "hinted handoff" transfers data from D to A/B when they recover
\`\`\`

**Trade-off**: Higher availability, eventual consistency.

**Used by**: DynamoDB, Riak, Voldemort

---

## Quorum in Real-World Systems

### 1. Apache Cassandra

**Configuration**:
\`\`\`sql
CREATE TABLE users (
    user_id UUID PRIMARY KEY,
    name TEXT
) WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 3};

-- Write with QUORUM
INSERT INTO users (user_id, name) VALUES (uuid(), 'Alice')
USING CONSISTENCY QUORUM;  -- W = 2 (majority of 3)

-- Read with QUORUM  
SELECT * FROM users WHERE user_id = ?
USING CONSISTENCY QUORUM;  -- R = 2 (majority of 3)
\`\`\`

**Consistency Levels** (configurable per query):
- \`ONE\`: W=1 or R=1 (fastest, eventual consistency)
- \`QUORUM\`: W=⌈N/2⌉, R=⌈N/2⌉ (strong consistency)
- \`ALL\`: W=N, R=N (strongest, low availability)
- \`LOCAL_QUORUM\`: Quorum within datacenter (geo-replication)

**Default**: Usually \`QUORUM\` for balance.

### 2. Amazon DynamoDB

**Configuration**:
\`\`\`python
# Default: Eventual consistency (R=1)
response = table.get_item(
    Key={'user_id': '123'}
)

# Strong consistency (R=2 for N=3)
response = table.get_item(
    Key={'user_id': '123'},
    ConsistentRead=True
)
\`\`\`

**Write path**: Always W=2 (for N=3 replicas)

**Read path**:
- Eventual consistency: R=1 (fast, may be stale)
- Strong consistency: R=2 (guaranteed latest)

**Sloppy quorum**: Writes accepted by any healthy nodes.

### 3. MongoDB

**Write Concern**:
\`\`\`javascript
db.users.insert(
  { name: "Alice" },
  { writeConcern: { w: "majority" } }  // Wait for majority
)
\`\`\`

**Read Concern**:
\`\`\`javascript
db.users.find().readConcern("majority")
\`\`\`

**Configuration**:
- \`w: 1\`: Write to primary only
- \`w: "majority"\`: Write to majority of replicas
- \`w: N\`: Write to N replicas

### 4. Riak

**N, W, R configurable per bucket**:
\`\`\`erlang
{n_val, 3},     % N=3 replicas
{w, 2},          % W=2 write quorum  
{r, 2},          % R=2 read quorum
{dw, 1}          % Durable writes (to disk)
\`\`\`

**Supports sloppy quorum with hinted handoff**.

---

## Handling Failures

### Scenario: Write with Partial Failure

\`\`\`
N=5, W=3, R=3

Write to "user:123":
Server A: Success ✓
Server B: Success ✓
Server C: Timeout ✗
Server D: Success ✓
Server E: Down ✗

Result: 3 successes ≥ W=3 → Write succeeds
\`\`\`

**Read repair**: When reading, if servers have different versions, update stale servers.

### Scenario: Temporary Network Partition

\`\`\`
Partition A: {Server 1, Server 2}
Partition B: {Server 3, Server 4, Server 5}

N=5, W=3, R=3

Write in Partition B:
- 3 servers available ≥ W=3 → Write succeeds

Write in Partition A:
- 2 servers available < W=3 → Write fails (availability trade-off)
\`\`\`

**Split-brain prevention**: Quorum ensures only one partition can accept writes.

---

## Versioning and Conflict Resolution

### Vector Clocks (for concurrent writes)

\`\`\`
T0: Initial state across 3 replicas
All: {name: "Alice"}

T1: Client 1 writes to {A, B}
A: {name: "Alice Smith", version: {A:1, B:1, C:0}}
B: {name: "Alice Smith", version: {A:1, B:1, C:0}}
C: {name: "Alice"}  (stale)

T2: Client 2 writes to {B, C} (before replication from T1)
B: {name: "Alicia", version: {A:0, B:2, C:1}}  
C: {name: "Alicia", version: {A:0, B:2, C:1}}

Conflict! Two concurrent writes
\`\`\`

**Resolution strategies**:
1. **Last-Write-Wins** (timestamp): Simple but loses data
2. **Vector Clocks**: Detect conflicts, application resolves
3. **CRDTs**: Automatic conflict resolution

---

## Trade-offs

### Availability vs Consistency

**Higher W**:
- ✅ More consistent (more replicas updated)
- ❌ Lower availability (need more replicas up)
- ❌ Higher write latency

**Lower W**:
- ✅ Higher availability
- ✅ Lower write latency
- ❌ Less consistent

### Latency Optimization

**Parallel Reads**:
\`\`\`python
# Send read to all N servers
# Return as soon as R respond (don't wait for all)
\`\`\`

**Speculative Reads**:
\`\`\`
Send to R servers
If no response after 10ms, send to R+1 servers
Return first R responses
\`\`\`

---

## Implementation Example

\`\`\`python
import time
from typing import List, Any

class QuorumStore:
    def __init__(self, replicas: List[str], n: int, w: int, r: int):
        self.replicas = replicas
        self.n = n
        self.w = w
        self.r = r
        
        # Validate quorum rule
        assert w + r > n, "Must satisfy W + R > N for strong consistency"
    
    def write(self, key: str, value: Any) -> bool:
        """Write with quorum"""
        timestamp = int(time.time() * 1000)
        version = {"value": value, "timestamp": timestamp}
        
        successes = 0
        for replica in self.replicas:
            try:
                self._send_write(replica, key, version)
                successes += 1
                
                # Early return once quorum achieved
                if successes >= self.w:
                    return True
            except Exception:
                continue
        
        return successes >= self.w
    
    def read(self, key: str) -> Any:
        """Read with quorum"""
        responses = []
        
        for replica in self.replicas:
            try:
                version = self._send_read(replica, key)
                responses.append(version)
                
                # Early return once quorum achieved
                if len(responses) >= self.r:
                    break
            except Exception:
                continue
        
        if len(responses) < self.r:
            raise Exception(f"Failed to achieve read quorum R={self.r}")
        
        # Return value with highest timestamp (latest write)
        latest = max(responses, key=lambda v: v["timestamp"])
        
        # Read repair: Update stale replicas
        self._read_repair(key, latest, responses)
        
        return latest["value"]
    
    def _read_repair(self, key: str, latest: dict, responses: List[dict]):
        """Update replicas with stale data"""
        for i, response in enumerate(responses):
            if response["timestamp"] < latest["timestamp"]:
                # This replica is stale, update it
                self._send_write(self.replicas[i], key, latest)

# Usage
replicas = ["server1", "server2", "server3", "server4", "server5"]
store = QuorumStore(replicas, n=5, w=3, r=3)

# Write with quorum
store.write("user:123", {"name": "Alice", "email": "alice@example.com"})

# Read with quorum (guaranteed to see latest write)
user = store.read("user:123")
\`\`\`

---

## Common Configurations

| N | W | R | Consistency | Availability | Use Case |
|---|---|---|-------------|--------------|----------|
| 3 | 2 | 2 | Strong | Medium | General purpose |
| 5 | 3 | 3 | Strong | High | Production default |
| 3 | 1 | 3 | Eventual (fast writes) | High | Logging |
| 3 | 3 | 1 | Eventual (fast reads) | High | Analytics |
| 3 | 1 | 1 | Eventual | Highest | Non-critical data |
| 3 | 3 | 3 | Strongest | Low | Financial systems |

---

## Interview Tips

### Key Talking Points

1. **Quorum rule**: W + R > N ensures strong consistency
2. **Trade-off triangle**: Consistency, Availability, Latency (pick 2)
3. **Real-world default**: N=3, W=2, R=2 (or N=5, W=3, R=3)
4. **Configurability**: Adjust W/R based on workload
5. **Systems using quorum**: Cassandra, DynamoDB, Riak, MongoDB

### Common Interview Questions

**"How does quorum ensure consistency?"**
- W + R > N guarantees read and write quorums overlap
- At least one server in read quorum has latest write
- Return highest version/timestamp

**"What is the trade-off with higher W?"**
- Higher W: More consistent, but slower writes and lower availability
- Lower W: Faster writes, but need higher R for consistency
- Balance based on read/write ratio

**"How do you handle replicas being down?"**
- Quorum allows some replicas to be down
- If too many down, writes fail (preserve consistency)
- Sloppy quorum: Accept writes from any servers (higher availability)

**"What is sloppy quorum?"**
- Accept writes from ANY N servers (not just designated replicas)
- Hinted handoff: Transfer data when designated replicas recover
- Trade-off: Higher availability, eventual consistency
- Used by: DynamoDB, Riak

### Design Exercise

Design a distributed cache with N=5 replicas:

\`\`\`
Requirement: Read-heavy workload (90% reads, 10% writes)

Configuration: N=5, W=4, R=2 (W + R = 6 > 5)

Reasoning:
- High W=4: Writes slow but very consistent
- Low R=2: Reads fast (only need 2 responses)
- Still strong consistency (W + R > N)
- Optimized for read-heavy pattern
- Can tolerate 3 failures for reads, 1 failure for writes

Alternative for write-heavy: N=5, W=2, R=4
\`\`\`

---

## Summary

**Quorum consensus** is the fundamental technique for achieving consistency in replicated distributed systems without requiring all replicas to agree.

**Key principles**:
- ✅ **N, W, R parameters**: Control consistency vs availability trade-off
- ✅ **Golden rule**: W + R > N ensures strong consistency
- ✅ **Flexibility**: Configure per workload (read-heavy vs write-heavy)
- ✅ **Fault tolerance**: Continue operating with some replicas down
- ✅ **Production standard**: N=3 (W=2, R=2) or N=5 (W=3, R=3)

**Industry adoption**: Cassandra, DynamoDB, Riak, MongoDB, Voldemort—every major distributed database uses quorum-based replication.

Understanding quorum consensus is **essential** for designing consistent, available, partition-tolerant distributed systems.`,
};
