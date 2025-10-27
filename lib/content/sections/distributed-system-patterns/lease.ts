/**
 * Lease Section
 */

export const leaseSection = {
  id: 'lease',
  title: 'Lease',
  content: `A Lease is a time-bound lock or permission that grants exclusive access to a resource for a limited duration. It\'s a fundamental pattern in distributed systems for managing resource ownership, preventing split-brain, and implementing distributed coordination.

## What is a Lease?

A **lease** is a **time-limited grant of exclusive rights** to a resource or role.

**Key Properties**:
- **Time-bound**: Expires after a fixed duration (e.g., 30 seconds)
- **Exclusive**: Only one holder at a time
- **Renewable**: Holder can extend before expiration
- **Automatic release**: Expires if not renewed (failure detection)

\`\`\`
Example:
  Node A acquires lease on partition-0 for 30 seconds
  
  T=0s:  Lease granted, expires at T=30s
  T=10s: Node A renews lease, expires at T=40s
  T=15s: Node A crashes
  T=40s: Lease expires automatically
  T=41s: Node B can acquire lease
\`\`\`

**vs Lock**: Lock is indefinite until released. Lease expires automatically.

---

## Why Leases Are Needed

### **1. Automatic Failure Recovery**

**Problem with Traditional Locks**:
\`\`\`
1. Node A acquires lock on resource
2. Node A crashes while holding lock
3. Lock never released
4. Resource is locked forever ❌
\`\`\`

**Solution with Lease**:
\`\`\`
1. Node A acquires lease for 30 seconds
2. Node A crashes at T=10s
3. Lease expires at T=30s
4. Node B acquires lease at T=31s ✅
\`\`\`

**Result**: System automatically recovers from failures without manual intervention.

### **2. Preventing Split-Brain**

In distributed leader election, leases prevent two nodes from acting as leader simultaneously.

\`\`\`
Without Lease:
  1. Node A is leader
  2. Network partition
  3. Node B thinks A is dead, becomes leader
  4. Both A and B act as leader (split-brain) ❌

With Lease:
  1. Node A acquires leader lease for 30s
  2. Network partition at T=10s
  3. Node B waits until T=30s (lease expiry)
  4. Node B becomes leader at T=31s
  5. Node A stopped acting as leader at T=30s ✅
\`\`\`

**Result**: At most one leader at any time (may be no leader during transition).

### **3. Read Optimization**

Leases allow caching without frequent revalidation.

\`\`\`
1. Client reads data from cache
2. Cache has lease on data (valid for 60s)
3. Client can read from cache without contacting origin
4. After 60s, cache must revalidate
\`\`\`

**Benefit**: Reduce load on primary database, lower latency.

### **4. Work Distribution**

Leases can assign work to nodes temporarily.

\`\`\`
1. Worker node acquires lease on task queue partition
2. Processes tasks from partition for lease duration
3. Renews lease if still healthy
4. If worker crashes, lease expires, another worker takes over
\`\`\`

**Benefit**: Dynamic load balancing, automatic failover.

---

## How Leases Work

### **Basic Protocol**

**Acquisition**:
\`\`\`
1. Client sends lease request to coordinator
2. Coordinator checks if lease is available
3. If available:
   - Grant lease with expiration time
   - Record (client_id, resource, expiry_time)
4. If unavailable:
   - Reject or queue request
\`\`\`

**Renewal**:
\`\`\`
1. Before expiration, client sends renewal request
2. Coordinator verifies client still holds lease
3. Extend lease expiration time
4. Return new expiration time to client
\`\`\`

**Expiration**:
\`\`\`
1. Coordinator monitors expiration times
2. When lease expires:
   - Remove lease record
   - Resource becomes available
   - Queue next requester (if any)
\`\`\`

**Release**:
\`\`\`
1. Client explicitly releases lease (before expiration)
2. Coordinator removes lease record
3. Resource immediately available
\`\`\`

### **Lease Table**

Coordinator maintains lease table:

\`\`\`
| Resource ID | Holder    | Granted At | Expires At | Lease Duration |
|-------------|-----------|------------|------------|----------------|
| partition-0 | node-A    | T=100      | T=130      | 30s            |
| partition-1 | node-B    | T=105      | T=135      | 30s            |
| partition-2 | node-C    | T=110      | T=140      | 30s            |
| partition-3 | (none)    | -          | -          | -              |
\`\`\`

### **Timeline Example**

\`\`\`
T=0s:   Node A requests lease on resource X
T=1s:   Coordinator grants lease (expires at T=31s)
T=20s:  Node A renews lease (expires at T=51s)
T=40s:  Node A renews lease (expires at T=71s)
T=45s:  Node A crashes
T=71s:  Lease expires (Node A didn't renew)
T=72s:  Node B requests lease
T=73s:  Coordinator grants lease to Node B
\`\`\`

**Key Point**: Even though Node A crashed at T=45s, resource wasn't available until T=71s (lease expiration). This is the **price of automatic recovery** - some unavailability.

---

## Lease Duration Selection

**Trade-off**: Short vs long leases

### **Short Leases** (e.g., 5-10 seconds)

**Advantages**:
- Fast failure detection
- Quick recovery (new node can acquire soon)
- Less time with "dead" resource

**Disadvantages**:
- Frequent renewal overhead (network, CPU)
- Risk of false failure (short network hiccup)
- Higher load on coordinator

**Use when**: Fast failover is critical, network is reliable.

### **Long Leases** (e.g., 60-300 seconds)

**Advantages**:
- Less renewal overhead
- More tolerant of transient network issues
- Lower load on coordinator

**Disadvantages**:
- Slow failure detection
- Longer unavailability after crashes
- Risk of stale leases

**Use when**: Network is unreliable, coordinator load is high.

### **Adaptive Leases**

Adjust lease duration based on conditions:

\`\`\`
if network_latency < 50ms:
    lease_duration = 10s  (fast network)
elif network_latency < 200ms:
    lease_duration = 30s  (normal)
else:
    lease_duration = 60s  (slow network)
\`\`\`

Or based on failure rate:

\`\`\`
if recent_failures > threshold:
    lease_duration *= 1.5  (increase to reduce false failures)
\`\`\`

---

## Clock Skew and Leases

**Critical Problem**: Leases depend on time, but clocks can be wrong.

### **Scenario: Clock Skew**

\`\`\`
Coordinator Clock: 12:00:00
Node A Clock:      12:00:30 (30 seconds ahead)

1. Coordinator grants lease to Node A (expires at 12:00:30)
2. Node A thinks current time is 12:00:30 (lease already expired!)
3. Node A immediately stops acting on lease
4. System unavailable ❌
\`\`\`

### **Solutions**

**1. Lease Expiry in Coordinator Time**

\`\`\`
Request:
  Node → Coordinator: "Grant me lease for 30s"

Response:
  Coordinator → Node: "Lease granted, valid until T=12:00:30 (my time)"
  
Node Logic:
  - Don't use own clock to check expiration
  - Renew lease before coordinator's stated expiry time
  - Account for network round-trip time
\`\`\`

**2. Time-to-Live (TTL) Instead of Absolute Time**

\`\`\`
Response:
  Coordinator → Node: "Lease granted, TTL=30s, granted_at=T"
  
Node Logic:
  elapsed = (current_time - granted_at)  (use node's clock for elapsed)
  if elapsed > TTL * 0.8:  (renew at 80% of TTL)
      renew_lease()
\`\`\`

**3. NTP Synchronization**

- Synchronize all clocks using NTP
- Acceptable skew: < 100ms
- Monitor clock drift, alert on large skew

**4. Conservative Safety Margin**

\`\`\`
Coordinator grants 30s lease.
Node treats it as 25s lease (5s safety margin).
Node renews at 20s (80% of 25s).
\`\`\`

**Trade-off**: Slightly more overhead, but much safer.

---

## Lease Renewal Strategies

### **1. Fixed Interval Renewal**

\`\`\`
lease_duration = 30s
renewal_interval = 10s

T=0s:   Acquire lease (expires at T=30s)
T=10s:  Renew (expires at T=40s)
T=20s:  Renew (expires at T=50s)
T=30s:  Renew (expires at T=60s)
\`\`\`

**Simple but wasteful**: Many renewals.

### **2. Percentage-Based Renewal**

\`\`\`
renewal_threshold = 0.5  (50% of lease duration)

T=0s:   Acquire lease (expires at T=30s)
T=15s:  Renew (50% elapsed, expires at T=45s)
T=30s:  Renew (50% elapsed, expires at T=60s)
\`\`\`

**Better**: Fewer renewals, scales with lease duration.

### **3. Exponential Backoff on Failure**

\`\`\`
T=15s:  Attempt to renew
        Coordinator unreachable, retry in 1s
T=16s:  Retry, failed, retry in 2s
T=18s:  Retry, failed, retry in 4s
T=22s:  Retry, failed, retry in 8s
T=30s:  Lease expires, stop acting on lease
\`\`\`

**Prevents**: Renewal storms, coordinator overload.

### **4. Renewal with Jitter**

\`\`\`
renewal_time = lease_duration * 0.5 + random(0, lease_duration * 0.1)

Example: 30s lease
  Node A renews at 15s
  Node B renews at 17s
  Node C renews at 16s
\`\`\`

**Prevents**: Thundering herd (all nodes renewing at same time).

---

## Lease-Based Leader Election

**Use Case**: Elect a leader using a lease on a special "leader" resource.

### **Protocol**

\`\`\`
1. All nodes attempt to acquire lease on "leader" resource
2. Coordinator grants lease to first requester (Node A)
3. Node A acts as leader while holding lease
4. Node A renews lease periodically
5. Other nodes periodically attempt to acquire (in case lease expires)
6. If Node A fails to renew:
   - Lease expires
   - Next node acquires lease, becomes leader
\`\`\`

### **Example: Kubernetes Leader Election**

Kubernetes uses leases for controller leader election.

\`\`\`yaml
apiVersion: coordination.k8s.io/v1
kind: Lease
metadata:
  name: my-controller-leader
spec:
  holderIdentity: pod-1
  leaseDurationSeconds: 15
  renewTime: "2024-01-15T10:00:30Z"
\`\`\`

**Process**:
1. Multiple controller pods try to acquire lease
2. First pod to acquire becomes leader
3. Leader renews lease every 10s (before 15s expiry)
4. If leader crashes, lease expires, another pod becomes leader

**Safety**: At most one leader at any time.

---

## Lease-Based Caching

**Use Case**: Cache data with lease to ensure freshness.

### **Protocol**

\`\`\`
1. Client requests data from cache
2. Cache doesn't have data or lease expired
3. Cache requests data + lease from origin
4. Origin grants data + lease (e.g., "valid for 60s")
5. Cache serves data to client
6. Future requests: Cache serves from local copy (fast)
7. After 60s: Lease expires, cache must revalidate
\`\`\`

### **Example**

\`\`\`
T=0s:   Client requests /user/123
T=1s:   Cache misses, requests from DB
T=2s:   DB returns data + lease (expires at T=62s)
T=3s:   Cache returns data to client

T=10s:  Another client requests /user/123
T=11s:  Cache serves from memory (lease still valid) ✅

T=70s:  Client requests /user/123
T=71s:  Lease expired, cache requests from DB
T=72s:  DB returns updated data + new lease
\`\`\`

**Benefit**: Reduced database load, faster reads, guaranteed freshness.

### **Lease Invalidation**

When data changes, invalidate leases:

\`\`\`
1. Write to /user/123 (update)
2. DB invalidates all leases for /user/123
3. Caches must refetch on next access
\`\`\`

**Trade-off**: More complexity, but tighter consistency.

---

## Leases in Real-World Systems

### **Google Chubby**

Chubby is a lock service used across Google for distributed coordination.

**Lease-Based Locks**:
- Clients acquire locks with leases (default: 12 seconds)
- Clients renew leases (KeepAlive messages)
- If client fails to renew, lock released automatically
- Used for leader election, resource management, configuration storage

**Grace Period**:
- When lease expires, Chubby provides grace period
- Client can recover if it was just slow
- Prevents unnecessary failures

### **Apache ZooKeeper**

ZooKeeper uses **session leases** for client liveness.

\`\`\`
1. Client connects to ZooKeeper, establishes session
2. Session has timeout (e.g., 30 seconds)
3. Client sends heartbeats to keep session alive
4. If no heartbeat for timeout period:
   - Session expires
   - Ephemeral nodes deleted
   - Watchers notified
\`\`\`

**Leader Election with ZooKeeper**:
- Clients create ephemeral sequential nodes under /election
- Lowest sequence number is leader
- Leader holds "lease" via session
- If leader's session expires, ephemeral node deleted, next node becomes leader

### **etcd (Kubernetes)**

etcd provides lease primitives for coordination.

\`\`\`
1. Create lease: TTL=30s
2. Attach key-value pairs to lease
3. Send KeepAlive requests to renew lease
4. If lease expires: All keys attached to lease are deleted
\`\`\`

**Use Case**: Service discovery
\`\`\`
1. Service starts, creates lease
2. Registers in etcd with lease: /services/my-service → "IP:Port"
3. Sends KeepAlives while running
4. If service crashes: Lease expires, registration removed
5. Clients querying /services/my-service see service disappeared
\`\`\`

### **Cassandra**

Cassandra\'s coordinator nodes use leases for consistent reads.

**Read Repair with Leases**:
- Coordinator acquires short lease on partition
- Reads from multiple replicas
- Repairs inconsistencies
- Lease ensures no concurrent writes during repair

---

## Implementation Considerations

### **Lease Storage**

**In-Memory** (fast):
\`\`\`
lease_map = {
  "resource-1": {holder: "node-A", expires_at: T+30},
  "resource-2": {holder: "node-B", expires_at: T+25},
}
\`\`\`

**Pros**: Fast lookups, fast updates
**Cons**: Lost on coordinator crash (need recovery)

**Persistent** (durable):
\`\`\`
Store leases in database or replicated log.
\`\`\`

**Pros**: Survives coordinator crash
**Cons**: Slower, more complex

**Hybrid**: In-memory with periodic persistence and WAL for durability.

### **Lease Expiry Handling**

**Passive Expiry**:
\`\`\`
Check expiry when lease is accessed:
  if now() > lease.expires_at:
      delete_lease()
\`\`\`

**Pros**: Simple
**Cons**: Leases linger in memory until accessed

**Active Expiry**:
\`\`\`
Background thread checks and deletes expired leases:
  every 1 second:
      for lease in all_leases:
          if now() > lease.expires_at:
              delete_lease()
\`\`\`

**Pros**: Timely cleanup, accurate expiry
**Cons**: More CPU overhead

**Best**: Combine both (active expiry + check on access).

### **Handling Coordinator Failure**

**Problem**: Coordinator crashes, all leases lost.

**Solutions**:

**1. Replicate Coordinator**:
- Use leader election among coordinators
- Only leader grants/renews leases
- On leader failure, new leader takes over

**2. Replicate Lease State**:
- Use consensus (Raft, Paxos) for lease state
- Majority agreement for lease grants
- Survives minority failures

**3. Conservative Recovery**:
- New coordinator waits for all previous leases to expire
- Then starts granting new leases
- Ensures no overlap

---

## Common Pitfalls

### **1. Not Accounting for Clock Skew**

**Problem**: Node's clock is fast, thinks lease expired early.

**Solution**: Use coordinator's time, add safety margins.

### **2. Renewing Too Late**

**Problem**: Network delay causes renewal to arrive after expiration.

\`\`\`
Lease expires at T=30s.
Node renews at T=29.5s.
Network delay: 1s.
Renewal arrives at T=30.5s (too late).
\`\`\`

**Solution**: Renew at 50-80% of lease duration.

### **3. Acting After Lease Expiry**

**Problem**: Node continues to act on resource after its lease expires.

**Solution**: Node must track its own lease expiry and stop immediately.

### **4. No Lease Renewal Jitter**

**Problem**: All nodes renew at same time, coordinator overload.

**Solution**: Add random jitter to renewal time.

### **5. Ignoring Network Partitions**

**Problem**: Node partitioned from coordinator can't renew lease.

**Solution**: Node must stop acting on lease if it can't reach coordinator for renewal.

---

## Interview Tips

### **Key Concepts to Explain**1. **What is a lease**: Time-bound exclusive permission
2. **Why needed**: Automatic failure recovery, prevent split-brain
3. **How it works**: Grant, renew, expire
4. **Trade-offs**: Lease duration (short = fast recovery, long = less overhead)
5. **Clock skew**: Use coordinator's time, add safety margins

### **Common Interview Questions**

**Q: What\'s the difference between a lock and a lease?**
A: "A lock is indefinite—held until explicitly released. If a node holding a lock crashes without releasing it, the lock is stuck forever. A lease is time-bound—automatically expires after a duration. If the holder crashes, the lease expires, and another node can acquire it. Leases trade immediate availability for automatic failure recovery."

**Q: How do you choose the lease duration?**
A: "Trade-off between recovery speed and renewal overhead. Short leases (5-10s) provide fast failover but require frequent renewals. Long leases (60s+) reduce overhead but increase downtime after failures. Consider network reliability: unreliable networks need longer leases to avoid false failures. Typical: 30s for leader election, 60s for caching."

**Q: What happens if a node's clock is wrong?**
A: "Clock skew can cause issues. If a node's clock is fast, it may think its lease expired prematurely and stop working. If slow, it may continue after expiry. Solution: (1) Use coordinator's time for expiry, not node's time. (2) Add safety margins—renew at 50-80% of lease duration. (3) Synchronize clocks with NTP. (4) Use TTL (relative time) instead of absolute timestamps."

**Q: How do you prevent split-brain with leases?**
A: "Leases ensure at most one holder at a time. For leader election: Only one node holds the leader lease. If it fails to renew (crash or partition), the lease expires. New leader can't be elected until expiry. This creates a brief period with no leader (availability hit) but guarantees safety (no two leaders). The unavailability period equals lease duration."

---

## Summary

Leases are fundamental for distributed coordination with automatic failure recovery:

1. **Core Idea**: Time-bound exclusive permission that expires automatically
2. **Benefits**: Automatic recovery, prevent split-brain, optimized caching
3. **Mechanism**: Grant, renew, expire
4. **Duration**: Trade-off between recovery speed and overhead
5. **Clock Skew**: Use coordinator's time, add safety margins
6. **Real-World**: Chubby, ZooKeeper, etcd, Kubernetes
7. **Trade-offs**: Brief unavailability after failures vs automatic recovery

**Interview Focus**: Understand the problem leases solve (automatic failure recovery), how they prevent split-brain, trade-offs in lease duration, and clock skew handling.
`,
};
