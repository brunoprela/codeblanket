/**
 * Heartbeat Section
 */

export const heartbeatSection = {
  id: 'heartbeat',
  title: 'Heartbeat',
  content: `Heartbeat is a simple yet critical pattern in distributed systems for detecting node failures and monitoring system health. It\'s the foundation for automatic failover, leader election, and cluster membership management.

## What is a Heartbeat?

A **heartbeat** is a **periodic signal** sent by a node to indicate it is alive and functioning.

**Analogy**: Like a medical heartbeat—as long as it beats, the patient is alive. When it stops, something is wrong.

\`\`\`
Node A → Node B: "I'm alive" (every 1 second)

If Node B doesn't receive heartbeat within timeout (e.g., 5 seconds):
  → Node B considers Node A dead
\`\`\`

**Key Characteristics**:
- **Periodic**: Sent at regular intervals (e.g., every 500ms, 1s, 5s)
- **Lightweight**: Minimal payload (often just timestamp or counter)
- **Bidirectional or Unidirectional**: A→B, or A↔B

---

## Why Heartbeats Are Needed

### **1. Failure Detection**

**Problem**: How do you know if a node has crashed or is just slow?

Without Heartbeat:
\`\`\`
T=0s:  Node A is processing requests
T=10s: Node A crashes
T=60s: Other nodes still think A is alive
T=120s: Requests still being routed to A (failing)
\`\`\`

With Heartbeat:
\`\`\`
T=0s:  Node A sends heartbeat every 1s
T=10s: Node A crashes, heartbeats stop
T=15s: Node B detects A missed 5 heartbeats (5s timeout)
T=16s: Node B marks A as dead, stops routing to A ✅
\`\`\`

**Result**: Fast failure detection, automatic recovery.

### **2. Leader Election**

Heartbeats are used to maintain leadership:

\`\`\`
Leader: Sends heartbeats to followers
Followers: Receive heartbeats → Leader is alive
If follower misses heartbeats → Trigger election
\`\`\`

**Raft Example**: Leader sends heartbeat every 150ms. Follower election timeout: 1500ms.

### **3. Cluster Membership**

Heartbeats track which nodes are part of the cluster:

\`\`\`
Node joins cluster:
  - Starts sending heartbeats
  - Other nodes add it to membership list

Node leaves/crashes:
  - Heartbeats stop
  - Other nodes remove it from membership
\`\`\`

### **4. Resource Leasing**

Heartbeats can act as lease renewals:

\`\`\`
Node A holds lease on resource X
Node A sends heartbeat = lease renewal
If heartbeats stop → Lease expires
\`\`\`

### **5. Load Balancer Health Checks**

Load balancers use heartbeats to route traffic:

\`\`\`
Load Balancer → Backend Server: Health check (heartbeat)
If server responds → Route traffic
If server doesn't respond → Remove from pool
\`\`\`

---

## How Heartbeats Work

### **Basic Protocol**

**Sender (Node A)**:
\`\`\`
Every heartbeat_interval:
    send_heartbeat (to=Node B, payload={
        node_id: "A",
        timestamp: current_time(),
        sequence: counter++
    })
\`\`\`

**Receiver (Node B)**:
\`\`\`
On receive heartbeat from Node A:
    last_heartbeat_time[A] = current_time()
    update_node_status(A, "alive")

Background thread:
    Every check_interval:
        for each node in cluster:
            time_since_last_heartbeat = current_time() - last_heartbeat_time[node]
            if time_since_last_heartbeat > timeout_threshold:
                mark_node_dead (node)
                trigger_failure_handling (node)
\`\`\`

### **Heartbeat Message**

Minimal payload:

\`\`\`json
{
  "node_id": "node-A",
  "timestamp": 1704067200000,
  "sequence": 1234
}
\`\`\`

Enhanced payload (more information):

\`\`\`json
{
  "node_id": "node-A",
  "timestamp": 1704067200000,
  "sequence": 1234,
  "load": 0.65,
  "free_memory_mb": 2048,
  "active_connections": 150,
  "version": "1.2.3"
}
\`\`\`

**Trade-off**: Larger payload provides more information but increases network overhead.

---

## Heartbeat Timing Parameters

### **Heartbeat Interval**

How often to send heartbeat.

**Short Interval** (100-500ms):
- **Pros**: Fast failure detection, quick response
- **Cons**: High network overhead, more CPU usage

**Long Interval** (5-10s):
- **Pros**: Low overhead, less network traffic
- **Cons**: Slow failure detection

**Typical Values**:
- Leader election: 150ms - 1s
- Cluster membership: 1s - 5s
- Health checks: 1s - 30s

**Rule of Thumb**: heartbeat_interval × 3 to 10 = timeout_threshold

### **Timeout Threshold**

How long to wait before declaring node dead.

**Short Timeout** (1-3 heartbeats):
- **Pros**: Fast failure detection
- **Cons**: False positives (transient network issues)

**Long Timeout** (5-10 heartbeats):
- **Pros**: Tolerant of transient issues
- **Cons**: Slow failure detection

**Example**:
\`\`\`
Heartbeat interval: 1s
Timeout threshold: 5s (5 missed heartbeats)

T=0s:  Last heartbeat received
T=1s:  Expected heartbeat (missed)
T=2s:  Expected heartbeat (missed)
T=3s:  Expected heartbeat (missed)
T=4s:  Expected heartbeat (missed)
T=5s:  Expected heartbeat (missed)
T=5s:  Declare node dead (5s elapsed, 5 missed)
\`\`\`

### **Check Interval**

How often receiver checks for missed heartbeats.

\`\`\`
Background thread runs every check_interval:
    check_all_nodes_for_missed_heartbeats()
\`\`\`

**Typical Value**: heartbeat_interval / 2 to heartbeat_interval

**Example**:
- Heartbeat interval: 1s
- Check interval: 500ms

**Trade-off**: Faster checks = more accurate but more CPU usage.

---

## Heartbeat Patterns

### **1. All-to-All Heartbeat**

Every node sends heartbeat to every other node.

\`\`\`
Cluster: {A, B, C, D}

A sends to: B, C, D
B sends to: A, C, D
C sends to: A, B, D
D sends to: A, B, C
\`\`\`

**Pros**: 
- Simple to implement
- Fast failure detection
- No single point of failure

**Cons**:
- Network overhead: O(n²) messages per interval
- Doesn't scale to large clusters (>100 nodes)

**Use When**: Small clusters (5-50 nodes), critical systems.

### **2. Leader-Based Heartbeat**

Followers send heartbeat to leader, leader monitors all.

\`\`\`
Leader: Node A
Followers: B, C, D

B → A: heartbeat
C → A: heartbeat
D → A: heartbeat

A detects failures, coordinates responses.
\`\`\`

**Pros**:
- Centralized monitoring
- O(n) messages per interval
- Easier to implement consistent failure handling

**Cons**:
- Single point of failure (leader)
- Leader can become bottleneck

**Use When**: Leader-follower architectures (databases, Raft).

**Enhancement**: Leader also sends heartbeats to followers (bidirectional).

### **3. Ring-Based Heartbeat**

Nodes arranged in a ring, each monitors next node.

\`\`\`
A → B → C → D → A

A monitors B
B monitors C
C monitors D
D monitors A
\`\`\`

**Pros**:
- O(n) messages per interval
- Scales well
- Distributed monitoring

**Cons**:
- Slow failure detection (must propagate around ring)
- Single failure can disrupt ring
- Complex to maintain ring topology

**Use When**: Large clusters, uniform monitoring sufficient.

### **4. Gossip-Based Heartbeat**

Each node randomly chooses peers to exchange heartbeat information.

\`\`\`
Every interval:
    Node A picks random nodes {B, C}
    A sends heartbeat info about all nodes it knows
    B, C update their view of cluster
\`\`\`

**Pros**:
- Highly scalable (O(log n) propagation)
- Fault-tolerant (no single point of failure)
- Self-healing (information spreads even with failures)

**Cons**:
- Eventual consistency (slower to detect failures)
- More complex to implement

**Use When**: Large clusters (100s-1000s of nodes), eventual consistency acceptable.

**Example Systems**: Cassandra, Consul, SWIM protocol.

---

## Failure Detection Strategies

### **1. Fixed Timeout**

\`\`\`
if current_time - last_heartbeat > timeout:
    mark_dead()
\`\`\`

**Simple but inflexible**: Doesn't adapt to network conditions.

### **2. Adaptive Timeout**

Adjust timeout based on observed network latency.

\`\`\`
timeout = mean_heartbeat_interval + (4 × stddev_heartbeat_interval)
\`\`\`

**Pros**: Adapts to network conditions, fewer false positives
**Cons**: More complex, requires maintaining statistics

**Phi Accrual Failure Detector** (Cassandra): More sophisticated adaptive approach.

### **3. Multiple Missed Heartbeats**

Require multiple consecutive missed heartbeats before declaring failure.

\`\`\`
missed_count = 0

On heartbeat received:
    missed_count = 0

On heartbeat expected but not received:
    missed_count++
    if missed_count >= threshold:
        mark_dead()
\`\`\`

**Pros**: Tolerant of packet loss
**Cons**: Slower detection

**Typical Threshold**: 3-5 missed heartbeats

### **4. Quorum-Based Detection**

Require multiple nodes to agree that a node is dead.

\`\`\`
Node B thinks A is dead.
Node C thinks A is dead.
Node D thinks A is alive.

Majority (2/3) say dead → Mark A as dead.
\`\`\`

**Pros**: Reduces false positives from network partitions
**Cons**: Slower detection, more coordination overhead

---

## Heartbeat in Real-World Systems

### **Apache Kafka**

**Controller Heartbeat**:
- Kafka controller (leader broker) manages cluster metadata
- Controller sends heartbeat to ZooKeeper (or KRaft)
- If heartbeat stops, controller loses leadership
- New controller elected

**Broker Heartbeat**:
- Brokers send heartbeat to controller
- Controller tracks live brokers
- Routes requests to live brokers only

**Configuration**:
\`\`\`
session.timeout.ms = 10000         (10 seconds)
heartbeat.interval.ms = 3000       (3 seconds)
\`\`\`

### **Kubernetes**

**Node Heartbeat**:
- Kubelet (on each node) sends heartbeat to API server
- Heartbeat includes node status (Ready, DiskPressure, MemoryPressure)
- If no heartbeat for 40s, node marked NotReady
- Pods on NotReady node rescheduled to other nodes

**Pod Health Checks**:
- Liveness probe: Is pod alive?
- Readiness probe: Is pod ready to serve traffic?
- Startup probe: Has pod started successfully?

**Configuration**:
\`\`\`yaml
livenessProbe:
  httpGet:
    path: /healthz
    port: 8080
  initialDelaySeconds: 10
  periodSeconds: 5          # Heartbeat interval
  timeoutSeconds: 1
  failureThreshold: 3       # Missed heartbeats before restart
\`\`\`

### **Cassandra**

**Gossip Protocol with Phi Accrual**:
- Nodes exchange gossip messages (heartbeat + metadata)
- Phi accrual failure detector calculates suspicion level
- High phi value → Likely dead
- Phi > 8: Mark node down (configurable)

**Configuration**:
\`\`\`
phi_convict_threshold: 8              (Suspicion level for failure)
gossip_interval: 1000ms               (Heartbeat frequency)
\`\`\`

### **etcd (Raft)**

**Leader Heartbeat**:
- Leader sends heartbeat to followers (AppendEntries with no data)
- Heartbeat interval: 100ms (typical)
- Follower election timeout: 1000-5000ms (randomized)
- If follower doesn't receive heartbeat, triggers election

**Lease-Based Heartbeat**:
- Leader has lease on leadership
- Heartbeats act as lease renewal
- If heartbeats stop, lease expires

### **Redis Sentinel**

**Sentinel Monitoring**:
- Sentinels monitor Redis master and replicas
- Send PING commands (heartbeat) every 1s
- If no PONG within timeout, mark as subjectively down (SDOWN)
- Query other sentinels for confirmation
- If quorum agrees, mark as objectively down (ODOWN)
- Trigger failover

**Configuration**:
\`\`\`
sentinel down-after-milliseconds mymaster 5000
sentinel parallel-syncs mymaster 1
sentinel failover-timeout mymaster 60000
\`\`\`

---

## Handling Network Partitions

**Scenario**: Network partition splits cluster.

\`\`\`
Cluster: {A, B, C, D, E}
Network partition:
  Partition 1: {A, B, C}
  Partition 2: {D, E}

From Partition 1 perspective:
  - A, B, C exchange heartbeats (alive)
  - D, E don't send heartbeats (dead?)

From Partition 2 perspective:
  - D, E exchange heartbeats (alive)
  - A, B, C don't send heartbeats (dead?)
\`\`\`

**Problem**: Each partition thinks the other is dead (split-brain risk).

**Solutions**:

### **1. Quorum-Based Decisions**

Require majority to make decisions:

\`\`\`
Partition 1: 3 nodes (majority of 5) → Can elect leader
Partition 2: 2 nodes (minority) → Cannot elect leader
\`\`\`

**Result**: Only one partition can make progress (consistency over availability).

### **2. Fencing / STONITH**

"Shoot The Other Node In The Head"

\`\`\`
Partition 1 detects Partition 2 is unreachable.
Partition 1 forcibly powers off nodes in Partition 2.
Prevents Partition 2 from operating.
\`\`\`

**Aggressive but effective** in preventing split-brain.

### **3. Witness / Tie-Breaker Node**

Place a witness node in a third location:

\`\`\`
Locations: DC1 (A, B), DC2 (C, D), DC3 (Witness E)

If DC1 and DC2 partitioned:
  - DC1 can reach E → Has majority (3/5)
  - DC2 cannot reach E → Minority (2/5)
\`\`\`

**Result**: Prevents 50/50 splits.

---

## Implementation Considerations

### **Heartbeat Transport**

**UDP**:
- Fast, low overhead
- Connectionless (no TCP handshake)
- Tolerant of packet loss (okay for heartbeats)
- **Used by**: Cassandra gossip, SWIM

**TCP**:
- Reliable delivery
- Connection-oriented (detects connection failures)
- Higher overhead
- **Used by**: etcd, Kubernetes

**HTTP/gRPC**:
- Easy to implement and debug
- Works through firewalls
- Higher latency
- **Used by**: Load balancer health checks, Kubernetes probes

### **Heartbeat Thread vs Event Loop**

**Dedicated Thread**:
\`\`\`
Thread {
  while (true) {
    send_heartbeat();
    sleep (interval);
  }
}
\`\`\`

**Pros**: Simple, predictable
**Cons**: Resource overhead (one thread per heartbeat destination)

**Event Loop / Timer**:
\`\`\`
EventLoop.schedule (interval) {
  send_heartbeat();
}
\`\`\`

**Pros**: Efficient (no thread per destination)
**Cons**: May be delayed if event loop is busy

**Best Practice**: Use event loop, but monitor for delays.

### **Monitoring Heartbeats**

**Key Metrics**:

1. **Heartbeat Latency**: Time between heartbeat send and receive
   - P50, P99, P99.9
   - Spikes indicate network issues

2. **Missed Heartbeats**: Number of missed heartbeats per node
   - High count: Network problems or slow nodes

3. **False Failure Rate**: Nodes marked dead but actually alive
   - High rate: Timeout too short

4. **Failure Detection Time**: Time from actual failure to detection
   - Long time: Timeout too long

5. **Heartbeat Message Size**: Size of heartbeat payload
   - Growing size: Too much information in heartbeat

**Alerts**:
- Heartbeat latency P99 > 1s (network issues)
- False failure rate > 1% (timeout too aggressive)
- Multiple nodes marked dead simultaneously (network partition?)

---

## Common Pitfalls

### **1. Timeout Too Short**

**Problem**: False positives from transient network delays.

\`\`\`
Timeout: 500ms
Network hiccup: 1s
Result: Node marked dead incorrectly, unnecessary failover ❌
\`\`\`

**Solution**: Timeout = 3-10× heartbeat interval.

### **2. Heartbeat Payload Too Large**

**Problem**: Heartbeat becomes expensive, defeats purpose.

\`\`\`
Heartbeat includes:
  - All node metadata
  - Full configuration
  - Recent logs
Size: 10KB+
Result: Network congestion, CPU overhead ❌
\`\`\`

**Solution**: Keep heartbeat minimal (<1KB), send full info separately.

### **3. Heartbeat Thread Blocked**

**Problem**: Long-running operation blocks heartbeat thread.

\`\`\`
Heartbeat thread:
  send_heartbeat()
  perform_expensive_operation()  // Blocks for 10s
  send_heartbeat()  // Next heartbeat delayed 10s!
\`\`\`

**Solution**: Dedicated heartbeat thread, don't mix with other work.

### **4. Not Handling Clock Skew**

**Problem**: Nodes have different clocks, timeouts incorrect.

\`\`\`
Node A clock: 12:00:00
Node B clock: 12:00:30 (30s ahead)

Node A sends heartbeat with timestamp 12:00:00
Node B checks: current_time (12:00:30) - 12:00:00 = 30s → Dead ❌
\`\`\`

**Solution**: Use time since last heartbeat (relative time) not absolute timestamps.

### **5. Symmetric Failure**

**Problem**: Both nodes think the other is dead.

\`\`\`
Network partition between A and B.
A thinks B is dead.
B thinks A is dead.
Both try to take over same resource. ❌
\`\`\`

**Solution**: Quorum-based decisions, fencing, lease-based coordination.

---

## Interview Tips

### **Key Concepts to Explain**

1. **What is heartbeat**: Periodic signal indicating liveness
2. **Why needed**: Failure detection, leader election, health monitoring
3. **How it works**: Send periodically, check for timeouts
4. **Timing**: Heartbeat interval vs timeout threshold (3-10× interval)
5. **Patterns**: All-to-all, leader-based, ring-based, gossip-based

### **Common Interview Questions**

**Q: How do you choose the heartbeat interval and timeout?**
A: "Trade-off between detection speed and false positives. Fast detection needs short interval (100ms-1s) and short timeout (3-5× interval). But short timeout increases false positives from transient network delays. For critical systems (leader election), use 100-500ms interval, 3-10× timeout. For health checks, 1-5s interval is common. Also consider network reliability: unreliable network needs longer timeout to avoid false positives."

**Q: What happens during a network partition?**
A: "Nodes in each partition stop receiving heartbeats from the other partition. They think the other nodes are dead. To prevent split-brain, use quorum: only partition with majority can elect a leader or make decisions. Minority partition can't make progress. This ensures consistency but sacrifices availability for the minority partition."

**Q: Why not just use TCP keepalive instead of application-level heartbeats?**
A: "TCP keepalive detects connection failures (network, OS crash) but not application-level issues. If application is stuck in infinite loop or deadlocked, TCP connection is alive but application can't serve requests. Application-level heartbeats ensure the application logic is functioning, not just the network connection. Also, TCP keepalive has long default timeouts (hours) and isn't customizable per connection."

**Q: How do you reduce false positives in failure detection?**
A: "(1) Adaptive timeouts: Adjust based on observed latency (Phi accrual). (2) Multiple missed heartbeats: Require 3-5 consecutive misses. (3) Quorum-based detection: Multiple nodes must agree on failure. (4) Bidirectional heartbeats: Check both directions. (5) Monitor heartbeat latency: If latency increases, adjust timeout dynamically."

---

## Summary

Heartbeat is a foundational pattern for distributed system health and failure detection:

1. **Core Idea**: Periodic signal indicating liveness
2. **Use Cases**: Failure detection, leader election, health monitoring, cluster membership
3. **Timing**: Heartbeat interval (how often), timeout threshold (when to declare dead)
4. **Patterns**: All-to-all, leader-based, ring-based, gossip-based
5. **Challenges**: Network partitions, false positives, clock skew
6. **Real-World**: Kafka (broker monitoring), Kubernetes (node heartbeat), Cassandra (gossip), etcd (Raft)
7. **Trade-offs**: Detection speed vs false positives, network overhead vs precision

**Interview Focus**: Understand timing parameters (interval vs timeout), different patterns, handling network partitions, and trade-offs in detection speed vs accuracy.
`,
};
