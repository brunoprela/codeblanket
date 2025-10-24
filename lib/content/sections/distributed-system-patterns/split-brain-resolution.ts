/**
 * Split-Brain Resolution Section
 */

export const splitbrainresolutionSection = {
  id: 'split-brain-resolution',
  title: 'Split-Brain Resolution',
  content: `Split-brain is one of the most dangerous scenarios in distributed systems, where the cluster divides into multiple partitions, each believing it's the only active group. This leads to data inconsistency, conflicting operations, and potential data loss. Understanding how to prevent and resolve split-brain is crucial for building reliable distributed systems.

## What is Split-Brain?

**Split-brain** occurs when a distributed system **divides into two or more independent partitions** that operate **independently**, each believing they are the only active partition.

**Analogy**: Like a person with split-brain syndrome—two halves of the brain operate independently, unaware of each other.

\`\`\`
Before Split-Brain:
  Cluster: [Node A (Leader), Node B, Node C]
  All nodes communicate, single leader

Network Partition:
  Partition 1: [Node A]
  Partition 2: [Node B, Node C]

Split-Brain Scenario:
  Partition 1: Node A thinks it's still leader
  Partition 2: Node B elected as new leader
  Both accept writes → Data diverges! ❌
\`\`\`

---

## Why Split-Brain is Dangerous

### **1. Data Inconsistency**

Two leaders accepting conflicting writes:

\`\`\`
T=0:  Account balance = $1000
T=1:  Network partition
T=2:  Partition 1: Withdraw $500 → Balance = $500
T=3:  Partition 2: Withdraw $600 → Balance = $400
T=4:  Network heals
T=5:  What is the balance? $500 or $400? ❌
\`\`\`

**Result**: Conflicting state, data corruption.

### **2. Duplicate Operations**

Both partitions perform same operation:

\`\`\`
Send email notification:
  Partition 1: Sends email to user
  Partition 2: Sends email to user
  Result: User receives duplicate emails ❌
\`\`\`

### **3. Resource Conflicts**

Both partitions manage same resource:

\`\`\`
Resource: Lock on critical section
  Partition 1: Node A holds lock
  Partition 2: Node B holds lock
  Both execute critical section simultaneously ❌
\`\`\`

### **4. Lost Updates**

After partition heals, one partition's work may be discarded:

\`\`\`
Partition 1: 100 transactions
Partition 2: 150 transactions
After healing: Only one partition's transactions kept
Result: Lost work ❌
\`\`\`

---

## Common Causes of Split-Brain

### **1. Network Partition**

\`\`\`
Data Center 1: Nodes A, B
Data Center 2: Nodes C, D, E

Network link between DCs fails:
  DC1 can't reach DC2
  DC2 can't reach DC1
  Each DC thinks the other is dead
\`\`\`

### **2. Network Asymmetry**

\`\`\`
Node A can send to Node B: ✓
Node B can send to Node A: ✗

Node B thinks A is dead (no responses)
Node A thinks B is alive (can send)
\`\`\`

### **3. Switch/Router Failure**

\`\`\`
  [Node A] ─┐
  [Node B] ─┤─── [Switch] ─┬─── [Node D]
  [Node C] ─┘              └─── [Node E]

Switch fails:
  {A, B, C} can't reach {D, E}
\`\`\`

### **4. Firewall Rule Changes**

\`\`\`
Admin accidentally adds firewall rule:
  Block traffic between Node A and Nodes {B, C, D}
  
Node A isolated, thinks it's the only survivor
\`\`\`

### **5. GC Pauses / Process Suspension**

\`\`\`
Node A: Long GC pause (10 seconds)
Other nodes: Can't reach A, think it's dead
Node A: Resumes, thinks it's still leader
Result: Two leaders ❌
\`\`\`

---

## Prevention Strategies

### **1. Quorum-Based Decisions**

**Principle**: Require **majority** for any operation that changes system state.

\`\`\`
Cluster of 5 nodes: {A, B, C, D, E}
Quorum = 3 (majority)

Network partition:
  Partition 1: {A, B, C} (3 nodes) → Has quorum ✓
  Partition 2: {D, E} (2 nodes) → No quorum ✗

Result:
  Partition 1 can elect leader, accept writes
  Partition 2 cannot elect leader, read-only or unavailable
\`\`\`

**Why It Works**: Only one partition can have majority.

**Trade-off**: Availability (minority partition becomes unavailable).

### **2. Fencing Tokens**

**Concept**: Leader gets a monotonically increasing token. Resources reject operations from stale tokens.

\`\`\`
T=0:  Node A elected leader, token = 1
T=10: Network partition
T=15: Node B elected leader, token = 2
T=20: Both A (token=1) and B (token=2) try to write

Database:
  Receives write from A (token=1) → Reject (old token)
  Receives write from B (token=2) → Accept (newer token)
\`\`\`

**Implementation**:
\`\`\`
Every operation includes fencing token:
  write(key, value, token)

Resource checks:
  if token < last_seen_token:
      reject_operation()
  else:
      last_seen_token = token
      perform_operation()
\`\`\`

**Real-World**: Google Chubby uses sequence numbers for fencing.

### **3. Witness / Tie-Breaker Node**

**Concept**: Add a third location with a lightweight witness node.

\`\`\`
Setup:
  DC1 (US-East): Nodes A, B
  DC2 (US-West): Nodes C, D
  DC3 (US-Central): Witness E

Network partition between DC1 and DC2:
  DC1: {A, B} + Witness E (can reach) = 3 nodes (quorum!)
  DC2: {C, D} (can't reach E) = 2 nodes (no quorum)

Result: Only DC1 can operate
\`\`\`

**Benefit**: Prevents 50/50 splits, ensures quorum is achievable.

**Witness Node**: Doesn't store data, only participates in quorum votes.

### **4. Lease-Based Leadership**

**Concept**: Leader must renew lease periodically. If partition prevents renewal, leadership expires.

\`\`\`
T=0:  Node A elected, lease expires at T=30
T=10: Network partition (A can't reach majority)
T=15: A tries to renew lease (fails, can't reach quorum)
T=20: A's lease expires, A stops acting as leader
T=25: Partition 2 elects Node B (new leader)
T=30: Even if partition heals, A won't act as leader (lease expired)
\`\`\`

**Safety**: At most one leader active at any time.

**Trade-off**: Brief period with no leader (lease expiration time).

### **5. STONITH (Shoot The Other Node In The Head)**

**Concept**: Forcibly power off nodes in minority partition.

\`\`\`
Cluster: {A, B, C, D, E}
Network partition: {A, B, C} vs {D, E}

Majority partition {A, B, C}:
  - Detects partition
  - Uses out-of-band management (IPMI, iLO) to power off D and E
  - Ensures D and E can't operate

Result: Only one partition can operate (forcibly)
\`\`\`

**Aggressive but Effective**: Guarantees no split-brain.

**Used in**: Highly available systems (Pacemaker, RHCS).

**Requirements**: Out-of-band management hardware.

---

## Detection Strategies

### **1. Heartbeat Monitoring**

Detect when nodes can't communicate:

\`\`\`
Node A sends heartbeat to {B, C, D, E}
Heartbeats to {B, C} succeed
Heartbeats to {D, E} fail

Node A detects potential partition:
  - Can reach some nodes, not others
  - Possible split-brain scenario
\`\`\`

### **2. Consensus Protocol State**

Monitor consensus state (Raft, Paxos):

\`\`\`
Raft:
  - Leader can't reach majority of followers
  - Leader steps down
  - Minority partition can't elect new leader

If two leaders exist:
  - Both think they have majority
  - Split-brain detected! ❌
\`\`\`

### **3. External Coordination Service**

Use external service to detect split:

\`\`\`
All nodes report to external ZooKeeper cluster:
  "I'm alive, my view: {A, B, C}"

ZooKeeper sees:
  {A, B} report: "Can't reach {C, D, E}"
  {C, D, E} report: "Can't reach {A, B}"

ZooKeeper detects split-brain, alerts operators
\`\`\`

### **4. Quorum Check**

Before critical operations, verify quorum:

\`\`\`
def execute_critical_operation():
    nodes_reachable = ping_all_nodes()
    if nodes_reachable < quorum:
        raise InsufficientQuorumError()
    
    // Proceed with operation
    perform_operation()
\`\`\`

**Benefit**: Fail safe if quorum is lost.

---

## Resolution Strategies

When split-brain is detected, how to resolve?

### **1. Manual Intervention**

**Concept**: Stop system, let operator decide.

\`\`\`
Detect split-brain:
  - Halt all operations
  - Alert operators
  - Operator manually inspects state
  - Operator chooses which partition's data to keep
  - Restart system with chosen state
\`\`\`

**Pros**: Safest (human decides)
**Cons**: Downtime, requires operator expertise

**Used when**: Data is critical, consistency is paramount (financial systems).

### **2. Last-Write-Wins (LWW)**

**Concept**: Use timestamps to decide which operation wins.

\`\`\`
Partition 1: Write X=5 at T=100
Partition 2: Write X=7 at T=150

After healing: X=7 (later timestamp wins)
\`\`\`

**Pros**: Simple, automatic resolution
**Cons**: Clock synchronization required, potential data loss

**Used in**: Eventually consistent systems (Dynamo, Riak).

### **3. Vector Clocks / Version Vectors**

**Concept**: Track causality to detect conflicts.

\`\`\`
Initial: X=1, vector={A:1, B:0, C:0}

Partition 1: Write X=5, vector={A:2, B:0, C:0}
Partition 2: Write X=7, vector={A:1, B:1, C:0}

After healing:
  - Vectors are concurrent (neither dominates)
  - Conflict detected
  - Keep both versions: X=[5, 7]
  - Application resolves conflict
\`\`\`

**Pros**: Preserves all data, detects conflicts accurately
**Cons**: Application must handle conflicts

**Used in**: Amazon Dynamo, Riak.

### **4. Application-Specific Merge**

**Concept**: Custom logic to merge conflicting states.

\`\`\`
Shopping cart:
  Partition 1: Add item A
  Partition 2: Add item B
  
Merge: Cart = {item A, item B} (union)
\`\`\`

**CRDTs**: Conflict-free Replicated Data Types provide automatic merge.

### **5. Pick Winner Based on Policy**

**Concept**: Policy decides which partition wins.

\`\`\`
Policy: "Partition with most nodes wins"
Partition 1: 3 nodes
Partition 2: 2 nodes
Result: Keep Partition 1's state, discard Partition 2's
\`\`\`

**Alternative Policies**:
- Partition with primary data center
- Partition with lowest node ID
- Partition that saw more operations

---

## Real-World Examples

### **MongoDB**

**Prevention**: Quorum-based elections

\`\`\`
Replica set: {Primary, Secondary1, Secondary2}
Majority required for: Elections, write acknowledgment

Network partition:
  {Primary, Secondary1}: Can elect primary (2/3 majority)
  {Secondary2}: Cannot elect (1/3, no majority)
\`\`\`

**Automatic**: If primary loses connection to majority, it steps down.

### **Cassandra**

**Detection**: Gossip protocol tracks node states

\`\`\`
Each node gossips with random nodes.
If partition, nodes in each partition gossip internally.
After heal: Gossip spreads across, nodes detect partition happened.
\`\`\`

**Resolution**: Read repair and anti-entropy resolve inconsistencies.

**Tunable Consistency**: Quorum reads/writes prevent split-brain data issues.

### **etcd / Raft**

**Prevention**: Leader election requires majority

\`\`\`
Cluster: {Node1, Node2, Node3, Node4, Node5}
Quorum: 3

Partition: {1, 2} vs {3, 4, 5}
{3, 4, 5}: Can elect leader (3/5 majority)
{1, 2}: Cannot elect (2/5, no majority)
\`\`\`

**Lease**: Leader has time-bound lease, must renew with majority.

**Safety**: Only one partition can have leader.

### **PostgreSQL Replication**

**Problem**: Traditional streaming replication is susceptible to split-brain.

**Solution**: Use tools like **repmgr** or **Patroni**:

\`\`\`
Patroni + etcd:
  - Use etcd (Raft-based) for leader election
  - PostgreSQL primary holds lease in etcd
  - If primary loses lease, it stops accepting writes
  - Secondary with lease becomes new primary
\`\`\`

**Prevents**: Two primaries accepting writes.

### **Kafka**

**Prevention**: Controller election via ZooKeeper (or KRaft)

\`\`\`
Controller (leader broker) manages cluster metadata.
Election requires ZooKeeper quorum (majority).

Network partition:
  Partition with ZooKeeper quorum: Can elect controller
  Partition without: Cannot elect controller
\`\`\`

**Safety**: Single controller at any time.

---

## Best Practices

### **1. Always Use Odd Number of Nodes**

\`\`\`
3 nodes: Quorum = 2, can survive 1 failure
4 nodes: Quorum = 3, can survive 1 failure (same as 3!)
5 nodes: Quorum = 3, can survive 2 failures
\`\`\`

**Reason**: Even numbers don't improve fault tolerance but increase cost.

### **2. Distribute Nodes Across Failure Domains**

\`\`\`
Bad: All nodes in same datacenter (DC failure = all down)

Good:
  Node1: US-East-1a
  Node2: US-East-1b
  Node3: US-West-1a
\`\`\`

**Benefit**: Increase availability across zone/region failures.

### **3. Monitor Quorum State**

\`\`\`
Alert: "Cluster does not have quorum"
Alert: "Multiple leaders detected"
Alert: "Node count below quorum threshold"
\`\`\`

**Critical**: Detect split-brain scenarios immediately.

### **4. Test Network Partitions**

\`\`\`
Chaos Engineering:
  - Simulate network partition (iptables, tc)
  - Verify system behavior
  - Ensure split-brain prevention works
  - Test recovery after partition heals
\`\`\`

**Tools**: Jepsen, Chaos Monkey, Blockade.

### **5. Implement Fencing**

For critical systems:
\`\`\`
- Use fencing tokens in every operation
- Reject operations from old tokens
- Consider STONITH for hardware-level fencing
\`\`\`

### **6. Document Recovery Procedures**

\`\`\`
Runbook: "Split-Brain Resolution"
  1. Detect split-brain (multiple leaders)
  2. Identify partition with majority
  3. Manually verify data in both partitions
  4. Choose winning partition
  5. Shut down losing partition
  6. Resync losing partition from winner
  7. Verify consistency
  8. Resume operations
\`\`\`

---

## Common Pitfalls

### **1. Not Using Quorum**

**Problem**: Even number of nodes, 50/50 split.

\`\`\`
4 nodes: {A, B} vs {C, D}
Both can elect leader ❌
\`\`\`

**Solution**: Use odd number or witness node.

### **2. Ignoring Clock Skew**

**Problem**: Last-write-wins doesn't work if clocks are wrong.

\`\`\`
Partition 1 (clock ahead): Write X=5 at T=200
Partition 2 (clock correct): Write X=7 at T=100

LWW chooses X=5 (later timestamp), but X=7 was actually later! ❌
\`\`\`

**Solution**: Use logical clocks (vector clocks) or require NTP sync.

### **3. Assuming Network Is Reliable**

**Problem**: "Network partitions are rare, we don't need protection."

**Reality**: Cloud networks, cross-datacenter links, misconfigurations—partitions happen.

**Solution**: Always design for partitions (CAP theorem).

### **4. Not Testing Partition Scenarios**

**Problem**: Split-brain prevention works in theory but fails in practice.

**Solution**: Regularly test with chaos engineering (Jepsen, Chaos Monkey).

### **5. Silent Split-Brain**

**Problem**: Split-brain occurs but no alerts.

\`\`\`
System continues operating in both partitions.
Data diverges.
Eventually discovered days later. ❌
\`\`\`

**Solution**: Proactive monitoring, assert invariants (exactly one leader).

---

## Interview Tips

### **Key Concepts to Explain**

1. **What is split-brain**: Cluster divides, both partitions think they're primary
2. **Why dangerous**: Data inconsistency, conflicting operations, lost updates
3. **Prevention**: Quorum, fencing tokens, leases, STONITH
4. **Detection**: Heartbeat monitoring, consensus state, quorum checks
5. **Resolution**: Manual, LWW, vector clocks, application merge

### **Common Interview Questions**

**Q: How do you prevent split-brain in a distributed system?**
A: "Use quorum-based decisions. Require majority (n/2 + 1) for leadership election and critical operations. For 5 nodes, quorum is 3. If network partitions into {3} and {2}, only the 3-node partition has quorum and can elect a leader. The 2-node partition cannot, ensuring only one active leader. Also use fencing tokens: each leader gets monotonically increasing token. Resources reject operations from old tokens, preventing stale leaders from causing issues."

**Q: What's the trade-off of using quorum to prevent split-brain?**
A: "Trade-off is availability. With quorum, the minority partition becomes unavailable—it can't elect a leader or accept writes. For 5 nodes split {3, 2}, the 2-node partition is down. This is CAP theorem in action: we choose consistency (no split-brain) over availability (minority down). Alternative approaches like multi-master allow both partitions to operate but accept eventual consistency and conflict resolution."

**Q: How would you detect if split-brain has occurred?**
A: "Monitor for: (1) Multiple leaders: Use monitoring to assert 'exactly one leader' invariant. (2) Consensus state: Check if multiple nodes think they're leader. (3) Quorum checks: Before critical operations, verify you can reach quorum. (4) Heartbeat patterns: If you can reach some nodes but not others, potential partition. (5) External coordination: Use external service (ZooKeeper) to get global view and detect inconsistencies. Alert immediately on any detection."

**Q: After a network partition heals, how do you resolve data inconsistencies?**
A: "Depends on consistency model: (1) Strong consistency (Raft, Paxos): Only one partition was active (had quorum), no inconsistency. (2) Eventual consistency (Dynamo): Use vector clocks to detect conflicts. If writes are concurrent, keep both versions and let application merge (e.g., shopping cart union). (3) Last-write-wins: Use timestamps, keep write with latest timestamp (requires synchronized clocks). (4) Manual: If critical, halt system, operator inspects and chooses correct state. (5) CRDTs: Automatic conflict-free merge."

---

## Summary

Split-brain resolution is critical for distributed system reliability:

1. **Problem**: Network partition causes multiple independent leaders, leading to data inconsistency
2. **Prevention**: Quorum (majority required), fencing tokens, leases, witness nodes, STONITH
3. **Detection**: Monitor for multiple leaders, heartbeat failures, consensus state inconsistencies
4. **Resolution**: Manual intervention, last-write-wins, vector clocks, application-specific merge
5. **Best Practices**: Odd number of nodes, distribute across failure domains, test partitions, monitor quorum
6. **Trade-offs**: Consistency vs availability (CAP theorem)
7. **Real-World**: MongoDB (quorum elections), etcd (Raft majority), Cassandra (tunable consistency)

**Interview Focus**: Understand the danger of split-brain (data inconsistency), prevention using quorum, trade-off with availability, and resolution strategies (vector clocks, LWW, manual).
`,
};
