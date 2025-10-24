/**
 * Leader Election Section
 */

export const leaderelectionSection = {
  id: 'leader-election',
  title: 'Leader Election',
  content: `Leader election is a fundamental pattern in distributed systems that determines which node acts as the coordinator or primary for a given resource or set of operations. This pattern ensures that despite failures, the system maintains progress and consistency.

## What is Leader Election?

Leader election is the process by which a group of nodes in a distributed system **agree on a single node** to take on special responsibilities. This elected leader typically:
- Coordinates writes to ensure consistency
- Makes decisions on behalf of the system
- Serves as a single source of truth
- Manages distributed locks or resources
- Handles critical operations requiring coordination

**Key Guarantee**: At any given time, there is **at most one leader** recognized by all nodes.

---

## Why Leader Election is Needed

### **1. Avoiding Conflicts**
Without a leader, multiple nodes might try to:
- Write to the same data simultaneously (race conditions)
- Make conflicting decisions
- Coordinate the same operation

**Example**: In a database, if two nodes both think they're the primary and accept writes, you get **split-brain** - leading to data inconsistency.

### **2. Simplifying Coordination**
Many distributed algorithms require coordination:
- Assigning work to workers
- Managing distributed transactions
- Coordinating replica updates
- Making system-wide decisions

Having a single leader simplifies these tasks tremendously.

### **3. Performance Optimization**
Leaders can:
- Cache frequently accessed data
- Make decisions without consensus overhead
- Optimize request routing

### **4. Failure Recovery**
When the leader fails:
- System detects the failure
- Automatically elects a new leader
- Operations resume with minimal downtime

This provides **high availability** without manual intervention.

---

## Real-World Use Cases

### **Distributed Databases**
- **MongoDB**: Primary node handles all writes, secondaries replicate
- **PostgreSQL**: In replication setup, primary accepts writes
- **MySQL**: Master-slave replication with single master

### **Distributed Coordination**
- **ZooKeeper**: Elects a leader among servers
- **etcd**: Leader handles write operations
- **Consul**: Leader coordinates cluster operations

### **Distributed Locking**
- **Redis Redlock**: Acquires locks across multiple instances
- **DynamoDB**: Leader coordinates distributed transactions

### **Job Schedulers**
- **Apache Spark**: Master node coordinates job execution
- **Hadoop**: ResourceManager (leader) manages cluster resources
- **Kubernetes**: Control plane components elect leader

---

## Leader Election Algorithms

### **1. Bully Algorithm**

Simple but effective algorithm where the node with the **highest ID wins**.

**How it works**:

1. **Detection**: A node detects the leader has failed
2. **Election**: It sends election message to all nodes with higher IDs
3. **Response**: Higher ID nodes respond and take over the election
4. **Victory**: If no higher ID responds, the node declares itself leader
5. **Announcement**: New leader broadcasts victory message

**Example**:
\`\`\`
Nodes: {1, 2, 3, 4, 5}, Leader: Node 5

1. Node 5 crashes
2. Node 3 detects failure, sends election to {4, 5}
3. Node 4 responds "I'll handle it"
4. Node 4 sends election to {5}
5. No response from 5
6. Node 4 declares itself leader
7. Node 4 broadcasts "I am the leader"
\`\`\`

**Advantages**:
- Simple to understand and implement
- Deterministic outcome
- Fast convergence

**Disadvantages**:
- High network traffic (O(n²) messages in worst case)
- Not suitable for large clusters
- Relies on node IDs (requires pre-configuration)

**When to use**: Small clusters (5-20 nodes) where deterministic selection is important.

---

### **2. Ring Algorithm**

Nodes are arranged in a logical ring, and election messages pass around the ring.

**How it works**:

1. **Detection**: Node detects leader failure
2. **Election**: Sends election message with its ID
3. **Propagation**: Each node adds its ID and forwards message
4. **Completion**: Message returns to initiator with all IDs
5. **Selection**: Node with highest ID becomes leader
6. **Announcement**: Coordinator message sent around ring

**Example**:
\`\`\`
Ring: 1 → 2 → 3 → 4 → 5 → 1

1. Node 3 detects leader failure
2. Sends election message with [3]
3. Node 4 receives, adds ID: [3, 4]
4. Node 5 receives, adds ID: [3, 4, 5]
5. Node 1 receives, adds ID: [3, 4, 5, 1]
6. Message returns to Node 3
7. Highest ID is 5, Node 5 is leader
\`\`\`

**Advantages**:
- Lower message complexity: O(n)
- No dependency on node IDs being pre-known

**Disadvantages**:
- Slow if ring is large
- Single point of failure (message loss)
- Ring topology must be maintained

**When to use**: Systems with natural ring topology or where message efficiency is important.

---

### **3. Paxos-Based Leader Election**

Used in production systems like **Chubby** (Google's lock service) and **ZooKeeper**.

**Concept**: Uses consensus algorithm (Paxos) to agree on the leader.

**Key Properties**:
- **Safety**: Never elects two leaders simultaneously
- **Liveness**: Eventually elects a leader (given enough time)
- **Fault tolerance**: Works as long as majority of nodes are available

**How it works** (simplified):

1. **Prepare Phase**: Proposer sends proposal with unique ID
2. **Promise Phase**: Acceptors promise not to accept lower IDs
3. **Accept Phase**: Proposer sends leader value
4. **Commit Phase**: Acceptors commit the value
5. **Learn Phase**: All nodes learn who the leader is

**Advantages**:
- Proven correctness (mathematically)
- Handles network partitions gracefully
- Ensures safety (no split-brain)

**Disadvantages**:
- Complex to implement correctly
- Multiple round trips (latency)
- Difficult to understand and debug

**When to use**: Production systems requiring strong consistency guarantees (ZooKeeper, etcd).

---

### **4. Raft Leader Election**

More understandable alternative to Paxos, used in **etcd**, **Consul**, **CockroachDB**.

**Concept**: Time divided into **terms**, each term has at most one leader.

**How it works**:

1. **Follower State**: All nodes start as followers
2. **Timeout**: If follower doesn't hear from leader, becomes **candidate**
3. **Request Votes**: Candidate requests votes from all nodes
4. **Vote**: Nodes vote for first candidate they hear from (per term)
5. **Majority**: Candidate with majority becomes leader
6. **Heartbeats**: Leader sends heartbeats to maintain authority

**Example**:
\`\`\`
Nodes: {A, B, C, D, E}, Current Leader: None

1. Node B timeout expires (randomized)
2. Node B becomes candidate for term 1
3. Node B requests votes from {A, C, D, E}
4. Nodes {A, C, D} vote for B (they haven't voted in term 1)
5. Node B has majority (4/5), becomes leader
6. Node B sends heartbeat to all nodes
7. Nodes update their term to 1, recognize B as leader
\`\`\`

**Key Features**:
- **Randomized timeouts**: Prevents split votes
- **Term numbers**: Ensures temporal consistency
- **Up-to-date logs**: Only candidates with latest data can win

**Advantages**:
- Easier to understand than Paxos
- Proven correct
- Efficient (few round trips)
- Handles network partitions

**Disadvantages**:
- Still complex to implement
- Requires majority for progress
- Not suitable for wide-area networks (high latency)

**When to use**: Production distributed systems requiring strong consistency and understandability (etcd, Consul).

---

### **5. ZooKeeper Leader Election**

ZooKeeper provides primitives for leader election using **ephemeral sequential nodes**.

**How it works**:

1. **Create Node**: Each candidate creates ephemeral sequential node (e.g., /election/n_0000000001)
2. **Get Children**: Get all nodes under /election
3. **Check Sequence**: If my node has lowest sequence number, I'm the leader
4. **Watch**: If not leader, watch the node right before mine
5. **Notification**: If watched node disappears, check if I'm now the leader

**Example**:
\`\`\`
Nodes: {A, B, C}

1. Node A creates /election/n_0000000001
2. Node B creates /election/n_0000000002
3. Node C creates /election/n_0000000003

4. Node A sees it has lowest number → Leader
5. Node B watches /election/n_0000000001
6. Node C watches /election/n_0000000002

7. Node A crashes (ephemeral node deleted)
8. Node B receives notification
9. Node B sees it has lowest number → New Leader
\`\`\`

**Advantages**:
- Simple to implement using ZooKeeper
- Automatic failure detection (ephemeral nodes)
- Fair (lowest sequence wins)
- No thundering herd (each node watches one specific node)

**Disadvantages**:
- Requires ZooKeeper infrastructure
- Single point of failure (ZooKeeper itself, though it's highly available)

**When to use**: Applications already using ZooKeeper for coordination.

---

## Split-Brain Problem

One of the most dangerous issues in leader election is **split-brain**: when **two nodes both think they're the leader**.

### **Causes**
- **Network partition**: Cluster splits into two groups
- **Slow heartbeats**: Nodes incorrectly detect leader failure
- **Clock skew**: Timeouts expire incorrectly

### **Consequences**
- **Data corruption**: Two leaders accept conflicting writes
- **Inconsistent state**: System state diverges
- **Lost updates**: Writes may be lost when partition heals

### **Solutions**

**1. Quorum-Based Election**
- Require **majority** to elect leader
- Only one partition can have majority
- If network splits 50/50, neither side can elect leader (system halts but stays consistent)

**Example**: 5 nodes, split into {3} and {2}
- Partition with 3 nodes: Can elect leader (3 > 5/2)
- Partition with 2 nodes: Cannot elect leader (2 < 5/2)

**2. Fencing Tokens**
- Leader gets monotonically increasing token
- All operations include token
- Resources reject operations from stale tokens

**Example**:
\`\`\`
1. Node A is leader with token 5
2. Network partition
3. Node B elected with token 6
4. Both A and B try to write to database
5. Database accepts B's write (token 6 > 5)
6. Database rejects A's write (token 5 < 6)
\`\`\`

**3. Witness Servers**
- Place tie-breaker node in different data center
- Provides odd number of voters
- Prevents 50/50 splits

---

## Leader Lease

To reduce heartbeat overhead, leaders acquire a **lease** (time-bound permission to act as leader).

**How it works**:
1. Leader elected and granted lease (e.g., 30 seconds)
2. During lease, no re-elections happen
3. Followers trust leader until lease expires
4. Leader must renew lease before expiration
5. If leader crashes, wait for lease to expire before re-election

**Advantages**:
- Reduces heartbeat frequency
- Lower network overhead
- Better for WAN (wide-area networks)

**Disadvantages**:
- Longer downtime if leader crashes (must wait for lease expiry)
- Clock synchronization required
- Complexity in handling lease renewal

**Trade-off**: Heartbeat frequency vs downtime during failure.

---

## Implementation Considerations

### **Failure Detection**
How do nodes know the leader has failed?

**1. Heartbeat Mechanism**
- Leader sends periodic heartbeat (e.g., every 150ms)
- Follower has timeout (e.g., 500ms)
- If no heartbeat within timeout, leader is considered dead

**2. Timeout Selection**
- Too short: False positives (premature elections)
- Too long: Longer downtime during actual failures
- Typically: 3-10x heartbeat interval

**3. Network Partitions**
- Must distinguish between leader crash vs network issue
- Quorum ensures only one partition can elect leader

### **Leadership Transition**
When leader changes, ensure smooth transition:

**1. Graceful Handoff**
- Old leader completes in-flight operations
- Transfers state to new leader
- Announces resignation

**2. Forced Takeover**
- New leader assumes control immediately
- May need to replay operations or reconcile state

### **Node Recovery**
When a failed node recovers:

**1. Discovery**
- Node queries cluster to find current leader
- Syncs its state with leader

**2. Integration**
- Becomes follower
- Starts participating in votes
- May become leader in future elections

---

## Common Pitfalls

### **1. False Failure Detection**
**Problem**: Network hiccup causes election even though leader is healthy.

**Solution**: 
- Use adaptive timeouts
- Require multiple missed heartbeats
- Exponential backoff for elections

### **2. Cascading Elections**
**Problem**: Elections trigger more elections, system unstable.

**Solution**:
- Randomized election timeouts (Raft approach)
- Backoff between election attempts
- Require supermajority for re-elections shortly after previous election

### **3. Forgotten Leader**
**Problem**: Old leader doesn't know it's been replaced, continues operating.

**Solution**:
- Use fencing tokens
- Leader must check with quorum before critical operations
- Ephemeral locks that expire

### **4. Unfair Elections**
**Problem**: Same node always becomes leader (e.g., lowest ID).

**Solution**:
- Use randomization (Raft)
- Consider node load and health in election
- Rotate leadership periodically

---

## Monitoring and Observability

### **Key Metrics**

**1. Leader Stability**
- Time since last election
- Number of elections per hour
- Average leadership duration

**High election frequency** → Network instability or configuration issues

**2. Election Duration**
- Time from leader failure to new leader elected
- P50, P99, P99.9 latencies

**Long elections** → Timeout too long or quorum issues

**3. Leadership Distribution**
- Which nodes become leader most often
- Distribution across cluster

**Skewed distribution** → Unfair election or node issues

### **Alerts**

- **Frequent elections**: More than 1 per minute
- **Long elections**: Takes >10 seconds
- **No leader**: Cluster without leader for >30 seconds
- **Split-brain detected**: Multiple leaders observed

---

## Interview Tips

### **What to Discuss**

**1. Start with Why**
- Explain why leader election is needed for the specific system
- What happens without coordination?

**2. Choose an Algorithm**
- Bully/Ring for small clusters
- Raft/Paxos for production systems
- ZooKeeper for applications already using it

**3. Handle Split-Brain**
- Always discuss quorum-based election
- Mention fencing tokens

**4. Failure Scenarios**
- Leader crashes
- Network partition
- Slow nodes

**5. Trade-offs**
- Consistency vs availability during partitions
- Heartbeat frequency vs false positives
- Election speed vs correctness

### **Common Interview Questions**

**Q: How does your system handle network partitions?**
A: "We use quorum-based election with majority requirement. If network splits into {3} and {2} partitions, only the 3-node partition can elect a leader. The 2-node partition cannot make progress, ensuring we don't have two leaders."

**Q: What happens if the leader crashes in the middle of a write operation?**
A: "The new leader will replay the write-ahead log or use a prepare-commit protocol. If the write wasn't committed, it's rolled back. Clients retry based on idempotency keys."

**Q: How do you prevent a slow leader from degrading performance?**
A: "Monitor leader performance. If leader is slow (e.g., P99 latency > threshold), trigger a controlled re-election. New leader is elected, old leader steps down gracefully."

---

## Real-World Examples

### **etcd (Used by Kubernetes)**
- Uses Raft consensus for leader election
- Leader handles all writes, followers replicate
- Automatic failover in seconds
- 3 or 5 node clusters common (odd for quorum)

**Configuration**:
\`\`\`
Election timeout: 1000ms
Heartbeat interval: 100ms
Quorum: Majority (e.g., 3 out of 5)
\`\`\`

### **MongoDB**
- Primary-secondary replication
- Uses modified Raft-like protocol
- Election triggered by:
  - Primary failure
  - Network partition
  - Manual stepdown (maintenance)
- Elections take 10-12 seconds typically

**Election Process**:
1. Secondary detects primary heartbeat missed
2. Initiates election
3. Nodes vote for candidate with most recent oplog
4. Candidate with majority becomes primary
5. Primary begins accepting writes

### **Apache Kafka** (Controller Election)
- One broker acts as controller (leader)
- Controller manages partition assignments
- Uses ZooKeeper for election (transitioning to KRaft)
- Automatic failover when controller fails

---

## Summary

Leader election is a critical building block for distributed systems. Key takeaways:

1. **Purpose**: Provide single coordinator to avoid conflicts and simplify coordination
2. **Algorithms**: Bully (simple), Ring (efficient), Raft (production-ready, understandable), Paxos (proven), ZooKeeper (convenient)
3. **Split-Brain Prevention**: Quorum-based election, fencing tokens, witness servers
4. **Trade-offs**: Consistency vs availability during partitions, heartbeat frequency vs false positives
5. **Production Systems**: etcd, ZooKeeper, MongoDB, Kafka all use leader election

**Interview Focus**: Understand the need for coordination, explain an algorithm (Raft recommended), handle failure scenarios, discuss split-brain prevention.
`,
};
