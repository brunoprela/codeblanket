/**
 * CAP Theorem Deep Dive Section
 */

export const captheoremSection = {
  id: 'cap-theorem',
  title: 'CAP Theorem Deep Dive',
  content: `The CAP Theorem is one of the most fundamental concepts in distributed systems. It explains the inherent trade-offs when designing distributed databases and why certain consistency guarantees cannot coexist with availability during network partitions.

## What is CAP Theorem?

**Definition**: In a distributed system, you can only guarantee **two out of three** properties simultaneously:
- **C**onsistency: All nodes see the same data at the same time
- **A**vailability: Every request receives a response (success or failure)
- **P**artition Tolerance: The system continues to operate despite network partitions

### **Formulated by Eric Brewer (2000)**
Computer scientist Eric Brewer proposed CAP theorem, proving that distributed systems must make trade-offs between consistency and availability when network partitions occur.

---

## The Three Properties Explained

### **Consistency (C)**

**Definition**: Every read receives the most recent write or an error. All nodes see the same data at the same time.

**Example**:
\`\`\`
User writes: balance = $100
Immediately after, any read from any node returns: balance = $100
        \`\`\`

**Strong Consistency Behavior:**
\`\`\`
Time 0: User writes balance = $100 to Node A
Time 1: System replicates to Nodes B, C, D
Time 2: User reads from any node → Gets $100(consistent)
        \`\`\`

**Not Consistent (Eventual Consistency):**
\`\`\`
Time 0: User writes balance = $100 to Node A
Time 1: User reads from Node B → Gets $90(stale data)
Time 2: Replication completes
Time 3: User reads from Node B → Gets $100(now consistent)
        \`\`\`

### **Availability (A)**

**Definition**: Every request receives a non-error response, without guarantee that it contains the most recent write.

**Key Points:**
- System always responds (doesn't hang or timeout)
- Response might be stale data (not most recent)
- No request fails due to non-responding nodes

**Example:**
\`\`\`
    Request: GET / balance
    Response: 200 OK, balance = $90(might be stale, but you get a response)
        \`\`\`

**Unavailable System:**
\`\`\`
    Request: GET / balance
    Response: 504 Gateway Timeout (system couldn't respond)
        \`\`\`

### **Partition Tolerance (P)**

**Definition**: The system continues to function despite network partitions (communication breakdowns between nodes).

**What is a Network Partition?**
When nodes in a distributed system cannot communicate with each other due to network failures.

**Example:**
\`\`\`
     Data Center 1 | NETWORK PARTITION | Data Center 2
┌─────────────────┐         |                     |    ┌─────────────────┐
│   Node A, B     │    ✗    | Can't communicate  |    │   Node C, D     │
│(West Coast)  │    ✗    |                     |    │   (East Coast)  │
└─────────────────┘         |                     |    └─────────────────┘
    \`\`\`

**In Real World**: Network partitions happen due to:
- Router failures
- Fiber cuts
- Datacenter network issues
- DNS failures
- Misconfigured firewalls

**Key Insight**: In distributed systems, **partition tolerance is not optional**. Networks are unreliable, partitions will happen. Therefore, the real choice is between **Consistency** and **Availability** during a partition.

---

## Why You Can Only Choose 2

### **The Fundamental Trade-off**

When a network partition occurs, you must choose:

**Option 1: Prioritize Consistency (CP)**
- **Reject writes/reads** that can't be replicated to all nodes
- System becomes **unavailable** during partition
- Data remains consistent across accessible nodes

**Option 2: Prioritize Availability (AP)**
- **Accept writes/reads** even if nodes are partitioned
- System remains **available** during partition
- Data becomes **inconsistent** across nodes (will be reconciled later)

### **Why You Can't Have All Three**

**Scenario:** Database with 2 nodes across 2 data centers

\`\`\`
Initial state:
Node A: balance = $100
Node B: balance = $100
        \`\`\`

**Network partition occurs** (A and B can't communicate)

**User writes to Node A:** balance = $150

**Now another user reads from Node B: What should it return?**

**Choice 1: Return $100 (stale data)**
- ✅ Available: System responds
- ❌ Not Consistent: Returns old data
- **This is AP (Availability + Partition Tolerance)**

**Choice 2: Reject the read (return error)**
- ❌ Not Available: System doesn't provide data
- ✅ Consistent: Won't return stale data
- **This is CP (Consistency + Partition Tolerance)**

**You cannot:**
- Return $150 ← Node B doesn't have it (partition)
- Return $100 and claim consistency ← That\'s stale data
- Return $150 without the network ← Impossible

---

## CP Systems (Consistency + Partition Tolerance)

**Trade-off**: Sacrifice **Availability** to maintain **Consistency**

**Behavior During Partition:**
- **Reject writes** that can't be replicated to majority of nodes
- **Reject reads** from nodes that might have stale data
- System becomes **unavailable** in the minority partition

### **CP System Examples**

#### **1. HBase**
- **Use Case**: BigTable-style wide-column store
- **Behavior**: If HBase RegionServer is partitioned from master, it stops serving requests
- **Rationale**: Prevents returning stale data

#### **2. MongoDB (with majority read/write concern)**
- **Use Case**: Document database
- **Behavior**: Writes require acknowledgment from majority of replica set nodes
- **During Partition**: If majority not available, writes are rejected
- **Rationale**: Ensures data is replicated before acknowledging write

#### **3. Redis (single instance)**
- **Use Case**: In-memory cache/database
- **Behavior**: Single-node, no partition issues
- **Note**: Redis Cluster with majority quorum is CP

#### **4. ZooKeeper / etcd / Consul**
- **Use Case**: Coordination services, configuration management
- **Behavior**: Require majority quorum for writes
- **During Partition**: Minority partition becomes read-only or unavailable
- **Rationale**: Coordination requires consistency (can't have split-brain)

### **When to Choose CP**

✅ **Banking and Financial Systems**
- Consistency is non-negotiable
- Better to reject transaction than have inconsistent balances

✅ **Inventory Management**
- Can't oversell products
- Better to show "out of stock" than sell items you don't have

✅ **Configuration Management**
- Distributed systems need consistent configuration
- Better to pause than have inconsistent config

✅ **Distributed Locks**
- Only one process should hold lock
- Consistency critical for correctness

---

## AP Systems (Availability + Partition Tolerance)

**Trade-off**: Sacrifice **Consistency** to maintain **Availability**

**Behavior During Partition:**
- **Accept writes** to any available node
- **Accept reads** from any available node (might return stale data)
- System remains **available** in all partitions
- Data becomes temporarily inconsistent, will **eventually converge**

### **AP System Examples**

#### **1. Cassandra**
- **Use Case**: Wide-column store for massive scale
- **Behavior**: Accepts writes even if some replicas are down
- **Consistency Level**: Configurable (ONE, QUORUM, ALL)
  - \`ONE\` = AP (availability prioritized)
  - \`ALL\` = CP (consistency prioritized)
- **Instagram/Netflix Use Case**: Feed data, where availability matters more than immediate consistency

#### **2. DynamoDB**
- **Use Case**: AWS managed NoSQL database
- **Behavior**: Eventually consistent reads by default
- **During Partition**: Remains available, accepts reads/writes
- **Rationale**: High availability for services like Amazon.com

#### **3. Riak**
- **Use Case**: Distributed key-value store
- **Behavior**: Uses **eventual consistency** with vector clocks for conflict resolution
- **During Partition**: Both sides accept writes, conflicts resolved later

#### **4. Couchbase**
- **Use Case**: Document database
- **Behavior**: Cross-datacenter replication with eventual consistency
- **During Partition**: Each datacenter operates independently

#### **5. DNS (Domain Name System)**
- **Use Case**: Global name resolution
- **Behavior**: Cached data, eventual consistency across nameservers
- **During Partition**: Nameservers continue serving cached/stale data
- **Rationale**: Availability critical for internet infrastructure

### **When to Choose AP**

✅ **Social Media Feeds (Twitter, Instagram)**
- User doesn't care if feed is 1 second stale
- Better to show slightly stale feed than no feed

✅ **Product Catalogs (Amazon, eBay)**
- Product descriptions rarely change
- Better to show product page than error

✅ **Analytics and Logging**
- Slight delays in data aggregation acceptable
- Better to collect data than lose it

✅ **Collaboration Tools (Google Docs)**
- Users can work during network issues
- Conflicts resolved when reconnected

✅ **Shopping Carts**
- Better to let user add items (might be stale inventory) than block them
- Can validate inventory at checkout

---

## CA Systems (Consistency + Availability)

**Can you have CA?**

**In theory**: Yes, if no network partitions occur.

**In practice**: **CA systems don't exist in distributed systems** because:
- Networks are unreliable; partitions **will** happen
- To be distributed (multiple nodes), you must tolerate partitions
- Therefore, **P is mandatory** in distributed systems

### **"CA" Systems (Actually Single-Node Systems)**

#### **Traditional RDBMS (PostgreSQL, MySQL) - Single Node**
- **Consistency**: ACID transactions
- **Availability**: Always available (no partitions, single node)
- **Not Partition Tolerant**: Because there's only one node

**Key Insight**: Once you replicate (master-slave, multi-master), you're distributed and must choose between CP or AP during partitions.

---

## Real-World Examples

### **Example 1: Banking System (CP Choice)**

**Scenario**: Money transfer between accounts

**Requirement**: **Strong consistency** - balance must always be accurate

**Architecture**: PostgreSQL with synchronous replication

**During Partition:**
\`\`\`
Master database in DC1
Replica database in DC2
Network partition occurs

User tries to transfer $100:
    - Master in DC1 can't reach replica in DC2
        - System rejects transaction (unavailable)
            - Better to reject than risk inconsistent balances
                \`\`\`

**Result**: **CP System** - Sacrificed availability to maintain consistency

---

### **Example 2: Instagram Feed (AP Choice)**

**Scenario**: User posts a photo to Instagram

**Requirement**: **High availability** - user should always be able to post

**Architecture**: Cassandra with replication factor 3

**During Partition:**
\`\`\`
    3 Cassandra nodes: A, B, C
Node C is partitioned from A and B

User posts photo to Node A:
    - Node A accepts write
        - Replicates to Node B(success)
            - Can't reach Node C (partition)
                - Write succeeds with 2 / 3 replicas

    User in different region reads from Node C:
    - Doesn't see new photo yet (stale data)
        - After partition heals, Node C catches up
            \`\`\`

**Result**: **AP System** - Sacrificed consistency (eventual) to maintain availability

---

### **Example 3: Amazon Product Catalog (AP Choice)**

**Scenario**: User views product page

**Requirement**: **Availability** - product page must always load

**Architecture**: DynamoDB with eventual consistency

**During Partition:**
\`\`\`
Product price updated: $50 → $45(sale)
Update propagating across regions

    User in Europe:
    - Reads from European replica
        - Sees $50(stale price for 1 - 2 seconds)
        - Eventually sees $45

Better than:
    - "Product page unavailable"(CP approach)
        \`\`\`

**Result**: **AP System** - Slightly stale data acceptable for availability

---

## CAP Theorem in Practice

### **Most Systems Choose AP**

**Why?**
- **Availability is critical** for user experience
- Users expect systems to always work
- Slight staleness is acceptable in most use cases

**Examples**: Social media, e-commerce, content sites

### **When to Choose CP**

- **Correctness over availability**: Financial transactions, inventory
- **Coordination**: Leader election, distributed locks
- **Metadata systems**: Configuration management

---

## Misconceptions About CAP

### **❌ Misconception 1: "You choose 2 forever"**
**Reality**: You can configure consistency levels **per query**.

**Example**: Cassandra
\`\`\`java
    // Strong consistency (CP)
    session.execute (query, ConsistencyLevel.QUORUM);

    // High availability (AP)
    session.execute (query, ConsistencyLevel.ONE);
    \`\`\`

### **❌ Misconception 2: "AP systems are always inconsistent"**
**Reality**: **Eventual consistency** means data *will* converge. Often happens in milliseconds.

### **❌ Misconception 3: "CP systems are always unavailable"**
**Reality**: Unavailable only during network partitions (rare). 99.9% of the time, they're available.

### **❌ Misconception 4: "NoSQL = AP, SQL = CP"**
**Reality**: 
- **MySQL with async replication** = AP (might read from stale replica)
- **MongoDB with majority write** = CP (requires quorum)
- **Cassandra with QUORUM read/write** = CP (during partition)

---

## Trade-offs Summary

| System | Type | Consistency | Availability During Partition | Use Case |
|--------|------|-------------|-------------------------------|----------|
| **HBase** | CP | Strong | Minority partition unavailable | BigTable workloads |
| **MongoDB** | CP | Strong (with majority) | Unavailable without majority | General purpose |
| **PostgreSQL** | CP | Strong (ACID) | Unavailable if master down | Transactional systems |
| **Cassandra** | AP | Eventual (configurable) | Always available | High-scale writes |
| **DynamoDB** | AP | Eventual (default) | Always available | AWS applications |
| **Riak** | AP | Eventual | Always available | High availability |

---

## Best Practices

### **✅ Understand Your Requirements**

Ask:
- Can I tolerate stale data? → **AP**
- Must data always be current? → **CP**
- What happens if system is unavailable for 30 seconds? → Determines choice

### **✅ Design for Partition Scenarios**

- **Simulate network partitions** in testing (chaos engineering)
- **Define behavior** when partition occurs
- **Monitor partition events** in production

### **✅ Use Tunable Consistency (Cassandra/DynamoDB)**

- **Critical writes** (payments): Use QUORUM or ALL → CP
- **Non-critical reads** (feed): Use ONE → AP
- **Balance**: Most queries use QUORUM

### **✅ Accept Trade-offs**

- **No perfect solution**: Every choice has drawbacks
- **Different consistency for different data**: User profile = CP, feed = AP
- **Hybrid architectures**: Use multiple databases

---

## Interview Tips

### **How to Discuss CAP:**

#### **1. Explain the Trade-off**
❌ "Cassandra is AP"
✅ "Cassandra is typically AP because it prioritizes availability and accepts eventual consistency, but you can configure QUORUM reads/writes for CP behavior during partitions."

#### **2. Relate to Requirements**
"For a banking system, I'd choose CP (PostgreSQL with sync replication) because balance correctness is more important than availability during the rare partition event."

#### **3. Show Nuance**
"CAP is not binary. Systems like Cassandra let you tune consistency per query. Critical writes can use QUORUM (CP) while reads use ONE (AP)."

#### **4. Real-World Examples**
- **Instagram**: AP (Cassandra) because feed staleness is acceptable
- **Stripe**: CP (PostgreSQL) because payment accuracy is critical

#### **5. Ask Clarifying Questions**
- "What are the consistency requirements?"
- "Can we tolerate eventual consistency?"
- "What's the impact if the system is unavailable for 1 minute?"

---

## Key Takeaways

1. **CAP Theorem**: You can only have 2 of 3 (Consistency, Availability, Partition Tolerance) during a network partition
2. **Partition Tolerance is mandatory** in distributed systems (networks fail)
3. **Real choice**: Consistency (CP) vs Availability (AP) during partitions
4. **CP systems**: Sacrifice availability for consistency (HBase, MongoDB, ZooKeeper)
5. **AP systems**: Sacrifice strong consistency for availability (Cassandra, DynamoDB, Riak)
6. **Most systems choose AP**: Availability critical for user experience, eventual consistency acceptable
7. **Tunable consistency**: Cassandra, DynamoDB allow per-query consistency configuration
8. **Context matters**: Banking = CP, Social media = AP
9. **CAP applies only during partitions**: Rest of the time, systems can have both C and A
10. **Design for partitions**: Simulate, define behavior, monitor

---

## Summary

CAP Theorem forces a fundamental trade-off in distributed database design: **Consistency or Availability during network partitions**. Understanding this trade-off is critical for:
- Choosing the right database (PostgreSQL vs Cassandra vs DynamoDB)
- Configuring consistency levels (QUORUM vs ONE)
- Setting correct expectations (eventual consistency acceptable?)

In system design interviews, demonstrate understanding by:
- Explaining the trade-off clearly
- Relating choice to specific requirements
- Providing real-world examples
- Showing awareness of tunable consistency

Most modern systems choose **AP (eventual consistency)** because availability is critical for user experience and slight staleness is acceptable. **CP systems** are chosen when correctness is more important than availability (banking, inventory, coordination).`,
};
