/**
 * Consistency Models Section
 */

export const consistencymodelsSection = {
  id: 'consistency-models',
  title: 'Consistency Models',
  content: `Consistency models define the rules about the order and visibility of updates in distributed systems. Understanding these models is crucial for designing systems with appropriate consistency guarantees.

## What Are Consistency Models?

**Definition**: A consistency model specifies a contract between the distributed system and the application developer about how reads and writes behave, particularly regarding the order and timing of operations across replicas.

### **Why Multiple Models Exist**

Different consistency models represent different trade-offs:
- **Strong consistency**: Easier to reason about, slower, less available
- **Weak consistency**: Faster, more available, harder to reason about

---

## Consistency Model Spectrum

\`\`\`
Stronger Consistency (Slower, Less Available)
↓
Linearizability (Strongest)
↓
Sequential Consistency
↓
Causal Consistency
↓
Eventual Consistency
↓
Weakest Consistency (Fastest, Most Available)
\`\`\`

---

## 1. Linearizability (Strong Consistency)

**Definition**: Operations appear to occur instantaneously at some point between their invocation and completion. Once a write completes, all subsequent reads see that write or later writes.

**Also called**: Strong consistency, atomic consistency, external consistency

### **Guarantees**

\`\`\`
Timeline:
T1: Writer writes X=1 (completes at T1_end)
T2: Reader reads X (after T1_end) → Must see X=1 or later value

Key property: Real-time ordering preserved
\`\`\`

### **Example**

\`\`\`
Time: 10:00:00 - Alice writes balance = $100
Time: 10:00:01 - Write completes
Time: 10:00:02 - Bob reads balance → Guaranteed to see $100

Linearizability guarantee:
Any read after write completion sees the write or later
\`\`\`

### **Use Cases**

- **Banking systems**: Account balance must be immediately consistent
- **Inventory management**: Stock levels must be accurate
- **Leader election**: All nodes must agree on leader
- **Distributed locks**: Mutual exclusion requires strong consistency

### **Cost**

- Higher latency (must coordinate across nodes)
- Lower availability (can't serve requests during partition)
- Typical: 10-100ms latency for distributed systems

### **Systems Providing Linearizability**

- **Google Spanner**: Uses TrueTime for linearizability
- **etcd**: Raft consensus provides linearizability
- **ZooKeeper**: ZAB protocol provides linearizability
- **PostgreSQL single node**: ACID provides linearizability

---

## 2. Sequential Consistency

**Definition**: Operations from all processes appear to execute in some sequential order, and operations of each individual process appear in program order.

**Key Difference from Linearizability**: No real-time ordering requirement between different processes.

### **Guarantees**

\`\`\`
Process A: Write X=1, Write X=2
Process B: Read X

Sequential Consistency allows:
- B sees X=0, X=1, X=2 (respects A's order)

Does NOT guarantee:
- If B's read happens "after" A's write in real-time, B might still see old value
\`\`\`

### **Example**

\`\`\`
Process A writes: X=1 then X=2
Process B writes: Y=1 then Y=2

Sequential consistency allows:
Order 1: X=1, X=2, Y=1, Y=2 ✓
Order 2: X=1, Y=1, X=2, Y=2 ✓
Order 3: Y=1, X=1, Y=2, X=2 ✓

NOT allowed:
Order: X=2, X=1, Y=1, Y=2 ✗ (violates A's program order)
\`\`\`

### **Use Cases**

- **Multi-processor caches**: CPU cache coherence protocols
- **Less critical distributed systems**: Where exact real-time ordering not required

### **Cost**

- Moderate latency (less than linearizability)
- Moderate complexity

---

## 3. Causal Consistency

**Definition**: Operations that are causally related must be seen in the same order by all processes. Concurrent operations can be seen in different orders.

**Key Idea**: "Cause must precede effect"

### **Guarantees**

\`\`\`
If operation A causally influences operation B:
- All processes see A before B

If operations are concurrent (no causal relationship):
- Processes may see them in different orders
\`\`\`

### **Example: Social Media Post**

\`\`\`
Alice: Posts "I'm getting married!" (Event A)
Alice: Posts "Here's the wedding photo!" (Event B - caused by A)

Causal Consistency guarantees:
- All users see post A before post B
- Photo post must come after announcement

Bob: Posts "Congrats Alice!" (Event C - caused by A)

Causal Consistency guarantees:
- Bob's post C comes after seeing A
- But C and B are concurrent (no causal link)
- Different users may see B then C, or C then B
\`\`\`

### **Causal Dependencies**

\`\`\`
Causally related:
- Read-then-write: User reads value X, then writes Y based on X
- Write-then-read: User writes X, then reads own write
- Transitive: A→B and B→C implies A→C

NOT causally related (concurrent):
- Two users write independently without knowledge of each other
\`\`\`

### **Use Cases**

- **Social media feeds**: Comments after posts, replies after comments
- **Collaborative editing**: User edits must respect causal order
- **Chat applications**: Messages in conversation must preserve causality

### **Systems Providing Causal Consistency**

- **COPS (Clusters of Order-Preserving Servers)**
- **MongoDB with causal consistency sessions**
- **Some eventual consistency systems with version vectors**

### **Cost**

- Moderate latency
- More complex to implement (need to track causal dependencies)
- Better availability than strong consistency

---

## 4. Eventual Consistency

**Definition**: If no new updates are made, eventually all replicas will converge to the same value. No guarantees about when convergence occurs.

### **Guarantees**

\`\`\`
Time T1: Write X=1 to Replica A
Time T2: Read X from Replica B → Might see X=0 (stale)
Time T3: Read X from Replica B → Might see X=0 (still stale)
...
Time T_n: Eventually, read from Replica B → X=1

Guarantee: Eventually consistent (no time bound)
\`\`\`

### **Example: DNS**

\`\`\`
Time 10:00 - Update DNS record: example.com → 1.2.3.4
Time 10:01 - Query DNS server A → Gets old IP (cached)
Time 10:05 - Query DNS server B → Gets new IP (propagated)
Time 10:30 - Query DNS server A → Gets new IP (eventually updated)

Eventually consistent: All DNS servers converge, but takes time
\`\`\`

### **Variations of Eventual Consistency**

#### **Read-Your-Writes Consistency**

**Guarantee**: A process always sees its own writes.

\`\`\`
Alice writes X=1
Alice reads X → Guaranteed to see X=1 (her own write)
Bob reads X → Might see X=0 (stale, eventual)
\`\`\`

**Use case**: User updates profile, must see own changes immediately

**Implementation**: Route user's reads to same replica that handled write, or use session tokens

#### **Monotonic Read Consistency**

**Guarantee**: If a process reads value X, subsequent reads never return older values.

\`\`\`
Alice reads X=1 at time T1
Alice reads X at time T2 → Sees X=1 or X=2, never X=0

Prevents: Time travel (going backwards in time)
\`\`\`

**Use case**: Once user sees new data, don't show old data later

**Implementation**: Track version per client, only serve newer versions

#### **Monotonic Write Consistency**

**Guarantee**: Writes from a process are applied in order.

\`\`\`
Alice writes X=1, then X=2
All replicas see: X=0 → X=1 → X=2
Never see: X=0 → X=2 → X=1
\`\`\`

**Use case**: Sequential updates must preserve order

#### **Session Consistency**

**Guarantee**: Within a session, read-your-writes + monotonic reads.

\`\`\`
Session S1 (Alice):
- Alice writes X=1 → Sees X=1 in all subsequent reads
- Alice writes X=2 → Sees X=2 in all subsequent reads

Different session S2 (Bob):
- Bob might see X=0, X=1, or X=2 (eventual)
\`\`\`

**Use case**: User session needs consistency, but cross-user eventual consistency OK

**Implementation**: Session affinity (sticky sessions to same replica)

### **Use Cases for Eventual Consistency**

- **Social media feeds**: Likes, comments can be slightly delayed
- **Product catalogs**: Product descriptions rarely change
- **DNS**: DNS records propagate slowly
- **Shopping carts**: Can tolerate brief inconsistency
- **Analytics**: Aggregated metrics don't need real-time accuracy

### **Systems Using Eventual Consistency**

- **Cassandra** (default: CL=ONE)
- **DynamoDB** (default: eventually consistent reads)
- **Riak**
- **Couchbase**
- **DNS system**

### **Cost**

- Lowest latency
- Highest availability
- Application complexity (must handle stale data)

---

## Consistency Model Comparison

| Model | Ordering Guarantee | Real-Time | Latency | Availability | Complexity |
|-------|-------------------|-----------|---------|--------------|------------|
| **Linearizability** | Total order, real-time | Yes | High | Low | Low (easy to reason) |
| **Sequential** | Total order, no real-time | No | Medium | Medium | Medium |
| **Causal** | Causal order only | No | Low | High | High (track causality) |
| **Eventual** | No guarantees (eventual) | No | Lowest | Highest | Highest (app handles stale) |

---

## Choosing the Right Consistency Model

### **Strong Consistency (Linearizability) When:**

✅ **Correctness is critical**
- Banking (account balances)
- Inventory (prevent overselling)
- Auctions (bids must be ordered)

✅ **Coordination needed**
- Leader election
- Distributed locks
- Configuration management

**Trade-off**: Accept higher latency and lower availability for correctness

### **Eventual Consistency When:**

✅ **Availability is critical**
- Social media (feeds must always load)
- Content sites (articles must be viewable)
- E-commerce catalogs (product pages always accessible)

✅ **Slight staleness acceptable**
- Like counts
- Product reviews
- Analytics dashboards

**Trade-off**: Accept complexity of handling stale data for availability

### **Causal Consistency When:**

✅ **Causality matters, but not total order**
- Social media (comments after posts)
- Collaborative editing
- Chat applications

**Trade-off**: Balance between consistency and performance

---

## Real-World Example: Amazon's Shopping Cart

**Problem**: User adds items to cart, must see items immediately.

**Consistency Choice**: **Eventual consistency with session consistency**

**Why**:
- User must see their own cart updates (read-your-writes)
- Slight delay in cart sync across devices acceptable
- Availability critical (cart must always work)

**Implementation**:
- Session affinity (user's requests go to same replica)
- Conflict resolution (merge carts if diverge)
- Background sync for cross-device consistency

**Result**: Fast, available cart that works during network issues

---

## Interview Tips

### **Question**: "What consistency model would you use for a banking system?"

**Bad Answer**: "Eventual consistency because it's faster."

**Good Answer**: "Linearizability (strong consistency) because account balances must be immediately accurate. If Alice transfers $100 to Bob, Bob must see the $100 immediately, and Alice's balance must reflect the deduction. Eventual consistency could cause Alice to spend the same $100 twice. I'd use a system like PostgreSQL with ACID transactions or Google Spanner for distributed strong consistency. The higher latency (50-100ms) is acceptable because correctness is more important than speed for financial transactions."

### **Show Depth**

- Mention specific consistency levels (Cassandra QUORUM, DynamoDB strongly consistent reads)
- Discuss trade-offs explicitly (latency vs consistency)
- Relate to CAP/PACELC theorem
- Give real-world examples (Amazon DynamoDB eventual, Google Spanner linearizable)

---

## Key Takeaways

1. **Linearizability**: Strongest consistency, appears instantaneous, preserves real-time order
2. **Sequential Consistency**: Total order but no real-time guarantees
3. **Causal Consistency**: Only causally-related operations ordered, concurrent operations can differ
4. **Eventual Consistency**: Eventually converges, no timing guarantees, highest availability
5. **Variations**: Read-your-writes, monotonic reads, session consistency add useful guarantees
6. **Choose based on requirements**: Banking=strong, social media=eventual
7. **Trade-off spectrum**: Consistency ↔ Latency ↔ Availability
8. **Modern systems**: Often tunable consistency (Cassandra, DynamoDB)
9. **Application complexity**: Weaker consistency = more app logic to handle staleness
10. **In interviews**: Justify consistency choice based on use case requirements

---

## Summary

Consistency models define the contract between the system and application regarding order and visibility of updates. The choice ranges from **linearizability** (strongest, slowest, least available) to **eventual consistency** (weakest, fastest, most available). Most modern distributed databases offer tunable consistency, allowing you to choose per-operation. The key is matching the consistency model to your requirements: use strong consistency when correctness is critical (banking, inventory) and eventual consistency when availability is critical (social feeds, catalogs).`,
};
