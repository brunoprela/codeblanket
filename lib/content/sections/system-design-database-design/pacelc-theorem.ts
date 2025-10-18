/**
 * PACELC Theorem Section
 */

export const pacelctheoremSection = {
  id: 'pacelc-theorem',
  title: 'PACELC Theorem',
  content: `PACELC is an extension of CAP theorem that provides a more complete picture of distributed database trade-offs. While CAP focuses only on behavior during network partitions, PACELC addresses the trade-offs that exist even when the system is functioning normally.

## What is PACELC?

**Definition**: PACELC theorem states that in a distributed system:
- **P**artition: If there is a **P**artition, choose between **A**vailability and **C**onsistency
- **E**lse: **E**lse (when no partition), choose between **L**atency and **C**onsistency

**Formulated by**: Daniel Abadi (2012)

---

## Understanding the Extension

### **CAP Theorem's Limitation**

CAP only describes behavior **during network partitions** (which are rare events). But systems must make trade-off decisions **all the time**, not just during partitions.

**Question CAP doesn't answer:**
"When there's NO partition and the system is healthy, how should reads/writes behave?"

### **PACELC Fills the Gap**

PACELC adds the **"ELC"** part:
- **E**lse (when no partition exists)
- **L**atency vs **C**onsistency trade-off

Even without partitions, you must choose:
- **Low Latency**: Fast responses, might read slightly stale data (eventual consistency)
- **Consistency**: Always read latest data, but slower (must check all replicas)

---

## The Four PACELC Categories

### **PA/EL Systems (Availability + Latency)**

**During Partition**: Choose **Availability** over Consistency (AP)
**No Partition**: Choose **Latency** over Consistency (EL)

**Characteristics:**
- Always optimize for speed and availability
- Eventual consistency both during and after partitions
- Examples: Cassandra (CL=ONE), DynamoDB, Riak

**Example: Cassandra with Consistency Level ONE**

\`\`\`
During Partition (PA):
- Accept writes on any available node
- High availability, eventual consistency

No Partition (EL):
- Read from closest/fastest replica (ONE)
- Low latency, but might read stale data
- Eventual consistency
\`\`\`

**Use Cases:**
- Social media feeds (Instagram, Twitter)
- Product catalogs (Amazon)
- Activity logs
- Analytics data

---

### **PC/EC Systems (Consistency Always)**

**During Partition**: Choose **Consistency** over Availability (CP)
**No Partition**: Choose **Consistency** over Latency (EC)

**Characteristics:**
- Always prioritize strong consistency
- Sacrifice availability during partitions
- Sacrifice latency for consistency normally
- Examples: HBase, MongoDB (majority), BigTable

**Example: HBase**

\`\`\`
During Partition (PC):
- Reject writes if can't reach majority
- Strong consistency maintained

No Partition (EC):
- Wait for acknowledgment from multiple replicas
- Higher latency, but guaranteed consistency
\`\`\`

**Use Cases:**
- Financial systems (banking, payments)
- Inventory management
- Coordination services (ZooKeeper, etcd)

---

### **PA/EC Systems (Mixed Approach)**

**During Partition**: Choose **Availability** over Consistency (AP)
**No Partition**: Choose **Consistency** over Latency (EC)

**Characteristics:**
- Prioritize availability during rare partition events
- Prioritize consistency during normal operation
- Less common architecture
- Examples: Some configurations of MongoDB

**Example: MongoDB with Read Preference Primary**

\`\`\`
During Partition (PA):
- Secondary nodes can serve stale reads
- Availability prioritized

No Partition (EC):
- All reads go to primary
- Wait for replication before acknowledging writes
- Consistency prioritized, higher latency
\`\`\`

---

### **PC/EL Systems (Rare)**

**During Partition**: Choose **Consistency** over Availability (CP)
**No Partition**: Choose **Latency** over Consistency (EL)

**Characteristics:**
- Uncommon configuration
- Strict during partitions, relaxed normally
- Rarely used in practice

---

## Real-World Examples

### **Cassandra (PA/EL)**

\`\`\`
Partition Behavior (PA):
- Consistency Level: ONE
- Accepts writes even if some replicas down
- High availability
- Eventual consistency

Normal Behavior (EL):
- Reads from nearest replica
- Low latency (sub-millisecond)
- Might read stale data
- Eventual consistency
\`\`\`

**Why PA/EL for Cassandra:**
- Used by Netflix, Instagram for feeds
- Availability and speed critical
- Eventual consistency acceptable

**Trade-off:** Fast and always available, but data might be briefly inconsistent

---

### **HBase (PC/EC)**

\`\`\`
Partition Behavior (PC):
- Requires connection to HMaster
- Rejects operations if can't maintain consistency
- Strong consistency

Normal Behavior (EC):
- Reads must go to authoritative server
- Writes wait for WAL sync
- Higher latency (10-50ms)
- Strong consistency
\`\`\`

**Why PC/EC for HBase:**
- Used for systems requiring strong consistency
- Correctness more important than speed
- Based on Google BigTable design

**Trade-off:** Consistent and correct, but slower and less available

---

### **DynamoDB (PA/EL)**

\`\`\`
Partition Behavior (PA):
- Eventually consistent reads (default)
- Accepts writes to available replicas
- High availability

Normal Behavior (EL):
- Eventually consistent reads by default
- Low latency (<10ms)
- Optional strongly consistent reads (higher latency)

Configuration:
- Default: PA/EL (eventual consistency)
- Optional: PA/EC (strongly consistent reads, higher latency)
\`\`\`

**Why PA/EL for DynamoDB:**
- AWS needs high availability for services
- Most use cases tolerate eventual consistency
- Can opt-in to consistency when needed

---

### **Google Spanner (PC/EC)**

\`\`\`
Partition Behavior (PC):
- Requires Paxos quorum
- Rejects operations without majority
- Strong consistency

Normal Behavior (EC):
- External consistency (stricter than strong consistency)
- Uses TrueTime API for global ordering
- Higher latency (commit waits for 2 datacenters + TrueTime uncertainty)
- Strong consistency guaranteed

Latency Cost:
- Cross-region commit: 50-100ms (waits for Paxos quorum + TrueTime)
\`\`\`

**Why PC/EC for Spanner:**
- Google needs strong consistency for critical data (Ads billing, etc.)
- Willing to pay latency cost for correctness

**Trade-off:** Global strong consistency, but higher latency than eventual consistency systems

---

## PACELC Trade-off Comparison

| System | CAP | PACELC | Partition | Normal Operation | Use Case |
|--------|-----|--------|-----------|------------------|----------|
| **Cassandra** | AP | PA/EL | Available, eventual | Low latency, eventual | Social feeds, logs |
| **DynamoDB** | AP | PA/EL | Available, eventual | Low latency, eventual | AWS services, catalogs |
| **Riak** | AP | PA/EL | Available, eventual | Low latency, eventual | High availability needs |
| **HBase** | CP | PC/EC | Consistent, unavailable | Consistent, higher latency | Strong consistency needs |
| **MongoDB** | CP | PC/EC | Consistent, unavailable | Consistent, higher latency | General purpose |
| **Spanner** | CP | PC/EC | Consistent, unavailable | Consistent, high latency | Global consistency |

---

## Why PACELC Matters

### **CAP Only Tells Part of the Story**

**Example: Cassandra vs HBase**

Both can be configured as CP or AP during partitions, but:

**Cassandra (PA/EL):**
- Even without partition, prioritizes low latency over consistency
- Reads from nearest replica (might be stale)

**HBase (PC/EC):**
- Even without partition, prioritizes consistency over latency
- Reads must go to authoritative source (slower)

**PACELC explains the behavior difference even when both are healthy.**

---

## Practical Implications

### **For System Design:**

**1. Consider BOTH states:**
- **Partition behavior** (rare, but critical)
- **Normal behavior** (99.9% of the time)

**2. Normal operation matters more:**
- Partitions are rare (minutes per year)
- Normal operation is constant (millions of requests per day)
- The ELC trade-off affects performance daily

**3. Choose based on requirements:**
- Need speed? → PA/EL (Cassandra, DynamoDB)
- Need correctness? → PC/EC (HBase, Spanner)

---

### **Interview Example:**

**Question:** "Why does Cassandra have higher throughput than HBase?"

**Bad Answer:** "Because Cassandra is AP and HBase is CP."

**Good Answer:** "Cassandra is PA/EL - it optimizes for availability during partitions AND low latency during normal operation by reading from any replica and accepting eventual consistency. HBase is PC/EC - it requires consistency both during partitions and normal operation, meaning reads must go to the authoritative server and writes wait for replication, resulting in higher latency but stronger guarantees."

---

## Best Practices

### **✅ Understand Your True Requirements**

Ask:
1. **During partition** (rare): Can I sacrifice availability for consistency?
2. **During normal operation** (constant): Can I sacrifice latency for consistency?

Example answers:
- **Banking**: No (PC) + No (EC) → PC/EC system (PostgreSQL, Spanner)
- **Social feed**: Yes (PA) + Yes (EL) → PA/EL system (Cassandra, DynamoDB)

### **✅ Optimize for the Common Case**

- Partitions: Minutes per year
- Normal operation: 99.9% of uptime

**The ELC trade-off affects you every day, so choose wisely.**

### **✅ Consider Latency Budget**

**PA/EL systems (Cassandra, DynamoDB):**
- Read latency: 1-5ms (local replica)
- Write latency: 1-10ms (async replication)

**PC/EC systems (HBase, Spanner):**
- Read latency: 5-50ms (authoritative source)
- Write latency: 10-100ms (sync replication + quorum)

If you need <10ms latency, PA/EL is likely required.

---

## Key Takeaways

1. **PACELC extends CAP** by considering normal operation (no partition) behavior
2. **PA/EL systems** (Cassandra, DynamoDB) optimize for availability and low latency, accept eventual consistency
3. **PC/EC systems** (HBase, Spanner) optimize for strong consistency, accept higher latency and less availability
4. **Normal operation trade-off (ELC) matters more** than partition behavior (99.9% vs 0.1% of time)
5. **Choose based on requirements**: Speed-critical (PA/EL) vs Correctness-critical (PC/EC)
6. **PACELC explains why Cassandra is faster than HBase** even when both are healthy
7. **In interviews, show depth** by discussing both partition AND normal operation trade-offs

---

## Summary

PACELC provides a more complete framework than CAP by addressing the reality that trade-offs exist not just during partitions but during normal operation:

- **P**artition → **A**vailability vs **C**onsistency (CAP's domain)
- **E**lse (no partition) → **L**atency vs **C**onsistency (PACELC's addition)

This explains why Cassandra (PA/EL) is faster than MongoDB (PC/EC) even when both systems are healthy - Cassandra prioritizes low latency while MongoDB prioritizes consistency.`,
};
