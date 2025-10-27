/**
 * Database Replication Section
 */

export const databasereplicationSection = {
  id: 'database-replication',
  title: 'Database Replication',
  content: `Database replication copies data from one database to another to improve availability, fault tolerance, and read scalability.

## What is Database Replication?

**Definition**: The process of copying and maintaining database data in multiple database instances to ensure data availability, reliability, and performance.

### **Why Replicate Databases?**

**Without Replication:**
- Single point of failure (database crashes = entire app down)
- Limited read capacity (single database can't handle high read traffic)
- No disaster recovery
- Maintenance requires downtime

**With Replication:**
- High availability (if primary fails, replica takes over)
- Read scalability (distribute reads across replicas)
- Disaster recovery (data backup in different locations)
- Zero-downtime maintenance (update replicas one at a time)

**Real-world**: Facebook has thousands of MySQL replicas to handle billions of reads per day.

---

## Primary-Replica (Master-Slave) Architecture

**Most common replication pattern.**

### **Architecture**

**Components:**
- **Primary (Master)**: Accepts writes, source of truth
- **Replicas (Slaves)**: Receive data from primary, handle reads

**Data flow:**1. Application writes to Primary
2. Primary logs changes
3. Changes replicated to Replicas
4. Application reads from Replicas

**Example:**
- Primary: Handles 10K writes/sec
- Replica 1: Handles 50K reads/sec
- Replica 2: Handles 50K reads/sec
- Replica 3: Handles 50K reads/sec
- Total read capacity: 150K reads/sec

---

## Synchronous vs Asynchronous Replication

### **Synchronous Replication**

**How it works**: Write confirmed only after data written to BOTH primary and replica (s).

**Flow:**1. App writes to Primary
2. Primary writes to disk
3. Primary sends data to Replica
4. Replica writes to disk
5. Replica acknowledges to Primary
6. Primary confirms write to App

**Pros:**
- **Strong consistency**: Replica always has latest data
- **No data loss**: If primary fails immediately after write, data exists on replica
- **Guaranteed durability**

**Cons:**
- **Slower writes**: Must wait for replica acknowledgment (network latency)
- **Availability risk**: If replica down, writes fail or block
- **Geographic limitations**: High latency if replica in different region

**When to use**: Banking, financial transactions (data loss unacceptable).

---

### **Asynchronous Replication**

**How it works**: Write confirmed after data written to primary, replica updated later.

**Flow:**1. App writes to Primary
2. Primary writes to disk
3. Primary confirms write to App immediately
4. Primary asynchronously sends data to Replica (in background)
5. Replica writes to disk eventually

**Pros:**
- **Fast writes**: No wait for replica
- **High availability**: Primary can write even if replica down
- **Works across regions**: No latency penalty

**Cons:**
- **Eventual consistency**: Replica may lag behind primary (replication lag)
- **Data loss risk**: If primary fails before replicating, recent writes lost
- **Stale reads**: Reading from replica may return old data

**When to use**: Most web applications (social media, e-commerce), where slight delay acceptable.

**Replication lag**: Time between write to primary and appearance on replica.
- Typical: 0-5 seconds
- High load: Up to 60 seconds or more

---

### **Semi-Synchronous Replication**

**Compromise between sync and async.**

**How it works**: Write confirmed after at least ONE replica acknowledges, others updated asynchronously.

**Flow:**1. App writes to Primary
2. Primary writes to disk
3. Primary sends to all replicas
4. Wait for ONE replica to acknowledge
5. Confirm write to App
6. Other replicas update asynchronously

**Pros:**
- Faster than fully synchronous
- Better data durability than async (at least 1 replica has data)
- Balance of performance and safety

**When to use**: Production systems needing durability without full sync penalty.

---

## Multi-Master Replication

**Multiple databases accept writes simultaneously.**

### **Architecture**

**Setup:**
- Multiple primary nodes (no single primary)
- Each node can accept writes
- Changes replicate to other nodes

**Example:**
- Primary 1 (US): Handles US writes
- Primary 2 (EU): Handles EU writes
- Both replicate to each other

**Pros:**
- **Write scalability**: Distribute writes across multiple nodes
- **No single point of failure**: Any node can accept writes
- **Geographic distribution**: Users write to nearest node (low latency)

**Cons:**
- **Conflict resolution**: Two users update same data on different nodes → conflict
- **Complexity**: Much harder to implement and maintain
- **Consistency challenges**: Eventual consistency, not immediate

---

### **Conflict Resolution**

**Problem**: User A updates record on Primary 1, User B updates same record on Primary 2 simultaneously.

**Strategies:**

**1. Last Write Wins (LWW)**
- Use timestamp to determine which write is newer
- Discard older write
- **Problem**: Clocks may not be synchronized, data loss possible

**2. Version Vectors**
- Track version history of each record
- Merge conflicting updates
- **Example**: CouchDB uses this approach

**3. Application-Level Resolution**
- Store both versions, let application decide
- **Example**: Shopping cart (merge both carts)

**4. Conflict-Free Replicated Data Types (CRDTs)**
- Data structures designed to merge automatically
- **Example**: Counters, sets, maps

---

## Read Replicas and Read Scaling

**Purpose**: Offload read traffic from primary to replicas.

### **Configuration**

**Application routing:**
- Writes → Primary
- Reads → Replicas (round-robin or load balanced)

**Example:**
\`\`\`
Write Path:
App → Primary DB

Read Path:
App → Load Balancer → Replica 1, 2, 3, ...
\`\`\`

**Scaling reads:**
- 1 primary + 0 replicas: 10K reads/sec
- 1 primary + 5 replicas: 50K reads/sec
- 1 primary + 10 replicas: 100K reads/sec

**Linear read scaling** (add replicas → more read capacity).

---

### **Handling Replication Lag**

**Problem**: User writes data, immediately reads from replica, data not yet replicated → appears data is lost.

**Solutions:**

**1. Read Your Own Writes**
- After user writes, route their reads to primary for short time (e.g., 5 seconds)
- Then route to replicas
- Ensures user sees their own changes immediately

**2. Sticky Sessions**
- Route user's requests to same replica consistently
- Replica eventually catches up
- User experiences consistency (even if stale)

**3. Monitor Replication Lag**
- Track lag: \`primary_log_position - replica_log_position\`
- If lag > threshold, don't route reads to that replica
- Alert if lag consistently high

**4. Causal Consistency**
- Use version numbers or timestamps
- Read from replica only if version >= version of last write

---

## Failover and Promotion

**Scenario**: Primary database crashes. How to maintain availability?

### **Automatic Failover**

**Process:**1. **Detection**: Monitor detects primary is down (heartbeat timeout)
2. **Election**: Choose which replica to promote (typically most up-to-date)
3. **Promotion**: Promote replica to new primary
4. **Reconfiguration**: Update application to write to new primary
5. **Recovery**: When old primary recovers, it becomes a replica

**Challenges:**

**1. Split-Brain Problem**
- Network partition isolates primary
- System thinks primary is down, promotes replica
- Now two primaries (both accepting writes)
- **Solution**: Use consensus algorithm (Raft, Paxos) or fencing

**2. Data Loss**
- If using async replication, promoted replica may not have latest writes
- **Solution**: Accept data loss or use semi-sync replication

**3. Failover Time**
- Detection: 10-30 seconds
- Promotion: 10-60 seconds
- Total downtime: 30-90 seconds
- **Solution**: Use automated tools (Orchestrator, ProxySQL)

---

### **Manual Failover**

**Process:**1. Administrator manually promotes replica
2. Update DNS or load balancer configuration
3. Restart application with new primary connection

**When to use:**
- Planned maintenance
- Upgrading database version
- When automatic failover risky

**Downtime**: Minutes to hours (depending on planning).

---

## Real-World Replication Examples

### **MySQL Replication**

**Configuration:**
\`\`\`
Primary → Replica1, Replica2, Replica3
    \`\`\`

**Replication methods:**
- **Statement-based**: Replicates SQL statements
- **Row-based**: Replicates actual data changes (more reliable)
- **Mixed**: Hybrid approach

**Lag monitoring:**
- \`SHOW SLAVE STATUS\` → check \`Seconds_Behind_Master\`

---

### **PostgreSQL Streaming Replication**

**Configuration:**
- Primary streams WAL (Write-Ahead Log) to replicas
- Replicas replay WAL to stay in sync

**Modes:**
- **Asynchronous**: Default, fast
- **Synchronous**: Wait for replica acknowledgment
- **Quorum-based**: Wait for N of M replicas

**Hot Standby**: Replicas accept read queries while replicating.

---

### **MongoDB Replica Sets**

**Configuration:**
- 3+ nodes: Primary + Secondaries
- Automatic failover via election (Raft-like consensus)

**Example:**
- 3-node replica set
- Primary accepts writes
- If primary fails, secondaries elect new primary (takes ~10 seconds)

**Read preferences:**
- Primary: Always read from primary (strong consistency)
- PrimaryPreferred: Primary if available, secondary otherwise
- Secondary: Always read from secondary (stale reads possible)
- Nearest: Lowest latency node

---

## Replication Topologies

### **1. Simple Primary-Replica**

\`\`\`
Primary → Replica1
       → Replica2
       → Replica3
    \`\`\`

**Pros**: Simple, easy to understand
**Cons**: Primary is bottleneck for replication

---

### **2. Cascading Replication**

\`\`\`
Primary → Replica1 → Replica2
                   → Replica3
    \`\`\`

**Pros**: Reduces replication load on primary
**Cons**: Increased replication lag for downstream replicas

---

### **3. Circular Replication (Multi-Master)**

\`\`\`
Primary1 ↔ Primary2 ↔ Primary3
    \`\`\`

**Pros**: Multi-region writes
**Cons**: Conflict resolution complexity

---

## Interview Tips

### **Common Questions:**

**Q: "How would you scale reads for a database receiving 100K reads/sec?"**

✅ Good answer: "Add read replicas:
1. Current: 1 primary handling all 100K reads/sec (overloaded)
2. Add 10 read replicas: Each handles 10K reads/sec
3. Load balancer distributes reads across replicas
4. Writes still go to primary
5. Monitor replication lag to ensure data not too stale"

**Q: "What happens if the primary database fails?"**

✅ Good answer: "Failover process:
1. Monitor detects primary down (heartbeat failure)
2. Promote most up-to-date replica to new primary
3. Reconfigure app to write to new primary
4. Total downtime: 30-90 seconds with auto-failover
5. Risk: If async replication, may lose recent writes (last few seconds)
6. Mitigation: Use semi-sync replication for critical data"

**Q: "How do you handle replication lag?"**

✅ Good answer: "Several strategies:
1. Monitor lag metrics (alert if >5 seconds)
2. Read-your-own-writes: Route user's reads to primary after they write
3. Remove lagging replicas from load balancer
4. Use semi-sync replication to reduce lag
5. Accept eventual consistency for non-critical reads"

---

## Key Takeaways

1. **Replication = copying data to multiple databases** for availability and read scaling
2. **Primary-replica** most common: primary handles writes, replicas handle reads
3. **Async replication**: Fast but eventual consistency (most common in practice)
4. **Sync replication**: Slow but strong consistency (use for critical data)
5. **Read replicas** enable horizontal read scaling (add replicas → more read capacity)
6. **Replication lag**: Monitor and handle (read-your-own-writes pattern)
7. **Failover**: Automatic preferred, 30-90 second downtime typical`,
};
