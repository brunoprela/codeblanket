/**
 * Apache Cassandra Section
 */

export const cassandraSection = {
  id: 'apache-cassandra',
  title: 'Apache Cassandra',
  content: `Apache Cassandra is a highly available, masterless distributed database designed for handling massive amounts of data across commodity servers with no single point of failure.

## Overview

**Cassandra** = Masterless, eventually consistent, wide-column NoSQL database

**Created**: Facebook (2008) → Apache (2009)

**Design influences**:
- **Data model**: Google BigTable (wide-column)
- **Architecture**: Amazon Dynamo (masterless, eventually consistent)

**Used by**:
- Netflix (millions of writes/sec)
- Apple (75,000+ nodes, 10+ PB)
- Instagram (feed storage)
- Uber (time-series data)

**Scale**:
- Linear scalability (add nodes = increase throughput)
- Petabytes of data
- Millions of operations per second
- Multi-datacenter replication

---

## Key Characteristics

**1. Masterless (Peer-to-Peer)**:
- No master node → no single point of failure
- All nodes are equal
- Any node can handle any request

**2. Eventually Consistent**:
- Favors availability over consistency (AP in CAP)
- Tunable consistency levels
- Conflict resolution via timestamps

**3. Wide-Column Data Model**:
- Similar to BigTable
- Flexible schema
- Denormalization encouraged

**4. High Availability**:
- Replication across nodes
- Multi-datacenter support
- Automatic failover

**5. Linear Scalability**:
- Add nodes without downtime
- Throughput scales linearly
- No need to over-provision

---

## Data Model

### Hierarchy

\`\`\`
Keyspace (database)
  └── Table
       └── Row (identified by primary key)
            └── Columns
\`\`\`

### Example Schema

\`\`\`cql
CREATE KEYSPACE social WITH replication = {
  'class': 'NetworkTopologyStrategy',
  'datacenter1': 3,
  'datacenter2': 2
};

CREATE TABLE users (
  user_id uuid PRIMARY KEY,
  username text,
  email text,
  created_at timestamp
);

CREATE TABLE posts (
  user_id uuid,
  post_id timeuuid,
  content text,
  likes int,
  PRIMARY KEY (user_id, post_id)
) WITH CLUSTERING ORDER BY (post_id DESC);
\`\`\`

### Primary Key

**Components**:
1. **Partition Key**: Determines which nodes store the data
2. **Clustering Key**: Sorts data within partition

\`\`\`cql
PRIMARY KEY (partition_key)
PRIMARY KEY (partition_key, clustering_key)
PRIMARY KEY ((compound, partition, key), clustering_key)
\`\`\`

**Example**:
\`\`\`cql
PRIMARY KEY (user_id, post_id)
           ─────────  ────────
           Partition  Clustering
              Key       Key
\`\`\`

**Partition key** determines data location:
\`\`\`
hash(user_id) → Token → Node(s)
\`\`\`

**Clustering key** sorts within partition:
\`\`\`
user_id=123:
  post_id=uuid4 → content="Hello"
  post_id=uuid3 → content="World"
  post_id=uuid2 → content="Foo"
(Sorted by post_id DESC)
\`\`\`

---

## Architecture

### Ring Topology

\`\`\`
              Node A
             (Token: 0)
               /   \\
              /     \\
    Node D   /       \\   Node B
(Token: 768)         (Token: 256)
            \\       /
             \\     /
              \\   /
              Node C
           (Token: 512)
\`\`\`

**Ring properties**:
- Token range: 0 to 2^64-1
- Each node owns a range of tokens
- Data distributed by consistent hashing
- No master, all nodes equal

### Virtual Nodes (vnodes)

**Old approach**: Each physical node = 1 token
**New approach**: Each physical node = 256 tokens (vnodes)

\`\`\`
Physical Node 1:
  vnode: token=100
  vnode: token=200
  vnode: token=350
  ... (253 more)

Physical Node 2:
  vnode: token=50
  vnode: token=175
  vnode: token=400
  ... (253 more)
\`\`\`

**Benefits**:
- ✅ Faster rebalancing (distribute vnodes, not entire node)
- ✅ Better load distribution
- ✅ Easier to add/remove nodes

---

## Replication Strategy

### Replication Factor (RF)

**RF = 3** means each row stored on 3 nodes

\`\`\`
Write to partition key "user_id=123"
Token: hash(123) = 500

Coordinator determines replicas:
  Primary: Node C (token 500)
  Replica 1: Node D (next in ring)
  Replica 2: Node A (next in ring)
\`\`\`

### Replication Strategies

**1. SimpleStrategy** (single datacenter):
\`\`\`cql
CREATE KEYSPACE test WITH replication = {
  'class': 'SimpleStrategy',
  'replication_factor': 3
};
\`\`\`
Replicas placed on next N nodes in ring

**2. NetworkTopologyStrategy** (multi-datacenter):
\`\`\`cql
CREATE KEYSPACE prod WITH replication = {
  'class': 'NetworkTopologyStrategy',
  'us-east': 3,
  'us-west': 2,
  'eu-west': 2
};
\`\`\`
Specify RF per datacenter

**Best practice**: Always use NetworkTopologyStrategy (even single DC)

---

## Consistency Levels

### Tunable Consistency

**Write consistency**:
- **ANY**: Hinted handoff OK (fastest, least consistent)
- **ONE**: At least 1 replica ACK
- **TWO**: At least 2 replicas ACK
- **THREE**: At least 3 replicas ACK
- **QUORUM**: Majority (RF/2 + 1) ACK
- **ALL**: All replicas ACK (slowest, most consistent)
- **LOCAL_QUORUM**: Quorum in local DC
- **EACH_QUORUM**: Quorum in each DC

**Read consistency**:
- **ONE**: Read from 1 replica (fastest)
- **TWO**: Read from 2, compare
- **THREE**: Read from 3, compare
- **QUORUM**: Read from majority
- **ALL**: Read from all replicas (slowest, most consistent)
- **LOCAL_QUORUM**: Quorum in local DC

### Strong Consistency Formula

**Strong consistency** achieved when:
\`\`\`
Write CL + Read CL > RF
\`\`\`

**Example** (RF=3):
\`\`\`
QUORUM write + QUORUM read = Strong consistency
  (2 writes + 2 reads > 3)
  
ONE write + ALL read = Strong consistency
  (1 write + 3 reads > 3)

ONE write + ONE read = Eventually consistent
  (1 write + 1 read < 3)
\`\`\`

---

## Write Path

### Write Flow

\`\`\`
Client
  ↓
Coordinator Node (any node can coordinate)
  ↓ (determines replicas based on partition key)
  ├─→ Node A (replica 1) ─→ CommitLog → Memtable
  ├─→ Node B (replica 2) ─→ CommitLog → Memtable
  └─→ Node C (replica 3) ─→ CommitLog → Memtable
  ↓ (wait for CL acknowledgments)
Success returned to client
\`\`\`

### Detailed Write Process on Each Node

**Step 1**: Write to **CommitLog** (on disk, append-only)
- Durability guarantee
- Sequential writes (fast!)

**Step 2**: Write to **Memtable** (in memory, sorted)
- Fast lookups
- Sorted by clustering key

**Step 3**: Return ACK to coordinator

**Step 4** (later): When memtable full → flush to **SSTable**
- Immutable sorted file on disk
- Memtable cleared

### SSTables (Sorted String Tables)

**Immutable files on disk**:
\`\`\`
Data.db:        Actual data
Index.db:       Index for faster lookups
Filter.db:      Bloom filter
Statistics.db:  Metadata
Summary.db:     Index summary
\`\`\`

**Multiple SSTables accumulate over time** → compaction needed

---

## Read Path

### Read Flow

\`\`\`
Client → Coordinator
  ↓
Determine replicas
  ↓
Read from CL nodes
  ├─→ Node A: Check Memtable + SSTables
  ├─→ Node B: Check Memtable + SSTables
  └─→ Node C: Check Memtable + SSTables
  ↓
Merge results (latest timestamp wins)
  ↓
Return to client
\`\`\`

### Read on Each Node

**Step 1**: Check **row cache** (if enabled)
- Caches entire rows
- Skip memtable + SSTable lookup if hit

**Step 2**: Check **Memtable**
- Most recent writes

**Step 3**: Check **Bloom filters**
- Quickly determine if key might be in SSTable
- Avoids unnecessary disk reads

**Step 4**: Check **Partition Key Cache**
- Caches partition key locations

**Step 5**: Read **SSTables** (oldest to newest)
- May need to check multiple SSTables
- Merge results (latest timestamp wins)

**Step 6**: Perform **read repair** (if enabled)
- Detect inconsistencies
- Send updates to out-of-date replicas

---

## Compaction

### Why Compaction?

**Problem**: Accumulation of SSTables
- Slow reads (check many files)
- Wasted space (old versions, tombstones)

**Solution**: Compact SSTables

### Compaction Strategies

**1. SizeTiered Compaction (STCS)**:
- Merge SSTables of similar size
- Good for: Write-heavy workloads
- Bad for: Can cause temporary 2x space usage

**2. Leveled Compaction (LCS)**:
- SSTables organized in levels
- Each level 10x larger than previous
- Good for: Read-heavy, limited space
- Bad for: Write amplification

**3. Time-Window Compaction (TWCS)**:
- Compact data in time windows
- Good for: Time-series data
- Data never crosses time window boundaries

**4. Incremental Compaction (ICS)**:
- Unified compaction strategy (Cassandra 4.1+)
- Adaptive, combines benefits of STCS and LCS

### Example: STCS

\`\`\`
4 SSTables of ~100 MB each
  ↓ Compact
1 SSTable of 400 MB

4 SSTables of ~400 MB each
  ↓ Compact
1 SSTable of 1.6 GB
\`\`\`

---

## Hinted Handoff

### Temporary Failures

**Scenario**: Node B is temporarily down

\`\`\`
Write (RF=3) → Should go to Node A, B, C
Node B is down!
  ↓
Coordinator writes hint to Node D
  ↓
When Node B comes back online:
Node D replays hints to Node B
\`\`\`

**Hints**:
- Stored on coordinator (or another node)
- Replayed when target node recovers
- TTL: 3 hours by default
- Not a substitute for repair!

**Benefits**:
- ✅ Improve eventual consistency
- ✅ Reduce repair workload

---

## Anti-Entropy Repair

### Background Repair

**Purpose**: Ensure all replicas are in sync

**Process**:
1. Build Merkle trees for each replica
2. Compare Merkle trees
3. Identify differences
4. Stream missing data

**Types**:
- **Read repair**: On-demand during reads (automatic)
- **Manual repair**: \`nodetool repair\` (manual)
- **Subrange repair**: Repair specific token ranges

**Best practice**: Run full cluster repair every GC grace seconds (10 days)

---

## Gossip Protocol

### Peer-to-Peer Communication

**Purpose**: Nodes discover cluster state

**How it works**:
1. Every second, node gossips with 1-3 random nodes
2. Exchange state information:
   - Which nodes are alive
   - Load information
   - Schema version
   - Token ranges

**Failure detection**: Phi Accrual Failure Detector
- Adaptive (learns normal latency)
- Suspicion score (phi value)
- phi > 8 → node considered down

---

## Multi-Datacenter Replication

### Architecture

\`\`\`
Datacenter 1 (us-east)          Datacenter 2 (eu-west)
   Node A1                          Node B1
   Node A2                          Node B2
   Node A3                          Node B3
      ↓                                ↓
    Write replicated across datacenters
\`\`\`

**Replication**:
\`\`\`cql
CREATE KEYSPACE global WITH replication = {
  'class': 'NetworkTopologyStrategy',
  'us-east': 3,
  'eu-west': 3
};
\`\`\`

**Consistency Levels**:
- **LOCAL_QUORUM**: Quorum in local DC (fast, no cross-DC latency)
- **EACH_QUORUM**: Quorum in each DC (slow, strong consistency)

**Best practice**: Use LOCAL_QUORUM for most operations

---

## Cassandra Query Language (CQL)

### Basic Operations

**Insert/Update**:
\`\`\`cql
INSERT INTO users (user_id, username, email)
VALUES (uuid(), 'john', 'john@example.com');

UPDATE users
SET email = 'newemail@example.com'
WHERE user_id = 123e4567-e89b-12d3-a456-426614174000;
\`\`\`

**Select**:
\`\`\`cql
SELECT * FROM users WHERE user_id = ?;

SELECT * FROM posts WHERE user_id = ? AND post_id > ?;

SELECT * FROM posts WHERE user_id = ? ORDER BY post_id DESC LIMIT 10;
\`\`\`

**Delete**:
\`\`\`cql
DELETE FROM users WHERE user_id = ?;

DELETE email FROM users WHERE user_id = ?;  -- Delete single column
\`\`\`

### Tombstones

**Deletes create tombstones** (markers):
\`\`\`
DELETE FROM users WHERE user_id = 123;
  ↓
Writes tombstone with timestamp
  ↓
Tombstone removed after gc_grace_seconds (10 days)
\`\`\`

**Why?**
- Eventual consistency: Replicas must know about delete
- Deleted data might be replayed from hints or repair
- Tombstone prevents resurrection

**Problem**: Too many tombstones slow reads

---

## Data Modeling

### Cassandra-Specific Principles

**1. Query-Driven Design**:
- Design tables for queries, not normalization
- One table per query pattern

**2. Denormalization is Normal**:
- Duplicate data across tables
- Update in multiple places

**3. Avoid JOINS**:
- No joins in Cassandra
- Embed related data or use multiple queries

**4. Partition Size Matters**:
- Keep partitions < 100 MB
- Avoid unbounded partitions

### Example: Twitter-Like Feed

**Bad design** (unbounded partition):
\`\`\`cql
CREATE TABLE user_posts (
  user_id uuid,
  post_id timeuuid,
  content text,
  PRIMARY KEY (user_id, post_id)
);
-- Problem: Popular user = huge partition!
\`\`\`

**Good design** (bucketing):
\`\`\`cql
CREATE TABLE user_posts (
  user_id uuid,
  bucket int,  -- 0-99
  post_id timeuuid,
  content text,
  PRIMARY KEY ((user_id, bucket), post_id)
);
-- Query: Fetch from multiple buckets
\`\`\`

---

## Use Cases

**1. Time-Series Data**:
- IoT sensor data
- Logs and metrics
- User activity tracking

**Why Cassandra?**
- High write throughput
- Time-based partitioning
- Easy to query recent data

**2. User Profiles**:
- User accounts
- Preferences
- Session data

**Why Cassandra?**
- Fast lookups by user ID
- Flexible schema

**3. Product Catalog**:
- E-commerce products
- Inventory

**Why Cassandra?**
- Multiple query patterns (by category, by tag)
- Denormalization works well

**Not suitable for**:
- ❌ Complex joins
- ❌ Aggregations (SUM, AVG)
- ❌ Strong consistency requirements
- ❌ Ad-hoc queries

---

## Operations

### Adding Nodes

\`\`\`bash
# Add new node to cluster
# 1. Configure new node with seed nodes
# 2. Start Cassandra on new node
# 3. Node automatically joins via gossip

# Rebalance data
nodetool cleanup  -- Remove data new node now owns
\`\`\`

**Zero downtime**: Cluster continues operating

### Removing Nodes

\`\`\`bash
# Decommission node
nodetool decommission  -- Streams data to other nodes
\`\`\`

### Monitoring

**Key metrics**:
- Read/write latency (p99)
- Read/write throughput
- Pending compactions
- Disk usage
- GC pause time

**Tools**:
- nodetool
- DataStax OpsCenter
- Prometheus + Grafana

---

## Performance Tuning

### 1. Choose Right Consistency Level

- LOCAL_QUORUM for most use cases
- ONE for non-critical reads (faster)

### 2. Tune JVM

- Heap size: 8-16 GB (not more!)
- G1GC collector
- Monitor GC pauses

### 3. Compaction Strategy

- STCS for write-heavy
- LCS for read-heavy
- TWCS for time-series

### 4. Partition Size

- Keep < 100 MB
- Use bucketing for large partitions

### 5. Caching

- Enable row cache for hot data
- Partition key cache always on

---

## Interview Tips

**Explain Cassandra in 2 minutes**:
"Cassandra is a masterless, highly available, eventually consistent wide-column database. All nodes are equal, no single point of failure. Data is distributed via consistent hashing and replicated across nodes. Tunable consistency levels allow trading consistency for availability and performance. Writes go to commit log and memtable, then flushed to immutable SSTables. Compaction merges SSTables. Gossip protocol for cluster state, hinted handoff for temporary failures. Best for high write throughput, linear scalability, and multi-datacenter replication. Not suitable for complex queries or strong consistency."

**Key trade-offs**:
- Availability vs consistency (tunable)
- Write performance vs eventual consistency
- Denormalization vs storage efficiency
- Masterless vs operational simplicity

**Common mistakes**:
- ❌ Trying to use like SQL database (joins, transactions)
- ❌ Not considering partition size
- ❌ Using wrong consistency level
- ❌ Not running regular repairs

---

## Key Takeaways

🔑 Masterless architecture = no single point of failure
🔑 Tunable consistency: trade-off availability and consistency
🔑 Linear scalability: add nodes = increase throughput
🔑 Write-optimized: CommitLog + Memtable + SSTable (LSM-tree)
🔑 Partition key determines data placement
🔑 Denormalization is expected (query-driven design)
🔑 Multi-datacenter replication built-in
🔑 Gossip protocol for cluster coordination
🔑 Hinted handoff improves eventual consistency
🔑 Best for: time-series, high write throughput, multi-DC
`,
};
