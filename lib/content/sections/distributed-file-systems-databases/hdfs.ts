/**
 * HDFS (Hadoop Distributed File System) Section
 */

export const hdfsSection = {
  id: 'hdfs',
  title: 'HDFS (Hadoop Distributed File System)',
  content: `HDFS (Hadoop Distributed File System) is the open-source implementation of Google File System concepts, designed to run on commodity hardware and store massive datasets for Hadoop processing.

## Overview

**What is HDFS?**
- Distributed file system for Hadoop ecosystem
- Stores very large files (GBs to TBs) across clusters
- Optimized for **batch processing**, not real-time access
- Inspired by Google\'s GFS paper (2003)
- Written in Java, part of Apache Hadoop

**Scale**: Single clusters with 10,000+ nodes storing petabytes

---

## HDFS Architecture

\`\`\`
                    ┌──────────────────┐
                    │    NameNode      │  (Master)
                    │   (Metadata)     │
                    │                  │
                    │  - Namespace     │
                    │  - Block mapping │
                    │  - Access control│
                    └──────────────────┘
                           │
                           │ Heartbeat + Block reports
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
  ┌──────────┐       ┌──────────┐      ┌──────────┐
  │DataNode 1│       │DataNode 2│      │DataNode 3│
  │          │       │          │      │          │
  │ Blocks:  │       │ Blocks:  │      │ Blocks:  │
  │ B1, B3   │       │ B1, B2   │      │ B2, B3   │
  └──────────┘       └──────────┘      └──────────┘
        ▲                  ▲                  ▲
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │ Read/Write
                    ┌──────────────┐
                    │   Client     │
                    └──────────────┘
\`\`\`

### Core Components

**1. NameNode (Master)**
- Single master (like GFS Master)
- Manages file system namespace
- Manages block-to-DataNode mapping
- Handles metadata operations

**2. DataNodes (Workers)**
- Store actual data blocks
- Serve read/write requests
- Report block health to NameNode
- Execute block creation/deletion/replication

**3. Client**
- HDFS client library
- Interacts with NameNode for metadata
- Directly reads/writes data to DataNodes

---

## NameNode Deep Dive

### Responsibilities

**1. Namespace management**:
\`\`\`
/user/hadoop/input/data.txt → [Block1, Block2, Block3]
/user/hive/warehouse/table1 → [Block4, Block5]
\`\`\`

**2. Block mapping**:
\`\`\`
Block1 → [DataNode1, DataNode2, DataNode3]  (3 replicas)
Block2 → [DataNode2, DataNode4, DataNode5]
Block3 → [DataNode1, DataNode3, DataNode6]
\`\`\`

**3. Coordination**:
- Heartbeat from DataNodes (3 seconds)
- Block reports from DataNodes (6 hours)
- Client operations (create, delete, open, close)

### Metadata Storage

**In-memory structures** (for speed):
- Entire file system namespace
- Block → DataNode mapping
- Typical: 150 bytes per block, 1GB metadata = ~5-6 million blocks

**On-disk structures** (for persistence):
- **FSImage**: Snapshot of file system namespace
- **EditLog**: Transaction log of changes

**How they work together**:
\`\`\`
1. NameNode starts → loads FSImage into memory
2. Every modification → written to EditLog
3. EditLog grows → eventually merged into FSImage (checkpointing)
4. Crash recovery → load FSImage + replay EditLog
\`\`\`

### Example EditLog entries:
\`\`\`
1. OP_ADD: /user/data.txt, replication=3
2. OP_ALLOCATE_BLOCK_ID: Block_1001
3. OP_ADD_BLOCK: /user/data.txt → Block_1001
4. OP_CLOSE: /user/data.txt
5. OP_DELETE: /user/old.txt
\`\`\`

---

## DataNode Deep Dive

### Block Storage

**Block = unit of storage**:
- Default size: **128 MB** (configurable: 64 MB to 256 MB)
- Stored as regular files on DataNode's local file system
- Each block stored in separate file

**Example DataNode storage**:
\`\`\`
/hdfs/data/current/
  blk_1001        (128 MB - data)
  blk_1001.meta   (checksum metadata)
  blk_1002        (128 MB - data)
  blk_1002.meta   (checksum metadata)
  ...
\`\`\`

### Why 128 MB blocks?

**Larger blocks = fewer blocks** = less metadata at NameNode

**Example**:
- 1 TB file with 128 MB blocks = 8,192 blocks
- 1 TB file with 1 MB blocks = 1,048,576 blocks (130x more metadata!)

**Trade-offs**:
- ✅ Reduced metadata overhead
- ✅ Reduced NameNode memory usage
- ✅ Fewer network connections
- ✅ Less seek time overhead (sequential I/O)
- ❌ Wastes space for small files (1 KB file uses 1 block)
- ❌ Fewer parallel operations (for small files)

### Heartbeat and Block Reports

**Heartbeat (every 3 seconds)**:
- DataNode → NameNode: "I'm alive"
- NameNode → DataNode: Commands (replicate, delete blocks)
- Missing heartbeat (10 minutes) = DataNode presumed dead

**Block report (every 6 hours)**:
- Complete list of blocks stored on DataNode
- NameNode builds/updates block→DataNode mapping
- Detects missing or corrupt blocks

---

## HDFS Read Flow

**Scenario**: Client reads \`/user/data.txt\` (300 MB, 3 blocks)

\`\`\`
Client              NameNode            DataNodes
  |                     |                    |
  |-- (1) Open file -->|                    |
  |                     |                    |
  |<-- (2) Block ------ |                    |
  |   locations         |                    |
  |   B1:[DN1,DN2,DN3]  |                    |
  |   B2:[DN2,DN4,DN5]  |                    |
  |   B3:[DN1,DN3,DN6]  |                    |
  |                     |                    |
  |-- (3) Read B1 ------|-----------------> DN1
  |                     |                    |
  |<-- (4) B1 data -----|-------------------|
  |                     |                    |
  |-- (5) Read B2 ------|-----------------> DN2
  |<-- (6) B2 data -----|-------------------|
  |                     |                    |
  |-- (7) Read B3 ------|-----------------> DN1
  |<-- (8) B3 data -----|-------------------|
  |                     |                    |
  |-- (9) Close file -->|                    |
\`\`\`

**Detailed steps**:

**1. Client opens file**:
\`\`\`java
FileSystem fs = FileSystem.get (conf);
FSDataInputStream in = fs.open("/user/data.txt");
\`\`\`

**2. NameNode returns block locations**:
- Sorted by distance to client (network topology)
- Client caches this info

**3-8. Client reads blocks**:
- Reads from closest DataNode
- Verifies checksums
- If failure, tries next DataNode replica
- Continues to next block

**9. Client closes stream**:
- Releases resources

**Key points**:
- NameNode NOT involved in data transfer
- Client reads directly from DataNodes
- Client intelligently chooses nearest replica
- Checksum verification at read time

---

## HDFS Write Flow

**Scenario**: Client writes 300 MB file (3 blocks, replication=3)

\`\`\`
Client              NameNode            DataNodes
  |                     |                    |
  |-- (1) Create -->----|                    |
  |                     |                    |
  |<-- (2) OK ----------|                    |
  |                     |                    |
  |-- (3) Add block --->|                    |
  |                     |                    |
  |<-- (4) B1 ---------|                    |
  |   write to:         |                    |
  |   DN1,DN2,DN3       |                    |
  |                     |                    |
  |-- (5) Setup --------|-----------------> DN1
  |   pipeline          |       |
  |                     |       └----------> DN2
  |                     |              |
  |                     |              └----> DN3
  |                     |                    |
  |-- (6) Write --------|-----------------> DN1
  |   data packets      |       |
  |                     |       └----------> DN2
  |                     |              |
  |                     |              └----> DN3
  |                     |                    |
  |<-- (7) ACK ---------|-------------------|
  |   pipeline          |                    |
  |                     |                    |
  |-- (8) Close ------->|                    |
\`\`\`

**Detailed steps**:

**1-2. Create file**:
- Client asks NameNode to create file
- NameNode checks permissions, creates entry in namespace
- Initially: file exists but has 0 blocks

**3-4. Request block allocation**:
- Client requests block for writing
- NameNode allocates block ID
- Chooses 3 DataNodes based on: rack diversity, disk space, load

**5. Setup write pipeline**:
- DN1 → DN2 → DN3 (linear pipeline)
- Each DataNode establishes connection to next

**6. Client writes data**:
- Data split into 64 KB packets
- Packet sent to DN1 → DN1 forwards to DN2 → DN2 forwards to DN3
- **Pipeline replication** (not star topology!)

**7. ACK pipeline**:
- DN3 ACKs to DN2 → DN2 ACKs to DN1 → DN1 ACKs to Client
- Only when all replicas ACK, packet considered written

**8. Complete and close**:
- Client signals completion
- NameNode commits file (marks as complete)

### Why Pipeline Replication?

**Pipeline (HDFS approach)**:
\`\`\`
Client → DN1 → DN2 → DN3
        (128 MB)
Network: 128 MB
\`\`\`

**Star topology (alternative)**:
\`\`\`
Client → DN1
Client → DN2
Client → DN3
        (128 MB each)
Network: 384 MB from client!
\`\`\`

Pipeline uses **1/3 the client bandwidth**!

---

## Replication Strategy

### Rack Awareness

HDFS is **rack-aware**:
\`\`\`
Rack 1              Rack 2              Rack 3
┌──────────┐       ┌──────────┐       ┌──────────┐
│  DN1     │       │  DN3     │       │  DN5     │
│  DN2     │       │  DN4     │       │  DN6     │
└──────────┘       └──────────┘       └──────────┘
     │                  │                  │
     └──────────────────┼──────────────────┘
            Rack switch (expensive)
\`\`\`

**Default replication policy** (replication=3):
1. **First replica**: Same node as client (or random if client outside cluster)
2. **Second replica**: Different rack (rack fault tolerance)
3. **Third replica**: Same rack as second (save bandwidth)

**Example**:
- Client on DN1 (Rack 1)
- First replica: DN1 (Rack 1) - local
- Second replica: DN3 (Rack 2) - different rack
- Third replica: DN4 (Rack 2) - same rack as second

**Benefits**:
- ✅ Tolerates single rack failure (have replica in another rack)
- ✅ Minimizes inter-rack traffic (2 replicas in same rack)
- ✅ Write performance (first write is local)

### Re-replication

**Triggers for re-replication**:
1. DataNode dies (blocks under-replicated)
2. Disk failure (blocks lost)
3. Replication factor increased
4. Corruption detected

**Re-replication priority**:
1. **Critical**: Blocks with only 1 replica
2. **High**: Blocks with < replication factor
3. **Normal**: Blocks evenly distributed

**Process**:
- NameNode chooses source DataNode (has block)
- NameNode chooses destination DataNode (needs block)
- Destination DataNode copies block from source

---

## NameNode High Availability

**Problem**: NameNode is single point of failure!

### Solution: HA with Standby NameNode

\`\`\`
      ┌──────────────┐              ┌──────────────┐
      │Active        │              │Standby       │
      │NameNode      │◄────────────►│NameNode      │
      └──────────────┘              └──────────────┘
             │                             │
             │                             │
             └─────────────┬───────────────┘
                           │
                    ┌──────────────┐
                    │ Shared       │
                    │ EditLog      │
                    │ (JournalNodes│
                    │  or NFS)     │
                    └──────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
   DataNode1          DataNode2          DataNode3
\`\`\`

**Key components**:

**1. Active NameNode**:
- Serves all client requests
- Writes to shared EditLog

**2. Standby NameNode**:
- Reads from shared EditLog
- Maintains up-to-date namespace
- Performs checkpointing
- Ready for instant failover

**3. Shared EditLog**:
- QJM (Quorum Journal Manager): Paxos-like consensus (typical)
- NFS: Shared storage (simpler but less reliable)

**4. ZooKeeper**:
- Coordinates failover
- Ensures only one active NameNode
- Prevents split-brain

**Failover**:
- Automatic via ZKFC (ZooKeeper Failover Controller)
- ZooKeeper detects Active NameNode failure
- Standby promoted to Active (~30 seconds)
- Fence previous Active (prevent split-brain)

---

## HDFS Federation

**Problem**: Single NameNode limits:
- Memory: All metadata must fit in RAM
- Throughput: All metadata ops go through one NameNode
- Isolation: One namespace for all users

**Solution**: HDFS Federation = Multiple NameNodes

\`\`\`
NameNode1          NameNode2          NameNode3
/user              /data              /tmp
  |                  |                  |
  ├─ hadoop          ├─ input           ├─ temp1
  ├─ hive            └─ output          └─ temp2
  └─ spark
  
     └──────────────────┴──────────────────┘
                    │
            Shared DataNode pool
\`\`\`

**Benefits**:
- ✅ Horizontal scaling of metadata
- ✅ Namespace isolation
- ✅ Better performance (distributed load)

**Trade-off**: No single global namespace

---

## HDFS Snapshots

**Create point-in-time copies** of directory:

\`\`\`bash
# Enable snapshots on directory
hdfs dfsadmin -allowSnapshot /user/data

# Create snapshot
hdfs dfs -createSnapshot /user/data snapshot-2024-01

# List snapshots
hdfs lsSnapshottableDir

# Restore from snapshot
hdfs dfs -cp /user/data/.snapshot/snapshot-2024-01/file.txt /user/data/
\`\`\`

**Implementation**:
- Copy-on-write (no data duplication on snapshot creation)
- Only delta stored
- Fast snapshot creation (metadata operation)

**Use cases**:
- Backup before risky operation
- Data recovery
- Testing (rollback if experiment fails)

---

## HDFS vs GFS Comparison

| Feature             | GFS (Google)         | HDFS (Hadoop)           |
|---------------------|----------------------|-------------------------|
| Language            | C++                  | Java                    |
| Master              | GFS Master           | NameNode                |
| Worker              | Chunkserver          | DataNode                |
| Block size          | 64 MB                | 128 MB (default)        |
| Replication         | 3x (default)         | 3x (default)            |
| Consistency         | Weak (relaxed)       | Strong (for closed files)|
| Append              | Record append        | Append (added later)    |
| HA                  | Shadow masters       | Standby NameNode + ZK   |
| Open source         | No                   | Yes                     |
| Federation          | Unclear              | Yes                     |

---

## HDFS Use Cases

### 1. Data Lake Storage

\`\`\`
HDFS as central repository:
- Raw logs from servers
- Database dumps
- IoT sensor data
- Machine learning datasets
\`\`\`

**Why HDFS?**
- ✅ Cheap storage (commodity hardware)
- ✅ Petabyte scale
- ✅ Integrates with Hadoop ecosystem (Spark, Hive, Pig)

### 2. MapReduce/Spark Processing

\`\`\`
1. Upload data to HDFS
2. Run MapReduce/Spark job
3. Results written back to HDFS
4. Further processing or export
\`\`\`

**Why HDFS?**
- ✅ Data locality (compute moves to data)
- ✅ High throughput for batch reads
- ✅ Fault tolerance during long-running jobs

### 3. Log Aggregation

\`\`\`
Web servers → Flume/Kafka → HDFS → Hive/Spark
\`\`\`

**Why HDFS?**
- ✅ Append-only writes (perfect for logs)
- ✅ Cheap long-term storage
- ✅ SQL-like queries via Hive

---

## HDFS Operations

### Common commands:

\`\`\`bash
# Upload file
hdfs dfs -put local.txt /user/hadoop/

# Download file
hdfs dfs -get /user/hadoop/data.txt local.txt

# List directory
hdfs dfs -ls /user/hadoop/

# Create directory
hdfs dfs -mkdir /user/hadoop/input

# Delete file
hdfs dfs -rm /user/hadoop/old.txt

# Cat file (small files only!)
hdfs dfs -cat /user/hadoop/data.txt

# Check file replication and block info
hdfs fsck /user/hadoop/data.txt -files -blocks -locations

# Get file stats
hdfs dfs -stat "%b %r %n" /user/hadoop/data.txt
\`\`\`

### Administrative commands:

\`\`\`bash
# Report cluster health
hdfs dfsadmin -report

# Safe mode (for maintenance)
hdfs dfsadmin -safemode enter
hdfs dfsadmin -safemode leave

# Balance DataNodes
hdfs balancer -threshold 10

# Decommission DataNode
hdfs dfsadmin -refreshNodes
\`\`\`

---

## HDFS Performance Tuning

### 1. Block Size

**When to increase block size** (e.g., 256 MB):
- Very large files (TBs)
- Reduce NameNode memory usage
- Sequential processing (MapReduce)

**When to keep smaller** (128 MB or 64 MB):
- Many small to medium files
- More parallelism (more blocks = more map tasks)

### 2. Replication Factor

**Increase replication** (e.g., 4 or 5):
- Very important data
- Hot data (many reads)
- Small files (improve read parallelism)

**Decrease replication** (e.g., 2):
- Temporary data
- Data easily regenerated
- Save storage space

### 3. Short-circuit reads

**Optimization**: When client and block are on same node, read directly from local disk (bypass DataNode process)

\`\`\`xml
<property>
  <name>dfs.client.read.shortcircuit</name>
  <value>true</value>
</property>
\`\`\`

**Benefit**: 2-3x faster reads for local blocks

---

## HDFS Limitations

**Not suitable for**:
- ❌ Low-latency access (use HBase, Cassandra)
- ❌ Lots of small files (NameNode memory pressure)
- ❌ Multiple writers, random writes (append-only)
- ❌ POSIX semantics (close file before reading writes)

**Suitable for**:
- ✅ Batch processing (MapReduce, Spark)
- ✅ Large files (GBs, TBs)
- ✅ Sequential reads/writes
- ✅ Write-once, read-many workload
- ✅ Throughput > latency

---

## Interview Tips

**Explain HDFS in 2 minutes**:
"HDFS is Hadoop\'s distributed file system, inspired by Google's GFS. It has a single NameNode managing metadata and multiple DataNodes storing 128 MB blocks with 3x replication. Files are written once and read many times. HDFS optimizes for high throughput over low latency, making it perfect for batch processing. NameNode HA is achieved with Standby NameNode and ZooKeeper. The system is rack-aware for fault tolerance and network efficiency. It's not suitable for small files or low-latency access, but excels at storing and processing massive datasets."

**Key differences from GFS**:
- Open source vs proprietary
- 128 MB blocks vs 64 MB
- Stronger consistency (for closed files)
- HA with ZooKeeper integration
- Federation support

**Common interview questions**:
- Why 128 MB blocks? (Reduce metadata, improve throughput)
- How does rack awareness work? (2 replicas in one rack, 1 in another)
- What happens when NameNode fails? (Standby promoted via ZK)
- Why is HDFS bad for small files? (Each file = metadata in NameNode RAM)

---

## Key Takeaways

🔑 HDFS = open-source GFS for Hadoop ecosystem
🔑 NameNode (master) + DataNodes (workers) architecture
🔑 128 MB blocks with 3x replication by default
🔑 Rack-aware replication for fault tolerance
🔑 Optimized for throughput, not latency
🔑 HA via Standby NameNode + ZooKeeper
🔑 Federation enables horizontal scaling of metadata
🔑 Perfect for batch processing, bad for random access
🔑 Write-once, read-many workload pattern
🔑 Foundation of Hadoop ecosystem (Hive, Spark, HBase)
`,
};
