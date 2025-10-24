/**
 * Apache HBase Section
 */

export const hbaseSection = {
  id: 'apache-hbase',
  title: 'Apache HBase',
  content: `Apache HBase is an open-source, distributed, scalable, big data store modeled after Google BigTable, providing real-time read/write access to large datasets on HDFS.

## Overview

**HBase** = Open-source BigTable implementation

**Created**: 2006 (originally part of Hadoop)

**Key characteristics**:
- Wide-column NoSQL database
- Built on HDFS
- Strong consistency (unlike Cassandra)
- Linear scalability
- Random read/write access
- No SQL (uses custom API and HBase shell)

**Used by**:
- Facebook (Messages storage)
- Twitter (monitoring data)
- Yahoo! (user data)
- Adobe (analytics)

---

## Architecture

\`\`\`
             ZooKeeper
            (Coordination)
                 │
         ┌───────┴────────┐
         │                │
      HMaster          HMaster
     (Active)        (Standby)
         │
    ┌────┴────┬────────┐
    │         │        │
RegionServer  RegionServer  RegionServer
    │         │        │
 Regions   Regions   Regions
    │         │        │
    └─────────┴────────┘
            │
          HDFS
\`\`\`

### Components

**1. HMaster**:
- Assign regions to RegionServers
- Handle region splitting
- Detect RegionServer failures
- Schema changes (create/delete tables)
- Load balancing

**2. RegionServer**:
- Serve reads and writes
- Manage regions
- Handle region splits
- Flush memstore to HDFS
- Compact HFiles

**3. ZooKeeper**:
- Coordinate cluster
- Track active HMaster
- Region assignment state
- Server liveness detection

**4. HDFS**:
- Persistent storage for HFiles
- Write-Ahead Log (WAL) storage

---

## Data Model

### Similar to BigTable

\`\`\`
Table: users
Row Key: "user_123"
Column Family: personal_info
  Column: personal_info:name → "Alice"
  Column: personal_info:email → "alice@example.com"
  Column: personal_info:age → "30"
Column Family: activity
  Column: activity:last_login → "2024-01-15T10:30:00Z"
  Column: activity:login_count → "42"
Timestamps: Each cell can have multiple versions
\`\`\`

### Key Concepts

**1. Row Key**:
- Unique identifier (byte array)
- Rows sorted lexicographically by row key
- **Design critical for performance!**

**2. Column Family**:
- Group related columns
- Must be defined at table creation
- Unit of access control
- Stored together on disk

**3. Column Qualifier**:
- Within column family
- Can be added dynamically
- Full name: \`family: qualifier\`

**4. Cell**:
- Intersection of row key, column family, column qualifier, timestamp
- Stores byte array value

**5. Timestamp**:
- Each cell can have multiple versions
- Automatic (system time) or manual
- Used for versioning and TTL

---

## Regions

### What is a Region?

**Region** = Contiguous range of rows

\`\`\`
Table: users (100M rows)
  ↓
Region 1: row_key [aaa → mmm]
Region 2: row_key [mmm → zzz]
\`\`\`

**Properties**:
- Initial table = 1 region
- Splits when reaches threshold (default 10 GB)
- Unit of distribution and load balancing
- Served by single RegionServer

### Region Split

\`\`\`
Region 1 (20 GB)
  ↓ (reaches split threshold)
Region 1a: [aaa → kkk] (10 GB)
Region 1b: [kkk → zzz] (10 GB)
\`\`\`

**Automatic** but can be pre-split for better distribution

---

## Read Path

\`\`\`
Client → RegionServer
  ↓
Check BlockCache (read cache)
  ↓ (cache miss)
Check MemStore (recent writes)
  ↓ (not found)
Read HFiles from HDFS
  ↓
Merge results (latest timestamp wins)
  ↓
Return to client
\`\`\`

**Read optimizations**:
- **BlockCache**: LRU cache of HFile blocks
- **Bloom filters**: Quickly determine if key might be in HFile
- **Block index**: Binary search within HFile

---

## Write Path

\`\`\`
Client → RegionServer
  ↓
Write to WAL (Write-Ahead Log) in HDFS
  ↓
Write to MemStore (in-memory sorted buffer)
  ↓
Return success to client
  ↓
(When MemStore full)
Flush MemStore to HFile in HDFS
\`\`\`

**Write durability**:
1. **WAL first**: Ensures durability (crash recovery)
2. **MemStore**: Fast in-memory writes
3. **Flush to HFile**: Periodic or when MemStore full

**HFiles**:
- Immutable sorted files on HDFS
- Similar to BigTable's SSTables

---

## Compaction

### Why Compaction?

**Problem**: Accumulation of HFiles
- Slow reads (check many files)
- Wasted space (old versions, deleted cells)

**Solution**: Compact HFiles

### Types of Compaction

**1. Minor Compaction**:
- Merge smaller HFiles into larger ones
- Doesn't remove deleted cells (tombstones)
- Fast, runs frequently

**2. Major Compaction**:
- Merge ALL HFiles in region
- Remove deleted cells and old versions
- Reclaim disk space
- Slow, runs weekly/monthly

**Process**:
\`\`\`
HFile1 + HFile2 + HFile3
  ↓ Merge sort
New HFile (compacted)
  ↓
Delete old HFiles
\`\`\`

---

## Row Key Design

### Critical for Performance!

**Bad designs** (hot spots):

❌ **Sequential keys** (timestamp, auto-increment):
\`\`\`
row_key: 2024-01-15-00:00:01
row_key: 2024-01-15-00:00:02
row_key: 2024-01-15-00:00:03
→ All writes go to last region (hot spot!)
\`\`\`

❌ **Domain-based keys** (URL):
\`\`\`
row_key: www.google.com
row_key: www.facebook.com
→ Popular domains = hot regions
\`\`\`

**Good designs** (distributed):

✅ **Reverse timestamp**:
\`\`\`
row_key: Long.MAX_VALUE - timestamp
→ Recent data (most accessed) distributed
\`\`\`

✅ **Hashed prefix**:
\`\`\`
row_key: hash(user_id) + user_id + timestamp
→ Writes distributed across regions
\`\`\`

✅ **Salted keys**:
\`\`\`
row_key: (id % num_buckets) + id
→ Distribute sequential IDs
\`\`\`

**Design principles**:
1. **Distribute writes** across regions
2. **Co-locate related data** for efficient scans
3. **Think about access patterns**

---

## HBase API

### Basic Operations

\`\`\`java
// Put (insert/update)
Put put = new Put(Bytes.toBytes("row_123"));
put.addColumn(
    Bytes.toBytes("personal_info"),
    Bytes.toBytes("name"),
    Bytes.toBytes("Alice")
);
table.put(put);

// Get (read)
Get get = new Get(Bytes.toBytes("row_123"));
Result result = table.get(get);
byte[] value = result.getValue(
    Bytes.toBytes("personal_info"),
    Bytes.toBytes("name")
);
String name = Bytes.toString(value);  // "Alice"

// Delete
Delete delete = new Delete(Bytes.toBytes("row_123"));
table.delete(delete);

// Scan (range query)
Scan scan = new Scan();
scan.setStartRow(Bytes.toBytes("row_100"));
scan.setStopRow(Bytes.toBytes("row_200"));
ResultScanner scanner = table.getScanner(scan);
for (Result r : scanner) {
    // Process result
}
scanner.close();
\`\`\`

---

## Consistency Model

### Strong Consistency

**HBase provides strong consistency** (unlike Cassandra):

\`\`\`
Write to row_key="user_123"
  ↓
Write to single RegionServer
  ↓
WAL + MemStore updated
  ↓
Immediately readable (same or different client)
\`\`\`

**Why strong consistency?**
- Single RegionServer serves each row
- No distributed coordination needed for single-row operations
- WAL ensures durability

**Limitation**: No multi-row transactions (until Phoenix)

---

## HBase vs Cassandra

| Feature         | HBase                  | Cassandra              |
|-----------------|------------------------|------------------------|
| Consistency     | Strong                 | Tunable (eventual)     |
| Architecture    | Master-slave           | Masterless (P2P)       |
| Storage         | HDFS                   | Local disk             |
| Availability    | Lower (master failure) | Higher (no SPOF)       |
| Write speed     | Moderate (WAL overhead)| Fast (no WAL by default)|
| Read speed      | Fast with cache        | Fast                   |
| Operations      | Simpler                | More complex           |
| Use case        | Strong consistency     | High availability      |

**Choose HBase if**:
- Need strong consistency
- Already using Hadoop ecosystem
- Complex scans and filtering
- Can tolerate brief unavailability

**Choose Cassandra if**:
- Need high availability (no downtime)
- Write-heavy workload
- Multi-datacenter replication
- Tunable consistency acceptable

---

## HBase with Phoenix

### SQL on HBase

**Phoenix** = SQL layer on top of HBase

\`\`\`sql
-- Create table
CREATE TABLE users (
    user_id BIGINT PRIMARY KEY,
    name VARCHAR,
    email VARCHAR,
    age INTEGER
);

-- Insert
UPSERT INTO users VALUES (123, 'Alice', 'alice@example.com', 30);

-- Query
SELECT * FROM users WHERE age > 25;

-- Join (yes, joins!)
SELECT u.name, o.amount
FROM users u
JOIN orders o ON u.user_id = o.user_id;
\`\`\`

**Benefits**:
- ✅ Familiar SQL interface
- ✅ Secondary indexes
- ✅ Query optimization
- ✅ JDBC/ODBC drivers
- ✅ Transactions (limited)

**Performance**:
- Compiles SQL to HBase scans
- Pushes down predicates
- Parallel execution
- Often faster than raw HBase API!

---

## Secondary Indexes

### Problem

**HBase only indexes by row key**:
\`\`\`
Fast: Get by row_key
Slow: Get by email (full table scan!)
\`\`\`

### Solutions

**1. Phoenix Secondary Indexes**:
\`\`\`sql
CREATE INDEX email_idx ON users(email);
-- Now fast: SELECT * FROM users WHERE email = 'alice@example.com';
\`\`\`

**2. Manual Index Table**:
\`\`\`
Main table: user_id → user data
Index table: email → user_id
\`\`\`

**3. Denormalization**:
Store data multiple ways with different row keys

---

## Filters

### Server-Side Filtering

\`\`\`java
// SingleColumnValueFilter
SingleColumnValueFilter filter = new SingleColumnValueFilter(
    Bytes.toBytes("personal_info"),
    Bytes.toBytes("age"),
    CompareOperator.GREATER,
    Bytes.toBytes(25)
);
scan.setFilter(filter);

// RowFilter
RowFilter rowFilter = new RowFilter(
    CompareOperator.EQUAL,
    new SubstringComparator("user")
);

// FilterList (AND/OR multiple filters)
FilterList filterList = new FilterList(FilterList.Operator.MUST_PASS_ALL);
filterList.addFilter(filter1);
filterList.addFilter(filter2);
\`\`\`

**Benefits**: Reduce data transferred to client

---

## Coprocessors

### Server-Side Processing

**Coprocessors** = HBase's "stored procedures"

**Two types**:

**1. Observers** (like triggers):
\`\`\`java
public class AuditObserver implements RegionObserver {
    @Override
    public void prePut(...) {
        // Log before write
    }
}
\`\`\`

**2. Endpoints** (like stored procedures):
\`\`\`java
public class AggregationEndpoint {
    public long sum(column) {
        // Sum values server-side
        return total;
    }
}
\`\`\`

**Use cases**:
- Aggregations (SUM, COUNT, AVG)
- Data validation
- Audit logging
- Access control

---

## Best Practices

### 1. Row Key Design

- Avoid sequential keys (hot spots)
- Use hashed prefixes or salting
- Design for access patterns

### 2. Column Families

- Keep few column families (1-3 max)
- Group frequently accessed columns together
- Different compaction/compression per family

### 3. Pre-splitting

**Avoid hot spot during initial load**:
\`\`\`java
byte[][] splits = new byte[][] {
    Bytes.toBytes("row_1000"),
    Bytes.toBytes("row_2000"),
    Bytes.toBytes("row_3000")
};
admin.createTable(tableDescriptor, splits);
\`\`\`

### 4. Bloom Filters

Enable to reduce disk I/O:
\`\`\`java
columnDescriptor.setBloomFilterType(BloomType.ROW);
\`\`\`

### 5. Compression

Enable compression to save space:
\`\`\`java
columnDescriptor.setCompressionType(Compression.Algorithm.SNAPPY);
\`\`\`

### 6. Block Cache

Tune for read-heavy workloads:
\`\`\`
hbase.bucketcache.size = 8192  # MB
\`\`\`

---

## Use Cases

**1. Time-Series Data**:
- IoT sensor data
- Log aggregation
- Metrics storage

**Why HBase?**
- Efficient range scans (by time)
- Strong consistency
- Append-only workload

**2. Real-Time Analytics**:
- Click stream analysis
- User behavior tracking

**Why HBase?**
- Random reads and writes
- Large datasets
- Integration with Hadoop

**3. Content Storage**:
- Messages (Facebook Messenger)
- Emails
- Documents

**Why HBase?**
- Flexible schema
- Large objects
- Strong consistency

**Not suitable for**:
- ❌ Transactional workloads (ACID across rows)
- ❌ Small datasets (< 1 GB)
- ❌ Complex joins (use Phoenix or external tool)
- ❌ High availability critical (brief downtime during master failover)

---

## Monitoring

**Key metrics**:
\`\`\`
# RegionServer metrics
hbase.regionserver.requests  (read/write rate)
hbase.regionserver.memstore_size
hbase.regionserver.blockCacheHitRatio

# HMaster metrics
hbase.master.numRegionServers
hbase.master.numDeadRegionServers

# Region metrics
hbase.region.compactionQueueLength
hbase.region.flushQueueLength
\`\`\`

**Tools**:
- HBase Master UI (http://master:16010)
- RegionServer UI (http://regionserver:16030)
- Ganglia, Prometheus, Grafana

---

## Interview Tips

**Explain HBase in 2 minutes**:
"HBase is an open-source distributed database modeled after Google BigTable, built on HDFS. It provides random read/write access to large datasets with strong consistency. Data organized as wide-column store with row key, column families, and timestamps. Regions are ranges of rows served by RegionServers. Writes go to WAL and MemStore, then flushed to immutable HFiles. Compaction merges HFiles. ZooKeeper coordinates cluster and tracks HMaster. Row key design critical - avoid sequential keys to prevent hot spots. Phoenix adds SQL layer. Best for time-series data, real-time analytics, and large datasets requiring strong consistency."

**Key concepts**:
- Wide-column model (row key, column family, qualifier, timestamp)
- Regions and RegionServers
- Write path: WAL → MemStore → HFile
- Compaction (minor and major)
- Strong consistency (single RegionServer per row)
- Row key design (avoid hot spots)

**Common mistakes**:
- ❌ Sequential row keys (hot spots)
- ❌ Too many column families (performance hit)
- ❌ Not pre-splitting tables (hot spot during load)
- ❌ Treating like RDBMS (no joins, no transactions across rows)

---

## Key Takeaways

🔑 HBase = open-source BigTable on HDFS
🔑 Wide-column store with strong consistency
🔑 Regions = ranges of rows, unit of distribution
🔑 Write path: WAL → MemStore → HFile → Compaction
🔑 Row key design critical for performance (avoid hot spots)
🔑 HMaster assigns regions, RegionServers serve data, ZooKeeper coordinates
🔑 Phoenix adds SQL layer with secondary indexes
🔑 Strong consistency (vs Cassandra's eventual consistency)
🔑 Best for: time-series, real-time analytics, strong consistency needs
🔑 Integration with Hadoop ecosystem (MapReduce, Spark, Hive)
`,
};
