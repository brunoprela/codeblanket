/**
 * Google BigTable Section
 */

export const bigtableSection = {
  id: 'google-bigtable',
  title: 'Google BigTable',
  content: `Google BigTable is a pioneering distributed database that revolutionized wide-column storage and became the inspiration for HBase, Cassandra, and other NoSQL databases.

## Overview

**BigTable** (2006) = Google's distributed wide-column database

**What it powers**:
- Google Search (web index)
- Google Maps (geo data)
- Gmail (email storage)
- YouTube (video metadata)
- Google Analytics (event data)

**Scale**:
- Petabytes of data
- Thousands of servers
- Millions of operations per second
- Millisecond latency

**Key paper**: "Bigtable: A Distributed Storage System for Structured Data" (2006)

---

## Data Model

### Not a Relational Database!

BigTable is a **sparse, distributed, persistent multi-dimensional sorted map**

\`\`\`
(row key, column key, timestamp) → value
\`\`\`

**Example**: Web page storage

\`\`\`
Row key: "com.example.www"  (reversed domain for locality)
Column family: contents
  Column: contents:html  → "<html>...</html>"
  Column: contents:title → "Example Page"
Column family: anchor
  Column: anchor:cnnsi.com → "CNN Sports"
  Column: anchor:sports.com → "Sports News"
Timestamp: Multiple versions with timestamps
\`\`\`

### Key Concepts

**1. Row Key**:
- Unique identifier (up to 64 KB)
- **Lexicographically sorted**
- Data co-location by row key prefix
- Example: \`com.example.www\`, \`com.example.mail\`

**2. Column Families**:
- Group related columns
- Unit of access control
- Must be defined at table creation
- Example: \`contents\`, \`anchor\`, \`metadata\`

**3. Columns**:
- Within column families
- Can be created dynamically
- Format: \`family:qualifier\`
- Example: \`anchor:cnnsi.com\`, \`contents:html\`

**4. Timestamps**:
- Multiple versions per cell
- 64-bit timestamp (microseconds)
- Versions garbage collected automatically

**5. Cells**:
- Intersection of row, column, timestamp
- Uninterpreted byte array

### Example Table

\`\`\`
Table: webtable

Row: "com.cnn.www"
  contents:html              t3:  "<html>...</html>"
  contents:html              t2:  "<html>...</html>"  (old version)
  anchor:cnnsi.com           t2:  "CNN"

Row: "com.example.www"
  contents:html              t1:  "<html>...</html>"
  anchor:sports.com          t1:  "Sports"
  anchor:news.com            t1:  "News Site"
\`\`\`

**Key properties**:
- Rows sorted by key
- Sparse (empty cells don't take space)
- Each cell can have multiple timestamped versions

---

## BigTable Architecture

### High-Level Overview

\`\`\`
                    ┌──────────────┐
                    │   Master     │
                    │  (Metadata)  │
                    └──────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
  ┌──────────┐       ┌──────────┐      ┌──────────┐
  │ Tablet   │       │ Tablet   │      │ Tablet   │
  │ Server 1 │       │ Server 2 │      │ Server 3 │
  └──────────┘       └──────────┘      └──────────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                    ┌──────────────┐
                    │     GFS      │
                    │  (Storage)   │
                    └──────────────┘
                           │
                    ┌──────────────┐
                    │   Chubby     │
                    │   (Lock      │
                    │   Service)   │
                    └──────────────┘
\`\`\`

### Core Components

**1. Master Server**:
- Assigns tablets to tablet servers
- Detects tablet server failures
- Balances tablet server load
- Garbage collects files in GFS
- Handles schema changes

**2. Tablet Servers**:
- Manages 10-1000 tablets
- Handles read/write requests
- Splits tablets that grow too large
- Typically 100-1000 servers per cluster

**3. Client Library**:
- Talks directly to tablet servers for data
- Talks to master only for metadata
- Caches tablet locations

**4. Chubby (Lock Service)**:
- Distributed lock service (Paxos-based)
- Ensures one active master
- Stores root tablet location
- Tablet server lease management

**5. GFS (Google File System)**:
- Persistent storage for all data
- SSTables and logs stored in GFS

---

## Tablets

### What is a Tablet?

**Tablet** = Range of rows

Example table split into tablets:
\`\`\`
Tablet 1:  aaa...com.apple.www
Tablet 2:  com.cnn.www...com.google.www
Tablet 3:  com.google.www...zzz
\`\`\`

**Properties**:
- Default size: ~100-200 MB
- Splits when grows too large
- Merges when too small (rare)
- Unit of distribution and load balancing

### Three-Level Hierarchy

BigTable uses B+ tree-like structure:

\`\`\`
           Root Tablet (Chubby)
                  │
         ┌────────┴────────┐
         ▼                 ▼
    META Tablet 1     META Tablet 2
         │                 │
    ┌────┴────┐       ┌────┴────┐
    ▼         ▼       ▼         ▼
  User     User     User     User
 Tablet1  Tablet2  Tablet3  Tablet4
\`\`\`

**Level 1**: Chubby file with root tablet location
**Level 2**: META tablets (map: tablet location → user tablet)
**Level 3**: User tablets (actual data)

**Each META tablet can address**: ~128 MB / ~1KB per entry = ~128K tablets
**Total addressable**: 128K * 128K = ~2^34 tablets = enormous!

---

## SSTable

### Structure

**SSTable** (Sorted String Table) = Immutable, sorted key-value file

\`\`\`
SSTable file in GFS:
┌─────────────────────────┐
│  Index Block            │  (maps keys to block offsets)
├─────────────────────────┤
│  Data Block 1           │
│    key1 → value1        │
│    key2 → value2        │
├─────────────────────────┤
│  Data Block 2           │
│    key3 → value3        │
│    key4 → value4        │
├─────────────────────────┤
│  ...                    │
├─────────────────────────┤
│  Data Block N           │
└─────────────────────────┘
\`\`\`

**Properties**:
- Immutable (never modified after creation)
- Sorted by key
- Block-compressed
- Indexed for efficient lookups

**Operations**:
- **Read**: Binary search in index, read block from GFS
- **Write**: Create new SSTable (via compaction)
- **Delete**: Mark as deleted in new SSTable, garbage collect later

---

## Write Path (LSM-Tree)

BigTable uses **LSM-tree** (Log-Structured Merge-tree)

### Write Flow

\`\`\`
Write request
    ↓
1. Append to commit log (WAL) in GFS
    ↓
2. Insert into memtable (in-memory sorted buffer)
    ↓
3. Return success to client
    ↓
(When memtable full)
4. Freeze memtable → immutable memtable
5. Create new memtable
6. Write immutable memtable to SSTable in GFS
7. Delete old commit log
\`\`\`

**Example**:
\`\`\`
1. Client writes: row="user123", column="name:first", value="John"
2. Tablet server appends to commit log (for durability)
3. Tablet server inserts into memtable (sorted in memory)
4. Returns success to client
5. (Later) Memtable reaches 64 MB → flushed to SSTable
\`\`\`

**Why this design?**
- ✅ Fast writes (sequential log append + memory insert)
- ✅ Durability (commit log)
- ✅ No random disk writes

---

## Read Path

### Read Flow

\`\`\`
Read request
    ↓
1. Check memtable (most recent writes)
    ↓
2. If not found, check immutable memtables
    ↓
3. If not found, check SSTables (newest to oldest)
    ↓
4. Merge results (handle timestamps, deletions)
    ↓
5. Return to client
\`\`\`

**Example**:
\`\`\`
Read: row="user123", column="name:first"

1. Check memtable: Not found
2. Check SSTable-3 (newest): Found "John" at t=1000
3. Check SSTable-2: Found "Jon" at t=500 (older, ignore)
4. Return "John"
\`\`\`

**Optimization**: Bloom filters
- Each SSTable has Bloom filter
- Quickly determine "definitely not present" without disk I/O
- Avoids unnecessary SSTable reads

---

## Compaction

### Why Compaction?

**Problem**: Over time, many SSTables accumulate
- Reads become slow (check many files)
- Disk space wasted (old versions, deleted cells)

**Solution**: Periodic compaction

### Types of Compaction

**1. Minor Compaction**:
- Memtable → SSTable
- Reduces memory usage
- Frequent (every few minutes)

**2. Major Compaction**:
- Merges multiple SSTables into one
- Removes deleted entries and old versions
- Reclaims disk space
- Infrequent (daily or weekly)

### Compaction Process

\`\`\`
SSTable-1 (0-100)     SSTable-2 (0-100)     SSTable-3 (0-100)
    ↓                      ↓                      ↓
        Merge sort (keeping only latest versions)
                           ↓
              New SSTable-4 (0-100, compacted)
                           ↓
         Delete SSTable-1, SSTable-2, SSTable-3
\`\`\`

**Benefits**:
- ✅ Faster reads (fewer files to check)
- ✅ Reclaimed disk space
- ✅ Better performance

**Cost**:
- ❌ I/O and CPU overhead during compaction
- ❌ Can impact foreground operations

---

## Locality Groups

**Optimization**: Group related column families

\`\`\`
Table: webtable
Locality Group 1 (frequently accessed):
  - contents:html
  - contents:title
Locality Group 2 (rarely accessed):
  - anchor:*
\`\`\`

**Benefits**:
- ✅ Separate SSTables per locality group
- ✅ Read only needed data
- ✅ Different compression for each group

**Example**: Compress text heavily, compress images lightly

---

## Compression

**Block-level compression** in SSTables:

**Compression algorithms**:
- BMDiff (for similar data, like web pages)
- Zippy (fast compression/decompression)

**Two-pass compression**:
1. Compress similar values together (BMDiff)
2. Compress blocks (Zippy)

**Results**: 10:1 compression ratio for web data!

---

## Bloom Filters

**Problem**: Checking if key exists requires reading SSTable from disk

**Solution**: Bloom filter (probabilistic data structure)

\`\`\`
Bloom Filter for SSTable-1:
- Size: 10 KB (in memory)
- False positive rate: 1%
- Can definitively say "key NOT present"
- Might falsely say "key present" (1% of time)
\`\`\`

**Usage**:
\`\`\`
1. Check Bloom filter: Key might be in SSTable-1
2. Read SSTable-1 from disk
3. Either find key or confirm false positive
\`\`\`

**Benefits**: Avoids 99% of unnecessary disk reads!

---

## Caching

**Two-level caching**:

**1. Scan Cache**:
- Caches (key, value) pairs
- Good for repeated access to same data

**2. Block Cache**:
- Caches SSTable blocks
- Good for sequential scans

**LRU eviction** for both caches

---

## Tablet Location

### How Client Finds Tablet

\`\`\`
Client wants to read row="com.example.www"

1. Check cache: Is location cached?
   ↓ (cache miss)
2. Read root tablet location from Chubby
   ↓
3. Read META tablet to find user tablet location
   ↓
4. Cache location
   ↓
5. Contact tablet server directly

(Future reads: Use cached location!)
\`\`\`

**Three network round trips** worst case (first access)
**Zero round trips** best case (cached location)

---

## Tablet Assignment

### Master's Role

**Tablet lifecycle**:

**1. Tablet Server Registration**:
- Tablet server acquires lease in Chubby
- Master monitors Chubby directory
- Master detects new servers

**2. Tablet Assignment**:
- Master assigns unassigned tablets to servers
- Master sends "load tablet" request
- Tablet server loads tablet metadata from META

**3. Tablet Serving**:
- Tablet server handles read/write requests
- Periodically renews Chubby lease

**4. Tablet Server Failure**:
- Chubby lease expires (no heartbeat)
- Master detects failure
- Master reassigns tablets to other servers

**5. Master Failure**:
- New master elected via Chubby
- New master scans META tablets
- Rebuilds tablet assignment

---

## Failure Handling

### Tablet Server Failure

\`\`\`
Tablet Server crashes
    ↓
Chubby lease expires
    ↓
Master detects failure
    ↓
Master marks tablets as unassigned
    ↓
Master reassigns to other servers
    ↓
New server loads tablet from GFS
    ↓
Replays commit log
    ↓
Tablet available again
\`\`\`

**Recovery time**: Seconds to minutes

### Master Failure

- New master elected via Chubby
- Scans Chubby for live tablet servers
- Scans META tablets for tablet assignments
- Rebuilds in-memory state
- Resumes operations

**Downtime**: Typically < 10 seconds

### Network Partition

- Tablet servers lose connection to Chubby
- Cannot renew lease
- Master considers them dead
- Tablets reassigned
- Old tablet server stops serving (cannot renew lease)

---

## BigTable Performance

### Typical Numbers (from paper)

**Random reads**: 1000 ops/sec per server
**Random writes**: 100K ops/sec per server (batched)
**Sequential scans**: 600 MB/s per server

**Latency**:
- Read: ~10ms (95th percentile)
- Write: ~10ms (95th percentile)

### Optimizations

**1. Batching**:
- Group multiple writes
- Amortize commit log overhead

**2. Bloom filters**:
- Reduce disk reads

**3. Compression**:
- Reduce I/O

**4. Locality groups**:
- Read only needed columns

**5. Caching**:
- Block cache + Scan cache

---

## Use Cases

### 1. Google Search (Web Index)

\`\`\`
Table: webtable
Row key: Reversed URL (com.google.www)
Column families:
  - contents:html (page content)
  - anchor:* (backlinks)
  - pagerank
\`\`\`

**Why BigTable?**
- Massive scale (billions of pages)
- Sparse (not all columns present for all pages)
- Sorted access by URL

### 2. Google Earth (Map Tiles)

\`\`\`
Table: map-tiles
Row key: (lat, lon, zoom)
Column: tile image
\`\`\`

**Why BigTable?**
- Billions of tiles
- Fast lookups by coordinates
- Easy replication

### 3. Google Analytics (Events)

\`\`\`
Table: events
Row key: (customer, timestamp)
Column families:
  - event:type
  - event:data
  - user:id
\`\`\`

**Why BigTable?**
- High write rate (millions of events/sec)
- Time-series data (sorted by timestamp)
- Efficient range scans

---

## BigTable Impact and Legacy

### Inspired NoSQL Systems

**HBase** (open-source BigTable):
- Nearly identical design
- Runs on HDFS instead of GFS
- Uses ZooKeeper instead of Chubby

**Cassandra** (Facebook → Apache):
- Wide-column model from BigTable
- Masterless architecture from Dynamo

**Many others**:
- Hypertable
- Accumulo
- Cloud Bigtable (Google Cloud managed service)

### Key Innovations

**1. LSM-tree for writes**:
- Fast writes via memtable + commit log
- Compaction for read optimization

**2. SSTable format**:
- Immutable, sorted, compressed
- Efficient storage and retrieval

**3. Wide-column data model**:
- Flexible schema
- Sparse data support
- Column families for access control

**4. Separation of compute and storage**:
- Tablet servers (compute) separate from GFS (storage)
- Enables independent scaling

---

## Interview Tips

**Explain BigTable in 2 minutes**:
"BigTable is Google's distributed wide-column database. Data is organized as a sparse, sorted map: (row key, column, timestamp) → value. Tablets are ranges of rows served by tablet servers. Writes go to memtable (in memory) and commit log (in GFS) for durability, then memtable is flushed to immutable SSTables. Reads check memtable first, then SSTables. Bloom filters avoid unnecessary disk reads. Compaction merges SSTables to improve read performance. Master assigns tablets to servers, Chubby provides coordination. It powers Google Search, Maps, Gmail, Analytics."

**Key concepts to mention**:
- Wide-column data model
- LSM-tree (memtable + SSTable)
- Tablets and tablet servers
- Compaction (minor and major)
- Bloom filters
- Chubby for coordination
- GFS for storage

**Common interview questions**:
- How do writes work in BigTable? (Memtable + commit log)
- Why use SSTables? (Immutable, sorted, compressed)
- How do Bloom filters help? (Avoid unnecessary disk reads)
- What is compaction? (Merge SSTables, remove old versions)

---

## Key Takeaways

🔑 BigTable = distributed wide-column database at Google
🔑 Data model: (row, column, timestamp) → value
🔑 LSM-tree: Fast writes via memtable, reads from memtable + SSTables
🔑 SSTables are immutable, sorted, compressed files in GFS
🔑 Compaction merges SSTables for better read performance
🔑 Bloom filters reduce unnecessary disk reads
🔑 Tablets = ranges of rows, unit of distribution
🔑 Master assigns tablets, Chubby coordinates, GFS stores data
🔑 Powers Google Search, Maps, Gmail, Analytics at massive scale
🔑 Inspired HBase, Cassandra, and many NoSQL databases
`,
};
