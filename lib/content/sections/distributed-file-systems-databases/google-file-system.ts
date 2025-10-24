/**
 * Google File System (GFS) Section
 */

export const gfsSection = {
  id: 'google-file-system',
  title: 'Google File System (GFS)',
  content: `Google File System (GFS) revolutionized distributed storage and became the foundation for modern distributed file systems. Published in 2003, it powers Google's infrastructure and inspired Hadoop's HDFS.

## Why GFS Was Created

### The Problem Google Faced

**Scale**: Store petabytes of data across thousands of commodity machines
- Web crawl data: Billions of web pages
- Search index: Massive inverted indexes
- MapReduce outputs: Terabytes of intermediate data
- User data: Gmail, YouTube (later)

**Traditional file systems failed**:
- ❌ NAS/SAN too expensive at Google's scale
- ❌ Traditional distributed FS designed for small scale
- ❌ Needed to handle **component failures as the norm, not exception**
- ❌ Optimized for different workload (small files, random access)

### Google's Unique Requirements

1. **Component failures are common**: With 1000s of machines, something is always failing
2. **Huge files**: Multi-GB files are the norm (not KB/MB files)
3. **Append-heavy workload**: Most writes are appends, not random writes
4. **Sequential reads**: Most reads scan large portions of files
5. **High sustained bandwidth** > Low latency: Batch processing priority

---

## GFS Architecture

### Three Key Components

\`\`\`
                    ┌──────────────┐
                    │   Master     │  (single, but replicated)
                    │  (Metadata)  │
                    └──────────────┘
                           │
                           │ Control messages (metadata)
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
  ┌──────────┐       ┌──────────┐      ┌──────────┐
  │ Chunk    │       │ Chunk    │      │ Chunk    │
  │ Server 1 │       │ Server 2 │      │ Server 3 │
  │ (Data)   │       │ (Data)   │      │ (Data)   │
  └──────────┘       └──────────┘      └──────────┘
        ▲                  ▲                  ▲
        │ Data flow        │                  │
        │ (no master)      │                  │
        └──────────────────┼──────────────────┘
                           │
                    ┌──────────────┐
                    │   Client     │
                    └──────────────┘
\`\`\`

### 1. GFS Master (Single Master)

**Responsibilities**:
- **Namespace management**: Directory hierarchy, file-to-chunk mapping
- **Chunk location**: Which chunkservers hold which chunks
- **Chunk lease management**: Grant write leases (primary replica)
- **Garbage collection**: Delete orphaned chunks
- **Chunk migration**: Balance load across chunkservers

**Metadata stored in memory** (fast access):
1. File namespace (directory structure)
2. File → Chunk mapping
3. Chunk → Chunkserver location (built at startup via heartbeat)

**Persistent state** (on disk, replicated):
- Operation log (journal) - mutations to namespace/file-to-chunk mapping
- Checkpoints (snapshots of state)

**Why single master?**
- ✅ Simplified design - no distributed consensus needed
- ✅ Fast metadata operations (all in memory)
- ✅ Easy to maintain consistency
- ❌ Potential bottleneck (mitigated by keeping master out of data path)
- ❌ Single point of failure (mitigated by replication + shadow masters)

### 2. Chunkservers (Data Storage)

**Chunk storage**:
- Files divided into **64 MB chunks** (unusually large!)
- Each chunk identified by unique 64-bit chunk handle
- Chunks stored as regular Linux files
- Each chunk replicated across **3 chunkservers** (default)

**Why 64 MB chunks?**
- ✅ Reduces metadata size (fewer chunks = less memory at master)
- ✅ Reduces client-master communication (one request covers more data)
- ✅ Allows keeping TCP connections open longer
- ❌ Wastes space for small files (but Google has few small files)
- ❌ Hot spots for popular small files (mitigated by higher replication)

**Chunkserver responsibilities**:
- Store chunks as Linux files
- Serve read/write requests
- No caching (Linux buffer cache does it)
- Heartbeat to master
- Chunk checksum verification

### 3. GFS Client

**Client library** (linked into applications):
- Translates file operations to GFS protocol
- Communicates with master and chunkservers
- No POSIX API - custom API optimized for GFS workload

**Client caching**:
- Caches metadata (from master)
- Does NOT cache data (files too large, Linux cache sufficient)

---

## GFS Read Flow

**Goal**: Read 1 GB file starting at offset 2 GB

\`\`\`
Client                Master              Chunkservers
  |                      |                      |
  |--- (1) Lookup ------>|                      |
  |  file, offset        |                      |
  |                      |                      |
  |<-- (2) Chunk IDs ----|                      |
  |  + Locations         |                      |
  |  [C100: S1,S2,S3]    |                      |
  |  [C101: S2,S4,S5]    |                      |
  |                      |                      |
  |--- (3) Read C100 ----|-------------------->|
  |  from S1 (closest)   |                     S1
  |                      |                      |
  |<-- (4) Data ---------|---------------------|
  |                      |                      |
\`\`\`

**Steps**:

1. **Client → Master**: "Give me chunks for file X starting at offset Y"
   - Master calculates chunk indexes
   - Returns chunk handles + locations (ordered by network distance)

2. **Client caches** this metadata

3. **Client → Chunkserver**: Directly requests data from closest chunkserver
   - Master NOT involved in data transfer!
   - Client can specify offset + length within chunk

4. **Chunkserver → Client**: Returns data
   - Verifies checksum before sending
   - Client verifies checksum after receiving

**Key insight**: Master only involved in (1). All data flows directly between client and chunkservers.

---

## GFS Write Flow

**Two types**:
1. **Append** (most common) - add data to end of file
2. **Random write** (rare) - write at specific offset

### Record Append Flow (Common Case)

\`\`\`
Client                Master              Chunkservers
  |                      |                      |
  |--- (1) Append ------>|                      |
  |  to file X           |                      |
  |                      |                      |
  |<-- (2) Last chunk ---|                      |
  |  + Locations         |                      |
  |  + Primary lease     |                      |
  |                      |                      |
  |--- (3) Push data ----|-------------------->|
  |  to all replicas     |                  S1,S2,S3
  |  (pipelined)         |                      |
  |                      |                      |
  |--- (4) Write cmd ----|-------------------->|
  |  to primary          |                  S1 (primary)
  |                      |                      |
  |                   S1 assigns serial order   |
  |                   S1 forwards to S2, S3     |
  |                      |                      |
  |<-- (5) Success ------|---------------------|
  |  or retry            |                      |
\`\`\`

**Detailed steps**:

**Step 1: Find chunk and lease**
- Client asks master: "Where is last chunk of file X?"
- If no lease exists, master grants lease to one replica (becomes primary)
- Master returns: chunk handle, locations, which is primary

**Step 2: Client pushes data to all replicas**
- Client pushes data to ALL replicas (primary + secondaries)
- Data pushed **linearly** in pipeline (not from client to all)
- Each chunkserver forwards to next in chain
- Data stored in internal LRU buffer (not yet written to chunk file)

**Step 3: Client sends write command to primary**
- After all replicas ACK data receipt
- Primary assigns consecutive serial numbers to all mutations
- Ensures deterministic order across all replicas

**Step 4: Primary forwards write command to secondaries**
- Secondaries apply mutations in the same order
- Each secondary ACKs to primary

**Step 5: Primary replies to client**
- Success: All replicas applied mutation
- Failure: Some replicas failed → inconsistent state
- Client retries from step 2 (pushes data again)

### Lease Mechanism

**Lease properties**:
- 60-second timeout (extended via heartbeats)
- Primary holds lease = has authority to order mutations
- Prevents split-brain (two primaries)
- Master can wait for lease expiry before granting new lease

**Why leases?**
- Master doesn't need to track every write
- Primary can order writes without master coordination
- Reduces master bottleneck

---

## Consistency Model

GFS provides **relaxed consistency** - NOT strict consistency!

### Consistency Guarantees

**File namespace mutations** (create file, delete) are **atomic and serializable**:
- Master holds all namespace in memory
- Uses operation log for atomicity
- File creation is atomic

**Data mutations** have weaker guarantees:

| Operation      | Concurrent Writers | Guarantee                        |
|----------------|-------------------|----------------------------------|
| Write          | No                | Consistent and defined           |
| Write          | Yes               | Consistent but undefined         |
| Record Append  | No or Yes         | Defined but possibly inconsistent|

**Defined**: Client sees exactly what mutation wrote
**Consistent**: All clients see same data
**Undefined**: Clients see same data but not necessarily what any mutation wrote

### Why Weak Consistency?

**GFS prioritizes**: Availability + Performance > Strict consistency

**Google's workload allows this**:
- Most apps do append-only writes (log files, MapReduce output)
- Readers can detect inconsistency via checksums and application-level markers
- Applications designed to tolerate inconsistency

**Example**: MapReduce writes
- Each mapper appends records to output file
- Records may be duplicated or out of order
- MapReduce framework handles this (deduplication in reduce phase)

---

## Consistency in Practice

### Record Append Semantics

**At-least-once semantics**: Record guaranteed to be appended at least once
- May be duplicated if retry happens
- May have padding or gaps (if append doesn't fit in current chunk)

**Application responsibility**:
- Include checksums in records
- Include unique IDs for deduplication
- Filter out padding records

**Example**:
\`\`\`
Chunk:
[Record A][Record B][Padding][Record C][Record A duplicate][Record D]
\`\`\`

Reader must:
1. Verify checksums
2. Deduplicate based on record ID
3. Skip padding

---

## Fault Tolerance

### Handling Chunkserver Failure

**Detection**:
- Master expects heartbeat every few seconds
- Missing heartbeat → chunkserver assumed dead

**Recovery**:
1. Master identifies under-replicated chunks
2. Prioritizes chunks with < 2 replicas (critical)
3. Orders chunkservers to re-replicate chunks
4. New replica created by copying from existing replica

### Handling Master Failure

**Master state is persistent**:
- Operation log replicated to multiple remote machines
- Checkpoints created periodically
- State can be reconstructed from log + checkpoint

**Recovery**:
1. Master restarts and loads checkpoint
2. Replays operation log
3. Polls chunkservers for chunk locations
4. Resumes operation

**Shadow masters**:
- Read-only replicas of master
- Lag slightly behind primary
- Can serve read-only requests
- Can take over if primary dies (with brief unavailability)

### Data Integrity

**Checksumming**:
- Each 64 KB block has 32-bit checksum
- Checksum stored separately from data
- Verified on every read
- Verified periodically in background

**Corruption detection**:
1. Chunkserver detects corrupt chunk (checksum mismatch)
2. Chunkserver informs master
3. Master triggers re-replication from good replica
4. Corrupt chunk deleted after re-replication

---

## GFS in Action: Real-World Usage

### Google Search

**Crawl data storage**:
- Web crawler writes crawled pages to GFS
- Each page appended to large file
- Batch processing reads all pages

**Index building**:
- MapReduce reads crawl data from GFS
- Processes and builds inverted index
- Writes index back to GFS
- Index served to search frontend

### Gmail (Early Days)

**Email storage**:
- Each email appended to user's log file in GFS
- Metadata stored in Bigtable (which uses GFS)
- Attachments stored as chunks in GFS

**Why GFS for email?**
- Immutable append-only writes (email never changes)
- High durability (3x replication)
- Seamless scaling as users grew

---

## Garbage Collection

**Lazy deletion**:
- When file deleted, master logs deletion
- File renamed to hidden name with deletion timestamp
- Actual deletion happens during background scan (3 days later)

**Benefits**:
- Simple and robust (no complex distributed deletion protocol)
- Deletion batched and spread over time
- Easy to undelete (during 3-day window)

**Chunk garbage collection**:
- Master maintains all chunk references
- During periodic scan, finds orphaned chunks
- Orphaned chunks removed from chunkservers

---

## Chunk Replication Strategy

**Placement goals**:
1. Spread across different racks (rack failure tolerance)
2. Utilize disk bandwidth across all chunkservers
3. Minimize write traffic across racks (expensive)

**Creation**:
- Below-average disk utilization
- Recent creations limited per chunkserver (avoid write hotspot)
- Across racks

**Re-replication**:
- Prioritized by replication factor
- Chunk with 1 replica > chunk with 2 replicas
- Prioritize live files over deleted files

**Rebalancing**:
- Master periodically examines distribution
- Moves replicas for better disk utilization and load

---

## Master Operation Log

**Critical for durability**:
- Defines **persistent record of metadata changes**
- Replicated to multiple remote machines
- Master responds to client only after log flushed to disk (local + remote)

**Log structure**:
\`\`\`
1: CREATE /users/john/data.txt → chunk C100
2: APPEND /users/john/data.txt → chunk C101
3: DELETE /users/jane/old.txt
4: RENAME /users/bob/temp → /users/bob/final
...
\`\`\`

**Checkpoints**:
- Compact checkpoint created when log grows large
- Checkpoint = snapshot of master state
- Recovery = load checkpoint + replay log since checkpoint
- Creation done in background (new checkpoint, switch, delete old)

---

## Limitations and Lessons

### Limitations

**1. Single master bottleneck**:
- Large namespace = lots of metadata = memory pressure
- Many clients = high metadata request rate
- Solution: Shadow masters, metadata sharding (later systems)

**2. Weak consistency model**:
- Complex for application developers
- Not suitable for POSIX workloads (random writes)
- Later systems (Colossus) improved this

**3. Small file inefficiency**:
- 64 MB chunks waste space for small files
- Hot small files become hot spots

**4. Master/chunkserver protocol complexity**:
- Many edge cases
- Replication, re-replication, balancing logic intricate

### Lessons Learned

✅ **Component failures are normal**: Design for failure
✅ **Single master simplifies design**: Worth the trade-off for many workloads
✅ **Large chunks reduce metadata**: Good for large file workloads
✅ **Separating data and control flow**: Master out of data path = scalability
✅ **Application-level recovery**: Push some consistency responsibility to apps
✅ **Checksumming is mandatory**: Data corruption is real

---

## GFS Successor: Colossus

Google replaced GFS with **Colossus** (not publicly documented):

**Improvements**:
- Distributed master (no single master bottleneck)
- Automatic sharding of metadata
- Better small file performance
- Stronger consistency guarantees
- Better integration with Google infrastructure

**GFS legacy**: Inspired HDFS, Ceph, and many other distributed file systems

---

## Interview Tips

**Explain GFS in 2 minutes**:
"GFS is Google's distributed file system for storing petabytes. It has a single master holding metadata in memory, and thousands of chunkservers storing 64 MB chunks with 3x replication. The key insight is separating control (master) from data flow (direct client-to-chunkserver). It's optimized for large files and append-heavy workloads. The weak consistency model is acceptable for Google's batch processing. Master failure is handled by replication and operation log. It inspired HDFS and modern distributed storage."

**Common mistakes**:
- ❌ Saying "master is single point of failure" without mentioning replication
- ❌ Not explaining why 64 MB chunks
- ❌ Thinking it provides strong consistency
- ❌ Not mentioning data flows directly between client and chunkserver

**Key trade-offs to discuss**:
- Single master: Simplicity vs scalability
- 64 MB chunks: Reduced metadata vs small file inefficiency
- Weak consistency: Performance + availability vs application complexity
- Lazy deletion: Simplicity + recoverability vs delayed space reclamation

---

## Key Takeaways

🔑 GFS pioneered modern distributed file systems
🔑 Single master + many chunkservers = simple but effective
🔑 64 MB chunks = less metadata, good for large files
🔑 3x replication + checksums = durability and integrity
🔑 Weak consistency = performance + availability at cost of complexity
🔑 Master out of data path = no bottleneck for data transfer
🔑 Designed for Google's workload: large files, appends, batch processing
🔑 Influenced all modern distributed storage systems
`,
};
