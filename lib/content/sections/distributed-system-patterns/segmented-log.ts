/**
 * Segmented Log Section
 */

export const segmentedlogSection = {
  id: 'segmented-log',
  title: 'Segmented Log',
  content: `Segmented Log is a pattern where a single large log file is broken into multiple smaller segments. This pattern is essential for managing large-scale logs in distributed systems and is used extensively in systems like Apache Kafka, Cassandra, and various distributed databases.

## What is a Segmented Log?

A **Segmented Log** divides a potentially infinite append-only log into **multiple fixed-size or time-bounded segments**.

Instead of:
\`\`\`
log.dat (growing indefinitely)
\`\`\`

You have:
\`\`\`
segment-00000.dat (100MB, closed)
segment-00001.dat (100MB, closed)
segment-00002.dat (100MB, closed)
segment-00003.dat (50MB, active)
\`\`\`

**Active Segment**: Currently being written to
**Closed Segments**: Immutable, no longer accepting writes

---

## Why Segment Logs?

### **1. Easier Deletion/Cleanup**

**Problem**: With single large log, can't delete old entries without rewriting entire file.

**Solution**: Delete entire old segments.

\`\`\`
Without Segmentation:
  log.dat (10GB)
  Want to delete entries older than 7 days:
    → Must rewrite entire 10GB file ❌

With Segmentation:
  segment-00000.dat (100MB, 30 days old)
  segment-00005.dat (100MB, 8 days old)
  segment-00010.dat (100MB, 6 days old) ← Keep from here
  
  Delete segments 0-9:
    → Simple file deletion ✅
\`\`\`

**Kafka Example**: Retaining only last 7 days of messages—just delete segments older than 7 days.

### **2. Efficient Compaction**

**Problem**: Log may contain redundant or outdated entries.

**Solution**: Compact individual segments without affecting active writes.

\`\`\`
Segment 0:
  key=A, value=1
  key=B, value=2
  key=A, value=3  ← Latest value for A

Compacted Segment 0:
  key=B, value=2
  key=A, value=3  ← Only keep latest
\`\`\`

**Used by**: Kafka log compaction, RocksDB, Cassandra

### **3. Faster Recovery**

**Problem**: Reading single 100GB log file on recovery is slow.

**Solution**: Parallel recovery—read multiple segments concurrently.

\`\`\`
Single Log:
  Recovery: Read 100GB sequentially (single thread)
  Time: 1000 seconds

Segmented (10 x 10GB):
  Recovery: Read 10 segments in parallel (10 threads)
  Time: 100 seconds
\`\`\`

**Improvement**: 10x faster recovery

### **4. Better I/O Performance**

**Problem**: Single large file causes:
- File descriptor contention
- OS-level caching issues
- Difficult to optimize for access patterns

**Solution**: Segments enable:
- Separate caching policies per segment
- Concurrent reads from different segments
- Hot segments in memory, cold on disk

### **5. Simplified Archival**

**Problem**: How to archive old data to cheaper storage?

**Solution**: Move entire closed segments to S3, Glacier, or tape.

\`\`\`
Hot Tier (SSD):
  segment-00010.dat to segment-00015.dat (recent)

Warm Tier (HDD):
  segment-00005.dat to segment-00009.dat (older)

Cold Tier (S3):
  segment-00000.dat to segment-00004.dat (archive)
\`\`\`

**Tiered Storage**: Different performance/cost for different ages of data.

---

## How Segmented Logs Work

### **Basic Structure**

\`\`\`
Log Directory:
  /var/log/myapp/
    segment-00000.dat       (100MB, closed)
    segment-00000.index     (Index for segment-00000)
    segment-00001.dat       (100MB, closed)
    segment-00001.index
    segment-00002.dat       (50MB, active)
    segment-00002.index     (Index being built)
    active-segment.ptr      (Pointer to current active segment)
\`\`\`

**Segment File**: Contains actual log entries
**Index File**: Maps keys/offsets to positions in segment (for fast lookups)

### **Writing to Segmented Log**

\`\`\`
1. Append entry to active segment
2. Update index (if key-based)
3. Check segment size:
   - If < threshold (e.g., 100MB):
     → Continue writing
   - If ≥ threshold:
     → Close current segment
     → Create new active segment
     → Start writing to new segment
\`\`\`

**Example**:
\`\`\`
Active: segment-00002.dat (99.8 MB)

Write entry (0.5 MB):
  → Total would be 100.3 MB (exceeds 100 MB limit)
  
Actions:
  1. Close segment-00002.dat
  2. Finalize segment-00002.index
  3. Create segment-00003.dat
  4. Write entry to segment-00003.dat
\`\`\`

### **Reading from Segmented Log**

**By Offset**:
\`\`\`
Read offset 1,500,000:

1. Determine which segment contains offset:
   segment-00000: offsets 0 - 999,999
   segment-00001: offsets 1,000,000 - 1,999,999 ← This one
   segment-00002: offsets 2,000,000 - ...

2. Open segment-00001.dat
3. Use segment-00001.index to find position
4. Read entry from position
\`\`\`

**By Key** (with index):
\`\`\`
Read key "user:12345":

1. Check most recent segments first (likely to have latest value)
2. For each segment (newest to oldest):
   - Check index for key
   - If found: Read from segment
   - If not found: Check next segment
3. Return value or "not found"
\`\`\`

### **Index Structure**

**Purpose**: Fast lookup without scanning entire segment.

**Index Entry**:
\`\`\`
{
  key: "user:12345",          (or offset: 1500000)
  position: 45678,            Byte position in segment file
  timestamp: 1704067200000,   Optional, for time-based queries
}
\`\`\`

**Index Types**:

1. **Offset Index**: Maps offset → position
   \`\`\`
   Offset 1,000,000 → Position 0
   Offset 1,000,100 → Position 15000
   Offset 1,000,200 → Position 30000
   \`\`\`

2. **Key Index**: Maps key → position (sparse)
   \`\`\`
   "user:10000" → Position 0
   "user:10050" → Position 15000
   \`\`\`

3. **Timestamp Index**: Maps timestamp → offset
   \`\`\`
   Timestamp 1704067200 → Offset 1,000,000
   Timestamp 1704067260 → Offset 1,000,500
   \`\`\`

**Sparse Index**: Not every entry indexed, only periodic entries (e.g., every 4KB)
- Reduces index size
- Binary search in index, then linear scan in segment

---

## Segment Rotation

**Triggers for Creating New Segment**:

### **1. Size-Based**
\`\`\`
if segment.size >= max_segment_size:
    rotate()
\`\`\`

**Common sizes**: 100MB, 1GB, 10GB

**Trade-offs**:
- Small segments: More files, faster cleanup, better parallelism
- Large segments: Fewer files, less overhead, slower cleanup

### **2. Time-Based**
\`\`\`
if time_since_segment_created >= max_segment_age:
    rotate()
\`\`\`

**Common durations**: 1 hour, 1 day, 1 week

**Use case**: Ensure segments align with retention policy (e.g., daily segments for "keep 30 days")

### **3. Record-Count-Based**
\`\`\`
if segment.record_count >= max_records_per_segment:
    rotate()
\`\`\`

**Common counts**: 1 million, 10 million

**Use case**: Predictable segment sizes for fixed-size records

### **Combined Triggers**
\`\`\`
if segment.size >= max_size OR 
   time_since_creation >= max_age OR
   record_count >= max_records:
    rotate()
\`\`\`

**Best practice**: Use multiple triggers to handle various load patterns.

---

## Retention Policies

How long to keep segments?

### **Time-Based Retention**
\`\`\`
retention_policy = 7 days

Delete segments where:
  segment.last_modified < (now - 7 days)
\`\`\`

**Kafka Example**: \`log.retention.hours = 168\` (7 days)

### **Size-Based Retention**
\`\`\`
retention_policy = 100 GB total

if total_size > 100 GB:
    delete oldest segments until total_size <= 100 GB
\`\`\`

**Kafka Example**: \`log.retention.bytes = 107374182400\` (100GB)

### **Hybrid Retention**
\`\`\`
Keep whichever is more restrictive:
  - At least 7 days of data
  - At most 100 GB of data
\`\`\`

### **Infinite Retention (with Compaction)**
\`\`\`
Never delete segments, but compact them:
  - Remove duplicate keys (keep latest)
  - Remove tombstones after grace period
  - Keep log bounded by unique keys (not time)
\`\`\`

**Kafka Example**: \`log.cleanup.policy = compact\`

---

## Log Compaction

**Goal**: Retain only the latest value for each key.

### **Compaction Process**

**Before Compaction**:
\`\`\`
Segment 1:
  key=A, value=1, offset=0
  key=B, value=2, offset=1
  key=A, value=3, offset=2

Segment 2:
  key=C, value=4, offset=3
  key=B, value=5, offset=4
  key=A, value=6, offset=5
\`\`\`

**After Compaction**:
\`\`\`
Compacted Segment:
  key=C, value=4, offset=3
  key=B, value=5, offset=4
  key=A, value=6, offset=5
\`\`\`

**Result**: Only latest value for each key retained.

### **Kafka Log Compaction**

Kafka\'s compaction ensures:
- Latest value for each key is always available
- Old values are eventually removed
- Log size bounded by number of unique keys (not time)

**Use Cases**:
- Change Data Capture (CDC)
- Database table as Kafka topic
- Materialized views
- Caching invalidation

**Configuration**:
\`\`\`
log.cleanup.policy=compact
min.cleanable.dirty.ratio=0.5    # Trigger compaction
delete.retention.ms=86400000     # Keep tombstones for 1 day
\`\`\`

### **Tombstones**

**Deletion in Compacted Log**: Set value to null (tombstone).

\`\`\`
key=A, value=1    (original)
key=A, value=null (deletion, tombstone)

After compaction:
  key=A entry completely removed
\`\`\`

**Grace Period**: Keep tombstones for some time (ensure all consumers see deletion).

---

## Segmented Log in Real-World Systems

### **Apache Kafka**

Kafka topics are segmented logs.

**Structure**:
\`\`\`
/var/kafka-logs/topic-0/
  00000000000000000000.log       (1GB)
  00000000000000000000.index
  00000000000000000000.timeindex
  00000000001000000000.log       (1GB)
  00000000001000000000.index
  00000000001000000000.timeindex
  00000000002000000000.log       (500MB, active)
  00000000002000000000.index
\`\`\`

**File naming**: Starting offset of segment (e.g., 00000000001000000000 = offset 1,000,000,000)

**Configuration**:
\`\`\`
log.segment.bytes=1073741824        # 1GB per segment
log.segment.ms=604800000            # 7 days max age
log.retention.hours=168             # Keep 7 days
log.retention.bytes=107374182400    # Or 100GB total
log.cleanup.policy=delete           # Or "compact"
\`\`\`

**Performance**: 
- Sequential writes to active segment (very fast)
- Can serve millions of messages per second
- Segments can be read in parallel by different consumers

### **Apache Cassandra**

Cassandra uses **SSTables** (Sorted String Tables), which are immutable segmented files.

**Write Path**:
\`\`\`
1. Write to CommitLog (WAL, segmented)
2. Write to Memtable (in-memory)
3. Memtable full → Flush to SSTable (new segment)
4. Over time: Compact SSTables (merge segments)
\`\`\`

**SSTable Structure**:
\`\`\`
/var/cassandra/data/keyspace/table/
  mc-1-big-Data.db           (10MB, immutable)
  mc-1-big-Index.db
  mc-1-big-Summary.db
  mc-2-big-Data.db           (10MB, immutable)
  mc-2-big-Index.db
  mc-3-big-Data.db           (50MB, result of compaction)
\`\`\`

**Compaction Strategies**:
- **SizeTieredCompactionStrategy (STCS)**: Merge similar-sized SSTables
- **LeveledCompactionStrategy (LCS)**: Organize SSTables into levels (like LSM tree)
- **TimeWindowCompactionStrategy (TWCS)**: Group by time window (great for time-series)

### **RocksDB / LevelDB**

LSM tree-based storage engine with segmented SSTable files.

**Structure**:
\`\`\`
/var/rocksdb/
  000001.log            (WAL, segmented)
  000002.log
  000003.log
  000010.sst            (Level 0, 2MB)
  000011.sst            (Level 0, 2MB)
  000020.sst            (Level 1, 10MB)
  000021.sst            (Level 1, 10MB)
  000030.sst            (Level 2, 50MB)
\`\`\`

**Levels**:
- Level 0: Newly flushed from memtable, may have overlapping keys
- Level 1+: Non-overlapping key ranges, compacted from above

**Compaction**: Merge segments from Level N to Level N+1

### **Elasticsearch**

Elasticsearch uses **Lucene segments** under the hood.

**Segment Properties**:
- Immutable: Once written, never modified
- Merge: Background process merges small segments into larger ones
- Deleted docs: Marked in separate file (not immediately removed)

**Refresh**: New segments made searchable (default: 1 second)
**Flush**: Segments flushed to disk periodically

---

## Implementation Considerations

### **Concurrency Control**

**Single Writer**:
- Only one thread writes to active segment
- Simplifies synchronization
- Used by most systems

**Multiple Readers**:
- Many threads can read from closed segments concurrently
- No locking needed (segments are immutable)

### **Crash Recovery**

**Corrupted Active Segment**:
\`\`\`
1. Detect corruption (e.g., incomplete write)
2. Truncate active segment to last valid entry
3. Discard incomplete entry
4. Continue writing
\`\`\`

**Missing Segments**:
\`\`\`
If segments 0, 1, 3 exist but 2 is missing:
  → Critical error, cannot recover
  → Restore from backup or replicas
\`\`\`

**Checksums**: Each entry has checksum to detect corruption.

### **Monitoring**

**Key Metrics**:

1. **Segment Count**: Number of active segments
   - Too many: Compaction not keeping up
   - Too few: Segments too large

2. **Active Segment Size**: Current size of active segment
   - Approaching limit: Rotation imminent

3. **Segment Creation Rate**: New segments per hour
   - High rate: High write throughput
   - Sudden increase: Investigate

4. **Compaction Lag**: Segments waiting for compaction
   - Growing: Compaction can't keep up
   - Need more resources

5. **Disk Usage**: Total size of all segments
   - Approaching capacity: Adjust retention or add storage

**Alerts**:
- Segment count > 1000 (too many small segments)
- Compaction lag > 100 segments
- Disk usage > 80%

---

## Common Pitfalls

### **1. Too Small Segments**

**Problem**: Creating 1MB segments results in millions of files.

**Issues**:
- File system performance degrades
- Too many open file descriptors
- Metadata overhead

**Solution**: Use reasonable segment size (100MB - 1GB for most systems).

### **2. No Compaction**

**Problem**: Segments accumulate with redundant data.

**Issues**:
- Disk space wasted
- Read performance degrades (must check many segments)
- Eventually run out of disk

**Solution**: Implement background compaction.

### **3. Ignoring Retention Policies**

**Problem**: Segments never deleted.

**Issues**:
- Infinite disk growth
- Old data retained indefinitely

**Solution**: Implement and monitor retention policies.

### **4. Not Handling Segment Corruption**

**Problem**: Corrupted segment causes reads to fail.

**Issues**:
- Data loss
- Service outage

**Solution**:
- Checksums for all entries
- Replication across nodes
- Regular backups

### **5. Synchronous Compaction**

**Problem**: Compaction blocks writes.

**Issues**:
- Write latency spikes
- Reduced availability

**Solution**: Background compaction (separate thread/process).

---

## Interview Tips

### **Key Concepts**1. **Why Segment**: Easier deletion, compaction, recovery, archival
2. **How it works**: Multiple fixed-size files, active vs closed segments
3. **Rotation**: Size, time, or record-count triggers
4. **Compaction**: Merge segments, remove duplicates/deletions
5. **Real-world**: Kafka, Cassandra, RocksDB

### **Common Interview Questions**

**Q: Why not use a single large log file?**
A: "Single large file has issues: (1) Can't delete old data without rewriting entire file. (2) Slow recovery (must read entire file). (3) Compaction requires rewriting all data. (4) Difficult to archive old data. Segmenting solves all these—just delete old segments, recover segments in parallel, compact individual segments, and move segments to cold storage."

**Q: How do you decide when to create a new segment?**
A: "Typically use multiple triggers: (1) Size-based: Create new segment when current reaches 100MB-1GB. (2) Time-based: Create new segment every hour/day (aligns with retention). (3) Record-count: Create after N records. Use whichever trigger hits first. This handles both high-throughput and low-throughput scenarios."

**Q: What's the difference between deletion and compaction?**
A: "Deletion removes entire segments (usually based on time or size retention policy). Compaction merges segments and removes redundant entries (keeping latest value for each key). Deletion is simple (delete files). Compaction is complex (must merge, deduplicate, handle tombstones) but keeps log bounded by unique keys, not time."

**Q: How does segmentation improve read performance?**
A: "Multiple ways: (1) Parallel reads from different segments. (2) Hot segments cached in memory, cold on disk. (3) Indices per segment reduce search space. (4) Compaction reduces total segments to check. (5) Can stop searching once key found in recent segment."

---

## Summary

Segmented Log is a fundamental pattern for managing large append-only logs:

1. **Core Idea**: Break single log into multiple fixed-size segments
2. **Benefits**: Easy deletion, compaction, recovery, archival
3. **Rotation**: Triggered by size, time, or record count
4. **Compaction**: Merge segments, remove redundant data
5. **Real-World**: Kafka (topics), Cassandra (SSTables), RocksDB (LSM tree)
6. **Trade-offs**: Segment size (small = many files, large = slow operations)

**Interview Focus**: Understand the problems segmentation solves (deletion, compaction, recovery), how rotation works, and real-world examples (Kafka, Cassandra).
`,
};
